"""Calibrate response overrides against actual-opponent counterfactuals."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import _familywise_95_t_critical


THRESHOLDS = (
    0.0,
    0.025,
    0.05,
    0.075,
    0.1,
    0.125,
    0.15,
    0.175,
    0.2,
    0.225,
    0.25,
    0.275,
    0.3,
    0.325,
    0.35,
    0.4,
    0.45,
    0.5,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument(
        "--evidence-input",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evidence = _load_evidence(args.evidence_input)
    challenger_rows = list(
        manifest.get("response_challenger_rows") or ()
    )
    if args.benchmark_report is not None:
        benchmark = json.loads(
            args.benchmark_report.read_text(encoding="utf-8")
        )
        challenger_rows = rows_from_benchmark(
            challenger_rows,
            benchmark_rows=benchmark.get("rows") or (),
        )
    rows = build_calibration_rows(
        challenger_rows,
        evidence=evidence,
    )
    sweeps = [
        summarize_threshold(rows, threshold=threshold)
        for threshold in THRESHOLDS
    ]
    selected = choose_conservative_threshold(sweeps)
    report = {
        "ok": bool(rows) and len(rows) == len(challenger_rows),
        "schema_version": "response-override-calibration-v1",
        "manifest": str(args.manifest),
        "benchmark_report": (
            str(args.benchmark_report)
            if args.benchmark_report is not None
            else None
        ),
        "evidence_inputs": [str(path) for path in args.evidence_input],
        "states": len(rows),
        "opponents": sorted(
            {str(row["opponent"]) for row in rows}
        ),
        "selected_threshold": selected["threshold"],
        "selected_summary": selected,
        "threshold_sweep": sweeps,
        "leave_one_opponent_out": leave_one_opponent_out(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "ok",
                    "states",
                    "opponents",
                    "selected_threshold",
                    "selected_summary",
                    "leave_one_opponent_out",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["ok"] else 1


def rows_from_benchmark(
    challenger_rows: Sequence[Mapping[str, Any]],
    *,
    benchmark_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_public_id = {
        str(row["public_view_id"]): row for row in benchmark_rows
    }
    if len(by_public_id) != len(benchmark_rows):
        raise ValueError("duplicate_response_benchmark_state")
    rows: list[dict[str, Any]] = []
    for source in challenger_rows:
        public_id = str(source["public_state_id"])
        benchmark = by_public_id.get(public_id)
        if benchmark is None:
            raise ValueError(
                f"missing_response_benchmark_state:{public_id}"
            )
        production = str(benchmark["production_key"])
        advantages = list(benchmark.get("paired_advantages") or ())
        best = max(
            advantages,
            key=lambda row: (
                float(row.get("mean_delta") or 0.0),
                str(row.get("candidate_key") or ""),
            ),
            default=None,
        )
        empirical = (
            str(best.get("candidate_key") or "")
            if best is not None
            and float(best.get("mean_delta") or 0.0) > 0.0
            else production
        )
        rows.append(
            {
                **dict(source),
                "production_key": production,
                "selected_key": str(benchmark["selected_key"]),
                "empirical_best_key": empirical,
                "applied_override": (
                    str(benchmark["selected_key"]) != production
                ),
                "confidence_override": bool(
                    benchmark.get("confidence_override")
                ),
                "candidate_count": int(
                    benchmark.get("candidate_count") or 0
                ),
                "online_mean_delta": (
                    float(best["mean_delta"])
                    if best is not None and empirical != production
                    else None
                ),
                "online_standard_error": (
                    float(best["standard_error"])
                    if best is not None
                    and empirical != production
                    and best.get("standard_error") is not None
                    else None
                ),
                "online_lower_confidence_bound": (
                    float(best["lower_confidence_bound"])
                    if best is not None
                    and empirical != production
                    and best.get("lower_confidence_bound") is not None
                    else None
                ),
                "online_samples": (
                    int(best.get("samples") or 0)
                    if best is not None and empirical != production
                    else 0
                ),
            }
        )
    return rows


def build_calibration_rows(
    challenger_rows: Iterable[Mapping[str, Any]],
    *,
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for challenger in challenger_rows:
        state_hash = str(challenger["state_before_hash"])
        if state_hash in seen:
            raise ValueError(f"duplicate_challenger_state:{state_hash}")
        seen.add(state_hash)
        production = str(challenger["production_key"])
        candidate = str(challenger["empirical_best_key"])
        audits = list(evidence.get(state_hash) or ())
        if not audits:
            raise ValueError(f"missing_response_evidence:{state_hash}")
        deltas: list[float] = []
        audit_seeds: set[int] = set()
        for audit in audits:
            if not bool(audit.get("complete")):
                raise ValueError(f"incomplete_response_evidence:{state_hash}")
            audit_seed = int(audit.get("audit_seed") or 0)
            if audit_seed in audit_seeds:
                raise ValueError(
                    f"duplicate_response_audit_seed:{state_hash}:{audit_seed}"
                )
            audit_seeds.add(audit_seed)
            for world in audit.get("paired_worlds") or ():
                rewards = {
                    str(outcome["candidate_key"]): float(outcome["reward"])
                    for outcome in world.get("outcomes") or ()
                }
                if production not in rewards or candidate not in rewards:
                    raise ValueError(
                        f"response_candidate_missing:{state_hash}"
                    )
                deltas.append(rewards[candidate] - rewards[production])
        comparisons = max(
            1,
            int(challenger.get("candidate_count") or 0) - 1,
        )
        rows.append(
            {
                **dict(challenger),
                "opponent": str(
                    next(iter(challenger.get("opponents") or ()), "")
                ),
                "actual_audit_seeds": sorted(audit_seeds),
                "actual": summarize_deltas(
                    deltas,
                    comparisons=comparisons,
                ),
            }
        )
    return rows


def summarize_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    selected_rows = [
        row
        for row in rows
        if row.get("online_lower_confidence_bound") is not None
        and float(row["online_lower_confidence_bound"]) > threshold
    ]
    selected_hashes = {
        str(row["state_before_hash"]) for row in selected_rows
    }
    gains = [
        (
            float(row["actual"]["mean_delta"])
            if str(row["state_before_hash"]) in selected_hashes
            else 0.0
        )
        for row in rows
    ]
    return {
        "threshold": round(float(threshold), 6),
        "states": len(rows),
        "overrides": len(selected_rows),
        "beneficial_overrides": sum(
            float(row["actual"]["mean_delta"]) > 0.0
            for row in selected_rows
        ),
        "harmful_overrides": sum(
            float(row["actual"]["mean_delta"]) < 0.0
            for row in selected_rows
        ),
        "confidently_beneficial_overrides": sum(
            bool(row["actual"]["confidently_positive"])
            for row in selected_rows
        ),
        "confidently_harmful_overrides": sum(
            row["actual"]["upper_confidence_bound"] is not None
            and float(row["actual"]["upper_confidence_bound"]) < 0.0
            for row in selected_rows
        ),
        "total_actual_gain": round(sum(gains), 6),
        "mean_actual_gain": round(mean(gains) if gains else 0.0, 6),
        "state_level_gain": summarize_deltas(gains, comparisons=1),
    }


def choose_conservative_threshold(
    sweeps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not sweeps:
        raise ValueError("empty_response_threshold_sweep")
    zero_harm = [
        row for row in sweeps if int(row["harmful_overrides"]) == 0
    ]
    candidates = zero_harm or list(sweeps)
    return dict(
        max(
            candidates,
            key=lambda row: (
                -int(row["confidently_harmful_overrides"]),
                -int(row["harmful_overrides"]),
                float(row["total_actual_gain"]),
                int(row["confidently_beneficial_overrides"]),
                int(row["beneficial_overrides"]),
                -float(row["threshold"]),
            ),
        )
    )


def leave_one_opponent_out(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    opponents = sorted({str(row["opponent"]) for row in rows})
    for opponent in opponents:
        training = [
            row for row in rows if str(row["opponent"]) != opponent
        ]
        holdout = [
            row for row in rows if str(row["opponent"]) == opponent
        ]
        training_sweeps = [
            summarize_threshold(training, threshold=threshold)
            for threshold in THRESHOLDS
        ]
        selected = choose_conservative_threshold(training_sweeps)
        holdout_summary = summarize_threshold(
            holdout,
            threshold=float(selected["threshold"]),
        )
        results.append(
            {
                "held_out_opponent": opponent,
                "training_states": len(training),
                "holdout_states": len(holdout),
                "selected_threshold": selected["threshold"],
                "holdout": holdout_summary,
            }
        )
    return results


def summarize_deltas(
    deltas: Sequence[float],
    *,
    comparisons: int,
) -> dict[str, Any]:
    samples = len(deltas)
    average = mean(deltas) if deltas else 0.0
    if samples >= 2:
        squared_error = sum(
            (delta - average) ** 2 for delta in deltas
        )
        sample_stddev = math.sqrt(squared_error / (samples - 1))
        standard_error = sample_stddev / math.sqrt(samples)
        radius = _familywise_95_t_critical(
            samples - 1,
            comparisons=max(1, comparisons),
        ) * standard_error
        lower_bound: float | None = average - radius
        upper_bound: float | None = average + radius
    else:
        sample_stddev = 0.0
        standard_error = 0.0
        lower_bound = None
        upper_bound = None
    return {
        "samples": samples,
        "mean_delta": round(average, 6),
        "standard_error": round(standard_error, 6),
        "lower_confidence_bound": (
            round(lower_bound, 6)
            if lower_bound is not None
            else None
        ),
        "upper_confidence_bound": (
            round(upper_bound, 6)
            if upper_bound is not None
            else None
        ),
        "confidently_positive": bool(
            lower_bound is not None and lower_bound > 0.0
        ),
    }


def _load_evidence(
    paths: Sequence[Path],
) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for audit in _read_jsonl(path):
            evidence[str(audit.get("state_before_hash") or "")].append(audit)
    return dict(evidence)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
