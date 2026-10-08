"""Pool paired counterfactual deltas across independent audit seeds."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterator


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import _familywise_95_t_critical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-input",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--state-hash", required=True)
    parser.add_argument("--preferred-key")
    parser.add_argument("--candidate-key")
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--public-view-id")
    parser.add_argument("--familywise-comparisons", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preferred_key, candidate_key = _resolve_action_keys(
        preferred_key=args.preferred_key,
        candidate_key=args.candidate_key,
        benchmark_report=args.benchmark_report,
        public_view_id=args.public_view_id,
    )

    deltas: list[float] = []
    fingerprints: set[tuple[str, int, str]] = set()
    rows: list[dict[str, Any]] = []
    for path in args.evidence_input:
        input_deltas: list[float] = []
        input_fingerprints: set[tuple[str, int, str]] = set()
        matching_audits = 0
        for audit in _read_jsonl(path):
            if str(audit.get("state_before_hash") or "") != args.state_hash:
                continue
            matching_audits += 1
            for world in audit.get("paired_worlds") or ():
                rewards = {
                    str(outcome["candidate_key"]): float(outcome["reward"])
                    for outcome in world.get("outcomes") or ()
                }
                if (
                    preferred_key not in rewards
                    or candidate_key not in rewards
                ):
                    continue
                delta = (
                    rewards[candidate_key]
                    - rewards[preferred_key]
                )
                input_deltas.append(delta)
                input_fingerprints.add(
                    (
                        str(world.get("world_fingerprint") or ""),
                        int(world.get("rollout_seed") or 0),
                        str(world.get("opponent_policy") or ""),
                    )
                )
        if matching_audits != 1:
            raise ValueError(
                f"expected_one_matching_audit:{path}:{matching_audits}"
            )
        rows.append(
            {
                "evidence_input": str(path),
                **_summary(
                    input_deltas,
                    comparisons=max(1, args.familywise_comparisons),
                ),
                "unique_worlds": len(input_fingerprints),
            }
        )
        deltas.extend(input_deltas)
        fingerprints.update(input_fingerprints)

    report = {
        "ok": bool(deltas)
        and all(row["samples"] > 0 for row in rows),
        "schema_version": "counterfactual-seed-pool-v1",
        "state_hash": args.state_hash,
        "preferred_key": preferred_key,
        "candidate_key": candidate_key,
        "familywise_comparisons": max(
            1,
            int(args.familywise_comparisons),
        ),
        "inputs": rows,
        "pooled": {
            **_summary(
                deltas,
                comparisons=max(1, args.familywise_comparisons),
            ),
            "unique_worlds": len(fingerprints),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


def _resolve_action_keys(
    *,
    preferred_key: str | None,
    candidate_key: str | None,
    benchmark_report: Path | None,
    public_view_id: str | None,
) -> tuple[str, str]:
    if benchmark_report is not None or public_view_id:
        if benchmark_report is None or not public_view_id:
            raise ValueError(
                "benchmark_report_and_public_view_id_required_together"
            )
        report = json.loads(
            benchmark_report.read_text(encoding="utf-8")
        )
        matches = [
            row
            for row in report.get("rows") or ()
            if str(row.get("public_view_id") or "") == public_view_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "expected_one_benchmark_row:"
                f"{public_view_id}:{len(matches)}"
            )
        row = matches[0]
        derived_preferred = str(row["source_selected_label"])
        derived_candidate = str(row["selected_label"])
        if (
            preferred_key
            and preferred_key != derived_preferred
        ):
            raise ValueError("preferred_key_benchmark_mismatch")
        if (
            candidate_key
            and candidate_key != derived_candidate
        ):
            raise ValueError("candidate_key_benchmark_mismatch")
        return derived_preferred, derived_candidate
    if not preferred_key or not candidate_key:
        raise ValueError(
            "preferred_key_and_candidate_key_required"
        )
    return preferred_key, candidate_key


def _summary(
    deltas: list[float],
    *,
    comparisons: int,
) -> dict[str, Any]:
    samples = len(deltas)
    average = mean(deltas) if deltas else 0.0
    if samples >= 2:
        squared_error = sum(
            (delta - average) ** 2
            for delta in deltas
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
        "sample_stddev": round(sample_stddev, 6),
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
        "positive_samples": sum(delta > 0.0 for delta in deltas),
        "tied_samples": sum(abs(delta) <= 1e-12 for delta in deltas),
        "negative_samples": sum(delta < 0.0 for delta in deltas),
        "confidently_positive": bool(
            lower_bound is not None and lower_bound > 0.0
        ),
    }


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
