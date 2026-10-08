"""Run the concurrent frozen-baseline plus dual-validator discard candidate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean


APP_ROOT = Path(__file__).resolve().parents[1]
while str(APP_ROOT) in sys.path:
    sys.path.remove(str(APP_ROOT))

import concurrent.futures


sys.path.insert(0, str(APP_ROOT))

from ai.dual_discard_validator import close_shared_dual_discard_executors
from ai.dual_validated_candidate import (
    ProfessionalParallelDualValidatedCandidatePolicy,
)
from ai.ismcts import public_view_from_dict
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from engine.rules import rules_for_room
from tools.evaluate_discard_racing_rechecks import _load_cases


ROLLOUT_FACTORIES = (
    IndependentFastRolloutPolicy,
    IndependentFastPressurePolicy,
    IndependentFastDenialPolicy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-report", type=Path, required=True)
    parser.add_argument("--override-manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--state-hash", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = _load_cases(
        args.league_report,
        args.override_manifest,
        evidence_paths=args.evidence,
        evidence_aggregation="pooled",
    )
    requested = {str(value) for value in args.state_hash if str(value)}
    if requested:
        cases = [
            case
            for case in cases
            if str(case["state_before_hash"]) in requested
        ]
    if args.limit is not None:
        cases = cases[: max(1, args.limit)]

    rows = []
    try:
        for index, case in enumerate(cases, start=1):
            row = _evaluate_case(case)
            rows.append(row)
            progress_every = max(0, int(args.progress_every))
            if (
                progress_every
                and (
                    index % progress_every == 0
                    or index == len(cases)
                )
            ):
                print(
                    json.dumps(
                        {
                            "progress": index,
                            "total": len(cases),
                            "last_elapsed_ms": row["elapsed_ms"],
                            "maximum_elapsed_ms": max(
                                float(item["elapsed_ms"])
                                for item in rows
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        close_shared_dual_discard_executors()
    report = _summarize(rows)
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
                    "baseline_mismatches",
                    "candidate_changes",
                    "improved_vs_current",
                    "regressed_vs_current",
                    "maximum_elapsed_ms",
                    "p95_elapsed_ms",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def _evaluate_case(case: dict) -> dict:
    event = case["event"]
    view = public_view_from_dict(event["public_view"])
    rules = rules_for_room(
        wildcard_enabled=bool(case["game"]["wildcard_enabled"]),
        players=int(case["game"]["players"]),
    )
    policy = ProfessionalParallelDualValidatedCandidatePolicy(
        rollout_policy_factories=ROLLOUT_FACTORIES,
    )
    started = time.perf_counter()
    error = None
    try:
        selected = policy.choose_discard(view, rules)
    except Exception as exc:  # pragma: no cover - report health
        selected = str(event["search_selected_label"])
        error = f"{type(exc).__name__}:{exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    baseline = policy.last_baseline_search
    validation = policy.last_validation
    deep = case["deep"]
    rewards = {
        str(item["label"]): float(item.get("average_reward") or 0.0)
        for item in deep.get("candidate_stats") or ()
    }
    current = str(event["search_selected_label"])
    recorded_baseline = str(
        event.get("baseline_selected_label")
        or current
    )
    best_reward = max(rewards.values())

    def regret(label: str) -> float | None:
        reward = rewards.get(label)
        return best_reward - reward if reward is not None else None

    return {
        "state_before_hash": case["state_before_hash"],
        "production_label": str(event["production_label"]),
        "recorded_current_label": current,
        "recorded_baseline_label": recorded_baseline,
        "recomputed_baseline_label": (
            baseline.selected_label if baseline is not None else None
        ),
        "candidate_label": selected,
        "baseline_matches_recorded": bool(
            baseline is not None
            and baseline.selected_label == recorded_baseline
        ),
        "validation_selected_label": (
            validation.selected_label if validation is not None else None
        ),
        "validation_complete": bool(
            validation is not None and validation.complete
        ),
        "validation_confidence_override": bool(
            validation is not None
            and validation.confidence_override
        ),
        "current_regret": regret(current),
        "candidate_regret": regret(selected),
        "elapsed_ms": round(elapsed_ms, 3),
        "baseline_elapsed_ms": round(
            float(baseline.elapsed_ms) if baseline is not None else 0.0,
            3,
        ),
        "validation_elapsed_ms": round(
            float(validation.elapsed_ms)
            if validation is not None
            else 0.0,
            3,
        ),
        "error": error or policy.last_validation_error,
    }


def _summarize(rows: list[dict]) -> dict:
    comparable = [
        row
        for row in rows
        if row["current_regret"] is not None
        and row["candidate_regret"] is not None
    ]
    changed = [
        row
        for row in comparable
        if row["candidate_label"] != row["recorded_current_label"]
    ]
    improved = [
        row
        for row in changed
        if float(row["candidate_regret"])
        < float(row["current_regret"]) - 1e-12
    ]
    regressed = [
        row
        for row in changed
        if float(row["candidate_regret"])
        > float(row["current_regret"]) + 1e-12
    ]
    elapsed = sorted(float(row["elapsed_ms"]) for row in rows)
    p95_index = min(
        len(elapsed) - 1,
        int((len(elapsed) - 1) * 0.95),
    ) if elapsed else 0
    return {
        "ok": (
            all(row["error"] is None for row in rows)
            and all(row["validation_complete"] for row in rows)
            and all(row["baseline_matches_recorded"] for row in rows)
            and len(comparable) == len(rows)
        ),
        "schema_version": "parallel-dual-candidate-state-evaluation-v1",
        "states": len(rows),
        "baseline_mismatches": sum(
            not row["baseline_matches_recorded"] for row in rows
        ),
        "candidate_changes": len(changed),
        "improved_vs_current": len(improved),
        "regressed_vs_current": len(regressed),
        "mean_regret_reduction": (
            round(
                mean(float(row["current_regret"]) for row in comparable)
                - mean(
                    float(row["candidate_regret"])
                    for row in comparable
                ),
                6,
            )
            if comparable
            else None
        ),
        "maximum_elapsed_ms": max(elapsed, default=0.0),
        "p95_elapsed_ms": (
            elapsed[p95_index] if elapsed else 0.0
        ),
        "rows": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
