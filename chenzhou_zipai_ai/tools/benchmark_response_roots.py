"""Benchmark recorded response roots without replaying whole games."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import RootResponseCandidate, public_view_from_dict
from ai.opponent_league import create_policy
from ai.simulation_trace import public_state_identity
from engine.rules import rules_for_room


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--state-id", action="append", default=[])
    parser.add_argument("--exclude-seeds-from-report", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--max-ms", type=float, default=10_000.0)
    parser.add_argument("--expected-changed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    cases = response_cases(report)
    excluded_seeds: set[int] = set()
    if args.exclude_seeds_from_report is not None:
        exclusion = json.loads(
            args.exclude_seeds_from_report.read_text(encoding="utf-8")
        )
        excluded_seeds = {
            int(row["seed"])
            for row in exclusion.get("rows") or ()
        }
        cases = [
            case for case in cases if int(case["seed"]) not in excluded_seeds
        ]
    requested = {str(value) for value in args.state_id if str(value)}
    if requested:
        cases = [case for case in cases if case["public_view_id"] in requested]
        missing = requested - {str(case["public_view_id"]) for case in cases}
        if missing:
            raise ValueError(f"response_state_ids_missing:{len(missing)}")
    selected_cases = cases[: max(1, int(args.top))]
    policy = create_policy(args.candidate)
    rows = [
        benchmark_case(
            case,
            policy=policy,
            max_allowed_ms=max(1.0, float(args.max_ms)),
        )
        for case in selected_cases
    ]
    latencies = sorted(float(row["elapsed_ms"]) for row in rows)
    source_latencies = sorted(float(row["source_elapsed_ms"]) for row in rows)
    changed = sum(
        str(row["selected_key"]) != str(row["source_selected_key"])
        for row in rows
    )
    expected_changed = max(0, int(args.expected_changed))
    output = {
        "ok": bool(rows)
        and len(rows) == min(max(1, int(args.top)), len(cases))
        and all(bool(row["within_limit"]) for row in rows)
        and changed == expected_changed
        and all(bool(row["used_search"]) for row in rows)
        and all(int(row["invariant_violations"]) == 0 for row in rows)
        and all(int(row["coverage_failures"]) == 0 for row in rows),
        "schema_version": "response-root-latency-benchmark-v1",
        "source_report": str(args.source_report),
        "candidate": args.candidate,
        "available_roots": len(cases),
        "benchmarked_roots": len(rows),
        "root_selection": "source_elapsed_ms_desc_then_public_view_id_desc",
        "excluded_seed_report": (
            str(args.exclude_seeds_from_report)
            if args.exclude_seeds_from_report is not None
            else None
        ),
        "excluded_game_seeds": len(excluded_seeds),
        "selected_game_seeds": len({int(row["seed"]) for row in rows}),
        "max_allowed_ms": max(1.0, float(args.max_ms)),
        "mean_ms": round(mean(latencies), 3) if latencies else 0.0,
        "median_ms": round(median(latencies), 3) if latencies else 0.0,
        "p95_ms": round(_nearest_rank(latencies, 0.95), 3),
        "max_ms": round(max(latencies), 3) if latencies else 0.0,
        "source_mean_ms": (
            round(mean(source_latencies), 3) if source_latencies else 0.0
        ),
        "source_p95_ms": round(_nearest_rank(source_latencies, 0.95), 3),
        "source_max_ms": (
            round(max(source_latencies), 3) if source_latencies else 0.0
        ),
        "mean_reduction_fraction": (
            round(1.0 - mean(latencies) / mean(source_latencies), 6)
            if latencies and mean(source_latencies) > 0.0
            else 0.0
        ),
        "over_limit": sum(not bool(row["within_limit"]) for row in rows),
        "changed_from_source": changed,
        "expected_changed_from_source": expected_changed,
        "incomplete": sum(not bool(row["used_search"]) for row in rows),
        "invariant_violations": sum(
            int(row["invariant_violations"]) for row in rows
        ),
        "coverage_failures": sum(
            int(row["coverage_failures"]) for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: output[key]
                for key in (
                    "ok",
                    "benchmarked_roots",
                    "mean_ms",
                    "p95_ms",
                    "max_ms",
                    "source_mean_ms",
                    "mean_reduction_fraction",
                    "over_limit",
                    "changed_from_source",
                    "expected_changed_from_source",
                    "incomplete",
                    "invariant_violations",
                    "coverage_failures",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if output["ok"] else 1


def response_cases(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for game in report.get("rows") or ():
        for event_index, event in enumerate(
            game.get("candidate_response_events") or ()
        ):
            public_view = dict(event.get("public_view") or {})
            candidate_rows = list(event.get("candidates") or ())
            if not public_view or not candidate_rows:
                continue
            cases.append(
                {
                    "seed": int(game["seed"]),
                    "candidate_seat": int(game["candidate_seat"]),
                    "dealer": int(game["dealer"]),
                    "players": int(game["players"]),
                    "wildcard_enabled": bool(
                        event.get(
                            "wildcard_enabled",
                            game.get(
                                "wildcard_enabled",
                                report.get("wildcard_enabled", False),
                            ),
                        )
                    ),
                    "opponents": list(game.get("opponents") or ()),
                    "event_index": event_index,
                    "public_view_id": public_state_identity(public_view),
                    "public_view": public_view,
                    "candidates": candidate_rows,
                    "production_key": str(event.get("production_key") or ""),
                    "source_selected_key": str(
                        event.get("search_selected_key")
                        or event.get("production_key")
                        or ""
                    ),
                    "source_elapsed_ms": float(event.get("elapsed_ms") or 0.0),
                    "source_used_search": bool(event.get("used_search")),
                    "source_reason": str(event.get("reason") or ""),
                }
            )
    return sorted(
        cases,
        key=lambda case: (
            float(case["source_elapsed_ms"]),
            str(case["public_view_id"]),
        ),
        reverse=True,
    )


def benchmark_case(
    case: Mapping[str, Any],
    *,
    policy: Any,
    max_allowed_ms: float,
) -> dict[str, Any]:
    view = public_view_from_dict(case["public_view"])
    candidates = response_candidates(case["candidates"])
    production_key = str(case["production_key"])
    if production_key not in {candidate.key for candidate in candidates}:
        raise ValueError("response_production_key_not_legal")
    rules = rules_for_room(
        wildcard_enabled=bool(case["wildcard_enabled"]),
        players=int(case["players"]),
    )
    started = time.perf_counter()
    result = policy.response_search.search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
        preferred_key=production_key,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        **{
            key: value
            for key, value in case.items()
            if key not in {"public_view", "candidates"}
        },
        "selected_key": result.selected_key,
        "empirical_best_key": result.empirical_best_key,
        "confidence_override": result.confidence_override,
        "used_search": result.used_search,
        "reason": result.reason,
        "candidate_count": len(candidates),
        "elapsed_ms": round(elapsed_ms, 3),
        "within_limit": elapsed_ms <= max_allowed_ms,
        "simulations": result.simulations,
        "paired_determinizations": result.paired_determinizations,
        "deadline_interruptions": result.deadline_interruptions,
        "invariant_violations": result.rollout_invariant_violations,
        "coverage_failures": result.rollout_coverage_failures,
        "paired_advantages": [
            advantage.to_dict() for advantage in result.paired_advantages
        ],
    }


def response_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[RootResponseCandidate, ...]:
    return tuple(
        RootResponseCandidate(
            key=str(row["key"]),
            action_type=str(row["action_type"]),
            heuristic_value=float(row.get("heuristic_value") or 0.0),
            option_id=(
                str(row["option_id"])
                if row.get("option_id") is not None
                else None
            ),
            consumed_from_hand=tuple(
                str(label) for label in row.get("consumed_from_hand") or ()
            ),
            meld_groups=tuple(
                tuple(str(label) for label in group)
                for group in row.get("meld_groups") or ()
            ),
            followup_discard=(
                str(row["followup_discard"])
                if row.get("followup_discard") is not None
                else None
            ),
        )
        for row in rows
    )


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return values[max(0, math.ceil(len(values) * quantile) - 1)]


if __name__ == "__main__":
    raise SystemExit(main())
