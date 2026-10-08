"""Replay recorded discard roots through a league candidate with wall-clock limits."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterator, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import public_view_from_dict
from ai.opponent_league import (
    _effective_league_workers,
    create_policy,
)
from ai.simulation_trace import public_state_identity
from engine.rules import rules_for_room


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-report", type=Path, required=True)
    parser.add_argument("--trace-input", type=Path)
    parser.add_argument("--source-policy", default="")
    parser.add_argument(
        "--candidate",
        default="professional_parallel_dual_validated_candidate",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max-ms", type=float, default=15_000.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--state-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.league_report.read_text(encoding="utf-8"))
    cases = (
        _discard_trace_cases(
            _read_jsonl(args.trace_input),
            source_policy=str(args.source_policy),
        )
        if args.trace_input is not None
        else _discard_cases(source)
    )
    requested = {str(value) for value in args.state_id if str(value)}
    if requested:
        cases = [
            case
            for case in cases
            if case["public_view_id"] in requested
        ]
        missing = requested - {
            case["public_view_id"]
            for case in cases
        }
        if missing:
            raise ValueError(
                f"candidate_root_state_ids_missing:{len(missing)}"
            )
    selected_cases = sorted(
        cases,
        key=lambda case: (
            float(case["source_elapsed_ms"]),
            str(case["public_view_id"]),
        ),
        reverse=True,
    )[: max(1, args.top)]

    requested_workers = max(1, int(args.workers))
    effective_workers = _effective_league_workers(
        candidate=args.candidate,
        requested=requested_workers,
        remaining_jobs=len(selected_cases),
    )
    tasks = [
        (
            case,
            args.candidate,
            max(1.0, args.max_ms),
        )
        for case in selected_cases
    ]
    if effective_workers > 1:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=get_context("spawn"),
        ) as pool:
            rows = list(
                pool.map(
                    _benchmark_case_task,
                    tasks,
                    chunksize=1,
                )
            )
    else:
        rows = [
            _benchmark_case_task(task)
            for task in tasks
        ]
    latencies = sorted(float(row["elapsed_ms"]) for row in rows)
    strategy_runtime_error_reasons = _strategy_runtime_error_reasons(rows)
    report = {
        "ok": bool(rows)
        and all(row["within_limit"] for row in rows)
        and all(row["invariant_violations"] == 0 for row in rows)
        and all(row["coverage_failures"] == 0 for row in rows)
        and not strategy_runtime_error_reasons,
        "schema_version": "candidate-root-latency-benchmark-v1",
        "source_report": str(args.league_report),
        "candidate": args.candidate,
        "available_roots": len(cases),
        "benchmarked_roots": len(rows),
        "max_allowed_ms": max(1.0, args.max_ms),
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "worker_model": (
            "spawn_process_executor"
            if effective_workers > 1
            else "serial"
        ),
        "mean_ms": round(mean(latencies), 3) if latencies else 0.0,
        "median_ms": round(median(latencies), 3) if latencies else 0.0,
        "p95_ms": round(_nearest_rank(latencies, 0.95), 3),
        "max_ms": round(max(latencies), 3) if latencies else 0.0,
        "over_limit": sum(
            int(not row["within_limit"])
            for row in rows
        ),
        "changed_actions": sum(
            int(row["selected_label"] != row["source_selected_label"])
            for row in rows
        ),
        "baseline_overrides": sum(
            int(
                row["selected_label"]
                != row.get("baseline_selected_label")
            )
            for row in rows
        ),
        "decision_budget_fallbacks": sum(
            int(row["decision_budget_fallbacks"])
            for row in rows
        ),
        "validation_timeouts": sum(
            int(row["validation_timeouts"])
            for row in rows
        ),
        "validation_incomplete_fallbacks": sum(
            int(row["validation_incomplete_fallbacks"])
            for row in rows
        ),
        "late_full_coverage_fallbacks": sum(
            int(bool(row["late_full_coverage_fallback"]))
            for row in rows
        ),
        "reconfirmation_budget_fallbacks": sum(
            int(row["reconfirmation_budget_fallbacks"])
            for row in rows
        ),
        "reconfirmation_incomplete_fallbacks": sum(
            int(row["reconfirmation_incomplete_fallbacks"])
            for row in rows
        ),
        "deadline_interruptions": sum(
            int(row["deadline_interruptions"])
            for row in rows
        ),
        "baseline_phase_deadline_interruptions": sum(
            int(row["baseline_phase_deadline_interruptions"])
            for row in rows
        ),
        "invariant_violations": sum(
            int(row["invariant_violations"])
            for row in rows
        ),
        "coverage_failures": sum(
            int(row["coverage_failures"])
            for row in rows
        ),
        "strategy_runtime_errors": sum(
            int(
                row.get("validation_error") is not None
                and not str(row["validation_error"]).startswith(
                    "decision_budget_"
                )
            )
            for row in rows
        ),
        "strategy_runtime_error_reasons": (
            strategy_runtime_error_reasons
        ),
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
                    "candidate",
                    "benchmarked_roots",
                    "mean_ms",
                    "p95_ms",
                    "max_ms",
                    "over_limit",
                    "changed_actions",
                    "baseline_overrides",
                    "decision_budget_fallbacks",
                    "validation_timeouts",
                    "validation_incomplete_fallbacks",
                    "late_full_coverage_fallbacks",
                    "reconfirmation_budget_fallbacks",
                    "reconfirmation_incomplete_fallbacks",
                    "deadline_interruptions",
                    "baseline_phase_deadline_interruptions",
                    "invariant_violations",
                    "coverage_failures",
                    "strategy_runtime_errors",
                    "strategy_runtime_error_reasons",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def _strategy_runtime_error_reasons(
    rows: list[dict[str, Any]],
) -> list[str]:
    return sorted(
        {
            str(row["validation_error"])
            for row in rows
            if row.get("validation_error")
            and not str(row["validation_error"]).startswith(
                "decision_budget_"
            )
        }
    )


def _discard_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    wildcard_enabled = bool(report.get("wildcard_enabled"))
    for row in report.get("rows") or ():
        for event in row.get("candidate_discard_events") or ():
            if not event.get("public_view"):
                continue
            cases.append(
                {
                    "matchup": list(row.get("opponents") or ()),
                    "seed": int(row["seed"]),
                    "candidate_seat": int(row["candidate_seat"]),
                    "dealer": int(row["dealer"]),
                    "players": int(row["players"]),
                    "wildcard_enabled": wildcard_enabled,
                    "public_view_id": str(event["public_view_id"]),
                    "public_view": event["public_view"],
                    "source_elapsed_ms": float(
                        event.get("elapsed_ms") or 0.0
                    ),
                    "source_selected_label": str(
                        event.get("search_selected_label")
                        or event.get("production_label")
                        or ""
                    ),
                }
            )
    return cases


def _discard_trace_cases(
    games: Iterator[Mapping[str, Any]],
    *,
    source_policy: str = "",
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for game in games:
        candidate_seat = int(game.get("candidate_seat") or 0)
        for trace in game.get("decision_trace") or ():
            selected_key = str(trace.get("selected_key") or "")
            public_view = trace.get("public_state")
            if (
                str(trace.get("phase") or "") != "discard"
                or int(trace.get("seat", -1)) != candidate_seat
                or (
                    source_policy
                    and str(trace.get("policy") or "") != source_policy
                )
                or not selected_key.startswith("DISCARD:")
                or not isinstance(public_view, Mapping)
            ):
                continue
            cases.append(
                {
                    "matchup": list(game.get("opponents") or ()),
                    "seed": int(game.get("seed") or 0),
                    "candidate_seat": candidate_seat,
                    "dealer": int(game.get("dealer") or 0),
                    "players": int(game.get("players") or 0),
                    "wildcard_enabled": bool(
                        game.get("wildcard_enabled")
                    ),
                    "public_view_id": public_state_identity(public_view),
                    "public_view": dict(public_view),
                    "source_elapsed_ms": float(
                        trace.get("elapsed_ms") or 0.0
                    ),
                    "source_selected_label": selected_key.split(":", 1)[1],
                }
            )
    return cases


def _benchmark_case(
    case: dict[str, Any],
    *,
    candidate: str,
    max_allowed_ms: float,
) -> dict[str, Any]:
    policy = create_policy(candidate)
    view = public_view_from_dict(case["public_view"])
    rules = rules_for_room(
        wildcard_enabled=bool(case["wildcard_enabled"]),
        players=int(case["players"]),
    )
    started = time.perf_counter()
    selected = policy.choose_discard(view, rules)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    search = getattr(policy, "last_search", None)
    diagnostics = policy.diagnostics()
    discard_events = tuple(policy.discard_events())
    event = discard_events[-1] if discard_events else {}
    validation = getattr(policy, "last_validation", None)
    return {
        **{
            key: value
            for key, value in case.items()
            if key != "public_view"
        },
        "selected_label": selected,
        "production_label": event.get("production_label"),
        "proposed_search_selected_label": event.get(
            "proposed_search_selected_label"
        ),
        "final_authorization_reason": event.get(
            "final_authorization_reason"
        ),
        "elapsed_ms": round(elapsed_ms, 3),
        "within_limit": elapsed_ms <= max_allowed_ms,
        "decision_budget_fallbacks": int(
            diagnostics.get("decision_budget_fallbacks") or 0
        ),
        "validation_timeouts": int(
            diagnostics.get("validation_timeouts") or 0
        ),
        "validation_incomplete_fallbacks": int(
            diagnostics.get("validation_incomplete_fallbacks") or 0
        ),
        "reconfirmation_budget_fallbacks": int(
            diagnostics.get("reconfirmation_budget_fallbacks") or 0
        ),
        "reconfirmation_incomplete_fallbacks": int(
            diagnostics.get("reconfirmation_incomplete_fallbacks") or 0
        ),
        "deadline_interruptions": int(
            getattr(search, "deadline_interruptions", 0) or 0
        ),
        "baseline_phase_deadline_interruptions": int(
            event.get("baseline_phase_deadline_interruptions") or 0
        ),
        "invariant_violations": int(
            getattr(search, "rollout_invariant_violations", 0) or 0
        ),
        "coverage_failures": int(
            getattr(search, "rollout_coverage_failures", 0) or 0
        ),
        "validation_complete": event.get("validation_complete"),
        "validation_attempted": event.get("validation_attempted", True),
        "late_full_coverage_fallback": event.get(
            "late_full_coverage_fallback",
            False,
        ),
        "validation_reconfirmation_attempted": event.get(
            "validation_reconfirmation_attempted"
        ),
        "validation_error": event.get("validation_error"),
        "baseline_selected_label": event.get(
            "baseline_selected_label"
        ),
        "validation_preferred_label": getattr(
            validation,
            "preferred_label",
            None,
        ),
        "validation_challenger_label": getattr(
            validation,
            "challenger_label",
            None,
        ),
        "validation_challenger_labels": [
            evidence.challenger_label
            for evidence in getattr(
                validation,
                "challenger_evidence",
                (),
            )
        ],
        "validation_selected_label": getattr(
            validation,
            "selected_label",
            None,
        ),
        "validation_coverage_ranking": [
            {
                "label": candidate_stats.label,
                "visits": candidate_stats.visits,
                "average_reward": candidate_stats.average_reward,
                "win_rate": candidate_stats.win_rate,
                "mean_outcome_score": (
                    candidate_stats.mean_outcome_score
                ),
                "heuristic_value": candidate_stats.heuristic_value,
            }
            for candidate_stats in (
                getattr(
                    getattr(validation, "coverage", None),
                    "candidates",
                    (),
                )
            )
        ],
        "validation_diagnostics": event.get("validation_diagnostics"),
        "baseline_diagnostics": event.get("baseline_diagnostics"),
        "decision_time_budget_ms": event.get("decision_time_budget_ms"),
        "confirmation_reserve_ms": event.get("confirmation_reserve_ms"),
        "validation_cleanup_elapsed_ms": event.get(
            "validation_cleanup_elapsed_ms"
        ),
        "reason": str(getattr(search, "reason", "")),
    }


def _benchmark_case_task(
    task: tuple[dict[str, Any], str, float],
) -> dict[str, Any]:
    case, candidate, max_allowed_ms = task
    return _benchmark_case(
        case,
        candidate=candidate,
        max_allowed_ms=max_allowed_ms,
    )


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(
        0,
        min(
            len(values) - 1,
            int((len(values) * percentile) + 0.999999) - 1,
        ),
    )
    return values[index]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
