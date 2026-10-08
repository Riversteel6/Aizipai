"""Replay recorded response roots through a candidate search policy."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Iterator, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.counterfactual_audit import _response_candidates
from ai.ismcts import public_view_from_dict
from ai.opponent_league import create_policy
from ai.simulation_trace import canonical_public_state, public_state_identity
from engine.rules import rules_for_room


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-report", type=Path, required=True)
    parser.add_argument("--trace-input", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--state-id", action="append", default=[])
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--max-ms", type=float, default=15_000.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.league_report.read_text(encoding="utf-8"))
    cases = response_cases(report, traces=_read_jsonl(args.trace_input))
    requested = {str(value) for value in args.state_id if str(value)}
    if requested:
        cases = [
            case for case in cases if case["public_view_id"] in requested
        ]
        missing = requested - {
            str(case["public_view_id"]) for case in cases
        }
        if missing:
            raise ValueError(
                f"candidate_response_state_ids_missing:{len(missing)}"
            )
    selected_cases = cases[: max(1, int(args.top))]
    rows = [
        _benchmark_case(
            case,
            candidate=args.candidate,
            max_allowed_ms=max(1.0, float(args.max_ms)),
        )
        for case in selected_cases
    ]
    latencies = sorted(float(row["elapsed_ms"]) for row in rows)
    output = {
        "ok": bool(rows)
        and all(bool(row["within_limit"]) for row in rows)
        and all(int(row["invariant_violations"]) == 0 for row in rows)
        and all(int(row["coverage_failures"]) == 0 for row in rows),
        "schema_version": "candidate-response-latency-benchmark-v1",
        "source_report": str(args.league_report),
        "source_trace": str(args.trace_input),
        "candidate": args.candidate,
        "available_roots": len(cases),
        "benchmarked_roots": len(rows),
        "max_allowed_ms": max(1.0, float(args.max_ms)),
        "mean_ms": round(mean(latencies), 3) if latencies else 0.0,
        "median_ms": round(median(latencies), 3) if latencies else 0.0,
        "p95_ms": round(_nearest_rank(latencies, 0.95), 3),
        "max_ms": round(max(latencies), 3) if latencies else 0.0,
        "over_limit": sum(not bool(row["within_limit"]) for row in rows),
        "changed_from_production": sum(
            str(row["selected_key"]) != str(row["production_key"])
            for row in rows
        ),
        "changed_from_source": sum(
            str(row["selected_key"]) != str(row["source_selected_key"])
            for row in rows
        ),
        "confidence_overrides": sum(
            bool(row["confidence_override"]) for row in rows
        ),
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
                    "candidate",
                    "benchmarked_roots",
                    "mean_ms",
                    "p95_ms",
                    "max_ms",
                    "over_limit",
                    "changed_from_production",
                    "changed_from_source",
                    "confidence_overrides",
                    "invariant_violations",
                    "coverage_failures",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if output["ok"] else 1


def response_cases(
    report: Mapping[str, Any],
    *,
    traces: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    trace_games = {_game_identity(game): game for game in traces}
    report_rows = list(report.get("rows") or ())
    report_ids = {_game_identity(game) for game in report_rows}
    if set(trace_games) != report_ids:
        raise ValueError("candidate_response_trace_game_mismatch")
    cases: list[dict[str, Any]] = []
    for game in report_rows:
        identity = _game_identity(game)
        events = list(game.get("candidate_response_events") or ())
        steps = [
            step
            for step in trace_games[identity].get("decision_trace") or ()
            if step.get("phase") == "response_root"
            and int(step.get("seat", -1))
            == int(game.get("candidate_seat", -1))
        ]
        if len(events) != len(steps):
            raise ValueError(
                f"candidate_response_event_count_mismatch:{identity}"
            )
        for event_index, (event, step) in enumerate(zip(events, steps)):
            event_view = event.get("public_view") or {}
            trace_view = step.get("public_state") or {}
            if canonical_public_state(event_view) != canonical_public_state(
                trace_view
            ):
                raise ValueError(
                    f"candidate_response_public_state_mismatch:{identity}"
                )
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
                    "public_view_id": public_state_identity(trace_view),
                    "public_view": dict(trace_view),
                    "trace": dict(step),
                    "production_key": str(
                        event.get("production_key") or ""
                    ),
                    "source_selected_key": str(
                        event.get("search_selected_key")
                        or event.get("production_key")
                        or ""
                    ),
                }
            )
    return cases


def _benchmark_case(
    case: Mapping[str, Any],
    *,
    candidate: str,
    max_allowed_ms: float,
) -> dict[str, Any]:
    policy = create_policy(candidate)
    view = public_view_from_dict(case["public_view"])
    candidates = _response_candidates(case["trace"])
    production_key = str(case["production_key"])
    if production_key not in {item.key for item in candidates}:
        raise ValueError("candidate_response_production_not_legal")
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
            if key not in {"public_view", "trace"}
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
            advantage.to_dict()
            for advantage in result.paired_advantages
        ],
    }


def _game_identity(
    row: Mapping[str, Any],
) -> tuple[int, int, int, tuple[str, ...]]:
    return (
        int(row["seed"]),
        int(row["candidate_seat"]),
        int(row["dealer"]),
        tuple(str(item) for item in row.get("opponents") or ()),
    )


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil(len(values) * quantile) - 1)
    return values[index]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
