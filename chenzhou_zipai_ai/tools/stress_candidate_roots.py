"""Continuously replay recorded discard roots through one candidate instance."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.dual_discard_validator import close_shared_dual_discard_executors
from ai.ismcts import public_view_from_dict
from ai.opponent_league import create_policy
from engine.rules import rules_for_room


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-report", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        default="professional_parallel_dual_validated_candidate",
    )
    parser.add_argument("--state-id", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-ms", type=float, default=15_000.0)
    parser.add_argument(
        "--require-validation-complete",
        action="store_true",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.league_report.read_text(encoding="utf-8"))
    reference = json.loads(
        args.reference_report.read_text(encoding="utf-8")
    )
    requested = tuple(
        dict.fromkeys(str(value) for value in args.state_id if str(value))
    )
    source_cases = _discard_cases(source)
    reference_cases = _discard_cases(reference)
    state_ids = requested or tuple(sorted(source_cases))
    missing_source = set(state_ids) - set(source_cases)
    missing_reference = set(state_ids) - set(reference_cases)
    if missing_source or missing_reference:
        raise ValueError(
            "candidate_root_state_ids_missing:"
            f"source={len(missing_source)}:"
            f"reference={len(missing_reference)}"
        )

    policy = create_policy(args.candidate)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for round_number in range(1, max(1, int(args.rounds)) + 1):
            for public_view_id in state_ids:
                case = source_cases[public_view_id]
                expected = reference_cases[public_view_id][
                    "source_selected_label"
                ]
                diagnostics_before = policy.diagnostics()
                call_started = time.perf_counter()
                selected = policy.choose_discard(
                    public_view_from_dict(case["public_view"]),
                    rules_for_room(
                        wildcard_enabled=bool(case["wildcard_enabled"]),
                        players=int(case["players"]),
                    ),
                )
                elapsed_ms = (
                    time.perf_counter() - call_started
                ) * 1000.0
                diagnostics_after = policy.diagnostics()
                event = policy.discard_events()[-1]
                time.sleep(0.02)
                lingering_threads = [
                    thread.name
                    for thread in threading.enumerate()
                    if thread.name.startswith("ThreadPoolExecutor")
                ]
                row = {
                    "round": round_number,
                    "public_view_id": public_view_id,
                    "expected_label": expected,
                    "selected_label": selected,
                    "matches_expected": selected == expected,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "within_limit": elapsed_ms <= max(1.0, args.max_ms),
                    "validation_complete": bool(
                        event.get("validation_complete")
                    ),
                    "validation_error": event.get("validation_error"),
                    "validation_worker_pending_at_cleanup": bool(
                        event.get(
                            "validation_worker_pending_at_cleanup"
                        )
                    ),
                    "validation_worker_done_after_cleanup": bool(
                        event.get(
                            "validation_worker_done_after_cleanup"
                        )
                    ),
                    "validation_cleanup_elapsed_ms": float(
                        event.get("validation_cleanup_elapsed_ms") or 0.0
                    ),
                    "lingering_per_decision_threads": lingering_threads,
                    "diagnostic_delta": {
                        key: int(diagnostics_after.get(key) or 0)
                        - int(diagnostics_before.get(key) or 0)
                        for key in (
                            "decision_budget_fallbacks",
                            "validation_timeouts",
                            "validation_incomplete_fallbacks",
                            "validation_cleanup_waits",
                        )
                    },
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "round",
                                "public_view_id",
                                "selected_label",
                                "expected_label",
                                "elapsed_ms",
                                "validation_complete",
                                "validation_error",
                                "lingering_per_decision_threads",
                            )
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        close_shared_dual_discard_executors()

    report = {
        "ok": bool(rows)
        and all(row["matches_expected"] for row in rows)
        and all(row["within_limit"] for row in rows)
        and all(not row["validation_error"] for row in rows)
        and all(
            row["validation_worker_done_after_cleanup"]
            for row in rows
        )
        and all(
            not row["lingering_per_decision_threads"]
            for row in rows
        )
        and (
            not args.require_validation_complete
            or all(row["validation_complete"] for row in rows)
        ),
        "schema_version": "candidate-root-continuity-stress-v1",
        "source_report": str(args.league_report),
        "reference_report": str(args.reference_report),
        "candidate": args.candidate,
        "rounds": max(1, int(args.rounds)),
        "roots": len(state_ids),
        "decisions": len(rows),
        "max_allowed_ms": max(1.0, args.max_ms),
        "require_validation_complete": bool(
            args.require_validation_complete
        ),
        "total_elapsed_ms": round(
            (time.perf_counter() - started) * 1000.0,
            3,
        ),
        "max_ms": round(
            max(float(row["elapsed_ms"]) for row in rows),
            3,
        ),
        "action_mismatches": sum(
            int(not row["matches_expected"])
            for row in rows
        ),
        "over_limit": sum(
            int(not row["within_limit"])
            for row in rows
        ),
        "validation_incomplete": sum(
            int(not row["validation_complete"])
            for row in rows
        ),
        "validation_errors": sum(
            int(bool(row["validation_error"]))
            for row in rows
        ),
        "lingering_per_decision_thread_observations": sum(
            int(bool(row["lingering_per_decision_threads"]))
            for row in rows
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
                    "decisions",
                    "total_elapsed_ms",
                    "max_ms",
                    "action_mismatches",
                    "over_limit",
                    "validation_incomplete",
                    "validation_errors",
                    "lingering_per_decision_thread_observations",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def _discard_cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for row in report.get("rows") or ():
        for event in row.get("candidate_discard_events") or ():
            public_view_id = str(event.get("public_view_id") or "")
            if not public_view_id or not event.get("public_view"):
                continue
            cases[public_view_id] = {
                "public_view": event["public_view"],
                "players": int(row["players"]),
                "wildcard_enabled": bool(
                    report.get("wildcard_enabled")
                ),
                "source_selected_label": str(
                    event.get("search_selected_label")
                    or event.get("production_label")
                    or ""
                ),
            }
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
