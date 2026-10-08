"""Verify exact response-confirmation equivalence on fixed recorded roots."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import public_view_from_dict
from ai.opponent_league import create_policy
from ai.dual_discard_validator import _balanced_world_shards, _shared_executor
from ai.parallel_response_search import (
    _merge_response_confirmation_shards,
    _response_search_task,
)
from engine.rules import rules_for_room
from tools.benchmark_response_roots import response_candidates, response_cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument(
        "--inherited-candidate",
        default=(
            "professional_parallel_multi_opponent_production_anchored_"
            "rank1_sharded_research"
        ),
    )
    parser.add_argument("--state-id", action="append", default=[])
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    cases = response_cases(report)
    requested = {str(value) for value in args.state_id if str(value)}
    if requested:
        cases = [case for case in cases if case["public_view_id"] in requested]
        missing = requested - {str(case["public_view_id"]) for case in cases}
        if missing:
            raise ValueError(f"response_state_ids_missing:{len(missing)}")
    cases = cases[: max(1, int(args.top))]
    rows = [
        verify_case(
            case,
            inherited_candidate=args.inherited_candidate,
            shard_count=max(1, int(args.shards)),
        )
        for case in cases
    ]
    failures = [
        failure
        for row in rows
        for failure in row.get("failures") or ()
    ]
    output = {
        "ok": bool(rows) and not failures,
        "schema_version": "exact-sharded-response-equivalence-v1",
        "source_report": str(args.source_report),
        "inherited_candidate": args.inherited_candidate,
        "fixed_root_selection": (
            "source_elapsed_ms_desc_then_public_view_id_desc"
        ),
        "states": len(rows),
        "shards": max(1, int(args.shards)),
        "worlds_compared": sum(int(row["worlds"]) for row in rows),
        "failure_count": len(failures),
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
                    "states",
                    "shards",
                    "worlds_compared",
                    "failure_count",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if output["ok"] else 1


def verify_case(
    case: dict[str, Any],
    *,
    inherited_candidate: str,
    shard_count: int,
) -> dict[str, Any]:
    policy = create_policy(inherited_candidate)
    progressive = policy.response_search
    progressive.response_binary_confirmation.config = replace(
        progressive.response_binary_confirmation.config,
        record_paired_worlds=True,
    )
    progressive.response_confirmation.config = replace(
        progressive.response_confirmation.config,
        record_paired_worlds=True,
    )
    view = public_view_from_dict(case["public_view"])
    candidates = response_candidates(case["candidates"])
    by_key = {candidate.key: candidate for candidate in candidates}
    preferred = str(case["production_key"])
    rules = rules_for_room(
        wildcard_enabled=bool(case["wildcard_enabled"]),
        players=int(case["players"]),
    )
    started = time.perf_counter()
    progressive_result = progressive.search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
        preferred_key=preferred,
    )
    monolithic_elapsed_ms = (time.perf_counter() - started) * 1000.0
    monolithic = progressive.last_confirmation
    failures: list[str] = []
    if monolithic is None:
        failures.append("monolithic_confirmation_missing")
        return _failure_row(
            case,
            failures=failures,
            elapsed_ms=monolithic_elapsed_ms,
            progressive_result=progressive_result,
        )
    confirmation_keys = [preferred]
    confirmation_keys.extend(
        candidate.candidate.key
        for candidate in monolithic.candidates
        if candidate.candidate.key != preferred
    )
    confirmation_candidates = tuple(by_key[key] for key in confirmation_keys)
    confirmation_policy = (
        progressive.response_binary_confirmation
        if progressive.last_response_selection is None
        else progressive.response_confirmation
    )
    config = confirmation_policy.config
    worlds = max(1, config.max_iterations // len(confirmation_candidates))
    shards = _balanced_world_shards(worlds, shard_count)
    executor = _shared_executor(max(2, shard_count))
    futures = [
        executor.submit(
            _response_search_task,
            {
                "view": view,
                "rules": rules,
                "candidates": confirmation_candidates,
                "preferred_key": preferred,
                "base_config": config,
                "rollout_policy_factories": (
                    confirmation_policy.rollout_policy_factories
                ),
                "worlds": amount,
                "paired_world_offset": offset,
            },
        )
        for offset, amount in shards
    ]
    shard_results = [future.result() for future in futures]
    merged = _merge_response_confirmation_shards(
        shard_results,
        candidates=confirmation_candidates,
        preferred_key=preferred,
        config=config,
        expected_worlds=worlds,
        root_seat=int(view.seat),
        elapsed_ms=monolithic.elapsed_ms,
    )
    if tuple(world.to_dict() for world in merged.paired_worlds) != tuple(
        world.to_dict() for world in monolithic.paired_worlds
    ):
        failures.append("paired_worlds_mismatch")
    if merged.candidates != monolithic.candidates:
        failures.append("candidate_statistics_mismatch")
    if merged.paired_advantages != monolithic.paired_advantages:
        failures.append("paired_advantages_mismatch")
    if replace(merged, elapsed_ms=0.0) != replace(
        monolithic,
        elapsed_ms=0.0,
    ):
        failures.append("full_confirmation_result_mismatch")
    return {
        "public_view_id": case["public_view_id"],
        "source_elapsed_ms": case["source_elapsed_ms"],
        "monolithic_progressive_elapsed_ms": round(monolithic_elapsed_ms, 3),
        "candidate_count": len(candidates),
        "confirmation_keys": confirmation_keys,
        "worlds": worlds,
        "shards": [list(shard) for shard in shards],
        "monolithic_selected_key": monolithic.selected_key,
        "merged_selected_key": merged.selected_key,
        "progressive_selected_key": progressive_result.selected_key,
        "failures": failures,
    }


def _failure_row(
    case: dict[str, Any],
    *,
    failures: list[str],
    elapsed_ms: float,
    progressive_result: Any,
) -> dict[str, Any]:
    return {
        "public_view_id": case["public_view_id"],
        "source_elapsed_ms": case["source_elapsed_ms"],
        "monolithic_progressive_elapsed_ms": round(elapsed_ms, 3),
        "candidate_count": len(case["candidates"]),
        "confirmation_keys": [],
        "worlds": 0,
        "shards": [],
        "monolithic_selected_key": None,
        "merged_selected_key": None,
        "progressive_selected_key": progressive_result.selected_key,
        "failures": failures,
    }
if __name__ == "__main__":
    raise SystemExit(main())
