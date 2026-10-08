"""Verify that confirmation sharding preserves every frozen world outcome."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.dual_discard_validator import (
    _balanced_world_shards,
    _merge_confirmation_shards,
    _root_search_task,
)
from ai.full_game_simulator import _discardable_labels
from ai.ismcts import public_view_from_dict
from ai.opponent_league import create_policy
from engine.rules import rules_for_room


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--state-id", action="append", required=True)
    parser.add_argument(
        "--candidate",
        default=(
            "professional_parallel_multi_opponent_"
            "production_anchored_rank1_sharded_research"
        ),
    )
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    cases = {
        str(event["public_view_id"]): {
            "players": int(row["players"]),
            "wildcard_enabled": bool(source.get("wildcard_enabled")),
            "public_view": event["public_view"],
        }
        for row in source.get("rows") or ()
        for event in row.get("candidate_discard_events") or ()
        if event.get("public_view_id") and event.get("public_view")
    }
    requested = tuple(dict.fromkeys(str(value) for value in args.state_id))
    missing = [state_id for state_id in requested if state_id not in cases]
    if missing:
        raise ValueError(f"sharded_equivalence_state_ids_missing:{len(missing)}")

    rows = [
        _verify_state(
            state_id,
            cases[state_id],
            candidate=args.candidate,
            shard_count=max(1, int(args.shards)),
        )
        for state_id in requested
    ]
    failures = [
        failure
        for row in rows
        for failure in row["failures"]
    ]
    report = {
        "ok": not failures,
        "schema_version": "exact-sharded-confirmation-equivalence-v1",
        "source_report": str(args.source_report),
        "candidate": args.candidate,
        "states": len(rows),
        "confirmation_seeds_per_state": 2,
        "shards_per_seed": max(1, int(args.shards)),
        "worlds_compared": sum(row["worlds_compared"] for row in rows),
        "failures": failures,
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
                "ok": report["ok"],
                "states": report["states"],
                "confirmation_seeds_per_state": report[
                    "confirmation_seeds_per_state"
                ],
                "shards_per_seed": report["shards_per_seed"],
                "worlds_compared": report["worlds_compared"],
                "failures": len(failures),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def _verify_state(
    state_id: str,
    case: dict[str, Any],
    *,
    candidate: str,
    shard_count: int,
) -> dict[str, Any]:
    policy = create_policy(candidate)
    view = public_view_from_dict(case["public_view"])
    rules = rules_for_room(
        wildcard_enabled=bool(case["wildcard_enabled"]),
        players=int(case["players"]),
    )
    decision = policy._choose(view, rules, legal_actions=[{"type": "DISCARD"}])
    legal = set(_discardable_labels(view.hand))
    by_label = {}
    for item in decision.action_evals:
        label = item.action.label
        if item.type != "DISCARD" or not label or label not in legal:
            continue
        previous = by_label.get(label)
        if previous is None or item.ev > previous.ev:
            by_label[label] = item
    ranked = sorted(
        by_label.values(),
        key=lambda item: (float(item.ev), str(item.action.label)),
        reverse=True,
    )
    labels = [str(item.action.label) for item in ranked]
    priors = {
        str(item.action.label): float(item.ev)
        for item in ranked
    }
    validator = policy.discard_validator
    screened = validator.screen_discard(
        view,
        rules=rules,
        candidate_labels=labels,
        candidate_priors=priors,
        absolute_deadline=None,
    )
    preferred = str(decision.selected_label)
    challenger = next(
        candidate_stats.label
        for candidate_stats in screened.coverage.candidates
        if candidate_stats.label != preferred
    )
    confirmation_labels = (preferred, challenger)
    config = validator.config
    seeds = (
        int(config.first_confirmation_seed),
        int(config.second_confirmation_seed),
    )
    worlds = int(config.confirmation_worlds)
    shared = {
        "view": view,
        "rules": rules,
        "labels": list(confirmation_labels),
        "priors": priors,
        "preferred": preferred,
        "time_budget_ms": int(config.time_budget_ms),
        "absolute_deadline": None,
        "rollout_max_turns": int(config.rollout_max_turns),
        "rollout_policy_factories": validator.rollout_policy_factories,
        "record_paired_worlds": True,
    }
    failures: list[str] = []
    seed_rows = []
    for seed in seeds:
        monolithic_started = time.perf_counter()
        monolithic = _root_search_task(
            {**shared, "worlds": worlds, "seed": seed}
        )
        monolithic_ms = (time.perf_counter() - monolithic_started) * 1000.0
        shard_ranges = _balanced_world_shards(worlds, shard_count)
        sharded_started = time.perf_counter()
        futures = [
            screened.executor.submit(
                _root_search_task,
                {
                    **shared,
                    "worlds": amount,
                    "paired_world_offset": offset,
                    "seed": seed,
                },
            )
            for offset, amount in shard_ranges
        ]
        parts = [future.result() for future in futures]
        sharded_ms = (time.perf_counter() - sharded_started) * 1000.0
        merged = _merge_confirmation_shards(
            parts,
            labels=confirmation_labels,
            priors=priors,
            preferred=preferred,
            expected_worlds=worlds,
            root_seat=int(view.seat),
            elapsed_ms=sharded_ms,
        )
        sharded_worlds = sorted(
            (
                world
                for part in parts
                for world in part.paired_worlds
            ),
            key=lambda world: world.world_index,
        )
        checks = {
            "worlds": tuple(monolithic.paired_worlds)
            == tuple(sharded_worlds),
            "selected_label": monolithic.selected_label
            == merged.selected_label,
            "candidate_statistics": monolithic.candidates == merged.candidates,
            "candidate_statistics_by_label": {
                item.label: asdict(item) for item in monolithic.candidates
            }
            == {item.label: asdict(item) for item in merged.candidates},
            "paired_advantages": monolithic.paired_advantages
            == merged.paired_advantages,
            "simulations": monolithic.simulations == merged.simulations,
            "paired_determinizations": (
                monolithic.paired_determinizations
                == merged.paired_determinizations
                == worlds
            ),
            "integrity": (
                monolithic.used_search
                and merged.used_search
                and monolithic.rollout_invariant_violations == 0
                and merged.rollout_invariant_violations == 0
                and monolithic.rollout_coverage_failures == 0
                and merged.rollout_coverage_failures == 0
            ),
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(f"{state_id}:{seed}:{name}")
        seed_rows.append(
            {
                "seed": seed,
                "worlds": worlds,
                "shard_ranges": [list(item) for item in shard_ranges],
                "monolithic_ms": round(monolithic_ms, 3),
                "sharded_ms": round(sharded_ms, 3),
                "checks": checks,
                "monolithic_selected_label": monolithic.selected_label,
                "sharded_selected_label": merged.selected_label,
                "monolithic_candidate_order": [
                    item.label for item in monolithic.candidates
                ],
                "sharded_candidate_order": [
                    item.label for item in merged.candidates
                ],
                "monolithic_candidates": [
                    asdict(item) for item in monolithic.candidates
                ],
                "sharded_candidates": [
                    asdict(item) for item in merged.candidates
                ],
                "first_world_difference": _first_world_difference(
                    monolithic.paired_worlds,
                    sharded_worlds,
                ),
            }
        )
    return {
        "public_view_id": state_id,
        "production_label": preferred,
        "challenger_label": challenger,
        "worlds_compared": worlds * len(seeds),
        "failures": failures,
        "seeds": seed_rows,
    }


def _first_world_difference(monolithic, sharded):
    for first, second in zip(monolithic, sharded):
        if first != second:
            return {
                "monolithic": asdict(first),
                "sharded": asdict(second),
            }
    if len(monolithic) != len(sharded):
        return {
            "monolithic_worlds": len(monolithic),
            "sharded_worlds": len(sharded),
        }
    return None


if __name__ == "__main__":
    raise SystemExit(main())
