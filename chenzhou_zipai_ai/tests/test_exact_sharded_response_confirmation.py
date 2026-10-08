from __future__ import annotations

from dataclasses import replace

from ai.full_game_simulator import BaselinePolicy, PublicView
from ai.ismcts import (
    ProgressiveRootISMCTSPolicy,
    RootISMCTSConfig,
    RootISMCTSPolicy,
    RootResponseCandidate,
)
from ai.parallel_response_search import (
    ExactShardedProgressiveResponsePolicy,
    _merge_response_confirmation_shards,
    _response_search_task,
)
from engine.deck import full_deck_counts
from engine.rules import rules_for_room


def test_exact_response_shards_match_monolithic_confirmation() -> None:
    view, rules = _response_public_view()
    candidates = (
        RootResponseCandidate("PASS", "PASS", 100.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=200.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )
    config = RootISMCTSConfig(
        time_budget_ms=30_000,
        max_iterations=8,
        max_candidates=2,
        rollout_max_turns=8,
        require_confident_override=True,
        min_confidence_pairs=2,
        minimum_confident_advantage=0.02,
        complete_first_paired_batch=True,
        require_complete_iteration_budget_for_override=True,
        record_paired_worlds=True,
        seed=20260803,
    )
    monolithic = RootISMCTSPolicy(
        config,
        rollout_policy_factories=(BaselinePolicy,),
    ).search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
        preferred_key="PASS",
    )
    shards = [
        _response_search_task(
            {
                "view": view,
                "rules": rules,
                "candidates": candidates,
                "preferred_key": "PASS",
                "base_config": config,
                "rollout_policy_factories": (BaselinePolicy,),
                "worlds": 2,
                "paired_world_offset": offset,
            }
        )
        for offset in (0, 2)
    ]
    merged = _merge_response_confirmation_shards(
        shards,
        candidates=candidates,
        preferred_key="PASS",
        config=config,
        expected_worlds=4,
        root_seat=view.seat,
        elapsed_ms=monolithic.elapsed_ms,
    )

    assert replace(merged, elapsed_ms=0.0) == replace(
        monolithic,
        elapsed_ms=0.0,
    )


def test_exact_response_wrapper_shards_coverage_stage_without_drift() -> None:
    view, rules = _response_public_view()
    candidates = (
        RootResponseCandidate("PASS", "PASS", 100.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=200.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )
    config = RootISMCTSConfig(
        time_budget_ms=30_000,
        max_iterations=8,
        max_candidates=2,
        rollout_max_turns=8,
        require_confident_override=False,
        complete_first_paired_batch=True,
        record_paired_worlds=True,
        seed=20260803,
    )
    monolithic = RootISMCTSPolicy(
        config,
        rollout_policy_factories=(BaselinePolicy,),
    ).search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
        preferred_key="PASS",
    )
    base = ProgressiveRootISMCTSPolicy(
        coverage_config=config,
        rollout_policy_factories=(BaselinePolicy,),
    )
    wrapper = ExactShardedProgressiveResponsePolicy(
        base,
        parallel_shards=2,
        parallel_workers=2,
    )

    sharded = wrapper._sharded_confirmation(
        view,
        rules=rules,
        candidates=candidates,
        preferred_key="PASS",
        stage=base.coverage,
        absolute_deadline=None,
    )

    assert replace(sharded, elapsed_ms=0.0) == replace(
        monolithic,
        elapsed_ms=0.0,
    )


def _response_public_view() -> tuple[PublicView, dict]:
    rules = rules_for_room(wildcard_enabled=False, players=3)
    hand = ("二", "二", "四", "六", "九")
    pending = "二"
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(hand)
    remaining.subtract((pending,))
    opponent_size = 5
    stock_count = sum(remaining.values()) - opponent_size * 2
    return (
        PublicView(
            seat=0,
            hand=hand,
            own_melds=(),
            all_melds=((), (), ()),
            discards=((), (), ()),
            remaining_counts=tuple(sorted((+remaining).items())),
            stock_count=stock_count,
            hand_sizes=(len(hand), opponent_size, opponent_size),
            pending_card=pending,
            pending_source_seat=1,
        ),
        rules,
    )
