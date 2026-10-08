"""Bound cross-round strategy memory without changing decision semantics."""

from __future__ import annotations

import sys
from typing import Any

from ai.features import (
    _quick_potential_cached,
    _quick_potential_counts_cached,
    _suit_presence_score,
    quick_potential_after_each_removal_from_counts,
)
from ai.opponent_proxy import _cached_quick_potential
from engine.hu_checker import _solve_grouping, _solve_grouping_counts


_ROUND_CACHES: tuple[Any, ...] = (
    _solve_grouping_counts,
    _solve_grouping,
    quick_potential_after_each_removal_from_counts,
    _quick_potential_cached,
    _quick_potential_counts_cached,
    _suit_presence_score,
    _cached_quick_potential,
)


def clear_round_strategy_caches() -> dict[str, int]:
    cleared_entries = 0
    for cached in _ROUND_CACHES:
        cleared_entries += int(cached.cache_info().currsize)
        cached.cache_clear()

    telemetry_events = 0
    frozen = sys.modules.get("ai.frozen_two_player_strategy")
    policy_factory: Any = getattr(frozen, "_frozen_policy", None) if frozen is not None else None
    if policy_factory is not None and policy_factory.cache_info().currsize:
        policy = policy_factory()
        for attribute in ("_discard_events", "_response_events"):
            events = getattr(policy, attribute, None)
            if isinstance(events, list):
                telemetry_events += len(events)
                events.clear()
    return {
        "cleared_cache_entries": cleared_entries,
        "cleared_telemetry_events": telemetry_events,
    }


__all__ = ["clear_round_strategy_caches"]
