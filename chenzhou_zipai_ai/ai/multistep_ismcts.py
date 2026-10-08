"""Bounded multi-step information-set continuation research policy."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Sequence

from ai.full_game_simulator import (
    InformationSetSearchPolicy,
    PublicView,
    SimulationPolicy,
    _discardable_labels,
)
from ai.ismcts import RootISMCTSConfig, RootISMCTSPolicy, RootSearchResult


class BoundedMultiStepInformationSetPolicy(InformationSetSearchPolicy):
    """Re-search the root player's later discards to a bounded decision depth."""

    name = "bounded_multistep_ismcts_v1"

    def __init__(
        self,
        *,
        decision_depth: int = 2,
        root_config: RootISMCTSConfig | None = None,
        opponent_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
    ) -> None:
        factories = tuple(opponent_policy_factories)
        if not factories:
            raise ValueError("multistep_requires_opponent_factories")
        self.decision_depth = max(0, int(decision_depth))
        self.root_config = root_config or RootISMCTSConfig(
            time_budget_ms=1_500,
            max_iterations=24,
            max_candidates=4,
            skip_search_gap=math.inf,
            rollout_max_turns=80,
            require_confident_override=False,
        )
        self.opponent_policy_factories = factories
        self.last_search: RootSearchResult | None = None

    def _next_self_policy(self) -> SimulationPolicy:
        return BoundedMultiStepInformationSetPolicy(
            decision_depth=max(0, self.decision_depth - 1),
            root_config=self.root_config,
            opponent_policy_factories=self.opponent_policy_factories,
        )

    def choose_discard(self, view: PublicView, rules: dict) -> str:
        preferred = super().choose_discard(view, rules)
        if self.decision_depth <= 0:
            self.last_search = None
            return preferred
        labels = _discardable_labels(view.hand)
        if len(labels) < 2:
            self.last_search = None
            return preferred
        search = RootISMCTSPolicy(
            replace(
                self.root_config,
                max_candidates=max(
                    2,
                    min(len(labels), self.root_config.max_candidates),
                ),
            ),
            rollout_policy_factories=self.opponent_policy_factories,
            root_continuation_policy_factory=self._next_self_policy,
        )
        self.last_search = search.search_discard(
            view,
            rules=rules,
            candidate_labels=labels,
            force_search=True,
            paired_candidates=True,
            preferred_label=preferred,
        )
        return self.last_search.selected_label


__all__ = ["BoundedMultiStepInformationSetPolicy"]
