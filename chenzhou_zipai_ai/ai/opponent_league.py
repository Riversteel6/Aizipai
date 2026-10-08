"""Reproducible multi-style opponent league for production strategy audits."""

from __future__ import annotations

import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from statistics import mean
from typing import Any, Callable, Mapping

from audit.independent_opponent import (
    IndependentBalancedPolicy,
    IndependentDenialPolicy,
    IndependentExploitPolicy,
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
    IndependentPressurePolicy,
)
from ai.features import quick_potential
from ai.full_game_simulator import (
    BaselinePolicy,
    ChiPlan,
    FullGameSimulator,
    InformationSetSearchPolicy,
    ProfessionalBrainSimulationPolicy,
    PublicView,
    SimMeld,
    SimulationPolicy,
    _chi_kind,
    _discardable_labels,
    _position_value,
    _production_state_from_public_view,
    _public_discard_danger,
    _remove_many,
    _remove_one,
    _replace_view,
)
from ai.ismcts import (
    ProfessionalConfidenceRootCandidatePolicy,
    ProfessionalConfidenceRootShadowPolicy,
    ProfessionalDiscardResponseRiskCandidatePolicy,
    ProfessionalDiscardResponseRiskShadowPolicy,
    ProfessionalDiscardShadowSimulationPolicy,
    ProfessionalFullActionTeacherPolicy,
    ProfessionalProgressiveAllActionCandidatePolicy,
    ProfessionalResponseShadowSimulationPolicy,
    ProfessionalSearchSimulationPolicy,
    RootISMCTSPolicy,
)
from ai.dual_validated_candidate import (
    ProfessionalParallelMultiCalibratedCandidateV4Policy,
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy,
    ProfessionalParallelMultiOpponentRobustV82ResearchPolicy,
    ProfessionalParallelMultiOpponentRobustV83ResearchPolicy,
    ProfessionalParallelMultiOpponentRobustV8ResearchPolicy,
    ProfessionalParallelMultiOpponentProductionAnchoredRank1ResearchPolicy,
    ProfessionalParallelMultiOpponentProductionAnchoredRank1ShardedResearchPolicy,
    ProfessionalParallelMultiOpponentProductionAnchoredRank1ShardedResponseResearchPolicy,
    ProfessionalParallelMultiOpponentUnifiedGateResearchPolicy,
    ProfessionalParallelMultiOpponentUnifiedGateRank1ResearchPolicy,
    ProfessionalParallelMultiSearchProxyResearchPolicy,
    ProfessionalParallelMultiSparseProxyCalibratedV5ResearchPolicy,
    ProfessionalParallelMultiSparseProxyCalibratedV6ResearchPolicy,
    ProfessionalParallelMultiSparseProxyCalibratedV7ResearchPolicy,
    ProfessionalParallelMultiSparseProxyResearchPolicy,
    ProfessionalParallelMultiValidatedCandidatePolicy,
    ProfessionalParallelDualValidatedCandidatePolicy,
    ProfessionalV81TwoPlayerResponseGate014ResearchPolicy,
    ProfessionalV81TwoPlayerExactDiscardShardedResearchPolicy,
    ProfessionalV81TwoPlayerWangTingGuardResearchPolicy,
)
from ai.opponent_proxy import FastInformationSetProxyPolicy
from engine.cards import RED_LABELS, WILD_LABEL
from engine.red_black_rules import red_black_target_distance
from engine.rules import rules_for_room


_NESTED_SEARCH_WORKERS = {
    "professional_parallel_dual_validated_candidate": 4,
    "professional_parallel_multi_validated_candidate": 10,
    "professional_parallel_multi_calibrated_candidate_v4": 10,
    "professional_parallel_multi_validated_candidate_v3_2": 10,
    "professional_parallel_multi_search_proxy_research": 10,
    "professional_parallel_multi_sparse_proxy_research": 10,
    "professional_parallel_multi_sparse_proxy_calibrated_v5_research": 10,
    "professional_parallel_multi_sparse_proxy_calibrated_v6_research": 10,
    "professional_parallel_multi_sparse_proxy_calibrated_v7_research": 10,
    "professional_parallel_multi_opponent_robust_v8_research": 10,
    "professional_parallel_multi_opponent_robust_v8_1_research": 10,
    "professional_v81_two_player_response_gate_014_research": 10,
    "professional_v81_two_player_wang_ting_guard_research": 10,
    # Empirical process-tree measurements on the 32-thread acceptance host
    # show this candidate averaging about 9 logical CPUs while its shared
    # pools are alive. Budget it as 15 inner workers so two independent league
    # games can run concurrently without changing either game's search config.
    "professional_v81_two_player_exact_discard_sharded_research": 15,
    "professional_parallel_multi_opponent_robust_v8_2_research": 10,
    "professional_parallel_multi_opponent_robust_v8_3_research": 10,
    "professional_parallel_multi_opponent_unified_gate_research": 10,
    "professional_parallel_multi_opponent_unified_gate_rank1_research": 10,
    "professional_parallel_multi_opponent_production_anchored_rank1_research": 10,
    "professional_parallel_multi_opponent_production_anchored_rank1_sharded_research": 10,
}

_FROZEN_PRODUCT_CANDIDATE = (
    "professional_v81_two_player_exact_discard_sharded_research"
)


class ProductDecisionKernelSimulationPolicy(ProfessionalBrainSimulationPolicy):
    """Simulator adapter that calls the exact live product kernel."""

    name = "product_decision_kernel_v1"
    seat_aware_opponents = True

    def __init__(self) -> None:
        super().__init__()
        self._product_discard_events: list[dict[str, Any]] = []
        self._product_response_events: list[dict[str, Any]] = []

    def _choose(
        self,
        view: PublicView,
        rules: dict[str, Any],
        *,
        legal_actions: list[dict[str, str]],
        pending_card: str | None = None,
        chi_options: list[dict[str, object]] | None = None,
    ) -> Any:
        from ai.frozen_two_player_strategy import (
            PRODUCT_DECISION_KERNEL,
            _frozen_policy,
        )

        raw_policy = _frozen_policy()
        before_discard = len(raw_policy.discard_events())
        before_response = len(raw_policy.response_events())
        state = _production_state_from_public_view(
            view,
            legal_actions=legal_actions,
            pending_card=pending_card,
            chi_options=chi_options,
            seat_aware_opponents=True,
        )
        decision = PRODUCT_DECISION_KERNEL.choose_action(state, rules=rules)
        self._product_discard_events.extend(
            raw_policy.discard_events()[before_discard:]
        )
        self._product_response_events.extend(
            raw_policy.response_events()[before_response:]
        )
        return decision

    def choose_response(
        self,
        view: PublicView,
        legal_actions: tuple[Any, ...],
        plans: tuple[ChiPlan, ...],
        hu: Any | None,
        rules: dict[str, Any],
    ) -> str:
        legal_types = tuple(dict.fromkeys(str(action.type).upper() for action in legal_actions))
        option_groups: list[tuple[str, str, str]] = []
        for plan in plans:
            if plan.initial_group not in option_groups:
                option_groups.append(plan.initial_group)
        chi_options = [
            {
                "option_id": f"sim_chi_{index:03d}",
                "labels": list(group),
                "confidence": 1.0,
            }
            for index, group in enumerate(option_groups, start=1)
        ]
        decision = self._choose(
            view,
            rules,
            legal_actions=[{"type": action_type} for action_type in legal_types],
            pending_card=view.pending_card,
            chi_options=chi_options or None,
        )
        self._remember_decision(decision)
        selected_type = str(decision.selected_action).upper()
        if selected_type == "CHI":
            selected_plan = self._product_chi_plan_for_decision(
                decision,
                list(plans),
            )
            for action in legal_actions:
                if (
                    str(action.type).upper() == "CHI"
                    and tuple(action.consumed_from_hand)
                    == tuple(selected_plan.consumed_from_hand)
                    and tuple(action.meld_groups) == tuple(selected_plan.groups)
                ):
                    return str(action.key)
            raise ValueError("product_kernel_chi_plan_not_legal")
        for action in legal_actions:
            if str(action.type).upper() == selected_type:
                return str(action.key)
        raise ValueError(
            f"product_kernel_response_not_legal:{selected_type}"
        )

    @staticmethod
    def _product_chi_plan_for_decision(
        decision: Any,
        plans: list[ChiPlan],
    ) -> ChiPlan:
        """Map a product CHI even when production's strategy gate rejected it.

        Product search is allowed to override a production preference such as
        ``chi_ev_not_enough``.  The offline adapter must therefore use the
        selected option's exact card plan, not the baseline ``allowed`` bit.
        Legality is checked again against the simulator's legal TraceAction.
        """

        selected_eval = next(
            (
                item
                for item in decision.action_evals
                if item.action.type == "CHI"
                and item.action.option_id == decision.selected_option_id
            ),
            None,
        )
        if selected_eval is None:
            raise ValueError("product_kernel_chi_eval_missing")
        consumed = Counter(selected_eval.debug_details.get("consumed_from_hand") or ())
        initial = Counter(selected_eval.debug_details.get("meld_cards") or ())
        compares = tuple(
            sorted(
                tuple(sorted(group))
                for group in selected_eval.debug_details.get("compare_groups") or ()
            )
        )
        for plan in plans:
            plan_compares = tuple(
                sorted(tuple(sorted(group)) for group in plan.compare_groups)
            )
            if (
                Counter(plan.initial_group) == initial
                and Counter(plan.consumed_from_hand) == consumed
                and plan_compares == compares
            ):
                return plan
        raise ValueError("product_kernel_chi_plan_mapping_failed")

    def discard_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._product_discard_events)

    def response_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._product_response_events)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "formal_route": "product_decision_kernel",
            "product_discard_events": len(self._product_discard_events),
            "product_response_events": len(self._product_response_events),
        }


class AggressiveMeldPolicy(InformationSetSearchPolicy):
    """Pressure policy that claims legal melds whenever shape remains playable."""

    name = "aggressive_meld"

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        return view.hand.count(label) == 2 and bool(_discardable_labels(_remove_many(view.hand, [label, label])))

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        scored: list[tuple[float, ChiPlan]] = []
        for plan in plans:
            after = _remove_many(view.hand, plan.consumed_from_hand)
            if not _discardable_labels(after):
                continue
            meld_bonus = sum(
                18.0 if _chi_kind(group) in {"special_123", "special_2710"} else 5.0
                for group in plan.groups
            )
            scored.append((quick_potential(after) * 12.0 + meld_bonus, plan))
        return max(scored, key=lambda item: item[0])[1] if scored else None


class ConservativeSearchPolicy(InformationSetSearchPolicy):
    """Search policy that demands a larger margin before exposing the hand."""

    name = "conservative_search"
    peng_margin = 65.0
    chi_margin = 80.0

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        if view.hand.count(label) != 2:
            return False
        pass_value = _position_value(view.hand, view.own_melds, view.remaining_counts, rules)
        hand_after = tuple(_remove_many(view.hand, [label, label]))
        melds = (*view.own_melds, SimMeld("peng", (label, label, label)))
        claim_view = _replace_view(view, hand=hand_after, own_melds=melds)
        try:
            discard = self.choose_discard(claim_view, rules)
        except ValueError:
            return False
        claim_value = _position_value(
            tuple(_remove_one(hand_after, discard)),
            melds,
            view.remaining_counts,
            rules,
        )
        return claim_value > pass_value + self.peng_margin

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        pass_value = _position_value(view.hand, view.own_melds, view.remaining_counts, rules)
        best: tuple[float, ChiPlan] | None = None
        for plan in plans:
            hand_after = tuple(_remove_many(view.hand, plan.consumed_from_hand))
            melds = (
                *view.own_melds,
                *(SimMeld(_chi_kind(group), tuple(group)) for group in plan.groups),
            )
            claim_view = _replace_view(view, hand=hand_after, own_melds=melds)
            try:
                discard = self.choose_discard(claim_view, rules)
            except ValueError:
                continue
            value = _position_value(
                tuple(_remove_one(hand_after, discard)),
                melds,
                view.remaining_counts,
                rules,
            )
            if best is None or value > best[0]:
                best = value, plan
        return best[1] if best is not None and best[0] > pass_value + self.chi_margin else None


class DefensiveSearchPolicy(InformationSetSearchPolicy):
    """Search policy that overweights public discard danger."""

    name = "defensive_search"

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        labels = _discardable_labels(view.hand)
        if not labels:
            raise ValueError("no_discardable_card")
        return max(
            labels,
            key=lambda label: (
                self._discard_value(view, label, rules) - _public_discard_danger(label, view) * 2.5,
                label,
            ),
        )


class RedBlackSearchPolicy(InformationSetSearchPolicy):
    """Search policy that commits harder to a visible red or black route."""

    name = "red_black_search"

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        labels = _discardable_labels(view.hand)
        if not labels:
            raise ValueError("no_discardable_card")
        current_red = sum(label in RED_LABELS for label in view.hand)
        chase_red = current_red >= 7

        def route_value(label: str) -> float:
            after_red = current_red - int(label in RED_LABELS)
            if chase_red:
                route_bonus = after_red * 18.0
            else:
                route_bonus = -min(abs(after_red), abs(after_red - 1)) * 22.0
            return self._discard_value(view, label, rules) + route_bonus

        return max(labels, key=lambda label: (route_value(label), label))


class _WildcardRouteProfessionalPolicy(ProfessionalBrainSimulationPolicy):
    route_bonus_per_step = 0.0
    require_wildcard_in_hand = False

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        decision = self._choose(view, rules, legal_actions=[{"type": "DISCARD"}])
        production_label = decision.selected_label
        if decision.selected_action != "DISCARD" or not production_label:
            return super().choose_discard(view, rules)
        if (
            not rules.get("wildcard", {}).get("enabled", False)
            or self.require_wildcard_in_hand
            and WILD_LABEL not in view.hand
        ):
            return production_label
        candidates = [
            item
            for item in decision.action_evals
            if item.type == "DISCARD" and item.allowed and item.label
        ]
        if not candidates:
            return production_label
        selected = max(
            candidates,
            key=lambda item: (
                item.ev
                + _wildcard_route_bonus(
                    view,
                    str(item.label),
                    rules,
                    per_step=self.route_bonus_per_step,
                ),
                int(item.label == production_label),
                item.ev,
                str(item.label),
            ),
        )
        return str(selected.label)


class WildcardRouteLightPolicy(_WildcardRouteProfessionalPolicy):
    """Offline candidate that modestly values reachable red/black outcomes."""

    name = "professional_wang_route_light"
    route_bonus_per_step = 30.0


class WildcardRouteFocusPolicy(_WildcardRouteProfessionalPolicy):
    """Offline candidate that becomes route-focused only while holding Wang."""

    name = "professional_wang_route_focus"
    route_bonus_per_step = 60.0
    require_wildcard_in_hand = True


class _WildcardUtilizationProfessionalPolicy(ProfessionalBrainSimulationPolicy):
    max_ev_gap = 0.0
    require_wildcard_in_hand = True

    def __init__(self) -> None:
        self.utilization_opportunities = 0
        self.utilization_overrides = 0

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        decision = self._choose(view, rules, legal_actions=[{"type": "DISCARD"}])
        production_label = decision.selected_label
        if decision.selected_action != "DISCARD" or not production_label:
            return super().choose_discard(view, rules)
        if (
            not rules.get("wildcard", {}).get("enabled", False)
            or WILD_LABEL not in view.hand
        ):
            return production_label
        self.utilization_opportunities += 1
        candidates = [
            item
            for item in decision.action_evals
            if item.type == "DISCARD" and item.allowed and item.label
        ]
        production_evals = [
            item for item in candidates if item.label == production_label
        ]
        if not production_evals:
            return production_label
        production_eval = max(production_evals, key=lambda item: item.ev)
        near_ties = [
            item
            for item in candidates
            if item.ev >= production_eval.ev - max(0.0, float(self.max_ev_gap))
        ]
        selected = max(
            near_ties,
            key=lambda item: (
                _wildcard_utilization_quality(item),
                item.ev,
                int(item.label == production_label),
                str(item.label),
            ),
        )
        self.utilization_overrides += int(selected.label != production_label)
        return str(selected.label)

    def diagnostics(self) -> dict[str, int]:
        return {
            "wang_utilization_opportunities": self.utilization_opportunities,
            "wang_utilization_overrides": self.utilization_overrides,
        }


class WildcardUtilizationNarrowPolicy(_WildcardUtilizationProfessionalPolicy):
    """Offline candidate using Wang quality only inside a 20-EV near tie."""

    name = "professional_wang_util_narrow"
    max_ev_gap = 20.0


class WildcardUtilizationBalancedPolicy(_WildcardUtilizationProfessionalPolicy):
    """Offline candidate allowing Wang quality inside a 60-EV near tie."""

    name = "professional_wang_util_balanced"
    max_ev_gap = 60.0


class SeatAwareOpponentProfessionalPolicy(ProfessionalBrainSimulationPolicy):
    """Offline candidate that keeps each public opponent in its own seat."""

    name = "professional_seat_aware_opponents"
    seat_aware_opponents = True


def _wildcard_utilization_quality(action_eval: Any) -> tuple[int, ...]:
    debug = action_eval.debug_details or {}
    simulation = debug.get("simulation") or {}
    allocation = simulation.get("allocation_after") or {}
    weak_potentials = allocation.get("weak_potentials") or []
    option_keys: set[tuple[str, tuple[str, ...], str]] = set()
    waits: set[str] = set()
    special_options = 0
    for potential in weak_potentials:
        waiting_for = str(potential.get("waiting_for") or "")
        reason = str(potential.get("reason") or "")
        labels = tuple(sorted(str(label) for label in potential.get("labels") or []))
        key = (waiting_for, labels, reason)
        if key in option_keys:
            continue
        option_keys.add(key)
        if waiting_for:
            waits.add(waiting_for)
        if "2710" in reason or "二七十" in reason or "123" in reason or "一二三" in reason:
            special_options += 1
    ting = debug.get("ting_after") or {}
    information = debug.get("information_set") or {}
    return (
        int(bool(ting.get("is_ting"))),
        int(information.get("hu_outs") or 0),
        int(ting.get("expected_xi_if_hu") or 0),
        special_options,
        len(waits),
        len(option_keys),
        int(information.get("improving_outs") or 0),
    )


def _wildcard_route_bonus(
    view: PublicView,
    discard: str,
    rules: dict[str, Any],
    *,
    per_step: float,
) -> float:
    mode = str(rules.get("rules", {}).get("red_black_mode", "")).lower()
    if mode in {"none", "off", "disabled"}:
        return 0.0
    hand_after = list(view.hand)
    hand_after.remove(discard)
    meld_red = sum(
        label in RED_LABELS
        for meld in view.own_melds
        for label in meld.cards
    )
    hand_counts = Counter(hand_after)
    locked_red = meld_red + sum(
        amount
        for label, amount in hand_counts.items()
        if label in RED_LABELS and amount >= 3
    )
    red_count = meld_red + sum(label in RED_LABELS for label in hand_after)
    distances = red_black_target_distance(red_count, locked_red, rules)
    nearest = min(distances.values(), default=99)
    return max(0.0, 3.0 - float(nearest)) * max(0.0, float(per_step))


_OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES = (
    IndependentFastRolloutPolicy,
    IndependentFastPressurePolicy,
    IndependentFastDenialPolicy,
    IndependentFastRolloutPolicy,
    IndependentFastPressurePolicy,
    IndependentFastDenialPolicy,
    IndependentFastRolloutPolicy,
    IndependentFastPressurePolicy,
    IndependentFastDenialPolicy,
    IndependentFastRolloutPolicy,
    IndependentFastPressurePolicy,
    IndependentFastDenialPolicy,
    IndependentFastRolloutPolicy,
    IndependentFastPressurePolicy,
    IndependentFastDenialPolicy,
    IndependentFastRolloutPolicy,
    IndependentFastPressurePolicy,
    IndependentFastDenialPolicy,
    FastInformationSetProxyPolicy,
    FastInformationSetProxyPolicy,
    FastInformationSetProxyPolicy,
)


POLICY_FACTORIES: dict[str, Callable[[], SimulationPolicy]] = {
    "baseline": BaselinePolicy,
    "information_set_search": InformationSetSearchPolicy,
    "professional_brain": ProfessionalBrainSimulationPolicy,
    "aggressive_meld": AggressiveMeldPolicy,
    "conservative_search": ConservativeSearchPolicy,
    "defensive_search": DefensiveSearchPolicy,
    "red_black_search": RedBlackSearchPolicy,
    "root_ismcts": RootISMCTSPolicy,
    "professional_search": ProfessionalSearchSimulationPolicy,
    "professional_response_shadow": ProfessionalResponseShadowSimulationPolicy,
    "professional_discard_shadow": ProfessionalDiscardShadowSimulationPolicy,
    "professional_discard_risk_shadow": ProfessionalDiscardResponseRiskShadowPolicy,
    "professional_discard_risk_candidate": ProfessionalDiscardResponseRiskCandidatePolicy,
    "professional_confidence_root_shadow": lambda: ProfessionalConfidenceRootShadowPolicy(
        rollout_policy_factories=(
            IndependentFastRolloutPolicy,
            IndependentFastPressurePolicy,
            IndependentFastDenialPolicy,
        )
    ),
    "professional_confidence_root_candidate": lambda: ProfessionalConfidenceRootCandidatePolicy(
        rollout_policy_factories=(
            IndependentFastRolloutPolicy,
            IndependentFastPressurePolicy,
            IndependentFastDenialPolicy,
        )
    ),
    "professional_full_action_teacher": lambda: ProfessionalFullActionTeacherPolicy(
        rollout_policy_factories=(
            IndependentFastRolloutPolicy,
            IndependentFastPressurePolicy,
            IndependentFastDenialPolicy,
        )
    ),
    "professional_progressive_all_action_candidate": lambda: (
        ProfessionalProgressiveAllActionCandidatePolicy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
            )
        )
    ),
    "professional_parallel_dual_validated_candidate": lambda: (
        ProfessionalParallelDualValidatedCandidatePolicy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
            )
        )
    ),
    "professional_parallel_multi_validated_candidate": lambda: (
        ProfessionalParallelMultiCalibratedCandidateV4Policy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
            )
        )
    ),
    "professional_parallel_multi_calibrated_candidate_v4": lambda: (
        ProfessionalParallelMultiCalibratedCandidateV4Policy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
            )
        )
    ),
    "professional_parallel_multi_validated_candidate_v3_2": lambda: (
        ProfessionalParallelMultiValidatedCandidatePolicy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
            )
        )
    ),
    "professional_parallel_multi_search_proxy_research": lambda: (
        ProfessionalParallelMultiSearchProxyResearchPolicy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                FastInformationSetProxyPolicy,
            )
        )
    ),
    "professional_parallel_multi_sparse_proxy_research": lambda: (
        ProfessionalParallelMultiSparseProxyResearchPolicy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                FastInformationSetProxyPolicy,
            )
        )
    ),
    "professional_parallel_multi_sparse_proxy_calibrated_v5_research": (
        lambda: ProfessionalParallelMultiSparseProxyCalibratedV5ResearchPolicy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                FastInformationSetProxyPolicy,
            )
        )
    ),
    "professional_parallel_multi_sparse_proxy_calibrated_v6_research": (
        lambda: ProfessionalParallelMultiSparseProxyCalibratedV6ResearchPolicy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                FastInformationSetProxyPolicy,
            )
        )
    ),
    "professional_parallel_multi_sparse_proxy_calibrated_v7_research": (
        lambda: ProfessionalParallelMultiSparseProxyCalibratedV7ResearchPolicy(
            rollout_policy_factories=(
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                IndependentFastRolloutPolicy,
                IndependentFastPressurePolicy,
                IndependentFastDenialPolicy,
                FastInformationSetProxyPolicy,
            )
        )
    ),
    "professional_parallel_multi_opponent_robust_v8_research": (
        lambda: ProfessionalParallelMultiOpponentRobustV8ResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_parallel_multi_opponent_robust_v8_1_research": (
        lambda: ProfessionalParallelMultiOpponentRobustV81ResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_v81_two_player_response_gate_014_research": (
        lambda: ProfessionalV81TwoPlayerResponseGate014ResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_v81_two_player_wang_ting_guard_research": (
        lambda: ProfessionalV81TwoPlayerWangTingGuardResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_v81_two_player_exact_discard_sharded_research": (
        lambda: ProfessionalV81TwoPlayerExactDiscardShardedResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_parallel_multi_opponent_robust_v8_2_research": (
        lambda: ProfessionalParallelMultiOpponentRobustV82ResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_parallel_multi_opponent_robust_v8_3_research": (
        lambda: ProfessionalParallelMultiOpponentRobustV83ResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_parallel_multi_opponent_unified_gate_research": (
        lambda: ProfessionalParallelMultiOpponentUnifiedGateResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_parallel_multi_opponent_unified_gate_rank1_research": (
        lambda: ProfessionalParallelMultiOpponentUnifiedGateRank1ResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_parallel_multi_opponent_production_anchored_rank1_research": (
        lambda: ProfessionalParallelMultiOpponentProductionAnchoredRank1ResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_parallel_multi_opponent_production_anchored_rank1_sharded_research": (
        lambda: ProfessionalParallelMultiOpponentProductionAnchoredRank1ShardedResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_parallel_multi_opponent_production_anchored_rank1_sharded_response_research": (
        lambda: ProfessionalParallelMultiOpponentProductionAnchoredRank1ShardedResponseResearchPolicy(
            rollout_policy_factories=(
                _OPPONENT_ROBUST_ROLLOUT_POLICY_FACTORIES
            )
        )
    ),
    "professional_wang_route_light": WildcardRouteLightPolicy,
    "professional_wang_route_focus": WildcardRouteFocusPolicy,
    "professional_wang_util_narrow": WildcardUtilizationNarrowPolicy,
    "professional_wang_util_balanced": WildcardUtilizationBalancedPolicy,
    "professional_seat_aware_opponents": SeatAwareOpponentProfessionalPolicy,
    "independent_balanced": IndependentBalancedPolicy,
    "independent_pressure": IndependentPressurePolicy,
    "independent_denial": IndependentDenialPolicy,
    "independent_exploit_untrained": IndependentExploitPolicy,
    "independent_fast_rollout": IndependentFastRolloutPolicy,
    "independent_fast_pressure": IndependentFastPressurePolicy,
    "independent_fast_denial": IndependentFastDenialPolicy,
}

DEFAULT_MATCHUPS: tuple[tuple[str, str], ...] = (
    ("independent_balanced", "independent_pressure"),
    ("independent_denial", "information_set_search"),
    ("aggressive_meld", "defensive_search"),
    ("red_black_search", "information_set_search"),
)

DEFAULT_HEADS_UP_MATCHUPS: tuple[tuple[str], ...] = (
    ("independent_balanced",),
    ("independent_pressure",),
    ("independent_denial",),
    ("information_set_search",),
    ("aggressive_meld",),
    ("defensive_search",),
    ("red_black_search",),
)


@dataclass(frozen=True)
class LeagueJob:
    candidate: str
    opponents: tuple[str, ...]
    candidate_seat: int
    dealer: int
    seed: int
    wildcard_enabled: bool
    players: int
    record_decisions: bool = False


def create_policy(name: str) -> SimulationPolicy:
    try:
        return POLICY_FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(f"unknown league policy: {name}") from exc


def build_league_jobs(
    *,
    candidate: str,
    matchups: tuple[tuple[str, ...], ...],
    deals_per_matchup: int,
    wildcard_enabled: bool,
    seed: int,
    players: int = 3,
    record_decisions: bool = False,
) -> list[LeagueJob]:
    if players not in {2, 3}:
        raise ValueError("players must be 2 or 3")
    jobs: list[LeagueJob] = []
    for matchup_index, matchup in enumerate(matchups):
        if len(matchup) != players - 1:
            raise ValueError(
                f"matchup requires {players - 1} opponents for {players} players: {matchup}"
            )
        for deal_index in range(deals_per_matchup):
            game_seed = seed + matchup_index * 100_000 + deal_index
            for candidate_seat in range(players):
                for dealer in range(players):
                    rotation = (candidate_seat + dealer + deal_index) % len(matchup)
                    ordered_opponents = matchup[rotation:] + matchup[:rotation]
                    jobs.append(
                        LeagueJob(
                            candidate=candidate,
                            opponents=ordered_opponents,
                            candidate_seat=candidate_seat,
                            dealer=dealer,
                            seed=game_seed,
                            wildcard_enabled=wildcard_enabled,
                            players=players,
                            record_decisions=record_decisions,
                        )
                    )
    return jobs


def run_opponent_league(
    *,
    candidate: str = "professional_brain",
    matchups: tuple[tuple[str, ...], ...] | None = None,
    deals_per_matchup: int = 1,
    wildcard_enabled: bool,
    seed: int = 20260726,
    workers: int = 1,
    players: int = 3,
    record_decisions: bool = False,
    balanced_rotation_only: bool = False,
    completed_rows: list[dict[str, Any]] | None = None,
    progress_callback: (
        Callable[[int, int, dict[str, Any]], None] | None
    ) = None,
    early_stop_callback: (
        Callable[
            [list[dict[str, Any]], int],
            Mapping[str, Any] | None,
        ]
        | None
    ) = None,
) -> dict[str, Any]:
    if candidate not in POLICY_FACTORIES:
        raise ValueError(f"unknown candidate policy: {candidate}")
    if players not in {2, 3}:
        raise ValueError("players must be 2 or 3")
    if deals_per_matchup < 1:
        raise ValueError("deals_per_matchup must be positive")
    if matchups is None:
        matchups = DEFAULT_HEADS_UP_MATCHUPS if players == 2 else DEFAULT_MATCHUPS
    for matchup in matchups:
        if len(matchup) != players - 1 or any(name not in POLICY_FACTORIES for name in matchup):
            raise ValueError(f"invalid matchup: {matchup}")

    jobs = build_league_jobs(
        candidate=candidate,
        matchups=matchups,
        deals_per_matchup=deals_per_matchup,
        wildcard_enabled=wildcard_enabled,
        seed=seed,
        players=players,
        record_decisions=record_decisions,
    )
    if balanced_rotation_only:
        jobs = [
            job
            for job in jobs
            if _is_balanced_league_job(job)
        ]
    resumed_rows = list(completed_rows or [])
    _validate_completed_league_prefix(
        jobs,
        resumed_rows,
        record_decisions=record_decisions,
    )
    resumed_count = len(resumed_rows)
    remaining_jobs = jobs[resumed_count:]
    rows = list(resumed_rows)
    early_stop = (
        early_stop_callback(rows, len(jobs))
        if rows and early_stop_callback is not None
        else None
    )
    effective_workers = _effective_league_workers(
        candidate=candidate,
        requested=workers,
        remaining_jobs=len(remaining_jobs),
    )
    if workers > 1 and not early_stop:
        if remaining_jobs:
            pool = ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=get_context("spawn"),
            )
            try:
                for row in pool.map(
                    _play_league_job,
                    remaining_jobs,
                    chunksize=1,
                ):
                    rows.append(row)
                    if progress_callback is not None:
                        progress_callback(len(rows), len(jobs), row)
                    if early_stop_callback is not None:
                        early_stop = early_stop_callback(rows, len(jobs))
                        if early_stop:
                            break
            finally:
                pool.shutdown(
                    wait=not bool(early_stop),
                    cancel_futures=bool(early_stop),
                )
    elif not early_stop:
        effective_workers = 1
        for job in remaining_jobs:
            row = _play_league_job(job)
            rows.append(row)
            if progress_callback is not None:
                progress_callback(len(rows), len(jobs), row)
            if early_stop_callback is not None:
                early_stop = early_stop_callback(rows, len(jobs))
                if early_stop:
                    break
    report = _summarize_league(
        rows,
        candidate=candidate,
        wildcard_enabled=wildcard_enabled,
        seed=seed,
        deals_per_matchup=deals_per_matchup,
        players=players,
    )
    report["requested_workers"] = workers
    report["effective_workers"] = effective_workers
    report["worker_model"] = (
        "spawn_process_executor"
        if workers > 1 and remaining_jobs
        else "serial"
    )
    report["balanced_rotation_only"] = balanced_rotation_only
    report["resumed_games"] = resumed_count
    report["expected_games"] = len(jobs)
    report["early_stopped"] = bool(early_stop)
    report["early_stop"] = dict(early_stop or {})
    report["ok"] = bool(report["ok"] and not early_stop)
    return report


def _effective_league_workers(
    *,
    candidate: str,
    requested: int,
    remaining_jobs: int,
) -> int:
    logical_cpus = os.cpu_count() or 1
    inner_workers = _NESTED_SEARCH_WORKERS.get(candidate, 0)
    cpu_limit = (
        max(1, logical_cpus // (inner_workers + 1))
        if inner_workers
        else logical_cpus
    )
    return max(
        1,
        min(
            max(1, int(requested)),
            max(1, int(remaining_jobs)),
            cpu_limit,
        ),
    )


def _is_balanced_league_job(job: LeagueJob) -> bool:
    seed = abs(int(job.seed))
    return (
        job.candidate_seat == seed % job.players
        and job.dealer == (seed // job.players) % job.players
    )


def _validate_completed_league_prefix(
    jobs: list[LeagueJob],
    rows: list[dict[str, Any]],
    *,
    record_decisions: bool,
) -> None:
    if len(rows) > len(jobs):
        raise ValueError("completed_league_rows_exceed_schedule")
    for index, row in enumerate(rows):
        expected = _league_job_identity(jobs[index])
        observed = _league_row_identity(row)
        if observed != expected:
            raise ValueError(
                "completed_league_row_mismatch:"
                f"index={index}:expected={expected}:observed={observed}"
            )
        if (
            record_decisions
            and "decision_trace" not in row
            and not row.get("_decision_trace_checkpointed")
        ):
            raise ValueError(
                f"completed_league_row_missing_decision_trace:index={index}"
            )


def _league_job_identity(job: LeagueJob) -> tuple[Any, ...]:
    return (
        job.candidate,
        tuple(sorted(job.opponents)),
        job.players,
        job.candidate_seat,
        job.dealer,
        job.seed,
    )


def _league_row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    try:
        return (
            str(row["candidate"]),
            tuple(sorted(str(name) for name in row["opponents"])),
            int(row["players"]),
            int(row["candidate_seat"]),
            int(row["dealer"]),
            int(row["seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid_completed_league_row") from exc


def _candidate_policy_for_job(job: LeagueJob) -> SimulationPolicy:
    if job.players == 2 and job.candidate == _FROZEN_PRODUCT_CANDIDATE:
        return ProductDecisionKernelSimulationPolicy()
    return create_policy(job.candidate)


def _play_league_job(job: LeagueJob) -> dict[str, Any]:
    policies: list[SimulationPolicy | None] = [None] * job.players
    policies[job.candidate_seat] = _candidate_policy_for_job(job)
    opponent_seats = [seat for seat in range(job.players) if seat != job.candidate_seat]
    for seat, opponent in zip(opponent_seats, job.opponents, strict=True):
        policies[seat] = create_policy(opponent)
    concrete = [policy for policy in policies if policy is not None]
    simulator = FullGameSimulator(
        concrete,
        wildcard_enabled=job.wildcard_enabled,
        dealer=job.dealer,
        rules=rules_for_room(
            wildcard_enabled=job.wildcard_enabled,
            players=job.players,
        ),
        record_decisions=job.record_decisions,
    )
    result = simulator.play(job.seed)
    candidate_policy = concrete[job.candidate_seat]
    diagnostics = (
        candidate_policy.diagnostics()
        if callable(getattr(candidate_policy, "diagnostics", None))
        else {}
    )
    response_events = (
        [
            {
                **event,
                "seed": job.seed,
                "candidate_seat": job.candidate_seat,
                "dealer": job.dealer,
                "players": job.players,
                "wildcard_enabled": job.wildcard_enabled,
            }
            for event in candidate_policy.response_events()
        ]
        if callable(getattr(candidate_policy, "response_events", None))
        else []
    )
    discard_events = (
        [
            {
                **event,
                "seed": job.seed,
                "candidate_seat": job.candidate_seat,
                "dealer": job.dealer,
                "players": job.players,
                "wildcard_enabled": job.wildcard_enabled,
            }
            for event in candidate_policy.discard_events()
        ]
        if callable(getattr(candidate_policy, "discard_events", None))
        else []
    )
    winner_policy = None
    if result.winner is not None:
        winner_policy = concrete[result.winner].name
    candidate_outcome_score = (
        result.score
        if result.winner == job.candidate_seat
        else -result.score
        if result.winner is not None
        else 0.0
    )
    row = {
        "candidate": job.candidate,
        "opponents": sorted(job.opponents),
        "players": job.players,
        "wildcard_enabled": job.wildcard_enabled,
        "candidate_seat": job.candidate_seat,
        "dealer": job.dealer,
        "seed": job.seed,
        "winner": result.winner,
        "winner_policy": winner_policy,
        "candidate_won": result.winner == job.candidate_seat,
        "draw": result.winner is None,
        "candidate_outcome_score": candidate_outcome_score,
        "candidate_diagnostics": diagnostics,
        "candidate_response_events": response_events,
        "candidate_discard_events": discard_events,
        "result": result.to_dict(),
    }
    if job.record_decisions:
        row["decision_trace"] = simulator.decision_trace()
    return row


def _summarize_league(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    wildcard_enabled: bool,
    seed: int,
    deals_per_matchup: int,
    players: int,
) -> dict[str, Any]:
    candidate_wins = sum(int(row["candidate_won"]) for row in rows)
    draws = sum(int(row["draw"]) for row in rows)
    losses = len(rows) - candidate_wins - draws
    violations = sum(len(row["result"]["violations"]) for row in rows)
    coverage_failures = sum(len(row["result"]["coverage_failures"]) for row in rows)
    matchup_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = " + ".join(row["opponents"])
        matchup_rows.setdefault(key, []).append(row)
    by_matchup = {
        key: _summarize_rows(group)
        for key, group in sorted(matchup_rows.items())
    }
    decisive = candidate_wins + losses
    worst_matchup = min(
        by_matchup,
        key=lambda key: (
            by_matchup[key]["candidate_win_share_all"],
            by_matchup[key]["mean_candidate_outcome_score"],
        ),
    )
    response_events = [
        event
        for row in rows
        for event in row.get("candidate_response_events") or []
    ]
    discard_events = [
        event
        for row in rows
        for event in row.get("candidate_discard_events") or []
    ]
    strategy_runtime_error_reasons = Counter(
        str(event["validation_error"])
        for event in discard_events
        if event.get("validation_error")
        and not str(event["validation_error"]).startswith(
            "decision_budget_"
        )
    )
    strategy_runtime_errors = sum(
        strategy_runtime_error_reasons.values()
    )
    return {
        "ok": (
            violations == 0
            and coverage_failures == 0
            and strategy_runtime_errors == 0
        ),
        "candidate": candidate,
        "players": players,
        "wildcard_enabled": wildcard_enabled,
        "seed": seed,
        "deals_per_matchup": deals_per_matchup,
        "games": len(rows),
        "candidate_wins": candidate_wins,
        "losses": losses,
        "draws": draws,
        "candidate_win_share_all": round(candidate_wins / len(rows), 4) if rows else 0.0,
        "candidate_win_share_decisive": round(candidate_wins / decisive, 4) if decisive else 0.0,
        "mean_candidate_outcome_score": round(
            mean(float(row["candidate_outcome_score"]) for row in rows),
            4,
        )
        if rows
        else 0.0,
        "invariant_violations": violations,
        "coverage_failures": coverage_failures,
        "strategy_runtime_errors": strategy_runtime_errors,
        "strategy_runtime_error_reasons": dict(
            sorted(strategy_runtime_error_reasons.items())
        ),
        "decision_trace_recorded": any("decision_trace" in row for row in rows),
        "decision_trace_steps": sum(
            len(row.get("decision_trace") or [])
            for row in rows
        ),
        "candidate_diagnostics": _summarize_candidate_diagnostics(rows),
        "response_shadow": _summarize_response_shadow(response_events),
        "discard_shadow": _summarize_discard_shadow(discard_events),
        "worst_matchup": worst_matchup,
        "by_matchup": by_matchup,
        "rows": rows,
    }


def _summarize_candidate_diagnostics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    numeric: Counter[str] = Counter()
    categorical: dict[str, Counter[str]] = {}
    for row in rows:
        for key, value in (row.get("candidate_diagnostics") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[str(key)] += value
            elif value is not None:
                categorical.setdefault(str(key), Counter())[str(value)] += 1
    result: dict[str, Any] = dict(sorted(numeric.items()))
    for key, counts in sorted(categorical.items()):
        if len(counts) == 1:
            result[key] = next(iter(counts))
        else:
            result[f"{key}_counts"] = dict(sorted(counts.items()))
    return result


def _summarize_response_shadow(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    attempted = [event for event in events if event.get("search_attempted")]
    usable = [event for event in attempted if event.get("used_search")]
    disagreements = [event for event in usable if event.get("disagreement")]
    candidate_coverage_failures = [
        event
        for event in attempted
        if (
            len(event.get("candidates") or [])
            != int(event.get("candidate_count") or 0)
            or any(
                int(candidate.get("visits") or 0) == 0
                for candidate in event.get("candidates") or []
            )
        )
    ]
    latencies = sorted(
        float(event["elapsed_ms"])
        for event in attempted
        if isinstance(event.get("elapsed_ms"), (int, float))
    )
    return {
        "opportunities": len(events),
        "attempts": len(attempted),
        "usable": len(usable),
        "disagreements": len(disagreements),
        "attempt_rate": round(len(attempted) / len(events), 4) if events else 0.0,
        "usable_rate": round(len(usable) / len(attempted), 4) if attempted else 0.0,
        "disagreement_rate": round(len(disagreements) / len(usable), 4) if usable else 0.0,
        "simulations": sum(int(event.get("simulations") or 0) for event in attempted),
        "paired_determinizations": sum(
            int(event.get("paired_determinizations") or 0)
            for event in attempted
        ),
        "deadline_interruptions": sum(
            int(event.get("deadline_interruptions") or 0)
            for event in attempted
        ),
        "rollout_invariant_violations": sum(
            int(event.get("rollout_invariant_violations") or 0)
            for event in attempted
        ),
        "rollout_coverage_failures": sum(
            int(event.get("rollout_coverage_failures") or 0)
            for event in attempted
        ),
        "candidate_coverage_failures": len(candidate_coverage_failures),
        "by_response_type": dict(
            sorted(Counter(str(event.get("response_type") or "unknown") for event in events).items())
        ),
        "by_reason": dict(
            sorted(Counter(str(event.get("reason") or "unknown") for event in events).items())
        ),
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "p99": round(_percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "disagreement_samples": disagreements[:20],
    }


def _summarize_discard_shadow(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [event for event in events if event.get("eligible")]
    root_events = [
        event
        for event in events
        if "search_attempted" in event
    ]
    root_attempted = [
        event
        for event in root_events
        if event.get("search_attempted")
    ]
    root_usable = [
        event
        for event in root_attempted
        if event.get("used_search")
    ]
    root_latencies = sorted(
        float(event.get("elapsed_ms") or 0.0)
        for event in root_attempted
    )
    root_candidate_coverage_failures = [
        event
        for event in root_attempted
        if (
            int(event.get("searched_candidate_count") or 0)
            != int(event.get("legal_candidate_count") or 0)
            or len(event.get("candidates") or [])
            != int(event.get("legal_candidate_count") or 0)
            or any(
                int(candidate.get("visits") or 0) == 0
                for candidate in event.get("candidates") or []
            )
        )
    ]
    validation_events = [
        event
        for event in root_attempted
        if "validation_complete" in event
        and event.get("validation_attempted", True)
    ]
    validation_stages = [
        stage
        for event in validation_events
        for stage in (
            (event.get("validation_diagnostics") or {}).get(
                "coverage"
            ),
            (event.get("validation_diagnostics") or {}).get(
                "first_confirmation"
            ),
            (event.get("validation_diagnostics") or {}).get(
                "second_confirmation"
            ),
        )
        if isinstance(stage, dict)
    ]
    risk_events = [event for event in events if isinstance(event.get("risk_search"), dict)]
    risk_usable = [
        event
        for event in risk_events
        if event["risk_search"].get("used_search")
    ]
    latencies = sorted(
        float(event["risk_search"].get("elapsed_ms") or 0.0)
        for event in risk_events
    )
    return {
        "opportunities": len(events),
        "captured_roots": len(eligible),
        "capture_rate": round(len(eligible) / len(events), 4) if events else 0.0,
        "by_reason": dict(
            sorted(Counter(str(event.get("reason") or "unknown") for event in events).items())
        ),
        "response_risk": {
            "attempts": len(risk_events),
            "usable": len(risk_usable),
            "usable_rate": round(len(risk_usable) / len(risk_events), 4)
            if risk_events
            else 0.0,
            "disagreements": sum(int(bool(event.get("disagreement"))) for event in risk_events),
            "simulations": sum(
                int(event["risk_search"].get("simulations") or 0)
                for event in risk_events
            ),
            "paired_determinizations": sum(
                int(event["risk_search"].get("paired_determinizations") or 0)
                for event in risk_events
            ),
            "deadline_interruptions": sum(
                int(event["risk_search"].get("deadline_interruptions") or 0)
                for event in risk_events
            ),
            "invariant_violations": sum(
                int(event["risk_search"].get("rollout_invariant_violations") or 0)
                for event in risk_events
            ),
            "coverage_failures": sum(
                int(event["risk_search"].get("rollout_coverage_failures") or 0)
                for event in risk_events
            ),
            "latency_ms": {
                "p50": round(_percentile(latencies, 0.50), 3),
                "p95": round(_percentile(latencies, 0.95), 3),
                "p99": round(_percentile(latencies, 0.99), 3),
                "max": round(max(latencies), 3) if latencies else 0.0,
            },
            "disagreement_samples": [
                event
                for event in risk_events
                if event.get("disagreement")
            ][:20],
        },
        "root_search": {
            "opportunities": len(root_events),
            "attempts": len(root_attempted),
            "usable": len(root_usable),
            "usable_rate": round(
                len(root_usable) / len(root_attempted),
                4,
            )
            if root_attempted
            else 0.0,
            "disagreements": sum(
                int(bool(event.get("disagreement")))
                for event in root_usable
            ),
            "simulations": sum(
                int(event.get("simulations") or 0)
                for event in root_attempted
            ),
            "paired_determinizations": sum(
                int(event.get("paired_determinizations") or 0)
                for event in root_attempted
            ),
            "deadline_interruptions": sum(
                int(event.get("deadline_interruptions") or 0)
                for event in root_attempted
            ),
            "invariant_violations": sum(
                int(event.get("rollout_invariant_violations") or 0)
                for event in root_attempted
            ),
            "rollout_coverage_failures": sum(
                int(event.get("rollout_coverage_failures") or 0)
                for event in root_attempted
            ),
            "candidate_coverage_failures": len(
                root_candidate_coverage_failures
            ),
            "dual_validation": {
                "attempts": len(validation_events),
                "complete": sum(
                    int(bool(event.get("validation_complete")))
                    for event in validation_events
                ),
                "incomplete": sum(
                    int(not bool(event.get("validation_complete")))
                    for event in validation_events
                ),
                "errors": sum(
                    int(bool(event.get("validation_error")))
                    for event in validation_events
                ),
                "reconfirmation_attempts": sum(
                    int(
                        bool(
                            event.get(
                                "validation_reconfirmation_attempted"
                            )
                        )
                    )
                    for event in validation_events
                ),
                "decision_budget_fallbacks": sum(
                    int(
                        str(event.get("validation_error") or "").startswith(
                            "decision_budget_"
                        )
                    )
                    for event in validation_events
                ),
                "deadline_interruptions": sum(
                    int(stage.get("deadline_interruptions") or 0)
                    for stage in validation_stages
                ),
                "invariant_violations": sum(
                    int(
                        stage.get(
                            "rollout_invariant_violations"
                        )
                        or 0
                    )
                    for stage in validation_stages
                ),
                "coverage_failures": sum(
                    int(stage.get("rollout_coverage_failures") or 0)
                    for stage in validation_stages
                ),
                "zero_visit_candidates": sum(
                    int(stage.get("zero_visit_candidates") or 0)
                    for stage in validation_stages
                ),
            },
            "by_reason": dict(
                sorted(
                    Counter(
                        str(event.get("reason") or "unknown")
                        for event in root_events
                    ).items()
                )
            ),
            "latency_ms": {
                "p50": round(_percentile(root_latencies, 0.50), 3),
                "p95": round(_percentile(root_latencies, 0.95), 3),
                "p99": round(_percentile(root_latencies, 0.99), 3),
                "max": (
                    round(max(root_latencies), 3)
                    if root_latencies
                    else 0.0
                ),
            },
        },
        "samples": eligible[:20],
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(int(row["candidate_won"]) for row in rows)
    draws = sum(int(row["draw"]) for row in rows)
    losses = len(rows) - wins - draws
    decisive = wins + losses
    low, high = _wilson_interval(wins, decisive)
    return {
        "games": len(rows),
        "candidate_wins": wins,
        "losses": losses,
        "draws": draws,
        "candidate_win_share_all": round(wins / len(rows), 4) if rows else 0.0,
        "candidate_win_share_decisive": round(wins / decisive, 4) if decisive else 0.0,
        "wilson_95_decisive": [round(low, 4), round(high, 4)],
        "mean_candidate_outcome_score": round(
            mean(float(row["candidate_outcome_score"]) for row in rows),
            4,
        )
        if rows
        else 0.0,
        "invariant_violations": sum(len(row["result"]["violations"]) for row in rows),
        "coverage_failures": sum(
            len(row["result"]["coverage_failures"])
            for row in rows
        ),
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


__all__ = [
    "AggressiveMeldPolicy",
    "ConservativeSearchPolicy",
    "DEFAULT_HEADS_UP_MATCHUPS",
    "DEFAULT_MATCHUPS",
    "DefensiveSearchPolicy",
    "LeagueJob",
    "POLICY_FACTORIES",
    "RedBlackSearchPolicy",
    "build_league_jobs",
    "create_policy",
    "run_opponent_league",
    "_summarize_discard_shadow",
]
