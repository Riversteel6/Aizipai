"""Low-latency rollout proxies for expensive public opponent policies."""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
import os
from typing import Any

from ai.features import (
    QUICK_POTENTIAL_LABELS,
    best_quick_potential_after_discard_from_counts,
    quick_potential,
    quick_potential_after_each_removal_from_counts,
)
from ai.full_game_simulator import (
    ChiPlan,
    InformationSetSearchPolicy,
    PublicView,
    SimMeld,
    _chi_kind,
    _discardable_labels,
    _draw_shape_relevance_values,
    _public_discard_danger,
    _remove_many,
    _remove_one,
    _replace_view,
    evaluate_hu,
)
from audit.independent_opponent import (
    IndependentValue,
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
    _exposed_xi,
    _red_black_route_score,
    _shape_score,
)
from engine.cards import RED_LABELS, WILD_LABEL
from engine.xi_calculator import meld_xi


def _draw_shape_relevance(label: str, hand: tuple[str, ...]) -> int:
    """Compatibility scalar for audit callers; production ranks draws in one batch."""

    return _draw_shape_relevance_values((label,), hand)[label]


class FastInformationSetProxyPolicy(InformationSetSearchPolicy):
    """Approximate the search opponent without full wait enumeration."""

    name = "fast_information_set_proxy_v1"
    peng_margin = 15.0
    chi_margin = 20.0

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        labels = _discardable_labels(view.hand)
        if not labels:
            raise ValueError("no_discardable_card")
        wildcard_enabled = bool(rules.get("wildcard", {}).get("enabled", False))
        shortlist_size = 8 if wildcard_enabled else 4
        shortlist = sorted(
            labels,
            key=lambda label: self._cheap_discard_value(view, label, rules),
            reverse=True,
        )[:shortlist_size]
        values = _fast_discard_values(view, rules, shortlist)
        return max(shortlist, key=values.__getitem__)

    def choose_peng(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ) -> bool:
        gain = self._peng_response_gain(view, label, rules)
        return gain is not None and gain > self.peng_margin

    def _peng_response_gain(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ) -> float | None:
        if view.hand.count(label) != 2:
            return None
        pass_value = _fast_position_value(
            view.hand,
            view.own_melds,
            rules,
        )
        hand_after = tuple(_remove_many(view.hand, [label, label]))
        melds = (
            *view.own_melds,
            SimMeld("peng", (label, label, label)),
        )
        claim_view = _replace_view(
            view,
            hand=hand_after,
            own_melds=melds,
        )
        try:
            discard = self.choose_discard(claim_view, rules)
        except ValueError:
            return None
        claim_value = _fast_position_value(
            tuple(_remove_one(hand_after, discard)),
            melds,
            rules,
        )
        return claim_value - pass_value

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        best_gain, best_plan = self._best_chi_response_gain(view, plans, rules)
        return (
            best_plan
            if best_plan is not None and best_gain > self.chi_margin
            else None
        )

    def _best_chi_response_gain(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> tuple[float, ChiPlan | None]:
        pass_value = _fast_position_value(
            view.hand,
            view.own_melds,
            rules,
        )
        best_plan: ChiPlan | None = None
        best_value = -math.inf
        for plan in plans:
            hand_after = tuple(
                _remove_many(view.hand, plan.consumed_from_hand)
            )
            melds = (
                *view.own_melds,
                *(
                    SimMeld(_chi_kind(group), tuple(group))
                    for group in plan.groups
                ),
            )
            claim_view = _replace_view(
                view,
                hand=hand_after,
                own_melds=melds,
            )
            try:
                discard = self.choose_discard(claim_view, rules)
            except ValueError:
                continue
            value = _fast_position_value(
                tuple(_remove_one(hand_after, discard)),
                melds,
                rules,
            )
            if value > best_value:
                best_value = value
                best_plan = plan
        return best_value - pass_value, best_plan

    def _discard_value(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ) -> float:
        after = tuple(_remove_one(view.hand, label))
        current = _fast_position_value(
            after,
            view.own_melds,
            rules,
        )
        unseen = Counter(dict(view.remaining_counts))
        if sum(unseen.values()) <= 0:
            return current
        draw_sample_size = (
            12
            if rules.get("wildcard", {}).get("enabled", False)
            else 8
        )
        sampled_draws = sorted(
            unseen.items(),
            key=lambda item: (
                item[1],
                _draw_shape_relevance(item[0], after),
            ),
            reverse=True,
        )[:draw_sample_size]
        sampled_total = sum(
            amount
            for _draw, amount in sampled_draws
        )
        expected = sum(
            amount
            / max(1, sampled_total)
            * _fast_followup_value(
                (*after, draw),
                view.own_melds,
                rules,
            )
            for draw, amount in sampled_draws
            if amount > 0
        )
        return (
            current
            + expected * 0.35
            - _public_discard_danger(label, view)
        )


class FastAggressiveMeldProxyPolicy(FastInformationSetProxyPolicy):
    """Fast information-set discard with the aggressive teacher's claims."""

    name = "fast_aggressive_meld_proxy_v1"

    def choose_peng(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ) -> bool:
        return view.hand.count(label) == 2 and bool(
            _discardable_labels(_remove_many(view.hand, [label, label]))
        )

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
            bonus = sum(
                18.0
                if _chi_kind(group) in {"special_123", "special_2710"}
                else 5.0
                for group in plan.groups
            )
            scored.append((quick_potential(after) * 12.0 + bonus, plan))
        return max(scored, key=lambda item: item[0])[1] if scored else None


class FastDefensiveSearchProxyPolicy(FastInformationSetProxyPolicy):
    """Fast information-set proxy retaining the defensive danger modifier."""

    name = "fast_defensive_search_proxy_v1"

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        labels = _discardable_labels(view.hand)
        if not labels:
            raise ValueError("no_discardable_card")
        values = _fast_discard_values(view, rules, labels)
        return max(
            labels,
            key=lambda label: (
                values[label]
                - _public_discard_danger(label, view) * 2.5,
                label,
            ),
        )


class FastRedBlackSearchProxyPolicy(FastInformationSetProxyPolicy):
    """Fast information-set proxy retaining the red/black route modifier."""

    name = "fast_red_black_search_proxy_v1"

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        labels = _discardable_labels(view.hand)
        if not labels:
            raise ValueError("no_discardable_card")
        current_red = sum(label in RED_LABELS for label in view.hand)
        chase_red = current_red >= 7
        values = _fast_discard_values(view, rules, labels)

        def route_value(label: str) -> float:
            after_red = current_red - int(label in RED_LABELS)
            route_bonus = (
                after_red * 18.0
                if chase_red
                else -min(abs(after_red), abs(after_red - 1)) * 22.0
            )
            return values[label] + route_bonus

        return max(labels, key=lambda label: (route_value(label), label))


class FastIndependentStructuralDiscardProxyPolicy(IndependentFastRolloutPolicy):
    """Retain the exact independent teacher's own discard prefilter."""

    name = "fast_independent_structural_discard_proxy_v1"

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        labels = _discardable_labels(view.hand)
        if not labels:
            raise ValueError("no_discardable_card")
        return max(
            labels,
            key=lambda label: (
                self._cheap_discard_value(view, label, rules),
                label,
            ),
        )


class FastIndependentDenialStructuralDiscardProxyPolicy(
    FastIndependentStructuralDiscardProxyPolicy,
):
    name = "fast_independent_denial_structural_discard_proxy_v1"
    weights = IndependentFastDenialPolicy.weights


class FastIndependentPressureStructuralDiscardProxyPolicy(
    FastIndependentStructuralDiscardProxyPolicy,
):
    name = "fast_independent_pressure_structural_discard_proxy_v1"
    weights = IndependentFastPressurePolicy.weights


class HybridInformationSetResponseProxyPolicy(FastInformationSetProxyPolicy):
    """Exact search response values with the frozen fast discard proxy."""

    name = "hybrid_information_set_response_proxy_v1"

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        return InformationSetSearchPolicy.choose_peng(self, view, label, rules)

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        return InformationSetSearchPolicy.choose_chi(self, view, plans, rules)


class HybridDefensiveSearchResponseProxyPolicy(FastDefensiveSearchProxyPolicy):
    name = "hybrid_defensive_search_response_proxy_v1"

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        return InformationSetSearchPolicy.choose_peng(self, view, label, rules)

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        return InformationSetSearchPolicy.choose_chi(self, view, plans, rules)


class HybridRedBlackSearchResponseProxyPolicy(FastRedBlackSearchProxyPolicy):
    name = "hybrid_red_black_search_response_proxy_v1"

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        return InformationSetSearchPolicy.choose_peng(self, view, label, rules)

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        return InformationSetSearchPolicy.choose_chi(self, view, plans, rules)


class _ProductionHuIndependentResponseMixin:
    """Use the cross-validated production Hu solver only for response value."""

    def _position_value(
        self,
        hand: tuple[str, ...] | list[str],
        melds: tuple[Any, ...] | list[Any],
        remaining_counts: tuple[tuple[str, int], ...] | list[tuple[str, int]],
        rules: dict[str, Any],
        *,
        exposed_xi: int | None = None,
        shape_score: float | None = None,
        route_score: float | None = None,
        shape_draws: tuple[int, float] | None = None,
    ) -> IndependentValue:
        hand_tuple = tuple(hand)
        meld_tuple = tuple(melds)
        required_groups = int(rules.get("rules", {}).get("required_meld_groups", 7))
        hu_outs = 0
        partition_outs = 0
        weighted_hu_score = 0.0
        for label, raw_amount in remaining_counts:
            amount = int(raw_amount)
            if amount <= 0:
                continue
            hu = evaluate_hu((*hand_tuple, label), meld_tuple, rules)
            partition_complete = (
                sum(len(group) for group in hu.groups) == len(hand_tuple) + 1
                and len(hu.groups) + len(meld_tuple) == required_groups
            )
            if partition_complete:
                partition_outs += amount
            if hu.can_hu:
                hu_outs += amount
                weighted_hu_score += amount * float(hu.score)
        if exposed_xi is None:
            exposed_xi = _exposed_xi(meld_tuple, rules)
        if shape_score is None:
            shape_score = _shape_score(hand_tuple, rules)
        if route_score is None:
            route_score = _red_black_route_score(hand_tuple, meld_tuple, rules)
        wildcard_count = hand_tuple.count(WILD_LABEL)
        weights = self.weights
        total = (
            hu_outs * weights.hu_out
            + weighted_hu_score * weights.hu_score
            + partition_outs * weights.partition_out
            + exposed_xi * weights.exposed_xi
            + shape_score * weights.shape
            + route_score * weights.route
            + wildcard_count * weights.wildcard
        )
        return IndependentValue(
            total=total,
            hu_outs=hu_outs,
            partition_outs=partition_outs,
            weighted_hu_score=weighted_hu_score,
            exposed_xi=exposed_xi,
            shape_score=shape_score,
            route_score=route_score,
        )


class HybridIndependentResponseProxyPolicy(
    _ProductionHuIndependentResponseMixin,
    FastIndependentStructuralDiscardProxyPolicy,
):
    name = "hybrid_independent_response_proxy_v1"


class HybridIndependentDenialResponseProxyPolicy(
    _ProductionHuIndependentResponseMixin,
    FastIndependentDenialStructuralDiscardProxyPolicy,
):
    name = "hybrid_independent_denial_response_proxy_v1"


class HybridIndependentPressureResponseProxyPolicy(
    _ProductionHuIndependentResponseMixin,
    FastIndependentPressureStructuralDiscardProxyPolicy,
):
    name = "hybrid_independent_pressure_response_proxy_v1"


_QUICK_INDEX = {
    label: index for index, label in enumerate(QUICK_POTENTIAL_LABELS)
}


def _fast_discard_values(
    view: PublicView,
    rules: dict[str, Any],
    labels: list[str],
) -> dict[str, float]:
    base_counts = [0] * len(QUICK_POTENTIAL_LABELS)
    for label in view.hand:
        index = _QUICK_INDEX.get(label)
        if index is not None:
            base_counts[index] += 1
    existing_xi = sum(
        meld_xi(list(meld.cards), kind=meld.kind, rules=rules)
        for meld in view.own_melds
    )
    unseen = Counter(dict(view.remaining_counts))
    base_removal_values = quick_potential_after_each_removal_from_counts(
        tuple(base_counts)
    )
    red_count = sum(card in RED_LABELS for card in view.hand)
    draw_sample_size = (
        12 if rules.get("wildcard", {}).get("enabled", False) else 8
    )
    values: dict[str, float] = {}
    for label in labels:
        discard_index = _QUICK_INDEX[label]
        after_counts = list(base_counts)
        after_counts[discard_index] -= 1
        after = tuple(_remove_one(view.hand, label))
        current = (
            int(base_removal_values[discard_index]) * 12.0
            + existing_xi * 45.0
            + (red_count - int(label in RED_LABELS)) * 3.0
        )
        draw_relevance = _draw_shape_relevance_values(
            tuple(unseen),
            after,
        )
        sampled_draws = sorted(
            unseen.items(),
            key=lambda item: (
                item[1],
                draw_relevance[item[0]],
            ),
            reverse=True,
        )[:draw_sample_size]
        sampled_total = sum(amount for _draw, amount in sampled_draws)
        expected = 0.0
        for draw, amount in sampled_draws:
            if amount <= 0:
                continue
            draw_index = _QUICK_INDEX.get(draw)
            if draw_index is None:
                future = _fast_followup_value(
                    (*after, draw),
                    view.own_melds,
                    rules,
                )
            else:
                drawn_counts = list(after_counts)
                drawn_counts[draw_index] += 1
                best_followup = best_quick_potential_after_discard_from_counts(
                    tuple(drawn_counts)
                )
                future = (
                    best_followup * 12.0 + existing_xi * 45.0
                    if best_followup is not None
                    else -500.0
                )
            expected += amount / max(1, sampled_total) * future
        values[label] = round(
            current + expected * 0.35 - _public_discard_danger(label, view),
            9,
        )
    return values


def _fast_position_value(
    hand: tuple[str, ...] | list[str],
    melds: tuple[SimMeld, ...] | list[SimMeld],
    rules: dict[str, Any],
) -> float:
    existing_xi = sum(
        meld_xi(
            list(meld.cards),
            kind=meld.kind,
            rules=rules,
        )
        for meld in melds
    )
    return (
        _cached_quick_potential(tuple(sorted(hand))) * 12.0
        + existing_xi * 45.0
        + sum(label in RED_LABELS for label in hand) * 3.0
    )


def _fast_followup_value(
    hand: tuple[str, ...],
    melds: tuple[SimMeld, ...],
    rules: dict[str, Any],
) -> float:
    labels = _discardable_labels(hand)
    if not labels:
        return -500.0
    existing_xi = sum(
        meld_xi(
            list(meld.cards),
            kind=meld.kind,
            rules=rules,
        )
        for meld in melds
    )
    return (
        max(
            _cached_quick_potential(
                tuple(sorted(_remove_one(hand, label)))
            )
            for label in labels
        )
        * 12.0
        + existing_xi * 45.0
    )


@lru_cache(
    maxsize=(
        32_768
        if os.environ.get("AIZIPAI_MOBILE_RUNTIME") == "1"
        else 131_072
    )
)
def _cached_quick_potential(hand: tuple[str, ...]) -> int:
    return quick_potential(list(hand))


__all__ = [
    "FastAggressiveMeldProxyPolicy",
    "FastDefensiveSearchProxyPolicy",
    "FastIndependentDenialStructuralDiscardProxyPolicy",
    "FastIndependentPressureStructuralDiscardProxyPolicy",
    "FastIndependentStructuralDiscardProxyPolicy",
    "FastInformationSetProxyPolicy",
    "FastRedBlackSearchProxyPolicy",
    "HybridDefensiveSearchResponseProxyPolicy",
    "HybridIndependentDenialResponseProxyPolicy",
    "HybridIndependentPressureResponseProxyPolicy",
    "HybridIndependentResponseProxyPolicy",
    "HybridInformationSetResponseProxyPolicy",
    "HybridRedBlackSearchResponseProxyPolicy",
]
