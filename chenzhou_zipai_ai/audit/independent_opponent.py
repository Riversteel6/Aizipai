"""Validation-only opponents with a value model independent from production."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

from audit.independent_rules import (
    BIG_LABELS,
    NORMAL_LABELS,
    RED_LABELS,
    SMALL_LABELS,
    WILD_LABEL,
    evaluate_hu_oracle,
)


_LABEL_POSITION = {
    label: (suit, rank)
    for suit, labels in (("small", SMALL_LABELS), ("big", BIG_LABELS))
    for rank, label in enumerate(labels, start=1)
}
_LABEL_INDEX = {
    label: index
    for index, label in enumerate(NORMAL_LABELS)
}


@dataclass(frozen=True)
class IndependentPolicyWeights:
    hu_out: float = 145.0
    hu_score: float = 24.0
    partition_out: float = 38.0
    exposed_xi: float = 52.0
    shape: float = 9.0
    route: float = 8.0
    wildcard: float = 55.0
    discard_danger: float = 1.0
    peng_margin: float = 18.0
    chi_margin: float = 24.0
    discard_shortlist: int = 8


@dataclass(frozen=True)
class IndependentMeld:
    kind: str
    cards: tuple[str, ...]


@dataclass(frozen=True)
class IndependentValue:
    total: float
    hu_outs: int
    partition_outs: int
    weighted_hu_score: float
    exposed_xi: int
    shape_score: float
    route_score: float


class IndependentBalancedPolicy:
    """Public-information policy that shares rules, but no production valuation."""

    name = "independent_balanced"
    weights = IndependentPolicyWeights()

    def choose_discard(self, view: Any, rules: dict[str, Any]) -> str:
        label, _value = self._best_discard(view, rules)
        return label

    def choose_hu(self, view: Any, hu: Any, rules: dict[str, Any]) -> bool:
        return bool(getattr(hu, "can_hu", False))

    def choose_peng(self, view: Any, label: str, rules: dict[str, Any]) -> bool:
        gain = self._peng_response_gain(view, label, rules)
        return gain is not None and gain > self.weights.peng_margin

    def _peng_response_gain(
        self,
        view: Any,
        label: str,
        rules: dict[str, Any],
    ) -> float | None:
        if tuple(view.hand).count(label) != 2:
            return None
        hand_after = _remove_many(view.hand, (label, label))
        melds_after = (
            *tuple(view.own_melds),
            IndependentMeld("peng", (label, label, label)),
        )
        if not _discardable_labels(hand_after):
            return None
        pass_value = self._position_value(
            view.hand,
            view.own_melds,
            view.remaining_counts,
            rules,
        )
        claim_view = replace(
            view,
            hand=hand_after,
            own_melds=melds_after,
        )
        _discard, claim_value = self._best_discard(claim_view, rules)
        return claim_value.total - pass_value.total

    def choose_chi(
        self,
        view: Any,
        plans: Sequence[Any],
        rules: dict[str, Any],
    ) -> Any | None:
        best_gain, best_plan = self._best_chi_response_gain(view, plans, rules)
        return (
            best_plan
            if best_plan is not None and best_gain > self.weights.chi_margin
            else None
        )

    def _best_chi_response_gain(
        self,
        view: Any,
        plans: Sequence[Any],
        rules: dict[str, Any],
    ) -> tuple[float, Any | None]:
        pass_value = self._position_value(
            view.hand,
            view.own_melds,
            view.remaining_counts,
            rules,
        )
        best: tuple[float, str, Any] | None = None
        for plan in plans:
            hand_after = _remove_many(view.hand, tuple(plan.consumed_from_hand))
            if not _discardable_labels(hand_after):
                continue
            melds_after = (
                *tuple(view.own_melds),
                *(
                    IndependentMeld(_group_kind(group), tuple(group))
                    for group in plan.groups
                ),
            )
            claim_view = replace(
                view,
                hand=hand_after,
                own_melds=melds_after,
            )
            discard, value = self._best_discard(claim_view, rules)
            key = (value.total, discard, plan)
            if best is None or key[:2] > best[:2]:
                best = key
        if best is None:
            return float("-inf"), None
        return best[0] - pass_value.total, best[2]

    def _best_discard(
        self,
        view: Any,
        rules: dict[str, Any],
    ) -> tuple[str, IndependentValue]:
        scored = self.discard_scores(view, rules)
        label, (_adjusted, value) = max(
            scored.items(),
            key=lambda item: (item[1][0], item[0]),
        )
        return label, value

    def discard_scores(
        self,
        view: Any,
        rules: dict[str, Any],
        *,
        all_candidates: bool = False,
    ) -> dict[str, tuple[float, IndependentValue]]:
        """Expose exact teacher values without changing runtime selection."""
        hand = tuple(view.hand)
        hand_counts = Counter(hand)
        labels = sorted(
            label
            for label, amount in hand_counts.items()
            if label != WILD_LABEL and amount < 3
        )
        if not labels:
            raise ValueError("no_discardable_card")
        exposed_xi = _exposed_xi(view.own_melds, rules)
        meld_red_count = _meld_red_count(view.own_melds)
        wildcard_count = hand_counts[WILD_LABEL]
        allow_1510 = bool(rules.get("rules", {}).get("allow_1510", False))
        normal_counts = tuple(hand_counts[label] for label in NORMAL_LABELS)
        base_shape_score = _shape_score(hand, rules)
        total_red_count = (
            sum(hand_counts[label] for label in RED_LABELS)
            + meld_red_count
        )
        route_scores = {
            is_red: _red_black_route_score_from_counts(
                red_count=total_red_count - int(is_red),
                wildcard_count=wildcard_count,
                rules=rules,
            )
            for is_red in (False, True)
        }
        shape_draw_context = self._prepare_shape_draw_context(
            view.remaining_counts,
            rules,
        )
        prepared: dict[str, tuple[float, float, float]] = {}
        for label in labels:
            shape_score = base_shape_score - _shape_removal_loss(
                normal_counts,
                label,
                allow_1510=allow_1510,
            )
            route_score = route_scores[label in RED_LABELS]
            danger = _public_discard_danger(label, view)
            prepared[label] = (shape_score, route_score, danger)
        cheap_rows = sorted(
            (
                (
                    exposed_xi * self.weights.exposed_xi
                    + prepared[label][0] * self.weights.shape
                    + prepared[label][1] * self.weights.route
                    + wildcard_count * self.weights.wildcard
                    - self.weights.discard_danger * prepared[label][2]
                ),
                label,
            )
            for label in labels
        )
        shortlist = {
            label
            for _value, label in cheap_rows[
                -max(1, int(self.weights.discard_shortlist)) :
            ]
        }
        scored: dict[str, tuple[float, IndependentValue]] = {}
        evaluated_labels = labels if all_candidates else sorted(shortlist)
        for label in evaluated_labels:
            shape_score, route_score, danger = prepared[label]
            after = _remove_one(hand, label)
            value = self._position_value(
                after,
                view.own_melds,
                view.remaining_counts,
                rules,
                exposed_xi=exposed_xi,
                shape_score=shape_score,
                route_score=route_score,
                shape_draws=self._shape_draws_after_discard(
                    normal_counts,
                    label,
                    shape_draw_context,
                ),
            )
            adjusted = (
                value.total
                - self.weights.discard_danger * danger
            )
            scored[label] = (adjusted, value)
        return scored

    def _prepare_shape_draw_context(
        self,
        remaining_counts: Sequence[tuple[str, int]],
        rules: Mapping[str, Any],
    ) -> Any | None:
        return None

    def _shape_draws_after_discard(
        self,
        normal_counts: Sequence[int],
        label: str,
        context: Any | None,
    ) -> tuple[int, float] | None:
        return None

    def _cheap_discard_value(
        self,
        view: Any,
        label: str,
        rules: Mapping[str, Any],
    ) -> float:
        after = _remove_one(view.hand, label)
        return (
            _exposed_xi(view.own_melds, rules) * self.weights.exposed_xi
            + _shape_score(after, rules) * self.weights.shape
            + _red_black_route_score(after, view.own_melds, rules) * self.weights.route
            + after.count(WILD_LABEL) * self.weights.wildcard
            - self.weights.discard_danger * _public_discard_danger(label, view)
        )

    def _position_value(
        self,
        hand: Sequence[str],
        melds: Sequence[Any],
        remaining_counts: Sequence[tuple[str, int]],
        rules: Mapping[str, Any],
        *,
        exposed_xi: int | None = None,
        shape_score: float | None = None,
        route_score: float | None = None,
        shape_draws: tuple[int, float] | None = None,
    ) -> IndependentValue:
        hu_outs = 0
        partition_outs = 0
        weighted_hu_score = 0.0
        for label, amount in remaining_counts:
            count = int(amount)
            if count <= 0 or label not in (*NORMAL_LABELS, WILD_LABEL):
                continue
            hu = evaluate_hu_oracle((*tuple(hand), label), melds, rules)
            if hu.partition_complete:
                partition_outs += count
            if hu.can_hu:
                hu_outs += count
                weighted_hu_score += count * float(hu.score)
        if exposed_xi is None:
            exposed_xi = _exposed_xi(melds, rules)
        if shape_score is None:
            shape_score = _shape_score(hand, rules)
        if route_score is None:
            route_score = _red_black_route_score(hand, melds, rules)
        wildcard_count = tuple(hand).count(WILD_LABEL)
        total = (
            hu_outs * self.weights.hu_out
            + weighted_hu_score * self.weights.hu_score
            + partition_outs * self.weights.partition_out
            + exposed_xi * self.weights.exposed_xi
            + shape_score * self.weights.shape
            + route_score * self.weights.route
            + wildcard_count * self.weights.wildcard
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


class IndependentPressurePolicy(IndependentBalancedPolicy):
    """Lower claim thresholds to pressure passive production candidates."""

    name = "independent_pressure"
    weights = replace(
        IndependentPolicyWeights(),
        hu_out=132.0,
        shape=10.5,
        discard_danger=1.25,
        peng_margin=-4.0,
        chi_margin=2.0,
    )


class IndependentDenialPolicy(IndependentBalancedPolicy):
    """Prioritize low-risk discards and preserve concealed routes."""

    name = "independent_denial"
    weights = replace(
        IndependentPolicyWeights(),
        hu_out=155.0,
        hu_score=28.0,
        route=10.0,
        discard_danger=2.8,
        peng_margin=38.0,
        chi_margin=52.0,
    )


class IndependentExploitPolicy(IndependentBalancedPolicy):
    """Reserved frozen slot for weights selected on isolated training seeds."""

    name = "independent_exploit_untrained"
    weights = replace(
        IndependentPolicyWeights(),
        hu_out=150.0,
        shape=10.0,
        discard_danger=1.8,
        peng_margin=8.0,
        chi_margin=12.0,
    )


class IndependentFastRolloutPolicy(IndependentBalancedPolicy):
    """Independent low-latency policy for full-game search rollouts."""

    name = "independent_fast_rollout"

    def _prepare_shape_draw_context(
        self,
        remaining_counts: Sequence[tuple[str, int]],
        rules: Mapping[str, Any],
    ) -> tuple[tuple[int, ...], bool]:
        return (
            _indexed_remaining_counts(remaining_counts),
            bool(rules.get("rules", {}).get("allow_1510", False)),
        )

    def _shape_draws_after_discard(
        self,
        normal_counts: Sequence[int],
        label: str,
        context: tuple[tuple[int, ...], bool] | None,
    ) -> tuple[int, float] | None:
        if context is None:
            return None
        counts = list(normal_counts)
        counts[_LABEL_INDEX[label]] -= 1
        remaining, allow_1510 = context
        return _shape_improving_draws_from_counts(
            tuple(counts),
            remaining,
            allow_1510=allow_1510,
        )

    def _position_value(
        self,
        hand: Sequence[str],
        melds: Sequence[Any],
        remaining_counts: Sequence[tuple[str, int]],
        rules: Mapping[str, Any],
        *,
        exposed_xi: int | None = None,
        shape_score: float | None = None,
        route_score: float | None = None,
        shape_draws: tuple[int, float] | None = None,
    ) -> IndependentValue:
        if exposed_xi is None:
            exposed_xi = _exposed_xi(melds, rules)
        if shape_score is None:
            shape_score = _shape_score(hand, rules)
        if route_score is None:
            route_score = _red_black_route_score(hand, melds, rules)
        improving_outs, weighted_shape_gain = (
            shape_draws
            if shape_draws is not None
            else _shape_improving_draws(
                hand,
                remaining_counts,
                rules,
            )
        )
        wildcard_count = tuple(hand).count(WILD_LABEL)
        total = (
            exposed_xi * self.weights.exposed_xi
            + shape_score * self.weights.shape
            + route_score * self.weights.route
            + wildcard_count * self.weights.wildcard
            + improving_outs * 14.0
            + weighted_shape_gain * 3.0
        )
        return IndependentValue(
            total=total,
            hu_outs=0,
            partition_outs=improving_outs,
            weighted_hu_score=weighted_shape_gain,
            exposed_xi=exposed_xi,
            shape_score=shape_score,
            route_score=route_score,
        )


class IndependentFastPressurePolicy(IndependentFastRolloutPolicy):
    name = "independent_fast_pressure"
    weights = IndependentPressurePolicy.weights


class IndependentFastDenialPolicy(IndependentFastRolloutPolicy):
    name = "independent_fast_denial"
    weights = IndependentDenialPolicy.weights


def _discardable_labels(hand: Sequence[str]) -> list[str]:
    counts = Counter(hand)
    return sorted(
        label
        for label, amount in counts.items()
        if label != WILD_LABEL and amount < 3
    )


def _remove_one(hand: Sequence[str], label: str) -> tuple[str, ...]:
    result = list(hand)
    result.remove(label)
    return tuple(result)


def _remove_many(hand: Sequence[str], labels: Sequence[str]) -> tuple[str, ...]:
    result = list(hand)
    for label in labels:
        result.remove(label)
    return tuple(result)


def _group_kind(group: Sequence[str]) -> str:
    labels = tuple(group)
    positions = [_LABEL_POSITION[label] for label in labels]
    suits = {item[0] for item in positions}
    ranks = sorted(item[1] for item in positions)
    if len(suits) == 2 and len(set(ranks)) == 1:
        return "mixed_same_rank"
    if ranks == [1, 2, 3]:
        return "special_123"
    if ranks == [2, 7, 10]:
        return "special_2710"
    if ranks == [1, 5, 10]:
        return "special_1510"
    return "normal_sequence"


def _shape_score(hand: Sequence[str], rules: Mapping[str, Any]) -> float:
    labels = tuple(sorted(label for label in hand if label != WILD_LABEL))
    allow_1510 = bool(rules.get("rules", {}).get("allow_1510", False))
    return _cached_shape_score(labels, allow_1510=allow_1510)


@lru_cache(maxsize=32_768)
def _cached_shape_score(
    labels: tuple[str, ...],
    *,
    allow_1510: bool,
) -> float:
    counts = Counter(labels)
    score = 0.0
    for amount in counts.values():
        if amount >= 3:
            score += 13.0
        elif amount == 2:
            score += 6.0
        else:
            score += 0.5
    for first, second, third in _shape_templates(allow_1510=allow_1510):
        held = (
            int(counts[first] > 0)
            + int(counts[second] > 0)
            + int(counts[third] > 0)
        )
        if held == 3:
            score += 5.0
        elif held == 2:
            score += 2.5
    for rank in range(10):
        small = SMALL_LABELS[rank]
        big = BIG_LABELS[rank]
        if counts[small] and counts[big]:
            score += 2.0
    return score


@lru_cache(maxsize=2)
def _shape_templates(*, allow_1510: bool) -> tuple[tuple[str, str, str], ...]:
    groups: list[tuple[str, str, str]] = []
    for labels in (SMALL_LABELS, BIG_LABELS):
        groups.extend(tuple(labels[start : start + 3]) for start in range(8))
        groups.append((labels[1], labels[6], labels[9]))
        if allow_1510:
            groups.append((labels[0], labels[4], labels[9]))
    return tuple(groups)


@lru_cache(maxsize=2)
def _shape_companion_indices(
    *,
    allow_1510: bool,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    templates = _shape_templates(allow_1510=allow_1510)
    return tuple(
        tuple(
            tuple(
                _LABEL_INDEX[item]
                for item in template
                if item != label
            )
            for template in templates
            if label in template
        )
        for label in NORMAL_LABELS
    )


def _shape_improving_draws(
    hand: Sequence[str],
    remaining_counts: Sequence[tuple[str, int]],
    rules: Mapping[str, Any],
) -> tuple[int, float]:
    counts = [0] * len(NORMAL_LABELS)
    for label in hand:
        index = _LABEL_INDEX.get(label)
        if index is not None:
            counts[index] += 1
    allow_1510 = bool(rules.get("rules", {}).get("allow_1510", False))
    return _shape_improving_draws_from_counts(
        tuple(counts),
        _indexed_remaining_counts(remaining_counts),
        allow_1510=allow_1510,
    )


def _indexed_remaining_counts(
    remaining_counts: Sequence[tuple[str, int]],
) -> tuple[int, ...]:
    indexed = [0] * len(NORMAL_LABELS)
    for label, raw_amount in remaining_counts:
        index = _LABEL_INDEX.get(label)
        if index is not None:
            indexed[index] = max(0, int(raw_amount))
    return tuple(indexed)


def _shape_improving_draws_from_counts(
    counts: tuple[int, ...],
    remaining_counts: tuple[int, ...],
    *,
    allow_1510: bool,
) -> tuple[int, float]:
    gains = _cached_shape_draw_gains(
        counts,
        allow_1510=allow_1510,
    )
    improving_outs = 0
    weighted_gain = 0.0
    for index, amount in enumerate(remaining_counts):
        if amount <= 0:
            continue
        gain = gains[index]
        if gain > 0.0:
            improving_outs += amount
            weighted_gain += amount * gain
    return improving_outs, weighted_gain


def _shape_removal_loss(
    counts: Sequence[int],
    label: str,
    *,
    allow_1510: bool,
) -> float:
    index = _LABEL_INDEX[label]
    before_amount = counts[index]
    if before_amount == 1:
        companions = _shape_companion_indices(allow_1510=allow_1510)
        loss = 0.5 + 2.5 * sum(
            counts[first] > 0 or counts[second] > 0
            for first, second in companions[index]
        )
        counterpart = index + 10 if index < 10 else index - 10
        return loss + (2.0 if counts[counterpart] > 0 else 0.0)
    if before_amount == 2:
        return 5.5
    if before_amount == 3:
        return 7.0
    return 0.0


@lru_cache(maxsize=32_768)
def _cached_shape_draw_gains(
    counts: tuple[int, ...],
    *,
    allow_1510: bool,
) -> tuple[float, ...]:
    companions = _shape_companion_indices(allow_1510=allow_1510)
    gains: list[float] = []
    for index in range(len(NORMAL_LABELS)):
        before_amount = counts[index]
        if before_amount == 0:
            gain = 0.5
        elif before_amount == 1:
            gain = 5.5
        elif before_amount == 2:
            gain = 7.0
        else:
            gain = 0.0
        if before_amount == 0:
            gain += 2.5 * sum(
                counts[first] > 0 or counts[second] > 0
                for first, second in companions[index]
            )
            counterpart = index + 10 if index < 10 else index - 10
            if counts[counterpart] > 0:
                gain += 2.0
        gains.append(gain)
    return tuple(gains)


def _shape_draw_gain(
    counts: Counter[str],
    label: str,
    *,
    allow_1510: bool,
) -> float:
    if label == WILD_LABEL:
        return 0.0
    before_amount = counts[label]
    gain = _count_shape_value(before_amount + 1) - _count_shape_value(before_amount)
    if before_amount == 0:
        for first, second in _shape_template_companions(
            label,
            allow_1510=allow_1510,
        ):
            held = int(counts[first] > 0) + int(counts[second] > 0)
            gain += _template_shape_value(held + 1) - _template_shape_value(held)
        suit, rank = _LABEL_POSITION[label]
        counterpart = (
            BIG_LABELS[rank - 1]
            if suit == "small"
            else SMALL_LABELS[rank - 1]
        )
        if counts[counterpart] > 0:
            gain += 2.0
    return gain


@lru_cache(maxsize=40)
def _shape_template_companions(
    label: str,
    *,
    allow_1510: bool,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        tuple(item for item in template if item != label)
        for template in _shape_templates(allow_1510=allow_1510)
        if label in template
    )


def _count_shape_value(amount: int) -> float:
    if amount >= 3:
        return 13.0
    if amount == 2:
        return 6.0
    if amount == 1:
        return 0.5
    return 0.0


def _template_shape_value(held: int) -> float:
    if held >= 3:
        return 5.0
    if held == 2:
        return 2.5
    return 0.0


def _exposed_xi(melds: Sequence[Any], rules: Mapping[str, Any]) -> int:
    xi_rules = rules.get("xi", {})
    total = 0
    aliases = {
        "hidden_quad": "ti",
        "quad": "pao",
        "hidden_triplet": "wei",
        "exact_triplet": "wei",
        "sequence": "normal_sequence",
    }
    for meld in melds:
        cards = tuple(getattr(meld, "cards", getattr(meld, "labels", ())))
        kind = aliases.get(
            str(getattr(meld, "kind", getattr(meld, "type", ""))),
            str(getattr(meld, "kind", getattr(meld, "type", ""))),
        )
        normal = [label for label in cards if label in _LABEL_POSITION]
        suits = {_LABEL_POSITION[label][0] for label in normal}
        suit = next(iter(suits)) if len(suits) == 1 else None
        if suit is not None:
            total += int(xi_rules.get(kind, {}).get(suit, 0))
    return total


def _red_black_route_score(
    hand: Sequence[str],
    melds: Sequence[Any],
    rules: Mapping[str, Any],
    *,
    meld_red_count: int | None = None,
) -> float:
    red = sum(label in RED_LABELS for label in hand)
    red += (
        _meld_red_count(melds)
        if meld_red_count is None
        else meld_red_count
    )
    return _red_black_route_score_from_counts(
        red_count=red,
        wildcard_count=tuple(hand).count(WILD_LABEL),
        rules=rules,
    )


def _red_black_route_score_from_counts(
    *,
    red_count: int,
    wildcard_count: int,
    rules: Mapping[str, Any],
) -> float:
    scoring = rules.get("scoring", {})
    black = int(scoring.get("black_hu_red_count", 0))
    one_red = int(scoring.get("one_red_hu_red_count", 1))
    red_min = int(scoring.get("red_hu_min_red", 13))
    low_distance = min(
        abs(red_count - black),
        abs(red_count - one_red),
    )
    high_distance = max(0, red_min - red_count - wildcard_count)
    return max(-2.0 * low_distance, -1.2 * high_distance)


def _meld_red_count(melds: Sequence[Any]) -> int:
    return sum(
        label in RED_LABELS
        for meld in melds
        for label in getattr(meld, "cards", getattr(meld, "labels", ()))
    )


def _public_discard_danger(label: str, view: Any) -> float:
    danger = 0.0
    player_count = len(view.all_melds)
    next_seat = (int(view.seat) + 1) % max(1, player_count)
    position = _LABEL_POSITION.get(label)
    for seat, melds in enumerate(view.all_melds):
        if seat == view.seat:
            continue
        for meld in melds:
            cards = tuple(getattr(meld, "cards", getattr(meld, "labels", ())))
            kind = str(getattr(meld, "kind", getattr(meld, "type", "")))
            if cards and cards[0] == label and kind in {"peng", "wei", "hidden_triplet"}:
                danger += 190.0
            if seat == next_seat and position is not None:
                related = sum(
                    1
                    for card in cards
                    if card in _LABEL_POSITION
                    and _LABEL_POSITION[card][0] == position[0]
                    and abs(_LABEL_POSITION[card][1] - position[1]) <= 2
                )
                danger += related * 7.0
        danger += len(melds) * 5.0
    if label in RED_LABELS:
        danger += 10.0
    if int(getattr(view, "stock_count", 0)) <= 8:
        danger *= 1.35
    return danger


__all__ = [
    "IndependentBalancedPolicy",
    "IndependentDenialPolicy",
    "IndependentExploitPolicy",
    "IndependentFastDenialPolicy",
    "IndependentFastPressurePolicy",
    "IndependentFastRolloutPolicy",
    "IndependentPolicyWeights",
    "IndependentPressurePolicy",
    "IndependentValue",
]
