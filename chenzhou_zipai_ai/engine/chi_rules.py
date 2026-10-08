"""Exact Chenzhou chi and mandatory compare-plan enumeration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from engine.cards import WILD_LABEL, label_for, normalize_card_label, parse_label
from engine.melds import classify_meld


@dataclass(frozen=True)
class ChiPlan:
    initial_group: tuple[str, str, str]
    compare_groups: tuple[tuple[str, str, str], ...]
    consumed_from_hand: tuple[str, ...]

    @property
    def groups(self) -> tuple[tuple[str, str, str], ...]:
        return (self.initial_group, *self.compare_groups)

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_group": list(self.initial_group),
            "compare_groups": [list(group) for group in self.compare_groups],
            "consumed_from_hand": list(self.consumed_from_hand),
        }


def enumerate_chi_plans(
    hand: list[str],
    external_label: str,
    *,
    allow_1510: bool = False,
) -> list[ChiPlan]:
    """Enumerate initial chi plus every mandatory compare group.

    The external card is available once. Every same-label card left in hand after
    the initial group must be consumed by another legal chi group.
    """

    external = normalize_card_label(external_label)
    normalized_hand = [normalize_card_label(label) for label in hand]
    if not external or external == WILD_LABEL or any(not label for label in normalized_hand):
        return []
    if normalized_hand.count(external) >= 3:
        return []

    pool = Counter(normalized_hand)
    pool[external] += 1
    initial_groups = _groups_containing(
        pool,
        external,
        allow_1510=allow_1510,
        max_external_count=3,
    )
    plans: list[ChiPlan] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    for initial in initial_groups:
        remaining = _subtract_group(pool, initial)
        if remaining is None:
            continue
        for compares in _complete_compare_groups(remaining, external, allow_1510=allow_1510):
            groups = (initial, *compares)
            key = tuple(tuple(sorted(group, key=_label_sort_key)) for group in groups)
            if key in seen:
                continue
            seen.add(key)
            consumed = [label for group in groups for label in group]
            consumed.remove(external)
            plans.append(
                ChiPlan(
                    initial_group=initial,
                    compare_groups=compares,
                    consumed_from_hand=tuple(consumed),
                )
            )
    return sorted(
        plans,
        key=lambda plan: (
            len(plan.compare_groups),
            tuple(_label_sort_key(label) for label in plan.consumed_from_hand),
        ),
    )


def _complete_compare_groups(
    counts: Counter[str],
    external: str,
    *,
    allow_1510: bool,
) -> list[tuple[tuple[str, str, str], ...]]:
    if counts.get(external, 0) <= 0:
        return [()]
    groups = _groups_containing(
        counts,
        external,
        allow_1510=allow_1510,
        max_external_count=2,
    )
    results: list[tuple[tuple[str, str, str], ...]] = []
    for group in groups:
        remaining = _subtract_group(counts, group)
        if remaining is None:
            continue
        for suffix in _complete_compare_groups(remaining, external, allow_1510=allow_1510):
            results.append((group, *suffix))
    return results


def _groups_containing(
    counts: Counter[str],
    external: str,
    *,
    allow_1510: bool,
    max_external_count: int,
) -> list[tuple[str, str, str]]:
    if counts.get(external, 0) <= 0 or counts.get(external, 0) > max_external_count:
        return []
    usable = counts.copy()
    for label, amount in list(usable.items()):
        if label != external and amount >= 3:
            usable[label] = 0

    parsed = parse_label(external)
    if parsed is None:
        return []
    suit, rank = parsed
    opposite = "big" if suit == "small" else "small"
    candidates: list[list[str]] = []
    for start in range(max(1, rank - 2), min(rank, 8) + 1):
        candidates.append([label_for(suit, value) for value in (start, start + 1, start + 2)])
    if rank in {2, 7, 10}:
        candidates.append([label_for(suit, value) for value in (2, 7, 10)])
    if allow_1510 and rank in {1, 5, 10}:
        candidates.append([label_for(suit, value) for value in (1, 5, 10)])
    candidates.extend(
        [
            [external, external, label_for(opposite, rank)],
            [external, label_for(opposite, rank), label_for(opposite, rank)],
        ]
    )

    groups: list[tuple[str, str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        kind = classify_meld(candidate).kind
        is_optional_1510 = allow_1510 and _is_same_suit_ranks(candidate, {1, 5, 10})
        if external not in candidate or (
            kind not in {"sequence", "mixed_same_rank", "special_123", "special_2710"}
            and not is_optional_1510
        ):
            continue
        needed = Counter(candidate)
        if any(usable.get(label, 0) < amount for label, amount in needed.items()):
            continue
        group = tuple(sorted(candidate, key=_label_sort_key))
        if group not in seen:
            seen.add(group)
            groups.append(group)
    return groups


def _subtract_group(counts: Counter[str], group: tuple[str, str, str]) -> Counter[str] | None:
    needed = Counter(group)
    if any(counts.get(label, 0) < amount for label, amount in needed.items()):
        return None
    remaining = counts.copy()
    remaining.subtract(needed)
    return +remaining


def _label_sort_key(label: str) -> tuple[int, int, str]:
    parsed = parse_label(label)
    if parsed is None:
        return (99, 99, label)
    suit, rank = parsed
    return (rank, 0 if suit == "small" else 1, label)


def _is_same_suit_ranks(labels: list[str], ranks: set[int]) -> bool:
    parsed = [parse_label(label) for label in labels]
    return (
        all(item is not None for item in parsed)
        and len({item[0] for item in parsed if item is not None}) == 1
        and {item[1] for item in parsed if item is not None} == ranks
    )


__all__ = ["ChiPlan", "enumerate_chi_plans"]
