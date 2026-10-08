"""Hu checking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import os
from itertools import combinations_with_replacement

from engine.cards import WILD_LABEL, normal_labels, normalize_cards, parse_label
from engine.melds import (
    classify_meld,
    complete_three_card_meld,
    complete_three_card_melds,
)
from engine.rules import load_rules, min_xi
from engine.xi_calculator import meld_xi, total_xi


@dataclass(frozen=True)
class HuBreakdown:
    can_hu: bool
    total_xi: int
    min_xi: int
    hand_groups: list[list[str]] = field(default_factory=list)
    existing_groups: list[list[str]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    partition_complete: bool = False
    group_count: int = 0

    @property
    def best_partition(self) -> list[list[str]]:
        return [*self.existing_groups, *self.hand_groups]

    def to_dict(self) -> dict:
        return {
            "can_hu": self.can_hu,
            "total_xi": self.total_xi,
            "min_xi": self.min_xi,
            "best_partition": self.best_partition,
            "hand_groups": self.hand_groups,
            "existing_groups": self.existing_groups,
            "reasons": self.reasons,
            "partition_complete": self.partition_complete,
            "group_count": self.group_count,
        }


def can_form_three_card_meld(labels: list[str]) -> bool:
    return complete_three_card_meld(labels) is not None


def best_grouping(
    labels: list[str],
    *,
    config_path: str = "config/rules.yaml",
    required_pair_count: int = 0,
) -> list[list[str]]:
    normalized = normalize_cards(labels)
    if required_pair_count not in {0, 1}:
        return []
    pair_cards = required_pair_count * 2
    if (
        len(normalized) != len(labels)
        or len(normalized) < pair_cards
        or (len(normalized) - pair_cards) % 3
    ):
        return []
    result = _solve_grouping_counts(
        _count_state(normalized),
        str(config_path),
        required_pair_count,
    )
    return [list(group) for group in result[1]] if result is not None else []


def best_grouping_normalized(
    labels: list[str] | tuple[str, ...],
    *,
    config_path: str = "config/rules.yaml",
    required_pair_count: int = 0,
) -> tuple[tuple[str, ...], ...]:
    """Return the exact grouping for engine-owned, already-normalized labels."""
    if required_pair_count not in {0, 1}:
        return ()
    pair_cards = required_pair_count * 2
    if len(labels) < pair_cards or (len(labels) - pair_cards) % 3:
        return ()
    counts = [0] * len(_GROUPING_LABELS)
    try:
        for label in labels:
            counts[_GROUPING_INDEX[label]] += 1
    except KeyError:
        return ()
    result = _solve_grouping_counts(
        tuple(counts),
        str(config_path),
        required_pair_count,
    )
    return result[1] if result is not None else ()


def _state_key(counts: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((label, amount) for label, amount in counts.items() if amount > 0))


def _count_state(labels: list[str]) -> tuple[int, ...]:
    counts = [0] * len(_GROUPING_LABELS)
    for label in labels:
        counts[_GROUPING_INDEX[label]] += 1
    return tuple(counts)


_GROUPING_CACHE_SIZE = 32_768 if os.environ.get("AIZIPAI_MOBILE_RUNTIME") == "1" else 200_000


@lru_cache(maxsize=_GROUPING_CACHE_SIZE)
def _solve_grouping_counts(
    counts: tuple[int, ...],
    config_path: str,
    required_pairs: int,
) -> tuple[int, tuple[tuple[str, ...], ...]] | None:
    first_index = None
    for index, amount in enumerate(counts):
        if amount > 0:
            first_index = index
            break
    if first_index is None:
        return (0, ()) if required_pairs == 0 else None

    best: tuple[int, tuple[tuple[str, ...], ...]] | None = None
    for group, required in _indexed_legal_groups()[first_index]:
        enough_cards = True
        for index, amount in required:
            if counts[index] < amount:
                enough_cards = False
                break
        if not enough_cards:
            continue
        remaining = list(counts)
        for index, amount in required:
            remaining[index] -= amount
        child = _solve_grouping_counts(
            tuple(remaining),
            config_path,
            required_pairs,
        )
        if child is None:
            continue
        score = concealed_group_xi(
            list(group),
            config_path=config_path,
        ) + child[0]
        candidate = (score, (group, *child[1]))
        if best is None or candidate[0] > best[0]:
            best = candidate

    if required_pairs:
        pair_groups: list[tuple[int, ...]] = []
        if counts[first_index] >= 2:
            pair_groups.append((first_index, first_index))
        wildcard_index = _GROUPING_INDEX[WILD_LABEL]
        if first_index != wildcard_index and counts[wildcard_index] >= 1:
            pair_groups.append((first_index, wildcard_index))
        for pair_indices in pair_groups:
            remaining = list(counts)
            for index in pair_indices:
                remaining[index] -= 1
            child = _solve_grouping_counts(
                tuple(remaining),
                config_path,
                required_pairs - 1,
            )
            if child is None:
                continue
            pair = tuple(_GROUPING_LABELS[index] for index in pair_indices)
            candidate = (child[0], (pair, *child[1]))
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


@lru_cache(maxsize=1)
def _indexed_legal_groups() -> tuple[
    tuple[
        tuple[
            tuple[str, ...],
            tuple[tuple[int, int], ...],
        ],
        ...,
    ],
    ...,
]:
    by_label = _legal_source_groups_by_label()
    return tuple(
        tuple(
            (
                tuple(group),
                tuple(
                    (_GROUPING_INDEX[label], amount)
                    for label, amount in required
                ),
            )
            for group, required in by_label[label]
        )
        for label in _GROUPING_LABELS
    )


@lru_cache(maxsize=_GROUPING_CACHE_SIZE)
def _solve_grouping(
    state: tuple[tuple[str, int], ...],
    config_path: str,
    required_pairs: int,
) -> tuple[int, tuple[tuple[str, ...], ...]] | None:
    counts = Counter(dict(state))
    if not counts:
        return (0, ()) if required_pairs == 0 else None
    first = min(counts, key=_label_sort_key)
    best: tuple[int, tuple[tuple[str, ...], ...]] | None = None
    for group in _candidate_groups_with(first, counts):
        needed = Counter(group)
        if any(counts[label] < amount for label, amount in needed.items()):
            continue
        remaining = counts.copy()
        remaining.subtract(needed)
        child = _solve_grouping(_state_key(remaining), config_path, required_pairs)
        if child is None:
            continue
        score = concealed_group_xi(group, config_path=config_path) + child[0]
        candidate = (score, (tuple(group), *child[1]))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if required_pairs:
        for pair in _candidate_pairs_with(first, counts):
            remaining = counts.copy()
            remaining.subtract(Counter(pair))
            child = _solve_grouping(_state_key(remaining), config_path, required_pairs - 1)
            if child is None:
                continue
            candidate = (child[0], (tuple(pair), *child[1]))
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def greedy_grouping(labels: list[str]) -> list[list[str]]:
    counts = Counter(labels)
    groups: list[list[str]] = []
    while True:
        candidates: list[tuple[int, list[str]]] = []
        for first, amount in counts.items():
            if amount <= 0:
                continue
            for group in _candidate_groups_with(first, counts):
                candidates.append((meld_xi(group), group))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], _group_priority(item[1])), reverse=True)
        _score, group = candidates[0]
        if _score <= 0 and len(groups) >= len(labels) // 3:
            break
        for label in group:
            counts[label] -= 1
        groups.append(group)
    return groups


def _group_priority(group: list[str]) -> int:
    if set(group) & {"王"}:
        return 1
    return 2


def hand_xi_potential(labels: list[str], *, config_path: str = "config/rules.yaml") -> int:
    return sum(
        concealed_group_xi(group, config_path=config_path)
        for group in best_grouping(labels, config_path=config_path)
    )


def can_hu(
    labels: list[str],
    *,
    existing_xi: int = 0,
    existing_group_count: int = 0,
    existing_quad_count: int = 0,
    config_path: str = "config/rules.yaml",
) -> bool:
    config = load_rules(config_path)
    groups = best_grouping(
        labels,
        config_path=config_path,
        required_pair_count=int(existing_quad_count >= 1),
    )
    required_groups = int(config.get("rules", {}).get("required_meld_groups", 7))
    complete = sum(len(group) for group in groups) == len(normalize_cards(labels))
    return (
        complete
        and len(groups) + existing_group_count == required_groups
        and existing_xi + sum(concealed_group_xi(group, config_path=config_path) for group in groups) >= min_xi(config)
    )


def explain_hu(
    labels: list[str],
    *,
    existing_groups: list[list[str]] | None = None,
    config_path: str = "config/rules.yaml",
) -> HuBreakdown:
    config = load_rules(config_path)
    existing = existing_groups or []
    groups = best_grouping(
        labels,
        config_path=config_path,
        required_pair_count=int(any(len(group) == 4 for group in existing)),
    )
    hand_xi = sum(concealed_group_xi(group, config_path=config_path) for group in groups)
    existing_xi = total_xi(existing, config_path=config_path)
    total = hand_xi + existing_xi
    required = min_xi(config)
    required_groups = int(config.get("rules", {}).get("required_meld_groups", 7))
    normalized = normalize_cards(labels)
    hand_complete = sum(len(group) for group in groups) == len(normalized)
    existing_complete = all(_is_existing_group_legal(group) for group in existing)
    group_count = len(groups) + len(existing)
    partition_complete = hand_complete and existing_complete and group_count == required_groups
    reasons = [f"手牌组合胡息 {hand_xi}", f"落地组合胡息 {existing_xi}"]
    if not hand_complete:
        reasons.append("手牌不能被合法门子完整覆盖")
    if not existing_complete:
        reasons.append("落地牌中存在无法确认的门子")
    if group_count != required_groups:
        reasons.append(f"当前共 {group_count} 方门子，需要 {required_groups} 方")
    if partition_complete and total >= required:
        reasons.append(f"总胡息 {total} 达到起胡 {required}")
    else:
        reasons.append(f"当前不能胡：总胡息 {total}，起胡 {required}")
    return HuBreakdown(
        can_hu=partition_complete and total >= required,
        total_xi=total,
        min_xi=required,
        hand_groups=groups,
        existing_groups=existing,
        reasons=reasons,
        partition_complete=partition_complete,
        group_count=group_count,
    )


def check_hu(state, allocation=None, rules: dict | None = None):
    from ai.pro_brain import check_hu as _check_hu

    return _check_hu(state, allocation, rules)


def _candidate_groups_with(first: str, counts: Counter[str]) -> list[list[str]]:
    return [
        list(group)
        for group, required in _legal_source_groups_by_label().get(first, ())
        if all(counts[label] >= amount for label, amount in required)
    ]


@lru_cache(maxsize=1)
def _legal_source_groups_by_label() -> dict[
    str,
    tuple[
        tuple[
            tuple[str, ...],
            tuple[tuple[str, int], ...],
        ],
        ...,
    ],
]:
    labels = sorted((*normal_labels(), WILD_LABEL), key=_label_sort_key)
    grouped: dict[
        str,
        list[
            tuple[
                tuple[str, ...],
                tuple[tuple[str, int], ...],
            ]
        ],
    ] = {label: [] for label in labels}
    for group in combinations_with_replacement(labels, 3):
        if complete_three_card_meld(list(group)) is None:
            continue
        required = tuple(Counter(group).items())
        for label in dict.fromkeys(group):
            grouped[label].append((group, required))
    return {
        label: tuple(candidates)
        for label, candidates in grouped.items()
    }


def _candidate_pairs_with(first: str, counts: Counter[str]) -> list[list[str]]:
    pairs: list[list[str]] = []
    if counts[first] >= 2:
        pairs.append([first, first])
    if first != WILD_LABEL and counts[WILD_LABEL] >= 1:
        pairs.append([first, WILD_LABEL])
    return pairs


def _label_sort_key(label: str) -> tuple[int, int, str]:
    if label == WILD_LABEL:
        return (99, 99, label)
    parsed = parse_label(label)
    if parsed is None:
        return (98, 98, label)
    suit, rank = parsed
    return (rank, 0 if suit == "small" else 1, label)


_GROUPING_LABELS = tuple(
    sorted((*normal_labels(), WILD_LABEL), key=_label_sort_key)
)
_GROUPING_INDEX = {
    label: index for index, label in enumerate(_GROUPING_LABELS)
}


def _is_existing_group_legal(group: list[str]) -> bool:
    pattern = classify_meld(group)
    return pattern.kind not in {"unknown", "hidden_meld"}


def concealed_group_xi(
    group: list[str],
    *,
    config_path: str = "config/rules.yaml",
    rules: dict | None = None,
) -> int:
    config = rules or load_rules(config_path)
    wildcard_count = group.count(WILD_LABEL)
    if (
        wildcard_count
        and not bool(
            config.get("wildcard", {}).get(
                "wildcard_groups_count_xi",
                True,
            )
        )
    ):
        return 0
    normal = [label for label in group if label != WILD_LABEL]
    if normal and len(set(normal)) == 1 and len(group) == 3:
        return meld_xi(group, kind="wei", config_path=config_path, rules=config)
    if normal and len(set(normal)) == 1 and len(group) == 4:
        return meld_xi(group, kind="ti", config_path=config_path, rules=config)
    if wildcard_count and len(group) == 3:
        resolved_values = []
        for resolved_pattern in complete_three_card_melds(group):
            resolved = list(resolved_pattern.labels)
            if WILD_LABEL in resolved:
                continue
            kind = "wei" if len(set(resolved)) == 1 else None
            resolved_values.append(
                meld_xi(
                    resolved,
                    kind=kind,
                    config_path=config_path,
                    rules=config,
                )
            )
        if resolved_values:
            return max(resolved_values)
    return meld_xi(group, config_path=config_path, rules=config)
