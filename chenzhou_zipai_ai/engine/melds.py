"""Meld definitions and matching."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from engine.cards import WILD_LABEL, label_for, parse_label


@dataclass(frozen=True)
class MeldPattern:
    labels: tuple[str, ...]
    kind: str
    suit: str | None = None
    wildcards_used: int = 0
    wildcard_mapping: dict[str, str] = field(default_factory=dict)
    source_labels: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "cards": list(self.labels),
            "source_labels": list(self.source_labels or self.labels),
            "kind": self.kind,
            "type": self.kind,
            "suit": self.suit,
            "wildcards_used": self.wildcards_used,
            "wildcard_mapping": dict(self.wildcard_mapping),
        }


def is_special_2710(labels: list[str]) -> bool:
    parsed = [parse_label(label) for label in labels]
    if any(item is None for item in parsed):
        return False
    suits = {item[0] for item in parsed if item}
    ranks = sorted(item[1] for item in parsed if item)
    return len(suits) == 1 and ranks == [2, 7, 10]


def is_special_123(labels: list[str]) -> bool:
    parsed = [parse_label(label) for label in labels]
    if any(item is None for item in parsed):
        return False
    suits = {item[0] for item in parsed if item}
    ranks = sorted(item[1] for item in parsed if item)
    return len(suits) == 1 and ranks == [1, 2, 3]


def is_sequence(labels: list[str]) -> bool:
    parsed = [parse_label(label) for label in labels]
    if any(item is None for item in parsed):
        return False
    suits = {item[0] for item in parsed if item}
    ranks = sorted(item[1] for item in parsed if item)
    return len(suits) == 1 and len(ranks) == 3 and ranks == list(range(ranks[0], ranks[0] + 3))


def classify_meld(labels: list[str]) -> MeldPattern:
    normal = [label for label in labels if label != WILD_LABEL and label != "暗"]
    wildcard_count = sum(1 for label in labels if label == WILD_LABEL)
    hidden_count = sum(1 for label in labels if label == "暗")
    if hidden_count:
        return MeldPattern(tuple(labels), "hidden_triplet" if len(labels) == 3 else "hidden_meld", None, source_labels=tuple(labels))
    if len(set(normal)) == 1 and len(normal) + wildcard_count >= 3:
        kind = "ti" if len(labels) >= 4 else "peng"
        parsed = parse_label(normal[0]) if normal else None
        mapping = _wildcard_mapping(labels, [normal[0]] * wildcard_count) if normal else {}
        return MeldPattern(
            tuple(labels),
            kind,
            parsed[0] if parsed else None,
            wildcard_count,
            mapping,
            tuple(labels),
        )
    parsed_normal = [parse_label(label) for label in normal]
    if len(labels) == 3 and parsed_normal and all(item is not None for item in parsed_normal):
        ranks = {item[1] for item in parsed_normal if item is not None}
        if len(ranks) == 1 and len(set(normal)) == 2:
            return MeldPattern(tuple(labels), "mixed_same_rank", None, wildcard_count, source_labels=tuple(labels))
    completed = complete_three_card_meld(labels)
    if completed:
        if is_special_2710(list(completed.labels)):
            return MeldPattern(
                completed.labels,
                "special_2710",
                completed.suit,
                completed.wildcards_used,
                completed.wildcard_mapping,
                completed.source_labels,
            )
        if is_special_123(list(completed.labels)):
            return MeldPattern(
                completed.labels,
                "special_123",
                completed.suit,
                completed.wildcards_used,
                completed.wildcard_mapping,
                completed.source_labels,
            )
        if is_sequence(list(completed.labels)):
            return MeldPattern(
                completed.labels,
                "sequence",
                completed.suit,
                completed.wildcards_used,
                completed.wildcard_mapping,
                completed.source_labels,
            )
    return MeldPattern(tuple(labels), "unknown", None, wildcard_count, source_labels=tuple(labels))


def complete_three_card_meld(labels: list[str]) -> MeldPattern | None:
    return _complete_three_card_meld(tuple(labels))


@lru_cache(maxsize=4096)
def _complete_three_card_meld(labels: tuple[str, ...]) -> MeldPattern | None:
    candidates = _complete_three_card_melds(labels)
    return candidates[0] if candidates else None


def complete_three_card_melds(labels: list[str]) -> tuple[MeldPattern, ...]:
    """Return every legal wildcard resolution for one three-card source group."""

    return _complete_three_card_melds(tuple(labels))


@lru_cache(maxsize=4096)
def _complete_three_card_melds(
    labels: tuple[str, ...],
) -> tuple[MeldPattern, ...]:
    if len(labels) != 3:
        return ()
    wildcards = labels.count(WILD_LABEL)
    fixed = [label for label in labels if label != WILD_LABEL]
    results: list[MeldPattern] = []
    seen: set[tuple[tuple[str, ...], str, str | None]] = set()

    def add(pattern: MeldPattern) -> None:
        key = (tuple(pattern.labels), pattern.kind, pattern.suit)
        if key not in seen:
            seen.add(key)
            results.append(pattern)

    if not fixed:
        add(
            MeldPattern(
                (WILD_LABEL, WILD_LABEL, WILD_LABEL),
                "wild_triplet",
                None,
                wildcards,
                source_labels=labels,
            )
        )
    parsed_fixed = [parse_label(label) for label in fixed]
    if parsed_fixed and all(item is not None for item in parsed_fixed):
        ranks = {item[1] for item in parsed_fixed if item is not None}
        suits = {item[0] for item in parsed_fixed if item is not None}
        if len(ranks) == 1 and len(suits) >= 2 and len(fixed) + wildcards == 3:
            rank = next(iter(ranks))
            replacement_suit = parsed_fixed[0][0] if parsed_fixed else "small"
            replacements = [label_for(replacement_suit, rank)] * wildcards
            completed_labels = tuple([*fixed, *replacements])
            add(
                MeldPattern(
                    completed_labels,
                    "mixed_same_rank",
                    None,
                    wildcards,
                    _wildcard_mapping(list(labels), replacements),
                    labels,
                )
            )
    candidates = _candidate_three_sets()
    fixed_counter = Counter(fixed)
    for candidate in candidates:
        candidate_counter = Counter(candidate)
        missing = sum(max(0, fixed_counter[label] - candidate_counter[label]) for label in fixed_counter)
        if missing:
            continue
        needed = 3 - len(fixed)
        if needed <= wildcards:
            parsed = parse_label(candidate[0])
            replacements = _candidate_replacements(candidate, fixed)
            add(
                MeldPattern(
                    tuple(candidate),
                    "complete",
                    parsed[0] if parsed else None,
                    wildcards,
                    _wildcard_mapping(list(labels), replacements),
                    labels,
                )
            )
    return tuple(results)


def _candidate_replacements(candidate: tuple[str, ...], fixed: list[str]) -> list[str]:
    remaining = Counter(candidate)
    for label in fixed:
        remaining[label] -= 1
    replacements: list[str] = []
    for label, amount in remaining.items():
        replacements.extend([label] * max(0, amount))
    return replacements


def _wildcard_mapping(labels: list[str], replacements: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    index = 0
    for position, label in enumerate(labels, start=1):
        if label != WILD_LABEL:
            continue
        replacement = replacements[index] if index < len(replacements) else WILD_LABEL
        mapping[f"{WILD_LABEL}#{position}"] = replacement
        index += 1
    return mapping


@lru_cache(maxsize=1)
def _candidate_three_sets() -> tuple[tuple[str, str, str], ...]:
    result: list[tuple[str, str, str]] = []
    for suit in ("small", "big"):
        for start in range(1, 9):
            result.append(tuple(label_for(suit, rank) for rank in range(start, start + 3)))
        result.append(tuple(label_for(suit, rank) for rank in (2, 7, 10)))
        for rank in range(1, 11):
            label = label_for(suit, rank)
            result.append((label, label, label))
    return tuple(result)
