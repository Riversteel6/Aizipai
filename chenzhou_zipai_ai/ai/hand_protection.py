"""Hand structure protection analysis for safer discard decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from engine.cards import (
    BIG_LABELS,
    RED_LABELS,
    WILD_LABEL,
    is_big as _is_big_label,
    is_small as _is_small_label,
    label_for,
    normalize_cards,
    parse_label,
    same_rank as _same_rank,
)


SMALL_LABELS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
BIG_LABELS = BIG_LABELS


@dataclass(frozen=True)
class ProtectedMeld:
    type: str
    cards: tuple[str, ...]
    labels: tuple[str, ...]
    protect_level: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "cards": list(self.cards),
            "labels": list(self.labels),
            "protect_level": self.protect_level,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HandProtection:
    counts: dict[str, int]
    hard_protected: set[str]
    soft_protected: set[str]
    orphan_cards: list[str]
    low_value_singles: list[str]
    structure_notes: dict[str, list[str]] = field(default_factory=dict)
    protected_melds: list[ProtectedMeld] = field(default_factory=list)


@dataclass(frozen=True)
class StructureAllocation:
    normalized_hand: list[str]
    total_counts: dict[str, int]
    locked_melds: list[ProtectedMeld]
    locked_counts: dict[str, int]
    free_counts_after_locked: dict[str, int]
    soft_melds: list[ProtectedMeld]
    soft_counts: dict[str, int]
    orphan_cards: list[str]
    weak_potential_cards: list[str]
    protected_cards: list[str]
    allocation_notes: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_hand": list(self.normalized_hand),
            "total_counts": dict(self.total_counts),
            "locked_melds": [item.to_dict() for item in self.locked_melds],
            "locked_counts": dict(self.locked_counts),
            "free_counts_after_locked": dict(self.free_counts_after_locked),
            "soft_melds": [item.to_dict() for item in self.soft_melds],
            "soft_counts": dict(self.soft_counts),
            "orphan_cards": list(self.orphan_cards),
            "weak_potential_cards": list(self.weak_potential_cards),
            "protected_cards": list(self.protected_cards),
            "allocation_notes": {
                key: list(values) for key, values in self.allocation_notes.items()
            },
        }


def rank_of(card: str) -> int | None:
    parsed = parse_label(card)
    return parsed[1] if parsed is not None else None


def is_big(card: str) -> bool:
    return _is_big_label(card)


def is_small(card: str) -> bool:
    return _is_small_label(card)


def same_rank(left: str, right: str) -> bool:
    return _same_rank(left, right)


def analyze_complete_melds(hand: list[str], rules: dict | None = None) -> list[ProtectedMeld]:
    normalized = normalize_cards(hand)
    rules = rules or {}
    counter = Counter(normalized)
    wild_count = counter.get(WILD_LABEL, 0)
    melds: list[ProtectedMeld] = []
    seen = set[tuple[str, tuple[str, ...], str]]()

    def add_meld(
        meld_type: str,
        cards: list[str],
        reason: str,
        protect_level: str | None = None,
    ) -> None:
        cards_tuple = tuple(cards)
        protect = protect_level or _meld_protect_level(meld_type, rules)
        key = (meld_type, tuple(sorted(cards_tuple)), protect)
        if key in seen:
            return
        seen.add(key)
        labels = tuple(sorted({label for label in cards_tuple}))
        melds.append(
            ProtectedMeld(
                type=meld_type,
                cards=cards_tuple,
                labels=labels,
                protect_level=protect,
                reason=reason,
            )
        )

    # Exact triplet / quad of same label.
    for label, amount in sorted(counter.items()):
        if label == WILD_LABEL:
            continue
        if amount >= 4:
            add_meld("exact_quad", [label] * 4, f"{label}x{amount} 同点四张")
        elif amount >= 3:
            add_meld("exact_triplet", [label] * 3, f"{label}x{amount} 同点三张")

    # Same-rank mixed triples across small/big variants.
    rank_groups: dict[int, list[str]] = defaultdict(list)
    for label, amount in counter.items():
        if label == WILD_LABEL:
            continue
        parsed = parse_label(label)
        if parsed is None:
            continue
        _, rank = parsed
        rank_groups[rank].extend([label] * amount)
    for rank, labels in sorted(rank_groups.items()):
        if len(set(labels)) <= 1:
            continue
        if len(labels) < 3:
            continue
        cards = labels[:3]
        add_meld(
            "mixed_same_rank_triplet",
            cards,
            f"同点混搭 rank={rank}: {''.join(cards)}",
            protect_level=_mixed_rank_protect_level(rules),
        )

    # Concrete sequence and special complete melds in each suit.
    for suit in ("small", "big"):
        suit_ranks = {
            parsed[1]
            for label in counter
            for parsed in [parse_label(label)]
            if parsed is not None and parsed[0] == suit
        }
        if {1, 2, 3} <= suit_ranks:
            labels = tuple(label_for(suit, rank) for rank in (1, 2, 3))
            add_meld("special_123", list(labels), "完整一二三" if suit == "small" else "完整壹贰叁")
        if {2, 7, 10} <= suit_ranks:
            labels = tuple(label_for(suit, rank) for rank in (2, 7, 10))
            add_meld("special_2710", list(labels), "完整二七十" if suit == "small" else "完整贰柒拾")
        for start in range(1, 9):
            needed = tuple(label_for(suit, rank) for rank in (start, start + 1, start + 2))
            if all(counter.get(label, 0) >= 1 for label in needed):
                add_meld("normal_sequence", list(needed), f"{''.join(needed)}")

        if wild_count <= 0:
            continue
        for start in range(1, 9):
            needed = tuple(label_for(suit, rank) for rank in (start, start + 1, start + 2))
            present = [label for label in needed if counter.get(label, 0) > 0]
            missing = [label for label in needed if counter.get(label, 0) == 0]
            if len(missing) == 1 and len(present) == 2 and wild_count >= 1:
                add_meld(
                    "wildcard_meld",
                    sorted(present) + [WILD_LABEL],
                    f"{''.join(needed)} 缺一张补王",
                    protect_level="soft",
                )
        if {1, 2} <= suit_ranks and counter.get(label_for(suit, 3), 0) == 0 and wild_count >= 1:
            add_meld(
                "wildcard_meld",
                [label_for(suit, 1), label_for(suit, 2), WILD_LABEL],
                f"{label_for(suit, 1)}{label_for(suit, 2)}缺一张补王",
                protect_level="soft",
            )
        if {1, 3} <= suit_ranks and counter.get(label_for(suit, 2), 0) == 0 and wild_count >= 1:
            add_meld(
                "wildcard_meld",
                [label_for(suit, 1), label_for(suit, 3), WILD_LABEL],
                f"{label_for(suit, 1)}{label_for(suit, 3)}缺一张补王",
                protect_level="soft",
            )

        if {2, 3} <= suit_ranks and counter.get(label_for(suit, 1), 0) == 0 and wild_count >= 1:
            add_meld(
                "wildcard_meld",
                [label_for(suit, 2), label_for(suit, 3), WILD_LABEL],
                f"{label_for(suit, 2)}{label_for(suit, 3)}缺一张补王",
                protect_level="soft",
            )
        if {2, 7} <= suit_ranks and counter.get(label_for(suit, 10), 0) == 0 and wild_count >= 1:
            add_meld(
                "wildcard_meld",
                [label_for(suit, 2), label_for(suit, 7), WILD_LABEL],
                f"{label_for(suit, 2)}{label_for(suit, 7)}缺一张补王",
                protect_level="soft",
            )
        if {2, 10} <= suit_ranks and counter.get(label_for(suit, 7), 0) == 0 and wild_count >= 1:
            add_meld(
                "wildcard_meld",
                [label_for(suit, 2), label_for(suit, 10), WILD_LABEL],
                f"{label_for(suit, 2)}{label_for(suit, 10)}缺一张补王",
                protect_level="soft",
            )
        if {7, 10} <= suit_ranks and counter.get(label_for(suit, 2), 0) == 0 and wild_count >= 1:
            add_meld(
                "wildcard_meld",
                [label_for(suit, 7), label_for(suit, 10), WILD_LABEL],
                f"{label_for(suit, 7)}{label_for(suit, 10)}缺一张补王",
                protect_level="soft",
            )

    if wild_count > 0:
        for label, amount in counter.items():
            if label == WILD_LABEL:
                continue
            if amount == 2:
                add_meld(
                    "wildcard_meld",
                    [label, label, WILD_LABEL],
                    f"{label}x2+王可补刻",
                    protect_level="soft",
                )

    return melds


def _meld_required_counts(meld: ProtectedMeld) -> Counter:
    return Counter(meld.cards)


def _can_allocate(counts: Counter[str], meld: ProtectedMeld) -> bool:
    required = _meld_required_counts(meld)
    return all(counts.get(label, 0) >= amount for label, amount in required.items())


def _consume_meld(counts: Counter[str], meld: ProtectedMeld) -> bool:
    required = _meld_required_counts(meld)
    if not _can_allocate(counts, meld):
        return False
    for label, amount in required.items():
        counts[label] -= amount
    return True


def _hard_meld_priority(meld: ProtectedMeld) -> tuple[int, str, str]:
    hard_priority = {
        "exact_quad": 0,
        "exact_triplet": 1,
    }
    return (hard_priority.get(meld.type, 10), meld.type, "".join(meld.cards))


def _soft_meld_priority(meld: ProtectedMeld) -> tuple[int, str, str]:
    soft_priority = {
        "special_123": 0,
        "special_2710": 1,
        "normal_sequence": 2,
        "mixed_same_rank_triplet": 3,
        "wildcard_meld": 4,
    }
    return (soft_priority.get(meld.type, 10), meld.type, "".join(meld.cards))


def _allocate_hand_structures(
    normalized: list[str],
    rules: dict | None = None,
) -> StructureAllocation:
    from ai.pro_brain import allocate_hand_structures as allocate_professional_structures
    from ai.pro_brain import build_decision_context as build_professional_context

    context = build_professional_context(normalize_cards(normalized), rules=rules)
    allocation = allocate_professional_structures(context)
    notes: dict[str, list[str]] = defaultdict(list)
    for item in allocation.allocation_notes:
        bucket = "hard" if "locked" in item or "hard" in item else "weak"
        notes[bucket].append(item)
    soft_counts = Counter(
        label for meld in allocation.soft_melds for label in meld.labels
    )
    weak_labels = [
        label for meld in allocation.weak_potentials for label in meld.labels
    ]
    return StructureAllocation(
        normalized_hand=list(allocation.normalized_hand),
        total_counts=dict(allocation.total_counts),
        locked_melds=[_legacy_meld_from_professional(meld) for meld in allocation.locked_melds],
        locked_counts=dict(allocation.locked_counts),
        free_counts_after_locked=dict(allocation.free_counts_after_locked),
        soft_melds=[_legacy_meld_from_professional(meld) for meld in allocation.soft_melds],
        soft_counts=dict(soft_counts),
        orphan_cards=list(allocation.orphan_cards),
        weak_potential_cards=weak_labels,
        protected_cards=sorted(set(allocation.hard_protected_labels) | set(allocation.soft_protected_labels)),
        allocation_notes=notes,
    )


def analyze_hand_protection(hand: list[str], rules=None) -> HandProtection:
    normalized = normalize_cards(hand)
    counts = Counter(normalized)
    rules = rules or {}

    allocation = _allocate_hand_structures(normalized, rules=rules)
    protected_melds = [*allocation.locked_melds, *allocation.soft_melds]
    hard_protected = {label for meld in allocation.locked_melds for label in meld.labels}
    soft_protected = {label for meld in allocation.soft_melds for label in meld.labels}
    free_counts = Counter(allocation.free_counts_after_locked)
    used_soft_counts = Counter(label for meld in allocation.soft_melds for label in meld.labels)
    for label, amount in used_soft_counts.items():
        free_counts[label] = max(0, free_counts.get(label, 0) - amount)
    free_counts = Counter({label: amount for label, amount in free_counts.items() if amount > 0})
    orphan_cards = list(allocation.orphan_cards)
    low_value_singles = list(allocation.orphan_cards)

    return HandProtection(
        counts=dict(counts),
        hard_protected=hard_protected,
        soft_protected=soft_protected,
        orphan_cards=orphan_cards,
        low_value_singles=low_value_singles,
        structure_notes=dict(allocation.allocation_notes),
        protected_melds=protected_melds,
    )


def analyze_structure_allocation(
    hand: list[str],
    rules: dict | None = None,
) -> StructureAllocation:
    """Expose structure allocation for downstream logging and validation."""

    normalized = normalize_cards(hand)
    return _allocate_hand_structures(normalized, rules=rules)


def _legacy_meld_from_professional(meld) -> ProtectedMeld:
    return ProtectedMeld(
        type=meld.type,
        cards=tuple(meld.labels),
        labels=tuple(sorted(set(meld.labels))),
        protect_level=meld.protect_level,
        reason=meld.reason,
    )


def _is_low_value_singleton(
    label: str,
    amount: int,
    counts: Counter[str],
    soft_protected: set[str],
    hard_protected: set[str],
    rules=None,
) -> bool:
    if amount != 1:
        return False
    if label == WILD_LABEL:
        return False
    if label in hard_protected or label in soft_protected:
        return False

    parsed = parse_label(label)
    if parsed is None:
        return False
    suit, rank = parsed

    suit_ranks = {
        parsed_item[1]
        for item in counts
        for parsed_item in [parse_label(item)]
        if parsed_item is not None and parsed_item[0] == suit
    }
    if {1, 2, 3} <= suit_ranks and rank in {1, 2, 3}:
        return False
    if {2, 7, 10} <= suit_ranks and rank in {2, 7, 10}:
        return False
    for start in range(1, 9):
        needed = {start, start + 1, start + 2}
        if needed <= suit_ranks and rank in needed:
            return False

    if (rules or {}).get("red_black_core_only"):
        return False

    return True


def _meld_protect_level(meld_type: str, rules: dict | None = None) -> str:
    rules = rules or {}
    forced = rules.get("protection_levels", {})
    if isinstance(forced, dict):
        level = forced.get(meld_type)
        if level in {"hard", "soft"}:
            return level
    if meld_type in {"exact_triplet", "exact_quad"}:
        return "hard"
    if meld_type == "wildcard_meld":
        return "soft"
    return "soft"


def _mixed_rank_protect_level(rules: dict | None = None) -> str:
    rules = rules or {}
    forced = rules.get("protection_levels", {})
    if isinstance(forced, dict):
        level = forced.get("mixed_same_rank_triplet")
        if level in {"hard", "soft"}:
            return level
    return "soft"
