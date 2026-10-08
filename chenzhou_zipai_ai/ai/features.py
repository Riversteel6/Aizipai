"""Fast hand feature extraction for first-generation policy."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
import os

from engine.cards import BIG_LABELS, RED_LABELS, SMALL_LABELS, WILD_LABEL, parse_label


QUICK_POTENTIAL_LABELS = (*SMALL_LABELS, *BIG_LABELS, WILD_LABEL)
_LARGE_FEATURE_CACHE_SIZE = 32_768 if os.environ.get("AIZIPAI_MOBILE_RUNTIME") == "1" else 200_000
_REMOVAL_FEATURE_CACHE_SIZE = 32_768 if os.environ.get("AIZIPAI_MOBILE_RUNTIME") == "1" else 65_536
_SUIT_FEATURE_CACHE_SIZE = 32_768 if os.environ.get("AIZIPAI_MOBILE_RUNTIME") == "1" else 200_000


def _presence_score_from_mask(mask: int, weight: int) -> int:
    score = sum(bool(mask & (1 << index)) for index in (0, 1, 2)) * weight
    score += sum(bool(mask & (1 << index)) for index in (1, 6, 9)) * weight
    for start in range(8):
        connected = ((mask >> start) & 0b111).bit_count()
        if connected >= 2:
            score += connected
    return score


_SMALL_PRESENCE_SCORES = tuple(
    _presence_score_from_mask(mask, 2) for mask in range(1 << 10)
)
_BIG_PRESENCE_SCORES = tuple(
    _presence_score_from_mask(mask, 3) for mask in range(1 << 10)
)
_PRESENCE_SCORE_TABLES = (_SMALL_PRESENCE_SCORES, _BIG_PRESENCE_SCORES)
_RED_LABEL_INDICES = frozenset(
    index
    for index, label in enumerate(QUICK_POTENTIAL_LABELS[:-1])
    if label in RED_LABELS
)
_MULTIPLICITY_SCORE_TABLE = tuple(
    tuple(
        amount * 4
        if index == len(QUICK_POTENTIAL_LABELS) - 1
        else (
            (6 if index in _RED_LABEL_INDICES else 3)
            if amount >= 3
            else (3 if index in _RED_LABEL_INDICES else 2)
            if amount == 2
            else 0
        )
        for amount in range(5)
    )
    for index in range(len(QUICK_POTENTIAL_LABELS))
)


@lru_cache(maxsize=_SUIT_FEATURE_CACHE_SIZE)
def _suit_feature_summary(
    counts: tuple[int, ...],
    suit_index: int,
) -> tuple[int, int]:
    """Return exact multiplicity+presence score and mask for one ten-rank suit."""

    offset = suit_index * 10
    mask = 0
    score = 0
    for rank, amount in enumerate(counts):
        if amount > 0:
            mask |= 1 << rank
        if amount <= 4:
            score += _MULTIPLICITY_SCORE_TABLE[offset + rank][amount]
        else:
            score += _multiplicity_score(offset + rank, amount)
    return score + _PRESENCE_SCORE_TABLES[suit_index][mask], mask


def quick_potential(hand: Sequence[str]) -> int:
    return _quick_potential_cached(tuple(sorted(hand)))


def quick_potential_from_counts(counts: tuple[int, ...]) -> int:
    """Evaluate the same feature from canonical card counts."""
    if len(counts) != len(QUICK_POTENTIAL_LABELS):
        raise ValueError("quick_potential_count_shape")
    return _quick_potential_counts_cached(counts)


@lru_cache(maxsize=_REMOVAL_FEATURE_CACHE_SIZE)
def quick_potential_after_each_removal_from_counts(
    counts: tuple[int, ...],
) -> tuple[int | None, ...]:
    """Return the exact feature value after removing each available card."""
    if len(counts) != len(QUICK_POTENTIAL_LABELS):
        raise ValueError("quick_potential_count_shape")
    base = _quick_potential_counts_cached(counts)
    _small_score, small_mask = _suit_feature_summary(counts[:10], 0)
    _big_score, big_mask = _suit_feature_summary(counts[10:20], 1)
    values: list[int | None] = [None] * len(counts)
    for index, amount in enumerate(counts):
        if amount <= 0:
            continue
        if index == len(counts) - 1:
            multiplicity_delta = -4
        elif amount >= 4:
            multiplicity_delta = 0
        elif amount == 3:
            multiplicity_delta = -3 if QUICK_POTENTIAL_LABELS[index] in RED_LABELS else -1
        elif amount == 2:
            multiplicity_delta = -3 if QUICK_POTENTIAL_LABELS[index] in RED_LABELS else -2
        else:
            multiplicity_delta = 0
        value = base + multiplicity_delta
        if index < len(counts) - 1 and amount == 1:
            offset = 0 if index < 10 else 10
            presence_mask = small_mask if offset == 0 else big_mask
            score_table = _SMALL_PRESENCE_SCORES if offset == 0 else _BIG_PRESENCE_SCORES
            removed_mask = presence_mask & ~(1 << (index - offset))
            value += score_table[removed_mask] - score_table[presence_mask]
        values[index] = value
    return tuple(values)


@lru_cache(maxsize=_REMOVAL_FEATURE_CACHE_SIZE)
def best_quick_potential_after_discard_from_counts(
    counts: tuple[int, ...],
) -> int | None:
    """Return the exact best legal-discard feature without a 21-item tuple."""

    if len(counts) != len(QUICK_POTENTIAL_LABELS):
        raise ValueError("quick_potential_count_shape")
    small_total, small_mask = _suit_feature_summary(counts[:10], 0)
    big_total, big_mask = _suit_feature_summary(counts[10:20], 1)
    wildcard_amount = counts[-1]
    wildcard_total = (
        _MULTIPLICITY_SCORE_TABLE[-1][wildcard_amount]
        if wildcard_amount <= 4
        else _multiplicity_score(len(counts) - 1, wildcard_amount)
    )
    base = small_total + big_total + wildcard_total
    small_score = _SMALL_PRESENCE_SCORES[small_mask]
    big_score = _BIG_PRESENCE_SCORES[big_mask]
    best: int | None = None
    for index, amount in enumerate(counts[:-1]):
        if amount <= 0 or amount >= 3:
            continue
        if amount == 2:
            delta = -3 if index in _RED_LABEL_INDICES else -2
        elif index < 10:
            delta = (
                _SMALL_PRESENCE_SCORES[small_mask & ~(1 << index)]
                - small_score
            )
        else:
            rank = index - 10
            delta = (
                _BIG_PRESENCE_SCORES[big_mask & ~(1 << rank)]
                - big_score
            )
        value = base + delta
        if best is None or value > best:
            best = value
    return best


@lru_cache(maxsize=_LARGE_FEATURE_CACHE_SIZE)
def _quick_potential_cached(hand: tuple[str, ...]) -> int:
    score = 0
    counts = Counter(hand)
    for label, amount in counts.items():
        if label == WILD_LABEL:
            score += amount * 4
        elif amount >= 3:
            score += 6 if label in RED_LABELS else 3
        elif amount == 2:
            score += 3 if label in RED_LABELS else 2
    for suit in ("small", "big"):
        ranks = {
            rank
            for label in hand
            if (parsed := parse_label(label)) is not None
            for item_suit, rank in [parsed]
            if item_suit == suit
        }
        for group in ({1, 2, 3}, {2, 7, 10}):
            score += len(ranks & group) * (3 if suit == "big" else 2)
        for start in range(1, 9):
            connected = len(ranks & {start, start + 1, start + 2})
            if connected >= 2:
                score += connected
    return score


@lru_cache(maxsize=_LARGE_FEATURE_CACHE_SIZE)
def _quick_potential_counts_cached(counts: tuple[int, ...]) -> int:
    small_score, _small_mask = _suit_feature_summary(counts[:10], 0)
    big_score, _big_mask = _suit_feature_summary(counts[10:20], 1)
    wildcard_amount = counts[-1]
    wildcard_score = (
        _MULTIPLICITY_SCORE_TABLE[-1][wildcard_amount]
        if wildcard_amount <= 4
        else _multiplicity_score(len(counts) - 1, wildcard_amount)
    )
    return small_score + big_score + wildcard_score


def _multiplicity_score(index: int, amount: int) -> int:
    if index == len(QUICK_POTENTIAL_LABELS) - 1:
        return amount * 4
    if amount >= 3:
        return 6 if QUICK_POTENTIAL_LABELS[index] in RED_LABELS else 3
    if amount == 2:
        return 3 if QUICK_POTENTIAL_LABELS[index] in RED_LABELS else 2
    return 0


@lru_cache(maxsize=2_048)
def _suit_presence_score(present: tuple[bool, ...], weight: int) -> int:
    mask = 0
    for index, available in enumerate(present):
        if available:
            mask |= 1 << index
    if weight == 2:
        return _SMALL_PRESENCE_SCORES[mask]
    if weight == 3:
        return _BIG_PRESENCE_SCORES[mask]
    return _presence_score_from_mask(mask, weight)
