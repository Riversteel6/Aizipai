"""Card model and helpers."""

from __future__ import annotations

import re
import itertools
from dataclasses import dataclass
from functools import lru_cache

SMALL_LABELS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
BIG_LABELS = ("壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "拾")

_LABEL_NORMALIZE_MAP = {
    "參": "叁",
    "参": "叁",
    "貳": "贰",
    "貮": "贰",
}
RED_LABELS = {"二", "七", "十", "贰", "柒", "拾"}
WILD_LABEL = "王"
_VALID_LABELS = (*SMALL_LABELS, *BIG_LABELS, WILD_LABEL)


@dataclass(frozen=True, order=True)
class Card:
    label: str

    @property
    def suit(self) -> str:
        if self.label in SMALL_LABELS:
            return "small"
        if self.label in BIG_LABELS:
            return "big"
        if self.is_wild:
            return "wild"
        raise ValueError(f"Unknown card label: {self.label}")

    @property
    def rank(self) -> int | None:
        if self.label in SMALL_LABELS:
            return SMALL_LABELS.index(self.label) + 1
        if self.label in BIG_LABELS:
            return BIG_LABELS.index(self.label) + 1
        return None

    @property
    def is_wild(self) -> bool:
        return self.label == WILD_LABEL

    @property
    def is_red(self) -> bool:
        return self.label in RED_LABELS or self.is_wild

    @property
    def is_black(self) -> bool:
        return not self.is_wild and self.label in (*SMALL_LABELS, *BIG_LABELS) and self.label not in RED_LABELS

    @property
    def is_big(self) -> bool:
        return self.label in BIG_LABELS

    @property
    def is_small(self) -> bool:
        return self.label in SMALL_LABELS


@dataclass(frozen=True, order=True)
class CardInstance:
    id: str
    label: str
    rank: int | None
    size: str | None
    is_red: bool
    is_wildcard: bool
    source: str
    x: int | None
    y: int | None
    confidence: float | None
    clickable: bool

    @property
    def card_id(self) -> str:
        return self.id

    @property
    def is_black(self) -> bool:
        return not self.is_wildcard and self.label in (*SMALL_LABELS, *BIG_LABELS) and self.label not in RED_LABELS

    def to_dict(self) -> dict[str, object]:
        return {
            "card_id": self.id,
            "id": self.id,
            "label": self.label,
            "rank": self.rank,
            "size": self.size,
            "is_red": self.is_red,
            "is_black": self.is_black,
            "is_wildcard": self.is_wildcard,
            "source": self.source,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "clickable": self.clickable,
        }


_INSTANCE_COUNTER = itertools.count()


def card(label: str | Card) -> Card:
    return label if isinstance(label, Card) else Card(label)


def labels(cards: list[str | Card]) -> list[str]:
    return [card(item).label for item in cards]


def normal_labels() -> list[str]:
    return [*SMALL_LABELS, *BIG_LABELS]


def is_red(label: str | Card) -> bool:
    return card(label).is_red


def is_black(label: str | Card) -> bool:
    return card(label).is_black


def is_big(label: str | Card) -> bool:
    return card(label).is_big


def is_small(label: str | Card) -> bool:
    return card(label).is_small


def same_rank(left: str | Card, right: str | Card) -> bool:
    left_card = card(left)
    right_card = card(right)
    return left_card.rank is not None and left_card.rank == right_card.rank


def label_for(suit: str, rank: int) -> str:
    if suit == "small":
        return SMALL_LABELS[rank - 1]
    if suit == "big":
        return BIG_LABELS[rank - 1]
    raise ValueError(f"Unknown suit: {suit}")


@lru_cache(maxsize=1024)
def normalize_card_label(label: str) -> str:
    text = (label or "").strip()
    if not text:
        return ""

    # OCR/template source sometimes comes with extension/suffix/path.
    text = text.replace("\u3000", "").replace(" ", "")
    text = text.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    text = re.sub(r"\.[a-zA-Z0-9]+$", "", text)
    text = re.sub(r"[_-].*$", "", text)
    text = _LABEL_NORMALIZE_MAP.get(text, text)
    if text in _VALID_LABELS:
        return text
    for original, normalized in _LABEL_NORMALIZE_MAP.items():
        text = text.replace(original, normalized)
    for candidate in _VALID_LABELS:
        if candidate in text:
            return candidate
    return text


def normalize_cards(cards: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in cards:
        label = normalize_card_label(item)
        if label:
            normalized.append(label)
    return normalized


@lru_cache(maxsize=256)
def parse_label(label: str) -> tuple[str, int] | None:
    normalized = normalize_card_label(label)
    if normalized in SMALL_LABELS:
        return ("small", SMALL_LABELS.index(normalized) + 1)
    if normalized in BIG_LABELS:
        return ("big", BIG_LABELS.index(normalized) + 1)
    return None


def rank_of(label: str) -> int | None:
    if isinstance(label, CardInstance):
        return label.rank
    parsed = parse_label(label)
    return parsed[1] if parsed is not None else None


def normalize_card_instance_id(prefix: str | None = None) -> str:
    if not prefix:
        prefix = "card"
    return f"{prefix}_{next(_INSTANCE_COUNTER):06d}"


def card_instance(
    label: str,
    *,
    source: str = "unknown",
    x: int | None = None,
    y: int | None = None,
    confidence: float | None = None,
    clickable: bool = True,
    instance_id: str | None = None,
) -> CardInstance:
    normalized = normalize_card_label(label)
    parsed = parse_label(normalized)
    size: str | None = None
    rank: int | None = None
    if parsed:
        size, rank = parsed
    is_wild = normalized == WILD_LABEL
    if is_wild:
        is_wild = True
        rank = None
        size = "wild"
    return CardInstance(
        id=instance_id or normalize_card_instance_id(),
        label=normalized,
        rank=rank,
        size=size,
        is_red=(normalized in RED_LABELS or is_wild),
        is_wildcard=is_wild,
        source=source,
        x=x,
        y=y,
        confidence=confidence,
        clickable=clickable,
    )


def card_instances(
    labels: list[str],
    *,
    source: str = "unknown",
) -> list[CardInstance]:
    return [card_instance(label, source=source) for label in labels]
