"""Deck construction."""

from __future__ import annotations

from collections import Counter
from typing import Any

from engine.cards import WILD_LABEL, normal_labels
from engine.rules import load_rules


def full_deck_counts(
    *,
    config_path: str = "config/rules.yaml",
    include_wild: bool = True,
    rules: dict[str, Any] | None = None,
) -> Counter[str]:
    config = rules or load_rules(config_path)
    copies = int(config.get("game", {}).get("deck_copies", 4))
    counts = Counter({label: copies for label in normal_labels()})
    if include_wild and config.get("wildcard", {}).get("enabled", True):
        wildcard = config.get("wildcard", {})
        wildcard_copies = max(0, int(wildcard.get("copies", copies)))
        counts[wildcard.get("name", WILD_LABEL)] = wildcard_copies
    return counts


def visible_counts(
    *,
    hand: list[str] | None = None,
    memory: dict | None = None,
    meld_groups: dict[str, list[list[dict]]] | None = None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts.update(hand or [])
    if memory:
        counts.update(memory.get("my_discards", []))
        counts.update(memory.get("opponent_discards", []))
        for key in ("my_meld_groups", "opponent_meld_groups"):
            for group in memory.get(key, []):
                counts.update(label for label in group if label != "暗")
    if meld_groups:
        for groups in meld_groups.values():
            for group in groups:
                counts.update(item["name"] for item in group if item["name"] != "暗")
    return counts


def remaining_counts(
    *,
    hand: list[str] | None = None,
    memory: dict | None = None,
    meld_groups: dict[str, list[list[dict]]] | None = None,
    config_path: str = "config/rules.yaml",
    rules: dict[str, Any] | None = None,
) -> Counter[str]:
    remaining = full_deck_counts(config_path=config_path, rules=rules)
    for label, amount in visible_counts(hand=hand, memory=memory, meld_groups=meld_groups).items():
        remaining[label] -= amount
        if remaining[label] < 0:
            remaining[label] = 0
    return remaining


def expanded_deck(counts: Counter[str]) -> list[str]:
    return [label for label, amount in counts.items() for _ in range(amount)]
