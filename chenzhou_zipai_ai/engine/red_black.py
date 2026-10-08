"""Red-black route planner and lightweight rule helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.pro_brain import RedBlackPlan, evaluate_red_black_plan
from engine.cards import RED_LABELS, WILD_LABEL, normalize_card_label
from engine.red_black_rules import classify_red_black
from engine.rules import load_rules


@dataclass(frozen=True)
class RedBlackBreakdown:
    red_count: int
    black_count: int
    wildcard_count: int
    red_cards: list[str] = field(default_factory=list)
    black_cards: list[str] = field(default_factory=list)
    wildcard_cards: list[str] = field(default_factory=list)
    red_black_points: float = 0.0
    mode: str = "red_black_mingtang"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "red_count": self.red_count,
            "black_count": self.black_count,
            "wildcard_count": self.wildcard_count,
            "red_cards": list(self.red_cards),
            "black_cards": list(self.black_cards),
            "wildcard_cards": list(self.wildcard_cards),
            "red_black_points": self.red_black_points,
            "mode": self.mode,
            "details": dict(self.details),
        }


class RedBlackPlanner:
    def evaluate(self, context, allocation) -> RedBlackPlan:
        return evaluate_red_black_plan(context, allocation)


def count_red_black(
    cards: list[str],
    *,
    rules: dict[str, Any] | None = None,
    config_path: str | Path = "config/rules.yaml",
) -> RedBlackBreakdown:
    config = rules or load_rules(config_path)
    red_labels = set(config.get("cards", {}).get("red") or RED_LABELS)
    normal_labels = set(config.get("cards", {}).get("normal") or [])
    red_cards: list[str] = []
    black_cards: list[str] = []
    wildcard_cards: list[str] = []
    for raw in cards:
        label = normalize_card_label(str(raw))
        if not label:
            continue
        if label == WILD_LABEL:
            wildcard_cards.append(label)
        elif label in red_labels:
            red_cards.append(label)
        elif not normal_labels or label in normal_labels:
            black_cards.append(label)
    return RedBlackBreakdown(
        red_count=len(red_cards),
        black_count=len(black_cards),
        wildcard_count=len(wildcard_cards),
        red_cards=red_cards,
        black_cards=black_cards,
        wildcard_cards=wildcard_cards,
        mode=str(config.get("rules", {}).get("red_black_mode", "red_black_mingtang")),
        details={"red_labels": sorted(red_labels)},
    )


def calculate_red_black_points(
    cards: list[str],
    *,
    rules: dict[str, Any] | None = None,
    config_path: str | Path = "config/rules.yaml",
) -> RedBlackBreakdown:
    config = rules or load_rules(config_path)
    breakdown = count_red_black(cards, rules=config)
    mode = str(config.get("rules", {}).get("red_black_mode", "red_black_mingtang"))
    outcome = classify_red_black(
        breakdown.red_count,
        config,
        card_count=breakdown.red_count + breakdown.black_count,
    )
    return RedBlackBreakdown(
        red_count=breakdown.red_count,
        black_count=breakdown.black_count,
        wildcard_count=breakdown.wildcard_count,
        red_cards=breakdown.red_cards,
        black_cards=breakdown.black_cards,
        wildcard_cards=breakdown.wildcard_cards,
        red_black_points=round(outcome.points, 3),
        mode=mode,
        details={
            **breakdown.details,
            "special_hand": outcome.kind,
            "qualifies": outcome.qualifies,
            "raw_points": outcome.raw_points,
            "multiplier": outcome.multiplier,
        },
    )


analyze_red_black = calculate_red_black_points


__all__ = [
    "RedBlackBreakdown",
    "RedBlackPlan",
    "RedBlackPlanner",
    "analyze_red_black",
    "calculate_red_black_points",
    "count_red_black",
    "evaluate_red_black_plan",
]
