"""Pure red/black special-hand rules shared by scoring and strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


DISABLED_MODES = {"none", "off", "disabled"}
DOUBLE_MODES = {"red_black_point_double", "red_black_point_2x", "red_black_double", "double"}


@dataclass(frozen=True)
class RedBlackOutcome:
    kind: str | None
    qualifies: bool
    raw_points: float
    multiplier: float
    points: float


def classify_red_black(
    red_count: int,
    rules: Mapping[str, Any],
    *,
    card_count: int | None = None,
) -> RedBlackOutcome:
    rule_section = rules.get("rules", {}) if isinstance(rules.get("rules"), Mapping) else {}
    scoring = rules.get("scoring", {}) if isinstance(rules.get("scoring"), Mapping) else {}
    mode = str(rule_section.get("red_black_mode", "red_black_mingtang")).strip().lower()
    if mode in DISABLED_MODES:
        return RedBlackOutcome(None, False, 0.0, 1.0, 0.0)
    if card_count is not None and card_count <= 0:
        return RedBlackOutcome(None, False, 0.0, 1.0, 0.0)

    black_count = int(scoring.get("black_hu_red_count", 0))
    one_red_count = int(scoring.get("one_red_hu_red_count", 1))
    red_minimum = int(scoring.get("red_hu_min_red", 13))
    if red_count == black_count:
        kind = "black_hu"
    elif red_count == one_red_count:
        kind = "one_red_hu"
    elif red_count >= red_minimum:
        kind = "red_hu"
    else:
        return RedBlackOutcome(None, False, 0.0, 1.0, 0.0)

    by_kind = scoring.get("red_black_special_points", {})
    if not isinstance(by_kind, Mapping):
        by_kind = {}
    raw_points = float(by_kind.get(kind, scoring.get("red_black_special_point", 1)))
    multiplier = float(scoring.get("red_black_point_multiplier", 1))
    if mode in DOUBLE_MODES:
        multiplier = float(scoring.get("red_black_double_multiplier", 2))
    return RedBlackOutcome(kind, True, raw_points, multiplier, raw_points * multiplier)


def red_black_target_distance(red_count: int, locked_red_count: int, rules: Mapping[str, Any]) -> dict[str, int]:
    scoring = rules.get("scoring", {}) if isinstance(rules.get("scoring"), Mapping) else {}
    red_minimum = int(scoring.get("red_hu_min_red", 13))
    distances = {"red_hu": max(0, red_minimum - red_count)}
    if locked_red_count == 0:
        distances["black_hu"] = red_count
    if locked_red_count <= 1:
        distances["one_red_hu"] = abs(red_count - 1)
    return distances


__all__ = ["RedBlackOutcome", "classify_red_black", "red_black_target_distance"]
