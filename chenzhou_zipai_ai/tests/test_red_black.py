"""Tests for red-black rule helpers."""

from __future__ import annotations

from pathlib import Path

from engine.rules import load_rules
from engine.red_black import RedBlackBreakdown, calculate_red_black_points, count_red_black


def test_count_red_black_excludes_wildcards_from_red_and_black_counts():
    result = count_red_black(["二", "七", "三", "壹", "王"])

    assert isinstance(result, RedBlackBreakdown)
    assert result.red_count == 2
    assert result.black_count == 2
    assert result.wildcard_count == 1
    assert result.red_cards == ["二", "七"]
    assert result.black_cards == ["三", "壹"]
    assert result.wildcard_cards == ["王"]


def test_red_black_points_only_reward_named_special_hands():
    result = calculate_red_black_points(["二", "七", "三", "王"])

    assert result.red_black_points == 0
    assert result.details["special_hand"] is None
    assert calculate_red_black_points(["二", "三"]).details["special_hand"] == "one_red_hu"
    assert calculate_red_black_points(["一", "三"]).details["special_hand"] == "black_hu"
    assert calculate_red_black_points(["二"] * 13).details["special_hand"] == "red_hu"


def test_red_black_special_points_are_configurable():
    rules = {
        "cards": {
            "normal": ["一", "二", "三", "四"],
            "red": ["一", "四"],
        },
        "rules": {"red_black_mode": "red_black_mingtang"},
        "scoring": {
            "red_black_special_points": {"one_red_hu": 2.5},
        },
    }

    result = calculate_red_black_points(["一", "二", "三", "王"], rules=rules)

    assert result.red_cards == ["一"]
    assert result.black_cards == ["二", "三"]
    assert result.red_black_points == 2.5


def test_red_black_points_can_be_disabled_by_mode():
    rules = {
        "cards": {"normal": ["一", "二"], "red": ["二"]},
        "rules": {"red_black_mode": "none"},
        "scoring": {"red_black_special_point": 3},
    }

    result = calculate_red_black_points(["二", "一"], rules=rules)

    assert result.red_count == 1
    assert result.black_count == 1
    assert result.red_black_points == 0


def test_default_rule_files_expose_red_black_scoring_config():
    nested_root = Path(__file__).resolve().parents[1]
    config_paths = [nested_root / "config" / "rules.yaml", nested_root.parent / "config" / "rules.yaml"]

    for config_path in config_paths:
        rules = load_rules(config_path)
        scoring = rules["scoring"]
        assert scoring["red_hu_min_red"] == 13
        assert scoring["black_hu_red_count"] == 0
        assert scoring["one_red_hu_red_count"] == 1
        result = calculate_red_black_points(["二", "三", "四"], rules=rules)
        assert result.red_black_points == 1
        assert result.details["special_hand"] == "one_red_hu"
