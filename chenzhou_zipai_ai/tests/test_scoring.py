"""Tests for settlement scoring."""

from __future__ import annotations

from types import SimpleNamespace

from ai.pro_brain import HuResult
from engine.rules import RuleConfig
from engine.scoring import ScoreResult, calculate_score, score_hu


def _rules(**overrides) -> RuleConfig:
    rules = {
        "min_xi": 9,
        "zimo_double": False,
        "red_black_mode": "red_black_mingtang",
        "piao_mode": "none",
        "xi_to_tun": "3_to_1",
        "base_tun_at_9_xi": 1,
    }
    scoring = {
        "red_hu_min_red": 13,
        "black_hu_red_count": 0,
        "one_red_hu_red_count": 1,
        "red_black_special_point": 1,
    }
    rules.update(overrides.pop("rules", {}))
    scoring.update(overrides.pop("scoring", {}))
    return RuleConfig({"rules": rules, "scoring": scoring, **overrides})


def _hu(**overrides) -> SimpleNamespace:
    data = {
        "can_hu": True,
        "total_xi": 9,
        "min_xi": 9,
        "partition": [],
        "wildcard_mapping": {},
        "red_black_bonus": 0,
        "reason": "test",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_score_hu_uses_three_xi_to_one_tun_and_base_at_nine():
    result = score_hu(_hu(total_xi=15), _rules())

    assert isinstance(result, ScoreResult)
    assert result.base_tun == 1
    assert result.extra_tun == 2
    assert result.tun == 3
    assert result.final_score == 3


def test_score_hu_accepts_real_hu_result_and_rule_config():
    hu_result = HuResult(
        can_hu=True,
        total_xi=9,
        min_xi=9,
        partition=[{"labels": ["二", "三", "四"]}],
        wildcard_mapping={},
        red_black_bonus=0,
        reason="test",
    )

    result = score_hu(hu_result, _rules())

    assert result.tun == 1
    assert result.red_black_points == 1
    assert result.details["red_black"]["special_hand"] == "one_red_hu"
    assert result.final_score == 2


def test_score_hu_supports_one_xi_to_one_tun_and_base_three():
    result = score_hu(
        _hu(total_xi=12),
        _rules(rules={"xi_to_tun": "1_to_1", "base_tun_at_9_xi": 3}),
    )

    assert result.xi_to_tun == "1_to_1"
    assert result.base_tun == 3
    assert result.extra_tun == 3
    assert result.final_score == 6


def test_score_hu_applies_zimo_red_black_double_and_piao_fields():
    result = calculate_score(
        _hu(
            total_xi=9,
            partition=[
                {"labels": ["二", "三", "四"]},
                {"labels": ["一", "三", "四"]},
            ],
            piao_points=2,
            win_type="zimo",
        ),
        _rules(rules={"zimo_double": True, "red_black_mode": "red_black_point_double"}),
    )

    assert result.red_count == 1
    assert result.black_count == 5
    assert result.red_black_points == 2
    assert result.piao_points == 2
    assert result.zimo_multiplier == 2
    assert result.final_score == 10


def test_score_hu_black_special_points_are_configurable():
    result = score_hu(
        _hu(partition=[{"labels": ["一", "三", "四"]}]),
        _rules(scoring={"red_black_special_points": {"black_hu": 1.5}}),
    )

    assert result.red_black_points == 1.5
    assert result.details["red_black"]["special_hand"] == "black_hu"


def test_score_hu_returns_zero_for_rejected_or_short_hu():
    result = score_hu(_hu(can_hu=False, total_xi=8, reject_reason="xi_not_enough"), _rules())

    assert result.can_hu is False
    assert result.tun == 0
    assert result.final_score == 0
    assert result.details["reject_reason"] == "xi_not_enough"
