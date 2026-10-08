"""Tests for xi calculation."""

from engine.xi_calculator import calculate_xi, meld_xi, total_xi


def test_xi_for_special_melds_and_runs():
    assert meld_xi(["二", "七", "十"]) == 3
    assert meld_xi(["贰", "柒", "拾"]) == 6
    assert meld_xi(["一", "二", "三"]) == 3
    assert meld_xi(["壹", "贰", "叁"]) == 6


def test_xi_for_peng_pao_ti_by_size():
    assert meld_xi(["二", "二", "二"], kind="peng") == 1
    assert meld_xi(["贰", "贰", "贰"], kind="peng") == 3
    assert meld_xi(["二", "二", "二", "二"], kind="pao") == 6
    assert meld_xi(["贰", "贰", "贰", "贰"], kind="ti") == 12


def test_calculate_xi_outputs_total_and_details_for_raw_melds():
    result = calculate_xi(
        [
            ["二", "七", "十"],
            {"type": "peng", "labels": ["贰", "贰", "贰"]},
        ]
    )

    assert result.total_xi == 6
    assert result.min_xi == 9
    assert result.xi_gap == 3
    assert result.details == [
        {"type": "special_2710", "cards": ["二", "七", "十"], "labels": ["二", "七", "十"], "xi": 3, "suit": "small"},
        {"type": "peng", "cards": ["贰", "贰", "贰"], "labels": ["贰", "贰", "贰"], "xi": 3, "suit": "big"},
    ]
    assert result.to_dict()["is_enough_to_hu"] is False


def test_total_xi_can_use_explicit_rules_dict():
    rules = {
        "rules": {"min_xi": 9},
        "xi": {
            "ti": {"small": 9, "big": 12},
            "pao": {"small": 6, "big": 9},
            "wei": {"small": 3, "big": 6},
            "peng": {"small": 2, "big": 4},
            "special_123": {"small": 3, "big": 6},
            "special_2710": {"small": 3, "big": 6},
        },
    }

    assert total_xi([["二", "二", "二"], ["贰", "贰", "贰"]], rules=rules) == 6
