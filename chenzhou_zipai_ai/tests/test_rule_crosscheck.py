"""Differential rule audit smoke tests."""

from audit.differential_rules import run_rule_crosscheck


def test_independent_rule_crosscheck_covers_positive_hu_and_chi_cases():
    report = run_rule_crosscheck(
        hu_cases=30,
        chi_cases=30,
        discard_cases=30,
        response_cases=30,
        draw_auto_cases=30,
        seed=20260726,
        wildcard_enabled=False,
    )

    assert report["hu_complete_cases"] > 0
    assert report["chi_positive_cases"] > 0
    assert report["discard_locked_cases"] > 0
    assert report["response_positive_cases"] > 0
    assert report["response_joint_cases"] > 0
    assert report["response_transition_cases"] > 0
    assert report["draw_auto_positive_cases"] > 0
    assert report["oracle_independent"]


def test_heads_up_rule_crosscheck_audits_84_card_room_shape():
    report = run_rule_crosscheck(
        hu_cases=10,
        chi_cases=10,
        discard_cases=10,
        response_cases=10,
        draw_auto_cases=10,
        seed=20260726,
        wildcard_enabled=True,
        players=2,
    )

    assert report["ok"]
    assert report["room_shape"] == {
        "players": 2,
        "wildcard_enabled": True,
        "deck_size": 84,
        "initial_dealt": 41,
        "initial_stock": 43,
    }
