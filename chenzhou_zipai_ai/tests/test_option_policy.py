"""Tests for chi/compare option selection."""

from ai.option_policy import choose_option, evaluate_option


def test_option_policy_prefers_special_2710_over_plain_sequence():
    best = choose_option([["三", "四", "五"], ["二", "七", "十"]])

    assert best is not None
    assert best.labels == ["二", "七", "十"]
    assert best.type == "special_2710"


def test_option_policy_scores_123_as_valuable():
    evaluation = evaluate_option(["壹", "贰", "叁"])

    assert evaluation.type == "special_123"
    assert evaluation.xi == 6
    assert evaluation.score > 0


def test_option_policy_recognizes_mixed_same_rank_candidate():
    evaluation = evaluate_option(["拾", "拾", "十"])

    assert evaluation.type == "mixed_same_rank"
    assert evaluation.score > 0


def test_option_policy_rejects_sandwiched_mixed_same_rank_candidate():
    evaluation = evaluate_option(["伍", "五", "伍"])

    assert evaluation.type == "unknown"
    assert evaluation.score < 0


def test_option_policy_accepts_valid_mixed_same_rank_orders():
    assert evaluate_option(["五", "五", "伍"]).type == "mixed_same_rank"
    assert evaluate_option(["伍", "伍", "五"]).type == "mixed_same_rank"
