"""Tests for first-generation AlphaDog policy."""

from ai.policy import choose_action, choose_discard, evaluate_response_actions


def test_policy_does_not_discard_wildcard_when_other_cards_exist():
    decision = choose_discard(["王", "二", "七", "四", "九"])

    assert decision.label != "王"
    assert "王" in decision.hard_protected


def test_policy_avoids_breaking_complete_2710():
    decision = choose_discard(["二", "七", "十", "四", "九"])

    assert decision.label not in {"二", "七", "十"}


def test_policy_prefers_isolated_card_over_complete_structures():
    decision = choose_discard(["二", "七", "十", "一", "二", "三", "九"])

    assert decision.label == "九"


def test_policy_avoids_discarding_complete_triplet():
    decision = choose_discard(["叁", "叁", "叁", "三", "肆", "伍", "陆", "捌", "九", "拾", "二", "七"])

    assert decision.label not in {"叁", "三"}


def test_policy_prefers_normal_unprotected_not_triplet():
    decision = choose_discard(["叁", "叁", "叁", "四", "九"])

    assert decision.action == "discard"
    assert decision.label in {"四", "九"}
    assert "叁" not in {
        decision.label,
    }
    assert decision.candidate_stage == "normal_unprotected"


def test_policy_prefers_normal_unprotected_not_triplet_with_2710_side_cards():
    decision = choose_discard(["叁", "叁", "叁", "四", "九", "二", "七"])

    assert decision.action == "discard"
    assert decision.label in {"四", "九"}
    assert "叁" not in {decision.label}
    assert decision.candidate_stage in {"normal_unprotected", "weak_potential"}


def test_policy_does_not_force_break_when_unprotected_clickable_exists():
    decision = choose_discard(["叁", "叁", "叁", "四", "九"], clickable_cards=["叁", "四"])

    assert decision.action == "discard"
    assert decision.label == "四"
    assert decision.candidate_stage == "normal_unprotected"


def test_policy_alias_normalization_blocks_mixed_triplet():
    decision = choose_discard(["參", "叁", "参", "四", "九"])

    assert decision.action == "discard"
    assert decision.label in {"四", "九"}
    assert "叁" not in {decision.label}


def test_policy_hard_protects_wild_and_triplet():
    decision = choose_discard(["叁", "叁", "叁", "王", "九"])

    assert decision.action == "discard"
    assert decision.label == "九"
    assert set(decision.hard_protected) == {"叁", "王"}


def test_policy_forced_break_when_all_clickable_are_hard():
    decision = choose_discard(["叁", "叁", "叁", "肆", "肆", "肆"], clickable_cards=["叁", "肆"])

    assert decision.action == "discard"
    assert decision.candidate_stage == "forced_break_hard_protection"
    assert decision.break_hard_protection is True
    assert decision.break_reason == "all_clickable_cards_are_hard_protected"
    assert decision.label in {"叁", "肆"}


def test_policy_do_not_force_break_when_unprotected_cards_exist_outside_clickable():
    decision = choose_discard(["叁", "叁", "叁", "四", "九"], clickable_cards=["叁"])

    assert decision.action == "safe_halt"
    assert decision.candidate_stage == "safe_halt"
    assert decision.break_reason == "only_hard_protected_clickable_but_unprotected_cards_exist_in_hand"


def test_policy_fallback_keeps_something_when_all_cards_are_triplets():
    decision = choose_discard(["叁", "叁", "叁", "四", "四", "四"])

    assert decision.action == "discard"
    assert decision.label in {"叁", "四"}


def test_policy_only_hu_when_hu_is_legal_in_live_state():
    waiting = choose_action(["二", "七", "十"], legal_actions=["pass"])
    hu = choose_action(["二", "七", "十"], legal_actions=["hu", "pass"])
    complete = choose_action(
        [
            "贰", "柒", "拾",
            "壹", "贰", "叁",
            "一", "二", "三",
            "四", "五", "六",
            "七", "八", "九",
            "肆", "伍", "陆",
            "柒", "捌", "玖",
        ],
        legal_actions=["hu", "pass"],
    )

    assert waiting.action == "pass"
    assert hu.action == "pass"
    assert {"meld_group_count_invalid", "xi_not_enough"} & {
        item["reject_reason"] for item in hu.response_evaluations
    }
    assert complete.action == "hu"


def test_policy_response_priority_for_visible_buttons():
    assert choose_action(["一", "一"], legal_actions=["chi", "peng", "pass"]).action == "expand_chi_options"
    assert choose_action(["二", "二", "四", "六", "九"], legal_actions=["peng", "pass"]).action == "peng"
    assert choose_action(
        ["二", "十", "四", "六", "九"],
        legal_actions=["chi", "pass"],
        option_details=[{"labels": ["二", "七", "十"], "confidence": 1.0}],
    ).action == "chi"
    assert choose_action(["一"], legal_actions=["pao", "pass"]).action == "wait_auto_meld"


def test_policy_waits_for_auto_ti():
    assert choose_action(["一"], legal_actions=["ti", "pass"]).action == "wait_auto_meld"


def test_response_actions_include_explainable_scores():
    result = evaluate_response_actions(["peng", "pass"], hand=["二", "二", "四", "六", "九"])

    by_action = {item["action"]: item for item in result}
    assert set(by_action) == {"peng", "pass"}
    assert by_action["pass"]["allowed"]
    assert by_action["peng"]["allowed"]
    assert by_action["peng"]["score"] > by_action["pass"]["score"]


def test_chi_uncertain_options_only_go_pass():
    result = choose_action(
        ["一", "二", "三", "四", "伍"],
        legal_actions=["chi", "pass"],
        option_count=1,
        option_details=[
            {
                "region_name": "chi_options",
                "index": 1,
                "labels": ["五", "七", "九"],
                "confidence": 0.40,
                "center": [100, 200],
            }
        ],
    )

    assert result.action == "pass"
    assert not result.label
    assert result.response_evaluations is not None
    assert any(item["action"] == "pass" for item in result.response_evaluations)


def test_policy_avoids_discarding_伍伍五():
    decision = choose_discard(["伍", "伍", "五", "九", "八"])

    assert decision.action == "discard"
    assert decision.label not in {"伍", "五"}
    assert decision.label in {"九", "八"}
    assert any(item["type"] == "mixed_same_rank_triplet" for item in decision.protected_melds)


def test_policy_avoids_discarding_四五六():
    decision = choose_discard(["四", "五", "六", "九", "八"])

    assert decision.action == "discard"
    assert decision.label not in {"四", "五", "六"}
    assert decision.label in {"九", "八"}
    assert any(item["type"] == "normal_sequence" for item in decision.protected_melds)


def test_policy_avoids_discarding_mixed_triplet_and_sequence_when_singleton_exist():
    decision = choose_discard(["伍", "伍", "五", "四", "五", "六", "九"])

    assert decision.action == "discard"
    assert decision.label == "九"
    assert decision.candidate_stage in {"normal_unprotected", "break_soft_protection"}
    assert any(item["type"] == "mixed_same_rank_triplet" for item in decision.protected_melds)
    assert any(item["type"] == "normal_sequence" for item in decision.protected_melds)


def test_policy_avoid_discarding_二七十():
    decision = choose_discard(["二", "七", "十", "九", "八"])

    assert decision.action == "discard"
    assert decision.label not in {"二", "七", "十"}
    assert decision.label in {"九", "八"}


def test_policy_avoid_discarding_一二三():
    decision = choose_discard(["一", "二", "三", "九", "八"])

    assert decision.action == "discard"
    assert decision.label not in {"一", "二", "三"}
    assert decision.label in {"九", "八"}


def test_policy_avoid_discarding_大字顺子():
    decision = choose_discard(["肆", "伍", "陆", "九", "八"])

    assert decision.action == "discard"
    assert decision.label not in {"肆", "伍", "陆"}
    assert decision.label in {"九", "八"}


def test_policy_all_soft_protected_falls_back_to_soft_break_with_reason():
    decision = choose_discard(["四", "五", "六", "七", "八", "九"])

    assert decision.action == "discard"
    assert decision.candidate_stage == "break_soft_protection"
    assert decision.label in {"四", "五", "六", "七", "八", "九"}
    assert any(item["type"] == "normal_sequence" for item in decision.protected_melds)


def test_policy_forced_break_hard_requires_all_cards_hard():
    decision = choose_discard(["叁", "叁", "叁", "伍", "伍", "伍"])

    assert decision.action == "discard"
    assert decision.candidate_stage == "forced_break_hard_protection"
    assert decision.break_hard_protection is True


def test_policy_candidate_stage_and_log_contains_protected_melds():
    decision = choose_discard(["伍", "伍", "五", "四", "五", "六", "九"])

    assert decision.candidate_stage in {"normal_unprotected", "break_soft_protection"}
    assert isinstance(decision.protected_melds, list)
    assert {"type", "cards", "protect_level", "labels", "reason"} <= set(decision.protected_melds[0].keys())
    assert decision.label == "九"
