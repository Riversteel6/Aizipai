"""Tests for hand-structure protection analysis."""

from ai.hand_protection import analyze_complete_melds, analyze_hand_protection, same_rank, rank_of
from engine.cards import normalize_cards


def test_hand_protection_marks_triplet_as_hard():
    analysis = analyze_hand_protection(["叁", "叁", "叁", "四", "九"])

    assert "叁" in analysis.hard_protected
    assert analysis.counts["叁"] == 3


def test_hand_protection_marks_alias_triplet_and_wild_as_hard():
    analysis = analyze_hand_protection(["參", "参", "叁", "王", "九"])

    assert analysis.counts["叁"] == 3
    assert set(analysis.hard_protected) == {"叁", "王"}


def test_hand_protection_identifies_low_value_single_forbidden_set():
    analysis = analyze_hand_protection(["叁", "叁", "叁", "四", "九", "二", "七"])

    assert "叁" in analysis.hard_protected
    assert "四" in analysis.low_value_singles
    assert "九" in analysis.low_value_singles


def test_same_rank_mapping_between_small_and_big():
    assert rank_of("五") == 5
    assert rank_of("伍") == 5
    assert same_rank("五", "伍")
    assert same_rank("六", "陆")
    assert same_rank("十", "拾")
    assert not same_rank("五", "六")


def test_analyze_complete_melds_detects_mixed_same_rank_triplet():
    melds = analyze_complete_melds(["伍", "伍", "五", "九", "八"], rules={})

    assert any(item.type == "mixed_same_rank_triplet" and set(item.labels) == {"五", "伍"} for item in melds)


def test_analyze_complete_melds_detects_sequence():
    melds = analyze_complete_melds(["肆", "伍", "陆", "九", "八"], rules={})

    assert any(item.type == "normal_sequence" and list(item.cards) == ["肆", "伍", "陆"] for item in melds)


def test_analyze_complete_melds_detects_special_123_and_2710():
    three = analyze_complete_melds(["一", "二", "三", "二", "七", "十"], rules={})
    assert any(item.type == "special_123" and item.cards == ("一", "二", "三") for item in three)
    assert any(item.type == "special_2710" and item.cards == ("二", "七", "十") for item in three)


def test_hand_protection_protects_mixed_triplet_and_sequence_labels():
    analysis = analyze_hand_protection(["肆", "伍", "陆", "四", "八", "九"], rules={})

    for label in normalize_cards(["肆", "伍", "陆"]):
        assert label in analysis.soft_protected
    assert "九" in analysis.low_value_singles


def test_hand_protection_not_reuse_locked_triplet_for_sequence():
    analysis = analyze_hand_protection(["一", "二", "二", "二", "三"], rules={})

    assert "二" in analysis.hard_protected
    assert "一" not in analysis.soft_protected
    assert "三" not in analysis.soft_protected
    assert "一" in analysis.low_value_singles
    assert "三" in analysis.low_value_singles
    assert not any(item.type == "special_123" for item in analysis.protected_melds)
