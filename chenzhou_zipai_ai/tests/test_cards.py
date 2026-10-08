"""Tests for card helpers."""

from engine.cards import (
    Card,
    CardInstance,
    card_instance,
    card_instances,
    is_black,
    is_red,
    normalize_card_label,
    normalize_cards,
    same_rank,
)


def test_card_helpers_identify_suit_rank_and_color():
    assert Card("一").suit == "small"
    assert Card("壹").suit == "big"
    assert Card("十").rank == 10
    assert is_red("二")
    assert not is_red("三")
    assert is_black("三")
    assert not is_black("二")
    assert not is_black("王")
    assert same_rank("一", "壹")


def test_normalize_card_label_aliases():
    assert normalize_card_label("參") == "叁"
    assert normalize_card_label("参") == "叁"
    assert normalize_card_label("貳") == "贰"
    assert normalize_card_label("貮") == "贰"
    assert normalize_card_label(" 叁 ") == "叁"
    assert normalize_card_label("叁.png") == "叁"
    assert normalize_card_label("叁_001") == "叁"


def test_normalize_cards_removes_invalid_and_keeps_standard_labels():
    assert normalize_cards(["參", "叁", " 参 ", "未知牌", "叁.png"]) == ["叁", "叁", "叁", "未知牌", "叁"]


def test_card_instance_builds_clickable_attrs():
    instance = card_instance("二", source="hand", x=10, y=20, confidence=0.93, clickable=True)
    assert isinstance(instance, CardInstance)
    assert instance.label == "二"
    assert instance.size == "small"
    assert instance.rank == 2
    assert instance.is_red is True
    assert instance.is_black is False
    assert instance.source == "hand"
    assert instance.x == 10
    assert instance.y == 20
    assert instance.confidence == 0.93
    assert instance.clickable is True

    black = card_instance("三")
    assert black.is_black is True
    assert black.to_dict()["is_black"] is True


def test_card_instances_are_independent_instances():
    instances = card_instances(["叁", "叁"], source="hand")
    assert instances[0].label == "叁"
    assert instances[1].label == "叁"
    assert instances[0].id != instances[1].id
