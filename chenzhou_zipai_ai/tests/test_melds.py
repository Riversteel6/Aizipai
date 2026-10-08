"""Tests for meld pattern contracts."""

from __future__ import annotations

from engine.melds import classify_meld, complete_three_card_meld


def test_complete_meld_reports_wildcard_replacement_identity():
    pattern = complete_three_card_meld(["二", "七", "王"])

    assert pattern is not None
    assert pattern.labels == ("二", "七", "十")
    assert pattern.source_labels == ("二", "七", "王")
    assert pattern.wildcards_used == 1
    assert pattern.wildcard_mapping == {"王#3": "十"}
    assert pattern.to_dict()["wildcard_mapping"] == {"王#3": "十"}


def test_classify_meld_preserves_wildcard_mapping_for_special_melds():
    pattern = classify_meld(["二", "七", "王"])

    assert pattern.kind == "special_2710"
    assert pattern.labels == ("二", "七", "十")
    assert pattern.wildcard_mapping == {"王#3": "十"}
    assert pattern.to_dict()["source_labels"] == ["二", "七", "王"]


def test_classify_triplet_maps_wildcard_to_triplet_label():
    pattern = classify_meld(["八", "八", "王"])

    assert pattern.kind == "peng"
    assert pattern.wildcards_used == 1
    assert pattern.wildcard_mapping == {"王#3": "八"}
