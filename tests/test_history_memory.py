from collections import Counter
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vision.history_memory import (
    VisionMemory,
    build_sanity_checks,
    infer_meld_type,
    recover_temporally_hidden_hand,
)


def _item(name: str) -> dict:
    return {"name": name}


def test_memory_extends_discard_sequences_without_dropping_existing_cards():
    memory = VisionMemory()
    memory.update_from_snapshot(
        {
            "screenshot": "a.png",
            "remaining_deck_count": 8,
            "discards": {"my_discards": [_item("一"), _item("二")]},
            "meld_groups": {},
            "sanity_checks": {"warnings": []},
        }
    )
    memory.update_from_snapshot(
        {
            "screenshot": "b.png",
            "remaining_deck_count": 7,
            "discards": {"my_discards": [_item("一"), _item("二"), _item("三")]},
            "meld_groups": {},
            "sanity_checks": {"warnings": []},
        }
    )

    assert memory.frames_seen == 2
    assert memory.remaining_deck_count == 7
    assert memory.my_discards == ["一", "二", "三"]


def test_memory_keeps_meld_groups_as_vertical_units():
    memory = VisionMemory()
    snapshot = {
        "screenshot": "a.png",
        "discards": {},
        "meld_groups": {
            "opponent_melds": [
                [_item("柒"), _item("暗"), _item("暗")],
                [_item("壹"), _item("壹"), _item("一")],
            ]
        },
        "sanity_checks": {"warnings": []},
    }

    memory.update_from_snapshot(snapshot)
    memory.update_from_snapshot(snapshot)

    assert memory.opponent_meld_groups == [["柒", "暗", "暗"], ["壹", "壹", "一"]]


def test_infers_conservative_meld_types():
    assert infer_meld_type(["一", "暗", "暗"]) == "hidden_triplet"
    assert infer_meld_type(["一", "二", "三"]) == "chi_sequence"
    assert infer_meld_type(["壹", "壹", "一"]) == "chi_same_rank"
    assert infer_meld_type(["九", "九", "九"]) == "peng"


def test_sanity_check_counts_hand_plus_my_meld_cells():
    snapshot = {
        "hand_count": 17,
        "meld_groups": {"my_melds": [[_item("一"), _item("暗"), _item("暗")]]},
    }

    assert build_sanity_checks(snapshot, expected_total=20)["ok"]
    failed = build_sanity_checks(snapshot, expected_total=21)
    assert not failed["ok"]
    assert failed["controlled_card_count"] == 20


def test_sanity_allows_one_extra_card_when_waiting_to_discard():
    snapshot = {
        "phase": "await_action",
        "discard_button": {"name": "discard"},
        "hand_count": 21,
        "meld_groups": {"my_melds": []},
    }

    result = build_sanity_checks(snapshot, expected_total=20)

    assert result["ok"]
    assert result["accepted_counts"] == [20, 21]
    assert result["transient_source_card"]


def test_temporal_recovery_restores_one_card_from_last_validated_hand():
    memory = VisionMemory(
        last_screenshot="previous.png",
        last_hand=["一", "二", "三"],
        last_hand_details=[
            {"name": "一", "x": 10, "y": 20, "w": 30, "h": 40, "clickable": True},
            {"name": "二", "x": 50, "y": 20, "w": 30, "h": 40, "clickable": True},
            {"name": "三", "x": 90, "y": 20, "w": 30, "h": 40, "clickable": True},
        ],
        last_hand_expected_total=3,
    )
    snapshot = {
        "hand": ["一", "三"],
        "raw_hand": ["一", "三"],
        "hand_count": 2,
        "hand_details": [],
        "discard_button": {"name": "discard"},
        "buttons": [],
        "discards": {"my_discards": []},
        "meld_groups": {"my_melds": []},
        "metadata": {},
    }

    result = recover_temporally_hidden_hand(snapshot, memory, expected_total=3)

    assert result["hand"] == ["一", "三", "二"]
    recovered = result["hand_details"][-1]
    assert recovered["name"] == "二"
    assert not recovered["clickable"]
    assert recovered["recognition_source"] == "temporal_validated_hand"
    assert result["sanity_checks"]["ok"]


def test_temporal_recovery_restores_three_cards_only_for_dealer_opening_occlusion():
    previous = [
        "一",
        "一",
        "二",
        "七",
        "叁",
        "四",
        "肆",
        "伍",
        "五",
        "五",
        "陆",
        "陆",
        "柒",
        "捌",
        "捌",
        "九",
        "九",
        "九",
        "拾",
        "拾",
        "拾",
    ]
    current = list(previous)
    current.remove("九")
    current.remove("拾")
    current.remove("拾")
    memory = VisionMemory(
        last_screenshot="dealer_opening_visible.png",
        last_hand=previous,
        last_hand_expected_total=21,
    )
    snapshot = {
        "hand": current,
        "raw_hand": current,
        "hand_count": 18,
        "hand_details": [],
        "discard_button": {"name": "discard"},
        "buttons": [],
        "discards": {"my_discards": []},
        "meld_groups": {"my_melds": []},
        "metadata": {},
    }

    result = recover_temporally_hidden_hand(snapshot, memory, expected_total=21)

    assert Counter(result["hand"]) == Counter(previous)
    recovered = result["hand_details"][-3:]
    assert len(recovered) == 3
    assert all(not card["clickable"] for card in recovered)
    assert result["sanity_checks"]["ok"]


def test_temporal_recovery_does_not_restore_three_cards_outside_dealer_opening():
    memory = VisionMemory(last_hand=["一", "二", "三", "四"], last_hand_expected_total=4)
    snapshot = {
        "hand": ["一"],
        "hand_count": 1,
        "discard_button": {"name": "discard"},
        "buttons": [],
        "discards": {"my_discards": []},
        "meld_groups": {"my_melds": []},
        "metadata": {},
    }

    assert recover_temporally_hidden_hand(snapshot, memory, expected_total=4) is snapshot


def test_temporal_recovery_does_not_readd_card_outside_decision_window():
    memory = VisionMemory(last_hand=["一", "二", "三"], last_hand_expected_total=3)
    snapshot = {
        "hand": ["一", "三"],
        "hand_count": 2,
        "buttons": [],
        "discards": {"my_discards": []},
        "meld_groups": {"my_melds": []},
        "metadata": {},
    }

    assert recover_temporally_hidden_hand(snapshot, memory, expected_total=3) is snapshot


def test_temporal_recovery_rejects_changed_hand():
    memory = VisionMemory(last_hand=["一", "二", "三"], last_hand_expected_total=3)
    snapshot = {
        "hand": ["一", "四"],
        "hand_count": 2,
        "discard_button": {"name": "discard"},
        "buttons": [],
        "discards": {"my_discards": []},
        "meld_groups": {"my_melds": []},
        "metadata": {},
    }

    assert recover_temporally_hidden_hand(snapshot, memory, expected_total=3) is snapshot


def test_memory_keeps_last_validated_hand_when_current_count_is_invalid():
    memory = VisionMemory(last_hand=["一", "二", "三"], last_hand_expected_total=3)
    memory.update_from_snapshot(
        {
            "screenshot": "bad.png",
            "hand": ["一", "三"],
            "hand_details": [],
            "discards": {},
            "meld_groups": {},
            "sanity_checks": {"ok": False, "warnings": ["count_mismatch"], "expected_total": 3},
        }
    )

    assert memory.last_hand == ["一", "二", "三"]
