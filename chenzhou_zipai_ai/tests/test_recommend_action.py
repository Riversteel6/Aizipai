"""Tests for screenshot-to-policy recommendation."""

from pathlib import Path
from types import SimpleNamespace

from chenzhou_zipai_ai.game_logging import GameLogger, LoggerConfig
from control.action_plan import build_action_plan
from tools.recommend_action import (
    _apply_hidden_cards,
    _apply_protocol_state,
    _clear_stale_options_when_discard_button_visible,
    _force_visible_hu_priority,
    _hold_for_opponent_priority,
    _inject_semantic_chi_options_before_policy,
    _repair_chi_options_before_policy,
    _resolve_project_path,
    _validate_pass_response_context,
    _visible_hu_override_plan,
    _visible_option_override_plan,
    recommend_from_screenshot,
)
from tools.qs_packet_extract import extract_qs_packets, write_qs_jsonl
from tools.live_assistant import _should_hold_after_chi_click
from vision.history_memory import VisionMemory, save_memory


def test_recommend_action_resolves_nested_screen_config_from_workspace_root():
    resolved = _resolve_project_path("config/screen_1080x2400.yaml")

    assert resolved.exists()
    assert resolved.name == "screen_1080x2400.yaml"


def test_apply_hidden_cards_adds_nonclickable_card_and_rechecks_sanity():
    state = {
        "raw_hand": ["一", "二"],
        "hand": ["一", "二"],
        "hand_details": [
            {"card_id": "h001", "name": "一", "confidence": 0.99, "clickable": True},
            {"card_id": "h002", "name": "二", "confidence": 0.99, "clickable": True},
        ],
        "meld_groups": {"my_melds": []},
        "hand_count": 2,
        "metadata": {},
    }

    patched = _apply_hidden_cards(state, ["拾"], expected_total=3)

    assert patched["hand"] == ["一", "二", "拾"]
    assert patched["hand_details"][-1]["clickable"] is False
    assert patched["hand_details"][-1]["source"] == "hidden_manual_override"
    assert patched["metadata"]["hidden_cards"] == ["拾"]
    assert patched["sanity_checks"]["ok"] is True


def test_apply_protocol_state_overlays_hand_labels_and_keeps_coordinates():
    state = {
        "raw_hand": ["一", "八", "三"],
        "hand": ["一", "八", "三"],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 10, "y": 20, "clickable": True},
            {"card_id": "h002", "name": "八", "x": 30, "y": 20, "clickable": True},
            {"card_id": "h003", "name": "三", "x": 50, "y": 20, "clickable": True},
        ],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }

    patched = _apply_protocol_state(
        state,
        {"playerholdcards": [1, 2, 3], "canchi": True, "showGuo": True},
        expected_total=3,
    )

    assert patched["hand"] == ["一", "二", "三"]
    assert patched["hand_details"][1]["x"] == 30
    assert patched["hand_details"][1]["name"] == "二"
    assert patched["hand_details"][1]["source"] == "protocol_overlay"
    assert patched["metadata"]["protocol_hand_binding"]["clickable_bound_count"] == 3
    assert [item["type"] for item in patched["legal_actions"]] == ["chi", "pass"]
    assert patched["metadata"]["recognition_source"] == "protocol_with_screenshot_coordinates"
    assert patched["sanity_checks"]["ok"] is True


def test_apply_protocol_state_sorts_slots_and_keeps_protocol_click_target():
    state = {
        "raw_hand": ["bad1", "bad2"],
        "hand": ["bad1", "bad2"],
        "hand_details": [
            {"card_id": "h002", "name": "bad2", "x": 120, "y": 20, "w": 100, "h": 120, "clickable": True},
            {"card_id": "h001", "name": "bad1", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True},
        ],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }

    patched = _apply_protocol_state(state, {"playerholdcards": ["1s", "2s"]}, expected_total=2)
    plan = build_action_plan(patched, {"action": "discard", "label": "二", "selected_card_id": "h002"})

    assert patched["hand"] == ["一", "二"]
    assert [item["card_id"] for item in patched["hand_details"]] == ["h001", "h002"]
    assert plan["ready"] is True
    assert plan["target_card_id"] == "h002"
    assert plan["clicks"][0] == {"target": "hand:二", "x": 170, "y": 80, "delay_ms": 80}


def test_apply_protocol_state_prefers_label_match_when_protocol_order_differs_from_screen():
    state = {
        "raw_hand": ["王", "壹", "壹", "王"],
        "hand": ["王", "壹", "壹", "王"],
        "hand_details": [
            {"card_id": "h001", "name": "王", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True},
            {"card_id": "h002", "name": "壹", "x": 10, "y": 150, "w": 100, "h": 120, "clickable": True},
            {"card_id": "h003", "name": "壹", "x": 10, "y": 280, "w": 100, "h": 120, "clickable": True},
            {"card_id": "h004", "name": "王", "x": 120, "y": 20, "w": 100, "h": 120, "clickable": True},
        ],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }

    patched = _apply_protocol_state(state, {"playerholdcards": ["1b", "1b", "Qh", "Qh"]}, expected_total=4)

    assert [item["card_id"] for item in patched["hand_details"]] == ["h002", "h003", "h001", "h004"]
    assert [item["protocol_binding"] for item in patched["hand_details"]] == ["exact_label"] * 4


def test_apply_protocol_state_marks_cards_without_coordinates_nonclickable():
    state = {
        "raw_hand": ["一"],
        "hand": ["一"],
        "hand_details": [{"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True}],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }

    patched = _apply_protocol_state(state, {"playerholdcards": ["1s", "2s", "Qh"]}, expected_total=3)

    assert patched["hand"] == ["一", "二", "王"]
    assert [item["clickable"] for item in patched["hand_details"]] == [True, False, False]
    assert [item["card_id"] for item in patched["hand_details"]] == ["h001", "protocol_002", "protocol_003"]
    assert patched["metadata"]["protocol_hand_binding"]["missing_coordinate_count"] == 2


def test_apply_protocol_state_keeps_visible_button_coordinates_for_protocol_actions():
    state = {
        "raw_hand": ["一"],
        "hand": ["一"],
        "hand_details": [{"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True}],
        "buttons": [
            {"name": "pass", "x": 100, "y": 200, "w": 80, "h": 40},
            {"name": "peng", "x": 220, "y": 200, "w": 80, "h": 40},
        ],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }

    patched = _apply_protocol_state(state, {"playerholdcards": ["1s"], "showGuo": 1}, expected_total=1)
    plan = build_action_plan(patched, {"action": "pass"})

    assert patched["buttons"] == [{"name": "pass", "x": 100, "y": 200, "w": 80, "h": 40, "source": "protocol_allowed_visible_button"}]
    assert plan["ready"] is True
    assert plan["clicks"] == [{"target": "button:pass", "x": 140, "y": 220, "delay_ms": 80}]


def test_apply_protocol_state_drops_protocol_buttons_without_coordinates():
    state = {
        "raw_hand": ["一"],
        "hand": ["一"],
        "hand_details": [{"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True}],
        "buttons": [],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }

    patched = _apply_protocol_state(state, {"playerholdcards": ["1s"], "showGuo": 1}, expected_total=1)

    assert patched["buttons"] == []


def test_apply_protocol_state_ignores_empty_pending_pass_without_clickable_button():
    state = {
        "raw_hand": ["一", "二"],
        "hand": ["一", "二"],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True},
            {"card_id": "h002", "name": "二", "x": 120, "y": 20, "w": 100, "h": 120, "clickable": True},
        ],
        "buttons": [],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }
    protocol = {
        "playerholdcards": ["1s", "2s"],
        "showGuo": 1,
        "metadata": {"qs_payload": {"cmd": 1014, "precardval": ""}},
    }

    patched = _apply_protocol_state(state, protocol, expected_total=2)

    assert patched.get("legal_actions") is None
    assert patched["buttons"] == []
    assert patched["metadata"]["ignored_protocol_legal_actions"]["reason"] == "empty_pending_pass_without_clickable_button"


def test_apply_protocol_state_skips_stale_overlay_without_visual_game_anchor():
    state = {
        "raw_hand": [],
        "hand": [],
        "hand_details": [],
        "buttons": [],
        "discard_button": None,
        "options": [],
        "option_details": [],
        "discards": {},
        "melds": {},
        "meld_groups": {"my_melds": []},
        "metadata": {},
    }
    protocol = {
        "playerholdcards": ["1s", "1s", "1s", "7b", "9b", "9b"],
        "showGuo": 1,
        "metadata": {"qs_payload": {"cmd": 1014, "precardval": ""}},
    }

    patched = _apply_protocol_state(state, protocol, expected_total=None)

    assert patched["hand"] == []
    assert patched["hand_details"] == []
    assert patched["legal_actions"] == [{"type": "WAIT", "source": "no_visual_game_anchor"}]
    assert patched["metadata"]["protocol_overlay_skipped"]["reason"] == "no_visual_game_anchor"


def test_apply_protocol_state_keeps_empty_pending_pass_when_visible_button_is_clickable():
    state = {
        "raw_hand": ["一"],
        "hand": ["一"],
        "hand_details": [{"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True}],
        "buttons": [{"name": "pass", "x": 100, "y": 200, "w": 80, "h": 40}],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }
    protocol = {
        "playerholdcards": ["1s"],
        "showGuo": 1,
        "metadata": {"qs_payload": {"cmd": 1014, "precardval": ""}},
    }

    patched = _apply_protocol_state(state, protocol, expected_total=1)

    assert [item["type"] for item in patched["legal_actions"]] == ["pass"]
    assert patched["buttons"][0]["name"] == "pass"


def test_apply_protocol_state_infers_missing_hu_from_four_button_protocol():
    state = {
        "raw_hand": ["一"],
        "hand": ["一"],
        "hand_details": [{"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True}],
        "buttons": [
            {"name": "chi", "x": 1610, "y": 435, "w": 150, "h": 170},
            {"name": "peng", "x": 1886, "y": 450, "w": 150, "h": 170},
            {"name": "pass", "x": 2090, "y": 420, "w": 200, "h": 205},
        ],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }

    patched = _apply_protocol_state(
        state,
        {"playerholdcards": ["1s"], "canhu": 1, "canpeng": 1, "canchi": 1, "showGuo": 1},
        expected_total=1,
    )
    buttons = {button["name"]: button for button in patched["buttons"]}
    plan = build_action_plan(patched, {"action": "hu"})

    assert set(buttons) == {"hu", "chi", "peng", "pass"}
    assert buttons["hu"]["source"] == "protocol_inferred_visible_button"
    assert 1350 <= buttons["hu"]["x"] + buttons["hu"]["w"] // 2 <= 1500
    assert plan["ready"] is True
    assert plan["clicks"][0]["target"] == "button:hu"


def test_apply_protocol_state_preserves_visible_discard_phase_over_stale_pass_prompt():
    state = {
        "raw_hand": ["一"],
        "hand": ["一"],
        "hand_details": [{"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True}],
        "legal_actions": [{"type": "discard"}],
        "buttons": [],
        "discard_button": {"name": "discard", "x": 100, "y": 200, "w": 80, "h": 40},
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }
    protocol = {
        "playerholdcards": ["1s"],
        "showGuo": 1,
        "metadata": {"qs_payload": {"cmd": 1014, "precardval": "-"}},
    }

    patched = _apply_protocol_state(state, protocol, expected_total=1)

    assert patched["legal_actions"] == [{"type": "discard"}]
    assert patched["buttons"] == []


def test_apply_protocol_state_preserves_visible_discard_button_without_visible_legal_actions():
    state = {
        "raw_hand": ["一"],
        "hand": ["一"],
        "hand_details": [{"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True}],
        "buttons": [],
        "discard_button": {"name": "discard", "x": 100, "y": 200, "w": 80, "h": 40},
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }
    protocol = {
        "playerholdcards": ["1s"],
        "showGuo": 1,
        "metadata": {"qs_payload": {"cmd": 1014, "precardval": "-"}},
    }

    patched = _apply_protocol_state(state, protocol, expected_total=1)

    assert patched.get("legal_actions") is None
    assert patched["buttons"] == []


def test_apply_protocol_state_preserves_visible_discard_button_with_protocol_pending_card():
    state = {
        "raw_hand": ["一"],
        "hand": ["一"],
        "hand_details": [{"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True}],
        "buttons": [],
        "discard_button": {"name": "discard", "x": 100, "y": 200, "w": 80, "h": 40},
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }
    protocol = {
        "playerholdcards": ["1s"],
        "showGuo": 1,
        "precardval": "9b",
        "metadata": {"qs_payload": {"cmd": 1014, "precardval": "9b"}},
    }

    patched = _apply_protocol_state(state, protocol, expected_total=1)

    assert patched.get("legal_actions") is None
    assert patched["buttons"] == []


def test_apply_protocol_state_drops_extra_visible_slots_from_hand_details():
    state = {
        "raw_hand": ["一", "二", "三"],
        "hand": ["一", "二", "三"],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 10, "y": 20, "clickable": True},
            {"card_id": "h002", "name": "二", "x": 30, "y": 20, "clickable": True},
            {"card_id": "h003", "name": "三", "x": 50, "y": 20, "clickable": True},
        ],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }

    patched = _apply_protocol_state(state, {"playerholdcards": ["1s", "2s"]}, expected_total=2)

    assert patched["hand"] == ["一", "二"]
    assert len(patched["hand_details"]) == 2
    assert patched["metadata"]["protocol_hand_binding"]["extra_visible_slot_count"] == 1


def test_apply_protocol_state_reads_qs_packet_jsonl(tmp_path):
    state = {
        "raw_hand": ["一"],
        "hand": ["一"],
        "hand_details": [{"card_id": "h001", "name": "一", "x": 10, "y": 20, "clickable": True}],
        "metadata": {},
        "meld_groups": {"my_melds": []},
    }
    packet = _qs_packet(
        1014,
        _i(2) + _s("7s") + _i(1) + _i(0) + _i(1) + _i(0) + _s("a") + _i(0) + _i(0) + _i(0),
    )
    protocol_log = write_qs_jsonl(extract_qs_packets(packet), tmp_path / "qs_packets.jsonl")

    patched = _apply_protocol_state(state, protocol_log, expected_total=1)

    assert patched["pending_card"] == "七"
    assert [item["type"] for item in patched["legal_actions"]] == ["chi"]
    assert patched["metadata"]["protocol_state"]["qs_cmd"] == 1014


def test_pass_context_blocks_when_only_pass_button_is_detected():
    reason = _validate_pass_response_context(
        {"buttons": [{"name": "pass"}]},
        {"action_evals": [{"type": "PASS"}]},
    )

    assert reason == "只识别到过按钮，疑似吃碰胡按钮漏识别，暂停自动过"


def test_pass_context_allows_only_pass_when_candidate_ui_is_open():
    reason = _validate_pass_response_context(
        {
            "buttons": [{"name": "pass"}],
            "option_details": [
                {"region_name": "chi_options", "labels": ["伍", "伍", "伍"], "center": [1710, 208]}
            ],
        },
        {
            "action_evals": [
                {"type": "PASS"},
                {"type": "CHI", "allowed": False, "reject_reason": "chi_option_binding_missing"},
            ]
        },
    )

    assert reason is None


def test_recommend_action_passes_without_expanding_when_semantic_chi_is_worse(monkeypatch):
    fake_state = {
        "screenshot": "fake.png",
        "screenshot_path": "fake.png",
        "raw_hand": ["二", "七", "十", "九"],
        "hand": ["二", "七", "十", "九"],
        "remaining_deck_count": 30,
        "pending_card": "八",
        "pending_action_card": {"name": "八", "confidence": 0.98},
        "opponent_pending_card": {"name": "八", "confidence": 0.98},
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "chi"}, {"type": "pass"}],
        "buttons": [
            {"name": "chi", "x": 1825, "y": 420, "w": 200, "h": 200},
            {"name": "pass", "x": 2090, "y": 420, "w": 200, "h": 205},
        ],
        "hand_details": [
            {"card_id": "h001", "name": "二", "x": 10, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h002", "name": "七", "x": 120, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h003", "name": "十", "x": 230, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h004", "name": "九", "x": 340, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": None,
        "hand_count": 4,
    }

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)

    result = recommend_from_screenshot("fake.png")

    assert result["decision"]["action"] == "pass"
    assert result["decision"]["candidate_stage"] == "pass"
    assert result["action_plan"]["action"] == "pass"
    assert result["pending_action_card"]["name"] == "八"
    assert result["opponent_pending_card"]["name"] == "八"
    assert result["action_plan"]["clicks"][0]["target"] == "button:pass"
    assert result["conflict_guard_result"]["passed"] is True


def test_recommend_action_expands_only_after_semantic_chi_is_selected(monkeypatch):
    hand = ["七", "九", "一", "四", "六"]
    pending = "八"
    fake_state = {
        "screenshot": "fake.png",
        "screenshot_path": "fake.png",
        "raw_hand": hand,
        "hand": hand,
        "remaining_deck_count": 30,
        "pending_card": pending,
        "pending_action_card": {"name": pending, "confidence": 0.99},
        "opponent_pending_card": {"name": pending, "confidence": 0.99},
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "chi"}, {"type": "pass"}],
        "buttons": [
            {"name": "chi", "x": 1825, "y": 420, "w": 200, "h": 200},
            {"name": "pass", "x": 2090, "y": 420, "w": 200, "h": 205},
        ],
        "hand_details": [
            {
                "card_id": f"h{index:03d}",
                "name": label,
                "x": index * 100,
                "y": 20,
                "w": 80,
                "h": 120,
                "confidence": 0.99,
                "clickable": True,
            }
            for index, label in enumerate(hand, start=1)
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": None,
        "hand_count": len(hand),
    }
    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)

    result = recommend_from_screenshot("fake.png")

    assert result["decision"]["action"] == "chi"
    assert result["action_plan"]["action"] == "chi"
    assert result["action_plan"]["ready"] is True
    assert result["action_plan"]["target_option_cards"] == ["七", "八", "九"]
    assert [click["target"] for click in result["action_plan"]["clicks"]] == ["button:chi"]


def test_recommend_action_rechecks_instead_of_clicking_chi_from_untrusted_pending_card(monkeypatch):
    hand = ["七", "九", "一", "四", "六"]
    fake_state = {
        "screenshot": "fake.png",
        "screenshot_path": "fake.png",
        "raw_hand": hand,
        "hand": hand,
        "remaining_deck_count": 30,
        "pending_action_card": {"name": "五", "confidence": 0.56},
        "opponent_pending_card": {"name": "五", "confidence": 0.56},
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "chi"}, {"type": "pass"}],
        "buttons": [
            {"name": "chi", "x": 1825, "y": 420, "w": 200, "h": 200},
            {"name": "pass", "x": 2090, "y": 420, "w": 200, "h": 205},
        ],
        "hand_details": [
            {
                "card_id": f"h{index:03d}",
                "name": label,
                "x": index * 100,
                "y": 20,
                "w": 80,
                "h": 120,
                "confidence": 0.99,
                "clickable": True,
            }
            for index, label in enumerate(hand, start=1)
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": None,
        "hand_count": len(hand),
    }
    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)

    result = recommend_from_screenshot("fake.png")

    assert result["action_plan"]["action"] == "wait_chi_candidate_recheck"
    assert result["action_plan"]["ready"] is False
    assert result["action_plan"]["clicks"] == []
    assert result["action_plan"]["validation"]["reason_code"] == "pending_card_untrusted"


def test_semantic_chi_options_are_generated_without_visible_candidate_ui():
    state = {
        "hand": ["七", "九", "二", "十"],
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "pending_action_card": {"name": "八", "confidence": 0.99},
        "option_details": [],
        "options": [],
    }

    patched = _inject_semantic_chi_options_before_policy(
        state,
        {"rules": {"allow_1510": False}},
    )

    assert patched["chi_options"]
    assert any(option["labels"] == ["七", "八", "九"] for option in patched["chi_options"])
    assert all(option["semantic_only"] for option in patched["chi_options"])
    assert patched["option_details"] == []


def test_semantic_chi_does_not_commit_from_a_low_confidence_visual_pending_card():
    state = {
        "hand": ["七", "九", "二", "十"],
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "pending_action_card": {"name": "五", "confidence": 0.56},
        "option_details": [],
        "options": [],
    }

    patched = _inject_semantic_chi_options_before_policy(
        state,
        {"rules": {"allow_1510": False}},
    )

    assert "chi_options" not in patched
    assert patched["metadata"]["semantic_chi_pending_card_untrusted"]["reason"] == (
        "no_exact_protocol_or_confident_visual_pending_card"
    )


def test_semantic_chi_prefers_exact_protocol_card_over_visual_guess():
    state = {
        "hand": ["柒", "捌", "拾", "二"],
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "pending_card": "玖",
        "pending_action_card": {"name": "五", "confidence": 0.99},
        "option_details": [],
        "options": [],
    }

    patched = _inject_semantic_chi_options_before_policy(
        state,
        {"rules": {"allow_1510": False}},
    )

    assert any(option["labels"] == ["柒", "捌", "玖"] for option in patched["chi_options"])
    assert patched["metadata"]["semantic_chi_plans"]["pending_card"] == "玖"
    assert patched["metadata"]["semantic_chi_plans"]["pending_card_source"] == "protocol"


def test_visible_chi_options_are_never_replaced_by_semantic_options():
    visible = {
        "option_id": "visible_001",
        "region_name": "chi_options",
        "labels": ["七", "八", "九"],
        "center": [1600, 200],
    }
    state = {
        "hand": ["七", "九"],
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "pending_action_card": {"name": "八", "confidence": 0.99},
        "option_details": [visible],
        "options": [visible],
    }

    patched = _inject_semantic_chi_options_before_policy(
        state,
        {"rules": {"allow_1510": False}},
    )

    assert "chi_options" not in patched
    assert patched["option_details"] == [visible]


def test_recommend_action_reuses_precomputed_vision_state_for_policy_rerun(monkeypatch):
    inspect_calls = []
    fake_state = {
        "screenshot": "fake.png",
        "screenshot_path": "fake.png",
        "raw_hand": ["二", "七", "十", "九"],
        "hand": ["二", "七", "十", "九"],
        "remaining_deck_count": 30,
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "discard"}],
        "buttons": [],
        "hand_details": [
            {
                "card_id": f"h{index:03d}",
                "name": label,
                "x": index * 100,
                "y": 20,
                "w": 80,
                "h": 120,
                "confidence": 0.99,
                "clickable": True,
            }
            for index, label in enumerate(["二", "七", "十", "九"], start=1)
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": {"x": 2000, "y": 400, "w": 100, "h": 100},
        "hand_count": 4,
        "metadata": {},
        "discards": {},
    }

    def fake_inspect(*args, **kwargs):
        inspect_calls.append(True)
        return fake_state

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", fake_inspect)

    first = recommend_from_screenshot("fake.png", include_recognized_state=True)
    second = recommend_from_screenshot(
        "fake.png",
        expected_total=4,
        precomputed_state=first["_recognized_state"],
        update_memory=False,
    )

    assert inspect_calls == [True]
    assert second["sanity_checks"]["ok"] is True


def test_recommend_action_uses_same_memory_context_for_policy_and_guard(monkeypatch, tmp_path):
    labels = ["二", "七", "十", "九"]
    fake_state = {
        "screenshot": "fake.png",
        "screenshot_path": "fake.png",
        "raw_hand": labels,
        "hand": labels,
        "remaining_deck_count": 39,
        "sanity_checks": {"ok": True, "expected_total": 4, "controlled_card_count": 4},
        "legal_actions": [{"type": "discard"}],
        "buttons": [],
        "hand_details": [
            {
                "card_id": f"h{index:03d}",
                "name": label,
                "x": index * 100,
                "y": 20,
                "w": 80,
                "h": 120,
                "confidence": 0.99,
                "clickable": True,
            }
            for index, label in enumerate(labels, start=1)
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": {"x": 2000, "y": 400, "w": 100, "h": 100},
        "hand_count": 4,
        "metadata": {},
        "discards": {},
    }
    memory_file = tmp_path / "memory.json"
    assert save_memory(VisionMemory(my_meld_groups=[["壹", "贰", "叁"]]), memory_file)
    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)

    result = recommend_from_screenshot(
        "fake.png",
        expected_total=4,
        memory_file=memory_file,
        update_memory=False,
    )

    professional = result["professional_brain"]
    expected_context = professional["decision"]["context_snapshot"]["context"]
    assert professional["guard_result"]["passed"] is True
    assert professional["context"]["existing_melds"] == expected_context["existing_melds"]
    assert len(expected_context["existing_melds"]) == 1


def test_recommend_action_uses_protocol_opponent_priority(monkeypatch):
    fake_state = {
        "screenshot": "fake.png",
        "screenshot_path": "fake.png",
        "raw_hand": ["一", "二"],
        "hand": ["一", "二"],
        "remaining_deck_count": 39,
        "pending_card": "十",
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "chi"}, {"type": "pass"}],
        "buttons": [
            {"name": "chi", "x": 1845, "y": 390, "w": 230, "h": 235},
            {"name": "pass", "x": 2090, "y": 420, "w": 200, "h": 205},
        ],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h002", "name": "二", "x": 120, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": None,
        "metadata": {},
    }
    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)

    result = recommend_from_screenshot(
        "fake.png",
        protocol_payload={
            "playerholdcards": ["1s", "2s"],
            "precardval": "Ts",
            "canchi": 1,
            "showGuo": 1,
            "selfChairId": 1,
            "drawChairId": 2,
        },
    )

    assert result["metadata"]["opponent_priority_pending"] is True
    assert result["action_plan"]["action"] == "pass"
    assert result["action_plan"]["clicks"] != [{"target": "button:chi", "x": 1960, "y": 507, "delay_ms": 80}]


def test_visible_chi_option_override_holds_low_confidence_choice_without_reselection():
    chosen = SimpleNamespace(labels=["九", "十"], reason="顺子", type="sequence")
    plan = _visible_option_override_plan(
        {"pending_card": "八"},
        option_stage="chi",
        stage_options=[
            {"labels": ["六", "七"], "confidence": 0.977, "x": 1824, "y": 129, "center": [1866, 194]},
            {"labels": ["九", "十"], "confidence": 0.485, "x": 1881, "y": 129, "center": [1922, 194]},
        ],
        chosen_option=chosen,
        all_unclear=False,
        current_action="chi_option",
    )

    assert plan["ready"] is False
    assert plan["target"] == "chi:九十"
    assert plan["target_option_cards"] == ["九", "十"]
    assert plan["clicks"] == []
    assert "置信度 0.485 过低" in plan["reason"]


def test_visible_chi_option_override_holds_when_policy_option_is_missing():
    plan = _visible_option_override_plan(
        {"pending_card": "壹"},
        option_stage="chi",
        stage_options=[
            {"labels": ["贰", "叁"], "confidence": 0.977, "x": 1824, "y": 129, "center": [1866, 194]},
            {"labels": ["玖", "拾"], "confidence": 0.985, "x": 1881, "y": 129, "center": [1922, 194]},
        ],
        chosen_option=None,
        all_unclear=False,
        current_action="chi_option",
    )

    assert plan["ready"] is False
    assert plan["target"] is None
    assert plan["target_option_cards"] == []
    assert plan["clicks"] == []
    assert plan["validation"]["reason_code"] == "policy_option_missing"


def test_visible_chi_option_override_rejects_illegal_candidate():
    chosen = SimpleNamespace(labels=["六", "八", "捌"], reason="组合价值不明确", type="unknown")
    plan = _visible_option_override_plan(
        {"pending_card": "七"},
        option_stage="chi",
        stage_options=[
            {"labels": ["六", "八", "捌"], "confidence": 0.91, "x": 1646, "y": 11, "center": [1710, 208]},
        ],
        chosen_option=chosen,
        all_unclear=False,
        current_action="chi_option",
    )

    assert plan["ready"] is False
    assert plan["clicks"] == []
    assert plan["validation"]["reason_code"] == "invalid_chi_option"


def test_visible_chi_option_override_repairs_size_mismatch_from_hand():
    chosen = SimpleNamespace(labels=["伍", "六", "七"], reason="可信候选", type="sequence")
    plan = _visible_option_override_plan(
        {"hand": ["四", "五", "六"]},
        option_stage="chi",
        stage_options=[
            {"labels": ["伍", "六", "七"], "confidence": 0.91, "x": 1646, "y": 11, "center": [1710, 208]},
        ],
        chosen_option=chosen,
        all_unclear=False,
        current_action="chi_option",
    )

    assert plan["ready"] is True
    assert plan["target"] == "chi:五六七"
    assert plan["target_option_cards"] == ["五", "六", "七"]
    assert plan["clicks"] == [{"target": "chi:五六七", "x": 1710, "y": 208, "delay_ms": 80}]
    assert "按手牌纠正为 五六七" in plan["reason"]


def test_visible_chi_option_override_repairs_same_rank_size_mismatch_from_hand():
    chosen = SimpleNamespace(labels=["伍", "伍", "伍"], reason="可信候选", type="unknown")
    plan = _visible_option_override_plan(
        {"hand": ["五", "伍", "陆"], "pending_card": "五"},
        option_stage="chi",
        stage_options=[
            {"labels": ["伍", "伍", "伍"], "confidence": 0.91, "x": 1646, "y": 11, "center": [1710, 208]},
        ],
        chosen_option=chosen,
        all_unclear=False,
        current_action="chi_option",
    )

    assert plan["ready"] is True
    assert plan["target"] == "chi:五五伍"
    assert plan["target_option_cards"] == ["五", "五", "伍"]
    assert plan["clicks"] == [{"target": "chi:五五伍", "x": 1710, "y": 208, "delay_ms": 80}]
    assert "按手牌纠正为 五五伍" in plan["reason"]


def test_visible_chi_option_override_repairs_pending_same_rank_misread_from_hand():
    chosen = SimpleNamespace(labels=["拾", "拾", "七"], reason="可信候选", type="unknown")
    plan = _visible_option_override_plan(
        {"hand": ["十", "拾", "三"], "pending_card": "拾"},
        option_stage="chi",
        stage_options=[
            {"labels": ["拾", "拾", "七"], "confidence": 0.91, "x": 1646, "y": 11, "center": [1710, 208]},
        ],
        chosen_option=chosen,
        all_unclear=False,
        current_action="chi_option",
    )

    assert plan["ready"] is True
    assert plan["target"] == "chi:拾拾十"
    assert plan["target_option_cards"] == ["拾", "拾", "十"]
    assert plan["clicks"] == [{"target": "chi:拾拾十", "x": 1710, "y": 208, "delay_ms": 80}]
    assert "按手牌纠正为 拾拾十" in plan["reason"]


def test_visible_chi_option_override_repairs_pending_sequence_misread_from_hand():
    chosen = SimpleNamespace(labels=["二", "八", "伍"], reason="可信候选", type="unknown")
    plan = _visible_option_override_plan(
        {"hand": ["二", "三", "七"], "pending_card": "四"},
        option_stage="chi",
        stage_options=[
            {"labels": ["二", "八", "伍"], "confidence": 0.91, "x": 1646, "y": 11, "center": [1710, 208]},
        ],
        chosen_option=chosen,
        all_unclear=False,
        current_action="chi_option",
    )

    assert plan["ready"] is True
    assert plan["target"] == "chi:二三四"
    assert plan["target_option_cards"] == ["二", "三", "四"]
    assert plan["clicks"] == [{"target": "chi:二三四", "x": 1710, "y": 208, "delay_ms": 80}]
    assert "按手牌纠正为 二三四" in plan["reason"]


def test_repairs_chi_option_labels_before_policy_evaluation():
    state = {
        "hand": ["二", "三", "七"],
        "pending_card": "四",
        "option_details": [
            {
                "region_name": "chi_options",
                "option_id": "chi_001",
                "labels": ["二", "八", "伍"],
                "center": [1710, 208],
            }
        ],
    }

    patched = _repair_chi_options_before_policy(state)

    assert patched["option_details"][0]["labels"] == ["二", "三", "四"]
    assert patched["options"][0]["labels"] == ["二", "三", "四"]
    assert patched["chi_options"][0]["labels"] == ["二", "三", "四"]
    assert patched["option_details"][0]["raw_labels"] == ["二", "八", "伍"]


def test_visible_chi_option_does_not_override_policy_discard():
    chosen = SimpleNamespace(labels=["玖", "拾"], reason="含红牌", type="sequence")
    plan = _visible_option_override_plan(
        {},
        option_stage="chi",
        stage_options=[
            {"labels": ["玖", "拾"], "confidence": 0.9, "x": 1881, "y": 129, "center": [1922, 194]},
        ],
        chosen_option=chosen,
        all_unclear=False,
        current_action="discard",
    )

    assert plan is None


def test_visible_chi_option_does_not_override_policy_pass():
    chosen = SimpleNamespace(labels=["七", "七", "柒"], reason="同点混搭候选", type="mixed_same_rank")
    plan = _visible_option_override_plan(
        {},
        option_stage="chi",
        stage_options=[
            {"labels": ["七", "七", "柒"], "confidence": 0.804, "x": 1646, "y": 11, "center": [1710, 208]},
        ],
        chosen_option=chosen,
        all_unclear=False,
        current_action="pass",
    )

    assert plan is None


def test_clear_stale_history_options_when_discard_button_is_visible():
    state = {
        "discard_button": {"name": "discard", "center": [1172, 511]},
        "buttons": [],
        "options": [{"region_name": "chi_options"}],
        "option_details": [
            {"region_name": "chi_options", "labels": ["玖", "八"], "center": [1922, 194]},
        ],
        "metadata": {},
    }

    patched = _clear_stale_options_when_discard_button_visible(state)

    assert patched["options"] == []
    assert patched["option_details"] == []
    assert patched["metadata"]["stale_options_cleared"]["reason"] == "discard_button_visible_without_response_buttons"


def test_keep_options_when_response_buttons_are_visible():
    state = {
        "discard_button": {"name": "discard", "center": [1172, 511]},
        "buttons": [{"name": "chi", "center": [1960, 507]}],
        "options": [{"region_name": "chi_options"}],
        "option_details": [
            {"region_name": "chi_options", "labels": ["捌", "玖", "拾"], "center": [1710, 208]},
        ],
        "metadata": {},
    }

    assert _clear_stale_options_when_discard_button_visible(state) is state


def _qs_packet(cmd: int, body: bytes) -> bytes:
    return b"QS" + len(body).to_bytes(2, "little", signed=True) + cmd.to_bytes(2, "little", signed=True) + body


def _i(value: int) -> bytes:
    return value.to_bytes(4, "little", signed=True)


def _s(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return _i(len(raw)) + raw


def test_pass_context_blocks_when_visible_response_button_was_not_evaluated():
    reason = _validate_pass_response_context(
        {"buttons": [{"name": "pass"}, {"name": "peng"}]},
        {"action_evals": [{"type": "PASS"}]},
    )

    assert "peng" in reason


def test_pass_context_blocks_when_hu_button_is_visible():
    reason = _validate_pass_response_context(
        {"buttons": [{"name": "pass"}, {"name": "hu"}]},
        {"action_evals": [{"type": "PASS"}, {"type": "HU", "allowed": True}]},
    )

    assert reason == "胡按钮可见，禁止自动过"


def test_pass_context_allows_when_visible_response_button_was_evaluated():
    reason = _validate_pass_response_context(
        {"buttons": [{"name": "pass"}, {"name": "peng"}]},
        {"action_evals": [{"type": "PASS"}, {"type": "PENG", "allowed": False}]},
    )

    assert reason is None


def test_visible_hu_overrides_non_hu_plan():
    plan = _visible_hu_override_plan(
        {"buttons": [{"name": "pass"}, {"name": "hu", "x": 100, "y": 200, "w": 80, "h": 60}]},
        {"action": "pass"},
    )

    assert plan is not None
    assert plan["action"] == "hu"
    assert plan["ready"] is True


def test_protocol_hu_legal_action_halts_when_button_is_not_visible_by_default():
    plan = _visible_hu_override_plan(
        {"buttons": [], "legal_actions": [{"type": "HU"}, {"type": "PASS"}]},
        {"action": "pass"},
    )

    assert plan is not None
    assert plan["action"] == "hu"
    assert plan["ready"] is False
    assert plan["clicks"] == []
    assert plan["validation"]["reason_code"] == "hu_button_not_visible"


def test_protocol_hu_without_visible_button_fails_guard_instead_of_faking_validation():
    plan, snapshot = _force_visible_hu_priority(
        {"buttons": [], "legal_actions": [{"type": "HU"}, {"type": "PASS"}]},
        {
            "action": "pass",
            "ready": True,
            "clicks": [{"target": "button:pass", "x": 100, "y": 200}],
            "validation": {"passed": True},
        },
        {"guard_result": {"passed": True}},
    )

    assert plan["action"] == "hu"
    assert plan["ready"] is False
    assert plan["validation"]["passed"] is False
    assert plan["validation"]["reason_code"] == "hu_button_not_visible"
    assert snapshot["guard_result"]["passed"] is False
    assert snapshot["guard_result"]["safe_halt"] is True
    assert snapshot["guard_result"]["reason"] == "hu_button_not_visible"


def test_selected_hu_without_visible_button_uses_the_same_hard_stop():
    plan, snapshot = _force_visible_hu_priority(
        {"buttons": [], "legal_actions": [{"type": "HU"}]},
        {
            "action": "hu",
            "ready": False,
            "reason": "未找到 hu 按钮",
            "clicks": [],
            "validation": {"passed": False, "reason_code": "action_plan_policy_mismatch"},
        },
        {"guard_result": {"passed": True}},
    )

    assert plan["ready"] is False
    assert plan["validation"]["reason_code"] == "hu_button_not_visible"
    assert snapshot["guard_result"]["passed"] is False
    assert snapshot["guard_result"]["reason"] == "hu_button_not_visible"


def test_visible_hu_keeps_priority_after_option_override():
    plan, snapshot = _force_visible_hu_priority(
        {
            "buttons": [
                {"name": "pass"},
                {"name": "hu", "x": 1910, "y": 425, "w": 150, "h": 170},
            ],
            "option_details": [
                {"region_name": "chi_options", "labels": ["六", "八"], "center": [1753, 171]},
            ],
        },
        {
            "action": "chi_option",
            "ready": True,
            "reason": "候选区已展开，选择 六八",
            "clicks": [{"target": "chi:六八", "x": 1753, "y": 171, "delay_ms": 80}],
        },
        {"guard_result": {"checks": ["visible_option_override"]}},
    )

    assert plan["action"] == "hu"
    assert plan["clicks"] == [{"target": "button:hu", "x": 1985, "y": 510, "delay_ms": 80}]
    assert snapshot["guard_result"]["checks"] == ["visible_hu_override", "target_clickable"]


def test_protocol_opponent_priority_blocks_response_and_option_clicks():
    plan, snapshot = _hold_for_opponent_priority(
        {
            "buttons": [{"name": "pass"}, {"name": "hu", "center": [1985, 510]}],
            "discard_button": None,
            "option_stage": "chi",
            "option_details": [
                {"region_name": "chi_options", "labels": ["叁", "叁", "叁"], "center": [1838, 208]},
            ],
            "metadata": {
                "opponent_priority_pending": True,
                "opponent_priority_source": "qs_last_draw_chair_id",
                "pending_action_card": {
                    "region_name": "opponent_pending_card",
                    "name": "三",
                    "confidence": 0.982,
                },
            },
        },
        {
            "action": "hu",
            "ready": True,
            "clicks": [{"target": "button:hu", "x": 1985, "y": 510, "delay_ms": 80}],
        },
        {"guard_result": {"checks": ["visible_hu_override"]}},
    )

    assert plan["action"] == "hu"
    assert plan["ready"] is True
    assert plan["clicks"] == [{"target": "button:hu", "x": 1985, "y": 510, "delay_ms": 80}]
    assert snapshot["guard_result"]["checks"] == ["visible_hu_override"]


def test_opponent_priority_guard_allows_visible_planned_response_button():
    original = {
        "action": "pass",
        "ready": True,
        "clicks": [{"target": "button:pass", "x": 2190, "y": 522, "delay_ms": 80}],
    }
    plan, _snapshot = _hold_for_opponent_priority(
        {
            "buttons": [{"name": "pass"}, {"name": "peng", "center": [1961, 535]}, {"name": "chi"}],
            "discard_button": None,
            "metadata": {
                "opponent_priority_pending": True,
                "opponent_priority_source": "stale_protocol_state",
            },
        },
        original,
        {"guard_result": {"checks": ["action_plan_matches_policy"]}},
    )

    assert plan is original


def test_visual_opponent_pending_card_alone_does_not_block_after_priority_passes():
    original = {
        "action": "chi",
        "ready": True,
        "clicks": [{"target": "button:chi", "x": 1925, "y": 520, "delay_ms": 80}],
    }
    plan, _snapshot = _hold_for_opponent_priority(
        {
            "buttons": [{"name": "pass"}, {"name": "chi", "center": [1925, 520]}],
            "discard_button": None,
            "metadata": {
                "pending_action_card": {
                    "region_name": "opponent_pending_card",
                    "name": "三",
                    "confidence": 0.982,
                },
            },
        },
        original,
        {},
    )

    assert plan is original


def test_opponent_priority_guard_does_not_block_own_discard_button():
    plan, _snapshot = _hold_for_opponent_priority(
        {
            "buttons": [],
            "discard_button": {"name": "discard", "center": [1172, 511]},
            "metadata": {
                "pending_action_card": {
                    "region_name": "opponent_pending_card",
                    "name": "三",
                    "confidence": 0.982,
                },
            },
        },
        {"action": "discard", "ready": True, "clicks": [{"target": "hand:三", "x": 877, "y": 864}]},
        {},
    )

    assert plan["action"] == "discard"


def test_live_guard_holds_pass_after_chi_click():
    result = {
        "buttons": [{"name": "pass"}, {"name": "chi"}],
        "options": [],
        "option_details": [],
        "action_plan": {"action": "pass", "ready": True},
    }
    import time

    guard = {"last_executed_signature": ["chi", [["button:chi", 1960, 507]]], "last_executed_at": time.time()}

    assert _should_hold_after_chi_click(result, guard) is True


def test_recommend_action_logs_round_relative_screenshot_for_replay(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    source.write_text("fake screenshot", encoding="utf-8")
    fake_state = {
        "screenshot": str(source),
        "screenshot_path": str(source),
        "raw_hand": ["一", "二", "二", "二", "三"],
        "hand": ["一", "二", "二", "二", "三"],
        "remaining_deck_count": 40,
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "discard"}],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h002", "name": "二", "x": 120, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h003", "name": "二", "x": 230, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h004", "name": "二", "x": 340, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h005", "name": "三", "x": 450, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": {"x": 100, "y": 200, "w": 100, "h": 40},
        "hand_count": 5,
        "phase": "play",
        "metadata": {"remaining_deck_count": 40},
    }

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)
    logger = GameLogger(
        tmp_path / "logs",
        config=LoggerConfig(save_raw_screenshot=True, save_debug_screenshot=False),
    )
    session_id = logger.start_session(device_id="dev")
    round_id = logger.start_round()

    result = recommend_from_screenshot(source, logger=logger, frame_id="frame_000001")
    logger.end_round(result="test")

    round_path = logger.paths.round_dir(session_id, round_id)
    copied = round_path / result["screenshot"]
    assert result["screenshot"] == "screenshots/frame_000001_raw.png"
    assert copied.exists()


def test_recommend_action_returns_seat_role_metadata(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    source.write_text("fake screenshot", encoding="utf-8")
    fake_state = {
        "screenshot": str(source),
        "screenshot_path": str(source),
        "raw_hand": ["一", "二", "二", "二", "三"],
        "hand": ["一", "二", "二", "二", "三"],
        "remaining_deck_count": 40,
        "seat_role": {"role": "dealer", "expected_total": 21, "confidence": 1.0},
        "metadata": {"seat_role": {"role": "dealer", "expected_total": 21, "confidence": 1.0}},
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "discard"}],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h002", "name": "二", "x": 120, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h003", "name": "二", "x": 230, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h004", "name": "二", "x": 340, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h005", "name": "三", "x": 450, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": {"x": 100, "y": 200, "w": 100, "h": 40},
        "hand_count": 5,
        "phase": "play",
    }

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)

    result = recommend_from_screenshot(source, memory_file=None)

    assert result["seat_role"]["role"] == "dealer"
    assert result["metadata"]["seat_role"]["expected_total"] == 21


def test_recommend_action_from_known_screenshot():
    screenshot = Path("data/screenshots/screenshot_20260527_105209.png")
    if not screenshot.exists():
        return

    result = recommend_from_screenshot(
        screenshot,
        expected_total=20,
        memory_file=None,
    )

    assert result["sanity_checks"]["ok"]
    assert "hu_breakdown" in result
    assert "existing_xi" in result
    assert result["decision"]["action"] in {"discard", "hu", "pass"}
    if result["decision"]["action"] == "discard":
        assert result["decision"]["label"] != "王"


def test_recommend_action_waits_during_opponent_priority_pending():
    screenshot = Path("data/screenshots/screenshot_20260527_132742.png")
    if not screenshot.exists():
        return

    result = recommend_from_screenshot(
        screenshot,
        expected_total=20,
        memory_file=None,
        opponent_priority_pending=True,
    )

    assert result["decision"]["action"] == "pass"
    assert result["action_plan"]["action"] == "wait_opponent_priority"
    assert result["action_plan"]["clicks"] == []


def test_recommend_action_blocks_discard_when_sanity_fails():
    screenshot = Path("data/screenshots/screenshot_20260527_142258.png")
    if not screenshot.exists():
        return

    result = recommend_from_screenshot(
        screenshot,
        expected_total=21,
        memory_file=None,
    )

    if result["decision"]["action"] == "discard" and not result["sanity_checks"]["ok"]:
        assert result["action_plan"]["ready"] is False
        assert result["action_plan"]["clicks"] == []


def test_unknown_chi_option_halts_instead_of_left_fallback(monkeypatch):
    class FakeDecision:
        action = "chi"

        def to_dict(self):
            return {
                "action": "chi",
                "label": None,
                "score": 1.0,
                "reason": "fake",
                "evaluations": [],
                "response_evaluations": [],
            }

    fake_state = {
        "screenshot": "fake.png",
        "hand": ["一", "二"],
        "hand_details": [],
        "remaining_deck_count": 40,
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "chi"}],
        "buttons": [{"name": "chi", "x": 50, "y": 50, "w": 80, "h": 40}],
        "meld_groups": {"my_melds": []},
        "options": [{"region_name": "chi_options"}],
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["一", "二", "二"],
                "confidence": 0.9,
                "center": (100, 100),
                "index": 1,
                "x": 80,
                "y": 40,
            },
        ],
    }

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)
    monkeypatch.setattr("tools.recommend_action.choose_action", lambda *args, **kwargs: FakeDecision())

    result = recommend_from_screenshot("fake.png")

    assert result["chosen_option"]["type"] == "unknown"
    assert result["action_plan"]["ready"] is False
    assert result["action_plan"]["clicks"] == []
    assert result["action_plan"]["validation"]["reason_code"] == "chi_option_recognition_uncertain"


def test_unknown_reason_chi_option_halts_instead_of_left_fallback(monkeypatch):
    class FakeDecision:
        action = "chi"

        def to_dict(self):
            return {
                "action": "chi",
                "label": None,
                "score": 1.0,
                "reason": "fake",
                "evaluations": [],
                "response_evaluations": [],
            }

    class FakeEvaluation:
        def __init__(self, labels: list[str], index: int):
            self.labels = labels
            self.type = "complete"
            self.reason = "组合价值不明确"
            self.score = -10.0 + index
            self.action = "chi"
            self.xi = 0

        def to_dict(self) -> dict:
            return {
                "labels": self.labels,
                "action": self.action,
                "score": self.score,
                "type": self.type,
                "xi": self.xi,
                "reason": self.reason,
            }

    fake_state = {
        "screenshot": "fake.png",
        "hand": ["一", "二"],
        "hand_details": [],
        "remaining_deck_count": 40,
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "chi"}],
        "buttons": [{"name": "chi", "x": 50, "y": 50, "w": 80, "h": 40}],
        "meld_groups": {"my_melds": []},
        "options": [{"region_name": "chi_options"}],
        "option_details": [
            {
                "region_name": "chi_options",
                "labels": ["一", "三", "王"],
                "confidence": 0.9,
                "center": (220, 120),
                "index": 1,
                "x": 120,
                "y": 40,
            },
            {
                "region_name": "chi_options",
                "labels": ["四", "六", "王"],
                "confidence": 0.9,
                "center": (180, 120),
                "index": 2,
                "x": 60,
                "y": 40,
            },
        ],
    }

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)
    monkeypatch.setattr("tools.recommend_action.choose_action", lambda *args, **kwargs: FakeDecision())
    monkeypatch.setattr(
        "tools.recommend_action.evaluate_option",
        lambda labels, action="chi", config_path="config/rules.yaml": FakeEvaluation(labels, len(labels)),
    )

    result = recommend_from_screenshot("fake.png")

    assert result["action_plan"]["ready"] is False
    assert result["action_plan"]["clicks"] == []
    assert result["action_plan"]["validation"]["reason_code"] == "chi_option_recognition_uncertain"


def test_recommend_action_includes_full_audit_fields(monkeypatch):
    fake_state = {
        "screenshot": "fake.png",
        "raw_hand": ["參", "叁", "参", "四", "九"],
        "hand": ["參", "叁", "参", "四", "九"],
        "remaining_deck_count": 40,
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "discard"}],
        "hand_details": [
            {"name": "參", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True},
            {"name": "叁", "x": 120, "y": 20, "w": 100, "h": 120, "clickable": True},
            {"name": "参", "x": 230, "y": 20, "w": 100, "h": 120, "clickable": True},
            {"name": "四", "x": 340, "y": 20, "w": 100, "h": 120, "clickable": True},
            {"name": "九", "x": 450, "y": 20, "w": 100, "h": 120, "clickable": True},
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": {"x": 100, "y": 200, "w": 100, "h": 40},
        "hand_count": 5,
        "metadata": {
            "remaining_deck_count": 40,
        },
    }

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)

    result = recommend_from_screenshot("fake.png")

    decision = result["decision"]
    assert decision["action"] == "discard"
    assert decision["selected_discard"] in {"四", "九"}
    assert decision["policy_action"] == "discard"
    assert "hard_protected" in decision and "叁" in decision["hard_protected"]
    assert "candidate_stage" in decision
    assert "discard_candidates_before_filter" in decision
    assert "discard_candidates_after_hard_filter" in decision
    assert "discard_candidates_after_soft_filter" in decision
    assert "selected_reason" in decision
    assert "break_hard_protection" in decision
    assert "break_reason" in decision
    assert result["action_plan_target"] == decision["selected_discard"]
    assert result["action_plan_ready"] == result["action_plan"]["ready"]


def test_recommend_action_uses_professional_brain_as_live_policy(monkeypatch):
    fake_state = {
        "screenshot": "fake.png",
        "screenshot_path": "fake.png",
        "raw_hand": ["一", "二", "二", "二", "三"],
        "hand": ["一", "二", "二", "二", "三"],
        "remaining_deck_count": 40,
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "discard"}],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 10, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h002", "name": "二", "x": 120, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h003", "name": "二", "x": 230, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h004", "name": "二", "x": 340, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h005", "name": "三", "x": 450, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": {"x": 100, "y": 200, "w": 100, "h": 40},
        "hand_count": 5,
    }

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)

    result = recommend_from_screenshot("fake.png")

    assert result["professional_brain"]["live_decision_source"] == "professional_policy_brain"
    assert "legacy_policy_delta" not in result["professional_brain"]
    assert result["decision"]["action"] == "discard"
    assert result["decision"]["label"] in {"一", "三"}
    assert result["decision"]["label"] != "二"
    assert result["action_plan"]["target_card_id"] == result["decision"]["selected_card_id"]
    assert result["action_plan"]["execution_mode"] == "select_then_discard_button"
    assert [click["target"] for click in result["action_plan"]["clicks"]] == [
        f"hand:{result['decision']['label']}",
        "button:discard",
    ]
    assert result["conflict_guard_result"]["passed"] is True
    assert result["action_plan"]["validation"]["passed"] is True
    assert result["action_plan"]["validation"]["self_check"] == "run_decision_self_check"
    assert result["action_plan"]["validation"]["guard_result"]["passed"] is True


def test_recommend_action_prefers_compare_stage_when_visible():
    screenshot = Path("data/screenshots/screenshot_20260527_170608.png")
    if not screenshot.exists():
        return

    result = recommend_from_screenshot(
        screenshot,
        expected_total=20,
        memory_file=None,
    )

    assert result["option_stage"] == "compare"
    assert result["action_plan"]["action"] == "compare_option"
    assert result["action_plan"]["clicks"][0]["target"] == "compare:捌玖拾"


def test_recommend_action_handles_compare_stage_after_professional_chi_selection(monkeypatch):
    fake_state = {
        "screenshot": "fake.png",
        "hand": ["二", "十", "四", "六", "九"],
        "remaining_deck_count": 40,
        "pending_card": "七",
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "hand_details": [
            {"card_id": "h001", "name": "二", "x": 10, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h002", "name": "十", "x": 120, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h003", "name": "四", "x": 230, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h004", "name": "六", "x": 340, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
            {"card_id": "h005", "name": "九", "x": 450, "y": 20, "w": 100, "h": 120, "confidence": 0.99, "clickable": True},
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [
            {
                "region_name": "compare_options",
                "option_id": "compare_123",
                "labels": ["一", "二", "三"],
                "center": (300, 120),
                "index": 1,
                "x": 260,
                "y": 40,
            },
            {
                "region_name": "compare_options",
                "option_id": "compare_234",
                "labels": ["二", "三", "四"],
                "center": (440, 120),
                "index": 2,
                "x": 400,
                "y": 40,
            },
            {
                "region_name": "chi_options",
                "option_id": "chi_2710",
                "labels": ["二", "七", "十"],
                "center": (600, 120),
                "index": 1,
                "x": 560,
                "y": 40,
            },
        ],
        "discard_button": None,
        "hand_count": 5,
    }

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)

    result = recommend_from_screenshot("fake.png")

    assert result["decision"]["action"] == "chi"
    assert result["decision"]["selected_option_id"] == "chi_2710"
    assert result["option_stage"] == "compare"
    assert result["action_plan"]["action"] == "compare_option"
    assert result["action_plan"]["target_option_id"] == "compare_123"
    assert result["action_plan"]["clicks"] == [
        {"target": "compare:一二三", "x": 300, "y": 120, "delay_ms": 80}
    ]
    assert result["conflict_guard_result"]["passed"] is True


def test_recommend_action_ignores_disabled_discard_label(monkeypatch):
    class FakeDecision:
        action = "discard"

        def __init__(self, label: str):
            self.label = label

        def to_dict(self):
            return {
                "action": "discard",
                "label": self.label,
                "score": 1.0,
                "reason": "fake",
                "evaluations": [],
                "response_evaluations": [],
            }

    class FakeDiscardDecision:
        action = "discard"

        def __init__(self, label: str):
            self.label = label

        def to_dict(self):
            return {
                "action": "discard",
                "label": self.label,
                "score": 2.0,
                "reason": "fallback",
                "evaluations": [],
                "response_evaluations": [],
            }

    fake_state = {
        "screenshot": "fake.png",
        "hand": ["叁", "九", "八"],
        "remaining_deck_count": 40,
        "sanity_checks": {"ok": True},
        "legal_actions": [{"type": "discard"}],
        "hand_details": [
            {"name": "叁", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": False},
            {"name": "九", "x": 120, "y": 20, "w": 100, "h": 120, "clickable": True},
            {"name": "八", "x": 230, "y": 20, "w": 100, "h": 120, "clickable": True},
        ],
        "meld_groups": {"my_melds": []},
        "options": [],
        "option_details": [],
        "discard_button": {"x": 100, "y": 200, "w": 100, "h": 40},
        "hand_count": 3,
    }

    monkeypatch.setattr("tools.recommend_action.inspect_screenshot", lambda *args, **kwargs: fake_state)
    monkeypatch.setattr("tools.recommend_action.choose_action", lambda *args, **kwargs: FakeDecision("叁"))
    monkeypatch.setattr("tools.recommend_action.choose_discard", lambda *args, **kwargs: FakeDiscardDecision("九"))

    result = recommend_from_screenshot("fake.png")

    assert result["decision"]["action"] == "discard"
    assert result["decision"]["label"] == "叁"
    assert result["action_plan"]["ready"] is False
    assert result["action_plan"]["reason"] == "conflict_action_plan_reselected_card"
    assert result["action_plan"]["validation"]["self_check"] == "run_decision_self_check"
    assert result["action_plan"]["validation"]["guard_result"]["passed"] is False
