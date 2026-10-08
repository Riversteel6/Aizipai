from vision.protocol_state import normalize_protocol_card, protocol_state_from_payload


def test_protocol_state_normalizes_apk_exposed_fields():
    state = protocol_state_from_payload(
        {
            "playerholdcards": [1, 2, 3, 101, 107, "Tb"],
            "curr_card": 7,
            "out_cards": "8,9,10",
            "canpeng": 1,
            "canchi": True,
            "canhu": 0,
            "showGuo": "true",
            "card_num": "35",
        }
    )

    assert state["hand"] == ["一", "二", "三", "壹", "柒", "拾"]
    assert state["pending_card"] == "七"
    assert state["discards"]["protocol_out_cards"] == ["八", "九", "十"]
    assert [item["type"] for item in state["legal_actions"]] == ["peng", "chi", "pass"]
    assert state["remaining_deck_count"] == 35
    assert state["metadata"]["screenshot_fallback_required"] is False


def test_protocol_state_keeps_apk_seat_fields_for_later_mapping():
    state = protocol_state_from_payload({"playerholdcards": [1, 2], "zhuang": 2, "selfChairId": 1})

    assert state["metadata"]["seat_fields"] == {"zhuang": 2, "selfChairId": 1}


def test_protocol_state_does_not_default_hand_snapshot_to_discard():
    state = protocol_state_from_payload({"playerholdcards": ["1s", "2s"]})

    assert state["hand"] == ["一", "二"]
    assert state["legal_actions"] == []


def test_protocol_state_marks_opponent_priority_when_source_chair_differs():
    state = protocol_state_from_payload(
        {
            "playerholdcards": ["1s", "2s"],
            "precardval": "Ts",
            "canchi": 1,
            "showGuo": 1,
            "selfChairId": 1,
            "drawChairId": 2,
        }
    )

    assert state["pending_card"] == "十"
    assert [item["type"] for item in state["legal_actions"]] == ["chi", "pass"]
    assert state["metadata"]["opponent_priority_pending"] is True


def test_protocol_state_keeps_own_response_actionable_when_source_chair_matches():
    state = protocol_state_from_payload(
        {
            "playerholdcards": ["1s", "2s"],
            "precardval": "Ts",
            "canchi": 1,
            "selfChairId": 1,
            "drawChairId": 1,
        }
    )

    assert state["metadata"]["opponent_priority_pending"] is False


def test_protocol_state_keeps_self_prompt_actionable_after_opponent_draw():
    state = protocol_state_from_payload(
        {
            "playerholdcards": ["1s", "2s"],
            "precardval": "Ts",
            "canchi": 1,
            "showGuo": 1,
            "selfChairId": 1,
            "chairId": 1,
            "drawChairId": 2,
        }
    )

    assert [item["type"] for item in state["legal_actions"]] == ["chi", "pass"]
    assert state["metadata"]["opponent_priority_pending"] is False


def test_protocol_state_parses_frida_message_text():
    state = protocol_state_from_payload(
        {
            "type": "recv",
            "text": "playerholdcards=[1,2,103] curr_card=10 canpeng=1 zhuang=2 chairId=1",
        }
    )

    assert state["hand"] == ["一", "二", "叁"]
    assert state["pending_card"] == "十"
    assert [item["type"] for item in state["legal_actions"]] == ["peng"]
    assert state["metadata"]["seat_fields"] == {"zhuang": 2, "chairId": 1}


def test_protocol_state_uses_latest_protocol_line_from_log(tmp_path):
    log = tmp_path / "hook.log"
    log.write_text(
        "{\"type\":\"recv\",\"text\":\"playerholdcards=[1,2] canchi=true\"}\n"
        "{\"type\":\"recv\",\"text\":\"playerholdcards=[3,4] canhu=true\"}\n",
        encoding="utf-8",
    )

    state = protocol_state_from_payload(log)

    assert state["hand"] == ["三", "四"]
    assert [item["type"] for item in state["legal_actions"]] == ["hu"]


def test_protocol_state_parses_hook_log_text():
    state = protocol_state_from_payload("playerholdcards=[1,2,103] curr_card=10 canchi=true showGuo=1")

    assert state["hand"] == ["一", "二", "叁"]
    assert state["pending_card"] == "十"
    assert [item["type"] for item in state["legal_actions"]] == ["chi", "pass"]


def test_protocol_card_keeps_unknown_numeric_visible():
    assert normalize_protocol_card(77) == "77"


def test_protocol_card_maps_qh_to_wildcard():
    assert normalize_protocol_card("Qh") == "王"
    assert protocol_state_from_payload({"playerholdcards": ["1s", "Qh"], "precardval": "Qh"})["hand"] == ["一", "王"]
