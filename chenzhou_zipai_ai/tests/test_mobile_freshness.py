from tools.mobile_freshness import (
    changed_fresh_context,
    changed_fresh_context_for_action,
    mobile_guard_signature,
    planned_hand_target,
    settlement_ready_click_is_fresh,
)


def _context() -> dict:
    return {
        "flow_state": "play",
        "controlled_card_count": 21,
        "hand_signature": [["一", 1], ["玖", 1]],
        "button_names": [],
        "discard_button_visible": True,
        "pending_action_card": None,
        "opponent_pending_card": None,
        "option_stage": None,
        "decision_action": "discard",
    }


def test_fresh_context_ignores_internal_decision_and_auxiliary_deck_count():
    expected = _context()
    actual = {
        **expected,
        "decision_action": None,
        "remaining_deck_count": 42,
    }

    assert changed_fresh_context(expected, actual) == []


def test_fresh_context_rejects_hand_button_pending_and_option_changes():
    for key, value in (
        ("hand_signature", [["一", 1]]),
        ("button_names", ["hu"]),
    ):
        actual = {**_context(), key: value}

        assert changed_fresh_context(_context(), actual) == [key]


def test_own_discard_phase_ignores_stale_pending_regions():
    actual = {
        **_context(),
        "pending_action_card": "五",
        "opponent_pending_card": "伍",
    }

    assert changed_fresh_context(_context(), actual) == []


def test_response_phase_still_requires_pending_card_and_option_stage():
    expected = {
        **_context(),
        "button_names": ["chi", "pass"],
        "discard_button_visible": False,
        "opponent_pending_card": "四",
    }
    for key, value in (
        ("opponent_pending_card", "五"),
        ("option_stage", "chi"),
    ):
        actual = {**expected, key: value}

        assert changed_fresh_context(expected, actual) == [key]


def test_opponent_response_uses_opponent_card_when_center_region_is_noisy():
    expected = {
        **_context(),
        "button_names": ["chi", "pass"],
        "discard_button_visible": False,
        "pending_action_card": "八",
        "opponent_pending_card": "八",
        "option_stage": "chi",
    }
    actual = {**expected, "pending_action_card": "五"}

    assert changed_fresh_context_for_action(expected, actual, "pass") == []


def test_settlement_ready_freshness_uses_visible_button_not_play_fields():
    flow = {
        "state": "settlement_ready",
        "x": 1795,
        "y": 954,
        "w": 320,
        "h": 101,
    }

    assert settlement_ready_click_is_fresh(
        [["settlement_ready", 1901, 1004]],
        flow,
    )


def test_settlement_ready_freshness_rejects_missing_or_moved_button():
    click = [["settlement_ready", 1901, 1004]]

    assert not settlement_ready_click_is_fresh(click, {"state": "already_ready"})
    assert not settlement_ready_click_is_fresh(
        click,
        {"state": "settlement_ready", "x": 900, "y": 500, "w": 220, "h": 80},
    )


def test_pass_after_evaluated_options_tolerates_only_option_layer_dropout():
    expected = {
        **_context(),
        "button_names": ["chi", "pass"],
        "discard_button_visible": False,
        "option_stage": "chi",
    }
    actual = {**expected, "option_stage": None}

    assert changed_fresh_context_for_action(expected, actual, "pass") == []
    assert changed_fresh_context_for_action(expected, actual, "chi_option") == ["option_stage"]
    assert changed_fresh_context_for_action(
        expected,
        {**actual, "opponent_pending_card": "五"},
        "pass",
    ) == ["opponent_pending_card"]


def test_discard_freshness_ignores_occluded_locked_hand_variation():
    expected = _context()
    actual = {
        **expected,
        "controlled_card_count": 20,
        "hand_signature": [["一", 1]],
    }

    assert changed_fresh_context_for_action(expected, actual, "discard") == []


def test_discard_freshness_still_rejects_turn_or_response_changes():
    expected = _context()

    assert changed_fresh_context_for_action(
        expected,
        {**expected, "discard_button_visible": False},
        "discard",
    ) == ["discard_button_visible"]
    assert changed_fresh_context_for_action(
        expected,
        {**expected, "button_names": ["hu", "pass"]},
        "discard",
    ) == ["button_names"]


def test_settlement_ready_guard_signature_ignores_button_box_jitter():
    first = ("settlement_ready", (("settlement_ready", 1901, 1004),))
    second = ("settlement_ready", (("settlement_ready", 1955, 1001),))

    assert mobile_guard_signature(first) == mobile_guard_signature(second)
    assert mobile_guard_signature(("pass", (("button:pass", 2190, 522),))) == (
        "pass",
        (("button:pass", 2190, 522),),
    )


def test_planned_hand_target_uses_drag_source_not_drop_coordinate():
    plan = {
        "action": "discard",
        "clicks": [
            {
                "target": "hand:一",
                "from_x": 509,
                "from_y": 1004,
                "x": 509,
                "y": 320,
            }
        ],
    }

    assert planned_hand_target(plan) == {
        "target": "hand:一",
        "label": "一",
        "card_id": None,
        "x": 509,
        "y": 1004,
    }
