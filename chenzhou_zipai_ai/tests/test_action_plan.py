"""Tests for non-executing action plans."""

from control.action_plan import action_plan_contract_error, build_action_plan


def test_builds_button_click_plan_when_button_visible():
    state = {"buttons": [{"name": "hu", "x": 100, "y": 200, "w": 80, "h": 40}]}
    plan = build_action_plan(state, {"action": "hu"})

    assert plan["ready"]
    assert plan["clicks"] == [{"target": "button:hu", "x": 140, "y": 220, "delay_ms": 80}]


def test_does_not_plan_click_when_button_is_missing():
    plan = build_action_plan({"buttons": []}, {"action": "hu"})

    assert not plan["ready"]
    assert plan["clicks"] == []


def test_hu_plan_rejects_protocol_fallback_when_hu_button_not_visible_by_default():
    state = {"buttons": [], "legal_actions": [{"type": "HU"}, {"type": "PASS"}]}

    plan = build_action_plan(state, {"action": "hu"})

    assert not plan["ready"]
    assert plan["clicks"] == []
    assert plan["validation"]["reason_code"] == "action_plan_policy_mismatch"


def test_action_plan_contract_rejects_unready_play_action_marked_valid():
    plan = {
        "action": "hu",
        "ready": False,
        "clicks": [],
        "validation": {"passed": True},
    }

    assert action_plan_contract_error(plan) == "unready_play_action_marked_valid"


def test_action_plan_contract_rejects_ready_plan_without_action_type():
    plan = {
        "ready": True,
        "clicks": [{"target": "button:pass", "x": 10, "y": 20}],
        "validation": {"passed": True},
    }

    assert action_plan_contract_error(plan) == "action_plan_missing_action"


def test_action_plan_contract_rejects_negative_tap_coordinates():
    plan = {
        "action": "pass",
        "ready": True,
        "clicks": [{"target": "button:pass", "x": -1, "y": 20}],
        "policy_selected_action": {"type": "pass", "label": None},
        "validation": {"passed": True, "target_matches_policy": True},
    }

    assert action_plan_contract_error(plan) == "action_plan_click_x_out_of_bounds"


def test_action_plan_contract_rejects_drag_without_start_coordinates():
    plan = {
        "action": "compact_hand",
        "ready": True,
        "execution_mode": "drag_sequence",
        "clicks": [{"target": "hand:compact", "x": 100, "y": 200}],
        "validation": {"passed": True},
    }

    assert action_plan_contract_error(plan) == "action_plan_click_from_x_invalid"


def test_hu_plan_uses_protocol_fallback_only_when_explicitly_trusted():
    state = {"buttons": [], "legal_actions": [{"type": "HU"}, {"type": "CHI"}, {"type": "PASS"}]}
    state["metadata"] = {"allow_protocol_fallback_buttons": True}

    plan = build_action_plan(state, {"action": "hu"})

    assert plan["ready"]
    assert plan["clicks"] == [{"target": "button:hu", "x": 1745, "y": 510, "delay_ms": 80}]


def test_auto_meld_actions_wait_instead_of_clicking_visible_buttons():
    state = {
        "buttons": [
            {"name": "pao", "x": 100, "y": 200, "w": 80, "h": 40},
            {"name": "ti", "x": 200, "y": 200, "w": 80, "h": 40},
        ]
    }

    plan = build_action_plan(state, {"action": "pao", "label": "二"})

    assert plan["action"] == "wait_auto_meld"
    assert not plan["ready"]
    assert plan["clicks"] == []
    assert plan["target_label"] == "二"
    assert plan["validation"]["reason_code"] == "wait_auto_meld"


def test_chi_plan_uses_policy_selected_option_id():
    state = {
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "pending_card": "七",
        "option_details": [
            {
                "region_name": "chi_options",
                "option_id": "chi_123",
                "labels": ["一", "二", "三"],
                "x": 300,
                "y": 40,
                "w": 90,
                "h": 180,
            },
            {
                "region_name": "chi_options",
                "option_id": "chi_2710",
                "labels": ["二", "七", "十"],
                "center": (520, 130),
                "x": 480,
                "y": 40,
                "w": 90,
                "h": 180,
            },
        ],
    }

    plan = build_action_plan(
        state,
        {
            "action": "chi",
            "selected_option_id": "chi_2710",
            "selected_option_cards": ["二", "七", "十"],
        },
    )

    assert plan["ready"]
    assert plan["action"] == "chi_option"
    assert plan["target_type"] == "option_column"
    assert plan["target_option_id"] == "chi_2710"
    assert plan["target_option_cards"] == ["二", "七", "十"]
    assert plan["clicks"] == [
        {"target": "chi:二七十", "x": 520, "y": 130, "delay_ms": 80},
    ]


def test_chi_plan_rejects_illegal_selected_option_cards():
    state = {
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "pending_card": "七",
        "option_details": [
            {
                "region_name": "chi_options",
                "option_id": "chi_bad",
                "labels": ["六", "八", "捌"],
                "center": (520, 130),
            },
        ],
    }

    plan = build_action_plan(
        state,
        {
            "action": "chi",
            "selected_option_id": "chi_bad",
            "selected_option_cards": ["六", "八", "捌"],
        },
    )

    assert not plan["ready"]
    assert plan["clicks"] == []
    assert plan["validation"]["reason_code"] == "invalid_chi_option"


def test_chi_plan_repairs_single_size_mismatch_from_hand_context():
    state = {
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "hand": ["四", "五", "六"],
        "hand_details": [
            {"name": "四", "card_id": "h001", "x": 10, "y": 20, "w": 80, "h": 100},
            {"name": "五", "card_id": "h002", "x": 100, "y": 20, "w": 80, "h": 100},
            {"name": "六", "card_id": "h003", "x": 190, "y": 20, "w": 80, "h": 100},
        ],
        "option_details": [
            {
                "region_name": "chi_options",
                "option_id": "chi_001",
                "labels": ["伍", "六", "七"],
                "center": (520, 130),
            },
        ],
    }

    plan = build_action_plan(
        state,
        {
            "action": "chi",
            "selected_option_id": "chi_001",
            "selected_option_cards": ["伍", "六", "七"],
        },
    )

    assert plan["ready"]
    assert plan["target_label"] == "五六七"
    assert plan["target_option_cards"] == ["五", "六", "七"]
    assert plan["clicks"] == [
        {"target": "chi:五六七", "x": 520, "y": 130, "delay_ms": 80},
    ]


def test_chi_plan_repairs_same_rank_size_mismatch_from_hand_context():
    state = {
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "hand": ["五", "伍", "陆"],
        "pending_card": "五",
        "hand_details": [
            {"name": "五", "card_id": "h001", "x": 10, "y": 20, "w": 80, "h": 100},
            {"name": "伍", "card_id": "h002", "x": 100, "y": 20, "w": 80, "h": 100},
        ],
        "option_details": [
            {
                "region_name": "chi_options",
                "option_id": "chi_001",
                "labels": ["伍", "伍", "伍"],
                "center": (520, 130),
            },
        ],
    }

    plan = build_action_plan(
        state,
        {
            "action": "chi",
            "selected_option_id": "chi_001",
            "selected_option_cards": ["伍", "伍", "伍"],
        },
    )

    assert plan["ready"]
    assert plan["target_label"] == "五五伍"
    assert plan["target_option_cards"] == ["五", "五", "伍"]
    assert plan["clicks"] == [
        {"target": "chi:五五伍", "x": 520, "y": 130, "delay_ms": 80},
    ]


def test_chi_plan_repairs_pending_same_rank_misread_from_hand_context():
    state = {
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "hand": ["十", "拾", "三"],
        "pending_card": "拾",
        "hand_details": [
            {"name": "十", "card_id": "h001", "x": 10, "y": 20, "w": 80, "h": 100},
            {"name": "拾", "card_id": "h002", "x": 100, "y": 20, "w": 80, "h": 100},
        ],
        "option_details": [
            {
                "region_name": "chi_options",
                "option_id": "chi_001",
                "labels": ["拾", "拾", "七"],
                "center": (520, 130),
            },
        ],
    }

    plan = build_action_plan(
        state,
        {
            "action": "chi",
            "selected_option_id": "chi_001",
            "selected_option_cards": ["拾", "拾", "七"],
        },
    )

    assert plan["ready"]
    assert plan["target_label"] == "拾拾十"
    assert plan["target_option_cards"] == ["拾", "拾", "十"]
    assert plan["clicks"] == [
        {"target": "chi:拾拾十", "x": 520, "y": 130, "delay_ms": 80},
    ]


def test_chi_plan_repairs_pending_sequence_misread_from_hand_context():
    state = {
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "hand": ["二", "三", "七"],
        "pending_card": "四",
        "hand_details": [
            {"name": "二", "card_id": "h001", "x": 10, "y": 20, "w": 80, "h": 100},
            {"name": "三", "card_id": "h002", "x": 100, "y": 20, "w": 80, "h": 100},
        ],
        "option_details": [
            {
                "region_name": "chi_options",
                "option_id": "chi_001",
                "labels": ["二", "八", "伍"],
                "center": (520, 130),
            },
        ],
    }

    plan = build_action_plan(
        state,
        {
            "action": "chi",
            "selected_option_id": "chi_001",
            "selected_option_cards": ["二", "八", "伍"],
        },
    )

    assert plan["ready"]
    assert plan["target_label"] == "二三四"
    assert plan["target_option_cards"] == ["二", "三", "四"]
    assert plan["clicks"] == [
        {"target": "chi:二三四", "x": 520, "y": 130, "delay_ms": 80},
    ]


def test_chi_plan_rejects_two_card_option_without_pending_card():
    state = {
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "option_details": [
            {
                "region_name": "chi_options",
                "option_id": "chi_two",
                "labels": ["二", "十"],
                "center": (520, 130),
            },
        ],
    }

    plan = build_action_plan(
        state,
        {
            "action": "chi",
            "selected_option_id": "chi_two",
            "selected_option_cards": ["二", "十"],
        },
    )

    assert not plan["ready"]
    assert plan["clicks"] == []
    assert plan["validation"]["reason_code"] == "chi_option_pending_card_missing"


def test_chi_plan_does_not_reselect_when_policy_option_missing():
    state = {
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "option_details": [
            {
                "region_name": "chi_options",
                "option_id": "chi_123",
                "labels": ["一", "二", "三"],
                "center": (320, 130),
            },
        ],
    }

    plan = build_action_plan(state, {"action": "chi", "selected_option_id": "chi_2710"})

    assert not plan["ready"]
    assert plan["clicks"] == []
    assert plan["reason"] == "selected_option_not_clickable"


def test_chi_plan_does_not_expand_options_as_real_chi_action():
    state = {"buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}]}

    plan = build_action_plan(state, {"action": "chi"})

    assert not plan["ready"]
    assert plan["clicks"] == []
    assert plan["validation"]["reason_code"] == "chi_options_not_visible"


def test_semantic_chi_intent_expands_only_after_policy_has_selected_chi():
    state = {
        "hand": ["七", "九", "一"],
        "pending_action_card": {"name": "八"},
        "buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}],
        "chi_options": [
            {
                "option_id": "semantic_chi_001",
                "labels": ["七", "八", "九"],
                "semantic_only": True,
            }
        ],
    }

    plan = build_action_plan(
        state,
        {
            "action": "chi",
            "selected_option_id": "semantic_chi_001",
            "selected_option_cards": ["七", "八", "九"],
        },
    )

    assert plan["ready"]
    assert plan["action"] == "chi"
    assert plan["target_option_cards"] == ["七", "八", "九"]
    assert plan["clicks"] == [{"target": "button:chi", "x": 140, "y": 220, "delay_ms": 80}]


def test_expand_chi_options_clicks_button_only_before_options_appear():
    state = {"buttons": [{"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40}]}

    plan = build_action_plan(state, {"action": "expand_chi_options"})

    assert plan["ready"]
    assert plan["action"] == "expand_chi_options"
    assert plan["target_type"] == "button"
    assert plan["clicks"] == [{"target": "button:chi", "x": 140, "y": 220, "delay_ms": 80}]


def test_compare_option_plan_clicks_policy_selected_compare_candidate_only():
    state = {
        "option_details": [
            {
                "region_name": "compare_options",
                "option_id": "compare_123",
                "labels": ["一", "二", "三"],
                "center": (360, 120),
            },
            {
                "region_name": "compare_options",
                "option_id": "compare_234",
                "labels": ["二", "三", "四"],
                "x": 480,
                "y": 40,
                "w": 90,
                "h": 180,
            },
        ]
    }

    plan = build_action_plan(
        state,
        {
            "action": "compare_option",
            "selected_option_id": "compare_234",
            "selected_option_cards": ["二", "三", "四"],
        },
    )

    assert plan["ready"]
    assert plan["action"] == "compare_option"
    assert plan["target_option_id"] == "compare_234"
    assert plan["target_option_cards"] == ["二", "三", "四"]
    assert plan["clicks"] == [{"target": "compare:二三四", "x": 525, "y": 130, "delay_ms": 80}]


def test_compare_option_plan_does_not_reselect_missing_candidate():
    state = {
        "option_details": [
            {
                "region_name": "compare_options",
                "option_id": "compare_123",
                "labels": ["一", "二", "三"],
                "center": (360, 120),
            },
        ]
    }

    plan = build_action_plan(state, {"action": "compare_option", "selected_option_id": "compare_234"})

    assert not plan["ready"]
    assert plan["clicks"] == []
    assert plan["reason"] == "selected_option_not_clickable"


def test_builds_discard_click_plan_from_hand_card():
    state = {"hand_details": [{"name": "九", "x": 10, "y": 20, "w": 100, "h": 120}]}
    plan = build_action_plan(state, {"action": "discard", "label": "九"})

    assert plan["ready"]
    assert plan["clicks"][0]["target"] == "hand:九"


def test_discard_button_uses_verified_select_then_confirm_sequence():
    state = {
        "hand_details": [{"name": "九", "x": 10, "y": 20, "w": 100, "h": 120}],
        "discard_button": {"x": 900, "y": 360, "w": 500, "h": 180},
    }

    plan = build_action_plan(state, {"action": "discard", "label": "九"})

    assert plan["ready"]
    assert plan["execution_mode"] == "select_then_discard_button"
    assert [click["target"] for click in plan["clicks"]] == ["hand:九", "button:discard"]


def test_blocks_discard_without_button_or_protocol_turn_signal():
    state = {
        "hand_details": [{"name": "九", "x": 10, "y": 20, "w": 100, "h": 120}],
        "legal_actions": [],
        "discard_button": None,
    }

    plan = build_action_plan(state, {"action": "discard", "label": "九"})

    assert not plan["ready"]
    assert plan["clicks"] == []
    assert plan["reason"] == "discard_not_actionable_without_turn_signal"


def test_allows_discard_drag_when_protocol_marks_own_draw():
    state = {
        "hand_details": [{"name": "九", "x": 10, "y": 20, "w": 100, "h": 120}],
        "legal_actions": [{"type": "discard", "source": "qs_own_draw"}],
        "discard_button": None,
    }

    plan = build_action_plan(state, {"action": "discard", "label": "九"})

    assert plan["ready"]
    assert plan["execution_mode"] == "discard_drag"
    assert plan["clicks"] == [{"target": "hand:九", "x": 60, "y": 80, "delay_ms": 80}]


def test_skips_hand_card_when_not_clickable():
    state = {
        "hand_details": [{"name": "叁", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": False}]
    }
    plan = build_action_plan(state, {"action": "discard", "label": "叁"})

    assert not plan["ready"]
    assert plan["reason"] == "selected_card_not_clickable"


def test_does_not_replace_policy_target_when_policy_label_not_clickable():
    state = {
        "hand_details": [
            {"name": "四", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": False},
            {"name": "叁", "x": 120, "y": 20, "w": 100, "h": 120, "clickable": True},
        ]
    }
    plan = build_action_plan(
        state,
        {
            "action": "discard",
            "label": "四",
            "candidate_stage": "normal_unprotected",
            "hard_protected": ["叁"],
            "break_hard_protection": False,
        },
    )

    assert not plan["ready"]
    assert plan["reason"] == "selected_card_not_clickable"


def test_blocks_hard_protected_discard_without_forced_break():
    state = {
        "hand_details": [
            {"name": "叁", "x": 10, "y": 20, "w": 100, "h": 120, "clickable": True},
            {"name": "四", "x": 120, "y": 20, "w": 100, "h": 120, "clickable": True},
        ]
    }
    plan = build_action_plan(
        state,
        {
            "action": "discard",
            "label": "叁",
            "candidate_stage": "normal_unprotected",
            "hard_protected": ["叁", "王"],
            "break_hard_protection": False,
        },
    )

    assert not plan["ready"]
    assert plan["reason"] == "blocked_discard_hard_protected"
