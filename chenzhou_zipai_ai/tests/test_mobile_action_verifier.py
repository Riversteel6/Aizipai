from tools.mobile_action_verifier import (
    evaluate_discard_selection,
    evaluate_post_action,
    evaluate_pre_action,
    required_pre_surface_features,
    required_surface_features,
)


def _surface(**changes):
    surface = {
        "flow_state": "play",
        "button_names": [],
        "discard_button_visible": True,
        "opponent_pending_card": None,
        "option_stage": None,
    }
    surface.update(changes)
    return surface


def test_discard_preflight_accepts_relocated_semantic_target():
    result = evaluate_pre_action(
        action="discard",
        expected=_surface(),
        actual=_surface(),
        target={"status": "relocated", "label": "一", "x": 800, "y": 870},
    )

    assert result["status"] == "relocated"


def test_discard_preflight_rejects_missing_target_and_preempts_hu():
    missing = evaluate_pre_action(
        action="discard",
        expected=_surface(),
        actual=_surface(),
        target={"status": "missing"},
    )
    hu = evaluate_pre_action(
        action="discard",
        expected=_surface(),
        actual=_surface(button_names=["hu", "pass"], discard_button_visible=False),
        target={"status": "matched"},
    )

    assert missing["status"] == "stale"
    assert hu["status"] == "preempt_hu"


def test_target_movement_alone_never_confirms_discard_result():
    result = evaluate_post_action(
        action="discard",
        expected=_surface(),
        actual=_surface(),
    )

    assert result["status"] == "pending"


def test_discard_result_requires_expected_turn_surface_transition():
    result = evaluate_post_action(
        action="discard",
        expected=_surface(),
        actual=_surface(discard_button_visible=False),
    )

    assert result["status"] == "confirmed_signal"


def test_discard_confirmation_requires_selected_semantic_target():
    selected = evaluate_discard_selection(
        expected=_surface(),
        actual=_surface(),
        target={"status": "matched", "label": "九", "selected": True},
    )
    not_selected = evaluate_discard_selection(
        expected=_surface(),
        actual=_surface(),
        target={"status": "matched", "label": "九", "selected": False},
    )

    assert selected["status"] == "selected"
    assert not_selected["status"] == "pending"


def test_discard_selection_never_overrides_visible_hu():
    result = evaluate_discard_selection(
        expected=_surface(),
        actual=_surface(button_names=["hu", "pass"], discard_button_visible=False),
        target={"status": "matched", "selected": True},
    )

    assert result["status"] == "preempt_hu"


def test_pass_result_requires_response_surface_to_change():
    expected = _surface(
        button_names=["chi", "pass"],
        discard_button_visible=False,
        opponent_pending_card="四",
    )

    assert evaluate_post_action(action="pass", expected=expected, actual=dict(expected))["status"] == "pending"
    assert evaluate_post_action(
        action="pass",
        expected=expected,
        actual=_surface(discard_button_visible=False),
    )["status"] == "confirmed_signal"


def test_chi_option_result_accepts_the_physically_required_compare_surface():
    expected = _surface(option_stage="chi", discard_button_visible=False)

    result = evaluate_post_action(
        action="chi_option",
        expected=expected,
        actual=_surface(option_stage="compare", discard_button_visible=False),
    )

    assert result["status"] == "confirmed_signal"
    assert result["strong_transition"] is True


def test_semantic_chi_button_result_accepts_visible_candidate_surface_as_strong_transition():
    expected = _surface(
        button_names=["chi", "pass"],
        discard_button_visible=False,
        option_stage=None,
    )

    result = evaluate_post_action(
        action="chi",
        expected=expected,
        actual=_surface(
            button_names=["chi", "pass"],
            discard_button_visible=False,
            option_stage="chi",
        ),
    )

    assert result["status"] == "confirmed_signal"
    assert result["strong_transition"] is True


def test_chi_option_result_does_not_accept_the_unchanged_chi_surface():
    expected = _surface(option_stage="chi", discard_button_visible=False)

    assert evaluate_post_action(
        action="chi_option",
        expected=expected,
        actual=dict(expected),
    )["status"] == "pending"


def test_known_mobile_actions_never_require_full_state_for_confirmation():
    actions = {
        "settlement_ready",
        "hu",
        "pass",
        "peng",
        "expand_chi_options",
        "chi",
        "chi_option",
        "compare_option",
        "compact_hand",
    }

    for action in actions:
        assert "full_state" not in required_surface_features(
            action,
            expected={"option_stage": "chi"},
        )

    assert required_surface_features("settlement_ready") == frozenset({"flow"})
    assert "opponent_pending_card" in required_surface_features("pass")


def test_preflight_surface_omits_features_not_read_by_pre_action_evaluator():
    assert required_pre_surface_features("pass") == frozenset({"flow", "buttons"})
    assert required_pre_surface_features("peng") == frozenset({"flow", "buttons"})
    assert required_pre_surface_features("expand_chi_options") == frozenset(
        {"flow", "buttons"}
    )
    assert required_pre_surface_features("chi") == frozenset({"flow", "buttons"})
    assert required_pre_surface_features("chi_option") == frozenset(
        {"flow", "buttons", "options"}
    )
    assert required_pre_surface_features("compare_option") == frozenset(
        {"flow", "buttons", "options"}
    )
    assert required_pre_surface_features("discard") == frozenset(
        {"flow", "buttons", "discard_button", "hand_target"}
    )
