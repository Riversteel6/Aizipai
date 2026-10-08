"""Pure action-specific decisions for Android preflight and result verification."""

from __future__ import annotations


RESPONSE_BUTTONS = {"chi", "peng", "hu", "pass", "pao"}


def required_pre_surface_features(action: str, *, expected: dict | None = None) -> frozenset[str]:
    """Return only the visual fields read by ``evaluate_pre_action``."""

    action = str(action).lower()
    del expected
    if action in {"settlement_ready", "compact_hand"}:
        return frozenset({"flow"})
    if action == "hu":
        return frozenset({"flow", "buttons"})
    if action == "discard":
        return frozenset({"flow", "buttons", "discard_button", "hand_target"})
    if action in {"chi_option", "compare_option"}:
        return frozenset({"flow", "buttons", "options"})
    if action in {"pass", "peng", "expand_chi_options", "chi"}:
        return frozenset({"flow", "buttons"})
    return frozenset({"full_state"})


def required_surface_features(action: str, *, expected: dict | None = None) -> frozenset[str]:
    """Return the smallest visual surface needed to verify one semantic action."""

    action = str(action).lower()
    expected = expected or {}
    if action == "settlement_ready":
        return frozenset({"flow"})
    if action == "compact_hand":
        return frozenset({"flow"})
    if action == "hu":
        return frozenset({"flow", "buttons"})
    if action == "discard":
        return frozenset({"flow", "buttons", "discard_button", "hand_target"})
    if action in {
        "pass",
        "peng",
        "expand_chi_options",
        "chi",
        "chi_option",
        "compare_option",
    }:
        features = {"flow", "buttons", "discard_button", "options"}
        if action == "pass":
            features.add("opponent_pending_card")
        return frozenset(features)
    return frozenset({"full_state"})


def evaluate_pre_action(
    *,
    action: str,
    expected: dict,
    actual: dict,
    target: dict | None = None,
) -> dict:
    action = str(action).lower()
    buttons = {str(name).lower() for name in actual.get("button_names") or []}
    if action != "hu" and "hu" in buttons:
        return {"status": "preempt_hu", "reason": "visible_hu_button"}
    if actual.get("flow_state") != expected.get("flow_state"):
        return {"status": "stale", "reason": "flow_state_changed"}

    if action == "discard":
        if not actual.get("discard_button_visible") or buttons & RESPONSE_BUTTONS:
            return {"status": "stale", "reason": "discard_turn_changed"}
        target_status = str((target or {}).get("status") or "missing")
        if target_status in {"matched", "relocated"}:
            return {
                "status": "relocated" if target_status == "relocated" else "execute",
                "reason": target_status,
                "target": target,
            }
        if target_status == "uncertain":
            return {"status": "uncertain", "reason": "target_recognition_uncertain"}
        return {"status": "stale", "reason": "target_card_missing"}

    required_button = {
        "expand_chi_options": "chi",
        "chi": "chi",
        "peng": "peng",
        "pass": "pass",
        "hu": "hu",
    }.get(action)
    if required_button is not None and required_button not in buttons:
        return {"status": "stale", "reason": f"button_missing:{required_button}"}
    required_stage = {
        "chi_option": "chi",
        "compare_option": "compare",
    }.get(action)
    if required_stage is not None and actual.get("option_stage") != required_stage:
        return {"status": "stale", "reason": f"option_stage_missing:{required_stage}"}
    return {"status": "execute", "reason": "action_surface_matches"}


def evaluate_post_action(
    *,
    action: str,
    expected: dict,
    actual: dict,
) -> dict:
    action = str(action).lower()
    buttons = {str(name).lower() for name in actual.get("button_names") or []}
    expected_buttons = {str(name).lower() for name in expected.get("button_names") or []}
    flow_changed = actual.get("flow_state") != expected.get("flow_state")
    strong_transition = False

    if action == "discard":
        confirmed = (
            flow_changed
            or not actual.get("discard_button_visible")
            or bool(buttons & RESPONSE_BUTTONS)
        )
    elif action in {"expand_chi_options", "chi"}:
        candidate_surface_visible = actual.get("option_stage") == "chi"
        if action == "expand_chi_options":
            confirmed = candidate_surface_visible
        else:
            confirmed = (
                flow_changed
                or candidate_surface_visible
                or actual.get("discard_button_visible")
                or (
                    actual.get("option_stage") is None
                    and not bool(buttons & RESPONSE_BUTTONS)
                )
            )
        strong_transition = candidate_surface_visible
    elif action == "chi_option":
        compare_surface_visible = actual.get("option_stage") == "compare"
        confirmed = (
            flow_changed
            or compare_surface_visible
            or actual.get("discard_button_visible")
            or (
                actual.get("option_stage") is None
                and not bool(buttons & RESPONSE_BUTTONS)
            )
        )
        strong_transition = compare_surface_visible
    elif action in {"compare_option", "peng"}:
        confirmed = (
            flow_changed
            or actual.get("discard_button_visible")
            or (
                actual.get("option_stage") is None
                and not bool(buttons & RESPONSE_BUTTONS)
            )
        )
    elif action == "pass":
        confirmed = (
            flow_changed
            or not bool(buttons & RESPONSE_BUTTONS)
            or buttons != expected_buttons
            or actual.get("opponent_pending_card") != expected.get("opponent_pending_card")
        )
    elif action == "hu":
        confirmed = flow_changed or "hu" not in buttons
    elif action == "settlement_ready":
        confirmed = actual.get("flow_state") != "settlement_ready"
    else:
        return {"status": "uncertain", "reason": f"unsupported_action:{action}"}

    return {
        "status": "confirmed_signal" if confirmed else "pending",
        "reason": "expected_transition_visible" if confirmed else "waiting_expected_transition",
        "strong_transition": bool(strong_transition),
    }


def evaluate_discard_selection(
    *,
    expected: dict,
    actual: dict,
    target: dict | None,
) -> dict:
    """Verify the semantic discard target after the first tap, before confirmation."""

    buttons = {str(name).lower() for name in actual.get("button_names") or []}
    if "hu" in buttons:
        return {"status": "preempt_hu", "reason": "visible_hu_button"}
    if actual.get("flow_state") != expected.get("flow_state"):
        return {"status": "stale", "reason": "flow_state_changed"}
    if not actual.get("discard_button_visible") or buttons & RESPONSE_BUTTONS:
        return {"status": "stale", "reason": "discard_turn_changed"}

    target_status = str((target or {}).get("status") or "missing")
    if target_status in {"matched", "relocated"}:
        if bool((target or {}).get("selected")):
            return {"status": "selected", "reason": "semantic_target_selected"}
        return {"status": "pending", "reason": "waiting_target_selected_state"}
    if target_status == "uncertain":
        return {"status": "uncertain", "reason": "selected_target_recognition_uncertain"}
    return {"status": "stale", "reason": "selected_target_missing"}


__all__ = [
    "evaluate_discard_selection",
    "evaluate_post_action",
    "evaluate_pre_action",
    "required_pre_surface_features",
    "required_surface_features",
]
