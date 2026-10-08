"""Pure helpers for rejecting Android plans computed from stale frames."""

from __future__ import annotations


FRESH_CONTEXT_KEYS = (
    "flow_state",
    "controlled_card_count",
    "hand_signature",
    "button_names",
    "discard_button_visible",
    "pending_action_card",
    "opponent_pending_card",
    "option_stage",
)


def changed_fresh_context(expected: dict, actual: dict) -> list[str]:
    expected = normalized_fresh_context(expected)
    actual = normalized_fresh_context(actual)
    return [
        key
        for key in FRESH_CONTEXT_KEYS
        if expected.get(key) != actual.get(key)
    ]


def changed_fresh_context_for_action(expected: dict, actual: dict, action: str) -> list[str]:
    changed = changed_fresh_context(expected, actual)
    normalized_action = str(action).lower()
    if (
        normalized_action in {"pass", "peng", "chi", "chi_option", "compare_option"}
        and expected.get("opponent_pending_card")
        and expected.get("opponent_pending_card") == actual.get("opponent_pending_card")
    ):
        changed = [key for key in changed if key != "pending_action_card"]
    if normalized_action == "discard":
        # The hand was already fully recognized when the plan was created. A
        # second full-hand read is especially unstable around locked cards and
        # transient finger occlusion, so execution freshness is based on the
        # current turn surface and the separately checked target slot.
        changed = [
            key
            for key in changed
            if key not in {
                "controlled_card_count",
                "hand_signature",
                "pending_action_card",
                "opponent_pending_card",
                "option_stage",
            }
        ]
    if (
        normalized_action == "pass"
        and expected.get("option_stage") in {"chi", "compare"}
        and actual.get("option_stage") is None
    ):
        changed = [key for key in changed if key != "option_stage"]
    return changed


def mobile_guard_signature(signature: tuple) -> tuple:
    action = str(signature[0] or "") if signature else ""
    if action != "settlement_ready":
        return signature
    return ("settlement_ready", (("settlement_ready", 0, 0),))


def planned_hand_target(plan: dict) -> dict | None:
    """Return the source hand coordinate for tap or drag discard plans."""

    if str(plan.get("action") or "").lower() != "discard":
        return None
    clicks = plan.get("clicks") or []
    if not clicks or not isinstance(clicks[0], dict):
        return None
    click = clicks[0]
    try:
        x = int(click.get("from_x", click.get("x")))
        y = int(click.get("from_y", click.get("y")))
    except (TypeError, ValueError):
        return None
    target = str(click.get("target") or "")
    label = plan.get("target_label")
    if not label and target.startswith("hand:"):
        label = target.split(":", 1)[1]
    return {
        "target": target,
        "label": str(label or ""),
        "card_id": plan.get("target_card_id"),
        "x": x,
        "y": y,
    }


def normalized_fresh_context(context: dict) -> dict:
    normalized = dict(context)
    response_buttons = {
        str(name).lower() for name in normalized.get("button_names") or ()
    } & {"chi", "peng", "hu", "pass"}
    if normalized.get("discard_button_visible") and not response_buttons:
        normalized["pending_action_card"] = None
        normalized["opponent_pending_card"] = None
        normalized["option_stage"] = None
    return normalized


def settlement_ready_click_is_fresh(raw_clicks: object, flow: dict) -> bool:
    if flow.get("state") != "settlement_ready":
        return False
    try:
        x = int(flow["x"])
        y = int(flow["y"])
        width = int(flow["w"])
        height = int(flow["h"])
    except (KeyError, TypeError, ValueError):
        return False
    if width <= 0 or height <= 0 or not isinstance(raw_clicks, list):
        return False

    margin_x = max(24, width // 5)
    margin_y = max(18, height // 4)
    for item in raw_clicks:
        if not isinstance(item, list) or len(item) != 3 or item[0] != "settlement_ready":
            continue
        try:
            click_x = int(item[1])
            click_y = int(item[2])
        except (TypeError, ValueError):
            return False
        return (
            x - margin_x <= click_x <= x + width + margin_x
            and y - margin_y <= click_y <= y + height + margin_y
        )
    return False
