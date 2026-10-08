"""Pure construction of mobile hard-priority plans."""

from __future__ import annotations

import json


def visible_hu_plan(buttons: list[dict]) -> dict | None:
    hu = next(
        (
            button
            for button in buttons
            if str(button.get("name") or "").lower() == "hu"
        ),
        None,
    )
    if hu is None:
        return None
    center = hu.get("center")
    if center is None:
        center = (
            int(hu["x"]) + int(hu["w"]) // 2,
            int(hu["y"]) + int(hu["h"]) // 2,
        )
    plan = {
        "action": "hu",
        "ready": True,
        "execution_mode": "tap_sequence",
        "reason": "visible_hu_mobile_priority",
        "validation": {
            "passed": True,
            "checks": ["visible_hu_button", "target_clickable"],
        },
        "clicks": [
            {
                "target": "button:hu",
                "x": int(center[0]),
                "y": int(center[1]),
                "delay_ms": 0,
            }
        ],
    }
    plan["_mobile_signature"] = json.dumps(
        {"priority": "hu"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return plan
