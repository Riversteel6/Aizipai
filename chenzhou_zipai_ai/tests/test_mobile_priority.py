import json

from tools.mobile_priority import visible_hu_plan


def test_visible_hu_plan_bypasses_strategy_with_zero_delay():
    plan = visible_hu_plan(
        [
            {"name": "pass", "center": [1800, 510]},
            {"name": "hu", "center": [1500, 510]},
        ]
    )

    assert plan is not None
    assert plan["action"] == "hu"
    assert plan["clicks"] == [
        {"target": "button:hu", "x": 1500, "y": 510, "delay_ms": 0}
    ]
    assert json.loads(plan["_mobile_signature"]) == {"priority": "hu"}


def test_visible_hu_plan_does_not_guess_when_button_is_absent():
    assert visible_hu_plan([{"name": "pass", "center": [1800, 510]}]) is None
