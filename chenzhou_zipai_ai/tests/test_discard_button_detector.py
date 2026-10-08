"""Tests for central discard button detection."""

from pathlib import Path

import cv2

from tools.inspect_state import inspect_screenshot
from vision.discard_button_detector import detect_discard_button


def test_detects_discard_button_on_own_turn():
    screenshot = Path("data/screenshots/screenshot_20260527_142258.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    result = detect_discard_button(image)

    assert result is not None
    assert 1000 <= result.center[0] <= 1300
    assert 430 <= result.center[1] <= 560


def test_discard_button_switches_state_to_discard_actions():
    screenshot = Path("data/screenshots/screenshot_20260527_142258.png")
    if not screenshot.exists():
        return

    result = inspect_screenshot(screenshot, expected_total=21)

    assert result["discard_button"] is not None
    assert "discard" in {item["type"] for item in result["legal_actions"]}


def test_countdown_panel_is_not_discard_button_after_discard():
    screenshot = Path("data/screenshots/screenshot_20260527_164100.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    result = detect_discard_button(image)

    assert result is None


def test_visible_action_button_prevents_discard_phase():
    screenshot = Path("data/screenshots/screenshot_20260527_154343.png")
    if not screenshot.exists():
        return

    result = inspect_screenshot(screenshot, expected_total=20)

    assert [button["name"] for button in result["buttons"]] == ["pass", "peng"]
    assert "discard" not in {item["type"] for item in result["legal_actions"]}
