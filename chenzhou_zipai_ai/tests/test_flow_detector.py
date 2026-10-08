"""Tests for high-level flow/screen detection."""

from pathlib import Path

import cv2
import numpy as np

from vision.flow_detector import detect_flow_state, detect_flow_state_from_path


def test_detects_settlement_ready_screen():
    screenshot = Path("data/screenshots/screenshot_20260527_141515.png")
    if not screenshot.exists():
        return

    result = detect_flow_state_from_path(screenshot)

    assert result.state == "settlement_ready"
    assert result.center is not None
    x, y = result.center
    assert 1850 <= x <= 2150
    assert 850 <= y <= 1060


def test_detects_center_room_ready_button():
    image = np.zeros((1080, 2344, 3), dtype=np.uint8)
    image[:, :] = (60, 55, 45)
    cv2.rectangle(image, (1020, 600), (1320, 690), (0, 170, 255), -1)

    result = detect_flow_state(image)

    assert result.state == "settlement_ready"
    assert result.center is not None
    x, y = result.center
    assert 1080 <= x <= 1260
    assert 620 <= y <= 680


def test_center_discard_button_with_hand_cards_is_play():
    image = np.zeros((1080, 2344, 3), dtype=np.uint8)
    image[:, :] = (60, 55, 45)
    cv2.rectangle(image, (1020, 600), (1320, 690), (0, 170, 255), -1)
    for x in (620, 770, 920, 1070):
        cv2.rectangle(image, (x, 800), (x + 140, 1040), (245, 245, 238), -1)
        cv2.rectangle(image, (x + 35, 850), (x + 95, 920), (20, 20, 20), -1)

    result = detect_flow_state(image)

    assert result.state == "play"


def test_real_dealer_discard_screen_is_not_center_ready():
    screenshot = Path("data/screenshots/screenshot_20260617_182456_890264_549900.png")
    if not screenshot.exists():
        return

    result = detect_flow_state_from_path(screenshot)

    assert result.state == "play"


def test_detects_final_score_screen():
    screenshot = Path("data/screenshots/screenshot_20260527_141556.png")
    if not screenshot.exists():
        return

    result = detect_flow_state_from_path(screenshot)

    assert result.state == "final_score"


def test_play_screen_is_not_settlement():
    screenshot = Path("data/screenshots/screenshot_20260527_142633.png")
    if not screenshot.exists():
        return

    result = detect_flow_state_from_path(screenshot)

    assert result.state == "play"


def test_detects_already_ready_waiting_state():
    screenshot = Path("data/screenshots/screenshot_20260527_164335.png")
    if not screenshot.exists():
        return

    result = detect_flow_state_from_path(screenshot)

    assert result.state == "already_ready"


def test_visible_ready_button_wins_over_opponent_already_ready_marker():
    image = np.zeros((1080, 2344, 3), dtype=np.uint8)
    image[:, :] = (60, 55, 45)
    cv2.rectangle(image, (330, 720), (370, 760), (0, 190, 220), -1)
    cv2.rectangle(image, (375, 780), (415, 820), (0, 190, 220), -1)
    cv2.rectangle(image, (1900, 900), (2180, 990), (0, 170, 255), -1)

    result = detect_flow_state(image)

    assert result.state == "settlement_ready"
    assert result.center is not None
    x, y = result.center
    assert 1900 <= x <= 2180
    assert 900 <= y <= 990


def test_own_turn_pointer_is_not_settlement_ready():
    screenshot = Path("data/screenshots/screenshot_20260527_165901.png")
    if not screenshot.exists():
        return

    result = detect_flow_state_from_path(screenshot)

    assert result.state == "play"
