from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from vision.button_detector import detect_buttons
from vision.option_recognizer import recognize_options


REGIONS = (
    (0.70, 0.24, 0.995, 0.72),
    (0.12, 0.58, 0.96, 0.995),
    (0.18, 0.01, 0.995, 0.36),
    (0.32, 0.24, 0.68, 0.76),
)
THRESHOLDS = (0.045, 0.045, 0.035, 0.08)


def _fingerprint(image) -> list[list[tuple[int, int, int]]]:
    height, width = image.shape[:2]
    result: list[list[tuple[int, int, int]]] = []
    for left, top, right, bottom in REGIONS:
        x1, y1 = int(width * left), int(height * top)
        x2, y2 = int(width * right), int(height * bottom)
        samples = []
        for row in range(10):
            y = min(y2 - 1, int(y1 + (row + 0.5) * (y2 - y1) / 10.0))
            for column in range(20):
                x = min(x2 - 1, int(x1 + (column + 0.5) * (x2 - x1) / 20.0))
                blue, green, red = image[y, x]
                samples.append((int(red), int(green), int(blue)))
        result.append(samples)
    return result


def _changed_ratio(before, after) -> float:
    changed = sum(
        max(abs(left[channel] - right[channel]) for channel in range(3)) >= 32
        for left, right in zip(before, after)
    )
    return changed / len(before)


def _overlay_variant(image, *, color: tuple[int, int, int], text: str):
    changed = image.copy()
    scale_x = image.shape[1] / 2344.0
    scale_y = image.shape[0] / 1080.0
    x1, y1 = round(331 * scale_x), round(3 * scale_y)
    x2, y2 = round(681 * scale_x), round(89 * scale_y)
    cv2.rectangle(changed, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.putText(
        changed,
        text,
        (x1 + 8, y1 + max(20, (y2 - y1) // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.35, scale_y * 0.7),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return changed


def test_overlay_text_and_color_do_not_trigger_or_change_game_recognition() -> None:
    screenshot = Path("data/screenshots/screenshot_20260527_160227.png")
    if not screenshot.is_file():
        pytest.skip(f"missing historical candidate frame: {screenshot}")
    original = cv2.imread(str(screenshot))
    assert original is not None

    original_fingerprint = _fingerprint(original)
    original_buttons = [button.to_dict() for button in detect_buttons(original)]
    original_options = [option.to_dict() for option in recognize_options(original)]
    assert original_options

    for changed in (
        _overlay_variant(original, color=(30, 130, 30), text="AI RUNNING"),
        _overlay_variant(original, color=(0, 190, 230), text="AI WARNING"),
    ):
        changed_fingerprint = _fingerprint(changed)
        assert all(
            _changed_ratio(before, after) < threshold
            for before, after, threshold in zip(
                original_fingerprint,
                changed_fingerprint,
                THRESHOLDS,
            )
        )
        assert [button.to_dict() for button in detect_buttons(changed)] == original_buttons
        assert [option.to_dict() for option in recognize_options(changed)] == original_options
