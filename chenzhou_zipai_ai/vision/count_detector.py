"""Detect numeric counters on the game screen."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision.regions import crop_region, load_regions_for_size

DIGIT_TEMPLATES = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def _template_image(pattern: tuple[str, ...]) -> np.ndarray:
    image = np.array([[255 if char == "1" else 0 for char in row] for row in pattern], dtype=np.uint8)
    return cv2.resize(image, (24, 40), interpolation=cv2.INTER_NEAREST)


def _digit_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.inRange(gray, 205, 255)


def _digit_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 8 <= w <= 45 and 25 <= h <= 70 and area >= 80:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda item: item[0])
    return boxes


def _classify_digit(mask: np.ndarray, box: tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    if w <= 14 and h >= 25:
        return "1"
    digit = mask[y : y + h, x : x + w]
    resized = cv2.resize(digit, (24, 40), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 80, 255, cv2.THRESH_BINARY)
    best_digit = ""
    best_score = -1.0
    for value, pattern in DIGIT_TEMPLATES.items():
        template = _template_image(pattern)
        score = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED).max()
        if score > best_score:
            best_digit = value
            best_score = float(score)
    return best_digit


def detect_remaining_deck_count(
    image: np.ndarray,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
) -> int | None:
    height, width = image.shape[:2]
    region = next(
        (item for item in load_regions_for_size(config_path, width, height) if item.name == "remaining_deck_count"),
        None,
    )
    if region is None:
        return None
    crop = crop_region(image, region)
    mask = _digit_mask(crop)
    boxes = _digit_boxes(mask)
    if not boxes:
        return None
    digits = "".join(_classify_digit(mask, box) for box in boxes)
    return int(digits) if digits else None
