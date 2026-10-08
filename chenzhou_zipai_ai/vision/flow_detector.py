"""Detect coarse game-flow screens that are outside normal card play."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision.image_source import read_bgr


@dataclass(frozen=True)
class FlowDetection:
    state: str
    confidence: float
    x: int | None = None
    y: int | None = None
    w: int | None = None
    h: int | None = None

    @property
    def center(self) -> tuple[int, int] | None:
        if self.x is None or self.y is None or self.w is None or self.h is None:
            return None
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> dict[str, float | int | str | tuple[int, int] | None]:
        return {
            "state": self.state,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "center": self.center,
        }


def detect_flow_state(image: np.ndarray) -> FlowDetection:
    """Classify high-level non-play states.

    This intentionally uses simple color/geometry signals instead of OCR. The
    result only gates obvious settlement automation; card-play decisions still
    come from the normal vision pipeline.
    """

    final = _detect_final_score_screen(image)
    if final is not None:
        return final
    ready = _detect_orange_ready_button(image)
    if ready is not None:
        return ready
    already_ready = _detect_already_ready_state(image)
    if already_ready is not None:
        return already_ready
    ready = _detect_center_ready_button(image)
    if ready is not None:
        return ready
    return FlowDetection("play", 0.0)


def detect_flow_state_from_path(path: str | Path) -> FlowDetection:
    image = read_bgr(path)
    return detect_flow_state(image)


def _detect_center_ready_button(image: np.ndarray) -> FlowDetection | None:
    if _has_lower_hand_card_tiles(image):
        return None
    height, width = image.shape[:2]
    # The room/lobby ready button is centered horizontally, unlike the
    # settlement ready button in the lower-right corner.
    x0 = int(width * 0.35)
    x1 = int(width * 0.65)
    y0 = int(height * 0.42)
    y1 = int(height * 0.72)
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([8, 80, 120]), np.array([35, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 3000 or w < 120 or h < 35:
            continue
        aspect = w / max(h, 1)
        center_x = x0 + x + w // 2
        center_y = y0 + y + h // 2
        if 1.8 <= aspect <= 5.0 and width * 0.40 <= center_x <= width * 0.60 and height * 0.45 <= center_y <= height * 0.75:
            candidates.append((x0 + x, y0 + y, w, h, area))
    if not candidates:
        return None
    x, y, w, h, area = max(candidates, key=lambda item: item[4])
    confidence = min(1.0, area / 12000.0)
    return FlowDetection("settlement_ready", confidence, x, y, w, h)


def _has_lower_hand_card_tiles(image: np.ndarray) -> bool:
    height, width = image.shape[:2]
    roi = image[int(height * 0.45) : int(height * 0.98), int(width * 0.02) : int(width * 0.95)]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 110, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tile_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect = w / max(h, 1)
        center_y = int(height * 0.45) + y + h / 2
        if (
            area >= 10000
            and 60 <= w <= 220
            and 80 <= h <= 420
            and 0.25 <= aspect <= 1.8
            and center_y >= height * 0.62
        ):
            tile_count += 1
            if tile_count >= 2:
                return True
    return False


def _detect_orange_ready_button(image: np.ndarray) -> FlowDetection | None:
    height, width = image.shape[:2]
    # The settlement "准备" button sits in the lower-right. Limit the search so
    # normal orange game decorations do not trigger a ready click.
    x0 = int(width * 0.72)
    y0 = int(height * 0.72)
    roi = image[y0:height, x0:width]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([8, 80, 120]), np.array([35, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 2500 or w < 90 or h < 35:
            continue
        if y0 + y + h // 2 < height * 0.82:
            continue
        aspect = w / max(h, 1)
        if 1.7 <= aspect <= 4.6:
            candidates.append((x0 + x, y0 + y, w, h, area))
    if not candidates:
        return None
    x, y, w, h, area = max(candidates, key=lambda item: item[4])
    confidence = min(1.0, area / 12000.0)
    return FlowDetection("settlement_ready", confidence, x, y, w, h)


def _detect_already_ready_state(image: np.ndarray) -> FlowDetection | None:
    height, width = image.shape[:2]
    x0 = 0
    y0 = int(height * 0.55)
    roi = image[y0 : int(height * 0.85), x0 : int(width * 0.35)]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 100, 120]), np.array([45, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 800 <= area <= 3500 and 30 <= w <= 80 and 30 <= h <= 80:
            boxes.append((x0 + x, y0 + y, w, h, area))
    if len(boxes) < 2:
        return None
    x = min(box[0] for box in boxes)
    y = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return FlowDetection("already_ready", min(1.0, len(boxes) / 3.0), x, y, x2 - x, y2 - y)


def _detect_final_score_screen(image: np.ndarray) -> FlowDetection | None:
    height, width = image.shape[:2]
    lower = image[int(height * 0.68) : height, int(width * 0.22) : int(width * 0.78)]
    if lower.size == 0:
        return None
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([40, 80, 90]), np.array([90, 255, 255]))
    blue = cv2.inRange(hsv, np.array([90, 70, 90]), np.array([125, 255, 255]))
    green_ratio = float(np.count_nonzero(green)) / green.size
    blue_ratio = float(np.count_nonzero(blue)) / blue.size
    if green_ratio > 0.01 and blue_ratio > 0.008:
        return FlowDetection("final_score", min(1.0, green_ratio + blue_ratio))
    return None
