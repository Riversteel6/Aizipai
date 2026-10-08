"""Detect the central discard confirmation button."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision.regions import crop_region, load_regions_for_size


@dataclass(frozen=True)
class DiscardButtonDetection:
    confidence: float
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> dict[str, float | int | str | tuple[int, int]]:
        return {
            "name": "discard",
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "center": self.center,
        }


def detect_discard_button(
    image: np.ndarray,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
) -> DiscardButtonDetection | None:
    height, width = image.shape[:2]
    region = next(
        (item for item in load_regions_for_size(config_path, width, height) if item.name == "discard_button"),
        None,
    )
    if region is None:
        return None

    crop = crop_region(image, region)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([12, 120, 150]), np.array([45, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 7000 or w < 130 or h < 45:
            continue
        aspect = w / max(h, 1)
        if 1.8 <= aspect <= 5.2:
            candidates.append((x, y, w, h, area))
    if not candidates:
        return None
    x, y, w, h, area = max(candidates, key=lambda item: item[4])
    confidence = min(1.0, area / 30000.0)
    return DiscardButtonDetection(confidence, region.x + x, region.y + y, w, h)
