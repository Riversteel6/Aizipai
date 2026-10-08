"""Detect whether the local player currently has the dealer marker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision.regions import crop_region, load_regions_for_size


@dataclass(frozen=True)
class SeatRoleDetection:
    role: str
    expected_total: int
    confidence: float
    reason: str
    region: dict[str, int | str] | None = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "expected_total": self.expected_total,
            "confidence": self.confidence,
            "reason": self.reason,
            "region": self.region,
        }


def _gold_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([10, 70, 90], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def detect_seat_role(
    image: np.ndarray,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    min_gold_pixels: int = 120,
    min_gold_ratio: float = 0.018,
) -> SeatRoleDetection:
    height, width = image.shape[:2]
    region = next(
        (item for item in load_regions_for_size(config_path, width, height) if item.name == "my_dealer_marker"),
        None,
    )
    if region is None:
        return SeatRoleDetection(
            role="unknown",
            expected_total=20,
            confidence=0.0,
            reason="dealer_marker_region_missing",
            region=None,
        )
    crop = crop_region(image, region)
    mask = _gold_mask(crop)
    gold_pixels = int(cv2.countNonZero(mask))
    ratio = gold_pixels / float(max(1, crop.shape[0] * crop.shape[1]))
    if gold_pixels >= min_gold_pixels and ratio >= min_gold_ratio:
        return SeatRoleDetection(
            role="dealer",
            expected_total=21,
            confidence=min(1.0, ratio / 0.08),
            reason=f"dealer_marker_gold_pixels:{gold_pixels}",
            region=region.to_dict(),
        )
    return SeatRoleDetection(
        role="player",
        expected_total=20,
        confidence=max(0.5, 1.0 - min(1.0, ratio / max(min_gold_ratio, 0.001))),
        reason=f"dealer_marker_absent_gold_pixels:{gold_pixels}",
        region=region.to_dict(),
    )
