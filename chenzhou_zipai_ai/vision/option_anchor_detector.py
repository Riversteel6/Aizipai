"""Detect the visible 比牌/吃牌 labels that anchor option panels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from vision.viewport import ViewportTransform


BASE_WIDTH = 2344
BASE_HEIGHT = 1080
NORMALIZED_ANCHOR_SIZE = (48, 96)
ASSET_DIR = Path(__file__).resolve().parent / "assets" / "option_anchors"


@dataclass(frozen=True)
class OptionAnchor:
    region_name: str
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> dict[str, float | int | str | tuple[int, int]]:
        return {
            "region_name": self.region_name,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "confidence": self.confidence,
            "center": self.center,
        }


def detect_option_anchors(
    image: np.ndarray,
    *,
    threshold: float = 0.75,
    min_margin: float = 0.10,
) -> list[OptionAnchor]:
    """Return high-confidence anchors, ordered from left to right."""

    if image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    transform = ViewportTransform(BASE_WIDTH, BASE_HEIGHT, width, height)
    scale_x = transform.scale_x
    scale_y = transform.scale_y
    scan_height = min(height, max(0, transform.map_y(540)))
    yellow = _yellow_mask(image[:scan_height])
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, round(9 * scale_x)), max(5, round(17 * scale_y))),
    )
    merged = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    templates = _anchor_templates()
    matches: list[OptionAnchor] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not _plausible_anchor_box(x, y, w, h, transform=transform):
            continue
        normalized = cv2.resize(
            yellow[y : y + h, x : x + w],
            NORMALIZED_ANCHOR_SIZE,
            interpolation=cv2.INTER_NEAREST,
        )
        scores = {
            region_name: float(cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            for region_name, template in templates.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        region_name, confidence = ranked[0]
        margin = confidence - ranked[1][1]
        if confidence < threshold or margin < min_margin:
            continue
        matches.append(
            OptionAnchor(
                region_name=region_name,
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=round(confidence, 3),
            )
        )

    anchors = _best_match_per_type(matches)
    by_type = {anchor.region_name: anchor for anchor in anchors}
    compare = by_type.get("compare_options")
    chi = by_type.get("chi_options")
    if compare is not None and chi is not None and compare.x >= chi.x:
        return []
    return sorted(anchors, key=lambda anchor: anchor.x)


def _yellow_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array([18, 120, 130], dtype=np.uint8),
        np.array([42, 255, 255], dtype=np.uint8),
    )


def _plausible_anchor_box(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    transform: ViewportTransform,
) -> bool:
    min_x, top_y = transform.map_point(500, 0)
    _, max_y = transform.map_point(500, 190)
    return (
        x >= min_x
        and top_y <= y <= max_y
        and round(25 * transform.scale_x) <= w <= round(65 * transform.scale_x)
        and round(70 * transform.scale_y) <= h <= round(125 * transform.scale_y)
    )


def _best_match_per_type(matches: list[OptionAnchor]) -> list[OptionAnchor]:
    best: dict[str, OptionAnchor] = {}
    for match in matches:
        current = best.get(match.region_name)
        if current is None or match.confidence > current.confidence:
            best[match.region_name] = match
    return list(best.values())


@lru_cache(maxsize=1)
def _anchor_templates() -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for filename, region_name in (
        ("compare.png", "compare_options"),
        ("chi.png", "chi_options"),
    ):
        image = cv2.imread(str(ASSET_DIR / filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Option anchor template not found: {ASSET_DIR / filename}")
        templates[region_name] = cv2.resize(
            image,
            NORMALIZED_ANCHOR_SIZE,
            interpolation=cv2.INTER_NEAREST,
        )
    return templates


__all__ = ["OptionAnchor", "detect_option_anchors"]
