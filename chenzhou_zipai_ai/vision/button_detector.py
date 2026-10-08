"""Action button detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from vision.regions import Region, crop_region, load_regions_for_size
from vision.image_source import read_bgr
from vision.template_loader import TemplateImage, load_templates

_IGNORED_TEMPLATE_FILES = {"hu_001.png", "hu_002.png", "hu_003.png"}
_DISPLAY_BOXES = {
    "hu": (40, -15, 150, 170),
    "pass": (30, 0, 200, 205),
}
_RESPONSE_BUTTON_NAMES = {"chi", "peng", "pass", "hu"}
_RESPONSE_FALLBACK_THRESHOLD = 0.64


@dataclass(frozen=True)
class ButtonDetection:
    name: str
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
            "name": self.name,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "center": self.center,
        }


def load_button_threshold(path: str | Path = "config/thresholds.yaml") -> float:
    config_path = Path(path)
    if not config_path.exists():
        return 0.82
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return float(data.get("vision", {}).get("button_match_confidence", 0.82))


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _buttons_region(config_path: str | Path, width: int, height: int) -> Region:
    for region in load_regions_for_size(config_path, width, height):
        if region.name == "buttons":
            return region
    raise ValueError(f"Region config has no 'buttons' region: {config_path}")


def _best_match(search_image: np.ndarray, template: TemplateImage) -> tuple[float, tuple[int, int]]:
    if template.height > search_image.shape[0] or template.width > search_image.shape[1]:
        return 0.0, (0, 0)
    result = cv2.matchTemplate(search_image, template.image, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_location = cv2.minMaxLoc(result)
    return float(max_value), max_location


def _overlap_ratio(first: ButtonDetection, second: ButtonDetection) -> float:
    x0 = max(first.x, second.x)
    y0 = max(first.y, second.y)
    x1 = min(first.x + first.w, second.x + second.w)
    y1 = min(first.y + first.h, second.y + second.h)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    smaller_area = min(first.w * first.h, second.w * second.h)
    return intersection / smaller_area if smaller_area else 0.0


def _non_overlapping_fallbacks(
    candidates: list[ButtonDetection],
    detected: list[ButtonDetection],
) -> list[ButtonDetection]:
    accepted: list[ButtonDetection] = []
    for candidate in sorted(candidates, key=lambda item: -item.confidence):
        if any(
            _overlap_ratio(candidate, existing) >= 0.5
            for existing in (*detected, *accepted)
        ):
            continue
        accepted.append(candidate)
    return accepted


def detect_buttons(
    image: np.ndarray,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    template_dir: str | Path = "data/templates/buttons",
    threshold: float | None = None,
    names: Iterable[str] | None = None,
    search_padding: int = 120,
) -> list[ButtonDetection]:
    """Detect visible action buttons in the configured buttons region."""

    height, width = image.shape[:2]
    region = _buttons_region(config_path, width, height)
    search_region = Region(
        name=region.name,
        x=max(0, region.x - search_padding),
        y=max(0, region.y - search_padding),
        w=min(width, region.x2 + search_padding) - max(0, region.x - search_padding),
        h=min(height, region.y2 + search_padding) - max(0, region.y - search_padding),
    )
    search = _as_gray(crop_region(image, search_region))
    templates = load_templates(template_dir)
    allowed = set(names) if names is not None else None
    min_confidence = (
        load_button_threshold(Path(config_path).with_name("thresholds.yaml"))
        if threshold is None
        else threshold
    )

    detections: list[ButtonDetection] = []
    fallback_candidates: list[ButtonDetection] = []
    for template in templates:
        if template.path.name in _IGNORED_TEMPLATE_FILES:
            continue
        if allowed is not None and template.name not in allowed:
            continue
        confidence, location = _best_match(search, template)
        x = search_region.x + location[0]
        y = search_region.y + location[1]
        output_x = x
        output_y = y
        output_w = template.width
        output_h = template.height
        if template.name in _DISPLAY_BOXES:
            offset_x, offset_y, width_override, height_override = _DISPLAY_BOXES[template.name]
            output_x = max(0, x + offset_x)
            output_y = max(0, y + offset_y)
            output_w = width_override
            output_h = height_override
        detection = ButtonDetection(
            name=template.name,
            confidence=confidence,
            x=output_x,
            y=output_y,
            w=output_w,
            h=output_h,
        )
        if confidence >= min_confidence:
            detections.append(detection)
        elif (
            template.name in _RESPONSE_BUTTON_NAMES
            and confidence >= _RESPONSE_FALLBACK_THRESHOLD
        ):
            fallback_candidates.append(detection)

    if any(detection.name in _RESPONSE_BUTTON_NAMES for detection in detections):
        detections.extend(_non_overlapping_fallbacks(fallback_candidates, detections))

    detections.sort(key=lambda item: (-item.confidence, item.name))
    deduped: list[ButtonDetection] = []
    seen: set[str] = set()
    for detection in detections:
        if detection.name in seen:
            continue
        deduped.append(detection)
        seen.add(detection.name)
    return deduped


def detect_buttons_from_path(
    screenshot: str | Path,
    **kwargs,
) -> list[ButtonDetection]:
    image = read_bgr(screenshot)
    return detect_buttons(image, **kwargs)
