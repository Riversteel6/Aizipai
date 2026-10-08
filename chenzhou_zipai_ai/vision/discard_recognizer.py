"""Recognize right-side historical discard cards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from vision.hand_recognizer import classify_hand_slot
from vision.regions import Region, crop_region, load_regions_for_size
from vision.template_loader import TemplateImage, load_templates_from_dirs


@dataclass(frozen=True)
class DiscardCard:
    region_name: str
    name: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    template: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "region_name": self.region_name,
            "name": self.name,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "template": self.template,
        }


def _target_regions(config_path: str | Path, width: int, height: int) -> list[Region]:
    wanted = {"opponent_discards", "my_discards"}
    return [region for region in load_regions_for_size(config_path, width, height) if region.name in wanted]


def _detect_slots(image, region: Region) -> list[tuple[int, int, int, int]]:
    crop = crop_region(image, region)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mask = cv2.inRange(gray, 180, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    slots: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 45 <= w <= 95 and 40 <= h <= 95 and area >= 1500:
            slots.append((x + region.x, y + region.y, w, h))
    slots.sort(key=lambda item: (round(item[1] / 55), item[0]))
    return slots


def _load_discard_templates(template_dirs: tuple[str | Path, ...]) -> list[TemplateImage]:
    return load_templates_from_dirs(template_dirs, grayscale=False)


def recognize_discards(
    image,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    template_dirs: tuple[str | Path, ...] = (
        "data/templates/discard_auto",
        "data/templates/hand_auto",
    ),
    threshold: float = 0.45,
) -> dict[str, list[DiscardCard]]:
    height, width = image.shape[:2]
    templates = _load_discard_templates(template_dirs)
    result: dict[str, list[DiscardCard]] = {}
    for region in _target_regions(config_path, width, height):
        cards: list[DiscardCard] = []
        for x, y, w, h in _detect_slots(image, region):
            slot_image = image[y : y + h, x : x + w]
            name, confidence, template = classify_hand_slot(slot_image, templates)
            if confidence < threshold:
                continue
            cards.append(
                DiscardCard(
                    region_name=region.name,
                    name=name,
                    confidence=confidence,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    template=template,
                )
            )
        result[region.name] = cards
    return result
