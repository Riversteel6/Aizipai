"""Recognize interactive pending cards from configured screenshot regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from engine.cards import normalize_card_label
from vision.card_classifier import CardClassification, classify_card_crop
from vision.regions import Region, crop_region, load_regions_for_size
from vision.template_loader import TemplateImage, load_templates_from_dirs

PENDING_REGION_NAMES = ("pending_action_card", "opponent_pending_card")


def load_templates(template_dirs: tuple[str | Path, ...], *, grayscale: bool = False) -> list[TemplateImage]:
    return load_templates_from_dirs(template_dirs, grayscale=grayscale)


@dataclass(frozen=True)
class PendingCard:
    region_name: str
    name: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    template: str

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> dict[str, float | int | str | tuple[int, int]]:
        return {
            "region_name": self.region_name,
            "name": self.name,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "center": self.center,
            "template": self.template,
        }


def recognize_pending_cards(
    image: np.ndarray,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    template_dirs: tuple[str | Path, ...] = (
        "data/templates/discard_auto",
        "data/templates/hand_auto",
    ),
    threshold: float = 0.45,
    region_names: tuple[str, ...] = PENDING_REGION_NAMES,
) -> dict[str, PendingCard | None]:
    """Recognize pending action cards without merging them into discard history."""

    height, width = image.shape[:2]
    templates = _load_pending_templates(template_dirs)
    result: dict[str, PendingCard | None] = {name: None for name in region_names}
    for region in _target_regions(config_path, width, height, region_names):
        surface = _pending_card_surface(image, region)
        if surface is None:
            continue
        surface_region, classifier_crop, is_tall_surface = surface
        classification = _classify_pending_surface(
            image,
            surface_region,
            classifier_crop,
            is_tall_surface=is_tall_surface,
            threshold=threshold,
            templates=templates,
        )
        if classification is None or not classification.accepted:
            continue
        label = normalize_card_label(classification.name)
        if not label:
            continue
        result[region.name] = _pending_from_classification(surface_region, label, classification)
    return result


def _classify_pending_surface(
    image: np.ndarray,
    surface_region: Region,
    primary_crop: np.ndarray,
    *,
    is_tall_surface: bool,
    threshold: float,
    templates: list[TemplateImage],
) -> CardClassification | None:
    if not is_tall_surface:
        classification = classify_card_crop(
            primary_crop,
            threshold=threshold,
            templates=templates,
        )
        return classification if classification.accepted else None

    # Big red glyphs (for example 贰) extend lower than compact black glyphs
    # (for example 玖).  A single fixed top-crop height therefore creates a
    # false threshold boundary.  Classify two nearby card-height crops and
    # require them to agree on the label; one of the two must still clear the
    # normal pending-card evidence threshold.  This is scale agreement, not a
    # card-name exception or a lowered confidence threshold.
    full_surface = crop_region(image, surface_region)
    surface_width = full_surface.shape[1]
    secondary_height = min(
        full_surface.shape[0],
        max(1, int(round(surface_width * 1.15))),
    )
    secondary_crop = full_surface[:secondary_height]
    primary = classify_card_crop(primary_crop, threshold=0.0, templates=templates)
    secondary = classify_card_crop(secondary_crop, threshold=0.0, templates=templates)
    primary_label = normalize_card_label(primary.name)
    secondary_label = normalize_card_label(secondary.name)
    if not primary_label or primary_label != secondary_label:
        return None
    best = max((primary, secondary), key=lambda item: item.confidence)
    required_confidence = max(threshold, 0.60)
    if best.confidence < required_confidence:
        return None
    return CardClassification(
        name=best.name,
        confidence=best.confidence,
        template=best.template,
        accepted=True,
    )


def _pending_card_surface(
    image: np.ndarray,
    region: Region,
) -> tuple[Region, np.ndarray, bool] | None:
    """Locate the actual white pending tile before classifying its glyph.

    Pending-card regions are deliberately much taller than one ordinary hand
    slot.  Classifying the entire configured region also classifies counters,
    table texture and blank space.  On the live CHI incident this turned a
    clearly visible ``玖`` into a low-confidence ``五``.  The response tile is
    the dominant filled white contour; on its tall presentation the reliable
    upright glyph is in the top card-height segment.
    """

    region_crop = crop_region(image, region)
    if region_crop.size == 0:
        return None
    gray = cv2.cvtColor(region_crop, cv2.COLOR_BGR2GRAY) if region_crop.ndim == 3 else region_crop
    mask = cv2.inRange(gray, 170, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_width = max(18, int(round(region.w * 0.30)))
    min_height = max(28, int(round(region.w * 0.45)))
    surfaces: list[tuple[float, int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_width or h < min_height:
            continue
        rectangle_area = max(1, w * h)
        fill_ratio = float(cv2.contourArea(contour)) / rectangle_area
        if fill_ratio < 0.55:
            continue
        surfaces.append((float(rectangle_area), x, y, w, h))
    if not surfaces:
        return None

    _area, x, y, w, h = max(surfaces)
    card_crop = region_crop[y : y + h, x : x + w]
    is_tall_surface = h >= max(1, int(round(w * 2.0)))
    classifier_crop = card_crop
    if is_tall_surface:
        upright_height = min(h, max(1, int(round(w * 1.05))))
        classifier_crop = card_crop[:upright_height]
    surface_region = Region(
        name=region.name,
        x=region.x + x,
        y=region.y + y,
        w=w,
        h=h,
    )
    return surface_region, classifier_crop, is_tall_surface


def _load_pending_templates(template_dirs: tuple[str | Path, ...]) -> list[TemplateImage]:
    return load_templates(template_dirs, grayscale=False)


def _target_regions(
    config_path: str | Path,
    width: int,
    height: int,
    region_names: tuple[str, ...],
) -> list[Region]:
    wanted = set(region_names)
    return [region for region in load_regions_for_size(config_path, width, height) if region.name in wanted]


def _pending_from_classification(region: Region, label: str, classification: CardClassification) -> PendingCard:
    return PendingCard(
        region_name=region.name,
        name=label,
        confidence=classification.confidence,
        x=region.x,
        y=region.y,
        w=region.w,
        h=region.h,
        template=classification.template,
    )


__all__ = ["PendingCard", "recognize_pending_cards"]
