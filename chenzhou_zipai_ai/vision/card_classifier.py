"""Card classification from cropped images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from vision.hand_recognizer import classify_hand_slot
from vision.template_loader import TemplateImage, load_templates_from_dirs

DEFAULT_CARD_TEMPLATE_DIRS: tuple[str | Path, ...] = (
    "data/templates/hand_auto",
)


def load_templates(template_dir: str | Path | tuple[str | Path, ...], *, grayscale: bool = False) -> list[TemplateImage]:
    return load_templates_from_dirs(template_dir, grayscale=grayscale)


@dataclass(frozen=True)
class CardClassification:
    name: str
    confidence: float
    template: str
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "template": self.template,
            "accepted": self.accepted,
        }


def classify_card_crop(
    crop: np.ndarray,
    *,
    template_dir: str | Path | tuple[str | Path, ...] = DEFAULT_CARD_TEMPLATE_DIRS,
    threshold: float = 0.72,
    templates: Sequence[TemplateImage] | None = None,
) -> CardClassification:
    """Classify one cropped card image with the hand template library."""

    loaded_templates = list(templates) if templates is not None else load_templates(template_dir, grayscale=False)
    name, confidence, template = classify_hand_slot(crop, loaded_templates)
    return CardClassification(
        name=name,
        confidence=confidence,
        template=template,
        accepted=bool(name) and confidence >= threshold,
    )


def classify_card_crops(
    crops: Sequence[np.ndarray],
    *,
    template_dir: str | Path | tuple[str | Path, ...] = DEFAULT_CARD_TEMPLATE_DIRS,
    threshold: float = 0.72,
    templates: Sequence[TemplateImage] | None = None,
) -> list[CardClassification]:
    """Classify multiple cropped card images while loading templates once."""

    loaded_templates = list(templates) if templates is not None else load_templates(template_dir, grayscale=False)
    return [
        classify_card_crop(crop, threshold=threshold, templates=loaded_templates)
        for crop in crops
    ]


__all__ = ["CardClassification", "classify_card_crop", "classify_card_crops"]
