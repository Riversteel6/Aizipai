"""Card region detection helpers.

The first vision milestone only detects the player's own hand cards. This
module provides the document-level detector entry points while delegating the
actual calibrated slot search to ``hand_recognizer``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision.hand_recognizer import CardSlot, detect_hand_slots


def detect_card_slots(
    image: np.ndarray,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    search_padding_left: int = 180,
    search_padding_top: int = 90,
    search_padding_right: int = 260,
) -> list[CardSlot]:
    """Detect visible own-hand card slots from a calibrated screenshot."""

    return detect_hand_slots(
        image,
        config_path=config_path,
        search_padding_left=search_padding_left,
        search_padding_top=search_padding_top,
        search_padding_right=search_padding_right,
    )


def crop_card_slots(image: np.ndarray, slots: list[CardSlot]) -> list[np.ndarray]:
    """Return independent image crops for detected card slots."""

    return [image[slot.y : slot.y + slot.h, slot.x : slot.x + slot.w].copy() for slot in slots]


detect_cards = detect_card_slots


__all__ = ["CardSlot", "detect_card_slots", "detect_cards", "crop_card_slots"]
