"""Own-hand card recognition from calibrated screenshots."""

from __future__ import annotations

import hashlib
import os
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from vision.hybrid_card_classifier import fuse_card_predictions
from vision.image_source import read_bgr
from vision.neural_card_classifier import (
    DEFAULT_MODEL_PATH,
    NeuralCardPrediction,
    OnnxCardClassifier,
)
from vision.regions import Region, crop_region, load_regions_for_size
from vision.template_loader import TemplateImage, load_templates_from_dirs

_IGNORED_TEMPLATE_FILES = {"贰_001.png", "玖_002.png"}
_TOP_ONLY_TEMPLATE_FILES = {"玖_003.png"}
_TEMPLATE_GRAY_CACHE: dict[str, np.ndarray] = {}
_TEMPLATE_RESIZE_CACHE: dict[tuple[str, int, int], np.ndarray] = {}
_TEMPLATE_GLYPH_CACHE: dict[str, np.ndarray | None] = {}
_RESIZED_TEMPLATE_MATRIX_CACHE_LIMIT = 8
_RESIZED_TEMPLATE_MATRIX_CACHE: OrderedDict[
    tuple[int, int, tuple[str, ...]],
    np.ndarray,
] = OrderedDict()
_RESIZED_TEMPLATE_MATRIX_LOCK = RLock()
_GLYPH_TEMPLATE_MATRIX_CACHE_LIMIT = 4
_GLYPH_TEMPLATE_MATRIX_CACHE: OrderedDict[tuple[str, ...], np.ndarray] = OrderedDict()
_GLYPH_TEMPLATE_MATRIX_LOCK = RLock()
_CLASSIFICATION_CACHE_LIMIT = 1024
_CLASSIFICATION_CACHE: OrderedDict[tuple[object, ...], tuple[tuple[float, str, str], ...]] = OrderedDict()
_CLASSIFICATION_CACHE_LOCK = RLock()
_GRID_FALLBACK_MIN_CARD_SURFACE_RATIO = 0.20

_LOW_CONFIDENCE_CONTOUR_SCORE = 0.64
_LOW_CONFIDENCE_CONTOUR_MARGIN = 0.07


def load_templates(template_dir: str | Path | tuple[str | Path, ...], *, grayscale: bool = False) -> list[TemplateImage]:
    return load_templates_from_dirs(template_dir, grayscale=grayscale)


@dataclass(frozen=True)
class HandCard:
    name: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    template: str
    clickable: bool = True
    card_id: str | None = None
    raw_confidence: float | None = None
    runner_up_name: str = ""
    runner_up_confidence: float = 0.0
    recognition_source: str = "template"
    neural_label: str = ""
    neural_confidence: float = 0.0
    hybrid_reason: str = ""

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> dict[str, float | int | str | tuple[int, int]]:
        raw_confidence = self.confidence if self.raw_confidence is None else self.raw_confidence
        return {
            "name": self.name,
            "card_id": self.card_id,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "center": self.center,
            "template": self.template,
            "clickable": self.clickable,
            "raw_confidence": raw_confidence,
            "runner_up_name": self.runner_up_name,
            "runner_up_confidence": self.runner_up_confidence,
            "confidence_margin": raw_confidence - self.runner_up_confidence,
            "recognition_source": self.recognition_source,
            "neural_label": self.neural_label,
            "neural_confidence": self.neural_confidence,
            "hybrid_reason": self.hybrid_reason,
        }


@dataclass(frozen=True)
class CardSlot:
    x: int
    y: int
    w: int
    h: int
    source: str = "contour"


def read_image(path: str | Path) -> np.ndarray:
    return read_bgr(path)


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _my_hand_region(config_path: str | Path, width: int, height: int) -> Region:
    for region in load_regions_for_size(config_path, width, height):
        if region.name == "my_hand":
            return region
    raise ValueError(f"Region config has no 'my_hand' region: {config_path}")


def _split_component(x: int, y: int, w: int, h: int) -> list[CardSlot]:
    if h >= 480:
        return [
            CardSlot(x, y, w, 128),
            CardSlot(x, y + 128, w, 127),
            CardSlot(x, y + 255, w, 128),
            CardSlot(x, y + h - 152, w, 152),
        ]
    if h >= 360:
        return [
            CardSlot(x, y, w, 128),
            CardSlot(x, y + 128, w, 127),
            CardSlot(x, y + h - 152, w, 152),
        ]
    if h >= 235:
        return [
            CardSlot(x, y, w, 127),
            CardSlot(x, y + h - 152, w, 152),
        ]
    return [CardSlot(x, y, w, h)]


def detect_hand_slots(
    image: np.ndarray,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    search_padding_left: int = 180,
    search_padding_top: int = 180,
    search_padding_right: int = 260,
    recover_missing_top_row: bool = False,
    expected_slot_count: int | None = None,
) -> list[CardSlot]:
    """Find visible card slots inside the configured own-hand region."""

    height, width = image.shape[:2]
    region = _my_hand_region(config_path, width, height)
    search_y = max(0, region.y - search_padding_top)
    search_region = Region(
        name=region.name,
        x=max(0, region.x - search_padding_left),
        y=search_y,
        w=min(width, region.x2 + search_padding_right) - max(0, region.x - search_padding_left),
        h=region.y2 - search_y,
    )
    hand = crop_region(image, search_region)
    gray = _as_gray(hand)
    mask = cv2.inRange(gray, 170, 255)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    slots: list[CardSlot] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (70 <= w <= 170 and 90 <= h <= 540 and area >= 5000):
            continue
        for slot in _split_component(x + search_region.x, y + search_region.y, w, h):
            if _is_floating_action_card(slot, region):
                continue
            slots.append(slot)

    slots.sort(key=lambda item: (item.x, item.y))
    if expected_slot_count is not None and len(slots) == expected_slot_count:
        return slots
    return _with_grid_fallback_slots(
        slots,
        region,
        width,
        search_padding_right,
        search_padding_top=search_padding_top,
        recover_missing_top_row=recover_missing_top_row,
    )


def hand_target_is_clickable(
    image: np.ndarray,
    x: int,
    y: int,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    point_margin: int = 14,
) -> bool:
    """Check one planned hand coordinate without classifying the whole hand."""

    height, width = image.shape[:2]
    x = int(x)
    y = int(y)
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    candidates = []
    for slot in detect_hand_slots(image, config_path=config_path):
        if not (
            slot.x - point_margin <= x <= slot.x + slot.w + point_margin
            and slot.y - point_margin <= y <= slot.y + slot.h + point_margin
        ):
            continue
        center_x = slot.x + slot.w // 2
        center_y = slot.y + slot.h // 2
        candidates.append(((center_x - x) ** 2 + (center_y - y) ** 2, slot))
    for _distance, slot in sorted(candidates, key=lambda item: item[0]):
        crop = image[
            max(0, slot.y) : min(height, slot.y + slot.h),
            max(0, slot.x) : min(width, slot.x + slot.w),
        ]
        if crop.size and _is_clickable_card(crop):
            return True
    return False


def _is_floating_action_card(slot: CardSlot, hand_region: Region) -> bool:
    return (
        slot.y < hand_region.y
        and slot.h >= hand_region.h * 0.38
        and slot.y + slot.h <= hand_region.y + hand_region.h * 0.10
    )


def _with_grid_fallback_slots(
    slots: list[CardSlot],
    region: Region,
    image_width: int,
    search_padding_right: int,
    *,
    search_padding_top: int,
    recover_missing_top_row: bool = False,
) -> list[CardSlot]:
    if len(slots) < 4:
        return slots
    compact_slots = _compact_four_stack_fallback_slots(slots)
    if compact_slots is not None:
        return compact_slots
    columns = sorted({slot.x for slot in slots})
    if len(columns) < 2:
        return slots
    gaps = [b - a for a, b in zip(columns, columns[1:]) if 90 <= b - a <= 180]
    if not gaps:
        return slots
    step = int(round(float(np.median(gaps))))
    rows = _row_prototypes(slots)
    if recover_missing_top_row:
        rows = _with_recovered_top_row(
            rows,
            region,
            search_padding_top=search_padding_top,
        )
    if not rows:
        return slots

    result = list(slots)
    x = columns[0]
    max_x = min(image_width - 70, region.x2 + search_padding_right)
    while x <= max_x:
        for y, w, h in rows:
            candidate = CardSlot(x, y, w, h, source="grid_fallback")
            if not _has_nearby_slot(result, candidate):
                result.append(candidate)
        x += step
    result.sort(key=lambda item: (item.x, item.y))
    return result


def _with_recovered_top_row(
    rows: list[tuple[int, int, int]],
    region: Region,
    *,
    search_padding_top: int,
) -> list[tuple[int, int, int]]:
    if len(rows) < 2:
        return rows
    ordered = sorted(rows)
    gaps = [
        next_y - y
        for (y, _, _), (next_y, _, _) in zip(ordered, ordered[1:])
        if 105 <= next_y - y <= 145
    ]
    if not gaps:
        return rows
    step = int(round(float(np.median(gaps))))
    first_y, first_w, first_h = ordered[0]
    recovered_y = first_y - step
    search_top = max(0, region.y - max(0, search_padding_top))
    if not (search_top <= recovered_y < first_y - 60):
        return rows
    if any(abs(y - recovered_y) <= 22 for y, _, _ in ordered):
        return rows
    return sorted([(recovered_y, first_w, first_h), *ordered])


def _compact_four_stack_fallback_slots(slots: list[CardSlot]) -> list[CardSlot] | None:
    rows = _row_prototypes(slots)
    if len(rows) < 4:
        return None
    rows = rows[:4]
    full_columns: list[int] = []
    bottom_only_columns: list[int] = []
    for column in sorted({slot.x for slot in slots}):
        column_slots = [slot for slot in slots if abs(slot.x - column) <= 28]
        matched_rows = sum(
            1
            for row_y, _, _ in rows
            if any(abs(slot.y - row_y) <= 24 for slot in column_slots)
        )
        if matched_rows >= 3:
            full_columns.append(column)
        elif matched_rows == 1 and any(abs(slot.y - rows[-1][0]) <= 32 for slot in column_slots):
            bottom_only_columns.append(column)
    if len(full_columns) < 2:
        return None

    median_width = int(round(float(np.median([slot.w for slot in slots if slot.w >= 100]))))
    step = max(120, median_width + 3)
    start_x = min(full_columns)
    stop_before = min((x for x in bottom_only_columns if x > start_x + step), default=None)
    if stop_before is None:
        return None

    result = list(slots)
    x = start_x
    while x < stop_before - step * 0.35:
        for row_y, row_w, row_h in rows:
            candidate = CardSlot(int(round(x)), row_y, row_w, row_h, source="grid_fallback")
            if not _has_nearby_slot(result, candidate):
                result.append(candidate)
        x += step
    result.sort(key=lambda item: (item.x, item.y))
    return result


def _row_prototypes(slots: list[CardSlot]) -> list[tuple[int, int, int]]:
    rows: list[list[CardSlot]] = []
    for slot in sorted(slots, key=lambda item: item.y):
        target = next((row for row in rows if abs(row[0].y - slot.y) <= 16), None)
        if target is None:
            rows.append([slot])
        else:
            target.append(slot)
    prototypes: list[tuple[int, int, int]] = []
    for row in rows:
        if len(row) < 2:
            continue
        y = int(round(float(np.median([slot.y for slot in row]))))
        w = int(round(float(np.median([slot.w for slot in row]))))
        h = int(round(float(np.median([slot.h for slot in row]))))
        prototypes.append((y, w, h))
    return prototypes


def _has_nearby_slot(slots: list[CardSlot], candidate: CardSlot) -> bool:
    return any(abs(slot.x - candidate.x) <= 28 and abs(slot.y - candidate.y) <= 22 for slot in slots)


def _card_name(template: TemplateImage) -> str:
    return template.label or template.name


def _template_key(template: TemplateImage) -> str:
    return str(template.path)


def _template_gray(template: TemplateImage) -> np.ndarray:
    key = _template_key(template)
    cached = _TEMPLATE_GRAY_CACHE.get(key)
    if cached is None:
        cached = _as_gray(template.image)
        _TEMPLATE_GRAY_CACHE[key] = cached
    return cached


def _resized_template_gray(template: TemplateImage, width: int, height: int) -> np.ndarray:
    key = (_template_key(template), width, height)
    cached = _TEMPLATE_RESIZE_CACHE.get(key)
    if cached is None:
        cached = cv2.resize(_template_gray(template), (width, height))
        _TEMPLATE_RESIZE_CACHE[key] = cached
    return cached


def _template_glyph_mask(template: TemplateImage) -> np.ndarray | None:
    key = _template_key(template)
    if key not in _TEMPLATE_GLYPH_CACHE:
        _TEMPLATE_GLYPH_CACHE[key] = _glyph_mask(template.image)
    return _TEMPLATE_GLYPH_CACHE[key]


def _match_slot(
    slot_image: np.ndarray,
    slot_gray: np.ndarray,
    slot_mask: np.ndarray | None,
    template: TemplateImage,
    *,
    resized_value: float | None = None,
    glyph_value: float | None = None,
) -> float:
    search = slot_gray
    template_gray = _template_gray(template)
    if resized_value is None:
        resized = _resized_template_gray(template, search.shape[1], search.shape[0])
        resized_score = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
        _, resized_value, _, _ = cv2.minMaxLoc(resized_score)
    if glyph_value is None:
        glyph_value = _glyph_match(slot_mask, template)
    if template.height > search.shape[0] or template.width > search.shape[1]:
        return max(float(resized_value), float(glyph_value))
    if template_gray.shape == search.shape:
        # ``resized`` is pixel-identical to the original template in this
        # case, so the second matchTemplate call would produce the same 1x1
        # score.  Keep the exact score and glyph fallback without doing the
        # duplicate OpenCV scan.
        return max(float(resized_value), float(glyph_value))
    result = cv2.matchTemplate(search, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, _ = cv2.minMaxLoc(result)
    return max(float(max_value), float(resized_value), float(glyph_value))


def _batch_resized_template_scores(
    slot_gray: np.ndarray,
    templates: list[TemplateImage],
) -> np.ndarray:
    """Compute equal-size TM_CCOEFF_NORMED scores in one native matrix product."""

    height, width = slot_gray.shape[:2]
    key = (width, height, tuple(_template_key(template) for template in templates))
    with _RESIZED_TEMPLATE_MATRIX_LOCK:
        matrix = _RESIZED_TEMPLATE_MATRIX_CACHE.get(key)
        if matrix is not None:
            _RESIZED_TEMPLATE_MATRIX_CACHE.move_to_end(key)
    if matrix is None:
        rows: list[np.ndarray] = []
        for template in templates:
            values = _resized_template_gray(template, width, height).astype(
                np.float32,
                copy=True,
            ).reshape(-1)
            values -= float(values.mean())
            norm = float(np.linalg.norm(values))
            if norm > 1e-12:
                values /= norm
            else:
                values.fill(0.0)
            rows.append(values)
        matrix = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
        with _RESIZED_TEMPLATE_MATRIX_LOCK:
            _RESIZED_TEMPLATE_MATRIX_CACHE[key] = matrix
            _RESIZED_TEMPLATE_MATRIX_CACHE.move_to_end(key)
            while len(_RESIZED_TEMPLATE_MATRIX_CACHE) > _RESIZED_TEMPLATE_MATRIX_CACHE_LIMIT:
                _RESIZED_TEMPLATE_MATRIX_CACHE.popitem(last=False)

    slot_values = slot_gray.astype(np.float32, copy=True).reshape(-1)
    slot_values -= float(slot_values.mean())
    slot_norm = float(np.linalg.norm(slot_values))
    if slot_norm <= 1e-12:
        return np.zeros(len(templates), dtype=np.float32)
    slot_values /= slot_norm
    return np.clip(matrix @ slot_values, -1.0, 1.0)


def _batch_glyph_template_scores(
    slot_mask: np.ndarray | None,
    templates: list[TemplateImage],
) -> np.ndarray:
    """Compute equal-size glyph TM_CCOEFF_NORMED scores in one native product."""

    if slot_mask is None:
        return np.zeros(len(templates), dtype=np.float32)
    key = tuple(_template_key(template) for template in templates)
    with _GLYPH_TEMPLATE_MATRIX_LOCK:
        matrix = _GLYPH_TEMPLATE_MATRIX_CACHE.get(key)
        if matrix is not None:
            _GLYPH_TEMPLATE_MATRIX_CACHE.move_to_end(key)
    if matrix is None:
        rows: list[np.ndarray] = []
        for template in templates:
            mask = _template_glyph_mask(template)
            if mask is None:
                rows.append(np.zeros(slot_mask.size, dtype=np.float32))
                continue
            values = mask.astype(np.float32, copy=True).reshape(-1)
            values -= float(values.mean())
            norm = float(np.linalg.norm(values))
            if norm > 1e-12:
                values /= norm
            else:
                values.fill(0.0)
            rows.append(values)
        matrix = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
        with _GLYPH_TEMPLATE_MATRIX_LOCK:
            _GLYPH_TEMPLATE_MATRIX_CACHE[key] = matrix
            _GLYPH_TEMPLATE_MATRIX_CACHE.move_to_end(key)
            while len(_GLYPH_TEMPLATE_MATRIX_CACHE) > _GLYPH_TEMPLATE_MATRIX_CACHE_LIMIT:
                _GLYPH_TEMPLATE_MATRIX_CACHE.popitem(last=False)

    slot_values = slot_mask.astype(np.float32, copy=True).reshape(-1)
    slot_values -= float(slot_values.mean())
    slot_norm = float(np.linalg.norm(slot_values))
    if slot_norm <= 1e-12:
        return np.zeros(len(templates), dtype=np.float32)
    slot_values /= slot_norm
    return np.clip(matrix @ slot_values, -1.0, 1.0)


def _glyph_mask(
    image: np.ndarray,
    *,
    remove_edge_artifacts: bool = False,
) -> np.ndarray | None:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image
    else:
        bgr = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((gray < 165) | ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 80))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    if remove_edge_artifacts:
        mask = _remove_edge_glyph_artifacts(mask)
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    if w < 8 or h < 8:
        return None

    crop = mask[y : y + h, x : x + w]
    canvas = np.zeros((80, 80), dtype=np.uint8)
    scale = min(70 / w, 70 / h)
    resized = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))))
    top = (80 - resized.shape[0]) // 2
    left = (80 - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _remove_edge_glyph_artifacts(mask: np.ndarray) -> np.ndarray:
    """Remove neighboring-card fragments without trimming the actual glyph."""

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if component_count <= 1:
        return mask

    height, width = mask.shape[:2]
    tiny_edge_area = max(24, round(height * width * 0.003))
    horizontal_edge_margin = max(2, round(width * 0.05))
    vertical_edge_margin = max(2, round(height * 0.05))
    keep: list[int] = []
    for index in range(1, component_count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[index]
        )
        near_left_or_right = (
            x <= horizontal_edge_margin
            or x + component_width >= width - horizontal_edge_margin
        )
        near_top_or_bottom = (
            y <= vertical_edge_margin
            or y + component_height >= height - vertical_edge_margin
        )
        side_spill = near_left_or_right and component_height >= round(height * 0.75)
        stacked_card_edge = (
            near_top_or_bottom
            and component_width >= round(width * 0.50)
            and component_height <= max(3, round(height * 0.05))
        )
        tiny_edge_fragment = (near_left_or_right or near_top_or_bottom) and area <= tiny_edge_area
        if side_spill or stacked_card_edge or tiny_edge_fragment:
            continue
        keep.append(index)

    if not keep:
        return mask
    return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)


def _glyph_match(slot_mask: np.ndarray | None, template: TemplateImage) -> float:
    template_mask = _template_glyph_mask(template)
    if slot_mask is None or template_mask is None:
        return 0.0
    result = cv2.matchTemplate(slot_mask, template_mask, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, _ = cv2.minMaxLoc(result)
    return float(max_value)


def _is_clickable_card(slot_image: np.ndarray) -> bool:
    if _is_selected_card(slot_image):
        return True
    gray = _as_gray(slot_image)
    if gray.size == 0:
        return False
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return False
    y0 = h // 6
    y1 = max(h - h // 6, y0 + 1)
    x0 = w // 6
    x1 = max(w - w // 6, x0 + 1)
    inner = gray[y0:y1, x0:x1]
    if inner.size == 0:
        inner = gray
    background = _card_background_pixels(gray)
    if background.size:
        background_mean = float(background.mean())
        background_bright_ratio = float((background > 185).mean())
        background_white_ratio = float((background > 205).mean())
        if background_mean < 195.0 and background_bright_ratio < 0.20 and background_white_ratio < 0.08:
            return False
    mean = float(inner.mean())
    very_dark_ratio = float((inner < 120).mean())
    dark_ratio = float((inner < 135).mean())
    return not (mean < 130.0 and (very_dark_ratio > 0.24 or dark_ratio > 0.35))


def _is_selected_card(slot_image: np.ndarray) -> bool:
    if slot_image.ndim != 3 or slot_image.shape[2] < 3:
        return False
    h, w = slot_image.shape[:2]
    if h < 16 or w < 16:
        return False
    background = _card_background_pixels(slot_image[:, :, :3])
    if not background.size:
        return False
    bgr = background.astype(np.int16, copy=False)
    blue, green, red = bgr[:, 0], bgr[:, 1], bgr[:, 2]
    pink_background = (
        (red >= 160)
        & (red - blue >= 35)
        & (red - green >= 20)
    )
    return float(pink_background.mean()) >= 0.65


def _card_background_pixels(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    patches = []
    boxes = (
        (0, h // 5, 0, w // 4),
        (0, h // 5, 3 * w // 4, w),
        (h // 5, 4 * h // 5, 0, max(1, w // 7)),
        (h // 5, 4 * h // 5, max(0, 6 * w // 7), w),
        (4 * h // 5, h, 0, w // 4),
        (4 * h // 5, h, 3 * w // 4, w),
    )
    for y0, y1, x0, x1 in boxes:
        patch = gray[y0:y1, x0:x1]
        if patch.size:
            patches.append(patch.reshape(-1, *gray.shape[2:]))
    if not patches:
        return np.array([], dtype=gray.dtype)
    return np.concatenate(patches)


def classify_hand_slot(slot_image: np.ndarray, templates: list[TemplateImage]) -> tuple[str, float, str]:
    scores = _candidate_scores(slot_image, templates)
    if not scores:
        return "", 0.0, ""
    confidence, name, template = scores[0]
    return name, confidence, template


def _classification_cache_key(
    slot_image: np.ndarray,
    templates: list[TemplateImage],
) -> tuple[object, ...] | None:
    if slot_image.size == 0 or not templates:
        return None
    small = cv2.resize(slot_image, (48, 48), interpolation=cv2.INTER_AREA)
    quantized = np.ascontiguousarray(small // 4)
    digest = hashlib.blake2b(quantized.tobytes(), digest_size=16).digest()
    template_bank = tuple(_template_key(template) for template in templates)
    return (slot_image.shape[:2], digest, template_bank)


def _cached_candidate_scores(key: tuple[object, ...] | None) -> list[tuple[float, str, str]] | None:
    if key is None:
        return None
    with _CLASSIFICATION_CACHE_LOCK:
        cached = _CLASSIFICATION_CACHE.get(key)
        if cached is None:
            return None
        _CLASSIFICATION_CACHE.move_to_end(key)
        return list(cached)


def _cache_candidate_scores(
    key: tuple[object, ...] | None,
    scores: list[tuple[float, str, str]],
) -> None:
    if key is None:
        return
    with _CLASSIFICATION_CACHE_LOCK:
        _CLASSIFICATION_CACHE[key] = tuple(scores)
        _CLASSIFICATION_CACHE.move_to_end(key)
        while len(_CLASSIFICATION_CACHE) > _CLASSIFICATION_CACHE_LIMIT:
            _CLASSIFICATION_CACHE.popitem(last=False)


def clear_classification_cache() -> None:
    with _CLASSIFICATION_CACHE_LOCK:
        _CLASSIFICATION_CACHE.clear()
    with _RESIZED_TEMPLATE_MATRIX_LOCK:
        _RESIZED_TEMPLATE_MATRIX_CACHE.clear()
    with _GLYPH_TEMPLATE_MATRIX_LOCK:
        _GLYPH_TEMPLATE_MATRIX_CACHE.clear()


@lru_cache(maxsize=4)
def _load_neural_classifier(model_path: str) -> OnnxCardClassifier:
    return OnnxCardClassifier(model_path)


def _candidate_scores(
    slot_image: np.ndarray,
    templates: list[TemplateImage],
) -> list[tuple[float, str, str]]:
    cache_key = _classification_cache_key(slot_image, templates)
    cached = _cached_candidate_scores(cache_key)
    if cached is not None:
        return cached
    slot_gray = _as_gray(slot_image)
    slot_mask = _glyph_mask(slot_image)
    eligible_templates: list[TemplateImage] = []
    for template in templates:
        if template.path.parent.name == "hand_auto" and template.path.name in _IGNORED_TEMPLATE_FILES:
            continue
        if template.path.parent.name == "hand_auto" and template.path.name in _TOP_ONLY_TEMPLATE_FILES and slot_image.shape[0] > 140:
            continue
        eligible_templates.append(template)
    if not eligible_templates:
        return []
    resized_scores = _batch_resized_template_scores(slot_gray, eligible_templates)
    glyph_scores = _batch_glyph_template_scores(slot_mask, eligible_templates)
    scored = [
        (
            _match_slot(
                slot_image,
                slot_gray,
                slot_mask,
                template,
                resized_value=float(resized_scores[index]),
                glyph_value=float(glyph_scores[index]),
            ),
            template,
        )
        for index, template in enumerate(eligible_templates)
    ]
    ranked = sorted(range(len(scored)), key=lambda index: scored[index][0], reverse=True)
    exact_indices = set(ranked[:6])
    if ranked:
        cutoff = scored[ranked[min(5, len(ranked) - 1)]][0] - 0.001
        exact_indices.update(
            index for index in ranked if scored[index][0] >= cutoff
        )
    exact_indices.update(
        index
        for index, (score, _template) in enumerate(scored)
        if any(abs(score - threshold) <= 0.001 for threshold in (0.60, 0.64, 0.70))
    )
    for index in exact_indices:
        _approximate, template = scored[index]
        scored[index] = (
            _match_slot(slot_image, slot_gray, slot_mask, template),
            template,
        )
    scores = [
        (score, _card_name(template), template.path.name)
        for score, template in scored
    ]
    scores.sort(reverse=True, key=lambda item: item[0])
    _cache_candidate_scores(cache_key, scores)
    return scores


def _runner_up_score(
    slot_image: np.ndarray,
    templates: list[TemplateImage],
    best_name: str,
) -> tuple[str, float]:
    scores = _candidate_scores(slot_image, templates)
    for score, name, _template in scores:
        if name != best_name:
            return name, score
    return "", 0.0


def _accept_low_confidence_contour(
    slot: CardSlot,
    confidence: float,
    runner_up_name: str,
    runner_up_confidence: float,
) -> bool:
    return (
        slot.source == "contour"
        and bool(runner_up_name)
        and confidence >= _LOW_CONFIDENCE_CONTOUR_SCORE
        and confidence - runner_up_confidence >= _LOW_CONFIDENCE_CONTOUR_MARGIN
    )


def _slot_grid_position(slot: CardSlot, slots: list[CardSlot]) -> tuple[int, int]:
    columns: list[list[CardSlot]] = []
    for candidate in sorted(slots, key=lambda item: item.x):
        for column in columns:
            if abs(column[0].x - candidate.x) <= 28:
                column.append(candidate)
                break
        else:
            columns.append([candidate])

    for column_index, column in enumerate(columns, start=1):
        if not any(abs(candidate.x - slot.x) <= 28 for candidate in column):
            continue
        ordered = sorted(column, key=lambda item: item.y)
        for layer_index, candidate in enumerate(ordered, start=1):
            if abs(candidate.x - slot.x) <= 28 and abs(candidate.y - slot.y) <= 22:
                return column_index, layer_index
    return 0, 0


def _secondary_low_confidence_acceptance(
    slot: CardSlot,
    slots: list[CardSlot],
    slot_image: np.ndarray,
    templates: list[TemplateImage],
    *,
    threshold: float,
) -> tuple[str, float, str] | None:
    if slot.source == "contour":
        original_slot_mask = _glyph_mask(slot_image)
        cleaned_slot_mask = _glyph_mask(slot_image, remove_edge_artifacts=True)
        edge_artifact_removed = (
            original_slot_mask is not None
            and cleaned_slot_mask is not None
            and not np.array_equal(original_slot_mask, cleaned_slot_mask)
        )
        if edge_artifact_removed:
            cleaned_scores = sorted(
                (
                    _glyph_match(cleaned_slot_mask, template),
                    _card_name(template),
                    template.path.name,
                )
                for template in templates
                if not (
                    template.path.parent.name == "hand_auto"
                    and template.path.name in _IGNORED_TEMPLATE_FILES
                )
            )
            cleaned_scores.reverse()
            if cleaned_scores:
                best_score, best_name, best_template = cleaned_scores[0]
                second_other = next(
                    (
                        score
                        for score, name, _template in cleaned_scores
                        if name != best_name
                    ),
                    0.0,
                )
                if best_score >= 0.86 and best_score - second_other >= 0.18:
                    return (
                        best_name,
                        best_score,
                        f"{best_template}:edge_artifact_recovery",
                    )
        return None

    if slot.source != "grid_fallback":
        return None

    scores = _candidate_scores(slot_image, templates)
    if not scores:
        return None
    best_score, best_name, best_template = scores[0]
    second_other = next((score for score, name, _template in scores if name != best_name), 0.0)
    if (
        0.60 <= best_score < threshold
        and best_score - second_other >= 0.12
    ):
        return best_name, best_score, f"{best_template}:secondary_grid_margin"
    return None


def _grid_fallback_has_card_surface(slot_image: np.ndarray) -> bool:
    if slot_image.size == 0:
        return False
    gray = _as_gray(slot_image)
    bright_ratio = float(np.count_nonzero(gray >= 170)) / float(gray.size)
    return bright_ratio >= _GRID_FALLBACK_MIN_CARD_SURFACE_RATIO


def recognize_hand(
    image: np.ndarray,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    template_dir: str | Path | tuple[str | Path, ...] = "data/templates/hand_auto",
    threshold: float = 0.70,
    search_padding_left: int = 180,
    search_padding_top: int = 180,
    search_padding_right: int = 260,
    recover_missing_top_row: bool = False,
    expected_hand_count: int | None = None,
    unknown_crop_dir: str | Path | None = None,
    debug_path: str | Path | None = None,
    diagnostics: dict[str, object] | None = None,
    hybrid_mode: str | None = None,
    neural_model_path: str | Path = DEFAULT_MODEL_PATH,
    neural_classifier: Any | None = None,
) -> list[HandCard]:
    """Recognize visible own-hand cards from a screenshot."""

    slots = detect_hand_slots(
        image,
        config_path=config_path,
        search_padding_left=search_padding_left,
        search_padding_top=search_padding_top,
        search_padding_right=search_padding_right,
        recover_missing_top_row=recover_missing_top_row,
        expected_slot_count=expected_hand_count,
    )
    raw_slot_count = len(slots)
    slots = [
        slot
        for slot in slots
        if slot.source != "grid_fallback"
        or _grid_fallback_has_card_surface(
            image[slot.y : slot.y + slot.h, slot.x : slot.x + slot.w]
        )
    ]
    templates = load_templates(template_dir, grayscale=False)
    slot_images = [
        image[slot.y : slot.y + slot.h, slot.x : slot.x + slot.w]
        for slot in slots
    ]
    resolved_hybrid_mode = (
        hybrid_mode
        or os.environ.get("AIZIPAI_CARD_HYBRID_MODE")
        or "off"
    ).strip().lower()
    if resolved_hybrid_mode not in {"off", "shadow", "enforce"}:
        raise ValueError(f"unsupported_hybrid_mode:{resolved_hybrid_mode}")
    neural_error = ""
    neural_elapsed_ms = 0.0
    predictions: list[NeuralCardPrediction | None] = [None] * len(slots)
    if resolved_hybrid_mode != "off":
        try:
            classifier = neural_classifier
            if classifier is None:
                model_path = Path(neural_model_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"neural_model_missing:{model_path}")
                classifier = _load_neural_classifier(str(model_path.resolve()))
            started = perf_counter()
            predictions = list(classifier.classify_batch(slot_images))
            neural_elapsed_ms = (perf_counter() - started) * 1000.0
            if len(predictions) != len(slots):
                raise ValueError("neural_prediction_count_mismatch")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            neural_error = str(exc)
            predictions = [None] * len(slots)
    cards: list[HandCard] = []
    rejected: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []
    hybrid_rows: list[dict[str, object]] = []
    for index, (slot, slot_image, neural) in enumerate(
        zip(slots, slot_images, predictions),
        start=1,
    ):
        if neural is None:
            name, confidence, template = classify_hand_slot(
                slot_image,
                templates,
            )
            scores: list[tuple[float, str, str]] = []
        else:
            scores = _candidate_scores(slot_image, templates)
            if scores:
                confidence, name, template = scores[0]
            else:
                confidence, name, template = 0.0, "", ""
        raw_confidence = confidence
        runner_up_name, runner_up_confidence = _runner_up_score(slot_image, templates, name)
        recognition_source = "template"
        hybrid_reason = ""
        allow_structural_recovery = False
        if neural is not None:
            hybrid = fuse_card_predictions(scores, neural)
            hybrid_reason = hybrid.reason
            hybrid_rows.append(
                {
                    "slot_index": index,
                    "template_label": name,
                    "template_confidence": round(float(raw_confidence), 4),
                    "neural_label": neural.label,
                    "neural_confidence": round(float(neural.confidence), 4),
                    "accepted": hybrid.accepted,
                    "hybrid_label": hybrid.label,
                    "reason": hybrid.reason,
                }
            )
            if resolved_hybrid_mode == "enforce":
                if not hybrid.accepted:
                    allow_structural_recovery = bool(
                        confidence < threshold
                        and hybrid.reason != "confident_disagreement"
                    )
                    if not allow_structural_recovery:
                        _save_unknown_crop(
                            slot_image,
                            unknown_crop_dir,
                            slot_index=index,
                            name=name,
                            confidence=confidence,
                        )
                        rejected.append(
                            {
                                "slot": slot,
                                "name": name,
                                "confidence": confidence,
                                "runner_up_name": runner_up_name,
                                "runner_up_confidence": runner_up_confidence,
                                "neural_label": neural.label,
                                "neural_confidence": neural.confidence,
                                "hybrid_reason": hybrid.reason,
                            }
                        )
                        debug_rows.append(
                            {
                                "slot": slot,
                                "name": "unknown",
                                "confidence": 0.0,
                                "template": hybrid.reason,
                                "accepted": False,
                            }
                        )
                        continue
                if hybrid.accepted:
                    name = hybrid.label
                    confidence = hybrid.confidence
                    template = f"hybrid:{hybrid.reason}"
                    recognition_source = "hybrid"
        if confidence < threshold and (
            resolved_hybrid_mode != "enforce" or allow_structural_recovery
        ):
            secondary = _secondary_low_confidence_acceptance(
                slot,
                slots,
                slot_image,
                templates,
                threshold=threshold,
            )
            if secondary is not None:
                name, confidence, template = secondary
                recognition_source = (
                    "edge_artifact_recovery"
                    if template.endswith(":edge_artifact_recovery")
                    else "secondary_grid_rule"
                )
            elif _accept_low_confidence_contour(
                slot,
                confidence,
                runner_up_name,
                runner_up_confidence,
            ):
                confidence = threshold
                template = f"{template}:contour_margin_recovery"
                recognition_source = "contour_margin_recovery"
            else:
                _save_unknown_crop(
                    slot_image,
                    unknown_crop_dir,
                    slot_index=index,
                    name=name,
                    confidence=confidence,
                )
                debug_rows.append(
                    {
                        "slot": slot,
                        "name": name or "unknown",
                        "confidence": confidence,
                        "template": template,
                        "accepted": False,
                    }
                )
                rejected.append(
                    {
                        "slot": slot,
                        "name": name,
                        "confidence": confidence,
                        "runner_up_name": runner_up_name,
                        "runner_up_confidence": runner_up_confidence,
                    }
                )
                continue
        debug_rows.append(
            {
                "slot": slot,
                "name": name,
                "confidence": confidence,
                "template": template,
                "accepted": True,
            }
        )
        cards.append(
            HandCard(
                name=name,
                confidence=confidence,
                x=slot.x,
                y=slot.y,
                w=slot.w,
                h=slot.h,
                template=template,
                clickable=_is_clickable_card(slot_image),
                raw_confidence=raw_confidence,
                runner_up_name=runner_up_name,
                runner_up_confidence=runner_up_confidence,
                recognition_source=recognition_source,
                neural_label=neural.label if neural is not None else "",
                neural_confidence=(
                    neural.confidence if neural is not None else 0.0
                ),
                hybrid_reason=hybrid_reason,
            )
        )
    cards = _remove_contained_duplicate_cards(cards)
    accepted_before_inference = len(cards)
    inferred_cards = _infer_occluded_column_cards(
        cards,
        rejected,
        expected_count=expected_hand_count,
    )
    cards = _remove_contained_duplicate_cards(cards + inferred_cards)
    cards = sorted(cards, key=lambda item: (item.x, item.y))
    result = [replace(card, card_id=f"h{index:03d}") for index, card in enumerate(cards, start=1)]
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "slot_count": len(slots),
                "raw_slot_count": raw_slot_count,
                "filtered_grid_slot_count": raw_slot_count - len(slots),
                "contour_slot_count": sum(slot.source == "contour" for slot in slots),
                "grid_fallback_slot_count": sum(slot.source == "grid_fallback" for slot in slots),
                "accepted_before_inference": accepted_before_inference,
                "accepted_count": len(result),
                "rejected_count": len(rejected),
                "inferred_count": len(inferred_cards),
                "nonclickable_count": sum(not card.clickable for card in result),
                "recognition_sources": dict(Counter(card.recognition_source for card in result)),
                "hybrid_mode": resolved_hybrid_mode,
                "neural_available": bool(hybrid_rows),
                "neural_error": neural_error,
                "neural_elapsed_ms": round(neural_elapsed_ms, 3),
                "hybrid_disagreements": sum(
                    row["reason"] == "confident_disagreement"
                    for row in hybrid_rows
                ),
                "hybrid_rows": hybrid_rows[:24],
                "rejected_candidates": [
                    {
                        "x": item["slot"].x,
                        "y": item["slot"].y,
                        "slot_source": item["slot"].source,
                        "best_label": item.get("name"),
                        "best_confidence": round(float(item.get("confidence") or 0.0), 4),
                        "runner_up_label": item.get("runner_up_name"),
                        "runner_up_confidence": round(
                            float(item.get("runner_up_confidence") or 0.0),
                            4,
                        ),
                    }
                    for item in rejected[:8]
                    if isinstance(item.get("slot"), CardSlot)
                ],
            }
        )
    _write_debug_hand(image, debug_path, debug_rows)
    return result


def _infer_occluded_column_cards(
    cards: list[HandCard],
    rejected: list[dict[str, object]],
    *,
    expected_count: int | None = None,
) -> list[HandCard]:
    result: list[HandCard] = []
    for item in rejected:
        if expected_count is not None and len(cards) + len(result) >= expected_count:
            break
        slot = item.get("slot")
        if not isinstance(slot, CardSlot):
            continue
        if _has_existing_card_near_slot(cards, slot):
            continue
        visible_cards = cards + result
        column_cards = [
            card
            for card in visible_cards
            if abs((card.x + card.w / 2) - (slot.x + slot.w / 2)) <= max(38, slot.w * 0.35)
            and abs(card.y - slot.y) <= 310
        ]
        label_counts: dict[str, int] = {}
        for card in column_cards:
            label_counts[card.name] = label_counts.get(card.name, 0) + 1
        label = next((name for name, count in label_counts.items() if count >= 2), None)
        raw_name = str(item.get("name") or "")
        raw_confidence = float(item.get("confidence") or 0.0)
        runner_up_name = str(item.get("runner_up_name") or "")
        runner_up_confidence = float(item.get("runner_up_confidence") or 0.0)
        single_anchor_expected_count = False
        if label is None:
            expected_shortage = (
                expected_count - len(cards)
                if expected_count is not None
                else 0
            )
            raw_margin = raw_confidence - runner_up_confidence
            matching_anchors = [
                card
                for card in column_cards
                if card.name == raw_name
                and 70 <= abs(card.y - slot.y) <= 170
            ]
            single_anchor_expected_count = bool(
                expected_shortage == 1
                and slot.source == "grid_fallback"
                and len(matching_anchors) == 1
                and raw_name
                and raw_confidence >= 0.54
                and raw_margin >= 0.08
            )
            if not single_anchor_expected_count:
                continue
            label = raw_name
        if label_counts[label] >= 4:
            continue
        same_label_cards = [card for card in column_cards if card.name == label]
        bracketed = (
            any(card.y < slot.y - 30 for card in same_label_cards)
            and any(card.y > slot.y + 30 for card in same_label_cards)
        )
        raw_label_supported = (
            raw_name == label
            and raw_confidence >= 0.40
            and raw_confidence - runner_up_confidence >= 0.04
        )
        runner_up_bracket_supported = (
            bracketed
            and runner_up_name == label
            and runner_up_confidence >= 0.38
            and raw_confidence - runner_up_confidence <= 0.08
        )
        if not (
            raw_label_supported
            or runner_up_bracket_supported
            or single_anchor_expected_count
        ):
            continue
        if (
            slot.source == "grid_fallback"
            and not runner_up_bracket_supported
            and not single_anchor_expected_count
            and not (
                raw_confidence >= 0.58
                and raw_confidence - runner_up_confidence >= 0.06
            )
        ):
            continue
        result.append(
            HandCard(
                name=label,
                confidence=0.73,
                x=slot.x,
                y=slot.y,
                w=slot.w,
                h=slot.h,
                template="occlusion_inferred_same_column",
                clickable=False,
                raw_confidence=raw_confidence,
                runner_up_name=runner_up_name,
                runner_up_confidence=runner_up_confidence,
                recognition_source=(
                    "stack_bracket_consensus"
                    if runner_up_bracket_supported
                    else (
                        "expected_count_single_anchor_consensus"
                        if single_anchor_expected_count
                        else "stack_consensus"
                    )
                ),
            )
        )
    return result


def _has_existing_card_near_slot(cards: list[HandCard], slot: CardSlot) -> bool:
    return any(abs(card.x - slot.x) <= 28 and abs(card.y - slot.y) <= 22 for card in cards)


def _save_unknown_crop(
    slot_image: np.ndarray,
    unknown_crop_dir: str | Path | None,
    *,
    slot_index: int,
    name: str,
    confidence: float,
) -> None:
    if unknown_crop_dir is None:
        return
    directory = Path(unknown_crop_dir)
    directory.mkdir(parents=True, exist_ok=True)
    label = _safe_filename_component(name or "unknown")
    score = max(0, min(999, int(round(confidence * 1000))))
    path = directory / f"unknown_{slot_index:03d}_{label}_{score:03d}.png"
    _write_image(path, slot_image)


def _write_debug_hand(image: np.ndarray, debug_path: str | Path | None, rows: list[dict[str, object]]) -> None:
    if debug_path is None:
        return
    output_path = Path(debug_path)
    debug = image.copy()
    for row in rows:
        slot = row["slot"]
        assert isinstance(slot, CardSlot)
        accepted = bool(row.get("accepted"))
        color = (0, 220, 0) if accepted else (0, 0, 255)
        cv2.rectangle(debug, (slot.x, slot.y), (slot.x + slot.w, slot.y + slot.h), color, 2)
        label = f"{row.get('name') or 'unknown'}:{float(row.get('confidence') or 0):.2f}"
        cv2.putText(
            debug,
            label,
            (slot.x, max(20, slot.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    _write_image(output_path, debug)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def _safe_filename_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe or "unknown"


def _remove_contained_duplicate_cards(cards: list[HandCard]) -> list[HandCard]:
    ranked = sorted(
        cards,
        key=lambda card: (
            card.confidence,
            card.recognition_source == "template",
            -(card.w * card.h),
        ),
        reverse=True,
    )
    kept: list[HandCard] = []
    for card in ranked:
        if any(other.name == card.name and _overlap_ratio(card, other) >= 0.72 for other in kept):
            continue
        kept.append(card)
    return kept


def _overlap_ratio(a: HandCard, b: HandCard) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    return intersection / max(1, min(a.w * a.h, b.w * b.h))


def recognize_hand_from_path(
    screenshot: str | Path,
    **kwargs,
) -> list[HandCard]:
    return recognize_hand(read_image(screenshot), **kwargs)
