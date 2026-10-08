"""Fast semantic relocation of one planned discard target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from engine.cards import normalize_card_label
from vision.hand_recognizer import (
    _candidate_scores,
    _is_clickable_card,
    _is_selected_card,
    _load_neural_classifier,
    detect_hand_slots,
    load_templates,
)
from vision.hybrid_card_classifier import fuse_card_predictions
from vision.neural_card_classifier import DEFAULT_MODEL_PATH, NeuralCardPrediction


@dataclass(frozen=True)
class TargetCandidate:
    label: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    clickable: bool
    accepted: bool
    reason: str
    selected: bool = False

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass(frozen=True, slots=True)
class _TargetSlotObservation:
    slot: object
    crop: np.ndarray
    clickable: bool
    selected: bool

    @property
    def classification_crop(self) -> np.ndarray:
        return _classification_crop(self.crop, selected=self.selected)


def choose_target_candidate(
    candidates: list[TargetCandidate],
    *,
    expected_label: str,
    original_x: int,
    original_y: int,
    original_tolerance: int = 18,
    prefer_selected: bool = False,
) -> dict:
    expected = normalize_card_label(expected_label)
    usable = [
        item
        for item in candidates
        if item.accepted and item.clickable and normalize_card_label(item.label) == expected
    ]
    if prefer_selected:
        selected_usable = [item for item in usable if item.selected]
        if selected_usable:
            usable = selected_usable
    if usable:
        chosen = min(
            usable,
            key=lambda item: (
                (item.center[0] - original_x) ** 2 + (item.center[1] - original_y) ** 2,
                -item.confidence,
            ),
        )
        moved = (
            abs(chosen.center[0] - original_x) > original_tolerance
            or abs(chosen.center[1] - original_y) > original_tolerance
        )
        return {
            "status": "relocated" if moved else "matched",
            "label": expected,
            "confidence": chosen.confidence,
            "x": chosen.center[0],
            "y": chosen.center[1],
            "reason": chosen.reason,
            "selected": chosen.selected,
        }
    uncertain = [
        item
        for item in candidates
        if item.clickable and normalize_card_label(item.label) == expected
    ]
    return {
        "status": "uncertain" if uncertain else "missing",
        "label": expected,
        "confidence": max((item.confidence for item in uncertain), default=0.0),
        "x": None,
        "y": None,
        "reason": "target_evidence_uncertain" if uncertain else "target_not_found",
        "selected": any(item.selected for item in uncertain),
    }


def locate_hand_target(
    image: np.ndarray,
    *,
    expected_label: str,
    original_x: int,
    original_y: int,
    config_path: str | Path,
    template_dir: str | Path,
    neural_model_path: str | Path = DEFAULT_MODEL_PATH,
    prefer_selected: bool = False,
) -> dict:
    started = perf_counter()
    slots = detect_hand_slots(image, config_path=config_path)
    height, width = image.shape[:2]
    slot_rows: list[_TargetSlotObservation] = []
    for slot in slots:
        crop = image[
            max(0, slot.y) : min(height, slot.y + slot.h),
            max(0, slot.x) : min(width, slot.x + slot.w),
        ]
        if not crop.size:
            continue
        slot_rows.append(
            _TargetSlotObservation(
                slot=slot,
                crop=crop,
                clickable=_is_clickable_card(crop),
                selected=_is_selected_card(crop),
            )
        )

    original_index = _nearest_slot_index(
        slot_rows,
        original_x=int(original_x),
        original_y=int(original_y),
    )
    if original_index is not None:
        original_row = slot_rows[original_index]
        original_prediction = _classify_slot_crops(
            [original_row.classification_crop],
            template_dir=template_dir,
            neural_model_path=neural_model_path,
        )[0]
        original_prediction = _selected_color_fallback_prediction(
            original_row,
            original_prediction,
            expected_label=expected_label,
            template_dir=template_dir,
            neural_model_path=neural_model_path,
        )
        original_candidate = _target_candidate(original_row, original_prediction)
        original_result = choose_target_candidate(
            [original_candidate],
            expected_label=expected_label,
            original_x=int(original_x),
            original_y=int(original_y),
            prefer_selected=prefer_selected,
        )
        if original_result["status"] in {"matched", "relocated"} and (
            not prefer_selected or bool(original_result.get("selected"))
        ):
            original_result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            original_result["slot_count"] = len(slot_rows)
            original_result["classified_slot_count"] = 1
            return original_result

    fallback_rows = [row for index, row in enumerate(slot_rows) if index != original_index]
    predictions = _classify_slot_crops(
        [row.classification_crop for row in fallback_rows],
        template_dir=template_dir,
        neural_model_path=neural_model_path,
    )
    predictions = _with_selected_color_fallbacks(
        fallback_rows,
        predictions,
        expected_label=expected_label,
        template_dir=template_dir,
        neural_model_path=neural_model_path,
    )
    candidates = [_target_candidate(row, prediction) for row, prediction in zip(fallback_rows, predictions)]
    if original_index is not None:
        candidates.append(original_candidate)
    result = choose_target_candidate(
        candidates,
        expected_label=expected_label,
        original_x=int(original_x),
        original_y=int(original_y),
        prefer_selected=prefer_selected,
    )
    result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
    result["slot_count"] = len(slot_rows)
    result["classified_slot_count"] = len(slot_rows)
    return result


def _classification_crop(crop: np.ndarray, *, selected: bool) -> np.ndarray:
    if not selected or crop.ndim != 3 or crop.shape[2] < 3:
        return crop
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _selected_color_fallback_prediction(
    row: _TargetSlotObservation,
    prediction: tuple[str, float, bool, str],
    *,
    expected_label: str,
    template_dir: str | Path,
    neural_model_path: str | Path,
) -> tuple[str, float, bool, str]:
    """Retry one rejected selected crop in color without weakening acceptance.

    Grayscale remains the established first path because it removes the pink
    selection overlay for most cards.  Some red glyphs lose discriminating
    color information in that conversion, though.  Only when the independent
    fusion rejects grayscale do we retry the same, already-selected crop in
    color, and only an independently accepted match for the planned semantic
    label may replace the rejected result.
    """

    if not row.selected or prediction[2]:
        return prediction
    color_prediction = _classify_slot_crops(
        [row.crop],
        template_dir=template_dir,
        neural_model_path=neural_model_path,
    )[0]
    if not color_prediction[2]:
        return prediction
    if normalize_card_label(color_prediction[0]) != normalize_card_label(expected_label):
        return prediction
    return (
        color_prediction[0],
        color_prediction[1],
        color_prediction[2],
        f"selected_color_fallback:{color_prediction[3]}",
    )


def _with_selected_color_fallbacks(
    rows: list[_TargetSlotObservation],
    predictions: list[tuple[str, float, bool, str]],
    *,
    expected_label: str,
    template_dir: str | Path,
    neural_model_path: str | Path,
) -> list[tuple[str, float, bool, str]]:
    """Batch the exceptional color retry for relocated selected targets."""

    retry_indexes = [
        index
        for index, (row, prediction) in enumerate(zip(rows, predictions))
        if row.selected and not prediction[2]
    ]
    if not retry_indexes:
        return predictions
    color_predictions = _classify_slot_crops(
        [rows[index].crop for index in retry_indexes],
        template_dir=template_dir,
        neural_model_path=neural_model_path,
    )
    output = list(predictions)
    expected = normalize_card_label(expected_label)
    for index, color_prediction in zip(retry_indexes, color_predictions):
        if color_prediction[2] and normalize_card_label(color_prediction[0]) == expected:
            output[index] = (
                color_prediction[0],
                color_prediction[1],
                color_prediction[2],
                f"selected_color_fallback:{color_prediction[3]}",
            )
    return output


def _nearest_slot_index(
    rows: list[_TargetSlotObservation],
    *,
    original_x: int,
    original_y: int,
) -> int | None:
    if not rows:
        return None
    index, row = min(
        enumerate(rows),
        key=lambda item: (
            (int(item[1].slot.x) + int(item[1].slot.w) // 2 - original_x) ** 2
            + (int(item[1].slot.y) + int(item[1].slot.h) // 2 - original_y) ** 2
        ),
    )
    slot = row.slot
    center_x = int(slot.x) + int(slot.w) // 2
    center_y = int(slot.y) + int(slot.h) // 2
    return index if abs(center_x - original_x) <= 24 and abs(center_y - original_y) <= 24 else None


def _target_candidate(
    row: _TargetSlotObservation,
    prediction: tuple[str, float, bool, str],
) -> TargetCandidate:
    slot = row.slot
    return TargetCandidate(
        label=prediction[0],
        confidence=prediction[1],
        x=int(slot.x),
        y=int(slot.y),
        w=int(slot.w),
        h=int(slot.h),
        clickable=row.clickable,
        accepted=prediction[2],
        reason=prediction[3],
        selected=row.selected,
    )


def _classify_slot_crops(
    crops: list[np.ndarray],
    *,
    template_dir: str | Path,
    neural_model_path: str | Path,
) -> list[tuple[str, float, bool, str]]:
    templates = load_templates(template_dir, grayscale=False)
    template_scores = [_candidate_scores(crop, templates) for crop in crops]
    neural: list[NeuralCardPrediction | None] = [None] * len(crops)
    model_path = Path(neural_model_path)
    if crops and model_path.exists():
        try:
            neural = list(_load_neural_classifier(str(model_path.resolve())).classify_batch(crops))
            if len(neural) != len(crops):
                raise ValueError("neural_prediction_count_mismatch")
        except (OSError, RuntimeError, ValueError):
            neural = [None] * len(crops)

    output: list[tuple[str, float, bool, str]] = []
    for scores, prediction in zip(template_scores, neural):
        if prediction is not None:
            fused = fuse_card_predictions(scores, prediction)
            output.append(
                (
                    normalize_card_label(fused.label),
                    float(fused.confidence),
                    bool(fused.accepted),
                    fused.reason,
                )
            )
            continue
        if not scores:
            output.append(("", 0.0, False, "template_missing"))
            continue
        confidence, label, _template = scores[0]
        runner_up = next((score for score, other, _ in scores[1:] if other != label), 0.0)
        accepted = confidence >= 0.82 and confidence - runner_up >= 0.14
        output.append(
            (
                normalize_card_label(label),
                float(confidence),
                accepted,
                "strong_template" if accepted else "template_uncertain",
            )
        )
    return output


__all__ = ["TargetCandidate", "choose_target_candidate", "locate_hand_target"]
