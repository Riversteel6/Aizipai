"""Recognize card labels inside chi/compare option columns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.cards import normalize_card_label
from vision.hand_recognizer import _candidate_scores, _load_neural_classifier
from vision.neural_card_classifier import DEFAULT_MODEL_PATH, NeuralCardPrediction
from vision.option_detector import OptionCandidate, detect_option_candidates
from vision.template_loader import TemplateImage, load_templates_from_dirs


@dataclass(frozen=True)
class RecognizedOption:
    region_name: str
    index: int
    labels: list[str]
    confidence: float
    x: int
    y: int
    w: int
    h: int
    card_count: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> dict:
        return {
            "region_name": self.region_name,
            "index": self.index,
            "labels": self.labels,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "card_count": self.card_count,
            "center": self.center,
        }


def load_option_templates(template_dirs: tuple[str | Path, ...]) -> list[TemplateImage]:
    return load_templates_from_dirs(template_dirs, grayscale=False)


def recognize_options(
    image,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    template_dirs: tuple[str | Path, ...] = (
        "data/templates/option_auto",
        "data/templates/discard_auto",
        "data/templates/meld_auto",
        "data/templates/hand_auto",
    ),
    threshold: float = 0.35,
) -> list[RecognizedOption]:
    templates = load_option_templates(template_dirs)
    candidates = detect_option_candidates(image, config_path=config_path)
    cells_by_candidate = [_candidate_cells(image, candidate) for candidate in candidates]
    all_cells = [cell for cells in cells_by_candidate for cell in cells]
    predictions = _classify_option_cells(all_cells)
    recognized: list[RecognizedOption] = []
    prediction_offset = 0
    for candidate, cells in zip(candidates, cells_by_candidate):
        candidate_predictions = predictions[
            prediction_offset : prediction_offset + len(cells)
        ]
        prediction_offset += len(cells)
        option = _recognize_candidate(
            candidate,
            cells,
            candidate_predictions,
            templates,
            threshold=threshold,
        )
        if option is not None:
            recognized.append(option)
    return _dedupe_options(recognized)


def _candidate_cells(image, candidate: OptionCandidate) -> list:
    if len(candidate.card_boxes) >= candidate.card_count:
        cells = []
        height, width = image.shape[:2]
        for x, y, w, h in candidate.card_boxes[: candidate.card_count]:
            inset = max(3, round(min(w, h) * 0.06))
            x1 = min(width, max(0, x + inset))
            y1 = min(height, max(0, y + inset))
            x2 = min(width, max(x1 + 1, x + w - inset))
            y2 = min(height, max(y1 + 1, y + h - inset))
            cells.append(image[y1:y2, x1:x2])
        return cells

    cells = []
    card_h = candidate.h / max(1, candidate.card_count)
    for index in range(candidate.card_count):
        y1 = round(candidate.y + index * card_h)
        y2 = round(candidate.y + (index + 1) * card_h)
        x1 = candidate.x + 4
        x2 = candidate.x + candidate.w - 10
        cells.append(image[y1:y2, x1:x2])
    return cells


def _classify_option_cells(cells: list) -> list[NeuralCardPrediction | None]:
    if not cells or not DEFAULT_MODEL_PATH.exists():
        return [None] * len(cells)
    try:
        return list(_load_neural_classifier(str(DEFAULT_MODEL_PATH)).classify_batch(cells))
    except (OSError, RuntimeError, ValueError):
        return [None] * len(cells)


def _recognize_candidate(
    candidate: OptionCandidate,
    cells: list,
    predictions: list[NeuralCardPrediction | None],
    templates,
    *,
    threshold: float,
) -> RecognizedOption | None:
    labels: list[str] = []
    confidences: list[float] = []
    if len(cells) != candidate.card_count:
        return None
    for cell, prediction in zip(cells, predictions):
        resolved = _resolve_option_cell(_candidate_scores(cell, templates), prediction)
        if resolved is None or resolved[1] < threshold:
            return None
        name, confidence, _source = resolved
        labels.append(normalize_card_label(name))
        confidences.append(confidence)
    labels = _normalize_labels(_correct_option_labels(labels))
    return RecognizedOption(
        region_name=candidate.region_name,
        index=candidate.index,
        labels=labels,
        confidence=round(sum(confidences) / len(confidences), 3),
        x=candidate.x,
        y=candidate.y,
        w=candidate.w,
        h=candidate.h,
        card_count=candidate.card_count,
    )


def _resolve_option_cell(
    scores: list[tuple[float, str, str]],
    prediction: NeuralCardPrediction | None,
) -> tuple[str, float, str] | None:
    normalized_scores = [
        (float(score), normalize_card_label(name), template)
        for score, name, template in scores
    ]
    template_label = normalized_scores[0][1] if normalized_scores else ""
    template_confidence = normalized_scores[0][0] if normalized_scores else 0.0
    template_runner_up = next(
        (
            score
            for score, label, _template in normalized_scores[1:]
            if label != template_label
        ),
        0.0,
    )
    template_margin = template_confidence - template_runner_up

    neural_label = normalize_card_label(prediction.label) if prediction else ""
    neural_confidence = float(prediction.confidence) if prediction else 0.0
    neural_margin = float(prediction.margin) if prediction else 0.0

    if neural_label and neural_confidence >= 0.42 and neural_margin >= 0.24:
        calibrated = max(neural_confidence, min(0.95, 0.62 + 0.5 * neural_margin))
        return neural_label, calibrated, "neural_clear_margin"
    if (
        neural_label
        and neural_label == template_label
        and neural_confidence >= 0.28
        and neural_margin >= 0.10
        and template_confidence >= 0.60
    ):
        return neural_label, max(neural_confidence, template_confidence), "agreement"
    if template_label and template_confidence >= 0.70 and template_margin >= 0.08:
        return template_label, template_confidence, "template_clear_margin"
    return None


def _correct_option_labels(labels: list[str]) -> list[str]:
    return [normalize_card_label(item) for item in labels]


def _normalize_labels(labels: list[str]) -> list[str]:
    return [normalize_card_label(item) for item in labels]


def _dedupe_options(options: list[RecognizedOption]) -> list[RecognizedOption]:
    result: list[RecognizedOption] = []
    seen: set[tuple[int, int, tuple[str, ...]]] = set()
    for option in sorted(options, key=lambda item: (item.x, item.y, item.region_name)):
        key = (round(option.x / 10), round(option.y / 10), tuple(option.labels))
        if key in seen:
            continue
        seen.add(key)
        result.append(option)
    return result
