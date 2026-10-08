"""Fuse independent template and neural card evidence without guessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vision.neural_card_classifier import NeuralCardPrediction


@dataclass(frozen=True)
class HybridCardDecision:
    label: str
    confidence: float
    accepted: bool
    reason: str


def fuse_card_predictions(
    template_scores: Sequence[tuple[float, str, str]],
    neural: NeuralCardPrediction,
) -> HybridCardDecision:
    if not template_scores:
        return HybridCardDecision("", 0.0, False, "template_missing")
    template_confidence, template_label, _ = template_scores[0]
    template_runner_up = next(
        (
            score
            for score, label, _template in template_scores[1:]
            if label != template_label
        ),
        0.0,
    )
    template_margin = template_confidence - template_runner_up
    neural_confident = neural.confidence >= 0.62 and neural.margin >= 0.12

    if neural_confident and neural.label == template_label:
        combined = 0.45 * template_confidence + 0.55 * neural.confidence
        partially_occluded_agreement = (
            template_confidence >= 0.52
            and template_margin >= 0.10
            and neural.confidence >= 0.90
        )
        return HybridCardDecision(
            template_label,
            float(combined),
            template_confidence >= 0.55 or partially_occluded_agreement,
            "independent_agreement",
        )
    if (
        neural_confident
        and template_confidence >= 0.70
        and template_margin >= 0.08
        and neural.label != template_label
    ):
        return HybridCardDecision("", 0.0, False, "confident_disagreement")
    if (
        neural.confidence >= 0.92
        and neural.margin >= 0.35
        and (template_confidence < 0.60 or template_margin < 0.05)
    ):
        return HybridCardDecision(
            neural.label,
            neural.confidence,
            True,
            "neural_rescue",
        )
    if template_confidence >= 0.82 and template_margin >= 0.14:
        return HybridCardDecision(
            template_label,
            template_confidence,
            True,
            "strong_template_neural_uncertain",
        )
    return HybridCardDecision("", 0.0, False, "insufficient_independent_evidence")


__all__ = ["HybridCardDecision", "fuse_card_predictions"]
