from vision.hybrid_card_classifier import fuse_card_predictions
from vision.neural_card_classifier import NeuralCardPrediction


def neural(
    label: str,
    confidence: float,
    runner_up_confidence: float,
) -> NeuralCardPrediction:
    return NeuralCardPrediction(
        label=label,
        confidence=confidence,
        runner_up_label="二",
        runner_up_confidence=runner_up_confidence,
    )


def test_independent_agreement_accepts_a_card() -> None:
    decision = fuse_card_predictions(
        [(0.72, "一", "a.png"), (0.61, "二", "b.png")],
        neural("一", 0.88, 0.08),
    )

    assert decision.accepted
    assert decision.label == "一"
    assert decision.reason == "independent_agreement"


def test_partial_occlusion_accepts_high_confidence_independent_agreement() -> None:
    decision = fuse_card_predictions(
        [(0.5446, "八", "a.png"), (0.4146, "九", "b.png")],
        neural("八", 0.9502, 0.10),
    )

    assert decision.accepted
    assert decision.label == "八"
    assert decision.reason == "independent_agreement"


def test_partial_occlusion_still_rejects_weak_neural_agreement() -> None:
    decision = fuse_card_predictions(
        [(0.5446, "八", "a.png"), (0.4146, "九", "b.png")],
        neural("八", 0.70, 0.10),
    )

    assert not decision.accepted


def test_confident_disagreement_returns_unknown_instead_of_guessing() -> None:
    decision = fuse_card_predictions(
        [(0.84, "一", "a.png"), (0.65, "二", "b.png")],
        neural("三", 0.90, 0.05),
    )

    assert not decision.accepted
    assert decision.label == ""
    assert decision.reason == "confident_disagreement"


def test_strong_neural_prediction_can_rescue_weak_template_evidence() -> None:
    decision = fuse_card_predictions(
        [(0.55, "一", "a.png"), (0.53, "二", "b.png")],
        neural("三", 0.95, 0.05),
    )

    assert decision.accepted
    assert decision.label == "三"
    assert decision.reason == "neural_rescue"


def test_two_weak_branches_return_unknown() -> None:
    decision = fuse_card_predictions(
        [(0.68, "一", "a.png"), (0.62, "二", "b.png")],
        neural("一", 0.58, 0.22),
    )

    assert not decision.accepted
    assert decision.reason == "insufficient_independent_evidence"
