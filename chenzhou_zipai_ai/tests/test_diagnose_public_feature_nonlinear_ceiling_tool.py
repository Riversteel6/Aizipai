from tools.diagnose_public_feature_nonlinear_ceiling import (
    classify_feature_ceiling,
    reliability_blend,
    selected_probability_index,
)


THRESHOLDS = {
    "minimum_material_log_loss_improvement_lower_95": 0.05,
    "minimum_material_true_probability_improvement_lower_95": 0.01,
    "minimum_material_top1_improvement_lower_95": 0.05,
    "maximum_feature_insufficient_log_loss_improvement_upper_95": 0.05,
    "maximum_feature_insufficient_true_probability_improvement_upper_95": 0.01,
    "maximum_feature_insufficient_top1_improvement_upper_95": 0.05,
}


def test_reliability_blend_is_uniform_without_public_actions() -> None:
    result = reliability_blend(
        (0.9, 0.1),
        public_actions=0,
        full_reliability_public_actions=6,
    )

    assert tuple(result) == (0.5, 0.5)


def test_selected_probability_index_matches_lexical_tie_break() -> None:
    assert selected_probability_index((0.5, 0.5), ("a", "b")) == 1


def test_classify_feature_ceiling_supports_linear_underfit() -> None:
    result = classify_feature_ceiling(
        {
            "log_loss": {"lower_95": 0.06, "upper_95": 0.08},
            "true_probability": {"lower_95": 0.02, "upper_95": 0.03},
            "top1": {"lower_95": 0.07, "upper_95": 0.09},
        },
        thresholds=THRESHOLDS,
    )

    assert result == "linear_underfit_supported"


def test_classify_feature_ceiling_supports_feature_insufficiency() -> None:
    result = classify_feature_ceiling(
        {
            "log_loss": {"lower_95": -0.01, "upper_95": 0.04},
            "true_probability": {"lower_95": -0.01, "upper_95": 0.009},
            "top1": {"lower_95": -0.02, "upper_95": 0.04},
        },
        thresholds=THRESHOLDS,
    )

    assert result == "aggregate_feature_insufficiency_supported"
