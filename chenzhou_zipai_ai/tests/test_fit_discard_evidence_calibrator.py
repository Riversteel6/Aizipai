import pytest

from tools.fit_discard_evidence_calibrator import (
    CandidateExample,
    evidence_features,
    evidence_usable,
    fit_model,
    predict,
    select_final_fit_setting,
    select_setting,
    validation_fold_assignments,
)


def test_ridge_calibrator_learns_positive_evidence_direction() -> None:
    examples = [
        CandidateExample(
            state_id=str(index),
            label="B",
            features=(value, 0.1, 0.0, value, 0.0, 0.0, 1.0, 1.0),
            target=value * 0.5,
            weight=1.0,
        )
        for index, value in enumerate((-0.4, -0.2, 0.2, 0.4))
    ]

    model = fit_model(examples, ridge=0.1)
    negative = predict(
        model,
        (-0.3, 0.1, 0.0, -0.3, 0.0, 0.0, 1.0, 1.0),
    )
    positive = predict(
        model,
        (0.3, 0.1, 0.0, 0.3, 0.0, 0.0, 1.0, 1.0),
    )

    assert negative < 0.0
    assert positive > 0.0


def test_select_setting_enforces_empirical_harm_limit() -> None:
    settings = [
        {
            "ridge": 1.0,
            "margin": 0.03,
            "mean_improvement_over_preferred": 0.2,
            "mean_regret_to_pooled_best": 0.1,
            "empirically_negative_overrides": 1,
            "overrides": 5,
        },
        {
            "ridge": 10.0,
            "margin": 0.12,
            "mean_improvement_over_preferred": 0.1,
            "mean_regret_to_pooled_best": 0.2,
            "empirically_negative_overrides": 0,
            "overrides": 3,
        },
    ]

    selected = select_setting(
        settings,
        max_empirically_negative_overrides=0,
    )

    assert selected["margin"] == 0.12


def test_select_setting_rejects_impossible_empirical_harm_limit() -> None:
    with pytest.raises(
        ValueError,
        match="discard_calibrator_no_setting_satisfies_harm_limit",
    ):
        select_setting(
            [
                {
                    "ridge": 1.0,
                    "margin": 0.03,
                    "mean_improvement_over_preferred": 0.2,
                    "mean_regret_to_pooled_best": 0.1,
                    "empirically_negative_overrides": 1,
                    "overrides": 5,
                }
            ],
            max_empirically_negative_overrides=0,
        )


def test_select_final_fit_setting_enforces_final_model_harm_limit() -> None:
    settings = [
        {
            "ridge": 1.0,
            "margin": 0.1,
            "mean_improvement_over_preferred": 0.2,
            "mean_regret_to_pooled_best": 0.1,
            "empirically_negative_overrides": 0,
            "overrides": 5,
            "final_fit": {"empirically_negative_overrides": 1},
        },
        {
            "ridge": 1.0,
            "margin": 0.2,
            "mean_improvement_over_preferred": 0.1,
            "mean_regret_to_pooled_best": 0.2,
            "empirically_negative_overrides": 0,
            "overrides": 3,
            "final_fit": {"empirically_negative_overrides": 0},
        },
    ]

    selected = select_final_fit_setting(
        settings,
        max_empirically_negative_overrides=0,
    )

    assert selected["margin"] == 0.2


def test_evidence_usable_requires_valid_partial_confirmation() -> None:
    partial = {
        "complete": False,
        "structurally_valid": True,
        "confirmation_fraction": 0.92,
    }

    assert evidence_usable(
        partial,
        minimum_confirmation_fraction=0.9,
    )
    assert not evidence_usable(
        partial,
        minimum_confirmation_fraction=0.95,
    )
    assert not evidence_usable(
        {**partial, "structurally_valid": False},
        minimum_confirmation_fraction=0.9,
    )


def test_evidence_features_include_combined_opponent_interactions() -> None:
    evidence = {
        "combined_mean_delta": 0.25,
        "combined_standard_error": 0.1,
        "first_mean_delta": 0.2,
        "second_mean_delta": 0.3,
        "coverage_rank": 1,
        "heuristic_rank": 2,
        "confirmation_fraction": 1.0,
    }
    context = (0.4, 0.2, 0.0, 0.5, 0.0, 0.0, 0.3, 0.1)

    features = evidence_features(evidence, opponent_context=context)

    assert len(features) == 25
    assert features[9:17] == context
    assert features[17:] == pytest.approx(
        tuple(0.25 * value for value in context)
    )


def test_validation_folds_keep_linked_states_together() -> None:
    assignments = validation_fold_assignments(
        [
            {
                "state_before_hash": "state-a",
                "validation_group_id": "shared-group",
            },
            {
                "state_before_hash": "state-b",
                "validation_group_id": "shared-group",
            },
        ],
        folds=5,
    )

    assert assignments["state-a"] == assignments["state-b"]
