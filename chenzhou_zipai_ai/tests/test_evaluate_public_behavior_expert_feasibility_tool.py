from __future__ import annotations

import pytest

from engine.cards import RED_LABELS, SMALL_LABELS
from tools.evaluate_public_behavior_expert_feasibility import (
    BehaviorCase,
    expert_features,
    fit_multiclass_ridge,
    trigger_features,
)


def test_missing_trigger_has_no_card_information() -> None:
    assert trigger_features(None) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_trigger_features_use_only_public_card_properties() -> None:
    label = next(item for item in SMALL_LABELS if item in RED_LABELS)
    rank = SMALL_LABELS.index(label) + 1

    assert trigger_features(label) == (
        1.0,
        1.0,
        1.0,
        rank / 10.0,
        float(rank in {2, 7, 10}),
    )


def test_expert_features_activate_only_selected_profile_block() -> None:
    features = expert_features(
        (2.0, 3.0),
        "beta",
        profiles=("alpha", "beta", "gamma"),
    )

    assert features == (
        2.0,
        3.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        2.0,
        3.0,
        0.0,
        0.0,
    )


def test_multiclass_ridge_returns_normalized_fixed_vocabulary() -> None:
    cases = [
        BehaviorCase("g1", "alpha", 0, "response", "A", (0.0,)),
        BehaviorCase("g2", "alpha", 1, "response", "B", (1.0,)),
        BehaviorCase("g3", "alpha", 2, "response", "A", (0.1,)),
        BehaviorCase("g4", "alpha", 3, "response", "B", (0.9,)),
    ]
    model = fit_multiclass_ridge(
        cases,
        [case.features for case in cases],
        tokens=("A", "B"),
        ridge=1.0,
        temperature=1.0,
    )

    posterior = model.posterior((0.2,))

    assert set(posterior) == {"A", "B"}
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior["A"] > posterior["B"]

