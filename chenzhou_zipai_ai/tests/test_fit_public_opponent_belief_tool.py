from __future__ import annotations

from ai.opponent_belief import PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES
from tools.fit_public_opponent_belief import (
    BeliefCase,
    fit_weighted_ridge_model,
    focused_gate_failures,
    mean_confidence_interval,
)


def test_weighted_ridge_model_learns_separable_public_profiles() -> None:
    cases = []
    for profile, value in (("a", -1.0), ("b", 1.0)):
        for game in range(10):
            for _state in range(1 + game % 3):
                cases.append(
                    BeliefCase(
                        game_id=f"{profile}:{game}",
                        profile=profile,
                        fold=game % 5,
                        public_actions=6,
                        features=(value,) + (0.0,) * (
                            len(PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES) - 1
                        ),
                    )
                )
    model = fit_weighted_ridge_model(
        cases,
        profiles=("a", "b"),
        ridge=0.01,
        temperature=1.0,
        full_reliability_public_actions=6,
    )
    a = model.posterior(cases[0].features, public_actions=6)
    b = model.posterior(cases[-1].features, public_actions=6)
    assert a["a"] > a["b"]
    assert b["b"] > b["a"]


def test_gate_requires_strict_better_than_uniform_confidence_bounds() -> None:
    gate = {
        "required_profiles": 7,
        "required_games": 504,
        "minimum_states": 2500,
        "maximum_game_mean_log_loss_upper_95": 1.9459101490553132,
        "minimum_game_mean_true_probability_lower_95": 1 / 7,
        "minimum_game_weighted_top1_accuracy": 0.25,
        "maximum_low_evidence_posterior": 0.35,
        "require_zero_group_leakage": True,
        "require_all_profiles_in_every_fold": True,
    }
    failures = focused_gate_failures(
        gate=gate,
        profiles=7,
        games=504,
        states=3181,
        log_loss_upper_95=gate["maximum_game_mean_log_loss_upper_95"],
        true_probability_lower_95=gate[
            "minimum_game_mean_true_probability_lower_95"
        ],
        top1_accuracy=0.25,
        low_evidence_maximum=0.35,
        group_leakage=0,
        all_profiles_every_fold=True,
    )
    assert failures == [
        "game_mean_log_loss_upper_95",
        "game_mean_true_probability_lower_95",
    ]


def test_mean_confidence_interval_is_game_level() -> None:
    result = mean_confidence_interval([0.1, 0.2, 0.3, 0.4])
    assert result["n"] == 4
    assert result["mean"] == 0.25
    assert result["lower_95"] < result["mean"] < result["upper_95"]
