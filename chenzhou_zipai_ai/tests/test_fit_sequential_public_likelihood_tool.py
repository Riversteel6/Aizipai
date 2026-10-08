import math

from engine.cards import SMALL_LABELS
from tools.fit_sequential_public_likelihood import (
    fit_likelihood_model,
    normalize_log_scores,
    update_log_scores,
)


def _event(profile: str, label: str) -> dict:
    return {
        "profile": profile,
        "action_kind": "DISCARD",
        "action_token": f"DISCARD:{label}",
        "public_actions_before": 0,
    }


def test_sequential_likelihood_update_favors_matching_profile() -> None:
    first, second = SMALL_LABELS[:2]
    events = [
        _event("a", first),
        _event("a", first),
        _event("b", second),
        _event("b", second),
    ]
    model = fit_likelihood_model(events, profiles=("a", "b"), alpha=1.0)
    scores = update_log_scores(
        {"a": -math.log(2), "b": -math.log(2)},
        _event("a", first),
        model=model,
        profiles=("a", "b"),
    )
    posterior = normalize_log_scores(scores, profiles=("a", "b"))

    assert posterior["a"] > posterior["b"]
    assert abs(sum(posterior.values()) - 1.0) < 1e-12


def test_sequential_response_channel_is_separate_from_discard() -> None:
    event = {
        "profile": "a",
        "action_kind": "NO_CLAIM",
        "action_token": "NO_CLAIM",
        "public_actions_before": 4,
    }
    model = fit_likelihood_model([event], profiles=("a", "b"), alpha=1.0)

    scores = update_log_scores(
        {"a": 0.0, "b": 0.0},
        event,
        model=model,
        profiles=("a", "b"),
    )

    assert scores["a"] > scores["b"]
