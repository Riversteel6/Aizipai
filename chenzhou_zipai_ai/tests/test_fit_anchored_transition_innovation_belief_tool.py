from __future__ import annotations

import pytest

from engine.cards import RED_LABELS, SMALL_LABELS
from tools.fit_anchored_transition_innovation_belief import (
    fit_transition_innovation_model,
    previous_event_class,
    update_transition_innovation_scores,
)


PROFILES = ("alpha", "beta")


def response_event(
    game_id: str,
    order: int,
    kind: str,
    *,
    profile: str = "alpha",
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "source_sequence": order + 1,
        "event_order": order,
        "action_kind": kind,
        "action_token": kind,
        "public_actions_before": order,
        "profile": profile,
    }


def test_previous_event_class_uses_observer_visible_card_groups() -> None:
    label = next(item for item in SMALL_LABELS if item in RED_LABELS)
    event = response_event("game-1", 0, "DISCARD")
    event["action_token"] = f"DISCARD:{label}"

    assert previous_event_class(event) == "DISCARD_SMALL_RED"


def test_first_event_adds_no_transition_innovation() -> None:
    scores = {"alpha": 0.25, "beta": -0.5}

    updated = update_transition_innovation_scores(
        scores,
        response_event("game-1", 0, "CHI"),
        None,
        model={"alpha": 1.0},
        profiles=PROFILES,
    )

    assert updated == scores


def test_equal_transition_and_marginal_probabilities_add_zero() -> None:
    model = {
        "alpha": 1.0,
        "marginal_counts": {
            profile: {"response|0_to_2|NO_CLAIM": 1} for profile in PROFILES
        },
        "marginal_totals": {
            profile: {"response|0_to_2": 2} for profile in PROFILES
        },
        "transition_counts": {
            profile: {"response|0_to_2|CHI|NO_CLAIM": 1}
            for profile in PROFILES
        },
        "transition_totals": {
            profile: {"response|0_to_2|CHI": 2} for profile in PROFILES
        },
    }

    updated = update_transition_innovation_scores(
        {profile: 0.0 for profile in PROFILES},
        response_event("game-1", 1, "NO_CLAIM"),
        response_event("game-1", 0, "CHI"),
        model=model,
        profiles=PROFILES,
    )

    assert updated == pytest.approx({profile: 0.0 for profile in PROFILES})


def test_transition_model_resets_previous_event_at_game_boundary() -> None:
    events = [
        response_event("game-1", 0, "CHI"),
        response_event("game-1", 1, "NO_CLAIM"),
        response_event("game-2", 0, "PENG"),
        response_event("game-2", 1, "NO_CLAIM"),
    ]

    model = fit_transition_innovation_model(
        events,
        profiles=PROFILES,
        alpha=1.0,
    )

    assert model["games"] == 2
    assert model["events"] == 4
    assert model["transition_events"] == 2

