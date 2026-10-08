from __future__ import annotations

import math

import pytest

from ai.opponent_belief import (
    PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES,
    PublicOpponentBeliefModel,
    public_opponent_features,
    smooth_weighted_profile_schedule,
)
from ai.opponent_belief_runtime import runtime_opponent_rollout_schedule
from ai.opponent_league import create_policy
from ai.ismcts import public_view_from_dict
from engine.rules import rules_for_room


def test_public_opponent_features_follow_registered_schema() -> None:
    view = {
        "seat": 0,
        "stock_count": 30,
        "hand_sizes": [18, 14],
        "all_melds": [
            [],
            [
                {"type": "peng", "labels": ["二", "二", "二"]},
                {"type": "special_2710", "labels": ["贰", "柒", "拾"]},
            ],
        ],
        "discards": [[], ["壹", "贰", "叁", "四"]],
    }
    evidence = public_opponent_features(
        view,
        rules_for_room(wildcard_enabled=False, players=2),
    )
    values = dict(zip(PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES, evidence.features))

    assert evidence.opponent_seat == 1
    assert evidence.public_actions == 6
    assert values["discard_count_scaled"] == pytest.approx(0.2)
    assert values["meld_count_scaled"] == pytest.approx(2 / 6)
    assert values["claim_share"] == pytest.approx(2 / 6)
    assert values["peng_share"] == pytest.approx(0.5)
    assert values["special_chi_share"] == pytest.approx(0.5)
    assert values["discard_big_share"] == pytest.approx(0.75)
    assert values["discard_2710_share"] == pytest.approx(0.25)
    assert values["opponent_hand_size_scaled"] == pytest.approx(0.7)
    assert values["opponent_meld_cards_scaled"] == pytest.approx(0.3)
    assert all(math.isfinite(value) for value in evidence.features)


def test_public_opponent_features_requires_seat_for_three_players() -> None:
    view = {
        "seat": 0,
        "stock_count": 20,
        "hand_sizes": [15, 15, 15],
        "all_melds": [[], [], []],
        "discards": [[], [], []],
    }
    rules = rules_for_room(wildcard_enabled=False, players=3)
    with pytest.raises(ValueError, match="public_opponent_seat_required"):
        public_opponent_features(view, rules)
    assert public_opponent_features(view, rules, opponent_seat=2).opponent_seat == 2


def test_public_opponent_belief_reliability_blends_with_uniform() -> None:
    feature_count = len(PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES)
    model = PublicOpponentBeliefModel(
        profile_names=("a", "b"),
        feature_names=PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES,
        coefficients=tuple((8.0, -8.0) for _ in range(feature_count)),
        intercepts=(0.0, 0.0),
        feature_means=(0.0,) * feature_count,
        feature_scales=(1.0,) * feature_count,
        full_reliability_public_actions=6,
    )
    features = (1.0,) * feature_count

    assert model.posterior(features, public_actions=0) == {"a": 0.5, "b": 0.5}
    partial = model.posterior(features, public_actions=1)
    assert partial["a"] < 0.59
    assert partial["a"] > 0.5
    assert sum(partial.values()) == pytest.approx(1.0)
    full = model.posterior(features, public_actions=6)
    assert full["a"] > 0.999


def test_smooth_weighted_profile_schedule_is_apportioned_and_deterministic() -> None:
    posterior = {"a": 0.5, "b": 0.3, "c": 0.2}
    first = smooth_weighted_profile_schedule(posterior, slots=10)
    second = smooth_weighted_profile_schedule(posterior, slots=10)

    assert first == second
    assert {name: first.count(name) for name in posterior} == {
        "a": 5,
        "b": 3,
        "c": 2,
    }
    assert len(set(first[:3])) >= 2


def test_runtime_belief_uses_public_actions_but_ignores_hidden_state_fields() -> None:
    rules = rules_for_room(wildcard_enabled=True, players=2)
    public_view = {
        "seat": 0,
        "stock_count": 30,
        "hand_sizes": [18, 14],
        "all_melds": [[], [{"type": "peng", "labels": ["二"] * 3}]],
        "discards": [[], ["壹", "贰", "叁", "四"]],
    }
    factories, belief = runtime_opponent_rollout_schedule(
        public_view,
        rules,
        slots=21,
    )
    hidden_variant = {
        **public_view,
        "opponent_hidden_hand": ["王", "王", "王"],
        "remaining_counts": [("王", 3)],
    }
    _hidden_factories, hidden_belief = runtime_opponent_rollout_schedule(
        hidden_variant,
        rules,
        slots=21,
    )

    assert len(factories) == 21
    assert sum(belief.posterior.values()) == pytest.approx(1.0)
    assert belief.public_actions == 5
    assert belief.posterior == hidden_belief.posterior
    assert belief.schedule == hidden_belief.schedule
    assert not belief.to_dict()["hidden_information_used"]


def test_frozen_candidate_installs_public_belief_schedule_on_every_search_stage() -> None:
    rules = rules_for_room(wildcard_enabled=True, players=2)
    view = public_view_from_dict(
        {
            "seat": 0,
            "hand": ["一", "二", "三", "四", "五", "六"],
            "own_melds": [],
            "all_melds": [[], [{"type": "peng", "labels": ["柒"] * 3}]],
            "discards": [[], ["贰", "柒", "拾"]],
            "remaining_counts": [],
            "stock_count": 20,
            "hand_sizes": [6, 12],
        }
    )
    policy = create_policy(
        "professional_v81_two_player_exact_discard_sharded_research"
    )

    policy._configure_opponent_belief(view, rules)

    assert policy.last_opponent_belief is not None
    assert policy.last_opponent_belief.public_actions == 4
    expected = policy.last_opponent_belief.schedule
    actual = tuple(
        factory().name
        for factory in policy.search.coverage.rollout_policy_factories
    )
    assert len(actual) == len(expected) == 112
    assert policy.discard_validator.rollout_policy_factories == (
        policy.search.coverage.rollout_policy_factories
    )
