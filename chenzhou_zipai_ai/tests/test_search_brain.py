"""Shadow-first production search wrapper tests."""

from __future__ import annotations

import random

from ai.ismcts import RootISMCTSConfig
from ai.search_brain import (
    SearchAugmentationConfig,
    choose_action_with_search,
    public_view_from_production_state,
)
from engine.deck import expanded_deck, full_deck_counts
from engine.rules import rules_for_room


WILD_CONTINUE_HAND = [
    "王", "王", "贰", "柒", "壹", "贰", "叁", "九", "九", "九", "二", "七",
    "四", "五", "六", "七", "八", "九", "肆", "伍", "陆",
]


def test_shadow_search_never_changes_production_action():
    rules = rules_for_room(wildcard_enabled=False)
    deck = expanded_deck(full_deck_counts(rules=rules))
    random.Random(20260729).shuffle(deck)
    state = {
        "hand": deck[:21],
        "legal_actions": [{"type": "DISCARD"}],
        "remaining_deck_count": 19,
        "memory": {
            "my_discards": [],
            "opponent_discards": [],
            "opponent_meld_groups": [],
        },
    }
    config = SearchAugmentationConfig(
        shadow_only=True,
        activation_ev_gap=100_000,
        max_candidates=2,
        root=RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
            skip_search_gap=100_000,
            rollout_max_turns=80,
        ),
    )

    augmented = choose_action_with_search(
        state,
        rules=rules,
        search_config=config,
    )

    assert augmented.search_attempted
    assert not augmented.search_applied
    assert augmented.reason == "shadow_only"
    assert augmented.search_result is not None
    assert augmented.decision.selected_action == augmented.production_decision.selected_action
    assert augmented.decision.selected_label == augmented.production_decision.selected_label


def test_heads_up_shadow_search_builds_a_two_player_information_set():
    rules = rules_for_room(wildcard_enabled=True, players=2)
    deck = expanded_deck(full_deck_counts(rules=rules))
    random.Random(20260730).shuffle(deck)
    state = {
        "hand": deck[:21],
        "legal_actions": [{"type": "DISCARD"}],
        "remaining_deck_count": 43,
        "room_players": 2,
        "memory": {
            "my_discards": [],
            "opponent_discards": [],
            "opponent_meld_groups": [],
            "my_passed_chi": ["七"],
            "my_passed_peng": ["陆"],
            "opponent_passed_chi": ["二"],
            "opponent_passed_peng": ["玖"],
        },
    }
    config = SearchAugmentationConfig(
        shadow_only=True,
        activation_ev_gap=100_000,
        max_candidates=2,
        root=RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=1,
            max_candidates=2,
            skip_search_gap=100_000,
            rollout_max_turns=80,
        ),
    )

    augmented = choose_action_with_search(
        state,
        rules=rules,
        search_config=config,
    )

    assert augmented.search_attempted
    assert augmented.search_result is not None
    assert augmented.search_result.determinization_failures == 0
    assert augmented.decision.selected_label == augmented.production_decision.selected_label
    view = public_view_from_production_state(
        state,
        augmented.production_decision,
        rules=rules,
    )
    assert view is not None
    assert view.passed_chi == (("七",), ("二",))
    assert view.passed_peng == (("陆",), ("玖",))


def test_response_shadow_search_compares_exact_chi_and_pass_without_promotion():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "七",
        "chi_options": [
            {
                "option_id": "chi_2710",
                "labels": ["二", "七", "十"],
                "confidence": 1.0,
            }
        ],
        "remaining_deck_count": 64,
        "metadata": {
            "self_chair_id": 0,
            "last_turn_event": {
                "type": "discard",
                "seat": 2,
                "cards": ["七"],
            },
        },
        "memory": {
            "my_discards": [],
            "opponent_discards": [],
            "opponent_meld_groups": [],
        },
    }
    config = SearchAugmentationConfig(
        shadow_only=True,
        max_response_candidates=2,
        root=RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
            rollout_max_turns=8,
        ),
    )

    augmented = choose_action_with_search(
        state,
        rules=rules,
        search_config=config,
    )

    assert augmented.production_decision.selected_action == "CHI"
    assert augmented.search_attempted
    assert not augmented.search_applied
    assert augmented.reason == "response_shadow_only"
    assert augmented.search_result is not None
    assert {item.candidate.action_type for item in augmented.search_result.candidates} == {
        "CHI",
        "PASS",
    }
    assert augmented.decision == augmented.production_decision


def test_three_player_peng_search_skips_without_proven_turn_direction():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    state = {
        "hand": ["二", "二", "四", "六", "九"],
        "legal_actions": [{"type": "PENG"}, {"type": "PASS"}],
        "pending_card": "二",
        "remaining_deck_count": 64,
    }

    augmented = choose_action_with_search(
        state,
        rules=rules,
        search_config=SearchAugmentationConfig(shadow_only=True),
    )

    assert augmented.production_decision.selected_action == "PENG"
    assert not augmented.search_attempted
    assert augmented.reason == "insufficient_response_public_state"
    assert augmented.decision == augmented.production_decision


def test_hu_keeps_hard_priority_and_never_enters_response_search():
    rules = rules_for_room(wildcard_enabled=True, players=3)
    state = {
        "hand": WILD_CONTINUE_HAND,
        "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
        "remaining_deck_count": 8,
    }

    augmented = choose_action_with_search(
        state,
        rules=rules,
        search_config=SearchAugmentationConfig(shadow_only=False),
    )

    assert augmented.production_decision.selected_action == "HU"
    assert not augmented.search_attempted
    assert not augmented.search_applied
    assert augmented.reason == "terminal_hu_keeps_hard_priority"
    assert augmented.decision == augmented.production_decision
