from __future__ import annotations

from tools.evaluate_opponent_distribution_split import (
    _canonical_opponent_name,
    evaluate_selection,
    focused_gate_failures,
    heldout_truth,
    select_challengers,
)


def test_canonical_opponent_name_accepts_versioned_search_policy() -> None:
    assert (
        _canonical_opponent_name("information_set_search_v2")
        == "information_set_search"
    )
    assert _canonical_opponent_name("red_black_search") == "red_black_search"


def test_select_challengers_reserves_prior_slots() -> None:
    selected = select_challengers(
        ("A", "B", "P", "C", "D", "E"),
        preferred_label="P",
        heuristics={
            "A": 1.0,
            "B": 2.0,
            "P": 3.0,
            "C": 4.0,
            "D": 100.0,
            "E": 90.0,
        },
        coverage_means={
            "A": 6.0,
            "B": 5.0,
            "P": 4.0,
            "C": 3.0,
            "D": 2.0,
            "E": 1.0,
        },
        limit=5,
        prior_slots=2,
    )

    assert selected == ("A", "B", "C", "D", "E")


def test_heldout_truth_and_selection_classification() -> None:
    worlds = [
        {"P": 0.0, "A": 1.0, "B": -1.0},
        {"P": 0.0, "A": 1.0, "B": -1.0},
        {"P": 0.0, "A": 1.0, "B": -1.0},
    ] * 20
    truth = heldout_truth(
        worlds,
        labels=("P", "A", "B"),
        preferred="P",
    )

    beneficial = evaluate_selection("A", preferred="P", truth=truth)
    harmful = evaluate_selection("B", preferred="P", truth=truth)

    assert truth["best_label"] == "A"
    assert beneficial["classification"] == "correct_override"
    assert harmful["classification"] == "harmful_override"


def test_focused_gate_requires_strict_harm_reduction() -> None:
    gate = {
        "states": 2,
        "maximum_heldout_harmful_overrides": 1,
        "maximum_confidently_negative_overrides": 0,
        "minimum_preserved_confidently_positive_overrides": 1,
        "minimum_mean_improvement_over_preferred": 0.01,
        "maximum_mean_regret_to_heldout_best": 0.1,
        "require_zero_search_health_errors": True,
    }
    candidate = {
        "harmful_overrides": 1,
        "confidently_negative_overrides": 0,
        "confidently_positive_overrides": 1,
        "mean_improvement_over_preferred": 0.02,
        "mean_regret_to_heldout_best": 0.05,
    }
    anchor = {"harmful_overrides": 1}

    failures = focused_gate_failures(
        candidate,
        anchor=anchor,
        gate=gate,
        health_errors=0,
        states=2,
    )

    assert failures == ["harm_not_strictly_better_than_anchor:1>=1"]
