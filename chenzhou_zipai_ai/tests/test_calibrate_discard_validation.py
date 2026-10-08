from pathlib import Path

import pytest
import tools.calibrate_discard_validation as calibration_module
from tools.calibrate_discard_validation import (
    _benchmark_index,
    _load_audits,
    classify_online_action,
    simulate_repeatable_threshold,
    summarize_online_performance,
    summarize_deltas,
)


def test_load_audits_accepts_disjoint_strata_with_equal_replicates(
    monkeypatch,
) -> None:
    rows = {
        "a1": [{"state_before_hash": "A", "audit_seed": 1}],
        "a2": [{"state_before_hash": "A", "audit_seed": 2}],
        "b1": [{"state_before_hash": "B", "audit_seed": 1}],
        "b2": [{"state_before_hash": "B", "audit_seed": 2}],
    }
    monkeypatch.setattr(
        calibration_module,
        "_read_jsonl",
        lambda path: iter(rows[path.name]),
    )

    grouped = _load_audits(
        [Path(name) for name in ("a1", "a2", "b1", "b2")]
    )

    assert set(grouped) == {"A", "B"}
    assert all(len(audits) == 2 for audits in grouped.values())


def test_load_audits_rejects_unequal_stratum_replicates(
    monkeypatch,
) -> None:
    rows = {
        "a1": [{"state_before_hash": "A", "audit_seed": 1}],
        "a2": [{"state_before_hash": "A", "audit_seed": 2}],
        "b1": [{"state_before_hash": "B", "audit_seed": 1}],
    }
    monkeypatch.setattr(
        calibration_module,
        "_read_jsonl",
        lambda path: iter(rows[path.name]),
    )

    with pytest.raises(
        ValueError,
        match="discard_calibration_evidence_replicate_count_mismatch",
    ):
        _load_audits(
            [Path(name) for name in ("a1", "a2", "b1")]
        )


def test_benchmark_index_flattens_saved_league_discard_events() -> None:
    indexed = _benchmark_index(
        {
            "rows": [
                {
                    "candidate_discard_events": [
                        {
                            "public_view_id": "view-a",
                            "production_label": "A",
                            "search_selected_label": "B",
                            "baseline_selected_label": "A",
                            "validation_preferred_label": "A",
                            "validation_challenger_label": "B",
                            "validation_challenger_labels": ["B"],
                            "validation_selected_label": "B",
                            "validation_complete": True,
                            "validation_error": None,
                            "candidates": [
                                {
                                    "label": "B",
                                    "average_reward": 0.2,
                                    "heuristic_value": 3.0,
                                }
                            ],
                            "validation_diagnostics": {
                                "expected_confirmation_worlds": 112,
                                "challengers": [],
                            },
                        }
                    ]
                }
            ]
        }
    )

    assert indexed["view-a"]["selected_label"] == "B"
    assert indexed["view-a"]["validation_complete"] is True
    assert indexed["view-a"]["validation_coverage_ranking"][0]["label"] == "B"


def test_summarize_deltas_applies_familywise_confidence() -> None:
    summary = summarize_deltas([1.0] * 64, comparisons=5)

    assert summary["samples"] == 64
    assert summary["mean_delta"] == 1.0
    assert summary["confidently_positive"] is True
    assert summary["confidently_negative"] is False


def test_summarize_online_performance_reports_net_and_strata() -> None:
    rows = [
        {
            "preferred_label": "A",
            "online_selected_label": "B",
            "pooled_best_label": "B",
            "classification": "correct_override",
            "preferred_confidently_suboptimal": True,
            "game": {"opponents": ["pressure"]},
            "online_selected_vs_preferred": {
                "mean_delta": 0.2,
                "lower_confidence_bound": 0.1,
                "upper_confidence_bound": 0.3,
            },
            "online_selected_vs_pooled_best": {"mean_delta": 0.0},
        },
        {
            "preferred_label": "A",
            "online_selected_label": "C",
            "pooled_best_label": "A",
            "classification": "harmful_override",
            "preferred_confidently_suboptimal": False,
            "game": {"opponents": ["pressure"]},
            "online_selected_vs_preferred": {
                "mean_delta": -0.1,
                "lower_confidence_bound": -0.2,
                "upper_confidence_bound": -0.01,
            },
            "online_selected_vs_pooled_best": {"mean_delta": -0.1},
        },
    ]

    summary = summarize_online_performance(rows)

    assert summary["overrides"] == 2
    assert summary["mean_improvement_over_preferred"] == 0.05
    assert summary["mean_regret_to_pooled_best"] == 0.05
    assert summary["confidently_positive_overrides"] == 1
    assert summary["confidently_negative_overrides"] == 1
    assert summary["opponent_breakdown"]["pressure"]["states"] == 2


def test_classify_online_action_distinguishes_override_quality() -> None:
    confident = {"confidently_positive": True, "mean_delta": 0.2}
    uncertain = {"confidently_positive": False, "mean_delta": 0.1}
    harmful = {"confidently_positive": False, "mean_delta": -0.1}

    assert (
        classify_online_action(
            preferred="A",
            online_selected="B",
            pooled_best="B",
            preferred_confidently_suboptimal=True,
            selected_vs_preferred=confident,
        )
        == "correct_override"
    )
    assert (
        classify_online_action(
            preferred="A",
            online_selected="B",
            pooled_best="C",
            preferred_confidently_suboptimal=True,
            selected_vs_preferred=confident,
        )
        == "beneficial_override"
    )
    assert (
        classify_online_action(
            preferred="A",
            online_selected="B",
            pooled_best="C",
            preferred_confidently_suboptimal=True,
            selected_vs_preferred=uncertain,
        )
        == "unproven_override"
    )
    assert (
        classify_online_action(
            preferred="A",
            online_selected="B",
            pooled_best="C",
            preferred_confidently_suboptimal=True,
            selected_vs_preferred=harmful,
        )
        == "harmful_override"
    )
    assert (
        classify_online_action(
            preferred="A",
            online_selected="A",
            pooled_best="B",
            preferred_confidently_suboptimal=True,
            selected_vs_preferred=harmful,
        )
        == "missed_override"
    )


def test_pooled_best_override_remains_unproven_without_confidence() -> None:
    result = classify_online_action(
        preferred="A",
        online_selected="B",
        pooled_best="B",
        preferred_confidently_suboptimal=False,
        selected_vs_preferred={
            "confidently_positive": False,
            "mean_delta": 0.1,
        },
    )

    assert result == "unproven_override"


def test_threshold_sweep_uses_repeatable_complete_evidence() -> None:
    candidate = {
        "label": "B",
        "mean_reward": 0.2,
        "vs_preferred": {
            "confidently_positive": True,
            "mean_delta": 0.2,
        },
    }
    preferred = {
        "label": "A",
        "mean_reward": 0.0,
        "vs_preferred": {
            "confidently_positive": False,
            "mean_delta": 0.0,
        },
    }
    rows = [
        {
            "preferred_label": "A",
            "pooled_best_label": "B",
            "preferred_confidently_suboptimal": True,
            "pooled_best_vs_preferred": {"mean_delta": 0.2},
            "candidates": [candidate, preferred],
            "online_challenger_evidence": [
                {
                    "challenger_label": "B",
                    "complete": True,
                    "familywise_confident": False,
                    "first_mean_delta": 0.1,
                    "second_mean_delta": 0.2,
                    "combined_mean_delta": 0.15,
                    "combined_lower_confidence_bound": -0.1,
                }
            ],
        }
    ]

    accepted = simulate_repeatable_threshold(rows, threshold=0.1)
    rejected = simulate_repeatable_threshold(rows, threshold=0.2)

    assert accepted["correct_overrides"] == 1
    assert rejected["missed_overrides"] == 1
