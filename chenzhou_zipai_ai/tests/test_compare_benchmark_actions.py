from tools.compare_benchmark_actions import compare_actions


def _stats(delta: float) -> dict[str, object]:
    return {
        "mean_delta": delta,
        "confidently_positive": delta > 0.05,
        "confidently_negative": delta < -0.05,
    }


def test_comparison_keeps_one_fixed_preferred_action() -> None:
    rows = [
        {
            "state_before_hash": "state",
            "public_view_id": "view",
            "preferred_label": "A",
            "online_selected_label": "B",
            "pooled_best_label": "C",
            "preferred_confidently_suboptimal": True,
            "candidates": [
                {"label": "A", "vs_preferred": _stats(0.0)},
                {"label": "B", "vs_preferred": _stats(-0.2)},
                {"label": "C", "vs_preferred": _stats(0.3)},
            ],
        }
    ]

    report = compare_actions(rows, selections={"view": "C"})

    assert report["ok"]
    assert report["correct_overrides"] == 1
    assert report["mean_improvement_over_fixed_preferred"] == 0.3
    assert report["mean_gain_over_recorded_action"] == 0.5
    assert report["improved_actions"] == 1


def test_subset_comparison_counts_unchanged_population_as_zero_gain() -> None:
    rows = [
        {
            "state_before_hash": "state",
            "public_view_id": "view",
            "preferred_label": "A",
            "online_selected_label": "B",
            "pooled_best_label": "C",
            "preferred_confidently_suboptimal": True,
            "candidates": [
                {"label": "A", "vs_preferred": _stats(0.0)},
                {"label": "B", "vs_preferred": _stats(-0.2)},
                {"label": "C", "vs_preferred": _stats(0.3)},
            ],
        }
    ]

    report = compare_actions(
        rows,
        selections={"view": "C"},
        comparison_population_states=5,
    )

    assert report["comparison_population_states"] == 5
    assert report["unchanged_population_states"] == 4
    assert report["population_mean_gain_over_recorded_action"] == 0.1
    assert report["population_improved_actions"] == 1
    assert report["population_regressed_actions"] == 0
    assert report["population_equal_actions"] == 4


def test_population_cannot_be_smaller_than_compared_subset() -> None:
    rows = [
        {
            "state_before_hash": "state",
            "public_view_id": "view",
            "preferred_label": "A",
            "online_selected_label": "A",
            "pooled_best_label": "A",
            "preferred_confidently_suboptimal": False,
            "candidates": [
                {"label": "A", "vs_preferred": _stats(0.0)},
            ],
        }
    ]

    try:
        compare_actions(
            rows,
            selections={"view": "A"},
            comparison_population_states=0,
        )
    except ValueError:
        raise AssertionError("zero should preserve the default population")

    try:
        compare_actions(
            rows,
            selections={"view": "A"},
            comparison_population_states=-1,
        )
    except ValueError as exc:
        assert "smaller_than_compared_states" in str(exc)
    else:
        raise AssertionError("negative population must be rejected")
