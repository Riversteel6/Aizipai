from tools.calibrate_response_validation import (
    choose_conservative_threshold,
    rows_from_benchmark,
    summarize_threshold,
)


def _row(state: str, online_lcb: float, actual_delta: float):
    return {
        "state_before_hash": state,
        "online_lower_confidence_bound": online_lcb,
        "actual": {
            "mean_delta": actual_delta,
            "upper_confidence_bound": actual_delta + 0.05,
            "confidently_positive": actual_delta > 0.05,
        },
    }


def test_response_threshold_rejects_harmful_low_confidence_override():
    rows = [
        _row("bad", 0.1, -0.2),
        _row("good", 0.35, 0.4),
    ]

    summary = summarize_threshold(rows, threshold=0.2)

    assert summary["overrides"] == 1
    assert summary["beneficial_overrides"] == 1
    assert summary["harmful_overrides"] == 0
    assert summary["total_actual_gain"] == 0.4


def test_conservative_selection_requires_zero_empirical_harm_when_available():
    selected = choose_conservative_threshold(
        [
            {
                "threshold": 0.1,
                "harmful_overrides": 1,
                "confidently_harmful_overrides": 0,
                "total_actual_gain": 1.0,
                "confidently_beneficial_overrides": 2,
                "beneficial_overrides": 3,
            },
            {
                "threshold": 0.3,
                "harmful_overrides": 0,
                "confidently_harmful_overrides": 0,
                "total_actual_gain": 0.4,
                "confidently_beneficial_overrides": 1,
                "beneficial_overrides": 1,
            },
        ]
    )

    assert selected["threshold"] == 0.3


def test_response_benchmark_replaces_stale_online_challenger():
    rows = rows_from_benchmark(
        [
            {
                "public_state_id": "public-1",
                "production_key": "PASS",
                "empirical_best_key": "CHI:old",
            }
        ],
        benchmark_rows=[
            {
                "public_view_id": "public-1",
                "production_key": "PASS",
                "selected_key": "PASS",
                "candidate_count": 3,
                "confidence_override": False,
                "paired_advantages": [
                    {
                        "candidate_key": "PENG:二",
                        "mean_delta": 0.4,
                        "standard_error": 0.1,
                        "lower_confidence_bound": 0.2,
                        "samples": 84,
                    }
                ],
            }
        ],
    )

    assert rows[0]["empirical_best_key"] == "PENG:二"
    assert rows[0]["online_lower_confidence_bound"] == 0.2
    assert not rows[0]["applied_override"]
