from tools.evaluate_discard_calibrator import evaluate_model


def _evidence(label: str, combined: float) -> dict[str, object]:
    return {
        "challenger_label": label,
        "complete": True,
        "first_mean_delta": combined,
        "second_mean_delta": combined,
        "combined_mean_delta": combined,
        "combined_standard_error": 0.0,
        "coverage_rank": 1,
        "heuristic_rank": 1,
        "coverage_reward_delta": 0.0,
        "heuristic_value_delta": 0.0,
    }


def _stats(delta: float) -> dict[str, object]:
    return {
        "mean_delta": delta,
        "confidently_positive": delta > 0.05,
        "confidently_negative": delta < -0.05,
    }


def _row(state: str, evidence_delta: float, truth_delta: float) -> dict:
    return {
        "state_before_hash": state,
        "public_view_id": f"view-{state}",
        "preferred_label": "A",
        "pooled_best_label": "B" if truth_delta > 0.0 else "A",
        "preferred_confidently_suboptimal": truth_delta > 0.05,
        "pooled_best_vs_preferred": _stats(max(0.0, truth_delta)),
        "online_selected_label": "B",
        "online_challenger_evidence": [_evidence("B", evidence_delta)],
        "candidates": [
            {"label": "A", "vs_preferred": _stats(0.0)},
            {"label": "B", "vs_preferred": _stats(truth_delta)},
        ],
    }


def test_heldout_margin_trades_override_for_keep() -> None:
    rows = [
        _row("positive", 0.2, 0.1),
        _row("weak", 0.01, -0.1),
    ]
    model = {
        "ridge": 1.0,
        "feature_mean": [0.0] * 8,
        "feature_scale": [1.0] * 8,
        "intercept": 0.0,
        "coefficients": [1.0] + [0.0] * 7,
    }

    permissive = evaluate_model(
        rows,
        model=model,
        decision_margin=0.0,
    )
    conservative = evaluate_model(
        rows,
        model=model,
        decision_margin=0.05,
    )

    assert permissive["overrides"] == 2
    assert permissive["harmful_overrides"] == 1
    assert conservative["overrides"] == 1
    assert conservative["harmful_overrides"] == 0
    assert conservative["correct_overrides"] == 1
    assert conservative["mean_improvement_over_preferred"] == 0.05


def test_heldout_evaluation_uses_opponent_context() -> None:
    positive = _row("positive-context", 0.0, 0.1)
    positive["opponent_context_features"] = [1.0] + [0.0] * 7
    negative = _row("negative-context", 0.0, -0.1)
    negative["opponent_context_features"] = [-1.0] + [0.0] * 7
    model = {
        "ridge": 1.0,
        "feature_mean": [0.0] * 17,
        "feature_scale": [1.0] * 17,
        "intercept": 0.0,
        "coefficients": [0.0] * 9 + [1.0] + [0.0] * 7,
    }

    evaluation = evaluate_model(
        [positive, negative],
        model=model,
        decision_margin=0.0,
    )

    assert evaluation["overrides"] == 1
    assert evaluation["correct_overrides"] == 1
    assert evaluation["harmful_overrides"] == 0
