from tools.evaluate_public_belief_schedule_resolution import focused_gate_failures


_GATE = {
    "required_states": 54,
    "maximum_health_errors": 0,
    "require_all_validations_complete": True,
    "maximum_harmful_overrides": 3,
    "maximum_missed_overrides": 16,
    "minimum_confidently_positive_overrides": 30,
    "maximum_mean_regret_to_heldout_best": 0.1383702,
    "minimum_mean_regret_improvement_lower_95": 0.0,
}


def test_schedule_resolution_gate_accepts_joint_improvement() -> None:
    candidate = {
        "harmful_overrides": 2,
        "missed_overrides": 15,
        "confidently_positive_overrides": 31,
        "mean_regret_to_heldout_best": 0.12,
    }

    assert focused_gate_failures(
        candidate,
        regret_improvement={"lower_95": 0.001},
        health_errors=0,
        incomplete=0,
        states=54,
        gate=_GATE,
    ) == []


def test_schedule_resolution_gate_reports_each_failure() -> None:
    candidate = {
        "harmful_overrides": 4,
        "missed_overrides": 17,
        "confidently_positive_overrides": 29,
        "mean_regret_to_heldout_best": 0.14,
    }

    assert focused_gate_failures(
        candidate,
        regret_improvement={"lower_95": 0.0},
        health_errors=1,
        incomplete=1,
        states=53,
        gate=_GATE,
    ) == [
        "required_states",
        "health_errors",
        "incomplete_validations",
        "harmful_overrides",
        "missed_overrides",
        "confidently_positive_overrides",
        "mean_regret_to_heldout_best",
        "regret_improvement_lower_95",
    ]
