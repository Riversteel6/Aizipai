from tools.evaluate_fast_discard_followup_heavy_state_gate import (
    focused_gate_failures,
)


_GATE = {
    "required_repeats": 2,
    "require_all_runs_complete": True,
    "maximum_health_errors": 0,
    "require_stable_selected_action": True,
}


def test_heavy_state_gate_accepts_two_stable_complete_runs() -> None:
    assert focused_gate_failures(
        _GATE,
        runs=2,
        complete_runs=2,
        health_errors=0,
        selected_labels=1,
    ) == []


def test_heavy_state_gate_reports_runtime_failures() -> None:
    assert focused_gate_failures(
        _GATE,
        runs=1,
        complete_runs=0,
        health_errors=2,
        selected_labels=0,
    ) == [
        "required_repeats",
        "incomplete_runs",
        "health_errors",
        "unstable_selected_action",
    ]
