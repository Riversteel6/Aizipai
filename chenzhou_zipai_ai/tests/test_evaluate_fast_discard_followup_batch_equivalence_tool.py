from tools.evaluate_fast_discard_followup_batch_equivalence import (
    _focused_gate_failures,
)


_GATE = {
    "minimum_trace_states": 3000,
    "require_zero_quick_potential_value_mismatches": True,
    "require_zero_proxy_value_mismatches": True,
    "require_zero_proxy_action_mismatches": True,
    "maximum_trace_p95_inference_ms": 5.0,
}


def test_focused_gate_accepts_exact_low_latency_equivalence() -> None:
    assert _focused_gate_failures(
        _GATE,
        states=3296,
        quick_mismatches=0,
        value_mismatches=0,
        action_mismatches=0,
        p95_inference_ms=4.0,
    ) == []


def test_focused_gate_reports_each_regression() -> None:
    assert _focused_gate_failures(
        _GATE,
        states=2999,
        quick_mismatches=1,
        value_mismatches=1,
        action_mismatches=1,
        p95_inference_ms=5.1,
    ) == [
        "minimum_trace_states",
        "quick_potential_value_mismatches",
        "proxy_value_mismatches",
        "proxy_action_mismatches",
        "p95_inference_ms",
    ]
