from tools.fit_teacher_discard_proxies import focused_gate_failures


def test_teacher_proxy_gate_requires_every_profile_and_latency() -> None:
    metrics = {
        "overall_state_top1_accuracy": 0.45,
        "overall_gain_over_legacy_proxy": 0.12,
        "game_mean_top1": {"lower_95": 0.31},
        "p95_inference_ms": 4.0,
        "illegal_predictions": 0,
        "by_profile": {
            name: {"state_top1_accuracy": 0.3}
            for name in "abcdefg"
        },
    }
    gate = {
        "required_profiles": 7,
        "required_folds": 5,
        "minimum_total_states": 3000,
        "minimum_states_per_profile": 350,
        "minimum_overall_state_top1_accuracy": 0.4,
        "minimum_profile_state_top1_accuracy": 0.25,
        "minimum_game_mean_top1_lower_95": 0.3,
        "minimum_overall_gain_over_legacy_proxy": 0.1,
        "maximum_p95_inference_ms": 5.0,
        "require_zero_group_leakage": True,
        "require_zero_illegal_predictions": True,
    }
    assert focused_gate_failures(
        metrics,
        gate=gate,
        profiles=7,
        folds=5,
        states=3200,
        states_by_profile={name: 400 for name in "abcdefg"},
        group_leakage=0,
        integrity_errors=0,
    ) == []

    metrics["by_profile"]["a"]["state_top1_accuracy"] = 0.24
    assert focused_gate_failures(
        metrics,
        gate=gate,
        profiles=7,
        folds=5,
        states=3200,
        states_by_profile={name: 400 for name in "abcdefg"},
        group_leakage=0,
        integrity_errors=0,
    ) == ["profile_state_top1_accuracy"]
