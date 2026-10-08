from tools.check_v81_calibrator_training_gate import evaluate_training_gate


def test_gate_refuses_rollout_only_labels() -> None:
    report = evaluate_training_gate(
        [
            {"acceptance": "PASS", "oracle_status": "EXACT_ACTION"},
            {"acceptance": "UNRESOLVED", "oracle_status": "ROLLOUT_ONLY"},
        ],
        expected_rows=2,
    )

    assert report["status"] == "FAIL"
    assert report["calibrator_training_allowed"] is False
    assert "rollout_only_labels:1" in report["failures"]


def test_gate_allows_complete_independently_accepted_labels() -> None:
    report = evaluate_training_gate(
        [
            {"acceptance": "PASS", "oracle_status": "EXACT_ACTION"},
            {"acceptance": "PASS", "oracle_status": "EXACT_ACTION"},
        ],
        expected_rows=2,
    )

    assert report["status"] == "PASS"
    assert report["calibrator_training_allowed"] is True
