"""Tests for implementation-document stop-condition auditing."""

from tools.audit_stop_conditions import build_audit_report


def _ok_command(command: str) -> dict:
    return {"ok": True, "command": command, "returncode": 0, "stdout": "", "stderr": ""}


def _fake_nodes() -> list[str]:
    nodes = []
    nodes.extend(f"chenzhou_zipai_ai/tests/test_policy.py::test_policy_case_{idx}" for idx in range(20))
    nodes.extend(
        f"chenzhou_zipai_ai/tests/test_professional_brain.py::test_structure_allocation_contract_cases[{idx}]"
        for idx in range(10)
    )
    nodes.extend(
        [
            "chenzhou_zipai_ai/tests/test_professional_brain.py::test_instance_allocation_does_not_reuse_triplet_for_123",
            "chenzhou_zipai_ai/tests/test_professional_brain.py::test_hu_strategy_late_game_hu_now_even_when_continue_ev_is_high",
            "chenzhou_zipai_ai/tests/test_hu_checker.py::test_explain_hu_returns_groups_and_xi",
            "chenzhou_zipai_ai/tests/test_hu_checker.py::test_wildcard_can_complete_key_melds",
            "chenzhou_zipai_ai/tests/test_xi_calculator.py::test_xi_for_special_melds_and_runs",
            "chenzhou_zipai_ai/tests/test_xi_calculator.py::test_xi_for_peng_pao_ti_by_size",
        ]
    )
    return nodes


def test_stop_condition_audit_reports_missing_real_replays_only():
    report = build_audit_report(
        pytest_result=_ok_command("pytest"),
        eval_policy_result=_ok_command("eval_policy_cases"),
        regression_result=_ok_command("regression_suite"),
        collected_nodes=_fake_nodes(),
        replay_result={
            "ok": False,
            "decisions_checked": 0,
            "min_decisions": 10,
        },
        readme_runtime_local_ok=True,
    )

    assert not report["ok"]
    assert report["passed"] == report["total"] - 1
    assert [row["item"] for row in report["unmet"]] == [9]
    assert report["unmet"][0]["gap"].startswith("缺真实设备日志样本")
    assert "collect_real_replay_samples.py" in report["unmet"][0]["gap"]


def test_stop_condition_audit_passes_when_replay_evidence_exists():
    report = build_audit_report(
        pytest_result=_ok_command("pytest"),
        eval_policy_result=_ok_command("eval_policy_cases"),
        regression_result=_ok_command("regression_suite"),
        collected_nodes=_fake_nodes(),
        replay_result={
            "ok": True,
            "decisions_checked": 10,
            "min_decisions": 10,
        },
        readme_runtime_local_ok=True,
    )

    assert report["ok"]
    assert not report["unmet"]
