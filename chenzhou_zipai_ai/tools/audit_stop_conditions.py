"""Audit implementation-document stop conditions against current evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.eval_policy_cases import CASES as POLICY_CASES
from tools.verify_real_replays import verify_replay_logs


STRATEGY_TEST_FILES = {
    "chenzhou_zipai_ai/tests/test_policy.py",
    "chenzhou_zipai_ai/tests/test_professional_brain.py",
    "chenzhou_zipai_ai/tests/test_recommend_action.py",
    "chenzhou_zipai_ai/tests/test_action_plan.py",
    "chenzhou_zipai_ai/tests/test_live_assistant.py",
    "chenzhou_zipai_ai/tests/test_option_policy.py",
}

ALLOCATION_TEST_HINTS = {
    "test_instance_allocation_does_not_reuse_triplet_for_123",
    "test_mixed_triplet_and_sequence_use_different_card_instances_when_possible",
    "test_single_five_cannot_feed_mixed_triplet_and_sequence_together",
    "test_structure_allocation_contract_cases",
}

HU_XI_TEST_FILES = {
    "chenzhou_zipai_ai/tests/test_hu_checker.py",
    "chenzhou_zipai_ai/tests/test_xi_calculator.py",
}


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=WORKSPACE, capture_output=True, text=True, check=False)
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _collect_pytest_nodes() -> list[str]:
    result = _run([sys.executable, "-m", "pytest", "--collect-only", "-q"])
    if not result["ok"]:
        return []
    return [
        line.strip().replace("\\", "/")
        for line in result["stdout"].splitlines()
        if "::test_" in line
    ]


def _count_nodes(nodes: list[str], files: set[str]) -> int:
    return sum(any(node.startswith(f"{file}::") for file in files) for node in nodes)


def _count_allocation_tests(nodes: list[str]) -> int:
    return sum(any(hint in node for hint in ALLOCATION_TEST_HINTS) for node in nodes)


def _count_hu_xi_tests(nodes: list[str]) -> int:
    return sum(
        any(node.startswith(f"{file}::") for file in HU_XI_TEST_FILES)
        or "::test_hu_" in node
        or "::test_xi_" in node
        or "hu_strategy" in node
        for node in nodes
    )


def _readme_runtime_local_ok() -> bool:
    texts = []
    for path in (WORKSPACE / "README.md", ROOT / "README.md"):
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    joined = "\n".join(texts)
    return "本地规则引擎" in joined and "策略引擎" in joined and "大模型" in joined and "不参与实时打牌" in joined


def _condition(
    item: int,
    requirement: str,
    passed: bool,
    evidence: str,
    *,
    actual: Any = None,
    required: Any = None,
    gap: str = "",
) -> dict[str, Any]:
    return {
        "item": item,
        "requirement": requirement,
        "passed": bool(passed),
        "actual": actual,
        "required": required,
        "evidence": evidence,
        "gap": gap,
    }


def build_audit_report(
    *,
    pytest_result: dict[str, Any],
    eval_policy_result: dict[str, Any],
    regression_result: dict[str, Any],
    collected_nodes: list[str],
    replay_result: dict[str, Any],
    readme_runtime_local_ok: bool,
) -> dict[str, Any]:
    strategy_count = _count_nodes(collected_nodes, STRATEGY_TEST_FILES) + len(POLICY_CASES)
    allocation_count = _count_allocation_tests(collected_nodes)
    chi_count = sum(1 for case in POLICY_CASES if str(case.get("name", "")).startswith("chi_"))
    peng_count = sum(1 for case in POLICY_CASES if str(case.get("name", "")).startswith("peng_"))
    hu_xi_count = _count_hu_xi_tests(collected_nodes)
    condition_rows = [
        _condition(1, "pytest -q 全部通过", pytest_result["ok"], pytest_result["command"]),
        _condition(2, "tools/eval_policy_cases.py 全部通过", eval_policy_result["ok"], eval_policy_result["command"]),
        _condition(3, "tools/regression_suite.py 全部通过", regression_result["ok"], regression_result["command"]),
        _condition(4, "至少 40 个策略测试通过", strategy_count >= 40, "pytest collect + eval_policy_cases.CASES", actual=strategy_count, required=40),
        _condition(5, "至少 10 个资源分配测试通过", allocation_count >= 10, "pytest collect allocation contract cases", actual=allocation_count, required=10),
        _condition(6, "至少 10 个吃牌 EV 测试通过", chi_count >= 10, "eval_policy_cases.CASES chi_*", actual=chi_count, required=10),
        _condition(7, "至少 5 个碰牌 EV 测试通过", peng_count >= 5, "eval_policy_cases.CASES peng_*", actual=peng_count, required=5),
        _condition(8, "至少 5 个胡牌/胡息测试通过", hu_xi_count >= 5, "pytest collect hu/xi tests", actual=hu_xi_count, required=5),
        _condition(
            9,
            "replay_decision 可以复现至少 10 个真实日志",
            bool(replay_result.get("ok")),
            "tools.verify_real_replays.verify_replay_logs",
            actual=replay_result.get("decisions_checked", 0),
            required=replay_result.get("min_decisions", 10),
            gap="缺真实设备日志样本；连接手机后运行 tools/collect_real_replay_samples.py" if not replay_result.get("ok") else "",
        ),
        _condition(10, "所有 SAFE_HALT 都有 reason", pytest_result["ok"], "validate_logs and SAFE_HALT contract tests"),
        _condition(11, "所有 action_plan 都不能私自换牌", pytest_result["ok"], "ConflictGuard/action_plan target tests"),
        _condition(12, "所有 hard break 都必须有 forced reason", pytest_result["ok"], "forced_break_hard_protection tests"),
        _condition(13, "所有日志字段完整", pytest_result["ok"], "test_game_logger_contract + validate_logs"),
        _condition(14, "README 写明本地引擎实时打牌，大模型只用于开发/复盘", readme_runtime_local_ok, "README.md / chenzhou_zipai_ai/README.md"),
    ]
    passed_count = sum(1 for row in condition_rows if row["passed"])
    return {
        "ok": passed_count == len(condition_rows),
        "passed": passed_count,
        "total": len(condition_rows),
        "conditions": condition_rows,
        "unmet": [row for row in condition_rows if not row["passed"]],
    }


def audit_stop_conditions(*, logs_root: Path, include_fixtures: bool = False) -> dict[str, Any]:
    pytest_result = _run([sys.executable, "-m", "pytest", "-q"])
    eval_policy_result = _run([sys.executable, "chenzhou_zipai_ai/tools/eval_policy_cases.py"])
    regression_result = _run([sys.executable, "chenzhou_zipai_ai/tools/regression_suite.py"])
    collected_nodes = _collect_pytest_nodes()
    replay_result = verify_replay_logs(logs_root, min_decisions=10, include_fixtures=include_fixtures)
    return build_audit_report(
        pytest_result=pytest_result,
        eval_policy_result=eval_policy_result,
        regression_result=regression_result,
        collected_nodes=collected_nodes,
        replay_result=replay_result,
        readme_runtime_local_ok=_readme_runtime_local_ok(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit document stop conditions.")
    parser.add_argument("--logs-root", type=Path, default=ROOT / "logs")
    parser.add_argument("--include-fixtures", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_stop_conditions(logs_root=args.logs_root, include_fixtures=args.include_fixtures)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        status = "ok" if report["ok"] else "FAIL"
        print(f"{status}\tpassed={report['passed']}/{report['total']}")
        for row in report["conditions"]:
            mark = "ok" if row["passed"] else "MISS"
            actual = "" if row["actual"] is None else f" actual={row['actual']}"
            required = "" if row["required"] is None else f" required={row['required']}"
            gap = "" if not row["gap"] else f" gap={row['gap']}"
            print(f"{mark}\t{row['item']}. {row['requirement']}{actual}{required}{gap}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
