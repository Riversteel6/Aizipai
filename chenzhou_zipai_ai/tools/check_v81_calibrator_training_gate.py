"""Refuse v8.1 WIN_FIRST calibrator training until every audit label is accepted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = WORKSPACE / "reports" / "v81_suspicious_decisions_relabel_20260822.jsonl"
DEFAULT_OUTPUT = WORKSPACE / "reports" / "v81_win_first_calibrator_training_gate_20260822.json"


def evaluate_training_gate(
    rows: Iterable[dict[str, Any]],
    *,
    expected_rows: int = 179,
) -> dict[str, Any]:
    items = list(rows)
    accepted = [row for row in items if row.get("acceptance") == "PASS"]
    exact = [row for row in items if row.get("oracle_status") == "EXACT_ACTION"]
    rollout_only = [row for row in items if row.get("oracle_status") == "ROLLOUT_ONLY"]
    product_errors = [row for row in items if row.get("product_error")]
    failures: list[str] = []
    if len(items) != expected_rows:
        failures.append(f"row_count:{len(items)}!={expected_rows}")
    if len(accepted) != expected_rows:
        failures.append(f"accepted_labels:{len(accepted)}!={expected_rows}")
    if rollout_only:
        failures.append(f"rollout_only_labels:{len(rollout_only)}")
    if product_errors:
        failures.append(f"product_errors:{len(product_errors)}")
    return {
        "schema_version": "v81-win-first-calibrator-training-gate-v1",
        "expected_rows": expected_rows,
        "observed_rows": len(items),
        "accepted_labels": len(accepted),
        "exact_oracle_actions": len(exact),
        "rollout_only_labels": len(rollout_only),
        "product_errors": len(product_errors),
        "legacy_calibrator_enabled": False,
        "calibrator_training_allowed": not failures,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-rows", type=int, default=179)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate_training_gate(rows, expected_rows=args.expected_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["calibrator_training_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
