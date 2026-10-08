"""Run the frozen mixed proxy schedule on the preregistered heavy state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.dual_discard_validator import close_shared_dual_discard_executors
from ai.dual_validated_candidate import (
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy,
)
from ai.opponent_belief import PublicOpponentBeliefModel
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from tools.evaluate_anchored_structural_discard_proxies import sha256_file
from tools.evaluate_hybrid_response_proxies import HYBRID_RESPONSE_PROXY_TYPES
from tools.evaluate_opponent_distribution_split import _load_audits
from tools.evaluate_public_belief_validator import evaluate_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_gate(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "focused_gate_pass": report["focused_gate_pass"],
                "complete_runs": report["complete_runs"],
                "health_errors": report["health_errors"],
                "elapsed_ms": [row["elapsed_ms"] for row in report["rows"]],
                "selected_labels": report["selected_labels"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] and report["focused_gate_pass"] else 1


def run_gate(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    _verify_hash(inputs, "equivalence_report")
    _verify_hash(inputs, "validator_preregistration")
    equivalence = _read_json(Path(inputs["equivalence_report"]))
    if not equivalence.get("focused_gate_pass"):
        raise ValueError("heavy_state_equivalence_anchor_did_not_pass")
    validator_prereg = _read_json(Path(inputs["validator_preregistration"]))
    validator_inputs = dict(validator_prereg["inputs"])
    state_id = str(inputs["state_before_hash"])
    benchmark = _read_json(Path(validator_inputs["benchmark"]))
    benchmark_rows = {
        str(row["state_before_hash"]): row for row in benchmark["rows"]
    }
    batch_4 = _load_audits(Path(validator_inputs["heldout_batch_4"]))
    batch_5 = _load_audits(Path(validator_inputs["heldout_batch_5"]))
    if state_id not in benchmark_rows or state_id not in batch_4 or state_id not in batch_5:
        raise ValueError(f"heavy_state_missing:{state_id}")
    belief_report = _read_json(Path(validator_inputs["belief_report"]))
    belief_model = PublicOpponentBeliefModel.from_dict(
        belief_report["final_model"]
    )
    anchor = ProfessionalParallelMultiOpponentRobustV81ResearchPolicy(
        rollout_policy_factories=(
            IndependentFastRolloutPolicy,
            IndependentFastPressurePolicy,
            IndependentFastDenialPolicy,
        )
    )
    validator_config = anchor.discard_validator.config
    slots = int(validator_prereg["distribution"]["slots"])
    repeats = int(preregistration["focused_gate"]["required_repeats"])
    rows: list[dict[str, Any]] = []
    try:
        for repeat in range(1, repeats + 1):
            row = evaluate_state(
                benchmark_rows[state_id],
                batch_4[state_id],
                batch_5[state_id],
                belief_model=belief_model,
                validator_config=validator_config,
                slots=slots,
                profile_factories=HYBRID_RESPONSE_PROXY_TYPES,
            )
            result = {
                "repeat": repeat,
                "elapsed_ms": row["elapsed_ms"],
                "validation_complete": row["validation_complete"],
                "health_errors": row["health_errors"],
                "health_details": row["health_details"],
                "selected_label": row["candidate_selected_label"],
                "schedule_counts": row["schedule_counts"],
            }
            rows.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            close_shared_dual_discard_executors()
    finally:
        close_shared_dual_discard_executors()

    health_errors = sum(int(row["health_errors"]) for row in rows)
    complete_runs = sum(bool(row["validation_complete"]) for row in rows)
    selected_labels = sorted({str(row["selected_label"]) for row in rows})
    failures = focused_gate_failures(
        preregistration["focused_gate"],
        runs=len(rows),
        complete_runs=complete_runs,
        health_errors=health_errors,
        selected_labels=len(selected_labels),
    )
    return {
        "ok": bool(rows),
        "schema_version": "fast-discard-followup-heavy-state-gate-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "state_before_hash": state_id,
        "runs": len(rows),
        "complete_runs": complete_runs,
        "health_errors": health_errors,
        "selected_labels": selected_labels,
        "focused_gate": dict(preregistration["focused_gate"]),
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
        "rows": rows,
    }


def focused_gate_failures(
    gate: Mapping[str, Any],
    *,
    runs: int,
    complete_runs: int,
    health_errors: int,
    selected_labels: int,
) -> list[str]:
    failures: list[str] = []
    required = int(gate["required_repeats"])
    if runs != required:
        failures.append("required_repeats")
    if gate["require_all_runs_complete"] and complete_runs != required:
        failures.append("incomplete_runs")
    if health_errors > int(gate["maximum_health_errors"]):
        failures.append("health_errors")
    if gate["require_stable_selected_action"] and selected_labels != 1:
        failures.append("unstable_selected_action")
    return failures


def _verify_hash(inputs: Mapping[str, Any], key: str) -> None:
    path = Path(str(inputs[key]))
    if sha256_file(path) != str(inputs[f"{key}_sha256"]):
        raise ValueError(f"heavy_state_{key}_hash_mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
