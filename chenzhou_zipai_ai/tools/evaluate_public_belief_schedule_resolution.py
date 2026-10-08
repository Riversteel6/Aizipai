"""Evaluate higher-resolution public-belief scheduling on frozen error states."""

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
from tools.evaluate_opponent_distribution_split import (
    _load_audits,
    _mean_confidence,
    summarize_policy,
)
from tools.evaluate_proxy_public_belief_validator import verify_capability_reports
from tools.evaluate_public_belief_validator import evaluate_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_evaluation(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(raw).hexdigest(),
        progress_every=max(0, int(args.progress_every)),
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
                "states": report["states"],
                "health_errors": report["health_errors"],
                "candidate": report["candidate"],
                "regret_improvement": report["regret_improvement"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] and report["focused_gate_pass"] else 1


def run_evaluation(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
    progress_every: int,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    for key in inputs:
        if f"{key}_sha256" in inputs:
            _verify_hash(inputs, key)
    source_report = _read_json(Path(inputs["source_report"]))
    validator_prereg = _read_json(Path(inputs["validator_preregistration"]))
    validator_inputs = dict(validator_prereg["inputs"])
    verify_capability_reports(validator_inputs)
    classes = set(preregistration["subset"]["source_candidate_classifications"])
    source_rows = {
        str(row["state_before_hash"]): row
        for row in source_report["rows"]
        if str(row["candidate"]["classification"]) in classes
    }
    required_states = int(preregistration["subset"]["required_states"])
    if len(source_rows) != required_states:
        raise ValueError(
            f"schedule_resolution_subset:{len(source_rows)}!={required_states}"
        )
    benchmark = _read_json(Path(validator_inputs["benchmark"]))
    benchmark_rows = {
        str(row["state_before_hash"]): row for row in benchmark["rows"]
    }
    batch_4 = _load_audits(Path(validator_inputs["heldout_batch_4"]))
    batch_5 = _load_audits(Path(validator_inputs["heldout_batch_5"]))
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
    slots = int(preregistration["distribution"]["slots"])
    rows: list[dict[str, Any]] = []
    try:
        for index, state_id in enumerate(sorted(source_rows), start=1):
            row = evaluate_state(
                benchmark_rows[state_id],
                batch_4[state_id],
                batch_5[state_id],
                belief_model=belief_model,
                validator_config=validator_config,
                slots=slots,
                profile_factories=HYBRID_RESPONSE_PROXY_TYPES,
            )
            row["source_candidate"] = source_rows[state_id]["candidate"]
            row["source_candidate_selected_label"] = source_rows[state_id][
                "candidate_selected_label"
            ]
            rows.append(row)
            if progress_every and (index % progress_every == 0 or index == len(source_rows)):
                print(
                    json.dumps(
                        {
                            "progress": index,
                            "total": len(source_rows),
                            "last_elapsed_ms": row["elapsed_ms"],
                            "health_errors": sum(
                                int(item["health_errors"]) for item in rows
                            ),
                        }
                    ),
                    flush=True,
                )
    finally:
        close_shared_dual_discard_executors()

    candidate = summarize_policy(rows, prefix="candidate")
    source_candidate = summarize_policy(rows, prefix="source_candidate")
    regret_improvement = _mean_confidence(
        [
            float(row["source_candidate"]["regret_to_heldout_best"])
            - float(row["candidate"]["regret_to_heldout_best"])
            for row in rows
        ]
    )
    health_errors = sum(int(row["health_errors"]) for row in rows)
    incomplete = sum(not bool(row["validation_complete"]) for row in rows)
    failures = focused_gate_failures(
        candidate,
        regret_improvement=regret_improvement,
        health_errors=health_errors,
        incomplete=incomplete,
        states=len(rows),
        gate=preregistration["focused_gate"],
    )
    return {
        "ok": bool(rows) and health_errors == 0,
        "schema_version": "public-belief-schedule-resolution-focused-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "states": len(rows),
        "health_errors": health_errors,
        "incomplete_validations": incomplete,
        "changed_selections": sum(
            str(row["candidate_selected_label"])
            != str(row["source_candidate_selected_label"])
            for row in rows
        ),
        "candidate": candidate,
        "source_candidate": source_candidate,
        "regret_improvement": regret_improvement,
        "focused_gate": dict(preregistration["focused_gate"]),
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
        "rows": rows,
    }


def focused_gate_failures(
    candidate: Mapping[str, Any],
    *,
    regret_improvement: Mapping[str, Any],
    health_errors: int,
    incomplete: int,
    states: int,
    gate: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    if states != int(gate["required_states"]):
        failures.append("required_states")
    if health_errors > int(gate["maximum_health_errors"]):
        failures.append("health_errors")
    if gate["require_all_validations_complete"] and incomplete:
        failures.append("incomplete_validations")
    if int(candidate["harmful_overrides"]) > int(gate["maximum_harmful_overrides"]):
        failures.append("harmful_overrides")
    if int(candidate["missed_overrides"]) > int(gate["maximum_missed_overrides"]):
        failures.append("missed_overrides")
    if int(candidate["confidently_positive_overrides"]) < int(
        gate["minimum_confidently_positive_overrides"]
    ):
        failures.append("confidently_positive_overrides")
    if float(candidate["mean_regret_to_heldout_best"]) >= float(
        gate["maximum_mean_regret_to_heldout_best"]
    ):
        failures.append("mean_regret_to_heldout_best")
    if float(regret_improvement["lower_95"]) <= float(
        gate["minimum_mean_regret_improvement_lower_95"]
    ):
        failures.append("regret_improvement_lower_95")
    return failures


def _verify_hash(inputs: Mapping[str, Any], key: str) -> None:
    path = Path(str(inputs[key]))
    if sha256_file(path) != str(inputs[f"{key}_sha256"]):
        raise ValueError(f"schedule_resolution_{key}_hash_mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
