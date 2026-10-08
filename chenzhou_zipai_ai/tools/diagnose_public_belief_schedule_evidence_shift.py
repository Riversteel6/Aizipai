"""Compare validator evidence for states changed by schedule resolution."""

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
from tools.evaluate_proxy_public_belief_validator import verify_capability_reports
from tools.evaluate_public_belief_validator import evaluate_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = diagnose(
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
                "states": report["states"],
                "runs": report["runs"],
                "health_errors": report["health_errors"],
                "summary_by_slots": report["summary_by_slots"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def diagnose(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    for key in inputs:
        if f"{key}_sha256" in inputs:
            _verify_hash(inputs, key)
    source_21 = _read_json(Path(inputs["source_21_report"]))
    source_112 = _read_json(Path(inputs["source_112_report"]))
    state_ids = changed_state_ids(source_21["rows"], source_112["rows"])
    required = int(preregistration["required_changed_states"])
    if len(state_ids) != required:
        raise ValueError(f"schedule_evidence_changed_states:{len(state_ids)}!={required}")
    validator_prereg = _read_json(Path(inputs["validator_preregistration"]))
    validator_inputs = dict(validator_prereg["inputs"])
    verify_capability_reports(validator_inputs)
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
    decision_margin = float(validator_config.evidence_calibrator.decision_margin)
    source_21_by_id = {str(row["state_before_hash"]): row for row in source_21["rows"]}
    source_112_by_id = {str(row["state_before_hash"]): row for row in source_112["rows"]}
    rows: list[dict[str, Any]] = []
    try:
        for slots in tuple(int(item) for item in preregistration["schedule_slots"]):
            for state_id in state_ids:
                row = evaluate_state(
                    benchmark_rows[state_id],
                    batch_4[state_id],
                    batch_5[state_id],
                    belief_model=belief_model,
                    validator_config=validator_config,
                    slots=slots,
                    profile_factories=HYBRID_RESPONSE_PROXY_TYPES,
                )
                heldout_best = str(row["heldout_best_label"])
                challenger = _challenger_evidence(row, heldout_best)
                result = {
                    "slots": slots,
                    "state_before_hash": state_id,
                    "opponent_stratum": row["opponent_stratum"],
                    "public_actions": row["public_actions"],
                    "posterior": row["posterior"],
                    "schedule_counts": row["schedule_counts"],
                    "preferred_label": row["preferred_label"],
                    "selected_label": row["candidate_selected_label"],
                    "heldout_best_label": heldout_best,
                    "validation_complete": row["validation_complete"],
                    "health_errors": row["health_errors"],
                    "heldout_best_in_confirmed_challengers": challenger is not None,
                    "heldout_best_calibrated_advantage": (
                        challenger.get("calibrated_advantage")
                        if challenger is not None
                        else None
                    ),
                    "heldout_best_passed_margin": bool(
                        challenger is not None
                        and challenger.get("calibrated_advantage") is not None
                        and float(challenger["calibrated_advantage"]) > decision_margin
                    ),
                    "heldout_best_evidence": challenger,
                    "validation_evidence": row["validation_evidence"],
                    "source_21_selected_label": source_21_by_id[state_id][
                        "candidate_selected_label"
                    ],
                    "source_112_selected_label": source_112_by_id[state_id][
                        "candidate_selected_label"
                    ],
                }
                rows.append(result)
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "slots",
                                "state_before_hash",
                                "selected_label",
                                "heldout_best_label",
                                "heldout_best_in_confirmed_challengers",
                                "heldout_best_calibrated_advantage",
                                "heldout_best_passed_margin",
                                "health_errors",
                            )
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        close_shared_dual_discard_executors()
    summary_by_slots = {
        str(slots): {
            "runs": len(selected),
            "complete": sum(bool(row["validation_complete"]) for row in selected),
            "heldout_best_in_confirmed_challengers": sum(
                bool(row["heldout_best_in_confirmed_challengers"])
                for row in selected
            ),
            "heldout_best_passed_margin": sum(
                bool(row["heldout_best_passed_margin"]) for row in selected
            ),
            "selected_heldout_best": sum(
                row["selected_label"] == row["heldout_best_label"]
                for row in selected
            ),
        }
        for slots in tuple(int(item) for item in preregistration["schedule_slots"])
        for selected in [[row for row in rows if row["slots"] == slots]]
    }
    return {
        "schema_version": "public-belief-schedule-evidence-shift-diagnostic-v1",
        "promotion_effect": "none",
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "decision_margin": decision_margin,
        "states": len(state_ids),
        "runs": len(rows),
        "health_errors": sum(int(row["health_errors"]) for row in rows),
        "summary_by_slots": summary_by_slots,
        "rows": rows,
    }


def changed_state_ids(
    source_21_rows: list[Mapping[str, Any]],
    source_112_rows: list[Mapping[str, Any]],
) -> tuple[str, ...]:
    selected_21 = {
        str(row["state_before_hash"]): str(row["candidate_selected_label"])
        for row in source_21_rows
    }
    selected_112 = {
        str(row["state_before_hash"]): str(row["candidate_selected_label"])
        for row in source_112_rows
    }
    if not set(selected_112).issubset(selected_21):
        raise ValueError("schedule_evidence_source_state_mismatch")
    return tuple(
        sorted(
            state_id
            for state_id, label in selected_112.items()
            if selected_21[state_id] != label
        )
    )


def _challenger_evidence(row: Mapping[str, Any], label: str) -> dict[str, Any] | None:
    evidence = row.get("validation_evidence") or {}
    return next(
        (
            dict(item)
            for item in evidence.get("challengers") or ()
            if str(item.get("challenger_label")) == label
        ),
        None,
    )


def _verify_hash(inputs: Mapping[str, Any], key: str) -> None:
    path = Path(str(inputs[key]))
    if sha256_file(path) != str(inputs[f"{key}_sha256"]):
        raise ValueError(f"schedule_evidence_{key}_hash_mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
