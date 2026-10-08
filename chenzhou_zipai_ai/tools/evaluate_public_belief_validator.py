"""Focused validation of a public-belief-conditioned V8.1 discard validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.dual_discard_validator import (
    DualBatchDiscardValidator,
    close_shared_dual_discard_executors,
)
from ai.dual_validated_candidate import (
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy,
)
from ai.full_game_simulator import InformationSetSearchPolicy, SimulationPolicy
from ai.ismcts import public_view_from_dict
from ai.opponent_belief import (
    PublicOpponentBeliefModel,
    public_opponent_features,
    smooth_weighted_profile_schedule,
)
from ai.opponent_league import (
    AggressiveMeldPolicy,
    DefensiveSearchPolicy,
    RedBlackSearchPolicy,
)
from audit.independent_opponent import (
    IndependentBalancedPolicy,
    IndependentDenialPolicy,
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
    IndependentPressurePolicy,
)
from engine.rules import rules_for_room
from tools.evaluate_opponent_distribution_split import (
    _load_audits,
    _world_rewards,
    evaluate_selection,
    heldout_truth,
    summarize_policy,
)


PROFILE_FACTORIES: dict[str, Callable[[], SimulationPolicy]] = {
    "aggressive_meld": AggressiveMeldPolicy,
    "defensive_search": DefensiveSearchPolicy,
    "independent_balanced": IndependentBalancedPolicy,
    "independent_denial": IndependentDenialPolicy,
    "independent_pressure": IndependentPressurePolicy,
    "information_set_search": InformationSetSearchPolicy,
    "red_black_search": RedBlackSearchPolicy,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    preregistration_bytes = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_bytes.decode("utf-8"))
    report = run_preregistered_validation(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(
            preregistration_bytes
        ).hexdigest(),
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
                "candidate_harmful_overrides": report["candidate"][
                    "harmful_overrides"
                ],
                "anchor_harmful_overrides": report["anchor"][
                    "harmful_overrides"
                ],
                "candidate_missed_overrides": report["candidate"][
                    "missed_overrides"
                ],
                "candidate_confidently_positive_overrides": report[
                    "candidate"
                ]["confidently_positive_overrides"],
                "candidate_mean_regret": report["candidate"][
                    "mean_regret_to_heldout_best"
                ],
                "p95_elapsed_ms": report["p95_elapsed_ms"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def run_preregistered_validation(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
    progress_every: int = 0,
    profile_factories: Mapping[str, Callable[[], SimulationPolicy]] | None = None,
) -> dict[str, Any]:
    resolved_factories = dict(profile_factories or PROFILE_FACTORIES)
    inputs = dict(preregistration["inputs"])
    belief_report = json.loads(
        Path(inputs["belief_report"]).read_text(encoding="utf-8")
    )
    if not belief_report.get("focused_gate_pass"):
        raise ValueError("public_belief_source_did_not_pass")
    belief_model = PublicOpponentBeliefModel.from_dict(
        belief_report["final_model"]
    )
    expected_profiles = tuple(
        str(item) for item in preregistration["distribution"]["profile_order"]
    )
    if belief_model.profile_names != expected_profiles:
        raise ValueError("public_belief_profile_order_mismatch")
    if set(expected_profiles) != set(resolved_factories):
        raise ValueError("public_belief_factory_coverage_mismatch")

    benchmark = json.loads(
        Path(inputs["benchmark"]).read_text(encoding="utf-8")
    )
    benchmark_rows = {
        str(row["state_before_hash"]): row
        for row in benchmark.get("rows") or ()
    }
    batch_4 = _load_audits(Path(inputs["heldout_batch_4"]))
    batch_5 = _load_audits(Path(inputs["heldout_batch_5"]))
    if not benchmark_rows or set(benchmark_rows) != set(batch_4) or set(batch_4) != set(batch_5):
        raise ValueError("public_belief_validator_state_mismatch")

    anchor_policy = ProfessionalParallelMultiOpponentRobustV81ResearchPolicy(
        rollout_policy_factories=(
            IndependentFastRolloutPolicy,
            IndependentFastPressurePolicy,
            IndependentFastDenialPolicy,
        )
    )
    validator_config = anchor_policy.discard_validator.config
    slots = int(preregistration["distribution"]["slots"])
    rows: list[dict[str, Any]] = []
    try:
        for index, state_id in enumerate(sorted(benchmark_rows), start=1):
            row = evaluate_state(
                benchmark_rows[state_id],
                batch_4[state_id],
                batch_5[state_id],
                belief_model=belief_model,
                validator_config=validator_config,
                slots=slots,
                profile_factories=resolved_factories,
            )
            rows.append(row)
            if progress_every and (index % progress_every == 0 or index == len(benchmark_rows)):
                print(
                    json.dumps(
                        {
                            "progress": index,
                            "total": len(benchmark_rows),
                            "last_elapsed_ms": row["elapsed_ms"],
                            "health_errors": sum(
                                int(item["health_errors"]) for item in rows
                            ),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        close_shared_dual_discard_executors()

    candidate = summarize_policy(rows, prefix="candidate")
    anchor = summarize_policy(rows, prefix="anchor")
    health_errors = sum(int(row["health_errors"]) for row in rows)
    incomplete = sum(not bool(row["validation_complete"]) for row in rows)
    elapsed = sorted(float(row["elapsed_ms"]) for row in rows)
    p95 = elapsed[min(len(elapsed) - 1, int((len(elapsed) - 1) * 0.95))] if elapsed else 0.0
    failures = focused_gate_failures(
        candidate,
        anchor=anchor,
        gate=preregistration["focused_gate"],
        health_errors=health_errors,
        incomplete=incomplete,
        states=len(rows),
    )
    return {
        "ok": bool(rows) and health_errors == 0,
        "schema_version": "public-belief-validator-focused-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "belief_report": str(inputs["belief_report"]),
        "belief_report_sha256": _sha256(Path(inputs["belief_report"])),
        "benchmark": str(inputs["benchmark"]),
        "benchmark_sha256": _sha256(Path(inputs["benchmark"])),
        "states": len(rows),
        "health_errors": health_errors,
        "incomplete_validations": incomplete,
        "candidate": candidate,
        "anchor": anchor,
        "p95_elapsed_ms": round(p95, 3),
        "maximum_elapsed_ms": round(max(elapsed, default=0.0), 3),
        "focused_gate": dict(preregistration["focused_gate"]),
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
        "schedule_profile_totals": dict(
            Counter(
                profile
                for row in rows
                for profile, count in row["schedule_counts"].items()
                for _ in range(int(count))
            )
        ),
        "rollout_profile_factories": {
            profile: factory.__name__
            for profile, factory in resolved_factories.items()
        },
        "rows": rows,
    }


def evaluate_state(
    benchmark: Mapping[str, Any],
    audit_4: Mapping[str, Any],
    audit_5: Mapping[str, Any],
    *,
    belief_model: PublicOpponentBeliefModel,
    validator_config: Any,
    slots: int,
    profile_factories: Mapping[str, Callable[[], SimulationPolicy]],
) -> dict[str, Any]:
    state_id = str(benchmark["state_before_hash"])
    if str(audit_4.get("state_before_hash")) != state_id or str(audit_5.get("state_before_hash")) != state_id:
        raise ValueError(f"public_belief_validator_state_identity:{state_id}")
    view = public_view_from_dict(audit_4["public_state"])
    game = dict(audit_4["game"])
    rules = rules_for_room(
        wildcard_enabled=bool(game["wildcard_enabled"]),
        players=int(game["players"]),
    )
    evidence = public_opponent_features(view, rules)
    posterior = belief_model.posterior(
        evidence.features,
        public_actions=evidence.public_actions,
    )
    schedule = smooth_weighted_profile_schedule(posterior, slots=slots)
    factories = tuple(profile_factories[name] for name in schedule)
    ranking = list(benchmark.get("validation_coverage_ranking") or ())
    labels = [str(item["label"]) for item in ranking]
    priors = {str(item["label"]): float(item["heuristic_value"]) for item in ranking}
    preferred = str(
        benchmark.get("validation_preferred_label")
        or benchmark.get("baseline_selected_label")
        or ""
    )
    if len(labels) < 2 or preferred not in priors:
        raise ValueError(f"public_belief_validator_candidates:{state_id}")
    validator = DualBatchDiscardValidator(
        rollout_policy_factories=factories,
        config=validator_config,
    )
    started = time.perf_counter()
    error: str | None = None
    try:
        screened = validator.screen_discard(
            view,
            rules=rules,
            candidate_labels=labels,
            candidate_priors=priors,
        )
        validation = validator.confirm_discard(
            screened,
            preferred_label=preferred,
            confirmation_worlds=validator_config.confirmation_worlds,
        )
        selected = validation.selected_label
        complete = validation.complete
        health_errors = validation_health_errors(validation)
        health_details = validation_health_details(validation)
    except Exception as exc:  # pragma: no cover - report health path
        validation = None
        selected = preferred
        complete = False
        health_errors = 1
        error = f"{type(exc).__name__}:{exc}"
        health_details = [{"stage": "exception", "error": error}]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    worlds = [*_world_rewards(audit_4), *_world_rewards(audit_5)]
    truth = heldout_truth(worlds, labels=labels, preferred=preferred)
    anchor_selected = str(
        benchmark.get("validation_selected_label") or preferred
    )
    return {
        "state_before_hash": state_id,
        "opponent_stratum": str(benchmark.get("opponent_stratum") or ""),
        "public_actions": evidence.public_actions,
        "posterior": {key: round(value, 8) for key, value in posterior.items()},
        "schedule_counts": dict(Counter(schedule)),
        "preferred_label": preferred,
        "anchor_selected_label": anchor_selected,
        "candidate_selected_label": selected,
        "validation_complete": complete,
        "validation_confidence_override": bool(
            validation is not None and validation.confidence_override
        ),
        "elapsed_ms": round(elapsed_ms, 3),
        "health_errors": health_errors,
        "health_details": health_details,
        "validation_evidence": validation_evidence_summary(validation),
        "error": error,
        "heldout_best_label": truth["best_label"],
        "candidate": evaluate_selection(selected, preferred=preferred, truth=truth),
        "anchor": evaluate_selection(anchor_selected, preferred=preferred, truth=truth),
    }


def validation_evidence_summary(validation: Any | None) -> dict[str, Any] | None:
    if validation is None:
        return None
    return {
        "coverage_candidates": [
            candidate.to_dict() for candidate in validation.coverage.candidates
        ],
        "challengers": [
            {
                "challenger_label": evidence.challenger_label,
                "confidence_override": evidence.confidence_override,
                "override_basis": evidence.override_basis,
                "calibrated_advantage": evidence.calibrated_advantage,
                "first_paired_advantages": [
                    item.to_dict()
                    for item in evidence.first_confirmation.paired_advantages
                ],
                "second_paired_advantages": [
                    item.to_dict()
                    for item in evidence.second_confirmation.paired_advantages
                ],
                "combined_paired_advantages": [
                    item.to_dict()
                    for item in evidence.combined.paired_advantages
                ],
            }
            for evidence in validation.challenger_evidence
        ],
    }


def validation_health_errors(validation: Any) -> int:
    errors = int(not validation.complete)
    stages = [validation.coverage]
    for evidence in validation.challenger_evidence:
        stages.extend((evidence.first_confirmation, evidence.second_confirmation))
    for stage in stages:
        errors += int(stage.deadline_interruptions != 0)
        errors += int(stage.rollout_invariant_violations != 0)
        errors += int(stage.rollout_coverage_failures != 0)
        errors += int(stage.determinization_failures != 0)
    return errors


def validation_health_details(validation: Any) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    stages: list[tuple[str, Any, int]] = [
        ("coverage", validation.coverage, validation.expected_coverage_worlds)
    ]
    for index, evidence in enumerate(validation.challenger_evidence):
        stages.extend(
            (
                (
                    f"challenger_{index}_first",
                    evidence.first_confirmation,
                    validation.expected_confirmation_worlds,
                ),
                (
                    f"challenger_{index}_second",
                    evidence.second_confirmation,
                    validation.expected_confirmation_worlds,
                ),
            )
        )
    for name, stage, expected_worlds in stages:
        visits = [int(candidate.visits) for candidate in stage.candidates]
        row = {
            "stage": name,
            "expected_worlds": int(expected_worlds),
            "paired_determinizations": int(stage.paired_determinizations),
            "minimum_candidate_visits": min(visits, default=0),
            "maximum_candidate_visits": max(visits, default=0),
            "deadline_interruptions": int(stage.deadline_interruptions),
            "rollout_invariant_violations": int(stage.rollout_invariant_violations),
            "rollout_coverage_failures": int(stage.rollout_coverage_failures),
            "determinization_failures": int(stage.determinization_failures),
        }
        if (
            row["paired_determinizations"] != row["expected_worlds"]
            or row["minimum_candidate_visits"] != row["expected_worlds"]
            or any(
                row[key]
                for key in (
                    "deadline_interruptions",
                    "rollout_invariant_violations",
                    "rollout_coverage_failures",
                    "determinization_failures",
                )
            )
        ):
            details.append(row)
    if not validation.complete and not details:
        details.append({"stage": "validation", "incomplete_without_stage_error": True})
    return details


def focused_gate_failures(
    candidate: Mapping[str, Any],
    *,
    anchor: Mapping[str, Any],
    gate: Mapping[str, Any],
    health_errors: int,
    incomplete: int,
    states: int,
) -> list[str]:
    failures: list[str] = []
    if states != int(gate["required_states"]):
        failures.append("required_states")
    if health_errors > int(gate["maximum_health_errors"]):
        failures.append("health_errors")
    if bool(gate["require_all_validations_complete"]) and incomplete:
        failures.append("incomplete_validations")
    if int(candidate["harmful_overrides"]) > int(gate["maximum_harmful_overrides"]):
        failures.append("harmful_overrides")
    if int(candidate["missed_overrides"]) > int(gate["maximum_missed_beneficial_actions"]):
        failures.append("missed_beneficial_actions")
    if int(candidate["confidently_positive_overrides"]) < int(
        gate["minimum_confidently_positive_overrides"]
    ):
        failures.append("confidently_positive_overrides")
    if bool(gate["require_fewer_harmful_overrides_than_anchor"]) and int(
        candidate["harmful_overrides"]
    ) >= int(anchor["harmful_overrides"]):
        failures.append("harm_not_better_than_anchor")
    if bool(gate["require_lower_mean_regret_than_anchor"]) and float(
        candidate["mean_regret_to_heldout_best"]
    ) >= float(anchor["mean_regret_to_heldout_best"]):
        failures.append("regret_not_better_than_anchor")
    if float(candidate["improvement_confidence"]["lower_95"]) <= float(
        gate["minimum_mean_improvement_lower_95"]
    ):
        failures.append("mean_improvement_lower_95")
    return failures


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
