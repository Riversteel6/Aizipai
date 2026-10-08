"""Attribute public-belief value error with crossed profile/world rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.dual_validated_candidate import (
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy,
)
from ai.full_game_simulator import SimulationPolicy
from ai.ismcts import (
    PairedWorldOutcome,
    RootISMCTSConfig,
    RootISMCTSPolicy,
    public_view_from_dict,
)
from ai.opponent_belief import PublicOpponentBeliefModel, public_opponent_features
from engine.rules import rules_for_room
from tools.calibrate_discard_validation import summarize_deltas
from tools.evaluate_anchored_structural_discard_proxies import sha256_file
from tools.evaluate_hybrid_response_proxies import HYBRID_RESPONSE_PROXY_TYPES
from tools.evaluate_opponent_distribution_split import (
    _load_audits,
    _world_rewards,
    evaluate_selection,
    heldout_truth,
    select_challengers,
    summarize_policy,
)
from tools.fit_discard_evidence_calibrator import predict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_diagnostic(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(raw).hexdigest(),
        workers=max(1, int(args.workers)),
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
                "integrity_gate_pass": report["integrity_gate_pass"],
                "states": report["states"],
                "health_errors": report["health_errors"],
                "world_identity_mismatches": report[
                    "world_identity_mismatches"
                ],
                "attribution": report["attribution"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] and report["integrity_gate_pass"] else 1


def run_diagnostic(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
    workers: int,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    for key in tuple(inputs):
        if key.endswith("_sha256"):
            continue
        expected_key = f"{key}_sha256"
        if expected_key in inputs:
            actual = sha256_file(Path(str(inputs[key])))
            if actual != str(inputs[expected_key]):
                raise ValueError(f"crossed_value_{key}_hash_mismatch")

    profiles = tuple(str(item) for item in preregistration["profiles"])
    if set(profiles) != set(HYBRID_RESPONSE_PROXY_TYPES):
        raise ValueError("crossed_value_profile_mapping_mismatch")
    stages = tuple(dict(item) for item in preregistration["stages"])
    if {str(item["id"]) for item in stages} != {
        "coverage",
        "confirmation_1",
        "confirmation_2",
    }:
        raise ValueError("crossed_value_stage_contract")

    schedule_report = _read_json(Path(str(inputs["schedule_diagnostic"])))
    exact_oracle = _read_json(Path(str(inputs["exact_oracle_report"])))
    benchmark = _read_json(Path(str(inputs["benchmark"])))
    belief_report = _read_json(Path(str(inputs["belief_report"])))
    _verify_capability_report(inputs, "discard_proxy_capability_report")
    _verify_capability_report(inputs, "response_proxy_capability_report")

    state_ids = tuple(str(item) for item in preregistration["state_ids"])
    schedule_rows = _index_schedule_rows(schedule_report, state_ids=state_ids)
    benchmark_rows = _unique_index(benchmark["rows"], key="state_before_hash")
    oracle_rows = _unique_index(exact_oracle["rows"], key="state_before_hash")
    batch_4 = _load_audits(Path(str(inputs["heldout_batch_4"])))
    batch_5 = _load_audits(Path(str(inputs["heldout_batch_5"])))
    for source_name, source in (
        ("benchmark", benchmark_rows),
        ("oracle", oracle_rows),
        ("batch_4", batch_4),
        ("batch_5", batch_5),
    ):
        missing = set(state_ids) - set(source)
        if missing:
            raise ValueError(
                f"crossed_value_{source_name}_state_mismatch:{len(missing)}"
            )

    belief_model = PublicOpponentBeliefModel.from_dict(
        belief_report["final_model"]
    )
    calibrator = (
        ProfessionalParallelMultiOpponentRobustV81ResearchPolicy
        .evidence_calibrator_config
    )
    calibrator_model = {
        "feature_mean": calibrator.feature_mean,
        "feature_scale": calibrator.feature_scale,
        "intercept": calibrator.intercept,
        "coefficients": calibrator.coefficients,
    }

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        for index, state_id in enumerate(state_ids, start=1):
            row = evaluate_crossed_state(
                benchmark_rows[state_id],
                schedule_rows[state_id],
                oracle_rows[state_id],
                batch_4[state_id],
                batch_5[state_id],
                belief_model=belief_model,
                calibrator_model=calibrator_model,
                decision_margin=float(calibrator.decision_margin),
                profiles=profiles,
                stages=stages,
                profile_factories=HYBRID_RESPONSE_PROXY_TYPES,
                executor=executor,
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "progress": index,
                        "total": len(state_ids),
                        "state": state_id[:8],
                        "elapsed_ms": row["elapsed_ms"],
                        "health_errors": sum(
                            int(item["health_errors"]) for item in rows
                        ),
                        "world_identity_mismatches": sum(
                            int(item["world_identity_mismatches"])
                            for item in rows
                        ),
                    }
                ),
                flush=True,
            )

    health_errors = sum(int(row["health_errors"]) for row in rows)
    identity_mismatches = sum(
        int(row["world_identity_mismatches"]) for row in rows
    )
    gate = dict(preregistration["integrity_gate"])
    failures = integrity_gate_failures(
        rows,
        profiles=profiles,
        stages=stages,
        health_errors=health_errors,
        world_identity_mismatches=identity_mismatches,
        gate=gate,
    )
    attribution = summarize_attribution(rows)
    return {
        "ok": bool(rows) and health_errors == 0 and identity_mismatches == 0,
        "schema_version": "crossed-profile-world-value-attribution-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "states": len(rows),
        "profiles": list(profiles),
        "stages": stages,
        "workers": workers,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "health_errors": health_errors,
        "world_identity_mismatches": identity_mismatches,
        "summaries": {
            name: summarize_policy(rows, prefix=name)
            for name in (
                "schedule_21",
                "schedule_112",
                "uniform_crossed",
                "posterior_crossed",
                "actual_profile_proxy",
                "exact_actual_oracle",
            )
        },
        "attribution": attribution,
        "integrity_gate": gate,
        "integrity_gate_pass": not failures,
        "integrity_gate_failures": failures,
        "rows": rows,
    }


def evaluate_crossed_state(
    benchmark: Mapping[str, Any],
    schedule_rows: Mapping[int, Mapping[str, Any]],
    oracle: Mapping[str, Any],
    audit_4: Mapping[str, Any],
    audit_5: Mapping[str, Any],
    *,
    belief_model: PublicOpponentBeliefModel,
    calibrator_model: Mapping[str, Any],
    decision_margin: float,
    profiles: Sequence[str],
    stages: Sequence[Mapping[str, Any]],
    profile_factories: Mapping[str, Callable[[], SimulationPolicy]],
    executor: ProcessPoolExecutor,
) -> dict[str, Any]:
    state_id = str(benchmark["state_before_hash"])
    view = public_view_from_dict(audit_4["public_state"])
    game = dict(audit_4["game"])
    rules = rules_for_room(
        wildcard_enabled=bool(game["wildcard_enabled"]),
        players=int(game["players"]),
    )
    ranking = list(benchmark.get("validation_coverage_ranking") or ())
    labels = tuple(str(item["label"]) for item in ranking)
    priors = {str(item["label"]): float(item["heuristic_value"]) for item in ranking}
    preferred = str(
        benchmark.get("validation_preferred_label")
        or benchmark.get("baseline_selected_label")
        or ""
    )
    if len(labels) < 2 or preferred not in priors or len(set(labels)) != len(labels):
        raise ValueError(f"crossed_value_candidate_contract:{state_id}")

    evidence = public_opponent_features(view, rules)
    posterior = belief_model.posterior(
        evidence.features,
        public_actions=evidence.public_actions,
    )
    actual_profile = str(schedule_rows[21]["opponent_stratum"])
    if actual_profile not in profiles:
        raise ValueError(f"crossed_value_actual_profile:{state_id}")

    started = time.perf_counter()
    stage_results: dict[str, dict[str, Any]] = {}
    futures = {}
    for stage in stages:
        stage_id = str(stage["id"])
        for profile in profiles:
            futures[(stage_id, profile)] = executor.submit(
                _profile_search_task,
                {
                    "view": view,
                    "rules": rules,
                    "labels": labels,
                    "priors": priors,
                    "preferred": preferred,
                    "worlds": int(stage["worlds"]),
                    "seed": int(stage["seed"]),
                    "factory": profile_factories[profile],
                },
            )
    for stage in stages:
        stage_id = str(stage["id"])
        stage_results[stage_id] = {
            profile: futures[(stage_id, profile)].result()
            for profile in profiles
        }

    health_errors = sum(
        search_health_errors(result, expected_worlds=int(stage["worlds"]))
        for stage in stages
        for result in stage_results[str(stage["id"])].values()
    )
    profile_worlds = {
        stage_id: {
            profile: result.paired_worlds
            for profile, result in results.items()
        }
        for stage_id, results in stage_results.items()
    }
    identity_mismatches = sum(
        crossed_world_identity_mismatches(worlds)
        for worlds in profile_worlds.values()
    )

    uniform = {profile: 1.0 / len(profiles) for profile in profiles}
    actual = {
        profile: float(profile == actual_profile) for profile in profiles
    }
    decisions = {
        "uniform_crossed": weighted_decision(
            profile_worlds,
            weights=uniform,
            labels=labels,
            preferred=preferred,
            priors=priors,
            calibrator_model=calibrator_model,
            decision_margin=decision_margin,
        ),
        "posterior_crossed": weighted_decision(
            profile_worlds,
            weights=posterior,
            labels=labels,
            preferred=preferred,
            priors=priors,
            calibrator_model=calibrator_model,
            decision_margin=decision_margin,
        ),
        "actual_profile_proxy": weighted_decision(
            profile_worlds,
            weights=actual,
            labels=labels,
            preferred=preferred,
            priors=priors,
            calibrator_model=calibrator_model,
            decision_margin=decision_margin,
        ),
    }
    profile_decisions = {
        profile: weighted_decision(
            profile_worlds,
            weights={name: float(name == profile) for name in profiles},
            labels=labels,
            preferred=preferred,
            priors=priors,
            calibrator_model=calibrator_model,
            decision_margin=decision_margin,
        )["selected_label"]
        for profile in profiles
    }

    heldout_worlds = [*_world_rewards(audit_4), *_world_rewards(audit_5)]
    truth = heldout_truth(heldout_worlds, labels=labels, preferred=preferred)
    evaluation = {
        name: evaluate_selection(
            str(decision["selected_label"]),
            preferred=preferred,
            truth=truth,
        )
        for name, decision in decisions.items()
    }
    evaluation["schedule_21"] = evaluate_selection(
        str(schedule_rows[21]["selected_label"]),
        preferred=preferred,
        truth=truth,
    )
    evaluation["schedule_112"] = evaluate_selection(
        str(schedule_rows[112]["selected_label"]),
        preferred=preferred,
        truth=truth,
    )
    evaluation["exact_actual_oracle"] = evaluate_selection(
        str(oracle["candidate_selected_label"]),
        preferred=preferred,
        truth=truth,
    )

    return {
        "state_before_hash": state_id,
        "opponent_stratum": actual_profile,
        "public_actions": evidence.public_actions,
        "posterior": {key: round(value, 8) for key, value in posterior.items()},
        "preferred_label": preferred,
        "legal_labels": list(labels),
        "heldout_best_label": truth["best_label"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "health_errors": health_errors,
        "world_identity_mismatches": identity_mismatches,
        "profile_decisions": profile_decisions,
        "profile_decision_diversity": len(set(profile_decisions.values())),
        "weighted_decisions": decisions,
        **evaluation,
    }


def _profile_search_task(payload: Mapping[str, Any]):
    labels = tuple(str(item) for item in payload["labels"])
    worlds = int(payload["worlds"])
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=3_600_000,
            max_iterations=worlds * len(labels),
            max_candidates=len(labels),
            skip_search_gap=math.inf,
            rollout_max_turns=120,
            require_confident_override=False,
            min_confidence_pairs=min(32, worlds),
            minimum_confident_advantage=0.0,
            complete_first_paired_batch=True,
            require_complete_iteration_budget_for_override=True,
            record_paired_worlds=True,
            seed=int(payload["seed"]),
        ),
        rollout_policy_factories=(payload["factory"],),
    )
    return policy.search_discard(
        payload["view"],
        rules=dict(payload["rules"]),
        candidate_labels=labels,
        candidate_priors=dict(payload["priors"]),
        force_search=True,
        paired_candidates=True,
        preferred_label=str(payload["preferred"]),
    )


def weighted_decision(
    profile_worlds: Mapping[str, Mapping[str, Sequence[PairedWorldOutcome]]],
    *,
    weights: Mapping[str, float],
    labels: Sequence[str],
    preferred: str,
    priors: Mapping[str, float],
    calibrator_model: Mapping[str, Any],
    decision_margin: float,
) -> dict[str, Any]:
    coverage = crossed_reward_batches(
        profile_worlds["coverage"], weights=weights, labels=labels
    )
    first = crossed_reward_batches(
        profile_worlds["confirmation_1"], weights=weights, labels=labels
    )
    second = crossed_reward_batches(
        profile_worlds["confirmation_2"], weights=weights, labels=labels
    )
    coverage_means = {
        label: fmean(batch[label] for batch in coverage) for label in labels
    }
    coverage_ranking = sorted(
        labels,
        key=lambda label: (coverage_means[label], priors[label], label),
        reverse=True,
    )
    challengers = select_challengers(
        coverage_ranking,
        preferred_label=preferred,
        heuristics=priors,
        coverage_means=coverage_means,
        limit=5,
        prior_slots=2,
    )
    heuristic_ranking = sorted(
        labels,
        key=lambda label: (priors[label], coverage_means[label], label),
        reverse=True,
    )
    evidence = []
    comparisons = max(1, len(challengers))
    for challenger in challengers:
        first_deltas = [
            batch[challenger] - batch[preferred] for batch in first
        ]
        second_deltas = [
            batch[challenger] - batch[preferred] for batch in second
        ]
        combined = summarize_deltas(
            [*first_deltas, *second_deltas], comparisons=comparisons
        )
        features = (
            float(combined["mean_delta"]),
            float(combined["standard_error"]),
            abs(fmean(first_deltas) - fmean(second_deltas)),
            min(fmean(first_deltas), fmean(second_deltas)),
            coverage_means[challenger] - coverage_means[preferred],
            (priors[challenger] - priors[preferred]) / 1000.0,
            1.0 / (coverage_ranking.index(challenger) + 1),
            1.0 / (heuristic_ranking.index(challenger) + 1),
            1.0,
        )
        evidence.append(
            {
                "challenger_label": challenger,
                "first_mean_delta": round(fmean(first_deltas), 8),
                "second_mean_delta": round(fmean(second_deltas), 8),
                "combined_mean_delta": float(combined["mean_delta"]),
                "combined_standard_error": float(combined["standard_error"]),
                "combined_lower_confidence_bound": float(
                    combined["lower_confidence_bound"]
                ),
                "coverage_reward_delta": round(
                    coverage_means[challenger] - coverage_means[preferred], 8
                ),
                "coverage_rank": coverage_ranking.index(challenger) + 1,
                "heuristic_rank": heuristic_ranking.index(challenger) + 1,
                "calibrated_advantage": round(
                    predict(calibrator_model, features), 12
                ),
            }
        )
    eligible = [
        item
        for item in evidence
        if float(item["calibrated_advantage"]) > decision_margin
    ]
    selected = (
        max(
            eligible,
            key=lambda item: (
                float(item["calibrated_advantage"]),
                float(item["combined_lower_confidence_bound"]),
                float(item["combined_mean_delta"]),
                coverage_means[str(item["challenger_label"])],
                str(item["challenger_label"]),
            ),
        )["challenger_label"]
        if eligible
        else preferred
    )
    return {
        "weights": {key: round(float(value), 8) for key, value in weights.items()},
        "selected_label": str(selected),
        "challenger_labels": list(challengers),
        "evidence": evidence,
    }


def crossed_reward_batches(
    profile_worlds: Mapping[str, Sequence[PairedWorldOutcome]],
    *,
    weights: Mapping[str, float],
    labels: Sequence[str],
) -> list[dict[str, float]]:
    profiles = tuple(sorted(profile_worlds))
    if set(profiles) != set(weights):
        raise ValueError("crossed_value_weight_profile_mismatch")
    total = sum(max(0.0, float(weights[profile])) for profile in profiles)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("crossed_value_invalid_weights")
    normalized = {
        profile: max(0.0, float(weights[profile])) / total
        for profile in profiles
    }
    world_counts = {len(profile_worlds[profile]) for profile in profiles}
    if len(world_counts) != 1:
        raise ValueError("crossed_value_world_count_mismatch")
    batches = []
    for world_index in range(next(iter(world_counts), 0)):
        rows = {
            profile: _world_reward_map(profile_worlds[profile][world_index])
            for profile in profiles
        }
        if any(set(row) != set(labels) for row in rows.values()):
            raise ValueError("crossed_value_legal_action_mismatch")
        batches.append(
            {
                label: sum(
                    normalized[profile] * rows[profile][label]
                    for profile in profiles
                )
                for label in labels
            }
        )
    return batches


def crossed_world_identity_mismatches(
    profile_worlds: Mapping[str, Sequence[PairedWorldOutcome]],
) -> int:
    profiles = tuple(sorted(profile_worlds))
    counts = {len(profile_worlds[profile]) for profile in profiles}
    if len(counts) != 1:
        return max(counts, default=0)
    mismatches = 0
    for index in range(next(iter(counts), 0)):
        identities = {
            (
                profile_worlds[profile][index].world_index,
                profile_worlds[profile][index].world_fingerprint,
                profile_worlds[profile][index].rollout_seed,
            )
            for profile in profiles
        }
        mismatches += int(len(identities) != 1)
    return mismatches


def search_health_errors(result: Any, *, expected_worlds: int) -> int:
    return sum(
        (
            int(not result.used_search),
            int(result.paired_determinizations != expected_worlds),
            int(len(result.paired_worlds) != expected_worlds),
            int(result.determinization_failures != 0),
            int(result.deadline_interruptions != 0),
            int(result.rollout_invariant_violations != 0),
            int(result.rollout_coverage_failures != 0),
            int(any(candidate.visits != expected_worlds for candidate in result.candidates)),
        )
    )


def summarize_attribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    schedule_changes = sum(
        row["schedule_112"]["selected_label"]
        != row["posterior_crossed"]["selected_label"]
        for row in rows
    )
    belief_changes = sum(
        row["posterior_crossed"]["selected_label"]
        != row["actual_profile_proxy"]["selected_label"]
        for row in rows
    )
    proxy_changes = sum(
        row["actual_profile_proxy"]["selected_label"]
        != row["exact_actual_oracle"]["selected_label"]
        for row in rows
    )
    return {
        "factory_world_assignment_selected_changes": schedule_changes,
        "factory_world_assignment_material": schedule_changes > 0,
        "public_belief_selected_changes": belief_changes,
        "public_belief_material": belief_changes > 0,
        "proxy_to_exact_selected_changes": proxy_changes,
        "proxy_to_exact_material": proxy_changes > 0,
        "states_with_profile_sensitive_proxy_actions": sum(
            int(row["profile_decision_diversity"] > 1) for row in rows
        ),
        "mean_regret": {
            name: round(
                fmean(float(row[name]["regret_to_heldout_best"]) for row in rows),
                8,
            )
            for name in (
                "schedule_21",
                "schedule_112",
                "uniform_crossed",
                "posterior_crossed",
                "actual_profile_proxy",
                "exact_actual_oracle",
            )
        },
    }


def integrity_gate_failures(
    rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    stages: Sequence[Mapping[str, Any]],
    health_errors: int,
    world_identity_mismatches: int,
    gate: Mapping[str, Any],
) -> list[str]:
    failures = []
    if len(rows) != int(gate["required_states"]):
        failures.append("required_states")
    if len(profiles) != int(gate["required_profiles"]):
        failures.append("required_profiles")
    if len(stages) != int(gate["required_stages"]):
        failures.append("required_stages")
    if health_errors > int(gate["maximum_health_errors"]):
        failures.append("health_errors")
    if world_identity_mismatches > int(
        gate["maximum_world_identity_mismatches"]
    ):
        failures.append("world_identity_mismatches")
    if gate["require_all_legal_actions"] and any(
        not row.get("legal_labels") for row in rows
    ):
        failures.append("all_legal_actions")
    return failures


def _world_reward_map(world: PairedWorldOutcome) -> dict[str, float]:
    return {
        str(outcome.candidate_key): float(outcome.reward)
        for outcome in world.outcomes
    }


def _index_schedule_rows(
    report: Mapping[str, Any],
    *,
    state_ids: Sequence[str],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    indexed: dict[str, dict[int, Mapping[str, Any]]] = {
        state_id: {} for state_id in state_ids
    }
    for row in report.get("rows") or ():
        state_id = str(row.get("state_before_hash") or "")
        if state_id in indexed:
            slots = int(row.get("slots") or 0)
            if slots in indexed[state_id]:
                raise ValueError("crossed_value_duplicate_schedule_row")
            indexed[state_id][slots] = row
    if any(set(rows) != {21, 112} for rows in indexed.values()):
        raise ValueError("crossed_value_schedule_state_mismatch")
    return indexed


def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> dict[str, Mapping[str, Any]]:
    indexed = {str(row[key]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"crossed_value_duplicate_{key}")
    return indexed


def _verify_capability_report(inputs: Mapping[str, Any], key: str) -> None:
    report = _read_json(Path(str(inputs[key])))
    if not report.get("ok") or not report.get("focused_gate_pass"):
        raise ValueError(f"crossed_value_{key}_did_not_pass")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
