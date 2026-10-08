"""Evaluate a preregistered opponent-distribution split without new rollouts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.calibrate_discard_validation import (
    classify_online_action,
    summarize_deltas,
)
from tools.fit_discard_evidence_calibrator import predict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preregistration_bytes = args.preregistration.read_bytes()
    preregistration = json.loads(
        preregistration_bytes.decode("utf-8")
    )
    report = run_preregistered_split(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(
            preregistration_bytes
        ).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "ok",
                    "focused_gate_pass",
                    "states",
                    "candidate_harmful_overrides",
                    "anchor_harmful_overrides",
                    "candidate_confidently_positive_overrides",
                    "candidate_mean_improvement_over_preferred",
                    "candidate_mean_regret_to_heldout_best",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def run_preregistered_split(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    benchmark = json.loads(
        Path(inputs["benchmark"]).read_text(encoding="utf-8")
    )
    benchmark_rows = list(benchmark.get("rows") or ())
    benchmark_by_state = _unique_index(
        benchmark_rows,
        key="state_before_hash",
        error_prefix="opponent_split_benchmark",
    )
    batches = {
        index: _load_audits(Path(inputs[f"batch_{index}"]))
        for index in range(1, 6)
    }
    model_report = json.loads(
        Path(inputs["frozen_model"]).read_text(encoding="utf-8")
    )
    model = dict(model_report["model"])
    model_features = tuple(model_report.get("feature_names") or ())
    if len(model_features) != len(model.get("coefficients") or ()):
        raise ValueError("opponent_split_model_feature_mismatch")
    expected_states = set(benchmark_by_state)
    for index, audits in batches.items():
        if set(audits) != expected_states:
            raise ValueError(
                "opponent_split_batch_state_mismatch:"
                f"{index}:{len(audits)}:{len(expected_states)}"
            )

    rows = [
        evaluate_state(
            benchmark_by_state[state_id],
            tuple(batches[index][state_id] for index in range(1, 6)),
            model=model,
            decision_margin=float(model["decision_margin"]),
        )
        for state_id in sorted(expected_states)
    ]
    candidate = summarize_policy(rows, prefix="candidate")
    anchor = summarize_policy(rows, prefix="anchor")
    gate = dict(preregistration["focused_gate"])
    failures = focused_gate_failures(
        candidate,
        anchor=anchor,
        gate=gate,
        health_errors=sum(int(row["health_errors"]) for row in rows),
        states=len(rows),
    )
    return {
        "ok": bool(rows)
        and all(int(row["health_errors"]) == 0 for row in rows),
        "schema_version": "opponent-distribution-oracle-split-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "frozen_model": str(inputs["frozen_model"]),
        "frozen_model_feature_names": list(model_features),
        "decision_margin": float(model["decision_margin"]),
        "states": len(rows),
        "health_errors": sum(int(row["health_errors"]) for row in rows),
        "candidate_harmful_overrides": candidate["harmful_overrides"],
        "anchor_harmful_overrides": anchor["harmful_overrides"],
        "candidate_confidently_positive_overrides": candidate[
            "confidently_positive_overrides"
        ],
        "candidate_mean_improvement_over_preferred": candidate[
            "mean_improvement_over_preferred"
        ],
        "candidate_mean_regret_to_heldout_best": candidate[
            "mean_regret_to_heldout_best"
        ],
        "candidate": candidate,
        "anchor": anchor,
        "focused_gate": gate,
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
        "rows": rows,
    }


def evaluate_state(
    benchmark: Mapping[str, Any],
    audits: Sequence[Mapping[str, Any]],
    *,
    model: Mapping[str, Any],
    decision_margin: float,
) -> dict[str, Any]:
    if len(audits) != 5:
        raise ValueError("opponent_split_requires_five_batches")
    state_id = str(benchmark["state_before_hash"])
    if any(str(audit.get("state_before_hash")) != state_id for audit in audits):
        raise ValueError(f"opponent_split_state_mismatch:{state_id}")
    preferred = str(
        benchmark.get("validation_preferred_label")
        or benchmark.get("baseline_selected_label")
        or ""
    )
    anchor_selected = str(benchmark.get("selected_label") or preferred)
    worlds = [_world_rewards(audit) for audit in audits]
    candidate_sets = [set(worlds_by_batch[0]) for worlds_by_batch in worlds]
    if (
        not preferred
        or not candidate_sets
        or any(not batch for batch in worlds)
        or any(labels != candidate_sets[0] for labels in candidate_sets[1:])
        or preferred not in candidate_sets[0]
    ):
        raise ValueError(f"opponent_split_candidate_mismatch:{state_id}")
    labels = sorted(candidate_sets[0])
    heuristics = _heuristics(audits[0])
    if set(heuristics) != set(labels):
        raise ValueError(f"opponent_split_heuristic_mismatch:{state_id}")
    opponents = {
        _canonical_opponent_name(
            str(world.get("opponent_policy") or "")
        )
        for audit in audits
        for world in audit.get("paired_worlds") or ()
    }
    expected_opponent = _canonical_opponent_name(
        str(benchmark.get("opponent_stratum") or "")
    )
    health_errors = _health_errors(audits)
    if opponents != {expected_opponent}:
        health_errors += 1

    coverage = worlds[0][:48]
    first = worlds[1][:112]
    second = worlds[2][:112]
    heldout = [*worlds[3], *worlds[4]]
    coverage_means = {
        label: mean(world[label] for world in coverage)
        for label in labels
    }
    coverage_ranking = sorted(
        labels,
        key=lambda label: (
            coverage_means[label],
            heuristics[label],
            label,
        ),
        reverse=True,
    )
    challengers = select_challengers(
        coverage_ranking,
        preferred_label=preferred,
        heuristics=heuristics,
        coverage_means=coverage_means,
        limit=5,
        prior_slots=2,
    )
    evidence = [
        build_evidence(
            label,
            preferred=preferred,
            labels=labels,
            coverage_ranking=coverage_ranking,
            coverage_means=coverage_means,
            heuristics=heuristics,
            first=first,
            second=second,
            model=model,
        )
        for label in challengers
    ]
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
                float(coverage_means[str(item["challenger_label"])]),
                str(item["challenger_label"]),
            ),
        )["challenger_label"]
        if eligible
        else preferred
    )
    truth = heldout_truth(
        heldout,
        labels=labels,
        preferred=preferred,
    )
    if selected not in truth["by_label"] or anchor_selected not in truth["by_label"]:
        raise ValueError(f"opponent_split_selected_missing:{state_id}")
    candidate_eval = evaluate_selection(
        str(selected),
        preferred=preferred,
        truth=truth,
    )
    anchor_eval = evaluate_selection(
        anchor_selected,
        preferred=preferred,
        truth=truth,
    )
    return {
        "state_before_hash": state_id,
        "public_view_id": str(benchmark.get("public_view_id") or ""),
        "validation_group_id": str(
            benchmark.get("validation_group_id") or state_id
        ),
        "opponent_stratum": expected_opponent,
        "preferred_label": preferred,
        "anchor_selected_label": anchor_selected,
        "candidate_selected_label": str(selected),
        "challenger_labels": list(challengers),
        "candidate_evidence": evidence,
        "heldout_best_label": truth["best_label"],
        "heldout_worlds": len(heldout),
        "candidate": candidate_eval,
        "anchor": anchor_eval,
        "health_errors": health_errors,
    }


def build_evidence(
    challenger: str,
    *,
    preferred: str,
    labels: Sequence[str],
    coverage_ranking: Sequence[str],
    coverage_means: Mapping[str, float],
    heuristics: Mapping[str, float],
    first: Sequence[Mapping[str, float]],
    second: Sequence[Mapping[str, float]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons = max(1, min(5, len(labels) - 1))
    first_deltas = [world[challenger] - world[preferred] for world in first]
    second_deltas = [world[challenger] - world[preferred] for world in second]
    combined = summarize_deltas(
        [*first_deltas, *second_deltas],
        comparisons=comparisons,
    )
    heuristic_ranking = sorted(
        labels,
        key=lambda label: (
            heuristics[label],
            coverage_means[label],
            label,
        ),
        reverse=True,
    )
    features = (
        float(combined["mean_delta"]),
        float(combined["standard_error"]),
        abs(mean(first_deltas) - mean(second_deltas)),
        min(mean(first_deltas), mean(second_deltas)),
        coverage_means[challenger] - coverage_means[preferred],
        (heuristics[challenger] - heuristics[preferred]) / 1000.0,
        1.0 / (coverage_ranking.index(challenger) + 1),
        1.0 / (heuristic_ranking.index(challenger) + 1),
        1.0,
    )
    model_width = len(model.get("coefficients") or ())
    if model_width != 9:
        raise ValueError(f"opponent_split_expected_v81_model:{model_width}")
    return {
        "challenger_label": challenger,
        "first_mean_delta": round(mean(first_deltas), 8),
        "second_mean_delta": round(mean(second_deltas), 8),
        "combined_mean_delta": float(combined["mean_delta"]),
        "combined_standard_error": float(combined["standard_error"]),
        "combined_lower_confidence_bound": float(
            combined["lower_confidence_bound"]
        ),
        "coverage_reward_delta": round(
            coverage_means[challenger] - coverage_means[preferred],
            8,
        ),
        "heuristic_value_delta": round(
            heuristics[challenger] - heuristics[preferred],
            8,
        ),
        "coverage_rank": coverage_ranking.index(challenger) + 1,
        "heuristic_rank": heuristic_ranking.index(challenger) + 1,
        "calibrated_advantage": round(
            predict(model, features[:model_width]),
            12,
        ),
    }


def heldout_truth(
    worlds: Sequence[Mapping[str, float]],
    *,
    labels: Sequence[str],
    preferred: str,
) -> dict[str, Any]:
    comparisons = max(1, len(labels) - 1)
    by_label = {
        label: summarize_deltas(
            [world[label] - world[preferred] for world in worlds],
            comparisons=comparisons,
        )
        for label in labels
    }
    best = max(
        labels,
        key=lambda label: (
            mean(world[label] for world in worlds),
            label,
        ),
    )
    return {
        "best_label": best,
        "best_vs_preferred": by_label[best],
        "by_label": by_label,
    }


def evaluate_selection(
    selected: str,
    *,
    preferred: str,
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    selected_truth = truth["by_label"][selected]
    best_delta = float(truth["best_vs_preferred"]["mean_delta"])
    selected_delta = float(selected_truth["mean_delta"])
    return {
        "selected_label": selected,
        "classification": classify_online_action(
            preferred=preferred,
            online_selected=selected,
            pooled_best=str(truth["best_label"]),
            preferred_confidently_suboptimal=bool(
                truth["best_vs_preferred"]["confidently_positive"]
            ),
            selected_vs_preferred=selected_truth,
        ),
        "improvement_over_preferred": selected_delta,
        "regret_to_heldout_best": best_delta - selected_delta,
        "selected_vs_preferred": selected_truth,
    }


def summarize_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any]:
    evaluations = [row[prefix] for row in rows]
    classifications = Counter(
        str(row["classification"]) for row in evaluations
    )
    improvements = [
        float(row["improvement_over_preferred"]) for row in evaluations
    ]
    regrets = [float(row["regret_to_heldout_best"]) for row in evaluations]
    overrides = [
        row
        for source, row in zip(rows, evaluations)
        if str(row["selected_label"]) != str(source["preferred_label"])
    ]
    return {
        "states": len(rows),
        "overrides": len(overrides),
        "correct_overrides": classifications["correct_override"],
        "beneficial_overrides": classifications["beneficial_override"],
        "unproven_overrides": classifications["unproven_override"],
        "harmful_overrides": classifications["harmful_override"],
        "missed_overrides": classifications["missed_override"],
        "correct_keeps": classifications["correct_keep"],
        "confidently_positive_overrides": sum(
            bool(row["selected_vs_preferred"]["confidently_positive"])
            for row in overrides
        ),
        "confidently_negative_overrides": sum(
            bool(row["selected_vs_preferred"]["confidently_negative"])
            for row in overrides
        ),
        "mean_improvement_over_preferred": round(
            mean(improvements) if improvements else 0.0,
            8,
        ),
        "improvement_confidence": _mean_confidence(improvements),
        "mean_regret_to_heldout_best": round(
            mean(regrets) if regrets else 0.0,
            8,
        ),
    }


def focused_gate_failures(
    candidate: Mapping[str, Any],
    *,
    anchor: Mapping[str, Any],
    gate: Mapping[str, Any],
    health_errors: int,
    states: int,
) -> list[str]:
    failures: list[str] = []
    if states != int(gate["states"]):
        failures.append(f"states:{states}!={gate['states']}")
    harmful = int(candidate["harmful_overrides"])
    if harmful > int(gate["maximum_heldout_harmful_overrides"]):
        failures.append(
            "harmful_overrides:"
            f"{harmful}>{gate['maximum_heldout_harmful_overrides']}"
        )
    if harmful >= int(anchor["harmful_overrides"]):
        failures.append(
            "harm_not_strictly_better_than_anchor:"
            f"{harmful}>={anchor['harmful_overrides']}"
        )
    if int(candidate["confidently_negative_overrides"]) > int(
        gate["maximum_confidently_negative_overrides"]
    ):
        failures.append("confidently_negative_overrides")
    if int(candidate["confidently_positive_overrides"]) < int(
        gate["minimum_preserved_confidently_positive_overrides"]
    ):
        failures.append("confidently_positive_overrides")
    if float(candidate["mean_improvement_over_preferred"]) < float(
        gate["minimum_mean_improvement_over_preferred"]
    ):
        failures.append("mean_improvement_over_preferred")
    if float(candidate["mean_regret_to_heldout_best"]) > float(
        gate["maximum_mean_regret_to_heldout_best"]
    ):
        failures.append("mean_regret_to_heldout_best")
    if bool(gate["require_zero_search_health_errors"]) and health_errors:
        failures.append(f"search_health_errors:{health_errors}")
    return failures


def select_challengers(
    coverage_ranking: Sequence[str],
    *,
    preferred_label: str,
    heuristics: Mapping[str, float],
    coverage_means: Mapping[str, float],
    limit: int,
    prior_slots: int,
) -> tuple[str, ...]:
    available = [
        label for label in coverage_ranking if label != preferred_label
    ]
    resolved_limit = min(max(1, int(limit)), len(available))
    resolved_prior_slots = min(
        max(0, int(prior_slots)),
        resolved_limit - 1 if resolved_limit > 1 else 0,
    )
    selected = list(available[: resolved_limit - resolved_prior_slots])
    prior_ranked = sorted(
        available,
        key=lambda label: (
            heuristics[label],
            coverage_means[label],
            label,
        ),
        reverse=True,
    )
    for label in (*prior_ranked, *available):
        if label not in selected:
            selected.append(label)
        if len(selected) >= resolved_limit:
            break
    return tuple(selected)


def _load_audits(path: Path) -> dict[str, dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    indexed: dict[str, dict[str, Any]] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            audit = json.loads(line)
            state_id = str(audit.get("state_before_hash") or "")
            if not state_id or state_id in indexed:
                raise ValueError(f"opponent_split_duplicate_state:{path}")
            indexed[state_id] = audit
    return indexed


def _world_rewards(audit: Mapping[str, Any]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for world in audit.get("paired_worlds") or ():
        outcomes = {
            _action_label(str(item["candidate_key"])): float(item["reward"])
            for item in world.get("outcomes") or ()
        }
        if not outcomes:
            raise ValueError("opponent_split_world_without_outcomes")
        rows.append(outcomes)
    return rows


def _heuristics(audit: Mapping[str, Any]) -> dict[str, float]:
    return {
        _action_label(str(item.get("key") or item.get("label") or "")):
        float(item.get("heuristic_value") or 0.0)
        for item in audit.get("candidate_stats") or ()
    }


def _health_errors(audits: Sequence[Mapping[str, Any]]) -> int:
    errors = 0
    for audit in audits:
        health = dict(audit.get("search_health") or {})
        errors += int(not bool(audit.get("complete")))
        errors += int(int(audit.get("completed_paired_worlds") or 0) != 128)
        errors += int(int(health.get("deadline_interruptions") or 0) != 0)
        errors += int(int(health.get("determinization_failures") or 0) != 0)
        errors += int(int(health.get("rollout_invariant_violations") or 0) != 0)
        errors += int(int(health.get("rollout_coverage_failures") or 0) != 0)
    return errors


def _unique_index(
    rows: Iterable[Mapping[str, Any]],
    *,
    key: str,
    error_prefix: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in indexed:
            raise ValueError(f"{error_prefix}_duplicate:{value}")
        indexed[value] = row
    return indexed


def _action_label(key: str) -> str:
    return key.split(":", 1)[1] if ":" in key else key


def _canonical_opponent_name(name: str) -> str:
    aliases = {
        "information_set_search_v2": "information_set_search",
    }
    return aliases.get(name, name)


def _mean_confidence(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    center = mean(values)
    standard_error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    radius = 1.96 * standard_error
    return {
        "samples": len(values),
        "mean": round(center, 8),
        "lower_95": round(center - radius, 8),
        "upper_95": round(center + radius, 8),
    }


if __name__ == "__main__":
    raise SystemExit(main())
