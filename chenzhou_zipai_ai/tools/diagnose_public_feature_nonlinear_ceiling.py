"""Test nonlinear separability of the frozen public-opponent feature tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.evaluate_anchored_structural_discard_proxies import sha256_file
from tools.fit_public_opponent_belief import (
    BeliefCase,
    collect_cases,
    fit_weighted_ridge_model,
    mean_confidence_interval,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_diagnostic(
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
                "integrity_gate_pass": report["integrity_gate_pass"],
                "diagnosis": report["diagnosis"],
                "paired_improvements": report["paired_improvements"],
                "baseline_reproduction_error": report[
                    "baseline_reproduction_error"
                ],
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
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    for key in ("source_report", "baseline_report"):
        if sha256_file(Path(str(inputs[key]))) != str(inputs[f"{key}_sha256"]):
            raise ValueError(f"nonlinear_ceiling_{key}_hash_mismatch")
    source = _read_json(Path(str(inputs["source_report"])))
    baseline_report = _read_json(Path(str(inputs["baseline_report"])))
    baseline_prereg = _read_json(Path(str(baseline_report["preregistration"])))
    baseline_spec = dict(baseline_prereg["model"])
    folds = int(baseline_spec["folds"])
    cases, game_profiles, fold_profiles = collect_cases(
        list(source.get("rows") or ()),
        folds=folds,
    )
    profiles = tuple(sorted(set(game_profiles.values())))
    profile_to_index = {profile: index for index, profile in enumerate(profiles)}
    reference_spec = dict(preregistration["reference_model"])

    predictions: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    state_predictions: list[dict[str, Any]] = []
    low_evidence_maximum = 0.0
    group_leakage = 0
    fold_reports = []
    for fold in range(folds):
        training = [case for case in cases if case.fold != fold]
        validation = [case for case in cases if case.fold == fold]
        training_games = {case.game_id for case in training}
        validation_games = {case.game_id for case in validation}
        overlap = training_games & validation_games
        group_leakage += len(overlap)

        baseline_model = fit_weighted_ridge_model(
            training,
            profiles=profiles,
            ridge=float(baseline_spec["ridge"]),
            temperature=float(baseline_spec["temperature"]),
            full_reliability_public_actions=int(
                baseline_spec["full_reliability_public_actions"]
            ),
        )
        weights = _per_game_weights(training)
        reference_model = HistGradientBoostingClassifier(
            learning_rate=float(reference_spec["learning_rate"]),
            max_iter=int(reference_spec["max_iter"]),
            max_leaf_nodes=int(reference_spec["max_leaf_nodes"]),
            min_samples_leaf=int(reference_spec["min_samples_leaf"]),
            l2_regularization=float(reference_spec["l2_regularization"]),
            early_stopping=bool(reference_spec["early_stopping"]),
            random_state=int(reference_spec["random_state"]),
        )
        reference_model.fit(
            np.asarray([case.features for case in training], dtype=float),
            np.asarray([profile_to_index[case.profile] for case in training]),
            sample_weight=weights,
        )
        raw_probabilities = reference_model.predict_proba(
            np.asarray([case.features for case in validation], dtype=float)
        )
        if tuple(int(item) for item in reference_model.classes_) != tuple(
            range(len(profiles))
        ):
            raise ValueError("nonlinear_ceiling_reference_class_order")

        fold_rows = []
        for case, raw_reference in zip(validation, raw_probabilities):
            baseline = baseline_model.posterior(
                case.features,
                public_actions=case.public_actions,
            )
            reference = reliability_blend(
                raw_reference,
                public_actions=case.public_actions,
                full_reliability_public_actions=int(
                    baseline_spec["full_reliability_public_actions"]
                ),
            )
            true_index = profile_to_index[case.profile]
            baseline_vector = np.asarray(
                [baseline[profile] for profile in profiles], dtype=float
            )
            metrics = {
                "baseline_log_loss": -math.log(
                    max(1e-15, float(baseline_vector[true_index]))
                ),
                "baseline_true_probability": float(
                    baseline_vector[true_index]
                ),
                "baseline_top1": int(
                    selected_probability_index(baseline_vector, profiles)
                    == true_index
                ),
                "reference_log_loss": -math.log(
                    max(1e-15, float(reference[true_index]))
                ),
                "reference_true_probability": float(reference[true_index]),
                "reference_top1": int(
                    selected_probability_index(reference, profiles)
                    == true_index
                ),
            }
            predictions[case.game_id].append(metrics)
            state_predictions.append(
                {
                    "game_id": case.game_id,
                    "profile": case.profile,
                    "public_actions": case.public_actions,
                    "baseline_selected": profiles[
                        selected_probability_index(baseline_vector, profiles)
                    ],
                    "reference_selected": profiles[
                        selected_probability_index(reference, profiles)
                    ],
                    **metrics,
                }
            )
            fold_rows.append(metrics)
            if case.public_actions <= int(baseline_spec["low_evidence_max_actions"]):
                low_evidence_maximum = max(
                    low_evidence_maximum,
                    float(np.max(reference)),
                )
        fold_reports.append(
            {
                "fold": fold,
                "training_games": len(training_games),
                "validation_games": len(validation_games),
                "training_states": len(training),
                "validation_states": len(validation),
                "group_overlap": len(overlap),
                "baseline_mean_log_loss": fmean(
                    float(row["baseline_log_loss"]) for row in fold_rows
                ),
                "reference_mean_log_loss": fmean(
                    float(row["reference_log_loss"]) for row in fold_rows
                ),
            }
        )

    per_game = [
        summarize_game_predictions(
            game_id,
            game_profiles[game_id],
            predictions[game_id],
        )
        for game_id in sorted(predictions)
    ]
    baseline_metrics = summarize_model(per_game, prefix="baseline")
    reference_metrics = summarize_model(per_game, prefix="reference")
    improvements = {
        "log_loss": mean_confidence_interval(
            [
                float(row["baseline_log_loss"])
                - float(row["reference_log_loss"])
                for row in per_game
            ]
        ),
        "true_probability": mean_confidence_interval(
            [
                float(row["reference_true_probability"])
                - float(row["baseline_true_probability"])
                for row in per_game
            ]
        ),
        "top1": mean_confidence_interval(
            [
                float(row["reference_top1"])
                - float(row["baseline_top1"])
                for row in per_game
            ]
        ),
    }
    reproduction_error = baseline_reproduction_error(
        baseline_metrics,
        baseline_report["metrics"],
    )
    diagnosis = classify_feature_ceiling(
        improvements,
        thresholds=preregistration["diagnostic_thresholds"],
    )
    gate = dict(preregistration["integrity_gate"])
    failures = integrity_gate_failures(
        profiles=profiles,
        games=len(per_game),
        states=len(cases),
        folds=folds,
        group_leakage=group_leakage,
        reproduction_error=reproduction_error,
        low_evidence_maximum=low_evidence_maximum,
        gate=gate,
    )
    return {
        "ok": bool(per_game) and len(predictions) == len(game_profiles),
        "schema_version": "public-feature-nonlinear-ceiling-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "profiles": list(profiles),
        "games": len(per_game),
        "states": len(cases),
        "folds": fold_reports,
        "group_leakage": group_leakage,
        "all_profiles_in_every_fold": all(
            set(fold_profiles[fold]) == set(profiles) for fold in range(folds)
        ),
        "baseline_metrics": baseline_metrics,
        "reference_metrics": reference_metrics,
        "by_profile": {
            profile: summarize_profile(
                [row for row in per_game if row["profile"] == profile]
            )
            for profile in profiles
        },
        "by_public_evidence_stratum": summarize_evidence_strata(
            state_predictions
        ),
        "state_confusion": {
            "baseline": confusion_matrix(
                state_predictions,
                profiles=profiles,
                selected_key="baseline_selected",
            ),
            "reference": confusion_matrix(
                state_predictions,
                profiles=profiles,
                selected_key="reference_selected",
            ),
        },
        "paired_improvements": improvements,
        "maximum_low_evidence_reference_posterior": low_evidence_maximum,
        "baseline_reproduction_error": reproduction_error,
        "diagnosis": diagnosis,
        "integrity_gate": gate,
        "integrity_gate_pass": not failures,
        "integrity_gate_failures": failures,
    }


def reliability_blend(
    probabilities: Sequence[float],
    *,
    public_actions: int,
    full_reliability_public_actions: int,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    values = values / float(np.sum(values))
    reliability = min(
        1.0,
        max(0, int(public_actions))
        / float(max(1, full_reliability_public_actions)),
    )
    uniform = np.full(len(values), 1.0 / len(values), dtype=float)
    return reliability * values + (1.0 - reliability) * uniform


def selected_probability_index(
    probabilities: Sequence[float],
    labels: Sequence[str],
) -> int:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("nonlinear_ceiling_probability_label_mismatch")
    return max(
        range(len(labels)),
        key=lambda index: (float(probabilities[index]), str(labels[index])),
    )


def classify_feature_ceiling(
    improvements: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any],
) -> str:
    material = all(
        float(improvements[name]["lower_95"])
        >= float(thresholds[f"minimum_material_{name}_improvement_lower_95"])
        for name in ("log_loss", "true_probability", "top1")
    )
    insufficient = all(
        float(improvements[name]["upper_95"])
        <= float(
            thresholds[
                f"maximum_feature_insufficient_{name}_improvement_upper_95"
            ]
        )
        for name in ("log_loss", "true_probability", "top1")
    )
    if material:
        return "linear_underfit_supported"
    if insufficient:
        return "aggregate_feature_insufficiency_supported"
    return "inconclusive_mixed_signal"


def summarize_game_predictions(
    game_id: str,
    profile: str,
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "profile": profile,
        "states": len(rows),
        **{
            key: fmean(float(row[key]) for row in rows)
            for key in (
                "baseline_log_loss",
                "baseline_true_probability",
                "baseline_top1",
                "reference_log_loss",
                "reference_true_probability",
                "reference_top1",
            )
        },
    }


def summarize_model(
    per_game: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any]:
    return {
        "game_mean_log_loss": mean_confidence_interval(
            [float(row[f"{prefix}_log_loss"]) for row in per_game]
        ),
        "game_mean_true_probability": mean_confidence_interval(
            [float(row[f"{prefix}_true_probability"]) for row in per_game]
        ),
        "game_weighted_top1_accuracy": fmean(
            float(row[f"{prefix}_top1"]) for row in per_game
        ),
    }


def summarize_profile(per_game: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = summarize_model(per_game, prefix="baseline")
    reference = summarize_model(per_game, prefix="reference")
    return {
        "games": len(per_game),
        "baseline": baseline,
        "reference": reference,
        "improvement": {
            "log_loss": (
                float(baseline["game_mean_log_loss"]["mean"])
                - float(reference["game_mean_log_loss"]["mean"])
            ),
            "true_probability": (
                float(reference["game_mean_true_probability"]["mean"])
                - float(baseline["game_mean_true_probability"]["mean"])
            ),
            "top1": (
                float(reference["game_weighted_top1_accuracy"])
                - float(baseline["game_weighted_top1_accuracy"])
            ),
        },
    }


def summarize_evidence_strata(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def stratum(public_actions: int) -> str:
        if public_actions <= 2:
            return "0_to_2"
        if public_actions <= 5:
            return "3_to_5"
        return "6_plus"

    result = {}
    for name in ("0_to_2", "3_to_5", "6_plus"):
        selected = [
            row for row in rows if stratum(int(row["public_actions"])) == name
        ]
        result[name] = {
            "states": len(selected),
            "baseline_log_loss": fmean(
                float(row["baseline_log_loss"]) for row in selected
            ),
            "reference_log_loss": fmean(
                float(row["reference_log_loss"]) for row in selected
            ),
            "baseline_true_probability": fmean(
                float(row["baseline_true_probability"]) for row in selected
            ),
            "reference_true_probability": fmean(
                float(row["reference_true_probability"]) for row in selected
            ),
            "baseline_top1": fmean(
                float(row["baseline_top1"]) for row in selected
            ),
            "reference_top1": fmean(
                float(row["reference_top1"]) for row in selected
            ),
        }
    return result


def confusion_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    selected_key: str,
) -> dict[str, dict[str, int]]:
    return {
        actual: {
            predicted: sum(
                row["profile"] == actual and row[selected_key] == predicted
                for row in rows
            )
            for predicted in profiles
        }
        for actual in profiles
    }


def baseline_reproduction_error(
    reproduced: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> float:
    return max(
        abs(
            float(reproduced["game_mean_log_loss"]["mean"])
            - float(expected["game_mean_log_loss"]["mean"])
        ),
        abs(
            float(reproduced["game_mean_true_probability"]["mean"])
            - float(expected["game_mean_true_probability"]["mean"])
        ),
        abs(
            float(reproduced["game_weighted_top1_accuracy"])
            - float(expected["game_weighted_top1_accuracy"])
        ),
    )


def integrity_gate_failures(
    *,
    profiles: Sequence[str],
    games: int,
    states: int,
    folds: int,
    group_leakage: int,
    reproduction_error: float,
    low_evidence_maximum: float,
    gate: Mapping[str, Any],
) -> list[str]:
    failures = []
    if len(profiles) != int(gate["required_profiles"]):
        failures.append("required_profiles")
    if games != int(gate["required_games"]):
        failures.append("required_games")
    if states < int(gate["minimum_states"]):
        failures.append("minimum_states")
    if folds != int(gate["required_folds"]):
        failures.append("required_folds")
    if group_leakage > int(gate["maximum_group_leakage"]):
        failures.append("group_leakage")
    if reproduction_error > float(
        gate["maximum_baseline_metric_reproduction_error"]
    ):
        failures.append("baseline_metric_reproduction")
    if low_evidence_maximum > float(gate["maximum_low_evidence_posterior"]):
        failures.append("maximum_low_evidence_posterior")
    return failures


def _per_game_weights(cases: Sequence[BeliefCase]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case.game_id] += 1
    return np.asarray([1.0 / counts[case.game_id] for case in cases], dtype=float)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
