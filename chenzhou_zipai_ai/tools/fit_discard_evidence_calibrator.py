"""Fit and cross-validate a ridge calibrator for online discard evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.opponent_context import (
    OPPONENT_CONTEXT_FEATURE_NAMES,
    OPPONENT_CONTEXT_INTERACTION_FEATURE_NAMES,
    opponent_context_interactions,
)


FEATURE_NAMES = (
    "combined_mean_delta",
    "combined_standard_error",
    "batch_disagreement",
    "minimum_batch_mean",
    "coverage_reward_delta",
    "heuristic_value_delta_scaled",
    "inverse_coverage_rank",
    "inverse_heuristic_rank",
    "confirmation_fraction",
    *OPPONENT_CONTEXT_FEATURE_NAMES,
    *OPPONENT_CONTEXT_INTERACTION_FEATURE_NAMES,
)


@dataclass(frozen=True)
class CandidateExample:
    state_id: str
    label: str
    features: tuple[float, ...]
    target: float
    weight: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-report",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--margin",
        type=float,
        action="append",
        default=[],
        help="Override decision-margin grid; repeat for multiple values.",
    )
    parser.add_argument(
        "--max-empirically-negative-overrides",
        type=int,
        help="Reject cross-validation settings above this harm count.",
    )
    parser.add_argument(
        "--minimum-confirmation-fraction",
        type=float,
        action="append",
        default=[],
        help="Confirmation-fraction grid; repeat for multiple values.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        row
        for path in args.calibration_report
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]
    ]
    state_ids = [str(row["state_before_hash"]) for row in rows]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("discard_calibrator_duplicate_state")
    confirmation_fractions = tuple(
        dict.fromkeys(
            min(1.0, max(0.0, float(value)))
            for value in (
                args.minimum_confirmation_fraction or (1.0,)
            )
        )
    )
    examples_by_fraction = {
        fraction: examples_from_rows(
            rows,
            minimum_confirmation_fraction=fraction,
        )
        for fraction in confirmation_fractions
    }
    settings = [
        {
            **setting,
            "minimum_confirmation_fraction": fraction,
        }
        for fraction, fraction_examples in examples_by_fraction.items()
        if fraction_examples
        for setting in cross_validate(
            rows,
            fraction_examples,
            folds=max(2, int(args.folds)),
            ridge_values=(0.01, 0.1, 1.0, 10.0, 100.0),
            margins=(
                tuple(float(value) for value in args.margin)
                if args.margin
                else (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)
            ),
        )
    ]
    initial_selected_setting = select_setting(
        settings,
        max_empirically_negative_overrides=(
            args.max_empirically_negative_overrides
        ),
    )
    minimum_confirmation_fraction = float(
        initial_selected_setting["minimum_confirmation_fraction"]
    )
    examples = examples_by_fraction[minimum_confirmation_fraction]
    model = fit_model(
        examples,
        ridge=float(initial_selected_setting["ridge"]),
    )
    final_predictions = {
        (example.state_id, example.label): predict(
            model,
            example.features,
        )
        for example in examples
    }
    final_fit_settings = [
        {
            **setting,
            "final_fit": evaluate_state_policy(
                rows,
                final_predictions,
                ridge=float(setting["ridge"]),
                margin=float(setting["margin"]),
            ),
        }
        for setting in settings
        if (
            float(setting["ridge"])
            == float(initial_selected_setting["ridge"])
            and float(setting["minimum_confirmation_fraction"])
            == minimum_confirmation_fraction
        )
    ]
    selected_setting = select_final_fit_setting(
        final_fit_settings,
        max_empirically_negative_overrides=(
            args.max_empirically_negative_overrides
        ),
    )
    report = {
        "ok": bool(rows) and bool(examples),
        "schema_version": "discard-evidence-calibrator-v1",
        "calibration_reports": [
            str(path)
            for path in args.calibration_report
        ],
        "states": len(rows),
        "candidate_examples": len(examples),
        "candidate_examples_by_confirmation_fraction": {
            str(fraction): len(fraction_examples)
            for fraction, fraction_examples in examples_by_fraction.items()
        },
        "feature_names": list(FEATURE_NAMES),
        "folds": max(2, int(args.folds)),
        "validation_groups": len(
            {
                str(row.get("validation_group_id") or row["state_before_hash"])
                for row in rows
            }
        ),
        "validation_groups_by_fold": {
            str(fold): count
            for fold, count in sorted(
                _validation_group_counts(
                    rows,
                    folds=max(2, int(args.folds)),
                ).items()
            )
        },
        "max_empirically_negative_overrides": (
            args.max_empirically_negative_overrides
        ),
        "confirmation_fraction_grid": list(confirmation_fractions),
        "cross_validation": settings,
        "initial_selected_setting": initial_selected_setting,
        "final_fit_candidates": final_fit_settings,
        "selected_setting": selected_setting,
        "final_fit_evaluation": selected_setting["final_fit"],
        "model": {
            **model,
            "decision_margin": float(selected_setting["margin"]),
            "minimum_confirmation_fraction": (
                minimum_confirmation_fraction
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "states": report["states"],
                "candidate_examples": report["candidate_examples"],
                "selected_setting": selected_setting,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def select_setting(
    settings: Sequence[Mapping[str, Any]],
    *,
    max_empirically_negative_overrides: int | None = None,
) -> Mapping[str, Any]:
    eligible = list(settings)
    if max_empirically_negative_overrides is not None:
        limit = max(0, int(max_empirically_negative_overrides))
        eligible = [
            setting
            for setting in eligible
            if int(setting["empirically_negative_overrides"]) <= limit
        ]
    if not eligible:
        raise ValueError("discard_calibrator_no_setting_satisfies_harm_limit")
    return max(
        eligible,
        key=lambda setting: (
            float(setting["mean_improvement_over_preferred"]),
            -float(setting["mean_regret_to_pooled_best"]),
            -int(setting["empirically_negative_overrides"]),
            -int(setting["overrides"]),
            -float(setting["ridge"]),
            -float(setting["margin"]),
            float(setting.get("minimum_confirmation_fraction", 1.0)),
        ),
    )


def select_final_fit_setting(
    settings: Sequence[Mapping[str, Any]],
    *,
    max_empirically_negative_overrides: int | None = None,
) -> Mapping[str, Any]:
    eligible = list(settings)
    if max_empirically_negative_overrides is not None:
        limit = max(0, int(max_empirically_negative_overrides))
        eligible = [
            setting
            for setting in eligible
            if int(
                setting["final_fit"][
                    "empirically_negative_overrides"
                ]
            )
            <= limit
        ]
    if not eligible:
        raise ValueError(
            "discard_calibrator_no_final_setting_satisfies_harm_limit"
        )
    return select_setting(
        eligible,
        max_empirically_negative_overrides=(
            max_empirically_negative_overrides
        ),
    )


def examples_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_confirmation_fraction: float = 1.0,
) -> list[CandidateExample]:
    examples: list[CandidateExample] = []
    for row in rows:
        state_id = str(row["state_before_hash"])
        truth_by_label = {
            str(candidate["label"]): float(
                candidate["vs_preferred"]["mean_delta"]
            )
            for candidate in row["candidates"]
        }
        weight = max(1.0, float(row.get("pooled_worlds") or 1) / 640.0)
        for evidence in row.get("online_challenger_evidence") or ():
            label = str(evidence["challenger_label"])
            if (
                not evidence_usable(
                    evidence,
                    minimum_confirmation_fraction=(
                        minimum_confirmation_fraction
                    ),
                )
                or label not in truth_by_label
            ):
                continue
            examples.append(
                CandidateExample(
                    state_id=state_id,
                    label=label,
                    features=evidence_features(
                        evidence,
                        opponent_context=row.get(
                            "opponent_context_features"
                        ),
                    ),
                    target=truth_by_label[label],
                    weight=weight,
                )
            )
    return examples


def evidence_usable(
    evidence: Mapping[str, Any],
    *,
    minimum_confirmation_fraction: float,
) -> bool:
    if evidence.get("complete"):
        return True
    fraction = min(
        1.0,
        max(0.0, float(minimum_confirmation_fraction)),
    )
    return bool(
        fraction < 1.0
        and evidence.get("structurally_valid")
        and float(evidence.get("confirmation_fraction") or 0.0) >= fraction
    )


def evidence_features(
    evidence: Mapping[str, Any],
    *,
    opponent_context: Sequence[float] | None = None,
) -> tuple[float, ...]:
    first = float(evidence.get("first_mean_delta") or 0.0)
    second = float(evidence.get("second_mean_delta") or 0.0)
    coverage_rank = max(1, int(evidence.get("coverage_rank") or 0))
    heuristic_rank = max(1, int(evidence.get("heuristic_rank") or 0))
    context = tuple(
        float(value)
        for value in (
            opponent_context
            or (0.0,) * len(OPPONENT_CONTEXT_FEATURE_NAMES)
        )
    )
    if len(context) != len(OPPONENT_CONTEXT_FEATURE_NAMES):
        raise ValueError("discard_calibrator_opponent_context_mismatch")
    core_features = (
        float(evidence.get("combined_mean_delta") or 0.0),
        float(evidence.get("combined_standard_error") or 0.0),
        abs(first - second),
        min(first, second),
        float(evidence.get("coverage_reward_delta") or 0.0),
        float(evidence.get("heuristic_value_delta") or 0.0) / 1000.0,
        1.0 / coverage_rank,
        1.0 / heuristic_rank,
        float(evidence.get("confirmation_fraction") or 0.0),
    )
    return (
        *core_features,
        *context,
        *opponent_context_interactions(core_features[0], context),
    )


def fit_model(
    examples: Sequence[CandidateExample],
    *,
    ridge: float,
) -> dict[str, Any]:
    x = np.asarray([example.features for example in examples], dtype=float)
    y = np.asarray([example.target for example in examples], dtype=float)
    weights = np.asarray([example.weight for example in examples], dtype=float)
    feature_mean = np.average(x, axis=0, weights=weights)
    variance = np.average(
        (x - feature_mean) ** 2,
        axis=0,
        weights=weights,
    )
    feature_scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (x - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(examples)), standardized))
    sqrt_weight = np.sqrt(weights)
    weighted_design = design * sqrt_weight[:, None]
    weighted_target = y * sqrt_weight
    penalty = np.eye(design.shape[1]) * max(0.0, float(ridge))
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    predictions = design @ coefficients
    return {
        "ridge": float(ridge),
        "feature_mean": [round(float(value), 12) for value in feature_mean],
        "feature_scale": [
            round(float(value), 12)
            for value in feature_scale
        ],
        "intercept": round(float(coefficients[0]), 12),
        "coefficients": [
            round(float(value), 12)
            for value in coefficients[1:]
        ],
        "weighted_rmse": round(
            math.sqrt(
                float(
                    np.average(
                        (predictions - y) ** 2,
                        weights=weights,
                    )
                )
            ),
            8,
        ),
    }


def predict(
    model: Mapping[str, Any],
    features: Sequence[float],
) -> float:
    standardized = [
        (float(value) - float(center)) / float(scale)
        for value, center, scale in zip(
            features,
            model["feature_mean"],
            model["feature_scale"],
        )
    ]
    return float(model["intercept"]) + sum(
        float(coefficient) * value
        for coefficient, value in zip(
            model["coefficients"],
            standardized,
        )
    )


def cross_validate(
    rows: Sequence[Mapping[str, Any]],
    examples: Sequence[CandidateExample],
    *,
    folds: int,
    ridge_values: Sequence[float],
    margins: Sequence[float],
) -> list[dict[str, Any]]:
    state_fold = validation_fold_assignments(rows, folds=folds)
    predictions_by_ridge: dict[float, dict[tuple[str, str], float]] = {}
    for ridge in ridge_values:
        predictions: dict[tuple[str, str], float] = {}
        for fold in range(folds):
            training = [
                example
                for example in examples
                if state_fold[example.state_id] != fold
            ]
            validation = [
                example
                for example in examples
                if state_fold[example.state_id] == fold
            ]
            if not training or not validation:
                continue
            model = fit_model(training, ridge=float(ridge))
            for example in validation:
                predictions[(example.state_id, example.label)] = predict(
                    model,
                    example.features,
                )
        predictions_by_ridge[float(ridge)] = predictions

    reports: list[dict[str, Any]] = []
    for ridge, predictions in predictions_by_ridge.items():
        for margin in margins:
            reports.append(
                evaluate_state_policy(
                    rows,
                    predictions,
                    ridge=ridge,
                    margin=float(margin),
                )
            )
    return reports


def evaluate_state_policy(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], float],
    *,
    ridge: float,
    margin: float,
) -> dict[str, Any]:
    improvements: list[float] = []
    regrets: list[float] = []
    overrides = 0
    empirically_negative = 0
    confidently_positive = 0
    robust_misses = 0
    exact_best = 0
    for row in rows:
        state_id = str(row["state_before_hash"])
        preferred = str(row["preferred_label"])
        candidate_by_label = {
            str(candidate["label"]): candidate
            for candidate in row["candidates"]
        }
        available = [
            (label, prediction)
            for (candidate_state, label), prediction in predictions.items()
            if candidate_state == state_id
        ]
        selected, selected_prediction = max(
            available or [(preferred, -math.inf)],
            key=lambda item: (item[1], item[0]),
        )
        if selected_prediction <= margin:
            selected = preferred
        selected_stats = candidate_by_label[selected]
        actual_improvement = float(
            selected_stats["vs_preferred"]["mean_delta"]
        )
        improvements.append(actual_improvement)
        regrets.append(
            float(row["pooled_best_vs_preferred"]["mean_delta"])
            - actual_improvement
        )
        overrides += int(selected != preferred)
        empirically_negative += int(
            selected != preferred and actual_improvement < 0.0
        )
        confidently_positive += int(
            selected != preferred
            and bool(
                selected_stats["vs_preferred"][
                    "confidently_positive"
                ]
            )
        )
        robust_misses += int(
            bool(row["preferred_confidently_suboptimal"])
            and not bool(
                selected_stats["vs_preferred"]["confidently_positive"]
            )
        )
        exact_best += int(selected == str(row["pooled_best_label"]))
    return {
        "ridge": float(ridge),
        "margin": float(margin),
        "states": len(rows),
        "overrides": overrides,
        "confidently_positive_overrides": confidently_positive,
        "empirically_negative_overrides": empirically_negative,
        "robust_misses": robust_misses,
        "exact_pooled_best": exact_best,
        "mean_improvement_over_preferred": round(
            mean(improvements) if improvements else 0.0,
            8,
        ),
        "mean_regret_to_pooled_best": round(
            mean(regrets) if regrets else 0.0,
            8,
        ),
    }


def _stable_fold(state_id: str, *, folds: int) -> int:
    digest = hashlib.sha256(state_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(2, int(folds))


def validation_fold_assignments(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int,
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for row in rows:
        state_id = str(row["state_before_hash"])
        group_id = str(row.get("validation_group_id") or state_id)
        assignments[state_id] = _stable_fold(group_id, folds=folds)
    return assignments


def _validation_group_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int,
) -> dict[int, int]:
    groups = {
        str(row.get("validation_group_id") or row["state_before_hash"])
        for row in rows
    }
    counts: dict[int, int] = {}
    for group_id in groups:
        fold = _stable_fold(group_id, folds=folds)
        counts[fold] = counts.get(fold, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
