"""Run the preregistered grouped-CV public opponent-belief experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.opponent_belief import (
    PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES,
    PublicOpponentBeliefModel,
    public_opponent_features,
)
from engine.rules import rules_for_room


@dataclass(frozen=True)
class BeliefCase:
    game_id: str
    profile: str
    fold: int
    public_actions: int
    features: tuple[float, ...]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preregistration_bytes = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_bytes.decode("utf-8"))
    report = run_preregistered_experiment(
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
                "ok": report["ok"],
                "focused_gate_pass": report["focused_gate_pass"],
                "games": report["games"],
                "states": report["states"],
                "game_mean_log_loss_upper_95": report["metrics"][
                    "game_mean_log_loss"
                ]["upper_95"],
                "game_mean_true_probability_lower_95": report["metrics"][
                    "game_mean_true_probability"
                ]["lower_95"],
                "game_weighted_top1_accuracy": report["metrics"][
                    "game_weighted_top1_accuracy"
                ],
                "maximum_low_evidence_posterior": report["metrics"][
                    "maximum_low_evidence_posterior"
                ],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def run_preregistered_experiment(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    model_spec = dict(preregistration["model"])
    expected_features = tuple(str(item) for item in model_spec["feature_names"])
    if expected_features != PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES:
        raise ValueError("registered_public_opponent_feature_schema_mismatch")
    folds = int(model_spec["folds"])
    if folds != 5:
        raise ValueError("registered_public_opponent_fold_mismatch")

    source_path = Path(preregistration["inputs"]["source_report"])
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    games = list(source.get("rows") or ())
    cases, game_profiles, fold_profiles = collect_cases(
        games,
        folds=folds,
    )
    profiles = tuple(sorted(set(game_profiles.values())))
    profile_to_index = {profile: index for index, profile in enumerate(profiles)}

    predictions: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    low_evidence_maximum = 0.0
    group_leakage = 0
    fold_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        training = [case for case in cases if case.fold != fold]
        validation = [case for case in cases if case.fold == fold]
        training_games = {case.game_id for case in training}
        validation_games = {case.game_id for case in validation}
        overlap = training_games & validation_games
        group_leakage += len(overlap)
        model = fit_weighted_ridge_model(
            training,
            profiles=profiles,
            ridge=float(model_spec["ridge"]),
            temperature=float(model_spec["temperature"]),
            full_reliability_public_actions=int(
                model_spec["full_reliability_public_actions"]
            ),
        )
        fold_correct = 0
        fold_log_losses: list[float] = []
        for case in validation:
            posterior = model.posterior(
                case.features,
                public_actions=case.public_actions,
            )
            true_probability = posterior[case.profile]
            selected = max(profiles, key=lambda name: (posterior[name], name))
            correct = int(selected == case.profile)
            log_loss = -math.log(max(1e-15, true_probability))
            predictions[case.game_id].append(
                (log_loss, true_probability, correct)
            )
            fold_correct += correct
            fold_log_losses.append(log_loss)
            if case.public_actions <= int(model_spec["low_evidence_max_actions"]):
                low_evidence_maximum = max(
                    low_evidence_maximum,
                    max(posterior.values()),
                )
        fold_rows.append(
            {
                "fold": fold,
                "training_games": len(training_games),
                "validation_games": len(validation_games),
                "training_states": len(training),
                "validation_states": len(validation),
                "validation_profiles": sorted(
                    {case.profile for case in validation}
                ),
                "group_overlap": len(overlap),
                "state_top1_accuracy": fold_correct / max(1, len(validation)),
                "state_mean_log_loss": fmean(fold_log_losses),
            }
        )

    per_game = []
    for game_id in sorted(predictions):
        values = predictions[game_id]
        per_game.append(
            {
                "game_id": game_id,
                "profile": game_profiles[game_id],
                "states": len(values),
                "mean_log_loss": fmean(item[0] for item in values),
                "mean_true_probability": fmean(item[1] for item in values),
                "top1_accuracy": fmean(item[2] for item in values),
            }
        )
    log_loss_summary = mean_confidence_interval(
        [row["mean_log_loss"] for row in per_game]
    )
    true_probability_summary = mean_confidence_interval(
        [row["mean_true_probability"] for row in per_game]
    )
    top1_accuracy = fmean(row["top1_accuracy"] for row in per_game)
    profile_metrics = {
        profile: summarize_game_rows(
            [row for row in per_game if row["profile"] == profile]
        )
        for profile in profiles
    }
    final_model = fit_weighted_ridge_model(
        cases,
        profiles=profiles,
        ridge=float(model_spec["ridge"]),
        temperature=float(model_spec["temperature"]),
        full_reliability_public_actions=int(
            model_spec["full_reliability_public_actions"]
        ),
    )

    gate = dict(preregistration["focused_gate"])
    all_profiles_every_fold = all(
        set(fold_profiles[fold]) == set(profiles)
        for fold in range(folds)
    )
    failures = focused_gate_failures(
        gate=gate,
        profiles=len(profiles),
        games=len(games),
        states=len(cases),
        log_loss_upper_95=log_loss_summary["upper_95"],
        true_probability_lower_95=true_probability_summary["lower_95"],
        top1_accuracy=top1_accuracy,
        low_evidence_maximum=low_evidence_maximum,
        group_leakage=group_leakage,
        all_profiles_every_fold=all_profiles_every_fold,
    )
    integrity_errors = []
    if len(predictions) != len(games):
        integrity_errors.append("missing_game_predictions")
    if len(cases) != sum(len(game.get("candidate_discard_events") or ()) for game in games):
        integrity_errors.append("missing_state_predictions")
    if any(not values for values in predictions.values()):
        integrity_errors.append("empty_game_predictions")
    return {
        "ok": not integrity_errors,
        "schema_version": "public-opponent-belief-grouped-cv-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "source_report": str(source_path),
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "profiles": list(profiles),
        "games": len(games),
        "states": len(cases),
        "folds": fold_rows,
        "group_leakage": group_leakage,
        "all_profiles_in_every_fold": all_profiles_every_fold,
        "metrics": {
            "game_mean_log_loss": log_loss_summary,
            "game_mean_true_probability": true_probability_summary,
            "game_weighted_top1_accuracy": top1_accuracy,
            "maximum_low_evidence_posterior": low_evidence_maximum,
            "by_profile": profile_metrics,
        },
        "focused_gate": gate,
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
        "integrity_errors": integrity_errors,
        "final_model": final_model.to_dict(),
    }


def collect_cases(
    games: Sequence[Mapping[str, Any]],
    *,
    folds: int,
) -> tuple[list[BeliefCase], dict[str, str], dict[int, set[str]]]:
    groups: dict[str, list[tuple[tuple[int, int, int, int], str, Mapping[str, Any]]]] = defaultdict(list)
    for source_index, game in enumerate(games):
        opponents = tuple(str(item) for item in game.get("opponents") or ())
        if len(opponents) != 1:
            raise ValueError("public_opponent_belief_requires_one_opponent")
        profile = opponents[0]
        sort_key = (
            int(game.get("seed") or 0),
            int(game.get("candidate_seat") or 0),
            int(game.get("dealer") or 0),
            source_index,
        )
        game_id = (
            f"{profile}:{sort_key[0]}:{sort_key[1]}:"
            f"{sort_key[2]}:{source_index}"
        )
        groups[profile].append((sort_key, game_id, game))

    cases: list[BeliefCase] = []
    game_profiles: dict[str, str] = {}
    fold_profiles: dict[int, set[str]] = {fold: set() for fold in range(folds)}
    for profile in sorted(groups):
        for profile_index, (_, game_id, game) in enumerate(sorted(groups[profile])):
            fold = profile_index % folds
            fold_profiles[fold].add(profile)
            game_profiles[game_id] = profile
            rules = rules_for_room(
                wildcard_enabled=bool(game.get("wildcard_enabled")),
                players=int(game.get("players") or 2),
            )
            events = list(game.get("candidate_discard_events") or ())
            if not events:
                raise ValueError(f"public_opponent_game_without_states:{game_id}")
            for event in events:
                view = event.get("public_view")
                if not isinstance(view, Mapping):
                    raise ValueError(f"public_opponent_missing_view:{game_id}")
                evidence = public_opponent_features(view, rules)
                cases.append(
                    BeliefCase(
                        game_id=game_id,
                        profile=profile,
                        fold=fold,
                        public_actions=evidence.public_actions,
                        features=evidence.features,
                    )
                )
    return cases, game_profiles, fold_profiles


def fit_weighted_ridge_model(
    cases: Sequence[BeliefCase],
    *,
    profiles: Sequence[str],
    ridge: float,
    temperature: float,
    full_reliability_public_actions: int,
) -> PublicOpponentBeliefModel:
    if not cases:
        raise ValueError("public_opponent_belief_empty_training_set")
    profile_to_index = {profile: index for index, profile in enumerate(profiles)}
    game_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        game_counts[case.game_id] += 1
    weights = np.asarray(
        [1.0 / game_counts[case.game_id] for case in cases],
        dtype=float,
    )
    features = np.asarray([case.features for case in cases], dtype=float)
    targets = np.zeros((len(cases), len(profiles)), dtype=float)
    for row, case in enumerate(cases):
        targets[row, profile_to_index[case.profile]] = 1.0
    weight_total = float(np.sum(weights))
    means = np.sum(features * weights[:, None], axis=0) / weight_total
    variances = (
        np.sum(((features - means) ** 2) * weights[:, None], axis=0)
        / weight_total
    )
    scales = np.sqrt(np.maximum(variances, 0.0))
    scales[scales < 1e-12] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack((standardized, np.ones(len(cases))))
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_targets = targets * np.sqrt(weights)[:, None]
    penalty = np.eye(design.shape[1], dtype=float) * float(ridge)
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_targets,
    )
    return PublicOpponentBeliefModel(
        profile_names=tuple(profiles),
        feature_names=PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES,
        coefficients=tuple(
            tuple(float(value) for value in row)
            for row in solution[:-1]
        ),
        intercepts=tuple(float(value) for value in solution[-1]),
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        temperature=float(temperature),
        full_reliability_public_actions=int(full_reliability_public_actions),
    )


def mean_confidence_interval(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    average = fmean(values)
    standard_error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    half_width = 1.96 * standard_error
    return {
        "n": len(values),
        "mean": average,
        "lower_95": average - half_width,
        "upper_95": average + half_width,
    }


def summarize_game_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(rows),
        "mean_log_loss": fmean(float(row["mean_log_loss"]) for row in rows),
        "mean_true_probability": fmean(
            float(row["mean_true_probability"]) for row in rows
        ),
        "mean_top1_accuracy": fmean(float(row["top1_accuracy"]) for row in rows),
    }


def focused_gate_failures(
    *,
    gate: Mapping[str, Any],
    profiles: int,
    games: int,
    states: int,
    log_loss_upper_95: float,
    true_probability_lower_95: float,
    top1_accuracy: float,
    low_evidence_maximum: float,
    group_leakage: int,
    all_profiles_every_fold: bool,
) -> list[str]:
    failures: list[str] = []
    if profiles != int(gate["required_profiles"]):
        failures.append("required_profiles")
    if games != int(gate["required_games"]):
        failures.append("required_games")
    if states < int(gate["minimum_states"]):
        failures.append("minimum_states")
    if log_loss_upper_95 >= float(gate["maximum_game_mean_log_loss_upper_95"]):
        failures.append("game_mean_log_loss_upper_95")
    if true_probability_lower_95 <= float(
        gate["minimum_game_mean_true_probability_lower_95"]
    ):
        failures.append("game_mean_true_probability_lower_95")
    if top1_accuracy < float(gate["minimum_game_weighted_top1_accuracy"]):
        failures.append("game_weighted_top1_accuracy")
    if low_evidence_maximum > float(gate["maximum_low_evidence_posterior"]):
        failures.append("maximum_low_evidence_posterior")
    if bool(gate["require_zero_group_leakage"]) and group_leakage:
        failures.append("group_leakage")
    if bool(gate["require_all_profiles_in_every_fold"]) and not all_profiles_every_fold:
        failures.append("all_profiles_in_every_fold")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
