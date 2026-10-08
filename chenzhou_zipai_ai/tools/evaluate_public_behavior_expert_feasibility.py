"""Grouped-CV feasibility test for public-context behavior experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from engine.cards import BIG_LABELS, RED_LABELS, SMALL_LABELS
from tools.evaluate_anchored_structural_discard_proxies import sha256_file
from tools.fit_public_opponent_belief import mean_confidence_interval
from tools.fit_sequential_public_likelihood import load_events


DISCARD_VOCABULARY = tuple(
    f"DISCARD:{label}" for label in (*SMALL_LABELS, *BIG_LABELS)
)
RESPONSE_VOCABULARY = ("CHI", "PENG", "HU", "NO_CLAIM")


@dataclass(frozen=True)
class BehaviorCase:
    game_id: str
    profile: str
    fold: int
    channel: str
    token: str
    features: tuple[float, ...]


@dataclass(frozen=True)
class MulticlassRidgeModel:
    tokens: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    temperature: float

    def posterior(self, features: Sequence[float]) -> dict[str, float]:
        values = np.asarray(tuple(float(item) for item in features), dtype=float)
        if values.shape != (len(self.feature_means),):
            raise ValueError("public_behavior_feature_shape")
        means = np.asarray(self.feature_means, dtype=float)
        scales = np.asarray(self.feature_scales, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        logits = (
            ((values - means) / scales) @ coefficients
            + np.asarray(self.intercepts, dtype=float)
        ) / self.temperature
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        return {
            token: float(probabilities[index])
            for index, token in enumerate(self.tokens)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "public-behavior-multiclass-ridge-v1",
            "tokens": list(self.tokens),
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "coefficients": [list(row) for row in self.coefficients],
            "intercepts": list(self.intercepts),
            "temperature": self.temperature,
        }


def trigger_features(label: Any) -> tuple[float, ...]:
    if label is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    text = str(label)
    if text in SMALL_LABELS:
        small = 1.0
        rank = SMALL_LABELS.index(text) + 1
    elif text in BIG_LABELS:
        small = 0.0
        rank = BIG_LABELS.index(text) + 1
    else:
        raise KeyError(text)
    return (
        1.0,
        small,
        float(text in RED_LABELS),
        rank / 10.0,
        float(rank in {2, 7, 10}),
    )


def expert_features(
    base_features: Sequence[float],
    profile: str,
    *,
    profiles: Sequence[str],
) -> tuple[float, ...]:
    if profile not in profiles:
        raise KeyError(profile)
    base = tuple(float(item) for item in base_features)
    one_hot = tuple(float(item == profile) for item in profiles)
    interactions = tuple(
        value * float(item == profile)
        for item in profiles
        for value in base
    )
    return (*base, *one_hot, *interactions)


def collect_cases(
    events: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
) -> tuple[list[BehaviorCase], int]:
    cases = []
    unknown_tokens = 0
    for event in events:
        kind = str(event["action_kind"])
        channel = "discard" if kind == "DISCARD" else "response"
        token = str(event["action_token"] if channel == "discard" else kind)
        vocabulary = (
            DISCARD_VOCABULARY if channel == "discard" else RESPONSE_VOCABULARY
        )
        if token not in vocabulary:
            unknown_tokens += 1
            continue
        profile = str(event["profile"])
        if profile not in profiles:
            raise ValueError("public_behavior_unknown_profile")
        context = tuple(float(item) for item in event["context_features"])
        if len(context) != 21:
            raise ValueError("public_behavior_context_feature_count")
        try:
            public_trigger = trigger_features(event.get("trigger_label"))
        except KeyError:
            unknown_tokens += 1
            continue
        cases.append(
            BehaviorCase(
                game_id=str(event["game_id"]),
                profile=profile,
                fold=int(event["fold"]),
                channel=channel,
                token=token,
                features=(*context, *public_trigger),
            )
        )
    return cases, unknown_tokens


def fit_multiclass_ridge(
    cases: Sequence[BehaviorCase],
    feature_rows: Sequence[Sequence[float]],
    *,
    tokens: Sequence[str],
    ridge: float,
    temperature: float,
) -> MulticlassRidgeModel:
    if not cases or len(cases) != len(feature_rows):
        raise ValueError("public_behavior_empty_or_misaligned_training")
    if ridge <= 0.0 or temperature <= 0.0:
        raise ValueError("public_behavior_model_parameters")
    token_to_index = {token: index for index, token in enumerate(tokens)}
    game_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        game_counts[case.game_id] += 1
    weights = np.asarray(
        [1.0 / game_counts[case.game_id] for case in cases], dtype=float
    )
    features = np.asarray(feature_rows, dtype=float)
    targets = np.zeros((len(cases), len(tokens)), dtype=float)
    for row, case in enumerate(cases):
        targets[row, token_to_index[case.token]] = 1.0
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
    penalty = np.eye(design.shape[1], dtype=float) * ridge
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_targets,
    )
    return MulticlassRidgeModel(
        tokens=tuple(tokens),
        feature_means=tuple(float(item) for item in means),
        feature_scales=tuple(float(item) for item in scales),
        coefficients=tuple(
            tuple(float(item) for item in row) for row in solution[:-1]
        ),
        intercepts=tuple(float(item) for item in solution[-1]),
        temperature=temperature,
    )


def evaluate_fold(
    cases: Sequence[BehaviorCase],
    *,
    fold: int,
    profiles: Sequence[str],
    ridge: float,
    temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    elapsed_samples = []
    overlaps = 0
    channel_rows = []
    for channel, vocabulary in (
        ("discard", DISCARD_VOCABULARY),
        ("response", RESPONSE_VOCABULARY),
    ):
        training = [
            case for case in cases if case.channel == channel and case.fold != fold
        ]
        validation = [
            case for case in cases if case.channel == channel and case.fold == fold
        ]
        training_games = {case.game_id for case in training}
        validation_games = {case.game_id for case in validation}
        overlap = training_games & validation_games
        overlaps += len(overlap)
        pooled = fit_multiclass_ridge(
            training,
            [case.features for case in training],
            tokens=vocabulary,
            ridge=ridge,
            temperature=temperature,
        )
        expert = fit_multiclass_ridge(
            training,
            [
                expert_features(case.features, case.profile, profiles=profiles)
                for case in training
            ],
            tokens=vocabulary,
            ridge=ridge,
            temperature=temperature,
        )
        for case in validation:
            started = time.perf_counter()
            pooled_posterior = pooled.posterior(case.features)
            expert_posterior = expert.posterior(
                expert_features(case.features, case.profile, profiles=profiles)
            )
            elapsed_samples.append((time.perf_counter() - started) * 1000.0)
            pooled_selected = max(
                vocabulary, key=lambda item: (pooled_posterior[item], item)
            )
            expert_selected = max(
                vocabulary, key=lambda item: (expert_posterior[item], item)
            )
            records.append(
                {
                    "game_id": case.game_id,
                    "profile": case.profile,
                    "fold": case.fold,
                    "channel": case.channel,
                    "pooled_log_loss": -math.log(
                        max(1e-15, pooled_posterior[case.token])
                    ),
                    "expert_log_loss": -math.log(
                        max(1e-15, expert_posterior[case.token])
                    ),
                    "pooled_true_probability": pooled_posterior[case.token],
                    "expert_true_probability": expert_posterior[case.token],
                    "pooled_top1": int(pooled_selected == case.token),
                    "expert_top1": int(expert_selected == case.token),
                }
            )
        channel_rows.append(
            {
                "channel": channel,
                "training_games": len(training_games),
                "validation_games": len(validation_games),
                "training_events": len(training),
                "validation_events": len(validation),
                "group_overlap": len(overlap),
            }
        )
    return records, {
        "fold": fold,
        "channels": channel_rows,
        "group_overlap": overlaps,
        "elapsed_samples": elapsed_samples,
    }


def per_game_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["game_id"])].append(record)
    rows = []
    for game_id in sorted(grouped):
        game_records = grouped[game_id]
        row: dict[str, Any] = {
            "game_id": game_id,
            "profile": str(game_records[0]["profile"]),
            "events": len(game_records),
        }
        for channel in ("all", "discard", "response"):
            selected = (
                game_records
                if channel == "all"
                else [item for item in game_records if item["channel"] == channel]
            )
            if not selected:
                continue
            row[channel] = {
                "events": len(selected),
                "log_loss_improvement": fmean(
                    float(item["pooled_log_loss"])
                    - float(item["expert_log_loss"])
                    for item in selected
                ),
                "true_probability_improvement": fmean(
                    float(item["expert_true_probability"])
                    - float(item["pooled_true_probability"])
                    for item in selected
                ),
                "top1_improvement": fmean(
                    int(item["expert_top1"]) - int(item["pooled_top1"])
                    for item in selected
                ),
            }
        rows.append(row)
    return rows


def summarize_improvements(
    rows: Sequence[Mapping[str, Any]],
    *,
    channel: str,
) -> dict[str, Any]:
    selected = [row for row in rows if channel in row]
    return {
        "games": len(selected),
        "log_loss": mean_confidence_interval(
            [float(row[channel]["log_loss_improvement"]) for row in selected]
        ),
        "true_probability": mean_confidence_interval(
            [
                float(row[channel]["true_probability_improvement"])
                for row in selected
            ]
        ),
        "top1": mean_confidence_interval(
            [float(row[channel]["top1_improvement"]) for row in selected]
        ),
    }


def gate_failures(
    *,
    gate: Mapping[str, Any],
    profiles: Sequence[str],
    games: int,
    events: int,
    folds: Sequence[Mapping[str, Any]],
    group_leakage: int,
    unknown_tokens: int,
    improvements: Mapping[str, Mapping[str, Any]],
    by_profile: Mapping[str, Mapping[str, Any]],
    p95_ms: float,
) -> list[str]:
    failures = []
    if len(profiles) != int(gate["required_profiles"]):
        failures.append("required_profiles")
    if games != int(gate["required_games"]):
        failures.append("required_games")
    if events != int(gate["required_events"]):
        failures.append("required_events")
    if len(DISCARD_VOCABULARY) != int(gate["required_discard_tokens"]):
        failures.append("required_discard_tokens")
    if len(RESPONSE_VOCABULARY) != int(gate["required_response_tokens"]):
        failures.append("required_response_tokens")
    if len(folds) != int(gate["required_folds"]):
        failures.append("required_folds")
    if group_leakage > int(gate["maximum_group_leakage"]):
        failures.append("group_leakage")
    if unknown_tokens > int(gate["maximum_unknown_tokens"]):
        failures.append("unknown_tokens")
    if float(improvements["all"]["log_loss"]["lower_95"]) < float(
        gate["minimum_overall_log_loss_improvement_lower_95"]
    ):
        failures.append("overall_log_loss_improvement_lower_95")
    for channel in ("discard", "response"):
        if float(improvements[channel]["log_loss"]["lower_95"]) < float(
            gate[f"minimum_{channel}_log_loss_improvement_lower_95"]
        ):
            failures.append(f"{channel}_log_loss_improvement_lower_95")
    if float(improvements["all"]["true_probability"]["lower_95"]) < float(
        gate["minimum_overall_true_probability_improvement_lower_95"]
    ):
        failures.append("overall_true_probability_improvement_lower_95")
    if float(improvements["all"]["top1"]["lower_95"]) < float(
        gate["minimum_overall_top1_improvement_lower_95"]
    ):
        failures.append("overall_top1_improvement_lower_95")
    for profile, summary in by_profile.items():
        if float(summary["log_loss"]["mean"]) < float(
            gate["minimum_profile_mean_log_loss_improvement"]
        ):
            failures.append(f"profile_log_loss_regression:{profile}")
        if float(summary["true_probability"]["mean"]) < float(
            gate["minimum_profile_mean_true_probability_improvement"]
        ):
            failures.append(f"profile_true_probability_regression:{profile}")
    if p95_ms > float(gate["maximum_prediction_p95_ms"]):
        failures.append("prediction_p95_ms")
    return list(dict.fromkeys(failures))


def run_diagnostic(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    for key in ("dataset", "dataset_audit", "failed_transition_report"):
        if sha256_file(Path(str(inputs[key]))) != str(inputs[f"{key}_sha256"]):
            raise ValueError(f"public_behavior_{key}_hash_mismatch")
    dataset_audit = _read_json(Path(str(inputs["dataset_audit"])))
    if not dataset_audit.get("integrity_gate_pass"):
        raise ValueError("public_behavior_dataset_audit_did_not_pass")
    failed_transition = _read_json(Path(str(inputs["failed_transition_report"])))
    if failed_transition.get("focused_gate_pass"):
        raise ValueError("public_behavior_expected_failed_transition_anchor")
    if float(failed_transition["final_power"]) != 0.0:
        raise ValueError("public_behavior_transition_anchor_not_zero")

    profiles = tuple(str(item) for item in preregistration["profiles"])
    if tuple(str(item) for item in failed_transition["profiles"]) != profiles:
        raise ValueError("public_behavior_profile_mismatch")
    model_spec = dict(preregistration["model"])
    folds_count = int(model_spec["folds"])
    events = load_events(Path(str(inputs["dataset"])))
    cases, unknown_tokens = collect_cases(events, profiles=profiles)
    fold_reports = []
    records = []
    elapsed_samples = []
    group_leakage = 0
    for fold in range(folds_count):
        fold_records, diagnostics = evaluate_fold(
            cases,
            fold=fold,
            profiles=profiles,
            ridge=float(model_spec["ridge"]),
            temperature=float(model_spec["temperature"]),
        )
        records.extend(fold_records)
        elapsed_samples.extend(diagnostics["elapsed_samples"])
        group_leakage += int(diagnostics["group_overlap"])
        fold_reports.append(
            {
                "fold": fold,
                "profiles": sorted(
                    {case.profile for case in cases if case.fold == fold}
                ),
                "channels": diagnostics["channels"],
                "group_overlap": diagnostics["group_overlap"],
            }
        )

    game_rows = per_game_rows(records)
    improvements = {
        channel: summarize_improvements(game_rows, channel=channel)
        for channel in ("all", "discard", "response")
    }
    by_profile = {
        profile: summarize_improvements(
            [row for row in game_rows if row["profile"] == profile],
            channel="all",
        )
        for profile in profiles
    }
    elapsed_samples.sort()
    p95_index = (
        min(len(elapsed_samples) - 1, int((len(elapsed_samples) - 1) * 0.95))
        if elapsed_samples
        else 0
    )
    p95_ms = elapsed_samples[p95_index] if elapsed_samples else 0.0
    gate = dict(preregistration["focused_gate"])
    failures = gate_failures(
        gate=gate,
        profiles=profiles,
        games=len(game_rows),
        events=len(records),
        folds=fold_reports,
        group_leakage=group_leakage,
        unknown_tokens=unknown_tokens,
        improvements=improvements,
        by_profile=by_profile,
        p95_ms=p95_ms,
    )

    final_models = {}
    for channel, vocabulary in (
        ("discard", DISCARD_VOCABULARY),
        ("response", RESPONSE_VOCABULARY),
    ):
        selected = [case for case in cases if case.channel == channel]
        final_models[channel] = {
            "pooled": fit_multiclass_ridge(
                selected,
                [case.features for case in selected],
                tokens=vocabulary,
                ridge=float(model_spec["ridge"]),
                temperature=float(model_spec["temperature"]),
            ).to_dict(),
            "expert": fit_multiclass_ridge(
                selected,
                [
                    expert_features(
                        case.features, case.profile, profiles=profiles
                    )
                    for case in selected
                ],
                tokens=vocabulary,
                ridge=float(model_spec["ridge"]),
                temperature=float(model_spec["temperature"]),
            ).to_dict(),
        }
    all_profiles_every_fold = all(
        set(report["profiles"]) == set(profiles) for report in fold_reports
    )
    integrity_errors = []
    if len(cases) != len(events):
        integrity_errors.append("missing_event_cases")
    if len(records) != len(cases):
        integrity_errors.append("missing_oof_predictions")
    if not all_profiles_every_fold:
        integrity_errors.append("profile_fold_coverage")
    return {
        "ok": not integrity_errors and unknown_tokens == 0,
        "schema_version": "public-behavior-expert-feasibility-grouped-cv-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "profiles": list(profiles),
        "games": len(game_rows),
        "events": len(records),
        "discard_events": sum(record["channel"] == "discard" for record in records),
        "response_events": sum(record["channel"] == "response" for record in records),
        "discard_vocabulary": list(DISCARD_VOCABULARY),
        "response_vocabulary": list(RESPONSE_VOCABULARY),
        "pooled_feature_count": 26,
        "expert_feature_count": 26 + len(profiles) + 26 * len(profiles),
        "folds": fold_reports,
        "all_profiles_in_every_fold": all_profiles_every_fold,
        "group_leakage": group_leakage,
        "unknown_tokens": unknown_tokens,
        "integrity_errors": integrity_errors,
        "paired_improvements": improvements,
        "by_profile": by_profile,
        "prediction_p95_ms": p95_ms,
        "final_models": final_models,
        "focused_gate": gate,
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
    }


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
                "focused_gate_pass": report["focused_gate_pass"],
                "paired_improvements": report["paired_improvements"],
                "prediction_p95_ms": report["prediction_p95_ms"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] and report["focused_gate_pass"] else 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
