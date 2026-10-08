"""Distill seven low-latency discard teachers with game-grouped CV."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.action_value_model import fit_fixed_action_value_model
from ai.ismcts import public_view_from_dict
from ai.opponent_proxy import FastInformationSetProxyPolicy
from ai.value_corpus import _state_features
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from engine.rules import rules_for_room


@dataclass(frozen=True)
class TeacherDecision:
    group_id: str
    game_id: str
    profile: str
    fold: int
    selected_label: str
    legal_labels: tuple[str, ...]
    row_indices: tuple[int, ...]
    legacy_correct: bool


LEGACY_PROXY_TYPES = {
    "aggressive_meld": FastInformationSetProxyPolicy,
    "defensive_search": FastInformationSetProxyPolicy,
    "independent_balanced": IndependentFastRolloutPolicy,
    "independent_denial": IndependentFastDenialPolicy,
    "independent_pressure": IndependentFastPressurePolicy,
    "information_set_search": FastInformationSetProxyPolicy,
    "red_black_search": FastInformationSetProxyPolicy,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    preregistration_bytes = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_bytes.decode("utf-8"))
    report = run_preregistered_distillation(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(
            preregistration_bytes
        ).hexdigest(),
        model_dir=args.model_dir,
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
                "overall_top1_accuracy": report["metrics"][
                    "overall_state_top1_accuracy"
                ],
                "legacy_top1_accuracy": report["metrics"][
                    "legacy_state_top1_accuracy"
                ],
                "game_mean_top1_lower_95": report["metrics"][
                    "game_mean_top1"
                ]["lower_95"],
                "p95_inference_ms": report["metrics"]["p95_inference_ms"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def run_preregistered_distillation(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
    model_dir: Path,
) -> dict[str, Any]:
    profiles = tuple(str(item) for item in preregistration["profiles"])
    if set(profiles) != set(LEGACY_PROXY_TYPES):
        raise ValueError("teacher_proxy_profile_mapping_mismatch")
    model_spec = dict(preregistration["model"])
    folds = int(model_spec["folds"])
    trace_path = Path(preregistration["inputs"]["trace"])
    trace_sha256 = _sha256(trace_path)
    rows_by_profile, decisions_by_profile, integrity_errors = collect_teacher_rows(
        trace_path,
        profiles=profiles,
        folds=folds,
    )

    state_predictions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    final_models: dict[str, dict[str, Any]] = {}
    inference_times: list[float] = []
    for profile_index, profile in enumerate(profiles):
        rows = rows_by_profile[profile]
        decisions = decisions_by_profile[profile]
        seed = int(model_spec["base_seed"]) + profile_index
        for fold in range(folds):
            training_indices = [
                index
                for decision in decisions
                if decision.fold != fold
                for index in decision.row_indices
            ]
            model, fit_report = fit_fixed_action_value_model(
                rows,
                seed=seed,
                hidden_dimension=int(model_spec["hidden_dimension"]),
                ridge=float(model_spec["ridge"]),
                activation=str(model_spec["activation"]),
                training_indices=training_indices,
            )
            validation_decisions = [
                decision for decision in decisions if decision.fold == fold
            ]
            correct = 0
            legacy_correct = 0
            for decision in validation_decisions:
                candidate_rows = [rows[index] for index in decision.row_indices]
                scores = model.predict_rows(candidate_rows)
                selected = max(
                    zip(decision.legal_labels, scores),
                    key=lambda item: (float(item[1]), item[0]),
                )[0]
                is_correct = selected == decision.selected_label
                correct += int(is_correct)
                legacy_correct += int(decision.legacy_correct)
                state_predictions.append(
                    {
                        "group_id": decision.group_id,
                        "game_id": decision.game_id,
                        "profile": profile,
                        "fold": fold,
                        "correct": is_correct,
                        "legacy_correct": decision.legacy_correct,
                        "illegal": selected not in decision.legal_labels,
                    }
                )
            fold_reports.append(
                {
                    "profile": profile,
                    "fold": fold,
                    "training_games": len(
                        {
                            decision.game_id
                            for decision in decisions
                            if decision.fold != fold
                        }
                    ),
                    "validation_games": len(
                        {decision.game_id for decision in validation_decisions}
                    ),
                    "training_states": sum(
                        decision.fold != fold for decision in decisions
                    ),
                    "validation_states": len(validation_decisions),
                    "top1_accuracy": correct / max(1, len(validation_decisions)),
                    "legacy_top1_accuracy": legacy_correct
                    / max(1, len(validation_decisions)),
                    "fit": fit_report,
                }
            )

        final_model, final_fit = fit_fixed_action_value_model(
            rows,
            seed=seed,
            hidden_dimension=int(model_spec["hidden_dimension"]),
            ridge=float(model_spec["ridge"]),
            activation=str(model_spec["activation"]),
        )
        model_path = model_dir / f"stage4_teacher_discard_proxy_{profile}_v1.npz"
        final_model.save(
            model_path,
            metadata={
                "profile": profile,
                "experiment_id": preregistration["experiment_id"],
                "preregistration_sha256": preregistration_sha256,
                "trace_sha256": trace_sha256,
                "fit": final_fit,
            },
        )
        final_models[profile] = {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "fit": final_fit,
        }
        for decision in decisions:
            candidate_rows = [rows[index] for index in decision.row_indices]
            started = time.perf_counter()
            scores = final_model.predict_rows(candidate_rows)
            max(
                zip(decision.legal_labels, scores),
                key=lambda item: (float(item[1]), item[0]),
            )
            inference_times.append((time.perf_counter() - started) * 1000.0)

    metrics = summarize_predictions(
        state_predictions,
        profiles=profiles,
        inference_times=inference_times,
    )
    leakage = grouped_fold_leakage(decisions_by_profile)
    failures = focused_gate_failures(
        metrics,
        gate=preregistration["focused_gate"],
        profiles=len(profiles),
        folds=folds,
        states=len(state_predictions),
        states_by_profile={
            profile: len(decisions_by_profile[profile])
            for profile in profiles
        },
        group_leakage=leakage,
        integrity_errors=len(integrity_errors),
    )
    return {
        "ok": not integrity_errors and len(state_predictions) > 0,
        "schema_version": "seven-teacher-discard-proxy-grouped-cv-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "trace": str(trace_path),
        "trace_sha256": trace_sha256,
        "profiles": list(profiles),
        "states": len(state_predictions),
        "states_by_profile": {
            profile: len(decisions_by_profile[profile])
            for profile in profiles
        },
        "legal_action_rows": sum(len(rows) for rows in rows_by_profile.values()),
        "group_leakage": leakage,
        "integrity_errors": integrity_errors,
        "metrics": metrics,
        "folds": fold_reports,
        "final_models": final_models,
        "focused_gate": dict(preregistration["focused_gate"]),
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
    }


def collect_teacher_rows(
    trace_path: Path,
    *,
    profiles: Sequence[str],
    folds: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[TeacherDecision]],
    list[str],
]:
    games_by_profile: dict[str, list[tuple[tuple[int, int, int, int], str, dict[str, Any]]]] = defaultdict(list)
    with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            game = json.loads(line)
            opponents = tuple(str(item) for item in game.get("opponents") or ())
            if len(opponents) != 1 or opponents[0] not in profiles:
                continue
            profile = opponents[0]
            key = (
                int(game.get("seed") or 0),
                int(game.get("candidate_seat") or 0),
                int(game.get("dealer") or 0),
                source_index,
            )
            game_id = f"{profile}:{key[0]}:{key[1]}:{key[2]}:{source_index}"
            games_by_profile[profile].append((key, game_id, game))

    rows_by_profile: dict[str, list[dict[str, Any]]] = {profile: [] for profile in profiles}
    decisions_by_profile: dict[str, list[TeacherDecision]] = {profile: [] for profile in profiles}
    integrity_errors: list[str] = []
    for profile in profiles:
        legacy_policy = LEGACY_PROXY_TYPES[profile]()
        for profile_game_index, (_, game_id, game) in enumerate(sorted(games_by_profile[profile])):
            fold = profile_game_index % folds
            rules = rules_for_room(
                wildcard_enabled=bool(game.get("wildcard_enabled")),
                players=int(game.get("players") or 2),
            )
            teacher_names = {profile}
            if profile == "information_set_search":
                teacher_names.add("information_set_search_v2")
            for trace in game.get("decision_trace") or ():
                if str(trace.get("phase") or "") != "discard" or str(
                    trace.get("policy") or ""
                ) not in teacher_names:
                    continue
                selected_key = str(trace.get("selected_key") or "")
                if not selected_key.startswith("DISCARD:"):
                    integrity_errors.append("teacher_selected_non_discard")
                    continue
                selected_label = selected_key.split(":", 1)[1]
                legal_actions = [
                    dict(action)
                    for action in trace.get("legal_actions") or ()
                    if str(action.get("type") or "") == "DISCARD"
                ]
                legal_labels = tuple(str(action.get("label") or "") for action in legal_actions)
                if (
                    len(legal_labels) < 1
                    or len(set(legal_labels)) != len(legal_labels)
                    or selected_label not in legal_labels
                ):
                    integrity_errors.append("teacher_legal_action_contract")
                    continue
                public_state = dict(trace.get("public_state") or {})
                group_id = (
                    f"{game_id}:{int(trace.get('sequence') or 0)}:"
                    f"{trace.get('state_before_hash')}"
                )
                start = len(rows_by_profile[profile])
                state_features = _state_features(public_state)
                for action in legal_actions:
                    label = str(action["label"])
                    rows_by_profile[profile].append(
                        {
                            "game": {
                                "players": int(game.get("players") or 2),
                                "wildcard_enabled": bool(game.get("wildcard_enabled")),
                                "candidate_seat": int(trace.get("seat") or 0),
                                "dealer": int(game.get("dealer") or 0),
                            },
                            "phase": "discard",
                            "turn": int(trace.get("turn") or 0),
                            "seat": int(trace.get("seat") or 0),
                            "group_id": group_id,
                            "selected": label == selected_label,
                            "action": {
                                "key": str(action.get("key") or f"DISCARD:{label}"),
                                "type": "DISCARD",
                                "label": label,
                                "option_id": None,
                                "consumed_from_hand": [],
                                "meld_groups": [],
                                "followup_discard": None,
                                "heuristic_value": 0.0,
                            },
                            "state_features": state_features,
                            "targets": {"mean_reward": float(label == selected_label)},
                            "audit_health": {"paired_worlds": 8},
                        }
                    )
                stop = len(rows_by_profile[profile])
                try:
                    legacy_selected = legacy_policy.choose_discard(
                        public_view_from_dict(public_state),
                        rules,
                    )
                except (TypeError, ValueError):
                    legacy_selected = ""
                decisions_by_profile[profile].append(
                    TeacherDecision(
                        group_id=group_id,
                        game_id=game_id,
                        profile=profile,
                        fold=fold,
                        selected_label=selected_label,
                        legal_labels=legal_labels,
                        row_indices=tuple(range(start, stop)),
                        legacy_correct=legacy_selected == selected_label,
                    )
                )
    return rows_by_profile, decisions_by_profile, integrity_errors


def summarize_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    inference_times: Sequence[float],
) -> dict[str, Any]:
    by_profile = {}
    for profile in profiles:
        rows = [row for row in predictions if row["profile"] == profile]
        by_profile[profile] = {
            "states": len(rows),
            "state_top1_accuracy": fmean(bool(row["correct"]) for row in rows),
            "legacy_state_top1_accuracy": fmean(
                bool(row["legacy_correct"]) for row in rows
            ),
        }
    games: dict[str, list[bool]] = defaultdict(list)
    for row in predictions:
        games[str(row["game_id"])].append(bool(row["correct"]))
    game_means = [fmean(values) for values in games.values()]
    elapsed = sorted(float(value) for value in inference_times)
    p95_index = min(len(elapsed) - 1, int((len(elapsed) - 1) * 0.95)) if elapsed else 0
    return {
        "overall_state_top1_accuracy": fmean(
            bool(row["correct"]) for row in predictions
        ),
        "legacy_state_top1_accuracy": fmean(
            bool(row["legacy_correct"]) for row in predictions
        ),
        "overall_gain_over_legacy_proxy": fmean(
            int(bool(row["correct"])) - int(bool(row["legacy_correct"]))
            for row in predictions
        ),
        "illegal_predictions": sum(bool(row["illegal"]) for row in predictions),
        "game_mean_top1": mean_confidence(game_means),
        "p95_inference_ms": elapsed[p95_index] if elapsed else 0.0,
        "maximum_inference_ms": max(elapsed, default=0.0),
        "by_profile": by_profile,
    }


def focused_gate_failures(
    metrics: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    profiles: int,
    folds: int,
    states: int,
    states_by_profile: Mapping[str, int],
    group_leakage: int,
    integrity_errors: int,
) -> list[str]:
    failures: list[str] = []
    if profiles != int(gate["required_profiles"]):
        failures.append("required_profiles")
    if folds != int(gate["required_folds"]):
        failures.append("required_folds")
    if states < int(gate["minimum_total_states"]):
        failures.append("minimum_total_states")
    if any(
        count < int(gate["minimum_states_per_profile"])
        for count in states_by_profile.values()
    ):
        failures.append("minimum_states_per_profile")
    if float(metrics["overall_state_top1_accuracy"]) < float(
        gate["minimum_overall_state_top1_accuracy"]
    ):
        failures.append("overall_state_top1_accuracy")
    if any(
        float(row["state_top1_accuracy"])
        < float(gate["minimum_profile_state_top1_accuracy"])
        for row in metrics["by_profile"].values()
    ):
        failures.append("profile_state_top1_accuracy")
    if float(metrics["game_mean_top1"]["lower_95"]) <= float(
        gate["minimum_game_mean_top1_lower_95"]
    ):
        failures.append("game_mean_top1_lower_95")
    if float(metrics["overall_gain_over_legacy_proxy"]) < float(
        gate["minimum_overall_gain_over_legacy_proxy"]
    ):
        failures.append("overall_gain_over_legacy_proxy")
    if float(metrics["p95_inference_ms"]) > float(
        gate["maximum_p95_inference_ms"]
    ):
        failures.append("p95_inference_ms")
    if bool(gate["require_zero_group_leakage"]) and group_leakage:
        failures.append("group_leakage")
    if bool(gate["require_zero_illegal_predictions"]) and int(
        metrics["illegal_predictions"]
    ):
        failures.append("illegal_predictions")
    if integrity_errors:
        failures.append("integrity_errors")
    return failures


def grouped_fold_leakage(
    decisions_by_profile: Mapping[str, Sequence[TeacherDecision]],
) -> int:
    leakage = 0
    for decisions in decisions_by_profile.values():
        folds_by_game: dict[str, set[int]] = defaultdict(set)
        for decision in decisions:
            folds_by_game[decision.game_id].add(decision.fold)
        leakage += sum(len(folds) != 1 for folds in folds_by_game.values())
    return leakage


def mean_confidence(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    center = fmean(values)
    error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": center,
        "lower_95": center - 1.96 * error,
        "upper_95": center + 1.96 * error,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
