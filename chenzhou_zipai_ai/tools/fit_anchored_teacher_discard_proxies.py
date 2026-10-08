"""Fit and validate the anchored structural+dense seven-teacher proxies."""

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
from statistics import fmean, pstdev, stdev
from typing import Any, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.action_value_model import fit_fixed_action_value_model
from ai.ismcts import public_view_from_dict
from ai.opponent_proxy import (
    FastAggressiveMeldProxyPolicy,
    FastDefensiveSearchProxyPolicy,
    FastInformationSetProxyPolicy,
    FastRedBlackSearchProxyPolicy,
)
from ai.value_corpus import _state_features
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from ai.opponent_league import create_policy
from engine.rules import rules_for_room


@dataclass(frozen=True)
class DecisionRecord:
    group_id: str
    game_id: str
    profile: str
    fold: int
    selected_label: str
    legal_labels: tuple[str, ...]
    public_state: dict[str, Any]
    players: int
    wildcard_enabled: bool
    dealer: int
    turn: int
    seat: int
    legacy_selected: str
    structural_selected: str | None


INDEPENDENT_PROFILES = (
    "independent_balanced",
    "independent_denial",
    "independent_pressure",
)

LEGACY_PROXY_TYPES = {
    "aggressive_meld": FastInformationSetProxyPolicy,
    "defensive_search": FastInformationSetProxyPolicy,
    "independent_balanced": IndependentFastRolloutPolicy,
    "independent_denial": IndependentFastDenialPolicy,
    "independent_pressure": IndependentFastPressurePolicy,
    "information_set_search": FastInformationSetProxyPolicy,
    "red_black_search": FastInformationSetProxyPolicy,
}

STRUCTURAL_PROXY_TYPES = {
    "aggressive_meld": FastAggressiveMeldProxyPolicy,
    "defensive_search": FastDefensiveSearchProxyPolicy,
    "information_set_search": FastInformationSetProxyPolicy,
    "red_black_search": FastRedBlackSearchProxyPolicy,
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
                "teacher_target_mismatches": report[
                    "teacher_target_mismatches"
                ],
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
        raise ValueError("anchored_proxy_profile_mapping_mismatch")
    model_spec = dict(preregistration["independent_dense_model"])
    if tuple(model_spec["profiles"]) != INDEPENDENT_PROFILES:
        raise ValueError("anchored_proxy_independent_profile_order_mismatch")
    folds = int(model_spec["folds"])
    trace_path = Path(preregistration["inputs"]["trace"])
    trace_sha256 = _sha256(trace_path)
    decisions, integrity_errors = collect_decisions(
        trace_path,
        profiles=profiles,
        folds=folds,
    )

    predictions: list[dict[str, Any]] = []
    inference_times: list[float] = []
    target_mismatches = 0
    final_models: dict[str, dict[str, Any]] = {}
    fold_reports: list[dict[str, Any]] = []

    for profile in profiles:
        profile_decisions = decisions[profile]
        if profile not in INDEPENDENT_PROFILES:
            proxy = STRUCTURAL_PROXY_TYPES[profile]()
            for decision in profile_decisions:
                rules = rules_for_room(
                    wildcard_enabled=decision.wildcard_enabled,
                    players=decision.players,
                )
                view = public_view_from_dict(decision.public_state)
                started = time.perf_counter()
                selected = proxy.choose_discard(view, rules)
                inference_times.append((time.perf_counter() - started) * 1000.0)
                predictions.append(prediction_row(decision, selected=selected))
            continue

        profile_index = INDEPENDENT_PROFILES.index(profile)
        rows, row_indices_by_group, mismatches = build_dense_teacher_rows(
            profile_decisions,
            profile=profile,
        )
        target_mismatches += mismatches
        seed = int(model_spec["base_seed"]) + profile_index
        for fold in range(folds):
            training_indices = [
                index
                for decision in profile_decisions
                if decision.fold != fold
                for index in row_indices_by_group[decision.group_id]
            ]
            model, fit_report = fit_fixed_action_value_model(
                rows,
                seed=seed,
                hidden_dimension=int(model_spec["hidden_dimension"]),
                ridge=float(model_spec["ridge"]),
                activation=str(model_spec["activation"]),
                training_indices=training_indices,
            )
            validation = [
                decision for decision in profile_decisions if decision.fold == fold
            ]
            correct = 0
            for decision in validation:
                candidate_rows = [
                    rows[index]
                    for index in row_indices_by_group[decision.group_id]
                ]
                scores = model.predict_rows(candidate_rows)
                selected = max(
                    zip(decision.legal_labels, scores),
                    key=lambda item: (float(item[1]), item[0]),
                )[0]
                correct += int(selected == decision.selected_label)
                predictions.append(prediction_row(decision, selected=selected))
            fold_reports.append(
                {
                    "profile": profile,
                    "fold": fold,
                    "training_games": len(
                        {
                            decision.game_id
                            for decision in profile_decisions
                            if decision.fold != fold
                        }
                    ),
                    "validation_games": len({decision.game_id for decision in validation}),
                    "training_states": sum(
                        decision.fold != fold for decision in profile_decisions
                    ),
                    "validation_states": len(validation),
                    "top1_accuracy": correct / max(1, len(validation)),
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
        model_path = model_dir / f"stage4_anchored_discard_proxy_{profile}_v1.npz"
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
        for decision in profile_decisions:
            candidate_rows = [
                rows[index]
                for index in row_indices_by_group[decision.group_id]
            ]
            started = time.perf_counter()
            scores = final_model.predict_rows(candidate_rows)
            max(
                zip(decision.legal_labels, scores),
                key=lambda item: (float(item[1]), item[0]),
            )
            inference_times.append((time.perf_counter() - started) * 1000.0)

    metrics = summarize_predictions(
        predictions,
        profiles=profiles,
        inference_times=inference_times,
    )
    leakage = grouped_fold_leakage(decisions)
    states_by_profile = {
        profile: len(decisions[profile]) for profile in profiles
    }
    failures = focused_gate_failures(
        metrics,
        gate=preregistration["focused_gate"],
        profiles=len(profiles),
        folds=folds,
        states=len(predictions),
        states_by_profile=states_by_profile,
        group_leakage=leakage,
        target_mismatches=target_mismatches,
        integrity_errors=len(integrity_errors),
    )
    return {
        "ok": not integrity_errors and len(predictions) > 0,
        "schema_version": "anchored-structural-dense-discard-proxies-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "trace": str(trace_path),
        "trace_sha256": trace_sha256,
        "profiles": list(profiles),
        "states": len(predictions),
        "states_by_profile": states_by_profile,
        "group_leakage": leakage,
        "teacher_target_mismatches": target_mismatches,
        "integrity_errors": integrity_errors,
        "metrics": metrics,
        "independent_folds": fold_reports,
        "final_models": final_models,
        "focused_gate": dict(preregistration["focused_gate"]),
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
    }


def collect_decisions(
    trace_path: Path,
    *,
    profiles: Sequence[str],
    folds: int,
) -> tuple[dict[str, list[DecisionRecord]], list[str]]:
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

    decisions: dict[str, list[DecisionRecord]] = {profile: [] for profile in profiles}
    errors: list[str] = []
    for profile in profiles:
        legacy = LEGACY_PROXY_TYPES[profile]()
        structural = (
            STRUCTURAL_PROXY_TYPES[profile]()
            if profile in STRUCTURAL_PROXY_TYPES
            else None
        )
        for game_index, (_, game_id, game) in enumerate(sorted(games_by_profile[profile])):
            fold = game_index % folds
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
                legal_labels = tuple(
                    str(action.get("label") or "")
                    for action in trace.get("legal_actions") or ()
                    if str(action.get("type") or "") == "DISCARD"
                )
                selected = (
                    selected_key.split(":", 1)[1]
                    if selected_key.startswith("DISCARD:")
                    else ""
                )
                if (
                    not selected
                    or selected not in legal_labels
                    or len(set(legal_labels)) != len(legal_labels)
                ):
                    errors.append("teacher_legal_action_contract")
                    continue
                public_state = dict(trace.get("public_state") or {})
                view = public_view_from_dict(public_state)
                try:
                    legacy_selected = legacy.choose_discard(view, rules)
                    structural_selected = (
                        structural.choose_discard(view, rules)
                        if structural is not None
                        else None
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(f"proxy_runtime:{profile}:{type(exc).__name__}")
                    continue
                decisions[profile].append(
                    DecisionRecord(
                        group_id=(
                            f"{game_id}:{int(trace.get('sequence') or 0)}:"
                            f"{trace.get('state_before_hash')}"
                        ),
                        game_id=game_id,
                        profile=profile,
                        fold=fold,
                        selected_label=selected,
                        legal_labels=legal_labels,
                        public_state=public_state,
                        players=int(game.get("players") or 2),
                        wildcard_enabled=bool(game.get("wildcard_enabled")),
                        dealer=int(game.get("dealer") or 0),
                        turn=int(trace.get("turn") or 0),
                        seat=int(trace.get("seat") or 0),
                        legacy_selected=legacy_selected,
                        structural_selected=structural_selected,
                    )
                )
    return decisions, errors


def build_dense_teacher_rows(
    decisions: Sequence[DecisionRecord],
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, ...]], int]:
    teacher = create_policy(profile)
    rows: list[dict[str, Any]] = []
    indices_by_group: dict[str, tuple[int, ...]] = {}
    mismatches = 0
    for decision in decisions:
        rules = rules_for_room(
            wildcard_enabled=decision.wildcard_enabled,
            players=decision.players,
        )
        view = public_view_from_dict(decision.public_state)
        shortlist = teacher.discard_scores(view, rules)
        all_scores = teacher.discard_scores(
            view,
            rules,
            all_candidates=True,
        )
        targets = dense_teacher_targets(shortlist, all_scores)
        target_selected = max(targets, key=lambda label: (targets[label], label))
        mismatches += int(target_selected != decision.selected_label)
        start = len(rows)
        state_features = _state_features(decision.public_state)
        for label in decision.legal_labels:
            rows.append(
                {
                    "game": {
                        "players": decision.players,
                        "wildcard_enabled": decision.wildcard_enabled,
                        "candidate_seat": decision.seat,
                        "dealer": decision.dealer,
                    },
                    "phase": "discard",
                    "turn": decision.turn,
                    "seat": decision.seat,
                    "group_id": decision.group_id,
                    "selected": label == decision.selected_label,
                    "action": {
                        "key": f"DISCARD:{label}",
                        "type": "DISCARD",
                        "label": label,
                        "option_id": None,
                        "consumed_from_hand": [],
                        "meld_groups": [],
                        "followup_discard": None,
                        "heuristic_value": 0.0,
                    },
                    "state_features": state_features,
                    "targets": {"mean_reward": targets[label]},
                    "audit_health": {"paired_worlds": 8},
                }
            )
        indices_by_group[decision.group_id] = tuple(range(start, len(rows)))
    return rows, indices_by_group, mismatches


def dense_teacher_targets(
    shortlist: Mapping[str, tuple[float, Any]],
    all_scores: Mapping[str, tuple[float, Any]],
) -> dict[str, float]:
    if not shortlist or not all_scores or not set(shortlist).issubset(all_scores):
        raise ValueError("dense_teacher_scores_incomplete")
    shortlisted_values = [float(item[0]) for item in shortlist.values()]
    rejection_gap = max(1.0, pstdev(shortlisted_values))
    rejection_floor = min(shortlisted_values) - rejection_gap
    raw = {
        label: (
            float(shortlist[label][0])
            if label in shortlist
            else rejection_floor
        )
        for label in all_scores
    }
    center = fmean(raw.values())
    scale = max(1e-8, pstdev(raw.values()))
    return {label: (value - center) / scale for label, value in raw.items()}


def prediction_row(
    decision: DecisionRecord,
    *,
    selected: str,
) -> dict[str, Any]:
    return {
        "group_id": decision.group_id,
        "game_id": decision.game_id,
        "profile": decision.profile,
        "fold": decision.fold,
        "correct": selected == decision.selected_label,
        "legacy_correct": decision.legacy_selected == decision.selected_label,
        "illegal": selected not in decision.legal_labels,
    }


def summarize_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    inference_times: Sequence[float],
) -> dict[str, Any]:
    by_profile = {
        profile: {
            "states": len(rows),
            "state_top1_accuracy": fmean(bool(row["correct"]) for row in rows),
            "legacy_state_top1_accuracy": fmean(
                bool(row["legacy_correct"]) for row in rows
            ),
        }
        for profile in profiles
        for rows in [[row for row in predictions if row["profile"] == profile]]
    }
    games: dict[str, list[bool]] = defaultdict(list)
    for row in predictions:
        games[str(row["game_id"])].append(bool(row["correct"]))
    elapsed = sorted(float(value) for value in inference_times)
    p95_index = min(len(elapsed) - 1, int((len(elapsed) - 1) * 0.95)) if elapsed else 0
    return {
        "overall_state_top1_accuracy": fmean(bool(row["correct"]) for row in predictions),
        "legacy_state_top1_accuracy": fmean(
            bool(row["legacy_correct"]) for row in predictions
        ),
        "overall_gain_over_legacy_proxy": fmean(
            int(bool(row["correct"])) - int(bool(row["legacy_correct"]))
            for row in predictions
        ),
        "illegal_predictions": sum(bool(row["illegal"]) for row in predictions),
        "game_mean_top1": mean_confidence(
            [fmean(values) for values in games.values()]
        ),
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
    target_mismatches: int,
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
    minimums = dict(gate["minimum_profile_top1_accuracy"])
    if any(
        float(metrics["by_profile"][profile]["state_top1_accuracy"])
        < float(minimums[profile])
        for profile in minimums
    ):
        failures.append("profile_top1_accuracy")
    if float(metrics["overall_state_top1_accuracy"]) < float(
        gate["minimum_overall_state_top1_accuracy"]
    ):
        failures.append("overall_state_top1_accuracy")
    if float(metrics["game_mean_top1"]["lower_95"]) <= float(
        gate["minimum_game_mean_top1_lower_95"]
    ):
        failures.append("game_mean_top1_lower_95")
    if float(metrics["overall_gain_over_legacy_proxy"]) < float(
        gate["minimum_overall_gain_over_legacy_proxy"]
    ):
        failures.append("overall_gain_over_legacy_proxy")
    if float(metrics["p95_inference_ms"]) > float(gate["maximum_p95_inference_ms"]):
        failures.append("p95_inference_ms")
    if bool(gate["require_zero_group_leakage"]) and group_leakage:
        failures.append("group_leakage")
    if bool(gate["require_zero_teacher_target_mismatches"]) and target_mismatches:
        failures.append("teacher_target_mismatches")
    if bool(gate["require_zero_illegal_predictions"]) and int(
        metrics["illegal_predictions"]
    ):
        failures.append("illegal_predictions")
    if integrity_errors:
        failures.append("integrity_errors")
    return failures


def grouped_fold_leakage(
    decisions: Mapping[str, Sequence[DecisionRecord]],
) -> int:
    leakage = 0
    for profile_decisions in decisions.values():
        folds_by_game: dict[str, set[int]] = defaultdict(set)
        for decision in profile_decisions:
            folds_by_game[decision.game_id].add(decision.fold)
        leakage += sum(len(values) != 1 for values in folds_by_game.values())
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
