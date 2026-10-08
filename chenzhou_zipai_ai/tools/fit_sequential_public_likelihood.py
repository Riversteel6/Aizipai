"""Fit and group-validate observer-safe sequential opponent likelihoods."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.opponent_belief import public_opponent_features
from engine.cards import BIG_LABELS, SMALL_LABELS
from engine.rules import rules_for_room
from tools.build_sequential_public_evidence_dataset import assign_game_folds
from tools.evaluate_anchored_structural_discard_proxies import sha256_file
from tools.fit_public_opponent_belief import (
    BeliefCase,
    fit_weighted_ridge_model,
    mean_confidence_interval,
)


DISCARD_TOKENS = tuple((*SMALL_LABELS, *BIG_LABELS))
RESPONSE_TOKENS = ("NO_CLAIM", "CHI", "PENG", "HU")
STRATA = ("0_to_2", "3_to_5", "6_plus")


@dataclass(frozen=True)
class Checkpoint:
    game_id: str
    profile: str
    fold: int
    sequence: int
    public_actions: int
    features: tuple[float, ...]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_experiment(
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
                "games": report["games"],
                "checkpoints": report["checkpoints"],
                "paired_improvements": report["paired_improvements"],
                "maximum_low_evidence_posterior": report[
                    "maximum_low_evidence_posterior"
                ],
                "update_p95_ms": report["update_p95_ms"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] and report["focused_gate_pass"] else 1


def run_experiment(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    for key in ("dataset", "dataset_audit", "trace", "baseline_report"):
        if sha256_file(Path(str(inputs[key]))) != str(inputs[f"{key}_sha256"]):
            raise ValueError(f"sequential_likelihood_{key}_hash_mismatch")
    dataset_audit = _read_json(Path(str(inputs["dataset_audit"])))
    if not dataset_audit.get("ok") or not dataset_audit.get("integrity_gate_pass"):
        raise ValueError("sequential_likelihood_dataset_audit_did_not_pass")
    baseline_report = _read_json(Path(str(inputs["baseline_report"])))
    baseline_prereg = _read_json(Path(str(baseline_report["preregistration"])))
    baseline_spec = dict(baseline_prereg["model"])

    profiles = tuple(str(item) for item in preregistration["profiles"])
    model_spec = dict(preregistration["model"])
    folds = int(model_spec["folds"])
    events = load_events(Path(str(inputs["dataset"])))
    checkpoints = collect_checkpoints(
        Path(str(inputs["trace"])),
        profiles=profiles,
        folds=folds,
    )
    events_by_game = _group_by_game(events)
    checkpoints_by_game = _group_checkpoints(checkpoints)
    if set(events_by_game) != set(checkpoints_by_game):
        raise ValueError("sequential_likelihood_game_set_mismatch")

    per_checkpoint: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    update_times = []
    unknown_events = 0
    group_leakage = 0
    maximum_low_evidence_posterior = 0.0
    fold_reports = []
    for fold in range(folds):
        training_events = [event for event in events if int(event["fold"]) != fold]
        training_checkpoints = [
            checkpoint for checkpoint in checkpoints if checkpoint.fold != fold
        ]
        validation_games = {
            checkpoint.game_id for checkpoint in checkpoints if checkpoint.fold == fold
        }
        training_games = {
            checkpoint.game_id for checkpoint in training_checkpoints
        }
        overlap = training_games & validation_games
        group_leakage += len(overlap)
        likelihood = fit_likelihood_model(
            training_events,
            profiles=profiles,
            alpha=float(model_spec["dirichlet_alpha"]),
        )
        baseline = fit_weighted_ridge_model(
            [
                BeliefCase(
                    game_id=item.game_id,
                    profile=item.profile,
                    fold=item.fold,
                    public_actions=item.public_actions,
                    features=item.features,
                )
                for item in training_checkpoints
            ],
            profiles=profiles,
            ridge=float(baseline_spec["ridge"]),
            temperature=float(baseline_spec["temperature"]),
            full_reliability_public_actions=int(
                baseline_spec["full_reliability_public_actions"]
            ),
        )
        fold_checkpoint_count = 0
        for game_id in sorted(validation_games):
            game_events = sorted(
                events_by_game[game_id],
                key=lambda item: (
                    int(item["source_sequence"]),
                    int(item["event_order"]),
                ),
            )
            game_checkpoints = sorted(
                checkpoints_by_game[game_id], key=lambda item: item.sequence
            )
            log_scores = {profile: -math.log(len(profiles)) for profile in profiles}
            event_index = 0
            evidence_count = 0
            for checkpoint in game_checkpoints:
                while (
                    event_index < len(game_events)
                    and int(game_events[event_index]["source_sequence"])
                    < checkpoint.sequence
                ):
                    event = game_events[event_index]
                    started = time.perf_counter()
                    try:
                        log_scores = update_log_scores(
                            log_scores,
                            event,
                            model=likelihood,
                            profiles=profiles,
                        )
                    except KeyError:
                        unknown_events += 1
                    update_times.append((time.perf_counter() - started) * 1000.0)
                    evidence_count += 1
                    event_index += 1
                posterior = normalize_log_scores(log_scores, profiles=profiles)
                baseline_posterior = baseline.posterior(
                    checkpoint.features,
                    public_actions=checkpoint.public_actions,
                )
                true_profile = checkpoint.profile
                metrics = checkpoint_metrics(
                    baseline_posterior,
                    posterior,
                    true_profile=true_profile,
                    profiles=profiles,
                    evidence_count=evidence_count,
                )
                per_checkpoint[game_id].append(metrics)
                fold_checkpoint_count += 1
                if evidence_count <= 2:
                    maximum_low_evidence_posterior = max(
                        maximum_low_evidence_posterior,
                        max(posterior.values()),
                    )
        fold_reports.append(
            {
                "fold": fold,
                "training_games": len(training_games),
                "validation_games": len(validation_games),
                "training_events": len(training_events),
                "validation_checkpoints": fold_checkpoint_count,
                "group_overlap": len(overlap),
            }
        )

    per_game = [
        summarize_game(game_id, per_checkpoint[game_id])
        for game_id in sorted(per_checkpoint)
    ]
    baseline_metrics = summarize_model(per_game, prefix="baseline")
    sequence_metrics = summarize_model(per_game, prefix="sequence")
    improvements = paired_improvements(per_game)
    reproduction_error = baseline_reproduction_error(
        baseline_metrics,
        baseline_report["metrics"],
    )
    update_times.sort()
    p95_index = (
        min(len(update_times) - 1, int((len(update_times) - 1) * 0.95))
        if update_times
        else 0
    )
    update_p95 = update_times[p95_index] if update_times else 0.0
    gate = dict(preregistration["focused_gate"])
    failures = focused_gate_failures(
        profiles=profiles,
        games=len(per_game),
        checkpoints=sum(len(values) for values in per_checkpoint.values()),
        folds=folds,
        group_leakage=group_leakage,
        unknown_events=unknown_events,
        reproduction_error=reproduction_error,
        improvements=improvements,
        maximum_low_evidence_posterior=maximum_low_evidence_posterior,
        update_p95_ms=update_p95,
        gate=gate,
    )
    final_model = fit_likelihood_model(
        events,
        profiles=profiles,
        alpha=float(model_spec["dirichlet_alpha"]),
    )
    return {
        "ok": bool(per_game) and unknown_events == 0,
        "schema_version": "sequential-public-likelihood-grouped-cv-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "profiles": list(profiles),
        "games": len(per_game),
        "checkpoints": sum(len(values) for values in per_checkpoint.values()),
        "events": len(events),
        "folds": fold_reports,
        "group_leakage": group_leakage,
        "unknown_events": unknown_events,
        "baseline_metrics": baseline_metrics,
        "sequence_metrics": sequence_metrics,
        "paired_improvements": improvements,
        "maximum_low_evidence_posterior": maximum_low_evidence_posterior,
        "update_p95_ms": update_p95,
        "baseline_reproduction_error": reproduction_error,
        "by_profile": summarize_by_profile(per_game, profiles=profiles),
        "by_evidence_count": summarize_by_evidence_count(per_checkpoint),
        "final_model": final_model,
        "focused_gate": gate,
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
    }


def fit_likelihood_model(
    events: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    alpha: float,
) -> dict[str, Any]:
    if alpha <= 0.0:
        raise ValueError("sequential_likelihood_alpha")
    counts: dict[str, Counter[str]] = {
        profile: Counter() for profile in profiles
    }
    totals: dict[str, Counter[str]] = {
        profile: Counter() for profile in profiles
    }
    for event in events:
        profile = str(event["profile"])
        channel, stratum, token = event_likelihood_key(event)
        counts[profile][f"{channel}|{stratum}|{token}"] += 1
        totals[profile][f"{channel}|{stratum}"] += 1
    return {
        "schema_version": "profile-conditional-public-event-likelihood-v1",
        "profiles": list(profiles),
        "alpha": alpha,
        "discard_tokens": list(DISCARD_TOKENS),
        "response_tokens": list(RESPONSE_TOKENS),
        "strata": list(STRATA),
        "counts": {
            profile: dict(counts[profile]) for profile in profiles
        },
        "totals": {
            profile: dict(totals[profile]) for profile in profiles
        },
    }


def update_log_scores(
    log_scores: Mapping[str, float],
    event: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    profiles: Sequence[str],
) -> dict[str, float]:
    channel, stratum, token = event_likelihood_key(event)
    vocabulary = DISCARD_TOKENS if channel == "discard" else RESPONSE_TOKENS
    if token not in vocabulary:
        raise KeyError(token)
    alpha = float(model["alpha"])
    result = {}
    for profile in profiles:
        count = int(
            model["counts"][profile].get(
                f"{channel}|{stratum}|{token}", 0
            )
        )
        total = int(
            model["totals"][profile].get(f"{channel}|{stratum}", 0)
        )
        probability = (count + alpha) / (
            total + alpha * len(vocabulary)
        )
        result[profile] = float(log_scores[profile]) + math.log(probability)
    return result


def normalize_log_scores(
    log_scores: Mapping[str, float],
    *,
    profiles: Sequence[str],
) -> dict[str, float]:
    maximum = max(float(log_scores[profile]) for profile in profiles)
    values = {
        profile: math.exp(float(log_scores[profile]) - maximum)
        for profile in profiles
    }
    total = sum(values.values())
    return {profile: values[profile] / total for profile in profiles}


def event_likelihood_key(event: Mapping[str, Any]) -> tuple[str, str, str]:
    kind = str(event["action_kind"])
    if kind == "DISCARD":
        channel = "discard"
        token = str(event["action_token"]).split(":", 1)[1]
    else:
        channel = "response"
        token = kind
    return channel, evidence_stratum(int(event["public_actions_before"])), token


def evidence_stratum(public_actions: int) -> str:
    if public_actions <= 2:
        return "0_to_2"
    if public_actions <= 5:
        return "3_to_5"
    return "6_plus"


def collect_checkpoints(
    trace: Path,
    *,
    profiles: Sequence[str],
    folds: int,
) -> list[Checkpoint]:
    fold_by_source, game_ids, profile_by_source = assign_game_folds(
        trace,
        profiles=profiles,
        folds=folds,
    )
    checkpoints = []
    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            game = json.loads(line)
            candidate_seat = int(game["candidate_seat"])
            rules = rules_for_room(
                wildcard_enabled=bool(game.get("wildcard_enabled")),
                players=int(game.get("players") or 2),
            )
            for trace_event in game.get("decision_trace") or ():
                if (
                    int(trace_event.get("seat") or 0) != candidate_seat
                    or str(trace_event.get("phase") or "") != "discard"
                ):
                    continue
                evidence = public_opponent_features(
                    trace_event["public_state"], rules
                )
                checkpoints.append(
                    Checkpoint(
                        game_id=game_ids[source_index],
                        profile=profile_by_source[source_index],
                        fold=fold_by_source[source_index],
                        sequence=int(trace_event.get("sequence") or 0),
                        public_actions=evidence.public_actions,
                        features=evidence.features,
                    )
                )
    return checkpoints


def checkpoint_metrics(
    baseline: Mapping[str, float],
    sequence: Mapping[str, float],
    *,
    true_profile: str,
    profiles: Sequence[str],
    evidence_count: int,
) -> dict[str, float | int]:
    return {
        "baseline_log_loss": -math.log(max(1e-15, baseline[true_profile])),
        "baseline_true_probability": baseline[true_profile],
        "baseline_top1": int(
            selected_profile(baseline, profiles=profiles) == true_profile
        ),
        "sequence_log_loss": -math.log(max(1e-15, sequence[true_profile])),
        "sequence_true_probability": sequence[true_profile],
        "sequence_top1": int(
            selected_profile(sequence, profiles=profiles) == true_profile
        ),
        "evidence_count": evidence_count,
    }


def selected_profile(
    posterior: Mapping[str, float],
    *,
    profiles: Sequence[str],
) -> str:
    return max(profiles, key=lambda profile: (posterior[profile], profile))


def summarize_game(
    game_id: str,
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    profile = game_id.split(":", 1)[0]
    return {
        "game_id": game_id,
        "profile": profile,
        "checkpoints": len(rows),
        **{
            key: fmean(float(row[key]) for row in rows)
            for key in (
                "baseline_log_loss",
                "baseline_true_probability",
                "baseline_top1",
                "sequence_log_loss",
                "sequence_true_probability",
                "sequence_top1",
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


def paired_improvements(
    per_game: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "log_loss": mean_confidence_interval(
            [
                float(row["baseline_log_loss"])
                - float(row["sequence_log_loss"])
                for row in per_game
            ]
        ),
        "true_probability": mean_confidence_interval(
            [
                float(row["sequence_true_probability"])
                - float(row["baseline_true_probability"])
                for row in per_game
            ]
        ),
        "top1": mean_confidence_interval(
            [
                float(row["sequence_top1"])
                - float(row["baseline_top1"])
                for row in per_game
            ]
        ),
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


def summarize_by_profile(
    per_game: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
) -> dict[str, Any]:
    return {
        profile: {
            "games": len(rows),
            "baseline": summarize_model(rows, prefix="baseline"),
            "sequence": summarize_model(rows, prefix="sequence"),
            "improvements": paired_improvements(rows),
        }
        for profile in profiles
        for rows in [[row for row in per_game if row["profile"] == profile]]
    }


def summarize_by_evidence_count(
    per_checkpoint: Mapping[str, Sequence[Mapping[str, float | int]]],
) -> dict[str, Any]:
    rows = [row for values in per_checkpoint.values() for row in values]
    result = {}
    for name, predicate in (
        ("0_to_2", lambda count: count <= 2),
        ("3_to_5", lambda count: 3 <= count <= 5),
        ("6_plus", lambda count: count >= 6),
    ):
        selected = [row for row in rows if predicate(int(row["evidence_count"]))]
        result[name] = {
            "checkpoints": len(selected),
            "baseline_log_loss": fmean(
                float(row["baseline_log_loss"]) for row in selected
            ),
            "sequence_log_loss": fmean(
                float(row["sequence_log_loss"]) for row in selected
            ),
            "baseline_true_probability": fmean(
                float(row["baseline_true_probability"]) for row in selected
            ),
            "sequence_true_probability": fmean(
                float(row["sequence_true_probability"]) for row in selected
            ),
            "baseline_top1": fmean(
                float(row["baseline_top1"]) for row in selected
            ),
            "sequence_top1": fmean(
                float(row["sequence_top1"]) for row in selected
            ),
        }
    return result


def focused_gate_failures(
    *,
    profiles: Sequence[str],
    games: int,
    checkpoints: int,
    folds: int,
    group_leakage: int,
    unknown_events: int,
    reproduction_error: float,
    improvements: Mapping[str, Mapping[str, Any]],
    maximum_low_evidence_posterior: float,
    update_p95_ms: float,
    gate: Mapping[str, Any],
) -> list[str]:
    failures = []
    if len(profiles) != int(gate["required_profiles"]):
        failures.append("required_profiles")
    if games != int(gate["required_games"]):
        failures.append("required_games")
    if checkpoints != int(gate["required_checkpoints"]):
        failures.append("required_checkpoints")
    if folds != int(gate["required_folds"]):
        failures.append("required_folds")
    if group_leakage > int(gate["maximum_group_leakage"]):
        failures.append("group_leakage")
    if unknown_events > int(gate["maximum_unknown_events"]):
        failures.append("unknown_events")
    if reproduction_error > float(
        gate["maximum_baseline_metric_reproduction_error"]
    ):
        failures.append("baseline_metric_reproduction")
    for name in ("log_loss", "true_probability", "top1"):
        if float(improvements[name]["lower_95"]) < float(
            gate[f"minimum_paired_{name}_improvement_lower_95"]
        ):
            failures.append(f"paired_{name}_improvement_lower_95")
    if maximum_low_evidence_posterior > float(
        gate["maximum_low_evidence_posterior"]
    ):
        failures.append("maximum_low_evidence_posterior")
    if update_p95_ms > float(gate["maximum_update_p95_ms"]):
        failures.append("update_p95_ms")
    return failures


def load_events(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _group_by_game(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["game_id"])].append(event)
    return grouped


def _group_checkpoints(
    checkpoints: Sequence[Checkpoint],
) -> dict[str, list[Checkpoint]]:
    grouped: dict[str, list[Checkpoint]] = defaultdict(list)
    for checkpoint in checkpoints:
        grouped[checkpoint.game_id].append(checkpoint)
    return grouped


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
