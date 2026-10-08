"""Nested grouped validation for order-only opponent-belief evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from engine.cards import BIG_LABELS, RED_LABELS, SMALL_LABELS
from tools.evaluate_anchored_structural_discard_proxies import sha256_file
from tools.fit_anchored_sequential_belief_fusion import (
    baseline_reproduction_error,
    choose_power,
    evaluate_fusion_records,
    focused_gate_failures,
    group_checkpoints,
    group_events,
    nested_calibration_limit,
    paired_improvements,
    resolved_power_grid,
    summarize_by_profile,
    summarize_model,
    summarize_records_by_game,
)
from tools.fit_public_opponent_belief import (
    BeliefCase,
    fit_weighted_ridge_model,
)
from tools.fit_sequential_public_likelihood import (
    DISCARD_TOKENS,
    RESPONSE_TOKENS,
    Checkpoint,
    collect_checkpoints,
    event_likelihood_key,
    load_events,
)


PREVIOUS_EVENT_CLASSES = (
    "DISCARD_SMALL_RED",
    "DISCARD_SMALL_BLACK",
    "DISCARD_BIG_RED",
    "DISCARD_BIG_BLACK",
    "NO_CLAIM",
    "CHI",
    "PENG",
    "HU",
)


def previous_event_class(event: Mapping[str, Any]) -> str:
    kind = str(event["action_kind"])
    if kind != "DISCARD":
        if kind not in PREVIOUS_EVENT_CLASSES:
            raise KeyError(kind)
        return kind
    token = str(event["action_token"])
    if not token.startswith("DISCARD:"):
        raise KeyError(token)
    label = token.split(":", 1)[1]
    if label in SMALL_LABELS:
        size = "SMALL"
    elif label in BIG_LABELS:
        size = "BIG"
    else:
        raise KeyError(label)
    color = "RED" if label in RED_LABELS else "BLACK"
    return f"DISCARD_{size}_{color}"


def fit_transition_innovation_model(
    events: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    alpha: float,
) -> dict[str, Any]:
    if alpha <= 0.0:
        raise ValueError("transition_innovation_alpha")
    marginal_counts = {profile: Counter() for profile in profiles}
    marginal_totals = {profile: Counter() for profile in profiles}
    transition_counts = {profile: Counter() for profile in profiles}
    transition_totals = {profile: Counter() for profile in profiles}
    grouped = group_events(events)
    transition_events = 0
    for game_id in sorted(grouped):
        game_events = sorted(
            grouped[game_id],
            key=lambda item: (
                int(item["source_sequence"]),
                int(item["event_order"]),
            ),
        )
        previous_class: str | None = None
        for event in game_events:
            profile = str(event["profile"])
            channel, stratum, token = event_likelihood_key(event)
            marginal_counts[profile][f"{channel}|{stratum}|{token}"] += 1
            marginal_totals[profile][f"{channel}|{stratum}"] += 1
            if previous_class is not None:
                transition_counts[profile][
                    f"{channel}|{stratum}|{previous_class}|{token}"
                ] += 1
                transition_totals[profile][
                    f"{channel}|{stratum}|{previous_class}"
                ] += 1
                transition_events += 1
            previous_class = previous_event_class(event)
    return {
        "schema_version": "profile-conditional-transition-innovation-v1",
        "profiles": list(profiles),
        "alpha": alpha,
        "previous_event_classes": list(PREVIOUS_EVENT_CLASSES),
        "discard_tokens": list(DISCARD_TOKENS),
        "response_tokens": list(RESPONSE_TOKENS),
        "games": len(grouped),
        "events": len(events),
        "transition_events": transition_events,
        "marginal_counts": {
            profile: dict(marginal_counts[profile]) for profile in profiles
        },
        "marginal_totals": {
            profile: dict(marginal_totals[profile]) for profile in profiles
        },
        "transition_counts": {
            profile: dict(transition_counts[profile]) for profile in profiles
        },
        "transition_totals": {
            profile: dict(transition_totals[profile]) for profile in profiles
        },
    }


def update_transition_innovation_scores(
    log_scores: Mapping[str, float],
    event: Mapping[str, Any],
    previous_event: Mapping[str, Any] | None,
    *,
    model: Mapping[str, Any],
    profiles: Sequence[str],
) -> dict[str, float]:
    if previous_event is None:
        return {profile: float(log_scores[profile]) for profile in profiles}
    channel, stratum, token = event_likelihood_key(event)
    vocabulary = DISCARD_TOKENS if channel == "discard" else RESPONSE_TOKENS
    if token not in vocabulary:
        raise KeyError(token)
    previous_class = previous_event_class(previous_event)
    if previous_class not in PREVIOUS_EVENT_CLASSES:
        raise KeyError(previous_class)
    alpha = float(model["alpha"])
    result = {}
    for profile in profiles:
        marginal_count = int(
            model["marginal_counts"][profile].get(
                f"{channel}|{stratum}|{token}", 0
            )
        )
        marginal_total = int(
            model["marginal_totals"][profile].get(
                f"{channel}|{stratum}", 0
            )
        )
        transition_count = int(
            model["transition_counts"][profile].get(
                f"{channel}|{stratum}|{previous_class}|{token}", 0
            )
        )
        transition_total = int(
            model["transition_totals"][profile].get(
                f"{channel}|{stratum}|{previous_class}", 0
            )
        )
        marginal_probability = (marginal_count + alpha) / (
            marginal_total + alpha * len(vocabulary)
        )
        transition_probability = (transition_count + alpha) / (
            transition_total + alpha * len(vocabulary)
        )
        result[profile] = (
            float(log_scores[profile])
            + math.log(transition_probability)
            - math.log(marginal_probability)
        )
    return result


def transition_component_records(
    events: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Checkpoint],
    *,
    events_by_game: Mapping[str, Sequence[Mapping[str, Any]]],
    checkpoints_by_game: Mapping[str, Sequence[Checkpoint]],
    training_folds: Sequence[int],
    validation_folds: Sequence[int],
    profiles: Sequence[str],
    alpha: float,
    baseline_spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_fold_set = set(training_folds)
    validation_fold_set = set(validation_folds)
    training_checkpoints = [
        item for item in checkpoints if item.fold in training_fold_set
    ]
    training_events = [
        item for item in events if int(item["fold"]) in training_fold_set
    ]
    validation_games = {
        item.game_id for item in checkpoints if item.fold in validation_fold_set
    }
    training_games = {item.game_id for item in training_checkpoints}
    overlap = training_games & validation_games
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
    transition = fit_transition_innovation_model(
        training_events,
        profiles=profiles,
        alpha=alpha,
    )
    records = []
    unknown_events = 0
    elapsed_samples = []
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
        sequence_scores = {profile: 0.0 for profile in profiles}
        event_index = 0
        evidence_count = 0
        previous_event: Mapping[str, Any] | None = None
        for checkpoint in game_checkpoints:
            while (
                event_index < len(game_events)
                and int(game_events[event_index]["source_sequence"])
                < checkpoint.sequence
            ):
                current_event = game_events[event_index]
                started = time.perf_counter()
                try:
                    sequence_scores = update_transition_innovation_scores(
                        sequence_scores,
                        current_event,
                        previous_event,
                        model=transition,
                        profiles=profiles,
                    )
                    previous_event_class(current_event)
                except KeyError:
                    unknown_events += 1
                elapsed_samples.append((time.perf_counter() - started) * 1000.0)
                previous_event = current_event
                evidence_count += 1
                event_index += 1
            records.append(
                {
                    "game_id": game_id,
                    "profile": checkpoint.profile,
                    "fold": checkpoint.fold,
                    "evidence_count": evidence_count,
                    "baseline_posterior": baseline.posterior(
                        checkpoint.features,
                        public_actions=checkpoint.public_actions,
                    ),
                    "sequence_scores": dict(sequence_scores),
                }
            )
    return records, {
        "unknown_events": unknown_events,
        "group_overlap": len(overlap),
        "elapsed_samples": elapsed_samples,
    }


def run_experiment(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    for key in (
        "dataset",
        "dataset_audit",
        "trace",
        "baseline_report",
        "failed_raw_fusion_report",
    ):
        if sha256_file(Path(str(inputs[key]))) != str(inputs[f"{key}_sha256"]):
            raise ValueError(f"transition_innovation_{key}_hash_mismatch")
    dataset_audit = _read_json(Path(str(inputs["dataset_audit"])))
    if not dataset_audit.get("integrity_gate_pass"):
        raise ValueError("transition_innovation_dataset_audit_did_not_pass")
    raw_fusion = _read_json(Path(str(inputs["failed_raw_fusion_report"])))
    if raw_fusion.get("focused_gate_pass"):
        raise ValueError("transition_innovation_expected_failed_raw_anchor")

    baseline_report = _read_json(Path(str(inputs["baseline_report"])))
    baseline_prereg = _read_json(Path(str(baseline_report["preregistration"])))
    baseline_spec = dict(baseline_prereg["model"])
    profiles = tuple(str(item) for item in preregistration["profiles"])
    for source_name, source_report in (
        ("baseline", baseline_report),
        ("raw_fusion", raw_fusion),
    ):
        source_profiles = tuple(str(item) for item in source_report["profiles"])
        if source_profiles != profiles:
            raise ValueError(f"transition_innovation_{source_name}_profile_mismatch")
    configured_classes = tuple(
        str(item)
        for item in preregistration["transition_representation"][
            "previous_event_classes"
        ]
    )
    if configured_classes != PREVIOUS_EVENT_CLASSES:
        raise ValueError("transition_innovation_class_contract_mismatch")

    nested = dict(preregistration["nested_calibration"])
    folds = int(nested["outer_folds"])
    events = load_events(Path(str(inputs["dataset"])))
    checkpoints = collect_checkpoints(
        Path(str(inputs["trace"])),
        profiles=profiles,
        folds=folds,
    )
    events_by_game = group_events(events)
    checkpoints_by_game = group_checkpoints(checkpoints)
    if set(events_by_game) != set(checkpoints_by_game):
        raise ValueError("transition_innovation_game_set_mismatch")
    alpha = float(raw_fusion["final_model"]["sequence_likelihood"]["alpha"])
    power_grid = resolved_power_grid(nested["power_grid"])

    all_outer_records = []
    outer_reports = []
    outer_powers = []
    unknown_events = 0
    group_leakage = 0
    elapsed_samples = []
    for outer_fold in range(folds):
        outer_training_folds = tuple(
            fold for fold in range(folds) if fold != outer_fold
        )
        inner_records = []
        inner_reports = []
        for inner_fold in outer_training_folds:
            inner_training_folds = tuple(
                fold for fold in outer_training_folds if fold != inner_fold
            )
            records, diagnostics = transition_component_records(
                events,
                checkpoints,
                events_by_game=events_by_game,
                checkpoints_by_game=checkpoints_by_game,
                training_folds=inner_training_folds,
                validation_folds=(inner_fold,),
                profiles=profiles,
                alpha=alpha,
                baseline_spec=baseline_spec,
            )
            inner_records.extend(records)
            unknown_events += int(diagnostics["unknown_events"])
            group_leakage += int(diagnostics["group_overlap"])
            elapsed_samples.extend(diagnostics["elapsed_samples"])
            inner_reports.append(
                {
                    "heldout_fold": inner_fold,
                    "training_folds": list(inner_training_folds),
                    "records": len(records),
                    "group_overlap": diagnostics["group_overlap"],
                }
            )
        selected_power, calibration = choose_power(
            inner_records,
            profiles=profiles,
            grid=power_grid,
            maximum_low_evidence_posterior=float(
                nested_calibration_limit(preregistration)
            ),
        )
        outer_powers.append(selected_power)
        outer_records, diagnostics = transition_component_records(
            events,
            checkpoints,
            events_by_game=events_by_game,
            checkpoints_by_game=checkpoints_by_game,
            training_folds=outer_training_folds,
            validation_folds=(outer_fold,),
            profiles=profiles,
            alpha=alpha,
            baseline_spec=baseline_spec,
        )
        unknown_events += int(diagnostics["unknown_events"])
        group_leakage += int(diagnostics["group_overlap"])
        elapsed_samples.extend(diagnostics["elapsed_samples"])
        evaluated = evaluate_fusion_records(
            outer_records,
            power=selected_power,
            profiles=profiles,
            elapsed_samples=elapsed_samples,
        )
        all_outer_records.extend(evaluated)
        outer_reports.append(
            {
                "outer_fold": outer_fold,
                "training_folds": list(outer_training_folds),
                "selected_power": selected_power,
                "inner_folds": inner_reports,
                "inner_calibration": calibration,
                "outer_records": len(evaluated),
                "group_overlap": diagnostics["group_overlap"],
            }
        )

    final_power, final_calibration = choose_power(
        all_outer_records,
        profiles=profiles,
        grid=power_grid,
        maximum_low_evidence_posterior=float(
            nested_calibration_limit(preregistration)
        ),
    )
    per_game = summarize_records_by_game(all_outer_records)
    baseline_metrics = summarize_model(per_game, prefix="baseline")
    fusion_metrics = summarize_model(per_game, prefix="fusion")
    improvements = paired_improvements(per_game)
    by_profile = summarize_by_profile(per_game, profiles=profiles)
    reproduction_error = baseline_reproduction_error(
        baseline_metrics,
        baseline_report["metrics"],
    )
    maximum_low_evidence = max(
        (
            float(record["fusion_max_probability"])
            for record in all_outer_records
            if int(record["evidence_count"]) <= 2
        ),
        default=0.0,
    )
    elapsed_samples.sort()
    p95_index = (
        min(len(elapsed_samples) - 1, int((len(elapsed_samples) - 1) * 0.95))
        if elapsed_samples
        else 0
    )
    p95_ms = elapsed_samples[p95_index] if elapsed_samples else 0.0
    gate = dict(preregistration["focused_gate"])
    failures = focused_gate_failures(
        profiles=profiles,
        games=len(per_game),
        checkpoints=len(all_outer_records),
        outer_reports=outer_reports,
        group_leakage=group_leakage,
        unknown_events=unknown_events,
        reproduction_error=reproduction_error,
        improvements=improvements,
        by_profile=by_profile,
        maximum_low_evidence_posterior=maximum_low_evidence,
        p95_ms=p95_ms,
        outer_powers=outer_powers,
        final_power=final_power,
        gate=gate,
    )
    final_transition = fit_transition_innovation_model(
        events,
        profiles=profiles,
        alpha=alpha,
    )
    return {
        "ok": bool(all_outer_records) and unknown_events == 0,
        "schema_version": "anchored-transition-innovation-nested-cv-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "profiles": list(profiles),
        "games": len(per_game),
        "checkpoints": len(all_outer_records),
        "events": len(events),
        "outer_folds": outer_reports,
        "outer_powers": outer_powers,
        "final_power": final_power,
        "final_power_calibration": final_calibration,
        "group_leakage": group_leakage,
        "unknown_events": unknown_events,
        "baseline_metrics": baseline_metrics,
        "fusion_metrics": fusion_metrics,
        "paired_improvements": improvements,
        "by_profile": by_profile,
        "maximum_low_evidence_posterior": maximum_low_evidence,
        "update_and_fusion_p95_ms": p95_ms,
        "baseline_reproduction_error": reproduction_error,
        "failed_raw_fusion_anchor": {
            "report": str(inputs["failed_raw_fusion_report"]),
            "paired_improvements": raw_fusion["paired_improvements"],
            "focused_gate_failures": raw_fusion["focused_gate_failures"],
        },
        "final_model": {
            "schema_version": "anchored-transition-innovation-belief-v1",
            "aggregate_prior": baseline_report["final_model"],
            "transition_innovation": final_transition,
            "likelihood_power": final_power,
        },
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
                "outer_powers": report["outer_powers"],
                "final_power": report["final_power"],
                "paired_improvements": report["paired_improvements"],
                "maximum_low_evidence_posterior": report[
                    "maximum_low_evidence_posterior"
                ],
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
