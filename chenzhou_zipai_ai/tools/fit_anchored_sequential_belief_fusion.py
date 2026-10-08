"""Nested grouped validation for anchored sequential opponent-belief fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.evaluate_anchored_structural_discard_proxies import sha256_file
from tools.fit_public_opponent_belief import (
    BeliefCase,
    fit_weighted_ridge_model,
    mean_confidence_interval,
)
from tools.fit_sequential_public_likelihood import (
    Checkpoint,
    collect_checkpoints,
    fit_likelihood_model,
    load_events,
    selected_profile,
    update_log_scores,
)


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
        "failed_sequence_report",
    ):
        if sha256_file(Path(str(inputs[key]))) != str(inputs[f"{key}_sha256"]):
            raise ValueError(f"anchored_fusion_{key}_hash_mismatch")
    dataset_audit = _read_json(Path(str(inputs["dataset_audit"])))
    failed_sequence = _read_json(Path(str(inputs["failed_sequence_report"])))
    if not dataset_audit.get("integrity_gate_pass"):
        raise ValueError("anchored_fusion_dataset_audit_did_not_pass")
    if failed_sequence.get("focused_gate_pass"):
        raise ValueError("anchored_fusion_expected_failed_sequence_anchor")

    baseline_report = _read_json(Path(str(inputs["baseline_report"])))
    baseline_prereg = _read_json(Path(str(baseline_report["preregistration"])))
    baseline_spec = dict(baseline_prereg["model"])
    profiles = tuple(str(item) for item in preregistration["profiles"])
    for source_name, source_report in (
        ("baseline", baseline_report),
        ("failed_sequence", failed_sequence),
    ):
        source_profiles = tuple(str(item) for item in source_report["profiles"])
        if source_profiles != profiles:
            raise ValueError(f"anchored_fusion_{source_name}_profile_mismatch")
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
        raise ValueError("anchored_fusion_game_set_mismatch")
    alpha = float(failed_sequence["final_model"]["alpha"])
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
            records, diagnostics = component_records(
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
        outer_records, diagnostics = component_records(
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
    final_likelihood = fit_likelihood_model(
        events,
        profiles=profiles,
        alpha=alpha,
    )
    return {
        "ok": bool(all_outer_records) and unknown_events == 0,
        "schema_version": "anchored-sequential-belief-fusion-nested-cv-v1",
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
        "final_model": {
            "schema_version": "anchored-sequential-belief-fusion-v1",
            "aggregate_prior": baseline_report["final_model"],
            "sequence_likelihood": final_likelihood,
            "likelihood_power": final_power,
        },
        "focused_gate": gate,
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
    }


def component_records(
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
    likelihood = fit_likelihood_model(
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
        for checkpoint in game_checkpoints:
            while (
                event_index < len(game_events)
                and int(game_events[event_index]["source_sequence"])
                < checkpoint.sequence
            ):
                started = time.perf_counter()
                try:
                    sequence_scores = update_log_scores(
                        sequence_scores,
                        game_events[event_index],
                        model=likelihood,
                        profiles=profiles,
                    )
                except KeyError:
                    unknown_events += 1
                elapsed_samples.append((time.perf_counter() - started) * 1000.0)
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


def fused_posterior(
    baseline_posterior: Mapping[str, float],
    sequence_scores: Mapping[str, float],
    *,
    power: float,
    profiles: Sequence[str],
) -> dict[str, float]:
    values = {
        profile: math.log(max(1e-15, float(baseline_posterior[profile])))
        + float(power) * float(sequence_scores[profile])
        for profile in profiles
    }
    maximum = max(values.values())
    exponentials = {
        profile: math.exp(values[profile] - maximum) for profile in profiles
    }
    total = sum(exponentials.values())
    return {profile: exponentials[profile] / total for profile in profiles}


def choose_power(
    records: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    grid: Sequence[float],
    maximum_low_evidence_posterior: float,
) -> tuple[float, dict[str, Any]]:
    candidates = []
    for power in grid:
        per_game: dict[str, list[float]] = defaultdict(list)
        low_evidence_maximum = 0.0
        for record in records:
            posterior = fused_posterior(
                record["baseline_posterior"],
                record["sequence_scores"],
                power=power,
                profiles=profiles,
            )
            per_game[str(record["game_id"])].append(
                -math.log(max(1e-15, posterior[str(record["profile"])]))
            )
            if int(record["evidence_count"]) <= 2:
                low_evidence_maximum = max(
                    low_evidence_maximum, max(posterior.values())
                )
        if low_evidence_maximum > maximum_low_evidence_posterior + 1e-12:
            continue
        objective = fmean(
            fmean(losses) for losses in per_game.values()
        )
        candidates.append((objective, float(power), low_evidence_maximum))
    if not candidates:
        raise ValueError("anchored_fusion_no_feasible_power")
    objective, power, low_evidence = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return power, {
        "feasible_powers": len(candidates),
        "selected_objective": objective,
        "selected_low_evidence_maximum": low_evidence,
        "minimum_feasible_power": min(item[1] for item in candidates),
        "maximum_feasible_power": max(item[1] for item in candidates),
    }


def evaluate_fusion_records(
    records: Sequence[Mapping[str, Any]],
    *,
    power: float,
    profiles: Sequence[str],
    elapsed_samples: list[float],
) -> list[dict[str, Any]]:
    evaluated = []
    for record in records:
        started = time.perf_counter()
        fusion = fused_posterior(
            record["baseline_posterior"],
            record["sequence_scores"],
            power=power,
            profiles=profiles,
        )
        elapsed_samples.append((time.perf_counter() - started) * 1000.0)
        true_profile = str(record["profile"])
        baseline = record["baseline_posterior"]
        evaluated.append(
            {
                **record,
                "baseline_log_loss": -math.log(
                    max(1e-15, float(baseline[true_profile]))
                ),
                "baseline_true_probability": float(baseline[true_profile]),
                "baseline_top1": int(
                    selected_profile(baseline, profiles=profiles) == true_profile
                ),
                "fusion_log_loss": -math.log(
                    max(1e-15, float(fusion[true_profile]))
                ),
                "fusion_true_probability": float(fusion[true_profile]),
                "fusion_top1": int(
                    selected_profile(fusion, profiles=profiles) == true_profile
                ),
                "fusion_max_probability": max(fusion.values()),
            }
        )
    return evaluated


def summarize_records_by_game(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["game_id"])].append(record)
    return [
        {
            "game_id": game_id,
            "profile": rows[0]["profile"],
            "checkpoints": len(rows),
            **{
                key: fmean(float(row[key]) for row in rows)
                for key in (
                    "baseline_log_loss",
                    "baseline_true_probability",
                    "baseline_top1",
                    "fusion_log_loss",
                    "fusion_true_probability",
                    "fusion_top1",
                )
            },
        }
        for game_id, rows in sorted(grouped.items())
    ]


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
                - float(row["fusion_log_loss"])
                for row in per_game
            ]
        ),
        "true_probability": mean_confidence_interval(
            [
                float(row["fusion_true_probability"])
                - float(row["baseline_true_probability"])
                for row in per_game
            ]
        ),
        "top1": mean_confidence_interval(
            [
                float(row["fusion_top1"])
                - float(row["baseline_top1"])
                for row in per_game
            ]
        ),
    }


def summarize_by_profile(
    per_game: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
) -> dict[str, Any]:
    return {
        profile: {
            "games": len(rows),
            "baseline": summarize_model(rows, prefix="baseline"),
            "fusion": summarize_model(rows, prefix="fusion"),
            "improvements": paired_improvements(rows),
        }
        for profile in profiles
        for rows in [[row for row in per_game if row["profile"] == profile]]
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


def focused_gate_failures(
    *,
    profiles: Sequence[str],
    games: int,
    checkpoints: int,
    outer_reports: Sequence[Mapping[str, Any]],
    group_leakage: int,
    unknown_events: int,
    reproduction_error: float,
    improvements: Mapping[str, Mapping[str, Any]],
    by_profile: Mapping[str, Any],
    maximum_low_evidence_posterior: float,
    p95_ms: float,
    outer_powers: Sequence[float],
    final_power: float,
    gate: Mapping[str, Any],
) -> list[str]:
    failures = []
    if len(profiles) != int(gate["required_profiles"]):
        failures.append("required_profiles")
    if games != int(gate["required_games"]):
        failures.append("required_games")
    if checkpoints != int(gate["required_checkpoints"]):
        failures.append("required_checkpoints")
    if len(outer_reports) != int(gate["required_outer_folds"]):
        failures.append("required_outer_folds")
    if any(
        len(report["inner_folds"])
        != int(gate["required_inner_folds_per_outer"])
        for report in outer_reports
    ):
        failures.append("required_inner_folds_per_outer")
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
    for profile, summary in by_profile.items():
        for name in ("log_loss", "true_probability", "top1"):
            if float(summary["improvements"][name]["mean"]) < float(
                gate[f"minimum_profile_mean_{name}_improvement"]
            ):
                failures.append(f"profile_{name}_regression:{profile}")
    if maximum_low_evidence_posterior > float(
        gate["maximum_low_evidence_posterior"]
    ):
        failures.append("maximum_low_evidence_posterior")
    if p95_ms > float(gate["maximum_update_and_fusion_p95_ms"]):
        failures.append("update_and_fusion_p95_ms")
    if any(
        power < float(gate["minimum_outer_fold_power"])
        for power in outer_powers
    ):
        failures.append("minimum_outer_fold_power")
    if final_power < float(gate["minimum_final_power"]):
        failures.append("minimum_final_power")
    return list(dict.fromkeys(failures))


def resolved_power_grid(spec: Mapping[str, Any]) -> tuple[float, ...]:
    minimum = float(spec["minimum"])
    maximum = float(spec["maximum"])
    step = float(spec["step"])
    count = int(round((maximum - minimum) / step))
    return tuple(round(minimum + index * step, 10) for index in range(count + 1))


def nested_calibration_limit(preregistration: Mapping[str, Any]) -> float:
    return float(preregistration["focused_gate"]["maximum_low_evidence_posterior"])


def group_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["game_id"])].append(event)
    return grouped


def group_checkpoints(
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
