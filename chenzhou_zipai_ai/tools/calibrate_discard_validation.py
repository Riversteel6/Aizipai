"""Compare online discard validation against pooled counterfactual evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import _familywise_95_t_critical
from ai.opponent_context import (
    OPPONENT_CONTEXT_FEATURE_NAMES,
    opponent_context_features,
)
from ai.simulation_trace import public_state_identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-input",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audits_by_state = _load_audits(args.evidence_input)
    benchmark = json.loads(
        args.benchmark_report.read_text(encoding="utf-8")
    )
    benchmark_by_view = _benchmark_index(benchmark)
    rows = [
        build_state_calibration(
            audits,
            benchmark_by_view=benchmark_by_view,
        )
        for _, audits in sorted(audits_by_state.items())
    ]
    report = build_report(
        rows,
        evidence_inputs=args.evidence_input,
        benchmark_report=args.benchmark_report,
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
                    "states",
                    "complete_online_validations",
                    "incomplete_online_validations",
                    "preferred_confidently_suboptimal",
                    "correct_overrides",
                    "beneficial_overrides",
                    "unproven_overrides",
                    "harmful_overrides",
                    "missed_overrides",
                    "mean_improvement_over_preferred",
                    "mean_regret_to_pooled_best",
                    "confidently_negative_overrides",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def _benchmark_index(
    report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _benchmark_rows(report):
        public_view_id = str(row.get("public_view_id") or "")
        if not public_view_id:
            raise ValueError("discard_calibration_benchmark_view_missing")
        if public_view_id in indexed:
            raise ValueError(
                "discard_calibration_benchmark_view_duplicate:"
                f"{public_view_id}"
            )
        indexed[public_view_id] = row
    return indexed


def _benchmark_rows(
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = list(report.get("rows") or ())
    if not any("candidate_discard_events" in row for row in rows):
        return [dict(row) for row in rows]

    flattened: list[dict[str, Any]] = []
    for game in rows:
        for event in game.get("candidate_discard_events") or ():
            if not event.get("public_view_id"):
                continue
            flattened.append(
                {
                    "public_view_id": str(event["public_view_id"]),
                    "selected_label": str(
                        event.get("search_selected_label")
                        or event.get("production_label")
                        or ""
                    ),
                    "baseline_selected_label": event.get(
                        "baseline_selected_label"
                    ),
                    "validation_preferred_label": event.get(
                        "validation_preferred_label"
                    ),
                    "validation_challenger_label": event.get(
                        "validation_challenger_label"
                    ),
                    "validation_challenger_labels": list(
                        event.get("validation_challenger_labels") or ()
                    ),
                    "validation_selected_label": event.get(
                        "validation_selected_label"
                    ),
                    "validation_complete": bool(
                        event.get("validation_complete")
                    ),
                    "validation_error": event.get("validation_error"),
                    "validation_coverage_ranking": list(
                        event.get("candidates") or ()
                    ),
                    "validation_diagnostics": event.get(
                        "validation_diagnostics"
                    ),
                }
            )
    return flattened


def build_state_calibration(
    audits: Sequence[Mapping[str, Any]],
    *,
    benchmark_by_view: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not audits:
        raise ValueError("discard_calibration_requires_audits")
    state_hashes = {
        str(audit.get("state_before_hash") or "")
        for audit in audits
    }
    if len(state_hashes) != 1 or not next(iter(state_hashes)):
        raise ValueError("discard_calibration_state_hash_mismatch")
    if any(not audit.get("complete") for audit in audits):
        raise ValueError("discard_calibration_incomplete_audit")

    audited_selected_labels = {
        _action_label(str(audit.get("selected_key") or ""))
        for audit in audits
    }
    if len(audited_selected_labels) != 1:
        raise ValueError("discard_calibration_preferred_mismatch")
    audited_selected = next(iter(audited_selected_labels))
    public_ids = {
        public_state_identity(audit["public_state"])
        for audit in audits
    }
    if len(public_ids) != 1:
        raise ValueError("discard_calibration_public_state_mismatch")
    public_view_id = next(iter(public_ids))
    context_features = opponent_context_features(audits[0]["public_state"])
    benchmark = benchmark_by_view.get(public_view_id)
    if benchmark is None:
        raise ValueError(
            f"discard_calibration_benchmark_missing:{public_view_id}"
        )
    preferred = str(
        benchmark.get("baseline_selected_label")
        or audited_selected
    )
    validation_preferred = benchmark.get("validation_preferred_label")
    if (
        validation_preferred is not None
        and str(validation_preferred) != preferred
    ):
        raise ValueError(
            "discard_calibration_validation_preferred_mismatch:"
            f"{public_view_id}:{preferred}:{validation_preferred}"
        )

    seed_worlds = [_world_rewards(audit) for audit in audits]
    candidate_sets = [
        {label for world in worlds for label in world}
        for worlds in seed_worlds
    ]
    if not candidate_sets or any(
        labels != candidate_sets[0] for labels in candidate_sets[1:]
    ):
        raise ValueError("discard_calibration_candidate_set_mismatch")
    labels = sorted(candidate_sets[0])
    if preferred not in labels:
        raise ValueError("discard_calibration_preferred_missing")
    comparisons = max(1, len(labels) - 1)

    pooled_rewards = {
        label: [
            world[label]
            for worlds in seed_worlds
            for world in worlds
        ]
        for label in labels
    }
    pooled_best = max(
        labels,
        key=lambda label: (
            mean(pooled_rewards[label]),
            label,
        ),
    )
    candidates = [
        {
            "label": label,
            "mean_reward": round(mean(pooled_rewards[label]), 6),
            "seed_mean_deltas": [
                round(
                    mean(
                        world[label] - world[preferred]
                        for world in worlds
                    ),
                    6,
                )
                for worlds in seed_worlds
            ],
            "positive_seed_batches": sum(
                mean(
                    world[label] - world[preferred]
                    for world in worlds
                )
                > 0.0
                for worlds in seed_worlds
            ),
            "vs_preferred": summarize_deltas(
                [
                    candidate_reward - preferred_reward
                    for candidate_reward, preferred_reward in zip(
                        pooled_rewards[label],
                        pooled_rewards[preferred],
                    )
                ],
                comparisons=comparisons,
            ),
        }
        for label in labels
    ]
    candidate_by_label = {
        candidate["label"]: candidate
        for candidate in candidates
    }

    online_selected = str(benchmark.get("selected_label") or "")
    if online_selected not in candidate_by_label:
        raise ValueError(
            f"discard_calibration_online_candidate_missing:{online_selected}"
        )
    best_vs_preferred = candidate_by_label[pooled_best]["vs_preferred"]
    selected_vs_preferred = candidate_by_label[online_selected][
        "vs_preferred"
    ]
    selected_vs_best = summarize_deltas(
        [
            selected_reward - best_reward
            for selected_reward, best_reward in zip(
                pooled_rewards[online_selected],
                pooled_rewards[pooled_best],
            )
        ],
        comparisons=comparisons,
    )
    classification = classify_online_action(
        preferred=preferred,
        online_selected=online_selected,
        pooled_best=pooled_best,
        preferred_confidently_suboptimal=bool(
            best_vs_preferred["confidently_positive"]
        ),
        selected_vs_preferred=selected_vs_preferred,
    )
    online_challenger_evidence = [
        _online_challenger_evidence(
            challenger,
            expected_worlds=int(
                (benchmark.get("validation_diagnostics") or {}).get(
                    "expected_confirmation_worlds"
                )
                or 0
            ),
            coverage_ranking=(
                benchmark.get("validation_coverage_ranking") or ()
            ),
            preferred_label=preferred,
        )
        for challenger in (
            (benchmark.get("validation_diagnostics") or {}).get(
                "challengers"
            )
            or ()
        )
    ]
    return {
        "state_before_hash": next(iter(state_hashes)),
        "public_view_id": public_view_id,
        "validation_group_id": str(
            benchmark.get("validation_group_id")
            or next(iter(state_hashes))
        ),
        "development_group_id": benchmark.get(
            "development_group_id"
        ),
        "cohort": benchmark.get("cohort"),
        "opponent_stratum": benchmark.get("opponent_stratum"),
        "opponent_context_feature_names": list(
            OPPONENT_CONTEXT_FEATURE_NAMES
        ),
        "opponent_context_features": list(context_features),
        "game": dict(audits[0].get("game") or {}),
        "trace_sequence": int(audits[0].get("trace_sequence") or 0),
        "audit_seeds": [
            int(audit.get("audit_seed") or 0)
            for audit in audits
        ],
        "pooled_worlds": len(pooled_rewards[preferred]),
        "candidate_count": len(labels),
        "audited_selected_label": audited_selected,
        "baseline_changed_from_audited": preferred != audited_selected,
        "preferred_label": preferred,
        "pooled_best_label": pooled_best,
        "preferred_confidently_suboptimal": bool(
            best_vs_preferred["confidently_positive"]
        ),
        "pooled_best_vs_preferred": best_vs_preferred,
        "online_selected_label": online_selected,
        "online_validation_complete": bool(
            benchmark.get("validation_complete")
        ),
        "online_validation_error": benchmark.get("validation_error"),
        "online_challenger_labels": list(
            benchmark.get("validation_challenger_labels") or ()
        ),
        "online_challenger_evidence": online_challenger_evidence,
        "online_selected_vs_preferred": selected_vs_preferred,
        "online_selected_vs_pooled_best": selected_vs_best,
        "classification": classification,
        "candidates": sorted(
            candidates,
            key=lambda candidate: (
                candidate["mean_reward"],
                candidate["label"],
            ),
            reverse=True,
        ),
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    evidence_inputs: Sequence[Path],
    benchmark_report: Path,
) -> dict[str, Any]:
    classifications = [
        str(row["classification"])
        for row in rows
    ]
    complete = sum(
        bool(row["online_validation_complete"])
        for row in rows
    )
    performance = summarize_online_performance(rows)
    return {
        "ok": bool(rows)
        and all(
            int(row["pooled_worlds"]) > 0
            for row in rows
        ),
        "schema_version": "discard-validator-calibration-v1",
        "evidence_inputs": [str(path) for path in evidence_inputs],
        "benchmark_report": str(benchmark_report),
        "states": len(rows),
        "complete_online_validations": complete,
        "incomplete_online_validations": len(rows) - complete,
        "preferred_confidently_suboptimal": sum(
            bool(row["preferred_confidently_suboptimal"])
            for row in rows
        ),
        "correct_overrides": classifications.count("correct_override"),
        "beneficial_overrides": classifications.count(
            "beneficial_override"
        ),
        "unproven_overrides": classifications.count(
            "unproven_override"
        ),
        "harmful_overrides": classifications.count("harmful_override"),
        "missed_overrides": classifications.count("missed_override"),
        "correct_keeps": classifications.count("correct_keep"),
        **performance,
        "threshold_sweep": [
            simulate_repeatable_threshold(rows, threshold=threshold)
            for threshold in (
                0.0,
                0.025,
                0.05,
                0.075,
                0.1,
                0.125,
                0.15,
                0.2,
            )
        ],
        "rows": list(rows),
    }


def summarize_online_performance(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    improvements = [
        float(row["online_selected_vs_preferred"]["mean_delta"])
        for row in rows
    ]
    regrets = [
        -float(row["online_selected_vs_pooled_best"]["mean_delta"])
        for row in rows
    ]
    overrides = [
        row
        for row in rows
        if str(row["online_selected_label"])
        != str(row["preferred_label"])
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        opponents = tuple(
            str(value)
            for value in (row.get("game") or {}).get("opponents") or ()
        )
        grouped[",".join(opponents) or "unknown"].append(row)
    return {
        "overrides": len(overrides),
        "mean_improvement_over_preferred": round(
            mean(improvements) if improvements else 0.0,
            6,
        ),
        "state_level_improvement": summarize_deltas(
            improvements,
            comparisons=1,
        ),
        "mean_regret_to_pooled_best": round(
            mean(regrets) if regrets else 0.0,
            6,
        ),
        "confidently_positive_overrides": sum(
            (
                row["online_selected_vs_preferred"][
                    "lower_confidence_bound"
                ]
                is not None
                and float(
                    row["online_selected_vs_preferred"][
                        "lower_confidence_bound"
                    ]
                )
                > 0.0
            )
            for row in overrides
        ),
        "confidently_negative_overrides": sum(
            (
                row["online_selected_vs_preferred"][
                    "upper_confidence_bound"
                ]
                is not None
                and float(
                    row["online_selected_vs_preferred"][
                        "upper_confidence_bound"
                    ]
                )
                < 0.0
            )
            for row in overrides
        ),
        "opponent_breakdown": {
            opponent: _summarize_online_group(group_rows)
            for opponent, group_rows in sorted(grouped.items())
        },
    }


def _summarize_online_group(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    improvements = [
        float(row["online_selected_vs_preferred"]["mean_delta"])
        for row in rows
    ]
    regrets = [
        -float(row["online_selected_vs_pooled_best"]["mean_delta"])
        for row in rows
    ]
    return {
        "states": len(rows),
        "overrides": sum(
            str(row["online_selected_label"])
            != str(row["preferred_label"])
            for row in rows
        ),
        "mean_improvement_over_preferred": round(
            mean(improvements) if improvements else 0.0,
            6,
        ),
        "state_level_improvement": summarize_deltas(
            improvements,
            comparisons=1,
        ),
        "mean_regret_to_pooled_best": round(
            mean(regrets) if regrets else 0.0,
            6,
        ),
        "preferred_confidently_suboptimal": sum(
            bool(row["preferred_confidently_suboptimal"])
            for row in rows
        ),
        "missed_overrides": sum(
            str(row["classification"]) == "missed_override"
            for row in rows
        ),
    }


def classify_online_action(
    *,
    preferred: str,
    online_selected: str,
    pooled_best: str,
    preferred_confidently_suboptimal: bool,
    selected_vs_preferred: Mapping[str, Any],
) -> str:
    if online_selected == preferred:
        return (
            "missed_override"
            if preferred_confidently_suboptimal
            else "correct_keep"
        )
    if bool(selected_vs_preferred["confidently_positive"]):
        return (
            "correct_override"
            if online_selected == pooled_best
            else "beneficial_override"
        )
    if float(selected_vs_preferred["mean_delta"]) > 0.0:
        return "unproven_override"
    return "harmful_override"


def simulate_repeatable_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    classifications: list[str] = []
    selected_labels: list[str] = []
    mean_improvements: list[float] = []
    mean_regrets: list[float] = []
    exact_best = 0
    for row in rows:
        preferred = str(row["preferred_label"])
        evidence = [
            candidate
            for candidate in row.get("online_challenger_evidence") or ()
            if bool(candidate.get("familywise_confident"))
            or (
                bool(candidate.get("complete"))
                and float(candidate.get("first_mean_delta") or 0.0) > 0.0
                and float(candidate.get("second_mean_delta") or 0.0) > 0.0
                and float(candidate.get("combined_mean_delta") or 0.0)
                >= float(threshold)
            )
        ]
        selected = (
            max(
                evidence,
                key=lambda candidate: (
                    float(
                        candidate["combined_lower_confidence_bound"]
                        if candidate.get(
                            "combined_lower_confidence_bound"
                        )
                        is not None
                        else -math.inf
                    ),
                    float(
                        candidate["combined_mean_delta"]
                        if candidate.get("combined_mean_delta")
                        is not None
                        else -math.inf
                    ),
                    str(candidate.get("challenger_label") or ""),
                ),
            )["challenger_label"]
            if evidence
            else preferred
        )
        candidate_by_label = {
            str(candidate["label"]): candidate
            for candidate in row["candidates"]
        }
        selected_stats = candidate_by_label[str(selected)]
        classification = classify_online_action(
            preferred=preferred,
            online_selected=str(selected),
            pooled_best=str(row["pooled_best_label"]),
            preferred_confidently_suboptimal=bool(
                row["preferred_confidently_suboptimal"]
            ),
            selected_vs_preferred=selected_stats["vs_preferred"],
        )
        selected_labels.append(str(selected))
        classifications.append(classification)
        mean_improvements.append(
            float(selected_stats["vs_preferred"]["mean_delta"])
        )
        mean_regrets.append(
            float(row["pooled_best_vs_preferred"]["mean_delta"])
            - float(selected_stats["vs_preferred"]["mean_delta"])
        )
        exact_best += int(str(selected) == str(row["pooled_best_label"]))
    return {
        "threshold": round(float(threshold), 6),
        "overrides": sum(
            selected != str(row["preferred_label"])
            for selected, row in zip(selected_labels, rows)
        ),
        "exact_pooled_best": exact_best,
        "correct_overrides": classifications.count("correct_override"),
        "beneficial_overrides": classifications.count(
            "beneficial_override"
        ),
        "unproven_overrides": classifications.count(
            "unproven_override"
        ),
        "harmful_overrides": classifications.count("harmful_override"),
        "missed_overrides": classifications.count("missed_override"),
        "mean_improvement_over_preferred": round(
            mean(mean_improvements) if mean_improvements else 0.0,
            6,
        ),
        "mean_regret_to_pooled_best": round(
            mean(mean_regrets) if mean_regrets else 0.0,
            6,
        ),
    }


def summarize_deltas(
    deltas: Sequence[float],
    *,
    comparisons: int,
) -> dict[str, Any]:
    samples = len(deltas)
    average = mean(deltas) if deltas else 0.0
    if samples >= 2:
        squared_error = sum(
            (delta - average) ** 2
            for delta in deltas
        )
        sample_stddev = math.sqrt(squared_error / (samples - 1))
        standard_error = sample_stddev / math.sqrt(samples)
        radius = _familywise_95_t_critical(
            samples - 1,
            comparisons=max(1, comparisons),
        ) * standard_error
        lower_bound: float | None = average - radius
        upper_bound: float | None = average + radius
    else:
        sample_stddev = 0.0
        standard_error = 0.0
        lower_bound = None
        upper_bound = None
    return {
        "samples": samples,
        "mean_delta": round(average, 6),
        "sample_stddev": round(sample_stddev, 6),
        "standard_error": round(standard_error, 6),
        "lower_confidence_bound": (
            round(lower_bound, 6)
            if lower_bound is not None
            else None
        ),
        "upper_confidence_bound": (
            round(upper_bound, 6)
            if upper_bound is not None
            else None
        ),
        "positive_samples": sum(delta > 0.0 for delta in deltas),
        "tied_samples": sum(abs(delta) <= 1e-12 for delta in deltas),
        "negative_samples": sum(delta < 0.0 for delta in deltas),
        "confidently_positive": bool(
            lower_bound is not None and lower_bound > 0.0
        ),
        "confidently_negative": bool(
            upper_bound is not None and upper_bound < 0.0
        ),
    }


def _online_challenger_evidence(
    challenger: Mapping[str, Any],
    *,
    expected_worlds: int,
    coverage_ranking: Sequence[Mapping[str, Any]],
    preferred_label: str,
) -> dict[str, Any]:
    first = challenger.get("first_confirmation") or {}
    second = challenger.get("second_confirmation") or {}
    first_advantage = next(
        iter(challenger.get("first_paired_advantages") or ()),
        {},
    )
    second_advantage = next(
        iter(challenger.get("second_paired_advantages") or ()),
        {},
    )
    combined_advantage = next(
        iter(challenger.get("combined_paired_advantages") or ()),
        {},
    )
    challenger_label = str(challenger.get("challenger_label") or "")
    coverage_by_label = {
        str(candidate.get("label") or ""): candidate
        for candidate in coverage_ranking
    }
    coverage_candidate = coverage_by_label.get(challenger_label, {})
    coverage_preferred = coverage_by_label.get(preferred_label, {})
    first_worlds = int(first.get("paired_determinizations") or 0)
    second_worlds = int(second.get("paired_determinizations") or 0)
    confirmation_fraction = (
        min(first_worlds, second_worlds) / expected_worlds
        if expected_worlds > 0
        else 0.0
    )
    structurally_valid = bool(
        expected_worlds > 0
        and first_worlds >= 2
        and second_worlds >= 2
        and bool(first.get("used_search"))
        and bool(second.get("used_search"))
        and int(first.get("rollout_invariant_violations") or 0) == 0
        and int(second.get("rollout_invariant_violations") or 0) == 0
        and int(first.get("rollout_coverage_failures") or 0) == 0
        and int(second.get("rollout_coverage_failures") or 0) == 0
        and int(first.get("candidate_count") or 0) == 2
        and int(second.get("candidate_count") or 0) == 2
        and int(first.get("zero_visit_candidates") or 0) == 0
        and int(second.get("zero_visit_candidates") or 0) == 0
    )
    heuristic_ranking = sorted(
        coverage_ranking,
        key=lambda candidate: (
            float(candidate.get("heuristic_value") or 0.0),
            float(candidate.get("average_reward") or 0.0),
            str(candidate.get("label") or ""),
        ),
        reverse=True,
    )
    return {
        "challenger_label": challenger_label,
        "complete": bool(
            expected_worlds > 0
            and first_worlds == expected_worlds
            and second_worlds == expected_worlds
            and int(first.get("deadline_interruptions") or 0) == 0
            and int(second.get("deadline_interruptions") or 0) == 0
        ),
        "structurally_valid": structurally_valid,
        "expected_confirmation_worlds": expected_worlds,
        "first_confirmation_worlds": first_worlds,
        "second_confirmation_worlds": second_worlds,
        "confirmation_fraction": round(confirmation_fraction, 6),
        "familywise_confident": (
            challenger.get("override_basis") == "familywise_confident"
        ),
        "override_basis": challenger.get("override_basis"),
        "first_mean_delta": float(
            first_advantage.get("mean_delta") or 0.0
        ),
        "second_mean_delta": float(
            second_advantage.get("mean_delta") or 0.0
        ),
        "combined_mean_delta": float(
            combined_advantage.get("mean_delta") or 0.0
        ),
        "combined_lower_confidence_bound": float(
            combined_advantage.get("lower_confidence_bound") or 0.0
        ),
        "combined_standard_error": float(
            combined_advantage.get("standard_error") or 0.0
        ),
        "combined_sample_stddev": float(
            combined_advantage.get("sample_stddev") or 0.0
        ),
        "coverage_rank": next(
            (
                index
                for index, candidate in enumerate(
                    coverage_ranking,
                    start=1,
                )
                if str(candidate.get("label") or "")
                == challenger_label
            ),
            0,
        ),
        "heuristic_rank": next(
            (
                index
                for index, candidate in enumerate(
                    heuristic_ranking,
                    start=1,
                )
                if str(candidate.get("label") or "")
                == challenger_label
            ),
            0,
        ),
        "coverage_reward_delta": (
            float(coverage_candidate.get("average_reward") or 0.0)
            - float(coverage_preferred.get("average_reward") or 0.0)
        ),
        "heuristic_value_delta": (
            float(coverage_candidate.get("heuristic_value") or 0.0)
            - float(coverage_preferred.get("heuristic_value") or 0.0)
        ),
    }


def _load_audits(
    paths: Sequence[Path],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        audits = list(_read_jsonl(path))
        state_hashes = {
            str(audit.get("state_before_hash") or "")
            for audit in audits
        }
        if not state_hashes or "" in state_hashes:
            raise ValueError(f"discard_calibration_state_missing:{path}")
        if len(state_hashes) != len(audits):
            raise ValueError(f"discard_calibration_duplicate_state:{path}")
        for audit in audits:
            grouped[str(audit["state_before_hash"])].append(audit)
    replicate_counts = {
        len(audits)
        for audits in grouped.values()
    }
    if len(replicate_counts) != 1:
        raise ValueError(
            "discard_calibration_evidence_replicate_count_mismatch"
        )
    for state_hash, audits in grouped.items():
        audit_seeds = [
            int(audit.get("audit_seed") or 0)
            for audit in audits
        ]
        if len(audit_seeds) != len(set(audit_seeds)):
            raise ValueError(
                "discard_calibration_duplicate_audit_seed:"
                f"{state_hash}"
            )
    return dict(grouped)


def _world_rewards(
    audit: Mapping[str, Any],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for world in audit.get("paired_worlds") or ():
        rewards = {
            str(outcome["candidate_key"]): float(outcome["reward"])
            for outcome in world.get("outcomes") or ()
        }
        if rewards:
            rows.append(rewards)
    if not rows:
        raise ValueError("discard_calibration_worlds_missing")
    labels = set(rows[0])
    if any(set(row) != labels for row in rows[1:]):
        raise ValueError("discard_calibration_world_candidate_mismatch")
    return rows


def _action_label(key: str) -> str:
    return key.rsplit(":", 1)[-1]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
