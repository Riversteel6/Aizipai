"""Compare replayed actions on one fixed counterfactual truth frame."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.calibrate_discard_validation import (
    classify_online_action,
    summarize_deltas,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument(
        "--comparison-population-states",
        type=int,
        default=0,
        help=(
            "Total comparison population when the truth report contains "
            "only states whose actions differ; omitted states count as "
            "zero gain."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    truth = json.loads(args.truth_report.read_text(encoding="utf-8"))
    benchmark = json.loads(
        args.benchmark_report.read_text(encoding="utf-8")
    )
    selections = {
        str(row["public_view_id"]): str(row["selected_label"])
        for row in benchmark.get("rows") or ()
    }
    report = compare_actions(
        truth.get("rows") or (),
        selections=selections,
        truth_report=args.truth_report,
        benchmark_report=args.benchmark_report,
        comparison_population_states=args.comparison_population_states,
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
                    "overrides",
                    "harmful_overrides",
                    "missed_overrides",
                    "mean_improvement_over_fixed_preferred",
                    "mean_gain_over_recorded_action",
                    "population_mean_gain_over_recorded_action",
                    "improved_actions",
                    "regressed_actions",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def compare_actions(
    rows: Sequence[Mapping[str, Any]],
    *,
    selections: Mapping[str, str],
    truth_report: Path | str = "",
    benchmark_report: Path | str = "",
    comparison_population_states: int = 0,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        public_view_id = str(row["public_view_id"])
        selected = selections.get(public_view_id)
        if selected is None:
            missing.append(public_view_id)
            continue
        candidates = {
            str(candidate["label"]): candidate
            for candidate in row["candidates"]
        }
        if selected not in candidates:
            raise ValueError(
                f"benchmark_selected_label_missing_from_truth:{selected}"
            )
        preferred = str(row["preferred_label"])
        recorded = str(row["online_selected_label"])
        selected_stats = candidates[selected]["vs_preferred"]
        recorded_stats = candidates[recorded]["vs_preferred"]
        selected_delta = float(selected_stats["mean_delta"])
        recorded_delta = float(recorded_stats["mean_delta"])
        gain = selected_delta - recorded_delta
        classification = classify_online_action(
            preferred=preferred,
            online_selected=selected,
            pooled_best=str(row["pooled_best_label"]),
            preferred_confidently_suboptimal=bool(
                row["preferred_confidently_suboptimal"]
            ),
            selected_vs_preferred=selected_stats,
        )
        results.append(
            {
                "state_before_hash": str(row["state_before_hash"]),
                "public_view_id": public_view_id,
                "fixed_preferred_label": preferred,
                "recorded_label": recorded,
                "selected_label": selected,
                "pooled_best_label": str(row["pooled_best_label"]),
                "classification": classification,
                "improvement_over_fixed_preferred": selected_delta,
                "recorded_improvement_over_fixed_preferred": (
                    recorded_delta
                ),
                "gain_over_recorded_action": gain,
                "selected_vs_fixed_preferred": selected_stats,
            }
        )

    classifications = Counter(
        str(row["classification"])
        for row in results
    )
    improvements = [
        float(row["improvement_over_fixed_preferred"])
        for row in results
    ]
    gains = [
        float(row["gain_over_recorded_action"])
        for row in results
    ]
    population_states = (
        int(comparison_population_states)
        if comparison_population_states
        else len(results)
    )
    if population_states < len(results):
        raise ValueError(
            "comparison_population_states_smaller_than_compared_states:"
            f"{population_states}:{len(results)}"
        )
    unchanged_population_states = population_states - len(results)
    population_gains = gains + [0.0] * unchanged_population_states
    return {
        "ok": bool(results) and not missing,
        "schema_version": "benchmark-action-fixed-truth-comparison-v2",
        "truth_report": str(truth_report),
        "benchmark_report": str(benchmark_report),
        "states": len(results),
        "missing_states": len(missing),
        "missing_public_view_ids": missing,
        "overrides": sum(
            row["selected_label"] != row["fixed_preferred_label"]
            for row in results
        ),
        "correct_overrides": classifications["correct_override"],
        "beneficial_overrides": classifications["beneficial_override"],
        "unproven_overrides": classifications["unproven_override"],
        "harmful_overrides": classifications["harmful_override"],
        "missed_overrides": classifications["missed_override"],
        "correct_keeps": classifications["correct_keep"],
        "confidently_positive_overrides": sum(
            bool(
                row["selected_vs_fixed_preferred"][
                    "confidently_positive"
                ]
            )
            for row in results
            if row["selected_label"] != row["fixed_preferred_label"]
        ),
        "confidently_negative_overrides": sum(
            bool(
                row["selected_vs_fixed_preferred"][
                    "confidently_negative"
                ]
            )
            for row in results
            if row["selected_label"] != row["fixed_preferred_label"]
        ),
        "mean_improvement_over_fixed_preferred": round(
            mean(improvements) if improvements else 0.0,
            8,
        ),
        "fixed_preferred_improvement": summarize_deltas(
            improvements,
            comparisons=1,
        ),
        "mean_gain_over_recorded_action": round(
            mean(gains) if gains else 0.0,
            8,
        ),
        "gain_over_recorded_action": summarize_deltas(
            gains,
            comparisons=1,
        ),
        "comparison_population_states": population_states,
        "unchanged_population_states": unchanged_population_states,
        "population_mean_gain_over_recorded_action": round(
            mean(population_gains) if population_gains else 0.0,
            8,
        ),
        "population_gain_over_recorded_action": summarize_deltas(
            population_gains,
            comparisons=1,
        ),
        "improved_actions": sum(gain > 1e-12 for gain in gains),
        "regressed_actions": sum(gain < -1e-12 for gain in gains),
        "equal_actions": sum(abs(gain) <= 1e-12 for gain in gains),
        "population_improved_actions": sum(
            gain > 1e-12 for gain in population_gains
        ),
        "population_regressed_actions": sum(
            gain < -1e-12 for gain in population_gains
        ),
        "population_equal_actions": sum(
            abs(gain) <= 1e-12 for gain in population_gains
        ),
        "rows": results,
    }


if __name__ == "__main__":
    raise SystemExit(main())
