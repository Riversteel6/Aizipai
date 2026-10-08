"""Evaluate a frozen discard calibrator on held-out evidence."""

from __future__ import annotations

import argparse
import json
import math
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
from tools.fit_discard_evidence_calibrator import (
    FEATURE_NAMES,
    evidence_features,
    predict,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-report",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--decision-margin", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        row
        for path in args.calibration_report
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]
    ]
    state_ids = [str(row["state_before_hash"]) for row in rows]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("discard_calibrator_evaluation_duplicate_state")
    model_report = json.loads(
        args.model_report.read_text(encoding="utf-8")
    )
    model = model_report["model"]
    feature_names = tuple(model_report.get("feature_names") or ())
    if feature_names != FEATURE_NAMES:
        raise ValueError("discard_calibrator_feature_schema_mismatch")
    margin = (
        float(args.decision_margin)
        if args.decision_margin is not None
        else float(model.get("decision_margin") or 0.0)
    )
    report = evaluate_model(
        rows,
        model=model,
        decision_margin=margin,
        model_report=args.model_report,
        calibration_reports=args.calibration_report,
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
                    "mean_improvement_over_preferred",
                    "mean_regret_to_pooled_best",
                    "online_selection_matches",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def evaluate_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Mapping[str, Any],
    decision_margin: float,
    model_report: Path | str = "",
    calibration_reports: Sequence[Path | str] = (),
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    for row in rows:
        preferred = str(row["preferred_label"])
        truth_by_label = {
            str(candidate["label"]): candidate
            for candidate in row["candidates"]
        }
        predictions = []
        for evidence in row.get("online_challenger_evidence") or ():
            label = str(evidence.get("challenger_label") or "")
            if (
                not evidence.get("complete")
                or label not in truth_by_label
            ):
                continue
            predictions.append(
                {
                    "label": label,
                    "prediction": predict(
                        model,
                        evidence_features(
                            evidence,
                            opponent_context=row.get(
                                "opponent_context_features"
                            ),
                        ),
                    ),
                }
            )
        selected_prediction = -math.inf
        selected = preferred
        if predictions:
            best = max(
                predictions,
                key=lambda item: (
                    float(item["prediction"]),
                    str(item["label"]),
                ),
            )
            selected_prediction = float(best["prediction"])
            if selected_prediction > float(decision_margin):
                selected = str(best["label"])

        selected_stats = truth_by_label[selected]
        selected_vs_preferred = selected_stats["vs_preferred"]
        improvement = float(selected_vs_preferred["mean_delta"])
        regret = (
            float(row["pooled_best_vs_preferred"]["mean_delta"])
            - improvement
        )
        classification = classify_online_action(
            preferred=preferred,
            online_selected=selected,
            pooled_best=str(row["pooled_best_label"]),
            preferred_confidently_suboptimal=bool(
                row["preferred_confidently_suboptimal"]
            ),
            selected_vs_preferred=selected_vs_preferred,
        )
        evaluations.append(
            {
                "state_before_hash": str(row["state_before_hash"]),
                "public_view_id": str(row["public_view_id"]),
                "preferred_label": preferred,
                "selected_label": selected,
                "online_selected_label": str(
                    row.get("online_selected_label") or ""
                ),
                "selected_prediction": (
                    round(selected_prediction, 12)
                    if math.isfinite(selected_prediction)
                    else None
                ),
                "decision_margin": float(decision_margin),
                "classification": classification,
                "improvement_over_preferred": improvement,
                "regret_to_pooled_best": regret,
                "selected_vs_preferred": selected_vs_preferred,
                "candidate_predictions": sorted(
                    predictions,
                    key=lambda item: (
                        float(item["prediction"]),
                        str(item["label"]),
                    ),
                    reverse=True,
                ),
            }
        )

    classifications = Counter(
        str(row["classification"])
        for row in evaluations
    )
    improvements = [
        float(row["improvement_over_preferred"])
        for row in evaluations
    ]
    regrets = [
        float(row["regret_to_pooled_best"])
        for row in evaluations
    ]
    overrides = [
        row
        for row in evaluations
        if str(row["selected_label"]) != str(row["preferred_label"])
    ]
    matches = sum(
        str(row["selected_label"])
        == str(row["online_selected_label"])
        for row in evaluations
    )
    return {
        "ok": bool(evaluations)
        and all(
            row["selected_label"]
            for row in evaluations
        ),
        "schema_version": "discard-calibrator-heldout-evaluation-v1",
        "model_report": str(model_report),
        "calibration_reports": [
            str(path)
            for path in calibration_reports
        ],
        "feature_names": list(FEATURE_NAMES),
        "model_ridge": float(model.get("ridge") or 0.0),
        "decision_margin": float(decision_margin),
        "states": len(evaluations),
        "overrides": len(overrides),
        "correct_overrides": classifications["correct_override"],
        "beneficial_overrides": classifications["beneficial_override"],
        "unproven_overrides": classifications["unproven_override"],
        "harmful_overrides": classifications["harmful_override"],
        "missed_overrides": classifications["missed_override"],
        "correct_keeps": classifications["correct_keep"],
        "confidently_positive_overrides": sum(
            bool(
                row["selected_vs_preferred"][
                    "confidently_positive"
                ]
            )
            for row in overrides
        ),
        "confidently_negative_overrides": sum(
            bool(
                row["selected_vs_preferred"][
                    "confidently_negative"
                ]
            )
            for row in overrides
        ),
        "empirically_negative_overrides": sum(
            float(row["improvement_over_preferred"]) < 0.0
            for row in overrides
        ),
        "exact_pooled_best": sum(
            abs(float(row["regret_to_pooled_best"])) <= 1e-12
            for row in evaluations
        ),
        "mean_improvement_over_preferred": round(
            mean(improvements) if improvements else 0.0,
            8,
        ),
        "state_level_improvement": summarize_deltas(
            improvements,
            comparisons=1,
        ),
        "mean_regret_to_pooled_best": round(
            mean(regrets) if regrets else 0.0,
            8,
        ),
        "online_selection_matches": matches,
        "online_selection_mismatches": len(evaluations) - matches,
        "rows": evaluations,
    }


if __name__ == "__main__":
    raise SystemExit(main())
