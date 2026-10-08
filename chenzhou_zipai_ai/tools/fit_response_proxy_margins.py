"""Fit deterministic response margins for frozen low-latency opponent proxies."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.full_game_simulator import _replace_view, evaluate_hu
from ai.ismcts import public_view_from_dict
from engine.chi_rules import ChiPlan
from engine.rules import rules_for_room
from tools.evaluate_anchored_structural_discard_proxies import (
    PROXY_TYPES,
    sha256_file,
)


@dataclass(frozen=True)
class ResponseCase:
    group_id: str
    game_id: str
    profile: str
    players: int
    wildcard_enabled: bool
    public_state: dict[str, Any]
    legal_actions: tuple[dict[str, Any], ...]
    selected_key: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_fit(
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
                "response_roots": report["response_roots"],
                "selected_margins": report["selected_margins"],
                "training_exact_key_accuracy": report["training_metrics"][
                    "exact_key_accuracy"
                ],
                "training_claim_recall": report["training_metrics"][
                    "claim_recall"
                ],
                "training_claim_precision": report["training_metrics"][
                    "claim_precision"
                ],
                "failures": report["integrity_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def run_fit(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    profiles = tuple(str(item) for item in preregistration["profiles"])
    calibrated = tuple(str(item) for item in preregistration["calibrated_profiles"])
    if set(profiles) != set(PROXY_TYPES) or not set(calibrated).issubset(PROXY_TYPES):
        raise ValueError("response_margin_profile_mapping_mismatch")
    inputs = dict(preregistration["inputs"])
    trace_path = Path(inputs["trace"])
    trace_sha256 = sha256_file(trace_path)
    if trace_sha256 != str(inputs["trace_sha256"]):
        raise ValueError("response_margin_trace_hash_mismatch")
    cases, integrity_errors = collect_response_cases(trace_path, profiles=profiles)

    fit_reports: dict[str, dict[str, Any]] = {}
    selected_margins: dict[str, dict[str, float]] = {}
    for profile in calibrated:
        policy = PROXY_TYPES[profile]()
        profile_cases = [case for case in cases if case.profile == profile]
        action_reports: dict[str, Any] = {}
        margins: dict[str, float] = {}
        for action in ("PENG", "CHI"):
            rows = response_margin_rows(profile_cases, policy=policy, action=action)
            fitted = fit_binary_margin(rows)
            action_reports[action] = fitted
            margins[f"{action.lower()}_margin"] = float(fitted["threshold"])
        fit_reports[profile] = action_reports
        selected_margins[profile] = margins

    predictions: list[dict[str, Any]] = []
    inference_times: list[float] = []
    for case in cases:
        policy = configured_policy(
            case.profile,
            selected_margins.get(case.profile),
        )
        started = time.perf_counter()
        selected = predict_response(case, policy=policy)
        inference_times.append((time.perf_counter() - started) * 1000.0)
        predictions.append(
            {
                "profile": case.profile,
                "game_id": case.game_id,
                "target": case.selected_key,
                "selected": selected,
                "legal": selected in {str(item["key"]) for item in case.legal_actions},
            }
        )
    metrics = response_metrics(predictions, inference_times=inference_times)
    states_by_profile = Counter(case.profile for case in cases)
    failures = integrity_failures(
        cases,
        fit_reports=fit_reports,
        metrics=metrics,
        states_by_profile=states_by_profile,
        integrity_errors=integrity_errors,
        gate=preregistration["integrity_gate"],
        calibrated_profiles=calibrated,
    )
    return {
        "ok": not failures,
        "schema_version": "response-proxy-margin-fit-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "trace": str(trace_path),
        "trace_sha256": trace_sha256,
        "response_roots": len(cases),
        "states_by_profile": dict(states_by_profile),
        "selected_margins": selected_margins,
        "fit_reports": fit_reports,
        "training_metrics": metrics,
        "integrity_errors": integrity_errors,
        "integrity_failures": failures,
    }


def collect_response_cases(
    trace_path: Path,
    *,
    profiles: Sequence[str],
) -> tuple[list[ResponseCase], list[str]]:
    cases: list[ResponseCase] = []
    errors: list[str] = []
    seen: set[str] = set()
    with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            game = json.loads(line)
            opponents = tuple(str(item) for item in game.get("opponents") or ())
            if len(opponents) != 1 or opponents[0] not in profiles:
                continue
            profile = opponents[0]
            teacher_names = {profile}
            if profile == "information_set_search":
                teacher_names.add("information_set_search_v2")
            game_id = (
                f"{profile}:{int(game.get('seed') or 0)}:"
                f"{int(game.get('candidate_seat') or 0)}:"
                f"{int(game.get('dealer') or 0)}:{source_index}"
            )
            for trace in game.get("decision_trace") or ():
                if (
                    str(trace.get("phase") or "") != "response_root"
                    or str(trace.get("policy") or "") not in teacher_names
                ):
                    continue
                group_id = (
                    f"{game_id}:{int(trace.get('sequence') or 0)}:"
                    f"{trace.get('state_before_hash')}"
                )
                actions = tuple(dict(item) for item in trace.get("legal_actions") or ())
                legal_keys = {str(item.get("key") or "") for item in actions}
                selected = str(trace.get("selected_key") or "")
                public_state = trace.get("public_state")
                if (
                    not group_id
                    or group_id in seen
                    or not isinstance(public_state, Mapping)
                    or selected not in legal_keys
                    or "PASS" not in legal_keys
                ):
                    errors.append("response_trace_contract")
                    continue
                seen.add(group_id)
                cases.append(
                    ResponseCase(
                        group_id=group_id,
                        game_id=game_id,
                        profile=profile,
                        players=int(game.get("players") or 2),
                        wildcard_enabled=bool(game.get("wildcard_enabled")),
                        public_state=dict(public_state),
                        legal_actions=actions,
                        selected_key=selected,
                    )
                )
    return cases, errors


def response_margin_rows(
    cases: Sequence[ResponseCase],
    *,
    policy: Any,
    action: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        action_types = {str(item.get("type") or "") for item in case.legal_actions}
        if "HU" in action_types or action not in action_types:
            continue
        if action == "CHI" and case.selected_key.startswith("PENG:"):
            continue
        view = public_view_from_dict(case.public_state)
        rules = rules_for_room(
            wildcard_enabled=case.wildcard_enabled,
            players=case.players,
        )
        if action == "PENG":
            gain = policy._peng_response_gain(view, str(view.pending_card), rules)
            predicted_key = f"PENG:{view.pending_card}"
        else:
            plans, keys = chi_plans_and_keys(case.legal_actions)
            gain, selected_plan = policy._best_chi_response_gain(view, plans, rules)
            predicted_key = (
                keys[plans.index(selected_plan)]
                if selected_plan is not None and selected_plan in plans
                else "PASS"
            )
        if gain is None or not math.isfinite(float(gain)):
            continue
        rows.append(
            {
                "score": float(gain),
                "target_positive": case.selected_key.startswith(f"{action}:"),
                "positive_key_correct": predicted_key == case.selected_key,
            }
        )
    return rows


def fit_binary_margin(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("response_margin_rows_empty")
    scores = sorted({float(row["score"]) for row in rows})
    thresholds = [math.nextafter(scores[0], -math.inf)]
    thresholds.extend(
        (left + right) / 2.0 for left, right in zip(scores, scores[1:])
    )
    thresholds.append(scores[-1])
    positives = sum(bool(row["target_positive"]) for row in rows)
    negatives = len(rows) - positives
    evaluated = [
        threshold_metrics(rows, threshold=threshold)
        for threshold in thresholds
    ]
    selected = max(
        evaluated,
        key=lambda item: (
            float(item["balanced_accuracy"]),
            float(item["exact_key_accuracy"]),
            float(item["precision"]),
            float(item["threshold"]),
        ),
    )
    return {
        **selected,
        "examples": len(rows),
        "positive_examples": positives,
        "negative_examples": negatives,
        "candidate_thresholds": len(thresholds),
    }


def threshold_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, float]:
    tp = fp = tn = fn = exact = 0
    for row in rows:
        predicted = float(row["score"]) > threshold
        target = bool(row["target_positive"])
        tp += int(predicted and target)
        fp += int(predicted and not target)
        tn += int(not predicted and not target)
        fn += int(not predicted and target)
        exact += int(
            (not predicted and not target)
            or predicted and target and bool(row["positive_key_correct"])
        )
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "threshold": threshold,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "exact_key_accuracy": exact / max(1, len(rows)),
        "precision": tp / max(1, tp + fp),
        "recall": recall,
        "specificity": specificity,
    }


def configured_policy(
    profile: str,
    margins: Mapping[str, float] | None,
) -> Any:
    policy = PROXY_TYPES[profile]()
    if not margins:
        return policy
    if hasattr(policy, "weights"):
        policy.weights = replace(
            policy.weights,
            peng_margin=float(margins["peng_margin"]),
            chi_margin=float(margins["chi_margin"]),
        )
    else:
        policy.peng_margin = float(margins["peng_margin"])
        policy.chi_margin = float(margins["chi_margin"])
    return policy


def predict_response(case: ResponseCase, *, policy: Any) -> str:
    view = public_view_from_dict(case.public_state)
    rules = rules_for_room(
        wildcard_enabled=case.wildcard_enabled,
        players=case.players,
    )
    action_types = {str(item.get("type") or "") for item in case.legal_actions}
    selected = "PASS"
    pending = str(view.pending_card or "")
    if "HU" in action_types:
        hu = evaluate_hu((*view.hand, pending), view.own_melds, rules)
        hu_view = _replace_view(
            view,
            hand=(*view.hand, pending),
            own_melds=view.own_melds,
        )
        if policy.choose_hu(hu_view, hu, rules):
            selected = "HU"
    if (
        selected == "PASS"
        and "PENG" in action_types
        and policy.choose_peng(view, pending, rules)
    ):
        selected = f"PENG:{pending}"
    plans, keys = chi_plans_and_keys(case.legal_actions)
    if selected == "PASS" and plans:
        plan = policy.choose_chi(view, plans, rules)
        if plan is not None and plan in plans:
            selected = keys[plans.index(plan)]
        elif plan is not None:
            selected = "ILLEGAL_CHI"
    return selected


def chi_plans_and_keys(
    actions: Sequence[Mapping[str, Any]],
) -> tuple[list[ChiPlan], list[str]]:
    plans: list[ChiPlan] = []
    keys: list[str] = []
    for action in actions:
        if str(action.get("type") or "") != "CHI":
            continue
        groups = tuple(tuple(group) for group in action.get("meld_groups") or ())
        if not groups:
            continue
        plans.append(
            ChiPlan(
                initial_group=groups[0],
                compare_groups=groups[1:],
                consumed_from_hand=tuple(action.get("consumed_from_hand") or ()),
            )
        )
        keys.append(str(action.get("key") or ""))
    return plans, keys


def response_metrics(
    predictions: Sequence[Mapping[str, Any]],
    *,
    inference_times: Sequence[float],
) -> dict[str, Any]:
    summary = response_metric_summary(predictions)
    by_profile = {
        profile: response_metric_summary(
            [row for row in predictions if row["profile"] == profile]
        )
        for profile in sorted({str(row["profile"]) for row in predictions})
    }
    elapsed = sorted(float(value) for value in inference_times)
    p95_index = max(0, math.ceil(0.95 * len(elapsed)) - 1) if elapsed else 0
    return {
        **summary,
        "p95_inference_ms": elapsed[p95_index] if elapsed else 0.0,
        "maximum_inference_ms": max(elapsed, default=0.0),
        "by_profile": by_profile,
    }


def response_metric_summary(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact = sum(row["target"] == row["selected"] for row in predictions)
    target_claims = sum(row["target"] != "PASS" for row in predictions)
    predicted_claims = sum(row["selected"] != "PASS" for row in predictions)
    correct_claims = sum(
        row["target"] == row["selected"] and row["target"] != "PASS"
        for row in predictions
    )
    by_action = {}
    for action in ("HU", "PENG", "CHI"):
        target_count = sum(_action_type(row["target"]) == action for row in predictions)
        predicted_count = sum(
            _action_type(row["selected"]) == action for row in predictions
        )
        correct_count = sum(
            row["target"] == row["selected"]
            and _action_type(row["target"]) == action
            for row in predictions
        )
        by_action[action] = {
            "target": target_count,
            "predicted": predicted_count,
            "correct": correct_count,
            "recall": correct_count / max(1, target_count),
            "precision": correct_count / max(1, predicted_count),
        }
    return {
        "responses": len(predictions),
        "exact_key_accuracy": exact / max(1, len(predictions)),
        "claim_recall": correct_claims / max(1, target_claims),
        "claim_precision": correct_claims / max(1, predicted_claims),
        "illegal_predictions": sum(not bool(row["legal"]) for row in predictions),
        "by_action": by_action,
    }


def _action_type(key: Any) -> str:
    return str(key or "PASS").split(":", 1)[0]


def integrity_failures(
    cases: Sequence[ResponseCase],
    *,
    fit_reports: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    states_by_profile: Mapping[str, int],
    integrity_errors: Sequence[str],
    gate: Mapping[str, Any],
    calibrated_profiles: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    if len(cases) < int(gate["minimum_response_roots"]):
        failures.append("minimum_response_roots")
    if any(
        int(states_by_profile.get(profile, 0)) < int(gate["minimum_cases_per_profile"])
        for profile in states_by_profile
    ):
        failures.append("minimum_cases_per_profile")
    for profile in calibrated_profiles:
        for action in ("PENG", "CHI"):
            report = fit_reports[profile][action]
            if int(report["examples"]) < int(gate["minimum_binary_examples_per_action_profile"]):
                failures.append("minimum_binary_examples_per_action_profile")
            if bool(gate["require_both_classes_per_action_profile"]) and (
                not int(report["positive_examples"]) or not int(report["negative_examples"])
            ):
                failures.append("both_classes_per_action_profile")
    if bool(gate["require_zero_illegal_predictions"]) and int(
        metrics["illegal_predictions"]
    ):
        failures.append("illegal_predictions")
    if integrity_errors:
        failures.append("integrity_errors")
    return sorted(set(failures))


if __name__ == "__main__":
    raise SystemExit(main())
