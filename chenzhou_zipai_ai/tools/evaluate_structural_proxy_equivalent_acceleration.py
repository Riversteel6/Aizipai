"""Verify action equivalence and latency of count-state proxy acceleration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.full_game_simulator import _discardable_labels, _public_discard_danger
from ai.ismcts import public_view_from_dict
from engine.cards import RED_LABELS
from engine.rules import rules_for_room
from tools.evaluate_anchored_structural_discard_proxies import (
    PROXY_TYPES,
    sha256_file,
)
from tools.fit_anchored_teacher_discard_proxies import (
    collect_decisions,
    focused_gate_failures,
    grouped_fold_leakage,
    prediction_row,
    summarize_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_equivalence(
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
                "states": report["states"],
                "action_mismatches": report["action_mismatches"],
                "overall_top1_accuracy": report["metrics"][
                    "overall_state_top1_accuracy"
                ],
                "p95_inference_ms": report["metrics"]["p95_inference_ms"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def legacy_unbatched_selected(
    profile: str,
    policy: Any,
    view: Any,
    rules: dict[str, Any],
) -> str:
    labels = _discardable_labels(view.hand)
    if not labels:
        raise ValueError("no_discardable_card")
    if profile in {"aggressive_meld", "information_set_search"}:
        shortlist_size = 8 if rules.get("wildcard", {}).get("enabled", False) else 4
        shortlist = sorted(
            labels,
            key=lambda label: policy._cheap_discard_value(view, label, rules),
            reverse=True,
        )[:shortlist_size]
        return max(
            shortlist,
            key=lambda label: policy._discard_value(view, label, rules),
        )
    if profile == "defensive_search":
        return max(
            labels,
            key=lambda label: (
                policy._discard_value(view, label, rules)
                - _public_discard_danger(label, view) * 2.5,
                label,
            ),
        )
    if profile == "red_black_search":
        current_red = sum(label in RED_LABELS for label in view.hand)
        chase_red = current_red >= 7

        def route_value(label: str) -> float:
            after_red = current_red - int(label in RED_LABELS)
            route_bonus = (
                after_red * 18.0
                if chase_red
                else -min(abs(after_red), abs(after_red - 1)) * 22.0
            )
            return policy._discard_value(view, label, rules) + route_bonus

        return max(labels, key=lambda label: (route_value(label), label))
    raise ValueError(f"unaffected_profile:{profile}")


def run_equivalence(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    profiles = tuple(str(item) for item in preregistration["profiles"])
    affected = set(str(item) for item in preregistration["affected_profiles"])
    if set(profiles) != set(PROXY_TYPES) or not affected.issubset(PROXY_TYPES):
        raise ValueError("equivalent_acceleration_profile_mapping_mismatch")
    inputs = dict(preregistration["inputs"])
    trace_path = Path(inputs["trace"])
    prior_path = Path(inputs["failed_reproduction_report"])
    if sha256_file(trace_path) != str(inputs["trace_sha256"]):
        raise ValueError("equivalent_acceleration_trace_hash_mismatch")
    if sha256_file(prior_path) != str(inputs["failed_reproduction_report_sha256"]):
        raise ValueError("equivalent_acceleration_prior_report_hash_mismatch")

    folds = int(preregistration["folds"])
    decisions, integrity_errors = collect_decisions(
        trace_path,
        profiles=profiles,
        folds=folds,
    )
    predictions: list[dict[str, Any]] = []
    inference_times: list[float] = []
    mismatch_by_profile = Counter()
    for profile in profiles:
        policy = PROXY_TYPES[profile]()
        for decision in decisions[profile]:
            rules = rules_for_room(
                wildcard_enabled=decision.wildcard_enabled,
                players=decision.players,
            )
            view = public_view_from_dict(decision.public_state)
            old_selected = (
                legacy_unbatched_selected(profile, policy, view, rules)
                if profile in affected
                else None
            )
            started = time.perf_counter()
            selected = policy.choose_discard(view, rules)
            inference_times.append((time.perf_counter() - started) * 1000.0)
            if old_selected is not None and selected != old_selected:
                mismatch_by_profile[profile] += 1
            predictions.append(prediction_row(decision, selected=selected))

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
        target_mismatches=0,
        integrity_errors=len(integrity_errors),
    )
    action_mismatches = sum(mismatch_by_profile.values())
    if (
        preregistration["focused_gate"].get("require_zero_action_mismatches")
        and action_mismatches
    ):
        failures.append("action_mismatches")
    return {
        "ok": not integrity_errors and bool(predictions),
        "schema_version": "structural-discard-proxy-equivalent-acceleration-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "prior_report": str(prior_path),
        "prior_report_sha256": sha256_file(prior_path),
        "profiles": list(profiles),
        "affected_profiles": sorted(affected),
        "states": len(predictions),
        "states_by_profile": states_by_profile,
        "group_leakage": leakage,
        "action_mismatches": action_mismatches,
        "action_mismatches_by_profile": dict(mismatch_by_profile),
        "integrity_errors": integrity_errors,
        "metrics": metrics,
        "focused_gate": dict(preregistration["focused_gate"]),
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
    }


if __name__ == "__main__":
    raise SystemExit(main())
