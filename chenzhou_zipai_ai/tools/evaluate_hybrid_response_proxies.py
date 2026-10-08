"""Evaluate frozen hybrid response proxies against teacher response traces."""

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

from ai.opponent_proxy import (
    FastAggressiveMeldProxyPolicy,
    HybridDefensiveSearchResponseProxyPolicy,
    HybridIndependentDenialResponseProxyPolicy,
    HybridIndependentPressureResponseProxyPolicy,
    HybridIndependentResponseProxyPolicy,
    HybridInformationSetResponseProxyPolicy,
    HybridRedBlackSearchResponseProxyPolicy,
)
from tools.evaluate_anchored_structural_discard_proxies import sha256_file
from tools.fit_response_proxy_margins import (
    collect_response_cases,
    predict_response,
    response_metrics,
)


HYBRID_RESPONSE_PROXY_TYPES = {
    "aggressive_meld": FastAggressiveMeldProxyPolicy,
    "defensive_search": HybridDefensiveSearchResponseProxyPolicy,
    "independent_balanced": HybridIndependentResponseProxyPolicy,
    "independent_denial": HybridIndependentDenialResponseProxyPolicy,
    "independent_pressure": HybridIndependentPressureResponseProxyPolicy,
    "information_set_search": HybridInformationSetResponseProxyPolicy,
    "red_black_search": HybridRedBlackSearchResponseProxyPolicy,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_evaluation(
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
                "response_roots": report["response_roots"],
                "state_hash_overlap": report["state_hash_overlap"],
                "exact_key_accuracy": report["metrics"]["exact_key_accuracy"],
                "claim_recall": report["metrics"]["claim_recall"],
                "claim_precision": report["metrics"]["claim_precision"],
                "p95_inference_ms": report["metrics"]["p95_inference_ms"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def run_evaluation(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    profiles = tuple(str(item) for item in preregistration["profiles"])
    if set(profiles) != set(HYBRID_RESPONSE_PROXY_TYPES):
        raise ValueError("hybrid_response_profile_mapping_mismatch")
    inputs = dict(preregistration["inputs"])
    trace_path = Path(inputs["trace"])
    trace_sha256 = sha256_file(trace_path)
    if trace_sha256 != str(inputs["trace_sha256"]):
        raise ValueError("hybrid_response_trace_hash_mismatch")
    development_report_path = inputs.get("development_report")
    development_report_sha256 = None
    if development_report_path:
        development_report_sha256 = sha256_file(Path(str(development_report_path)))
        if development_report_sha256 != str(inputs["development_report_sha256"]):
            raise ValueError("hybrid_response_development_report_hash_mismatch")
    cases, integrity_errors = collect_response_cases(trace_path, profiles=profiles)

    overlap = 0
    reference_path = inputs.get("zero_overlap_reference_trace")
    reference_sha256 = None
    if reference_path:
        resolved_reference = Path(str(reference_path))
        reference_sha256 = sha256_file(resolved_reference)
        if reference_sha256 != str(inputs["zero_overlap_reference_trace_sha256"]):
            raise ValueError("hybrid_response_reference_hash_mismatch")
        reference_cases, reference_errors = collect_response_cases(
            resolved_reference,
            profiles=profiles,
        )
        integrity_errors.extend(reference_errors)
        state_hashes = {_state_hash(case.group_id) for case in cases}
        reference_hashes = {_state_hash(case.group_id) for case in reference_cases}
        overlap = len(state_hashes & reference_hashes)

    predictions: list[dict[str, Any]] = []
    inference_times: list[float] = []
    times_by_profile: dict[str, list[float]] = defaultdict(list)
    policies = {
        profile: HYBRID_RESPONSE_PROXY_TYPES[profile]() for profile in profiles
    }
    for case in cases:
        started = time.perf_counter()
        selected = predict_response(case, policy=policies[case.profile])
        elapsed = (time.perf_counter() - started) * 1000.0
        inference_times.append(elapsed)
        times_by_profile[case.profile].append(elapsed)
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
    for profile, values in times_by_profile.items():
        metrics["by_profile"][profile]["p95_inference_ms"] = percentile_95(values)
        metrics["by_profile"][profile]["maximum_inference_ms"] = max(values, default=0.0)
    states_by_profile = Counter(case.profile for case in cases)
    failures = evaluation_failures(
        metrics,
        states=len(cases),
        states_by_profile=states_by_profile,
        overlap=overlap,
        integrity_errors=integrity_errors,
        integrity_gate=preregistration["integrity_gate"],
        acceptance_gate=preregistration.get("acceptance_gate"),
    )
    return {
        "ok": not integrity_errors and bool(cases),
        "schema_version": "hybrid-response-proxy-evaluation-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "trace": str(trace_path),
        "trace_sha256": trace_sha256,
        "development_report": (
            str(development_report_path) if development_report_path else None
        ),
        "development_report_sha256": development_report_sha256,
        "reference_trace": str(reference_path) if reference_path else None,
        "reference_trace_sha256": reference_sha256,
        "response_roots": len(cases),
        "states_by_profile": dict(states_by_profile),
        "state_hash_overlap": overlap,
        "metrics": metrics,
        "integrity_errors": integrity_errors,
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
    }


def evaluation_failures(
    metrics: Mapping[str, Any],
    *,
    states: int,
    states_by_profile: Mapping[str, int],
    overlap: int,
    integrity_errors: Sequence[str],
    integrity_gate: Mapping[str, Any],
    acceptance_gate: Mapping[str, Any] | None,
) -> list[str]:
    failures: list[str] = []
    if states < int(integrity_gate["minimum_response_roots"]):
        failures.append("minimum_response_roots")
    if len(states_by_profile) != len(HYBRID_RESPONSE_PROXY_TYPES) or any(
        int(states_by_profile.get(profile, 0))
        < int(integrity_gate["minimum_cases_per_profile"])
        for profile in HYBRID_RESPONSE_PROXY_TYPES
    ):
        failures.append("minimum_cases_per_profile")
    if bool(integrity_gate["require_zero_illegal_predictions"]) and int(
        metrics["illegal_predictions"]
    ):
        failures.append("illegal_predictions")
    if bool(integrity_gate["require_zero_integrity_errors"]) and integrity_errors:
        failures.append("integrity_errors")
    if bool(integrity_gate.get("require_zero_state_hash_overlap")) and overlap:
        failures.append("state_hash_overlap")
    if not acceptance_gate:
        return failures
    if float(metrics["exact_key_accuracy"]) < float(
        acceptance_gate["minimum_exact_key_accuracy"]
    ):
        failures.append("exact_key_accuracy")
    if float(metrics["claim_recall"]) < float(
        acceptance_gate["minimum_claim_recall"]
    ):
        failures.append("claim_recall")
    if float(metrics["claim_precision"]) < float(
        acceptance_gate["minimum_claim_precision"]
    ):
        failures.append("claim_precision")
    for profile, minimum in dict(
        acceptance_gate["minimum_profile_exact_key_accuracy"]
    ).items():
        if float(metrics["by_profile"][profile]["exact_key_accuracy"]) < float(minimum):
            failures.append(f"profile_exact_key_accuracy:{profile}")
    for profile, minimum in dict(
        acceptance_gate.get("minimum_profile_claim_recall", {})
    ).items():
        if float(metrics["by_profile"][profile]["claim_recall"]) < float(minimum):
            failures.append(f"profile_claim_recall:{profile}")
    for profile, minimum in dict(
        acceptance_gate.get("minimum_profile_claim_precision", {})
    ).items():
        if float(metrics["by_profile"][profile]["claim_precision"]) < float(minimum):
            failures.append(f"profile_claim_precision:{profile}")
    for action, minimum in dict(acceptance_gate["minimum_action_recall"]).items():
        if float(metrics["by_action"][action]["recall"]) < float(minimum):
            failures.append(f"action_recall:{action}")
    for action, minimum in dict(acceptance_gate["minimum_action_precision"]).items():
        if float(metrics["by_action"][action]["precision"]) < float(minimum):
            failures.append(f"action_precision:{action}")
    if float(metrics["p95_inference_ms"]) > float(
        acceptance_gate["maximum_overall_p95_inference_ms"]
    ):
        failures.append("overall_p95_inference_ms")
    if any(
        float(row["p95_inference_ms"])
        > float(acceptance_gate["maximum_profile_p95_inference_ms"])
        for row in metrics["by_profile"].values()
    ):
        failures.append("profile_p95_inference_ms")
    return failures


def percentile_95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _state_hash(group_id: str) -> str:
    return str(group_id).rsplit(":", 1)[-1]


if __name__ == "__main__":
    raise SystemExit(main())
