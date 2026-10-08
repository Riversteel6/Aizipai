"""Zero-update reproduction for the anchored seven structural discard proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import public_view_from_dict
from ai.opponent_proxy import (
    FastAggressiveMeldProxyPolicy,
    FastDefensiveSearchProxyPolicy,
    FastIndependentDenialStructuralDiscardProxyPolicy,
    FastIndependentPressureStructuralDiscardProxyPolicy,
    FastIndependentStructuralDiscardProxyPolicy,
    FastInformationSetProxyPolicy,
    FastRedBlackSearchProxyPolicy,
)
from engine.rules import rules_for_room
from tools.fit_anchored_teacher_discard_proxies import (
    collect_decisions,
    focused_gate_failures,
    grouped_fold_leakage,
    prediction_row,
    summarize_predictions,
)


PROXY_TYPES = {
    "aggressive_meld": FastAggressiveMeldProxyPolicy,
    "defensive_search": FastDefensiveSearchProxyPolicy,
    "independent_balanced": FastIndependentStructuralDiscardProxyPolicy,
    "independent_denial": FastIndependentDenialStructuralDiscardProxyPolicy,
    "independent_pressure": FastIndependentPressureStructuralDiscardProxyPolicy,
    "information_set_search": FastInformationSetProxyPolicy,
    "red_black_search": FastRedBlackSearchProxyPolicy,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preregistration_bytes = args.preregistration.read_bytes()
    preregistration = json.loads(preregistration_bytes.decode("utf-8"))
    report = run_reproduction(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(
            preregistration_bytes
        ).hexdigest(),
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


def run_reproduction(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    profiles = tuple(str(item) for item in preregistration["profiles"])
    if set(profiles) != set(PROXY_TYPES):
        raise ValueError("structural_proxy_profile_mapping_mismatch")
    folds = int(preregistration["folds"])
    inputs = dict(preregistration["inputs"])
    trace_path = Path(inputs["trace"])
    trace_sha256 = sha256_file(trace_path)
    if trace_sha256 != str(inputs["trace_sha256"]):
        raise ValueError("structural_proxy_trace_hash_mismatch")

    decisions, integrity_errors = collect_decisions(
        trace_path,
        profiles=profiles,
        folds=folds,
    )
    predictions: list[dict[str, Any]] = []
    inference_times: list[float] = []
    for profile in profiles:
        proxy = PROXY_TYPES[profile]()
        for decision in decisions[profile]:
            rules = rules_for_room(
                wildcard_enabled=decision.wildcard_enabled,
                players=decision.players,
            )
            view = public_view_from_dict(decision.public_state)
            started = time.perf_counter()
            selected = proxy.choose_discard(view, rules)
            inference_times.append((time.perf_counter() - started) * 1000.0)
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
    return {
        "ok": not integrity_errors and bool(predictions),
        "schema_version": "anchored-structural-discard-proxy-reproduction-v1",
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
        "integrity_errors": integrity_errors,
        "metrics": metrics,
        "focused_gate": dict(preregistration["focused_gate"]),
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
