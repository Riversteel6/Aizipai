"""Replay the runtime progressive policy against 64-world decision labels."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
while str(APP_ROOT) in sys.path:
    sys.path.remove(str(APP_ROOT))

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from statistics import mean, stdev
from typing import Any, Iterable


sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import (
    ProgressiveRootISMCTSPolicy,
    RootISMCTSConfig,
    RootResponseCandidate,
    public_view_from_dict,
)
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from engine.rules import rules_for_room


ROLLOUT_FACTORIES = (
    IndependentFastRolloutPolicy,
    IndependentFastPressurePolicy,
    IndependentFastDenialPolicy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extra-evidence", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--state-hash", action="append", default=[])
    parser.add_argument(
        "--mode",
        choices=("2p_off", "2p_on", "3p_off", "3p_on"),
    )
    parser.add_argument(
        "--phase",
        choices=("discard", "response_root", "hu"),
    )
    parser.add_argument("--coverage-time-budget-ms", type=int, default=1_800)
    parser.add_argument("--refinement-candidates", type=int, default=5)
    parser.add_argument("--refinement-worlds", type=int, default=8)
    parser.add_argument("--refinement-time-budget-ms", type=int, default=2_500)
    parser.add_argument("--discard-selection-worlds", type=int, default=16)
    parser.add_argument("--discard-selection-time-budget-ms", type=int, default=4_000)
    parser.add_argument("--confirmation-worlds", type=int, default=96)
    parser.add_argument("--confirmation-time-budget-ms", type=int, default=12_000)
    parser.add_argument(
        "--minimum-confident-advantage",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--discard-confirmation-alternatives",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    cases, source_rows = _load_cases(
        args.manifest,
        extra_evidence=args.extra_evidence,
    )
    requested_hashes = {str(value) for value in args.state_hash if str(value)}
    if requested_hashes:
        cases = [
            case
            for case in cases
            if str(case["state_before_hash"]) in requested_hashes
        ]
    if args.mode:
        cases = [case for case in cases if _mode(case) == args.mode]
    if args.phase:
        phases = (
            {"self_hu", "post_action_hu", "post_auto_hu"}
            if args.phase == "hu"
            else {args.phase}
        )
        cases = [
            case
            for case in cases
            if str(case["phase"]) in phases
        ]
    if args.limit is not None:
        cases = cases[: max(1, args.limit)]
    policy_config = {
        "coverage_time_budget_ms": max(1, args.coverage_time_budget_ms),
        "refinement_candidates": max(2, args.refinement_candidates),
        "refinement_worlds": max(1, args.refinement_worlds),
        "refinement_time_budget_ms": max(
            1,
            args.refinement_time_budget_ms,
        ),
        "discard_selection_worlds": max(
            1,
            args.discard_selection_worlds,
        ),
        "discard_selection_time_budget_ms": max(
            1,
            args.discard_selection_time_budget_ms,
        ),
        "confirmation_worlds": max(1, args.confirmation_worlds),
        "confirmation_time_budget_ms": max(
            1,
            args.confirmation_time_budget_ms,
        ),
        "minimum_confident_advantage": max(
            0.0,
            args.minimum_confident_advantage,
        ),
        "discard_confirmation_alternatives": max(
            1,
            args.discard_confirmation_alternatives,
        ),
    }
    for case in cases:
        case["policy_config"] = policy_config
    if args.workers > 1 and len(cases) > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=get_context("spawn"),
        ) as executor:
            rows = list(executor.map(_evaluate_case, cases))
    else:
        rows = [_evaluate_case(case) for case in cases]
    report = _summarize(
        rows,
        manifest=args.manifest,
        source_rows=source_rows,
        policy_config=policy_config,
        extra_evidence=args.extra_evidence,
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
                    "source_rows",
                    "unique_states",
                    "evaluated_states",
                    "errors",
                    "candidate_coverage_failures",
                    "baseline_mean_regret",
                    "runtime_mean_regret",
                    "mean_regret_reduction",
                    "runtime_overrides",
                    "supported_overrides",
                    "harmful_overrides",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def _load_cases(
    manifest_path: Path,
    *,
    extra_evidence: Iterable[Path] = (),
) -> tuple[list[dict[str, Any]], int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_rows = 0
    seen_paths: set[Path] = set()
    evidence_sources = [
        (Path(str(entry["output_evidence"])), True)
        for entry in manifest.get("entries") or []
    ]
    evidence_sources.extend(
        (Path(path), False)
        for path in extra_evidence
    )
    for evidence_path, from_manifest in evidence_sources:
        if not evidence_path.is_absolute():
            evidence_path = (
                manifest_path.parent.parent / evidence_path
                if from_manifest
                else evidence_path
            )
        evidence_path = evidence_path.resolve()
        if evidence_path in seen_paths:
            continue
        seen_paths.add(evidence_path)
        with gzip.open(evidence_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                source_rows += 1
                if not row.get("complete"):
                    continue
                game = row.get("game") or {}
                identity = (
                    str(row["state_before_hash"]),
                    str(row["phase"]),
                    int(row["seat"]),
                    int(game["players"]),
                    bool(game["wildcard_enabled"]),
                )
                case = {
                    key: row[key]
                    for key in (
                        "state_before_hash",
                        "phase",
                        "seat",
                        "selected_key",
                        "best_key",
                        "legal_action_count",
                        "completed_paired_worlds",
                        "public_state",
                        "candidate_stats",
                    )
                }
                case["confidently_suboptimal"] = bool(
                    row.get("confidently_suboptimal")
                )
                case["deep_paired_advantages"] = list(
                    row.get("paired_advantages") or ()
                )
                case["game"] = {
                    "players": int(game["players"]),
                    "wildcard_enabled": bool(game["wildcard_enabled"]),
                }
                case["source_evidence"] = str(evidence_path)
                previous = by_identity.get(identity)
                if (
                    previous is None
                    or int(case["completed_paired_worlds"])
                    > int(previous["completed_paired_worlds"])
                ):
                    by_identity[identity] = case
    return (
        sorted(
            by_identity.values(),
            key=lambda case: (
                int(case["game"]["players"]),
                bool(case["game"]["wildcard_enabled"]),
                str(case["phase"]),
                str(case["state_before_hash"]),
            ),
        ),
        source_rows,
    )


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    started_error: str | None = None
    result: Any | None = None
    runtime_key: str | None = None
    try:
        view = public_view_from_dict(case["public_state"])
        rules = rules_for_room(
            wildcard_enabled=bool(case["game"]["wildcard_enabled"]),
            players=int(case["game"]["players"]),
        )
        search = _progressive_search(case.get("policy_config") or {})
        phase = str(case["phase"])
        if phase == "discard":
            labels = [str(item["label"]) for item in case["candidate_stats"]]
            result = search.search_discard(
                view,
                rules=rules,
                candidate_labels=labels,
                candidate_priors={
                    str(item["label"]): float(
                        item.get("heuristic_value") or 0.0
                    )
                    for item in case["candidate_stats"]
                },
                preferred_label=_discard_label(str(case["selected_key"])),
            )
            runtime_key = f"DISCARD:{result.selected_label}"
        elif phase == "response_root":
            candidates = [
                _response_candidate(item)
                for item in case["candidate_stats"]
            ]
            result = search.search_response(
                view,
                rules=rules,
                candidates=candidates,
                preferred_key=str(case["selected_key"]),
            )
            runtime_key = result.selected_key
        elif phase in {"self_hu", "post_action_hu", "post_auto_hu"}:
            result = search.search_hu(
                view,
                rules=rules,
                preferred_key=str(case["selected_key"]),
            )
            runtime_key = result.selected_key
        else:
            raise ValueError(f"unsupported_phase:{phase}")
    except Exception as exc:  # pragma: no cover - exercised by report health
        started_error = f"{type(exc).__name__}:{exc}"

    rewards = {
        _candidate_key(case["phase"], item): float(
            item.get("average_reward") or 0.0
        )
        for item in case["candidate_stats"]
    }
    baseline_key = str(case["selected_key"])
    best_key = str(case["best_key"])
    best_reward = max(rewards.values())
    baseline_reward = rewards.get(baseline_key)
    runtime_reward = rewards.get(runtime_key) if runtime_key is not None else None
    if started_error is None and runtime_reward is None:
        started_error = f"runtime_key_missing_from_deep_labels:{runtime_key}"
    deep_override_advantage = _deep_override_advantage(
        case,
        baseline_key=baseline_key,
        runtime_key=runtime_key,
    )
    covered = (
        result is not None
        and len(result.candidates) == int(case["legal_action_count"])
        and all(int(item.visits) > 0 for item in result.candidates)
    )
    return {
        "state_before_hash": case["state_before_hash"],
        "source_evidence": case["source_evidence"],
        "players": int(case["game"]["players"]),
        "wildcard_enabled": bool(case["game"]["wildcard_enabled"]),
        "mode": _mode(case),
        "phase": case["phase"],
        "legal_action_count": int(case["legal_action_count"]),
        "deep_worlds": int(case["completed_paired_worlds"]),
        "baseline_key": baseline_key,
        "runtime_key": runtime_key,
        "best_key": best_key,
        "baseline_reward": baseline_reward,
        "runtime_reward": runtime_reward,
        "best_reward": best_reward,
        "baseline_regret": (
            best_reward - float(baseline_reward)
            if baseline_reward is not None
            else None
        ),
        "runtime_regret": (
            best_reward - float(runtime_reward)
            if runtime_reward is not None
            else None
        ),
        "regret_reduction": (
            float(runtime_reward) - float(baseline_reward)
            if runtime_reward is not None and baseline_reward is not None
            else None
        ),
        "runtime_override": runtime_key is not None and runtime_key != baseline_key,
        "runtime_matches_deep_best": runtime_key == best_key,
        "baseline_matches_deep_best": baseline_key == best_key,
        "baseline_confidently_suboptimal": bool(
            case.get("confidently_suboptimal")
        ),
        "deep_override_advantage": deep_override_advantage,
        "deep_override_confidently_supported": bool(
            deep_override_advantage
            and deep_override_advantage.get("lower_confidence_bound")
            is not None
            and float(deep_override_advantage["lower_confidence_bound"]) > 0.0
        ),
        "deep_override_confidently_harmful": bool(
            deep_override_advantage
            and deep_override_advantage.get("upper_confidence_bound")
            is not None
            and float(deep_override_advantage["upper_confidence_bound"]) < 0.0
        ),
        "all_candidates_covered": covered,
        "used_search": bool(result.used_search) if result is not None else False,
        "reason": result.reason if result is not None else None,
        "simulations": int(result.simulations) if result is not None else 0,
        "paired_determinizations": (
            int(result.paired_determinizations)
            if result is not None
            else 0
        ),
        "elapsed_ms": float(result.elapsed_ms) if result is not None else 0.0,
        "confidence_override": (
            bool(result.confidence_override)
            if result is not None
            else False
        ),
        "paired_advantages": (
            [item.to_dict() for item in result.paired_advantages]
            if result is not None
            else []
        ),
        "search_stages": {
            "coverage": _stage_summary(search.last_coverage),
            "refinement": _stage_summary(search.last_refinement),
            "discard_selection": _stage_summary(
                search.last_discard_selection
            ),
            "confirmation": _stage_summary(search.last_confirmation),
        },
        "error": started_error,
    }


def _deep_override_advantage(
    case: dict[str, Any],
    *,
    baseline_key: str,
    runtime_key: str | None,
) -> dict[str, Any] | None:
    if runtime_key is None or runtime_key == baseline_key:
        return None
    if str(case.get("phase")) == "discard":
        expected_preferred = _discard_label(baseline_key)
        expected_candidate = _discard_label(runtime_key)
    else:
        expected_preferred = baseline_key
        expected_candidate = runtime_key
    return next(
        (
            dict(item)
            for item in case.get("deep_paired_advantages") or ()
            if str(item.get("preferred_key")) == expected_preferred
            and str(item.get("candidate_key")) == expected_candidate
        ),
        None,
    )


def _progressive_search(
    config: dict[str, Any],
) -> ProgressiveRootISMCTSPolicy:
    refinement_candidates = max(
        2,
        int(config.get("refinement_candidates", 3)),
    )
    refinement_worlds = max(1, int(config.get("refinement_worlds", 8)))
    discard_selection_worlds = max(
        1,
        int(config.get("discard_selection_worlds", 16)),
    )
    confirmation_worlds = max(
        1,
        int(config.get("confirmation_worlds", 96)),
    )
    discard_confirmation_alternatives = max(
        1,
        int(config.get("discard_confirmation_alternatives", 2)),
    )
    confirmation_candidates = 1 + discard_confirmation_alternatives
    return ProgressiveRootISMCTSPolicy(
        coverage_config=RootISMCTSConfig(
            time_budget_ms=max(
                1,
                int(config.get("coverage_time_budget_ms", 1_800)),
            ),
            max_iterations=24,
            max_candidates=24,
            skip_search_gap=math.inf,
            rollout_max_turns=120,
            require_confident_override=False,
            seed=20260726,
        ),
        refinement_config=RootISMCTSConfig(
            time_budget_ms=max(
                1,
                int(config.get("refinement_time_budget_ms", 2_500)),
            ),
            max_iterations=refinement_worlds * refinement_candidates,
            max_candidates=refinement_candidates,
            skip_search_gap=math.inf,
            rollout_max_turns=120,
            require_confident_override=True,
            min_confidence_pairs=min(6, refinement_worlds),
            seed=20260727,
        ),
        discard_selection_config=RootISMCTSConfig(
            time_budget_ms=max(
                1,
                int(config.get("discard_selection_time_budget_ms", 4_000)),
            ),
            max_iterations=(
                discard_selection_worlds * refinement_candidates
            ),
            max_candidates=refinement_candidates,
            skip_search_gap=math.inf,
            rollout_max_turns=120,
            require_confident_override=False,
            require_complete_iteration_budget_for_override=True,
            seed=20260729,
        ),
        confirmation_config=RootISMCTSConfig(
            time_budget_ms=max(
                1,
                int(config.get("confirmation_time_budget_ms", 6_000)),
            ),
            max_iterations=confirmation_worlds * confirmation_candidates,
            max_candidates=confirmation_candidates,
            skip_search_gap=math.inf,
            rollout_max_turns=120,
            require_confident_override=True,
            min_confidence_pairs=min(32, confirmation_worlds),
            minimum_confident_advantage=max(
                0.0,
                float(config.get("minimum_confident_advantage", 0.0)),
            ),
            seed=20260728,
        ),
        refinement_candidates=refinement_candidates,
        discard_confirmation_alternatives=(
            discard_confirmation_alternatives
        ),
        rollout_policy_factories=ROLLOUT_FACTORIES,
    )


def _stage_summary(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    candidates = []
    for item in result.candidates:
        candidate = getattr(item, "candidate", None)
        key = (
            str(candidate.key)
            if candidate is not None
            else str(item.label)
        )
        candidates.append(
            {
                "key": key,
                "visits": int(item.visits),
                "average_reward": round(float(item.average_reward), 6),
            }
        )
    return {
        "selected_key": str(
            getattr(
                result,
                "selected_key",
                getattr(result, "selected_label", ""),
            )
        ),
        "empirical_best_key": getattr(
            result,
            "empirical_best_key",
            getattr(result, "empirical_best_label", None),
        ),
        "used_search": bool(result.used_search),
        "reason": str(result.reason),
        "simulations": int(result.simulations),
        "paired_determinizations": int(result.paired_determinizations),
        "elapsed_ms": round(float(result.elapsed_ms), 3),
        "confidence_override": bool(result.confidence_override),
        "candidates": candidates,
        "paired_advantages": [
            item.to_dict() for item in result.paired_advantages
        ],
    }


def _response_candidate(item: dict[str, Any]) -> RootResponseCandidate:
    return RootResponseCandidate(
        key=str(item["key"]),
        action_type=str(item["action_type"]),
        heuristic_value=float(item.get("heuristic_value") or 0.0),
        option_id=(
            str(item["option_id"])
            if item.get("option_id") is not None
            else None
        ),
        consumed_from_hand=tuple(
            str(label)
            for label in item.get("consumed_from_hand") or ()
        ),
        meld_groups=tuple(
            tuple(str(label) for label in group)
            for group in item.get("meld_groups") or ()
        ),
        followup_discard=(
            str(item["followup_discard"])
            if item.get("followup_discard") is not None
            else None
        ),
    )


def _candidate_key(phase: str, item: dict[str, Any]) -> str:
    if phase == "discard":
        return f"DISCARD:{item['label']}"
    return str(item["key"])


def _discard_label(key: str) -> str:
    prefix = "DISCARD:"
    if not key.startswith(prefix):
        raise ValueError(f"invalid_discard_key:{key}")
    return key[len(prefix) :]


def _mode(case: dict[str, Any]) -> str:
    return (
        f"{int(case['game']['players'])}p_"
        f"{'on' if case['game']['wildcard_enabled'] else 'off'}"
    )


def _summarize(
    rows: list[dict[str, Any]],
    *,
    manifest: Path,
    source_rows: int,
    policy_config: dict[str, Any] | None = None,
    extra_evidence: Iterable[Path] = (),
) -> dict[str, Any]:
    healthy = [row for row in rows if row["error"] is None]
    errors = [row for row in rows if row["error"] is not None]
    coverage_failures = [
        row for row in healthy if not row["all_candidates_covered"]
    ]
    overrides = [row for row in healthy if row["runtime_override"]]
    supported = [
        row for row in overrides if float(row["regret_reduction"]) > 0.0
    ]
    harmful = [
        row for row in overrides if float(row["regret_reduction"]) < 0.0
    ]
    confidently_supported = [
        row
        for row in overrides
        if row.get("deep_override_confidently_supported")
    ]
    confidently_harmful = [
        row
        for row in overrides
        if row.get("deep_override_confidently_harmful")
    ]
    confidence_inconclusive = [
        row
        for row in overrides
        if not row.get("deep_override_confidently_supported")
        and not row.get("deep_override_confidently_harmful")
    ]
    by_mode = {
        mode: _group_summary(group)
        for mode, group in _group_by(healthy, "mode")
    }
    by_phase = {
        phase: _group_summary(group)
        for phase, group in _group_by(healthy, "phase")
    }
    baseline_regrets = [
        float(row["baseline_regret"]) for row in healthy
    ]
    runtime_regrets = [
        float(row["runtime_regret"]) for row in healthy
    ]
    reductions = [
        float(row["regret_reduction"]) for row in healthy
    ]
    no_mode_regression = all(
        float(summary["runtime_mean_regret"])
        <= float(summary["baseline_mean_regret"]) + 1e-12
        for summary in by_mode.values()
    )
    required_modes = {"2p_off", "2p_on", "3p_off", "3p_on"}
    required_modes_present = set(by_mode) == required_modes
    all_modes_positive_95ci = (
        required_modes_present
        and all(
            float(summary["regret_reduction_95ci"]["low"]) > 0.0
            for summary in by_mode.values()
        )
    )
    safety_gate = (
        not errors
        and not coverage_failures
        and not harmful
        and not confidently_harmful
        and not confidence_inconclusive
        and no_mode_regression
        and bool(healthy)
    )
    development_advantage_gate = safety_gate and all_modes_positive_95ci
    return {
        "schema_version": "progressive-deep-recheck-evaluation-v2",
        "ok": not errors and not coverage_failures,
        "manifest": str(manifest),
        "extra_evidence": [str(path) for path in extra_evidence],
        "development_evidence_only": True,
        "policy_config": dict(policy_config or {}),
        "source_rows": source_rows,
        "unique_states": len(rows),
        "evaluated_states": len(healthy),
        "errors": len(errors),
        "candidate_coverage_failures": len(coverage_failures),
        "baseline_mean_regret": _rounded_mean(baseline_regrets),
        "runtime_mean_regret": _rounded_mean(runtime_regrets),
        "mean_regret_reduction": _rounded_mean(reductions),
        "regret_reduction_95ci": _mean_ci(reductions),
        "baseline_best_matches": sum(
            int(row["baseline_matches_deep_best"]) for row in healthy
        ),
        "runtime_best_matches": sum(
            int(row["runtime_matches_deep_best"]) for row in healthy
        ),
        "runtime_overrides": len(overrides),
        "supported_overrides": len(supported),
        "neutral_overrides": len(overrides) - len(supported) - len(harmful),
        "harmful_overrides": len(harmful),
        "confidently_supported_overrides": len(confidently_supported),
        "confidently_harmful_overrides": len(confidently_harmful),
        "confidence_inconclusive_overrides": len(confidence_inconclusive),
        "no_mode_mean_regret_regression": no_mode_regression,
        "required_modes_present": required_modes_present,
        "all_modes_positive_regret_reduction_95ci": (
            all_modes_positive_95ci
        ),
        "development_safety_gate_passed": safety_gate,
        "development_advantage_gate_passed": development_advantage_gate,
        "accepted_as_runtime_candidate": development_advantage_gate,
        "by_mode": by_mode,
        "by_phase": by_phase,
        "error_samples": errors[:20],
        "coverage_failure_samples": coverage_failures[:20],
        "override_rows": overrides,
        "rows": rows,
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [float(row["baseline_regret"]) for row in rows]
    runtime = [float(row["runtime_regret"]) for row in rows]
    reductions = [float(row["regret_reduction"]) for row in rows]
    overrides = [row for row in rows if row["runtime_override"]]
    return {
        "states": len(rows),
        "baseline_mean_regret": _rounded_mean(baseline),
        "runtime_mean_regret": _rounded_mean(runtime),
        "mean_regret_reduction": _rounded_mean(reductions),
        "regret_reduction_95ci": _mean_ci(reductions),
        "baseline_best_matches": sum(
            int(row["baseline_matches_deep_best"]) for row in rows
        ),
        "runtime_best_matches": sum(
            int(row["runtime_matches_deep_best"]) for row in rows
        ),
        "overrides": len(overrides),
        "supported_overrides": sum(
            int(float(row["regret_reduction"]) > 0.0)
            for row in overrides
        ),
        "harmful_overrides": sum(
            int(float(row["regret_reduction"]) < 0.0)
            for row in overrides
        ),
        "confidently_supported_overrides": sum(
            int(bool(row.get("deep_override_confidently_supported")))
            for row in overrides
        ),
        "confidently_harmful_overrides": sum(
            int(bool(row.get("deep_override_confidently_harmful")))
            for row in overrides
        ),
        "confidence_inconclusive_overrides": sum(
            int(
                not row.get("deep_override_confidently_supported")
                and not row.get("deep_override_confidently_harmful")
            )
            for row in overrides
        ),
        "mean_elapsed_ms": _rounded_mean(
            [float(row["elapsed_ms"]) for row in rows]
        ),
        "max_elapsed_ms": round(
            max((float(row["elapsed_ms"]) for row in rows), default=0.0),
            3,
        ),
    }


def _group_by(
    rows: Iterable[dict[str, Any]],
    key: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return sorted(groups.items())


def _rounded_mean(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0, "low": 0.0, "high": 0.0}
    center = mean(values)
    if len(values) < 2:
        return {
            "samples": len(values),
            "low": round(center, 6),
            "high": round(center, 6),
        }
    standard_error = stdev(values) / math.sqrt(len(values))
    radius = _t_critical_95(len(values) - 1) * standard_error
    return {
        "samples": len(values),
        "low": round(center - radius, 6),
        "high": round(center + radius, 6),
    }


def _t_critical_95(degrees_of_freedom: int) -> float:
    table = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    if degrees_of_freedom <= len(table):
        return table[max(1, degrees_of_freedom) - 1]
    return 2.0 if degrees_of_freedom <= 120 else 1.96


if __name__ == "__main__":
    raise SystemExit(main())
