"""Recheck real full-game discard roots with paired hidden-state rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import RootISMCTSConfig, RootISMCTSPolicy, public_view_from_dict
from ai.opponent_league import create_policy
from engine.rules import rules_for_room
from tools.recheck_response_disagreements import (
    DEFAULT_OPPONENT_PROFILES,
    _stable_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--simulations", type=int, default=12)
    parser.add_argument("--time-budget-ms", type=int, default=1_000)
    parser.add_argument("--min-confidence-pairs", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--search-mode",
        choices=("full_rollout", "immediate_response"),
        default="full_rollout",
    )
    parser.add_argument(
        "--opponent-profile",
        action="append",
        choices=DEFAULT_OPPONENT_PROFILES,
    )
    parser.add_argument("--mixed-opponents", action="store_true")
    parser.add_argument(
        "--source-selected-only",
        action="store_true",
        help=(
            "Compare only the recorded source selection against production; "
            "useful for high-depth veto audits."
        ),
    )
    parser.add_argument("--state-id", action="append")
    parser.add_argument("--public-view-id", action="append")
    parser.add_argument(
        "--compare-pair",
        action="append",
        default=[],
        metavar="PUBLIC_VIEW_ID=OLD_LABEL,NEW_LABEL",
        help="Limit one public root to an explicit two-label comparison.",
    )
    args = parser.parse_args()

    cases = _load_cases(args.input)
    if args.state_id:
        requested = set(args.state_id)
        cases = [case for case in cases if case["state_id"] in requested]
    if args.public_view_id:
        requested = set(args.public_view_id)
        cases = [
            case for case in cases
            if case.get("public_view_id") in requested
        ]
    compare_pairs = _parse_compare_pairs(args.compare_pair)
    if compare_pairs:
        if args.source_selected_only:
            parser.error("--compare-pair conflicts with --source-selected-only")
        cases = _apply_compare_pairs(cases, compare_pairs)
    selected = sorted(
        cases,
        key=lambda case: (
            float(case["heuristic_gap"]),
            str(case["state_id"]),
        ),
    )[: max(1, args.top)]
    profiles = tuple(args.opponent_profile or ("baseline",))
    tasks = [
        (
            case,
            profiles,
            max(1, args.repeats),
            args.seed,
            max(2, args.simulations),
            max(1, args.time_budget_ms),
            args.search_mode,
            max(2, args.min_confidence_pairs),
            bool(args.mixed_opponents),
            bool(args.source_selected_only),
        )
        for case in selected
    ]
    requested_workers = max(1, int(args.workers))
    effective_workers = min(
        requested_workers,
        max(1, len(tasks)),
        os.cpu_count() or 1,
    )
    if effective_workers > 1:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=get_context("spawn"),
        ) as pool:
            audited = list(
                pool.map(
                    _audit_case_task,
                    tasks,
                    chunksize=1,
                )
            )
    else:
        audited = [_audit_case_task(task) for task in tasks]
    report = {
        "ok": all(
            case["all_runs_usable"]
            and case["zero_invariant_violations"]
            and case["zero_coverage_failures"]
            for case in audited
        ),
        "source_reports": [str(path) for path in args.input],
        "available_roots": len(cases),
        "audited_cases": len(audited),
        "repeats": max(1, args.repeats),
        "simulations": max(2, args.simulations),
        "time_budget_ms": max(1, args.time_budget_ms),
        "search_mode": args.search_mode,
        "min_confidence_pairs": max(2, args.min_confidence_pairs),
        "opponent_profiles": list(profiles),
        "mixed_opponents": bool(args.mixed_opponents),
        "source_selected_only": bool(args.source_selected_only),
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "worker_model": (
            "spawn_process_executor"
            if effective_workers > 1
            else "serial"
        ),
        "stable_overrides": sum(
            int(case["stable_override_candidate"])
            for case in audited
        ),
        "promotion_gate_passes": sum(
            int(case["passes_promotion_gate"])
            for case in audited
        ),
        "rollout_coverage_failures": sum(
            int(result["rollout_coverage_failures"])
            for case in audited
            for result in case["profile_results"]
        ),
        "cases": audited,
    }
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
                    "available_roots",
                    "audited_cases",
                    "repeats",
                    "simulations",
                    "stable_overrides",
                    "promotion_gate_passes",
                    "rollout_coverage_failures",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def _audit_case_task(
    task: tuple[
        dict[str, Any],
        tuple[str, ...],
        int,
        int,
        int,
        int,
        str,
        int,
        bool,
        bool,
    ],
) -> dict[str, Any]:
    (
        case,
        profiles,
        repeats,
        base_seed,
        simulations,
        time_budget_ms,
        search_mode,
        min_confidence_pairs,
        mixed_opponents,
        source_selected_only,
    ) = task
    return _audit_case(
        case,
        profiles=profiles,
        repeats=repeats,
        base_seed=base_seed,
        simulations=simulations,
        time_budget_ms=time_budget_ms,
        search_mode=search_mode,
        min_confidence_pairs=min_confidence_pairs,
        mixed_opponents=mixed_opponents,
        source_selected_only=source_selected_only,
    )


def _load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    by_state: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report.get("rows") or []:
            for event in row.get("candidate_discard_events") or []:
                is_supported_root = bool(
                    event.get("eligible")
                    or event.get("search_attempted")
                )
                if not is_supported_root or not event.get("public_view"):
                    continue
                candidates = [
                    {
                        "label": str(candidate["label"]),
                        "heuristic_value": float(candidate["heuristic_value"]),
                    }
                    for candidate in event.get("candidates") or []
                ]
                if len(candidates) < 2:
                    continue
                ranked_priors = sorted(
                    (
                        float(candidate["heuristic_value"])
                        for candidate in candidates
                    ),
                    reverse=True,
                )
                heuristic_gap = (
                    ranked_priors[0] - ranked_priors[1]
                    if len(ranked_priors) >= 2
                    else math.inf
                )
                payload = {
                    "public_view": event["public_view"],
                    "production_label": str(event["production_label"]),
                    "candidates": candidates,
                }
                state_id = _state_id(payload)
                by_state.setdefault(
                    state_id,
                    {
                        "state_id": state_id,
                        "public_view_id": str(
                            event.get("public_view_id") or state_id
                        ),
                        "source_report": str(path),
                        "players": int(event["players"]),
                        "wildcard_enabled": bool(event["wildcard_enabled"]),
                        "seed": int(event["seed"]),
                        "candidate_seat": int(event["candidate_seat"]),
                        "dealer": int(event["dealer"]),
                        "production_label": str(event["production_label"]),
                        "source_search_selected_label": (
                            str(event["search_selected_label"])
                            if event.get("search_selected_label") is not None
                            else None
                        ),
                        "source_disagreement": bool(event.get("disagreement")),
                        "heuristic_gap": float(
                            event.get("heuristic_gap", heuristic_gap)
                        ),
                        "public_view": event["public_view"],
                        "candidates": candidates,
                    },
                )
    return list(by_state.values())


def _parse_compare_pairs(values: list[str]) -> dict[str, tuple[str, str]]:
    pairs: dict[str, tuple[str, str]] = {}
    for value in values:
        public_view_id, separator, labels_text = value.partition("=")
        labels = tuple(
            label.strip() for label in labels_text.split(",") if label.strip()
        )
        if not separator or not public_view_id.strip() or len(labels) != 2:
            raise ValueError(
                "compare_pair_format:PUBLIC_VIEW_ID=OLD_LABEL,NEW_LABEL"
            )
        if labels[0] == labels[1]:
            raise ValueError("compare_pair_requires_distinct_labels")
        view_id = public_view_id.strip()
        if view_id in pairs:
            raise ValueError(f"duplicate_compare_pair:{view_id}")
        pairs[view_id] = (labels[0], labels[1])
    return pairs


def _apply_compare_pairs(
    cases: list[dict[str, Any]],
    pairs: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    found: set[str] = set()
    for case in cases:
        public_view_id = str(case.get("public_view_id") or "")
        labels = pairs.get(public_view_id)
        if labels is None:
            continue
        available = {
            str(candidate["label"])
            for candidate in case.get("candidates") or ()
        }
        if not set(labels).issubset(available):
            raise ValueError(f"compare_pair_candidate_missing:{public_view_id}")
        if str(case["production_label"]) != labels[0]:
            raise ValueError(f"compare_pair_old_label_mismatch:{public_view_id}")
        selected.append({**case, "comparison_labels": labels})
        found.add(public_view_id)
    missing = sorted(set(pairs) - found)
    if missing:
        raise ValueError("compare_pair_public_view_missing:" + ",".join(missing))
    return selected


def _audit_case(
    case: dict[str, Any],
    *,
    profiles: tuple[str, ...],
    repeats: int,
    base_seed: int,
    simulations: int,
    time_budget_ms: int,
    search_mode: str,
    min_confidence_pairs: int,
    mixed_opponents: bool,
    source_selected_only: bool = False,
) -> dict[str, Any]:
    view = public_view_from_dict(case["public_view"])
    rules = rules_for_room(
        wildcard_enabled=bool(case["wildcard_enabled"]),
        players=int(case["players"]),
    )
    labels = [str(candidate["label"]) for candidate in case["candidates"]]
    priors = {
        str(candidate["label"]): float(candidate["heuristic_value"])
        for candidate in case["candidates"]
    }
    comparison_labels = tuple(case.get("comparison_labels") or ())
    if comparison_labels:
        labels = [label for label in labels if label in comparison_labels]
        if set(labels) != set(comparison_labels):
            raise ValueError("compare_pair_candidate_missing")
        priors = {label: priors[label] for label in labels}
    if source_selected_only:
        source_selected = str(
            case.get("source_search_selected_label") or ""
        )
        production = str(case["production_label"])
        if not source_selected or source_selected == production:
            raise ValueError(
                "source_selected_only_requires_recorded_override"
            )
        labels = [
            label
            for label in labels
            if label in {production, source_selected}
        ]
        if set(labels) != {production, source_selected}:
            raise ValueError(
                "source_selected_only_candidate_missing"
            )
        priors = {label: priors[label] for label in labels}
    profile_groups = (
        (profiles,)
        if mixed_opponents
        else tuple((profile,) for profile in profiles)
    )
    profile_results = [
        _audit_profile(
            case,
            view=view,
            rules=rules,
            labels=labels,
            priors=priors,
            profiles=profile_group,
            repeats=repeats,
            base_seed=base_seed,
            simulations=simulations,
            time_budget_ms=time_budget_ms,
            search_mode=search_mode,
            min_confidence_pairs=min_confidence_pairs,
        )
        for profile_group in profile_groups
    ]
    all_runs_usable = all(result["usable_runs"] == repeats for result in profile_results)
    zero_invariant_violations = all(
        result["rollout_invariant_violations"] == 0
        for result in profile_results
    )
    zero_coverage_failures = all(
        result["rollout_coverage_failures"] == 0
        for result in profile_results
    )
    stable_selected_label = _stable_selection(
        [
            {
                "usable_runs": result["usable_runs"],
                "stable_selected_key": result["stable_selected_label"],
            }
            for result in profile_results
        ],
        repeats=repeats,
    )
    stable_override_label = _stable_selection(
        [
            {
                "usable_runs": result["usable_runs"],
                "stable_selected_key": result["stable_override_label"],
            }
            for result in profile_results
        ],
        repeats=repeats,
    )
    latencies = sorted(
        float(run["elapsed_ms"])
        for result in profile_results
        for run in result["runs"]
    )
    p95_ms = round(_percentile(latencies, 0.95), 3)
    stable_override = stable_override_label is not None
    return {
        **case,
        "opponent_profiles": list(profiles),
        "search_mode": search_mode,
        "min_confidence_pairs": min_confidence_pairs,
        "mixed_opponents": mixed_opponents,
        "source_selected_only": source_selected_only,
        "repeats": repeats,
        "all_runs_usable": all_runs_usable,
        "zero_invariant_violations": zero_invariant_violations,
        "zero_coverage_failures": zero_coverage_failures,
        "stable_selected_label": stable_selected_label,
        "stable_override_label": stable_override_label,
        "stable_override_candidate": stable_override,
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": p95_ms,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "passes_promotion_gate": (
            stable_override
            and all_runs_usable
            and zero_invariant_violations
            and zero_coverage_failures
            and p95_ms <= 1_000.0
        ),
        "profile_results": profile_results,
    }


def _audit_profile(
    case: dict[str, Any],
    *,
    view: Any,
    rules: dict[str, Any],
    labels: list[str],
    priors: dict[str, float],
    profiles: tuple[str, ...],
    repeats: int,
    base_seed: int,
    simulations: int,
    time_budget_ms: int,
    search_mode: str,
    min_confidence_pairs: int,
) -> dict[str, Any]:
    if not profiles:
        raise ValueError("at_least_one_opponent_profile_required")
    runs: list[dict[str, Any]] = []
    for offset in range(repeats):
        seed = base_seed + offset
        search_policy = RootISMCTSPolicy(
            RootISMCTSConfig(
                time_budget_ms=time_budget_ms,
                max_iterations=simulations,
                max_candidates=max(2, len(labels)),
                rollout_max_turns=1 if search_mode == "immediate_response" else 120,
                seed=seed,
                require_confident_override=search_mode == "full_rollout",
                min_confidence_pairs=min_confidence_pairs,
            ),
            rollout_policy_factories=tuple(
                (
                    lambda profile=profile: create_policy(profile)
                )
                for profile in profiles
            ),
        )
        if search_mode == "immediate_response":
            search = search_policy.search_discard_response_risk(
                view,
                rules=rules,
                candidate_labels=labels,
                candidate_priors=priors,
                preferred_label=str(case["production_label"]),
            )
        else:
            search = search_policy.search_discard(
                view,
                rules=rules,
                candidate_labels=labels,
                candidate_priors=priors,
                force_search=True,
                paired_candidates=True,
                preferred_label=str(case["production_label"]),
            )
        candidate_rows = [candidate.to_dict() for candidate in search.candidates]
        expected_paired_worlds = (
            simulations // max(1, len(labels))
            if search_mode == "full_rollout"
            else None
        )
        complete = _search_run_complete(
            search,
            candidate_rows=candidate_rows,
            search_mode=search_mode,
            expected_paired_worlds=expected_paired_worlds,
        )
        selection_margin = (
            _selected_vs_production_adjusted_margin(
                candidate_rows,
                selected_label=search.selected_label,
                production_label=str(case["production_label"]),
            )
            if search_mode == "immediate_response"
            else _selected_vs_production_reward_margin(
                candidate_rows,
                selected_label=search.selected_label,
                production_label=str(case["production_label"]),
            )
        )
        confidence_lower_bound = (
            _selected_vs_production_confidence_lower_bound(
                [item.to_dict() for item in search.paired_advantages],
                selected_label=search.selected_label,
                production_label=str(case["production_label"]),
            )
            if search_mode == "full_rollout"
            else None
        )
        runs.append(
            {
                "seed": seed,
                "selected_label": search.selected_label,
                "empirical_best_label": search.empirical_best_label,
                "used_search": search.used_search,
                "complete": complete,
                "reason": search.reason,
                "simulations": search.simulations,
                "expected_paired_worlds": expected_paired_worlds,
                "paired_determinizations": search.paired_determinizations,
                "deadline_interruptions": search.deadline_interruptions,
                "rollout_invariant_violations": search.rollout_invariant_violations,
                "rollout_violations": list(search.rollout_violations),
                "rollout_coverage_failures": search.rollout_coverage_failures,
                "rollout_coverage_reasons": list(search.rollout_coverage_reasons),
                "elapsed_ms": round(search.elapsed_ms, 3),
                "overrides_production": (
                    search.used_search
                    and search.selected_label != case["production_label"]
                ),
                "confidence_override": search.confidence_override,
                "paired_advantages": [
                    item.to_dict()
                    for item in search.paired_advantages
                ],
                "selected_vs_production_margin": selection_margin,
                "selected_vs_production_confidence_lower_bound": (
                    confidence_lower_bound
                ),
                "strictly_beats_production": (
                    complete
                    and search.selected_label != case["production_label"]
                    and (
                        (
                            search_mode == "full_rollout"
                            and search.confidence_override
                            and confidence_lower_bound is not None
                            and confidence_lower_bound > 1e-9
                        )
                        or (
                            search_mode == "immediate_response"
                            and selection_margin is not None
                            and selection_margin > 1e-9
                        )
                    )
                ),
                "candidates": candidate_rows,
            }
        )
    selections = Counter(
        str(run["selected_label"])
        for run in runs
        if run["used_search"]
    )
    stable_selected_label = (
        next(iter(selections))
        if sum(selections.values()) == repeats and len(selections) == 1
        else None
    )
    stable_override_label = _strict_stable_override_label(
        runs,
        stable_selected_label=stable_selected_label,
        production_label=str(case["production_label"]),
    )
    return {
        "profile": (
            profiles[0]
            if len(profiles) == 1
            else "mixed:" + ",".join(profiles)
        ),
        "profiles": list(profiles),
        "search_mode": search_mode,
        "repeats": repeats,
        "usable_runs": sum(int(run["complete"]) for run in runs),
        "override_runs": sum(int(run["overrides_production"]) for run in runs),
        "rollout_invariant_violations": sum(
            int(run["rollout_invariant_violations"])
            for run in runs
        ),
        "rollout_coverage_failures": sum(
            int(run["rollout_coverage_failures"])
            for run in runs
        ),
        "stable_selected_label": stable_selected_label,
        "stable_override_label": stable_override_label,
        "selection_counts": dict(sorted(selections.items())),
        "runs": runs,
    }


def _search_run_complete(
    search: Any,
    *,
    candidate_rows: list[dict[str, Any]],
    search_mode: str,
    expected_paired_worlds: int | None,
) -> bool:
    structurally_complete = (
        bool(search.used_search)
        and int(search.deadline_interruptions) == 0
        and int(search.rollout_invariant_violations) == 0
        and int(search.rollout_coverage_failures) == 0
    )
    if not structurally_complete:
        return False
    if search_mode != "full_rollout":
        return True
    if expected_paired_worlds is None or expected_paired_worlds < 1:
        return False
    return (
        int(search.paired_determinizations)
        == expected_paired_worlds
        and len(candidate_rows) >= 2
        and all(
            int(candidate.get("visits") or 0)
            == expected_paired_worlds
            for candidate in candidate_rows
        )
    )


def _selected_vs_production_reward_margin(
    candidates: list[dict[str, Any]],
    *,
    selected_label: str | None,
    production_label: str,
) -> float | None:
    by_label = {
        str(candidate.get("label")): candidate
        for candidate in candidates
        if candidate.get("label") is not None
    }
    selected = by_label.get(str(selected_label))
    production = by_label.get(production_label)
    if (
        selected is None
        or production is None
        or int(selected.get("visits") or 0) <= 0
        or int(production.get("visits") or 0) <= 0
    ):
        return None
    return round(
        float(selected.get("average_reward") or 0.0)
        - float(production.get("average_reward") or 0.0),
        9,
    )


def _selected_vs_production_adjusted_margin(
    candidates: list[dict[str, Any]],
    *,
    selected_label: str | None,
    production_label: str,
) -> float | None:
    by_label = {
        str(candidate.get("label")): candidate
        for candidate in candidates
        if candidate.get("label") is not None
    }
    selected = by_label.get(str(selected_label))
    production = by_label.get(production_label)
    if (
        selected is None
        or production is None
        or int(selected.get("samples") or 0) <= 0
        or int(production.get("samples") or 0) <= 0
    ):
        return None
    return round(
        float(selected.get("adjusted_value") or 0.0)
        - float(production.get("adjusted_value") or 0.0),
        9,
    )


def _selected_vs_production_confidence_lower_bound(
    paired_advantages: list[dict[str, Any]],
    *,
    selected_label: str | None,
    production_label: str,
) -> float | None:
    if selected_label is None or selected_label == production_label:
        return None
    row = next(
        (
            item
            for item in paired_advantages
            if item.get("candidate_key") == selected_label
            and item.get("preferred_key") == production_label
        ),
        None,
    )
    if row is None or row.get("lower_confidence_bound") is None:
        return None
    return round(float(row["lower_confidence_bound"]), 9)


def _strict_stable_override_label(
    runs: list[dict[str, Any]],
    *,
    stable_selected_label: str | None,
    production_label: str,
) -> str | None:
    if (
        not runs
        or stable_selected_label is None
        or stable_selected_label == production_label
        or not all(run.get("strictly_beats_production") is True for run in runs)
    ):
        return None
    return stable_selected_label


def _state_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


if __name__ == "__main__":
    raise SystemExit(main())
