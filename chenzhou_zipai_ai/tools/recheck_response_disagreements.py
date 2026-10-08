"""Repeat response-search disagreements across stronger multi-seed rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import (
    RootISMCTSConfig,
    RootISMCTSPolicy,
    RootResponseCandidate,
    public_view_from_dict,
)
from ai.opponent_league import POLICY_FACTORIES, create_policy
from engine.rules import rules_for_room


DEFAULT_OPPONENT_PROFILES = (
    "baseline",
    "information_set_search",
    "aggressive_meld",
    "conservative_search",
    "defensive_search",
    "red_black_search",
    "independent_balanced",
    "independent_pressure",
    "independent_denial",
    "independent_fast_rollout",
    "independent_fast_pressure",
    "independent_fast_denial",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--time-budget-ms", type=int, default=10_000)
    parser.add_argument("--min-confidence-pairs", type=int, default=6)
    parser.add_argument(
        "--opponent-profile",
        action="append",
        choices=DEFAULT_OPPONENT_PROFILES,
    )
    parser.add_argument("--mixed-opponents", action="store_true")
    parser.add_argument("--include-all-eligible", action="store_true")
    parser.add_argument("--state-id", action="append")
    args = parser.parse_args()

    cases = _load_cases(
        args.input,
        include_all_eligible=bool(args.include_all_eligible),
    )
    if args.state_id:
        requested = set(args.state_id)
        cases = [case for case in cases if case["state_id"] in requested]
    opponent_profiles = tuple(args.opponent_profile or ("baseline",))
    selected = sorted(
        cases,
        key=lambda case: (
            float(case["search_advantage"]),
            str(case["state_id"]),
        ),
        reverse=True,
    )[: max(1, args.top)]
    audited = [
        _audit_case(
            case,
            repeats=max(1, args.repeats),
            base_seed=args.seed,
            simulations=max(2, args.simulations),
            time_budget_ms=max(1, args.time_budget_ms),
            opponent_profiles=opponent_profiles,
            min_confidence_pairs=max(2, args.min_confidence_pairs),
            mixed_opponents=bool(args.mixed_opponents),
        )
        for case in selected
    ]
    report = {
        "ok": all(
            case["all_runs_usable"]
            and case["zero_invariant_violations"]
            and case["zero_coverage_failures"]
            for case in audited
        ),
        "source_reports": [str(path) for path in args.input],
        "available_disagreements": len(cases),
        "audited_cases": len(audited),
        "repeats": max(1, args.repeats),
        "simulations": max(2, args.simulations),
        "time_budget_ms": max(1, args.time_budget_ms),
        "min_confidence_pairs": max(2, args.min_confidence_pairs),
        "opponent_profiles": list(opponent_profiles),
        "mixed_opponents": bool(args.mixed_opponents),
        "include_all_eligible": bool(args.include_all_eligible),
        "stable_disagreements": sum(
            int(case["stable_disagreement_across_profiles"])
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
                    "available_disagreements",
                    "audited_cases",
                    "repeats",
                    "simulations",
                    "stable_disagreements",
                    "rollout_coverage_failures",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def _load_cases(
    paths: list[Path],
    *,
    include_all_eligible: bool = False,
) -> list[dict[str, Any]]:
    by_state: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report.get("rows") or []:
            for event in row.get("candidate_response_events") or []:
                if not event.get("public_view"):
                    continue
                if not include_all_eligible and not event.get("disagreement"):
                    continue
                production = _candidate_stats(event, str(event["production_key"]))
                selected = _candidate_stats(event, str(event["search_selected_key"]))
                if production is None or selected is None:
                    continue
                state_id = _state_id(event)
                by_state.setdefault(
                    state_id,
                    {
                        "state_id": state_id,
                        "source_report": str(path),
                        "players": int(event["players"]),
                        "wildcard_enabled": bool(event["wildcard_enabled"]),
                        "response_type": str(event["response_type"]),
                        "pending_card": str(event["pending_card"]),
                        "production_key": str(event["production_key"]),
                        "original_search_key": str(event["search_selected_key"]),
                        "search_advantage": round(
                            float(selected["average_reward"])
                            - float(production["average_reward"]),
                            6,
                        ),
                        "public_view": event["public_view"],
                        "candidates": [
                            _candidate_payload(candidate)
                            for candidate in event.get("candidates") or []
                        ],
                    },
                )
    return list(by_state.values())


def _audit_case(
    case: dict[str, Any],
    *,
    repeats: int,
    base_seed: int,
    simulations: int,
    time_budget_ms: int,
    opponent_profiles: tuple[str, ...],
    min_confidence_pairs: int,
    mixed_opponents: bool,
) -> dict[str, Any]:
    view = public_view_from_dict(case["public_view"])
    rules = rules_for_room(
        wildcard_enabled=bool(case["wildcard_enabled"]),
        players=int(case["players"]),
    )
    candidates = tuple(
        RootResponseCandidate(
            key=str(candidate["key"]),
            action_type=str(candidate["action_type"]),
            heuristic_value=float(candidate["heuristic_value"]),
            option_id=candidate.get("option_id"),
            consumed_from_hand=tuple(candidate.get("consumed_from_hand") or []),
            meld_groups=tuple(
                tuple(group)
                for group in candidate.get("meld_groups") or []
            ),
            followup_discard=candidate.get("followup_discard"),
        )
        for candidate in case["candidates"]
    )
    profile_groups = (
        (opponent_profiles,)
        if mixed_opponents
        else tuple((profile,) for profile in opponent_profiles)
    )
    profile_results = [
        _audit_profile(
            case,
            view=view,
            rules=rules,
            candidates=candidates,
            profiles=profile_group,
            repeats=repeats,
            base_seed=base_seed,
            simulations=simulations,
            time_budget_ms=time_budget_ms,
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
    stable_selected_key = _stable_selection(
        profile_results,
        repeats=repeats,
    )
    stable_confidence_override_key = (
        stable_selected_key
        if stable_selected_key is not None
        and stable_selected_key != case["production_key"]
        and all(
            result["confidence_override_runs"] == repeats
            for result in profile_results
        )
        else None
    )
    return {
        **case,
        "repeats": repeats,
        "opponent_profiles": list(opponent_profiles),
        "mixed_opponents": mixed_opponents,
        "all_runs_usable": all_runs_usable,
        "zero_invariant_violations": zero_invariant_violations,
        "zero_coverage_failures": zero_coverage_failures,
        "stable_selected_key": stable_selected_key,
        "stable_confidence_override_key": stable_confidence_override_key,
        "stable_disagreement_across_profiles": (
            stable_confidence_override_key is not None
        ),
        "profile_results": profile_results,
    }


def _audit_profile(
    case: dict[str, Any],
    *,
    view: Any,
    rules: dict[str, Any],
    candidates: tuple[RootResponseCandidate, ...],
    profiles: tuple[str, ...],
    repeats: int,
    base_seed: int,
    simulations: int,
    time_budget_ms: int,
    min_confidence_pairs: int,
) -> dict[str, Any]:
    if not profiles:
        raise ValueError("at_least_one_opponent_profile_required")
    unknown = [profile for profile in profiles if profile not in POLICY_FACTORIES]
    if unknown:
        raise ValueError(f"unknown opponent profile: {unknown[0]}")
    runs: list[dict[str, Any]] = []
    for offset in range(repeats):
        seed = base_seed + offset
        search = RootISMCTSPolicy(
            RootISMCTSConfig(
                time_budget_ms=time_budget_ms,
                max_iterations=simulations,
                max_candidates=max(2, len(candidates)),
                rollout_max_turns=120,
                seed=seed,
                require_confident_override=True,
                min_confidence_pairs=min_confidence_pairs,
            ),
            rollout_policy_factories=tuple(
                (
                    lambda profile=profile: create_policy(profile)
                )
                for profile in profiles
            ),
        ).search_response(
            view,
            rules=rules,
            candidates=candidates,
            force_search=True,
            preferred_key=str(case["production_key"]),
        )
        runs.append(
            {
                "seed": seed,
                "selected_key": search.selected_key,
                "empirical_best_key": search.empirical_best_key,
                "used_search": search.used_search,
                "reason": search.reason,
                "simulations": search.simulations,
                "paired_determinizations": search.paired_determinizations,
                "deadline_interruptions": search.deadline_interruptions,
                "rollout_invariant_violations": search.rollout_invariant_violations,
                "rollout_violations": list(search.rollout_violations),
                "rollout_coverage_failures": search.rollout_coverage_failures,
                "rollout_coverage_reasons": list(search.rollout_coverage_reasons),
                "elapsed_ms": round(search.elapsed_ms, 3),
                "disagrees_with_production": (
                    search.used_search
                    and search.selected_key != case["production_key"]
                ),
                "confidence_override": search.confidence_override,
                "paired_advantages": [
                    advantage.to_dict()
                    for advantage in search.paired_advantages
                ],
                "candidates": [
                    candidate.to_dict()
                    for candidate in search.candidates
                ],
            }
        )
    selections = Counter(
        str(run["selected_key"])
        for run in runs
        if run["used_search"]
    )
    stable_selected_key = (
        next(iter(selections))
        if sum(selections.values()) == repeats and len(selections) == 1
        else None
    )
    return {
        "profile": (
            profiles[0]
            if len(profiles) == 1
            else "mixed:" + ",".join(profiles)
        ),
        "profiles": list(profiles),
        "repeats": repeats,
        "usable_runs": sum(int(run["used_search"]) for run in runs),
        "disagreement_runs": sum(
            int(run["disagrees_with_production"])
            for run in runs
        ),
        "confidence_override_runs": sum(
            int(run["confidence_override"])
            for run in runs
        ),
        "original_selection_runs": sum(
            int(run["used_search"] and run["selected_key"] == case["original_search_key"])
            for run in runs
        ),
        "rollout_invariant_violations": sum(
            int(run["rollout_invariant_violations"])
            for run in runs
        ),
        "rollout_coverage_failures": sum(
            int(run["rollout_coverage_failures"])
            for run in runs
        ),
        "stable_selected_key": stable_selected_key,
        "selection_counts": dict(sorted(selections.items())),
        "runs": runs,
    }


def _stable_selection(
    profile_results: list[dict[str, Any]],
    *,
    repeats: int,
) -> str | None:
    if not profile_results:
        return None
    stable_keys = [result.get("stable_selected_key") for result in profile_results]
    if (
        any(result.get("usable_runs") != repeats for result in profile_results)
        or any(key is None for key in stable_keys)
        or len(set(stable_keys)) != 1
    ):
        return None
    return str(stable_keys[0])


def _candidate_stats(
    event: dict[str, Any],
    key: str,
) -> dict[str, Any] | None:
    return next(
        (
            candidate
            for candidate in event.get("candidates") or []
            if candidate.get("key") == key
        ),
        None,
    )


def _candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate.get(key)
        for key in (
            "key",
            "action_type",
            "heuristic_value",
            "option_id",
            "consumed_from_hand",
            "meld_groups",
            "followup_discard",
        )
    }


def _state_id(event: dict[str, Any]) -> str:
    payload = {
        "public_view": event["public_view"],
        "production_key": event["production_key"],
        "candidates": [
            _candidate_payload(candidate)
            for candidate in event.get("candidates") or []
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
