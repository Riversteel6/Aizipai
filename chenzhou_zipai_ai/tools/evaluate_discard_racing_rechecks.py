"""Evaluate exhaustive discard racing against deep counterfactual labels."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


APP_ROOT = Path(__file__).resolve().parents[1]
while str(APP_ROOT) in sys.path:
    sys.path.remove(str(APP_ROOT))

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context


sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import (
    RootISMCTSConfig,
    RootISMCTSPolicy,
    _combined_discard_confirmations,
    public_view_from_dict,
)
from ai.dual_discard_validator import (
    DualBatchDiscardValidator,
    DualDiscardValidatorConfig,
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

_SEARCH_EXECUTOR: ProcessPoolExecutor | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-report", type=Path, required=True)
    parser.add_argument("--override-manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument(
        "--evidence-aggregation",
        choices=("deepest", "pooled"),
        default="pooled",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--search-workers",
        type=int,
        default=1,
        help="Persistent workers used inside one paired root search.",
    )
    parser.add_argument(
        "--strategy",
        choices=(
            "racing24_split",
            "adaptive16_96",
            "split24_pair",
            "split24_dual192",
            "successive24_48",
        ),
        default="racing24_split",
    )
    parser.add_argument(
        "--preferred-source",
        choices=("production", "current"),
        default="production",
        help="Action that challengers must confidently beat.",
    )
    parser.add_argument("--coverage-worlds", type=int, default=24)
    parser.add_argument("--pair-confirmation-worlds", type=int, default=384)
    parser.add_argument("--dual-confirmation-worlds", type=int, default=192)
    parser.add_argument("--selection-worlds", type=int, default=48)
    parser.add_argument("--selection-challengers", type=int, default=5)
    parser.add_argument("--split-require-confidence", action="store_true")
    parser.add_argument(
        "--split-selection-rule",
        choices=("empirical", "confidence", "guarded_consensus"),
        default="empirical",
    )
    parser.add_argument("--confirmation-worlds-small", type=int, default=192)
    parser.add_argument("--confirmation-worlds-large", type=int, default=96)
    parser.add_argument("--small-candidate-threshold", type=int, default=6)
    parser.add_argument("--minimum-confident-advantage", type=float, default=0.02)
    parser.add_argument("--time-budget-ms", type=int, default=30_000)
    parser.add_argument("--state-hash", action="append", default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers > 1 and args.search_workers > 1:
        parser.error("--workers and --search-workers cannot both exceed one")

    cases = _load_cases(
        args.league_report,
        args.override_manifest,
        evidence_paths=args.evidence,
        evidence_aggregation=args.evidence_aggregation,
    )
    requested_hashes = {str(value) for value in args.state_hash if str(value)}
    if requested_hashes:
        cases = [
            case
            for case in cases
            if str(case["state_before_hash"]) in requested_hashes
        ]
    if args.limit is not None:
        cases = cases[: max(1, args.limit)]
    config = {
        "strategy": args.strategy,
        "evidence_aggregation": args.evidence_aggregation,
        "preferred_source": args.preferred_source,
        "coverage_worlds": max(1, args.coverage_worlds),
        "pair_confirmation_worlds": max(2, args.pair_confirmation_worlds),
        "dual_confirmation_worlds": max(2, args.dual_confirmation_worlds),
        "selection_worlds": max(2, args.selection_worlds),
        "selection_challengers": max(1, args.selection_challengers),
        "split_require_confidence": bool(args.split_require_confidence),
        "split_selection_rule": (
            "confidence"
            if args.split_require_confidence
            else args.split_selection_rule
        ),
        "confirmation_worlds_small": max(1, args.confirmation_worlds_small),
        "confirmation_worlds_large": max(1, args.confirmation_worlds_large),
        "small_candidate_threshold": max(2, args.small_candidate_threshold),
        "minimum_confident_advantage": max(
            0.0,
            args.minimum_confident_advantage,
        ),
        "time_budget_ms": max(1, args.time_budget_ms),
        "search_workers": max(1, args.search_workers),
    }
    for case in cases:
        case["config"] = config
    global _SEARCH_EXECUTOR
    if args.search_workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.search_workers,
            mp_context=get_context("spawn"),
        ) as executor:
            _SEARCH_EXECUTOR = executor
            try:
                rows = [_evaluate_case(case) for case in cases]
            finally:
                _SEARCH_EXECUTOR = None
    elif args.workers > 1 and len(cases) > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=get_context("spawn"),
        ) as executor:
            rows = list(executor.map(_evaluate_case, cases))
    else:
        rows = [_evaluate_case(case) for case in cases]
    report = _summarize(
        rows,
        league_report=args.league_report,
        override_manifest=args.override_manifest,
        evidence_paths=args.evidence,
        config=config,
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
                    "errors",
                    "incomplete_searches",
                    "candidate_changes",
                    "improved_vs_current",
                    "regressed_vs_current",
                    "equal_vs_current",
                    "current_mean_regret",
                    "candidate_mean_regret",
                    "mean_regret_reduction",
                    "maximum_elapsed_ms",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def _load_cases(
    league_report_path: Path,
    override_manifest_path: Path,
    *,
    evidence_paths: Iterable[Path],
    evidence_aggregation: str = "pooled",
) -> list[dict[str, Any]]:
    report = json.loads(league_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(override_manifest_path.read_text(encoding="utf-8"))
    event_rows = {
        (
            int(row["seed"]),
            int(row["candidate_seat"]),
            int(row["dealer"]),
            tuple(sorted(str(value) for value in row.get("opponents") or ())),
        ): row
        for row in report.get("rows") or ()
    }
    evidence_by_state = (
        _load_pooled_evidence(evidence_paths)
        if evidence_aggregation == "pooled"
        else _load_deepest_evidence(evidence_paths)
    )
    cases = []
    for entry in manifest.get("rows") or ():
        if str(entry.get("kind") or entry.get("phase")) != "discard":
            continue
        state_hash = str(entry["state_before_hash"])
        evidence = evidence_by_state.get(state_hash)
        if evidence is None:
            raise ValueError(f"deep_evidence_missing:{state_hash}")
        identity = (
            int(entry["seed"]),
            int(entry["candidate_seat"]),
            int(entry["dealer"]),
            tuple(
                sorted(str(value) for value in entry.get("opponents") or ())
            ),
        )
        league_row = event_rows.get(identity)
        if league_row is None:
            raise ValueError(f"league_row_missing:{identity}")
        event_index = int(entry["event_index"])
        events = league_row.get("candidate_discard_events") or ()
        if event_index >= len(events):
            raise ValueError(
                f"discard_event_missing:{identity}:index={event_index}"
            )
        event = dict(events[event_index])
        cases.append(
            {
                "state_before_hash": state_hash,
                "game": {
                    "players": int(event["players"]),
                    "wildcard_enabled": bool(event["wildcard_enabled"]),
                    "seed": int(entry["seed"]),
                    "candidate_seat": int(entry["candidate_seat"]),
                    "dealer": int(entry["dealer"]),
                    "opponents": list(entry.get("opponents") or ()),
                    "event_index": event_index,
                },
                "event": event,
                "deep": evidence,
            }
        )
    return sorted(cases, key=lambda case: str(case["state_before_hash"]))


def _load_deepest_evidence(
    paths: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    deepest: dict[str, dict[str, Any]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("complete") or row.get("phase") != "discard":
                    continue
                state_hash = str(row["state_before_hash"])
                previous = deepest.get(state_hash)
                if (
                    previous is None
                    or int(row["completed_paired_worlds"])
                    > int(previous["completed_paired_worlds"])
                ):
                    deepest[state_hash] = row
    return deepest


def _load_pooled_evidence(
    paths: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    by_state_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("complete") or row.get("phase") != "discard":
                    continue
                state_hash = str(row["state_before_hash"])
                audit_seed = int(row["audit_seed"])
                key = (state_hash, audit_seed)
                previous = by_state_seed.get(key)
                if (
                    previous is None
                    or int(row["completed_paired_worlds"])
                    > int(previous["completed_paired_worlds"])
                ):
                    by_state_seed[key] = row

    grouped: dict[str, list[dict[str, Any]]] = {}
    for (state_hash, _audit_seed), row in by_state_seed.items():
        grouped.setdefault(state_hash, []).append(row)

    pooled: dict[str, dict[str, Any]] = {}
    for state_hash, rows in grouped.items():
        rewards: dict[str, list[float]] = {}
        for row in rows:
            for world in row.get("paired_worlds") or ():
                for outcome in world.get("outcomes") or ():
                    label = str(
                        outcome.get("candidate_key")
                        or outcome.get("label")
                        or ""
                    ).removeprefix("DISCARD:")
                    if label:
                        rewards.setdefault(label, []).append(
                            float(outcome.get("reward") or 0.0)
                        )
        if not rewards:
            continue
        sample_counts = {len(values) for values in rewards.values()}
        if len(sample_counts) != 1:
            raise ValueError(
                f"pooled_evidence_candidate_coverage_mismatch:{state_hash}"
            )
        candidate_stats = [
            {
                "label": label,
                "average_reward": mean(values),
            }
            for label, values in rewards.items()
        ]
        candidate_stats.sort(
            key=lambda item: (
                float(item["average_reward"]),
                str(item["label"]),
            ),
            reverse=True,
        )
        pooled[state_hash] = {
            **max(
                rows,
                key=lambda row: int(row["completed_paired_worlds"]),
            ),
            "audit_seed": None,
            "completed_paired_worlds": sample_counts.pop(),
            "candidate_stats": candidate_stats,
            "best_key": "DISCARD:" + str(candidate_stats[0]["label"]),
            "pooled_audit_seeds": sorted(
                int(row["audit_seed"])
                for row in rows
            ),
        }
    return pooled


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    event = case["event"]
    config = case["config"]
    error = None
    coverage = None
    confirmation = None
    combined = None
    try:
        labels = [str(item["label"]) for item in event["candidates"]]
        priors = {
            str(item["label"]): float(item.get("heuristic_value") or 0.0)
            for item in event["candidates"]
        }
        preferred = (
            str(event["search_selected_label"])
            if config["preferred_source"] == "current"
            else str(event["production_label"])
        )
        view = public_view_from_dict(event["public_view"])
        rules = rules_for_room(
            wildcard_enabled=bool(case["game"]["wildcard_enabled"]),
            players=int(case["game"]["players"]),
        )
        if config["strategy"] == "adaptive16_96":
            coverage, confirmation, combined = _adaptive_search(
                view,
                rules=rules,
                labels=labels,
                priors=priors,
                preferred=preferred,
                config=config,
            )
        elif config["strategy"] == "split24_pair":
            coverage, confirmation, combined = _split_pair_search(
                view,
                rules=rules,
                labels=labels,
                priors=priors,
                preferred=preferred,
                config=config,
            )
        elif config["strategy"] == "split24_dual192":
            coverage, confirmation, combined = _split_dual_search(
                view,
                rules=rules,
                labels=labels,
                priors=priors,
                preferred=preferred,
                config=config,
            )
        elif config["strategy"] == "successive24_48":
            coverage, confirmation, combined = _successive_search(
                view,
                rules=rules,
                labels=labels,
                priors=priors,
                preferred=preferred,
                config=config,
            )
        else:
            coverage, confirmation, combined = _racing24_split_search(
                view,
                rules=rules,
                labels=labels,
                priors=priors,
                preferred=preferred,
                config=config,
            )
    except Exception as exc:  # pragma: no cover - summarized as report health
        error = f"{type(exc).__name__}:{exc}"

    deep = case["deep"]
    rewards = {
        str(item["label"]): float(item.get("average_reward") or 0.0)
        for item in deep.get("candidate_stats") or ()
    }
    production = str(event["production_label"])
    current = str(event["search_selected_label"])
    preferred = current if config["preferred_source"] == "current" else production
    candidate = str(combined.selected_label) if combined is not None else preferred
    best = str(deep["best_key"]).removeprefix("DISCARD:")
    best_reward = max(rewards.values())

    def regret(label: str) -> float | None:
        reward = rewards.get(label)
        return best_reward - reward if reward is not None else None

    small = len(event.get("candidates") or ()) <= int(
        config["small_candidate_threshold"]
    )
    if config["strategy"] == "adaptive16_96":
        expected_coverage = 96 if small else 16
        expected_confirmation = 0 if small else 96
    elif config["strategy"] == "split24_pair":
        expected_coverage = int(config["coverage_worlds"])
        expected_confirmation = int(config["pair_confirmation_worlds"])
    elif config["strategy"] == "split24_dual192":
        expected_coverage = int(config["coverage_worlds"])
        expected_confirmation = int(config["dual_confirmation_worlds"])
    elif config["strategy"] == "successive24_48":
        expected_coverage = int(config["coverage_worlds"])
        expected_confirmation = int(config["selection_worlds"])
    else:
        expected_coverage = int(config["coverage_worlds"])
        expected_confirmation = (
            int(config["confirmation_worlds_small"])
            if small
            else int(config["confirmation_worlds_large"])
        )
    complete = (
        coverage is not None
        and coverage.paired_determinizations == expected_coverage
        and coverage.deadline_interruptions == 0
        and (
            expected_confirmation == 0
            or (
                confirmation is not None
                and confirmation.paired_determinizations
                == expected_confirmation
                and confirmation.deadline_interruptions == 0
            )
        )
        and (
            config["strategy"] != "split24_dual192"
            or bool(combined and combined.used_search)
        )
    )
    return {
        "state_before_hash": case["state_before_hash"],
        "game": case["game"],
        "deep_worlds": int(deep["completed_paired_worlds"]),
        "legal_candidate_count": len(event.get("candidates") or ()),
        "production_label": production,
        "preferred_label": preferred,
        "current_label": current,
        "candidate_label": candidate,
        "deep_best_label": best,
        "production_regret": regret(production),
        "current_regret": regret(current),
        "candidate_regret": regret(candidate),
        "current_reward": rewards.get(current),
        "candidate_reward": rewards.get(candidate),
        "candidate_changed": candidate != current,
        "complete": complete,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "coverage": _stage_summary(coverage),
        "confirmation": _stage_summary(confirmation),
        "combined": _stage_summary(combined),
        "error": error,
    }


def _racing24_split_search(
    view: Any,
    *,
    rules: dict[str, Any],
    labels: list[str],
    priors: dict[str, float],
    preferred: str,
    config: dict[str, Any],
) -> tuple[Any, Any, Any]:
    coverage = _search(
        view,
        rules=rules,
        labels=labels,
        priors=priors,
        preferred=preferred,
        worlds=int(config["coverage_worlds"]),
        seed=20260726,
        time_budget_ms=int(config["time_budget_ms"]),
        require_confident_override=False,
        minimum_confident_advantage=float(
            config["minimum_confident_advantage"]
        ),
    )
    alternatives = [
        item.label
        for item in coverage.candidates
        if item.label != preferred
    ][:2]
    confirmation_worlds = (
        int(config["confirmation_worlds_small"])
        if len(labels) <= int(config["small_candidate_threshold"])
        else int(config["confirmation_worlds_large"])
    )
    confirmation = _search(
        view,
        rules=rules,
        labels=[preferred, *alternatives],
        priors={
            item.label: float(item.average_reward)
            for item in coverage.candidates
            if item.label == preferred or item.label in alternatives
        },
        preferred=preferred,
        worlds=confirmation_worlds,
        seed=20260728,
        time_budget_ms=int(config["time_budget_ms"]),
        require_confident_override=True,
        minimum_confident_advantage=float(
            config["minimum_confident_advantage"]
        ),
    )
    combined = _combined_discard_confirmations(
        coverage,
        confirmation,
        preferred_label=preferred,
        alternative_labels=alternatives,
        minimum_samples=int(config["coverage_worlds"])
        + confirmation_worlds,
        minimum_advantage=float(config["minimum_confident_advantage"]),
        familywise_comparisons=max(1, len(labels) - 1),
    )
    return coverage, confirmation, combined


def _adaptive_search(
    view: Any,
    *,
    rules: dict[str, Any],
    labels: list[str],
    priors: dict[str, float],
    preferred: str,
    config: dict[str, Any],
) -> tuple[Any, Any | None, Any]:
    minimum_advantage = float(config["minimum_confident_advantage"])
    time_budget_ms = int(config["time_budget_ms"])
    if len(labels) <= int(config["small_candidate_threshold"]):
        exhaustive = _search(
            view,
            rules=rules,
            labels=labels,
            priors=priors,
            preferred=preferred,
            worlds=96,
            seed=20260730,
            time_budget_ms=time_budget_ms,
            require_confident_override=True,
            minimum_confident_advantage=minimum_advantage,
        )
        return exhaustive, None, exhaustive
    breadth = _search(
        view,
        rules=rules,
        labels=labels,
        priors=priors,
        preferred=preferred,
        worlds=16,
        seed=20260730,
        time_budget_ms=time_budget_ms,
        require_confident_override=False,
        minimum_confident_advantage=minimum_advantage,
    )
    alternatives = [
        item.label
        for item in breadth.candidates
        if item.label != preferred
    ][:4]
    confirmation = _search(
        view,
        rules=rules,
        labels=[preferred, *alternatives],
        priors={
            item.label: float(item.average_reward)
            for item in breadth.candidates
            if item.label == preferred or item.label in alternatives
        },
        preferred=preferred,
        worlds=96,
        seed=20260728,
        time_budget_ms=time_budget_ms,
        require_confident_override=True,
        minimum_confident_advantage=minimum_advantage,
    )
    combined = _combined_discard_confirmations(
        breadth,
        confirmation,
        preferred_label=preferred,
        alternative_labels=alternatives,
        minimum_samples=112,
        minimum_advantage=minimum_advantage,
        familywise_comparisons=max(1, len(labels) - 1),
    )
    return breadth, confirmation, combined


def _split_pair_search(
    view: Any,
    *,
    rules: dict[str, Any],
    labels: list[str],
    priors: dict[str, float],
    preferred: str,
    config: dict[str, Any],
) -> tuple[Any, Any, Any]:
    screening = _search(
        view,
        rules=rules,
        labels=labels,
        priors=priors,
        preferred=preferred,
        worlds=int(config["coverage_worlds"]),
        seed=20260730,
        time_budget_ms=int(config["time_budget_ms"]),
        require_confident_override=False,
        minimum_confident_advantage=0.0,
    )
    challenger = next(
        item.label
        for item in screening.candidates
        if item.label != preferred
    )
    confirmation = _search(
        view,
        rules=rules,
        labels=[preferred, challenger],
        priors=priors,
        preferred=preferred,
        worlds=int(config["pair_confirmation_worlds"]),
        seed=20260732,
        time_budget_ms=int(config["time_budget_ms"]),
        require_confident_override=False,
        minimum_confident_advantage=float(
            config["minimum_confident_advantage"]
        ),
    )
    confirmation = _apply_split_selection_rule(
        screening,
        confirmation,
        labels=labels,
        preferred=preferred,
        challenger=challenger,
        config=config,
    )
    return screening, confirmation, confirmation


def _apply_split_selection_rule(
    screening: Any,
    confirmation: Any,
    *,
    labels: list[str],
    preferred: str,
    challenger: str,
    config: dict[str, Any],
) -> Any:
    rule = str(config["split_selection_rule"])
    if rule == "empirical":
        return confirmation
    advantage = next(
        item
        for item in confirmation.paired_advantages
        if item.preferred_key == preferred
        and item.candidate_key == challenger
    )
    minimum = float(config["minimum_confident_advantage"])
    strict = (
        advantage.lower_confidence_bound is not None
        and advantage.lower_confidence_bound > minimum
    )
    ranked = {
        item.label: (index + 1, float(item.average_reward))
        for index, item in enumerate(screening.candidates)
    }
    challenger_rank, challenger_reward = ranked[challenger]
    preferred_rank, preferred_reward = ranked[preferred]
    consensus = (
        challenger_rank == 1
        and preferred_rank > math.ceil(len(labels) / 2)
        and challenger_reward - preferred_reward >= 0.15
        and advantage.mean_delta > 0.0
    )
    override = strict or (
        rule == "guarded_consensus"
        and consensus
    )
    return replace(
        confirmation,
        selected_label=challenger if override else preferred,
        reason=(
            "split_pair_guarded_override"
            if override
            else "split_pair_guarded_kept_preferred"
        ),
        confidence_override=strict,
    )


def _split_dual_search(
    view: Any,
    *,
    rules: dict[str, Any],
    labels: list[str],
    priors: dict[str, float],
    preferred: str,
    config: dict[str, Any],
) -> tuple[Any, Any, Any]:
    if _SEARCH_EXECUTOR is not None:
        result = DualBatchDiscardValidator(
            rollout_policy_factories=ROLLOUT_FACTORIES,
            config=DualDiscardValidatorConfig(
                coverage_worlds=int(config["coverage_worlds"]),
                confirmation_worlds=int(
                    config["dual_confirmation_worlds"]
                ),
                parallel_workers=int(config["search_workers"]),
                time_budget_ms=int(config["time_budget_ms"]),
            ),
            executor=_SEARCH_EXECUTOR,
        ).search_discard(
            view,
            rules=rules,
            candidate_labels=labels,
            candidate_priors=priors,
            preferred_label=preferred,
        )
        return (
            result.coverage,
            result.second_confirmation,
            result.combined,
        )
    screening = _search(
        view,
        rules=rules,
        labels=labels,
        priors=priors,
        preferred=preferred,
        worlds=int(config["coverage_worlds"]),
        seed=20260730,
        time_budget_ms=int(config["time_budget_ms"]),
        require_confident_override=False,
        minimum_confident_advantage=0.0,
    )
    challenger = next(
        item.label
        for item in screening.candidates
        if item.label != preferred
    )
    confirmation_worlds = int(config["dual_confirmation_worlds"])
    first = _search(
        view,
        rules=rules,
        labels=[preferred, challenger],
        priors=priors,
        preferred=preferred,
        worlds=confirmation_worlds,
        seed=20260732,
        time_budget_ms=int(config["time_budget_ms"]),
        require_confident_override=False,
        minimum_confident_advantage=0.0,
    )
    if not _confirmation_supports_challenger(
        first,
        challenger=challenger,
    ):
        kept = replace(
            first,
            selected_label=preferred,
            reason="split_pair_dual_first_batch_rejected_challenger",
            confidence_override=False,
        )
        return screening, first, kept
    second = _search(
        view,
        rules=rules,
        labels=[preferred, challenger],
        priors=priors,
        preferred=preferred,
        worlds=confirmation_worlds,
        seed=20260733,
        time_budget_ms=int(config["time_budget_ms"]),
        require_confident_override=False,
        minimum_confident_advantage=0.0,
    )
    combined = _combined_discard_confirmations(
        first,
        second,
        preferred_label=preferred,
        alternative_labels=(challenger,),
        minimum_samples=confirmation_worlds * 2,
        minimum_advantage=0.0,
        familywise_comparisons=1,
    )
    override = _dual_confirmation_override(
        first,
        second,
        combined,
        challenger=challenger,
    )
    return screening, second, replace(
        combined,
        selected_label=challenger if override else preferred,
        reason=(
            "split_pair_dual_repeatable_confident_override"
            if override
            else "split_pair_dual_kept_preferred"
        ),
        confidence_override=override,
    )


def _confirmation_supports_challenger(
    confirmation: Any,
    *,
    challenger: str,
) -> bool:
    advantage = next(
        item
        for item in confirmation.paired_advantages
        if item.candidate_key == challenger
    )
    return advantage.mean_delta > 0.0


def _dual_confirmation_override(
    first: Any,
    second: Any,
    combined: Any,
    *,
    challenger: str,
) -> bool:
    first_advantage = next(
        item
        for item in first.paired_advantages
        if item.candidate_key == challenger
    )
    second_advantage = next(
        item
        for item in second.paired_advantages
        if item.candidate_key == challenger
    )
    return (
        first_advantage.mean_delta > 0.0
        and second_advantage.mean_delta > 0.0
        and combined.selected_label == challenger
    )


def _successive_search(
    view: Any,
    *,
    rules: dict[str, Any],
    labels: list[str],
    priors: dict[str, float],
    preferred: str,
    config: dict[str, Any],
) -> tuple[Any, Any, Any]:
    screening = _search(
        view,
        rules=rules,
        labels=labels,
        priors=priors,
        preferred=preferred,
        worlds=int(config["coverage_worlds"]),
        seed=20260730,
        time_budget_ms=int(config["time_budget_ms"]),
        require_confident_override=False,
        minimum_confident_advantage=0.0,
    )
    challengers = [
        item.label
        for item in screening.candidates
        if item.label != preferred
    ][: int(config["selection_challengers"])]
    selection = _search(
        view,
        rules=rules,
        labels=[preferred, *challengers],
        priors=priors,
        preferred=preferred,
        worlds=int(config["selection_worlds"]),
        seed=20260731,
        time_budget_ms=int(config["time_budget_ms"]),
        require_confident_override=False,
        minimum_confident_advantage=0.0,
    )
    return screening, selection, selection


def _search(
    view: Any,
    *,
    rules: dict[str, Any],
    labels: list[str],
    priors: dict[str, float],
    preferred: str,
    worlds: int,
    seed: int,
    time_budget_ms: int,
    require_confident_override: bool,
    minimum_confident_advantage: float,
) -> Any:
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=time_budget_ms,
            max_iterations=worlds * len(labels),
            max_candidates=len(labels),
            skip_search_gap=math.inf,
            rollout_max_turns=120,
            require_confident_override=require_confident_override,
            min_confidence_pairs=min(32, worlds),
            minimum_confident_advantage=minimum_confident_advantage,
            complete_first_paired_batch=True,
            require_complete_iteration_budget_for_override=True,
            seed=seed,
        ),
        rollout_policy_factories=ROLLOUT_FACTORIES,
    )
    return policy.search_discard(
        view,
        rules=rules,
        candidate_labels=labels,
        candidate_priors=priors,
        force_search=True,
        paired_candidates=True,
        preferred_label=preferred,
    )


def _stage_summary(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "selected_label": str(result.selected_label),
        "empirical_best_label": result.empirical_best_label,
        "reason": str(result.reason),
        "used_search": bool(result.used_search),
        "simulations": int(result.simulations),
        "paired_determinizations": int(result.paired_determinizations),
        "deadline_interruptions": int(result.deadline_interruptions),
        "elapsed_ms": round(float(result.elapsed_ms), 3),
        "confidence_override": bool(result.confidence_override),
        "candidates": [
            {
                "label": item.label,
                "visits": int(item.visits),
                "average_reward": round(float(item.average_reward), 6),
            }
            for item in result.candidates
        ],
        "paired_advantages": [
            item.to_dict() for item in result.paired_advantages
        ],
    }


def _summarize(
    rows: list[dict[str, Any]],
    *,
    league_report: Path,
    override_manifest: Path,
    evidence_paths: Iterable[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors = [row for row in rows if row["error"] is not None]
    incomplete = [row for row in rows if not row["complete"]]
    comparable = [
        row
        for row in rows
        if row["current_regret"] is not None
        and row["candidate_regret"] is not None
    ]
    improved = [
        row
        for row in comparable
        if float(row["candidate_regret"]) < float(row["current_regret"]) - 1e-12
    ]
    regressed = [
        row
        for row in comparable
        if float(row["candidate_regret"]) > float(row["current_regret"]) + 1e-12
    ]
    equal = [
        row
        for row in comparable
        if abs(float(row["candidate_regret"]) - float(row["current_regret"]))
        <= 1e-12
    ]
    current_regrets = [float(row["current_regret"]) for row in comparable]
    candidate_regrets = [float(row["candidate_regret"]) for row in comparable]
    return {
        "ok": not errors and not incomplete and len(comparable) == len(rows),
        "schema_version": "discard-racing-recheck-evaluation-v1",
        "league_report": str(league_report),
        "override_manifest": str(override_manifest),
        "evidence": [str(path) for path in evidence_paths],
        "config": config,
        "states": len(rows),
        "errors": len(errors),
        "incomplete_searches": len(incomplete),
        "candidate_changes": sum(row["candidate_changed"] for row in rows),
        "improved_vs_current": len(improved),
        "regressed_vs_current": len(regressed),
        "equal_vs_current": len(equal),
        "current_mean_regret": round(mean(current_regrets), 6)
        if current_regrets
        else None,
        "candidate_mean_regret": round(mean(candidate_regrets), 6)
        if candidate_regrets
        else None,
        "mean_regret_reduction": round(
            mean(current_regrets) - mean(candidate_regrets),
            6,
        )
        if current_regrets
        else None,
        "maximum_elapsed_ms": max(
            (float(row["elapsed_ms"]) for row in rows),
            default=0.0,
        ),
        "mean_elapsed_ms": round(
            mean(float(row["elapsed_ms"]) for row in rows),
            3,
        )
        if rows
        else 0.0,
        "improved_states": [
            row["state_before_hash"] for row in improved
        ],
        "regressed_states": [
            row["state_before_hash"] for row in regressed
        ],
        "rows": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
