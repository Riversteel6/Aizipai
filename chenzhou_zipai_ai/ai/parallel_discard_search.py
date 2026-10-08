"""Exact world-sharded execution for paired root discard stages."""

from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Sequence

from ai.decision_objective import outcome_vector
from ai.dual_discard_validator import (
    _balanced_world_shards,
    _future_results_before_deadline,
    _root_search_task,
    _shared_executor,
)
from ai.full_game_simulator import PublicView, _discardable_labels
from ai.ismcts import (
    RootCandidateStats,
    RootISMCTSConfig,
    RootISMCTSPolicy,
    RootSearchResult,
    _override_iteration_budget_incomplete,
    _paired_advantage_stats,
)


class ExactShardedRootDiscardPolicy:
    """Run one paired root-search stage on exact deterministic world shards."""

    name = "exact_sharded_root_discard_v1"

    def __init__(
        self,
        base: RootISMCTSPolicy,
        *,
        parallel_shards: int = 8,
        parallel_workers: int = 12,
    ) -> None:
        self.base = base
        self.parallel_shards = max(1, int(parallel_shards))
        self.parallel_workers = max(2, int(parallel_workers))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def search_discard(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        candidate_labels: Sequence[str] | None = None,
        candidate_priors: dict[str, float] | None = None,
        force_search: bool = False,
        paired_candidates: bool = False,
        preferred_label: str | None = None,
        absolute_deadline: float | None = None,
    ) -> RootSearchResult:
        prepared = _prepare_stage_candidates(
            self.base.config,
            view,
            candidate_labels=candidate_labels,
            candidate_priors=candidate_priors,
            preferred_label=preferred_label,
        )
        if (
            not force_search
            or not paired_candidates
            or prepared is None
            or self.parallel_shards <= 1
        ):
            return self.base.search_discard(
                view,
                rules=rules,
                candidate_labels=candidate_labels,
                candidate_priors=candidate_priors,
                force_search=force_search,
                paired_candidates=paired_candidates,
                preferred_label=preferred_label,
                absolute_deadline=absolute_deadline,
            )

        labels, priors, preferred = prepared
        worlds = max(1, self.base.config.max_iterations) // len(labels)
        if worlds < 2:
            return self.base.search_discard(
                view,
                rules=rules,
                candidate_labels=candidate_labels,
                candidate_priors=candidate_priors,
                force_search=force_search,
                paired_candidates=paired_candidates,
                preferred_label=preferred_label,
                absolute_deadline=absolute_deadline,
            )

        base_offset = max(0, int(self.base.config.paired_world_offset))
        shards = _balanced_world_shards(worlds, self.parallel_shards)
        started = time.perf_counter()
        executor = _shared_executor(self.parallel_workers)
        futures = [
            executor.submit(
                _root_search_task,
                {
                    "view": view,
                    "rules": rules,
                    "labels": labels,
                    "priors": priors,
                    "preferred": preferred,
                    "worlds": amount,
                    "seed": self.base.config.seed,
                    "time_budget_ms": self.base.config.time_budget_ms,
                    "absolute_deadline": absolute_deadline,
                    "rollout_max_turns": self.base.config.rollout_max_turns,
                    "objective_mode": self.base.config.objective_mode.value,
                    "objective_version": self.base.config.objective_version,
                    "rollout_policy_factories": (
                        self.base.rollout_policy_factories
                    ),
                    "root_continuation_policy_factory": (
                        self.base.root_continuation_policy_factory
                    ),
                    "record_paired_worlds": True,
                    "paired_world_offset": base_offset + offset,
                },
            )
            for offset, amount in shards
        ]
        try:
            results = _future_results_before_deadline(
                futures,
                absolute_deadline=absolute_deadline,
            )
        except FutureTimeoutError:
            return self.base.search_discard(
                view,
                rules=rules,
                candidate_labels=candidate_labels,
                candidate_priors=candidate_priors,
                force_search=force_search,
                paired_candidates=paired_candidates,
                preferred_label=preferred_label,
                absolute_deadline=absolute_deadline,
            )
        return _merge_exact_discard_shards(
            results,
            labels=labels,
            priors=priors,
            preferred=preferred,
            config=self.base.config,
            expected_worlds=worlds,
            world_offset=base_offset,
            root_seat=int(view.seat),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


def _prepare_stage_candidates(
    config: RootISMCTSConfig,
    view: PublicView,
    *,
    candidate_labels: Sequence[str] | None,
    candidate_priors: dict[str, float] | None,
    preferred_label: str | None,
) -> tuple[list[str], dict[str, float], str] | None:
    if candidate_priors is None:
        return None
    legal = set(_discardable_labels(view.hand))
    labels = list(dict.fromkeys(candidate_labels or sorted(legal)))
    labels = [label for label in labels if label in legal]
    if not labels or any(label not in candidate_priors for label in labels):
        return None
    preferred = preferred_label or labels[0]
    if preferred not in labels:
        return None
    priors = {label: float(candidate_priors[label]) for label in labels}
    ranked = sorted(
        labels,
        key=lambda label: (priors[label], label),
        reverse=True,
    )
    shortlist = ranked[: max(1, config.max_candidates)]
    if preferred not in shortlist:
        shortlist = [
            preferred,
            *(label for label in ranked if label != preferred),
        ][: max(1, config.max_candidates)]
    if len(shortlist) < 2:
        return None
    return shortlist, {label: priors[label] for label in shortlist}, preferred


def _merge_exact_discard_shards(
    results: Sequence[RootSearchResult],
    *,
    labels: Sequence[str],
    priors: dict[str, float],
    preferred: str,
    config: RootISMCTSConfig,
    expected_worlds: int,
    world_offset: int,
    root_seat: int,
    elapsed_ms: float,
) -> RootSearchResult:
    ordered_labels = tuple(labels)
    worlds = sorted(
        (world for result in results for world in result.paired_worlds),
        key=lambda world: world.world_index,
    )
    expected_indices = tuple(
        range(world_offset, world_offset + max(1, int(expected_worlds)))
    )
    actual_indices = tuple(world.world_index for world in worlds)
    reward_batches: list[dict[str, float]] = []
    outcome_batches: list[dict[str, Any]] = []
    outcomes_by_label = {label: [] for label in ordered_labels}
    complete_outcomes = True
    for world in worlds:
        by_label = {
            outcome.candidate_key: outcome
            for outcome in world.outcomes
        }
        if set(by_label) != set(ordered_labels):
            complete_outcomes = False
            continue
        reward_batches.append(
            {label: by_label[label].reward for label in ordered_labels}
        )
        outcome_batches.append(
            {
                label: outcome_vector(by_label[label], root_seat=root_seat)
                for label in ordered_labels
            }
        )
        for label in ordered_labels:
            outcomes_by_label[label].append(by_label[label])

    stats = tuple(
        _candidate_stats(
            label,
            outcomes_by_label[label],
            heuristic_value=priors[label],
            root_seat=root_seat,
        )
        for label in ordered_labels
    )
    invariant_violations = sum(
        result.rollout_invariant_violations for result in results
    )
    coverage_failures = sum(
        result.rollout_coverage_failures for result in results
    )
    deadline_interruptions = sum(
        result.deadline_interruptions for result in results
    )
    complete = (
        bool(results)
        and all(result.used_search for result in results)
        and actual_indices == expected_indices
        and complete_outcomes
        and all(
            len(outcomes_by_label[label]) == len(expected_indices)
            for label in ordered_labels
        )
        and invariant_violations == 0
        and coverage_failures == 0
        and deadline_interruptions == 0
    )
    paired_advantages = _paired_advantage_stats(
        reward_batches,
        candidate_keys=ordered_labels,
        preferred_key=preferred,
        outcome_batches=outcome_batches,
        objective_mode=config.objective_mode,
        familywise_comparisons=(
            config.confidence_familywise_comparisons
        ),
    )
    empirical_best = max(
        stats,
        key=lambda item: (
            item.average_reward,
            item.label == preferred,
            item.visits,
            item.heuristic_value,
            item.label,
        ),
    )
    selected = empirical_best.label if complete else preferred
    confidence_override = False
    reason = (
        "root_discard_ismcts_completed"
        if complete
        else "exact_sharded_discard_incomplete"
    )
    if complete and config.require_confident_override:
        selected = preferred
        if _override_iteration_budget_incomplete(
            config,
            candidate_count=len(ordered_labels),
            paired_determinizations=len(worlds),
            deadline_interruptions=deadline_interruptions,
        ):
            reason = "root_discard_confidence_budget_incomplete"
        elif empirical_best.label == preferred:
            reason = "root_discard_confidence_kept_preferred"
        else:
            advantage = next(
                (
                    item
                    for item in paired_advantages
                    if item.candidate_key == empirical_best.label
                ),
                None,
            )
            if (
                advantage is not None
                and advantage.samples >= max(2, config.min_confidence_pairs)
                and advantage.lower_confidence_bound is not None
                and advantage.lower_confidence_bound
                > config.minimum_confident_advantage
            ):
                selected = empirical_best.label
                confidence_override = True
                reason = "root_discard_confidence_override"
            else:
                reason = "root_discard_confidence_insufficient"

    return RootSearchResult(
        selected_label=selected,
        used_search=complete,
        reason=reason,
        simulations=sum(len(world.outcomes) for world in worlds),
        elapsed_ms=elapsed_ms,
        candidates=tuple(
            sorted(
                stats,
                key=lambda item: (
                    item.average_reward,
                    item.visits,
                    item.heuristic_value,
                ),
                reverse=True,
            )
        ),
        determinization_failures=sum(
            result.determinization_failures for result in results
        ),
        paired_determinizations=len(worlds),
        deadline_interruptions=deadline_interruptions,
        rollout_invariant_violations=invariant_violations,
        rollout_violations=tuple(
            dict.fromkeys(
                violation
                for result in results
                for violation in result.rollout_violations
            )
        ),
        rollout_coverage_failures=coverage_failures,
        rollout_coverage_reasons=tuple(
            dict.fromkeys(
                reason
                for result in results
                for reason in result.rollout_coverage_reasons
            )
        ),
        empirical_best_label=empirical_best.label,
        confidence_override=confidence_override,
        paired_advantages=paired_advantages,
        paired_worlds=(
            tuple(worlds) if config.record_paired_worlds else ()
        ),
    )


def _candidate_stats(
    label: str,
    outcomes: Sequence[Any],
    *,
    heuristic_value: float,
    root_seat: int,
) -> RootCandidateStats:
    reward_sum = sum(float(outcome.reward) for outcome in outcomes)
    wins = sum(int(outcome.winner == root_seat) for outcome in outcomes)
    losses = sum(
        int(outcome.winner is not None and outcome.winner != root_seat)
        for outcome in outcomes
    )
    draws = sum(int(outcome.winner is None) for outcome in outcomes)
    outcome_score_sum = sum(
        float(outcome.score)
        if outcome.winner == root_seat
        else -float(outcome.score)
        if outcome.winner is not None
        else 0.0
        for outcome in outcomes
    )
    signed_xi_sum = sum(
        float(outcome.total_xi)
        if outcome.winner == root_seat
        else -float(outcome.total_xi)
        if outcome.winner is not None
        else 0.0
        for outcome in outcomes
    )
    visits = len(outcomes)
    return RootCandidateStats(
        label=label,
        visits=visits,
        reward_sum=reward_sum,
        average_reward=reward_sum / max(1, visits),
        win_rate=wins / max(1, visits),
        heuristic_value=float(heuristic_value),
        wins=wins,
        losses=losses,
        draws=draws,
        loss_rate=losses / max(1, visits),
        draw_rate=draws / max(1, visits),
        mean_outcome_score=outcome_score_sum / max(1, visits),
        mean_signed_xi=signed_xi_sum / max(1, visits),
    )
