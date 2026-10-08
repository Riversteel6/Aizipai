"""Exact parallel execution for progressive response confirmations."""

from __future__ import annotations

import math
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
from typing import Any, Callable, Sequence

from ai.decision_objective import outcome_vector
from ai.dual_discard_validator import (
    _balanced_world_shards,
    _future_results_before_deadline,
    _shared_executor,
)
from ai.full_game_simulator import PublicView
from ai.ismcts import (
    ProgressiveRootISMCTSPolicy,
    RootISMCTSConfig,
    RootISMCTSPolicy,
    RootResponseCandidate,
    RootResponseCandidateStats,
    RootResponseSearchResult,
    _merge_progressive_response,
    _override_iteration_budget_incomplete,
    _paired_advantage_stats,
    _progressive_response_shortlist,
)


class ExactShardedProgressiveResponsePolicy:
    """Run every frozen progressive response stage in exact world shards."""

    name = "exact_sharded_progressive_response_v1"

    def __init__(
        self,
        base: ProgressiveRootISMCTSPolicy,
        *,
        parallel_shards: int = 5,
        parallel_workers: int = 10,
        forced_binary_confirmation: (
            Callable[
                [
                    PublicView,
                    dict[str, Any],
                    Sequence[RootResponseCandidate],
                    str,
                ],
                tuple[str, float] | None,
            ]
            | None
        ) = None,
    ) -> None:
        self.base = base
        self.parallel_shards = max(1, int(parallel_shards))
        self.parallel_workers = max(2, int(parallel_workers))
        self.forced_binary_confirmation = forced_binary_confirmation

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def search_response(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        candidates: Sequence[RootResponseCandidate],
        force_search: bool = False,
        preferred_key: str | None = None,
        absolute_deadline: float | None = None,
    ) -> RootResponseSearchResult:
        del force_search
        if not candidates:
            raise ValueError("no_progressive_response_candidate")
        self.base._ensure_coverage_capacity(len(candidates))
        preferred = preferred_key or candidates[0].key
        coverage = self._sharded_confirmation(
            view,
            rules=rules,
            candidates=candidates,
            preferred_key=preferred,
            stage=self.base.coverage,
            absolute_deadline=absolute_deadline,
        )
        self.base.last_coverage = coverage
        if not coverage.used_search:
            self._clear_after_coverage()
            return coverage

        shortlist_keys = _progressive_response_shortlist(
            coverage,
            preferred_key=preferred,
            maximum=self.base.refinement_candidates,
        )
        by_key = {candidate.key: candidate for candidate in candidates}
        shortlist = [by_key[key] for key in shortlist_keys]
        refinement = self._sharded_confirmation(
            view,
            rules=rules,
            candidates=shortlist,
            preferred_key=preferred,
            stage=self.base.refinement,
            absolute_deadline=absolute_deadline,
        )
        self.base.last_refinement = refinement
        self.base.last_discard_selection = None
        self.base.last_response_selection = None
        forced_confirmation = (
            self.forced_binary_confirmation(
                view,
                rules,
                candidates,
                preferred,
            )
            if self.forced_binary_confirmation is not None
            else None
        )
        if forced_confirmation is not None:
            challenger_key, minimum_advantage = forced_confirmation
            if (
                challenger_key in by_key
                and challenger_key != preferred
                and preferred in by_key
            ):
                confirmation = self._sharded_confirmation(
                    view,
                    rules=rules,
                    candidates=[by_key[preferred], by_key[challenger_key]],
                    preferred_key=preferred,
                    stage=self.base.response_binary_confirmation,
                    config=replace(
                        self.base.response_binary_confirmation.config,
                        minimum_confident_advantage=float(
                            minimum_advantage
                        ),
                    ),
                    absolute_deadline=absolute_deadline,
                )
                self.base.last_confirmation = confirmation
                return _merge_progressive_response(
                    coverage,
                    preferred_key=preferred,
                    refinement=refinement,
                    confirmation=confirmation,
                )
        empirical_alternative = refinement.empirical_best_key
        if (
            not refinement.used_search
            or empirical_alternative is None
            or empirical_alternative == preferred
        ):
            self.base.last_confirmation = None
            return _merge_progressive_response(
                coverage,
                preferred_key=preferred,
                refinement=refinement,
            )

        if len(shortlist) == 2:
            confirmation = self._sharded_confirmation(
                view,
                rules=rules,
                candidates=shortlist,
                preferred_key=preferred,
                stage=self.base.response_binary_confirmation,
                config=self.base.response_binary_confirmation.config,
                absolute_deadline=absolute_deadline,
            )
            self.base.last_confirmation = confirmation
            return _merge_progressive_response(
                coverage,
                preferred_key=preferred,
                refinement=refinement,
                confirmation=confirmation,
            )

        selection = self._sharded_confirmation(
            view,
            rules=rules,
            candidates=shortlist,
            preferred_key=preferred,
            stage=self.base.response_selection,
            absolute_deadline=absolute_deadline,
        )
        if _override_iteration_budget_incomplete(
            self.base.response_selection.config,
            candidate_count=len(shortlist),
            paired_determinizations=selection.paired_determinizations,
            deadline_interruptions=selection.deadline_interruptions,
        ):
            selection = replace(
                selection,
                selected_key=preferred,
                used_search=False,
                reason="root_response_selection_budget_incomplete",
                confidence_override=False,
            )
        self.base.last_response_selection = selection
        confirmation = None
        empirical_alternative = selection.empirical_best_key
        if (
            selection.used_search
            and empirical_alternative is not None
            and empirical_alternative != preferred
        ):
            confirmation = self._sharded_confirmation(
                view,
                rules=rules,
                candidates=[by_key[preferred], by_key[empirical_alternative]],
                preferred_key=preferred,
                stage=self.base.response_confirmation,
                config=self.base.response_confirmation.config,
                absolute_deadline=absolute_deadline,
            )
        self.base.last_confirmation = confirmation
        return _merge_progressive_response(
            coverage,
            preferred_key=preferred,
            refinement=refinement,
            selection=selection,
            confirmation=confirmation,
        )

    def search_hu(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        preferred_key: str = "HU",
    ) -> RootResponseSearchResult:
        return self.base.search_hu(
            view,
            rules=rules,
            preferred_key=preferred_key,
        )

    def _clear_after_coverage(self) -> None:
        self.base.last_refinement = None
        self.base.last_discard_selection = None
        self.base.last_response_selection = None
        self.base.last_confirmation = None

    def _sharded_confirmation(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        candidates: Sequence[RootResponseCandidate],
        preferred_key: str,
        stage: RootISMCTSPolicy,
        config: RootISMCTSConfig | None = None,
        absolute_deadline: float | None,
    ) -> RootResponseSearchResult:
        config = config or stage.config
        worlds = max(1, config.max_iterations // len(candidates))
        shards = _balanced_world_shards(worlds, self.parallel_shards)
        started = time.perf_counter()
        executor = _shared_executor(self.parallel_workers)
        futures = [
            executor.submit(
                _response_search_task,
                {
                    "view": view,
                    "rules": rules,
                    "candidates": tuple(candidates),
                    "preferred_key": preferred_key,
                    "base_config": config,
                    "rollout_policy_factories": stage.rollout_policy_factories,
                    "root_continuation_policy_factory": (
                        stage.root_continuation_policy_factory
                    ),
                    "worlds": amount,
                    "paired_world_offset": offset,
                    "absolute_deadline": absolute_deadline,
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
            return stage.search_response(
                view,
                rules=rules,
                candidates=candidates,
                force_search=True,
                preferred_key=preferred_key,
                absolute_deadline=absolute_deadline,
            )
        return _merge_response_confirmation_shards(
            results,
            candidates=candidates,
            preferred_key=preferred_key,
            config=config,
            expected_worlds=worlds,
            root_seat=int(view.seat),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


def _response_search_task(payload: dict[str, Any]) -> RootResponseSearchResult:
    candidates = tuple(payload["candidates"])
    worlds = int(payload["worlds"])
    config = replace(
        payload["base_config"],
        time_budget_ms=max(1, int(payload["base_config"].time_budget_ms)),
        max_iterations=worlds * len(candidates),
        max_candidates=len(candidates),
        skip_search_gap=math.inf,
        complete_first_paired_batch=True,
        record_paired_worlds=True,
        paired_world_offset=max(0, int(payload["paired_world_offset"])),
    )
    policy = RootISMCTSPolicy(
        config,
        rollout_policy_factories=payload["rollout_policy_factories"],
        root_continuation_policy_factory=payload.get(
            "root_continuation_policy_factory"
        ),
    )
    return policy.search_response(
        payload["view"],
        rules=payload["rules"],
        candidates=candidates,
        force_search=True,
        preferred_key=str(payload["preferred_key"]),
        absolute_deadline=payload.get("absolute_deadline"),
    )


def _merge_response_confirmation_shards(
    results: Sequence[RootResponseSearchResult],
    *,
    candidates: Sequence[RootResponseCandidate],
    preferred_key: str,
    config: RootISMCTSConfig,
    expected_worlds: int,
    root_seat: int,
    elapsed_ms: float,
) -> RootResponseSearchResult:
    by_key = {candidate.key: candidate for candidate in candidates}
    ordered_keys = tuple(by_key)
    worlds = sorted(
        (world for result in results for world in result.paired_worlds),
        key=lambda world: world.world_index,
    )
    expected_indices = tuple(range(max(1, int(expected_worlds))))
    actual_indices = tuple(world.world_index for world in worlds)
    reward_batches: list[dict[str, float]] = []
    outcome_batches: list[dict[str, Any]] = []
    outcomes_by_key = {key: [] for key in ordered_keys}
    complete_outcomes = True
    for world in worlds:
        outcomes = {outcome.candidate_key: outcome for outcome in world.outcomes}
        if set(outcomes) != set(ordered_keys):
            complete_outcomes = False
            continue
        reward_batches.append({key: outcomes[key].reward for key in ordered_keys})
        outcome_batches.append(
            {
                key: outcome_vector(outcomes[key], root_seat=root_seat)
                for key in ordered_keys
            }
        )
        for key in ordered_keys:
            outcomes_by_key[key].append(outcomes[key])

    aggregates: dict[str, dict[str, float | int]] = {}
    for key in ordered_keys:
        reward_sum = 0.0
        outcome_score_sum = 0.0
        signed_xi_sum = 0.0
        wins = 0
        losses = 0
        draws = 0
        for outcome in outcomes_by_key[key]:
            reward_sum += float(outcome.reward)
            if outcome.winner == root_seat:
                wins += 1
                outcome_score_sum += float(outcome.score)
                signed_xi_sum += float(outcome.total_xi)
            elif outcome.winner is None:
                draws += 1
            else:
                losses += 1
                outcome_score_sum -= float(outcome.score)
                signed_xi_sum -= float(outcome.total_xi)
        aggregates[key] = {
            "reward_sum": reward_sum,
            "outcome_score_sum": outcome_score_sum,
            "signed_xi_sum": signed_xi_sum,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }

    stats = tuple(
        RootResponseCandidateStats(
            candidate=by_key[key],
            visits=len(outcomes_by_key[key]),
            reward_sum=float(aggregates[key]["reward_sum"]),
            average_reward=(
                float(aggregates[key]["reward_sum"])
                / max(1, len(outcomes_by_key[key]))
            ),
            win_rate=(
                int(aggregates[key]["wins"])
                / max(1, len(outcomes_by_key[key]))
            ),
            wins=int(aggregates[key]["wins"]),
            losses=int(aggregates[key]["losses"]),
            draws=int(aggregates[key]["draws"]),
            loss_rate=(
                int(aggregates[key]["losses"])
                / max(1, len(outcomes_by_key[key]))
            ),
            draw_rate=(
                int(aggregates[key]["draws"])
                / max(1, len(outcomes_by_key[key]))
            ),
            mean_outcome_score=(
                float(aggregates[key]["outcome_score_sum"])
                / max(1, len(outcomes_by_key[key]))
            ),
            mean_signed_xi=(
                float(aggregates[key]["signed_xi_sum"])
                / max(1, len(outcomes_by_key[key]))
            ),
        )
        for key in ordered_keys
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
            len(outcomes_by_key[key]) == len(expected_indices)
            for key in ordered_keys
        )
        and invariant_violations == 0
        and coverage_failures == 0
        and deadline_interruptions == 0
    )
    paired_advantages = _paired_advantage_stats(
        reward_batches,
        candidate_keys=ordered_keys,
        preferred_key=preferred_key,
        outcome_batches=outcome_batches,
        objective_mode=config.objective_mode,
    )
    empirical_best = max(
        stats,
        key=lambda item: (
            item.average_reward,
            item.candidate.key == preferred_key,
            item.visits,
            item.candidate.heuristic_value,
            item.candidate.key,
        ),
    )
    selected_key = empirical_best.candidate.key if complete else preferred_key
    confidence_override = False
    reason = (
        "root_response_ismcts_completed"
        if complete
        else "sharded_response_confirmation_incomplete"
    )
    if complete and config.require_confident_override:
        selected_key = preferred_key
        if empirical_best.candidate.key == preferred_key:
            reason = "root_response_confidence_kept_preferred"
        else:
            advantage = next(
                (
                    item
                    for item in paired_advantages
                    if item.candidate_key == empirical_best.candidate.key
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
                selected_key = empirical_best.candidate.key
                confidence_override = True
                reason = "root_response_confidence_override"
            else:
                reason = "root_response_confidence_insufficient"
    return RootResponseSearchResult(
        selected_key=selected_key,
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
                    item.candidate.heuristic_value,
                ),
                reverse=True,
            )
        ),
        determinization_failures=max(
            (result.determinization_failures for result in results),
            default=0,
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
        empirical_best_key=empirical_best.candidate.key,
        confidence_override=confidence_override,
        paired_advantages=paired_advantages,
        paired_worlds=tuple(worlds),
    )
