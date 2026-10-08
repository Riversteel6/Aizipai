"""Parallel, repeatable validation for uncertain discard overrides."""

from __future__ import annotations

import atexit
import math
import os
import threading
import time
from concurrent.futures import (
    ALL_COMPLETED,
    Future,
    ProcessPoolExecutor,
    TimeoutError as FutureTimeoutError,
    wait,
)
from dataclasses import dataclass, replace
from multiprocessing import get_context
from typing import Any, Callable, Sequence

from ai.decision_objective import OBJECTIVE_VERSION, ObjectiveMode, outcome_vector
from ai.full_game_simulator import PublicView, SimulationPolicy
from ai.opponent_context import (
    opponent_context_features,
    opponent_context_interactions,
)
from ai.ismcts import (
    RootISMCTSConfig,
    RootCandidateStats,
    RootISMCTSPolicy,
    RootSearchResult,
    _combined_discard_confirmations,
    _paired_advantage_stats,
)


@dataclass(frozen=True)
class DualDiscardValidatorConfig:
    coverage_worlds: int = 24
    coverage_parallel_shards: int = 1
    confirmation_worlds: int = 192
    reconfirmation_worlds: int = 128
    confirmation_challengers: int = 1
    confirmation_prior_challengers: int = 0
    confirmation_familywise_comparisons: int | None = None
    reconfirmation_challengers: int = 1
    parallel_workers: int = 4
    confirmation_parallel_shards: int = 1
    time_budget_ms: int = 30_000
    rollout_max_turns: int = 120
    coverage_seed: int = 20260730
    first_confirmation_seed: int = 20260732
    second_confirmation_seed: int = 20260733
    repeatable_mean_override_threshold: float | None = None
    evidence_calibrator: DualDiscardEvidenceCalibratorConfig | None = None
    adaptive_confirmation_worlds_per_second: float | None = None
    adaptive_confirmation_min_worlds: int = 32
    adaptive_confirmation_guard_ms: int = 250
    objective_mode: ObjectiveMode = ObjectiveMode.SCORE_FIRST
    objective_version: str = OBJECTIVE_VERSION


@dataclass(frozen=True)
class DualDiscardEvidenceCalibratorConfig:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    decision_margin: float = 0.0
    trained_confirmation_worlds: int | None = None
    minimum_confirmation_fraction: float = 1.0


@dataclass(frozen=True)
class DualDiscardChallengerEvidence:
    challenger_label: str
    confidence_override: bool
    first_confirmation: RootSearchResult
    second_confirmation: RootSearchResult
    combined: RootSearchResult
    override_basis: str | None = None
    calibrated_advantage: float | None = None


@dataclass(frozen=True)
class DualDiscardValidationResult:
    selected_label: str
    preferred_label: str
    challenger_label: str
    confidence_override: bool
    elapsed_ms: float
    coverage: RootSearchResult
    first_confirmation: RootSearchResult
    second_confirmation: RootSearchResult
    combined: RootSearchResult
    expected_coverage_worlds: int
    expected_confirmation_worlds: int
    expected_candidate_count: int
    minimum_confirmation_fraction: float = 1.0
    challenger_evidence: tuple[DualDiscardChallengerEvidence, ...] = ()

    @property
    def selected_challenger_evidence(
        self,
    ) -> DualDiscardChallengerEvidence | None:
        return next(
            (
                evidence
                for evidence in self.challenger_evidence
                if evidence.challenger_label == self.challenger_label
            ),
            None,
        )

    @property
    def usable_confident_override(self) -> bool:
        evidence = self.selected_challenger_evidence
        return bool(
            self.confidence_override
            and evidence is not None
            and evidence.confidence_override
            and _confirmation_pair_usable(
                evidence.first_confirmation,
                evidence.second_confirmation,
                expected_worlds=self.expected_confirmation_worlds,
                minimum_fraction=self.minimum_confirmation_fraction,
            )
        )

    @property
    def complete(self) -> bool:
        confirmation_pairs = (
            tuple(
                (
                    evidence.first_confirmation,
                    evidence.second_confirmation,
                )
                for evidence in self.challenger_evidence
            )
            or ((self.first_confirmation, self.second_confirmation),)
        )
        return (
            self.coverage.used_search
            and self.coverage.deadline_interruptions == 0
            and self.coverage.rollout_invariant_violations == 0
            and self.coverage.rollout_coverage_failures == 0
            and self.coverage.paired_determinizations
            == self.expected_coverage_worlds
            and len(self.coverage.candidates)
            == self.expected_candidate_count
            and all(
                candidate.visits == self.expected_coverage_worlds
                for candidate in self.coverage.candidates
            )
            and all(
                _confirmation_pair_complete(
                    first,
                    second,
                    expected_worlds=self.expected_confirmation_worlds,
                )
                for first, second in confirmation_pairs
            )
        )


@dataclass(frozen=True)
class DualDiscardCoverage:
    view: PublicView
    rules: dict[str, Any]
    labels: tuple[str, ...]
    candidate_priors: dict[str, float]
    coverage: RootSearchResult
    executor: ProcessPoolExecutor
    elapsed_ms: float


_SHARED_EXECUTORS: dict[int, ProcessPoolExecutor] = {}
_SHARED_EXECUTORS_LOCK = threading.Lock()
_NUMERIC_WORKER_THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _configure_numeric_worker_threads() -> None:
    # Search parallelism comes from processes. Letting every worker create a
    # second CPU-sized thread pool exhausts memory and makes live latency worse.
    for name in _NUMERIC_WORKER_THREAD_ENV:
        os.environ[name] = "1"


def _executor_capacity(workers: int) -> int:
    resolved = max(2, int(workers))
    # The frozen live policy uses 20 workers for action evaluation and 12 for
    # confirmation, sequentially. One 20-process pool serves both workloads.
    return 20 if resolved in {12, 20} else resolved


def close_shared_dual_discard_executors(*, wait: bool = True) -> None:
    with _SHARED_EXECUTORS_LOCK:
        executors = tuple(_SHARED_EXECUTORS.values())
        _SHARED_EXECUTORS.clear()
    for executor in dict.fromkeys(executors):
        try:
            executor.shutdown(wait=wait, cancel_futures=True)
        except Exception:
            pass


atexit.register(close_shared_dual_discard_executors)


def _shared_executor(workers: int) -> ProcessPoolExecutor:
    capacity = _executor_capacity(workers)
    _configure_numeric_worker_threads()
    with _SHARED_EXECUTORS_LOCK:
        executor = _SHARED_EXECUTORS.get(capacity)
        if executor is None or bool(getattr(executor, "_broken", False)):
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            executor = ProcessPoolExecutor(
                max_workers=capacity,
                mp_context=get_context("spawn"),
            )
            _SHARED_EXECUTORS[capacity] = executor
        return executor


def _prewarm_worker(delay_seconds: float) -> int:
    if delay_seconds > 0.0:
        time.sleep(delay_seconds)
    return os.getpid()


def _future_results_before_deadline(
    futures: Sequence[Future],
    *,
    absolute_deadline: float | None,
) -> list[Any]:
    """Resolve completed futures without ever extending a foreground deadline."""

    ordered = tuple(futures)
    if not ordered:
        return []
    timeout = None
    if absolute_deadline is not None:
        timeout = max(0.0, absolute_deadline - time.perf_counter())
    if not all(hasattr(future, "_condition") for future in ordered):
        results: list[Any] = []
        try:
            for index, future in enumerate(ordered):
                remaining = None
                if absolute_deadline is not None:
                    remaining = absolute_deadline - time.perf_counter()
                    if remaining <= 0.0:
                        raise FutureTimeoutError("future_deadline_exhausted")
                results.append(future.result(timeout=remaining))
        except Exception:
            for pending in ordered[len(results):]:
                pending.cancel()
            raise
        return results
    _done, pending = wait(
        ordered,
        timeout=timeout,
        return_when=ALL_COMPLETED,
    )
    if pending:
        for future in pending:
            future.cancel()
        raise FutureTimeoutError("future_deadline_exhausted")
    return [future.result(timeout=0.0) for future in ordered]


def prewarm_shared_executor(
    workers: int,
    *,
    task_delay_seconds: float = 0.4,
) -> dict[str, int]:
    resolved = max(2, int(workers))
    executor = _shared_executor(resolved)
    futures = [
        executor.submit(
            _prewarm_worker,
            max(0.0, float(task_delay_seconds)),
        )
        for _ in range(resolved)
    ]
    results = _future_results_before_deadline(
        futures,
        absolute_deadline=(
            time.perf_counter()
            + max(5.0, float(task_delay_seconds) * 4.0)
        ),
    )
    process_ids = {int(result) for result in results}
    return {
        "workers": resolved,
        "tasks": len(futures),
        "processes_seen": len(process_ids),
    }


class DualBatchDiscardValidator:
    """All-action screening followed by two independent paired confirmations."""

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        config: DualDiscardValidatorConfig | None = None,
        executor: ProcessPoolExecutor | None = None,
    ) -> None:
        factories = tuple(rollout_policy_factories)
        if not factories:
            raise ValueError("dual_discard_validator_requires_rollout_policies")
        self.rollout_policy_factories = factories
        self.config = config or DualDiscardValidatorConfig()
        self.executor = executor

    def search_discard(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        candidate_labels: Sequence[str],
        candidate_priors: dict[str, float],
        preferred_label: str,
        absolute_deadline: float | None = None,
    ) -> DualDiscardValidationResult:
        deadline = _validation_deadline(
            absolute_deadline,
            time_budget_ms=self.config.time_budget_ms,
        )
        screened = self.screen_discard(
            view,
            rules=rules,
            candidate_labels=candidate_labels,
            candidate_priors=candidate_priors,
            absolute_deadline=deadline,
        )
        if not screened.coverage.used_search or _deadline_reached(deadline):
            return _incomplete_validation(
                screened,
                preferred_label=preferred_label,
                expected_coverage_worlds=self.config.coverage_worlds,
                confirmation_worlds=self.config.confirmation_worlds,
                reason="dual_validation_deadline_after_coverage",
            )
        return self.confirm_discard(
            screened,
            preferred_label=preferred_label,
            absolute_deadline=deadline,
        )

    def screen_discard(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        candidate_labels: Sequence[str],
        candidate_priors: dict[str, float],
        preferred_label: str | None = None,
        absolute_deadline: float | None = None,
    ) -> DualDiscardCoverage:
        started = time.perf_counter()
        labels = list(dict.fromkeys(str(label) for label in candidate_labels))
        if len(labels) < 2:
            raise ValueError("dual_discard_requires_two_candidates")
        if any(label not in candidate_priors for label in labels):
            raise ValueError("dual_discard_candidate_prior_missing")
        preferred = str(preferred_label or labels[0])
        if preferred not in labels:
            raise ValueError("dual_discard_preferred_candidate_missing")
        executor = self.executor or _shared_executor(
            self.config.parallel_workers
        )
        coverage = self._parallel_coverage(
            executor,
            view,
            rules=rules,
            labels=labels,
            priors=candidate_priors,
            preferred=preferred,
            absolute_deadline=absolute_deadline,
        )
        return DualDiscardCoverage(
            view=view,
            rules=dict(rules),
            labels=tuple(labels),
            candidate_priors=dict(candidate_priors),
            coverage=coverage,
            executor=executor,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def confirm_discard(
        self,
        screened: DualDiscardCoverage,
        *,
        preferred_label: str,
        confirmation_worlds: int | None = None,
        challenger_limit: int | None = None,
        confirmation_shards: int | None = None,
        familywise_comparisons: int | None = None,
        time_budget_ms: int | None = None,
        absolute_deadline: float | None = None,
    ) -> DualDiscardValidationResult:
        started = time.perf_counter()
        labels = list(screened.labels)
        if preferred_label not in labels:
            raise ValueError("dual_discard_preferred_candidate_missing")
        maximum_confirmation_worlds = max(
            2,
            int(
                confirmation_worlds
                if confirmation_worlds is not None
                else self.config.confirmation_worlds
            ),
        )
        resolved_confirmation_worlds = _adaptive_confirmation_worlds(
            maximum_worlds=maximum_confirmation_worlds,
            minimum_worlds=self.config.adaptive_confirmation_min_worlds,
            worlds_per_second=(
                self.config.adaptive_confirmation_worlds_per_second
                if confirmation_worlds is None
                else None
            ),
            guard_ms=self.config.adaptive_confirmation_guard_ms,
            absolute_deadline=absolute_deadline,
        )
        coverage = screened.coverage
        resolved_challenger_limit = max(
            1,
            int(
                challenger_limit
                if challenger_limit is not None
                else self.config.confirmation_challengers
            ),
        )
        challengers = _select_challenger_labels(
            coverage.candidates,
            preferred_label=preferred_label,
            limit=resolved_challenger_limit,
            prior_slots=self.config.confirmation_prior_challengers,
        )
        if not challengers:
            raise ValueError("dual_discard_challenger_missing")
        if _deadline_reached(absolute_deadline):
            return _incomplete_validation(
                screened,
                preferred_label=preferred_label,
                challenger_label=challengers[0],
                expected_coverage_worlds=self.config.coverage_worlds,
                confirmation_worlds=resolved_confirmation_worlds,
                reason="dual_confirmation_deadline_before_start",
            )
        resolved_time_budget_ms = _bounded_time_budget_ms(
            time_budget_ms,
            default_ms=self.config.time_budget_ms,
            absolute_deadline=absolute_deadline,
        )
        shared_payload = {
            "view": screened.view,
            "rules": screened.rules,
            "priors": screened.candidate_priors,
            "preferred": preferred_label,
            "worlds": resolved_confirmation_worlds,
            "time_budget_ms": resolved_time_budget_ms,
            "absolute_deadline": absolute_deadline,
            "rollout_max_turns": self.config.rollout_max_turns,
            "objective_mode": self.config.objective_mode.value,
            "objective_version": self.config.objective_version,
            "rollout_policy_factories": self.rollout_policy_factories,
        }
        confirmations = self._run_confirmations(
            screened.executor,
            shared_payload=shared_payload,
            preferred_label=preferred_label,
            challengers=challengers,
            shard_count=confirmation_shards,
        )
        resolved_familywise_comparisons = (
            int(familywise_comparisons)
            if familywise_comparisons is not None
            else int(self.config.confirmation_familywise_comparisons)
            if self.config.confirmation_familywise_comparisons is not None
            else len(confirmations)
        )
        if resolved_familywise_comparisons < len(confirmations):
            raise ValueError("dual_discard_familywise_comparisons_too_small")
        raw_evidence = tuple(
            _challenger_evidence(
                preferred_label=preferred_label,
                challenger=challenger,
                first=first,
                second=second,
                confirmation_worlds=resolved_confirmation_worlds,
                repeatable_mean_override_threshold=(
                    self.config.repeatable_mean_override_threshold
                ),
                familywise_comparisons=resolved_familywise_comparisons,
            )
            for challenger, first, second in confirmations
        )
        evidence = tuple(
            _calibrate_challenger_evidence(
                item,
                coverage=coverage,
                view=screened.view,
                preferred_label=preferred_label,
                expected_worlds=resolved_confirmation_worlds,
                calibrator=self.config.evidence_calibrator,
            )
            for item in raw_evidence
        )
        overrides = tuple(
            item for item in evidence if item.confidence_override
        )
        primary = max(
            overrides or evidence,
            key=lambda item: _challenger_evidence_rank(
                item,
                coverage=coverage,
                preferred_label=preferred_label,
            ),
        )
        override = bool(overrides)
        selected = (
            primary.challenger_label
            if override
            else preferred_label
        )
        combined = replace(
            primary.combined,
            selected_label=selected,
            reason=(
                "dual_parallel_multi_calibrated_override"
                if override
                and primary.override_basis
                == "ridge_calibrated_expected_value"
                else "dual_parallel_multi_repeatable_confident_override"
                if override and len(evidence) > 1
                else "dual_parallel_repeatable_confident_override"
                if override
                else "dual_parallel_multi_kept_preferred"
                if len(evidence) > 1
                else "dual_parallel_kept_preferred"
            ),
            confidence_override=override,
        )
        return DualDiscardValidationResult(
            selected_label=selected,
            preferred_label=preferred_label,
            challenger_label=primary.challenger_label,
            confidence_override=override,
            elapsed_ms=(
                screened.elapsed_ms
                + (time.perf_counter() - started) * 1000.0
            ),
            coverage=coverage,
            first_confirmation=primary.first_confirmation,
            second_confirmation=primary.second_confirmation,
            combined=combined,
            expected_coverage_worlds=self.config.coverage_worlds,
            expected_confirmation_worlds=resolved_confirmation_worlds,
            expected_candidate_count=len(labels),
            minimum_confirmation_fraction=(
                self.config.evidence_calibrator.minimum_confirmation_fraction
                if self.config.evidence_calibrator is not None
                else 1.0
            ),
            challenger_evidence=evidence,
        )

    def _run_confirmations(
        self,
        executor: ProcessPoolExecutor,
        *,
        shared_payload: dict[str, Any],
        preferred_label: str,
        challengers: Sequence[str],
        shard_count: int | None = None,
    ) -> list[tuple[str, RootSearchResult, RootSearchResult]]:
        resolved_shard_count = min(
            max(
                1,
                int(
                    shard_count
                    if shard_count is not None
                    else self.config.confirmation_parallel_shards
                ),
            ),
            max(1, int(shared_payload["worlds"])),
        )
        if resolved_shard_count > 1:
            return self._run_sharded_confirmations(
                executor,
                shared_payload=shared_payload,
                preferred_label=preferred_label,
                challengers=challengers,
                shard_count=resolved_shard_count,
            )
        futures = []
        for challenger in challengers:
            payload = {
                **shared_payload,
                "labels": [preferred_label, challenger],
            }
            futures.append(
                (
                    challenger,
                    executor.submit(
                        _root_search_task,
                        {
                            **payload,
                            "seed": self.config.first_confirmation_seed,
                        },
                    ),
                    executor.submit(
                        _root_search_task,
                        {
                            **payload,
                            "seed": self.config.second_confirmation_seed,
                        },
                    ),
                )
            )
        ordered_futures = [
            future
            for _challenger, first_future, second_future in futures
            for future in (first_future, second_future)
        ]
        resolved = dict(
            zip(
                ordered_futures,
                _future_results_before_deadline(
                    ordered_futures,
                    absolute_deadline=shared_payload.get("absolute_deadline"),
                ),
                strict=True,
            )
        )
        return [
            (
                challenger,
                resolved[first_future],
                resolved[second_future],
            )
            for challenger, first_future, second_future in futures
        ]

    def _run_sharded_confirmations(
        self,
        executor: ProcessPoolExecutor,
        *,
        shared_payload: dict[str, Any],
        preferred_label: str,
        challengers: Sequence[str],
        shard_count: int,
    ) -> list[tuple[str, RootSearchResult, RootSearchResult]]:
        started = time.perf_counter()
        worlds = max(1, int(shared_payload["worlds"]))
        shards = _balanced_world_shards(worlds, shard_count)
        futures = []
        for challenger in challengers:
            payload = {
                **shared_payload,
                "labels": [preferred_label, challenger],
                "record_paired_worlds": True,
            }
            first = [
                executor.submit(
                    _root_search_task,
                    {
                        **payload,
                        "worlds": amount,
                        "paired_world_offset": offset,
                        "seed": self.config.first_confirmation_seed,
                    },
                )
                for offset, amount in shards
            ]
            second = [
                executor.submit(
                    _root_search_task,
                    {
                        **payload,
                        "worlds": amount,
                        "paired_world_offset": offset,
                        "seed": self.config.second_confirmation_seed,
                    },
                )
                for offset, amount in shards
            ]
            futures.append((challenger, first, second))
        ordered_futures = [
            future
            for _challenger, first, second in futures
            for future in (*first, *second)
        ]
        resolved = dict(
            zip(
                ordered_futures,
                _future_results_before_deadline(
                    ordered_futures,
                    absolute_deadline=shared_payload.get("absolute_deadline"),
                ),
                strict=True,
            )
        )
        completed = [
            (
                challenger,
                [resolved[future] for future in first],
                [resolved[future] for future in second],
            )
            for challenger, first, second in futures
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [
            (
                challenger,
                _merge_confirmation_shards(
                    first,
                    labels=(preferred_label, challenger),
                    priors=shared_payload["priors"],
                    preferred=preferred_label,
                    expected_worlds=worlds,
                    root_seat=int(shared_payload["view"].seat),
                    elapsed_ms=elapsed_ms,
                    objective_mode=self.config.objective_mode,
                ),
                _merge_confirmation_shards(
                    second,
                    labels=(preferred_label, challenger),
                    priors=shared_payload["priors"],
                    preferred=preferred_label,
                    expected_worlds=worlds,
                    root_seat=int(shared_payload["view"].seat),
                    elapsed_ms=elapsed_ms,
                    objective_mode=self.config.objective_mode,
                ),
            )
            for challenger, first, second in completed
        ]

    def reconfirm_discard(
        self,
        previous: DualDiscardValidationResult,
        *,
        view: PublicView,
        rules: dict[str, Any],
        candidate_labels: Sequence[str],
        candidate_priors: dict[str, float],
        preferred_label: str,
        time_budget_ms: int | None = None,
        absolute_deadline: float | None = None,
    ) -> DualDiscardValidationResult:
        labels = tuple(
            dict.fromkeys(str(label) for label in candidate_labels)
        )
        coverage_labels = {
            candidate.label
            for candidate in previous.coverage.candidates
        }
        if set(labels) != coverage_labels:
            raise ValueError("dual_discard_reconfirmation_coverage_mismatch")
        executor = self.executor or _shared_executor(
            self.config.parallel_workers
        )
        return self.confirm_discard(
            DualDiscardCoverage(
                view=view,
                rules=dict(rules),
                labels=labels,
                candidate_priors=dict(candidate_priors),
                coverage=previous.coverage,
                executor=executor,
                elapsed_ms=previous.coverage.elapsed_ms,
            ),
            preferred_label=preferred_label,
            confirmation_worlds=self.config.reconfirmation_worlds,
            challenger_limit=self.config.reconfirmation_challengers,
            time_budget_ms=time_budget_ms,
            absolute_deadline=absolute_deadline,
        )

    def _parallel_coverage(
        self,
        executor: ProcessPoolExecutor,
        view: PublicView,
        *,
        rules: dict[str, Any],
        labels: list[str],
        priors: dict[str, float],
        preferred: str,
        absolute_deadline: float | None = None,
    ) -> RootSearchResult:
        started = time.perf_counter()
        if _deadline_reached(absolute_deadline):
            return _incomplete_root_search(
                labels,
                priors=priors,
                preferred=preferred,
                reason="parallel_coverage_deadline_before_start",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        coverage_shards = min(
            max(1, int(self.config.coverage_parallel_shards)),
            max(1, int(self.config.coverage_worlds)),
        )
        if coverage_shards > 1:
            shards = _balanced_world_shards(
                self.config.coverage_worlds,
                coverage_shards,
            )
            shared_payload = {
                "view": view,
                "rules": rules,
                "labels": labels,
                "priors": priors,
                "preferred": preferred,
                "seed": self.config.coverage_seed,
                "time_budget_ms": _bounded_time_budget_ms(
                    None,
                    default_ms=self.config.time_budget_ms,
                    absolute_deadline=absolute_deadline,
                ),
                "absolute_deadline": absolute_deadline,
                "rollout_max_turns": self.config.rollout_max_turns,
                "objective_mode": self.config.objective_mode.value,
                "objective_version": self.config.objective_version,
                "rollout_policy_factories": self.rollout_policy_factories,
                "record_paired_worlds": True,
            }
            futures = [
                executor.submit(
                    _root_search_task,
                    {
                        **shared_payload,
                        "worlds": amount,
                        "paired_world_offset": offset,
                    },
                )
                for offset, amount in shards
            ]
            results = _future_results_before_deadline(
                futures,
                absolute_deadline=absolute_deadline,
            )
            return _merge_world_sharded_coverage(
                results,
                labels=labels,
                priors=priors,
                preferred=preferred,
                expected_worlds=self.config.coverage_worlds,
                root_seat=int(view.seat),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                objective_mode=self.config.objective_mode,
            )
        worker_count = min(
            max(1, self.config.parallel_workers),
            max(1, len(labels) // 2),
        )
        partitions = [[] for _ in range(worker_count)]
        for index, label in enumerate(labels):
            partitions[index % worker_count].append(label)
        futures = [
            executor.submit(
                _root_search_task,
                {
                    "view": view,
                    "rules": rules,
                    "labels": partition,
                    "priors": {
                        label: priors[label]
                        for label in partition
                    },
                    "preferred": partition[0],
                    "worlds": self.config.coverage_worlds,
                    "seed": self.config.coverage_seed,
                    "time_budget_ms": _bounded_time_budget_ms(
                        None,
                        default_ms=self.config.time_budget_ms,
                        absolute_deadline=absolute_deadline,
                    ),
                    "absolute_deadline": absolute_deadline,
                    "rollout_max_turns": self.config.rollout_max_turns,
                    "objective_mode": self.config.objective_mode.value,
                    "objective_version": self.config.objective_version,
                    "rollout_policy_factories": (
                        self.rollout_policy_factories
                    ),
                },
            )
            for partition in partitions
        ]
        return _merge_parallel_coverage(
            _future_results_before_deadline(
                futures,
                absolute_deadline=absolute_deadline,
            ),
            preferred=preferred,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


def _root_search_task(payload: dict[str, Any]) -> RootSearchResult:
    labels = list(payload["labels"])
    worlds = int(payload["worlds"])
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=int(payload["time_budget_ms"]),
            max_iterations=worlds * len(labels),
            max_candidates=len(labels),
            skip_search_gap=math.inf,
            rollout_max_turns=int(payload["rollout_max_turns"]),
            require_confident_override=False,
            min_confidence_pairs=min(32, worlds),
            minimum_confident_advantage=0.0,
            complete_first_paired_batch=True,
            require_complete_iteration_budget_for_override=True,
            record_paired_worlds=bool(
                payload.get("record_paired_worlds", False)
            ),
            paired_world_offset=max(
                0,
                int(payload.get("paired_world_offset", 0)),
            ),
            seed=int(payload["seed"]),
            objective_mode=ObjectiveMode(
                payload.get("objective_mode", ObjectiveMode.SCORE_FIRST.value)
            ),
            objective_version=str(
                payload.get("objective_version", OBJECTIVE_VERSION)
            ),
        ),
        rollout_policy_factories=tuple(
            payload["rollout_policy_factories"]
        ),
        root_continuation_policy_factory=payload.get(
            "root_continuation_policy_factory"
        ),
    )
    return policy.search_discard(
        payload["view"],
        rules=payload["rules"],
        candidate_labels=labels,
        candidate_priors=payload["priors"],
        force_search=True,
        paired_candidates=True,
        preferred_label=str(payload["preferred"]),
        absolute_deadline=payload.get("absolute_deadline"),
    )


def _balanced_world_shards(
    worlds: int,
    shard_count: int,
) -> tuple[tuple[int, int], ...]:
    total = max(1, int(worlds))
    count = min(total, max(1, int(shard_count)))
    base, remainder = divmod(total, count)
    offset = 0
    shards = []
    for index in range(count):
        amount = base + int(index < remainder)
        shards.append((offset, amount))
        offset += amount
    return tuple(shards)


def _merge_confirmation_shards(
    results: Sequence[RootSearchResult],
    *,
    labels: Sequence[str],
    priors: dict[str, float],
    preferred: str,
    expected_worlds: int,
    root_seat: int,
    elapsed_ms: float,
    objective_mode: ObjectiveMode = ObjectiveMode.SCORE_FIRST,
) -> RootSearchResult:
    ordered_labels = tuple(dict.fromkeys(str(label) for label in labels))
    worlds = sorted(
        (
            world
            for result in results
            for world in result.paired_worlds
        ),
        key=lambda world: world.world_index,
    )
    expected_indices = tuple(range(max(1, int(expected_worlds))))
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
    aggregates = {}
    for label in ordered_labels:
        reward_sum = 0.0
        outcome_score_sum = 0.0
        signed_xi_sum = 0.0
        wins = 0
        losses = 0
        draws = 0
        for outcome in outcomes_by_label[label]:
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
        aggregates[label] = {
            "reward_sum": reward_sum,
            "outcome_score_sum": outcome_score_sum,
            "signed_xi_sum": signed_xi_sum,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }
    stats = tuple(
        RootCandidateStats(
            label=label,
            visits=len(outcomes_by_label[label]),
            reward_sum=aggregates[label]["reward_sum"],
            average_reward=(
                aggregates[label]["reward_sum"]
                / max(1, len(outcomes_by_label[label]))
            ),
            win_rate=(
                aggregates[label]["wins"]
                / max(1, len(outcomes_by_label[label]))
            ),
            heuristic_value=float(priors.get(label, 0.0)),
            wins=aggregates[label]["wins"],
            losses=aggregates[label]["losses"],
            draws=aggregates[label]["draws"],
            loss_rate=(
                aggregates[label]["losses"]
                / max(1, len(outcomes_by_label[label]))
            ),
            draw_rate=(
                aggregates[label]["draws"]
                / max(1, len(outcomes_by_label[label]))
            ),
            mean_outcome_score=(
                aggregates[label]["outcome_score_sum"]
                / max(1, len(outcomes_by_label[label]))
            ),
            mean_signed_xi=(
                aggregates[label]["signed_xi_sum"]
                / max(1, len(outcomes_by_label[label]))
            ),
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
        objective_mode=objective_mode,
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
    return RootSearchResult(
        selected_label=empirical_best.label if complete else preferred,
        used_search=complete,
        reason=(
            "root_discard_ismcts_completed"
            if complete
            else "sharded_confirmation_incomplete"
        ),
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
        empirical_best_label=empirical_best.label,
        confidence_override=False,
        paired_advantages=paired_advantages,
        paired_worlds=(),
    )


def _merge_world_sharded_coverage(
    results: Sequence[RootSearchResult],
    *,
    labels: Sequence[str],
    priors: dict[str, float],
    preferred: str,
    expected_worlds: int,
    root_seat: int,
    elapsed_ms: float,
    objective_mode: ObjectiveMode = ObjectiveMode.SCORE_FIRST,
) -> RootSearchResult:
    merged = _merge_confirmation_shards(
        results,
        labels=labels,
        priors=priors,
        preferred=preferred,
        expected_worlds=expected_worlds,
        root_seat=root_seat,
        elapsed_ms=elapsed_ms,
        objective_mode=objective_mode,
    )
    candidates = tuple(
        sorted(
            merged.candidates,
            key=lambda item: (
                item.average_reward,
                item.visits,
                item.heuristic_value,
                item.label,
            ),
            reverse=True,
        )
    )
    best = candidates[0]
    return replace(
        merged,
        selected_label=best.label if merged.used_search else preferred,
        reason=(
            "parallel_world_sharded_coverage_completed"
            if merged.used_search
            else "parallel_world_sharded_coverage_incomplete"
        ),
        candidates=candidates,
        determinization_failures=sum(
            result.determinization_failures for result in results
        ),
        empirical_best_label=best.label,
        confidence_override=False,
    )


def _challenger_evidence(
    *,
    preferred_label: str,
    challenger: str,
    first: RootSearchResult,
    second: RootSearchResult,
    confirmation_worlds: int,
    repeatable_mean_override_threshold: float | None,
    familywise_comparisons: int,
) -> DualDiscardChallengerEvidence:
    combined = _combined_discard_confirmations(
        first,
        second,
        preferred_label=preferred_label,
        alternative_labels=(challenger,),
        minimum_samples=confirmation_worlds * 2,
        minimum_advantage=0.0,
        familywise_comparisons=max(1, int(familywise_comparisons)),
    )
    override_basis = _dual_override_basis(
        first,
        second,
        combined,
        challenger=challenger,
        expected_worlds=confirmation_worlds,
        repeatable_mean_override_threshold=(
            repeatable_mean_override_threshold
        ),
    )
    return DualDiscardChallengerEvidence(
        challenger_label=challenger,
        confidence_override=override_basis is not None,
        first_confirmation=first,
        second_confirmation=second,
        combined=combined,
        override_basis=override_basis,
    )


def _select_challenger_labels(
    candidates: Sequence[RootCandidateStats],
    *,
    preferred_label: str,
    limit: int,
    prior_slots: int,
) -> tuple[str, ...]:
    available = [
        candidate
        for candidate in candidates
        if candidate.label != preferred_label
    ]
    resolved_limit = min(max(1, int(limit)), len(available))
    resolved_prior_slots = min(
        max(0, int(prior_slots)),
        resolved_limit - 1 if resolved_limit > 1 else 0,
    )
    coverage_slots = resolved_limit - resolved_prior_slots
    selected = [
        candidate.label
        for candidate in available[:coverage_slots]
    ]
    if len(selected) >= resolved_limit:
        return tuple(selected)
    prior_ranked = sorted(
        available,
        key=lambda candidate: (
            candidate.heuristic_value,
            candidate.average_reward,
            candidate.label,
        ),
        reverse=True,
    )
    for candidate in (*prior_ranked, *available):
        if candidate.label in selected:
            continue
        selected.append(candidate.label)
        if len(selected) >= resolved_limit:
            break
    return tuple(selected)


def _challenger_evidence_rank(
    evidence: DualDiscardChallengerEvidence,
    *,
    coverage: RootSearchResult,
    preferred_label: str,
) -> tuple[float, float, float, float, float, str]:
    advantage = next(
        (
            item
            for item in evidence.combined.paired_advantages
            if item.candidate_key == evidence.challenger_label
        ),
        None,
    )
    candidate = next(
        (
            item
            for item in evidence.combined.candidates
            if item.label == evidence.challenger_label
        ),
        None,
    )
    return (
        float(
            evidence.calibrated_advantage
            if evidence.calibrated_advantage is not None
            else -math.inf
        ),
        _pooled_challenger_mean_delta(
            evidence,
            coverage=coverage,
            preferred_label=preferred_label,
        ),
        float(
            advantage.lower_confidence_bound
            if advantage is not None
            else -math.inf
        ),
        float(
            advantage.mean_delta
            if advantage is not None
            else -math.inf
        ),
        float(
            candidate.average_reward
            if candidate is not None
            else -math.inf
        ),
        evidence.challenger_label,
    )


def _pooled_challenger_mean_delta(
    evidence: DualDiscardChallengerEvidence,
    *,
    coverage: RootSearchResult,
    preferred_label: str,
) -> float:
    confirmation = next(
        (
            item
            for item in evidence.combined.paired_advantages
            if item.candidate_key == evidence.challenger_label
        ),
        None,
    )
    coverage_challenger = next(
        (
            item
            for item in coverage.candidates
            if item.label == evidence.challenger_label
        ),
        None,
    )
    coverage_preferred = next(
        (
            item
            for item in coverage.candidates
            if item.label == preferred_label
        ),
        None,
    )
    if (
        confirmation is None
        or coverage_challenger is None
        or coverage_preferred is None
    ):
        return float(
            confirmation.mean_delta
            if confirmation is not None
            else -math.inf
        )
    coverage_samples = min(
        int(coverage_challenger.visits),
        int(coverage_preferred.visits),
    )
    confirmation_samples = int(confirmation.samples)
    total_samples = coverage_samples + confirmation_samples
    if total_samples <= 0:
        return -math.inf
    coverage_delta = (
        float(coverage_challenger.average_reward)
        - float(coverage_preferred.average_reward)
    )
    return (
        coverage_delta * coverage_samples
        + float(confirmation.mean_delta) * confirmation_samples
    ) / total_samples


def _calibrate_challenger_evidence(
    evidence: DualDiscardChallengerEvidence,
    *,
    coverage: RootSearchResult,
    view: PublicView | None = None,
    preferred_label: str,
    expected_worlds: int,
    calibrator: DualDiscardEvidenceCalibratorConfig | None,
) -> DualDiscardChallengerEvidence:
    if calibrator is None:
        return evidence
    if (
        calibrator.trained_confirmation_worlds is not None
        and int(calibrator.trained_confirmation_worlds) != expected_worlds
    ):
        return evidence
    if not _confirmation_pair_usable(
        evidence.first_confirmation,
        evidence.second_confirmation,
        expected_worlds=expected_worlds,
        minimum_fraction=calibrator.minimum_confirmation_fraction,
    ):
        return replace(
            evidence,
            confidence_override=False,
            override_basis=None,
            calibrated_advantage=None,
        )
    features = _calibrator_features(
        evidence,
        coverage=coverage,
        view=view,
        preferred_label=preferred_label,
        expected_worlds=expected_worlds,
    )
    if len(calibrator.coefficients) <= len(features):
        features = features[: len(calibrator.coefficients)]
    if not (
        len(features)
        == len(calibrator.feature_mean)
        == len(calibrator.feature_scale)
        == len(calibrator.coefficients)
    ):
        raise ValueError("dual_discard_calibrator_feature_mismatch")
    standardized = [
        (value - center) / scale
        for value, center, scale in zip(
            features,
            calibrator.feature_mean,
            calibrator.feature_scale,
        )
    ]
    calibrated_advantage = float(calibrator.intercept) + sum(
        coefficient * value
        for coefficient, value in zip(
            calibrator.coefficients,
            standardized,
        )
    )
    override = calibrated_advantage > float(calibrator.decision_margin)
    return replace(
        evidence,
        confidence_override=override,
        override_basis=(
            "ridge_calibrated_expected_value"
            if override
            else None
        ),
        calibrated_advantage=calibrated_advantage,
    )


def _calibrator_features(
    evidence: DualDiscardChallengerEvidence,
    *,
    coverage: RootSearchResult,
    view: PublicView | None = None,
    preferred_label: str,
    expected_worlds: int,
) -> tuple[float, ...]:
    first_advantage = _candidate_advantage(
        evidence.first_confirmation,
        evidence.challenger_label,
    )
    second_advantage = _candidate_advantage(
        evidence.second_confirmation,
        evidence.challenger_label,
    )
    combined_advantage = _candidate_advantage(
        evidence.combined,
        evidence.challenger_label,
    )
    if (
        first_advantage is None
        or second_advantage is None
        or combined_advantage is None
    ):
        raise ValueError("dual_discard_calibrator_advantage_missing")
    coverage_by_label = {
        candidate.label: candidate
        for candidate in coverage.candidates
    }
    challenger = coverage_by_label[evidence.challenger_label]
    preferred = coverage_by_label[preferred_label]
    heuristic_ranking = sorted(
        coverage.candidates,
        key=lambda candidate: (
            candidate.heuristic_value,
            candidate.average_reward,
            candidate.label,
        ),
        reverse=True,
    )
    coverage_rank = next(
        index
        for index, candidate in enumerate(coverage.candidates, start=1)
        if candidate.label == evidence.challenger_label
    )
    heuristic_rank = next(
        index
        for index, candidate in enumerate(heuristic_ranking, start=1)
        if candidate.label == evidence.challenger_label
    )
    core_features = (
        float(combined_advantage.mean_delta),
        float(combined_advantage.standard_error),
        abs(
            float(first_advantage.mean_delta)
            - float(second_advantage.mean_delta)
        ),
        min(
            float(first_advantage.mean_delta),
            float(second_advantage.mean_delta),
        ),
        float(challenger.average_reward - preferred.average_reward),
        float(
            challenger.heuristic_value - preferred.heuristic_value
        )
        / 1000.0,
        1.0 / coverage_rank,
        1.0 / heuristic_rank,
        min(
            evidence.first_confirmation.paired_determinizations,
            evidence.second_confirmation.paired_determinizations,
        )
        / max(1, expected_worlds),
    )
    context = opponent_context_features(view)
    return (
        *core_features,
        *context,
        *opponent_context_interactions(core_features[0], context),
    )


def _candidate_advantage(
    search: RootSearchResult,
    challenger_label: str,
):
    return next(
        (
            advantage
            for advantage in search.paired_advantages
            if advantage.candidate_key == challenger_label
        ),
        None,
    )


def _confirmation_pair_complete(
    first: RootSearchResult,
    second: RootSearchResult,
    *,
    expected_worlds: int,
) -> bool:
    return all(
        search.used_search
        and search.deadline_interruptions == 0
        and search.rollout_invariant_violations == 0
        and search.rollout_coverage_failures == 0
        and search.paired_determinizations == expected_worlds
        and len(search.candidates) == 2
        and all(
            candidate.visits == expected_worlds
            for candidate in search.candidates
        )
        for search in (first, second)
    )


def _confirmation_pair_usable(
    first: RootSearchResult,
    second: RootSearchResult,
    *,
    expected_worlds: int,
    minimum_fraction: float,
) -> bool:
    fraction = min(1.0, max(0.0, float(minimum_fraction)))
    if fraction >= 1.0:
        return _confirmation_pair_complete(
            first,
            second,
            expected_worlds=expected_worlds,
        )
    required_worlds = max(2, math.ceil(expected_worlds * fraction))
    return all(
        search.used_search
        and search.rollout_invariant_violations == 0
        and search.rollout_coverage_failures == 0
        and search.paired_determinizations >= required_worlds
        and len(search.candidates) == 2
        and all(
            candidate.visits == search.paired_determinizations
            for candidate in search.candidates
        )
        for search in (first, second)
    )


def _merge_parallel_coverage(
    results: Sequence[RootSearchResult],
    *,
    preferred: str,
    elapsed_ms: float,
) -> RootSearchResult:
    candidates = tuple(
        sorted(
            (
                candidate
                for result in results
                for candidate in result.candidates
            ),
            key=lambda item: (
                item.average_reward,
                item.visits,
                item.heuristic_value,
                item.label,
            ),
            reverse=True,
        )
    )
    if not candidates:
        raise ValueError("parallel_coverage_returned_no_candidates")
    best = candidates[0]
    usable = all(result.used_search for result in results)
    return RootSearchResult(
        selected_label=best.label if usable else preferred,
        used_search=usable,
        reason=(
            "parallel_all_action_coverage_completed"
            if usable
            else "parallel_all_action_coverage_incomplete"
        ),
        simulations=sum(result.simulations for result in results),
        elapsed_ms=elapsed_ms,
        candidates=candidates,
        determinization_failures=sum(
            result.determinization_failures for result in results
        ),
        paired_determinizations=min(
            result.paired_determinizations for result in results
        ),
        deadline_interruptions=sum(
            result.deadline_interruptions for result in results
        ),
        rollout_invariant_violations=sum(
            result.rollout_invariant_violations for result in results
        ),
        rollout_violations=tuple(
            dict.fromkeys(
                violation
                for result in results
                for violation in result.rollout_violations
            )
        ),
        rollout_coverage_failures=sum(
            result.rollout_coverage_failures for result in results
        ),
        rollout_coverage_reasons=tuple(
            dict.fromkeys(
                reason
                for result in results
                for reason in result.rollout_coverage_reasons
            )
        ),
        empirical_best_label=best.label,
        confidence_override=False,
    )


def _dual_override_basis(
    first: RootSearchResult,
    second: RootSearchResult,
    combined: RootSearchResult,
    *,
    challenger: str,
    expected_worlds: int,
    repeatable_mean_override_threshold: float | None,
) -> str | None:
    first_advantage = next(
        (
            item
            for item in first.paired_advantages
            if item.candidate_key == challenger
        ),
        None,
    )
    second_advantage = next(
        (
            item
            for item in second.paired_advantages
            if item.candidate_key == challenger
        ),
        None,
    )
    combined_advantage = next(
        (
            item
            for item in combined.paired_advantages
            if item.candidate_key == challenger
        ),
        None,
    )
    confirmation_pair_complete = bool(
        first_advantage is not None
        and second_advantage is not None
        and combined_advantage is not None
        and _confirmation_pair_complete(
            first,
            second,
            expected_worlds=expected_worlds,
        )
    )
    if not confirmation_pair_complete:
        return None
    split_evidence_not_negative = bool(
        first_advantage.upper_confidence_bound > 0.0
        and second_advantage.upper_confidence_bound > 0.0
    )
    if (
        combined.selected_label == challenger
        and split_evidence_not_negative
    ):
        return "familywise_confident"
    repeatable_positive = bool(
        first_advantage.mean_delta > 0.0
        and second_advantage.mean_delta > 0.0
    )
    if (
        repeatable_positive
        and repeatable_mean_override_threshold is not None
        and combined_advantage.mean_delta
        >= float(repeatable_mean_override_threshold)
    ):
        return "calibrated_repeatable_mean"
    return None


def _validation_deadline(
    absolute_deadline: float | None,
    *,
    time_budget_ms: int,
) -> float:
    local_deadline = (
        time.perf_counter() + max(1, int(time_budget_ms)) / 1000.0
    )
    if absolute_deadline is None:
        return local_deadline
    return min(local_deadline, float(absolute_deadline))


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _bounded_time_budget_ms(
    requested_ms: int | None,
    *,
    default_ms: int,
    absolute_deadline: float | None,
) -> int:
    budget_ms = max(
        1,
        int(requested_ms if requested_ms is not None else default_ms),
    )
    if absolute_deadline is None:
        return budget_ms
    remaining_ms = max(
        1,
        int((absolute_deadline - time.perf_counter()) * 1000.0),
    )
    return min(budget_ms, remaining_ms)


def _adaptive_confirmation_worlds(
    *,
    maximum_worlds: int,
    minimum_worlds: int,
    worlds_per_second: float | None,
    guard_ms: int,
    absolute_deadline: float | None,
) -> int:
    maximum = max(2, int(maximum_worlds))
    if worlds_per_second is None or absolute_deadline is None:
        return maximum
    minimum = min(maximum, max(2, int(minimum_worlds)))
    usable_seconds = max(
        0.0,
        (
            (float(absolute_deadline) - time.perf_counter()) * 1000.0
            - max(0, int(guard_ms))
        )
        / 1000.0,
    )
    budgeted = int(
        math.floor(
            usable_seconds * max(0.1, float(worlds_per_second))
        )
    )
    return min(maximum, max(minimum, budgeted))


def _incomplete_validation(
    screened: DualDiscardCoverage,
    *,
    preferred_label: str,
    reason: str,
    challenger_label: str | None = None,
    expected_coverage_worlds: int,
    confirmation_worlds: int | None = None,
) -> DualDiscardValidationResult:
    challenger = challenger_label or next(
        (
            candidate.label
            for candidate in screened.coverage.candidates
            if candidate.label != preferred_label
        ),
        next(
            label
            for label in screened.labels
            if label != preferred_label
        ),
    )
    worlds = max(
        2,
        int(
            confirmation_worlds
            if confirmation_worlds is not None
            else 0
        ),
    )
    confirmation = _incomplete_root_search(
        [preferred_label, challenger],
        priors=screened.candidate_priors,
        preferred=preferred_label,
        reason=reason,
        elapsed_ms=0.0,
    )
    return DualDiscardValidationResult(
        selected_label=preferred_label,
        preferred_label=preferred_label,
        challenger_label=challenger,
        confidence_override=False,
        elapsed_ms=screened.elapsed_ms,
        coverage=screened.coverage,
        first_confirmation=confirmation,
        second_confirmation=confirmation,
        combined=confirmation,
        expected_coverage_worlds=max(1, int(expected_coverage_worlds)),
        expected_confirmation_worlds=worlds,
        expected_candidate_count=len(screened.labels),
    )


def _incomplete_root_search(
    labels: Sequence[str],
    *,
    priors: dict[str, float],
    preferred: str,
    reason: str,
    elapsed_ms: float,
) -> RootSearchResult:
    return RootSearchResult(
        selected_label=preferred,
        used_search=False,
        reason=reason,
        simulations=0,
        elapsed_ms=elapsed_ms,
        candidates=tuple(
            RootCandidateStats(
                label=label,
                visits=0,
                reward_sum=0.0,
                average_reward=0.0,
                win_rate=0.0,
                heuristic_value=float(priors.get(label, 0.0)),
            )
            for label in labels
        ),
        deadline_interruptions=1,
        empirical_best_label=preferred,
    )


__all__ = [
    "DualBatchDiscardValidator",
    "DualDiscardChallengerEvidence",
    "DualDiscardCoverage",
    "DualDiscardValidationResult",
    "DualDiscardValidatorConfig",
    "close_shared_dual_discard_executors",
]
