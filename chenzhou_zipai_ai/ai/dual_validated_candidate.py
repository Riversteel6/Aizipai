"""Runtime candidate that overlaps the frozen search with dual validation."""

from __future__ import annotations

import os
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import replace
from typing import Any, Callable, Sequence

from ai.decision_objective import DecisionAnchor, OBJECTIVE_VERSION, ObjectiveMode
from ai.dual_discard_validator import (
    DualBatchDiscardValidator,
    DualDiscardEvidenceCalibratorConfig,
    DualDiscardValidationResult,
    DualDiscardValidatorConfig,
)
from ai.full_game_simulator import (
    PublicView,
    SimulationPolicy,
    _discardable_labels,
)
from ai.ismcts import (
    ProfessionalProgressiveAllActionCandidatePolicy,
    RootCandidateStats,
    RootSearchResult,
    public_view_to_dict,
)
from ai.parallel_response_search import ExactShardedProgressiveResponsePolicy
from ai.parallel_discard_search import ExactShardedRootDiscardPolicy
from ai.opponent_belief_runtime import (
    RuntimeOpponentBelief,
    runtime_opponent_rollout_schedule,
)
from ai.simulation_trace import public_state_identity
from engine.chi_rules import ChiPlan
from engine.melds import classify_meld


def _runtime_parallel_shards(configured: int) -> int:
    resolved = max(1, int(configured))
    if os.environ.get("AIZIPAI_MOBILE_RUNTIME") != "1":
        return resolved
    try:
        physical_workers = int(
            os.environ.get("AIZIPAI_MOBILE_STRATEGY_WORKERS", "2")
        )
    except ValueError:
        physical_workers = 2
    try:
        shards_per_worker = int(
            os.environ.get("AIZIPAI_MOBILE_SHARDS_PER_WORKER", "4")
        )
    except ValueError:
        shards_per_worker = 4
    # Physical concurrency and logical task grain are different controls.
    # Two phone workers still need several bounded shards each: collapsing a
    # whole stage into two giant tasks can exceed the per-request timeout,
    # while the old one-world-per-task layout can flood the shared queue.
    bounded_logical_shards = max(
        1,
        physical_workers * max(1, shards_per_worker),
    )
    return min(resolved, bounded_logical_shards)


def _install_rollout_policy_factories(
    search: Any,
    factories: tuple[Callable[[], SimulationPolicy], ...],
    *,
    seen: set[int] | None = None,
) -> None:
    visited = seen if seen is not None else set()
    if search is None or id(search) in visited:
        return
    visited.add(id(search))
    if hasattr(search, "rollout_policy_factories"):
        search.rollout_policy_factories = factories
        search.rollout_policy_factory = factories[0]
    base = getattr(search, "base", None)
    if base is not None:
        _install_rollout_policy_factories(base, factories, seen=visited)
    for attribute in (
        "coverage",
        "refinement",
        "discard_selection",
        "discard_confirmation",
        "response_selection",
        "response_binary_confirmation",
        "response_confirmation",
    ):
        child = getattr(search, attribute, None)
        if child is not None:
            _install_rollout_policy_factories(child, factories, seen=visited)


class ProfessionalParallelDualValidatedCandidatePolicy(
    ProfessionalProgressiveAllActionCandidatePolicy
):
    """Keep the frozen decision unless repeatable parallel evidence improves it."""

    name = "professional_parallel_dual_validated_candidate_v1"
    enable_late_full_coverage_fallback = False
    late_full_coverage_minimum_window_ms = 0
    reuse_validation_coverage_as_baseline = False
    single_pass_progressive_discard_evidence = False

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        validator_config: DualDiscardValidatorConfig | None = None,
        validator: DualBatchDiscardValidator | None = None,
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 0,
        response_minimum_confident_advantage: float = 0.02,
    ) -> None:
        factories = tuple(rollout_policy_factories)
        super().__init__(
            rollout_policy_factories=factories,
            response_minimum_confident_advantage=(
                response_minimum_confident_advantage
            ),
        )
        self.discard_validator = validator or DualBatchDiscardValidator(
            rollout_policy_factories=factories,
            config=validator_config,
        )
        self.decision_time_budget_ms = max(
            100,
            int(decision_time_budget_ms),
        )
        self.deadline_guard_ms = max(0, int(deadline_guard_ms))
        self.confirmation_reserve_ms = min(
            max(0, int(confirmation_reserve_ms)),
            max(
                0,
                self.decision_time_budget_ms
                - self.deadline_guard_ms
                - 1,
            ),
        )
        self.last_baseline_search: RootSearchResult | None = None
        self.last_validation: DualDiscardValidationResult | None = None
        self.last_validation_error: str | None = None
        self.validation_attempts = 0
        self.validation_overrides = 0
        self.validation_elapsed_ms = 0.0
        self.validation_reconfirmations = 0
        self.validation_reconfirmation_elapsed_ms = 0.0
        self.concurrent_discard_elapsed_ms = 0.0
        self.decision_budget_fallbacks = 0
        self.validation_timeouts = 0
        self.validation_incomplete_fallbacks = 0
        self.reconfirmation_budget_fallbacks = 0
        self.reconfirmation_incomplete_fallbacks = 0
        self.validation_cleanup_waits = 0
        self.validation_cleanup_elapsed_ms = 0.0
        self.validation_partial_overrides = 0
        self.late_full_coverage_fallbacks = 0
        self.validation_confirmation_skips = 0

    def choose_discard(
        self,
        view: PublicView,
        rules: dict[str, Any],
        *,
        absolute_deadline: float | None = None,
        production_decision: Any | None = None,
    ) -> str:
        started = time.perf_counter()
        local_deadline = started + self.decision_time_budget_ms / 1000.0
        decision_deadline = (
            local_deadline
            if absolute_deadline is None
            else min(local_deadline, float(absolute_deadline))
        )
        decision = production_decision or self._choose(
            view,
            rules,
            legal_actions=[{"type": "DISCARD"}],
        )
        if decision.selected_action != "DISCARD" or not decision.selected_label:
            raise ValueError(
                "parallel_dual_candidate_no_discard:"
                f"{decision.selected_action}"
            )
        anchor = DecisionAnchor(str(decision.selected_label))
        legal_labels = set(_discardable_labels(view.hand))
        by_label: dict[str, Any] = {}
        for item in decision.action_evals:
            label = item.action.label
            if (
                item.type != "DISCARD"
                or not label
                or label not in legal_labels
            ):
                continue
            previous = by_label.get(label)
            if previous is None or item.ev > previous.ev:
                by_label[label] = item
        ranked = sorted(
            by_label.values(),
            key=lambda item: (float(item.ev), str(item.action.label)),
            reverse=True,
        )
        public_view_payload = public_view_to_dict(view)
        base_event = {
            "production_label": decision.selected_label,
            "decision_anchor": {
                "label": anchor.label,
                "source": anchor.source,
            },
            "legal_candidate_count": len(ranked),
            "searched_candidate_count": len(ranked),
            "public_view": public_view_payload,
            "public_view_id": public_state_identity(public_view_payload),
        }
        if len(ranked) < 2:
            self.last_baseline_search = None
            self.last_validation = None
            self.last_validation_error = None
            self.last_search = None
            self._record_discard_event(
                {
                    **base_event,
                    "search_attempted": False,
                    "used_search": False,
                    "reason": "fewer_than_two_discard_candidates",
                }
            )
            return decision.selected_label

        labels = [str(item.action.label) for item in ranked]
        priors = {
            str(item.action.label): float(item.ev)
            for item in ranked
        }
        if self.single_pass_progressive_discard_evidence:
            return self._choose_discard_with_progressive_evidence(
                view,
                rules=rules,
                decision=decision,
                anchor=anchor,
                labels=labels,
                priors=priors,
                base_event=base_event,
                decision_deadline=decision_deadline,
                started=started,
            )
        self.last_validation = None
        self.last_validation_error = None
        validation_deadline = (
            decision_deadline - self.deadline_guard_ms / 1000.0
        )
        preconfirmation_deadline = max(
            started,
            validation_deadline
            - self.confirmation_reserve_ms / 1000.0,
        )
        late_full_coverage_fallback = bool(
            self.enable_late_full_coverage_fallback
            and _remaining_budget_ms(
                preconfirmation_deadline,
                guard_ms=0,
            )
            < self.late_full_coverage_minimum_window_ms
        )
        validation: DualDiscardValidationResult | None = None
        validation_confirmation_skipped = False
        validation_confirmation_skip_reason: str | None = None
        validation_confirmation_route: str | None = None
        reconfirmation_attempted = False
        validation_worker_pending_at_cleanup = False
        validation_worker_done_after_cleanup = True
        validation_cleanup_elapsed_ms = 0.0
        if late_full_coverage_fallback:
            self.late_full_coverage_fallbacks += 1
            baseline = self._search_discard_anchor(
                view,
                rules=rules,
                candidate_labels=labels,
                candidate_priors=priors,
                preferred_label=decision.selected_label,
                absolute_deadline=validation_deadline,
            )
        else:
            self.validation_attempts += 1
            thread = ThreadPoolExecutor(max_workers=1)
            validation_future = thread.submit(
                self.discard_validator.screen_discard,
                view,
                rules=rules,
                candidate_labels=labels,
                candidate_priors=priors,
                preferred_label=anchor.label,
                absolute_deadline=preconfirmation_deadline,
            )
            baseline: RootSearchResult | None = None
            try:
                if not self.reuse_validation_coverage_as_baseline:
                    baseline = self._search_discard_anchor(
                        view,
                        rules=rules,
                        candidate_labels=labels,
                        candidate_priors=priors,
                        preferred_label=decision.selected_label,
                        absolute_deadline=preconfirmation_deadline,
                    )
                try:
                    if validation_future.done():
                        # The baseline phase may consume its whole budget even
                        # though the concurrent coverage result is already
                        # available. Reading a completed future does not wait
                        # and must not be recorded as a budget fallback.
                        screened = validation_future.result(timeout=0)
                    else:
                        remaining_ms = _remaining_budget_ms(
                            validation_deadline,
                            guard_ms=0,
                        )
                        if remaining_ms <= 0:
                            raise FutureTimeoutError
                        screened = validation_future.result(
                            timeout=remaining_ms / 1000.0,
                        )
                    if self.reuse_validation_coverage_as_baseline:
                        # The validator coverage is already the complete,
                        # production-anchored top-set nomination over every
                        # legal discard. Reuse that exact evidence instead of
                        # running a second provisional all-action root search
                        # that competes for the same foreground deadline.
                        baseline = screened.coverage
                    validation_confirmation_skip_reason = (
                        self._discard_confirmation_skip_reason(
                            production_label=decision.selected_label,
                            baseline=baseline,
                            screened=screened,
                        )
                    )
                    if validation_confirmation_skip_reason is not None:
                        validation_confirmation_skipped = True
                        self.validation_confirmation_skips += 1
                    else:
                        remaining_ms = _remaining_budget_ms(
                            validation_deadline,
                            guard_ms=0,
                        )
                        if remaining_ms <= 0:
                            raise FutureTimeoutError
                        (
                            validation,
                            validation_confirmation_route,
                        ) = self._confirm_discard_validation(
                            screened,
                            preferred_label=anchor.label,
                            time_budget_ms=remaining_ms,
                            absolute_deadline=validation_deadline,
                        )
                        if (
                            not validation.complete
                            and time.perf_counter() >= validation_deadline
                        ):
                            self.last_validation_error = (
                                "decision_budget_validation_incomplete"
                            )
                except FutureTimeoutError:
                    validation_future.cancel()
                    validation = None
                    self.validation_timeouts += 1
                    self.last_validation_error = (
                        "decision_budget_waiting_for_validation"
                    )
                except Exception as exc:  # pragma: no cover - runtime fallback
                    validation = None
                    self.last_validation_error = f"{type(exc).__name__}:{exc}"
            finally:
                validation_worker_pending_at_cleanup = (
                    not validation_future.done()
                )
                cleanup_started = time.perf_counter()
                thread.shutdown(wait=False, cancel_futures=True)
                validation_cleanup_elapsed_ms = (
                    time.perf_counter() - cleanup_started
                ) * 1000.0
                validation_worker_done_after_cleanup = (
                    validation_future.done()
                )
                self.validation_cleanup_waits += int(
                    validation_worker_pending_at_cleanup
                )
                self.validation_cleanup_elapsed_ms += (
                    validation_cleanup_elapsed_ms
                )
            if baseline is None:
                baseline = _incomplete_production_anchor_search(
                    labels,
                    priors=priors,
                    preferred_label=anchor.label,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )

        self.last_baseline_search = baseline
        self.last_validation = validation
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.concurrent_discard_elapsed_ms += elapsed_ms
        partial_override_usable = bool(
            validation is not None
            and not validation.complete
            and validation.usable_confident_override
            and validation.preferred_label == anchor.label
        )
        validation_usable = bool(
            validation is not None
            and (validation.complete or partial_override_usable)
        )
        if not validation_usable:
            self.validation_incomplete_fallbacks += int(
                validation is not None
            )
            self.decision_budget_fallbacks += int(
                self.last_validation_error is not None
                and self.last_validation_error.startswith(
                    "decision_budget_"
                )
                and baseline.selected_label != anchor.label
            )
            merged = replace(
                baseline,
                reason=(
                    "parallel_dual_hybrid:"
                    + baseline.reason
                    + ":"
                    + (
                        "late_full_coverage_only"
                        if late_full_coverage_fallback
                        else validation_confirmation_skip_reason
                        if validation_confirmation_skipped
                        else self.last_validation_error
                        or "dual_validation_incomplete"
                    )
                ),
                elapsed_ms=elapsed_ms,
                confidence_override=False,
            )
            selected = baseline.selected_label
        else:
            selected = validation.selected_label
            merged = _merge_concurrent_discard_searches(
                baseline,
                validation,
                selected_label=selected,
                elapsed_ms=elapsed_ms,
                validation_usable=validation_usable,
            )
            self.validation_elapsed_ms += validation.elapsed_ms
            self.validation_overrides += int(
                selected != anchor.label
            )
            self.validation_partial_overrides += int(
                partial_override_usable
                and selected != anchor.label
            )
        proposed_selected = selected
        selected, authorization_reason = self._authorize_final_discard(
            production_label=decision.selected_label,
            proposed_label=proposed_selected,
            validation=validation,
        )
        if selected != proposed_selected:
            merged = replace(
                merged,
                selected_label=selected,
                reason=f"{merged.reason}:{authorization_reason}",
                confidence_override=False,
            )
        self.last_search = merged
        self.search_attempts += 1
        self.search_simulations += merged.simulations
        self.search_elapsed_ms += merged.elapsed_ms
        self.search_paired_determinizations += (
            merged.paired_determinizations
        )
        self.search_deadline_interruptions += (
            merged.deadline_interruptions
        )
        self.search_candidate_coverage_failures += int(
            any(item.visits == 0 for item in merged.candidates)
        )
        self.search_rollout_coverage_failures += (
            merged.rollout_coverage_failures
        )
        self.search_overrides += int(
            selected != decision.selected_label
        )
        self.search_confidence_overrides += int(
            merged.confidence_override
        )
        event = {
            **base_event,
            "search_attempted": True,
            "search_selected_label": selected,
            "proposed_search_selected_label": proposed_selected,
            "final_authorization_reason": authorization_reason,
            "baseline_selected_label": baseline.selected_label,
            "validation_selected_label": (
                validation.selected_label
                if validation is not None
                else None
            ),
            "validation_preferred_label": (
                validation.preferred_label
                if validation is not None
                else None
            ),
            "validation_challenger_label": (
                validation.challenger_label
                if validation is not None
                else None
            ),
            "validation_challenger_labels": (
                [
                    evidence.challenger_label
                    for evidence in validation.challenger_evidence
                ]
                if validation is not None
                else []
            ),
            "validation_complete": bool(
                validation is not None and validation.complete
            ),
            "validation_attempted": (
                not late_full_coverage_fallback
            ),
            "validation_reconfirmation_attempted": (
                reconfirmation_attempted
            ),
            "validation_confirmation_skipped": (
                validation_confirmation_skipped
            ),
            "validation_confirmation_skip_reason": (
                validation_confirmation_skip_reason
            ),
            "validation_confirmation_route": validation_confirmation_route,
            "validation_diagnostics": _validation_diagnostics(
                validation
            ),
            "baseline_diagnostics": _search_stage_diagnostics(baseline),
            "baseline_phase_deadline_interruptions": (
                baseline.deadline_interruptions
            ),
            "validation_error": self.last_validation_error,
            "decision_time_budget_ms": self.decision_time_budget_ms,
            "deadline_guard_ms": self.deadline_guard_ms,
            "confirmation_reserve_ms": self.confirmation_reserve_ms,
            "late_full_coverage_fallback": (
                late_full_coverage_fallback
            ),
            "validation_worker_pending_at_cleanup": (
                validation_worker_pending_at_cleanup
            ),
            "validation_worker_done_after_cleanup": (
                validation_worker_done_after_cleanup
            ),
            "validation_cleanup_elapsed_ms": round(
                validation_cleanup_elapsed_ms,
                3,
            ),
            "used_search": merged.used_search,
            "reason": merged.reason,
            "simulations": merged.simulations,
            "paired_determinizations": (
                merged.paired_determinizations
            ),
            "deadline_interruptions": (
                merged.deadline_interruptions
            ),
            "rollout_invariant_violations": (
                merged.rollout_invariant_violations
            ),
            "rollout_violations": list(merged.rollout_violations),
            "rollout_coverage_failures": (
                merged.rollout_coverage_failures
            ),
            "rollout_coverage_reasons": list(
                merged.rollout_coverage_reasons
            ),
            "elapsed_ms": round(merged.elapsed_ms, 3),
            "disagreement": selected != decision.selected_label,
            "empirical_best_label": merged.empirical_best_label,
            "confidence_override": merged.confidence_override,
            "paired_advantages": [
                advantage.to_dict()
                for advantage in merged.paired_advantages
            ],
            "candidates": [
                candidate.to_dict()
                for candidate in merged.candidates
            ],
        }
        self._record_discard_event(event)
        return selected

    def _choose_discard_with_progressive_evidence(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        decision: Any,
        anchor: DecisionAnchor,
        labels: Sequence[str],
        priors: dict[str, float],
        base_event: dict[str, Any],
        decision_deadline: float,
        started: float,
    ) -> str:
        """Run one progressive evidence chain around the production anchor."""

        evidence_deadline = (
            decision_deadline - self.deadline_guard_ms / 1000.0
        )
        self.last_validation = None
        self.last_validation_error = None
        evidence = self._search_discard_anchor(
            view,
            rules=rules,
            candidate_labels=labels,
            candidate_priors=priors,
            preferred_label=anchor.label,
            absolute_deadline=evidence_deadline,
        )
        self.last_baseline_search = evidence
        proposed_selected = evidence.selected_label
        selected, authorization_reason = (
            self._authorize_progressive_discard(
                production_label=anchor.label,
                proposed_label=proposed_selected,
                evidence=evidence,
                expected_labels=labels,
            )
        )
        self.decision_budget_fallbacks += int(
            proposed_selected != anchor.label
            and selected == anchor.label
            and authorization_reason == "single_pass_evidence_incomplete"
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.concurrent_discard_elapsed_ms += elapsed_ms
        merged = replace(
            evidence,
            selected_label=selected,
            reason=(
                "single_pass_progressive:"
                + evidence.reason
                + ":"
                + authorization_reason
            ),
            elapsed_ms=elapsed_ms,
            confidence_override=bool(
                selected != anchor.label and evidence.confidence_override
            ),
        )
        self.last_search = merged
        self.search_attempts += 1
        self.search_simulations += merged.simulations
        self.search_elapsed_ms += merged.elapsed_ms
        self.search_paired_determinizations += (
            merged.paired_determinizations
        )
        self.search_deadline_interruptions += merged.deadline_interruptions
        self.search_candidate_coverage_failures += int(
            any(item.visits == 0 for item in merged.candidates)
            and "production_prior_full_coverage" not in merged.reason
        )
        self.search_rollout_coverage_failures += (
            merged.rollout_coverage_failures
        )
        self.search_overrides += int(selected != anchor.label)
        self.search_confidence_overrides += int(
            merged.confidence_override
        )
        event = {
            **base_event,
            "search_attempted": True,
            "search_selected_label": selected,
            "proposed_search_selected_label": proposed_selected,
            "final_authorization_reason": authorization_reason,
            "baseline_selected_label": evidence.selected_label,
            "validation_selected_label": None,
            "validation_preferred_label": None,
            "validation_challenger_label": None,
            "validation_challenger_labels": [],
            "validation_complete": False,
            "validation_attempted": False,
            "validation_reconfirmation_attempted": False,
            "validation_confirmation_skipped": False,
            "validation_confirmation_skip_reason": None,
            "validation_confirmation_route": "single_pass_progressive",
            "discard_evidence_route": "single_pass_progressive",
            "validation_diagnostics": None,
            "baseline_diagnostics": _search_stage_diagnostics(evidence),
            "baseline_phase_deadline_interruptions": (
                evidence.deadline_interruptions
            ),
            "validation_error": self.last_validation_error,
            "decision_time_budget_ms": self.decision_time_budget_ms,
            "deadline_guard_ms": self.deadline_guard_ms,
            "confirmation_reserve_ms": 0,
            "late_full_coverage_fallback": False,
            "validation_worker_pending_at_cleanup": False,
            "validation_worker_done_after_cleanup": True,
            "validation_cleanup_elapsed_ms": 0.0,
            "used_search": merged.used_search,
            "reason": merged.reason,
            "simulations": merged.simulations,
            "paired_determinizations": merged.paired_determinizations,
            "deadline_interruptions": merged.deadline_interruptions,
            "rollout_invariant_violations": (
                merged.rollout_invariant_violations
            ),
            "rollout_violations": list(merged.rollout_violations),
            "rollout_coverage_failures": (
                merged.rollout_coverage_failures
            ),
            "rollout_coverage_reasons": list(
                merged.rollout_coverage_reasons
            ),
            "elapsed_ms": round(merged.elapsed_ms, 3),
            "disagreement": selected != decision.selected_label,
            "empirical_best_label": merged.empirical_best_label,
            "confidence_override": merged.confidence_override,
            "paired_advantages": [
                advantage.to_dict()
                for advantage in merged.paired_advantages
            ],
            "candidates": [
                candidate.to_dict()
                for candidate in merged.candidates
            ],
        }
        self._record_discard_event(event)
        return selected

    @staticmethod
    def _authorize_progressive_discard(
        *,
        production_label: str,
        proposed_label: str,
        evidence: RootSearchResult,
        expected_labels: Sequence[str],
    ) -> tuple[str, str]:
        if proposed_label == production_label:
            return production_label, "single_pass_production_kept"
        expected = set(expected_labels)
        observed = {item.label for item in evidence.candidates}
        production_prior_coverage = (
            "production_prior_full_coverage" in evidence.reason
        )
        evidence_complete = bool(
            evidence.used_search
            and evidence.deadline_interruptions == 0
            and evidence.rollout_invariant_violations == 0
            and evidence.rollout_coverage_failures == 0
            and expected == observed
            and (
                production_prior_coverage
                or all(item.visits > 0 for item in evidence.candidates)
            )
            and "overall_budget_minimum_regret" not in evidence.reason
            and "budget_incomplete" not in evidence.reason
        )
        if not evidence_complete:
            return production_label, "single_pass_evidence_incomplete"
        if not evidence.confidence_override:
            return production_label, "single_pass_unconfirmed_override"
        paired = next(
            (
                item
                for item in evidence.paired_advantages
                if item.candidate_key == proposed_label
                and item.preferred_key == production_label
            ),
            None,
        )
        if paired is None or not paired.confidently_positive:
            return production_label, "single_pass_missing_paired_evidence"
        return proposed_label, "single_pass_confidence_authorized"

    def _search_discard_anchor(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        candidate_labels: Sequence[str],
        candidate_priors: dict[str, float],
        preferred_label: str,
        absolute_deadline: float | None,
    ) -> RootSearchResult:
        return self.search.search_discard(
            view,
            rules=rules,
            candidate_labels=candidate_labels,
            candidate_priors=candidate_priors,
            force_search=True,
            paired_candidates=True,
            preferred_label=preferred_label,
            absolute_deadline=absolute_deadline,
        )

    def _authorize_final_discard(
        self,
        *,
        production_label: str,
        proposed_label: str,
        validation: DualDiscardValidationResult | None,
    ) -> tuple[str, str]:
        del production_label, validation
        return proposed_label, "legacy_final_authorization"

    def _should_skip_discard_confirmation(
        self,
        *,
        baseline: RootSearchResult,
        screened: Any,
    ) -> bool:
        del baseline, screened
        return False

    def _discard_confirmation_skip_reason(
        self,
        *,
        production_label: str,
        baseline: RootSearchResult,
        screened: Any,
    ) -> str | None:
        del production_label
        if self._should_skip_discard_confirmation(
            baseline=baseline,
            screened=screened,
        ):
            return "strong_agreement_confirmation_skipped"
        return None

    def _confirm_discard_validation(
        self,
        screened: Any,
        *,
        preferred_label: str,
        time_budget_ms: int,
        absolute_deadline: float | None,
    ) -> tuple[DualDiscardValidationResult, str]:
        return (
            self.discard_validator.confirm_discard(
                screened,
                preferred_label=preferred_label,
                time_budget_ms=time_budget_ms,
                absolute_deadline=absolute_deadline,
            ),
            "full_confirmation",
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            **super().diagnostics(),
            "validation_attempts": self.validation_attempts,
            "validation_overrides": self.validation_overrides,
            "validation_elapsed_ms": self.validation_elapsed_ms,
            "validation_reconfirmations": (
                self.validation_reconfirmations
            ),
            "validation_reconfirmation_elapsed_ms": (
                self.validation_reconfirmation_elapsed_ms
            ),
            "concurrent_discard_elapsed_ms": (
                self.concurrent_discard_elapsed_ms
            ),
            "decision_budget_fallbacks": self.decision_budget_fallbacks,
            "validation_timeouts": self.validation_timeouts,
            "validation_incomplete_fallbacks": (
                self.validation_incomplete_fallbacks
            ),
            "reconfirmation_budget_fallbacks": (
                self.reconfirmation_budget_fallbacks
            ),
            "reconfirmation_incomplete_fallbacks": (
                self.reconfirmation_incomplete_fallbacks
            ),
            "validation_cleanup_waits": self.validation_cleanup_waits,
            "validation_cleanup_elapsed_ms": (
                self.validation_cleanup_elapsed_ms
            ),
            "validation_partial_overrides": (
                self.validation_partial_overrides
            ),
            "late_full_coverage_fallbacks": (
                self.late_full_coverage_fallbacks
            ),
            "validation_confirmation_skips": (
                self.validation_confirmation_skips
            ),
        }


class ProfessionalParallelMultiValidatedCandidatePolicy(
    ProfessionalParallelDualValidatedCandidatePolicy
):
    """Frozen v3.2 calibrator with a reserved confirmation phase."""

    name = "professional_parallel_multi_calibrated_candidate_v3_2_budgeted"

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 6_000,
        evidence_calibrator: (
            DualDiscardEvidenceCalibratorConfig | None
        ) = None,
        response_minimum_confident_advantage: float = 0.02,
    ) -> None:
        super().__init__(
            rollout_policy_factories=rollout_policy_factories,
            validator_config=DualDiscardValidatorConfig(
                coverage_worlds=48,
                confirmation_worlds=112,
                reconfirmation_worlds=96,
                confirmation_challengers=5,
                confirmation_prior_challengers=2,
                reconfirmation_challengers=1,
                parallel_workers=10,
                time_budget_ms=30_000,
                rollout_max_turns=120,
                repeatable_mean_override_threshold=None,
                adaptive_confirmation_worlds_per_second=20.0,
                adaptive_confirmation_min_worlds=32,
                adaptive_confirmation_guard_ms=250,
                evidence_calibrator=(
                    evidence_calibrator
                    or DualDiscardEvidenceCalibratorConfig(
                        feature_mean=(
                            -0.130940537778,
                            0.057360539259,
                            0.102886659259,
                            -0.182383874074,
                            -0.091851851852,
                            -1.869717942222,
                            0.362361782251,
                            0.278436892326,
                        ),
                        feature_scale=(
                            0.206164335325,
                            0.0089161186,
                            0.076400585207,
                            0.209232034877,
                            0.220922391585,
                            9.341063884594,
                            0.278476674352,
                            0.170986951835,
                        ),
                        intercept=-0.129969585185,
                        coefficients=(
                            0.072196250267,
                            -0.003377056604,
                            0.007414678044,
                            0.069784006629,
                            0.044633213944,
                            0.00475275649,
                            -0.004226439627,
                            0.004457218883,
                        ),
                        decision_margin=0.0,
                        trained_confirmation_worlds=112,
                    )
                ),
            ),
            decision_time_budget_ms=decision_time_budget_ms,
            deadline_guard_ms=deadline_guard_ms,
            confirmation_reserve_ms=confirmation_reserve_ms,
            response_minimum_confident_advantage=(
                response_minimum_confident_advantage
            ),
        )


class ProfessionalParallelMultiCalibratedCandidateV4Policy(
    ProfessionalParallelMultiValidatedCandidatePolicy
):
    """Held-out-validated v4 calibrator for the current candidate."""

    name = "professional_parallel_multi_calibrated_candidate_v4"

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 6_000,
    ) -> None:
        super().__init__(
            rollout_policy_factories=rollout_policy_factories,
            decision_time_budget_ms=decision_time_budget_ms,
            deadline_guard_ms=deadline_guard_ms,
            confirmation_reserve_ms=confirmation_reserve_ms,
            evidence_calibrator=DualDiscardEvidenceCalibratorConfig(
                feature_mean=(
                    -0.149124015267,
                    0.057835089521,
                    0.09984529424,
                    -0.199046684941,
                    -0.117314677307,
                    -1.819442352533,
                    0.356111230122,
                    0.278130570948,
                ),
                feature_scale=(
                    0.220940054495,
                    0.009987302376,
                    0.078222618227,
                    0.226845841054,
                    0.244706138791,
                    8.750895858333,
                    0.270692812799,
                    0.163943313288,
                ),
                intercept=-0.14857630118,
                coefficients=(
                    0.081168523185,
                    -0.005049591649,
                    0.006656535454,
                    0.077907661404,
                    0.045965738816,
                    0.001859983576,
                    -0.005145938635,
                    0.003553237576,
                ),
                decision_margin=0.01,
                trained_confirmation_worlds=112,
            ),
        )


class ProfessionalParallelMultiSearchProxyResearchPolicy(
    ProfessionalParallelMultiCalibratedCandidateV4Policy
):
    """Uncalibrated research slot for expanded opponent coverage."""

    name = "professional_parallel_multi_search_proxy_research_v5"


class ProfessionalParallelMultiSparseProxyResearchPolicy(
    ProfessionalParallelMultiCalibratedCandidateV4Policy
):
    """Uncalibrated research slot with a sparse search-opponent prior."""

    name = "professional_parallel_multi_sparse_proxy_research_v5"

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 7_000,
    ) -> None:
        super().__init__(
            rollout_policy_factories=rollout_policy_factories,
            decision_time_budget_ms=decision_time_budget_ms,
            deadline_guard_ms=deadline_guard_ms,
            confirmation_reserve_ms=confirmation_reserve_ms,
        )


class ProfessionalParallelMultiSparseProxyCalibratedV5ResearchPolicy(
    ProfessionalParallelMultiValidatedCandidatePolicy
):
    """Opponent-conditioned v5 calibrator with sparse proxy coverage."""

    name = "professional_parallel_multi_sparse_proxy_calibrated_v5_research"

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 7_000,
    ) -> None:
        super().__init__(
            rollout_policy_factories=rollout_policy_factories,
            decision_time_budget_ms=decision_time_budget_ms,
            deadline_guard_ms=deadline_guard_ms,
            confirmation_reserve_ms=confirmation_reserve_ms,
            evidence_calibrator=DualDiscardEvidenceCalibratorConfig(
                feature_mean=(
                    -0.142637221991,
                    0.057595624283,
                    0.099401343929,
                    -0.192337909328,
                    -0.120216058711,
                    -2.146136677957,
                    0.355364667467,
                    0.279094616373,
                ),
                feature_scale=(
                    0.215212935802,
                    0.010174332761,
                    0.079296458704,
                    0.222306422202,
                    0.243330066516,
                    10.439692723432,
                    0.269012272842,
                    0.162309667716,
                ),
                intercept=-0.163898793643,
                coefficients=(
                    0.076858405184,
                    -0.00725249478,
                    0.012314122598,
                    0.072209682612,
                    0.042480154348,
                    0.003464836176,
                    -0.002606345321,
                    0.016020630681,
                ),
                decision_margin=0.03,
                trained_confirmation_worlds=112,
            ),
        )


class ProfessionalParallelMultiSparseProxyCalibratedV6ResearchPolicy(
    ProfessionalParallelMultiValidatedCandidatePolicy
):
    """Cross-validated v6 calibrator with a conservative blind-test margin."""

    name = "professional_parallel_multi_sparse_proxy_calibrated_v6_research"

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 7_000,
        evidence_decision_margin: float = 0.04,
        response_minimum_confident_advantage: float = 0.02,
        evidence_calibrator_config: (
            DualDiscardEvidenceCalibratorConfig | None
        ) = None,
    ) -> None:
        super().__init__(
            rollout_policy_factories=rollout_policy_factories,
            decision_time_budget_ms=decision_time_budget_ms,
            deadline_guard_ms=deadline_guard_ms,
            confirmation_reserve_ms=confirmation_reserve_ms,
            response_minimum_confident_advantage=(
                response_minimum_confident_advantage
            ),
            evidence_calibrator=(
                evidence_calibrator_config
                or DualDiscardEvidenceCalibratorConfig(
                feature_mean=(
                    -0.137167905378,
                    0.057294685259,
                    0.099234546315,
                    -0.186785190737,
                    -0.114874062085,
                    -2.086079219124,
                    0.357053384033,
                    0.27864162246,
                ),
                feature_scale=(
                    0.21288787832,
                    0.010313525844,
                    0.079211212794,
                    0.220232574116,
                    0.240564702987,
                    10.210360627322,
                    0.2707024722,
                    0.161393630157,
                ),
                intercept=-0.165296432769,
                coefficients=(
                    0.074997176762,
                    -0.000801656143,
                    0.014236318194,
                    0.069935741598,
                    0.043038078959,
                    0.003800234942,
                    -0.003758584783,
                    0.01778649876,
                ),
                decision_margin=evidence_decision_margin,
                trained_confirmation_worlds=112,
                )
            ),
        )


class ProfessionalParallelMultiSparseProxyCalibratedV7ResearchPolicy(
    ProfessionalParallelMultiSparseProxyCalibratedV6ResearchPolicy
):
    """V7 retains the v6 model and rejects marginal blind-test overrides."""

    name = "professional_parallel_multi_sparse_proxy_calibrated_v7_research"

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 7_000,
        response_minimum_confident_advantage: float = 0.02,
    ) -> None:
        super().__init__(
            rollout_policy_factories=rollout_policy_factories,
            decision_time_budget_ms=decision_time_budget_ms,
            deadline_guard_ms=deadline_guard_ms,
            confirmation_reserve_ms=confirmation_reserve_ms,
            evidence_decision_margin=0.05,
            response_minimum_confident_advantage=(
                response_minimum_confident_advantage
            ),
        )


class ProfessionalParallelMultiOpponentRobustV8ResearchPolicy(
    ProfessionalParallelMultiSparseProxyCalibratedV6ResearchPolicy
):
    """V8 expands opponent coverage and calibrates response overrides."""

    name = "professional_parallel_multi_opponent_robust_v8_research"
    evidence_calibrator_config = DualDiscardEvidenceCalibratorConfig(
        feature_mean=(
            -0.130127532654,
            0.057338807781,
            0.097921421491,
            -0.179088255211,
            -0.109096394936,
            -2.003221140343,
            0.359426378521,
            0.278467231009,
        ),
        feature_scale=(
            0.210318783175,
            0.010408255512,
            0.078534569675,
            0.217620979746,
            0.237820000029,
            9.853322547868,
            0.273932753276,
            0.160633132972,
        ),
        intercept=-0.163187077351,
        coefficients=(
            0.072319699814,
            -0.00465816027,
            0.016291474552,
            0.066953314957,
            0.042808253376,
            0.003683908561,
            -0.004040092782,
            0.020804545328,
        ),
        decision_margin=0.14,
        trained_confirmation_worlds=112,
    )

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 7_000,
    ) -> None:
        super().__init__(
            rollout_policy_factories=rollout_policy_factories,
            decision_time_budget_ms=decision_time_budget_ms,
            deadline_guard_ms=deadline_guard_ms,
            confirmation_reserve_ms=confirmation_reserve_ms,
            response_minimum_confident_advantage=0.175,
            evidence_calibrator_config=self.evidence_calibrator_config,
        )


class ProfessionalParallelMultiOpponentRobustV81ResearchPolicy(
    ProfessionalParallelMultiOpponentRobustV8ResearchPolicy
):
    """V8.1 uses same-distribution evidence and measured spare budget."""

    name = "professional_parallel_multi_opponent_robust_v8_1_research"
    evidence_calibrator_config = DualDiscardEvidenceCalibratorConfig(
        feature_mean=(
            -0.181363024444,
            0.058697715556,
            0.098445098519,
            -0.230585580741,
            -0.162514671858,
            -2.718086277778,
            0.35070507065,
            0.282373147429,
            1.0,
        ),
        feature_scale=(
            0.23373240012,
            0.011658319423,
            0.077732114479,
            0.241003560007,
            0.254824263069,
            12.390437621852,
            0.257620390917,
            0.146828696299,
            0.000001,
        ),
        intercept=-0.214135008889,
        coefficients=(
            0.07966510902,
            0.00429955342,
            0.008512115968,
            0.075888806227,
            0.046698412546,
            -0.001241217584,
            -0.005618689106,
            0.02271596429,
            0.0,
        ),
        decision_margin=0.14,
        trained_confirmation_worlds=112,
        minimum_confirmation_fraction=1.0,
    )

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 7_750,
    ) -> None:
        super().__init__(
            rollout_policy_factories=rollout_policy_factories,
            decision_time_budget_ms=decision_time_budget_ms,
            deadline_guard_ms=deadline_guard_ms,
            confirmation_reserve_ms=confirmation_reserve_ms,
        )


class ProfessionalV81TwoPlayerResponseGate014ResearchPolicy(
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy
):
    """V8.1 with the independently replayed two-player response margin."""

    name = "professional_v81_two_player_response_gate_014_research"
    two_player_response_margin = 0.14
    frozen_other_mode_response_margin = 0.175

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._configure_response_margin(players=2)

    def choose_peng(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ) -> bool:
        self._configure_response_margin(
            players=int(rules.get("game", {}).get("players", 3))
        )
        return super().choose_peng(view, label, rules)

    def choose_chi(
        self,
        view: PublicView,
        plans: list[Any],
        rules: dict[str, Any],
    ) -> Any | None:
        self._configure_response_margin(
            players=int(rules.get("game", {}).get("players", 3))
        )
        return super().choose_chi(view, plans, rules)

    def _configure_response_margin(self, *, players: int) -> None:
        margin = (
            self.two_player_response_margin
            if players == 2
            else self.frozen_other_mode_response_margin
        )
        for stage in (
            self.response_search.response_binary_confirmation,
            self.response_search.response_confirmation,
        ):
            if stage.config.minimum_confident_advantage != margin:
                stage.config = replace(
                    stage.config,
                    minimum_confident_advantage=margin,
                )


class ProfessionalV81TwoPlayerWangTingGuardResearchPolicy(
    ProfessionalV81TwoPlayerResponseGate014ResearchPolicy
):
    """Two-player response guards with enough time for full discard coverage."""

    name = "professional_v81_two_player_wang_ting_guard_research"
    equal_wait_peng_confirmation_margin = 0.02
    structural_response_confirmation_margin = 0.02
    expanded_wait_peng_max_danger = 40.0
    early_zero_xi_chi_minimum_stock = 35
    two_player_confirmation_reserve_ms = 3_500
    two_player_adaptive_confirmation_guard_ms = 1_250
    two_player_adaptive_confirmation_worlds_per_second = 15.0
    enable_late_full_coverage_fallback = True
    late_full_coverage_minimum_window_ms = 2_000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault(
            "confirmation_reserve_ms",
            self.two_player_confirmation_reserve_ms,
        )
        super().__init__(*args, **kwargs)
        validator_config = self.discard_validator.config
        worlds_per_second = (
            validator_config.adaptive_confirmation_worlds_per_second
        )
        if (
            validator_config.adaptive_confirmation_guard_ms
            < self.two_player_adaptive_confirmation_guard_ms
            or worlds_per_second is None
            or worlds_per_second
            > self.two_player_adaptive_confirmation_worlds_per_second
        ):
            self.discard_validator.config = replace(
                validator_config,
                adaptive_confirmation_guard_ms=(
                    self.two_player_adaptive_confirmation_guard_ms
                ),
                adaptive_confirmation_worlds_per_second=(
                    self.two_player_adaptive_confirmation_worlds_per_second
                ),
            )
        self.response_search = ExactShardedProgressiveResponsePolicy(
            self.response_search,
            parallel_shards=5,
            parallel_workers=10,
            forced_binary_confirmation=(
                self._no_wang_response_confirmation
            ),
        )

    def _no_wang_response_confirmation(
        self,
        view: PublicView,
        rules: dict[str, Any],
        candidates: Sequence[Any],
        preferred_key: str,
    ) -> tuple[str, float] | None:
        if (
            int(rules.get("game", {}).get("players", 3)) != 2
            or bool(rules.get("wildcard", {}).get("enabled"))
        ):
            return None
        peng_confirmation = self._equal_wait_xi_peng_confirmation(
            view,
            rules,
            candidates,
            preferred_key,
        )
        if peng_confirmation is not None:
            return peng_confirmation
        high_xi_chi_confirmation = (
            self._high_xi_soft_mixed_chi_confirmation(
                view,
                rules,
                candidates,
                preferred_key,
            )
        )
        if high_xi_chi_confirmation is not None:
            return high_xi_chi_confirmation
        return self._early_zero_xi_chi_pass_confirmation(
            view,
            rules,
            candidates,
            preferred_key,
        )

    def _choose_peng_decision(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ) -> Any:
        decision = super()._choose_peng_decision(view, label, rules)
        if not self._should_prefer_pass_for_weak_wang_ting_peng(
            decision,
            view=view,
            rules=rules,
        ):
            return decision
        pass_eval = next(
            item
            for item in decision.action_evals
            if item.allowed and item.action.type == "PASS"
        )
        return replace(
            decision,
            selected_action="PASS",
            selected_card_id=None,
            selected_label=None,
            selected_option_id=None,
            ev=float(pass_eval.ev),
            reason="two_player_wang_ting_weak_peng_preserved",
            requires_action_plan=False,
        )

    @staticmethod
    def _should_prefer_pass_for_weak_wang_ting_peng(
        decision: Any,
        *,
        view: PublicView,
        rules: dict[str, Any],
    ) -> bool:
        if (
            decision.selected_action != "PENG"
            or int(rules.get("game", {}).get("players", 3)) != 2
            or not bool(rules.get("wildcard", {}).get("enabled"))
        ):
            return False
        selected_eval = next(
            (
                item
                for item in decision.action_evals
                if item.allowed and item.action.type == "PENG"
            ),
            None,
        )
        if selected_eval is None:
            return False
        details = selected_eval.debug_details
        ting_before = details.get("ting_before") or {}
        ting_after = details.get("ting_after") or {}
        followup = details.get("followup_discard") or {}
        if (
            not bool(ting_before.get("is_ting"))
            or not bool(ting_after.get("is_ting"))
            or not bool(followup.get("breaks_soft"))
            or bool(followup.get("breaks_hard"))
        ):
            return False
        waiting_before = tuple(ting_before.get("waiting_cards") or ())
        waiting_after = tuple(ting_after.get("waiting_cards") or ())
        if not waiting_before or not waiting_after:
            return False
        remaining = dict(view.remaining_counts)
        before_outs = sum(int(remaining.get(label, 0)) for label in waiting_before)
        after_outs = sum(int(remaining.get(label, 0)) for label in waiting_after)
        before_xi = ting_before.get("expected_xi_if_hu")
        after_xi = ting_after.get("expected_xi_if_hu")
        if before_xi is None or after_xi is None:
            return False
        return (
            after_outs <= before_outs
            and float(after_xi) - float(before_xi) <= 1.0
        )

    def _equal_wait_xi_peng_confirmation(
        self,
        view: PublicView,
        rules: dict[str, Any],
        candidates: Sequence[Any],
        preferred_key: str,
    ) -> tuple[str, float] | None:
        if (
            preferred_key != "PASS"
            or int(rules.get("game", {}).get("players", 3)) != 2
            or bool(rules.get("wildcard", {}).get("enabled"))
            or view.pending_card is None
        ):
            return None
        peng_key = f"PENG:{view.pending_card}"
        if not any(candidate.key == peng_key for candidate in candidates):
            return None
        decision = self._choose_peng_decision(
            view,
            str(view.pending_card),
            rules,
        )
        peng_eval = next(
            (
                item
                for item in decision.action_evals
                if item.action.type == "PENG"
            ),
            None,
        )
        if peng_eval is None:
            return None
        details = peng_eval.debug_details
        ting_before = details.get("ting_before") or {}
        ting_after = details.get("ting_after") or {}
        impact = details.get("consumption_impact") or {}
        followup = details.get("followup_discard") or {}
        waiting_before = tuple(ting_before.get("waiting_cards") or ())
        waiting_after = tuple(ting_after.get("waiting_cards") or ())
        if (
            not bool(ting_before.get("is_ting"))
            or not bool(ting_after.get("is_ting"))
            or not waiting_before
            or not waiting_after
            or bool(impact.get("breaks_hard"))
            or not bool(impact.get("breaks_mixed_same_rank_triplet"))
            or bool(impact.get("breaks_normal_sequence"))
            or bool(impact.get("breaks_special_123"))
            or bool(impact.get("breaks_special_2710"))
            or bool(impact.get("breaks_wildcard_structure"))
            or float(impact.get("red_black_loss") or 0.0) != 0.0
            or bool(followup.get("breaks_hard"))
            or bool(followup.get("breaks_soft"))
        ):
            return None
        broken_soft_melds = tuple(impact.get("breaks_soft_melds") or ())
        if not broken_soft_melds or any(
            meld.get("type") != "mixed_same_rank_triplet"
            or float(meld.get("xi_value") or 0.0) != 0.0
            for meld in broken_soft_melds
        ):
            return None
        remaining = dict(view.remaining_counts)
        before_outs = sum(
            int(remaining.get(label, 0)) for label in waiting_before
        )
        after_outs = sum(
            int(remaining.get(label, 0)) for label in waiting_after
        )
        before_xi = ting_before.get("expected_xi_if_hu")
        after_xi = ting_after.get("expected_xi_if_hu")
        if before_xi is None or after_xi is None:
            return None
        same_waits = set(waiting_after) == set(waiting_before)
        danger_score = float(followup.get("danger_score") or 0.0)
        if same_waits:
            if (
                after_outs < before_outs
                or float(after_xi) - float(before_xi) < 3.0
                or danger_score > 30.0
            ):
                return None
        elif (
            after_outs <= before_outs
            or after_outs - before_outs < 4
            or after_outs < before_outs * 3
            or float(after_xi) < float(before_xi)
            or float(peng_eval.xi_gain or 0.0) < 3.0
            or danger_score > self.expanded_wait_peng_max_danger
        ):
            return None
        return peng_key, self.equal_wait_peng_confirmation_margin

    def _high_xi_soft_mixed_chi_confirmation(
        self,
        view: PublicView,
        rules: dict[str, Any],
        candidates: Sequence[Any],
        preferred_key: str,
    ) -> tuple[str, float] | None:
        preferred = next(
            (candidate for candidate in candidates if candidate.key == preferred_key),
            None,
        )
        if preferred is None or preferred.action_type != "CHI":
            return None
        for candidate in candidates:
            if (
                candidate.key == preferred_key
                or candidate.action_type != "CHI"
                or len(candidate.meld_groups) != 1
                or classify_meld(list(candidate.meld_groups[0])).kind
                != "special_123"
            ):
                continue
            chi_eval = self._chi_candidate_eval(
                view,
                rules,
                candidate,
            )
            if chi_eval is None or float(chi_eval.xi_gain or 0.0) < 6.0:
                continue
            details = chi_eval.debug_details
            ting_before = details.get("ting_before") or {}
            ting_after = details.get("ting_after") or {}
            impact = details.get("consumption_impact") or {}
            followup = details.get("followup_discard") or {}
            before_distance = ting_before.get("distance")
            after_distance = ting_after.get("distance")
            broken_soft_melds = tuple(
                impact.get("breaks_soft_melds") or ()
            )
            if (
                view.stock_count > 34
                or bool(ting_before.get("is_ting"))
                or bool(ting_after.get("is_ting"))
                or before_distance is None
                or after_distance is None
                or int(after_distance) > int(before_distance)
                or bool(impact.get("breaks_hard"))
                or bool(impact.get("breaks_normal_sequence"))
                or bool(impact.get("breaks_special_123"))
                or bool(impact.get("breaks_special_2710"))
                or bool(impact.get("breaks_wildcard_structure"))
                or float(impact.get("red_black_loss") or 0.0) != 3.0
                or len(broken_soft_melds) != 2
                or any(
                    meld.get("type") != "mixed_same_rank_triplet"
                    or float(meld.get("xi_value") or 0.0) != 0.0
                    for meld in broken_soft_melds
                )
                or bool(followup.get("breaks_hard"))
                or bool(followup.get("breaks_soft"))
                or float(followup.get("danger_score") or 0.0) > 40.0
            ):
                continue
            return (
                candidate.key,
                self.structural_response_confirmation_margin,
            )
        return None


    def _early_zero_xi_chi_pass_confirmation(
        self,
        view: PublicView,
        rules: dict[str, Any],
        candidates: Sequence[Any],
        preferred_key: str,
    ) -> tuple[str, float] | None:
        if (
            view.stock_count < self.early_zero_xi_chi_minimum_stock
            or view.own_melds
            or not any(candidate.key == "PASS" for candidate in candidates)
        ):
            return None
        preferred = next(
            (candidate for candidate in candidates if candidate.key == preferred_key),
            None,
        )
        if (
            preferred is None
            or preferred.action_type != "CHI"
            or len(preferred.meld_groups) != 1
            or classify_meld(list(preferred.meld_groups[0])).kind
            != "mixed_same_rank"
        ):
            return None
        chi_eval = self._chi_candidate_eval(view, rules, preferred)
        if chi_eval is None or float(chi_eval.xi_gain or 0.0) != 0.0:
            return None
        details = chi_eval.debug_details
        ting_before = details.get("ting_before") or {}
        ting_after = details.get("ting_after") or {}
        impact = details.get("consumption_impact") or {}
        followup = details.get("followup_discard") or {}
        before_distance = ting_before.get("distance")
        after_distance = ting_after.get("distance")
        raw_candidates = tuple(
            details.get("all_response_candidates") or ()
        )
        after_information_set = (
            raw_candidates[0].get("information_set") or {}
            if raw_candidates
            else {}
        )
        pass_information_set = details.get("pass_information_set") or {}
        if (
            bool(ting_before.get("is_ting"))
            or bool(ting_after.get("is_ting"))
            or before_distance is None
            or after_distance is None
            or int(after_distance) < int(before_distance)
            or bool(impact.get("breaks_hard"))
            or bool(impact.get("breaks_soft"))
            or bool(impact.get("breaks_wildcard_structure"))
            or float(impact.get("red_black_loss") or 0.0) != 0.0
            or bool(followup.get("breaks_hard"))
            or bool(followup.get("breaks_soft"))
            or float(followup.get("danger_score") or 0.0) > 10.0
            or int(after_information_set.get("improving_outs") or 0)
            - int(pass_information_set.get("improving_outs") or 0)
            < 9
        ):
            return None
        return "PASS", self.structural_response_confirmation_margin

    def _chi_candidate_eval(
        self,
        view: PublicView,
        rules: dict[str, Any],
        candidate: Any,
    ) -> Any | None:
        groups = tuple(tuple(group) for group in candidate.meld_groups)
        if not groups or any(len(group) != 3 for group in groups):
            return None
        plan = ChiPlan(
            initial_group=groups[0],
            compare_groups=groups[1:],
            consumed_from_hand=tuple(candidate.consumed_from_hand),
        )
        decision = self._choose_chi_decision(view, [plan], rules)
        return next(
            (
                item
                for item in decision.action_evals
                if item.action.type == "CHI"
            ),
            None,
        )


class ProfessionalV81TwoPlayerExactDiscardShardedResearchPolicy(
    ProfessionalV81TwoPlayerWangTingGuardResearchPolicy
):
    """V8.1 with exact world sharding inside progressive discard stages."""

    name = "professional_v81_two_player_exact_discard_sharded_research"
    parallel_production_evaluation = True
    reuse_validation_coverage_as_baseline = True
    single_pass_progressive_discard_evidence = True
    # The frozen ridge model was trained on the legacy composite reward. It is
    # deliberately disabled until a WIN_FIRST calibrator passes held-out tests.
    two_player_response_margin = 0.14
    confirmation_skip_minimum_coverage_gap = 0.08

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        mobile_shards = _runtime_parallel_shards(20)
        self.opponent_belief_slots = max(
            112,
            len(self.discard_validator.rollout_policy_factories),
        )
        self.last_opponent_belief: RuntimeOpponentBelief | None = None
        self._legacy_evidence_calibrator = (
            self.discard_validator.config.evidence_calibrator
        )
        self.discard_validator.config = replace(
            self.discard_validator.config,
            parallel_workers=20,
            coverage_parallel_shards=_runtime_parallel_shards(12),
            confirmation_parallel_shards=_runtime_parallel_shards(2),
        )
        self.response_search.parallel_shards = mobile_shards
        self.response_search.parallel_workers = 12
        progressive = self.search
        for attribute in (
            "refinement",
            "discard_selection",
            "discard_confirmation",
        ):
            setattr(
                progressive,
                attribute,
                ExactShardedRootDiscardPolicy(
                    getattr(progressive, attribute),
                    parallel_shards=mobile_shards,
                    parallel_workers=12,
                ),
            )
        progressive.discard_confirmation_alternatives = 1
        # Refinement already nominates the strongest challenger from the
        # complete production-anchored top set. Route it directly to the
        # independent binary confirmation instead of repeating a five-arm
        # ranking pass which consumes the mobile foreground deadline.
        progressive.direct_discard_confirmation_from_refinement = True
        progressive.reuse_candidate_priors_as_coverage = True
        confirmation = progressive.discard_confirmation.base
        confirmation_worlds = (
            confirmation.config.max_iterations
            // confirmation.config.max_candidates
        )
        # Keep every confirmation world exact, but submit one world per task so
        # the shared 20-process pool can dynamically absorb uneven rollout
        # lengths instead of waiting on the slowest static five-world shard.
        progressive.discard_confirmation.parallel_shards = (
            _runtime_parallel_shards(confirmation_worlds)
        )
        confirmation.config = replace(
            confirmation.config,
            max_candidates=2,
            max_iterations=confirmation_worlds * 2,
            confidence_familywise_comparisons=5,
        )

    def _configure_opponent_belief(
        self,
        view: PublicView,
        rules: dict[str, Any],
    ) -> None:
        if int(rules.get("game", {}).get("players", 3)) != 2:
            self.last_opponent_belief = None
            return
        factories, belief = runtime_opponent_rollout_schedule(
            view,
            rules,
            slots=self.opponent_belief_slots,
        )
        self.last_opponent_belief = belief
        self.discard_validator.rollout_policy_factories = factories
        _install_rollout_policy_factories(self.search, factories)
        _install_rollout_policy_factories(self.response_search, factories)

    def _attach_opponent_belief_trace(self, *, response: bool) -> None:
        if self.last_opponent_belief is None:
            return
        events = self._response_events if response else self._discard_events
        if events:
            events[-1]["opponent_belief"] = (
                self.last_opponent_belief.to_dict()
            )

    def _configure_objective(self, rules: dict[str, Any]) -> None:
        mode = (
            ObjectiveMode.WIN_FIRST
            if int(rules.get("game", {}).get("players", 3)) == 2
            and bool(rules.get("wildcard", {}).get("enabled"))
            else ObjectiveMode.SCORE_FIRST
        )
        self.objective_mode = mode
        self.objective_version = OBJECTIVE_VERSION
        self.two_player_response_margin = (
            0.0 if mode is ObjectiveMode.WIN_FIRST else 0.14
        )
        self._configure_response_margin(players=2)
        for search in (self.search, self.response_search):
            progressive = getattr(search, "base", search)
            for attribute in (
                "coverage",
                "refinement",
                "discard_selection",
                "discard_confirmation",
                "response_selection",
                "response_binary_confirmation",
                "response_confirmation",
            ):
                policy = getattr(progressive, attribute, None)
                policy = getattr(policy, "base", policy)
                if policy is not None and hasattr(policy, "config"):
                    policy.config = replace(
                        policy.config,
                        objective_mode=mode,
                        objective_version=OBJECTIVE_VERSION,
                    )
        self.discard_validator.config = replace(
            self.discard_validator.config,
            objective_mode=mode,
            objective_version=OBJECTIVE_VERSION,
            evidence_calibrator=(
                None
                if mode is ObjectiveMode.WIN_FIRST
                else self._legacy_evidence_calibrator
            ),
        )

    def choose_discard(
        self,
        view: PublicView,
        rules: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        self._configure_objective(rules)
        self._configure_opponent_belief(view, rules)
        selected = super().choose_discard(view, rules, **kwargs)
        self._attach_opponent_belief_trace(response=False)
        return selected

    def choose_response(
        self,
        view: PublicView,
        legal_actions: Sequence[Any],
        plans: Sequence[ChiPlan],
        hu: Any | None,
        rules: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        self._configure_objective(rules)
        self._configure_opponent_belief(view, rules)
        selected = super().choose_response(
            view,
            legal_actions,
            plans,
            hu,
            rules,
            **kwargs,
        )
        self._attach_opponent_belief_trace(response=True)
        return selected

    def choose_hu(
        self,
        view: PublicView,
        hu: Any,
        rules: dict[str, Any],
    ) -> bool:
        self._configure_objective(rules)
        self._configure_opponent_belief(view, rules)
        selected = super().choose_hu(view, hu, rules)
        self._attach_opponent_belief_trace(response=True)
        return selected

    def _should_skip_discard_confirmation(
        self,
        *,
        baseline: RootSearchResult,
        screened: Any,
    ) -> bool:
        coverage = screened.coverage
        expected_worlds = self.discard_validator.config.coverage_worlds
        ranked = sorted(
            coverage.candidates,
            key=lambda item: (
                item.average_reward,
                item.visits,
                item.heuristic_value,
                item.label,
            ),
            reverse=True,
        )
        return bool(
            baseline.used_search
            and coverage.used_search
            and len(ranked) >= 2
            and coverage.deadline_interruptions == 0
            and coverage.rollout_invariant_violations == 0
            and coverage.rollout_coverage_failures == 0
            and all(item.visits == expected_worlds for item in ranked)
            and baseline.selected_label == ranked[0].label
            and ranked[0].average_reward - ranked[1].average_reward
            >= self.confirmation_skip_minimum_coverage_gap
        )

    def _discard_confirmation_skip_reason(
        self,
        *,
        production_label: str,
        baseline: RootSearchResult,
        screened: Any,
    ) -> str | None:
        existing_reason = super()._discard_confirmation_skip_reason(
            production_label=production_label,
            baseline=baseline,
            screened=screened,
        )
        if (
            existing_reason is not None
            and baseline.selected_label == production_label
        ):
            return existing_reason

        coverage = screened.coverage
        expected_worlds = self.discard_validator.config.coverage_worlds
        coverage_complete = bool(
            coverage.used_search
            and coverage.deadline_interruptions == 0
            and coverage.rollout_invariant_violations == 0
            and coverage.rollout_coverage_failures == 0
            and coverage.paired_determinizations == expected_worlds
            and len(coverage.candidates) == len(screened.labels)
            and all(
                item.visits == expected_worlds
                for item in coverage.candidates
            )
        )
        if not coverage_complete:
            return "final_gate_incomplete_coverage"

        ranked = sorted(
            coverage.candidates,
            key=lambda item: (
                item.average_reward,
                item.visits,
                item.heuristic_value,
                item.label,
            ),
            reverse=True,
        )
        if ranked[0].label == production_label:
            return "final_gate_rank1_is_production"
        return None

    def _confirm_discard_validation(
        self,
        screened: Any,
        *,
        preferred_label: str,
        time_budget_ms: int,
        absolute_deadline: float | None,
    ) -> tuple[DualDiscardValidationResult, str]:
        config = self.discard_validator.config
        available_challengers = sum(
            item.label != preferred_label
            for item in screened.coverage.candidates
        )
        full_challenger_count = min(
            max(1, int(config.confirmation_challengers)),
            available_challengers,
        )
        familywise_comparisons = (
            int(config.confirmation_familywise_comparisons)
            if config.confirmation_familywise_comparisons is not None
            else full_challenger_count
        )
        prefilter_shards = min(
            max(
                int(config.confirmation_parallel_shards),
                max(1, int(config.parallel_workers) // 2),
            ),
            max(1, int(config.confirmation_worlds)),
        )
        prefilter = self.discard_validator.confirm_discard(
            screened,
            preferred_label=preferred_label,
            challenger_limit=1,
            confirmation_shards=prefilter_shards,
            familywise_comparisons=familywise_comparisons,
            time_budget_ms=time_budget_ms,
            absolute_deadline=absolute_deadline,
        )
        evidence = prefilter.selected_challenger_evidence
        if (
            not prefilter.complete
            or evidence is None
            or not evidence.confidence_override
        ):
            return prefilter, "rank1_prefilter_rejected"
        # Coverage already established the top-set ordering and the prefilter
        # independently confirmed its rank-1 challenger against the immutable
        # production anchor with the frozen familywise correction. Repeating
        # that same challenger inside a five-challenger batch adds no evidence
        # required by the final gate and used to consume the remaining deadline.
        # A wider pairwise matrix belongs only to an actually disputed top set;
        # it must not be an unconditional second confirmation pass.
        return prefilter, "rank1_prefilter_complete"

    def _authorize_final_discard(
        self,
        *,
        production_label: str,
        proposed_label: str,
        validation: DualDiscardValidationResult | None,
    ) -> tuple[str, str]:
        if proposed_label == production_label:
            return production_label, "coverage_gate_production_unchanged"
        if validation is None or not validation.complete:
            return production_label, "coverage_gate_incomplete_validation"
        if (
            not validation.confidence_override
            or validation.selected_label != proposed_label
        ):
            return production_label, "coverage_gate_unconfirmed_override"
        evidence = validation.selected_challenger_evidence
        if (
            evidence is None
            or not evidence.confidence_override
            or evidence.challenger_label != proposed_label
        ):
            return production_label, "coverage_gate_missing_selected_evidence"
        return proposed_label, "confirmation_gate_authorized"


class ProfessionalParallelMultiOpponentUnifiedGateResearchPolicy(
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy
):
    """V8.1 search evidence with one production-anchored return gate."""

    name = "professional_parallel_multi_opponent_unified_gate_research"
    minimum_final_familywise_lcb = 0.30

    def _authorize_final_discard(
        self,
        *,
        production_label: str,
        proposed_label: str,
        validation: DualDiscardValidationResult | None,
    ) -> tuple[str, str]:
        return _authorize_final_discard(
            production_label=production_label,
            proposed_label=proposed_label,
            validation=validation,
            minimum_familywise_lcb=self.minimum_final_familywise_lcb,
        )


class ProfessionalParallelMultiOpponentUnifiedGateRank1ResearchPolicy(
    ProfessionalParallelMultiOpponentUnifiedGateResearchPolicy
):
    """Keep the unified gate while confirming only its viable rank-1 input."""

    name = "professional_parallel_multi_opponent_unified_gate_rank1_research"

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ],
        decision_time_budget_ms: int = 14_500,
        deadline_guard_ms: int = 250,
        confirmation_reserve_ms: int = 7_750,
    ) -> None:
        super().__init__(
            rollout_policy_factories=rollout_policy_factories,
            decision_time_budget_ms=decision_time_budget_ms,
            deadline_guard_ms=deadline_guard_ms,
            confirmation_reserve_ms=confirmation_reserve_ms,
        )
        self.discard_validator.config = replace(
            self.discard_validator.config,
            confirmation_challengers=1,
            confirmation_familywise_comparisons=5,
        )


class ProfessionalParallelMultiOpponentProductionAnchoredRank1ResearchPolicy(
    ProfessionalParallelMultiOpponentUnifiedGateRank1ResearchPolicy
):
    """Validate the rank-1 challenger directly against production."""

    name = (
        "professional_parallel_multi_opponent_"
        "production_anchored_rank1_research"
    )

    def _search_discard_anchor(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        candidate_labels: Sequence[str],
        candidate_priors: dict[str, float],
        preferred_label: str,
        absolute_deadline: float | None,
    ) -> RootSearchResult:
        del view, rules, candidate_labels, candidate_priors, absolute_deadline
        return RootSearchResult(
            selected_label=preferred_label,
            used_search=False,
            reason="direct_production_anchor_without_provisional_search",
            simulations=0,
            elapsed_ms=0.0,
            candidates=(),
            empirical_best_label=preferred_label,
        )


class ProfessionalParallelMultiOpponentProductionAnchoredRank1ShardedResearchPolicy(
    ProfessionalParallelMultiOpponentProductionAnchoredRank1ResearchPolicy
):
    """Preserve exact confirmation worlds while evaluating them in shards."""

    name = (
        "professional_parallel_multi_opponent_"
        "production_anchored_rank1_sharded_research"
    )

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.discard_validator.config = replace(
            self.discard_validator.config,
            confirmation_parallel_shards=5,
        )


class ProfessionalParallelMultiOpponentProductionAnchoredRank1ShardedResponseResearchPolicy(
    ProfessionalParallelMultiOpponentProductionAnchoredRank1ShardedResearchPolicy
):
    """Keep the inherited candidate and shard only response confirmation."""

    name = (
        "professional_parallel_multi_opponent_production_anchored_"
        "rank1_sharded_response_research"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.response_search = ExactShardedProgressiveResponsePolicy(
            self.response_search,
            parallel_shards=5,
            parallel_workers=10,
        )


class ProfessionalParallelMultiOpponentRobustV82ResearchPolicy(
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy
):
    """V8.2 rejects formal hard-case harm with a final-fit-safe model."""

    name = "professional_parallel_multi_opponent_robust_v8_2_research"
    evidence_calibrator_config = DualDiscardEvidenceCalibratorConfig(
        feature_mean=(
            -0.16069262062,
            0.059010805256,
            0.098075173181,
            -0.209730217655,
            -0.143449330868,
            -2.632155000674,
            0.356837932209,
            0.282709068036,
            1.0,
        ),
        feature_scale=(
            0.242604336012,
            0.011830194445,
            0.07881096657,
            0.250046014516,
            0.264829378784,
            12.110084908459,
            0.264132566704,
            0.146572289498,
            0.000001,
        ),
        intercept=-0.197710291779,
        coefficients=(
            0.085001364888,
            0.0048375163,
            0.008696830695,
            0.080137637086,
            0.050220859018,
            -0.001699047155,
            -0.008864187275,
            0.02331468477,
            0.0,
        ),
        decision_margin=0.23,
        trained_confirmation_worlds=112,
        minimum_confirmation_fraction=1.0,
    )


class ProfessionalParallelMultiOpponentRobustV83ResearchPolicy(
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy
):
    """V8.3 calibrates search evidence against public opponent style."""

    name = "professional_parallel_multi_opponent_robust_v8_3_research"
    evidence_calibrator_config = DualDiscardEvidenceCalibratorConfig(
        feature_mean=(
            -0.16069262062,
            0.059010805256,
            0.098075173181,
            -0.209730217655,
            -0.143449330868,
            -2.632155000674,
            0.356837932209,
            0.282709068036,
            1.0,
            0.491878753539,
            0.218766846361,
            0.248652291105,
            0.339229559748,
            0.11320754717,
            0.309692273136,
            0.768766846361,
            0.231738544474,
            -0.079568613269,
            -0.043536462433,
            -0.051462469789,
            -0.074552439522,
            -0.017375722709,
            -0.044892052729,
            -0.112913026651,
            -0.047871917958,
        ),
        feature_scale=(
            0.242604336012,
            0.011830194449,
            0.07881096657,
            0.250046014516,
            0.264829378784,
            12.110084908459,
            0.264132566704,
            0.146572289498,
            0.000001,
            0.050553916665,
            0.155686990146,
            0.213048745566,
            0.422164818034,
            0.264767522595,
            0.41812841844,
            0.197012463055,
            0.197763973774,
            0.121893962413,
            0.092252257617,
            0.115563401975,
            0.163468848644,
            0.076459776987,
            0.136806044287,
            0.173983164306,
            0.108595802989,
        ),
        intercept=-0.197710291779,
        coefficients=(
            0.065059213617,
            0.001596488182,
            0.006405137619,
            0.062104101514,
            0.044854110397,
            -0.007504291719,
            -0.00623683122,
            0.024794550552,
            0.0,
            -0.001616997569,
            -0.005684532606,
            -0.130607617786,
            0.007411747024,
            0.00191108729,
            -0.005108209626,
            -0.009896494837,
            0.119476664681,
            -0.135624004148,
            -0.014948199408,
            -0.024942915558,
            -0.010361219055,
            -0.001593252681,
            0.00661513827,
            0.114804307215,
            0.137111184057,
        ),
        decision_margin=0.21,
        trained_confirmation_worlds=112,
        minimum_confirmation_fraction=1.0,
    )


def _authorize_final_discard(
    *,
    production_label: str,
    proposed_label: str,
    validation: DualDiscardValidationResult | None,
    minimum_familywise_lcb: float,
) -> tuple[str, str]:
    if proposed_label == production_label:
        return production_label, "unified_gate_production_unchanged"
    if validation is None:
        return production_label, "unified_gate_missing_validation"
    if not validation.complete:
        return production_label, "unified_gate_incomplete_validation"
    if not validation.confidence_override:
        return production_label, "unified_gate_no_confidence_override"
    if validation.selected_label != proposed_label:
        return production_label, "unified_gate_selected_label_mismatch"
    evidence = validation.selected_challenger_evidence
    if (
        evidence is None
        or not evidence.confidence_override
        or evidence.challenger_label != proposed_label
    ):
        return production_label, "unified_gate_missing_selected_evidence"
    advantage = next(
        (
            item
            for item in evidence.combined.paired_advantages
            if item.candidate_key == proposed_label
            and item.preferred_key == validation.preferred_label
        ),
        None,
    )
    if advantage is None or advantage.lower_confidence_bound is None:
        return production_label, "unified_gate_missing_familywise_lcb"
    if advantage.lower_confidence_bound < float(minimum_familywise_lcb):
        return production_label, "unified_gate_familywise_lcb_below_minimum"
    return proposed_label, "unified_gate_authorized"


def _merge_concurrent_discard_searches(
    baseline: RootSearchResult,
    validation: DualDiscardValidationResult,
    *,
    selected_label: str,
    elapsed_ms: float,
    validation_usable: bool = True,
) -> RootSearchResult:
    confirmation_stages = tuple(
        stage
        for evidence in validation.challenger_evidence
        for stage in (
            evidence.first_confirmation,
            evidence.second_confirmation,
        )
    ) or (
        validation.first_confirmation,
        validation.second_confirmation,
    )
    validation_stages = (
        validation.coverage,
        *confirmation_stages,
    )
    stages = (
        validation_stages
        if baseline is validation.coverage
        else (baseline, *validation_stages)
    )
    return RootSearchResult(
        selected_label=selected_label,
        used_search=validation.coverage.used_search and validation_usable,
        reason=(
            "parallel_dual_hybrid:"
            + baseline.reason
            + ":"
            + validation.combined.reason
        ),
        simulations=sum(stage.simulations for stage in stages),
        elapsed_ms=elapsed_ms,
        candidates=validation.coverage.candidates,
        determinization_failures=sum(
            stage.determinization_failures for stage in stages
        ),
        paired_determinizations=sum(
            stage.paired_determinizations for stage in stages
        ),
        deadline_interruptions=sum(
            stage.deadline_interruptions for stage in validation_stages
        ),
        rollout_invariant_violations=sum(
            stage.rollout_invariant_violations for stage in stages
        ),
        rollout_violations=tuple(
            dict.fromkeys(
                violation
                for stage in stages
                for violation in stage.rollout_violations
            )
        ),
        rollout_coverage_failures=sum(
            stage.rollout_coverage_failures for stage in stages
        ),
        rollout_coverage_reasons=tuple(
            dict.fromkeys(
                reason
                for stage in stages
                for reason in stage.rollout_coverage_reasons
            )
        ),
        empirical_best_label=validation.combined.empirical_best_label,
        confidence_override=(
            validation.confidence_override
            and selected_label != baseline.selected_label
        ),
        paired_advantages=validation.combined.paired_advantages,
        paired_worlds=validation.combined.paired_worlds,
    )


def _remaining_budget_ms(
    deadline: float,
    *,
    guard_ms: int,
) -> int:
    return max(
        0,
        int((deadline - time.perf_counter()) * 1000.0) - guard_ms,
    )


def _incomplete_production_anchor_search(
    labels: Sequence[str],
    *,
    priors: dict[str, float],
    preferred_label: str,
    elapsed_ms: float,
) -> RootSearchResult:
    """Keep the product action safe when shared coverage misses its deadline."""

    return RootSearchResult(
        selected_label=preferred_label,
        used_search=False,
        reason="validation_coverage_unavailable_before_deadline",
        simulations=0,
        elapsed_ms=elapsed_ms,
        candidates=tuple(
            RootCandidateStats(
                label=label,
                visits=0,
                reward_sum=0.0,
                average_reward=0.0,
                win_rate=0.0,
                heuristic_value=float(priors[label]),
            )
            for label in labels
        ),
        empirical_best_label=preferred_label,
    )


def _validation_diagnostics(
    validation: DualDiscardValidationResult | None,
) -> dict[str, Any] | None:
    if validation is None:
        return None
    return {
        "complete": validation.complete,
        "expected_coverage_worlds": (
            validation.expected_coverage_worlds
        ),
        "expected_confirmation_worlds": (
            validation.expected_confirmation_worlds
        ),
        "expected_candidate_count": (
            validation.expected_candidate_count
        ),
        "minimum_confirmation_fraction": (
            validation.minimum_confirmation_fraction
        ),
        "coverage": _search_stage_diagnostics(validation.coverage),
        "first_confirmation": _search_stage_diagnostics(
            validation.first_confirmation
        ),
        "second_confirmation": _search_stage_diagnostics(
            validation.second_confirmation
        ),
        "challengers": [
            {
                "challenger_label": evidence.challenger_label,
                "confidence_override": (
                    evidence.confidence_override
                ),
                "override_basis": evidence.override_basis,
                "calibrated_advantage": (
                    evidence.calibrated_advantage
                ),
                "first_confirmation": _search_stage_diagnostics(
                    evidence.first_confirmation
                ),
                "first_paired_advantages": [
                    advantage.to_dict()
                    for advantage in (
                        evidence.first_confirmation.paired_advantages
                    )
                ],
                "second_confirmation": _search_stage_diagnostics(
                    evidence.second_confirmation
                ),
                "second_paired_advantages": [
                    advantage.to_dict()
                    for advantage in (
                        evidence.second_confirmation.paired_advantages
                    )
                ],
                "combined_selected_label": (
                    evidence.combined.selected_label
                ),
                "combined_paired_advantages": [
                    advantage.to_dict()
                    for advantage in (
                        evidence.combined.paired_advantages
                    )
                ],
            }
            for evidence in validation.challenger_evidence
        ],
    }


def _search_stage_diagnostics(
    search: RootSearchResult,
) -> dict[str, Any]:
    return {
        "reason": search.reason,
        "used_search": search.used_search,
        "simulations": search.simulations,
        "paired_determinizations": search.paired_determinizations,
        "deadline_interruptions": search.deadline_interruptions,
        "rollout_invariant_violations": (
            search.rollout_invariant_violations
        ),
        "rollout_coverage_failures": (
            search.rollout_coverage_failures
        ),
        "candidate_count": len(search.candidates),
        "zero_visit_candidates": sum(
            int(candidate.visits == 0)
            for candidate in search.candidates
        ),
        "elapsed_ms": round(search.elapsed_ms, 3),
    }


__all__ = [
    "ProfessionalParallelDualValidatedCandidatePolicy",
    "ProfessionalParallelMultiValidatedCandidatePolicy",
]
