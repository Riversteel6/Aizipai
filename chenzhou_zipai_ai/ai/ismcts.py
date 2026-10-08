"""Selective root information-set Monte Carlo tree search.

The search samples complete hidden states consistent with ``PublicView`` and
uses UCB at the root. Subsequent decisions use a fast rollout policy. This is a
deliberately bounded first ISMCTS layer, not a policy/value-network substitute.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

from ai.decision_objective import (
    OBJECTIVE_VERSION,
    ObjectiveMode,
    outcome_vector,
    scalar_sampling_reward,
)
from ai.full_game_simulator import (
    BaselinePolicy,
    ChiPlan,
    FullGameSimulator,
    GameResult,
    InformationSetSearchPolicy,
    ProfessionalBrainSimulationPolicy,
    PublicView,
    SimMeld,
    SimPlayer,
    SimulationPolicy,
    _discardable_labels,
    _public_discard_danger,
    _remove_one,
    evaluate_hu,
)
from ai.features import quick_potential
from engine.cards import RED_LABELS, WILD_LABEL, normalize_card_label
from engine.chi_rules import enumerate_chi_plans
from engine.deck import full_deck_counts
from engine.xi_calculator import meld_xi


@dataclass(frozen=True)
class RootISMCTSConfig:
    time_budget_ms: int = 2_500
    max_iterations: int = 12
    max_candidates: int = 2
    exploration: float = 1.25
    skip_search_gap: float = 180.0
    rollout_max_turns: int = 120
    seed: int = 20260726
    require_confident_override: bool = False
    min_confidence_pairs: int = 6
    minimum_confident_advantage: float = 0.0
    confidence_familywise_comparisons: int | None = None
    record_paired_worlds: bool = False
    complete_first_paired_batch: bool = False
    require_complete_iteration_budget_for_override: bool = False
    paired_world_offset: int = 0
    objective_mode: ObjectiveMode = ObjectiveMode.SCORE_FIRST
    objective_version: str = OBJECTIVE_VERSION


@dataclass(frozen=True)
class RootCandidateStats:
    label: str
    visits: int
    reward_sum: float
    average_reward: float
    win_rate: float
    heuristic_value: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    loss_rate: float = 0.0
    draw_rate: float = 0.0
    mean_outcome_score: float = 0.0
    mean_signed_xi: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "visits": self.visits,
            "reward_sum": round(self.reward_sum, 4),
            "average_reward": round(self.average_reward, 4),
            "win_rate": round(self.win_rate, 4),
            "heuristic_value": round(self.heuristic_value, 4),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "loss_rate": round(self.loss_rate, 4),
            "draw_rate": round(self.draw_rate, 4),
            "mean_outcome_score": round(self.mean_outcome_score, 4),
            "mean_signed_xi": round(self.mean_signed_xi, 4),
        }


@dataclass(frozen=True)
class PairedAdvantageStats:
    candidate_key: str
    preferred_key: str
    samples: int
    mean_delta: float
    sample_stddev: float
    standard_error: float
    lower_confidence_bound: float | None
    upper_confidence_bound: float | None
    positive_samples: int
    tied_samples: int
    negative_samples: int
    familywise_comparisons: int = 1
    confidence_level: float = 0.95
    objective_metric: str = "delta_p_win"
    mean_delta_p_win: float | None = None
    mean_delta_p_loss: float | None = None
    mean_delta_score: float | None = None
    mean_delta_xi: float | None = None

    @property
    def confidently_positive(self) -> bool:
        return (
            self.lower_confidence_bound is not None
            and self.lower_confidence_bound > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "preferred_key": self.preferred_key,
            "samples": self.samples,
            "mean_delta": round(self.mean_delta, 6),
            "sample_stddev": round(self.sample_stddev, 6),
            "standard_error": round(self.standard_error, 6),
            "lower_confidence_bound": (
                round(self.lower_confidence_bound, 6)
                if self.lower_confidence_bound is not None
                else None
            ),
            "upper_confidence_bound": (
                round(self.upper_confidence_bound, 6)
                if self.upper_confidence_bound is not None
                else None
            ),
            "positive_samples": self.positive_samples,
            "tied_samples": self.tied_samples,
            "negative_samples": self.negative_samples,
            "familywise_comparisons": self.familywise_comparisons,
            "confidence_level": self.confidence_level,
            "confidently_positive": self.confidently_positive,
            "objective_metric": self.objective_metric,
            "mean_delta_p_win": (
                round(self.mean_delta_p_win, 6)
                if self.mean_delta_p_win is not None
                else round(self.mean_delta, 6)
                if self.objective_metric == "delta_p_win"
                else None
            ),
            "mean_delta_p_loss": (
                round(self.mean_delta_p_loss, 6)
                if self.mean_delta_p_loss is not None
                else None
            ),
            "mean_delta_score": (
                round(self.mean_delta_score, 6)
                if self.mean_delta_score is not None
                else None
            ),
            "mean_delta_xi": (
                round(self.mean_delta_xi, 6)
                if self.mean_delta_xi is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PairedCandidateOutcome:
    candidate_key: str
    reward: float
    winner: int | None
    score: float
    total_xi: int
    turns: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "reward": round(self.reward, 6),
            "winner": self.winner,
            "score": round(self.score, 6),
            "total_xi": self.total_xi,
            "turns": self.turns,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PairedWorldOutcome:
    world_index: int
    world_fingerprint: str
    rollout_seed: int
    opponent_policy: str
    outcomes: tuple[PairedCandidateOutcome, ...]
    root_continuation_policy: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_index": self.world_index,
            "world_fingerprint": self.world_fingerprint,
            "rollout_seed": self.rollout_seed,
            "opponent_policy": self.opponent_policy,
            "root_continuation_policy": self.root_continuation_policy,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True)
class RootSearchResult:
    selected_label: str
    used_search: bool
    reason: str
    simulations: int
    elapsed_ms: float
    candidates: tuple[RootCandidateStats, ...]
    determinization_failures: int = 0
    paired_determinizations: int = 0
    deadline_interruptions: int = 0
    rollout_invariant_violations: int = 0
    rollout_violations: tuple[str, ...] = ()
    rollout_coverage_failures: int = 0
    rollout_coverage_reasons: tuple[str, ...] = ()
    empirical_best_label: str | None = None
    confidence_override: bool = False
    paired_advantages: tuple[PairedAdvantageStats, ...] = ()
    paired_worlds: tuple[PairedWorldOutcome, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_label": self.selected_label,
            "used_search": self.used_search,
            "reason": self.reason,
            "simulations": self.simulations,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "determinization_failures": self.determinization_failures,
            "paired_determinizations": self.paired_determinizations,
            "deadline_interruptions": self.deadline_interruptions,
            "rollout_invariant_violations": self.rollout_invariant_violations,
            "rollout_violations": list(self.rollout_violations),
            "rollout_coverage_failures": self.rollout_coverage_failures,
            "rollout_coverage_reasons": list(self.rollout_coverage_reasons),
            "empirical_best_label": self.empirical_best_label,
            "confidence_override": self.confidence_override,
            "paired_advantages": [
                advantage.to_dict()
                for advantage in self.paired_advantages
            ],
            "paired_worlds": [
                world.to_dict()
                for world in self.paired_worlds
            ],
        }


@dataclass(frozen=True)
class DiscardResponseRiskCandidateStats:
    label: str
    samples: int
    pass_count: int
    claim_counts: tuple[tuple[str, int], ...]
    claim_seat_counts: tuple[tuple[str, int, int], ...]
    risk_sum: float
    expected_risk: float
    heuristic_value: float
    adjusted_value: float
    root_seat: int
    player_count: int

    def to_dict(self) -> dict[str, Any]:
        claims = dict(self.claim_counts)
        return {
            "label": self.label,
            "samples": self.samples,
            "pass_count": self.pass_count,
            "claim_counts": claims,
            "claim_rate": round(
                sum(claims.values()) / self.samples,
                4,
            )
            if self.samples
            else 0.0,
            "claim_rates": {
                action: round(count / self.samples, 4) if self.samples else 0.0
                for action, count in self.claim_counts
            },
            "claims_by_seat": [
                {
                    "action": action,
                    "seat": seat,
                    "relative_offset": (seat - self.root_seat) % self.player_count,
                    "is_next_seat": (seat - self.root_seat) % self.player_count == 1,
                    "count": count,
                    "rate": round(count / self.samples, 4) if self.samples else 0.0,
                }
                for action, seat, count in self.claim_seat_counts
            ],
            "risk_sum": round(self.risk_sum, 4),
            "expected_risk": round(self.expected_risk, 4),
            "heuristic_value": round(self.heuristic_value, 4),
            "adjusted_value": round(self.adjusted_value, 4),
        }


@dataclass(frozen=True)
class DiscardResponseRiskResult:
    selected_label: str
    used_search: bool
    reason: str
    simulations: int
    elapsed_ms: float
    candidates: tuple[DiscardResponseRiskCandidateStats, ...]
    determinization_failures: int = 0
    paired_determinizations: int = 0
    deadline_interruptions: int = 0
    rollout_invariant_violations: int = 0
    rollout_violations: tuple[str, ...] = ()
    rollout_coverage_failures: int = 0
    rollout_coverage_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_label": self.selected_label,
            "used_search": self.used_search,
            "reason": self.reason,
            "simulations": self.simulations,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "determinization_failures": self.determinization_failures,
            "paired_determinizations": self.paired_determinizations,
            "deadline_interruptions": self.deadline_interruptions,
            "rollout_invariant_violations": self.rollout_invariant_violations,
            "rollout_violations": list(self.rollout_violations),
            "rollout_coverage_failures": self.rollout_coverage_failures,
            "rollout_coverage_reasons": list(self.rollout_coverage_reasons),
        }


@dataclass(frozen=True)
class RootResponseCandidate:
    key: str
    action_type: str
    heuristic_value: float
    option_id: str | None = None
    consumed_from_hand: tuple[str, ...] = ()
    meld_groups: tuple[tuple[str, ...], ...] = ()
    followup_discard: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "action_type": self.action_type,
            "heuristic_value": round(self.heuristic_value, 4),
            "option_id": self.option_id,
            "consumed_from_hand": list(self.consumed_from_hand),
            "meld_groups": [list(group) for group in self.meld_groups],
            "followup_discard": self.followup_discard,
        }


@dataclass(frozen=True)
class RootResponseCandidateStats:
    candidate: RootResponseCandidate
    visits: int
    reward_sum: float
    average_reward: float
    win_rate: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    loss_rate: float = 0.0
    draw_rate: float = 0.0
    mean_outcome_score: float = 0.0
    mean_signed_xi: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "visits": self.visits,
            "reward_sum": round(self.reward_sum, 4),
            "average_reward": round(self.average_reward, 4),
            "win_rate": round(self.win_rate, 4),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "loss_rate": round(self.loss_rate, 4),
            "draw_rate": round(self.draw_rate, 4),
            "mean_outcome_score": round(self.mean_outcome_score, 4),
            "mean_signed_xi": round(self.mean_signed_xi, 4),
        }


@dataclass(frozen=True)
class RootResponseSearchResult:
    selected_key: str
    used_search: bool
    reason: str
    simulations: int
    elapsed_ms: float
    candidates: tuple[RootResponseCandidateStats, ...]
    determinization_failures: int = 0
    paired_determinizations: int = 0
    deadline_interruptions: int = 0
    rollout_invariant_violations: int = 0
    rollout_violations: tuple[str, ...] = ()
    rollout_coverage_failures: int = 0
    rollout_coverage_reasons: tuple[str, ...] = ()
    empirical_best_key: str | None = None
    confidence_override: bool = False
    paired_advantages: tuple[PairedAdvantageStats, ...] = ()
    paired_worlds: tuple[PairedWorldOutcome, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_key": self.selected_key,
            "used_search": self.used_search,
            "reason": self.reason,
            "simulations": self.simulations,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "determinization_failures": self.determinization_failures,
            "paired_determinizations": self.paired_determinizations,
            "deadline_interruptions": self.deadline_interruptions,
            "rollout_invariant_violations": self.rollout_invariant_violations,
            "rollout_violations": list(self.rollout_violations),
            "rollout_coverage_failures": self.rollout_coverage_failures,
            "rollout_coverage_reasons": list(self.rollout_coverage_reasons),
            "empirical_best_key": self.empirical_best_key,
            "confidence_override": self.confidence_override,
            "paired_advantages": [
                advantage.to_dict()
                for advantage in self.paired_advantages
            ],
            "paired_worlds": [
                world.to_dict()
                for world in self.paired_worlds
            ],
        }


def build_response_candidates(
    pending_card: str,
    evals: Sequence[Any],
    *,
    include_policy_rejected: bool = False,
) -> list[RootResponseCandidate]:
    """Build exact PASS/CHI/PENG commitments from production action evaluations."""

    pending = normalize_card_label(str(pending_card or ""))
    by_key: dict[str, RootResponseCandidate] = {}
    for item in evals:
        action_type = str(item.type).upper()
        if action_type not in {"PASS", "CHI", "PENG"}:
            continue
        if (
            not include_policy_rejected
            and not item.allowed
            and item.reject_reason not in {"chi_ev_not_enough", "peng_ev_not_enough"}
        ):
            continue
        if action_type == "PASS":
            candidate = RootResponseCandidate(
                key="PASS",
                action_type="PASS",
                heuristic_value=float(item.ev),
            )
        else:
            debug = item.debug_details
            consumed = tuple(_normalized_labels(debug.get("consumed_from_hand") or []))
            followup_raw = debug.get("followup_discard")
            followup = (
                normalize_card_label(str(followup_raw.get("label") or ""))
                if isinstance(followup_raw, Mapping)
                else ""
            )
            if not consumed or not followup:
                continue
            if action_type == "PENG":
                if not pending:
                    continue
                groups = ((pending, pending, pending),)
                key = f"PENG:{pending}"
            else:
                groups = _response_meld_groups(debug)
                option_id = item.action.option_id or item.action.action_id
                if not groups or not option_id:
                    continue
                key = f"CHI:{option_id}"
            candidate = RootResponseCandidate(
                key=key,
                action_type=action_type,
                heuristic_value=_response_candidate_prior(item),
                option_id=item.action.option_id,
                consumed_from_hand=consumed,
                meld_groups=groups,
                followup_discard=followup,
            )
        previous = by_key.get(candidate.key)
        if previous is None or candidate.heuristic_value > previous.heuristic_value:
            by_key[candidate.key] = candidate
    return sorted(
        by_key.values(),
        key=lambda candidate: (
            candidate.heuristic_value,
            candidate.key,
        ),
        reverse=True,
    )


def shortlist_response_candidates(
    candidates: Sequence[RootResponseCandidate],
    *,
    production_key: str,
    max_candidates: int,
) -> list[RootResponseCandidate]:
    """Keep the production commitment plus the strongest alternatives."""

    limit = max(2, max_candidates)
    production = next(
        (candidate for candidate in candidates if candidate.key == production_key),
        None,
    )
    if production is None:
        return []
    shortlist = list(candidates[:limit])
    if all(candidate.key != production_key for candidate in shortlist):
        shortlist[-1] = production
    return shortlist


def _ensure_unbounded_root_capacity(
    search: Any,
    *,
    candidate_count: int,
    unbounded: bool,
) -> None:
    if (
        not unbounded
        or candidate_count <= 0
        or not isinstance(search, RootISMCTSPolicy)
    ):
        return
    config = search.config
    target_pairs = (
        max(1, config.min_confidence_pairs)
        if config.require_confident_override
        else 1
    )
    required_iterations = candidate_count * target_pairs
    if (
        config.max_candidates < candidate_count
        or config.max_iterations < required_iterations
    ):
        search.config = replace(
            config,
            max_candidates=max(config.max_candidates, candidate_count),
            max_iterations=max(config.max_iterations, required_iterations),
        )


def public_view_to_dict(view: PublicView) -> dict[str, Any]:
    return {
        "seat": view.seat,
        "hand": list(view.hand),
        "all_melds": [
            [meld.to_dict() for meld in melds]
            for melds in view.all_melds
        ],
        "discards": [list(discards) for discards in view.discards],
        "remaining_counts": [list(item) for item in view.remaining_counts],
        "stock_count": view.stock_count,
        "hand_sizes": list(view.hand_sizes),
        "pending_card": view.pending_card,
        "pending_source_seat": view.pending_source_seat,
        "passed_chi": [list(labels) for labels in view.passed_chi],
        "passed_peng": [list(labels) for labels in view.passed_peng],
    }


def public_view_from_dict(payload: Mapping[str, Any]) -> PublicView:
    all_melds = tuple(
        tuple(_sim_meld_from_dict(meld) for meld in melds)
        for melds in payload.get("all_melds") or []
    )
    seat = int(payload["seat"])
    if not 0 <= seat < len(all_melds):
        raise ValueError("public_view_seat_out_of_range")
    return PublicView(
        seat=seat,
        hand=tuple(str(label) for label in payload.get("hand") or []),
        own_melds=all_melds[seat],
        all_melds=all_melds,
        discards=tuple(
            tuple(str(label) for label in discards)
            for discards in payload.get("discards") or []
        ),
        remaining_counts=tuple(
            (str(label), int(amount))
            for label, amount in payload.get("remaining_counts") or []
        ),
        stock_count=int(payload.get("stock_count") or 0),
        hand_sizes=tuple(int(size) for size in payload.get("hand_sizes") or []),
        pending_card=(
            str(payload["pending_card"])
            if payload.get("pending_card") is not None
            else None
        ),
        pending_source_seat=(
            int(payload["pending_source_seat"])
            if payload.get("pending_source_seat") is not None
            else None
        ),
        passed_chi=tuple(
            tuple(str(label) for label in labels)
            for labels in payload.get("passed_chi") or []
        ),
        passed_peng=tuple(
            tuple(str(label) for label in labels)
            for labels in payload.get("passed_peng") or []
        ),
    )


def _sim_meld_from_dict(payload: Mapping[str, Any]) -> SimMeld:
    kind = str(payload.get("type") or payload.get("kind") or "")
    cards = tuple(str(label) for label in payload.get("labels") or payload.get("cards") or [])
    if not kind or not cards:
        raise ValueError("invalid_public_view_meld")
    return SimMeld(kind, cards)


def _response_meld_groups(
    debug: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    plan_melds = debug.get("plan_melds")
    groups: list[tuple[str, ...]] = []
    if isinstance(plan_melds, Sequence) and not isinstance(plan_melds, (str, bytes)):
        for meld in plan_melds:
            if not isinstance(meld, Mapping):
                return ()
            group = tuple(_normalized_labels(meld.get("cards") or meld.get("labels") or []))
            if len(group) != 3:
                return ()
            groups.append(group)
    if groups:
        return tuple(groups)
    initial = tuple(_normalized_labels(debug.get("meld_cards") or []))
    compares = [
        tuple(_normalized_labels(group))
        for group in debug.get("compare_groups") or []
    ]
    if len(initial) != 3 or any(len(group) != 3 for group in compares):
        return ()
    return (initial, *compares)


def _response_candidate_prior(item: Any) -> float:
    if item.allowed:
        return float(item.ev)
    rows = item.debug_details.get("all_response_candidates") or []
    values = [
        float(row["ev"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("ev"), (int, float))
    ]
    return max(values, default=float(item.ev))


def _normalized_labels(values: Sequence[Any]) -> list[str]:
    labels: list[str] = []
    for raw in values:
        label = normalize_card_label(str(raw))
        if label:
            labels.append(label)
    return labels


class RootSelfContinuationPolicy(InformationSetSearchPolicy):
    """Low-latency self-policy approximation used after the forced root action."""

    name = "root_self_continuation_v1"


class RootISMCTSPolicy(InformationSetSearchPolicy):
    """Full-rollout information-set search for uncertain discard decisions."""

    name = "root_ismcts_v1"

    def __init__(
        self,
        config: RootISMCTSConfig | None = None,
        *,
        rollout_policy_factory: Callable[[], SimulationPolicy] | None = None,
        rollout_policy_factories: Sequence[Callable[[], SimulationPolicy]] | None = None,
        root_continuation_policy_factory: Callable[[], SimulationPolicy] | None = None,
    ) -> None:
        if rollout_policy_factory is not None and rollout_policy_factories is not None:
            raise ValueError("choose_one_rollout_policy_factory_argument")
        resolved_factories = tuple(rollout_policy_factories or ())
        if not resolved_factories:
            resolved_factories = (rollout_policy_factory or BaselinePolicy,)
        self.config = config or RootISMCTSConfig()
        self.rollout_policy_factories = resolved_factories
        self.rollout_policy_factory = resolved_factories[0]
        self.root_continuation_policy_factory = (
            root_continuation_policy_factory
            or RootSelfContinuationPolicy
        )
        self.last_search: RootSearchResult | None = None

    def _opponent_rollout_factory(
        self,
        sample_index: int,
    ) -> Callable[[], SimulationPolicy]:
        return self.rollout_policy_factories[
            sample_index % len(self.rollout_policy_factories)
        ]

    def _root_continuation_policy(self) -> SimulationPolicy:
        return self.root_continuation_policy_factory()

    def _rollout_policies(
        self,
        *,
        player_count: int,
        root_seat: int,
        root_policy: SimulationPolicy,
        opponent_policy_factory: Callable[[], SimulationPolicy] | None = None,
    ) -> list[SimulationPolicy]:
        factory = opponent_policy_factory or self.rollout_policy_factory
        return [
            root_policy if seat == root_seat else factory()
            for seat in range(player_count)
        ]

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        result = self.search_discard(view, rules=rules)
        self.last_search = result
        return result.selected_label

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
        started = time.perf_counter()
        legal = set(_discardable_labels(view.hand))
        labels = list(dict.fromkeys(candidate_labels or sorted(legal)))
        labels = [label for label in labels if label in legal]
        if not labels:
            raise ValueError("no_discardable_card")
        if preferred_label is not None and preferred_label not in labels:
            raise ValueError("preferred_discard_candidate_missing")
        if self.config.require_confident_override and not paired_candidates:
            raise ValueError("confidence_override_requires_paired_candidates")
        heuristic = {
            label: (
                float(candidate_priors[label])
                if candidate_priors is not None and label in candidate_priors
                else _root_heuristic_value(view, label, rules)
            )
            for label in labels
        }
        ranked_labels = sorted(
            labels,
            key=lambda label: (heuristic[label], label),
            reverse=True,
        )
        shortlist = ranked_labels[: max(1, self.config.max_candidates)]
        if preferred_label is not None and preferred_label not in shortlist:
            shortlist = [
                preferred_label,
                *(
                    label
                    for label in ranked_labels
                    if label != preferred_label
                ),
            ][: max(1, self.config.max_candidates)]
        fallback_label = preferred_label or shortlist[0]
        if len(shortlist) == 1:
            return _no_search_result(
                fallback_label,
                "single_legal_candidate",
                heuristic,
                started,
            )
        gap = heuristic[shortlist[0]] - heuristic[shortlist[1]]
        if not force_search and gap >= self.config.skip_search_gap:
            return _no_search_result(
                fallback_label,
                f"heuristic_gap={round(gap, 3)}",
                heuristic,
                started,
            )
        if paired_candidates:
            return self._search_discard_paired(
                view,
                rules=rules,
                shortlist=shortlist,
                heuristic=heuristic,
                started=started,
                preferred_label=fallback_label,
                absolute_deadline=absolute_deadline,
            )

        visits = Counter({label: 0 for label in shortlist})
        rewards = Counter({label: 0.0 for label in shortlist})
        wins = Counter({label: 0 for label in shortlist})
        losses = Counter({label: 0 for label in shortlist})
        draws = Counter({label: 0 for label in shortlist})
        outcome_scores = Counter({label: 0.0 for label in shortlist})
        signed_xi = Counter({label: 0.0 for label in shortlist})
        deadline = _bounded_search_deadline(
            started,
            time_budget_ms=self.config.time_budget_ms,
            absolute_deadline=absolute_deadline,
        )
        rng = random.Random(information_set_seed(view, self.config.seed))
        simulations = 0
        failures = 0
        rollout_coverage_failures = 0
        rollout_coverage_reasons: list[str] = []
        while simulations < max(1, self.config.max_iterations) and time.perf_counter() < deadline:
            label = _select_root_arm(
                shortlist,
                visits,
                rewards,
                heuristic,
                exploration=self.config.exploration,
            )
            determinization = determinize_public_view(view, rules=rules, rng=rng)
            if determinization is None:
                failures += 1
                break
            players, stock = determinization
            player_count = int(rules.get("game", {}).get("players", 3))
            rollout_factory = self._opponent_rollout_factory(simulations)
            policies = self._rollout_policies(
                player_count=player_count,
                root_seat=view.seat,
                root_policy=_ForcedFirstDiscardPolicy(
                    label,
                    continuation_policy=self._root_continuation_policy(),
                ),
                opponent_policy_factory=rollout_factory,
            )
            simulator = FullGameSimulator(
                policies,
                wildcard_enabled=bool(rules.get("wildcard", {}).get("enabled", False)),
                dealer=view.seat,
                rules=rules,
            )
            result = simulator.play_from_state(
                seed=rng.randrange(1, 2**31),
                players=players,
                stock=stock,
                current=view.seat,
                max_turns=self.config.rollout_max_turns,
            )
            if result.coverage_failures:
                rollout_coverage_failures += len(result.coverage_failures)
                rollout_coverage_reasons.extend(result.coverage_failures)
                break
            reward = _root_reward(
                result,
                root_seat=view.seat,
                objective_mode=self.config.objective_mode,
            )
            visits[label] += 1
            rewards[label] += reward
            wins[label] += int(result.winner == view.seat)
            losses[label] += int(
                result.winner is not None
                and result.winner != view.seat
            )
            draws[label] += int(result.winner is None)
            outcome_scores[label] += _root_outcome_score(
                result,
                root_seat=view.seat,
            )
            signed_xi[label] += _root_signed_xi(
                result,
                root_seat=view.seat,
            )
            simulations += 1

        if rollout_coverage_failures:
            return RootSearchResult(
                selected_label=shortlist[0],
                used_search=False,
                reason="rollout_coverage_incomplete",
                simulations=simulations,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=tuple(
                    RootCandidateStats(
                        label=label,
                        visits=visits[label],
                        reward_sum=rewards[label],
                        average_reward=rewards[label] / max(1, visits[label]),
                        win_rate=wins[label] / max(1, visits[label]),
                        heuristic_value=heuristic[label],
                        wins=wins[label],
                        losses=losses[label],
                        draws=draws[label],
                        loss_rate=losses[label] / max(1, visits[label]),
                        draw_rate=draws[label] / max(1, visits[label]),
                        mean_outcome_score=outcome_scores[label] / max(1, visits[label]),
                        mean_signed_xi=signed_xi[label] / max(1, visits[label]),
                    )
                    for label in shortlist
                ),
                determinization_failures=failures,
                rollout_coverage_failures=rollout_coverage_failures,
                rollout_coverage_reasons=tuple(dict.fromkeys(rollout_coverage_reasons)),
            )
        if simulations == 0:
            return RootSearchResult(
                selected_label=shortlist[0],
                used_search=False,
                reason="determinization_unavailable",
                simulations=0,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=tuple(
                    RootCandidateStats(
                        label=label,
                        visits=0,
                        reward_sum=0.0,
                        average_reward=0.0,
                        win_rate=0.0,
                        heuristic_value=heuristic[label],
                    )
                    for label in shortlist
                ),
                determinization_failures=failures,
            )
        stats = tuple(
            RootCandidateStats(
                label=label,
                visits=visits[label],
                reward_sum=rewards[label],
                average_reward=rewards[label] / max(1, visits[label]),
                win_rate=wins[label] / max(1, visits[label]),
                heuristic_value=heuristic[label],
                wins=wins[label],
                losses=losses[label],
                draws=draws[label],
                loss_rate=losses[label] / max(1, visits[label]),
                draw_rate=draws[label] / max(1, visits[label]),
                mean_outcome_score=outcome_scores[label] / max(1, visits[label]),
                mean_signed_xi=signed_xi[label] / max(1, visits[label]),
            )
            for label in shortlist
        )
        if any(item.visits == 0 for item in stats):
            return RootSearchResult(
                selected_label=shortlist[0],
                used_search=False,
                reason="candidate_coverage_incomplete",
                simulations=simulations,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=stats,
                determinization_failures=failures,
            )
        selected = max(
            stats,
            key=lambda item: (
                _candidate_rank_key(item),
                item.visits,
                item.heuristic_value,
                item.label,
            ),
        )
        return RootSearchResult(
            selected_label=selected.label,
            used_search=True,
            reason="root_ismcts_completed",
            simulations=simulations,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            candidates=tuple(
                sorted(
                    stats,
                    key=lambda item: (
                        _candidate_rank_key(item),
                        item.visits,
                        item.heuristic_value,
                    ),
                    reverse=True,
                )
            ),
            determinization_failures=failures,
        )

    def _search_discard_paired(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        shortlist: list[str],
        heuristic: dict[str, float],
        started: float,
        preferred_label: str,
        absolute_deadline: float | None = None,
    ) -> RootSearchResult:
        if max(1, self.config.max_iterations) < len(shortlist):
            return RootSearchResult(
                selected_label=preferred_label,
                used_search=False,
                reason="paired_candidate_budget_too_small",
                simulations=0,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=tuple(
                    RootCandidateStats(
                        label=label,
                        visits=0,
                        reward_sum=0.0,
                        average_reward=0.0,
                        win_rate=0.0,
                        heuristic_value=heuristic[label],
                    )
                    for label in shortlist
                ),
            )

        visits = Counter({label: 0 for label in shortlist})
        rewards = Counter({label: 0.0 for label in shortlist})
        wins = Counter({label: 0 for label in shortlist})
        losses = Counter({label: 0 for label in shortlist})
        draws = Counter({label: 0 for label in shortlist})
        outcome_scores = Counter({label: 0.0 for label in shortlist})
        signed_xi = Counter({label: 0.0 for label in shortlist})
        deadline = _bounded_search_deadline(
            started,
            time_budget_ms=self.config.time_budget_ms,
            absolute_deadline=absolute_deadline,
        )
        rng = random.Random(information_set_seed(view, self.config.seed))
        simulations = 0
        failures = 0
        paired_determinizations = 0
        deadline_interruptions = 0
        rollout_invariant_violations = 0
        rollout_violations: list[str] = []
        rollout_coverage_failures = 0
        rollout_coverage_reasons: list[str] = []
        paired_reward_batches: list[dict[str, float]] = []
        paired_outcome_batches: list[dict[str, Any]] = []
        paired_worlds: list[PairedWorldOutcome] = []
        paired_world_offset = max(0, int(self.config.paired_world_offset))
        for _ in range(paired_world_offset):
            determinization = determinize_public_view(view, rules=rules, rng=rng)
            if determinization is None:
                failures += 1
                break
            rng.randrange(1, 2**31)
        if failures:
            return RootSearchResult(
                selected_label=preferred_label,
                used_search=False,
                reason="determinization_unavailable_before_paired_shard",
                simulations=0,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=tuple(
                    RootCandidateStats(
                        label=label,
                        visits=0,
                        reward_sum=0.0,
                        average_reward=0.0,
                        win_rate=0.0,
                        heuristic_value=heuristic[label],
                    )
                    for label in shortlist
                ),
                determinization_failures=failures,
            )
        while simulations + len(shortlist) <= max(1, self.config.max_iterations):
            complete_first_batch = (
                self.config.complete_first_paired_batch
                and paired_determinizations == 0
                and (
                    absolute_deadline is None
                    or time.perf_counter() < absolute_deadline
                )
            )
            batch_deadline = (
                absolute_deadline
                if complete_first_batch and absolute_deadline is not None
                else None
                if complete_first_batch
                else deadline
            )
            if (
                batch_deadline is not None
                and time.perf_counter() >= batch_deadline
            ):
                break
            determinization = determinize_public_view(view, rules=rules, rng=rng)
            if determinization is None:
                failures += 1
                break
            base_players, base_stock = determinization
            rollout_seed = rng.randrange(1, 2**31)
            opponent_policy_factory = self._opponent_rollout_factory(
                paired_world_offset + paired_determinizations
            )
            opponent_policy_name = "unknown"
            root_continuation_policy_name = "unknown"
            batch_results: list[tuple[str, GameResult]] = []
            paired_hu_cache: dict[
                tuple[tuple[str, ...], tuple[SimMeld, ...]],
                Any,
            ] = {}
            paired_position_cache: dict[
                tuple[
                    tuple[str, ...],
                    tuple[SimMeld, ...],
                    tuple[tuple[str, int], ...],
                ],
                float,
            ] = {}
            paired_followup_cache: dict[
                tuple[tuple[str, ...], tuple[SimMeld, ...]],
                float,
            ] = {}
            for label in shortlist:
                if (
                    batch_deadline is not None
                    and time.perf_counter() >= batch_deadline
                ):
                    deadline_interruptions += 1
                    break
                player_count = int(rules.get("game", {}).get("players", 3))
                root_policy = _ForcedFirstDiscardPolicy(
                    label,
                    continuation_policy=self._root_continuation_policy(),
                )
                policies = self._rollout_policies(
                    player_count=player_count,
                    root_seat=view.seat,
                    root_policy=root_policy,
                    opponent_policy_factory=opponent_policy_factory,
                )
                for rollout_policy in policies:
                    cache_owner = getattr(
                        rollout_policy,
                        "continuation_policy",
                        rollout_policy,
                    )
                    if isinstance(cache_owner, InformationSetSearchPolicy):
                        cache_owner.bind_rollout_hu_cache(
                            rules,
                            paired_hu_cache,
                            position_cache=paired_position_cache,
                            followup_cache=paired_followup_cache,
                        )
                if root_continuation_policy_name == "unknown":
                    root_continuation_policy_name = str(
                        getattr(
                            root_policy.continuation_policy,
                            "name",
                            type(root_policy.continuation_policy).__name__,
                        )
                    )
                if opponent_policy_name == "unknown":
                    opponent_policy_name = next(
                        str(getattr(policy, "name", type(policy).__name__))
                        for seat, policy in enumerate(policies)
                        if seat != view.seat
                    )
                simulator = FullGameSimulator(
                    policies,
                    wildcard_enabled=bool(rules.get("wildcard", {}).get("enabled", False)),
                    dealer=view.seat,
                    rules=rules,
                )
                result = simulator.play_from_state(
                    seed=rollout_seed,
                    players=deepcopy(base_players),
                    stock=list(base_stock),
                    current=view.seat,
                    max_turns=self.config.rollout_max_turns,
                    deadline=batch_deadline,
                )
                rollout_invariant_violations += len(result.violations)
                rollout_violations.extend(result.violations)
                if result.violations:
                    break
                if result.coverage_failures:
                    rollout_coverage_failures += len(result.coverage_failures)
                    rollout_coverage_reasons.extend(result.coverage_failures)
                    break
                if result.reason == "time_budget":
                    deadline_interruptions += 1
                    break
                batch_results.append((label, result))
            if len(batch_results) != len(shortlist):
                break
            batch_rewards: dict[str, float] = {}
            for label, result in batch_results:
                reward = _root_reward(
                    result,
                    root_seat=view.seat,
                    objective_mode=self.config.objective_mode,
                )
                visits[label] += 1
                rewards[label] += reward
                wins[label] += int(result.winner == view.seat)
                losses[label] += int(
                    result.winner is not None
                    and result.winner != view.seat
                )
                draws[label] += int(result.winner is None)
                outcome_scores[label] += _root_outcome_score(
                    result,
                    root_seat=view.seat,
                )
                signed_xi[label] += _root_signed_xi(
                    result,
                    root_seat=view.seat,
                )
                simulations += 1
                batch_rewards[label] = reward
            paired_reward_batches.append(batch_rewards)
            paired_outcome_batches.append(
                _paired_outcome_batch(batch_results, root_seat=view.seat)
            )
            if self.config.record_paired_worlds:
                paired_worlds.append(
                    _paired_world_outcome(
                        world_index=(
                            paired_world_offset + paired_determinizations
                        ),
                        players=base_players,
                        stock=base_stock,
                        rollout_seed=rollout_seed,
                        opponent_policy=opponent_policy_name,
                        root_continuation_policy=root_continuation_policy_name,
                        batch_results=batch_results,
                        root_seat=view.seat,
                        objective_mode=self.config.objective_mode,
                    )
                )
            paired_determinizations += 1

        stats = tuple(
            RootCandidateStats(
                label=label,
                visits=visits[label],
                reward_sum=rewards[label],
                average_reward=rewards[label] / max(1, visits[label]),
                win_rate=wins[label] / max(1, visits[label]),
                heuristic_value=heuristic[label],
                wins=wins[label],
                losses=losses[label],
                draws=draws[label],
                loss_rate=losses[label] / max(1, visits[label]),
                draw_rate=draws[label] / max(1, visits[label]),
                mean_outcome_score=outcome_scores[label] / max(1, visits[label]),
                mean_signed_xi=signed_xi[label] / max(1, visits[label]),
            )
            for label in shortlist
        )
        paired_advantages = _paired_advantage_stats(
            paired_reward_batches,
            candidate_keys=shortlist,
            preferred_key=preferred_label,
            outcome_batches=paired_outcome_batches,
            objective_mode=self.config.objective_mode,
            familywise_comparisons=(
                self.config.confidence_familywise_comparisons
            ),
        )
        common = {
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "candidates": stats,
            "determinization_failures": failures,
            "paired_determinizations": paired_determinizations,
            "deadline_interruptions": deadline_interruptions,
            "rollout_invariant_violations": rollout_invariant_violations,
            "rollout_violations": tuple(dict.fromkeys(rollout_violations)),
            "rollout_coverage_failures": rollout_coverage_failures,
            "rollout_coverage_reasons": tuple(dict.fromkeys(rollout_coverage_reasons)),
            "paired_advantages": paired_advantages,
            "paired_worlds": tuple(paired_worlds),
        }
        if rollout_invariant_violations:
            return RootSearchResult(
                selected_label=preferred_label,
                used_search=False,
                reason="rollout_invariant_violation",
                simulations=simulations,
                **common,
            )
        if rollout_coverage_failures:
            return RootSearchResult(
                selected_label=preferred_label,
                used_search=False,
                reason="rollout_coverage_incomplete",
                simulations=simulations,
                **common,
            )
        if simulations == 0:
            return RootSearchResult(
                selected_label=preferred_label,
                used_search=False,
                reason=(
                    "discard_deadline_before_complete_pair"
                    if deadline_interruptions
                    else "determinization_unavailable"
                ),
                simulations=0,
                **common,
            )
        if any(item.visits == 0 for item in stats):
            return RootSearchResult(
                selected_label=preferred_label,
                used_search=False,
                reason="candidate_coverage_incomplete",
                simulations=simulations,
                **common,
            )
        empirical_best = max(
            stats,
            key=lambda item: (
                _candidate_rank_key(item),
                item.label == preferred_label,
                item.visits,
                item.heuristic_value,
                item.label,
            ),
        )
        selected_label = empirical_best.label
        confidence_override = False
        reason = (
            "root_discard_ismcts_completed_at_deadline"
            if deadline_interruptions
            else "root_discard_ismcts_completed"
        )
        if self.config.require_confident_override:
            selected_label = preferred_label
            if _override_iteration_budget_incomplete(
                self.config,
                candidate_count=len(shortlist),
                paired_determinizations=paired_determinizations,
                deadline_interruptions=deadline_interruptions,
            ):
                reason = "root_discard_confidence_budget_incomplete"
            elif empirical_best.label == preferred_label:
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
                    and advantage.samples
                    >= max(2, self.config.min_confidence_pairs)
                    and advantage.lower_confidence_bound is not None
                    and advantage.lower_confidence_bound
                    > self.config.minimum_confident_advantage
                ):
                    selected_label = empirical_best.label
                    confidence_override = True
                    reason = "root_discard_confidence_override"
                else:
                    reason = "root_discard_confidence_insufficient"
        return RootSearchResult(
            selected_label=selected_label,
            used_search=True,
            reason=reason,
            simulations=simulations,
            candidates=tuple(
                sorted(
                    stats,
                    key=lambda item: (
                        _candidate_rank_key(item),
                        item.visits,
                        item.heuristic_value,
                    ),
                    reverse=True,
                )
            ),
            elapsed_ms=common["elapsed_ms"],
            determinization_failures=failures,
            paired_determinizations=paired_determinizations,
            deadline_interruptions=deadline_interruptions,
            rollout_invariant_violations=rollout_invariant_violations,
            rollout_violations=common["rollout_violations"],
            rollout_coverage_failures=rollout_coverage_failures,
            rollout_coverage_reasons=common["rollout_coverage_reasons"],
            empirical_best_label=empirical_best.label,
            confidence_override=confidence_override,
            paired_advantages=paired_advantages,
            paired_worlds=tuple(paired_worlds),
        )

    def search_discard_response_risk(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        candidate_labels: Sequence[str],
        candidate_priors: dict[str, float],
        preferred_label: str | None = None,
    ) -> DiscardResponseRiskResult:
        """Sample only the immediate legal response to each candidate discard."""

        started = time.perf_counter()
        legal = set(_discardable_labels(view.hand))
        labels = [
            label
            for label in dict.fromkeys(candidate_labels)
            if label in legal and label in candidate_priors
        ]
        labels = sorted(
            labels,
            key=lambda label: (float(candidate_priors[label]), label),
            reverse=True,
        )[: max(1, self.config.max_candidates)]
        if not labels:
            raise ValueError("no_discard_response_risk_candidate")

        samples = Counter({label: 0 for label in labels})
        passes = Counter({label: 0 for label in labels})
        claims = {label: Counter() for label in labels}
        claim_seats = {label: Counter() for label in labels}
        risks = Counter({label: 0.0 for label in labels})
        deadline = started + max(1, self.config.time_budget_ms) / 1000.0
        rng = random.Random(information_set_seed(view, self.config.seed))
        simulations = 0
        failures = 0
        paired_determinizations = 0
        deadline_interruptions = 0
        rollout_invariant_violations = 0
        rollout_violations: list[str] = []
        rollout_coverage_failures = 0
        rollout_coverage_reasons: list[str] = []

        if len(labels) == 1:
            stats = _discard_response_risk_stats(
                labels,
                samples=samples,
                passes=passes,
                claims=claims,
                claim_seats=claim_seats,
                risks=risks,
                priors=candidate_priors,
                view=view,
            )
            return DiscardResponseRiskResult(
                selected_label=labels[0],
                used_search=False,
                reason="single_legal_candidate",
                simulations=0,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=stats,
            )

        while (
            simulations + len(labels) <= max(1, self.config.max_iterations)
            and time.perf_counter() < deadline
        ):
            determinization = determinize_public_view(view, rules=rules, rng=rng)
            if determinization is None:
                failures += 1
                break
            base_players, base_stock = determinization
            rollout_seed = rng.randrange(1, 2**31)
            opponent_policy_factory = self._opponent_rollout_factory(
                paired_determinizations
            )
            batch: list[tuple[str, str, int | None, float]] = []
            for label in labels:
                if time.perf_counter() >= deadline:
                    deadline_interruptions += 1
                    break
                player_count = int(rules.get("game", {}).get("players", 3))
                policies = self._rollout_policies(
                    player_count=player_count,
                    root_seat=view.seat,
                    root_policy=_ForcedFirstDiscardPolicy(
                        label,
                        continuation_policy=self._root_continuation_policy(),
                    ),
                    opponent_policy_factory=opponent_policy_factory,
                )
                simulator = FullGameSimulator(
                    policies,
                    wildcard_enabled=bool(rules.get("wildcard", {}).get("enabled", False)),
                    dealer=view.seat,
                    rules=rules,
                )
                result = simulator.play_from_state(
                    seed=rollout_seed,
                    players=deepcopy(base_players),
                    stock=list(base_stock),
                    current=view.seat,
                    max_turns=1,
                    deadline=deadline,
                )
                if result.coverage_failures:
                    rollout_coverage_failures += len(result.coverage_failures)
                    rollout_coverage_reasons.extend(result.coverage_failures)
                    break
                if result.violations:
                    rollout_invariant_violations += len(result.violations)
                    rollout_violations.extend(result.violations)
                    break
                if result.reason == "time_budget":
                    deadline_interruptions += 1
                    break
                try:
                    action, claimant = _immediate_response_from_result(
                        result,
                        discarder=view.seat,
                        player_count=player_count,
                    )
                except ValueError as exc:
                    rollout_invariant_violations += 1
                    rollout_violations.append(str(exc))
                    break
                batch.append(
                    (
                        label,
                        action,
                        claimant,
                        _immediate_response_risk(action, rules=rules),
                    )
                )
            if len(batch) != len(labels):
                break
            for label, action, claimant, risk in batch:
                samples[label] += 1
                risks[label] += risk
                if action == "pass":
                    passes[label] += 1
                else:
                    claims[label][action] += 1
                    if claimant is not None:
                        claim_seats[label][(action, claimant)] += 1
                simulations += 1
            paired_determinizations += 1

        stats = _discard_response_risk_stats(
            labels,
            samples=samples,
            passes=passes,
            claims=claims,
            claim_seats=claim_seats,
            risks=risks,
            priors=candidate_priors,
            view=view,
        )
        common = {
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "candidates": stats,
            "determinization_failures": failures,
            "paired_determinizations": paired_determinizations,
            "deadline_interruptions": deadline_interruptions,
            "rollout_invariant_violations": rollout_invariant_violations,
            "rollout_violations": tuple(dict.fromkeys(rollout_violations)),
            "rollout_coverage_failures": rollout_coverage_failures,
            "rollout_coverage_reasons": tuple(dict.fromkeys(rollout_coverage_reasons)),
        }
        if rollout_coverage_failures:
            return DiscardResponseRiskResult(
                selected_label=labels[0],
                used_search=False,
                reason="rollout_coverage_incomplete",
                simulations=simulations,
                **common,
            )
        if rollout_invariant_violations:
            return DiscardResponseRiskResult(
                selected_label=labels[0],
                used_search=False,
                reason="rollout_invariant_failure",
                simulations=simulations,
                **common,
            )
        if not paired_determinizations:
            return DiscardResponseRiskResult(
                selected_label=labels[0],
                used_search=False,
                reason=(
                    "discard_response_deadline_before_complete_pair"
                    if deadline_interruptions
                    else "determinization_unavailable"
                ),
                simulations=0,
                **common,
            )
        selected = max(
            stats,
            key=lambda item: (
                item.adjusted_value,
                item.heuristic_value,
                int(item.label == preferred_label),
                item.label,
            ),
        )
        return DiscardResponseRiskResult(
            selected_label=selected.label,
            used_search=True,
            reason=(
                "discard_response_risk_completed_at_deadline"
                if deadline_interruptions
                else "discard_response_risk_completed"
            ),
            simulations=simulations,
            **common,
        )

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
        """Compare exact PASS/CHI/PENG commitments from one public response state."""

        started = time.perf_counter()
        by_key = {candidate.key: candidate for candidate in candidates}
        if len(by_key) != len(candidates):
            raise ValueError("duplicate_response_candidate_key")
        if not by_key:
            raise ValueError("no_response_candidate")
        if preferred_key is not None and preferred_key not in by_key:
            raise ValueError("preferred_response_candidate_missing")
        for candidate in by_key.values():
            _validate_response_candidate(view, candidate, rules)

        heuristic = {
            key: candidate.heuristic_value
            for key, candidate in by_key.items()
        }
        ranked_keys = sorted(
            by_key,
            key=lambda key: (heuristic[key], key),
            reverse=True,
        )
        shortlist = ranked_keys[: max(1, self.config.max_candidates)]
        if preferred_key is not None and preferred_key not in shortlist:
            shortlist = [
                preferred_key,
                *(
                    key
                    for key in ranked_keys
                    if key != preferred_key
                ),
            ][: max(1, self.config.max_candidates)]
        fallback_key = preferred_key or shortlist[0]
        if len(shortlist) == 1:
            return _no_response_search_result(
                by_key[fallback_key],
                "single_legal_candidate",
                started,
            )
        gap = heuristic[shortlist[0]] - heuristic[shortlist[1]]
        if not force_search and gap >= self.config.skip_search_gap:
            return _no_response_search_result(
                by_key[fallback_key],
                f"heuristic_gap={round(gap, 3)}",
                started,
                candidates=tuple(by_key[key] for key in shortlist),
            )
        if max(1, self.config.max_iterations) < len(shortlist):
            return _no_response_search_result(
                by_key[fallback_key],
                "paired_candidate_budget_too_small",
                started,
                candidates=tuple(by_key[key] for key in shortlist),
            )

        visits = Counter({key: 0 for key in shortlist})
        rewards = Counter({key: 0.0 for key in shortlist})
        wins = Counter({key: 0 for key in shortlist})
        losses = Counter({key: 0 for key in shortlist})
        draws = Counter({key: 0 for key in shortlist})
        outcome_scores = Counter({key: 0.0 for key in shortlist})
        signed_xi = Counter({key: 0.0 for key in shortlist})
        deadline = _bounded_search_deadline(
            started,
            time_budget_ms=self.config.time_budget_ms,
            absolute_deadline=absolute_deadline,
        )
        rng = random.Random(information_set_seed(view, self.config.seed))
        simulations = 0
        failures = 0
        paired_determinizations = 0
        deadline_interruptions = 0
        rollout_invariant_violations = 0
        rollout_violations: list[str] = []
        rollout_coverage_failures = 0
        rollout_coverage_reasons: list[str] = []
        paired_reward_batches: list[dict[str, float]] = []
        paired_outcome_batches: list[dict[str, Any]] = []
        paired_worlds: list[PairedWorldOutcome] = []
        paired_world_offset = max(0, int(self.config.paired_world_offset))
        for _ in range(paired_world_offset):
            determinization = determinize_public_view(view, rules=rules, rng=rng)
            if determinization is None:
                failures += 1
                break
            rng.randrange(1, 2**31)
        if failures:
            return RootResponseSearchResult(
                selected_key=fallback_key,
                used_search=False,
                reason="determinization_unavailable_before_paired_shard",
                simulations=0,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=tuple(
                    RootResponseCandidateStats(
                        candidate=by_key[key],
                        visits=0,
                        reward_sum=0.0,
                        average_reward=0.0,
                        win_rate=0.0,
                    )
                    for key in shortlist
                ),
                determinization_failures=failures,
            )
        while simulations + len(shortlist) <= max(1, self.config.max_iterations):
            complete_first_batch = (
                self.config.complete_first_paired_batch
                and paired_determinizations == 0
                and absolute_deadline is None
            )
            if not complete_first_batch and time.perf_counter() >= deadline:
                break
            determinization = determinize_public_view(view, rules=rules, rng=rng)
            if determinization is None:
                failures += 1
                break
            base_players, base_stock = determinization
            rollout_seed = rng.randrange(1, 2**31)
            opponent_policy_factory = self._opponent_rollout_factory(
                paired_world_offset + paired_determinizations
            )
            opponent_policy_name = "unknown"
            root_continuation_policy_name = "unknown"
            batch_results: list[tuple[str, GameResult]] = []
            for key in shortlist:
                if not complete_first_batch and time.perf_counter() >= deadline:
                    deadline_interruptions += 1
                    break
                candidate = by_key[key]
                players = deepcopy(base_players)
                stock = list(base_stock)
                player_count = int(rules.get("game", {}).get("players", 3))
                root_policy = _ForcedRootResponsePolicy(
                    pending_card=str(view.pending_card),
                    candidate=candidate,
                    continuation_policy=self._root_continuation_policy(),
                )
                if root_continuation_policy_name == "unknown":
                    root_continuation_policy_name = str(
                        getattr(
                            root_policy.continuation_policy,
                            "name",
                            type(root_policy.continuation_policy).__name__,
                        )
                    )
                policies = self._rollout_policies(
                    player_count=player_count,
                    root_seat=view.seat,
                    root_policy=root_policy,
                    opponent_policy_factory=opponent_policy_factory,
                )
                if opponent_policy_name == "unknown":
                    opponent_policy_name = next(
                        str(getattr(policy, "name", type(policy).__name__))
                        for seat, policy in enumerate(policies)
                        if seat != view.seat
                    )
                simulator = FullGameSimulator(
                    policies,
                    wildcard_enabled=bool(rules.get("wildcard", {}).get("enabled", False)),
                    dealer=view.seat,
                    rules=rules,
                )
                result = simulator.play_from_pending_discard(
                    seed=rollout_seed,
                    players=players,
                    stock=stock,
                    discarder=int(view.pending_source_seat),
                    pending_card=str(view.pending_card),
                    blocked_auto_claim_seats=(
                        frozenset()
                        if candidate.action_type.upper() == "HU"
                        else frozenset((view.seat,))
                    ),
                    max_turns=self.config.rollout_max_turns,
                    deadline=None if complete_first_batch else deadline,
                )
                rollout_invariant_violations += len(result.violations)
                rollout_violations.extend(result.violations)
                if result.violations:
                    break
                if result.coverage_failures:
                    rollout_coverage_failures += len(result.coverage_failures)
                    rollout_coverage_reasons.extend(result.coverage_failures)
                    break
                if result.reason == "time_budget":
                    deadline_interruptions += 1
                    break
                batch_results.append((key, result))
            if len(batch_results) != len(shortlist):
                break
            batch_rewards: dict[str, float] = {}
            for key, result in batch_results:
                reward = _root_reward(
                    result,
                    root_seat=view.seat,
                    objective_mode=self.config.objective_mode,
                )
                visits[key] += 1
                rewards[key] += reward
                wins[key] += int(result.winner == view.seat)
                losses[key] += int(
                    result.winner is not None
                    and result.winner != view.seat
                )
                draws[key] += int(result.winner is None)
                outcome_scores[key] += _root_outcome_score(
                    result,
                    root_seat=view.seat,
                )
                signed_xi[key] += _root_signed_xi(
                    result,
                    root_seat=view.seat,
                )
                simulations += 1
                batch_rewards[key] = reward
            paired_reward_batches.append(batch_rewards)
            paired_outcome_batches.append(
                _paired_outcome_batch(batch_results, root_seat=view.seat)
            )
            if self.config.record_paired_worlds:
                paired_worlds.append(
                    _paired_world_outcome(
                        world_index=(
                            paired_world_offset + paired_determinizations
                        ),
                        players=base_players,
                        stock=base_stock,
                        rollout_seed=rollout_seed,
                        opponent_policy=opponent_policy_name,
                        root_continuation_policy=root_continuation_policy_name,
                        batch_results=batch_results,
                        root_seat=view.seat,
                        objective_mode=self.config.objective_mode,
                    )
                )
            paired_determinizations += 1

        stats = tuple(
            RootResponseCandidateStats(
                candidate=by_key[key],
                visits=visits[key],
                reward_sum=rewards[key],
                average_reward=rewards[key] / max(1, visits[key]),
                win_rate=wins[key] / max(1, visits[key]),
                wins=wins[key],
                losses=losses[key],
                draws=draws[key],
                loss_rate=losses[key] / max(1, visits[key]),
                draw_rate=draws[key] / max(1, visits[key]),
                mean_outcome_score=outcome_scores[key] / max(1, visits[key]),
                mean_signed_xi=signed_xi[key] / max(1, visits[key]),
            )
            for key in shortlist
        )
        paired_advantages = _paired_advantage_stats(
            paired_reward_batches,
            candidate_keys=shortlist,
            preferred_key=fallback_key,
            outcome_batches=paired_outcome_batches,
            objective_mode=self.config.objective_mode,
        )
        if rollout_invariant_violations:
            return RootResponseSearchResult(
                selected_key=fallback_key,
                used_search=False,
                reason="rollout_invariant_violation",
                simulations=simulations,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=stats,
                determinization_failures=failures,
                paired_determinizations=paired_determinizations,
                deadline_interruptions=deadline_interruptions,
                rollout_invariant_violations=rollout_invariant_violations,
                rollout_violations=tuple(dict.fromkeys(rollout_violations)),
                rollout_coverage_failures=rollout_coverage_failures,
                rollout_coverage_reasons=tuple(dict.fromkeys(rollout_coverage_reasons)),
                paired_advantages=paired_advantages,
                paired_worlds=tuple(paired_worlds),
            )
        if rollout_coverage_failures:
            return RootResponseSearchResult(
                selected_key=fallback_key,
                used_search=False,
                reason="rollout_coverage_incomplete",
                simulations=simulations,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=stats,
                determinization_failures=failures,
                paired_determinizations=paired_determinizations,
                deadline_interruptions=deadline_interruptions,
                rollout_invariant_violations=rollout_invariant_violations,
                rollout_violations=tuple(dict.fromkeys(rollout_violations)),
                rollout_coverage_failures=rollout_coverage_failures,
                rollout_coverage_reasons=tuple(dict.fromkeys(rollout_coverage_reasons)),
                paired_advantages=paired_advantages,
                paired_worlds=tuple(paired_worlds),
            )
        if simulations == 0:
            reason = (
                "response_deadline_before_complete_pair"
                if deadline_interruptions
                else "determinization_unavailable"
            )
            return RootResponseSearchResult(
                selected_key=fallback_key,
                used_search=False,
                reason=reason,
                simulations=0,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=stats,
                determinization_failures=failures,
                paired_determinizations=paired_determinizations,
                deadline_interruptions=deadline_interruptions,
                rollout_invariant_violations=rollout_invariant_violations,
                rollout_violations=tuple(dict.fromkeys(rollout_violations)),
                rollout_coverage_failures=rollout_coverage_failures,
                rollout_coverage_reasons=tuple(dict.fromkeys(rollout_coverage_reasons)),
                paired_advantages=paired_advantages,
                paired_worlds=tuple(paired_worlds),
            )
        if any(item.visits == 0 for item in stats):
            return RootResponseSearchResult(
                selected_key=fallback_key,
                used_search=False,
                reason="candidate_coverage_incomplete",
                simulations=simulations,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=stats,
                determinization_failures=failures,
                paired_determinizations=paired_determinizations,
                deadline_interruptions=deadline_interruptions,
                rollout_invariant_violations=rollout_invariant_violations,
                rollout_violations=tuple(dict.fromkeys(rollout_violations)),
                rollout_coverage_failures=rollout_coverage_failures,
                rollout_coverage_reasons=tuple(dict.fromkeys(rollout_coverage_reasons)),
                paired_advantages=paired_advantages,
                paired_worlds=tuple(paired_worlds),
            )
        empirical_best = max(
            stats,
            key=lambda item: (
                _candidate_rank_key(item),
                item.candidate.key == fallback_key,
                item.visits,
                item.candidate.heuristic_value,
                item.candidate.key,
            ),
        )
        selected_key = empirical_best.candidate.key
        confidence_override = False
        reason = (
            "root_response_ismcts_completed_at_deadline"
            if deadline_interruptions
            else "root_response_ismcts_completed"
        )
        if self.config.require_confident_override:
            selected_key = fallback_key
            if _override_iteration_budget_incomplete(
                self.config,
                candidate_count=len(shortlist),
                paired_determinizations=paired_determinizations,
                deadline_interruptions=deadline_interruptions,
            ):
                reason = "root_response_confidence_budget_incomplete"
            elif empirical_best.candidate.key == fallback_key:
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
                    and advantage.samples
                    >= max(2, self.config.min_confidence_pairs)
                    and advantage.lower_confidence_bound is not None
                    and advantage.lower_confidence_bound
                    > self.config.minimum_confident_advantage
                ):
                    selected_key = empirical_best.candidate.key
                    confidence_override = True
                    reason = "root_response_confidence_override"
                else:
                    reason = "root_response_confidence_insufficient"
        return RootResponseSearchResult(
            selected_key=selected_key,
            used_search=True,
            reason=reason,
            simulations=simulations,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            candidates=tuple(
                sorted(
                    stats,
                    key=lambda item: (
                        _candidate_rank_key(item),
                        item.visits,
                        item.candidate.heuristic_value,
                    ),
                    reverse=True,
                )
            ),
            determinization_failures=failures,
            paired_determinizations=paired_determinizations,
            deadline_interruptions=deadline_interruptions,
            rollout_invariant_violations=rollout_invariant_violations,
            rollout_violations=tuple(dict.fromkeys(rollout_violations)),
            rollout_coverage_failures=rollout_coverage_failures,
            rollout_coverage_reasons=tuple(dict.fromkeys(rollout_coverage_reasons)),
            empirical_best_key=empirical_best.candidate.key,
            confidence_override=confidence_override,
            paired_advantages=paired_advantages,
            paired_worlds=tuple(paired_worlds),
        )

    def search_hu(
        self,
        view: PublicView,
        *,
        rules: dict[str, Any],
        preferred_key: str = "HU",
    ) -> RootResponseSearchResult:
        """Compare accepting a self-Hu against continuing from paired worlds."""

        started = time.perf_counter()
        keys = ("HU", "PASS")
        if preferred_key not in keys:
            raise ValueError("preferred_hu_candidate_missing")
        if not evaluate_hu(view.hand, view.own_melds, rules).can_hu:
            raise ValueError("hu_candidate_not_legal")
        if self.config.objective_mode is ObjectiveMode.WIN_FIRST:
            return RootResponseSearchResult(
                selected_key="HU",
                used_search=True,
                reason="win_first_immediate_hu_dominance",
                simulations=0,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=(
                    RootResponseCandidateStats(
                        candidate=RootResponseCandidate("HU", "HU", 0.0),
                        visits=1,
                        reward_sum=1.0,
                        average_reward=1.0,
                        win_rate=1.0,
                        wins=1,
                    ),
                    RootResponseCandidateStats(
                        candidate=RootResponseCandidate("PASS", "PASS", 0.0),
                        visits=0,
                        reward_sum=0.0,
                        average_reward=0.0,
                        win_rate=0.0,
                    ),
                ),
                empirical_best_key="HU",
                confidence_override=preferred_key != "HU",
            )
        if max(1, self.config.max_iterations) < len(keys):
            return RootResponseSearchResult(
                selected_key=preferred_key,
                used_search=False,
                reason="paired_candidate_budget_too_small",
                simulations=0,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=tuple(
                    RootResponseCandidateStats(
                        candidate=RootResponseCandidate(key, key, 0.0),
                        visits=0,
                        reward_sum=0.0,
                        average_reward=0.0,
                        win_rate=0.0,
                    )
                    for key in keys
                ),
            )

        visits = Counter({key: 0 for key in keys})
        rewards = Counter({key: 0.0 for key in keys})
        wins = Counter({key: 0 for key in keys})
        losses = Counter({key: 0 for key in keys})
        draws = Counter({key: 0 for key in keys})
        outcome_scores = Counter({key: 0.0 for key in keys})
        signed_xi = Counter({key: 0.0 for key in keys})
        deadline = started + max(1, self.config.time_budget_ms) / 1000.0
        rng = random.Random(information_set_seed(view, self.config.seed))
        simulations = 0
        failures = 0
        paired_determinizations = 0
        deadline_interruptions = 0
        rollout_invariant_violations = 0
        rollout_violations: list[str] = []
        rollout_coverage_failures = 0
        rollout_coverage_reasons: list[str] = []
        paired_reward_batches: list[dict[str, float]] = []
        paired_outcome_batches: list[dict[str, Any]] = []
        paired_worlds: list[PairedWorldOutcome] = []
        while (
            simulations + len(keys) <= max(1, self.config.max_iterations)
            and time.perf_counter() < deadline
        ):
            determinization = determinize_public_view(view, rules=rules, rng=rng)
            if determinization is None:
                failures += 1
                break
            base_players, base_stock = determinization
            rollout_seed = rng.randrange(1, 2**31)
            rollout_factory = self._opponent_rollout_factory(
                paired_determinizations
            )
            opponent_policy_name = "unknown"
            root_continuation_policy_name = "unknown"
            batch_results: list[tuple[str, GameResult]] = []
            for key in keys:
                if time.perf_counter() >= deadline:
                    deadline_interruptions += 1
                    break
                player_count = int(rules.get("game", {}).get("players", 3))
                root_policy = _ForcedHuContinuationPolicy(
                    accept=key == "HU",
                    continuation_policy=self._root_continuation_policy(),
                )
                if root_continuation_policy_name == "unknown":
                    root_continuation_policy_name = str(
                        getattr(
                            root_policy.continuation_policy,
                            "name",
                            type(root_policy.continuation_policy).__name__,
                        )
                    )
                policies = self._rollout_policies(
                    player_count=player_count,
                    root_seat=view.seat,
                    root_policy=root_policy,
                    opponent_policy_factory=rollout_factory,
                )
                if opponent_policy_name == "unknown":
                    opponent_policy_name = next(
                        str(getattr(policy, "name", type(policy).__name__))
                        for seat, policy in enumerate(policies)
                        if seat != view.seat
                    )
                simulator = FullGameSimulator(
                    policies,
                    wildcard_enabled=bool(
                        rules.get("wildcard", {}).get("enabled", False)
                    ),
                    dealer=view.seat,
                    rules=rules,
                )
                result = simulator.play_from_state(
                    seed=rollout_seed,
                    players=deepcopy(base_players),
                    stock=list(base_stock),
                    current=view.seat,
                    needs_draw=False,
                    max_turns=max(1, self.config.rollout_max_turns),
                    deadline=deadline,
                )
                rollout_invariant_violations += len(result.violations)
                rollout_violations.extend(result.violations)
                if result.violations:
                    break
                if result.coverage_failures:
                    rollout_coverage_failures += len(result.coverage_failures)
                    rollout_coverage_reasons.extend(result.coverage_failures)
                    break
                if result.reason == "time_budget":
                    deadline_interruptions += 1
                    break
                batch_results.append((key, result))
            if len(batch_results) != len(keys):
                break
            batch_rewards: dict[str, float] = {}
            for key, result in batch_results:
                reward = _root_reward(
                    result,
                    root_seat=view.seat,
                    objective_mode=self.config.objective_mode,
                )
                visits[key] += 1
                rewards[key] += reward
                wins[key] += int(result.winner == view.seat)
                losses[key] += int(
                    result.winner is not None
                    and result.winner != view.seat
                )
                draws[key] += int(result.winner is None)
                outcome_scores[key] += _root_outcome_score(
                    result,
                    root_seat=view.seat,
                )
                signed_xi[key] += _root_signed_xi(
                    result,
                    root_seat=view.seat,
                )
                simulations += 1
                batch_rewards[key] = reward
            paired_reward_batches.append(batch_rewards)
            paired_outcome_batches.append(
                _paired_outcome_batch(batch_results, root_seat=view.seat)
            )
            if self.config.record_paired_worlds:
                paired_worlds.append(
                    _paired_world_outcome(
                        world_index=paired_determinizations,
                        players=base_players,
                        stock=base_stock,
                        rollout_seed=rollout_seed,
                        opponent_policy=opponent_policy_name,
                        root_continuation_policy=root_continuation_policy_name,
                        batch_results=batch_results,
                        root_seat=view.seat,
                        objective_mode=self.config.objective_mode,
                    )
                )
            paired_determinizations += 1

        stats = tuple(
            RootResponseCandidateStats(
                candidate=RootResponseCandidate(key, key, 0.0),
                visits=visits[key],
                reward_sum=rewards[key],
                average_reward=rewards[key] / max(1, visits[key]),
                win_rate=wins[key] / max(1, visits[key]),
                wins=wins[key],
                losses=losses[key],
                draws=draws[key],
                loss_rate=losses[key] / max(1, visits[key]),
                draw_rate=draws[key] / max(1, visits[key]),
                mean_outcome_score=(
                    outcome_scores[key] / max(1, visits[key])
                ),
                mean_signed_xi=signed_xi[key] / max(1, visits[key]),
            )
            for key in keys
        )
        paired_advantages = _paired_advantage_stats(
            paired_reward_batches,
            candidate_keys=keys,
            preferred_key=preferred_key,
            outcome_batches=paired_outcome_batches,
            objective_mode=self.config.objective_mode,
        )
        if (
            rollout_invariant_violations
            or rollout_coverage_failures
            or any(item.visits == 0 for item in stats)
        ):
            reason = (
                "rollout_invariant_violation"
                if rollout_invariant_violations
                else (
                    "rollout_coverage_incomplete"
                    if rollout_coverage_failures
                    else "candidate_coverage_incomplete"
                )
            )
            return RootResponseSearchResult(
                selected_key=preferred_key,
                used_search=False,
                reason=reason,
                simulations=simulations,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                candidates=stats,
                determinization_failures=failures,
                paired_determinizations=paired_determinizations,
                deadline_interruptions=deadline_interruptions,
                rollout_invariant_violations=rollout_invariant_violations,
                rollout_violations=tuple(dict.fromkeys(rollout_violations)),
                rollout_coverage_failures=rollout_coverage_failures,
                rollout_coverage_reasons=tuple(
                    dict.fromkeys(rollout_coverage_reasons)
                ),
                paired_advantages=paired_advantages,
                paired_worlds=tuple(paired_worlds),
            )
        empirical_best = max(
            stats,
            key=lambda item: (
                _candidate_rank_key(item),
                item.candidate.key == preferred_key,
                item.visits,
                item.candidate.key,
            ),
        )
        selected_key = empirical_best.candidate.key
        confidence_override = False
        reason = "root_hu_ismcts_completed"
        if self.config.require_confident_override:
            selected_key = preferred_key
            if _override_iteration_budget_incomplete(
                self.config,
                candidate_count=len(keys),
                paired_determinizations=paired_determinizations,
                deadline_interruptions=deadline_interruptions,
            ):
                reason = "root_hu_confidence_budget_incomplete"
            elif empirical_best.candidate.key == preferred_key:
                reason = "root_hu_confidence_kept_preferred"
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
                    and advantage.samples
                    >= max(2, self.config.min_confidence_pairs)
                    and advantage.lower_confidence_bound is not None
                    and advantage.lower_confidence_bound
                    > self.config.minimum_confident_advantage
                ):
                    selected_key = empirical_best.candidate.key
                    confidence_override = True
                    reason = "root_hu_confidence_override"
                else:
                    reason = "root_hu_confidence_insufficient"
        elif deadline_interruptions:
            reason = "root_hu_ismcts_completed_at_deadline"
        return RootResponseSearchResult(
            selected_key=selected_key,
            used_search=True,
            reason=reason,
            simulations=simulations,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            candidates=tuple(
                sorted(
                    stats,
                    key=lambda item: (
                        _candidate_rank_key(item),
                        item.visits,
                        item.candidate.key,
                    ),
                    reverse=True,
                )
            ),
            determinization_failures=failures,
            paired_determinizations=paired_determinizations,
            deadline_interruptions=deadline_interruptions,
            rollout_invariant_violations=rollout_invariant_violations,
            rollout_violations=tuple(dict.fromkeys(rollout_violations)),
            rollout_coverage_failures=rollout_coverage_failures,
            rollout_coverage_reasons=tuple(
                dict.fromkeys(rollout_coverage_reasons)
            ),
            empirical_best_key=empirical_best.candidate.key,
            confidence_override=confidence_override,
            paired_advantages=paired_advantages,
            paired_worlds=tuple(paired_worlds),
        )


class _ForcedFirstDiscardPolicy(BaselinePolicy):
    name = "ismcts_rollout"

    def __init__(
        self,
        first_label: str,
        *,
        continuation_policy: SimulationPolicy | None = None,
    ) -> None:
        self.first_label = first_label
        self.continuation_policy = continuation_policy or BaselinePolicy()
        self.used = False

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        if not self.used:
            self.used = True
            if self.first_label not in _discardable_labels(view.hand):
                raise ValueError(f"forced_root_discard_illegal:{self.first_label}")
            return self.first_label
        return self.continuation_policy.choose_discard(view, rules)

    def choose_hu(
        self,
        view: PublicView,
        hu: Any,
        rules: dict[str, Any],
    ) -> bool:
        return self.continuation_policy.choose_hu(view, hu, rules)

    def choose_peng(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ) -> bool:
        return self.continuation_policy.choose_peng(view, label, rules)

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        return self.continuation_policy.choose_chi(view, plans, rules)


class _ForcedRootResponsePolicy(BaselinePolicy):
    name = "ismcts_response_rollout"

    def __init__(
        self,
        *,
        pending_card: str,
        candidate: RootResponseCandidate,
        continuation_policy: SimulationPolicy | None = None,
    ) -> None:
        self.pending_card = pending_card
        self.candidate = candidate
        self.continuation_policy = continuation_policy or BaselinePolicy()
        self.response_active = True
        self.claim_committed = False
        self.followup_used = False

    def choose_hu(
        self,
        view: PublicView,
        hu: Any,
        rules: dict[str, Any],
    ) -> bool:
        if self.response_active:
            should_hu = self.candidate.action_type.upper() == "HU"
            self.claim_committed = should_hu
            return should_hu
        return self.continuation_policy.choose_hu(view, hu, rules)

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        if self.response_active and label == self.pending_card:
            should_peng = self.candidate.action_type.upper() == "PENG"
            self.claim_committed = should_peng
            return should_peng
        return self.continuation_policy.choose_peng(view, label, rules)

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        if self.response_active:
            if self.candidate.action_type.upper() != "CHI":
                return None
            selected = next(
                (
                    plan
                    for plan in plans
                    if _chi_plan_matches_candidate(plan, self.candidate)
                ),
                None,
            )
            if selected is None:
                raise ValueError("forced_root_chi_plan_missing")
            self.claim_committed = True
            return selected
        return self.continuation_policy.choose_chi(view, plans, rules)

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        followup = self.candidate.followup_discard
        if self.claim_committed and not self.followup_used and followup:
            self.followup_used = True
            if followup not in _discardable_labels(view.hand):
                raise ValueError(f"forced_response_followup_illegal:{followup}")
            return followup
        return self.continuation_policy.choose_discard(view, rules)

    def resolve_pending_response(self, executed: bool) -> None:
        if not executed:
            self.claim_committed = False
            self.followup_used = True

    def finish_pending_response(self) -> None:
        self.response_active = False
        if not self.claim_committed:
            self.followup_used = True


class _ForcedHuContinuationPolicy(BaselinePolicy):
    name = "ismcts_hu_rollout"

    def __init__(
        self,
        *,
        accept: bool,
        continuation_policy: SimulationPolicy | None = None,
    ) -> None:
        self.accept = bool(accept)
        self.continuation_policy = continuation_policy or BaselinePolicy()
        self.used = False

    def choose_hu(
        self,
        view: PublicView,
        hu: Any,
        rules: dict[str, Any],
    ) -> bool:
        if not self.used:
            self.used = True
            return self.accept
        return self.continuation_policy.choose_hu(view, hu, rules)

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        return self.continuation_policy.choose_discard(view, rules)

    def choose_peng(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ) -> bool:
        return self.continuation_policy.choose_peng(view, label, rules)

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        return self.continuation_policy.choose_chi(view, plans, rules)


def _validate_response_candidate(
    view: PublicView,
    candidate: RootResponseCandidate,
    rules: dict[str, Any],
) -> None:
    pending = view.pending_card
    source = view.pending_source_seat
    player_count = int(rules.get("game", {}).get("players", 3))
    if not pending:
        raise ValueError("response_pending_card_required")
    if source is None or not 0 <= source < player_count or source == view.seat:
        raise ValueError("response_source_seat_required")
    action_type = candidate.action_type.upper()
    if action_type == "PASS":
        if candidate.consumed_from_hand or candidate.meld_groups or candidate.followup_discard:
            raise ValueError("pass_candidate_has_claim_payload")
        return
    if action_type == "HU":
        if candidate.consumed_from_hand or candidate.meld_groups or candidate.followup_discard:
            raise ValueError("hu_candidate_has_claim_payload")
        hu = evaluate_hu(
            [*view.hand, pending],
            view.own_melds,
            rules,
        )
        if not hu.can_hu:
            raise ValueError("hu_candidate_not_legal")
        return
    if action_type == "PENG":
        if pending == WILD_LABEL or Counter(candidate.consumed_from_hand) != Counter((pending, pending)):
            raise ValueError("peng_candidate_consumption_mismatch")
        if view.hand.count(pending) != 2:
            raise ValueError("peng_candidate_pair_unavailable")
        if candidate.meld_groups != ((pending, pending, pending),):
            raise ValueError("peng_candidate_meld_mismatch")
        hand_after = list(view.hand)
        hand_after.remove(pending)
        hand_after.remove(pending)
    elif action_type == "CHI":
        if (source + 1) % player_count != view.seat:
            raise ValueError("chi_candidate_wrong_source_seat")
        plan = _matching_chi_plan(view, candidate, rules)
        if plan is None:
            raise ValueError("chi_candidate_plan_mismatch")
        hand_after = list(view.hand)
        for label in plan.consumed_from_hand:
            hand_after.remove(label)
    else:
        raise ValueError(f"unsupported_response_action:{action_type}")
    discardable = _discardable_labels(hand_after)
    if candidate.followup_discard is None:
        if not discardable:
            raise ValueError(f"{action_type.lower()}_followup_discard_unavailable")
    elif candidate.followup_discard not in discardable:
        raise ValueError(f"{action_type.lower()}_followup_discard_illegal")


def _matching_chi_plan(
    view: PublicView,
    candidate: RootResponseCandidate,
    rules: dict[str, Any],
) -> ChiPlan | None:
    expected_groups = _group_multiset_key(candidate.meld_groups)
    expected_consumed = Counter(candidate.consumed_from_hand)
    plans = enumerate_chi_plans(
        list(view.hand),
        str(view.pending_card),
        allow_1510=bool(rules.get("rules", {}).get("allow_1510", False)),
    )
    return next(
        (
            plan
            for plan in plans
            if Counter(plan.consumed_from_hand) == expected_consumed
            and _group_multiset_key(plan.groups) == expected_groups
        ),
        None,
    )


def _chi_plan_matches_candidate(
    plan: ChiPlan,
    candidate: RootResponseCandidate,
) -> bool:
    return (
        Counter(plan.consumed_from_hand) == Counter(candidate.consumed_from_hand)
        and _group_multiset_key(plan.groups) == _group_multiset_key(candidate.meld_groups)
    )


def _group_multiset_key(
    groups: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(tuple(sorted(group)) for group in groups))


class ProgressiveRootISMCTSPolicy:
    """Cover every root action once, then refine a small paired shortlist."""

    name = "progressive_root_ismcts_v1"

    def __init__(
        self,
        *,
        coverage_config: RootISMCTSConfig | None = None,
        refinement_config: RootISMCTSConfig | None = None,
        discard_selection_config: RootISMCTSConfig | None = None,
        confirmation_config: RootISMCTSConfig | None = None,
        response_selection_config: RootISMCTSConfig | None = None,
        response_binary_confirmation_config: RootISMCTSConfig | None = None,
        response_validation_config: RootISMCTSConfig | None = None,
        response_minimum_confident_advantage: float = 0.02,
        refinement_candidates: int = 5,
        discard_confirmation_alternatives: int = 2,
        direct_discard_confirmation_from_refinement: bool = False,
        reuse_candidate_priors_as_coverage: bool = False,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ] | None = None,
        root_continuation_policy_factory: Callable[[], SimulationPolicy] | None = None,
    ) -> None:
        self.refinement_candidates = max(2, refinement_candidates)
        self.discard_confirmation_alternatives = max(
            1,
            discard_confirmation_alternatives,
        )
        self.direct_discard_confirmation_from_refinement = bool(
            direct_discard_confirmation_from_refinement
        )
        self.reuse_candidate_priors_as_coverage = bool(
            reuse_candidate_priors_as_coverage
        )
        resolved_coverage = coverage_config or RootISMCTSConfig(
            time_budget_ms=1_800,
            max_iterations=24,
            max_candidates=24,
            skip_search_gap=math.inf,
            rollout_max_turns=120,
            require_confident_override=False,
            seed=20260726,
        )
        self.coverage = RootISMCTSPolicy(
            replace(
                resolved_coverage,
                complete_first_paired_batch=True,
            ),
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
        )
        resolved_refinement = refinement_config or RootISMCTSConfig(
            time_budget_ms=2_500,
            max_iterations=40,
            max_candidates=self.refinement_candidates,
            skip_search_gap=math.inf,
            rollout_max_turns=120,
            require_confident_override=True,
            min_confidence_pairs=6,
            seed=20260727,
        )
        self.refinement = RootISMCTSPolicy(
            resolved_refinement,
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
        )
        resolved_discard_selection = (
            discard_selection_config
            or RootISMCTSConfig(
                time_budget_ms=4_000,
                max_iterations=16 * self.refinement_candidates,
                max_candidates=self.refinement_candidates,
                skip_search_gap=math.inf,
                rollout_max_turns=120,
                require_confident_override=False,
                require_complete_iteration_budget_for_override=True,
                seed=20260729,
            )
        )
        if not resolved_discard_selection.require_complete_iteration_budget_for_override:
            resolved_discard_selection = replace(
                resolved_discard_selection,
                require_complete_iteration_budget_for_override=True,
            )
        used_discard_seeds = {
            resolved_coverage.seed,
            resolved_refinement.seed,
        }
        while resolved_discard_selection.seed in used_discard_seeds:
            resolved_discard_selection = replace(
                resolved_discard_selection,
                seed=resolved_discard_selection.seed + 1,
            )
        self.discard_selection = RootISMCTSPolicy(
            resolved_discard_selection,
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
        )
        resolved_confirmation = confirmation_config or RootISMCTSConfig(
            time_budget_ms=12_000,
            max_iterations=192,
            max_candidates=2,
            skip_search_gap=math.inf,
            rollout_max_turns=120,
            require_confident_override=True,
            min_confidence_pairs=32,
            minimum_confident_advantage=0.02,
            require_complete_iteration_budget_for_override=True,
            seed=20260728,
        )
        if not resolved_confirmation.require_confident_override:
            resolved_confirmation = replace(
                resolved_confirmation,
                require_confident_override=True,
            )
        if not resolved_confirmation.require_complete_iteration_budget_for_override:
            resolved_confirmation = replace(
                resolved_confirmation,
                require_complete_iteration_budget_for_override=True,
            )
        if resolved_confirmation.seed == resolved_refinement.seed:
            resolved_confirmation = replace(
                resolved_confirmation,
                seed=resolved_refinement.seed + 1,
            )
        while resolved_confirmation.seed == resolved_discard_selection.seed:
            resolved_confirmation = replace(
                resolved_confirmation,
                seed=resolved_confirmation.seed + 1,
            )
        confirmation_worlds = max(
            1,
            resolved_confirmation.max_iterations
            // max(1, resolved_confirmation.max_candidates),
        )
        self.discard_confirmation = RootISMCTSPolicy(
            replace(
                resolved_confirmation,
                max_candidates=1 + self.discard_confirmation_alternatives,
                max_iterations=(
                    confirmation_worlds
                    * (1 + self.discard_confirmation_alternatives)
                ),
            ),
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
        )
        resolved_response_selection = (
            response_selection_config
            or RootISMCTSConfig(
                time_budget_ms=6_000,
                max_iterations=160,
                max_candidates=self.refinement_candidates,
                skip_search_gap=math.inf,
                rollout_max_turns=120,
                require_confident_override=False,
                require_complete_iteration_budget_for_override=True,
                seed=20260728,
            )
        )
        resolved_response_binary = (
            response_binary_confirmation_config
            or RootISMCTSConfig(
                time_budget_ms=12_000,
                max_iterations=256,
                max_candidates=2,
                skip_search_gap=math.inf,
                rollout_max_turns=120,
                require_confident_override=True,
                min_confidence_pairs=32,
                minimum_confident_advantage=(
                    response_minimum_confident_advantage
                ),
                require_complete_iteration_budget_for_override=True,
                seed=20260728,
            )
        )
        resolved_response_validation = (
            response_validation_config
            or RootISMCTSConfig(
                time_budget_ms=8_000,
                max_iterations=168,
                max_candidates=2,
                skip_search_gap=math.inf,
                rollout_max_turns=120,
                require_confident_override=True,
                min_confidence_pairs=32,
                minimum_confident_advantage=(
                    response_minimum_confident_advantage
                ),
                require_complete_iteration_budget_for_override=True,
                seed=20260729,
            )
        )
        if not resolved_response_selection.require_complete_iteration_budget_for_override:
            resolved_response_selection = replace(
                resolved_response_selection,
                require_complete_iteration_budget_for_override=True,
            )
        if (
            not resolved_response_binary.require_confident_override
            or not resolved_response_binary.require_complete_iteration_budget_for_override
        ):
            resolved_response_binary = replace(
                resolved_response_binary,
                require_confident_override=True,
                require_complete_iteration_budget_for_override=True,
            )
        if (
            not resolved_response_validation.require_confident_override
            or not resolved_response_validation.require_complete_iteration_budget_for_override
        ):
            resolved_response_validation = replace(
                resolved_response_validation,
                require_confident_override=True,
                require_complete_iteration_budget_for_override=True,
            )
        if resolved_response_validation.seed == resolved_response_selection.seed:
            resolved_response_validation = replace(
                resolved_response_validation,
                seed=resolved_response_selection.seed + 1,
            )
        self.response_selection = RootISMCTSPolicy(
            resolved_response_selection,
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
        )
        self.response_binary_confirmation = RootISMCTSPolicy(
            resolved_response_binary,
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
        )
        self.response_confirmation = RootISMCTSPolicy(
            resolved_response_validation,
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
        )
        self.confirmation = self.discard_confirmation
        self.last_coverage: RootSearchResult | RootResponseSearchResult | None = None
        self.last_refinement: RootSearchResult | RootResponseSearchResult | None = None
        self.last_discard_selection: RootSearchResult | None = None
        self.last_response_selection: RootResponseSearchResult | None = None
        self.last_confirmation: RootSearchResult | RootResponseSearchResult | None = None

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
        labels = list(
            dict.fromkeys(
                candidate_labels or sorted(_discardable_labels(view.hand))
            )
        )
        if not labels:
            raise ValueError("no_progressive_discard_candidate")
        self._ensure_coverage_capacity(len(labels))
        preferred = preferred_label or labels[0]
        if (
            self.reuse_candidate_priors_as_coverage
            and candidate_priors is not None
            and all(label in candidate_priors for label in labels)
        ):
            coverage = _production_prior_discard_coverage(
                labels,
                candidate_priors=candidate_priors,
                preferred_label=preferred,
            )
        else:
            coverage = self.coverage.search_discard(
                view,
                rules=rules,
                candidate_labels=labels,
                candidate_priors=candidate_priors,
                force_search=True,
                paired_candidates=True,
                preferred_label=preferred,
                absolute_deadline=absolute_deadline,
            )
        self.last_coverage = coverage
        if not coverage.used_search:
            self.last_refinement = None
            self.last_discard_selection = None
            self.last_response_selection = None
            self.last_confirmation = None
            return coverage
        if _search_deadline_reached(absolute_deadline):
            self.last_refinement = None
            self.last_discard_selection = None
            self.last_response_selection = None
            self.last_confirmation = None
            return _progressive_discard_budget_fallback(
                coverage,
                preferred_label=preferred,
            )
        shortlist = _progressive_discard_shortlist(
            coverage,
            preferred_label=preferred,
            maximum=self.refinement_candidates,
        )
        refinement = self.refinement.search_discard(
            view,
            rules=rules,
            candidate_labels=shortlist,
            candidate_priors={
                item.label: item.average_reward
                for item in coverage.candidates
                if item.label in shortlist
            },
            force_search=True,
            paired_candidates=True,
            preferred_label=preferred,
            absolute_deadline=absolute_deadline,
        )
        self.last_refinement = refinement
        self.last_discard_selection = None
        self.last_response_selection = None
        confirmation = None
        empirical_alternative = refinement.empirical_best_label
        if (
            refinement.used_search
            and empirical_alternative is not None
            and empirical_alternative != preferred
        ):
            selection = None
            confirmation_source = refinement
            if (
                len(shortlist) > 2
                and not self.direct_discard_confirmation_from_refinement
            ):
                if _search_deadline_reached(absolute_deadline):
                    self.last_confirmation = None
                    return _progressive_discard_budget_fallback(
                        coverage,
                        refinement,
                        preferred_label=preferred,
                    )
                selection = self.discard_selection.search_discard(
                    view,
                    rules=rules,
                    candidate_labels=shortlist,
                    candidate_priors={
                        item.label: item.average_reward
                        for item in refinement.candidates
                    },
                    force_search=True,
                    paired_candidates=True,
                    preferred_label=preferred,
                    absolute_deadline=absolute_deadline,
                )
                if _override_iteration_budget_incomplete(
                    self.discard_selection.config,
                    candidate_count=len(shortlist),
                    paired_determinizations=selection.paired_determinizations,
                    deadline_interruptions=selection.deadline_interruptions,
                ):
                    selection = replace(
                        selection,
                        selected_label=preferred,
                        used_search=False,
                        reason="root_discard_selection_budget_incomplete",
                        confidence_override=False,
                    )
                self.last_discard_selection = selection
                confirmation_source = selection
            empirical_alternative = confirmation_source.empirical_best_label
            if (
                confirmation_source.used_search
                and empirical_alternative is not None
                and empirical_alternative != preferred
            ):
                if _search_deadline_reached(absolute_deadline):
                    self.last_confirmation = None
                    return _progressive_discard_budget_fallback(
                        coverage,
                        refinement,
                        selection,
                        preferred_label=preferred,
                    )
                confirmation_alternatives = _ranked_discard_alternatives(
                    confirmation_source,
                    preferred_label=preferred,
                    maximum=self.discard_confirmation_alternatives,
                )
                confirmation_labels = [preferred, *confirmation_alternatives]
                confirmation = self.discard_confirmation.search_discard(
                    view,
                    rules=rules,
                    candidate_labels=confirmation_labels,
                    candidate_priors={
                        item.label: item.average_reward
                        for item in confirmation_source.candidates
                        if item.label in confirmation_labels
                    },
                    force_search=True,
                    paired_candidates=True,
                    preferred_label=preferred,
                    absolute_deadline=absolute_deadline,
                )
        self.last_confirmation = confirmation
        return _merge_progressive_discard(
            coverage,
            refinement,
            preferred_label=preferred,
            selection=self.last_discard_selection,
            confirmation=confirmation,
        )

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
        if not candidates:
            raise ValueError("no_progressive_response_candidate")
        self._ensure_coverage_capacity(len(candidates))
        preferred = preferred_key or candidates[0].key
        coverage = self.coverage.search_response(
            view,
            rules=rules,
            candidates=candidates,
            force_search=True,
            preferred_key=preferred,
            absolute_deadline=absolute_deadline,
        )
        self.last_coverage = coverage
        if not coverage.used_search:
            self.last_refinement = None
            self.last_discard_selection = None
            self.last_response_selection = None
            self.last_confirmation = None
            return coverage
        if _search_deadline_reached(absolute_deadline):
            self.last_refinement = None
            self.last_response_selection = None
            self.last_confirmation = None
            return _progressive_response_budget_fallback(
                coverage,
                preferred_key=preferred,
            )
        shortlist_keys = _progressive_response_shortlist(
            coverage,
            preferred_key=preferred,
            maximum=self.refinement_candidates,
        )
        by_key = {candidate.key: candidate for candidate in candidates}
        shortlist = [by_key[key] for key in shortlist_keys]
        refinement = self.refinement.search_response(
            view,
            rules=rules,
            candidates=shortlist,
            force_search=True,
            preferred_key=preferred,
            absolute_deadline=absolute_deadline,
        )
        self.last_refinement = refinement
        self.last_discard_selection = None
        self.last_response_selection = None
        if _search_deadline_reached(absolute_deadline):
            self.last_confirmation = None
            return _progressive_response_budget_fallback(
                coverage,
                refinement,
                preferred_key=preferred,
            )
        empirical_alternative = refinement.empirical_best_key
        if (
            not refinement.used_search
            or empirical_alternative is None
            or empirical_alternative == preferred
        ):
            self.last_confirmation = None
            return _merge_progressive_response(
                coverage,
                preferred_key=preferred,
                refinement=refinement,
            )
        if len(shortlist) == 2:
            if _search_deadline_reached(absolute_deadline):
                self.last_confirmation = None
                return _progressive_response_budget_fallback(
                    coverage,
                    refinement,
                    preferred_key=preferred,
                )
            confirmation = self.response_binary_confirmation.search_response(
                view,
                rules=rules,
                candidates=shortlist,
                force_search=True,
                preferred_key=preferred,
                absolute_deadline=absolute_deadline,
            )
            self.last_confirmation = confirmation
            return _merge_progressive_response(
                coverage,
                preferred_key=preferred,
                refinement=refinement,
                confirmation=confirmation,
            )
        if _search_deadline_reached(absolute_deadline):
            self.last_confirmation = None
            return _progressive_response_budget_fallback(
                coverage,
                refinement,
                preferred_key=preferred,
            )
        selection = self.response_selection.search_response(
            view,
            rules=rules,
            candidates=shortlist,
            force_search=True,
            preferred_key=preferred,
            absolute_deadline=absolute_deadline,
        )
        if _override_iteration_budget_incomplete(
            self.response_selection.config,
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
        self.last_response_selection = selection
        confirmation = None
        empirical_alternative = selection.empirical_best_key
        if (
            selection.used_search
            and empirical_alternative is not None
            and empirical_alternative != preferred
        ):
            if _search_deadline_reached(absolute_deadline):
                self.last_confirmation = None
                return _progressive_response_budget_fallback(
                    coverage,
                    refinement,
                    selection,
                    preferred_key=preferred,
                )
            confirmation = self.response_confirmation.search_response(
                view,
                rules=rules,
                candidates=[
                    by_key[preferred],
                    by_key[empirical_alternative],
                ],
                force_search=True,
                preferred_key=preferred,
                absolute_deadline=absolute_deadline,
            )
        self.last_confirmation = confirmation
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
        refinement = self.refinement.search_hu(
            view,
            rules=rules,
            preferred_key=preferred_key,
        )
        self.last_coverage = None
        self.last_refinement = refinement
        self.last_discard_selection = None
        self.last_response_selection = None
        empirical_alternative = refinement.empirical_best_key
        if (
            not refinement.used_search
            or empirical_alternative is None
            or empirical_alternative == preferred_key
        ):
            self.last_confirmation = None
            return refinement
        confirmation = self.response_confirmation.search_hu(
            view,
            rules=rules,
            preferred_key=preferred_key,
        )
        self.last_confirmation = confirmation
        return _merge_progressive_response(
            refinement,
            preferred_key=preferred_key,
            confirmation=confirmation,
        )

    def _ensure_coverage_capacity(self, candidate_count: int) -> None:
        config = self.coverage.config
        required_candidates = max(config.max_candidates, candidate_count)
        required_iterations = max(config.max_iterations, candidate_count)
        if (
            required_candidates != config.max_candidates
            or required_iterations != config.max_iterations
        ):
            self.coverage.config = replace(
                config,
                max_candidates=required_candidates,
                max_iterations=required_iterations,
            )


def _progressive_discard_shortlist(
    coverage: RootSearchResult,
    *,
    preferred_label: str,
    maximum: int,
) -> list[str]:
    top_set = _confidence_top_set(coverage.candidates)
    reward_ranked = [
        item.label
        for item in sorted(
            coverage.candidates,
            key=lambda item: (
                _candidate_rank_key(item),
                item.heuristic_value,
                item.label,
            ),
            reverse=True,
        )
        if item.label in top_set
    ]
    prior_ranked = [
        item.label
        for item in sorted(
            coverage.candidates,
            key=lambda item: (
                item.heuristic_value,
                item.average_reward,
                item.label,
            ),
            reverse=True,
        )
    ]
    return _protected_progressive_shortlist(
        preferred_label,
        reward_ranked=reward_ranked,
        prior_ranked=prior_ranked,
        maximum=maximum,
    )


def _production_prior_discard_coverage(
    labels: Sequence[str],
    *,
    candidate_priors: Mapping[str, float],
    preferred_label: str,
) -> RootSearchResult:
    """Represent the already-complete production evaluation without rerolling."""

    candidates = tuple(
        RootCandidateStats(
            label=label,
            visits=0,
            reward_sum=0.0,
            average_reward=0.0,
            win_rate=0.0,
            heuristic_value=float(candidate_priors[label]),
        )
        for label in labels
    )
    empirical_best = max(
        candidates,
        key=lambda item: (
            item.heuristic_value,
            item.label == preferred_label,
            item.label,
        ),
    )
    return RootSearchResult(
        selected_label=preferred_label,
        used_search=True,
        reason="production_prior_full_coverage",
        simulations=0,
        elapsed_ms=0.0,
        candidates=candidates,
        empirical_best_label=empirical_best.label,
    )


def _progressive_response_shortlist(
    coverage: RootResponseSearchResult,
    *,
    preferred_key: str,
    maximum: int,
) -> list[str]:
    top_set = _confidence_top_set(coverage.candidates)
    reward_ranked = [
        item.candidate.key
        for item in sorted(
            coverage.candidates,
            key=lambda item: (
                _candidate_rank_key(item),
                item.candidate.heuristic_value,
                item.candidate.key,
            ),
            reverse=True,
        )
        if item.candidate.key in top_set
    ]
    prior_ranked = [
        item.candidate.key
        for item in sorted(
            coverage.candidates,
            key=lambda item: (
                item.candidate.heuristic_value,
                item.average_reward,
                item.candidate.key,
            ),
            reverse=True,
        )
    ]
    return _protected_progressive_shortlist(
        preferred_key,
        reward_ranked=reward_ranked,
        prior_ranked=prior_ranked,
        maximum=maximum,
    )


def _confidence_top_set(
    candidates: Sequence[Any],
    *,
    familywise_alpha: float = 0.05,
) -> set[str]:
    """Keep arms not yet statistically dominated on P(win)."""

    if not candidates:
        return set()
    keyed = [
        (
            _root_candidate_key(item),
            *_bernoulli_confidence_interval(
                float(item.win_rate),
                int(item.visits),
                comparisons=max(1, len(candidates)),
                alpha=familywise_alpha,
            ),
        )
        for item in candidates
    ]
    best_lower = max(lower for _key, lower, _upper in keyed)
    return {
        key
        for key, _lower, upper in keyed
        if upper >= best_lower
    }


def _bernoulli_confidence_interval(
    mean_value: float,
    samples: int,
    *,
    comparisons: int,
    alpha: float,
) -> tuple[float, float]:
    if samples <= 0:
        return 0.0, 1.0
    radius = math.sqrt(
        math.log(2.0 * max(1, comparisons) / max(1e-12, alpha))
        / (2.0 * samples)
    )
    return max(0.0, mean_value - radius), min(1.0, mean_value + radius)


def _minimum_regret_key(
    candidates: Sequence[Any],
    *,
    preferred_key: str,
) -> str:
    if not candidates:
        return preferred_key
    intervals = []
    for item in candidates:
        key = _root_candidate_key(item)
        lower, upper = _bernoulli_confidence_interval(
            float(item.win_rate),
            int(item.visits),
            comparisons=max(1, len(candidates)),
            alpha=0.05,
        )
        intervals.append((key, lower, upper, _candidate_rank_key(item)))
    best_upper = max(upper for _key, _lower, upper, _rank in intervals)
    return min(
        intervals,
        key=lambda row: (
            max(0.0, best_upper - row[1]),
            row[0] != preferred_key,
            tuple(-value for value in row[3]),
            row[0],
        ),
    )[0]


def _root_candidate_key(item: Any) -> str:
    label = getattr(item, "label", None)
    if label is not None:
        return str(label)
    return str(getattr(getattr(item, "candidate", None), "key", ""))


def _protected_progressive_shortlist(
    preferred_key: str,
    *,
    reward_ranked: Sequence[str],
    prior_ranked: Sequence[str],
    maximum: int,
) -> list[str]:
    limit = max(2, maximum)
    selected = [preferred_key]
    nominees = [
        *reward_ranked[:1],
        *[
            key
            for key in prior_ranked
            if key != preferred_key
        ][:1],
        *reward_ranked,
        *prior_ranked,
    ]
    for key in nominees:
        if key not in selected:
            selected.append(key)
        if len(selected) >= limit:
            break
    return selected


def _ranked_discard_alternatives(
    refinement: RootSearchResult,
    *,
    preferred_label: str,
    maximum: int,
) -> list[str]:
    return [
        item.label
        for item in sorted(
            refinement.candidates,
            key=lambda item: (
                _candidate_rank_key(item),
                item.heuristic_value,
                item.label,
            ),
            reverse=True,
        )
        if item.label != preferred_label
    ][: max(1, maximum)]


def _ranked_response_alternatives(
    refinement: RootResponseSearchResult,
    *,
    preferred_key: str,
    maximum: int,
) -> list[str]:
    return [
        item.candidate.key
        for item in sorted(
            refinement.candidates,
            key=lambda item: (
                _candidate_rank_key(item),
                item.candidate.heuristic_value,
                item.candidate.key,
            ),
            reverse=True,
        )
        if item.candidate.key != preferred_key
    ][: max(1, maximum)]


def _combined_discard_confirmation(
    refinement: RootSearchResult,
    confirmation: RootSearchResult,
    *,
    preferred_label: str,
    alternative_label: str,
    minimum_samples: int,
    minimum_advantage: float,
    familywise_comparisons: int,
) -> RootSearchResult:
    return _combined_discard_confirmations(
        refinement,
        confirmation,
        preferred_key=preferred_label,
        alternative_labels=(alternative_label,),
        minimum_samples=minimum_samples,
        minimum_advantage=minimum_advantage,
        familywise_comparisons=familywise_comparisons,
    )


def _combined_discard_confirmations(
    refinement: RootSearchResult,
    confirmation: RootSearchResult,
    *,
    preferred_label: str | None = None,
    preferred_key: str | None = None,
    alternative_labels: Sequence[str],
    minimum_samples: int,
    minimum_advantage: float,
    familywise_comparisons: int,
) -> RootSearchResult:
    preferred = preferred_label or preferred_key
    if preferred is None:
        raise ValueError("combined_discard_preferred_required")
    combined = tuple(
        advantage
        for alternative in alternative_labels
        if (
            advantage := _combine_paired_advantage(
                refinement.paired_advantages,
                confirmation.paired_advantages,
                preferred_key=preferred,
                alternative_key=alternative,
                familywise_comparisons=familywise_comparisons,
            )
        )
        is not None
    )
    usable = (
        refinement.used_search
        and confirmation.used_search
        and len(combined) == len(alternative_labels)
        and all(item.samples >= minimum_samples for item in combined)
    )
    confident = [
        item
        for item in combined
        if item.lower_confidence_bound is not None
        and item.lower_confidence_bound > minimum_advantage
    ]
    selected = (
        max(
            confident,
            key=lambda item: (
                item.mean_delta,
                item.lower_confidence_bound,
                item.candidate_key,
            ),
        ).candidate_key
        if usable and confident
        else preferred
    )
    empirical = max(
        combined,
        key=lambda item: (item.mean_delta, item.candidate_key),
        default=None,
    )
    override = usable and selected != preferred
    return replace(
        confirmation,
        selected_label=selected,
        used_search=usable,
        reason=(
            "progressive_combined_confirmation_override"
            if override
            else "progressive_combined_confirmation_kept_preferred"
            if usable
            else "progressive_combined_confirmation_unusable"
        ),
        empirical_best_label=(
            empirical.candidate_key
            if empirical is not None and empirical.mean_delta > 0.0
            else preferred
        ),
        confidence_override=override,
        paired_advantages=combined,
    )


def _combined_response_confirmation(
    refinement: RootResponseSearchResult,
    confirmation: RootResponseSearchResult,
    *,
    preferred_key: str,
    alternative_key: str,
    minimum_samples: int,
    minimum_advantage: float,
    familywise_comparisons: int,
) -> RootResponseSearchResult:
    return _combined_response_confirmations(
        refinement,
        confirmation,
        preferred_key=preferred_key,
        alternative_keys=(alternative_key,),
        minimum_samples=minimum_samples,
        minimum_advantage=minimum_advantage,
        familywise_comparisons=familywise_comparisons,
    )


def _combined_response_confirmations(
    refinement: RootResponseSearchResult,
    confirmation: RootResponseSearchResult,
    *,
    preferred_key: str,
    alternative_keys: Sequence[str],
    minimum_samples: int,
    minimum_advantage: float,
    familywise_comparisons: int,
) -> RootResponseSearchResult:
    combined = tuple(
        advantage
        for alternative in alternative_keys
        if (
            advantage := _combine_paired_advantage(
                refinement.paired_advantages,
                confirmation.paired_advantages,
                preferred_key=preferred_key,
                alternative_key=alternative,
                familywise_comparisons=familywise_comparisons,
            )
        )
        is not None
    )
    usable = (
        refinement.used_search
        and confirmation.used_search
        and len(combined) == len(alternative_keys)
        and all(item.samples >= minimum_samples for item in combined)
    )
    confident = [
        item
        for item in combined
        if item.lower_confidence_bound is not None
        and item.lower_confidence_bound > minimum_advantage
    ]
    selected = (
        max(
            confident,
            key=lambda item: (
                item.mean_delta,
                item.lower_confidence_bound,
                item.candidate_key,
            ),
        ).candidate_key
        if usable and confident
        else preferred_key
    )
    empirical = max(
        combined,
        key=lambda item: (item.mean_delta, item.candidate_key),
        default=None,
    )
    override = usable and selected != preferred_key
    return replace(
        confirmation,
        selected_key=selected,
        used_search=usable,
        reason=(
            "progressive_combined_confirmation_override"
            if override
            else "progressive_combined_confirmation_kept_preferred"
            if usable
            else "progressive_combined_confirmation_unusable"
        ),
        empirical_best_key=(
            empirical.candidate_key
            if empirical is not None and empirical.mean_delta > 0.0
            else preferred_key
        ),
        confidence_override=override,
        paired_advantages=combined,
    )


def _combine_paired_advantage(
    first: Sequence[PairedAdvantageStats],
    second: Sequence[PairedAdvantageStats],
    *,
    preferred_key: str,
    alternative_key: str,
    familywise_comparisons: int | None = None,
) -> PairedAdvantageStats | None:
    first_row = next(
        (
            row
            for row in first
            if row.preferred_key == preferred_key
            and row.candidate_key == alternative_key
        ),
        None,
    )
    second_row = next(
        (
            row
            for row in second
            if row.preferred_key == preferred_key
            and row.candidate_key == alternative_key
        ),
        None,
    )
    if first_row is None or second_row is None:
        return None
    first_n = first_row.samples
    second_n = second_row.samples
    samples = first_n + second_n
    if samples <= 0:
        return None
    mean_delta = (
        first_row.mean_delta * first_n
        + second_row.mean_delta * second_n
    ) / samples
    if samples >= 2:
        first_m2 = max(0, first_n - 1) * first_row.sample_stddev**2
        second_m2 = max(0, second_n - 1) * second_row.sample_stddev**2
        between_m2 = (
            (first_row.mean_delta - second_row.mean_delta) ** 2
            * first_n
            * second_n
            / samples
        )
        sample_stddev = math.sqrt(
            (first_m2 + second_m2 + between_m2) / (samples - 1)
        )
        standard_error = sample_stddev / math.sqrt(samples)
        comparisons = max(
            1,
            familywise_comparisons
            if familywise_comparisons is not None
            else max(
                first_row.familywise_comparisons,
                second_row.familywise_comparisons,
            ),
        )
        radius = _familywise_95_t_critical(
            samples - 1,
            comparisons=comparisons,
        ) * standard_error
        lower_bound: float | None = mean_delta - radius
        upper_bound: float | None = mean_delta + radius
    else:
        sample_stddev = 0.0
        standard_error = 0.0
        comparisons = max(
            1,
            familywise_comparisons
            if familywise_comparisons is not None
            else max(
                first_row.familywise_comparisons,
                second_row.familywise_comparisons,
            ),
        )
        lower_bound = None
        upper_bound = None
    return PairedAdvantageStats(
        candidate_key=alternative_key,
        preferred_key=preferred_key,
        samples=samples,
        mean_delta=mean_delta,
        sample_stddev=sample_stddev,
        standard_error=standard_error,
        lower_confidence_bound=lower_bound,
        upper_confidence_bound=upper_bound,
        positive_samples=(
            first_row.positive_samples + second_row.positive_samples
        ),
        tied_samples=first_row.tied_samples + second_row.tied_samples,
        negative_samples=(
            first_row.negative_samples + second_row.negative_samples
        ),
        familywise_comparisons=comparisons,
        confidence_level=min(
            first_row.confidence_level,
            second_row.confidence_level,
        ),
        objective_metric=first_row.objective_metric,
        mean_delta_p_win=_weighted_optional_mean(
            first_row.mean_delta_p_win,
            first_n,
            second_row.mean_delta_p_win,
            second_n,
        ),
        mean_delta_p_loss=_weighted_optional_mean(
            first_row.mean_delta_p_loss,
            first_n,
            second_row.mean_delta_p_loss,
            second_n,
        ),
        mean_delta_score=_weighted_optional_mean(
            first_row.mean_delta_score,
            first_n,
            second_row.mean_delta_score,
            second_n,
        ),
        mean_delta_xi=_weighted_optional_mean(
            first_row.mean_delta_xi,
            first_n,
            second_row.mean_delta_xi,
            second_n,
        ),
    )


def _weighted_optional_mean(
    first: float | None,
    first_n: int,
    second: float | None,
    second_n: int,
) -> float | None:
    if first is None or second is None or first_n + second_n <= 0:
        return None
    return (first * first_n + second * second_n) / (first_n + second_n)


def _merge_progressive_discard(
    coverage: RootSearchResult,
    refinement: RootSearchResult,
    *,
    preferred_label: str,
    selection: RootSearchResult | None = None,
    confirmation: RootSearchResult | None = None,
) -> RootSearchResult:
    stages = tuple(
        stage
        for stage in (coverage, refinement, selection, confirmation)
        if stage is not None
    )
    final = confirmation or selection or refinement
    usable = all(stage.used_search for stage in stages)
    return RootSearchResult(
        selected_label=(
            final.selected_label
            if usable
            else preferred_label
        ),
        used_search=usable,
        reason="progressive_all_action:" + ":".join(stage.reason for stage in stages),
        simulations=sum(stage.simulations for stage in stages),
        elapsed_ms=sum(stage.elapsed_ms for stage in stages),
        candidates=coverage.candidates,
        determinization_failures=sum(
            stage.determinization_failures for stage in stages
        ),
        paired_determinizations=sum(
            stage.paired_determinizations for stage in stages
        ),
        deadline_interruptions=sum(
            stage.deadline_interruptions for stage in stages
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
        empirical_best_label=final.empirical_best_label,
        confidence_override=usable and final.confidence_override,
        paired_advantages=final.paired_advantages,
        paired_worlds=final.paired_worlds,
    )


def _progressive_discard_budget_fallback(
    coverage: RootSearchResult,
    refinement: RootSearchResult | None = None,
    selection: RootSearchResult | None = None,
    *,
    preferred_label: str,
) -> RootSearchResult:
    stages = tuple(
        stage
        for stage in (coverage, refinement, selection)
        if stage is not None
    )
    final = stages[-1]
    minimum_regret_label = _minimum_regret_key(
        final.candidates,
        preferred_key=preferred_label,
    )
    return RootSearchResult(
        selected_label=minimum_regret_label,
        used_search=all(stage.used_search for stage in stages),
        reason=(
            "progressive_all_action:"
            + ":".join(stage.reason for stage in stages)
            + ":overall_budget_minimum_regret"
        ),
        simulations=sum(stage.simulations for stage in stages),
        elapsed_ms=sum(stage.elapsed_ms for stage in stages),
        candidates=coverage.candidates,
        determinization_failures=sum(
            stage.determinization_failures for stage in stages
        ),
        paired_determinizations=sum(
            stage.paired_determinizations for stage in stages
        ),
        deadline_interruptions=sum(
            stage.deadline_interruptions for stage in stages
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
        empirical_best_label=final.empirical_best_label,
        confidence_override=minimum_regret_label != preferred_label,
        paired_advantages=final.paired_advantages,
        paired_worlds=final.paired_worlds,
    )


def _search_deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _progressive_response_budget_fallback(
    coverage: RootResponseSearchResult,
    refinement: RootResponseSearchResult | None = None,
    selection: RootResponseSearchResult | None = None,
    *,
    preferred_key: str,
) -> RootResponseSearchResult:
    stages = tuple(
        stage
        for stage in (coverage, refinement, selection)
        if stage is not None
    )
    final = stages[-1]
    minimum_regret_key = _minimum_regret_key(
        final.candidates,
        preferred_key=preferred_key,
    )
    merged = _merge_progressive_response(
        coverage,
        preferred_key=preferred_key,
        refinement=refinement,
        selection=selection,
    )
    return replace(
        merged,
        selected_key=minimum_regret_key,
        reason=f"{merged.reason}:overall_budget_minimum_regret",
        confidence_override=minimum_regret_key != preferred_key,
    )


def _merge_progressive_response(
    coverage: RootResponseSearchResult,
    *,
    preferred_key: str,
    refinement: RootResponseSearchResult | None = None,
    selection: RootResponseSearchResult | None = None,
    confirmation: RootResponseSearchResult | None = None,
) -> RootResponseSearchResult:
    stages = tuple(
        stage
        for stage in (coverage, refinement, selection, confirmation)
        if stage is not None
    )
    final = confirmation or selection or refinement or coverage
    usable = all(stage.used_search for stage in stages)
    return RootResponseSearchResult(
        selected_key=(
            final.selected_key
            if usable
            else preferred_key
        ),
        used_search=usable,
        reason="progressive_all_action:" + ":".join(stage.reason for stage in stages),
        simulations=sum(stage.simulations for stage in stages),
        elapsed_ms=sum(stage.elapsed_ms for stage in stages),
        candidates=coverage.candidates,
        determinization_failures=sum(
            stage.determinization_failures for stage in stages
        ),
        paired_determinizations=sum(
            stage.paired_determinizations for stage in stages
        ),
        deadline_interruptions=sum(
            stage.deadline_interruptions for stage in stages
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
        empirical_best_key=final.empirical_best_key,
        confidence_override=usable and final.confidence_override,
        paired_advantages=final.paired_advantages,
        paired_worlds=final.paired_worlds,
    )


class ProfessionalSearchSimulationPolicy(ProfessionalBrainSimulationPolicy):
    """Production EV and safety filters followed by selective root ISMCTS."""

    name = "professional_search_v3_shadow"

    def __init__(
        self,
        *,
        root_config: RootISMCTSConfig | None = None,
        activation_ev_gap: float = 120.0,
        response_max_candidates: int | None = 4,
        discard_max_candidates: int | None = 2,
        apply_discard_search: bool = True,
        apply_discard_selection: bool = True,
        apply_response_selection: bool = False,
        apply_hu_selection: bool = False,
        rollout_policy_factories: Sequence[Callable[[], SimulationPolicy]] | None = None,
        root_continuation_policy_factory: Callable[[], SimulationPolicy] | None = None,
        capture_all_response_roots: bool = False,
        capture_all_discard_roots: bool = False,
        include_policy_rejected_actions: bool = False,
        respect_response_safety_flags: bool = True,
    ) -> None:
        resolved_config = root_config or RootISMCTSConfig()
        self.objective_mode = resolved_config.objective_mode
        self.objective_version = resolved_config.objective_version
        self.search = RootISMCTSPolicy(
            resolved_config,
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
        )
        self.response_search = RootISMCTSPolicy(
            replace(
                resolved_config,
                max_candidates=(
                    resolved_config.max_candidates
                    if response_max_candidates is None
                    else max(2, response_max_candidates)
                ),
            ),
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
        )
        self.activation_ev_gap = activation_ev_gap
        self.response_max_candidates = (
            None
            if response_max_candidates is None
            else max(2, response_max_candidates)
        )
        self.discard_max_candidates = (
            None
            if discard_max_candidates is None
            else max(2, discard_max_candidates)
        )
        self.apply_discard_search = apply_discard_search
        self.apply_discard_selection = apply_discard_selection
        self.apply_response_selection = apply_response_selection
        self.apply_hu_selection = apply_hu_selection
        self.capture_all_response_roots = capture_all_response_roots
        self.capture_all_discard_roots = capture_all_discard_roots
        self.include_policy_rejected_actions = include_policy_rejected_actions
        self.respect_response_safety_flags = respect_response_safety_flags
        self.last_search: RootSearchResult | None = None
        self.last_response_search: RootResponseSearchResult | None = None
        self.last_hu_search: RootResponseSearchResult | None = None
        self.search_attempts = 0
        self.search_overrides = 0
        self.search_confidence_overrides = 0
        self.search_simulations = 0
        self.search_elapsed_ms = 0.0
        self.search_paired_determinizations = 0
        self.search_deadline_interruptions = 0
        self.search_candidate_coverage_failures = 0
        self.search_rollout_coverage_failures = 0
        self.response_search_opportunities = 0
        self.response_search_attempts = 0
        self.response_search_usable = 0
        self.response_search_disagreements = 0
        self.response_search_simulations = 0
        self.response_paired_determinizations = 0
        self.response_deadline_interruptions = 0
        self.response_search_contract_failures = 0
        self.response_search_elapsed_ms = 0.0
        self.hu_search_opportunities = 0
        self.hu_search_attempts = 0
        self.hu_search_overrides = 0
        self.hu_search_simulations = 0
        self.hu_search_elapsed_ms = 0.0
        self._response_events: list[dict[str, Any]] = []
        self._response_events_dropped = 0
        self._discard_events: list[dict[str, Any]] = []
        self._discard_events_dropped = 0
        self._pending_response_key: str | None = None
        self._pending_response_candidate: RootResponseCandidate | None = None
        self._pending_response_card: str | None = None
        self._pending_response_source_seat: int | None = None

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        if not self.apply_discard_search:
            self.last_search = None
            return super().choose_discard(view, rules)
        decision = self._choose(view, rules, legal_actions=[{"type": "DISCARD"}])
        if decision.selected_action != "DISCARD" or not decision.selected_label:
            raise ValueError(f"professional_search_no_discard:{decision.selected_action}")
        legal_labels = set(_discardable_labels(view.hand))
        by_label: dict[str, Any] = {}
        for item in decision.action_evals:
            label = item.action.label
            if (
                item.type != "DISCARD"
                or not label
                or label not in legal_labels
                or (
                    not item.allowed
                    and not self.include_policy_rejected_actions
                )
            ):
                continue
            previous = by_label.get(label)
            if previous is None or item.ev > previous.ev:
                by_label[label] = item
        ranked_candidates = sorted(
            by_label.values(),
            key=lambda item: item.ev,
            reverse=True,
        )
        candidates = (
            ranked_candidates
            if self.discard_max_candidates is None
            else ranked_candidates[: self.discard_max_candidates]
        )
        base_event: dict[str, Any] = {
            "production_label": decision.selected_label,
            "legal_candidate_count": len(ranked_candidates),
            "searched_candidate_count": len(candidates),
        }
        if self.capture_all_discard_roots:
            base_event["public_view"] = public_view_to_dict(view)
        if len(candidates) < 2 or candidates[0].ev - candidates[1].ev > self.activation_ev_gap:
            self.last_search = None
            self._record_discard_event(
                {
                    **base_event,
                    "search_attempted": False,
                    "used_search": False,
                    "reason": (
                        "fewer_than_two_discard_candidates"
                        if len(candidates) < 2
                        else "heuristic_gap"
                    ),
                }
            )
            return decision.selected_label
        labels = [item.action.label for item in candidates]
        priors = {item.action.label: item.ev for item in candidates}
        _ensure_unbounded_root_capacity(
            self.search,
            candidate_count=len(labels),
            unbounded=self.discard_max_candidates is None,
        )
        self.last_search = self.search.search_discard(
            view,
            rules=rules,
            candidate_labels=labels,
            candidate_priors=priors,
            force_search=True,
            paired_candidates=True,
            preferred_label=decision.selected_label,
        )
        self.search_attempts += 1
        self.search_simulations += self.last_search.simulations
        self.search_elapsed_ms += self.last_search.elapsed_ms
        self.search_paired_determinizations += (
            self.last_search.paired_determinizations
        )
        self.search_deadline_interruptions += (
            self.last_search.deadline_interruptions
        )
        self.search_candidate_coverage_failures += int(
            any(item.visits == 0 for item in self.last_search.candidates)
        )
        self.search_rollout_coverage_failures += (
            self.last_search.rollout_coverage_failures
        )
        self.search_overrides += int(self.last_search.selected_label != decision.selected_label)
        self.search_confidence_overrides += int(self.last_search.confidence_override)
        disagreement = self.last_search.selected_label != decision.selected_label
        event = {
            **base_event,
            "search_attempted": True,
            "search_selected_label": self.last_search.selected_label,
            "used_search": self.last_search.used_search,
            "reason": self.last_search.reason,
            "simulations": self.last_search.simulations,
            "paired_determinizations": self.last_search.paired_determinizations,
            "deadline_interruptions": self.last_search.deadline_interruptions,
            "rollout_invariant_violations": (
                self.last_search.rollout_invariant_violations
            ),
            "rollout_violations": list(self.last_search.rollout_violations),
            "rollout_coverage_failures": (
                self.last_search.rollout_coverage_failures
            ),
            "rollout_coverage_reasons": list(
                self.last_search.rollout_coverage_reasons
            ),
            "elapsed_ms": round(self.last_search.elapsed_ms, 3),
            "disagreement": disagreement,
            "empirical_best_label": self.last_search.empirical_best_label,
            "confidence_override": self.last_search.confidence_override,
            "paired_advantages": [
                advantage.to_dict()
                for advantage in self.last_search.paired_advantages
            ],
            "candidates": [
                candidate.to_dict()
                for candidate in self.last_search.candidates
            ],
        }
        if disagreement or self.capture_all_discard_roots:
            event["public_view"] = public_view_to_dict(view)
        self._record_discard_event(event)
        if self.apply_discard_selection:
            return self.last_search.selected_label
        return decision.selected_label

    def choose_hu(
        self,
        view: PublicView,
        hu: Any,
        rules: dict[str, Any],
    ) -> bool:
        production_accepts = super().choose_hu(view, hu, rules)
        if hu.can_hu and self.objective_mode is ObjectiveMode.WIN_FIRST:
            self.last_hu_search = None
            return True
        if view.pending_card is not None or not hu.can_hu:
            self.last_hu_search = None
            return production_accepts
        self.hu_search_opportunities += 1
        self.hu_search_attempts += 1
        self.last_hu_search = self.search.search_hu(
            view,
            rules=rules,
            preferred_key="HU",
        )
        search = self.last_hu_search
        self.hu_search_simulations += search.simulations
        self.hu_search_elapsed_ms += search.elapsed_ms
        if not self.apply_hu_selection or not search.used_search:
            return production_accepts
        selected_hu = search.selected_key == "HU"
        self.hu_search_overrides += int(selected_hu != production_accepts)
        return selected_hu

    def choose_response(
        self,
        view: PublicView,
        legal_actions: Sequence[Any],
        plans: Sequence[ChiPlan],
        hu: Any | None,
        rules: dict[str, Any],
        *,
        absolute_deadline: float | None = None,
        production_decision: Any | None = None,
    ) -> str:
        candidates = [
            RootResponseCandidate(
                key=str(action.key),
                action_type=str(action.type).upper(),
                heuristic_value=0.0,
                option_id=(
                    str(action.option_id)
                    if action.option_id is not None
                    else None
                ),
                consumed_from_hand=tuple(action.consumed_from_hand),
                meld_groups=tuple(
                    tuple(group)
                    for group in action.meld_groups
                ),
            )
            for action in legal_actions
        ]
        if (
            self.objective_mode is ObjectiveMode.WIN_FIRST
            and any(candidate.action_type == "HU" for candidate in candidates)
        ):
            self.last_response_search = None
            return "HU"
        production_key = "PASS"
        safety_flags: list[str] = []
        production_action = str(
            getattr(production_decision, "selected_action", "")
        ).upper()
        production_reused = False
        if production_action == "PASS":
            production_reused = True
        elif production_action == "PENG" and any(
            candidate.action_type == "PENG" for candidate in candidates
        ):
            production_key = f"PENG:{view.pending_card}"
            production_reused = True
        if production_reused:
            safety_flags.extend(
                getattr(production_decision, "safety_flags", ()) or ()
            )
        if not production_reused and hu is not None:
            hu_view = replace(view, hand=(*view.hand, str(view.pending_card)))
            hu_decision = self._choose(
                hu_view,
                rules,
                legal_actions=[{"type": "HU"}, {"type": "PASS"}],
                pending_card=view.pending_card,
            )
            self._remember_decision(hu_decision)
            safety_flags.extend(hu_decision.safety_flags)
            if hu_decision.selected_action == "HU":
                production_key = "HU"
        if not production_reused and production_key == "PASS" and any(
            candidate.action_type == "PENG"
            for candidate in candidates
        ):
            peng_decision = self._choose_peng_decision(
                view,
                str(view.pending_card),
                rules,
            )
            self._remember_decision(peng_decision)
            safety_flags.extend(peng_decision.safety_flags)
            if peng_decision.selected_action == "PENG":
                production_key = f"PENG:{view.pending_card}"
        if not production_reused and production_key == "PASS" and plans:
            chi_decision = self._choose_chi_decision(
                view,
                list(plans),
                rules,
            )
            self._remember_decision(chi_decision)
            safety_flags.extend(chi_decision.safety_flags)
            selected_plan = self._chi_plan_for_decision(
                chi_decision,
                list(plans),
            )
            if selected_plan is not None:
                matched = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.action_type == "CHI"
                        and _chi_plan_matches_candidate(
                            selected_plan,
                            candidate,
                        )
                    ),
                    None,
                )
                if matched is None:
                    raise ValueError("joint_production_chi_candidate_missing")
                production_key = matched.key
        self._observe_response_search(
            view,
            rules,
            decision=None,
            response_type="JOINT",
            production_key=production_key,
            candidates_override=candidates,
            safety_flags=tuple(dict.fromkeys(safety_flags)),
            absolute_deadline=absolute_deadline,
        )
        if (
            self.apply_response_selection
            and self.last_response_search is not None
            and self.last_response_search.used_search
        ):
            return self.last_response_search.selected_key
        return production_key

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        self._clear_pending_response()
        peng_decision = self._choose_peng_decision(view, label, rules)
        chi_plans = self._joint_response_chi_plans(view, label, rules)
        if not chi_plans:
            production_key = (
                "PENG:" + label
                if peng_decision.selected_action == "PENG"
                else "PASS"
            )
            self._observe_response_search(
                view,
                rules,
                decision=peng_decision,
                response_type="PENG",
                production_key=production_key,
            )
            if (
                self.apply_response_selection
                and self.last_response_search is not None
                and self.last_response_search.confidence_override
            ):
                return self.last_response_search.selected_key == "PENG:" + label
            return peng_decision.selected_action == "PENG"

        chi_decision = self._choose_chi_decision(view, chi_plans, rules)
        production_key = (
            "PENG:" + label
            if peng_decision.selected_action == "PENG"
            else (
                f"CHI:{chi_decision.selected_option_id}"
                if chi_decision.selected_action == "CHI"
                and chi_decision.selected_option_id
                else "PASS"
            )
        )
        candidates = build_response_candidates(
            label,
            (
                *peng_decision.action_evals,
                *chi_decision.action_evals,
            ),
        )
        self._observe_response_search(
            view,
            rules,
            decision=None,
            response_type="PENG_OR_CHI",
            production_key=production_key,
            candidates_override=candidates,
            safety_flags=tuple(
                dict.fromkeys(
                    (
                        *peng_decision.safety_flags,
                        *chi_decision.safety_flags,
                    )
                )
            ),
        )
        selected_key = production_key
        if (
            self.apply_response_selection
            and self.last_response_search is not None
            and self.last_response_search.confidence_override
        ):
            selected_key = self.last_response_search.selected_key
        selected_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.key == selected_key
            ),
            None,
        )
        self._remember_pending_response(
            view,
            selected_key=selected_key,
            selected_candidate=selected_candidate,
        )
        return selected_key == "PENG:" + label

    def choose_chi(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ) -> ChiPlan | None:
        if self._pending_response_matches(view):
            selected_key = self._pending_response_key
            selected_candidate = self._pending_response_candidate
            if selected_key is None or not selected_key.startswith("CHI:"):
                return None
            if selected_candidate is None:
                raise ValueError("joint_chi_candidate_missing")
            matched_plan = next(
                (
                    plan
                    for plan in plans
                    if _chi_plan_matches_candidate(plan, selected_candidate)
                ),
                None,
            )
            if matched_plan is None:
                raise ValueError("joint_chi_plan_missing")
            return matched_plan
        self._clear_pending_response()
        decision = self._choose_chi_decision(view, plans, rules)
        selected_plan = self._chi_plan_for_decision(decision, plans)
        production_key = (
            f"CHI:{decision.selected_option_id}"
            if decision.selected_action == "CHI" and decision.selected_option_id
            else "PASS"
        )
        self._observe_response_search(
            view,
            rules,
            decision=decision,
            response_type="CHI",
            production_key=production_key,
        )
        if (
            self.apply_response_selection
            and self.last_response_search is not None
            and self.last_response_search.confidence_override
        ):
            if self.last_response_search.selected_key == "PASS":
                return None
            selected_candidate = next(
                (
                    item.candidate
                    for item in self.last_response_search.candidates
                    if item.candidate.key == self.last_response_search.selected_key
                ),
                None,
            )
            if selected_candidate is None:
                raise ValueError("confidence_response_candidate_missing")
            matched_plan = next(
                (
                    plan
                    for plan in plans
                    if _chi_plan_matches_candidate(plan, selected_candidate)
                ),
                None,
            )
            if matched_plan is None:
                raise ValueError("confidence_chi_plan_missing")
            return matched_plan
        return selected_plan

    def finish_pending_response(self) -> None:
        self._clear_pending_response()

    def _joint_response_chi_plans(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ) -> list[ChiPlan]:
        source = view.pending_source_seat
        player_count = int(rules.get("game", {}).get("players", 3))
        if (
            source is None
            or (source + 1) % player_count != view.seat
            or normalize_card_label(label) == WILD_LABEL
        ):
            return []
        return enumerate_chi_plans(
            list(view.hand),
            label,
            allow_1510=bool(rules.get("rules", {}).get("allow_1510", False)),
        )

    def _remember_pending_response(
        self,
        view: PublicView,
        *,
        selected_key: str,
        selected_candidate: RootResponseCandidate | None,
    ) -> None:
        self._pending_response_key = selected_key
        self._pending_response_candidate = selected_candidate
        self._pending_response_card = view.pending_card
        self._pending_response_source_seat = view.pending_source_seat

    def _pending_response_matches(self, view: PublicView) -> bool:
        return (
            self._pending_response_key is not None
            and self._pending_response_card == view.pending_card
            and self._pending_response_source_seat == view.pending_source_seat
        )

    def _clear_pending_response(self) -> None:
        self._pending_response_key = None
        self._pending_response_candidate = None
        self._pending_response_card = None
        self._pending_response_source_seat = None

    def _observe_response_search(
        self,
        view: PublicView,
        rules: dict[str, Any],
        *,
        decision: Any | None,
        response_type: str,
        production_key: str,
        candidates_override: Sequence[RootResponseCandidate] | None = None,
        safety_flags: Sequence[str] | None = None,
        absolute_deadline: float | None = None,
    ) -> None:
        self.response_search_opportunities += 1
        base_event: dict[str, Any] = {
            "response_type": response_type,
            "pending_card": view.pending_card,
            "pending_source_seat": view.pending_source_seat,
            "production_key": production_key,
        }
        if self.capture_all_response_roots:
            base_event["public_view"] = public_view_to_dict(view)
        resolved_safety_flags = tuple(
            safety_flags
            if safety_flags is not None
            else getattr(decision, "safety_flags", ())
        )
        if resolved_safety_flags and self.respect_response_safety_flags:
            self.last_response_search = None
            self._record_response_event(
                {
                    **base_event,
                    "search_attempted": False,
                    "used_search": False,
                    "reason": "production_safety_flag",
                    "safety_flags": list(resolved_safety_flags),
                }
            )
            return
        candidates = list(
            candidates_override
            if candidates_override is not None
            else build_response_candidates(
                str(view.pending_card or ""),
                getattr(decision, "action_evals", ()),
                include_policy_rejected=self.include_policy_rejected_actions,
            )
        )
        if len(candidates) < 2:
            self.last_response_search = None
            self._record_response_event(
                {
                    **base_event,
                    "search_attempted": False,
                    "used_search": False,
                    "reason": "fewer_than_two_safe_response_candidates",
                    "candidate_count": len(candidates),
                }
            )
            return
        shortlist = shortlist_response_candidates(
            candidates,
            production_key=production_key,
            max_candidates=(
                len(candidates)
                if self.response_max_candidates is None
                else self.response_max_candidates
            ),
        )
        if len(shortlist) < 2:
            self.last_response_search = None
            self._record_response_event(
                {
                    **base_event,
                    "search_attempted": False,
                    "used_search": False,
                    "reason": "production_response_candidate_missing",
                    "candidate_count": len(candidates),
                }
            )
            return
        if view.pending_card is None or view.pending_source_seat is None:
            self.last_response_search = None
            self._record_response_event(
                {
                    **base_event,
                    "search_attempted": False,
                    "used_search": False,
                    "reason": "insufficient_response_public_state",
                    "candidate_count": len(candidates),
                }
            )
            return
        self.response_search_attempts += 1
        _ensure_unbounded_root_capacity(
            self.response_search,
            candidate_count=len(shortlist),
            unbounded=self.response_max_candidates is None,
        )
        try:
            self.last_response_search = self.response_search.search_response(
                view,
                rules=rules,
                candidates=shortlist,
                force_search=True,
                preferred_key=production_key,
                absolute_deadline=absolute_deadline,
            )
        except ValueError as exc:
            self.last_response_search = None
            self.response_search_contract_failures += 1
            self._record_response_event(
                {
                    **base_event,
                    "search_attempted": True,
                    "used_search": False,
                    "reason": f"invalid_response_search_contract:{exc}",
                    "candidate_count": len(candidates),
                }
            )
            return
        search = self.last_response_search
        self.response_search_simulations += search.simulations
        self.response_paired_determinizations += search.paired_determinizations
        self.response_deadline_interruptions += search.deadline_interruptions
        self.response_search_elapsed_ms += search.elapsed_ms
        disagreement = search.used_search and search.selected_key != production_key
        self.response_search_usable += int(search.used_search)
        self.response_search_disagreements += int(disagreement)
        event = {
            **base_event,
            "search_attempted": True,
            "search_selected_key": search.selected_key,
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
            "disagreement": disagreement,
            "empirical_best_key": search.empirical_best_key,
            "confidence_override": search.confidence_override,
            "paired_advantages": [
                advantage.to_dict()
                for advantage in search.paired_advantages
            ],
            "candidate_count": len(candidates),
            "candidates": [
                candidate.to_dict()
                for candidate in search.candidates
            ],
        }
        if disagreement or self.capture_all_response_roots:
            event["public_view"] = public_view_to_dict(view)
        self._record_response_event(event)

    def _record_response_event(self, event: dict[str, Any]) -> None:
        if len(self._response_events) < 512:
            self._response_events.append(event)
        else:
            self._response_events_dropped += 1

    def _record_discard_event(self, event: dict[str, Any]) -> None:
        if len(self._discard_events) < 512:
            self._discard_events.append(event)
        else:
            self._discard_events_dropped += 1

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "search_attempts": self.search_attempts,
            "search_overrides": self.search_overrides,
            "search_confidence_overrides": self.search_confidence_overrides,
            "search_simulations": self.search_simulations,
            "search_elapsed_ms": round(self.search_elapsed_ms, 3),
            "search_paired_determinizations": (
                self.search_paired_determinizations
            ),
            "search_deadline_interruptions": (
                self.search_deadline_interruptions
            ),
            "search_candidate_coverage_failures": (
                self.search_candidate_coverage_failures
            ),
            "search_rollout_coverage_failures": (
                self.search_rollout_coverage_failures
            ),
            "response_search_opportunities": self.response_search_opportunities,
            "response_search_attempts": self.response_search_attempts,
            "response_search_usable": self.response_search_usable,
            "response_search_disagreements": self.response_search_disagreements,
            "response_search_simulations": self.response_search_simulations,
            "response_paired_determinizations": self.response_paired_determinizations,
            "response_deadline_interruptions": self.response_deadline_interruptions,
            "response_search_contract_failures": self.response_search_contract_failures,
            "response_search_elapsed_ms": round(self.response_search_elapsed_ms, 3),
            "hu_search_opportunities": self.hu_search_opportunities,
            "hu_search_attempts": self.hu_search_attempts,
            "hu_search_overrides": self.hu_search_overrides,
            "hu_search_simulations": self.hu_search_simulations,
            "hu_search_elapsed_ms": round(self.hu_search_elapsed_ms, 3),
            "response_events_dropped": self._response_events_dropped,
            "discard_events_dropped": self._discard_events_dropped,
        }

    def response_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._response_events)

    def discard_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._discard_events)


class ProfessionalResponseShadowSimulationPolicy(ProfessionalSearchSimulationPolicy):
    """Production actions with response ISMCTS diagnostics only."""

    name = "professional_response_shadow_v3"

    def __init__(
        self,
        *,
        root_config: RootISMCTSConfig | None = None,
        response_max_candidates: int = 4,
    ) -> None:
        super().__init__(
            root_config=root_config,
            response_max_candidates=response_max_candidates,
            apply_discard_search=False,
            capture_all_response_roots=True,
        )


class ProfessionalConfidenceRootShadowPolicy(ProfessionalSearchSimulationPolicy):
    """Production actions with confidence-bounded root search diagnostics."""

    name = "professional_confidence_root_shadow_v1"

    def __init__(
        self,
        *,
        root_config: RootISMCTSConfig | None = None,
        activation_ev_gap: float = 120.0,
        response_max_candidates: int = 4,
        rollout_policy_factories: Sequence[Callable[[], SimulationPolicy]] | None = None,
    ) -> None:
        super().__init__(
            root_config=root_config
            or RootISMCTSConfig(
                time_budget_ms=2_500,
                max_iterations=32,
                max_candidates=2,
                rollout_max_turns=120,
                require_confident_override=True,
                min_confidence_pairs=8,
            ),
            activation_ev_gap=activation_ev_gap,
            response_max_candidates=response_max_candidates,
            apply_discard_search=True,
            apply_discard_selection=False,
            apply_response_selection=False,
            rollout_policy_factories=rollout_policy_factories,
            capture_all_response_roots=True,
        )


class ProfessionalConfidenceRootCandidatePolicy(ProfessionalConfidenceRootShadowPolicy):
    """Offline-only candidate that applies confidence-bounded discard overrides."""

    name = "professional_confidence_root_candidate_v1"

    def __init__(
        self,
        *,
        root_config: RootISMCTSConfig | None = None,
        activation_ev_gap: float = 120.0,
        response_max_candidates: int = 4,
        rollout_policy_factories: Sequence[Callable[[], SimulationPolicy]] | None = None,
    ) -> None:
        super().__init__(
            root_config=root_config,
            activation_ev_gap=activation_ev_gap,
            response_max_candidates=response_max_candidates,
            rollout_policy_factories=rollout_policy_factories,
        )
        self.apply_discard_selection = True
        self.apply_response_selection = True
        self.apply_hu_selection = True


class ProfessionalFullActionTeacherPolicy(ProfessionalSearchSimulationPolicy):
    """Offline teacher that gives every legal root action paired full rollouts."""

    name = "professional_full_action_teacher_v1"

    def __init__(
        self,
        *,
        root_config: RootISMCTSConfig | None = None,
        rollout_policy_factories: Sequence[Callable[[], SimulationPolicy]] | None = None,
    ) -> None:
        super().__init__(
            root_config=root_config
            or RootISMCTSConfig(
                time_budget_ms=12_000,
                max_iterations=192,
                max_candidates=24,
                skip_search_gap=math.inf,
                rollout_max_turns=120,
                require_confident_override=True,
                min_confidence_pairs=8,
                record_paired_worlds=True,
                complete_first_paired_batch=True,
            ),
            activation_ev_gap=math.inf,
            response_max_candidates=None,
            discard_max_candidates=None,
            apply_discard_search=True,
            apply_discard_selection=True,
            apply_response_selection=True,
            apply_hu_selection=True,
            rollout_policy_factories=rollout_policy_factories,
            capture_all_response_roots=True,
            capture_all_discard_roots=True,
            include_policy_rejected_actions=True,
            respect_response_safety_flags=False,
        )


class ProfessionalProgressiveAllActionCandidatePolicy(
    ProfessionalSearchSimulationPolicy
):
    """Runtime candidate with one-world coverage and confident refinement."""

    name = "professional_progressive_all_action_candidate_v1"

    def __init__(
        self,
        *,
        rollout_policy_factories: Sequence[
            Callable[[], SimulationPolicy]
        ] | None = None,
        root_continuation_policy_factory: Callable[[], SimulationPolicy] | None = None,
        response_minimum_confident_advantage: float = 0.02,
    ) -> None:
        super().__init__(
            root_config=RootISMCTSConfig(
                time_budget_ms=2_500,
                max_iterations=24,
                max_candidates=3,
                skip_search_gap=math.inf,
                rollout_max_turns=120,
                require_confident_override=True,
                min_confidence_pairs=6,
            ),
            activation_ev_gap=math.inf,
            response_max_candidates=None,
            discard_max_candidates=None,
            apply_discard_search=True,
            apply_discard_selection=True,
            apply_response_selection=True,
            apply_hu_selection=True,
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
            capture_all_response_roots=True,
            capture_all_discard_roots=True,
            include_policy_rejected_actions=True,
            respect_response_safety_flags=False,
        )
        progressive = ProgressiveRootISMCTSPolicy(
            rollout_policy_factories=rollout_policy_factories,
            root_continuation_policy_factory=root_continuation_policy_factory,
            response_minimum_confident_advantage=(
                response_minimum_confident_advantage
            ),
        )
        self.search = progressive
        self.response_search = progressive


class ProfessionalDiscardShadowSimulationPolicy(ProfessionalBrainSimulationPolicy):
    """Production discards with exact public roots captured for offline search."""

    name = "professional_discard_shadow_v4"

    def __init__(
        self,
        *,
        activation_ev_gap: float = 120.0,
        max_candidates: int = 2,
    ) -> None:
        self.activation_ev_gap = activation_ev_gap
        self.max_candidates = max(2, max_candidates)
        self.discard_opportunities = 0
        self.discard_roots_captured = 0
        self.discard_roots_skipped = 0
        self._discard_events: list[dict[str, Any]] = []
        self._discard_events_dropped = 0

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        decision = self._choose(view, rules, legal_actions=[{"type": "DISCARD"}])
        if decision.selected_action != "DISCARD" or not decision.selected_label:
            raise ValueError(
                "professional_discard_shadow_no_discard:"
                f"{decision.selected_action}:reason={decision.reason}"
            )
        self.discard_opportunities += 1
        by_label: dict[str, Any] = {}
        for item in decision.action_evals:
            label = item.action.label
            if item.type != "DISCARD" or not item.allowed or not label:
                continue
            previous = by_label.get(label)
            if previous is None or item.ev > previous.ev:
                by_label[label] = item
        candidates = sorted(
            by_label.values(),
            key=lambda item: (float(item.ev), str(item.action.label)),
            reverse=True,
        )[: self.max_candidates]
        event: dict[str, Any] = {
            "production_label": decision.selected_label,
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "label": str(item.action.label),
                    "heuristic_value": round(float(item.ev), 4),
                }
                for item in candidates
            ],
            "safety_flags": list(decision.safety_flags),
        }
        if len(candidates) < 2:
            event.update({"eligible": False, "reason": "fewer_than_two_safe_candidates"})
            self.discard_roots_skipped += 1
        else:
            gap = float(candidates[0].ev) - float(candidates[1].ev)
            event["heuristic_gap"] = round(gap, 4)
            if decision.safety_flags:
                event.update({"eligible": False, "reason": "production_safety_flag"})
                self.discard_roots_skipped += 1
            elif gap > self.activation_ev_gap:
                event.update({"eligible": False, "reason": "heuristic_gap_too_large"})
                self.discard_roots_skipped += 1
            else:
                event.update(
                    {
                        "eligible": True,
                        "reason": "captured_uncertain_discard_root",
                        "public_view": public_view_to_dict(view),
                    }
                )
                self.discard_roots_captured += 1
        self._record_discard_event(event)
        return decision.selected_label

    def _record_discard_event(self, event: dict[str, Any]) -> None:
        if len(self._discard_events) < 512:
            self._discard_events.append(event)
        else:
            self._discard_events_dropped += 1

    def diagnostics(self) -> dict[str, int]:
        return {
            "discard_opportunities": self.discard_opportunities,
            "discard_roots_captured": self.discard_roots_captured,
            "discard_roots_skipped": self.discard_roots_skipped,
            "discard_events_dropped": self._discard_events_dropped,
        }

    def discard_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._discard_events)


class _ProfessionalDiscardResponseRiskPolicy(ProfessionalBrainSimulationPolicy):
    """Bounded immediate-response search around production-safe discards."""

    apply_risk_selection = False

    def __init__(
        self,
        *,
        root_config: RootISMCTSConfig | None = None,
        activation_ev_gap: float = 120.0,
    ) -> None:
        self.risk_search = RootISMCTSPolicy(
            root_config
            or RootISMCTSConfig(
                time_budget_ms=250,
                max_iterations=8,
                max_candidates=2,
                rollout_max_turns=1,
            )
        )
        self.activation_ev_gap = activation_ev_gap
        self.risk_opportunities = 0
        self.risk_attempts = 0
        self.risk_usable = 0
        self.risk_disagreements = 0
        self.risk_simulations = 0
        self.risk_paired_determinizations = 0
        self.risk_deadline_interruptions = 0
        self.risk_invariant_violations = 0
        self.risk_coverage_failures = 0
        self.risk_elapsed_ms = 0.0
        self._discard_events: list[dict[str, Any]] = []
        self._discard_events_dropped = 0

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        decision = self._choose(view, rules, legal_actions=[{"type": "DISCARD"}])
        production_label = decision.selected_label
        if decision.selected_action != "DISCARD" or not production_label:
            raise ValueError(
                "professional_discard_response_risk_no_discard:"
                f"{decision.selected_action}:reason={decision.reason}"
            )
        self.risk_opportunities += 1
        by_label: dict[str, Any] = {}
        for item in decision.action_evals:
            label = item.action.label
            if item.type != "DISCARD" or not item.allowed or not label:
                continue
            previous = by_label.get(label)
            if previous is None or item.ev > previous.ev:
                by_label[label] = item
        candidates = sorted(
            by_label.values(),
            key=lambda item: (float(item.ev), str(item.action.label)),
            reverse=True,
        )[:2]
        event: dict[str, Any] = {
            "production_label": production_label,
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "label": str(item.action.label),
                    "heuristic_value": round(float(item.ev), 4),
                }
                for item in candidates
            ],
            "safety_flags": list(decision.safety_flags),
            "selection_applied": self.apply_risk_selection,
        }
        if len(candidates) < 2:
            event.update({"eligible": False, "reason": "fewer_than_two_safe_candidates"})
            self._record_discard_event(event)
            return production_label
        gap = float(candidates[0].ev) - float(candidates[1].ev)
        event["heuristic_gap"] = round(gap, 4)
        if decision.safety_flags:
            event.update({"eligible": False, "reason": "production_safety_flag"})
            self._record_discard_event(event)
            return production_label
        if gap > self.activation_ev_gap:
            event.update({"eligible": False, "reason": "heuristic_gap_too_large"})
            self._record_discard_event(event)
            return production_label

        self.risk_attempts += 1
        search = self.risk_search.search_discard_response_risk(
            view,
            rules=rules,
            candidate_labels=[str(item.action.label) for item in candidates],
            candidate_priors={
                str(item.action.label): float(item.ev)
                for item in candidates
            },
            preferred_label=production_label,
        )
        disagreement = search.used_search and search.selected_label != production_label
        self.risk_usable += int(search.used_search)
        self.risk_disagreements += int(disagreement)
        self.risk_simulations += search.simulations
        self.risk_paired_determinizations += search.paired_determinizations
        self.risk_deadline_interruptions += search.deadline_interruptions
        self.risk_invariant_violations += search.rollout_invariant_violations
        self.risk_coverage_failures += search.rollout_coverage_failures
        self.risk_elapsed_ms += search.elapsed_ms
        event.update(
            {
                "eligible": search.used_search,
                "reason": search.reason,
                "shadow_label": search.selected_label,
                "disagreement": disagreement,
                "risk_search": search.to_dict(),
                "public_view": public_view_to_dict(view),
            }
        )
        self._record_discard_event(event)
        if self.apply_risk_selection and search.used_search:
            return search.selected_label
        return production_label

    def _record_discard_event(self, event: dict[str, Any]) -> None:
        if len(self._discard_events) < 512:
            self._discard_events.append(event)
        else:
            self._discard_events_dropped += 1

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "discard_risk_opportunities": self.risk_opportunities,
            "discard_risk_attempts": self.risk_attempts,
            "discard_risk_usable": self.risk_usable,
            "discard_risk_disagreements": self.risk_disagreements,
            "discard_risk_simulations": self.risk_simulations,
            "discard_risk_paired_determinizations": self.risk_paired_determinizations,
            "discard_risk_deadline_interruptions": self.risk_deadline_interruptions,
            "discard_risk_invariant_violations": self.risk_invariant_violations,
            "discard_risk_coverage_failures": self.risk_coverage_failures,
            "discard_risk_elapsed_ms": round(self.risk_elapsed_ms, 3),
            "discard_events_dropped": self._discard_events_dropped,
        }

    def discard_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._discard_events)


class ProfessionalDiscardResponseRiskShadowPolicy(_ProfessionalDiscardResponseRiskPolicy):
    """Production discard with immediate-response risk recorded only."""

    name = "professional_discard_risk_shadow_v1"


class ProfessionalDiscardResponseRiskCandidatePolicy(_ProfessionalDiscardResponseRiskPolicy):
    """Offline candidate that applies the immediate-response risk ranking."""

    name = "professional_discard_risk_candidate_v1"
    apply_risk_selection = True


def determinize_public_view(
    view: PublicView,
    *,
    rules: dict[str, Any],
    rng: random.Random,
) -> tuple[list[SimPlayer], list[str]] | None:
    """Sample opponent hands and stock while preserving every visible card."""

    player_count = int(rules.get("game", {}).get("players", 3))
    if (
        player_count not in {2, 3}
        or len(view.all_melds) != player_count
        or len(view.discards) != player_count
        or not 0 <= view.seat < player_count
    ):
        return None
    if view.passed_chi and len(view.passed_chi) != player_count:
        return None
    if view.passed_peng and len(view.passed_peng) != player_count:
        return None
    unknown = [
        label
        for label, amount in view.remaining_counts
        for _ in range(max(0, int(amount)))
    ]
    opponent_seats = [seat for seat in range(player_count) if seat != view.seat]
    sizes = _opponent_hand_sizes(
        view,
        unknown_total=len(unknown),
        player_count=player_count,
    )
    if sizes is None:
        return None
    expected = view.stock_count + sum(sizes[seat] for seat in opponent_seats)
    if expected != len(unknown):
        return None
    hidden_assignment = _sample_legal_hidden_assignment(
        unknown,
        opponent_seats=opponent_seats,
        sizes=sizes,
        rng=rng,
    )
    if hidden_assignment is None:
        return None
    opponent_hands, stock = hidden_assignment
    players: list[SimPlayer] = []
    for seat in range(player_count):
        if seat == view.seat:
            hand = list(view.hand)
        else:
            hand = opponent_hands[seat]
        melds = list(view.all_melds[seat])
        players.append(
            SimPlayer(
                seat=seat,
                hand=list(hand),
                melds=melds,
                discards=list(view.discards[seat]),
                passed_chi=set(
                    view.passed_chi[seat] if view.passed_chi else ()
                ),
                passed_peng=set(
                    view.passed_peng[seat] if view.passed_peng else ()
                ),
                quad_events=sum(meld.kind in {"ti", "pao"} for meld in melds),
            )
        )
    expected_counts = full_deck_counts(rules=rules)
    observed = Counter(stock)
    if view.pending_card:
        observed.update((view.pending_card,))
    for player in players:
        observed.update(player.hand)
        observed.update(player.discards)
        observed.update(card for meld in player.melds for card in meld.cards)
    if observed != expected_counts:
        return None
    return players, stock


def _sample_legal_hidden_assignment(
    unknown: list[str],
    *,
    opponent_seats: list[int],
    sizes: dict[int, int],
    rng: random.Random,
    max_attempts: int = 128,
) -> tuple[dict[int, list[str]], list[str]] | None:
    cards = list(unknown)
    for _ in range(max(1, max_attempts)):
        rng.shuffle(cards)
        hands: dict[int, list[str]] = {}
        offset = 0
        for seat in opponent_seats:
            size = sizes[seat]
            hands[seat] = cards[offset : offset + size]
            offset += size
        if any(_has_unresolved_auto_ti(hand) for hand in hands.values()):
            continue
        return hands, cards[offset:]
    return None


def _has_unresolved_auto_ti(hand: list[str]) -> bool:
    return any(
        label != WILD_LABEL and amount >= 4
        for label, amount in Counter(hand).items()
    )


def _opponent_hand_sizes(
    view: PublicView,
    *,
    unknown_total: int,
    player_count: int,
) -> dict[int, int] | None:
    if len(view.hand_sizes) == player_count:
        sizes = {seat: int(view.hand_sizes[seat]) for seat in range(player_count)}
        if sizes.get(view.seat) != len(view.hand) or any(size < 0 for size in sizes.values()):
            return None
        return sizes
    hidden_total = unknown_total - int(view.stock_count)
    if hidden_total < 0:
        return None
    opponents = [seat for seat in range(player_count) if seat != view.seat]
    if not opponents:
        return None
    base, remainder = divmod(hidden_total, len(opponents))
    sizes = {view.seat: len(view.hand)}
    sizes.update(
        {
            seat: base + int(index < remainder)
            for index, seat in enumerate(opponents)
        }
    )
    return sizes


def _select_root_arm(
    labels: list[str],
    visits: Counter[str],
    rewards: Counter[str],
    heuristic: dict[str, float],
    *,
    exploration: float,
) -> str:
    unvisited = [label for label in labels if visits[label] == 0]
    if unvisited:
        return max(unvisited, key=lambda label: (heuristic[label], label))
    total = sum(visits.values())
    return max(
        labels,
        key=lambda label: (
            rewards[label] / visits[label]
            + exploration * math.sqrt(math.log(max(2, total)) / visits[label]),
            heuristic[label],
            label,
        ),
    )


def _candidate_rank_key(item: Any) -> tuple[float, ...]:
    """Rank aggregate evidence without collapsing WIN_FIRST tie-breakers."""

    return (
        float(item.average_reward),
        -float(getattr(item, "loss_rate", 0.0)),
        float(getattr(item, "mean_outcome_score", 0.0)),
        float(getattr(item, "mean_signed_xi", 0.0)),
    )


def _root_heuristic_value(
    view: PublicView,
    label: str,
    rules: dict[str, Any],
) -> float:
    after = _remove_one(view.hand, label)
    existing_xi = sum(
        meld_xi(list(meld.cards), kind=meld.kind, rules=rules)
        for meld in view.own_melds
    )
    return (
        quick_potential(after) * 12.0
        + existing_xi * 45.0
        + sum(card in RED_LABELS for card in after) * 3.0
        - _public_discard_danger(label, view)
    )


def _paired_world_outcome(
    *,
    world_index: int,
    players: Sequence[SimPlayer],
    stock: Sequence[str],
    rollout_seed: int,
    opponent_policy: str,
    root_continuation_policy: str,
    batch_results: Sequence[tuple[str, GameResult]],
    root_seat: int,
    objective_mode: ObjectiveMode = ObjectiveMode.WIN_FIRST,
) -> PairedWorldOutcome:
    return PairedWorldOutcome(
        world_index=world_index,
        world_fingerprint=_determinization_fingerprint(players, stock),
        rollout_seed=rollout_seed,
        opponent_policy=opponent_policy,
        root_continuation_policy=root_continuation_policy,
        outcomes=tuple(
            PairedCandidateOutcome(
                candidate_key=key,
                reward=_root_reward(
                    result,
                    root_seat=root_seat,
                    objective_mode=objective_mode,
                ),
                winner=result.winner,
                score=float(result.score),
                total_xi=int(result.total_xi),
                turns=int(result.turns),
                reason=str(result.reason),
            )
            for key, result in batch_results
        ),
    )


def _determinization_fingerprint(
    players: Sequence[SimPlayer],
    stock: Sequence[str],
) -> str:
    payload = {
        "players": [
            {
                "seat": player.seat,
                "hand": list(player.hand),
                "melds": [meld.to_dict() for meld in player.melds],
                "discards": list(player.discards),
                "passed_chi": sorted(player.passed_chi),
                "passed_peng": sorted(player.passed_peng),
                "quad_events": player.quad_events,
            }
            for player in players
        ],
        "stock": list(stock),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _root_reward(
    result: Any,
    *,
    root_seat: int,
    objective_mode: ObjectiveMode = ObjectiveMode.WIN_FIRST,
) -> float:
    if result.coverage_failures:
        raise ValueError("rollout_coverage_incomplete")
    if result.violations:
        raise ValueError("rollout_invariant_violation")
    return scalar_sampling_reward(
        result,
        root_seat=root_seat,
        mode=objective_mode,
    )


def _root_outcome_score(result: Any, *, root_seat: int) -> float:
    if result.winner is None:
        return 0.0
    score = float(result.score)
    return score if result.winner == root_seat else -score


def _root_signed_xi(result: Any, *, root_seat: int) -> float:
    if result.winner is None:
        return 0.0
    total_xi = float(result.total_xi)
    return total_xi if result.winner == root_seat else -total_xi


def _bounded_search_deadline(
    started: float,
    *,
    time_budget_ms: int,
    absolute_deadline: float | None,
) -> float:
    local_deadline = started + max(1, time_budget_ms) / 1000.0
    if absolute_deadline is None:
        return local_deadline
    return min(local_deadline, absolute_deadline)


def _override_iteration_budget_incomplete(
    config: RootISMCTSConfig,
    *,
    candidate_count: int,
    paired_determinizations: int,
    deadline_interruptions: int,
) -> bool:
    if not config.require_complete_iteration_budget_for_override:
        return False
    target_pairs = max(1, config.max_iterations) // max(1, candidate_count)
    return (
        paired_determinizations < target_pairs
        or deadline_interruptions > 0
    )


def _paired_advantage_stats(
    reward_batches: Sequence[Mapping[str, float]],
    *,
    candidate_keys: Sequence[str],
    preferred_key: str,
    outcome_batches: Sequence[Mapping[str, Any]] | None = None,
    objective_mode: ObjectiveMode = ObjectiveMode.WIN_FIRST,
    familywise_comparisons: int | None = None,
) -> tuple[PairedAdvantageStats, ...]:
    if preferred_key not in candidate_keys:
        raise ValueError("preferred_candidate_missing_from_paired_rewards")
    rows: list[PairedAdvantageStats] = []
    resolved_comparisons = max(
        1,
        len(candidate_keys) - 1,
        int(familywise_comparisons or 1),
    )
    for candidate_key in candidate_keys:
        if candidate_key == preferred_key:
            continue
        deltas = [
            float(batch[candidate_key]) - float(batch[preferred_key])
            for batch in reward_batches
            if candidate_key in batch and preferred_key in batch
        ]
        samples = len(deltas)
        mean_delta = sum(deltas) / samples if samples else 0.0
        if samples >= 2:
            squared_error = sum(
                (delta - mean_delta) ** 2
                for delta in deltas
            )
            sample_stddev = math.sqrt(squared_error / (samples - 1))
            standard_error = sample_stddev / math.sqrt(samples)
            radius = _familywise_95_t_critical(
                samples - 1,
                comparisons=resolved_comparisons,
            ) * standard_error
            lower_bound: float | None = mean_delta - radius
            upper_bound: float | None = mean_delta + radius
        else:
            sample_stddev = 0.0
            standard_error = 0.0
            lower_bound = None
            upper_bound = None
        rows.append(
            PairedAdvantageStats(
                candidate_key=candidate_key,
                preferred_key=preferred_key,
                samples=samples,
                mean_delta=mean_delta,
                sample_stddev=sample_stddev,
                standard_error=standard_error,
                lower_confidence_bound=lower_bound,
                upper_confidence_bound=upper_bound,
                positive_samples=sum(delta > 0.0 for delta in deltas),
                tied_samples=sum(abs(delta) <= 1e-12 for delta in deltas),
                negative_samples=sum(delta < 0.0 for delta in deltas),
                familywise_comparisons=resolved_comparisons,
                objective_metric=(
                    "delta_p_win"
                    if objective_mode is ObjectiveMode.WIN_FIRST
                    else "legacy_scalar_reward"
                ),
                mean_delta_p_win=_mean_outcome_delta(
                    outcome_batches,
                    candidate_key=candidate_key,
                    preferred_key=preferred_key,
                    attribute="p_win",
                ),
                mean_delta_p_loss=_mean_outcome_delta(
                    outcome_batches,
                    candidate_key=candidate_key,
                    preferred_key=preferred_key,
                    attribute="p_loss",
                ),
                mean_delta_score=_mean_outcome_delta(
                    outcome_batches,
                    candidate_key=candidate_key,
                    preferred_key=preferred_key,
                    attribute="expected_score",
                ),
                mean_delta_xi=_mean_outcome_delta(
                    outcome_batches,
                    candidate_key=candidate_key,
                    preferred_key=preferred_key,
                    attribute="expected_xi",
                ),
            )
        )
    return tuple(rows)


def _paired_outcome_batch(
    batch_results: Sequence[tuple[str, GameResult]],
    *,
    root_seat: int,
) -> dict[str, Any]:
    return {
        key: outcome_vector(result, root_seat=root_seat)
        for key, result in batch_results
    }


def _mean_outcome_delta(
    outcome_batches: Sequence[Mapping[str, Any]] | None,
    *,
    candidate_key: str,
    preferred_key: str,
    attribute: str,
) -> float | None:
    if outcome_batches is None:
        return None
    deltas = [
        float(getattr(batch[candidate_key], attribute))
        - float(getattr(batch[preferred_key], attribute))
        for batch in outcome_batches
        if candidate_key in batch and preferred_key in batch
    ]
    return sum(deltas) / len(deltas) if deltas else None


def _two_sided_95_t_critical(degrees_of_freedom: int) -> float:
    """Conservative Student-t critical value without a scipy dependency."""

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
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom_must_be_positive")
    if degrees_of_freedom <= len(table):
        return table[degrees_of_freedom - 1]
    if degrees_of_freedom <= 60:
        return 2.042
    if degrees_of_freedom <= 120:
        return 2.0
    if degrees_of_freedom <= 500:
        return 1.98
    return 1.96


def _familywise_95_t_critical(
    degrees_of_freedom: int,
    *,
    comparisons: int,
) -> float:
    if comparisons <= 1:
        return _two_sided_95_t_critical(degrees_of_freedom)
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom_must_be_positive")
    alpha_per_comparison = 0.05 / comparisons
    quantile = 1.0 - alpha_per_comparison / 2.0
    z = NormalDist().inv_cdf(quantile)
    df = float(degrees_of_freedom)
    first = (z**3 + z) / (4.0 * df)
    second = (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
    third = (
        3.0 * z**7
        + 19.0 * z**5
        + 17.0 * z**3
        - 15.0 * z
    ) / (384.0 * df**3)
    approximation = z + first + second + third
    if degrees_of_freedom < 5:
        return max(approximation, 12.706)
    return approximation


def _discard_response_risk_stats(
    labels: Sequence[str],
    *,
    samples: Counter[str],
    passes: Counter[str],
    claims: Mapping[str, Counter[str]],
    claim_seats: Mapping[str, Counter[tuple[str, int]]],
    risks: Counter[str],
    priors: Mapping[str, float],
    view: PublicView,
) -> tuple[DiscardResponseRiskCandidateStats, ...]:
    action_order = ("hu", "pao", "peng", "chi")
    stats: list[DiscardResponseRiskCandidateStats] = []
    for label in labels:
        count = samples[label]
        expected_risk = risks[label] / count if count else 0.0
        stats.append(
            DiscardResponseRiskCandidateStats(
                label=label,
                samples=count,
                pass_count=passes[label],
                claim_counts=tuple(
                    (action, int(claims[label][action]))
                    for action in action_order
                ),
                claim_seat_counts=tuple(
                    (action, seat, amount)
                    for (action, seat), amount in sorted(
                        claim_seats[label].items(),
                        key=lambda item: (
                            (item[0][1] - view.seat) % len(view.discards),
                            action_order.index(item[0][0]),
                        ),
                    )
                ),
                risk_sum=risks[label],
                expected_risk=expected_risk,
                heuristic_value=float(priors[label]),
                adjusted_value=float(priors[label]) - expected_risk,
                root_seat=view.seat,
                player_count=len(view.discards),
            )
        )
    return tuple(stats)


def _immediate_response_from_result(
    result: GameResult,
    *,
    discarder: int,
    player_count: int,
) -> tuple[str, int | None]:
    action_counts = result.action_counts
    observed: list[tuple[str, int, int]] = []
    for counter_name, action in (
        ("discard_hu", "hu"),
        ("pao", "pao"),
        ("peng", "peng"),
        ("chi", "chi"),
    ):
        amount = int(action_counts.get(counter_name, 0))
        if amount <= 0:
            continue
        seat_rows = [
            (key, int(value))
            for key, value in action_counts.items()
            if key.startswith(f"{counter_name}:seat_") and int(value) > 0
        ]
        if len(seat_rows) != 1:
            raise ValueError(f"immediate_response_claimant_missing:{counter_name}")
        key, seat_amount = seat_rows[0]
        seat = int(key.rsplit("_", 1)[1])
        if amount != 1 or seat_amount != 1:
            raise ValueError(f"immediate_response_count_invalid:{counter_name}")
        observed.append((action, seat, amount))
    if len(observed) > 1:
        raise ValueError("multiple_immediate_responses")
    if not observed:
        if int(action_counts.get("discard", 0)) != 1:
            raise ValueError("root_discard_not_executed")
        return "pass", None
    action, claimant, _amount = observed[0]
    if not 0 <= claimant < player_count or claimant == discarder:
        raise ValueError(f"invalid_immediate_response_claimant:{claimant}")
    if action == "chi" and claimant != (discarder + 1) % player_count:
        raise ValueError(
            f"illegal_non_next_seat_chi:discarder={discarder}:claimant={claimant}"
        )
    return action, claimant


def _immediate_response_risk(action: str, *, rules: Mapping[str, Any]) -> float:
    weights = rules.get("weights") or rules.get("ai_weights") or {}
    base = abs(float(weights.get("danger_base_penalty", 160)))
    if action == "hu":
        return abs(float(weights.get("opponent_hu_risk_penalty", 1000)))
    if action == "pao":
        return abs(float(weights.get("opponent_pao_risk_penalty", 500)))
    if action == "peng":
        return base
    if action == "chi":
        return base * 0.75
    if action == "pass":
        return 0.0
    raise ValueError(f"unsupported_immediate_response:{action}")


def information_set_seed(view: PublicView, base_seed: int) -> int:
    payload = repr(
        (
            view.seat,
            view.hand,
            view.own_melds,
            view.all_melds,
            view.discards,
            view.remaining_counts,
            view.stock_count,
            view.hand_sizes,
            view.pending_card,
            view.pending_source_seat,
            view.passed_chi,
            view.passed_peng,
            base_seed,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _no_search_result(
    selected: str,
    reason: str,
    heuristic: dict[str, float],
    started: float,
) -> RootSearchResult:
    return RootSearchResult(
        selected_label=selected,
        used_search=False,
        reason=reason,
        simulations=0,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        candidates=tuple(
            RootCandidateStats(
                label=label,
                visits=0,
                reward_sum=0.0,
                average_reward=0.0,
                win_rate=0.0,
                heuristic_value=value,
            )
            for label, value in sorted(
                heuristic.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
        ),
    )


def _no_response_search_result(
    selected: RootResponseCandidate,
    reason: str,
    started: float,
    *,
    candidates: tuple[RootResponseCandidate, ...] | None = None,
) -> RootResponseSearchResult:
    rows = candidates or (selected,)
    return RootResponseSearchResult(
        selected_key=selected.key,
        used_search=False,
        reason=reason,
        simulations=0,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        candidates=tuple(
            RootResponseCandidateStats(
                candidate=candidate,
                visits=0,
                reward_sum=0.0,
                average_reward=0.0,
                win_rate=0.0,
            )
            for candidate in rows
        ),
    )


__all__ = [
    "DiscardResponseRiskCandidateStats",
    "DiscardResponseRiskResult",
    "PairedCandidateOutcome",
    "PairedWorldOutcome",
    "RootCandidateStats",
    "RootISMCTSConfig",
    "RootISMCTSPolicy",
    "ProgressiveRootISMCTSPolicy",
    "ProfessionalProgressiveAllActionCandidatePolicy",
    "ProfessionalResponseShadowSimulationPolicy",
    "ProfessionalDiscardResponseRiskCandidatePolicy",
    "ProfessionalDiscardResponseRiskShadowPolicy",
    "ProfessionalDiscardShadowSimulationPolicy",
    "ProfessionalFullActionTeacherPolicy",
    "ProfessionalSearchSimulationPolicy",
    "RootResponseCandidate",
    "RootResponseCandidateStats",
    "RootResponseSearchResult",
    "RootSearchResult",
    "build_response_candidates",
    "determinize_public_view",
    "information_set_seed",
    "public_view_from_dict",
    "public_view_to_dict",
    "shortlist_response_candidates",
]
