"""Decision-level counterfactual audits over paired hidden-state worlds."""

from __future__ import annotations

import random
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from ai.full_game_simulator import (
    BaselinePolicy,
    FullGameSimulator,
    PublicView,
    SimulationPolicy,
)
from ai.ismcts import (
    RootISMCTSConfig,
    RootISMCTSPolicy,
    RootResponseCandidate,
    RootSelfContinuationPolicy,
    _paired_advantage_stats,
    _paired_world_outcome,
    _root_outcome_score,
    _root_reward,
    _root_signed_xi,
    determinize_public_view,
    information_set_seed,
    public_view_from_dict,
)


DEFAULT_AUDIT_ROLLOUT_FACTORIES: tuple[Callable[[], SimulationPolicy], ...] = (
    IndependentFastRolloutPolicy,
    IndependentFastPressurePolicy,
    IndependentFastDenialPolicy,
)


@dataclass(frozen=True)
class CounterfactualAuditConfig:
    paired_worlds: int = 8
    min_confidence_pairs: int = 8
    time_budget_ms: int = 60_000
    rollout_max_turns: int = 120
    seed: int = 20260728


def _information_set_seed(
    base_seed: int,
    trace: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> int:
    """Resolve the exact RNG seed used by runtime search for this public state."""
    del rules
    view = public_view_from_dict(trace.get("public_state") or {})
    return information_set_seed(view, base_seed)


def _confidently_better_advantages(
    advantages: Sequence[Any],
    *,
    min_confidence_pairs: int,
) -> tuple[Any, ...]:
    minimum = max(2, int(min_confidence_pairs))
    return tuple(
        advantage
        for advantage in advantages
        if advantage.samples >= minimum
        and advantage.confidently_positive
    )


def audit_discard_trace(
    trace: Mapping[str, Any],
    *,
    rules: dict[str, Any],
    config: CounterfactualAuditConfig | None = None,
    rollout_policy_factories: Sequence[
        Callable[[], SimulationPolicy]
    ] = DEFAULT_AUDIT_ROLLOUT_FACTORIES,
    candidate_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    resolved = config or CounterfactualAuditConfig()
    if str(trace.get("phase") or "") != "discard":
        raise ValueError("counterfactual_discard_trace_required")
    view = public_view_from_dict(trace.get("public_state") or {})
    selected_key = str(trace.get("selected_key") or "")
    if not selected_key.startswith("DISCARD:"):
        raise ValueError("counterfactual_selected_discard_required")
    selected_label = selected_key.split(":", 1)[1]
    available_legal_labels = _discard_labels(trace)
    if selected_label not in available_legal_labels:
        raise ValueError("counterfactual_selected_discard_not_legal")
    legal_labels = available_legal_labels
    if candidate_labels is not None:
        requested = tuple(dict.fromkeys(str(label) for label in candidate_labels))
        if (
            len(requested) < 2
            or selected_label not in requested
            or any(label not in available_legal_labels for label in requested)
        ):
            raise ValueError(
                "counterfactual_discard_candidate_filter_invalid"
            )
        requested_set = set(requested)
        legal_labels = [
            label
            for label in available_legal_labels
            if label in requested_set
        ]
    paired_worlds = max(1, int(resolved.paired_worlds))
    audit_seed = _information_set_seed(resolved.seed, trace, rules)
    search = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=max(1, int(resolved.time_budget_ms)),
            max_iterations=paired_worlds * len(legal_labels),
            max_candidates=len(legal_labels),
            rollout_max_turns=max(1, int(resolved.rollout_max_turns)),
            seed=resolved.seed,
            record_paired_worlds=True,
            min_confidence_pairs=max(2, int(resolved.min_confidence_pairs)),
        ),
        rollout_policy_factories=tuple(rollout_policy_factories),
    )
    result = search.search_discard(
        view,
        rules=rules,
        candidate_labels=legal_labels,
        force_search=True,
        paired_candidates=True,
        preferred_label=selected_label,
    )
    stats_by_label = {
        candidate.label: candidate
        for candidate in result.candidates
    }
    selected_stats = stats_by_label.get(selected_label)
    empirical_best = (
        max(
            result.candidates,
            key=lambda item: (
                item.average_reward,
                item.label == selected_label,
                item.visits,
                item.heuristic_value,
                item.label,
            ),
        )
        if result.candidates
        else None
    )
    selected_ev = (
        float(selected_stats.average_reward)
        if selected_stats is not None and selected_stats.visits
        else None
    )
    best_ev = (
        float(empirical_best.average_reward)
        if empirical_best is not None and empirical_best.visits
        else None
    )
    regret = (
        max(0.0, best_ev - selected_ev)
        if selected_ev is not None and best_ev is not None
        else None
    )
    best_advantage = next(
        (
            advantage
            for advantage in result.paired_advantages
            if empirical_best is not None
            and advantage.candidate_key == empirical_best.label
        ),
        None,
    )
    confidently_better = _confidently_better_advantages(
        result.paired_advantages,
        min_confidence_pairs=resolved.min_confidence_pairs,
    )
    confidently_suboptimal = bool(confidently_better)
    complete = (
        result.used_search
        and result.paired_determinizations == paired_worlds
        and len(result.candidates) == len(legal_labels)
        and all(candidate.visits == paired_worlds for candidate in result.candidates)
        and not result.rollout_invariant_violations
        and not result.rollout_coverage_failures
        and len(result.paired_worlds) == paired_worlds
    )
    return {
        "schema_version": "counterfactual-decision-audit-v2",
        "trace_sequence": int(trace.get("sequence") or 0),
        "turn": int(trace.get("turn") or 0),
        "seat": int(trace.get("seat") or 0),
        "policy": str(trace.get("policy") or ""),
        "phase": "discard",
        "audit_seed": audit_seed,
        "state_before_hash": str(trace.get("state_before_hash") or ""),
        "public_state": dict(trace.get("public_state") or {}),
        "selected_key": selected_key,
        "available_legal_action_count": len(available_legal_labels),
        "legal_action_count": len(legal_labels),
        "audited_action_count": len(result.candidates),
        "requested_paired_worlds": paired_worlds,
        "completed_paired_worlds": result.paired_determinizations,
        "complete": complete,
        "reason": result.reason,
        "selected_expected_reward": selected_ev,
        "selected_win_rate": (
            selected_stats.win_rate
            if selected_stats is not None
            else None
        ),
        "selected_loss_rate": (
            selected_stats.loss_rate
            if selected_stats is not None
            else None
        ),
        "selected_draw_rate": (
            selected_stats.draw_rate
            if selected_stats is not None
            else None
        ),
        "selected_expected_score": (
            selected_stats.mean_outcome_score
            if selected_stats is not None
            else None
        ),
        "selected_expected_signed_xi": (
            selected_stats.mean_signed_xi
            if selected_stats is not None
            else None
        ),
        "best_key": (
            f"DISCARD:{empirical_best.label}"
            if empirical_best is not None
            else None
        ),
        "best_expected_reward": best_ev,
        "best_win_rate": (
            empirical_best.win_rate
            if empirical_best is not None
            else None
        ),
        "best_loss_rate": (
            empirical_best.loss_rate
            if empirical_best is not None
            else None
        ),
        "best_draw_rate": (
            empirical_best.draw_rate
            if empirical_best is not None
            else None
        ),
        "best_expected_score": (
            empirical_best.mean_outcome_score
            if empirical_best is not None
            else None
        ),
        "best_expected_signed_xi": (
            empirical_best.mean_signed_xi
            if empirical_best is not None
            else None
        ),
        "counterfactual_regret": regret,
        "selected_is_empirical_best": (
            empirical_best is not None
            and empirical_best.label == selected_label
        ),
        "confidently_suboptimal": confidently_suboptimal,
        "confidently_better_candidates": [
            advantage.to_dict()
            for advantage in confidently_better
        ],
        "best_vs_selected_confidence": (
            best_advantage.to_dict()
            if best_advantage is not None
            else None
        ),
        "candidate_stats": [
            {
                **candidate.to_dict(),
                "key": f"DISCARD:{candidate.label}",
            }
            for candidate in result.candidates
        ],
        "paired_advantages": [
            advantage.to_dict()
            for advantage in result.paired_advantages
        ],
        "paired_worlds": [
            world.to_dict()
            for world in result.paired_worlds
        ],
        "search_health": {
            "simulations": result.simulations,
            "elapsed_ms": round(result.elapsed_ms, 3),
            "determinization_failures": result.determinization_failures,
            "deadline_interruptions": result.deadline_interruptions,
            "rollout_invariant_violations": result.rollout_invariant_violations,
            "rollout_violations": list(result.rollout_violations),
            "rollout_coverage_failures": result.rollout_coverage_failures,
            "rollout_coverage_reasons": list(result.rollout_coverage_reasons),
        },
    }


def audit_response_trace(
    trace: Mapping[str, Any],
    *,
    rules: dict[str, Any],
    config: CounterfactualAuditConfig | None = None,
    rollout_policy_factories: Sequence[
        Callable[[], SimulationPolicy]
    ] = DEFAULT_AUDIT_ROLLOUT_FACTORIES,
) -> dict[str, Any]:
    resolved = config or CounterfactualAuditConfig()
    if str(trace.get("phase") or "") != "response_root":
        raise ValueError("counterfactual_response_trace_required")
    view = public_view_from_dict(trace.get("public_state") or {})
    candidates = _response_candidates(trace)
    selected_key = str(trace.get("selected_key") or "")
    legal_keys = [candidate.key for candidate in candidates]
    if selected_key not in legal_keys:
        raise ValueError("counterfactual_selected_response_not_legal")
    paired_worlds = max(1, int(resolved.paired_worlds))
    audit_seed = _information_set_seed(resolved.seed, trace, rules)
    search = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=max(1, int(resolved.time_budget_ms)),
            max_iterations=paired_worlds * len(candidates),
            max_candidates=len(candidates),
            rollout_max_turns=max(1, int(resolved.rollout_max_turns)),
            seed=resolved.seed,
            record_paired_worlds=True,
            min_confidence_pairs=max(2, int(resolved.min_confidence_pairs)),
        ),
        rollout_policy_factories=tuple(rollout_policy_factories),
    )
    result = search.search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
        preferred_key=selected_key,
    )
    stats_by_key = {
        candidate.candidate.key: candidate
        for candidate in result.candidates
    }
    selected_stats = stats_by_key.get(selected_key)
    empirical_best = (
        max(
            result.candidates,
            key=lambda item: (
                item.average_reward,
                item.candidate.key == selected_key,
                item.visits,
                item.candidate.heuristic_value,
                item.candidate.key,
            ),
        )
        if result.candidates
        else None
    )
    selected_ev = (
        float(selected_stats.average_reward)
        if selected_stats is not None and selected_stats.visits
        else None
    )
    best_ev = (
        float(empirical_best.average_reward)
        if empirical_best is not None and empirical_best.visits
        else None
    )
    regret = (
        max(0.0, best_ev - selected_ev)
        if selected_ev is not None and best_ev is not None
        else None
    )
    best_key = (
        empirical_best.candidate.key
        if empirical_best is not None
        else None
    )
    best_advantage = next(
        (
            advantage
            for advantage in result.paired_advantages
            if advantage.candidate_key == best_key
        ),
        None,
    )
    confidently_better = _confidently_better_advantages(
        result.paired_advantages,
        min_confidence_pairs=resolved.min_confidence_pairs,
    )
    confidently_suboptimal = bool(confidently_better)
    complete = (
        result.used_search
        and result.paired_determinizations == paired_worlds
        and len(result.candidates) == len(candidates)
        and all(candidate.visits == paired_worlds for candidate in result.candidates)
        and not result.rollout_invariant_violations
        and not result.rollout_coverage_failures
        and len(result.paired_worlds) == paired_worlds
        and all(
            len(world.outcomes) == len(candidates)
            for world in result.paired_worlds
        )
    )
    return {
        "schema_version": "counterfactual-decision-audit-v2",
        "trace_sequence": int(trace.get("sequence") or 0),
        "turn": int(trace.get("turn") or 0),
        "seat": int(trace.get("seat") or 0),
        "policy": str(trace.get("policy") or ""),
        "phase": "response_root",
        "audit_seed": audit_seed,
        "state_before_hash": str(trace.get("state_before_hash") or ""),
        "public_state": dict(trace.get("public_state") or {}),
        "selected_key": selected_key,
        "legal_action_count": len(candidates),
        "audited_action_count": len(result.candidates),
        "requested_paired_worlds": paired_worlds,
        "completed_paired_worlds": result.paired_determinizations,
        "complete": complete,
        "reason": result.reason,
        "selected_expected_reward": selected_ev,
        "selected_win_rate": (
            selected_stats.win_rate
            if selected_stats is not None
            else None
        ),
        "selected_loss_rate": (
            selected_stats.loss_rate
            if selected_stats is not None
            else None
        ),
        "selected_draw_rate": (
            selected_stats.draw_rate
            if selected_stats is not None
            else None
        ),
        "selected_expected_score": (
            selected_stats.mean_outcome_score
            if selected_stats is not None
            else None
        ),
        "selected_expected_signed_xi": (
            selected_stats.mean_signed_xi
            if selected_stats is not None
            else None
        ),
        "best_key": best_key,
        "best_expected_reward": best_ev,
        "best_win_rate": (
            empirical_best.win_rate
            if empirical_best is not None
            else None
        ),
        "best_loss_rate": (
            empirical_best.loss_rate
            if empirical_best is not None
            else None
        ),
        "best_draw_rate": (
            empirical_best.draw_rate
            if empirical_best is not None
            else None
        ),
        "best_expected_score": (
            empirical_best.mean_outcome_score
            if empirical_best is not None
            else None
        ),
        "best_expected_signed_xi": (
            empirical_best.mean_signed_xi
            if empirical_best is not None
            else None
        ),
        "counterfactual_regret": regret,
        "selected_is_empirical_best": best_key == selected_key,
        "confidently_suboptimal": confidently_suboptimal,
        "confidently_better_candidates": [
            advantage.to_dict()
            for advantage in confidently_better
        ],
        "best_vs_selected_confidence": (
            best_advantage.to_dict()
            if best_advantage is not None
            else None
        ),
        "candidate_stats": [
            candidate.to_dict()
            for candidate in result.candidates
        ],
        "paired_advantages": [
            advantage.to_dict()
            for advantage in result.paired_advantages
        ],
        "paired_worlds": [
            world.to_dict()
            for world in result.paired_worlds
        ],
        "search_health": {
            "simulations": result.simulations,
            "elapsed_ms": round(result.elapsed_ms, 3),
            "determinization_failures": result.determinization_failures,
            "deadline_interruptions": result.deadline_interruptions,
            "rollout_invariant_violations": result.rollout_invariant_violations,
            "rollout_violations": list(result.rollout_violations),
            "rollout_coverage_failures": result.rollout_coverage_failures,
            "rollout_coverage_reasons": list(result.rollout_coverage_reasons),
        },
    }


def audit_hu_trace(
    trace: Mapping[str, Any],
    *,
    rules: dict[str, Any],
    config: CounterfactualAuditConfig | None = None,
    rollout_policy_factories: Sequence[
        Callable[[], SimulationPolicy]
    ] = DEFAULT_AUDIT_ROLLOUT_FACTORIES,
    root_continuation_policy_factory: Callable[
        [], SimulationPolicy
    ] = RootSelfContinuationPolicy,
) -> dict[str, Any]:
    resolved = config or CounterfactualAuditConfig()
    phase = str(trace.get("phase") or "")
    if phase not in {"self_hu", "post_auto_hu", "post_action_hu"}:
        raise ValueError("counterfactual_hu_trace_required")
    if bool(rules.get("rules", {}).get("zimo_double", False)):
        raise ValueError("counterfactual_hu_zimo_multiplier_not_supported")
    view = public_view_from_dict(trace.get("public_state") or {})
    selected_key = str(trace.get("selected_key") or "")
    legal_keys = _hu_keys(trace)
    if selected_key not in legal_keys:
        raise ValueError("counterfactual_selected_hu_action_not_legal")
    paired_worlds_requested = max(1, int(resolved.paired_worlds))
    audit_seed = _information_set_seed(resolved.seed, trace, rules)
    factories = tuple(rollout_policy_factories)
    if not factories:
        raise ValueError("counterfactual_hu_opponent_factory_required")
    deadline = (
        time.perf_counter()
        + max(1, int(resolved.time_budget_ms)) / 1000.0
    )
    rng = random.Random(audit_seed)
    visits = Counter({key: 0 for key in legal_keys})
    rewards = Counter({key: 0.0 for key in legal_keys})
    wins = Counter({key: 0 for key in legal_keys})
    losses = Counter({key: 0 for key in legal_keys})
    draws = Counter({key: 0 for key in legal_keys})
    outcome_scores = Counter({key: 0.0 for key in legal_keys})
    signed_xi = Counter({key: 0.0 for key in legal_keys})
    reward_batches: list[dict[str, float]] = []
    paired_world_evidence = []
    failures = 0
    deadline_interruptions = 0
    invariant_violations = 0
    rollout_violations: list[str] = []
    coverage_failures = 0
    coverage_reasons: list[str] = []
    started = time.perf_counter()
    player_count = int(rules.get("game", {}).get("players", 3))
    for world_index in range(paired_worlds_requested):
        if time.perf_counter() >= deadline:
            deadline_interruptions += 1
            break
        determinization = determinize_public_view(view, rules=rules, rng=rng)
        if determinization is None:
            failures += 1
            break
        base_players, base_stock = determinization
        rollout_seed = rng.randrange(1, 2**31)
        opponent_factory = factories[world_index % len(factories)]
        batch_results = []
        opponent_policy_name = "unknown"
        root_continuation_policy_name = "unknown"
        for key in legal_keys:
            if time.perf_counter() >= deadline:
                deadline_interruptions += 1
                break
            players = deepcopy(base_players)
            stock = list(base_stock)
            policies = [
                opponent_factory()
                for _ in range(player_count)
            ]
            root_continuation_policy = root_continuation_policy_factory()
            policies[view.seat] = _ForcedHuAuditPolicy(
                accept=key == "HU",
                continuation_policy=root_continuation_policy,
            )
            root_continuation_policy_name = str(
                getattr(
                    root_continuation_policy,
                    "name",
                    type(root_continuation_policy).__name__,
                )
            )
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
                players=players,
                stock=stock,
                current=view.seat,
                needs_draw=False,
                initial_drawn_card=(
                    str((trace.get("metadata") or {}).get("drawn_card"))
                    if phase == "self_hu"
                    and (trace.get("metadata") or {}).get("drawn_card")
                    is not None
                    else None
                ),
                max_turns=max(1, int(resolved.rollout_max_turns)),
                deadline=deadline,
            )
            invariant_violations += len(result.violations)
            rollout_violations.extend(result.violations)
            if result.violations:
                break
            coverage_failures += len(result.coverage_failures)
            coverage_reasons.extend(result.coverage_failures)
            if result.coverage_failures:
                break
            if result.reason == "time_budget":
                deadline_interruptions += 1
                break
            batch_results.append((key, result))
        if len(batch_results) != len(legal_keys):
            break
        batch_rewards = {}
        for key, result in batch_results:
            reward = _root_reward(result, root_seat=view.seat)
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
            batch_rewards[key] = reward
        reward_batches.append(batch_rewards)
        paired_world_evidence.append(
            _paired_world_outcome(
                world_index=world_index,
                players=base_players,
                stock=base_stock,
                rollout_seed=rollout_seed,
                opponent_policy=opponent_policy_name,
                root_continuation_policy=root_continuation_policy_name,
                batch_results=batch_results,
                root_seat=view.seat,
            )
        )

    average_by_key = {
        key: rewards[key] / max(1, visits[key])
        for key in legal_keys
    }
    win_rate_by_key = {
        key: wins[key] / max(1, visits[key])
        for key in legal_keys
    }
    candidate_stats = [
        {
            "key": key,
            "action_type": key,
            "visits": visits[key],
            "reward_sum": round(rewards[key], 4),
            "average_reward": round(average_by_key[key], 4),
            "win_rate": round(win_rate_by_key[key], 4),
            "wins": wins[key],
            "losses": losses[key],
            "draws": draws[key],
            "loss_rate": round(losses[key] / max(1, visits[key]), 4),
            "draw_rate": round(draws[key] / max(1, visits[key]), 4),
            "mean_outcome_score": round(
                outcome_scores[key] / max(1, visits[key]),
                4,
            ),
            "mean_signed_xi": round(
                signed_xi[key] / max(1, visits[key]),
                4,
            ),
        }
        for key in legal_keys
    ]
    best_key = max(
        legal_keys,
        key=lambda key: (
            average_by_key[key],
            key == selected_key,
            visits[key],
            key,
        ),
    )
    paired_advantages = _paired_advantage_stats(
        reward_batches,
        candidate_keys=legal_keys,
        preferred_key=selected_key,
    )
    best_advantage = next(
        (
            advantage
            for advantage in paired_advantages
            if advantage.candidate_key == best_key
        ),
        None,
    )
    complete = (
        len(reward_batches) == paired_worlds_requested
        and all(visits[key] == paired_worlds_requested for key in legal_keys)
        and not invariant_violations
        and not coverage_failures
        and len(paired_world_evidence) == paired_worlds_requested
    )
    selected_ev = average_by_key[selected_key] if visits[selected_key] else None
    best_ev = (
        average_by_key[best_key]
        if visits[best_key]
        else None
    )
    selected_visits = visits[selected_key]
    best_visits = visits[best_key]
    regret = (
        max(0.0, best_ev - selected_ev)
        if selected_ev is not None and best_ev is not None
        else None
    )
    confidently_better = _confidently_better_advantages(
        paired_advantages,
        min_confidence_pairs=resolved.min_confidence_pairs,
    )
    confidently_suboptimal = bool(confidently_better)
    reason = "root_hu_counterfactual_completed"
    if not complete:
        if invariant_violations:
            reason = "rollout_invariant_violation"
        elif coverage_failures:
            reason = "rollout_coverage_incomplete"
        elif deadline_interruptions:
            reason = "hu_deadline_before_complete_pair"
        elif failures:
            reason = "determinization_unavailable"
        else:
            reason = "candidate_coverage_incomplete"
    return {
        "schema_version": "counterfactual-decision-audit-v2",
        "trace_sequence": int(trace.get("sequence") or 0),
        "turn": int(trace.get("turn") or 0),
        "seat": int(trace.get("seat") or 0),
        "policy": str(trace.get("policy") or ""),
        "phase": phase,
        "audit_seed": audit_seed,
        "state_before_hash": str(trace.get("state_before_hash") or ""),
        "public_state": dict(trace.get("public_state") or {}),
        "selected_key": selected_key,
        "legal_action_count": len(legal_keys),
        "audited_action_count": len(candidate_stats),
        "requested_paired_worlds": paired_worlds_requested,
        "completed_paired_worlds": len(reward_batches),
        "complete": complete,
        "reason": reason,
        "selected_expected_reward": selected_ev,
        "selected_win_rate": (
            wins[selected_key] / selected_visits
            if selected_visits
            else None
        ),
        "selected_loss_rate": (
            losses[selected_key] / selected_visits
            if selected_visits
            else None
        ),
        "selected_draw_rate": (
            draws[selected_key] / selected_visits
            if selected_visits
            else None
        ),
        "selected_expected_score": (
            outcome_scores[selected_key] / selected_visits
            if selected_visits
            else None
        ),
        "selected_expected_signed_xi": (
            signed_xi[selected_key] / selected_visits
            if selected_visits
            else None
        ),
        "best_key": best_key,
        "best_expected_reward": best_ev,
        "best_win_rate": (
            wins[best_key] / best_visits
            if best_visits
            else None
        ),
        "best_loss_rate": (
            losses[best_key] / best_visits
            if best_visits
            else None
        ),
        "best_draw_rate": (
            draws[best_key] / best_visits
            if best_visits
            else None
        ),
        "best_expected_score": (
            outcome_scores[best_key] / best_visits
            if best_visits
            else None
        ),
        "best_expected_signed_xi": (
            signed_xi[best_key] / best_visits
            if best_visits
            else None
        ),
        "counterfactual_regret": regret,
        "selected_is_empirical_best": best_key == selected_key,
        "confidently_suboptimal": confidently_suboptimal,
        "confidently_better_candidates": [
            advantage.to_dict()
            for advantage in confidently_better
        ],
        "best_vs_selected_confidence": (
            best_advantage.to_dict()
            if best_advantage is not None
            else None
        ),
        "candidate_stats": candidate_stats,
        "paired_advantages": [
            advantage.to_dict()
            for advantage in paired_advantages
        ],
        "paired_worlds": [
            world.to_dict()
            for world in paired_world_evidence
        ],
        "search_health": {
            "simulations": sum(visits.values()),
            "elapsed_ms": round(
                (time.perf_counter() - started) * 1000.0,
                3,
            ),
            "determinization_failures": failures,
            "deadline_interruptions": deadline_interruptions,
            "rollout_invariant_violations": invariant_violations,
            "rollout_violations": list(dict.fromkeys(rollout_violations)),
            "rollout_coverage_failures": coverage_failures,
            "rollout_coverage_reasons": list(
                dict.fromkeys(coverage_reasons)
            ),
        },
    }


def _discard_labels(trace: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for action in trace.get("legal_actions") or ():
        if not isinstance(action, Mapping):
            continue
        if str(action.get("type") or "").upper() != "DISCARD":
            continue
        label = str(action.get("label") or "")
        if label and label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError("counterfactual_discard_legal_actions_missing")
    return labels


def _response_candidates(
    trace: Mapping[str, Any],
) -> list[RootResponseCandidate]:
    candidates: list[RootResponseCandidate] = []
    seen: set[str] = set()
    for action in trace.get("legal_actions") or ():
        if not isinstance(action, Mapping):
            continue
        action_type = str(action.get("type") or "").upper()
        if action_type not in {"PASS", "HU", "PENG", "CHI"}:
            raise ValueError(f"counterfactual_response_action_unsupported:{action_type}")
        key = str(action.get("key") or "")
        if not key:
            raise ValueError("counterfactual_response_action_key_missing")
        if key in seen:
            raise ValueError(f"counterfactual_response_action_duplicate:{key}")
        if action_type == "PASS" and key != "PASS":
            raise ValueError("counterfactual_response_pass_key_invalid")
        candidates.append(
            RootResponseCandidate(
                key=key,
                action_type=action_type,
                heuristic_value=0.0,
                option_id=(
                    str(action["option_id"])
                    if action.get("option_id") is not None
                    else None
                ),
                consumed_from_hand=tuple(
                    str(label)
                    for label in action.get("consumed_from_hand") or ()
                ),
                meld_groups=tuple(
                    tuple(str(label) for label in group)
                    for group in action.get("meld_groups") or ()
                ),
            )
        )
        seen.add(key)
    if not candidates:
        raise ValueError("counterfactual_response_legal_actions_missing")
    if "PASS" not in seen:
        raise ValueError("counterfactual_response_pass_missing")
    return candidates


def _hu_keys(trace: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for action in trace.get("legal_actions") or ():
        if not isinstance(action, Mapping):
            continue
        key = str(action.get("key") or "")
        action_type = str(action.get("type") or "").upper()
        if action_type not in {"HU", "PASS"}:
            raise ValueError(f"counterfactual_hu_action_unsupported:{action_type}")
        if key != action_type:
            raise ValueError(f"counterfactual_hu_action_key_invalid:{key}")
        if key in keys:
            raise ValueError(f"counterfactual_hu_action_duplicate:{key}")
        keys.append(key)
    if set(keys) != {"HU", "PASS"}:
        raise ValueError("counterfactual_hu_legal_actions_incomplete")
    return keys


class _ForcedHuAuditPolicy(BaselinePolicy):
    name = "counterfactual_hu_rollout"

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


__all__ = [
    "CounterfactualAuditConfig",
    "audit_discard_trace",
    "audit_hu_trace",
    "audit_response_trace",
]
