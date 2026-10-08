"""Frozen production routing for the accepted two-player strategy."""

from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from ai.pro_brain import ActionEval, PolicyDecision, choose_action as choose_production_action
from ai.search_brain import public_view_from_production_state
from ai.ismcts import public_view_to_dict
from ai.simulation_trace import TraceAction
from ai.simulation_trace import public_state_identity
from engine.chi_rules import ChiPlan, enumerate_chi_plans
from engine.rules import load_rules


FROZEN_CANDIDATE = "professional_v81_two_player_exact_discard_sharded_research"
FROZEN_RELEASES = {
    False: "2p-no-wang-20260809-rc3",
    True: "v9",
}

_POLICY_LOCK = threading.Lock()
_SELECTION_CACHE_MAX_SIZE = 512
_SELECTION_CACHE: OrderedDict[tuple[object, ...], str] = OrderedDict()
_PREWARM_LOCK = threading.Lock()
_PREWARM_THREAD: threading.Thread | None = None
_PREWARM_RESULT: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def _frozen_policy() -> Any:
    # Keep the exact league factory accepted for this release candidate.
    from ai.opponent_league import create_policy

    return create_policy(FROZEN_CANDIDATE)


def prewarm_two_player_strategy_runtime() -> dict[str, Any]:
    global _PREWARM_RESULT, _PREWARM_THREAD

    from ai.dual_discard_validator import (
        close_shared_dual_discard_executors,
        prewarm_shared_executor,
    )

    started = time.perf_counter()
    try:
        pools = [prewarm_shared_executor(workers) for workers in (20, 12)]
        result = {
            "ok": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "pools": pools,
        }
    except Exception as exc:
        close_shared_dual_discard_executors(wait=False)
        result = {
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": f"{type(exc).__name__}:{exc}",
            "pools": [],
        }
    with _PREWARM_LOCK:
        _PREWARM_RESULT = result
        if not result["ok"]:
            _PREWARM_THREAD = None
    return result


def start_two_player_strategy_prewarm() -> threading.Thread:
    global _PREWARM_THREAD

    with _PREWARM_LOCK:
        if _PREWARM_THREAD is not None:
            return _PREWARM_THREAD
        thread = threading.Thread(
            target=prewarm_two_player_strategy_runtime,
            name="aizipai-strategy-prewarm",
            daemon=True,
        )
        _PREWARM_THREAD = thread
        thread.start()
        return thread


def stop_two_player_strategy_runtime(timeout_seconds: float = 8.0) -> dict[str, Any]:
    global _PREWARM_THREAD

    with _PREWARM_LOCK:
        thread = _PREWARM_THREAD
    if thread is not None and thread is not threading.current_thread():
        thread.join(max(0.0, float(timeout_seconds)))
    alive = bool(thread is not None and thread.is_alive())
    from ai.dual_discard_validator import close_shared_dual_discard_executors

    close_shared_dual_discard_executors(wait=not alive)
    with _PREWARM_LOCK:
        _PREWARM_THREAD = None
    return {"ok": not alive, "prewarm_thread_alive": alive}


class ProductDecisionKernel:
    """Single product decision kernel shared by live and formal evaluation."""

    def choose_action(
        self,
        state_or_hand: dict[str, Any] | list[str],
        *,
        rules: dict[str, Any] | None = None,
        config_path: str = "config/rules.yaml",
        absolute_deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
        baseline_decision: PolicyDecision | None = None,
    ) -> PolicyDecision:
        return _choose_action_impl(
            state_or_hand,
            rules=rules,
            config_path=config_path,
            absolute_deadline=absolute_deadline,
            cancelled=cancelled,
            baseline_decision=baseline_decision,
        )


PRODUCT_DECISION_KERNEL = ProductDecisionKernel()


def choose_action(
    state_or_hand: dict[str, Any] | list[str],
    *,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
    absolute_deadline: float | None = None,
    cancelled: Callable[[], bool] | None = None,
    baseline_decision: PolicyDecision | None = None,
) -> PolicyDecision:
    return PRODUCT_DECISION_KERNEL.choose_action(
        state_or_hand,
        rules=rules,
        config_path=config_path,
        absolute_deadline=absolute_deadline,
        cancelled=cancelled,
        baseline_decision=baseline_decision,
    )


def _choose_action_impl(
    state_or_hand: dict[str, Any] | list[str],
    *,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
    absolute_deadline: float | None = None,
    cancelled: Callable[[], bool] | None = None,
    baseline_decision: PolicyDecision | None = None,
) -> PolicyDecision:
    """Choose through the frozen candidate in two-player rooms.

    The production evaluator remains the legality and click-instance authority.
    The accepted information-set candidate may only select from actions exposed by
    that evaluator. Other room sizes retain the existing production behavior.
    """

    resolved_rules = rules or load_rules(config_path)
    player_count = int(resolved_rules.get("game", {}).get("players", 3))
    baseline = baseline_decision or choose_production_action(
        state_or_hand,
        rules=resolved_rules,
        config_path=config_path,
        parallel_evaluation=(
            isinstance(state_or_hand, dict) and player_count == 2
        ),
    )
    if _planning_interrupted(absolute_deadline, cancelled):
        return baseline
    if not isinstance(state_or_hand, dict):
        return baseline
    if player_count != 2:
        return baseline

    wildcard_enabled = bool(resolved_rules.get("wildcard", {}).get("enabled"))
    release_id = FROZEN_RELEASES[wildcard_enabled]
    if baseline.safety_flags or baseline.selected_action in {
        "HU",
        "PAO",
        "TI",
        "SAFE_HALT",
    }:
        return _stamp(
            baseline,
            release_id=release_id,
            route="hard_priority",
            candidate_applied=False,
        )

    raw_legal_types = {
        str(item.get("type") if isinstance(item, Mapping) else item).upper()
        for item in state_or_hand.get("legal_actions") or ()
    }
    if baseline.selected_action == "DISCARD" and "DISCARD" in raw_legal_types:
        return _choose_discard(
            state_or_hand,
            baseline,
            rules=resolved_rules,
            release_id=release_id,
            absolute_deadline=absolute_deadline,
            cancelled=cancelled,
        )
    if raw_legal_types & {"CHI", "PENG", "PASS"}:
        return _choose_response(
            state_or_hand,
            baseline,
            rules=resolved_rules,
            release_id=release_id,
            legal_types=raw_legal_types,
            absolute_deadline=absolute_deadline,
            cancelled=cancelled,
        )
    return _stamp(
        baseline,
        release_id=release_id,
        route="unsupported_root_kept_production",
        candidate_applied=False,
    )


def _choose_discard(
    state: dict[str, Any],
    baseline: PolicyDecision,
    *,
    rules: dict[str, Any],
    release_id: str,
    absolute_deadline: float | None,
    cancelled: Callable[[], bool] | None,
) -> PolicyDecision:
    view = public_view_from_production_state(state, baseline, rules=rules)
    if view is None:
        return _stamp(
            baseline,
            release_id=release_id,
            route="production_fallback",
            candidate_applied=False,
            fallback_reason="insufficient_public_state_for_frozen_discard",
        )
    started = time.perf_counter()
    cache_key = (
        FROZEN_CANDIDATE,
        release_id,
        "discard",
        public_state_identity(public_view_to_dict(view)),
    )
    with _POLICY_LOCK:
        if _planning_interrupted(absolute_deadline, cancelled):
            return baseline
        selected_label = _selection_cache_get(cache_key)
        cache_hit = selected_label is not None
        if selected_label is None:
            selected_label = str(
                _frozen_policy().choose_discard(
                    view,
                    rules,
                    absolute_deadline=absolute_deadline,
                    production_decision=baseline,
                )
            )
            _selection_cache_put(cache_key, selected_label)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    selected_eval = _best_eval(
        baseline.action_evals,
        action_type="DISCARD",
        label=selected_label,
    )
    if selected_eval is None:
        return _stamp(
            baseline,
            release_id=release_id,
            route="production_fallback",
            candidate_applied=False,
            fallback_reason="frozen_discard_not_executable",
            elapsed_ms=elapsed_ms,
        )
    return _select_eval(
        baseline,
        selected_eval,
        release_id=release_id,
        route="frozen_candidate_discard",
        candidate_applied=True,
        elapsed_ms=elapsed_ms,
        cache_hit=cache_hit,
    )


def _choose_response(
    state: dict[str, Any],
    baseline: PolicyDecision,
    *,
    rules: dict[str, Any],
    release_id: str,
    legal_types: set[str],
    absolute_deadline: float | None,
    cancelled: Callable[[], bool] | None,
) -> PolicyDecision:
    pending = str(state.get("pending_card") or state.get("external_card") or "")
    view = public_view_from_production_state(
        state,
        baseline,
        rules=rules,
        response_root=True,
    )
    if view is None or not pending:
        return _stamp(
            baseline,
            release_id=release_id,
            route="production_fallback",
            candidate_applied=False,
            fallback_reason="insufficient_public_state_for_frozen_response",
        )

    plans = (
        enumerate_chi_plans(
            list(view.hand),
            pending,
            allow_1510=bool(rules.get("rules", {}).get("allow_1510", False)),
        )
        if "CHI" in legal_types
        else []
    )
    trace_actions, plan_by_key = _response_trace_actions(
        pending,
        legal_types=legal_types,
        plans=plans,
    )
    if not trace_actions:
        return _stamp(
            baseline,
            release_id=release_id,
            route="production_fallback",
            candidate_applied=False,
            fallback_reason="no_frozen_response_candidates",
        )

    started = time.perf_counter()
    cache_key = (
        FROZEN_CANDIDATE,
        release_id,
        "response",
        public_state_identity(public_view_to_dict(view)),
        tuple(
            (
                action.key,
                action.type,
                action.label,
                action.option_id,
                action.consumed_from_hand,
                action.meld_groups,
            )
            for action in trace_actions
        ),
    )
    with _POLICY_LOCK:
        if _planning_interrupted(absolute_deadline, cancelled):
            return baseline
        selected_key = _selection_cache_get(cache_key)
        cache_hit = selected_key is not None
        if selected_key is None:
            policy = _frozen_policy()
            try:
                selected_key = str(
                    policy.choose_response(
                        view,
                        trace_actions,
                        plans,
                        None,
                        rules,
                        absolute_deadline=absolute_deadline,
                        production_decision=baseline,
                    )
                )
            finally:
                policy.finish_pending_response()
            _selection_cache_put(cache_key, selected_key)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if selected_key == "PASS":
        selected_eval = _best_eval(baseline.action_evals, action_type="PASS")
    elif selected_key.startswith("PENG:"):
        selected_eval = _best_eval(baseline.action_evals, action_type="PENG")
    elif selected_key.startswith("CHI:"):
        plan = plan_by_key.get(selected_key)
        selected_eval = _chi_eval_for_plan(baseline.action_evals, plan)
        if selected_eval is None and baseline.selected_action == "EXPAND_CHI_OPTIONS":
            return _stamp(
                baseline,
                release_id=release_id,
                route="frozen_candidate_requires_chi_options",
                candidate_applied=True,
                elapsed_ms=elapsed_ms,
                cache_hit=cache_hit,
            )
    else:
        selected_eval = None

    if selected_eval is None:
        return _stamp(
            baseline,
            release_id=release_id,
            route="production_fallback",
            candidate_applied=False,
            fallback_reason="frozen_response_not_executable",
            elapsed_ms=elapsed_ms,
        )
    return _select_eval(
        baseline,
        selected_eval,
        release_id=release_id,
        route="frozen_candidate_response",
        candidate_applied=True,
        elapsed_ms=elapsed_ms,
        cache_hit=cache_hit,
    )


def _selection_cache_get(key: tuple[object, ...]) -> str | None:
    value = _SELECTION_CACHE.get(key)
    if value is not None:
        _SELECTION_CACHE.move_to_end(key)
    return value


def _planning_interrupted(
    absolute_deadline: float | None,
    cancelled: Callable[[], bool] | None,
) -> bool:
    return bool(
        (cancelled is not None and cancelled())
        or (
            absolute_deadline is not None
            and time.perf_counter() >= absolute_deadline
        )
    )


def _selection_cache_put(key: tuple[object, ...], value: str) -> None:
    _SELECTION_CACHE[key] = value
    _SELECTION_CACHE.move_to_end(key)
    while len(_SELECTION_CACHE) > _SELECTION_CACHE_MAX_SIZE:
        _SELECTION_CACHE.popitem(last=False)


def _response_trace_actions(
    pending: str,
    *,
    legal_types: set[str],
    plans: Sequence[ChiPlan],
) -> tuple[list[TraceAction], dict[str, ChiPlan]]:
    actions: list[TraceAction] = []
    plan_by_key: dict[str, ChiPlan] = {}
    if "PASS" in legal_types:
        actions.append(TraceAction(key="PASS", type="PASS"))
    if "PENG" in legal_types:
        actions.append(
            TraceAction(
                key=f"PENG:{pending}",
                type="PENG",
                label=pending,
                consumed_from_hand=(pending, pending),
                meld_groups=((pending, pending, pending),),
                priority=2,
            )
        )
    for index, plan in enumerate(plans, start=1):
        key = f"CHI:runtime_{index:03d}"
        plan_by_key[key] = plan
        actions.append(
            TraceAction(
                key=key,
                type="CHI",
                option_id=key.removeprefix("CHI:"),
                consumed_from_hand=tuple(plan.consumed_from_hand),
                meld_groups=tuple(tuple(group) for group in plan.groups),
                priority=1,
            )
        )
    return actions, plan_by_key


def _best_eval(
    evals: Sequence[ActionEval],
    *,
    action_type: str,
    label: str | None = None,
) -> ActionEval | None:
    matches = [
        item
        for item in evals
        if item.action.type == action_type
        and (label is None or item.action.label == label)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: (bool(item.allowed), float(item.ev)))


def _chi_eval_for_plan(
    evals: Sequence[ActionEval],
    plan: ChiPlan | None,
) -> ActionEval | None:
    if plan is None:
        return None
    consumed = Counter(plan.consumed_from_hand)
    initial = Counter(plan.initial_group)
    compares = _group_multiset_key(plan.compare_groups)
    for item in evals:
        if item.action.type != "CHI":
            continue
        debug = item.debug_details
        if (
            Counter(debug.get("consumed_from_hand") or ()) == consumed
            and Counter(debug.get("meld_cards") or ()) == initial
            and _group_multiset_key(debug.get("compare_groups") or ()) == compares
        ):
            return item
    return None


def _group_multiset_key(groups: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(tuple(sorted(group)) for group in groups))


def _select_eval(
    baseline: PolicyDecision,
    selected_eval: ActionEval,
    *,
    release_id: str,
    route: str,
    candidate_applied: bool,
    elapsed_ms: float,
    cache_hit: bool = False,
) -> PolicyDecision:
    action = selected_eval.action
    stage = str(selected_eval.debug_details.get("candidate_stage") or action.type.lower())
    decision = replace(
        baseline,
        selected_action=action.type,
        selected_card_id=action.card_id,
        selected_label=action.label,
        selected_option_id=action.option_id,
        candidate_stage=stage,
        ev=float(selected_eval.ev),
        reason=f"{release_id}:{route}; {selected_eval.reason}",
        requires_action_plan=True,
        policy_version=release_id,
    )
    return _stamp(
        decision,
        release_id=release_id,
        route=route,
        candidate_applied=candidate_applied,
        elapsed_ms=elapsed_ms,
        cache_hit=cache_hit,
    )


def _stamp(
    decision: PolicyDecision,
    *,
    release_id: str,
    route: str,
    candidate_applied: bool,
    fallback_reason: str | None = None,
    elapsed_ms: float = 0.0,
    cache_hit: bool = False,
) -> PolicyDecision:
    snapshot = deepcopy(decision.context_snapshot)
    snapshot["product_strategy"] = {
        "release_id": release_id,
        "candidate": FROZEN_CANDIDATE,
        "route": route,
        "candidate_applied": candidate_applied,
        "fallback_reason": fallback_reason,
        "elapsed_ms": round(float(elapsed_ms), 3),
        "cache_hit": bool(cache_hit),
    }
    return replace(
        decision,
        context_snapshot=snapshot,
        policy_version=release_id,
    )


__all__ = [
    "FROZEN_CANDIDATE",
    "FROZEN_RELEASES",
    "choose_action",
    "start_two_player_strategy_prewarm",
    "stop_two_player_strategy_runtime",
]
