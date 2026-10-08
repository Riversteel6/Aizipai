"""Deterministic two- or three-player Chenzhou Zipai full-game simulation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from multiprocessing import get_context
from statistics import mean
from typing import Any, Protocol

from ai.features import (
    QUICK_POTENTIAL_LABELS,
    best_quick_potential_after_discard_from_counts,
    quick_potential,
    quick_potential_after_each_removal_from_counts,
)
from engine.cards import BIG_LABELS, RED_LABELS, SMALL_LABELS, WILD_LABEL, parse_label
from engine.chi_rules import ChiPlan, enumerate_chi_plans
from engine.deck import expanded_deck, full_deck_counts
from engine.hu_checker import best_grouping_normalized, concealed_group_xi
from engine.melds import classify_meld
from engine.red_black_rules import classify_red_black
from engine.rules import rules_for_room
from engine.xi_calculator import meld_xi
from ai.simulation_trace import SimulationDecisionTrace, TraceAction


_QUICK_POTENTIAL_INDEX = {
    label: index
    for index, label in enumerate(QUICK_POTENTIAL_LABELS)
}
_ROLLOUT_FEATURE_CACHE_SIZE = (
    16_384 if os.environ.get("AIZIPAI_MOBILE_RUNTIME") == "1" else 100_000
)


@dataclass(frozen=True)
class SimMeld:
    kind: str
    cards: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"type": self.kind, "labels": list(self.cards)}


@dataclass
class SimPlayer:
    seat: int
    hand: list[str] = field(default_factory=list)
    melds: list[SimMeld] = field(default_factory=list)
    discards: list[str] = field(default_factory=list)
    passed_chi: set[str] = field(default_factory=set)
    passed_peng: set[str] = field(default_factory=set)
    quad_events: int = 0


@dataclass(frozen=True)
class HuEvaluation:
    can_hu: bool
    total_xi: int
    groups: tuple[tuple[str, ...], ...]
    score: float
    red_count: int


@dataclass(frozen=True)
class PublicView:
    seat: int
    hand: tuple[str, ...]
    own_melds: tuple[SimMeld, ...]
    all_melds: tuple[tuple[SimMeld, ...], ...]
    discards: tuple[tuple[str, ...], ...]
    remaining_counts: tuple[tuple[str, int], ...]
    stock_count: int
    hand_sizes: tuple[int, ...] = ()
    pending_card: str | None = None
    pending_source_seat: int | None = None
    passed_chi: tuple[tuple[str, ...], ...] = ()
    passed_peng: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class GameResult:
    seed: int
    wildcard_enabled: bool
    winner: int | None
    dealer: int
    turns: int
    reason: str
    score: float
    total_xi: int
    action_counts: dict[str, int]
    violations: tuple[str, ...]
    coverage_failures: tuple[str, ...] = ()
    stalled_seat: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "wildcard_enabled": self.wildcard_enabled,
            "winner": self.winner,
            "dealer": self.dealer,
            "turns": self.turns,
            "reason": self.reason,
            "score": self.score,
            "total_xi": self.total_xi,
            "action_counts": dict(self.action_counts),
            "violations": list(self.violations),
            "coverage_failures": list(self.coverage_failures),
            "stalled_seat": self.stalled_seat,
        }


@dataclass(frozen=True)
class _DiscardResponseResolution:
    current: int
    needs_draw: bool
    continue_play: bool
    terminal_result: GameResult | None = None
    claim_type: str = "pass"
    claimant: int | None = None
    skip_hu_once: bool = False


@dataclass(frozen=True)
class _ResponseIntent:
    seat: int
    view: PublicView
    legal_actions: tuple[TraceAction, ...]
    selected_key: str
    chi_plan: ChiPlan | None
    hu: HuEvaluation | None
    state_before_hash: str


class SimulationPolicy(Protocol):
    name: str

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str: ...

    def choose_hu(
        self,
        view: PublicView,
        hu: HuEvaluation,
        rules: dict[str, Any],
    ) -> bool: ...

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool: ...

    def choose_chi(self, view: PublicView, plans: list[ChiPlan], rules: dict[str, Any]) -> ChiPlan | None: ...


class BaselinePolicy:
    name = "baseline"

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        labels = _discardable_labels(view.hand)
        if not labels:
            raise ValueError("no_discardable_card")
        return max(
            labels,
            key=lambda label: (
                quick_potential(_remove_one(view.hand, label)),
                -int(label in RED_LABELS),
                -_rank_distance_penalty(label, view.hand),
                label,
            ),
        )

    def choose_hu(
        self,
        view: PublicView,
        hu: HuEvaluation,
        rules: dict[str, Any],
    ) -> bool:
        return hu.can_hu

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        if view.hand.count(label) != 2:
            return False
        before = quick_potential(list(view.hand))
        after = quick_potential(_remove_many(view.hand, [label, label]))
        return meld_xi([label] * 3, kind="peng", rules=rules) > 0 and after + 2 >= before

    def choose_chi(self, view: PublicView, plans: list[ChiPlan], rules: dict[str, Any]) -> ChiPlan | None:
        before = quick_potential(list(view.hand))
        scored = [
            (quick_potential(_remove_many(view.hand, plan.consumed_from_hand)) - before, plan)
            for plan in plans
        ]
        best_gain, best = max(scored, key=lambda item: item[0])
        return best if best_gain >= 0 else None


class InformationSetSearchPolicy(BaselinePolicy):
    """Two-ply expectimax over the unseen-card distribution."""

    name = "information_set_search_v2"

    def bind_rollout_hu_cache(
        self,
        rules: dict[str, Any],
        cache: dict[
            tuple[tuple[str, ...], tuple[SimMeld, ...]],
            HuEvaluation,
        ],
        *,
        position_cache: dict[
            tuple[
                tuple[str, ...],
                tuple[SimMeld, ...],
                tuple[tuple[str, int], ...],
            ],
            float,
        ] | None = None,
        followup_cache: dict[
            tuple[tuple[str, ...], tuple[SimMeld, ...]],
            float,
        ] | None = None,
    ) -> None:
        """Bind an exact cache shared by paired paths of one hidden world."""

        self._hu_cache_rules_identity = id(rules)
        self._rollout_hu_cache = cache
        self._rollout_position_cache = (
            position_cache if position_cache is not None else {}
        )
        self._rollout_followup_cache = (
            followup_cache if followup_cache is not None else {}
        )

    def _hu_cache_for_rules(
        self,
        rules: dict[str, Any],
    ) -> dict[
        tuple[tuple[str, ...], tuple[SimMeld, ...]],
        HuEvaluation,
    ]:
        """Reuse exact Hu evaluations within one rollout policy instance.

        Rollout policies are instantiated for one determinized game.  The old
        implementation discarded this pure-result cache between discard, peng,
        and chi decisions in that same game, causing identical hand positions
        to be solved repeatedly.  Keep the cache local to the policy and clear
        it defensively if a caller reuses the policy with a different rules
        object.
        """

        rules_identity = id(rules)
        if getattr(self, "_hu_cache_rules_identity", None) != rules_identity:
            self._hu_cache_rules_identity = rules_identity
            self._rollout_hu_cache = {}
            self._rollout_position_cache = {}
            self._rollout_followup_cache = {}
            self._rollout_meld_xi_cache = {}
        return self._rollout_hu_cache

    def _existing_xi_cached(
        self,
        melds: tuple[SimMeld, ...] | list[SimMeld],
        rules: dict[str, Any],
    ) -> int:
        self._hu_cache_for_rules(rules)
        key = tuple(melds)
        cache = getattr(self, "_rollout_meld_xi_cache", None)
        if cache is None:
            cache = {}
            self._rollout_meld_xi_cache = cache
        cached = cache.get(key)
        if cached is None:
            cached = sum(
                meld_xi(list(meld.cards), kind=meld.kind, rules=rules)
                for meld in key
            )
            cache[key] = cached
        return cached

    def _position_value_cached(
        self,
        hand: tuple[str, ...] | list[str],
        melds: tuple[SimMeld, ...] | list[SimMeld],
        remaining_counts: tuple[tuple[str, int], ...],
        rules: dict[str, Any],
    ) -> float:
        hu_cache = self._hu_cache_for_rules(rules)
        cache = self._rollout_position_cache
        key = (tuple(sorted(hand)), tuple(melds), tuple(remaining_counts))
        value = cache.get(key)
        if value is None:
            value = _position_value(
                hand,
                melds,
                remaining_counts,
                rules,
                hu_cache=hu_cache,
                existing_xi=self._existing_xi_cached(melds, rules),
            )
            cache[key] = value
        return value

    def _best_quick_followup_value_cached(
        self,
        hand: tuple[str, ...],
        melds: tuple[SimMeld, ...],
        rules: dict[str, Any],
    ) -> float:
        return self._best_quick_followup_value_canonical_cached(
            tuple(sorted(hand)),
            melds,
            rules,
        )

    def _best_quick_followup_value_canonical_cached(
        self,
        hand: tuple[str, ...],
        melds: tuple[SimMeld, ...],
        rules: dict[str, Any],
    ) -> float:
        self._hu_cache_for_rules(rules)
        cache = self._rollout_followup_cache
        key = (hand, tuple(melds))
        value = cache.get(key)
        if value is None:
            value = _best_quick_followup_value_global(
                key[0],
                self._existing_xi_cached(melds, rules),
            )
            cache[key] = value
        return value

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        return self._choose_discard(
            view,
            rules,
            hu_cache=self._hu_cache_for_rules(rules),
        )

    def _choose_discard(
        self,
        view: PublicView,
        rules: dict[str, Any],
        *,
        hu_cache: dict[
            tuple[tuple[str, ...], tuple[SimMeld, ...]],
            HuEvaluation,
        ],
    ) -> str:
        labels = _discardable_labels(view.hand)
        if not labels:
            raise ValueError("no_discardable_card")
        wildcard_enabled = bool(rules.get("wildcard", {}).get("enabled", False))
        shortlist_size = 8 if wildcard_enabled else 4
        cheap_values = self._cheap_discard_values(view, rules, labels)
        shortlist = sorted(
            labels,
            key=cheap_values.__getitem__,
            reverse=True,
        )[:shortlist_size]
        unseen_items = tuple(dict(view.remaining_counts).items())
        unseen_total = sum(amount for _label, amount in unseen_items)
        return max(
            shortlist,
            key=lambda label: self._discard_value(
                view,
                label,
                rules,
                hu_cache=hu_cache,
                unseen_items=unseen_items,
                unseen_total=unseen_total,
            ),
        )

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        if view.hand.count(label) != 2:
            return False
        hu_cache = self._hu_cache_for_rules(rules)
        pass_value = self._position_value_cached(
            view.hand,
            view.own_melds,
            view.remaining_counts,
            rules,
        )
        hand_after_claim = tuple(_remove_many(view.hand, [label, label]))
        melds = (*view.own_melds, SimMeld("peng", (label, label, label)))
        claim_view = _replace_view(view, hand=hand_after_claim, own_melds=melds)
        try:
            discard = self._choose_discard(
                claim_view,
                rules,
                hu_cache=hu_cache,
            )
        except ValueError:
            return False
        after = tuple(_remove_one(hand_after_claim, discard))
        claim_value = self._position_value_cached(
            after,
            melds,
            view.remaining_counts,
            rules,
        )
        return claim_value > pass_value + 15

    def choose_chi(self, view: PublicView, plans: list[ChiPlan], rules: dict[str, Any]) -> ChiPlan | None:
        hu_cache = self._hu_cache_for_rules(rules)
        pass_value = self._position_value_cached(
            view.hand,
            view.own_melds,
            view.remaining_counts,
            rules,
        )
        best_plan: ChiPlan | None = None
        best_value = -math.inf
        for plan in plans:
            hand_after_claim = tuple(_remove_many(view.hand, plan.consumed_from_hand))
            melds = (
                *view.own_melds,
                *(SimMeld(_chi_kind(group), tuple(group)) for group in plan.groups),
            )
            claim_view = _replace_view(view, hand=hand_after_claim, own_melds=melds)
            try:
                discard = self._choose_discard(
                    claim_view,
                    rules,
                    hu_cache=hu_cache,
                )
            except ValueError:
                continue
            after = tuple(_remove_one(hand_after_claim, discard))
            value = self._position_value_cached(
                after,
                melds,
                view.remaining_counts,
                rules,
            )
            if value > best_value:
                best_value = value
                best_plan = plan
        return best_plan if best_plan is not None and best_value > pass_value + 20 else None

    def _discard_value(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
        *,
        hu_cache: dict[
            tuple[tuple[str, ...], tuple[SimMeld, ...]],
            HuEvaluation,
        ] | None = None,
        unseen_items: tuple[tuple[str, int], ...] | None = None,
        unseen_total: int | None = None,
    ) -> float:
        after = tuple(_remove_one(view.hand, label))
        after_sorted = tuple(sorted(after))
        current = self._position_value_cached(
            after,
            view.own_melds,
            view.remaining_counts,
            rules,
        )
        if unseen_items is None:
            unseen_items = tuple(dict(view.remaining_counts).items())
        total = (
            sum(amount for _label, amount in unseen_items)
            if unseen_total is None
            else unseen_total
        )
        if total <= 0:
            return current
        expected = 0.0
        wildcard_enabled = bool(rules.get("wildcard", {}).get("enabled", False))
        draw_sample_size = 12 if wildcard_enabled else 8
        draw_relevance = _draw_shape_relevance_values(
            tuple(label for label, _amount in unseen_items),
            after,
        )
        sampled_draws = sorted(
            unseen_items,
            key=lambda item: (item[1], draw_relevance[item[0]]),
            reverse=True,
        )[:draw_sample_size]
        sampled_total = sum(amount for _draw, amount in sampled_draws)
        for draw, amount in sampled_draws:
            if amount <= 0:
                continue
            drawn = _insert_sorted_label(after_sorted, draw)
            hu = _evaluate_hu_canonical_with_local_cache(
                drawn,
                view.own_melds,
                rules,
                hu_cache=hu_cache,
            )
            if hu.can_hu:
                future = 1200.0 + hu.score * 120.0
            else:
                future = self._best_quick_followup_value_canonical_cached(
                    drawn,
                    view.own_melds,
                    rules,
                )
            expected += amount / max(1, sampled_total) * future
        danger = _public_discard_danger(label, view)
        return current + expected * 0.35 - danger

    @staticmethod
    def _cheap_discard_value(view: PublicView, label: str, rules: dict[str, Any]) -> float:
        after = _remove_one(view.hand, label)
        existing_xi = sum(meld_xi(list(meld.cards), kind=meld.kind, rules=rules) for meld in view.own_melds)
        return quick_potential(after) * 12.0 + existing_xi * 45.0 + sum(card in RED_LABELS for card in after) * 3.0

    @staticmethod
    def _cheap_discard_values(
        view: PublicView,
        rules: dict[str, Any],
        labels: list[str],
    ) -> dict[str, float]:
        counts = [0] * len(QUICK_POTENTIAL_LABELS)
        for card in view.hand:
            counts[_QUICK_POTENTIAL_INDEX[card]] += 1
        removal_values = quick_potential_after_each_removal_from_counts(
            tuple(counts)
        )
        existing_xi = sum(
            meld_xi(list(meld.cards), kind=meld.kind, rules=rules)
            for meld in view.own_melds
        )
        red_count = sum(card in RED_LABELS for card in view.hand)
        return {
            label: (
                int(removal_values[_QUICK_POTENTIAL_INDEX[label]]) * 12.0
                + existing_xi * 45.0
                + (red_count - int(label in RED_LABELS)) * 3.0
            )
            for label in labels
        }


class ProfessionalBrainSimulationPolicy:
    """Adapt public simulator state to the production decision entry point."""

    name = "professional_brain_v2"
    seat_aware_opponents = False

    def __init__(self) -> None:
        self.last_decision_evidence: dict[str, Any] = {}

    def choose_discard(self, view: PublicView, rules: dict[str, Any]) -> str:
        decision = self._choose(view, rules, legal_actions=[{"type": "DISCARD"}])
        self._remember_decision(decision)
        if decision.selected_action != "DISCARD" or not decision.selected_label:
            reject_counts = Counter(
                item.reject_reason or "allowed"
                for item in decision.action_evals
                if item.type == "DISCARD"
            )
            raise ValueError(
                "professional_brain_no_discard:"
                f"{decision.selected_action}:"
                f"reason={decision.reason}:"
                f"safety={','.join(decision.safety_flags)}:"
                f"hand={''.join(view.hand)}:"
                f"melds={','.join(''.join(meld.cards) for meld in view.own_melds)}:"
                f"discard_evals={dict(reject_counts)}"
            )
        return decision.selected_label

    def choose_hu(
        self,
        view: PublicView,
        hu: HuEvaluation,
        rules: dict[str, Any],
    ) -> bool:
        decision = self._choose(
            view,
            rules,
            legal_actions=[{"type": "HU"}, {"type": "PASS"}],
            pending_card=view.pending_card,
        )
        self._remember_decision(decision)
        production_accepts = decision.selected_action == "HU"
        if production_accepts or not hu.can_hu:
            return production_accepts
        self.last_decision_evidence = {
            **self.last_decision_evidence,
            "production_selected_action": decision.selected_action,
            "selected_action": "HU",
            "reason": "authoritative_simulator_hu_overrode_local_false_negative",
            "simulation_rule_crosscheck": {
                "simulator_can_hu": True,
                "simulator_total_xi": hu.total_xi,
                "simulator_score": hu.score,
            },
        }
        return True

    def choose_peng(self, view: PublicView, label: str, rules: dict[str, Any]) -> bool:
        decision = self._choose_peng_decision(view, label, rules)
        self._remember_decision(decision)
        return decision.selected_action == "PENG"

    def _choose_peng_decision(
        self,
        view: PublicView,
        label: str,
        rules: dict[str, Any],
    ):
        return self._choose(
            view,
            rules,
            legal_actions=[{"type": "PENG"}, {"type": "PASS"}],
            pending_card=label,
        )

    def choose_chi(self, view: PublicView, plans: list[ChiPlan], rules: dict[str, Any]) -> ChiPlan | None:
        decision = self._choose_chi_decision(view, plans, rules)
        self._remember_decision(decision)
        return self._chi_plan_for_decision(decision, plans)

    def _remember_decision(self, decision: Any) -> None:
        to_dict = getattr(decision, "to_dict", None)
        self.last_decision_evidence = (
            to_dict()
            if callable(to_dict)
            else {
                "selected_action": getattr(decision, "selected_action", None),
                "reason": getattr(decision, "reason", None),
            }
        )

    def _choose_chi_decision(
        self,
        view: PublicView,
        plans: list[ChiPlan],
        rules: dict[str, Any],
    ):
        option_groups: list[tuple[str, str, str]] = []
        for plan in plans:
            if plan.initial_group not in option_groups:
                option_groups.append(plan.initial_group)
        chi_options = [
            {
                "option_id": f"sim_chi_{index:03d}",
                "labels": list(group),
                "confidence": 1.0,
            }
            for index, group in enumerate(option_groups, start=1)
        ]
        pending = _external_label_for_plans(plans)
        return self._choose(
            view,
            rules,
            legal_actions=[{"type": "CHI"}, {"type": "PASS"}],
            pending_card=pending,
            chi_options=chi_options,
        )

    @staticmethod
    def _chi_plan_for_decision(
        decision: Any,
        plans: list[ChiPlan],
    ) -> ChiPlan | None:
        if decision.selected_action != "CHI":
            return None
        selected_eval = next(
            (
                item
                for item in decision.action_evals
                if item.type == "CHI"
                and item.allowed
                and item.action.option_id == decision.selected_option_id
            ),
            None,
        )
        if selected_eval is None:
            raise ValueError("professional_brain_chi_eval_missing")
        consumed = Counter(selected_eval.debug_details.get("consumed_from_hand") or [])
        compare_groups = _group_multiset_key(selected_eval.debug_details.get("compare_groups") or [])
        initial_cards = Counter(selected_eval.debug_details.get("meld_cards") or [])
        for plan in plans:
            if (
                Counter(plan.initial_group) == initial_cards
                and Counter(plan.consumed_from_hand) == consumed
                and _group_multiset_key(plan.compare_groups) == compare_groups
            ):
                return plan
        raise ValueError("professional_brain_chi_plan_mapping_failed")

    def _choose(
        self,
        view: PublicView,
        rules: dict[str, Any],
        *,
        legal_actions: list[dict[str, str]],
        pending_card: str | None = None,
        chi_options: list[dict[str, object]] | None = None,
    ):
        from ai.pro_brain import choose_action as choose_production_action

        state = _production_state_from_public_view(
            view,
            legal_actions=legal_actions,
            pending_card=pending_card,
            chi_options=chi_options,
            seat_aware_opponents=self.seat_aware_opponents,
        )
        return choose_production_action(
            state,
            rules=rules,
            parallel_evaluation=bool(
                getattr(self, "parallel_production_evaluation", False)
            ),
        )


class FullGameSimulator:
    def __init__(
        self,
        policies: list[SimulationPolicy],
        *,
        wildcard_enabled: bool,
        dealer: int = 0,
        rules: dict[str, Any] | None = None,
        record_decisions: bool = False,
    ) -> None:
        self.rules = rules or rules_for_room(
            wildcard_enabled=wildcard_enabled,
            players=len(policies),
        )
        self.player_count = int(self.rules.get("game", {}).get("players", 3))
        if self.player_count not in {2, 3}:
            raise ValueError(f"unsupported player count: {self.player_count}")
        if len(policies) != self.player_count:
            raise ValueError(
                f"policy count {len(policies)} does not match room player count {self.player_count}"
            )
        self.policies = policies
        self.dealer = dealer % self.player_count
        self.wildcard_enabled = wildcard_enabled
        self.record_decisions = bool(record_decisions)
        self._decision_trace: list[SimulationDecisionTrace] = []

    def decision_trace(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self._decision_trace]

    def _record_decision(
        self,
        *,
        turn: int,
        phase: str,
        seat: int,
        view: PublicView,
        legal_actions: list[TraceAction],
        selected_key: str,
        players: list[SimPlayer],
        stock: list[str],
        state_before_hash: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.record_decisions:
            return
        policy = self.policies[seat]
        evidence = _policy_evidence(policy)
        selected_reason = (
            reason
            or str(evidence.get("reason") or "")
            or "policy_callback"
        )
        self._decision_trace.append(
            SimulationDecisionTrace(
                sequence=len(self._decision_trace) + 1,
                turn=turn,
                phase=phase,
                seat=seat,
                policy=_policy_name(policy),
                public_state=_public_view_trace_payload(view),
                legal_actions=tuple(legal_actions),
                selected_key=selected_key,
                reason=selected_reason,
                state_before_hash=state_before_hash,
                state_after_hash=_simulation_state_hash(players, stock),
                policy_evidence=evidence,
                metadata=dict(metadata or {}),
            )
        )

    def play(self, seed: int, *, max_turns: int = 200) -> GameResult:
        self._decision_trace = []
        rng = random.Random(seed)
        initial_counts = full_deck_counts(rules=self.rules)
        deck = expanded_deck(initial_counts)
        rng.shuffle(deck)
        players = [SimPlayer(seat=index) for index in range(self.player_count)]
        for _ in range(20):
            for player in players:
                player.hand.append(deck.pop())
        players[self.dealer].hand.append(deck.pop())
        stock = deck
        action_counts: Counter[str] = Counter()
        violations: list[str] = []
        self._lay_initial_quads(players, action_counts)
        violations.extend(_invariant_errors(players, stock, initial_counts))
        if violations:
            return GameResult(
                seed=seed,
                wildcard_enabled=self.wildcard_enabled,
                winner=None,
                dealer=self.dealer,
                turns=0,
                reason="invariant_failure",
                score=0.0,
                total_xi=0,
                action_counts=dict(action_counts),
                violations=tuple(violations),
            )
        return self._continue_play(
            seed=seed,
            players=players,
            stock=stock,
            initial_counts=initial_counts,
            current=self.dealer,
            needs_draw=False,
            turns=0,
            action_counts=action_counts,
            violations=violations,
            max_turns=max_turns,
        )

    def play_from_state(
        self,
        *,
        seed: int,
        players: list[SimPlayer],
        stock: list[str],
        current: int,
        needs_draw: bool = False,
        initial_drawn_card: str | None = None,
        turns: int = 0,
        max_turns: int = 200,
        deadline: float | None = None,
    ) -> GameResult:
        """Continue a fully determinized state for information-set rollouts."""

        self._decision_trace = []
        initial_counts = full_deck_counts(rules=self.rules)
        action_counts: Counter[str] = Counter()
        violations = []
        if len(players) != self.player_count:
            violations.append(
                f"player_count_mismatch:expected={self.player_count}:actual={len(players)}"
            )
        if initial_drawn_card is not None:
            if needs_draw:
                violations.append("initial_drawn_card_conflicts_with_needs_draw")
            elif initial_drawn_card not in players[current].hand:
                violations.append(
                    "initial_drawn_card_missing_from_current_hand:"
                    f"seat={current}:card={initial_drawn_card}"
                )
        violations.extend(_invariant_errors(players, stock, initial_counts))
        if violations:
            return GameResult(
                seed=seed,
                wildcard_enabled=self.wildcard_enabled,
                winner=None,
                dealer=self.dealer,
                turns=turns,
                reason="invariant_failure",
                score=0.0,
                total_xi=0,
                action_counts=dict(action_counts),
                violations=tuple(violations),
            )
        return self._continue_play(
            seed=seed,
            players=players,
            stock=stock,
            initial_counts=initial_counts,
            current=current % self.player_count,
            needs_draw=needs_draw,
            turns=turns,
            action_counts=action_counts,
            violations=[],
            max_turns=max_turns,
            deadline=deadline,
            initial_drawn_card=initial_drawn_card,
        )

    def play_from_pending_discard(
        self,
        *,
        seed: int,
        players: list[SimPlayer],
        stock: list[str],
        discarder: int,
        pending_card: str,
        turns: int = 0,
        max_turns: int = 200,
        blocked_auto_claim_seats: frozenset[int] = frozenset(),
        deadline: float | None = None,
    ) -> GameResult:
        """Resolve one public discard, then continue the determinized game."""

        self._decision_trace = []
        initial_counts = full_deck_counts(rules=self.rules)
        action_counts: Counter[str] = Counter()
        violations: list[str] = []
        if len(players) != self.player_count:
            violations.append(
                f"player_count_mismatch:expected={self.player_count}:actual={len(players)}"
            )
        if not 0 <= discarder < self.player_count:
            violations.append(f"invalid_discarder:{discarder}")
        if any(not 0 <= seat < self.player_count for seat in blocked_auto_claim_seats):
            violations.append(f"invalid_blocked_auto_claim_seats:{sorted(blocked_auto_claim_seats)}")
        violations.extend(
            _invariant_errors(
                players,
                stock,
                initial_counts,
                extra_cards=(pending_card,),
            )
        )
        if violations:
            return GameResult(
                seed=seed,
                wildcard_enabled=self.wildcard_enabled,
                winner=None,
                dealer=self.dealer,
                turns=turns,
                reason="invariant_failure",
                score=0.0,
                total_xi=0,
                action_counts=dict(action_counts),
                violations=tuple(violations),
            )
        resolution = self._resolve_discard_response(
            seed=seed,
            players=players,
            stock=stock,
            initial_counts=initial_counts,
            discarder=discarder,
            discard=pending_card,
            turns=turns,
            action_counts=action_counts,
            violations=violations,
            blocked_auto_claim_seats=blocked_auto_claim_seats,
        )
        self._finish_pending_response_policies()
        if resolution.terminal_result is not None:
            return resolution.terminal_result
        if not resolution.continue_play:
            return GameResult(
                seed=seed,
                wildcard_enabled=self.wildcard_enabled,
                winner=None,
                dealer=self.dealer,
                turns=turns,
                reason="invariant_failure",
                score=0.0,
                total_xi=0,
                action_counts=dict(action_counts),
                violations=tuple(violations),
            )
        return self._continue_play(
            seed=seed,
            players=players,
            stock=stock,
            initial_counts=initial_counts,
            current=resolution.current,
            needs_draw=resolution.needs_draw,
            turns=turns,
            action_counts=action_counts,
            violations=violations,
            max_turns=max_turns,
            deadline=deadline,
            skip_initial_hu=resolution.skip_hu_once,
        )

    def _continue_play(
        self,
        *,
        seed: int,
        players: list[SimPlayer],
        stock: list[str],
        initial_counts: Counter[str],
        current: int,
        needs_draw: bool,
        turns: int,
        action_counts: Counter[str],
        violations: list[str],
        max_turns: int,
        deadline: float | None = None,
        skip_initial_hu: bool = False,
        initial_drawn_card: str | None = None,
    ) -> GameResult:
        coverage_failures: list[str] = []
        skip_hu_once = bool(skip_initial_hu)
        pending_drawn_card = initial_drawn_card
        while (stock or pending_drawn_card is not None) and turns < max_turns:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            player = players[current]
            skip_current_hu = skip_hu_once
            skip_hu_once = False
            resumed_drawn_card = pending_drawn_card
            pending_drawn_card = None
            drew_this_turn = needs_draw or resumed_drawn_card is not None
            auto_resolution = "discard"
            draw_auto_meld_applied = False
            draw_hu: HuEvaluation | None = None
            if drew_this_turn:
                skip_current_hu = False
                if resumed_drawn_card is not None:
                    drawn = resumed_drawn_card
                else:
                    drawn = stock.pop()
                    player.hand.append(drawn)
                    action_counts["draw"] += 1
                invariant_errors = _invariant_errors(players, stock, initial_counts)
                if invariant_errors:
                    violations.extend(invariant_errors)
                    break
                draw_hu = evaluate_hu(player.hand, player.melds, self.rules)
                if draw_hu.can_hu:
                    accepted_hu = self._choose_and_record_hu(
                        player=player,
                        hu=draw_hu,
                        players=players,
                        stock=stock,
                        initial_counts=initial_counts,
                        turns=turns,
                        phase="self_hu",
                        metadata={"drawn_card": drawn},
                    )
                    if accepted_hu:
                        action_counts[f"win:{_policy_tag(self.policies[current])}"] += 1
                        return self._result(
                            seed,
                            current,
                            turns,
                            "self_draw",
                            draw_hu,
                            action_counts,
                            violations,
                        )
                    skip_current_hu = True
                hand_count_before_auto = len(player.hand)
                meld_count_before_auto = len(player.melds)
                quad_events_before_auto = player.quad_events
                auto_resolution = self._apply_draw_auto_meld(player, drawn, action_counts)
                draw_auto_meld_applied = (
                    len(player.hand) != hand_count_before_auto
                    or len(player.melds) != meld_count_before_auto
                    or player.quad_events != quad_events_before_auto
                )
                if auto_resolution != "discard":
                    skip_current_hu = False
                invariant_errors = _invariant_errors(players, stock, initial_counts)
                if invariant_errors:
                    violations.extend(invariant_errors)
                    break

            terminal_hu = (
                draw_hu
                if draw_hu is not None and not draw_auto_meld_applied
                else evaluate_hu(player.hand, player.melds, self.rules)
            )
            if terminal_hu.can_hu and not skip_current_hu and self._choose_and_record_hu(
                player=player,
                hu=terminal_hu,
                players=players,
                stock=stock,
                initial_counts=initial_counts,
                turns=turns,
                phase="post_auto_hu" if drew_this_turn else "post_action_hu",
                metadata={"drew_this_turn": drew_this_turn},
            ):
                action_counts[f"win:{_policy_tag(self.policies[current])}"] += 1
                reason = "post_auto_meld_hu" if drew_this_turn else "post_action_hu"
                return self._result(
                    seed,
                    current,
                    turns,
                    reason,
                    terminal_hu,
                    action_counts,
                    violations,
                )
            if auto_resolution == "skip_discard":
                current = (current + 1) % self.player_count
                needs_draw = True
                continue

            if deadline is not None and time.perf_counter() >= deadline:
                break
            turns += 1
            view = self._public_view(players, current, stock, initial_counts)
            legal_discard_labels = _discardable_labels(tuple(player.hand))
            if not legal_discard_labels and draw_auto_meld_applied:
                action_counts["auto_meld_no_discard"] += 1
                current = (current + 1) % self.player_count
                needs_draw = True
                continue
            legal_discard_actions = [
                TraceAction(
                    key=f"DISCARD:{label}",
                    type="DISCARD",
                    label=label,
                )
                for label in legal_discard_labels
            ]
            state_before_hash = _simulation_state_hash(players, stock)
            try:
                discard = self.policies[current].choose_discard(view, self.rules)
            except ValueError as exc:
                meld_summary = ",".join(
                    f"{meld.kind}:{''.join(meld.cards)}"
                    for meld in player.melds
                )
                if (
                    str(exc) == "no_discardable_card"
                    and _is_complete_hand_below_min_xi(
                        player,
                        terminal_hu,
                        self.rules,
                    )
                ):
                    action_counts["complete_hand_below_min_xi"] += 1
                    return GameResult(
                        seed=seed,
                        wildcard_enabled=self.wildcard_enabled,
                        winner=(
                            1 - current
                            if self.player_count == 2 and self.wildcard_enabled
                            else None
                        ),
                        dealer=self.dealer,
                        turns=turns,
                        reason="complete_hand_below_min_xi",
                        score=0.0,
                        total_xi=terminal_hu.total_xi,
                        action_counts=dict(action_counts),
                        violations=tuple(violations),
                        coverage_failures=tuple(coverage_failures),
                        stalled_seat=current,
                    )
                if (
                    str(exc) == "no_discardable_card"
                    and self.wildcard_enabled
                    and WILD_LABEL in player.hand
                ):
                    coverage_failures.append(
                        f"seat_{current}:unsupported_chenzhou_wildcard_terminal:"
                        f"policy={_policy_tag(self.policies[current])}:"
                        f"hand={''.join(player.hand)}:"
                        f"melds={meld_summary}:"
                        f"quad_events={player.quad_events}"
                    )
                    break
                violations.append(
                    f"seat_{current}:{exc}:"
                    f"policy={_policy_tag(self.policies[current])}:"
                    f"hand={''.join(player.hand)}:"
                    f"melds={meld_summary}"
                )
                break
            if discard not in legal_discard_labels:
                self._record_decision(
                    turn=turns,
                    phase="discard",
                    seat=current,
                    view=view,
                    legal_actions=legal_discard_actions,
                    selected_key=f"DISCARD:{discard}",
                    players=players,
                    stock=stock,
                    state_before_hash=state_before_hash,
                    reason="illegal_policy_selection",
                )
                violations.append(f"seat_{current}:illegal_discard:{discard}")
                break
            player.hand.remove(discard)
            self._record_decision(
                turn=turns,
                phase="discard",
                seat=current,
                view=view,
                legal_actions=legal_discard_actions,
                selected_key=f"DISCARD:{discard}",
                players=players,
                stock=stock,
                state_before_hash=state_before_hash,
                metadata={"drew_this_turn": drew_this_turn},
            )
            action_counts["discard"] += 1
            action_counts[f"discard:{_policy_tag(self.policies[current])}"] += 1

            resolution = self._resolve_discard_response(
                seed=seed,
                players=players,
                stock=stock,
                initial_counts=initial_counts,
                discarder=current,
                discard=discard,
                turns=turns,
                action_counts=action_counts,
                violations=violations,
            )
            self._finish_pending_response_policies()
            if resolution.terminal_result is not None:
                return resolution.terminal_result
            if not resolution.continue_play:
                break
            current = resolution.current
            needs_draw = resolution.needs_draw
            skip_hu_once = resolution.skip_hu_once

        if violations:
            reason = "invariant_failure"
        elif coverage_failures:
            reason = "rollout_coverage_incomplete"
        elif deadline is not None and time.perf_counter() >= deadline:
            reason = "time_budget"
        elif not stock:
            reason = "stock_exhausted"
        elif turns >= max_turns:
            reason = "max_turns"
        else:
            reason = "invariant_failure"
        return GameResult(
            seed=seed,
            wildcard_enabled=self.wildcard_enabled,
            winner=None,
            dealer=self.dealer,
            turns=turns,
            reason=reason,
            score=0.0,
            total_xi=0,
            action_counts=dict(action_counts),
            violations=tuple(violations),
            coverage_failures=tuple(coverage_failures),
        )

    def _resolve_discard_response(
        self,
        *,
        seed: int,
        players: list[SimPlayer],
        stock: list[str],
        initial_counts: Counter[str],
        discarder: int,
        discard: str,
        turns: int,
        action_counts: Counter[str],
        violations: list[str],
        blocked_auto_claim_seats: frozenset[int] = frozenset(),
    ) -> _DiscardResponseResolution:
        next_seat = (discarder + 1) % self.player_count
        pao_seat = self._first_pao_claim(
            players,
            discarder,
            discard,
            skip_seats=blocked_auto_claim_seats,
        )
        intents = self._collect_response_intents(
            players,
            discarder=discarder,
            discard=discard,
            stock=stock,
            initial_counts=initial_counts,
            blocked_auto_claim_seats=blocked_auto_claim_seats,
            automatic_pao_pending=pao_seat is not None,
        )
        hu_intent = next(
            (
                intent
                for intent in intents
                if intent.selected_key == "HU"
            ),
            None,
        )
        if hu_intent is not None:
            winner = hu_intent.seat
            hu = hu_intent.hu
            if hu is None:
                violations.append(f"selected_hu_without_evaluation:seat={winner}")
                return _DiscardResponseResolution(
                    current=discarder,
                    needs_draw=False,
                    continue_play=False,
                    claim_type="invariant_failure",
                )
            self._record_response_intents(
                intents,
                turns=turns,
                players=players,
                stock=stock,
                discarder=discarder,
                discard=discard,
                executed_seat=winner,
                executed_key="HU",
                resolution="hu",
            )
            invariant_errors = _invariant_errors(
                players,
                stock,
                initial_counts,
                extra_cards=(discard,),
            )
            if invariant_errors:
                violations.extend(invariant_errors)
                return _DiscardResponseResolution(
                    current=discarder,
                    needs_draw=False,
                    continue_play=False,
                    claim_type="invariant_failure",
                )
            action_counts["discard_hu"] += 1
            action_counts[f"discard_hu:seat_{winner}"] += 1
            action_counts[f"win:{_policy_tag(self.policies[winner])}"] += 1
            return _DiscardResponseResolution(
                current=winner,
                needs_draw=False,
                continue_play=False,
                terminal_result=self._result(
                    seed,
                    winner,
                    turns,
                    "discard_hu",
                    hu,
                    action_counts,
                    violations,
                ),
                claim_type="hu",
                claimant=winner,
            )

        if pao_seat is not None:
            self._record_response_intents(
                intents,
                turns=turns,
                players=players,
                stock=stock,
                discarder=discarder,
                discard=discard,
                executed_seat=pao_seat,
                executed_key=f"PAO:{discard}",
                resolution="automatic_pao",
            )
            self._upgrade_to_pao(players[pao_seat], discard)
            action_counts["pao"] += 1
            action_counts[f"pao:seat_{pao_seat}"] += 1
            action_counts[f"pao:{_policy_tag(self.policies[pao_seat])}"] += 1
            players[pao_seat].quad_events += 1
            invariant_errors = _invariant_errors(players, stock, initial_counts)
            if invariant_errors:
                violations.extend(invariant_errors)
                return _DiscardResponseResolution(
                    current=pao_seat,
                    needs_draw=False,
                    continue_play=False,
                    claim_type="invariant_failure",
                    claimant=pao_seat,
                )
            no_discard_after_pao = not _discardable_labels(
                tuple(players[pao_seat].hand)
            )
            if no_discard_after_pao:
                action_counts["auto_pao_no_discard"] += 1
            if players[pao_seat].quad_events >= 2 or no_discard_after_pao:
                return _DiscardResponseResolution(
                    current=(pao_seat + 1) % self.player_count,
                    needs_draw=True,
                    continue_play=True,
                    claim_type="pao",
                    claimant=pao_seat,
                )
            return _DiscardResponseResolution(
                current=pao_seat,
                needs_draw=False,
                continue_play=True,
                claim_type="pao",
                claimant=pao_seat,
            )

        self._apply_response_pass_restrictions(
            intents,
            players,
            discard,
        )
        peng_intent = next(
            (
                intent
                for intent in intents
                if intent.selected_key == f"PENG:{discard}"
            ),
            None,
        )
        chi_intent = next(
            (
                intent
                for intent in intents
                if intent.seat == next_seat
                and intent.selected_key.startswith("CHI:")
            ),
            None,
        )
        if peng_intent is not None:
            self._record_response_intents(
                intents,
                turns=turns,
                players=players,
                stock=stock,
                discarder=discarder,
                discard=discard,
                executed_seat=peng_intent.seat,
                executed_key=peng_intent.selected_key,
                resolution="peng",
            )
            peng_seat = peng_intent.seat
            claimant = players[peng_seat]
            claimant.hand.remove(discard)
            claimant.hand.remove(discard)
            claimant.melds.append(SimMeld("peng", (discard, discard, discard)))
            action_counts["peng"] += 1
            action_counts[f"peng:seat_{peng_seat}"] += 1
            action_counts[f"peng:{_policy_tag(self.policies[peng_seat])}"] += 1
            invariant_errors = _invariant_errors(players, stock, initial_counts)
            if invariant_errors:
                violations.extend(invariant_errors)
                return _DiscardResponseResolution(
                    current=peng_seat,
                    needs_draw=False,
                    continue_play=False,
                    claim_type="invariant_failure",
                    claimant=peng_seat,
                )
            return _DiscardResponseResolution(
                current=peng_seat,
                needs_draw=False,
                continue_play=True,
                claim_type="peng",
                claimant=peng_seat,
                skip_hu_once=peng_intent.hu is not None,
            )

        if chi_intent is not None:
            chi_plan = chi_intent.chi_plan
            if chi_plan is None:
                violations.append(
                    f"selected_chi_without_plan:seat={chi_intent.seat}"
                )
                return _DiscardResponseResolution(
                    current=next_seat,
                    needs_draw=False,
                    continue_play=False,
                    claim_type="invariant_failure",
                    claimant=next_seat,
                )
            self._record_response_intents(
                intents,
                turns=turns,
                players=players,
                stock=stock,
                discarder=discarder,
                discard=discard,
                executed_seat=next_seat,
                executed_key=chi_intent.selected_key,
                resolution="chi",
            )
            claimant = players[next_seat]
            for label in chi_plan.consumed_from_hand:
                claimant.hand.remove(label)
            claimant.melds.extend(
                SimMeld(_chi_kind(group), tuple(group))
                for group in chi_plan.groups
            )
            action_counts["chi"] += 1
            action_counts[f"chi:seat_{next_seat}"] += 1
            action_counts[f"chi:{_policy_tag(self.policies[next_seat])}"] += 1
            action_counts["compare"] += len(chi_plan.compare_groups)
            invariant_errors = _invariant_errors(players, stock, initial_counts)
            if invariant_errors:
                violations.extend(invariant_errors)
                return _DiscardResponseResolution(
                    current=next_seat,
                    needs_draw=False,
                    continue_play=False,
                    claim_type="invariant_failure",
                    claimant=next_seat,
                )
            return _DiscardResponseResolution(
                current=next_seat,
                needs_draw=False,
                continue_play=True,
                claim_type="chi",
                claimant=next_seat,
                skip_hu_once=chi_intent.hu is not None,
            )

        self._record_response_intents(
            intents,
            turns=turns,
            players=players,
            stock=stock,
            discarder=discarder,
            discard=discard,
            executed_seat=None,
            executed_key="PASS",
            resolution="all_pass",
        )
        players[discarder].discards.append(discard)
        invariant_errors = _invariant_errors(players, stock, initial_counts)
        if invariant_errors:
            violations.extend(invariant_errors)
            return _DiscardResponseResolution(
                current=next_seat,
                needs_draw=True,
                continue_play=False,
                claim_type="invariant_failure",
            )
        return _DiscardResponseResolution(
            current=next_seat,
            needs_draw=True,
            continue_play=True,
        )

    def _collect_response_intents(
        self,
        players: list[SimPlayer],
        *,
        discarder: int,
        discard: str,
        stock: list[str],
        initial_counts: Counter[str],
        blocked_auto_claim_seats: frozenset[int],
        automatic_pao_pending: bool,
    ) -> tuple[_ResponseIntent, ...]:
        next_seat = (discarder + 1) % self.player_count
        state_before_hash = _simulation_state_hash(players, stock)
        intents: list[_ResponseIntent] = []
        for offset in range(1, self.player_count):
            seat = (discarder + offset) % self.player_count
            player = players[seat]
            view = self._public_view(
                players,
                seat,
                stock,
                initial_counts,
                pending_card=discard,
                pending_source_seat=discarder,
            )
            legal_actions = [TraceAction(key="PASS", type="PASS")]
            hu = None
            if seat not in blocked_auto_claim_seats:
                candidate_hu = evaluate_hu(
                    [*player.hand, discard],
                    player.melds,
                    self.rules,
                )
                if candidate_hu.can_hu:
                    hu = candidate_hu
                    legal_actions.append(
                        TraceAction(
                            key="HU",
                            type="HU",
                            label=discard,
                            priority=4,
                        )
                    )
            if not automatic_pao_pending and discard != WILD_LABEL:
                if (
                    player.hand.count(discard) == 2
                    and discard not in player.passed_peng
                    and _claim_leaves_discardable_card(
                        player.hand,
                        (discard, discard),
                    )
                ):
                    legal_actions.append(
                        TraceAction(
                            key=f"PENG:{discard}",
                            type="PENG",
                            label=discard,
                            consumed_from_hand=(discard, discard),
                            meld_groups=((discard, discard, discard),),
                            priority=2,
                        )
                    )
                if seat == next_seat and discard not in player.passed_chi:
                    plans = enumerate_chi_plans(
                        player.hand,
                        discard,
                        allow_1510=bool(
                            self.rules.get("rules", {}).get(
                                "allow_1510",
                                False,
                            )
                        ),
                    )
                    plans = [
                        plan
                        for plan in plans
                        if _claim_leaves_discardable_card(
                            player.hand,
                            plan.consumed_from_hand,
                        )
                    ]
                    legal_actions.extend(
                        _chi_trace_action(plan)
                        for plan in plans
                    )
                else:
                    plans = []
            else:
                plans = []
            if len(legal_actions) == 1:
                continue

            selected_key = "PASS"
            selected_plan = None
            policy = self.policies[seat]
            choose_response = getattr(policy, "choose_response", None)
            if callable(choose_response):
                selected_key = str(
                    choose_response(
                        view,
                        tuple(legal_actions),
                        tuple(plans),
                        hu,
                        self.rules,
                    )
                )
                legal_keys = {action.key for action in legal_actions}
                if selected_key not in legal_keys:
                    raise ValueError(
                        f"joint_response_selection_not_legal:{selected_key}"
                    )
                if selected_key.startswith("CHI:"):
                    selected_plan = next(
                        (
                            plan
                            for plan in plans
                            if _chi_trace_key(plan) == selected_key
                        ),
                        None,
                    )
                    if selected_plan is None:
                        raise ValueError("joint_response_chi_plan_not_legal")
            else:
                if hu is not None:
                    choose_hu = getattr(policy, "choose_hu", None)
                    hu_view = _view_with_added_hand_card(view, discard)
                    if (
                        bool(choose_hu(hu_view, hu, self.rules))
                        if callable(choose_hu)
                        else True
                    ):
                        selected_key = "HU"
                if selected_key == "PASS" and any(
                    action.type == "PENG"
                    for action in legal_actions
                ):
                    if policy.choose_peng(view, discard, self.rules):
                        selected_key = f"PENG:{discard}"
                if selected_key == "PASS" and plans:
                    selected = policy.choose_chi(view, plans, self.rules)
                    if selected is not None:
                        selected_plan = next(
                            (
                                plan
                                for plan in plans
                                if _chi_trace_key(plan) == _chi_trace_key(selected)
                            ),
                            None,
                        )
                        if selected_plan is None:
                            raise ValueError("response_chi_plan_not_legal")
                        selected_key = _chi_trace_key(selected_plan)
            intents.append(
                _ResponseIntent(
                    seat=seat,
                    view=view,
                    legal_actions=tuple(legal_actions),
                    selected_key=selected_key,
                    chi_plan=selected_plan,
                    hu=hu,
                    state_before_hash=state_before_hash,
                )
            )
        return tuple(intents)

    def _choose_and_record_hu(
        self,
        *,
        player: SimPlayer,
        hu: HuEvaluation,
        players: list[SimPlayer],
        stock: list[str],
        initial_counts: Counter[str],
        turns: int,
        phase: str,
        metadata: dict[str, Any],
    ) -> bool:
        view = self._public_view(
            players,
            player.seat,
            stock,
            initial_counts,
        )
        state_before_hash = _simulation_state_hash(players, stock)
        choose_hu = getattr(self.policies[player.seat], "choose_hu", None)
        accepted = (
            bool(choose_hu(view, hu, self.rules))
            if callable(choose_hu)
            else True
        )
        self._record_decision(
            turn=turns,
            phase=phase,
            seat=player.seat,
            view=view,
            legal_actions=[
                TraceAction(key="PASS", type="PASS"),
                TraceAction(key="HU", type="HU", priority=4),
            ],
            selected_key="HU" if accepted else "PASS",
            players=players,
            stock=stock,
            state_before_hash=state_before_hash,
            reason="hu_accepted" if accepted else "hu_declined",
            metadata={
                **metadata,
                "hu_total_xi": hu.total_xi,
                "hu_score": hu.score,
                "hu_red_count": hu.red_count,
            },
        )
        return accepted

    def _record_response_intents(
        self,
        intents: tuple[_ResponseIntent, ...],
        *,
        turns: int,
        players: list[SimPlayer],
        stock: list[str],
        discarder: int,
        discard: str,
        executed_seat: int | None,
        executed_key: str,
        resolution: str,
    ) -> None:
        for intent in intents:
            executed = (
                intent.selected_key == executed_key
                and (
                    executed_seat is None
                    or intent.seat == executed_seat
                )
            )
            status = "executed" if executed else "not_executed"
            if intent.selected_key == "PASS" and resolution != "all_pass":
                status = "passed_before_resolution"
            elif intent.selected_key != "PASS" and not executed:
                status = "preempted_by_priority"
            resolve = getattr(
                self.policies[intent.seat],
                "resolve_pending_response",
                None,
            )
            if callable(resolve):
                resolve(executed)
            self._record_decision(
                turn=turns,
                phase="response_root",
                seat=intent.seat,
                view=intent.view,
                legal_actions=list(intent.legal_actions),
                selected_key=intent.selected_key,
                players=players,
                stock=stock,
                state_before_hash=intent.state_before_hash,
                reason=f"response_intent_{intent.selected_key.lower().split(':', 1)[0]}",
                metadata={
                    "pending_card": discard,
                    "pending_source_seat": discarder,
                    "resolution": resolution,
                    "resolution_status": status,
                    "executed_seat": executed_seat,
                    "executed_key": executed_key,
                },
            )

    @staticmethod
    def _apply_response_pass_restrictions(
        intents: tuple[_ResponseIntent, ...],
        players: list[SimPlayer],
        discard: str,
    ) -> None:
        for intent in intents:
            action_types = {action.type for action in intent.legal_actions}
            if "PENG" in action_types and intent.selected_key != f"PENG:{discard}":
                players[intent.seat].passed_peng.add(discard)
            if "CHI" in action_types and not intent.selected_key.startswith("CHI:"):
                players[intent.seat].passed_chi.add(discard)

    def _finish_pending_response_policies(self) -> None:
        for policy in self.policies:
            finish = getattr(policy, "finish_pending_response", None)
            if callable(finish):
                finish()

    def _lay_initial_quads(self, players: list[SimPlayer], actions: Counter[str]) -> None:
        for player in players:
            changed = True
            while changed:
                changed = False
                counts = Counter(player.hand)
                for label, amount in counts.items():
                    if label == WILD_LABEL or amount < 4:
                        continue
                    _remove_in_place(player.hand, [label] * 4)
                    player.melds.append(SimMeld("ti", (label,) * 4))
                    player.quad_events += 1
                    actions["ti"] += 1
                    changed = True
                    break

    def _apply_draw_auto_meld(
        self,
        player: SimPlayer,
        drawn: str,
        actions: Counter[str],
    ) -> str:
        if drawn == WILD_LABEL:
            return "discard"
        for index, meld in enumerate(player.melds):
            if meld.kind in {"peng", "wei"} and meld.cards[0] == drawn:
                player.hand.remove(drawn)
                player.melds[index] = SimMeld("pao", (drawn,) * 4)
                player.quad_events += 1
                actions["pao"] += 1
                return "skip_discard" if player.quad_events >= 2 else "discard"
        if player.hand.count(drawn) >= 4:
            _remove_in_place(player.hand, [drawn] * 4)
            player.melds.append(SimMeld("ti", (drawn,) * 4))
            player.quad_events += 1
            actions["ti"] += 1
            return "skip_discard" if player.quad_events >= 2 else "discard"
        if player.hand.count(drawn) == 3:
            _remove_in_place(player.hand, [drawn] * 3)
            player.melds.append(SimMeld("wei", (drawn,) * 3))
            actions["wei"] += 1
        return "discard"

    def _first_pao_claim(
        self,
        players: list[SimPlayer],
        discarder: int,
        label: str,
        *,
        skip_seats: frozenset[int] = frozenset(),
    ) -> int | None:
        for offset in range(1, self.player_count):
            seat = (discarder + offset) % self.player_count
            if seat in skip_seats:
                continue
            if any(meld.kind in {"peng", "wei"} and meld.cards[0] == label for meld in players[seat].melds):
                return seat
        return None

    @staticmethod
    def _upgrade_to_pao(player: SimPlayer, label: str) -> None:
        for index, meld in enumerate(player.melds):
            if meld.kind in {"peng", "wei"} and meld.cards[0] == label:
                player.melds[index] = SimMeld("pao", (label,) * 4)
                return
        raise ValueError("pao_source_meld_missing")

    @staticmethod
    def _public_view(
        players: list[SimPlayer],
        seat: int,
        stock: list[str],
        initial_counts: Counter[str],
        *,
        pending_card: str | None = None,
        pending_source_seat: int | None = None,
    ) -> PublicView:
        visible = Counter(players[seat].hand)
        for player in players:
            visible.update(card for meld in player.melds for card in meld.cards)
            visible.update(player.discards)
        if pending_card:
            visible.update((pending_card,))
        remaining = initial_counts.copy()
        remaining.subtract(visible)
        remaining = +remaining
        return PublicView(
            seat=seat,
            hand=tuple(players[seat].hand),
            own_melds=tuple(players[seat].melds),
            all_melds=tuple(tuple(player.melds) for player in players),
            discards=tuple(tuple(player.discards) for player in players),
            remaining_counts=tuple(sorted(remaining.items())),
            stock_count=len(stock),
            hand_sizes=tuple(len(player.hand) for player in players),
            pending_card=pending_card,
            pending_source_seat=pending_source_seat,
            passed_chi=tuple(
                tuple(sorted(player.passed_chi)) for player in players
            ),
            passed_peng=tuple(
                tuple(sorted(player.passed_peng)) for player in players
            ),
        )

    def _result(
        self,
        seed: int,
        winner: int,
        turns: int,
        reason: str,
        hu: HuEvaluation,
        actions: Counter[str],
        violations: list[str],
    ) -> GameResult:
        return GameResult(
            seed=seed,
            wildcard_enabled=self.wildcard_enabled,
            winner=winner,
            dealer=self.dealer,
            turns=turns,
            reason=reason,
            score=hu.score,
            total_xi=hu.total_xi,
            action_counts=dict(actions),
            violations=tuple(violations),
        )


def evaluate_hu(
    hand: list[str] | tuple[str, ...],
    melds: list[SimMeld] | tuple[SimMeld, ...],
    rules: dict[str, Any],
) -> HuEvaluation:
    labels = list(hand)
    groups = best_grouping_normalized(
        labels,
        required_pair_count=int(any(len(meld.cards) == 4 for meld in melds)),
    )
    complete = sum(len(group) for group in groups) == len(labels)
    required = int(rules.get("rules", {}).get("required_meld_groups", 7))
    if not complete or len(groups) + len(melds) != required:
        return HuEvaluation(False, 0, (), 0.0, 0)
    hand_xi = sum(concealed_group_xi(list(group), rules=rules) for group in groups)
    exposed_xi = sum(meld_xi(list(meld.cards), kind=meld.kind, rules=rules) for meld in melds)
    total_xi = hand_xi + exposed_xi
    if total_xi < int(rules.get("rules", {}).get("min_xi", 9)):
        return HuEvaluation(False, total_xi, groups, 0.0, 0)
    all_cards = [*labels, *(card for meld in melds for card in meld.cards)]
    red_count = sum(label in RED_LABELS for label in all_cards)
    base_tun = int(rules.get("rules", {}).get("base_tun_at_9_xi", 1))
    tun = base_tun + max(0, total_xi - 9) // 3
    red_black = classify_red_black(red_count, rules, card_count=len(all_cards))
    score = float(tun + red_black.points)
    return HuEvaluation(True, total_xi, groups, score, red_count)


def _is_complete_hand_below_min_xi(
    player: SimPlayer,
    hu: HuEvaluation,
    rules: dict[str, Any],
) -> bool:
    required = int(rules.get("rules", {}).get("required_meld_groups", 7))
    min_required_xi = int(rules.get("rules", {}).get("min_xi", 9))
    return (
        not hu.can_hu
        and sum(len(group) for group in hu.groups) == len(player.hand)
        and len(hu.groups) + len(player.melds) == required
        and hu.total_xi < min_required_xi
    )


def run_benchmark(
    games: int,
    *,
    wildcard_enabled: bool,
    seed: int = 20260722,
    workers: int = 1,
    candidate_policy: str = "information_set_search",
    players: int = 3,
) -> dict[str, object]:
    if games < 1:
        raise ValueError("games must be positive")
    if players not in {2, 3}:
        raise ValueError("players must be 2 or 3")
    if candidate_policy not in {"information_set_search", "professional_brain"}:
        raise ValueError(f"unknown candidate policy: {candidate_policy}")
    results: list[GameResult] = []
    search_wins = 0
    baseline_wins = 0
    draws = 0
    jobs = [
        (
            index,
            _benchmark_assignment(index, seed, players=players)[2],
            wildcard_enabled,
            candidate_policy,
            players,
        )
        for index in range(games)
    ]
    if workers > 1:
        with get_context("spawn").Pool(processes=min(workers, games, os.cpu_count() or 1)) as pool:
            game_rows = pool.map(_benchmark_game, jobs, chunksize=1)
    else:
        game_rows = [_benchmark_game(job) for job in jobs]
    for index, result in enumerate(game_rows):
        search_seat = index % players
        results.append(result)
        if result.winner is None:
            draws += 1
        elif result.winner == search_seat:
            search_wins += 1
        else:
            baseline_wins += 1
    decisive = search_wins + baseline_wins
    rate = search_wins / decisive if decisive else 0.0
    low, high = _wilson_interval(search_wins, decisive)
    equal_policy_share = 1.0 / players
    aggregate_actions: Counter[str] = Counter()
    for result in results:
        aggregate_actions.update(result.action_counts)
    seat_dealer_matrix: dict[str, dict[str, int]] = {}
    for index, result in enumerate(results):
        search_seat = index % players
        key = f"search_{search_seat}_dealer_{result.dealer}"
        row = seat_dealer_matrix.setdefault(key, {"games": 0, "search_wins": 0, "baseline_wins": 0, "draws": 0})
        row["games"] += 1
        if result.winner is None:
            row["draws"] += 1
        elif result.winner == search_seat:
            row["search_wins"] += 1
        else:
            row["baseline_wins"] += 1
    return {
        "games": games,
        "players": players,
        "wildcard_enabled": wildcard_enabled,
        "seed": seed,
        "workers": workers,
        "candidate_policy": candidate_policy,
        "search_wins": search_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "search_win_share_decisive": round(rate, 4),
        "equal_policy_expected_share": round(equal_policy_share, 4),
        "lift_vs_equal_policy": round(rate / equal_policy_share, 4) if decisive else 0.0,
        "wilson_95": [round(low, 4), round(high, 4)],
        "mean_turns": round(mean(result.turns for result in results), 3),
        "invariant_violations": sum(len(result.violations) for result in results),
        "coverage_failures": sum(len(result.coverage_failures) for result in results),
        "mean_action_counts": {
            key: round(amount / games, 4)
            for key, amount in sorted(aggregate_actions.items())
        },
        "independent_deal_groups": math.ceil(games / players),
        "seat_dealer_matrix": seat_dealer_matrix,
        "results": [result.to_dict() for result in results],
    }


def _benchmark_game(job: tuple[int, int, bool, str, int]) -> GameResult:
    index, game_seed, wildcard_enabled, candidate_policy, players = job
    search_seat = index % players
    policies: list[SimulationPolicy] = [BaselinePolicy() for _ in range(players)]
    policies[search_seat] = (
        ProfessionalBrainSimulationPolicy()
        if candidate_policy == "professional_brain"
        else InformationSetSearchPolicy()
    )
    simulator = FullGameSimulator(
        policies,
        wildcard_enabled=wildcard_enabled,
        dealer=(index // players) % players,
        rules=rules_for_room(wildcard_enabled=wildcard_enabled, players=players),
    )
    return simulator.play(game_seed)


def _benchmark_assignment(
    index: int,
    base_seed: int,
    *,
    players: int = 3,
) -> tuple[int, int, int]:
    if players not in {2, 3}:
        raise ValueError("players must be 2 or 3")
    return (
        index % players,
        (index // players) % players,
        base_seed + index // players,
    )


def _policy_tag(policy: SimulationPolicy) -> str:
    return (
        "search"
        if isinstance(policy, (InformationSetSearchPolicy, ProfessionalBrainSimulationPolicy))
        else "baseline"
    )


def _policy_name(policy: SimulationPolicy) -> str:
    return str(getattr(policy, "name", type(policy).__name__))


def _policy_evidence(policy: SimulationPolicy) -> dict[str, Any]:
    raw = getattr(policy, "last_decision_evidence", None)
    if not isinstance(raw, dict):
        return {}
    return json.loads(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    )


def _public_view_trace_payload(view: PublicView) -> dict[str, Any]:
    return {
        "seat": view.seat,
        "hand": list(view.hand),
        "own_melds": [meld.to_dict() for meld in view.own_melds],
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
    }


def _simulation_state_hash(players: list[SimPlayer], stock: list[str]) -> str:
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


def _chi_trace_key(plan: ChiPlan) -> str:
    payload = {
        "initial_group": list(plan.initial_group),
        "compare_groups": [list(group) for group in plan.compare_groups],
        "consumed_from_hand": list(plan.consumed_from_hand),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "CHI:" + hashlib.sha256(encoded).hexdigest()[:16]


def _chi_trace_action(plan: ChiPlan) -> TraceAction:
    key = _chi_trace_key(plan)
    return TraceAction(
        key=key,
        type="CHI",
        option_id=key.removeprefix("CHI:"),
        consumed_from_hand=tuple(plan.consumed_from_hand),
        meld_groups=tuple(tuple(group) for group in plan.groups),
        priority=1,
    )


def _production_state_from_public_view(
    view: PublicView,
    *,
    legal_actions: list[dict[str, str]],
    pending_card: str | None,
    chi_options: list[dict[str, object]] | None,
    seat_aware_opponents: bool = False,
) -> dict[str, object]:
    opponent_discards = [
        label
        for seat, discards in enumerate(view.discards)
        if seat != view.seat
        for label in discards
    ]
    opponent_melds = [
        meld.to_dict()
        for seat, melds in enumerate(view.all_melds)
        if seat != view.seat
        for meld in melds
    ]
    opponent_passed_chi = [
        label
        for seat, labels in enumerate(view.passed_chi)
        if seat != view.seat
        for label in labels
    ]
    opponent_passed_peng = [
        label
        for seat, labels in enumerate(view.passed_peng)
        if seat != view.seat
        for label in labels
    ]
    memory: dict[str, object] = {
        "my_discards": list(view.discards[view.seat]),
        "opponent_discards": opponent_discards,
        "opponent_meld_groups": opponent_melds,
        "opponent_passed_chi": opponent_passed_chi,
        "opponent_passed_peng": opponent_passed_peng,
        "my_passed_chi": list(
            view.passed_chi[view.seat]
            if view.seat < len(view.passed_chi)
            else ()
        ),
        "my_passed_peng": list(
            view.passed_peng[view.seat]
            if view.seat < len(view.passed_peng)
            else ()
        ),
    }
    if seat_aware_opponents:
        player_count = len(view.discards)
        opponent_seats = sorted(
            (seat for seat in range(player_count) if seat != view.seat),
            key=lambda seat: (seat - view.seat) % player_count,
        )
        memory["seat_aware_danger"] = True
        memory["opponents"] = [
            {
                "seat": seat,
                "relative_offset": (seat - view.seat) % player_count,
                "is_next_seat": (seat - view.seat) % player_count == 1,
                "hand_size": (
                    view.hand_sizes[seat]
                    if seat < len(view.hand_sizes)
                    else None
                ),
                "discards": list(view.discards[seat]),
                "meld_groups": [meld.to_dict() for meld in view.all_melds[seat]],
                "passed_chi": list(
                    view.passed_chi[seat]
                    if seat < len(view.passed_chi)
                    else ()
                ),
                "passed_peng": list(
                    view.passed_peng[seat]
                    if seat < len(view.passed_peng)
                    else ()
                ),
            }
            for seat in opponent_seats
        ]
    state: dict[str, object] = {
        "hand": list(view.hand),
        "strategy_existing_melds": [meld.to_dict() for meld in view.own_melds],
        "legal_actions": legal_actions,
        "remaining_deck_count": view.stock_count,
        "memory": memory,
    }
    if pending_card:
        state["pending_card"] = pending_card
    if view.pending_source_seat is not None:
        state["pending_source_seat"] = view.pending_source_seat
    if chi_options is not None:
        state["chi_options"] = chi_options
    return state


def _external_label_for_plans(plans: list[ChiPlan]) -> str:
    for plan in plans:
        exposed = Counter(label for group in plan.groups for label in group)
        exposed.subtract(plan.consumed_from_hand)
        external = [label for label, amount in exposed.items() for _ in range(max(0, amount))]
        if len(external) == 1:
            return external[0]
    raise ValueError("chi_plan_external_label_missing")


def _group_multiset_key(groups: tuple[tuple[str, ...], ...] | list[list[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(tuple(sorted(group)) for group in groups))


def _position_value(
    hand: tuple[str, ...] | list[str],
    melds: tuple[SimMeld, ...] | list[SimMeld],
    remaining_counts: tuple[tuple[str, int], ...],
    rules: dict[str, Any],
    *,
    hu_cache: dict[
        tuple[tuple[str, ...], tuple[SimMeld, ...]],
        HuEvaluation,
    ] | None = None,
    existing_xi: int | None = None,
) -> float:
    hand_tuple = tuple(sorted(hand))
    meld_tuple = tuple(melds)
    waits = _hu_waits(
        hand_tuple,
        meld_tuple,
        tuple(remaining_counts),
        rules,
        hu_cache=hu_cache,
    )
    outs = sum(amount for _label, amount, _score in waits)
    weighted_score = sum(amount * score for _label, amount, score in waits)
    if existing_xi is None:
        existing_xi = sum(
            meld_xi(list(meld.cards), kind=meld.kind, rules=rules)
            for meld in melds
        )
    red_count = sum(label in RED_LABELS for label in hand)
    return (
        quick_potential(list(hand)) * 12.0
        + existing_xi * 45.0
        + outs * 90.0
        + weighted_score * 4.0
        + red_count * 3.0
    )


def _hu_waits(
    hand: tuple[str, ...],
    melds: tuple[SimMeld, ...],
    remaining_counts: tuple[tuple[str, int], ...],
    rules: dict[str, Any],
    *,
    hu_cache: dict[
        tuple[tuple[str, ...], tuple[SimMeld, ...]],
        HuEvaluation,
    ] | None = None,
) -> tuple[tuple[str, int, float], ...]:
    waits: list[tuple[str, int, float]] = []
    for label, amount in remaining_counts:
        if amount <= 0:
            continue
        hu = _evaluate_hu_canonical_with_local_cache(
            _insert_sorted_label(hand, label),
            melds,
            rules,
            hu_cache=hu_cache,
        )
        if hu.can_hu:
            waits.append((label, amount, hu.score))
    return tuple(waits)


def _evaluate_hu_with_local_cache(
    hand: tuple[str, ...],
    melds: tuple[SimMeld, ...],
    rules: dict[str, Any],
    *,
    hu_cache: dict[
        tuple[tuple[str, ...], tuple[SimMeld, ...]],
        HuEvaluation,
    ] | None,
) -> HuEvaluation:
    return _evaluate_hu_canonical_with_local_cache(
        tuple(sorted(hand)),
        melds,
        rules,
        hu_cache=hu_cache,
    )


def _evaluate_hu_canonical_with_local_cache(
    hand: tuple[str, ...],
    melds: tuple[SimMeld, ...],
    rules: dict[str, Any],
    *,
    hu_cache: dict[
        tuple[tuple[str, ...], tuple[SimMeld, ...]],
        HuEvaluation,
    ] | None,
) -> HuEvaluation:
    if hu_cache is None:
        return evaluate_hu(hand, melds, rules)
    key = (hand, melds)
    cached = hu_cache.get(key)
    if cached is None:
        cached = evaluate_hu(key[0], melds, rules)
        hu_cache[key] = cached
    return cached


def _insert_sorted_label(hand: tuple[str, ...], label: str) -> tuple[str, ...]:
    index = bisect_right(hand, label)
    return (*hand[:index], label, *hand[index:])


def _best_quick_followup_value(hand: tuple[str, ...], melds: tuple[SimMeld, ...], rules: dict[str, Any]) -> float:
    existing_xi = sum(
        meld_xi(list(meld.cards), kind=meld.kind, rules=rules)
        for meld in melds
    )
    return _best_quick_followup_value_global(
        tuple(sorted(hand)),
        existing_xi,
    )


@lru_cache(maxsize=_ROLLOUT_FEATURE_CACHE_SIZE)
def _best_quick_followup_value_global(
    hand: tuple[str, ...],
    existing_xi: int,
) -> float:
    counts = [0] * len(QUICK_POTENTIAL_LABELS)
    for card in hand:
        counts[_QUICK_POTENTIAL_INDEX[card]] += 1
    best = best_quick_potential_after_discard_from_counts(tuple(counts))
    if best is None:
        return -500.0
    return best * 12.0 + existing_xi * 45.0


def _public_discard_danger(label: str, view: PublicView) -> float:
    danger = 0.0
    for seat, melds in enumerate(view.all_melds):
        if seat == view.seat:
            continue
        if any(meld.cards[0] == label and meld.kind in {"peng", "wei"} for meld in melds):
            danger += 180.0
        danger += len(melds) * 8.0
    if label in RED_LABELS:
        danger += 12.0
    return danger


def _discardable_labels(hand: tuple[str, ...] | list[str]) -> list[str]:
    counts = Counter(hand)
    return sorted(
        label
        for label, amount in counts.items()
        if label != WILD_LABEL and amount < 3
    )


def _claim_leaves_discardable_card(
    hand: tuple[str, ...] | list[str],
    consumed_from_hand: tuple[str, ...] | list[str],
) -> bool:
    hand_after = list(hand)
    _remove_in_place(hand_after, consumed_from_hand)
    return bool(_discardable_labels(hand_after))


def _remove_one(hand: tuple[str, ...] | list[str], label: str) -> list[str]:
    result = list(hand)
    result.remove(label)
    return result


def _remove_many(hand: tuple[str, ...] | list[str], labels: tuple[str, ...] | list[str]) -> list[str]:
    result = list(hand)
    _remove_in_place(result, labels)
    return result


def _remove_in_place(hand: list[str], labels: tuple[str, ...] | list[str]) -> None:
    for label in labels:
        hand.remove(label)


def _replace_view(
    view: PublicView,
    *,
    hand: tuple[str, ...],
    own_melds: tuple[SimMeld, ...],
) -> PublicView:
    all_melds = list(view.all_melds)
    all_melds[view.seat] = own_melds
    return PublicView(
        seat=view.seat,
        hand=hand,
        own_melds=own_melds,
        all_melds=tuple(all_melds),
        discards=view.discards,
        remaining_counts=view.remaining_counts,
        stock_count=view.stock_count,
        hand_sizes=view.hand_sizes,
        pending_card=view.pending_card,
        pending_source_seat=view.pending_source_seat,
        passed_chi=view.passed_chi,
        passed_peng=view.passed_peng,
    )


def _view_with_added_hand_card(
    view: PublicView,
    label: str,
) -> PublicView:
    return PublicView(
        seat=view.seat,
        hand=(*view.hand, label),
        own_melds=view.own_melds,
        all_melds=view.all_melds,
        discards=view.discards,
        remaining_counts=view.remaining_counts,
        stock_count=view.stock_count,
        hand_sizes=view.hand_sizes,
        pending_card=view.pending_card,
        pending_source_seat=view.pending_source_seat,
        passed_chi=view.passed_chi,
        passed_peng=view.passed_peng,
    )


def _chi_kind(group: tuple[str, ...] | list[str]) -> str:
    kind = classify_meld(list(group)).kind
    return {
        "sequence": "normal_sequence",
        "mixed_same_rank": "mixed_same_rank_triplet",
    }.get(kind, kind)


def _rank_distance_penalty(label: str, hand: tuple[str, ...]) -> int:
    parsed = parse_label(label)
    if parsed is None:
        return 99
    suit, rank = parsed
    neighbours = [
        abs(rank - other_rank)
        for other in hand
        if other != label
        and (other_parsed := parse_label(other)) is not None
        for other_suit, other_rank in [other_parsed]
        if other_suit == suit
    ]
    return min(neighbours, default=9)


def _draw_shape_relevance(label: str, hand: tuple[str, ...]) -> int:
    parsed = parse_label(label)
    if parsed is None:
        return 20 if label == WILD_LABEL else 0
    suit, rank = parsed
    score = hand.count(label) * 6
    for other in hand:
        other_parsed = parse_label(other)
        if other_parsed is None or other_parsed[0] != suit:
            continue
        distance = abs(rank - other_parsed[1])
        if distance <= 2:
            score += 3 - distance
    if rank in {1, 2, 3, 7, 10}:
        score += 2
    return score


def _draw_shape_relevance_values(
    labels: tuple[str, ...],
    hand: tuple[str, ...],
) -> dict[str, int]:
    suit_counts = {
        "small": [0] * 11,
        "big": [0] * 11,
    }
    for card in hand:
        parsed = parse_label(card)
        if parsed is not None:
            suit, rank = parsed
            suit_counts[suit][rank] += 1
    values: dict[str, int] = {}
    for label in labels:
        if label == WILD_LABEL:
            values[label] = 20
            continue
        parsed = parse_label(label)
        if parsed is None:
            values[label] = 0
            continue
        suit, rank = parsed
        counts = suit_counts[suit]
        score = counts[rank] * 9
        for distance, weight in ((1, 2), (2, 1)):
            lower = rank - distance
            upper = rank + distance
            if lower >= 1:
                score += counts[lower] * weight
            if upper <= 10:
                score += counts[upper] * weight
        if rank in {1, 2, 3, 7, 10}:
            score += 2
        values[label] = score
    return values


def _invariant_errors(
    players: list[SimPlayer],
    stock: list[str],
    initial_counts: Counter[str],
    *,
    extra_cards: tuple[str, ...] = (),
) -> list[str]:
    observed = Counter(stock)
    observed.update(extra_cards)
    for player in players:
        observed.update(player.hand)
        observed.update(player.discards)
        observed.update(card for meld in player.melds for card in meld.cards)
    errors: list[str] = []
    if observed != initial_counts:
        errors.append(f"card_conservation:{dict(initial_counts - observed)}:{dict(observed - initial_counts)}")
    for player in players:
        for meld in player.melds:
            if meld.kind in {"peng", "wei", "pao", "ti"}:
                expected_length = 4 if meld.kind in {"pao", "ti"} else 3
                if len(meld.cards) != expected_length or len(set(meld.cards)) != 1 or WILD_LABEL in meld.cards:
                    errors.append(f"seat_{player.seat}:invalid_{meld.kind}:{meld.cards}")
            elif classify_meld(list(meld.cards)).kind == "unknown":
                errors.append(f"seat_{player.seat}:invalid_chi:{meld.cards}")
    return errors


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


__all__ = [
    "BaselinePolicy",
    "FullGameSimulator",
    "GameResult",
    "HuEvaluation",
    "InformationSetSearchPolicy",
    "ProfessionalBrainSimulationPolicy",
    "PublicView",
    "SimMeld",
    "SimPlayer",
    "evaluate_hu",
    "run_benchmark",
]
