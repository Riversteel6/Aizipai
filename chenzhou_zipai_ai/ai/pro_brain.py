"""Unified local strategy brain for Chenzhou Zipai decisions.

This module is intentionally local-only: it builds one DecisionContext, allocates
CardInstance resources once, scores legal actions from that shared context, and
lets ConflictGuard stop execution when policy and action planning disagree.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import combinations, product
from typing import Any, Callable

from ai.opponent_model import infer_opponent_profile, infer_opponent_profiles
from ai.risk_model import (
    danger_score as legacy_danger_score,
    opponent_meld_risk,
    opponent_meld_risks_by_seat,
    visible_labels_from_memory,
)
from engine.cards import (
    BIG_LABELS,
    RED_LABELS,
    SMALL_LABELS,
    WILD_LABEL,
    CardInstance,
    card_instance,
    label_for,
    normalize_card_label,
    normalize_cards,
    parse_label,
)
from engine.chi_rules import enumerate_chi_plans
from engine.deck import full_deck_counts
from engine.hu_checker import best_grouping, best_grouping_normalized, concealed_group_xi
from engine.melds import classify_meld
from engine.red_black_rules import classify_red_black, red_black_target_distance
from engine.rules import RuleConfig, load_rules, normalize_rules


POLICY_VERSION = "v2.2.0"
HARD_BREAK_EV = -99999.0
SOFT_TYPES = {
    "normal_sequence",
    "special_123",
    "special_2710",
    "mixed_same_rank_triplet",
    "pair",
    "wildcard_meld",
}
AUTO_MELD_ACTIONS = {"PAO", "TI", "MING_LONG", "LONG", "AUTO_QUAD", "WEI"}
STANDARD_SAFE_HALT = {
    "recognition_uncertain",
    "no_clickable_cards",
    "selected_card_not_clickable",
    "blocked_discard_hard_protected",
    "only_hard_protected_clickable_but_unprotected_cards_exist_in_hand",
    "button_not_found",
    "option_not_clear",
    "sanity_check_failed",
    "opponent_priority_pending",
    "wait_auto_meld",
    "screen_not_stable",
    "action_plan_policy_mismatch",
    "duplicate_card_allocation",
    "chi_ev_not_enough",
    "peng_ev_not_enough",
    "hu_xi_not_enough",
    "unknown_flow_state",
    "unknown_error",
}
_TING_HU_PROBE_CACHE: ContextVar[
    dict[tuple[object, ...], tuple[bool, int, float]] | None
] = ContextVar("ting_hu_probe_cache", default=None)
_TING_DRAW_FEATURE_CACHE: ContextVar[
    dict[tuple[object, ...], tuple[int, int, int, tuple[bool, int, float] | None]] | None
] = ContextVar("ting_draw_feature_cache", default=None)
_FOLLOWUP_ANALYSIS_CACHE: ContextVar[dict[str, dict[str, Any]] | None] = ContextVar(
    "followup_analysis_cache",
    default=None,
)


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _counter_from_instances(cards: list[CardInstance]) -> Counter[str]:
    return Counter(card.label for card in cards)


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _card_sort_key(card: CardInstance) -> tuple[int, str, str]:
    if card.is_wildcard:
        return (99, card.label, card.id)
    return (card.rank or 99, card.size or "", card.id)


def _meld_rank_labels(labels: list[str]) -> list[int]:
    ranks: list[int] = []
    for label in labels:
        parsed = parse_label(label)
        if parsed is not None:
            ranks.append(parsed[1])
    return ranks


def _suit_for_labels(labels: list[str]) -> str:
    for label in labels:
        parsed = parse_label(label)
        if parsed is not None:
            return parsed[0]
    return "small"


def _meld_xi_value(meld_type: str, labels: list[str], rules: dict[str, Any]) -> int:
    xi = rules.get("xi", {})
    suit = _suit_for_labels(labels)
    if meld_type in {"exact_quad", "ti", "hidden_quad"}:
        return int(xi.get("ti", {}).get(suit, 0))
    if meld_type in {"pao", "quad"}:
        return int(xi.get("pao", {}).get(suit, 0))
    if meld_type in {"exact_triplet", "wei", "hidden_triplet"}:
        return int(xi.get("wei", {}).get(suit, 0))
    if meld_type == "peng":
        return int(xi.get("peng", {}).get(suit, 0))
    if meld_type == "special_123":
        return int(xi.get("special_123", {}).get(suit, 0))
    if meld_type == "special_2710":
        return int(xi.get("special_2710", {}).get(suit, 0))
    return 0


def _structure_value(meld_type: str, labels: list[str], rules: dict[str, Any]) -> float:
    weights = rules.get("weights") or rules.get("ai_weights", {})
    value_by_type = {
        "exact_quad": weights.get("exact_quad_break_penalty", 5000),
        "exact_triplet": weights.get("exact_triplet_break_penalty", 3000),
        "ti": weights.get("exact_quad_break_penalty", 5000),
        "pao": weights.get("exact_quad_break_penalty", 5000),
        "wei": weights.get("exact_triplet_break_penalty", 3000),
        "peng": weights.get("peng_meld_value", 120),
        "double_normal_sequence": weights.get("double_sequence_break_penalty", 1200),
        "special_2710": weights.get("special_2710_break_penalty", 1200),
        "special_123": weights.get("special_123_break_penalty", 1000),
        "mixed_same_rank_triplet": weights.get("mixed_same_rank_break_penalty", 1000),
        "normal_sequence": weights.get("normal_sequence_break_penalty", 500),
        "pair": weights.get("pair_break_penalty", 300),
        "wildcard_meld": weights.get("wildcard_keep", 500),
        "weak_wait": weights.get("weak_potential_discard_bonus", 120),
    }
    red_bonus = sum(1 for label in labels if label in RED_LABELS) * 20
    return float(value_by_type.get(meld_type, 0)) + red_bonus


def _claim_structure_value(meld_type: str, rules: dict[str, Any]) -> float:
    weights = rules.get("weights") or rules.get("ai_weights", {})
    value_by_type = {
        "peng": weights.get("peng_meld_value", 120),
        "normal_sequence": weights.get("chi_normal_sequence_value", 220),
        "special_2710": weights.get("chi_special_2710_value", 320),
        "special_123": weights.get("chi_special_123_value", 280),
        "mixed_same_rank_triplet": weights.get("chi_mixed_same_rank_value", 220),
        "exact_triplet": weights.get("chi_exact_triplet_value", 220),
    }
    return float(value_by_type.get(meld_type, 0))


@dataclass(frozen=True)
class ProtectedMeld:
    id: str
    type: str
    card_ids: list[str]
    labels: list[str]
    ranks: list[int]
    protect_level: str
    xi_value: int
    red_black_value: float
    structure_value: float
    reason: str
    used_counts: dict[str, int]
    waiting_for: str | None = None

    @property
    def cards(self) -> list[str]:
        return list(self.labels)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "meld_id": self.id,
            "id": self.id,
            "type": self.type,
            "cards": list(self.labels),
            "labels": list(self.labels),
            "card_ids": list(self.card_ids),
            "ranks": list(self.ranks),
            "protect_level": self.protect_level,
            "xi_value": self.xi_value,
            "red_black_value": self.red_black_value,
            "structure_value": self.structure_value,
            "reason": self.reason,
            "used_counts": dict(self.used_counts),
        }
        if self.waiting_for is not None:
            payload["waiting_for"] = self.waiting_for
        return payload


@dataclass(frozen=True)
class StructureAllocation:
    normalized_hand: list[str]
    card_instances: list[CardInstance]
    total_counts: dict[str, int]
    locked_melds: list[ProtectedMeld]
    locked_counts: dict[str, int]
    free_counts_after_locked: dict[str, int]
    soft_melds: list[ProtectedMeld]
    potential_melds: list[ProtectedMeld]
    weak_potentials: list[ProtectedMeld]
    orphan_cards: list[str]
    protected_melds: list[ProtectedMeld]
    hard_protected_instances: list[str]
    soft_protected_instances: list[str]
    hard_protected_labels: list[str]
    soft_protected_labels: list[str]
    free_discard_instances: list[CardInstance]
    allocation_candidates: list[dict[str, Any]]
    chosen_allocation: list[str]
    allocation_notes: list[str]
    allocation_conflicts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_hand": list(self.normalized_hand),
            "card_instances": [card.to_dict() for card in self.card_instances],
            "total_counts": dict(self.total_counts),
            "locked_melds": [meld.to_dict() for meld in self.locked_melds],
            "locked_counts": dict(self.locked_counts),
            "free_counts_after_locked": dict(self.free_counts_after_locked),
            "soft_melds": [meld.to_dict() for meld in self.soft_melds],
            "potential_melds": [meld.to_dict() for meld in self.potential_melds],
            "weak_potentials": [meld.to_dict() for meld in self.weak_potentials],
            "orphan_cards": list(self.orphan_cards),
            "protected_melds": [meld.to_dict() for meld in self.protected_melds],
            "hard_protected_instances": list(self.hard_protected_instances),
            "soft_protected_instances": list(self.soft_protected_instances),
            "hard_protected_card_ids": list(self.hard_protected_instances),
            "soft_protected_card_ids": list(self.soft_protected_instances),
            "hard_protected_labels": list(self.hard_protected_labels),
            "soft_protected_labels": list(self.soft_protected_labels),
            "free_discard_instances": [card.to_dict() for card in self.free_discard_instances],
            "free_discard_card_ids": [card.id for card in self.free_discard_instances],
            "weak_potential_cards": [label for meld in self.weak_potentials for label in meld.labels],
            "allocation_candidates": list(self.allocation_candidates),
            "chosen_allocation": list(self.chosen_allocation),
            "allocation_notes": list(self.allocation_notes),
            "allocation_conflicts": list(self.allocation_conflicts),
        }


@dataclass(frozen=True)
class DecisionContext:
    context_id: str
    state: dict[str, Any]
    raw_hand: list[str]
    normalized_hand: list[str]
    card_instances: list[CardInstance]
    rules: dict[str, Any]
    legal_action_types: list[str]
    buttons: list[dict[str, Any]]
    chi_options: list[dict[str, Any]]
    compare_options: list[dict[str, Any]]
    existing_melds: list[ProtectedMeld]
    recognition_warnings: list[str]
    recognition_errors: list[str]
    passed_chi: tuple[str, ...] = ()
    passed_peng: tuple[str, ...] = ()
    phase: str = "unknown"
    frame_id: str | None = None
    remaining_deck_count: int | None = None

    @property
    def state_fingerprint(self) -> str:
        return _context_state_fingerprint(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "frame_id": self.frame_id,
            "state_fingerprint": self.state_fingerprint,
            "phase": self.phase,
            "raw_hand": list(self.raw_hand),
            "normalized_hand": list(self.normalized_hand),
            "card_instances": [card.to_dict() for card in self.card_instances],
            "legal_action_types": list(self.legal_action_types),
            "buttons": deepcopy(self.buttons),
            "chi_options": deepcopy(self.chi_options),
            "compare_options": deepcopy(self.compare_options),
            "existing_melds": [meld.to_dict() for meld in self.existing_melds],
            "recognition_warnings": list(self.recognition_warnings),
            "recognition_errors": list(self.recognition_errors),
            "passed_chi": list(self.passed_chi),
            "passed_peng": list(self.passed_peng),
            "remaining_deck_count": self.remaining_deck_count,
        }


def _context_state_fingerprint(context: DecisionContext) -> str:
    pending_label = (
        context.state.get("pending_card")
        or context.state.get("external_card")
        or context.state.get("drawn_card")
    )
    payload = {
        "context_id": context.context_id,
        "frame_id": context.frame_id,
        "phase": context.phase,
        "raw_hand": list(context.raw_hand),
        "normalized_hand": list(context.normalized_hand),
        "card_instances": [
            {
                "card_id": card.id,
                "label": card.label,
                "x": card.x,
                "y": card.y,
                "confidence": card.confidence,
                "clickable": card.clickable,
            }
            for card in context.card_instances
        ],
        "legal_action_types": list(context.legal_action_types),
        "buttons": deepcopy(context.buttons),
        "chi_options": deepcopy(context.chi_options),
        "compare_options": deepcopy(context.compare_options),
        "existing_melds": [meld.to_dict() for meld in context.existing_melds],
        "pending_card": normalize_card_label(pending_label) if pending_label else None,
        "remaining_deck_count": context.remaining_deck_count,
        "room_players": int(context.rules.get("game", {}).get("players", 3)),
        "recognition_warnings": list(context.recognition_warnings),
        "recognition_errors": list(context.recognition_errors),
        "passed_chi": list(context.passed_chi),
        "passed_peng": list(context.passed_peng),
    }
    return _stable_hash(payload)


@dataclass(frozen=True)
class XiResult:
    total_xi: int
    details: list[dict[str, Any]]
    is_enough_to_hu: bool
    min_xi: int
    xi_gap: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_xi": self.total_xi,
            "details": list(self.details),
            "is_enough_to_hu": self.is_enough_to_hu,
            "min_xi": self.min_xi,
            "xi_gap": self.xi_gap,
        }


@dataclass(frozen=True)
class XiPotential:
    confirmed_xi: int
    potential_xi: int
    best_xi_route: list[dict[str, Any]]
    xi_sources: list[dict[str, Any]]
    missing_xi_sources: list[dict[str, Any]]
    cards_that_improve_xi: list[str]
    cards_that_reduce_xi: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed_xi": self.confirmed_xi,
            "potential_xi": self.potential_xi,
            "best_xi_route": list(self.best_xi_route),
            "xi_sources": list(self.xi_sources),
            "missing_xi_sources": list(self.missing_xi_sources),
            "cards_that_improve_xi": list(self.cards_that_improve_xi),
            "cards_that_reduce_xi": list(self.cards_that_reduce_xi),
        }


@dataclass(frozen=True)
class HuResult:
    can_hu: bool
    total_xi: int
    min_xi: int
    partition: list[dict[str, Any]]
    wildcard_mapping: dict[str, str]
    red_black_bonus: float
    reason: str
    reject_reason: str | None = None
    score_now: float = 0.0
    continue_ev: float = 0.0
    decision: str = "reject"

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_hu": self.can_hu,
            "total_xi": self.total_xi,
            "min_xi": self.min_xi,
            "partition": list(self.partition),
            "wildcard_mapping": dict(self.wildcard_mapping),
            "red_black_bonus": self.red_black_bonus,
            "reason": self.reason,
            "reject_reason": self.reject_reason,
            "score_now": self.score_now,
            "continue_ev": self.continue_ev,
            "decision": self.decision,
        }


@dataclass(frozen=True)
class WildcardValue:
    wildcard_count: int
    can_complete_hu: bool
    can_complete_xi: bool
    can_complete_2710: bool
    can_complete_123: bool
    can_complete_triplet: bool
    can_complete_red_black: bool
    best_usage: str | None
    future_flexibility_score: float
    discard_loss: float
    usage_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class RedBlackPlan:
    red_count: int
    black_count: int
    red_potential: float
    black_potential: float
    mode: str
    core_cards: list[str]
    cards_to_keep: list[str]
    cards_to_release: list[str]
    bonus_estimate: float
    risk_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class TingEstimate:
    shanten_like_distance: int
    is_ting: bool
    waiting_cards: list[str]
    improving_cards: list[str]
    hu_cards: list[str]
    expected_xi_if_hu: int
    expected_score_if_hu: float
    notes: list[str]
    hu_xi_by_label: dict[str, int] = field(default_factory=dict)
    hu_score_by_label: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["distance"] = self.shanten_like_distance
        return payload


@dataclass(frozen=True)
class HandAnalysis:
    confirmed_xi: int
    potential_xi: int
    min_xi: int
    xi_gap: int
    complete_melds: list[ProtectedMeld]
    protected_melds: list[ProtectedMeld]
    potential_melds: list[ProtectedMeld]
    weak_potentials: list[ProtectedMeld]
    orphan_cards: list[str]
    red_count: int
    black_count: int
    wildcard_count: int
    red_black_plan: RedBlackPlan
    wildcard_value: WildcardValue
    ting_estimate: TingEstimate
    hand_direction: str
    attack_defense_mode: str
    hand_value: float
    hu_result: HuResult
    xi_result: XiResult
    xi_potential: XiPotential
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed_xi": self.confirmed_xi,
            "potential_xi": self.potential_xi,
            "min_xi": self.min_xi,
            "xi_gap": self.xi_gap,
            "complete_melds": [meld.to_dict() for meld in self.complete_melds],
            "protected_melds": [meld.to_dict() for meld in self.protected_melds],
            "potential_melds": [meld.to_dict() for meld in self.potential_melds],
            "weak_potentials": [meld.to_dict() for meld in self.weak_potentials],
            "orphan_cards": list(self.orphan_cards),
            "red_count": self.red_count,
            "black_count": self.black_count,
            "wildcard_count": self.wildcard_count,
            "red_black_plan": self.red_black_plan.to_dict(),
            "wildcard_value": self.wildcard_value.to_dict(),
            "ting_estimate": self.ting_estimate.to_dict(),
            "hand_direction": self.hand_direction,
            "attack_defense_mode": self.attack_defense_mode,
            "hand_value": self.hand_value,
            "hu_result": self.hu_result.to_dict(),
            "xi_result": self.xi_result.to_dict(),
            "xi_potential": self.xi_potential.to_dict(),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DangerScore:
    danger_score: float
    risk_type: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class LegalAction:
    action_id: str
    type: str
    allowed: bool = True
    source: str = "policy"
    card_id: str | None = None
    label: str | None = None
    option_id: str | None = None
    option_cards: list[str] = field(default_factory=list)
    requires_button: bool = False
    requires_option: bool = False
    requires_followup_discard: bool = False
    uncertainty: float = 0.0
    reject_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "type": self.type,
            "allowed": self.allowed,
            "source": self.source,
            "card_id": self.card_id,
            "label": self.label,
            "option_id": self.option_id,
            "option_cards": list(self.option_cards),
            "requires_button": self.requires_button,
            "requires_option": self.requires_option,
            "requires_followup_discard": self.requires_followup_discard,
            "uncertainty": self.uncertainty,
            "allowed_by_rules": self.allowed,
            "reject_reason": self.reject_reason,
        }


@dataclass(frozen=True)
class SimulatedResult:
    action: LegalAction
    hand_after: list[str]
    allocation_after: StructureAllocation | None
    consumed_card_ids: list[str]
    breaks_hard: bool
    breaks_soft: bool
    followup_discard: dict[str, Any] | None
    notes: list[str]
    debug_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "hand_after": list(self.hand_after),
            "allocation_after": self.allocation_after.to_dict() if self.allocation_after else None,
            "consumed_card_ids": list(self.consumed_card_ids),
            "breaks_hard": self.breaks_hard,
            "breaks_soft": self.breaks_soft,
            "followup_discard": self.followup_discard,
            "notes": list(self.notes),
            "debug_details": deepcopy(self.debug_details),
        }


@dataclass(frozen=True)
class ActionEval:
    action: LegalAction
    allowed: bool
    ev: float
    score_gain: float
    hand_value_after: float
    xi_gain: float
    ting_gain: float
    red_black_gain: float
    wildcard_gain: float
    structure_loss: float
    danger_loss: float
    opponent_gain_risk: float
    uncertainty_penalty: float
    reject_reason: str | None
    reason: str
    debug_details: dict[str, Any]
    forced_followup_loss: float = 0.0
    safety_gain: float = 0.0
    immediate_score_gain: float = 0.0

    @property
    def label(self) -> str | None:
        return self.action.label

    @property
    def card_id(self) -> str | None:
        return self.action.card_id

    @property
    def type(self) -> str:
        return self.action.type

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **self.action.to_dict(),
            "action": self.action.to_dict(),
            "allowed": self.allowed,
            "ev": self.ev,
            "score": self.ev,
            "score_gain": self.score_gain,
            "hand_value_after": self.hand_value_after,
            "xi_gain": self.xi_gain,
            "ting_gain": self.ting_gain,
            "red_black_gain": self.red_black_gain,
            "wildcard_gain": self.wildcard_gain,
            "structure_loss": self.structure_loss,
            "danger_loss": self.danger_loss,
            "opponent_gain_risk": self.opponent_gain_risk,
            "uncertainty_penalty": self.uncertainty_penalty,
            "forced_followup_loss": self.forced_followup_loss,
            "reject_reason": self.reject_reason,
            "reason": self.reason,
            "reasons": [self.reason] if self.reason else [],
            "penalties": [self.reject_reason] if self.reject_reason else [],
            "after_xi_potential": self.xi_gain,
            "danger": self.danger_loss,
            "breaks_melds": self.debug_details.get("breaks_melds", []),
            "danger_reasons": self.debug_details.get("danger", {}).get("reasons", []),
            "score_breakdown": {
                "hand_value_after": self.hand_value_after,
                "xi_gain": self.xi_gain,
                "ting_gain": self.ting_gain,
                "ting_value_delta": self.debug_details.get("ting_value_delta", 0),
                "red_black_gain": self.red_black_gain,
                "wildcard_gain": self.wildcard_gain,
                "immediate_score_gain": self.immediate_score_gain,
                "safety_gain": self.safety_gain,
                "structure_loss": self.structure_loss,
                "danger_loss": self.danger_loss,
                "opponent_gain_risk": self.opponent_gain_risk,
                "uncertainty_penalty": self.uncertainty_penalty,
                "forced_followup_loss": self.forced_followup_loss,
            },
            "debug_details": deepcopy(self.debug_details),
        }
        if self.action.type == "EXPAND_CHI_OPTIONS":
            debug = payload["debug_details"]
            payload["pass_ev"] = debug.get("pass_ev")
            payload["chi_ev"] = debug.get("chi_ev", debug.get("response_ev"))
            payload["ev_delta_vs_pass"] = debug.get("ev_delta_vs_pass")
            payload["min_ev_gain"] = debug.get("min_ev_gain")
            payload["consumed_card_ids"] = list(debug.get("consumed_card_ids", []))
            payload["consumed_from_hand"] = list(debug.get("consumed_from_hand", []))
            payload["followup_discard"] = debug.get("followup_discard")
        if self.action.type in {"CHI", "PENG"}:
            debug = payload["debug_details"]
            response_key = "chi_ev" if self.action.type == "CHI" else "peng_ev"
            response_ev = debug.get(response_key, debug.get("response_ev"))
            if response_ev is None:
                response_ev = self.ev
            pass_ev = debug.get("pass_ev")
            payload["pass_ev"] = pass_ev
            payload[response_key] = response_ev
            payload["ev_delta_vs_pass"] = debug.get("ev_delta_vs_pass")
            payload["min_ev_gain"] = debug.get("min_ev_gain")
            payload["consumed_card_ids"] = list(debug.get("consumed_card_ids", []))
            payload["consumed_from_hand"] = list(debug.get("consumed_from_hand", []))
            payload["followup_discard"] = debug.get("followup_discard")
            breakdown = payload["score_breakdown"]
            breakdown[f"{self.action.type.lower()}_meld_value"] = self.score_gain
            breakdown["consumed_structure_cost"] = self.structure_loss
            breakdown["followup_discard_cost"] = self.forced_followup_loss
            breakdown["exposure_cost"] = self.opponent_gain_risk
        return payload


@dataclass(frozen=True)
class PolicyDecision:
    selected_action: str
    selected_card_id: str | None
    selected_label: str | None
    selected_option_id: str | None
    candidate_stage: str
    ev: float
    reason: str
    action_evals: list[ActionEval]
    context_snapshot: dict[str, Any]
    requires_action_plan: bool
    safety_flags: list[str]
    policy_version: str = POLICY_VERSION

    @property
    def action(self) -> str:
        return self.selected_action.lower()

    @property
    def label(self) -> str | None:
        return self.selected_label

    @property
    def selected_reason(self) -> str:
        return self.reason

    @property
    def evaluations(self) -> list[ActionEval]:
        return self.action_evals

    @property
    def response_evaluations(self) -> list[dict[str, Any]]:
        return [
            item.to_dict()
            for item in self.action_evals
            if item.action.type in {"CHI", "EXPAND_CHI_OPTIONS", "PENG", "PASS", "HU"}
        ]

    @property
    def hard_protected(self) -> list[str]:
        return list(self.context_snapshot.get("structure_allocation", {}).get("hard_protected_labels", []))

    @property
    def soft_protected(self) -> list[str]:
        return list(self.context_snapshot.get("structure_allocation", {}).get("soft_protected_labels", []))

    def to_dict(self) -> dict[str, Any]:
        selected = {
            "type": self.selected_action,
            "card_id": self.selected_card_id,
            "label": self.selected_label,
            "option_id": self.selected_option_id,
            "ev": self.ev,
        }
        context = self.context_snapshot.get("context", {})
        allocation = self.context_snapshot.get("structure_allocation", {})
        analysis = self.context_snapshot.get("hand_analysis", {})
        selected_eval = next(
            (
                item
                for item in self.action_evals
                if item.action.type == self.selected_action
                and item.action.card_id == self.selected_card_id
                and item.action.label == self.selected_label
                and item.action.option_id == self.selected_option_id
            ),
            None,
        )
        selected_option_cards = list(selected_eval.action.option_cards) if selected_eval else []
        selected["option_cards"] = selected_option_cards
        hard_labels = allocation.get("hard_protected_labels", [])
        soft_labels = allocation.get("soft_protected_labels", [])
        free_cards = allocation.get("free_discard_instances", [])
        weak_ids = {
            card_id
            for meld in allocation.get("weak_potentials", [])
            for card_id in meld.get("card_ids", [])
        }
        free_ids = {card.get("card_id") for card in free_cards if isinstance(card, dict)}
        soft_ids = set(allocation.get("soft_protected_card_ids", []))
        hard_ids = set(allocation.get("hard_protected_card_ids", []))
        serialized_evals = [item.to_dict() for item in self.action_evals]
        serialized_response_evals = [
            item
            for item in serialized_evals
            if item.get("type") in {"CHI", "EXPAND_CHI_OPTIONS", "PENG", "PASS", "HU"}
        ]
        return {
            "selected_action": selected,
            "action": self.action,
            "policy_action": self.action,
            "selected_card_id": self.selected_card_id,
            "selected_label": self.selected_label,
            "selected_option_id": self.selected_option_id,
            "selected_option_cards": selected_option_cards,
            "option_cards": selected_option_cards,
            "label": self.selected_label,
            "selected_discard": self.selected_label if self.selected_action == "DISCARD" else None,
            "candidate_stage": self.candidate_stage,
            "ev": self.ev,
            "score": self.ev,
            "reason": self.reason,
            "selected_reason": self.reason,
            "action_evals": list(serialized_evals),
            "evaluations": list(serialized_evals),
            "response_evaluations": list(serialized_response_evals),
            "context_snapshot": deepcopy(self.context_snapshot),
            "requires_action_plan": self.requires_action_plan,
            "safety_flags": list(self.safety_flags),
            "policy_version": self.policy_version,
            "raw_hand": list(context.get("raw_hand", [])),
            "normalized_hand": list(context.get("normalized_hand", [])),
            "counts": dict(allocation.get("total_counts", {})),
            "orphan_cards": list(allocation.get("orphan_cards", [])),
            "low_value_singles": [
                card.get("label")
                for card in free_cards
                if isinstance(card, dict) and card.get("label")
            ],
            "clickable_cards": [
                card.get("label")
                for card in context.get("card_instances", [])
                if isinstance(card, dict) and card.get("clickable", True)
            ],
            "discard_candidates_before_filter": [
                item.action.label
                for item in self.action_evals
                if item.action.type == "DISCARD" and item.action.label
            ],
            "discard_candidates_after_hard_filter": [
                item.action.label
                for item in self.action_evals
                if item.action.type == "DISCARD"
                and item.action.card_id not in hard_ids
                and item.action.label
            ],
            "discard_candidates_after_soft_filter": [
                item.action.label
                for item in self.action_evals
                if item.action.type == "DISCARD"
                and item.action.card_id in free_ids
                and item.action.card_id not in weak_ids
                and item.action.card_id not in soft_ids
                and item.action.card_id not in hard_ids
                and item.action.label
            ],
            "scores": {
                item.action.label: item.ev
                for item in self.action_evals
                if item.action.type == "DISCARD" and item.action.label
            },
            "hard_protected": list(hard_labels),
            "soft_protected": list(soft_labels),
            "protected_melds": list(allocation.get("protected_melds", [])),
            "break_hard_protection": self.candidate_stage == "forced_break_hard_protection",
            "break_reason": "all_clickable_cards_are_hard_protected"
            if self.candidate_stage == "forced_break_hard_protection"
            else None,
            "hu_breakdown": analysis.get("hu_result"),
            "selected_action_eval": selected_eval.to_dict() if selected_eval else None,
        }


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    safe_halt: bool
    reason: str | None
    checks: list[str]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "safe_halt": self.safe_halt,
            "reason": self.reason,
            "checks": list(self.checks),
            "details": deepcopy(self.details),
        }


def _meld_labels_from_observation(group: Any) -> tuple[list[str], str | None]:
    type_hint: str | None = None
    raw_group = group
    if isinstance(group, dict):
        type_hint = str(group.get("type") or group.get("kind") or "") or None
        raw_group = group.get("labels") or group.get("cards") or group.get("cells") or []
    labels: list[str] = []
    for item in raw_group if isinstance(raw_group, list) else []:
        raw_label = item.get("name") or item.get("label") if isinstance(item, dict) else item
        text = str(raw_label or "").strip()
        if text == "暗":
            labels.append(text)
            continue
        normalized = normalize_card_label(text)
        if normalized:
            labels.append(normalized)
    return labels, type_hint


def _existing_meld_kind(labels: list[str], type_hint: str | None) -> tuple[str, list[str]]:
    normalized_hint = str(type_hint or "").lower()
    visible = [label for label in labels if label != "暗"]
    hidden_count = len(labels) - len(visible)
    if normalized_hint in {
        "peng",
        "wei",
        "pao",
        "ti",
        "hidden_triplet",
        "hidden_quad",
        "special_123",
        "special_2710",
        "normal_sequence",
        "mixed_same_rank_triplet",
        "unknown",
    }:
        kind = {
            "hidden_triplet": "wei",
            "hidden_quad": "ti",
        }.get(normalized_hint, normalized_hint)
    elif hidden_count and len(labels) >= 4:
        kind = "ti"
    elif hidden_count and len(labels) == 3 and len(visible) == 1:
        kind = "wei"
    elif len(labels) == 4 and visible and len(set(visible)) == 1:
        kind = "pao"
    elif len(labels) == 3 and visible and len(set(visible)) == 1:
        kind = "peng"
    else:
        pattern = classify_meld(visible)
        kind = {
            "sequence": "normal_sequence",
            "mixed_same_rank": "mixed_same_rank_triplet",
        }.get(pattern.kind, pattern.kind)
    if hidden_count and len(set(visible)) == 1:
        scoring_labels = [visible[0]] * len(labels)
    else:
        scoring_labels = visible
    return kind, scoring_labels


def _existing_melds_from_state(state: dict[str, Any], rules: dict[str, Any]) -> list[ProtectedMeld]:
    observed_sources: list[list[Any]] = []
    strategy_melds = state.get("strategy_existing_melds")
    if isinstance(strategy_melds, list):
        observed_sources.append(strategy_melds)
    meld_groups = state.get("meld_groups")
    if isinstance(meld_groups, dict):
        my_melds = meld_groups.get("my_melds")
        if isinstance(my_melds, list):
            observed_sources.append(my_melds)
    memory = state.get("memory")
    if isinstance(memory, dict):
        remembered_melds = memory.get("my_meld_groups")
        if isinstance(remembered_melds, list):
            observed_sources.append(remembered_melds)

    result: list[ProtectedMeld] = []
    merged_multiplicity: Counter[tuple[str, tuple[str, ...]]] = Counter()
    for source in observed_sources:
        source_multiplicity: Counter[tuple[str, tuple[str, ...]]] = Counter()
        for group in source:
            labels, type_hint = _meld_labels_from_observation(group)
            if len(labels) < 3:
                continue
            kind, scoring_labels = _existing_meld_kind(labels, type_hint)
            key = (kind, tuple(sorted(labels)))
            source_multiplicity[key] += 1
            if source_multiplicity[key] <= merged_multiplicity[key]:
                continue
            index = len(result) + 1
            meld_id = f"e{index:03d}"
            result.append(
                ProtectedMeld(
                    id=meld_id,
                    type=kind,
                    card_ids=[f"{meld_id}_{cell_index}" for cell_index in range(1, len(labels) + 1)],
                    labels=scoring_labels,
                    ranks=_meld_rank_labels(scoring_labels),
                    protect_level="hard",
                    xi_value=_meld_xi_value(kind, scoring_labels, rules),
                    red_black_value=float(sum(1 for label in scoring_labels if label in RED_LABELS)),
                    structure_value=_structure_value(kind, scoring_labels, rules),
                    reason=f"existing_meld:{kind}",
                    used_counts=dict(Counter(scoring_labels)),
                )
            )
        for key, count in source_multiplicity.items():
            merged_multiplicity[key] = max(merged_multiplicity[key], count)
    return result


class GameStateBuilder:
    """Convert vision-state dictionaries into one normalized DecisionContext."""

    @staticmethod
    def build(
        state_or_hand: dict[str, Any] | list[str],
        *,
        rules: dict[str, Any] | None = None,
        config_path: str = "config/rules.yaml",
    ) -> DecisionContext:
        state = _state_from_input(state_or_hand)
        rules = normalize_rules(rules) if rules is not None else load_rules(config_path)
        raw_hand = _raw_hand_from_state(state)
        card_instances = _card_instances_from_state(state, raw_hand)
        normalized_hand = [card.label for card in card_instances]
        legal_action_types = _normalize_legal_action_types(state.get("legal_actions"))
        if not legal_action_types:
            legal_action_types = _infer_legal_action_types_from_buttons(state.get("buttons"))
        if not legal_action_types:
            legal_action_types = ["DISCARD", "PASS"]
        warnings = list(state.get("recognition_warnings") or [])
        errors = list(state.get("recognition_errors") or [])
        memory = state.get("memory") if isinstance(state.get("memory"), dict) else {}
        passed_chi = tuple(
            _ordered_unique(
                normalize_card_label(str(label))
                for label in (
                    state.get("passed_chi")
                    or memory.get("my_passed_chi")
                    or memory.get("passed_chi")
                    or []
                )
                if normalize_card_label(str(label))
            )
        )
        passed_peng = tuple(
            _ordered_unique(
                normalize_card_label(str(label))
                for label in (
                    state.get("passed_peng")
                    or memory.get("my_passed_peng")
                    or memory.get("passed_peng")
                    or []
                )
                if normalize_card_label(str(label))
            )
        )
        if not rules.get("wildcard", {}).get("enabled", False) and any(
            card.is_wildcard for card in card_instances
        ):
            errors.append("wildcard_seen_while_room_wildcard_disabled")
        min_conf = float(rules.get("vision", {}).get("hand_min_confidence", 0.70))
        secondary_confirmed_ids = {
            str(item.get("card_id") or item.get("id"))
            for item in state.get("hand_details") or []
            if "secondary_grid_" in str(item.get("template") or "")
        }
        for card in card_instances:
            if (
                card.confidence is not None
                and card.confidence < min_conf
                and card.id not in secondary_confirmed_ids
            ):
                warnings.append(f"low_confidence_card:{card.id}:{card.label}:{card.confidence:.2f}")
        return DecisionContext(
            context_id=str(state.get("context_id") or state.get("decision_id") or "context_local"),
            state=deepcopy(state),
            raw_hand=raw_hand,
            normalized_hand=normalized_hand,
            card_instances=card_instances,
            rules=rules,
            legal_action_types=legal_action_types,
            buttons=list(state.get("buttons") or []),
            chi_options=_state_options_for_region(state, "chi_options"),
            compare_options=_state_options_for_region(state, "compare_options"),
            existing_melds=_existing_melds_from_state(state, rules),
            recognition_warnings=_ordered_unique(warnings),
            recognition_errors=_ordered_unique(errors),
            passed_chi=passed_chi,
            passed_peng=passed_peng,
            phase=str(state.get("phase") or state.get("flow_state") or "unknown"),
            frame_id=state.get("frame_id"),
            remaining_deck_count=state.get("remaining_deck_count")
            or state.get("remaining_cards_estimate"),
        )


def _state_options_for_region(state: dict[str, Any], region_name: str) -> list[dict[str, Any]]:
    explicit_key = "chi_options" if region_name == "chi_options" else "compare_options"
    explicit = state.get(explicit_key)
    if explicit:
        return [dict(item) for item in explicit if isinstance(item, dict)]
    options = state.get("option_details") or []
    return [
        dict(item)
        for item in options
        if isinstance(item, dict) and item.get("region_name") == region_name
    ]


def build_decision_context(
    state_or_hand: dict[str, Any] | list[str],
    *,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
) -> DecisionContext:
    return GameStateBuilder.build(state_or_hand, rules=rules, config_path=config_path)


def _state_from_input(state_or_hand: dict[str, Any] | list[str]) -> dict[str, Any]:
    if isinstance(state_or_hand, dict):
        return deepcopy(state_or_hand)
    return {"hand": list(state_or_hand), "legal_actions": ["DISCARD"]}


def _raw_hand_from_state(state: dict[str, Any]) -> list[str]:
    if isinstance(state.get("raw_hand"), list):
        return [str(item) for item in state["raw_hand"]]
    if isinstance(state.get("hand"), list):
        return [str(item) for item in state["hand"]]
    details = state.get("hand_details") or []
    return [str(item.get("name") or item.get("label") or "") for item in details]


def _card_instances_from_state(state: dict[str, Any], raw_hand: list[str]) -> list[CardInstance]:
    details = state.get("hand_details") or []
    if details:
        result: list[CardInstance] = []
        for index, item in enumerate(details, start=1):
            label = item.get("label") or item.get("name") or (raw_hand[index - 1] if index <= len(raw_hand) else "")
            x = item.get("center_x")
            y = item.get("center_y")
            if x is None and item.get("x") is not None:
                x = int(item["x"] + item.get("w", 0) / 2)
            if y is None and item.get("y") is not None:
                y = int(item["y"] + item.get("h", 0) / 2)
            result.append(
                card_instance(
                    str(label),
                    source=str(item.get("source") or "hand"),
                    x=int(x) if x is not None else None,
                    y=int(y) if y is not None else None,
                    confidence=item.get("confidence"),
                    clickable=bool(item.get("clickable", True)),
                    instance_id=str(item.get("card_id") or item.get("id") or f"h{index:03d}"),
                )
            )
        return result
    return [
        card_instance(label, source="hand", instance_id=f"h{index:03d}")
        for index, label in enumerate(raw_hand, start=1)
        if normalize_card_label(label)
    ]


def _normalize_legal_action_types(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for item in values:
        action = item.get("type") if isinstance(item, dict) else item
        if not action:
            continue
        result.append(str(action).upper())
    return _ordered_unique(result)


def _infer_legal_action_types_from_buttons(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    button_map = {
        "hu": "HU",
        "chi": "CHI",
        "peng": "PENG",
        "pass": "PASS",
        "guo": "PASS",
        "过": "PASS",
        "吃": "CHI",
        "碰": "PENG",
        "胡": "HU",
    }
    result: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("name") or item.get("type") or item.get("label") or "").strip().lower()
        action = button_map.get(raw_name)
        if action:
            result.append(action)
    if result and "PASS" not in result:
        result.append("PASS")
    return _ordered_unique(result)


def allocate_hand_structures(state_or_context: DecisionContext | dict[str, Any] | list[str]) -> StructureAllocation:
    context = (
        state_or_context
        if isinstance(state_or_context, DecisionContext)
        else build_decision_context(state_or_context)
    )
    rules = context.rules
    all_cards = sorted(context.card_instances, key=_card_sort_key)
    total_counts = _counter_from_instances(all_cards)
    remaining = {card.id: card for card in all_cards}
    notes: list[str] = []
    conflicts: list[str] = []
    locked_melds: list[ProtectedMeld] = []
    locked_counts: Counter[str] = Counter()
    meld_seq = 0

    def next_meld_id() -> str:
        nonlocal meld_seq
        meld_seq += 1
        return f"m{meld_seq:03d}"

    def make_meld(
        meld_type: str,
        cards: list[CardInstance],
        protect_level: str,
        reason: str,
        waiting_for: str | None = None,
    ) -> ProtectedMeld:
        labels = [card.label for card in cards]
        return ProtectedMeld(
            id=next_meld_id(),
            type=meld_type,
            card_ids=[card.id for card in cards],
            labels=labels,
            ranks=_meld_rank_labels(labels),
            protect_level=protect_level,
            xi_value=_meld_xi_value(meld_type, labels, rules),
            red_black_value=sum(1 for label in labels if label in RED_LABELS),
            structure_value=_structure_value(meld_type, labels, rules),
            reason=reason,
            used_counts=dict(Counter(labels)),
            waiting_for=waiting_for,
        )

    for card in list(remaining.values()):
        if card.is_wildcard:
            meld = make_meld("wildcard_meld", [card], "hard", "wildcard_hard_protected")
            locked_melds.append(meld)
            locked_counts.update(meld.used_counts)
            del remaining[card.id]
            notes.append(f"王 hard protected: {card.id}")

    by_label = _instances_by_label(list(remaining.values()))
    for label in sorted(by_label):
        available = sorted(by_label[label], key=_card_sort_key)
        if len(available) >= 4:
            cards = available[:4]
            meld = make_meld("exact_quad", cards, "hard", "exact_quad_high_xi")
        elif len(available) >= 3:
            cards = available[:3]
            meld = make_meld("exact_triplet", cards, "hard", "exact_triplet_high_xi")
        else:
            continue
        locked_melds.append(meld)
        locked_counts.update(meld.used_counts)
        for card in cards:
            remaining.pop(card.id, None)
        notes.append(f"{''.join(meld.labels)} locked as {meld.type}")

    free_after_locked = dict(_counter_from_instances(list(remaining.values())))
    soft_candidates = _generate_soft_candidates(list(remaining.values()), rules)
    soft_melds = _choose_best_soft_allocation(soft_candidates)
    soft_ids = {card_id for meld in soft_melds for card_id in meld.card_ids}
    for card_id in soft_ids:
        remaining.pop(card_id, None)
    final_free_cards = sorted(remaining.values(), key=_card_sort_key)
    weak_potentials, weak_notes = _generate_weak_potentials(
        final_free_cards,
        total_counts=total_counts,
        free_after_locked=Counter(free_after_locked),
        make_meld=make_meld,
    )
    notes.extend(weak_notes)

    final_melds = [*locked_melds, *soft_melds]
    used_ids = [card_id for meld in final_melds for card_id in meld.card_ids]
    duplicate_ids = sorted({card_id for card_id, count in Counter(used_ids).items() if count > 1})
    if duplicate_ids:
        conflicts.append("duplicate_card_allocation:" + ",".join(duplicate_ids))

    hard_ids = [card_id for meld in locked_melds for card_id in meld.card_ids]
    hard_labels = _ordered_unique([label for meld in locked_melds for label in meld.labels])
    soft_labels = _ordered_unique([label for meld in soft_melds for label in meld.labels])

    return StructureAllocation(
        normalized_hand=list(context.normalized_hand),
        card_instances=list(context.card_instances),
        total_counts=dict(total_counts),
        locked_melds=locked_melds,
        locked_counts=dict(locked_counts),
        free_counts_after_locked=free_after_locked,
        soft_melds=soft_melds,
        potential_melds=list(weak_potentials),
        weak_potentials=list(weak_potentials),
        orphan_cards=[card.label for card in final_free_cards],
        protected_melds=final_melds,
        hard_protected_instances=hard_ids,
        soft_protected_instances=[card_id for meld in soft_melds for card_id in meld.card_ids],
        hard_protected_labels=hard_labels,
        soft_protected_labels=soft_labels,
        free_discard_instances=final_free_cards,
        allocation_candidates=[meld.to_dict() for meld in soft_candidates],
        chosen_allocation=[meld.id for meld in final_melds],
        allocation_notes=notes,
        allocation_conflicts=conflicts,
    )


def _instances_by_label(cards: list[CardInstance]) -> dict[str, list[CardInstance]]:
    result: dict[str, list[CardInstance]] = defaultdict(list)
    for card in cards:
        result[card.label].append(card)
    return {label: sorted(items, key=_card_sort_key) for label, items in result.items()}


def _candidate_instance_sets_for_labels(
    labels: list[str],
    by_label: dict[str, list[CardInstance]],
) -> list[list[CardInstance]]:
    required = Counter(labels)
    pools: list[list[tuple[CardInstance, ...]]] = []
    for label, amount in required.items():
        available = by_label.get(label, [])
        if len(available) < amount:
            return []
        pools.append(list(combinations(available, amount)))
    results: list[list[CardInstance]] = []
    for combo in product(*pools):
        cards = [card for group in combo for card in group]
        if len({card.id for card in cards}) == len(cards):
            results.append(sorted(cards, key=_card_sort_key))
    return results


def _generate_soft_candidates(cards: list[CardInstance], rules: dict[str, Any]) -> list[ProtectedMeld]:
    by_label = _instances_by_label(cards)
    candidates: list[ProtectedMeld] = []
    sequence = 0

    def next_id() -> str:
        nonlocal sequence
        sequence += 1
        return f"c{sequence:03d}"

    def add(meld_type: str, card_group: list[CardInstance], reason: str) -> None:
        labels = [card.label for card in card_group]
        candidates.append(
            ProtectedMeld(
                id=next_id(),
                type=meld_type,
                card_ids=[card.id for card in card_group],
                labels=labels,
                ranks=_meld_rank_labels(labels),
                protect_level="soft",
                xi_value=_meld_xi_value(meld_type, labels, rules),
                red_black_value=sum(1 for label in labels if label in RED_LABELS),
                structure_value=_structure_value(meld_type, labels, rules),
                reason=reason,
                used_counts=dict(Counter(labels)),
            )
        )

    for suit in ("small", "big"):
        labels_123 = [label_for(suit, rank) for rank in (1, 2, 3)]
        for group in _candidate_instance_sets_for_labels(labels_123, by_label):
            add("special_123", group, "complete_special_123")
        labels_2710 = [label_for(suit, rank) for rank in (2, 7, 10)]
        for group in _candidate_instance_sets_for_labels(labels_2710, by_label):
            add("special_2710", group, "complete_special_2710")
        for start in range(1, 9):
            labels = [label_for(suit, rank) for rank in (start, start + 1, start + 2)]
            for group in _candidate_instance_sets_for_labels(labels, by_label):
                add("normal_sequence", group, "complete_normal_sequence")

    by_rank: dict[int, list[CardInstance]] = defaultdict(list)
    for card in cards:
        if card.rank is not None and not card.is_wildcard:
            by_rank[card.rank].append(card)
    for rank, rank_cards in by_rank.items():
        if len(rank_cards) < 3:
            continue
        for group_tuple in combinations(sorted(rank_cards, key=_card_sort_key), 3):
            if len({card.label for card in group_tuple}) < 2:
                continue
            add("mixed_same_rank_triplet", list(group_tuple), f"mixed_same_rank_triplet_rank_{rank}")

    for label, same_label_cards in by_label.items():
        if label == WILD_LABEL or len(same_label_cards) < 2:
            continue
        for group_tuple in combinations(same_label_cards, 2):
            add("pair", list(group_tuple), "complete_pair")

    return sorted(
        candidates,
        key=lambda meld: (
            _soft_type_priority(meld.type),
            meld.structure_value,
            meld.xi_value,
            "".join(meld.labels),
        ),
        reverse=True,
    )


def _soft_type_priority(meld_type: str) -> int:
    priority = {
        "special_2710": 90,
        "special_123": 85,
        "mixed_same_rank_triplet": 80,
        "double_normal_sequence": 78,
        "normal_sequence": 70,
        "wildcard_meld": 60,
        "pair": 30,
    }
    return priority.get(meld_type, 0)


def _choose_best_soft_allocation(candidates: list[ProtectedMeld]) -> list[ProtectedMeld]:
    ordered = sorted(
        candidates,
        key=lambda meld: (_soft_type_priority(meld.type), meld.structure_value, meld.xi_value),
        reverse=True,
    )
    if not ordered:
        return []

    card_ids = sorted({card_id for meld in ordered for card_id in meld.card_ids})
    bit_by_id = {card_id: 1 << index for index, card_id in enumerate(card_ids)}
    best_by_mask: dict[int, tuple[ProtectedMeld, float, int]] = {}
    for index, meld in enumerate(ordered):
        mask = 0
        for card_id in meld.card_ids:
            mask |= bit_by_id[card_id]
        weight = (
            meld.structure_value
            + meld.xi_value * 40
            + _soft_type_priority(meld.type) * 10.0
        )
        current = best_by_mask.get(mask)
        if current is None or (weight, -index) > (current[1], -current[2]):
            best_by_mask[mask] = (meld, weight, index)

    compact = sorted(
        ((mask, meld, weight, index) for mask, (meld, weight, index) in best_by_mask.items()),
        key=lambda item: item[3],
    )
    masks = [item[0] for item in compact]
    melds = [item[1] for item in compact]
    weights = [item[2] for item in compact]
    by_bit: dict[int, list[int]] = defaultdict(list)
    for candidate_index, mask in enumerate(masks):
        bit = 1
        remaining = mask
        while remaining:
            if remaining & 1:
                by_bit[bit].append(candidate_index)
            remaining >>= 1
            bit <<= 1

    full_mask = (1 << len(card_ids)) - 1
    _score, selected_indexes = _solve_soft_allocation_state(
        full_mask,
        by_bit,
        masks,
        weights,
        {},
    )
    return sorted((melds[index] for index in selected_indexes), key=lambda meld: meld.id)


def _solve_soft_allocation_state(
    available: int,
    by_bit: dict[int, list[int]],
    masks: list[int],
    weights: list[float],
    memo: dict[int, tuple[float, tuple[int, ...]]],
) -> tuple[float, tuple[int, ...]]:
    cached = memo.get(available)
    if cached is not None:
        return cached
    applicable_by_bit: list[tuple[int, list[int]]] = []
    for bit, indexes in by_bit.items():
        if not available & bit:
            continue
        applicable = [index for index in indexes if masks[index] & available == masks[index]]
        if applicable:
            applicable_by_bit.append((bit, applicable))
    if not applicable_by_bit:
        result = (0.0, ())
        memo[available] = result
        return result
    pivot, options = min(applicable_by_bit, key=lambda item: (len(item[1]), item[0]))
    best_score, best_indexes = _solve_soft_allocation_state(
        available & ~pivot,
        by_bit,
        masks,
        weights,
        memo,
    )
    for candidate_index in options:
        child_score, child_indexes = _solve_soft_allocation_state(
            available & ~masks[candidate_index],
            by_bit,
            masks,
            weights,
            memo,
        )
        candidate_score = weights[candidate_index] + child_score
        candidate_indexes = (candidate_index, *child_indexes)
        if _allocation_result_key(candidate_score, candidate_indexes) > _allocation_result_key(
            best_score,
            best_indexes,
        ):
            best_score, best_indexes = candidate_score, candidate_indexes
    result = (best_score, best_indexes)
    memo[available] = result
    return result


def _allocation_result_key(score: float, indexes: tuple[int, ...]) -> tuple[float, int, tuple[int, ...]]:
    return score, len(indexes), tuple(-index for index in sorted(indexes))


def _generate_weak_potentials(
    free_cards: list[CardInstance],
    *,
    total_counts: Counter[str],
    free_after_locked: Counter[str],
    make_meld,
) -> tuple[list[ProtectedMeld], list[str]]:
    by_label = _instances_by_label(free_cards)
    notes: list[str] = []
    potentials: list[ProtectedMeld] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()

    def add_wait(meld_type: str, present_labels: list[str], missing_label: str, reason: str) -> None:
        key = (meld_type, tuple(sorted(present_labels)), missing_label)
        if key in seen:
            return
        seen.add(key)
        groups = _candidate_instance_sets_for_labels(present_labels, by_label)
        if not groups:
            return
        meld = make_meld(
            "weak_wait",
            groups[0],
            "weak",
            reason,
            waiting_for=missing_label,
        )
        potentials.append(meld)

    for suit in ("small", "big"):
        special_sets = [
            ("weak_gap_123", [1, 2, 3]),
            ("weak_gap_2710", [2, 7, 10]),
        ]
        sequence_sets = [("weak_gap_sequence", [start, start + 1, start + 2]) for start in range(1, 9)]
        for meld_type, ranks in [*special_sets, *sequence_sets]:
            labels = [label_for(suit, rank) for rank in ranks]
            present = [label for label in labels if by_label.get(label)]
            missing = [label for label in labels if not by_label.get(label)]
            if len(present) != 2 or len(missing) != 1:
                continue
            missing_label = missing[0]
            reason = f"{''.join(labels)} waiting_for {missing_label}"
            if total_counts.get(missing_label, 0) > 0 and free_after_locked.get(missing_label, 0) == 0:
                reason = (
                    f"{''.join(labels)} not available: {missing_label} free_count is 0 "
                    f"after locking {missing_label * total_counts[missing_label]}"
                )
                notes.append(reason)
            add_wait(meld_type, present, missing_label, reason)
    return potentials, notes


def calculate_xi(melds: list[ProtectedMeld], rules: dict[str, Any] | None = None) -> XiResult:
    rules = rules or load_rules()
    total = sum(int(meld.xi_value) for meld in melds)
    minimum = int(rules.get("rules", {}).get("min_xi", 9))
    details = [
        {"type": meld.type, "cards": list(meld.labels), "xi": meld.xi_value}
        for meld in melds
        if meld.xi_value
    ]
    return XiResult(
        total_xi=total,
        details=details,
        is_enough_to_hu=total >= minimum,
        min_xi=minimum,
        xi_gap=max(0, minimum - total),
    )


def estimate_potential_xi(
    allocation: StructureAllocation,
    rules: dict[str, Any] | None = None,
    existing_melds: list[ProtectedMeld] | None = None,
) -> XiPotential:
    rules = rules or load_rules()
    confirmed_melds = [*(existing_melds or []), *allocation.protected_melds]
    confirmed = calculate_xi(confirmed_melds, rules).total_xi
    weak_bonus = 0
    missing_sources: list[dict[str, Any]] = []
    improving: list[str] = []
    for weak in allocation.weak_potentials:
        if weak.waiting_for:
            improving.append(weak.waiting_for)
            weak_bonus += 3
            missing_sources.append(weak.to_dict())
    sources = [meld.to_dict() for meld in confirmed_melds if meld.xi_value]
    reducers = _ordered_unique(
        [label for meld in confirmed_melds if meld.xi_value for label in meld.labels]
    )
    return XiPotential(
        confirmed_xi=confirmed,
        potential_xi=confirmed + weak_bonus,
        best_xi_route=sources[:5],
        xi_sources=sources,
        missing_xi_sources=missing_sources,
        cards_that_improve_xi=_ordered_unique(improving),
        cards_that_reduce_xi=reducers,
    )


def _ting_probe_labels(context: DecisionContext) -> list[str]:
    counts = Counter(context.normalized_hand)
    limits = {label: int(context.rules.get("game", {}).get("deck_copies", 4)) for label in [*SMALL_LABELS, *BIG_LABELS]}
    if context.rules.get("wildcard", {}).get("enabled", False):
        limits[WILD_LABEL] = int(context.rules.get("wildcard", {}).get("copies", 4))
    return [label for label, limit in limits.items() if counts.get(label, 0) < limit]


def _context_after_ting_draw(context: DecisionContext, label: str) -> DecisionContext:
    drawn = card_instance(label, source="ting_probe", clickable=False)
    after_cards = [*context.card_instances, drawn]
    after_labels = [card.label for card in after_cards]
    after_state = {
        **context.state,
        "hand": after_labels,
        "raw_hand": after_labels,
        "hand_details": [card.to_dict() | {"name": card.label} for card in after_cards],
        "recognition_warnings": [],
        "recognition_errors": [],
    }
    min_conf = float(context.rules.get("vision", {}).get("hand_min_confidence", 0.70))
    warnings = [
        f"low_confidence_card:{card.id}:{card.label}:{card.confidence:.2f}"
        for card in after_cards
        if card.confidence is not None and card.confidence < min_conf
    ]
    errors = []
    if not context.rules.get("wildcard", {}).get("enabled", False) and any(
        card.is_wildcard for card in after_cards
    ):
        errors.append("wildcard_seen_while_room_wildcard_disabled")
    return replace(
        context,
        state=after_state,
        raw_hand=list(after_labels),
        normalized_hand=list(after_labels),
        card_instances=after_cards,
        recognition_warnings=_ordered_unique(warnings),
        recognition_errors=_ordered_unique(errors),
    )


def _ting_hu_probe_cache_key(
    context: DecisionContext,
    allocation: StructureAllocation,
    rules: dict[str, Any],
) -> tuple[object, ...]:
    memory = context.state.get("memory") if isinstance(context.state.get("memory"), dict) else None
    opponent = infer_opponent_profile(memory)
    existing_melds = tuple(
        sorted(
            (
                meld.type,
                tuple(sorted(meld.labels)),
                meld.xi_value,
                round(meld.structure_value, 6),
                meld.protect_level,
            )
            for meld in context.existing_melds
        )
    )
    return (
        id(rules),
        tuple(sorted(Counter(context.normalized_hand).items())),
        existing_melds,
        tuple(sorted(allocation.allocation_conflicts)),
        "HU" in context.legal_action_types,
        _pending_action_label(context),
        context.remaining_deck_count,
        str(opponent.get("profile") or "unknown"),
        int(opponent.get("meld_count") or 0),
    )


def _ting_draw_feature_cache_key(
    context: DecisionContext,
    rules: dict[str, Any],
    *,
    exact_probe_enabled: bool,
) -> tuple[object, ...]:
    """Canonical key for draw features that ignore interchangeable card IDs."""

    existing_melds = tuple(
        sorted(
            (
                meld.type,
                tuple(sorted(meld.labels)),
                meld.xi_value,
                round(meld.structure_value, 6),
                meld.protect_level,
            )
            for meld in context.existing_melds
        )
    )
    return (
        id(rules),
        tuple(sorted(Counter(context.normalized_hand).items())),
        existing_melds,
        bool(exact_probe_enabled),
    )


def _ting_draw_features(
    context: DecisionContext,
    rules: dict[str, Any],
    *,
    exact_probe_enabled: bool,
) -> tuple[int, int, int, tuple[bool, int, float] | None]:
    """Return the exact draw features, reusing only identical count states."""

    cache = _TING_DRAW_FEATURE_CACHE.get()
    key = _ting_draw_feature_cache_key(
        context,
        rules,
        exact_probe_enabled=exact_probe_enabled,
    )
    if cache is not None and key in cache:
        return cache[key]

    allocation = allocate_hand_structures(context)
    xi = estimate_potential_xi(allocation, rules, context.existing_melds)
    hu = (
        _cached_ting_hu_probe(context, allocation, rules)
        if exact_probe_enabled
        else None
    )
    result = (
        int(xi.confirmed_xi),
        int(xi.potential_xi),
        len(allocation.weak_potentials),
        hu,
    )
    if cache is not None:
        cache[key] = result
    return result


def _cached_ting_hu_probe(
    context: DecisionContext,
    allocation: StructureAllocation,
    rules: dict[str, Any],
) -> tuple[bool, int, float]:
    cache = _TING_HU_PROBE_CACHE.get()
    if cache is None:
        return _grouping_ting_hu_probe(context, rules)
    key = _ting_hu_probe_cache_key(context, allocation, rules)
    if key not in cache:
        cache[key] = _grouping_ting_hu_probe(context, rules)
    return cache[key]


def _grouping_ting_hu_probe(
    context: DecisionContext,
    rules: dict[str, Any],
) -> tuple[bool, int, float]:
    """Exact count-state HU projection used by high-volume ting probes.

    Ting evaluation consumes only can-HU, total xi, and immediate score. The
    engine count solver already maximizes concealed xi, so rebuilding every
    card-instance permutation and wildcard ID mapping is redundant here.
    """

    required_pair_count = int(
        any(len(meld.labels) == 4 for meld in context.existing_melds)
    )
    groups = best_grouping_normalized(
        tuple(context.normalized_hand),
        required_pair_count=required_pair_count,
    )
    if sum(len(group) for group in groups) != len(context.normalized_hand):
        return False, 0, 0.0
    required_groups = int(rules.get("rules", {}).get("required_meld_groups", 7))
    if len(context.existing_melds) + len(groups) != required_groups:
        return False, 0, 0.0
    if any(meld.type == "unknown" for meld in context.existing_melds):
        return False, 0, 0.0

    existing_xi = calculate_xi(context.existing_melds, rules).total_xi
    hand_xi = sum(
        concealed_group_xi(list(group), rules=rules)
        for group in groups
        if len(group) == 3
    )
    total_xi = existing_xi + hand_xi
    labels = [
        label
        for meld in context.existing_melds
        for label in meld.labels
        if label != WILD_LABEL
    ]
    labels.extend(
        label
        for group in groups
        for label in group
        if label != WILD_LABEL
    )
    red_count = sum(label in RED_LABELS for label in labels)
    outcome = classify_red_black(red_count, rules, card_count=len(labels))
    weights = rules.get("weights") or rules.get("ai_weights", {})
    red_black_bonus = float(outcome.points) * float(
        weights.get("red_black_special_ev_per_point", 240)
    )
    score_now = float(total_xi * 120) + red_black_bonus
    minimum = int(rules.get("rules", {}).get("min_xi", 9))
    return total_xi >= minimum, total_xi, score_now


def _ting_probe_partition_impossible(context: DecisionContext) -> bool:
    groups = best_grouping(
        context.normalized_hand,
        required_pair_count=int(any(len(meld.labels) == 4 for meld in context.existing_melds)),
    )
    return sum(len(group) for group in groups) != len(context.normalized_hand)


def _context_for_hu_evaluation(context: DecisionContext) -> DecisionContext:
    if "HU" not in context.legal_action_types:
        return context
    pending_label = _pending_action_label(context)
    if not pending_label:
        return context
    required_groups = int(context.rules.get("rules", {}).get("required_meld_groups", 7))
    hand_groups = required_groups - len(context.existing_melds)
    pair_groups = int(any(len(meld.labels) == 4 for meld in context.existing_melds))
    triple_groups = hand_groups - pair_groups
    if hand_groups < 0 or triple_groups < 0:
        return context
    expected_hand_cards = triple_groups * 3 + pair_groups * 2
    if len(context.card_instances) + 1 != expected_hand_cards:
        return context

    pending_id = "hu_pending_001"
    used_ids = {card.id for card in context.card_instances}
    suffix = 1
    while pending_id in used_ids:
        suffix += 1
        pending_id = f"hu_pending_{suffix:03d}"
    pending_card = card_instance(
        pending_label,
        source="pending_hu",
        clickable=False,
        instance_id=pending_id,
    )
    after_cards = [*context.card_instances, pending_card]
    metadata = deepcopy(context.state.get("metadata") or {})
    metadata["hu_pending_card_included"] = {
        "label": pending_label,
        "expected_hand_cards": expected_hand_cards,
    }
    after_state = {
        **context.state,
        "hand": [card.label for card in after_cards],
        "raw_hand": [card.label for card in after_cards],
        "hand_details": [card.to_dict() | {"name": card.label} for card in after_cards],
        "metadata": metadata,
    }
    return replace(
        context,
        state=after_state,
        raw_hand=[card.label for card in after_cards],
        normalized_hand=[card.label for card in after_cards],
        card_instances=after_cards,
    )


def _ting_distance_from_features(
    *,
    can_hu_now: bool,
    hu_card_count: int,
    improving_card_count: int,
    potential_gap: int,
    orphan_count: int,
) -> int:
    if can_hu_now:
        return 0
    if hu_card_count:
        return 1
    gap_distance = max(1, (potential_gap + 2) // 3)
    orphan_pressure = 1 if orphan_count >= 3 else 0
    if improving_card_count:
        return min(4, max(2, gap_distance + orphan_pressure))
    return min(4, max(2, gap_distance + orphan_pressure + 1))


def _ting_value_delta(
    before: TingEstimate,
    after: TingEstimate,
    weights: dict[str, Any],
    *,
    include_wait_breadth: bool = True,
) -> float:
    return round(
        _ting_state_value(after, weights, include_wait_breadth=include_wait_breadth)
        - _ting_state_value(before, weights, include_wait_breadth=include_wait_breadth),
        3,
    )


def _ting_state_value(
    estimate: TingEstimate,
    weights: dict[str, Any],
    *,
    include_wait_breadth: bool = True,
) -> float:
    hu_card_count = len(set(estimate.hu_cards))
    value = (
        -estimate.shanten_like_distance * float(weights.get("ting_distance_gain", 40))
        + len(set(estimate.improving_cards)) * float(weights.get("ting_improving_card_gain", 2))
        + estimate.expected_xi_if_hu * float(weights.get("ting_expected_xi_gain", 2))
    )
    if include_wait_breadth:
        value += hu_card_count * float(weights.get("ting_hu_card_gain", 8))
    if estimate.is_ting:
        value += float(weights.get("ting_ready", 500))
        if include_wait_breadth:
            value += hu_card_count * float(weights.get("ting_out_preservation", 40))
    return value


def check_hu(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    allocation: StructureAllocation | None = None,
    rules: dict[str, Any] | None = None,
) -> HuResult:
    context = state_or_context if isinstance(state_or_context, DecisionContext) else build_decision_context(state_or_context)
    rules = rules or context.rules
    hu_context = _context_for_hu_evaluation(context)
    if hu_context is not context:
        context = hu_context
        allocation = allocate_hand_structures(context)
    else:
        allocation = allocation or allocate_hand_structures(context)
    minimum = int(rules.get("rules", {}).get("min_xi", 9))
    required_groups = int(rules.get("rules", {}).get("required_meld_groups", 7))
    if allocation.allocation_conflicts:
        return HuResult(
            can_hu=False,
            total_xi=calculate_xi(context.existing_melds, rules).total_xi,
            min_xi=minimum,
            partition=[],
            wildcard_mapping={},
            red_black_bonus=0,
            reason="allocation_conflict",
            reject_reason="duplicate_card_allocation",
        )
    if _ting_probe_partition_impossible(context):
        hand_partition, wildcard_mapping = [], {}
    else:
        hand_partition, wildcard_mapping = _search_hu_partition(context, rules)
    hand_complete = not context.card_instances or bool(hand_partition)
    full_partition = [*context.existing_melds, *hand_partition]
    partition_xi = calculate_xi(full_partition, rules)
    red_black_bonus = _hu_red_black_bonus(context, full_partition, rules)
    partition_payload = [meld.to_dict() for meld in full_partition]
    if not hand_complete:
        return HuResult(
            can_hu=False,
            total_xi=partition_xi.total_xi,
            min_xi=minimum,
            partition=partition_payload,
            wildcard_mapping=wildcard_mapping,
            red_black_bonus=red_black_bonus,
            reason="hand cards cannot be fully partitioned into legal melds",
            reject_reason="partition_incomplete",
        )
    if any(meld.type == "unknown" for meld in context.existing_melds):
        return HuResult(
            can_hu=False,
            total_xi=partition_xi.total_xi,
            min_xi=minimum,
            partition=partition_payload,
            wildcard_mapping=wildcard_mapping,
            red_black_bonus=red_black_bonus,
            reason="existing meld type is unknown",
            reject_reason="existing_meld_unknown",
        )
    if len(full_partition) != required_groups:
        return HuResult(
            can_hu=False,
            total_xi=partition_xi.total_xi,
            min_xi=minimum,
            partition=partition_payload,
            wildcard_mapping=wildcard_mapping,
            red_black_bonus=red_black_bonus,
            reason=f"meld group count {len(full_partition)} != required {required_groups}",
            reject_reason="meld_group_count_invalid",
        )
    score_now, continue_ev, hu_decision, strategy_reason = _hu_strategy(
        context,
        full_partition,
        partition_xi.total_xi,
        rules,
        wildcard_mapping,
    )
    if partition_xi.total_xi < minimum:
        return HuResult(
            can_hu=False,
            total_xi=partition_xi.total_xi,
            min_xi=minimum,
            partition=partition_payload,
            wildcard_mapping=wildcard_mapping,
            red_black_bonus=red_black_bonus,
            reason=f"total_xi {partition_xi.total_xi} < min_xi {minimum}",
            reject_reason="xi_not_enough",
            score_now=score_now,
            continue_ev=continue_ev,
            decision="reject",
        )
    return HuResult(
        can_hu=True,
        total_xi=partition_xi.total_xi,
        min_xi=minimum,
        partition=partition_payload,
        wildcard_mapping=wildcard_mapping,
        red_black_bonus=red_black_bonus,
        reason=strategy_reason,
        reject_reason=None if hu_decision == "hu_now" else "hu_continue_ev_better",
        score_now=score_now,
        continue_ev=continue_ev,
        decision=hu_decision,
    )


def _hu_red_black_bonus(
    context: DecisionContext,
    partition: list[ProtectedMeld],
    rules: dict[str, Any],
) -> float:
    labels = [label for meld in partition for label in meld.labels if label != WILD_LABEL]
    red_count = sum(1 for label in labels if label in RED_LABELS)
    outcome = classify_red_black(red_count, rules, card_count=len(labels))
    weights = rules.get("weights") or rules.get("ai_weights", {})
    point_value = float(weights.get("red_black_special_ev_per_point", 240))
    return round(outcome.points * point_value, 3)


def _hu_strategy(
    context: DecisionContext,
    partition: list[ProtectedMeld],
    total_xi: int,
    rules: dict[str, Any],
    wildcard_mapping: dict[str, str],
) -> tuple[float, float, str, str]:
    must_hu = str(rules.get("rules", {}).get("must_hu", "none")).lower()
    red_black_bonus = _hu_red_black_bonus(context, partition, rules)
    score_now = float(total_xi * 120 + red_black_bonus)
    wildcard_count = sum(1 for card in context.card_instances if card.is_wildcard)
    remaining = context.remaining_deck_count
    defense_remaining, _danger_remaining = _room_phase_thresholds(rules)
    early_or_mid = remaining is None or remaining > defense_remaining
    weak_waits = len(allocate_hand_structures(context).weak_potentials)
    continue_ev = score_now
    reasons: list[str] = [f"score_now={score_now:.1f}", f"total_xi={total_xi}"]
    if red_black_bonus:
        reasons.append(f"red_black_bonus={red_black_bonus:.1f}")
    if wildcard_count:
        continue_ev += wildcard_count * 180
        reasons.append("手里有王，继续空间增加")
    if wildcard_mapping:
        continue_ev += len(wildcard_mapping) * 80
        reasons.append("王已可补胡，仍保留转更大牌空间")
    if weak_waits:
        continue_ev += weak_waits * 60
        reasons.append(f"weak_waits={weak_waits}")
    if remaining is not None and remaining <= defense_remaining:
        continue_ev -= 500
        reasons.append(f"后盘剩余={remaining}，有胡优先落袋")
    profile = infer_opponent_profile(context.state.get("memory") if isinstance(context.state.get("memory"), dict) else None)
    opponent_dangerous = profile.get("profile") in {"fast_meld", "red_value", "pair_triplet_value"} or int(profile.get("meld_count", 0)) >= 3
    if opponent_dangerous:
        continue_ev -= 500
        reasons.append(f"opponent_profile={profile.get('profile')}，对手危险，禁止贪忍")
    if must_hu in {"any", "dianpao"}:
        return score_now, continue_ev, "hu_now", "规则要求能胡即胡；" + "；".join(reasons)
    if remaining is not None and remaining <= defense_remaining:
        return score_now, continue_ev, "hu_now", "后盘有胡就胡；" + "；".join(reasons)
    if opponent_dangerous:
        return score_now, continue_ev, "hu_now", "对手危险时有胡就胡；" + "；".join(reasons)
    if early_or_mid and wildcard_count and continue_ev > score_now:
        return (
            score_now,
            continue_ev,
            "hu_now",
            "继续估值仅为启发式，未经整局配对反事实证明，不得据此拒胡；" + "；".join(reasons),
        )
    return score_now, continue_ev, "hu_now", "牌型完整且胡息达到起胡线；" + "；".join(reasons)


def _search_hu_partition(
    context: DecisionContext,
    rules: dict[str, Any],
) -> tuple[list[ProtectedMeld], dict[str, str]]:
    candidates = _generate_hu_partition_candidates(
        context.card_instances,
        rules,
        allow_pair=any(
            len(meld.labels) == 4
            for meld in context.existing_melds
        ),
    )
    ordered_ids = sorted(card.id for card in context.card_instances)
    if not ordered_ids:
        return [], {}
    bit_by_card_id = {card_id: 1 << index for index, card_id in enumerate(ordered_ids)}
    all_mask = (1 << len(ordered_ids)) - 1
    CandidateEntry = tuple[
        ProtectedMeld,
        dict[str, str],
        int,
        tuple[tuple[int, int], ...],
    ]
    by_card_index: list[list[CandidateEntry]] = [[] for _ in ordered_ids]
    label_order = (*SMALL_LABELS, *BIG_LABELS, WILD_LABEL)
    label_index = {label: index for index, label in enumerate(label_order)}
    for meld, mapping in candidates:
        meld_mask = 0
        for card_id in meld.card_ids:
            meld_mask |= bit_by_card_id[card_id]
        label_deltas = tuple(
            (label_index[label], amount)
            for label, amount in Counter(meld.labels).items()
        )
        entry = (meld, mapping, meld_mask, label_deltas)
        for card_id in meld.card_ids:
            by_card_index[bit_by_card_id[card_id].bit_length() - 1].append(entry)
    for rows in by_card_index:
        rows.sort(key=lambda item: (item[0].xi_value, item[0].structure_value), reverse=True)

    best_partition: list[ProtectedMeld] = []
    best_mapping: dict[str, str] = {}
    best_score = (-1, float("-inf"))
    fallback_partition: list[ProtectedMeld] = []
    fallback_mapping: dict[str, str] = {}
    fallback_score = (-1, float("-inf"))
    required_groups = int(rules.get("rules", {}).get("required_meld_groups", 7))
    required_hand_groups = required_groups - len(context.existing_melds)
    best_seen: dict[
        tuple[tuple[int, ...], int],
        tuple[int, float],
    ] = {}
    used_label_counts = [0] * len(label_order)

    def merged_mapping(chosen: list[CandidateEntry]) -> dict[str, str]:
        result: dict[str, str] = {}
        for _meld, wildcard_targets, _mask, _deltas in chosen:
            result.update(wildcard_targets)
        return result

    def search(
        used_mask: int,
        chosen: list[CandidateEntry],
        score: tuple[int, float],
    ) -> None:
        nonlocal best_partition, best_mapping, best_score
        nonlocal fallback_partition, fallback_mapping, fallback_score
        if len(chosen) > required_hand_groups:
            return
        state_key = (tuple(used_label_counts), len(chosen))
        if best_seen.get(
            state_key,
            (-1, float("-inf")),
        ) >= score:
            return
        best_seen[state_key] = score
        if used_mask == all_mask:
            if score > fallback_score:
                fallback_partition = [entry[0] for entry in chosen]
                fallback_mapping = merged_mapping(chosen)
                fallback_score = score
            if len(chosen) == required_hand_groups and score > best_score:
                best_partition = [entry[0] for entry in chosen]
                best_mapping = merged_mapping(chosen)
                best_score = score
            return
        remaining_mask = all_mask ^ used_mask
        pivot_index = (remaining_mask & -remaining_mask).bit_length() - 1
        for entry in by_card_index[pivot_index]:
            meld, _wildcard_targets, meld_mask, meld_label_deltas = entry
            if meld_mask & used_mask:
                continue
            chosen.append(entry)
            for index, amount in meld_label_deltas:
                used_label_counts[index] += amount
            search(
                used_mask | meld_mask,
                chosen,
                (
                    score[0] + meld.xi_value,
                    score[1] + meld.structure_value,
                ),
            )
            for index, amount in meld_label_deltas:
                used_label_counts[index] -= amount
            chosen.pop()

    search(0, [], (0, 0.0))
    if best_partition:
        return best_partition, best_mapping
    return fallback_partition, fallback_mapping


HU_COMPLETE_MELD_TYPES = {
    "exact_triplet",
    "exact_quad",
    "mixed_same_rank_triplet",
    "pair",
    "special_123",
    "special_2710",
    "normal_sequence",
    "wildcard_meld",
}


def _hu_complete_melds(melds: list[ProtectedMeld]) -> list[ProtectedMeld]:
    return [meld for meld in melds if meld.type in HU_COMPLETE_MELD_TYPES]


def _generate_hu_partition_candidates(
    cards: list[CardInstance],
    rules: dict[str, Any],
    *,
    allow_pair: bool = False,
) -> list[tuple[ProtectedMeld, dict[str, str]]]:
    non_wild = [card for card in cards if not card.is_wildcard]
    wildcards = [card for card in cards if card.is_wildcard]
    by_label = _instances_by_label(non_wild)
    by_rank: dict[int, list[CardInstance]] = defaultdict(list)
    for card in non_wild:
        if card.rank is not None:
            by_rank[card.rank].append(card)
    candidates: list[tuple[ProtectedMeld, dict[str, str]]] = []
    seq = 0

    def next_id() -> str:
        nonlocal seq
        seq += 1
        return f"hu{seq:03d}"

    def make_candidate(
        meld_type: str,
        selected_cards: list[CardInstance],
        *,
        reason: str,
        target_labels: list[str] | None = None,
        wildcard_mapping: dict[str, str] | None = None,
        xi_type: str | None = None,
    ) -> None:
        actual_labels = [card.label for card in selected_cards]
        scoring_labels = target_labels or actual_labels
        score_type = xi_type or meld_type
        xi_value = _meld_xi_value(score_type, scoring_labels, rules)
        structure_value = _structure_value(score_type, scoring_labels, rules)
        if wildcard_mapping:
            structure_value += 150.0
        candidates.append(
            (
                ProtectedMeld(
                    id=next_id(),
                    type=meld_type,
                    card_ids=[card.id for card in selected_cards],
                    labels=actual_labels,
                    ranks=_meld_rank_labels(scoring_labels),
                    protect_level="hard" if meld_type in {"exact_triplet", "exact_quad"} else "soft",
                    xi_value=xi_value,
                    red_black_value=sum(1 for label in scoring_labels if label in RED_LABELS),
                    structure_value=structure_value,
                    reason=reason,
                    used_counts=dict(Counter(actual_labels)),
                ),
                wildcard_mapping or {},
            )
        )

    def make_wildcard_resolutions(
        target_labels: list[str],
        *,
        xi_type: str,
        reason: str,
    ) -> None:
        if not wildcards:
            return
        target_counts = Counter(target_labels)
        eligible = [
            card
            for card in non_wild
            if target_counts.get(card.label, 0)
        ]
        minimum_real = max(0, 3 - len(wildcards))
        for real_count in range(minimum_real, min(2, len(eligible)) + 1):
            wildcard_count = 3 - real_count
            if wildcard_count < 1 or wildcard_count > len(wildcards):
                continue
            for real_group in combinations(eligible, real_count):
                real_counts = Counter(card.label for card in real_group)
                if any(
                    amount > target_counts.get(label, 0)
                    for label, amount in real_counts.items()
                ):
                    continue
                remaining_targets = list(target_labels)
                for card in real_group:
                    remaining_targets.remove(card.label)
                for wild_group in combinations(wildcards, wildcard_count):
                    mapping = {
                        wild.id: target
                        for wild, target in zip(wild_group, remaining_targets)
                    }
                    make_candidate(
                        "wildcard_meld",
                        [*real_group, *wild_group],
                        reason=reason,
                        target_labels=target_labels,
                        wildcard_mapping=mapping,
                        xi_type=xi_type,
                    )

    for label, label_cards in by_label.items():
        if allow_pair:
            for group in combinations(label_cards, 2):
                make_candidate("pair", list(group), reason="hu_quad_compensation_pair")
            for real_card in label_cards:
                for wildcard in wildcards:
                    make_candidate(
                        "pair",
                        [real_card, wildcard],
                        reason=f"hu_quad_compensation_wildcard_pair:{label}",
                        wildcard_mapping={wildcard.id: label},
                    )
        for group in combinations(label_cards, 3):
            make_candidate("exact_triplet", list(group), reason="hu_exact_triplet")
        for group in combinations(label_cards, 4):
            make_candidate("exact_quad", list(group), reason="hu_exact_quad")
    for label in (*SMALL_LABELS, *BIG_LABELS):
        make_wildcard_resolutions(
            [label, label, label],
            xi_type="exact_triplet",
            reason=f"hu_wildcard_triplet:{label}",
        )

    if allow_pair:
        for group in combinations(wildcards, 2):
            make_candidate("pair", list(group), reason="hu_quad_compensation_wild_pair")

    for rank, rank_cards in by_rank.items():
        if len(rank_cards) >= 3:
            for group in combinations(sorted(rank_cards, key=_card_sort_key), 3):
                if len({card.label for card in group}) >= 2:
                    make_candidate(
                        "mixed_same_rank_triplet",
                        list(group),
                        reason=f"hu_mixed_same_rank_triplet:{rank}",
                    )
    for rank in range(1, 11):
        small = label_for("small", rank)
        big = label_for("big", rank)
        for target_labels in ([small, small, big], [small, big, big]):
            make_wildcard_resolutions(
                target_labels,
                xi_type="mixed_same_rank_triplet",
                reason=f"hu_wildcard_mixed_rank:{rank}",
            )

    sequence_sets: list[tuple[str, list[int]]] = [
        ("special_123", [1, 2, 3]),
        ("special_2710", [2, 7, 10]),
        *[("normal_sequence", [start, start + 1, start + 2]) for start in range(1, 9)],
    ]
    for suit in ("small", "big"):
        for meld_type, ranks in sequence_sets:
            labels = [label_for(suit, rank) for rank in ranks]
            real_groups = _candidate_instance_sets_for_labels(labels, by_label)
            for group in real_groups:
                make_candidate(meld_type, group, reason=f"hu_{meld_type}")
            make_wildcard_resolutions(
                labels,
                xi_type=meld_type,
                reason=f"hu_wildcard_{meld_type}",
            )

    deduped: dict[tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]], tuple[ProtectedMeld, dict[str, str]]] = {}
    for meld, mapping in candidates:
        key = (meld.type, tuple(sorted(meld.card_ids)), tuple(sorted(mapping.items())))
        current = deduped.get(key)
        if current is None or (meld.xi_value, meld.structure_value) > (
            current[0].xi_value,
            current[0].structure_value,
        ):
            deduped[key] = (meld, mapping)
    return sorted(
        deduped.values(),
        key=lambda item: (item[0].xi_value, item[0].structure_value, len(item[0].card_ids)),
        reverse=True,
    )


def evaluate_wildcard_value(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    allocation: StructureAllocation,
) -> WildcardValue:
    context = state_or_context if isinstance(state_or_context, DecisionContext) else build_decision_context(state_or_context)
    wildcard_count = sum(1 for card in context.card_instances if card.is_wildcard)
    waits = {meld.waiting_for for meld in allocation.weak_potentials if meld.waiting_for}
    notes: list[str] = []
    if wildcard_count:
        notes.append("王 hard protected; default discard forbidden")
    can_2710 = any("2710" in meld.reason or "二七十" in meld.reason for meld in allocation.weak_potentials)
    can_123 = any("123" in meld.reason or "一二三" in meld.reason for meld in allocation.weak_potentials)
    can_triplet = any(count == 2 for label, count in allocation.total_counts.items() if label != WILD_LABEL)
    best_usage = None
    if wildcard_count and can_2710:
        best_usage = "complete_2710"
    elif wildcard_count and can_123:
        best_usage = "complete_123"
    elif wildcard_count and can_triplet:
        best_usage = "complete_triplet"
    return WildcardValue(
        wildcard_count=wildcard_count,
        can_complete_hu=bool(wildcard_count and waits),
        can_complete_xi=bool(wildcard_count and waits),
        can_complete_2710=bool(wildcard_count and can_2710),
        can_complete_123=bool(wildcard_count and can_123),
        can_complete_triplet=bool(wildcard_count and can_triplet),
        can_complete_red_black=bool(wildcard_count and waits & RED_LABELS),
        best_usage=best_usage,
        future_flexibility_score=float(wildcard_count * 500),
        discard_loss=float(wildcard_count * 10000),
        usage_notes=notes,
    )


def evaluate_red_black_plan(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    allocation: StructureAllocation,
) -> RedBlackPlan:
    context = state_or_context if isinstance(state_or_context, DecisionContext) else build_decision_context(state_or_context)
    existing_labels = [label for meld in context.existing_melds for label in meld.labels]
    red_cards = [card.label for card in context.card_instances if card.label in RED_LABELS]
    red_cards.extend(label for label in existing_labels if label in RED_LABELS)
    black_cards = [card.label for card in context.card_instances if card.label not in RED_LABELS and not card.is_wildcard]
    black_cards.extend(label for label in existing_labels if label not in RED_LABELS and label != WILD_LABEL)
    red_count = len(red_cards)
    black_count = len(black_cards)
    mode = "ignore_red_black"
    risk_notes: list[str] = []
    outcome = classify_red_black(red_count, context.rules, card_count=red_count + black_count)
    existing_red_count = sum(label in RED_LABELS for label in existing_labels)
    locked_red_count = existing_red_count + sum(
        label in RED_LABELS
        for meld in allocation.locked_melds
        for label in meld.labels
    )
    distances = red_black_target_distance(red_count, locked_red_count, context.rules)
    if str(context.rules.get("rules", {}).get("red_black_mode", "")).lower() in {"none", "off", "disabled"}:
        risk_notes.append("red_black_mode disabled; do not chase color blindly")
    elif outcome.kind is not None:
        mode = {
            "red_hu": "red_plan",
            "black_hu": "black_plan",
            "one_red_hu": "one_red_plan",
        }[outcome.kind]
    elif distances:
        target, distance = min(distances.items(), key=lambda item: (item[1], item[0]))
        mode = {
            "red_hu": "red_plan",
            "black_hu": "black_plan",
            "one_red_hu": "one_red_plan",
        }[target] if distance <= 3 else "mixed"
        if distance > 3:
            risk_notes.append("too far from red/black special thresholds; prioritize xi and ting")
    protected_red = _ordered_unique(
        [label for meld in allocation.protected_melds for label in meld.labels if label in RED_LABELS]
    )
    if mode == "red_plan":
        keep = protected_red
        release = [card.label for card in allocation.free_discard_instances if card.label not in RED_LABELS]
    elif mode == "one_red_plan":
        anchor = (protected_red or red_cards)[:1]
        keep = anchor
        release = [card.label for card in allocation.free_discard_instances if card.label in RED_LABELS and card.label not in keep]
    elif mode == "black_plan":
        keep = []
        release = [card.label for card in allocation.free_discard_instances if card.label in RED_LABELS]
    else:
        keep = protected_red
        release = [card.label for card in allocation.free_discard_instances if card.label not in keep]
    nearest_distance = min(distances.values(), default=99)
    bonus_estimate = max(0, 3 - nearest_distance) * 20.0
    return RedBlackPlan(
        red_count=red_count,
        black_count=black_count,
        red_potential=float(max(0, 13 - red_count)),
        black_potential=float(red_count if locked_red_count == 0 else 0),
        mode=mode,
        core_cards=keep,
        cards_to_keep=keep,
        cards_to_release=release,
        bonus_estimate=float(bonus_estimate),
        risk_notes=risk_notes,
    )


def estimate_ting_distance(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    allocation: StructureAllocation,
    *,
    current_hu: HuResult | None = None,
) -> TingEstimate:
    context = state_or_context if isinstance(state_or_context, DecisionContext) else build_decision_context(state_or_context)
    rules = context.rules
    xi = estimate_potential_xi(allocation, rules, context.existing_melds)
    minimum = int(rules.get("rules", {}).get("min_xi", 9))
    confirmed_gap = max(0, minimum - xi.confirmed_xi)
    potential_gap = max(0, minimum - xi.potential_xi)
    current_hu = current_hu or check_hu(context, allocation, rules)
    max_exact_cards = int(rules.get("ting", {}).get("max_exact_draw_enum_cards", 13))
    exact_probe_enabled = len(context.card_instances) <= max_exact_cards
    draw_results: list[dict[str, Any]] = []
    hu_cards: list[str] = []
    improving_cards: list[str] = []
    hu_xi_by_label: dict[str, int] = {}
    hu_score_by_label: dict[str, float] = {}
    best_xi_if_hu = current_hu.total_xi if current_hu.can_hu else 0
    best_score_if_hu = current_hu.score_now if current_hu.can_hu else 0.0
    for label in _ting_probe_labels(context):
        after_context = _context_after_ting_draw(context, label)
        (
            after_confirmed_xi,
            after_potential_xi,
            after_weak_waits,
            after_hu,
        ) = _ting_draw_features(
            after_context,
            rules,
            exact_probe_enabled=exact_probe_enabled,
        )
        after_confirmed_gap = max(0, minimum - after_confirmed_xi)
        after_potential_gap = max(0, minimum - after_potential_xi)
        improves = (
            after_confirmed_gap < confirmed_gap
            or after_potential_gap < potential_gap
            or after_potential_xi > xi.potential_xi
            or after_weak_waits > len(allocation.weak_potentials)
        )
        if after_hu is not None and after_hu[0]:
            hu_cards.append(label)
            hu_xi_by_label[label] = after_hu[1]
            hu_score_by_label[label] = after_hu[2]
            best_xi_if_hu = max(best_xi_if_hu, after_hu[1])
            best_score_if_hu = max(best_score_if_hu, after_hu[2])
            improves = True
        if improves:
            improving_cards.append(label)
        draw_results.append(
            {
                "label": label,
                "can_hu": bool(after_hu[0]) if after_hu is not None else False,
                "potential_xi": after_potential_xi,
                "confirmed_xi": after_confirmed_xi,
                "potential_gap": after_potential_gap,
                "weak_waits": after_weak_waits,
            }
        )

    if not improving_cards:
        improving_cards = list(xi.cards_that_improve_xi)
    hu_cards = _ordered_unique(hu_cards)
    improving_cards = _ordered_unique(improving_cards)
    if hu_cards:
        expected_xi_if_hu = round(sum(hu_xi_by_label[label] for label in hu_cards) / len(hu_cards))
        expected_score_if_hu = sum(hu_score_by_label[label] for label in hu_cards) / len(hu_cards)
    else:
        if not best_xi_if_hu:
            best_xi_if_hu = max(minimum, xi.potential_xi, *(item["potential_xi"] for item in draw_results))
        if not best_score_if_hu:
            best_score_if_hu = float(best_xi_if_hu * 40)
        expected_xi_if_hu = best_xi_if_hu
        expected_score_if_hu = best_score_if_hu
    distance = _ting_distance_from_features(
        can_hu_now=current_hu.can_hu,
        hu_card_count=len(hu_cards),
        improving_card_count=len(improving_cards),
        potential_gap=potential_gap,
        orphan_count=len(allocation.orphan_cards),
    )
    waiting_cards = hu_cards if hu_cards else (improving_cards if distance <= 1 else [])
    notes = [
        f"confirmed_xi_gap={confirmed_gap}",
        f"potential_xi_gap={potential_gap}",
        f"weak_potentials={len(allocation.weak_potentials)}",
        f"draw_probe_count={len(draw_results)}",
    ]
    if exact_probe_enabled:
        notes.append("exact_hu_draw_probe=enabled")
    else:
        notes.append(f"exact_hu_draw_probe=skipped_hand_size>{max_exact_cards}")
    return TingEstimate(
        shanten_like_distance=distance,
        is_ting=current_hu.can_hu or bool(hu_cards),
        waiting_cards=waiting_cards,
        improving_cards=improving_cards,
        hu_cards=hu_cards,
        expected_xi_if_hu=int(max(minimum, expected_xi_if_hu)),
        expected_score_if_hu=float(expected_score_if_hu),
        notes=notes,
        hu_xi_by_label={label: hu_xi_by_label[label] for label in hu_cards},
        hu_score_by_label={label: hu_score_by_label[label] for label in hu_cards},
    )


def analyze_hand(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    allocation: StructureAllocation | None = None,
) -> HandAnalysis:
    context = state_or_context if isinstance(state_or_context, DecisionContext) else build_decision_context(state_or_context)
    allocation = allocation or allocate_hand_structures(context)
    rules = context.rules
    controlled_melds = [*context.existing_melds, *allocation.protected_melds]
    xi_result = calculate_xi(controlled_melds, rules)
    xi_potential = estimate_potential_xi(allocation, rules, context.existing_melds)
    hu_result = check_hu(context, allocation, rules)
    red_black = evaluate_red_black_plan(context, allocation)
    wildcard_value = evaluate_wildcard_value(context, allocation)
    ting = estimate_ting_distance(context, allocation, current_hu=hu_result)
    red_count = red_black.red_count
    black_count = red_black.black_count
    wildcard_count = wildcard_value.wildcard_count
    if hu_result.can_hu:
        direction = "quick_hu"
    elif wildcard_count >= 2:
        direction = "wildcard_big_hand"
    elif red_black.mode == "red_plan":
        direction = "red_plan"
    elif red_black.mode == "black_plan":
        direction = "black_plan"
    elif xi_result.xi_gap:
        direction = "xi_build"
    else:
        direction = "mixed"
    remaining = context.remaining_deck_count
    defense_remaining, danger_remaining = _room_phase_thresholds(rules)
    if remaining is not None and remaining <= danger_remaining:
        mode = "danger_defense"
    elif remaining is not None and remaining <= defense_remaining:
        mode = "defense"
    elif xi_result.xi_gap <= 3:
        mode = "attack"
    else:
        mode = "balanced"
    hand_value = float(
        xi_result.total_xi * 40
        + xi_potential.potential_xi * 20
        + len(controlled_melds) * 50
        + wildcard_value.future_flexibility_score
        + red_black.bonus_estimate
    )
    return HandAnalysis(
        confirmed_xi=xi_result.total_xi,
        potential_xi=xi_potential.potential_xi,
        min_xi=xi_result.min_xi,
        xi_gap=xi_result.xi_gap,
        complete_melds=list(controlled_melds),
        protected_melds=list(controlled_melds),
        potential_melds=list(allocation.potential_melds),
        weak_potentials=list(allocation.weak_potentials),
        orphan_cards=list(allocation.orphan_cards),
        red_count=red_count,
        black_count=black_count,
        wildcard_count=wildcard_count,
        red_black_plan=red_black,
        wildcard_value=wildcard_value,
        ting_estimate=ting,
        hand_direction=direction,
        attack_defense_mode=mode,
        hand_value=round(hand_value, 3),
        hu_result=hu_result,
        xi_result=xi_result,
        xi_potential=xi_potential,
        notes=list(allocation.allocation_notes),
    )


def evaluate_discard_danger(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    card: CardInstance,
) -> DangerScore:
    context = state_or_context if isinstance(state_or_context, DecisionContext) else build_decision_context(state_or_context)
    memory = context.state.get("memory") if isinstance(context.state.get("memory"), dict) else None
    visible: list[str] = []
    visible_raw = context.state.get("visible_cards") or context.state.get("discards") or []
    if isinstance(visible_raw, list):
        visible.extend(str(item) for item in visible_raw)
    visible.extend(visible_labels_from_memory(memory))
    danger, reasons = legacy_danger_score(
        card.label,
        visible_labels=visible,
        remaining_deck_count=context.remaining_deck_count,
        rules=context.rules,
    )
    defense_remaining, _danger_remaining = _room_phase_thresholds(context.rules)
    seat_profiles = (
        infer_opponent_profiles(memory)
        if memory and memory.get("seat_aware_danger")
        else []
    )
    if seat_profiles:
        profile_rows: list[tuple[float, str, list[str]]] = []
        for profile in seat_profiles:
            profile_danger = 0.0
            profile_reasons: list[str] = []
            profile_name = str(profile.get("profile") or "unknown")
            seat_name = (
                "下家"
                if profile.get("is_next_seat")
                else f"座位{profile.get('seat')}"
            )
            if (
                profile_name == "fast_meld"
                and context.remaining_deck_count is not None
                and context.remaining_deck_count <= defense_remaining
            ):
                profile_danger += 40
                profile_reasons.append(f"{seat_name}落地多且后盘，放牌风险提高")
            if profile_name == "red_value" and card.label in RED_LABELS:
                profile_danger += 45 + max(0, int(profile.get("red_pressure", 0))) * 10
                profile_reasons.append(f"{seat_name}红牌压力高，红牌危险加重")
            if profile_name == "pair_triplet_value" and visible.count(card.label) <= 1:
                profile_danger += 30
                profile_reasons.append(f"{seat_name}可能做对子/刻子价值，未见同牌危险")
            profile_rows.append((profile_danger, profile_name, profile_reasons))
        profile_danger, profile_name, profile_reasons = max(
            profile_rows,
            key=lambda item: item[0],
        )
        danger += profile_danger
        reasons.extend(profile_reasons)
    else:
        profile = infer_opponent_profile(memory)
        profile_name = profile.get("profile")
        if (
            profile_name == "fast_meld"
            and context.remaining_deck_count is not None
            and context.remaining_deck_count <= defense_remaining
        ):
            danger += 40
            reasons.append("对手落地多且后盘，放牌风险提高")
        if profile_name == "red_value" and card.label in RED_LABELS:
            danger += 45 + max(0, int(profile.get("red_pressure", 0))) * 10
            reasons.append("对手红牌压力高，红牌危险加重")
        if profile_name == "pair_triplet_value" and visible.count(card.label) <= 1:
            danger += 30
            reasons.append("对手可能做对子/刻子价值，未见同牌危险")
    if seat_profiles:
        meld_rows = opponent_meld_risks_by_seat(
            card.label,
            memory,
            remaining_deck_count=context.remaining_deck_count,
            rules=context.rules,
        )
        meld_row = max(meld_rows, key=lambda item: item["danger_score"])
        meld_risk = float(meld_row["danger_score"])
        seat_name = "下家" if meld_row.get("is_next_seat") else f"座位{meld_row.get('seat')}"
        meld_reasons = [f"{seat_name}：{reason}" for reason in meld_row["reasons"]]
    else:
        meld_risk, meld_reasons = opponent_meld_risk(
            card.label,
            memory,
            remaining_deck_count=context.remaining_deck_count,
            rules=context.rules,
        )
    danger += meld_risk
    reasons.extend(meld_reasons)
    risk_type = (
        "opponent_pao_or_minglong"
        if meld_risk > 0
        else "red_or_key"
        if card.label in RED_LABELS
        else "opponent_profile"
        if profile_name not in {None, "unknown", "balanced"}
        else "late_unknown"
        if reasons
        else "low"
    )
    return DangerScore(danger, risk_type, reasons or ["early_stage_low_danger"])


def generate_legal_actions(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    allocation: StructureAllocation | None = None,
    analysis: HandAnalysis | None = None,
) -> tuple[list[LegalAction], list[LegalAction]]:
    context = state_or_context if isinstance(state_or_context, DecisionContext) else build_decision_context(state_or_context)
    allocation = allocation or allocate_hand_structures(context)
    analysis = analysis or analyze_hand(context, allocation)
    legal_types = set(context.legal_action_types or ["DISCARD", "PASS"])
    actions: list[LegalAction] = []
    rejected: list[LegalAction] = []
    seq = 0

    def next_id() -> str:
        nonlocal seq
        seq += 1
        return f"a{seq:03d}"

    def reject_missing_button(action_type: str, *, label: str | None = None) -> None:
        rejected.append(
            LegalAction(
                next_id(),
                action_type,
                allowed=False,
                source="button",
                label=label,
                requires_button=True,
                reject_reason="button_not_found",
            )
        )

    auto_meld = sorted(legal_types & AUTO_MELD_ACTIONS)
    if auto_meld:
        auto_type = auto_meld[0]
        auto_payload = _legal_action_payload(context, auto_type)
        actions.append(
            LegalAction(
                next_id(),
                "WAIT_AUTO_MELD",
                source="system",
                label=auto_type,
                option_cards=_auto_meld_payload_cards(context, auto_payload),
                requires_button=False,
                reject_reason=None,
            )
        )
    if "PASS" in legal_types or {"CHI", "PENG"} & legal_types:
        actions.append(LegalAction(next_id(), "PASS", source="button", requires_button=True))
    if "WAIT" in legal_types:
        actions.append(LegalAction(next_id(), "WAIT", source="system", requires_button=False))
    if "HU" in legal_types:
        if not _button_visible_or_unknown(context, "HU"):
            reject_missing_button("HU")
        else:
            allowed = analysis.hu_result.can_hu or _button_visible(context, "HU")
            action = LegalAction(
                next_id(),
                "HU",
                allowed=allowed,
                source="button",
                requires_button=True,
                reject_reason=None if allowed else analysis.hu_result.reject_reason or "hu_xi_not_enough",
            )
            (actions if allowed else rejected).append(action)
    if "DISCARD" in legal_types:
        protected_ids = set(allocation.hard_protected_instances) | set(allocation.soft_protected_instances)
        for card in allocation.card_instances:
            if not card.clickable:
                rejected.append(
                    LegalAction(
                        next_id(),
                        "DISCARD",
                        allowed=False,
                        source="hand",
                        card_id=card.id,
                        label=card.label,
                        reject_reason="selected_card_not_clickable",
                    )
                )
                continue
            actions.append(
                LegalAction(
                    next_id(),
                    "DISCARD",
                    allowed=True,
                    source="hand",
                    card_id=card.id,
                    label=card.label,
                    reject_reason="protected_candidate" if card.id in protected_ids else None,
                )
            )
    if "CHI" in legal_types:
        if not _button_visible_or_unknown(context, "CHI"):
            reject_missing_button("CHI")
        else:
            if not context.chi_options:
                metadata = context.state.get("metadata") or {}
                if not metadata.get("opponent_priority_pending"):
                    actions.append(
                        LegalAction(
                            next_id(),
                            "EXPAND_CHI_OPTIONS",
                            source="button",
                            requires_button=True,
                            requires_option=False,
                            requires_followup_discard=False,
                            reject_reason=None,
                        )
                    )
                actions.append(
                    LegalAction(
                        next_id(),
                        "CHI",
                        source="button",
                        requires_button=True,
                        requires_option=False,
                        requires_followup_discard=False,
                        reject_reason=None,
                    )
                )
            for index, option in enumerate(context.chi_options, start=1):
                labels = normalize_cards([str(label) for label in option.get("labels", [])])
                confidence = float(option.get("confidence", 1.0))
                actions.append(
                    LegalAction(
                        next_id(),
                        "CHI",
                        source="button",
                        option_id=str(option.get("option_id") or f"chi_{index:03d}"),
                        option_cards=labels,
                        requires_button=True,
                        requires_option=True,
                        requires_followup_discard=True,
                        uncertainty=max(0.0, 1.0 - confidence),
                    )
                )
    if "PENG" in legal_types:
        pending_label = _pending_action_label(context)
        if not _button_visible_or_unknown(context, "PENG"):
            reject_missing_button("PENG", label=pending_label)
        else:
            actions.append(
                LegalAction(
                    next_id(),
                    "PENG",
                    source="button",
                    label=pending_label,
                    requires_button=True,
                    requires_followup_discard=True,
                )
            )
    return actions, rejected


def _button_visible_or_unknown(context: DecisionContext, action_type: str) -> bool:
    if "buttons" not in context.state:
        return True
    return _button_visible(context, action_type)


def _button_visible(context: DecisionContext, action_type: str) -> bool:
    expected = action_type.lower()
    aliases = {
        "hu": {"hu", "胡"},
        "chi": {"chi", "吃"},
        "peng": {"peng", "碰"},
    }.get(expected, {expected})
    for button in context.buttons:
        if not isinstance(button, dict):
            continue
        name = str(button.get("name") or button.get("type") or button.get("label") or "").strip().lower()
        if name in aliases or name.upper() == action_type:
            return True
    return False


def _legal_action_payload(context: DecisionContext, action_type: str) -> dict[str, Any]:
    for item in context.state.get("legal_actions") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").upper() == action_type:
            return item
    return {}


def _auto_meld_payload_cards(context: DecisionContext, payload: dict[str, Any]) -> list[str]:
    for key in ("labels", "cards", "meld_cards", "option_cards"):
        values = payload.get(key)
        if isinstance(values, list):
            labels = normalize_cards([str(item) for item in values])
            if labels:
                return labels
    label = (
        payload.get("label")
        or payload.get("card")
        or context.state.get("pending_card")
        or context.state.get("external_card")
        or context.state.get("drawn_card")
    )
    labels = normalize_cards([str(label)]) if label else []
    return labels


def _discard_pools(allocation: StructureAllocation) -> tuple[str, set[str]]:
    hard_ids = {card_id for card_id in allocation.hard_protected_instances}
    clickable_non_hard = {
        card.id for card in allocation.card_instances if card.id not in hard_ids and card.clickable
    }
    if clickable_non_hard:
        return "non_hard_candidates", clickable_non_hard
    clickable_hard = {
        card.id for card in allocation.card_instances if card.id in hard_ids and card.clickable
    }
    if clickable_hard:
        return "forced_break_hard_protection", clickable_hard
    return "safe_halt", set()


def simulate_action(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    action: LegalAction,
    context: DecisionContext | None = None,
    allocation: StructureAllocation | None = None,
    followup_discard_cache: dict[str, dict[str, Any] | None] | None = None,
) -> SimulatedResult:
    context = context or (
        state_or_context if isinstance(state_or_context, DecisionContext) else build_decision_context(state_or_context)
        )
    allocation = allocation or allocate_hand_structures(context)
    if action.type in {"PASS", "HU", "WAIT", "WAIT_AUTO_MELD"}:
        debug: dict[str, Any] = {
            "state_change": "none",
            "hand_count_after": len(context.normalized_hand),
        }
        notes = [f"{action.type.lower()}_simulated", "keeps_current_hand"]
        if action.type == "HU":
            hu_result = check_hu(context, allocation, context.rules)
            debug["hu_result"] = hu_result.to_dict()
            notes.append("settles_round_if_allowed")
        if action.type in {"WAIT", "WAIT_AUTO_MELD"}:
            notes.append("wait_for_ui_or_auto_meld_resolution")
        return SimulatedResult(
            action=action,
            hand_after=list(context.normalized_hand),
            allocation_after=allocation,
            consumed_card_ids=[],
            breaks_hard=False,
            breaks_soft=False,
            followup_discard=None,
            notes=notes,
            debug_details=debug,
        )
    if action.type == "DISCARD" and action.card_id:
        after_cards = [card for card in context.card_instances if card.id != action.card_id]
        after_state = {
            **context.state,
            "hand": [card.label for card in after_cards],
            "hand_details": [card.to_dict() | {"name": card.label} for card in after_cards],
        }
        after_context = build_decision_context(after_state, rules=context.rules)
        after_allocation = allocate_hand_structures(after_context)
        breaks_hard = action.card_id in set(allocation.hard_protected_instances)
        breaks_soft = action.card_id in set(allocation.soft_protected_instances)
        return SimulatedResult(
            action=action,
            hand_after=after_context.normalized_hand,
            allocation_after=after_allocation,
            consumed_card_ids=[action.card_id],
            breaks_hard=breaks_hard,
            breaks_soft=breaks_soft,
            followup_discard=None,
            notes=["discard_simulated"],
            debug_details={
                "discarded_card_id": action.card_id,
                "discarded_label": action.label,
            },
        )
    if action.type in {"CHI", "PENG"}:
        candidates = (
            _chi_consumption_candidates(context, action)
            if action.type == "CHI"
            else _peng_consumption_candidates(context, action)
        )
        if not candidates:
            return SimulatedResult(
                action=action,
                hand_after=list(context.normalized_hand),
                allocation_after=allocation,
                consumed_card_ids=[],
                breaks_hard=False,
                breaks_soft=False,
                followup_discard=None,
                notes=[f"{action.type.lower()}_simulation_failed", "no_consumption_candidate"],
                debug_details={"candidate_count": 0},
            )
        weights = context.rules.get("weights") or context.rules.get("ai_weights", {})
        simulated_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            consumed_ids = candidate["consumed_card_ids"]
            meld_cards = candidate.get("meld_cards") or candidate["option_cards"]
            meld_type = _meld_type_for_option(meld_cards, action.type)
            structure_cost, cost_reasons, breaks_melds, breaks_hard, breaks_soft = _consumption_cost(
                consumed_ids,
                allocation,
                weights,
            )
            after_context = _context_after_consumption(
                context,
                consumed_ids,
                meld_type=meld_type,
                meld_cards=meld_cards,
                plan_melds=candidate.get("plan_melds"),
            )
            after_allocation = allocate_hand_structures(after_context)
            followup = _cached_best_followup_discard(
                after_context,
                followup_discard_cache,
            )
            simulated_candidates.append(
                {
                    "candidate": candidate,
                    "after_context": after_context,
                    "after_allocation": after_allocation,
                    "followup_discard": followup,
                    "structure_cost": structure_cost,
                    "cost_reasons": cost_reasons,
                    "breaks_melds": breaks_melds,
                    "breaks_hard": breaks_hard,
                    "breaks_soft": breaks_soft,
                    "meld_type": meld_type,
                }
            )
        best = sorted(
            simulated_candidates,
            key=lambda item: (
                bool(item["breaks_hard"]),
                bool(item["breaks_soft"]),
                item["structure_cost"],
                item["followup_discard"] is None,
            ),
        )[0]
        candidate = best["candidate"]
        notes = [
            f"{action.type.lower()}_simulated",
            f"meld_type={best['meld_type']}",
            "followup_discard_available" if best["followup_discard"] else "followup_discard_missing",
        ]
        return SimulatedResult(
            action=action,
            hand_after=list(best["after_context"].normalized_hand),
            allocation_after=best["after_allocation"],
            consumed_card_ids=list(candidate["consumed_card_ids"]),
            breaks_hard=bool(best["breaks_hard"]),
            breaks_soft=bool(best["breaks_soft"]),
            followup_discard=best["followup_discard"],
            notes=notes,
            debug_details={
                "option_cards": list(candidate["option_cards"]),
                "external_label": candidate["external_label"],
                "consumed_from_hand": list(candidate["consumed_labels"]),
                "meld_type": best["meld_type"],
                "compare_groups": list(candidate.get("compare_groups") or []),
                "plan_melds": list(candidate.get("plan_melds") or []),
                "structure_cost": best["structure_cost"],
                "cost_reasons": list(best["cost_reasons"]),
                "breaks_melds": list(best["breaks_melds"]),
                "candidate_count": len(simulated_candidates),
            },
        )
    return SimulatedResult(
        action=action,
        hand_after=list(context.normalized_hand),
        allocation_after=allocation,
        consumed_card_ids=[],
        breaks_hard=False,
        breaks_soft=False,
        followup_discard=None,
        notes=[f"{action.type.lower()} simulation is conservative placeholder"],
    )


def evaluate_action_ev(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    action: LegalAction,
    context: DecisionContext | None = None,
    allocation: StructureAllocation | None = None,
    analysis: HandAnalysis | None = None,
    post_discard_analysis_cache: (
        dict[tuple[tuple[object, ...], ...], HandAnalysis] | None
    ) = None,
) -> ActionEval:
    context = context or (
        state_or_context if isinstance(state_or_context, DecisionContext) else build_decision_context(state_or_context)
    )
    allocation = allocation or allocate_hand_structures(context)
    analysis = analysis or analyze_hand(context, allocation)
    weights = context.rules.get("weights") or context.rules.get("ai_weights", {})
    pool_stage, pool_ids = _discard_pools(allocation)
    if action.type == "HU":
        ui_hu_visible = _button_visible(context, "HU")
        rule_legal = analysis.hu_result.can_hu or ui_hu_visible
        strategy_accepts = analysis.hu_result.decision == "hu_now" or (
            not analysis.hu_result.can_hu and ui_hu_visible
        )
        allowed = action.allowed and rule_legal and strategy_accepts
        ev = float(weights.get("immediate_hu", 10000)) if allowed else HARD_BREAK_EV
        sim = simulate_action(context, action, context, allocation)
        reason = analysis.hu_result.reason
        if allowed and not analysis.hu_result.can_hu and ui_hu_visible:
            reason = f"visible_hu_button_trusted; local_check={analysis.hu_result.reject_reason or analysis.hu_result.reason}"
        return ActionEval(
            action=action,
            allowed=allowed,
            ev=ev,
            score_gain=ev,
            hand_value_after=analysis.hand_value,
            xi_gain=analysis.confirmed_xi,
            ting_gain=0,
            red_black_gain=analysis.red_black_plan.bonus_estimate,
            wildcard_gain=0,
            structure_loss=0,
            danger_loss=0,
            opponent_gain_risk=0,
            uncertainty_penalty=0,
            reject_reason=None if allowed else analysis.hu_result.reject_reason or "hu_xi_not_enough",
            reason=reason,
            debug_details={"hu_result": analysis.hu_result.to_dict(), "simulation": sim.to_dict()},
            immediate_score_gain=ev if allowed else 0,
        )
    if action.type == "PASS":
        sim = simulate_action(context, action, context, allocation)
        pending = _pending_action_label(context)
        information_set = _information_set_draw_value(
            context,
            analysis,
            newly_visible=[pending] if pending else [],
        )
        return ActionEval(
            action=action,
            allowed=True,
            ev=round(analysis.hand_value + information_set["value"], 3),
            score_gain=0,
            hand_value_after=analysis.hand_value,
            xi_gain=0,
            ting_gain=0,
            red_black_gain=0,
            wildcard_gain=0,
            structure_loss=0,
            danger_loss=0,
            opponent_gain_risk=0,
            uncertainty_penalty=action.uncertainty * 100,
            reject_reason=None,
            reason="PASS baseline keeps current structure",
            debug_details={
                "candidate_stage": "pass",
                "simulation": sim.to_dict(),
                "information_set": information_set,
            },
        )
    if action.type == "EXPAND_CHI_OPTIONS":
        sim = simulate_action(context, action, context, allocation)
        weights = context.rules.get("weights") or context.rules.get("ai_weights", {})
        probe_bonus = float(weights.get("chi_option_probe_bonus", 0.001))
        pending = _pending_action_label(context)
        information_set = _information_set_draw_value(
            context,
            analysis,
            newly_visible=[pending] if pending else [],
        )
        pass_ev = analysis.hand_value + information_set["value"]
        return ActionEval(
            action=action,
            allowed=True,
            ev=round(pass_ev + probe_bonus, 3),
            score_gain=0,
            hand_value_after=analysis.hand_value,
            xi_gain=0,
            ting_gain=0,
            red_black_gain=0,
            wildcard_gain=0,
            structure_loss=0,
            danger_loss=0,
            opponent_gain_risk=0,
            uncertainty_penalty=0,
            reject_reason=None,
            reason="EXPAND_CHI_OPTIONS reveal candidates; no chi meld selected",
            debug_details={
                "candidate_stage": "expand_chi_options",
                "pass_ev": round(pass_ev, 3),
                "response_ev": round(pass_ev, 3),
                "chi_ev": round(pass_ev, 3),
                "ev_delta_vs_pass": 0.0,
                "min_ev_gain": 0.0,
                "consumed_card_ids": [],
                "consumed_from_hand": [],
                "followup_discard": None,
                "information_set": information_set,
                "simulation": sim.to_dict(),
            },
        )
    if action.type == "WAIT":
        sim = simulate_action(context, action, context, allocation)
        wait_penalty = float(weights.get("wait_action_penalty", 5))
        return ActionEval(
            action=action,
            allowed=True,
            ev=round(analysis.hand_value - wait_penalty, 3),
            score_gain=0,
            hand_value_after=analysis.hand_value,
            xi_gain=0,
            ting_gain=0,
            red_black_gain=0,
            wildcard_gain=0,
            structure_loss=0,
            danger_loss=0,
            opponent_gain_risk=0,
            uncertainty_penalty=action.uncertainty * 100,
            reject_reason=None,
            reason="WAIT keeps current state until UI changes",
            debug_details={
                "candidate_stage": "wait",
                "simulation": sim.to_dict(),
                "wait_penalty": wait_penalty,
            },
            safety_gain=-wait_penalty,
        )
    if action.type == "WAIT_AUTO_MELD":
        sim = simulate_action(context, action, context, allocation)
        auto_details = _auto_meld_ev_details(context, action, allocation, analysis, weights)
        auto_debug = deepcopy(auto_details["debug_details"])
        auto_debug["simulation"] = sim.to_dict()
        return ActionEval(
            action=action,
            allowed=True,
            ev=auto_details["ev"],
            score_gain=auto_details["score_gain"],
            hand_value_after=analysis.hand_value + auto_details["score_gain"],
            xi_gain=auto_details["xi_gain"],
            ting_gain=0,
            red_black_gain=auto_details["red_black_gain"],
            wildcard_gain=0,
            structure_loss=0,
            danger_loss=0,
            opponent_gain_risk=auto_details["opponent_gain_risk"],
            uncertainty_penalty=0,
            reject_reason=None,
            reason=auto_details["reason"],
            debug_details=auto_debug,
        )
    if action.type in {"CHI", "PENG"}:
        return _evaluate_response_ev(context, action, analysis, allocation)
    if action.type != "DISCARD":
        return ActionEval(
            action=action,
            allowed=False,
            ev=HARD_BREAK_EV,
            score_gain=0,
            hand_value_after=analysis.hand_value,
            xi_gain=0,
            ting_gain=0,
            red_black_gain=0,
            wildcard_gain=0,
            structure_loss=0,
            danger_loss=0,
            opponent_gain_risk=0,
            uncertainty_penalty=0,
            reject_reason="unsupported_action",
            reason="unsupported action type",
            debug_details={},
        )

    card = next((item for item in context.card_instances if item.id == action.card_id), None)
    if card is None:
        return _rejected_eval(action, "selected_card_not_found", analysis)
    if not card.clickable:
        return _rejected_eval(action, "selected_card_not_clickable", analysis)
    if pool_stage == "safe_halt":
        return _rejected_eval(action, "no_clickable_cards", analysis)
    sim = simulate_action(context, action, context, allocation)
    after_context = _context_after_discard_id(context, action.card_id)
    after_analysis = analysis
    if sim.allocation_after:
        cache_key = tuple(
            sorted(
                (
                    item.label,
                    item.source,
                    (
                        float(item.confidence)
                        if item.confidence is not None
                        else -1.0
                    ),
                    item.clickable,
                )
                for item in after_context.card_instances
            )
        )
        if post_discard_analysis_cache is not None:
            after_analysis = post_discard_analysis_cache.get(cache_key)
        if after_analysis is None or after_analysis is analysis:
            after_analysis = analyze_hand(
                after_context,
                sim.allocation_after,
            )
            if post_discard_analysis_cache is not None:
                post_discard_analysis_cache[cache_key] = after_analysis
    would_break_ting = analysis.ting_estimate.is_ting and not after_analysis.ting_estimate.is_ting
    effective_pool_stage = pool_stage
    if pool_stage == "non_hard_candidates":
        free_ids = {item.id for item in allocation.free_discard_instances}
        weak_ids = {card_id for meld in allocation.weak_potentials for card_id in meld.card_ids}
        if card.id in free_ids - weak_ids:
            effective_pool_stage = "normal_unprotected"
        elif card.id in free_ids:
            effective_pool_stage = "weak_potential"
        else:
            effective_pool_stage = "break_soft_protection"
    if card.id not in pool_ids:
        if analysis.ting_estimate.is_ting and after_analysis.ting_estimate.is_ting and card.id not in set(
            allocation.hard_protected_instances
        ):
            effective_pool_stage = "ting_preserve_override"
        else:
            reject = (
                "hard_protected_exact_triplet"
                if card.id in allocation.hard_protected_instances
                else "not_in_current_candidate_pool"
            )
            return _rejected_eval(action, reject, analysis, candidate_stage=pool_stage)
    danger = evaluate_discard_danger(context, card)
    ting_value_delta = _ting_value_delta(
        analysis.ting_estimate,
        after_analysis.ting_estimate,
        weights,
        include_wait_breadth=False,
    )
    structure_loss = 0.0
    reasons: list[str] = []
    if sim.breaks_hard:
        structure_loss += float(weights.get("hard_protected_break_penalty", 10000))
        reasons.append("breaks hard protected structure")
    if sim.breaks_soft:
        structure_loss += _soft_structure_loss(card.id, allocation, weights)
        reasons.append("breaks soft protected structure")
    if effective_pool_stage == "normal_unprotected":
        score_gain = float(weights.get("free_card_discard_bonus", 200))
        reasons.append(f"{card.label} is free unprotected")
    elif effective_pool_stage == "weak_potential":
        score_gain = float(weights.get("weak_potential_discard_bonus", 120))
        reasons.append(f"{card.label} is weak potential only")
    elif effective_pool_stage == "break_soft_protection":
        score_gain = 0.0
        reasons.append("all free cards exhausted; choose lowest soft loss")
    elif effective_pool_stage == "ting_preserve_override":
        score_gain = 0.0
        reasons.append(f"{card.label} keeps current ting; allow soft structure break over breaking ting")
    else:
        score_gain = -structure_loss
        reasons.append("forced hard break because no other clickable candidate")
    if card.is_wildcard:
        structure_loss += float(weights.get("wildcard_discard_penalty", 10000))
        reasons.append("王 hard protected; discard forbidden except forced")
    shape_tiebreak, shape_reasons = _discard_duplicate_shape_tiebreak(card, allocation, weights)
    if effective_pool_stage == "forced_break_hard_protection":
        shape_tiebreak = 0.0
        shape_reasons = []
    reasons.extend(shape_reasons)
    if would_break_ting:
        reasons.append("discard loses current ting estimate; only allowed when no discard preserves ting")
    danger_loss = _scaled_discard_danger_loss(danger.danger_score, weights)
    information_set = _information_set_draw_value(
        after_context,
        after_analysis,
        newly_visible=[card.label],
    )
    ev = (
        after_analysis.hand_value
        + score_gain
        + ting_value_delta
        + shape_tiebreak
        - structure_loss
        - danger_loss
        - action.uncertainty * 100
        + information_set["value"]
    )
    return ActionEval(
        action=action,
        allowed=True,
        ev=round(ev, 3),
        score_gain=round(score_gain, 3),
        hand_value_after=after_analysis.hand_value,
        xi_gain=after_analysis.confirmed_xi - analysis.confirmed_xi,
        ting_gain=analysis.ting_estimate.shanten_like_distance - after_analysis.ting_estimate.shanten_like_distance,
        red_black_gain=after_analysis.red_black_plan.bonus_estimate - analysis.red_black_plan.bonus_estimate,
        wildcard_gain=after_analysis.wildcard_value.future_flexibility_score
        - analysis.wildcard_value.future_flexibility_score,
        structure_loss=round(structure_loss, 3),
        danger_loss=round(danger_loss, 3),
        opponent_gain_risk=0,
        uncertainty_penalty=round(action.uncertainty * 100, 3),
        reject_reason=None,
        reason="；".join(reasons + danger.reasons),
        debug_details={
            "candidate_stage": effective_pool_stage,
            "danger": danger.to_dict(),
            "raw_danger_loss": round(danger.danger_score, 3),
            "simulation": sim.to_dict(),
            "ting_value_delta": ting_value_delta,
            "information_set": information_set,
            "shape_tiebreak": round(shape_tiebreak, 3),
            "ting_before": analysis.ting_estimate.to_dict(),
            "ting_after": after_analysis.ting_estimate.to_dict(),
            "would_break_ting": would_break_ting,
        },
    )


def _scaled_discard_danger_loss(raw_score: float, weights: dict[str, Any]) -> float:
    score = float(raw_score or 0.0)
    if score <= 0:
        return 0.0
    high_threshold = float(weights.get("high_danger_threshold", 250))
    high_multiplier = float(weights.get("high_danger_multiplier", 1.6))
    return score * high_multiplier if score >= high_threshold else score


def _information_set_draw_value(
    context: DecisionContext,
    analysis: HandAnalysis | TingEstimate,
    *,
    newly_visible: list[str] | None = None,
) -> dict[str, Any]:
    remaining = _information_remaining_counts(context, newly_visible or [])
    unseen_total = sum(remaining.values())
    ting_estimate = analysis if isinstance(analysis, TingEstimate) else analysis.ting_estimate
    hu_labels = set(ting_estimate.hu_cards)
    improving_labels = set(ting_estimate.improving_cards) - hu_labels
    hu_outs_by_label = {label: remaining.get(label, 0) for label in sorted(hu_labels)}
    hu_outs = sum(hu_outs_by_label.values())
    improving_outs = sum(remaining.get(label, 0) for label in improving_labels)
    if unseen_total <= 0:
        return {
            "version": "information_set_search_v3",
            "value": 0.0,
            "unseen_total": 0,
            "hu_outs": 0,
            "improving_outs": 0,
            "hu_probability": 0.0,
            "improving_probability": 0.0,
            "hu_labels": sorted(hu_labels),
            "hu_outs_by_label": {label: 0 for label in sorted(hu_labels)},
            "hu_score_by_label": {
                label: round(
                    float(
                        ting_estimate.hu_score_by_label.get(
                            label,
                            ting_estimate.expected_score_if_hu,
                        )
                    ),
                    3,
                )
                for label in sorted(hu_labels)
            },
            "expected_hu_score": 0.0,
            "improving_labels": sorted(improving_labels),
        }
    weights = context.rules.get("weights") or context.rules.get("ai_weights", {})
    hu_probability = hu_outs / unseen_total
    improving_probability = improving_outs / unseen_total
    hu_reward = float(weights.get("information_set_hu_draw_value", 1000))
    hu_score_scale = float(weights.get("information_set_hu_score_scale", 0.2))
    improve_reward = float(weights.get("information_set_improve_draw_value", 220))
    score_by_label = ting_estimate.hu_score_by_label
    weighted_hu_score = sum(
        count * float(score_by_label.get(label, ting_estimate.expected_score_if_hu))
        for label, count in hu_outs_by_label.items()
    )
    expected_hu_score = weighted_hu_score / hu_outs if hu_outs else 0.0
    value = (
        hu_probability * hu_reward
        + weighted_hu_score / unseen_total * hu_score_scale
        + improving_probability * improve_reward
    )
    return {
        "version": "information_set_search_v3",
        "value": round(value, 3),
        "unseen_total": unseen_total,
        "hu_outs": hu_outs,
        "improving_outs": improving_outs,
        "hu_probability": round(hu_probability, 6),
        "improving_probability": round(improving_probability, 6),
        "hu_labels": sorted(hu_labels),
        "hu_outs_by_label": hu_outs_by_label,
        "hu_score_by_label": {
            label: round(float(score_by_label.get(label, ting_estimate.expected_score_if_hu)), 3)
            for label in sorted(hu_labels)
        },
        "expected_hu_score": round(expected_hu_score, 3),
        "improving_labels": sorted(improving_labels),
    }


def _information_remaining_counts(context: DecisionContext, newly_visible: list[str]) -> Counter[str]:
    known: Counter[str] = Counter(context.normalized_hand)
    for meld in context.existing_melds:
        known.update(label for label in meld.labels if label != "暗")

    memory = context.state.get("memory") if isinstance(context.state.get("memory"), dict) else None
    if memory:
        for key in ("my_discards", "opponent_discards"):
            known.update(_observed_labels(memory.get(key)))
        known.update(_observed_labels(memory.get("opponent_meld_groups")))
    else:
        known.update(_observed_labels(context.state.get("visible_cards")))
        known.update(_observed_labels(context.state.get("discards")))
        meld_groups = context.state.get("meld_groups")
        if isinstance(meld_groups, dict):
            known.update(_observed_labels(meld_groups.get("opponent_melds")))
    known.update(label for label in normalize_cards(newly_visible) if label != "暗")

    remaining = full_deck_counts(rules=context.rules)
    remaining.subtract(known)
    for label in list(remaining):
        if remaining[label] < 0:
            remaining[label] = 0
    return remaining


def _observed_labels(value: Any) -> list[str]:
    if isinstance(value, str):
        label = normalize_card_label(value)
        return [label] if label and label != "暗" else []
    if isinstance(value, dict):
        direct = value.get("name") or value.get("label") or value.get("card")
        if direct is not None:
            return _observed_labels(str(direct))
        return [label for item in value.values() for label in _observed_labels(item)]
    if isinstance(value, (list, tuple)):
        return [label for item in value for label in _observed_labels(item)]
    return []


def _auto_meld_ev_details(
    context: DecisionContext,
    action: LegalAction,
    allocation: StructureAllocation,
    analysis: HandAnalysis,
    weights: dict[str, Any],
) -> dict[str, Any]:
    auto_action = str(action.label or "AUTO_MELD").upper()
    labels = _infer_auto_meld_labels(context, action, allocation)
    xi_kind = _auto_meld_xi_kind(auto_action)
    xi_gain = _auto_meld_xi_value(xi_kind, labels, context.rules)
    structure_gain = _auto_meld_structure_gain(auto_action, labels, allocation, weights)
    red_black_gain = 0.0
    xi_score = float(xi_gain) * float(weights.get("xi_per_point", 40))
    defense_remaining, _danger_remaining = _room_phase_thresholds(context.rules)
    late_bonus = (
        120.0
        if context.remaining_deck_count is not None
        and context.remaining_deck_count <= defense_remaining
        else 0.0
    )
    exposure_risk = _auto_meld_exposure_risk(auto_action, labels, context, weights)
    wait_base = float(weights.get("wait_auto_meld", 8000))
    score_gain = round(structure_gain + xi_score + red_black_gain + late_bonus, 3)
    ev = round(wait_base + score_gain - exposure_risk, 3)
    xi_text = f"预估胡息 +{xi_gain}" if xi_gain else "胡息收益待真实落地确认"
    label_text = "".join(labels) if labels else "未识别具体落地牌"
    return {
        "ev": ev,
        "score_gain": score_gain,
        "xi_gain": xi_gain,
        "red_black_gain": red_black_gain,
        "opponent_gain_risk": exposure_risk,
        "reason": f"{auto_action} 会由游戏自动落地，等待动画完成；{xi_text}；落地牌={label_text}",
        "debug_details": {
            "candidate_stage": "wait_auto_meld",
            "auto_meld_action": auto_action,
            "auto_meld_kind": xi_kind,
            "auto_meld_cards": labels,
            "xi_gain": xi_gain,
            "xi_score": xi_score,
            "structure_gain": structure_gain,
            "red_black_gain": red_black_gain,
            "late_bonus": late_bonus,
            "opponent_gain_risk": exposure_risk,
            "score_gain": score_gain,
            "hand_value_before": analysis.hand_value,
            "hand_value_after": analysis.hand_value + score_gain,
        },
    }


def _infer_auto_meld_labels(
    context: DecisionContext,
    action: LegalAction,
    allocation: StructureAllocation,
) -> list[str]:
    if action.option_cards:
        labels = normalize_cards(action.option_cards)
        if len(labels) == 1:
            amount = 4 if _auto_meld_xi_kind(str(action.label or "").upper()) in {"ti", "pao"} else 3
            return labels * amount
        return labels
    auto_action = str(action.label or "").upper()
    if auto_action in {"TI", "AUTO_QUAD"}:
        quad = next((meld for meld in allocation.locked_melds if meld.type == "exact_quad"), None)
        if quad is not None:
            return list(quad.labels)
    if auto_action == "WEI":
        triplet = next((meld for meld in allocation.locked_melds if meld.type == "exact_triplet"), None)
        if triplet is not None:
            return list(triplet.labels)
    pending = normalize_cards(
        [
            str(
                context.state.get("pending_card")
                or context.state.get("external_card")
                or context.state.get("drawn_card")
                or ""
            )
        ]
    )
    if pending:
        label = pending[0]
        count = Counter(context.normalized_hand).get(label, 0)
        if auto_action in {"PAO", "MING_LONG", "LONG"}:
            return [label] * max(4, count + 1)
        if auto_action == "WEI":
            return [label] * max(3, count + 1)
        return [label] * max(1, count)
    return []


def _auto_meld_xi_kind(auto_action: str) -> str:
    if auto_action in {"TI", "AUTO_QUAD"}:
        return "ti"
    if auto_action in {"PAO", "MING_LONG", "LONG"}:
        return "pao"
    if auto_action == "WEI":
        return "wei"
    return "auto_meld"


def _auto_meld_xi_value(kind: str, labels: list[str], rules: dict[str, Any]) -> int:
    if kind not in {"ti", "pao", "wei"} or not labels:
        return 0
    xi = rules.get("xi", {})
    suit = _suit_for_labels(labels)
    return int(xi.get(kind, {}).get(suit, 0))


def _auto_meld_structure_gain(
    auto_action: str,
    labels: list[str],
    allocation: StructureAllocation,
    weights: dict[str, Any],
) -> float:
    if not labels:
        return float(weights.get("auto_meld_unknown_gain", 0))
    if auto_action in {"TI", "AUTO_QUAD"}:
        return float(weights.get("exact_quad_break_penalty", 5000))
    if auto_action in {"PAO", "MING_LONG", "LONG"}:
        return float(weights.get("exact_quad_break_penalty", 5000)) * 0.8
    if auto_action == "WEI":
        return float(weights.get("exact_triplet_break_penalty", 3000))
    label_set = set(labels)
    matched = [
        meld.structure_value
        for meld in allocation.protected_melds
        if label_set & set(meld.labels)
    ]
    return float(max(matched) if matched else 0.0)


def _auto_meld_exposure_risk(
    auto_action: str,
    labels: list[str],
    context: DecisionContext,
    weights: dict[str, Any],
) -> float:
    if auto_action not in {"PAO", "MING_LONG", "LONG"}:
        return 0.0
    red_pressure = sum(1 for label in labels if label in RED_LABELS) * 10.0
    defense_remaining, _danger_remaining = _room_phase_thresholds(context.rules)
    late_risk = (
        0.0
        if context.remaining_deck_count is None
        or context.remaining_deck_count > defense_remaining
        else 30.0
    )
    return min(float(weights.get("opponent_pao_risk_penalty", 500)), red_pressure + late_risk)


def _room_phase_thresholds(rules: dict[str, Any]) -> tuple[int, int]:
    """Scale legacy three-player phase boundaries to the room's initial stock."""

    player_count = int(rules.get("game", {}).get("players", 3))
    if player_count not in {2, 3}:
        player_count = 3
    deck_size = sum(full_deck_counts(rules=rules).values())
    initial_stock = max(1, deck_size - (player_count * 20 + 1))
    reference_stock = max(1, deck_size - (3 * 20 + 1))
    defense_remaining = max(1, round(15 * initial_stock / reference_stock))
    danger_remaining = max(1, round(8 * initial_stock / reference_stock))
    return defense_remaining, min(defense_remaining, danger_remaining)


class ActionSimulator:
    def simulate(self, context, action, allocation=None) -> SimulatedResult:
        return simulate_action(context, action, context if isinstance(context, DecisionContext) else None, allocation)


class EVScorer:
    def __init__(self) -> None:
        self._post_discard_analysis_cache: dict[
            tuple[tuple[object, ...], ...],
            HandAnalysis,
        ] = {}

    def evaluate(self, context, action, allocation=None, analysis=None) -> ActionEval:
        active_followup_cache = _FOLLOWUP_ANALYSIS_CACHE.get()
        followup_token = (
            _FOLLOWUP_ANALYSIS_CACHE.set({})
            if active_followup_cache is None
            else None
        )
        try:
            result = evaluate_action_ev(
                context,
                action,
                context if isinstance(context, DecisionContext) else None,
                allocation,
                analysis,
                self._post_discard_analysis_cache,
            )
        finally:
            if followup_token is not None:
                _FOLLOWUP_ANALYSIS_CACHE.reset(followup_token)
        if result.debug_details.get("scorer") == self.__class__.__name__:
            return result
        debug_details = deepcopy(result.debug_details)
        debug_details["scorer"] = self.__class__.__name__
        return replace(result, debug_details=debug_details)


def _evaluate_action_task(
    payload: tuple[
        DecisionContext,
        LegalAction,
        StructureAllocation,
        HandAnalysis,
    ],
) -> ActionEval:
    context, action, allocation, analysis = payload
    return EVScorer().evaluate(context, action, allocation, analysis)


def _evaluate_action_batch_task(
    payloads: list[
        tuple[
            DecisionContext,
            LegalAction,
            StructureAllocation,
            HandAnalysis,
        ]
    ],
) -> list[ActionEval]:
    """Evaluate one worker batch with the same pure post-discard cache as serial evaluation."""

    scorer = EVScorer()
    return [
        scorer.evaluate(context, action, allocation, analysis)
        for context, action, allocation, analysis in payloads
    ]


def _raise_if_policy_evaluation_interrupted(
    absolute_deadline: float | None,
    cancelled: Callable[[], bool] | None,
) -> None:
    if cancelled is not None and cancelled():
        raise TimeoutError("production_policy_cancelled")
    if absolute_deadline is not None and time.perf_counter() >= absolute_deadline:
        raise TimeoutError("production_policy_deadline_exceeded")


def _evaluate_legal_actions(
    context: DecisionContext,
    legal_actions: list[LegalAction],
    allocation: StructureAllocation,
    analysis: HandAnalysis,
    *,
    parallel: bool,
    absolute_deadline: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[ActionEval]:
    if not parallel or len(legal_actions) < 4:
        scorer = EVScorer()
        evaluations: list[ActionEval] = []
        for action in legal_actions:
            _raise_if_policy_evaluation_interrupted(absolute_deadline, cancelled)
            evaluations.append(scorer.evaluate(context, action, allocation, analysis))
        _raise_if_policy_evaluation_interrupted(absolute_deadline, cancelled)
        return evaluations

    from multiprocessing import current_process

    process = current_process()
    if process.name != "MainProcess" or process.daemon:
        return _evaluate_legal_actions(
            context,
            legal_actions,
            allocation,
            analysis,
            parallel=False,
            absolute_deadline=absolute_deadline,
            cancelled=cancelled,
        )

    futures = []
    try:
        from ai.dual_discard_validator import _shared_executor

        executor = _shared_executor(20)
        if absolute_deadline is None and cancelled is None:
            return list(
                executor.map(
                    _evaluate_action_task,
                    (
                        (context, action, allocation, analysis)
                        for action in legal_actions
                    ),
                    chunksize=1,
                )
            )
        from concurrent.futures import FIRST_COMPLETED, wait

        futures = [
            executor.submit(
                _evaluate_action_task,
                (context, action, allocation, analysis),
            )
            for action in legal_actions
        ]
        pending = set(futures)
        while pending:
            _raise_if_policy_evaluation_interrupted(absolute_deadline, cancelled)
            wait_seconds = 0.02
            if absolute_deadline is not None:
                wait_seconds = min(
                    wait_seconds,
                    max(0.0, absolute_deadline - time.perf_counter()),
                )
            _done, pending = wait(
                pending,
                timeout=wait_seconds,
                return_when=FIRST_COMPLETED,
            )
        _raise_if_policy_evaluation_interrupted(absolute_deadline, cancelled)
        return [future.result(timeout=0.0) for future in futures]
    except TimeoutError:
        for future in futures:
            future.cancel()
        raise
    except Exception:
        return _evaluate_legal_actions(
            context,
            legal_actions,
            allocation,
            analysis,
            parallel=False,
            absolute_deadline=absolute_deadline,
            cancelled=cancelled,
        )


def _rejected_eval(
    action: LegalAction,
    reason: str,
    analysis: HandAnalysis,
    *,
    candidate_stage: str = "rejected",
) -> ActionEval:
    return ActionEval(
        action=action,
        allowed=False,
        ev=HARD_BREAK_EV,
        score_gain=0,
        hand_value_after=analysis.hand_value,
        xi_gain=0,
        ting_gain=0,
        red_black_gain=0,
        wildcard_gain=0,
        structure_loss=0,
        danger_loss=0,
        opponent_gain_risk=0,
        uncertainty_penalty=0,
        reject_reason=reason,
        reason=reason,
        debug_details={"candidate_stage": candidate_stage},
    )


def _soft_structure_loss(card_id: str, allocation: StructureAllocation, weights: dict[str, Any]) -> float:
    total = 0.0
    for meld in allocation.soft_melds:
        if card_id not in meld.card_ids:
            continue
        if meld.type == "double_normal_sequence":
            total += float(weights.get("double_sequence_break_penalty", 1200))
        elif meld.type == "special_2710":
            total += float(weights.get("special_2710_break_penalty", 1200))
        elif meld.type == "special_123":
            total += float(weights.get("special_123_break_penalty", 1000))
        elif meld.type == "mixed_same_rank_triplet":
            total += float(weights.get("mixed_same_rank_break_penalty", 1000))
        elif meld.type == "normal_sequence":
            total += float(weights.get("normal_sequence_break_penalty", 500))
        elif meld.type == "pair":
            total += float(weights.get("pair_break_penalty", 300))
    return total


def _discard_duplicate_shape_tiebreak(
    card: CardInstance,
    allocation: StructureAllocation,
    weights: dict[str, Any],
) -> tuple[float, list[str]]:
    if card.is_wildcard:
        return 0.0, []
    count = int(allocation.total_counts.get(card.label, 0) or 0)
    singleton_bonus = float(weights.get("singleton_discard_tiebreak_bonus", 30))
    duplicate_penalty = float(
        weights.get(
            "duplicate_discard_tiebreak_penalty",
            weights.get("pair_preservation_tiebreak_penalty", 45),
        )
    )
    if count <= 0:
        return 0.0, []
    if count == 1:
        neighbor_count = _same_suit_neighbor_count(card.label, allocation.total_counts)
        isolated_bonus = float(weights.get("isolated_singleton_discard_bonus", 20))
        connected_penalty = float(weights.get("connected_singleton_keep_penalty", 20))
        adjustment = singleton_bonus
        reasons = [f"{card.label} is singleton; prefer isolated discard"]
        if neighbor_count == 0:
            adjustment += isolated_bonus
            reasons.append(f"{card.label} has no same-suit neighbors")
        elif neighbor_count >= 2:
            adjustment -= connected_penalty
            reasons.append(f"{card.label} has same-suit neighbors; keep flexibility")
        return adjustment, reasons
    if count == 2:
        return -duplicate_penalty, [f"{card.label} pair kept for future peng"]
    return -duplicate_penalty * 2, [f"{card.label} duplicate group kept for future meld"]


def _same_suit_neighbor_count(label: str, total_counts: dict[str, int]) -> int:
    parsed = parse_label(label)
    if parsed is None:
        return 0
    suit, rank = parsed
    count = 0
    for delta in (-2, -1, 1, 2):
        neighbor_rank = rank + delta
        if 1 <= neighbor_rank <= 10:
            count += min(1, int(total_counts.get(label_for(suit, neighbor_rank), 0) or 0))
    return count


def _evaluate_response_ev(
    context: DecisionContext,
    action: LegalAction,
    analysis: HandAnalysis,
    allocation: StructureAllocation,
) -> ActionEval:
    weights = context.rules.get("weights") or context.rules.get("ai_weights", {})
    min_gain = float(
        context.rules.get("chi" if action.type == "CHI" else "peng", {}).get(
            "min_ev_gain",
            weights.get("chi_min_ev_gain" if action.type == "CHI" else "peng_min_ev_gain", 20),
        )
    )
    pending = _pending_action_label(context)
    pass_information_set = _information_set_draw_value(
        context,
        analysis,
        newly_visible=[pending] if pending else [],
    )
    pass_ev = analysis.hand_value + pass_information_set["value"]
    option_cards = normalize_cards(action.option_cards)
    candidate_results: list[dict[str, Any]] = []
    followup_discard_cache: dict[str, dict[str, Any] | None] = {}
    if action.type == "CHI":
        if not option_cards and not context.chi_options:
            return _response_reject_eval(action, analysis, pass_ev, "chi_options_not_visible", min_gain)
        if action.uncertainty > 0.25 and context.rules.get("chi", {}).get("pass_if_uncertain", True):
            return _response_reject_eval(action, analysis, pass_ev, "chi_option_uncertain", min_gain)
        for candidate in _chi_consumption_candidates(context, action):
            candidate_results.append(
                _score_chi_candidate(
                    context,
                    action,
                    analysis,
                    allocation,
                    candidate,
                    min_gain,
                    pass_ev,
                    followup_discard_cache,
                )
            )
    else:
        for candidate in _peng_consumption_candidates(context, action):
            candidate_results.append(
                _score_peng_candidate(
                    context,
                    action,
                    analysis,
                    allocation,
                    candidate,
                    min_gain,
                    pass_ev,
                    followup_discard_cache,
                )
            )
    if not candidate_results:
        if (
            action.type == "CHI"
            and WILD_LABEL in option_cards
            and context.rules.get("chi", {}).get(
                "forbid_use_wildcard_unless_immediate_hu_or_big_gain",
                True,
            )
        ):
            return _response_reject_eval(
                action,
                analysis,
                pass_ev,
                "chi_consumes_wildcard_without_big_gain",
                min_gain,
            )
        if action.type == "CHI" and any(
            context.normalized_hand.count(label) >= 3
            for label in option_cards
            if label != _pending_action_label(context)
        ):
            return _response_reject_eval(
                action,
                analysis,
                pass_ev,
                "chi_consumes_hard_protected",
                min_gain,
            )
        if action.type == "CHI" and action.requires_option and _can_trust_visible_response_button(context, action, "CHI"):
            return _trusted_chi_option_eval(context, action, analysis, pass_ev, min_gain)
        if action.type == "PENG" and _can_trust_visible_response_button(context, action, "PENG"):
            return _trusted_peng_button_eval(context, action, analysis, pass_ev, min_gain)
        reject = "chi_option_not_in_hand" if action.type == "CHI" else "peng_pair_not_available"
        return _response_reject_eval(action, analysis, pass_ev, reject, min_gain)

    allowed_candidates = [item for item in candidate_results if item["allowed"]]
    best = max(allowed_candidates or candidate_results, key=lambda item: item["ev"])
    allowed = bool(best["allowed"])
    reject = None if allowed else best["reject_reason"]
    response_simulation = simulate_action(
        context,
        action,
        context,
        allocation,
        followup_discard_cache,
    ).to_dict()
    public_response_ev = round(best["ev"], 3) if allowed else HARD_BREAK_EV
    return ActionEval(
        action=action,
        allowed=allowed,
        ev=round(best["ev"] if allowed else HARD_BREAK_EV, 3),
        score_gain=round(best["score_gain"], 3),
        hand_value_after=round(best["hand_value_after"], 3),
        xi_gain=round(best["xi_gain"], 3),
        ting_gain=round(best["ting_gain"], 3),
        red_black_gain=round(best["red_black_gain"], 3),
        wildcard_gain=round(best["wildcard_gain"], 3),
        structure_loss=round(best["structure_loss"], 3),
        danger_loss=round(best["danger_loss"], 3),
        opponent_gain_risk=round(best["opponent_gain_risk"], 3),
        uncertainty_penalty=round(action.uncertainty * 100, 3),
        reject_reason=None if allowed else reject,
        reason=best["reason"] if best.get("reason") else (reject or "response_ev_not_enough"),
        debug_details={
            "pass_ev": pass_ev,
            "pass_information_set": pass_information_set,
            "response_ev": public_response_ev,
            "chi_ev" if action.type == "CHI" else "peng_ev": public_response_ev,
            "min_ev_gain": min_gain,
            "response_gain": best["score_gain"],
            "ev_delta_vs_pass": round(public_response_ev - pass_ev, 3),
            "all_response_candidates": candidate_results,
            "consumed_card_ids": best["consumed_card_ids"],
            "consumed_from_hand": best["consumed_from_hand"],
            "consumption_impact": best.get("consumption_impact", {}),
            "followup_discard": best.get("followup_discard"),
            "breaks_melds": best.get("breaks_melds", []),
            "simulation": response_simulation,
            "ting_value_delta": best.get("ting_value_delta", 0),
            "ting_before": best.get("ting_before"),
            "ting_after": best.get("ting_after"),
            "trusted_visible_response_override": best.get("trusted_visible_response_override", False),
            "meld_cards": best.get("meld_cards"),
            "compare_groups": best.get("compare_groups", []),
            "plan_melds": best.get("plan_melds", []),
            "recognized_option_cards": best.get("recognized_option_cards"),
        },
        forced_followup_loss=round(best["forced_followup_loss"], 3),
    )


def _response_reject_eval(
    action: LegalAction,
    analysis: HandAnalysis,
    pass_ev: float,
    reason: str,
    min_gain: float = 0.0,
    extra_debug: dict[str, Any] | None = None,
) -> ActionEval:
    response_key = "chi_ev" if action.type == "CHI" else "peng_ev"
    debug_details = {
        "pass_ev": pass_ev,
        "response_ev": HARD_BREAK_EV,
        response_key: HARD_BREAK_EV,
        "response_gain": 0,
        "ev_delta_vs_pass": -99999.0,
        "min_ev_gain": min_gain,
        "consumed_card_ids": [],
        "consumed_from_hand": [],
        "followup_discard": None,
    }
    if extra_debug:
        debug_details.update(extra_debug)
    return ActionEval(
        action=action,
        allowed=False,
        ev=HARD_BREAK_EV,
        score_gain=0,
        hand_value_after=analysis.hand_value,
        xi_gain=0,
        ting_gain=0,
        red_black_gain=0,
        wildcard_gain=0,
        structure_loss=0,
        danger_loss=0,
        opponent_gain_risk=0,
        uncertainty_penalty=round(action.uncertainty * 100, 3),
        reject_reason=reason,
        reason=reason,
        debug_details=debug_details,
    )


def _can_trust_visible_response_button(context: DecisionContext, action: LegalAction, action_type: str) -> bool:
    return action.requires_button and action.source == "button" and "buttons" in context.state and _button_visible(context, action_type)


def _trusted_peng_button_eval(
    context: DecisionContext,
    action: LegalAction,
    analysis: HandAnalysis,
    pass_ev: float,
    min_gain: float,
) -> ActionEval:
    ev = round(pass_ev + min_gain + 1.0, 3)
    pending = action.label or "unknown"
    return ActionEval(
        action=action,
        allowed=True,
        ev=ev,
        score_gain=round(min_gain + 1.0, 3),
        hand_value_after=analysis.hand_value,
        xi_gain=0,
        ting_gain=0,
        red_black_gain=0,
        wildcard_gain=0,
        structure_loss=0,
        danger_loss=0,
        opponent_gain_risk=25.0,
        uncertainty_penalty=round(action.uncertainty * 100, 3),
        reject_reason=None,
        reason="peng_protocol_trusted_pair_unavailable_locally",
        debug_details={
            "pass_ev": pass_ev,
            "response_ev": ev,
            "peng_ev": ev,
            "min_ev_gain": min_gain,
            "response_gain": round(min_gain + 1.0, 3),
            "ev_delta_vs_pass": round(ev - pass_ev, 3),
            "all_response_candidates": [],
            "consumed_card_ids": [],
            "consumed_from_hand": [],
            "consumption_impact": {
                "notes": ["trusted_visible_peng_button_when_local_pair_binding_missing"],
                "breaks_melds": [],
                "breaks_hard": False,
                "breaks_soft": False,
            },
            "followup_discard": None,
            "breaks_melds": [],
            "simulation": {
                "action": action.to_dict(),
                "hand_after": list(context.normalized_hand),
                "consumed_card_ids": [],
                "notes": ["peng_button_trusted_without_local_pair_consumption"],
            },
            "pending_label": pending,
            "trusted_button": True,
        },
    )


def _trusted_chi_option_eval(
    context: DecisionContext,
    action: LegalAction,
    analysis: HandAnalysis,
    pass_ev: float,
    min_gain: float,
) -> ActionEval:
    option_cards = normalize_cards(action.option_cards)
    return _response_reject_eval(
        action,
        analysis,
        pass_ev,
        "chi_option_binding_missing",
        min_gain,
        {
            "all_response_candidates": [],
            "consumed_card_ids": [],
            "consumed_from_hand": [],
            "consumption_impact": {
                "notes": ["visible_chi_option_without_local_consumption_binding"],
                "breaks_melds": [],
                "breaks_hard": False,
                "breaks_soft": False,
            },
            "followup_discard": None,
            "breaks_melds": [],
            "simulation": {
                "action": action.to_dict(),
                "hand_after": list(context.normalized_hand),
                "consumed_card_ids": [],
                "notes": ["chi_option_rejected_without_local_consumption"],
            },
            "selected_option_id": action.option_id,
            "selected_option_cards": option_cards,
            "trusted_button": True,
            "trusted_visible_response_override": False,
        },
    )


def _pending_action_label(context: DecisionContext) -> str | None:
    for key in (
        "pending_action_card",
        "pending_card",
        "current_card",
        "action_card",
        "last_discard",
        "discard_label",
        "response_card",
    ):
        value = context.state.get(key)
        if isinstance(value, dict):
            value = value.get("label") or value.get("name")
        if isinstance(value, str):
            normalized = normalize_card_label(value)
            if normalized:
                return normalized
    for item in context.state.get("legal_actions", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).upper() not in {"CHI", "PENG"}:
            continue
        value = item.get("label") or item.get("card") or item.get("pending_card")
        if isinstance(value, dict):
            value = value.get("label") or value.get("name")
        if isinstance(value, str):
            normalized = normalize_card_label(value)
            if normalized:
                return normalized
    return None


def _consume_labels_from_context(
    context: DecisionContext,
    labels: list[str],
) -> list[CardInstance] | None:
    by_label = _instances_by_label(list(context.card_instances))
    consumed: list[CardInstance] = []
    for label in labels:
        available = by_label.get(label, [])
        if not available:
            return None
        consumed_card = available.pop(0)
        consumed.append(consumed_card)
    return consumed


def _chi_consumption_candidates(context: DecisionContext, action: LegalAction) -> list[dict[str, Any]]:
    option_cards = normalize_cards(action.option_cards)
    if not option_cards:
        return []
    pending = _pending_action_label(context)
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], tuple[tuple[str, ...], ...]]] = set()
    possible_external = [pending] if pending else _ordered_unique(option_cards)
    allow_1510 = bool(context.rules.get("rules", {}).get("allow_1510", False))
    for external_label in possible_external:
        for plan in enumerate_chi_plans(
            context.normalized_hand,
            external_label,
            allow_1510=allow_1510,
        ):
            if not _chi_option_matches_initial_group(option_cards, plan.initial_group, external_label):
                continue
            consumed = _consume_labels_from_context(context, list(plan.consumed_from_hand))
            if consumed is None:
                continue
            plan_groups = [list(group) for group in plan.groups]
            plan_melds = [
                {"type": _meld_type_for_option(group, "CHI"), "cards": group}
                for group in plan_groups
            ]
            key = (
                external_label,
                tuple(card.id for card in consumed),
                tuple(tuple(group) for group in plan_groups),
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "external_label": external_label,
                    "option_cards": option_cards,
                    "meld_cards": list(plan.initial_group),
                    "compare_groups": [list(group) for group in plan.compare_groups],
                    "plan_melds": plan_melds,
                    "consumed_cards": consumed,
                    "consumed_labels": [card.label for card in consumed],
                    "consumed_card_ids": [card.id for card in consumed],
                }
            )
    return results


def _chi_option_matches_initial_group(
    option_cards: list[str],
    initial_group: tuple[str, str, str],
    external_label: str,
) -> bool:
    if Counter(option_cards) == Counter(initial_group):
        return True
    expected_hand_cards = list(initial_group)
    try:
        expected_hand_cards.remove(external_label)
    except ValueError:
        return False
    return any(
        Counter(option_cards[:index] + option_cards[index + 1 :]) == Counter(expected_hand_cards)
        for index in range(len(option_cards))
    )


def _canonical_meld_label_order(labels: list[str]) -> list[str]:
    return sorted(
        labels,
        key=lambda label: (
            parse_label(label)[1] if parse_label(label) else 99,
            parse_label(label)[0] if parse_label(label) else "",
            label,
        ),
    )


def _peng_consumption_candidates(context: DecisionContext, action: LegalAction) -> list[dict[str, Any]]:
    pending = action.label or _pending_action_label(context)
    if pending is None:
        pairs = [
            label
            for label, count in Counter(context.normalized_hand).items()
            if count >= 2 and label != WILD_LABEL
        ]
        pending = pairs[0] if len(pairs) == 1 else None
    if pending is None:
        return []
    consumed = _consume_labels_from_context(context, [pending, pending])
    if consumed is None:
        return []
    return [
        {
            "external_label": pending,
            "option_cards": [pending, pending, pending],
            "consumed_cards": consumed,
            "consumed_labels": [card.label for card in consumed],
            "consumed_card_ids": [card.id for card in consumed],
        }
    ]


def _context_after_consumption(
    context: DecisionContext,
    consumed_card_ids: list[str],
    *,
    meld_type: str | None = None,
    meld_cards: list[str] | None = None,
    plan_melds: list[dict[str, Any]] | None = None,
) -> DecisionContext:
    consumed = set(consumed_card_ids)
    after_cards = [card for card in context.card_instances if card.id not in consumed]
    strategy_existing_melds = list(context.state.get("strategy_existing_melds") or [])
    melds_to_add = list(plan_melds or [])
    if not melds_to_add and meld_type and meld_cards:
        melds_to_add.append({"type": meld_type, "cards": meld_cards})
    for meld in melds_to_add:
        strategy_existing_melds.append(
            {
                "type": meld.get("type"),
                "labels": list(meld.get("cards") or meld.get("labels") or []),
                "source": "simulated_response",
            }
        )
    after_state = {
        **context.state,
        "hand": [card.label for card in after_cards],
        "raw_hand": [card.label for card in after_cards],
        "hand_details": [card.to_dict() | {"name": card.label} for card in after_cards],
        "legal_actions": [{"type": "discard"}],
        "recognition_warnings": [],
        "recognition_errors": [],
        "strategy_existing_melds": strategy_existing_melds,
    }
    return build_decision_context(after_state, rules=context.rules)


def _context_after_discard_id(context: DecisionContext, card_id: str) -> DecisionContext:
    after_cards = [card for card in context.card_instances if card.id != card_id]
    after_state = {
        **context.state,
        "hand": [card.label for card in after_cards],
        "raw_hand": [card.label for card in after_cards],
        "hand_details": [card.to_dict() | {"name": card.label} for card in after_cards],
        "legal_actions": [{"type": "discard"}],
        "recognition_warnings": [],
        "recognition_errors": [],
    }
    return build_decision_context(after_state, rules=context.rules)


def _best_followup_discard(
    after_context: DecisionContext,
) -> dict[str, Any] | None:
    if not after_context.card_instances:
        return None
    after_decision = choose_action(after_context.state, rules=after_context.rules)
    if after_decision.selected_action != "DISCARD" or after_decision.selected_card_id is None:
        return None
    after_allocation = allocate_hand_structures(after_context)
    selected_card = next(
        (card for card in after_context.card_instances if card.id == after_decision.selected_card_id),
        None,
    )
    selected_eval = next(
        (
            item
            for item in after_decision.action_evals
            if item.action.type == "DISCARD"
            and item.action.card_id == after_decision.selected_card_id
        ),
        None,
    )
    followup_analysis_cache = _FOLLOWUP_ANALYSIS_CACHE.get()
    if selected_eval is not None and followup_analysis_cache is not None:
        before_analysis = after_decision.context_snapshot.get("hand_analysis") or {}
        ting_payload = selected_eval.debug_details.get("ting_after") or {}
        if ting_payload:
            ting_fields = {
                key: value
                for key, value in ting_payload.items()
                if key != "distance"
            }
            try:
                followup_analysis_cache[after_context.state_fingerprint] = {
                    "hand_value": float(selected_eval.hand_value_after),
                    "min_xi": int(before_analysis.get("min_xi", 0)),
                    "ting_estimate": TingEstimate(**ting_fields),
                    "red_black_bonus": float(
                        (before_analysis.get("red_black_plan") or {}).get(
                            "bonus_estimate",
                            0.0,
                        )
                    )
                    + float(selected_eval.red_black_gain),
                    "wildcard_flexibility": float(
                        (before_analysis.get("wildcard_value") or {}).get(
                            "future_flexibility_score",
                            0.0,
                        )
                    )
                    + float(selected_eval.wildcard_gain),
                }
            except (TypeError, ValueError):
                # The normal analysis path remains the exact fallback if a
                # diagnostic payload from an older frozen policy is incomplete.
                pass
    danger = evaluate_discard_danger(after_context, selected_card) if selected_card else DangerScore(0, "unknown", [])
    return {
        "type": "DISCARD",
        "card_id": after_decision.selected_card_id,
        "label": after_decision.selected_label,
        "ev": after_decision.ev,
        "reason": after_decision.reason,
        "breaks_hard": after_decision.selected_card_id in set(after_allocation.hard_protected_instances),
        "breaks_soft": after_decision.selected_card_id in set(after_allocation.soft_protected_instances),
        "is_wildcard": selected_card.is_wildcard if selected_card else False,
        "danger_score": danger.danger_score,
        "danger_reasons": danger.reasons,
    }


def _cached_best_followup_discard(
    after_context: DecisionContext,
    cache: dict[str, dict[str, Any] | None] | None,
) -> dict[str, Any] | None:
    if cache is None:
        return _best_followup_discard(after_context)
    cache_key = after_context.state_fingerprint
    if cache_key not in cache:
        cache[cache_key] = _best_followup_discard(after_context)
    return cache[cache_key]


def _response_after_analysis_values(
    claim_context: DecisionContext,
    after_context: DecisionContext,
) -> tuple[float, int, TingEstimate, float, float]:
    summary_cache = _FOLLOWUP_ANALYSIS_CACHE.get()
    summary = (
        summary_cache.get(claim_context.state_fingerprint)
        if summary_cache is not None
        else None
    )
    if summary is not None:
        return (
            float(summary["hand_value"]),
            int(summary["min_xi"]),
            summary["ting_estimate"],
            float(summary["red_black_bonus"]),
            float(summary["wildcard_flexibility"]),
        )
    after_allocation = allocate_hand_structures(after_context)
    after_analysis = analyze_hand(after_context, after_allocation)
    return (
        float(after_analysis.hand_value),
        int(after_analysis.min_xi),
        after_analysis.ting_estimate,
        float(after_analysis.red_black_plan.bonus_estimate),
        float(after_analysis.wildcard_value.future_flexibility_score),
    )


def _consumption_cost(
    consumed_card_ids: list[str],
    allocation: StructureAllocation,
    weights: dict[str, Any],
    *,
    preserved_soft_meld_ids: set[str] | None = None,
) -> tuple[float, list[str], list[list[str]], bool, bool]:
    preserved_soft_meld_ids = preserved_soft_meld_ids or set()
    hard_ids = set(allocation.hard_protected_instances)
    soft_ids = {
        card_id
        for meld in allocation.soft_melds
        if meld.id not in preserved_soft_meld_ids
        for card_id in meld.card_ids
    }
    cost = 0.0
    reasons: list[str] = []
    breaks_melds: list[list[str]] = []
    breaks_hard = False
    breaks_soft = False
    for meld in allocation.locked_melds:
        if set(consumed_card_ids) & set(meld.card_ids):
            breaks_hard = True
            cost += float(weights.get("hard_protected_break_penalty", 10000))
            reasons.append(f"consumes hard protected {''.join(meld.labels)}")
            breaks_melds.append(list(meld.labels))
    for meld in allocation.soft_melds:
        if meld.id in preserved_soft_meld_ids:
            continue
        if set(consumed_card_ids) & set(meld.card_ids):
            breaks_soft = True
            cost += _soft_structure_loss(next(iter(set(consumed_card_ids) & set(meld.card_ids))), allocation, weights)
            reasons.append(f"consumes soft protected {''.join(meld.labels)}")
            breaks_melds.append(list(meld.labels))
    if any(card_id in hard_ids for card_id in consumed_card_ids):
        breaks_hard = True
    if any(card_id in soft_ids for card_id in consumed_card_ids):
        breaks_soft = True
    return cost, reasons, breaks_melds, breaks_hard, breaks_soft


def _preserved_soft_melds_after_claim(
    consumed_card_ids: list[str],
    allocation: StructureAllocation,
    replacement_melds: list[dict[str, Any]],
) -> list[ProtectedMeld]:
    consumed = set(consumed_card_ids)
    replacement_counts = [
        Counter(normalize_cards(list(meld.get("cards") or meld.get("labels") or [])))
        for meld in replacement_melds
    ]
    preserved: list[ProtectedMeld] = []
    for meld in allocation.soft_melds:
        if not set(meld.card_ids) <= consumed:
            continue
        original = Counter(meld.labels)
        if any(all(replacement.get(label, 0) >= amount for label, amount in original.items()) for replacement in replacement_counts):
            preserved.append(meld)
    return preserved


def _consumption_impact(
    consumed_card_ids: list[str],
    allocation: StructureAllocation,
    weights: dict[str, Any],
    *,
    replacement_melds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    consumed = set(consumed_card_ids)
    consumed_labels = [
        card.label for card in allocation.card_instances if card.id in consumed
    ]
    touched_melds = [
        meld for meld in allocation.protected_melds if consumed & set(meld.card_ids)
    ]
    preserved_melds = _preserved_soft_melds_after_claim(
        consumed_card_ids,
        allocation,
        list(replacement_melds or []),
    )
    preserved_ids = {meld.id for meld in preserved_melds}
    broken_melds = [meld for meld in touched_melds if meld.id not in preserved_ids]
    break_types = {meld.type for meld in broken_melds}
    structure_loss, notes, breaks_melds, breaks_hard, breaks_soft = _consumption_cost(
        consumed_card_ids,
        allocation,
        weights,
        preserved_soft_meld_ids=preserved_ids,
    )
    normal_sequence_hits = [meld for meld in broken_melds if meld.type == "normal_sequence"]
    rank_span = {
        tuple(sorted({rank for rank in meld.ranks if rank is not None}))
        for meld in normal_sequence_hits
    }
    breaks_double_sequence = "double_normal_sequence" in break_types or any(
        len([meld for meld in normal_sequence_hits if tuple(sorted({rank for rank in meld.ranks if rank is not None})) == span]) >= 2
        for span in rank_span
    )
    if breaks_double_sequence and "double_normal_sequence" not in break_types:
        extra = float(weights.get("double_sequence_break_penalty", 1200))
        structure_loss += extra
        notes.append("consumes two copies of the same normal sequence")
    return {
        "consumed_card_ids": list(consumed_card_ids),
        "consumed_labels": consumed_labels,
        "consumed_from_melds": [meld.to_dict() for meld in touched_melds],
        "preserved_melds": [meld.to_dict() for meld in preserved_melds],
        "breaks_melds": breaks_melds,
        "breaks_hard_melds": [meld.to_dict() for meld in broken_melds if meld.protect_level == "hard"],
        "breaks_soft_melds": [meld.to_dict() for meld in broken_melds if meld.protect_level != "hard"],
        "breaks_hard": breaks_hard,
        "breaks_soft": breaks_soft,
        "breaks_exact_triplet": "exact_triplet" in break_types,
        "breaks_exact_quad": "exact_quad" in break_types,
        "breaks_double_sequence": breaks_double_sequence,
        "breaks_normal_sequence": "normal_sequence" in break_types,
        "breaks_special_123": "special_123" in break_types,
        "breaks_special_2710": "special_2710" in break_types,
        "breaks_mixed_same_rank_triplet": "mixed_same_rank_triplet" in break_types,
        "breaks_pair": "pair" in break_types,
        "breaks_wildcard_structure": "wildcard_meld" in break_types or WILD_LABEL in consumed_labels,
        "structure_loss": round(structure_loss, 3),
        "xi_loss": sum(meld.xi_value for meld in broken_melds),
        "red_black_loss": sum(meld.red_black_value for meld in broken_melds),
        "wildcard_loss": float(weights.get("wildcard_discard_penalty", 10000)) if WILD_LABEL in consumed_labels else 0.0,
        "notes": notes,
    }


def _response_reject_from_consumption(
    *,
    action_type: str,
    meld_type: str,
    impact: dict[str, Any],
    big_gain: bool,
) -> tuple[str | None, str | None]:
    if impact["breaks_hard"] or impact["breaks_wildcard_structure"]:
        return f"{action_type.lower()}_consumes_hard_protected", None
    if impact["breaks_double_sequence"]:
        return (
            f"{action_type.lower()}_breaks_double_sequence",
            f"{action_type} consumes cards from two complete normal sequences; PASS keeps the double sequence",
        )
    if action_type == "CHI" and impact["breaks_special_123"] and meld_type in {"mixed_same_rank_triplet", "exact_triplet"}:
        return (
            "chi_breaks_existing_123",
            "吃三三叁会消耗一二三中的三，导致一二三被拆成一二弱搭",
        )
    if impact["breaks_normal_sequence"] and not big_gain:
        return (
            f"{action_type.lower()}_breaks_complete_meld",
            f"{action_type} would break an existing complete sequence without enough gain",
        )
    return None, None


def _response_ting_reject_reason(
    action_type: str,
    before: TingEstimate,
    after: TingEstimate,
) -> tuple[str | None, str | None]:
    prefix = action_type.lower()
    if before.is_ting and not after.is_ting:
        return (
            f"{prefix}_breaks_ting",
            f"{action_type} would break current ting state; PASS keeps ting",
        )
    if action_type == "CHI" and before.is_ting:
        before_outs = _ting_out_count(before)
        after_outs = _ting_out_count(after)
        if after_outs <= before_outs:
            return (
                "chi_does_not_improve_ting",
                f"CHI keeps ting at {after_outs} outs; PASS keeps {before_outs} outs",
            )
    return None, None


def _ting_out_count(estimate: TingEstimate) -> int:
    return len(set(estimate.hu_cards or estimate.waiting_cards or []))


def _can_override_chi_soft_cost_for_new_ting(
    *,
    reject: str | None,
    meld_type: str,
    impact: dict[str, Any],
    before: TingEstimate,
    after: TingEstimate,
    min_xi: int,
    xi_gain: float,
    ting_value_delta: float,
    followup: dict[str, Any] | None,
) -> bool:
    if reject != "chi_ev_not_enough" or meld_type != "special_2710":
        return False
    if before.is_ting or not after.is_ting or _ting_out_count(after) < 3:
        return False
    if after.expected_xi_if_hu < min_xi or xi_gain <= 0 or ting_value_delta <= 0:
        return False
    if not impact.get("breaks_soft") or not impact.get("breaks_normal_sequence"):
        return False
    unsafe_breaks = (
        "breaks_hard",
        "breaks_wildcard_structure",
        "breaks_exact_quad",
        "breaks_exact_triplet",
        "breaks_mixed_same_rank_triplet",
        "breaks_special_123",
        "breaks_special_2710",
        "breaks_double_sequence",
    )
    if any(impact.get(key) for key in unsafe_breaks):
        return False
    broken_soft_melds = impact.get("breaks_soft_melds") or []
    if not broken_soft_melds or any(
        not isinstance(meld, dict) or str(meld.get("type")) != "normal_sequence"
        for meld in broken_soft_melds
    ):
        return False
    if followup is None or followup.get("breaks_hard") or followup.get("is_wildcard"):
        return False
    return True


def _zero_xi_mixed_soft_cost(impact: dict[str, Any]) -> float:
    broken_soft_melds = impact.get("breaks_soft_melds") or []
    if not broken_soft_melds or any(
        not isinstance(meld, dict)
        or str(meld.get("type")) != "mixed_same_rank_triplet"
        or float(meld.get("xi_value") or 0.0) != 0.0
        for meld in broken_soft_melds
    ):
        return 0.0
    return sum(
        max(0.0, float(meld.get("structure_value") or 0.0))
        for meld in broken_soft_melds
    )


def _is_two_player_no_wang(context: DecisionContext) -> bool:
    return (
        int(context.rules.get("game", {}).get("players", 3)) == 2
        and not bool(context.rules.get("wildcard", {}).get("enabled"))
    )


def _high_xi_123_soft_mixed_cost_relief(
    context: DecisionContext,
    *,
    meld_type: str,
    xi_gain: float,
    impact: dict[str, Any],
    before: TingEstimate,
    after: TingEstimate,
    followup: dict[str, Any] | None,
) -> float:
    broken_soft_melds = impact.get("breaks_soft_melds") or []
    if (
        not _is_two_player_no_wang(context)
        or meld_type != "special_123"
        or xi_gain < 6.0
        or context.remaining_deck_count is None
        or context.remaining_deck_count > 34
        or before.is_ting
        or after.is_ting
        or after.shanten_like_distance > before.shanten_like_distance
        or impact.get("breaks_hard")
        or impact.get("breaks_normal_sequence")
        or impact.get("breaks_special_123")
        or impact.get("breaks_special_2710")
        or impact.get("breaks_wildcard_structure")
        or len(broken_soft_melds) != 2
        or float(impact.get("red_black_loss") or 0.0) != 3.0
        or followup is None
        or followup.get("breaks_hard")
        or followup.get("breaks_soft")
        or followup.get("is_wildcard")
        or float(followup.get("danger_score") or 0.0) > 40.0
    ):
        return 0.0
    return _zero_xi_mixed_soft_cost(impact)


def _early_zero_xi_mixed_chi_flexibility_cost(
    context: DecisionContext,
    *,
    meld_type: str,
    plan_melds: list[dict[str, Any]],
    consumed_labels: list[str],
    xi_gain: float,
    impact: dict[str, Any],
    before: TingEstimate,
    after: TingEstimate,
    followup: dict[str, Any] | None,
    weights: dict[str, Any],
    before_analysis: HandAnalysis,
    after_information_set: dict[str, Any],
) -> float:
    if (
        not _is_two_player_no_wang(context)
        or meld_type != "mixed_same_rank_triplet"
        or len(plan_melds) != 1
        or len(consumed_labels) != 2
        or len(set(consumed_labels)) != 1
        or xi_gain != 0.0
        or context.existing_melds
        or context.remaining_deck_count is None
        or context.remaining_deck_count < 35
        or before.is_ting
        or after.is_ting
        or after.shanten_like_distance < before.shanten_like_distance
        or impact.get("breaks_hard")
        or impact.get("breaks_soft")
        or impact.get("breaks_wildcard_structure")
        or float(impact.get("red_black_loss") or 0.0) != 0.0
        or followup is None
        or followup.get("breaks_hard")
        or followup.get("breaks_soft")
        or followup.get("is_wildcard")
        or float(followup.get("danger_score") or 0.0) > 10.0
    ):
        return 0.0
    pending = _pending_action_label(context)
    pass_information_set = _information_set_draw_value(
        context,
        before_analysis,
        newly_visible=[pending] if pending else [],
    )
    if (
        int(after_information_set.get("improving_outs") or 0)
        - int(pass_information_set.get("improving_outs") or 0)
        < 9
    ):
        return 0.0
    return float(weights.get("pair_break_penalty", 300.0))


def _expanded_wait_peng_soft_mixed_cost_relief(
    context: DecisionContext,
    *,
    xi_gain: float,
    impact: dict[str, Any],
    before: TingEstimate,
    after: TingEstimate,
    followup: dict[str, Any] | None,
) -> float:
    before_waits = set(before.hu_cards or before.waiting_cards or [])
    after_waits = set(after.hu_cards or after.waiting_cards or [])
    pending = _pending_action_label(context)
    remaining = _information_remaining_counts(
        context,
        [pending] if pending else [],
    )
    before_outs = sum(int(remaining.get(label, 0)) for label in before_waits)
    after_outs = sum(int(remaining.get(label, 0)) for label in after_waits)
    if (
        not _is_two_player_no_wang(context)
        or xi_gain < 3.0
        or not before.is_ting
        or not after.is_ting
        or not before_waits
        or len(after_waits) <= len(before_waits)
        or after_outs - before_outs < 4
        or after_outs < before_outs * 3
        or after.expected_xi_if_hu < before.expected_xi_if_hu
        or impact.get("breaks_hard")
        or not impact.get("breaks_mixed_same_rank_triplet")
        or impact.get("breaks_normal_sequence")
        or impact.get("breaks_special_123")
        or impact.get("breaks_special_2710")
        or impact.get("breaks_wildcard_structure")
        or float(impact.get("red_black_loss") or 0.0) != 0.0
        or followup is None
        or followup.get("breaks_hard")
        or followup.get("breaks_soft")
        or followup.get("is_wildcard")
        or float(followup.get("danger_score") or 0.0) > 40.0
    ):
        return 0.0
    return _zero_xi_mixed_soft_cost(impact)


def _preserve_response_reject_over_ting(action_type: str, reject: str | None) -> bool:
    if not reject:
        return False
    prefix = action_type.lower()
    hard_rejects = {
        f"{prefix}_consumes_hard_protected",
        f"{prefix}_followup_forces_hard_break",
        f"{prefix}_followup_discards_wildcard",
    }
    if action_type == "CHI":
        hard_rejects.update(
            {
                "chi_breaks_double_sequence",
                "chi_breaks_existing_123",
                "chi_consumes_wildcard_without_big_gain",
            }
        )
    return reject in hard_rejects


def _meld_type_for_option(labels: list[str], action_type: str) -> str:
    normalized = normalize_cards(labels)
    if action_type == "PENG":
        return "peng"
    ranks = [parse_label(label)[1] for label in normalized if parse_label(label)]
    if set(ranks) == {2, 7, 10} and len(ranks) == 3:
        return "special_2710"
    if set(ranks) == {1, 2, 3} and len(ranks) == 3:
        return "special_123"
    if len(ranks) == 3 and max(ranks) - min(ranks) == 2 and len(set(ranks)) == 3:
        return "normal_sequence"
    if len(ranks) == 3 and len(set(ranks)) == 1 and len(set(normalized)) >= 2:
        return "mixed_same_rank_triplet"
    if len(set(normalized)) == 1:
        return "exact_triplet"
    return "unknown"


def _score_chi_candidate(
    context: DecisionContext,
    action: LegalAction,
    analysis: HandAnalysis,
    allocation: StructureAllocation,
    candidate: dict[str, Any],
    min_gain: float,
    pass_ev: float,
    followup_discard_cache: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    weights = context.rules.get("weights") or context.rules.get("ai_weights", {})
    option_cards = candidate["option_cards"]
    meld_cards = candidate.get("meld_cards") or option_cards
    meld_type = _meld_type_for_option(meld_cards, "CHI")
    plan_melds = candidate.get("plan_melds") or [{"type": meld_type, "cards": meld_cards}]
    xi_gain = sum(
        _meld_xi_value(str(meld["type"]), list(meld["cards"]), context.rules)
        for meld in plan_melds
    )
    structure_gain = sum(
        _claim_structure_value(str(meld["type"]), context.rules)
        for meld in plan_melds
    )
    consumed_ids = candidate["consumed_card_ids"]
    impact = _consumption_impact(
        consumed_ids,
        allocation,
        weights,
        replacement_melds=plan_melds,
    )
    structure_cost = float(impact["structure_loss"])
    cost_reasons = list(impact["notes"])
    breaks_melds = list(impact["breaks_melds"])
    breaks_hard = bool(impact["breaks_hard"])
    reject: str | None = None
    reject_explanation: str | None = None
    if breaks_hard and context.rules.get("chi", {}).get("forbid_break_hard_protection", True):
        reject = "chi_consumes_hard_protected"
    big_gain = meld_type in {"special_2710", "special_123"} or xi_gain > 0
    structural_reject, structural_reason = _response_reject_from_consumption(
        action_type="CHI",
        meld_type=meld_type,
        impact=impact,
        big_gain=big_gain,
    )
    if structural_reject:
        reject = reject or structural_reject
        reject_explanation = structural_reason
    if (
        WILD_LABEL in candidate["consumed_labels"] or WILD_LABEL in meld_cards
    ) and context.rules.get("chi", {}).get("forbid_use_wildcard_unless_immediate_hu_or_big_gain", True) and not big_gain:
        reject = reject or "chi_consumes_wildcard_without_big_gain"
    claim_context = _context_after_consumption(
        context,
        consumed_ids,
        meld_type=meld_type,
        meld_cards=meld_cards,
        plan_melds=plan_melds,
    )
    followup = _cached_best_followup_discard(
        claim_context,
        followup_discard_cache,
    )
    after_context = (
        _context_after_discard_id(claim_context, str(followup["card_id"]))
        if followup and followup.get("card_id")
        else claim_context
    )
    (
        after_hand_value,
        after_min_xi,
        after_ting_estimate,
        after_red_black_bonus,
        after_wildcard_flexibility,
    ) = _response_after_analysis_values(claim_context, after_context)
    ting_reject, ting_reject_explanation = _response_ting_reject_reason(
        "CHI",
        analysis.ting_estimate,
        after_ting_estimate,
    )
    if ting_reject and not _preserve_response_reject_over_ting("CHI", reject):
        reject = ting_reject
        reject_explanation = ting_reject_explanation
    ting_value_delta = _ting_value_delta(
        analysis.ting_estimate,
        after_ting_estimate,
        weights,
        include_wait_breadth=False,
    )
    raw_structure_cost = structure_cost
    high_xi_soft_cost_relief = _high_xi_123_soft_mixed_cost_relief(
        context,
        meld_type=meld_type,
        xi_gain=xi_gain,
        impact=impact,
        before=analysis.ting_estimate,
        after=after_ting_estimate,
        followup=followup,
    )
    structure_cost = max(
        0.0,
        structure_cost - high_xi_soft_cost_relief,
    )
    if high_xi_soft_cost_relief:
        cost_reasons.append(
            "6-xi 123 replaces only zero-xi mixed soft structures"
        )
    forced_followup_loss = 0.0
    if followup is None:
        forced_followup_loss += 200.0
        reject = reject or "chi_followup_missing"
    else:
        forced_followup_loss += float(followup["danger_score"])
        if followup["breaks_hard"]:
            forced_followup_loss += float(weights.get("hard_protected_break_penalty", 10000))
            reject = reject or "chi_followup_forces_hard_break"
        if followup["is_wildcard"]:
            forced_followup_loss += float(weights.get("wildcard_discard_penalty", 10000))
            reject = reject or "chi_followup_discards_wildcard"
    information_set = _information_set_draw_value(
        after_context,
        after_ting_estimate,
        newly_visible=[str(followup["label"])] if followup and followup.get("label") else [],
    )
    early_pair_flexibility_cost = (
        _early_zero_xi_mixed_chi_flexibility_cost(
            context,
            meld_type=meld_type,
            plan_melds=plan_melds,
            consumed_labels=list(candidate["consumed_labels"]),
            xi_gain=xi_gain,
            impact=impact,
            before=analysis.ting_estimate,
            after=after_ting_estimate,
            followup=followup,
            weights=weights,
            before_analysis=analysis,
            after_information_set=information_set,
        )
    )
    structure_cost += early_pair_flexibility_cost
    if early_pair_flexibility_cost:
        cost_reasons.append(
            "early zero-xi mixed claim spends an unimproved pair"
        )
    exposure_cost = 20.0
    uncertainty_penalty = action.uncertainty * 100
    ev = (
        after_hand_value
        + structure_gain
        + ting_value_delta
        - structure_cost
        - forced_followup_loss
        - exposure_cost
        - uncertainty_penalty
        + information_set["value"]
    )
    if ev < pass_ev + min_gain:
        reject = reject or "chi_ev_not_enough"
    new_ting_override = _can_override_chi_soft_cost_for_new_ting(
        reject=reject,
        meld_type=meld_type,
        impact=impact,
        before=analysis.ting_estimate,
        after=after_ting_estimate,
        min_xi=after_min_xi,
        xi_gain=xi_gain,
        ting_value_delta=ting_value_delta,
        followup=followup,
    )
    if new_ting_override:
        reject = None
        ev = max(ev, pass_ev + min_gain + 1.0)
    trusted_override = bool(
        new_ting_override or high_xi_soft_cost_relief
    )
    return {
        "allowed": reject is None,
        "reject_reason": reject,
        "ev": round(ev, 3),
        "score_gain": round(structure_gain, 3),
        "hand_value_after": after_hand_value,
        "xi_gain": xi_gain,
        "ting_gain": analysis.ting_estimate.shanten_like_distance - after_ting_estimate.shanten_like_distance,
        "ting_value_delta": ting_value_delta,
        "information_set": information_set,
        "red_black_gain": after_red_black_bonus - analysis.red_black_plan.bonus_estimate,
        "wildcard_gain": after_wildcard_flexibility
        - analysis.wildcard_value.future_flexibility_score,
        "structure_loss": structure_cost,
        "raw_structure_loss": raw_structure_cost,
        "structure_loss_relief": high_xi_soft_cost_relief,
        "pair_flexibility_cost": early_pair_flexibility_cost,
        "danger_loss": followup["danger_score"] if followup else 0.0,
        "opponent_gain_risk": exposure_cost,
        "forced_followup_loss": forced_followup_loss,
        "reason": (
            "chi_new_ting_overrides_soft_sequence_cost"
            if new_ting_override
            else "chi_high_xi_123_revalues_zero_xi_soft_mixed_cost"
            if high_xi_soft_cost_relief
            else "chi_early_zero_xi_mixed_preserves_pair_flexibility"
            if early_pair_flexibility_cost and reject is not None
            else reject_explanation
            or
            f"CHI {''.join(meld_cards)} consumes {''.join(candidate['consumed_labels'])}; "
            f"pass_ev={pass_ev:.1f} chi_ev={ev:.1f}"
        ),
        "consumed_card_ids": consumed_ids,
        "consumed_from_hand": candidate["consumed_labels"],
        "followup_discard": followup,
        "breaks_melds": breaks_melds,
        "cost_reasons": cost_reasons,
        "consumption_impact": impact,
        "meld_type": meld_type,
        "meld_cards": meld_cards,
        "compare_groups": candidate.get("compare_groups") or [],
        "plan_melds": plan_melds,
        "recognized_option_cards": option_cards,
        "trusted_visible_response_override": trusted_override,
        "xi_scored_in_hand_value": True,
        "ting_before": analysis.ting_estimate.to_dict(),
        "ting_after": after_ting_estimate.to_dict(),
    }


def _score_peng_candidate(
    context: DecisionContext,
    action: LegalAction,
    analysis: HandAnalysis,
    allocation: StructureAllocation,
    candidate: dict[str, Any],
    min_gain: float,
    pass_ev: float,
    followup_discard_cache: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    weights = context.rules.get("weights") or context.rules.get("ai_weights", {})
    option_cards = candidate["option_cards"]
    xi_gain = _meld_xi_value("peng", option_cards, context.rules)
    structure_gain = _claim_structure_value("peng", context.rules)
    consumed_ids = candidate["consumed_card_ids"]
    impact = _consumption_impact(
        consumed_ids,
        allocation,
        weights,
        replacement_melds=[{"type": "peng", "cards": option_cards}],
    )
    structure_cost = float(impact["structure_loss"])
    cost_reasons = list(impact["notes"])
    breaks_melds = list(impact["breaks_melds"])
    breaks_hard = bool(impact["breaks_hard"])
    reject: str | None = None
    reject_explanation: str | None = None
    if breaks_hard:
        reject = "peng_consumes_hard_protected"
    structural_reject, structural_reason = _response_reject_from_consumption(
        action_type="PENG",
        meld_type="peng",
        impact=impact,
        big_gain=xi_gain > 0,
    )
    if structural_reject:
        reject = reject or structural_reject
        reject_explanation = structural_reason
    claim_context = _context_after_consumption(
        context,
        consumed_ids,
        meld_type="peng",
        meld_cards=option_cards,
    )
    followup = _cached_best_followup_discard(
        claim_context,
        followup_discard_cache,
    )
    after_context = (
        _context_after_discard_id(claim_context, str(followup["card_id"]))
        if followup and followup.get("card_id")
        else claim_context
    )
    (
        after_hand_value,
        _after_min_xi,
        after_ting_estimate,
        after_red_black_bonus,
        after_wildcard_flexibility,
    ) = _response_after_analysis_values(claim_context, after_context)
    ting_reject, ting_reject_explanation = _response_ting_reject_reason(
        "PENG",
        analysis.ting_estimate,
        after_ting_estimate,
    )
    if ting_reject and not _preserve_response_reject_over_ting("PENG", reject):
        reject = ting_reject
        reject_explanation = ting_reject_explanation
    ting_value_delta = _ting_value_delta(
        analysis.ting_estimate,
        after_ting_estimate,
        weights,
        include_wait_breadth=False,
    )
    raw_structure_cost = structure_cost
    expanded_wait_soft_cost_relief = (
        _expanded_wait_peng_soft_mixed_cost_relief(
            context,
            xi_gain=xi_gain,
            impact=impact,
            before=analysis.ting_estimate,
            after=after_ting_estimate,
            followup=followup,
        )
    )
    structure_cost = max(
        0.0,
        structure_cost - expanded_wait_soft_cost_relief,
    )
    if expanded_wait_soft_cost_relief:
        cost_reasons.append(
            "expanded ting waits replace only zero-xi mixed soft structures"
        )
    forced_followup_loss = 0.0
    if followup is None:
        forced_followup_loss += 200.0
        if after_context.card_instances and after_allocation.hard_protected_instances and not after_allocation.free_discard_instances:
            forced_followup_loss += float(weights.get("hard_protected_break_penalty", 10000))
            reject = "peng_followup_forces_hard_break"
        else:
            reject = reject or "peng_followup_missing"
    else:
        forced_followup_loss += float(followup["danger_score"])
        if followup["breaks_hard"]:
            forced_followup_loss += float(weights.get("hard_protected_break_penalty", 10000))
            reject = reject or "peng_followup_forces_hard_break"
        if followup["is_wildcard"]:
            forced_followup_loss += float(weights.get("wildcard_discard_penalty", 10000))
            reject = reject or "peng_followup_discards_wildcard"
    information_set = _information_set_draw_value(
        after_context,
        after_ting_estimate,
        newly_visible=[str(followup["label"])] if followup and followup.get("label") else [],
    )
    exposure_cost = 25.0
    ev = (
        after_hand_value
        + structure_gain
        + ting_value_delta
        - structure_cost
        - forced_followup_loss
        - exposure_cost
        + information_set["value"]
    )
    if ev < pass_ev + min_gain:
        reject = reject or "peng_ev_not_enough"
    trusted_override = bool(expanded_wait_soft_cost_relief)
    return {
        "allowed": reject is None,
        "reject_reason": reject,
        "ev": round(ev, 3),
        "score_gain": round(structure_gain, 3),
        "hand_value_after": after_hand_value,
        "xi_gain": xi_gain,
        "ting_gain": analysis.ting_estimate.shanten_like_distance - after_ting_estimate.shanten_like_distance,
        "ting_value_delta": ting_value_delta,
        "information_set": information_set,
        "red_black_gain": after_red_black_bonus - analysis.red_black_plan.bonus_estimate,
        "wildcard_gain": after_wildcard_flexibility
        - analysis.wildcard_value.future_flexibility_score,
        "structure_loss": structure_cost,
        "raw_structure_loss": raw_structure_cost,
        "structure_loss_relief": expanded_wait_soft_cost_relief,
        "danger_loss": followup["danger_score"] if followup else 0.0,
        "opponent_gain_risk": exposure_cost,
        "forced_followup_loss": forced_followup_loss,
        "reason": (
            "peng_expanded_waits_revalue_zero_xi_soft_mixed_cost"
            if trusted_override
            else reject_explanation
            or
            f"PENG {candidate['external_label']} consumes {''.join(candidate['consumed_labels'])}; "
            f"pass_ev={pass_ev:.1f} peng_ev={ev:.1f}"
        ),
        "consumed_card_ids": consumed_ids,
        "consumed_from_hand": candidate["consumed_labels"],
        "followup_discard": followup,
        "breaks_melds": breaks_melds,
        "cost_reasons": cost_reasons,
        "consumption_impact": impact,
        "meld_type": "peng",
        "trusted_visible_response_override": trusted_override,
        "xi_scored_in_hand_value": True,
        "ting_before": analysis.ting_estimate.to_dict(),
        "ting_after": after_ting_estimate.to_dict(),
    }


def choose_action(
    state_or_hand: dict[str, Any] | list[str],
    *,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
    parallel_evaluation: bool = False,
    absolute_deadline: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> PolicyDecision:
    active_hu_cache = _TING_HU_PROBE_CACHE.get()
    if active_hu_cache is not None:
        return _choose_action_impl(
            state_or_hand,
            rules=rules,
            config_path=config_path,
            parallel_evaluation=parallel_evaluation,
            absolute_deadline=absolute_deadline,
            cancelled=cancelled,
        )
    hu_token = _TING_HU_PROBE_CACHE.set({})
    draw_token = _TING_DRAW_FEATURE_CACHE.set({})
    try:
        return _choose_action_impl(
            state_or_hand,
            rules=rules,
            config_path=config_path,
            parallel_evaluation=parallel_evaluation,
            absolute_deadline=absolute_deadline,
            cancelled=cancelled,
        )
    finally:
        _TING_DRAW_FEATURE_CACHE.reset(draw_token)
        _TING_HU_PROBE_CACHE.reset(hu_token)


def _choose_action_impl(
    state_or_hand: dict[str, Any] | list[str],
    *,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
    parallel_evaluation: bool = False,
    absolute_deadline: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> PolicyDecision:
    _raise_if_policy_evaluation_interrupted(absolute_deadline, cancelled)
    context = build_decision_context(state_or_hand, rules=rules, config_path=config_path)
    _raise_if_policy_evaluation_interrupted(absolute_deadline, cancelled)
    allocation = allocate_hand_structures(context)
    _raise_if_policy_evaluation_interrupted(absolute_deadline, cancelled)
    analysis = analyze_hand(context, allocation)
    _raise_if_policy_evaluation_interrupted(absolute_deadline, cancelled)
    legal_actions, rejected_actions = generate_legal_actions(context, allocation, analysis)
    evals = _evaluate_legal_actions(
        context,
        legal_actions,
        allocation,
        analysis,
        parallel=parallel_evaluation,
        absolute_deadline=absolute_deadline,
        cancelled=cancelled,
    )
    _raise_if_policy_evaluation_interrupted(absolute_deadline, cancelled)
    evals.extend(_rejected_eval(action, action.reject_reason or "rejected_by_rules", analysis) for action in rejected_actions)
    evals = _apply_discard_ting_guard(evals, analysis)
    safety_flags: list[str] = []
    if context.recognition_errors or context.recognition_warnings and context.rules.get("safety", {}).get(
        "halt_on_uncertain_recognition",
        True,
    ):
        safety_flags.append("recognition_uncertain")
    if allocation.allocation_conflicts:
        safety_flags.append("duplicate_card_allocation")
    pool_stage, pool_ids = _discard_pools(allocation)
    if (
        pool_stage == "forced_break_hard_protection"
        and allocation.free_discard_instances
        and context.rules.get("safety", {}).get(
            "halt_if_only_hard_protected_clickable_but_free_cards_exist",
            True,
        )
    ):
        safety_flags.append("only_hard_protected_clickable_but_unprotected_cards_exist_in_hand")
    if safety_flags:
        return _safe_halt_decision(context, allocation, analysis, evals, safety_flags[0])
    allowed = [item for item in evals if item.allowed]
    if not allowed:
        return _safe_halt_decision(context, allocation, analysis, evals, "no_clickable_cards")
    allowed_hu = [item for item in allowed if item.type == "HU"]
    selected = max(allowed_hu or allowed, key=lambda item: item.ev)
    stage = selected.debug_details.get("candidate_stage", selected.action.type.lower())
    return PolicyDecision(
        selected_action=selected.action.type,
        selected_card_id=selected.action.card_id,
        selected_label=selected.action.label,
        selected_option_id=selected.action.option_id,
        candidate_stage=str(stage),
        ev=selected.ev,
        reason=selected.reason,
        action_evals=sorted(evals, key=lambda item: item.ev, reverse=True),
        context_snapshot=_context_snapshot(context, allocation, analysis, legal_actions, rejected_actions),
        requires_action_plan=selected.action.type
        in {"DISCARD", "HU", "CHI", "EXPAND_CHI_OPTIONS", "PENG", "PAO", "TI", "PASS"},
        safety_flags=safety_flags,
    )


def _apply_discard_ting_guard(evals: list[ActionEval], analysis: HandAnalysis) -> list[ActionEval]:
    if not analysis.ting_estimate.is_ting:
        return evals
    preserving_ids = {
        item.action.card_id
        for item in evals
        if item.type == "DISCARD"
        and item.allowed
        and bool((item.debug_details.get("ting_after") or {}).get("is_ting"))
    }
    if not preserving_ids:
        return evals
    guarded: list[ActionEval] = []
    for item in evals:
        if item.type != "DISCARD" or not item.allowed or item.action.card_id in preserving_ids:
            guarded.append(item)
            continue
        debug_details = deepcopy(item.debug_details)
        debug_details["ting_guard"] = "rejected_because_another_discard_preserves_ting"
        guarded.append(
            replace(
                item,
                allowed=False,
                ev=HARD_BREAK_EV,
                reject_reason="discard_breaks_ting",
                reason="discard would break ting while another legal discard preserves it",
                debug_details=debug_details,
            )
        )
    return guarded


def _safe_halt_decision(
    context: DecisionContext,
    allocation: StructureAllocation,
    analysis: HandAnalysis,
    evals: list[ActionEval],
    reason: str,
) -> PolicyDecision:
    halt_reason = reason if reason in STANDARD_SAFE_HALT else "unknown_error"
    return PolicyDecision(
        selected_action="SAFE_HALT",
        selected_card_id=None,
        selected_label=None,
        selected_option_id=None,
        candidate_stage="safe_halt",
        ev=HARD_BREAK_EV,
        reason=halt_reason,
        action_evals=sorted(evals, key=lambda item: item.ev, reverse=True),
        context_snapshot=_context_snapshot(context, allocation, analysis, [], []),
        requires_action_plan=False,
        safety_flags=[halt_reason],
    )


def _context_snapshot(
    context: DecisionContext,
    allocation: StructureAllocation,
    analysis: HandAnalysis,
    legal_actions: list[LegalAction],
    rejected_actions: list[LegalAction],
) -> dict[str, Any]:
    return {
        "context": context.to_dict(),
        "structure_allocation": allocation.to_dict(),
        "hand_analysis": analysis.to_dict(),
        "legal_actions": [action.to_dict() for action in legal_actions],
        "rejected_actions": [action.to_dict() for action in rejected_actions],
    }


class PolicyBrain:
    def __init__(self, *, rules: dict[str, Any] | None = None, config_path: str = "config/rules.yaml") -> None:
        self.rules = rules
        self.config_path = config_path
        self.rule_config = RuleConfig(rules or load_rules(config_path))

    def choose_action(self, state_or_hand: dict[str, Any] | list[str]) -> PolicyDecision:
        return choose_action(state_or_hand, rules=self.rules, config_path=self.config_path)


def validate_decision_consistency(
    context: DecisionContext,
    decision: PolicyDecision,
    action_plan: dict[str, Any] | None = None,
) -> GuardResult:
    checks: list[str] = []
    details: dict[str, Any] = {}
    snapshot = decision.context_snapshot
    allocation_payload = snapshot.get("structure_allocation", {})
    if allocation_payload.get("allocation_conflicts"):
        return GuardResult(False, True, "conflict_duplicate_card_allocation", checks, allocation_payload)
    checks.append("allocation_no_overlap")

    snapshot_context = snapshot.get("context", {})
    expected_fingerprint = snapshot_context.get("state_fingerprint")
    actual_fingerprint = context.state_fingerprint
    if expected_fingerprint and actual_fingerprint != expected_fingerprint:
        actual_context = context.to_dict()
        changed_state_fields = sorted(
            key
            for key in set(snapshot_context) | set(actual_context)
            if key != "state_fingerprint" and snapshot_context.get(key) != actual_context.get(key)
        )
        return GuardResult(
            False,
            True,
            "conflict_state_mutation",
            checks,
            {
                "expected_state_fingerprint": expected_fingerprint,
                "actual_state_fingerprint": actual_fingerprint,
                "decision_context_id": snapshot_context.get("context_id"),
                "current_context_id": context.context_id,
                "decision_frame_id": snapshot_context.get("frame_id"),
                "current_frame_id": context.frame_id,
                "changed_state_fields": changed_state_fields,
            },
        )
    checks.append("state_unchanged_since_policy_decision")

    legal_actions = snapshot.get("legal_actions", [])
    legal_types = {item.get("type") for item in legal_actions}
    if decision.selected_action != "SAFE_HALT" and decision.selected_action not in legal_types:
        return GuardResult(False, True, "conflict_policy_action_not_legal", checks, {"legal_types": list(legal_types)})
    checks.append("policy_action_legal")

    hard_ids = set(allocation_payload.get("hard_protected_card_ids") or allocation_payload.get("hard_protected_instances") or [])
    free_ids = set(allocation_payload.get("free_discard_card_ids") or [])
    if (
        decision.selected_action == "DISCARD"
        and decision.selected_card_id in hard_ids
        and decision.candidate_stage != "forced_break_hard_protection"
    ):
        return GuardResult(False, True, "conflict_hard_protected_discard", checks, {})
    if (
        decision.selected_action == "DISCARD"
        and decision.selected_card_id in hard_ids
        and decision.candidate_stage == "forced_break_hard_protection"
        and free_ids
    ):
        return GuardResult(
            False,
            True,
            "conflict_only_hard_protected_clickable_but_free_cards_exist",
            checks,
            {
                "selected_card_id": decision.selected_card_id,
                "hard_protected_card_ids": sorted(hard_ids),
                "free_discard_card_ids": sorted(free_ids),
            },
        )
    checks.append("hard_protection_consistent")

    if context.recognition_warnings or context.recognition_errors:
        return GuardResult(False, True, "conflict_uncertain_recognition", checks, {})
    checks.append("recognition_confidence_ok")

    if action_plan is not None:
        if not action_plan.get("ready") and not action_plan.get("reason"):
            return GuardResult(False, True, "conflict_unknown", checks, {"action_plan": action_plan})
        if decision.selected_action == "EXPAND_CHI_OPTIONS":
            if action_plan.get("action") != "expand_chi_options":
                return GuardResult(
                    False,
                    True,
                    "conflict_action_plan_reselected_card",
                    checks,
                    {"action_plan": action_plan},
                )
            if context.chi_options or context.compare_options:
                return GuardResult(
                    False,
                    True,
                    "conflict_policy_action_not_legal",
                    checks,
                    {"reason": "candidate_options_already_visible"},
                )
            click_targets = [
                str(click.get("target") or "")
                for click in action_plan.get("clicks", []) or []
                if isinstance(click, dict)
            ]
            if action_plan.get("ready") and click_targets != ["button:chi"]:
                return GuardResult(
                    False,
                    True,
                    "conflict_action_plan_reselected_card",
                    checks,
                    {"action_plan": action_plan},
                )
            checks.append("expand_chi_options_button_only")
        plan_label = action_plan.get("target_label")
        plan_card_id = action_plan.get("target_card_id")
        if decision.selected_action == "DISCARD":
            if plan_card_id and decision.selected_card_id and plan_card_id != decision.selected_card_id:
                return GuardResult(False, True, "conflict_action_plan_reselected_card", checks, action_plan)
            if plan_label and decision.selected_label and normalize_card_label(plan_label) != decision.selected_label:
                return GuardResult(False, True, "conflict_action_plan_reselected_card", checks, action_plan)
            if plan_card_id in hard_ids and free_ids:
                return GuardResult(
                    False,
                    True,
                    "conflict_action_plan_targets_hard_protected_while_free_exists",
                    checks,
                    {
                        "action_plan": action_plan,
                        "hard_protected_card_ids": sorted(hard_ids),
                        "free_discard_card_ids": sorted(free_ids),
                    },
                )
        if action_plan.get("action") == "compare_option":
            compare_options = list(context.compare_options or snapshot_context.get("compare_options") or [])
            plan_option_id = action_plan.get("target_option_id")
            plan_option_cards = normalize_cards(action_plan.get("target_option_cards") or [])
            matched_compare = any(
                (
                    plan_option_id
                    and plan_option_id
                    == str(option.get("option_id") or f"compare_{int(option.get('index') or index):03d}")
                )
                or (
                    plan_option_cards
                    and plan_option_cards
                    == normalize_cards([str(label) for label in option.get("labels", [])])
                )
                for index, option in enumerate(compare_options, start=1)
                if isinstance(option, dict)
            )
            if action_plan.get("ready") and not matched_compare:
                return GuardResult(
                    False,
                    True,
                    "conflict_action_plan_reselected_card",
                    checks,
                    {
                        "action_plan": action_plan,
                        "compare_options": compare_options,
                    },
                )
            checks.append("compare_option_matches_visible_candidate")
        if decision.selected_action == "CHI" and action_plan.get("action") != "compare_option":
            plan_option_id = action_plan.get("target_option_id")
            if (
                action_plan.get("ready")
                and decision.selected_option_id
                and plan_option_id != decision.selected_option_id
            ):
                plan_option_cards = normalize_cards(action_plan.get("target_option_cards") or [])
                chi_options = list(context.chi_options or snapshot_context.get("chi_options") or [])
                matched_chi_option = any(
                    (
                        plan_option_id
                        and plan_option_id
                        == str(option.get("option_id") or f"chi_{int(option.get('index') or index):03d}")
                    )
                    or (
                        plan_option_cards
                        and plan_option_cards
                        == normalize_cards([str(label) for label in option.get("labels", [])])
                    )
                    for index, option in enumerate(chi_options, start=1)
                    if isinstance(option, dict)
                )
                plan_validation = action_plan.get("validation") if isinstance(action_plan.get("validation"), dict) else {}
                if matched_chi_option and plan_validation.get("reason_code") == "option_stage_evaluated":
                    checks.append("chi_option_stage_override_matches_visible_candidate")
                    return GuardResult(True, False, None, checks, details)
                return GuardResult(
                    False,
                    True,
                    "conflict_action_plan_reselected_card",
                    checks,
                    {
                        "selected_option_id": decision.selected_option_id,
                        "action_plan": action_plan,
                    },
                )
        checks.append("action_plan_matches_policy")

    return GuardResult(True, False, None, checks, details)


def run_decision_self_check(
    context: DecisionContext,
    decision: PolicyDecision,
    action_plan: dict[str, Any] | None = None,
) -> GuardResult:
    return validate_decision_consistency(context, decision, action_plan)
