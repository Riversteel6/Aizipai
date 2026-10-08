"""Action selection policy."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ai.evaluator import DiscardEvaluation
from ai.hand_protection import HandProtection, analyze_hand_protection
from ai.pro_brain import (
    choose_action as choose_professional_action,
)
from engine.cards import normalize_cards
from engine.hu_checker import HuBreakdown


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    label: str | None
    score: float
    reason: str
    evaluations: list[DiscardEvaluation]
    candidate_stage: str = "unknown"
    break_hard_protection: bool = False
    break_reason: str | None = None
    hard_protected: list[str] = field(default_factory=list)
    soft_protected: list[str] = field(default_factory=list)
    hu_breakdown: HuBreakdown | None = None
    response_evaluations: list[dict] | None = None
    raw_hand: list[str] = field(default_factory=list)
    normalized_hand: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    orphan_cards: list[str] = field(default_factory=list)
    low_value_singles: list[str] = field(default_factory=list)
    clickable_cards: list[str] = field(default_factory=list)
    discard_candidates_before_filter: list[str] = field(default_factory=list)
    discard_candidates_after_hard_filter: list[str] = field(default_factory=list)
    discard_candidates_after_soft_filter: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    selected_reason: str | None = None
    protected_melds: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "policy_action": self.action,
            "label": self.label,
            "score": self.score,
            "reason": self.reason,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "candidate_stage": self.candidate_stage,
            "break_hard_protection": self.break_hard_protection,
            "break_reason": self.break_reason,
            "protected_melds": self.protected_melds,
            "hard_protected": self.hard_protected,
            "soft_protected": self.soft_protected,
            "selected_discard": self.label,
            "selected_reason": self.selected_reason,
            "raw_hand": self.raw_hand,
            "normalized_hand": self.normalized_hand,
            "counts": self.counts,
            "orphan_cards": self.orphan_cards,
            "low_value_singles": self.low_value_singles,
            "clickable_cards": self.clickable_cards,
            "discard_candidates_before_filter": self.discard_candidates_before_filter,
            "discard_candidates_after_hard_filter": self.discard_candidates_after_hard_filter,
            "discard_candidates_after_soft_filter": self.discard_candidates_after_soft_filter,
            "scores": self.scores,
            "hu_breakdown": self.hu_breakdown.to_dict() if self.hu_breakdown else None,
            "response_evaluations": self.response_evaluations or [],
        }


def _professional_state_for_discard(
    hand: list[str],
    *,
    clickable_cards: list[str] | None = None,
    memory: dict | None = None,
    remaining_deck_count: int | None = None,
) -> dict:
    normalized = normalize_cards(hand)
    clickable_counter = Counter(normalize_cards(clickable_cards or normalized))
    hand_details = []
    for index, label in enumerate(normalized, start=1):
        if clickable_cards is None:
            clickable = True
        else:
            clickable = clickable_counter[label] > 0
            if clickable:
                clickable_counter[label] -= 1
        hand_details.append(
            {
                "card_id": f"h{index:03d}",
                "name": label,
                "label": label,
                "clickable": clickable,
            }
        )
    state = {
        "hand": normalized,
        "hand_details": hand_details,
        "legal_actions": [{"type": "DISCARD"}],
    }
    if memory is not None:
        state["memory"] = memory
    if remaining_deck_count is not None:
        state["remaining_deck_count"] = remaining_deck_count
    return state


def _professional_state_for_response(
    hand: list[str],
    *,
    legal_actions: list[str] | None,
    option_details: list[dict] | None,
    option_count: int,
    memory: dict | None,
    remaining_deck_count: int | None,
) -> dict:
    normalized_actions = [str(item).upper() for item in (legal_actions or ["PASS"])]
    state = _professional_state_for_discard(
        hand,
        memory=memory,
        remaining_deck_count=remaining_deck_count,
    )
    state["legal_actions"] = [{"type": action} for action in normalized_actions]
    if option_details:
        state["chi_options"] = [
            {
                "option_id": str(item.get("option_id") or item.get("index") or f"chi_{idx:03d}"),
                "labels": item.get("labels", []),
                "confidence": item.get("confidence", 1.0),
                "center": item.get("center"),
            }
            for idx, item in enumerate(option_details, start=1)
            if isinstance(item, dict) and (item.get("region_name") in {None, "chi_options"} or item.get("labels"))
        ]
    elif "CHI" in normalized_actions and option_count <= 0:
        state["chi_options"] = []
    return state


def _legacy_decision_from_professional(professional) -> PolicyDecision:
    snapshot = professional.context_snapshot
    context = snapshot.get("context", {})
    allocation = snapshot.get("structure_allocation", {})
    action = professional.action
    hard_protected = list(allocation.get("hard_protected_labels", []))
    soft_protected = list(allocation.get("soft_protected_labels", []))
    free_cards = [
        item.get("label")
        for item in allocation.get("free_discard_instances", [])
        if isinstance(item, dict) and item.get("label")
    ]
    hard_ids = set(allocation.get("hard_protected_card_ids") or allocation.get("hard_protected_instances") or [])
    soft_ids = set(allocation.get("soft_protected_card_ids") or allocation.get("soft_protected_instances") or [])
    response_evals = _legacy_response_evaluations(professional)
    discard_evals = [item for item in professional.action_evals if item.action.type == "DISCARD"]
    break_hard = professional.candidate_stage == "forced_break_hard_protection"
    break_reason = None
    if break_hard:
        break_reason = "all_clickable_cards_are_hard_protected"
    elif professional.selected_action == "SAFE_HALT":
        break_reason = professional.reason
    return PolicyDecision(
        action=action,
        label=professional.selected_label,
        score=professional.ev,
        reason=professional.reason,
        evaluations=discard_evals,
        candidate_stage=professional.candidate_stage,
        break_hard_protection=break_hard,
        break_reason=break_reason,
        hard_protected=hard_protected,
        soft_protected=soft_protected,
        hu_breakdown=None,
        response_evaluations=response_evals,
        raw_hand=list(context.get("raw_hand", [])),
        normalized_hand=list(context.get("normalized_hand", [])),
        counts=dict(allocation.get("total_counts", {})),
        orphan_cards=list(allocation.get("orphan_cards", [])),
        low_value_singles=free_cards,
        clickable_cards=[
            card.get("label")
            for card in context.get("card_instances", [])
            if isinstance(card, dict) and card.get("clickable", True)
        ],
        discard_candidates_before_filter=[
            item.action.label for item in discard_evals if item.action.label
        ],
        discard_candidates_after_hard_filter=[
            item.action.label
            for item in discard_evals
            if item.action.label and item.action.card_id not in hard_ids
        ],
        discard_candidates_after_soft_filter=[
            item.action.label
            for item in discard_evals
            if item.action.label and item.action.card_id not in hard_ids and item.action.card_id not in soft_ids
        ],
        scores={
            item.action.label: item.ev
            for item in discard_evals
            if item.action.label
        },
        selected_reason=professional.reason,
        protected_melds=list(allocation.get("protected_melds", [])),
    )


def _legacy_response_evaluations(professional) -> list[dict]:
    rows = []
    for item in professional.action_evals:
        if item.action.type not in {"CHI", "EXPAND_CHI_OPTIONS", "PENG", "PASS", "HU"}:
            continue
        rows.append(
            {
                "action": item.action.type.lower(),
                "score": item.ev,
                "reason": item.reason,
                "allowed": item.allowed,
                "reject_reason": item.reject_reason,
                "ev_delta_vs_pass": item.debug_details.get("ev_delta_vs_pass"),
                "option_id": item.action.option_id,
                "label": item.action.label,
            }
        )
    return sorted(rows, key=lambda item: item["score"], reverse=True)


def choose_discard(
    hand: list[str],
    *,
    legal_actions: list[str] | None = None,
    clickable_cards: list[str] | None = None,
    memory: dict | None = None,
    remaining_deck_count: int | None = None,
    config_path: str = "config/rules.yaml",
    rules: dict | None = None,
    protection: HandProtection | None = None,
) -> PolicyDecision:
    """Compatibility wrapper over the single professional policy brain."""

    raw_hand = list(hand)
    normalized_hand = normalize_cards(raw_hand)
    if not normalized_hand:
        return _safe_halt(
            raw_hand=raw_hand,
            normalized_hand=normalized_hand,
            reason="SAFE_HALT: hand empty after normalize",
            candidate_stage="safe_halt",
            break_reason="empty_hand",
        )

    if legal_actions is not None and "discard" not in legal_actions:
        return _safe_halt(
            raw_hand=raw_hand,
            normalized_hand=normalized_hand,
            reason="当前没有可执行的出牌/胡动作",
            candidate_stage="safe_halt",
            break_reason="discard_not_legal",
        )

    state = _professional_state_for_discard(
        raw_hand,
        clickable_cards=clickable_cards,
        memory=memory,
        remaining_deck_count=remaining_deck_count,
    )
    professional = choose_professional_action(state, rules=rules, config_path=config_path)
    return _legacy_decision_from_professional(professional)


def choose_action(
    hand: list[str],
    *,
    legal_actions: list[str] | None = None,
    existing_xi: int = 0,
    hu_breakdown: HuBreakdown | None = None,
    option_count: int = 0,
    clickable_cards: list[str] | None = None,
    memory: dict | None = None,
    remaining_deck_count: int | None = None,
    config_path: str = "config/rules.yaml",
    rules: dict | None = None,
    option_details: list[dict] | None = None,
) -> PolicyDecision:
    """Compatibility wrapper; real policy selection is delegated to PolicyBrain."""

    normalized_hand = normalize_cards(hand)
    if legal_actions and "ti" in legal_actions:
        return PolicyDecision("wait_auto_meld", None, 8000.0, "提会自动操作，等待动画完成", [])
    if legal_actions and "pao" in legal_actions:
        return PolicyDecision("wait_auto_meld", None, 7000.0, "跑会自动操作，等待动画完成", [])
    if legal_actions is not None and "discard" not in legal_actions:
        state = _professional_state_for_response(
            normalized_hand,
            legal_actions=legal_actions,
            option_details=option_details,
            option_count=option_count,
            memory=memory,
            remaining_deck_count=remaining_deck_count,
        )
        professional = choose_professional_action(state, rules=rules, config_path=config_path)
        return _legacy_decision_from_professional(professional)
    if legal_actions is None:
        professional = choose_professional_action(
            _professional_state_for_discard(
                hand,
                clickable_cards=clickable_cards,
                memory=memory,
                remaining_deck_count=remaining_deck_count,
            ),
            rules=rules,
            config_path=config_path,
        )
        return _legacy_decision_from_professional(professional)
    return choose_discard(
        normalized_hand,
        legal_actions=legal_actions,
        clickable_cards=clickable_cards,
        memory=memory,
        remaining_deck_count=remaining_deck_count,
        config_path=config_path,
        rules=rules,
    )


def evaluate_response_actions(
    legal_actions: list[str],
    *,
    option_count: int = 0,
    hand: list[str] | None = None,
    option_details: list[dict] | None = None,
    memory: dict | None = None,
    remaining_deck_count: int | None = None,
    config_path: str = "config/rules.yaml",
    rules: dict | None = None,
) -> list[dict]:
    if not hand:
        return [{"action": "pass", "score": 0.0, "reason": "缺少 DecisionContext，保守过牌"}]
    state = _professional_state_for_response(
        normalize_cards(hand),
        legal_actions=legal_actions,
        option_details=option_details,
        option_count=option_count,
        memory=memory,
        remaining_deck_count=remaining_deck_count,
    )
    professional = choose_professional_action(state, rules=rules, config_path=config_path)
    return _legacy_response_evaluations(professional)


def _safe_halt(
    *,
    raw_hand: list[str],
    normalized_hand: list[str],
    reason: str,
    candidate_stage: str,
    break_reason: str,
    protection: HandProtection | None = None,
    clickable_cards: list[str] | None = None,
    candidates_before: list[str] | None = None,
    after_hard: list[str] | None = None,
    after_soft: list[str] | None = None,
    protected_melds: list[dict[str, object]] | None = None,
) -> PolicyDecision:
    protection = protection or analyze_hand_protection(normalized_hand)
    candidate_before = candidates_before or []
    return PolicyDecision(
        action="safe_halt",
        label=None,
        score=0.0,
        reason=reason,
        evaluations=[],
        candidate_stage=candidate_stage,
        break_hard_protection=False,
        break_reason=break_reason,
        hard_protected=sorted(protection.hard_protected),
        soft_protected=sorted(protection.soft_protected),
        raw_hand=raw_hand,
        normalized_hand=normalized_hand,
        counts=dict(protection.counts),
        orphan_cards=protection.orphan_cards,
        low_value_singles=protection.low_value_singles,
        clickable_cards=clickable_cards or [],
        discard_candidates_before_filter=candidate_before,
        discard_candidates_after_hard_filter=after_hard or [],
        discard_candidates_after_soft_filter=after_soft or [],
        scores={},
        protected_melds=protected_melds or [meld.to_dict() for meld in protection.protected_melds],
        selected_reason=reason,
    )
