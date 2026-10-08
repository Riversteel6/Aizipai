"""Rule-engine action contract backed by the unified DecisionContext."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai.pro_brain import (
    ActionEval,
    DecisionContext,
    EVScorer,
    LegalAction,
    allocate_hand_structures,
    analyze_hand,
    build_decision_context,
    generate_legal_actions,
)


@dataclass(frozen=True)
class RuleScore:
    action: LegalAction
    legal: bool
    score: float
    reason: str
    reject_reason: str | None = None
    action_eval: ActionEval | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "legal": self.legal,
            "allowed": self.legal,
            "score": self.score,
            "ev": self.score,
            "reason": self.reason,
            "reject_reason": self.reject_reason,
            "action_eval": self.action_eval.to_dict() if self.action_eval else None,
            "details": dict(self.details),
        }


class LegalActionGenerator:
    """Small class wrapper for callers that prefer object-oriented wiring."""

    def generate(self, context, allocation=None, analysis=None):
        return generate_legal_actions(context, allocation, analysis)

    def legal_actions(self, state_or_context, *, rules=None, config_path: str = "config/rules.yaml"):
        return legal_actions(state_or_context, rules=rules, config_path=config_path)

    def is_legal(self, state_or_context, action, *, rules=None, config_path: str = "config/rules.yaml") -> bool:
        return is_legal(state_or_context, action, rules=rules, config_path=config_path)

    def score_action(self, state_or_context, action, *, rules=None, config_path: str = "config/rules.yaml") -> RuleScore:
        return score_action(state_or_context, action, rules=rules, config_path=config_path)

    def is_terminal(self, state_or_context) -> bool:
        return is_terminal(state_or_context)


def legal_actions(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    *,
    include_rejected: bool = False,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
) -> list[LegalAction] | tuple[list[LegalAction], list[LegalAction]]:
    """Return legal actions for the current rule state.

    `include_rejected=True` keeps the richer internal tuple for audit callers,
    while the default matches the public rules contract: list[Action].
    """

    context, allocation, analysis = _context_bundle(state_or_context, rules=rules, config_path=config_path)
    actions, rejected = generate_legal_actions(context, allocation, analysis)
    if include_rejected:
        return actions, rejected
    return actions


def is_legal(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    action: LegalAction | dict[str, Any] | str,
    *,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
) -> bool:
    requested = _normalize_action(action)
    actions = legal_actions(state_or_context, rules=rules, config_path=config_path)
    return any(_actions_match(candidate, requested) for candidate in actions)


def score_action(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    action: LegalAction | dict[str, Any] | str,
    *,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
) -> RuleScore:
    requested = _normalize_action(action)
    context, allocation, analysis = _context_bundle(state_or_context, rules=rules, config_path=config_path)
    actions, rejected = generate_legal_actions(context, allocation, analysis)
    matches = [candidate for candidate in actions if _actions_match(candidate, requested)]
    if not matches:
        rejected_match = next((candidate for candidate in rejected if _actions_match(candidate, requested)), None)
        reject_reason = rejected_match.reject_reason if rejected_match else "action_not_legal"
        return RuleScore(
            action=rejected_match or requested,
            legal=False,
            score=0.0,
            reason=reject_reason,
            reject_reason=reject_reason,
            details={"rejected_actions": [item.to_dict() for item in rejected]},
        )
    scorer = EVScorer()
    evals = [scorer.evaluate(context, candidate, allocation, analysis) for candidate in matches]
    best = max(evals, key=lambda item: item.ev)
    return RuleScore(
        action=best.action,
        legal=best.allowed,
        score=best.ev,
        reason=best.reason,
        reject_reason=best.reject_reason,
        action_eval=best,
        details={
            "scorer": best.debug_details.get("scorer"),
            "matched_action_count": len(matches),
        },
    )


def is_terminal(state_or_context: DecisionContext | dict[str, Any] | list[str]) -> bool:
    state = state_or_context.state if isinstance(state_or_context, DecisionContext) else state_or_context
    if not isinstance(state, dict):
        return False
    if any(bool(state.get(key)) for key in ("is_terminal", "terminal", "game_over", "round_over")):
        return True
    flow_state = str(state.get("flow_state") or state.get("phase") or state.get("state") or "").lower()
    if flow_state in {"final_score", "game_over", "round_over", "terminal"}:
        return True
    return state.get("final_score") is not None and flow_state != "settlement_ready"


def _context_bundle(
    state_or_context: DecisionContext | dict[str, Any] | list[str],
    *,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
):
    context = (
        state_or_context
        if isinstance(state_or_context, DecisionContext)
        else build_decision_context(state_or_context, rules=rules, config_path=config_path)
    )
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)
    return context, allocation, analysis


def _normalize_action(action: LegalAction | dict[str, Any] | str) -> LegalAction:
    if isinstance(action, LegalAction):
        return action
    if isinstance(action, str):
        return LegalAction(action_id="requested", type=_normalize_action_type(action), source="rules_contract")
    action_type = action.get("type") or action.get("action") or action.get("selected_action")
    if isinstance(action_type, dict):
        action_type = action_type.get("type")
    return LegalAction(
        action_id=str(action.get("action_id") or action.get("id") or "requested"),
        type=_normalize_action_type(str(action_type or "")),
        source=str(action.get("source") or "rules_contract"),
        card_id=action.get("card_id") or action.get("selected_card_id"),
        label=action.get("label") or action.get("selected_label"),
        option_id=action.get("option_id") or action.get("selected_option_id"),
        option_cards=list(action.get("option_cards") or []),
    )


def _normalize_action_type(action_type: str) -> str:
    normalized = action_type.strip().upper()
    aliases = {
        "WAIT_AUTO_MELD": "WAIT_AUTO_MELD",
        "AUTO_QUAD": "AUTO_QUAD",
        "MINGLONG": "MING_LONG",
        "MING_LONG": "MING_LONG",
    }
    return aliases.get(normalized, normalized)


def _actions_match(candidate: LegalAction, requested: LegalAction) -> bool:
    if candidate.type != requested.type:
        return False
    if requested.card_id is not None and candidate.card_id != requested.card_id:
        return False
    if requested.label is not None and candidate.label != requested.label:
        return False
    if requested.option_id is not None and candidate.option_id != requested.option_id:
        return False
    if requested.option_cards and list(candidate.option_cards) != list(requested.option_cards):
        return False
    return candidate.allowed

__all__ = [
    "LegalAction",
    "LegalActionGenerator",
    "RuleScore",
    "generate_legal_actions",
    "is_legal",
    "is_terminal",
    "legal_actions",
    "score_action",
]
