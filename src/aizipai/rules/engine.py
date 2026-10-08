from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aizipai.ai.heuristic import HeuristicAgent
from aizipai.state.models import Action, ActionType, GameState, Phase

ACTION_PRIORITY = {
    ActionType.HU: 0,
    ActionType.PAO: 1,
    ActionType.TI: 2,
    ActionType.PENG: 3,
    ActionType.CHI: 4,
    ActionType.PASS: 5,
    ActionType.DISCARD: 6,
}


@dataclass(frozen=True)
class RuleScore:
    action: Action
    legal: bool
    score: float
    reason: str
    reject_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "legal": self.legal,
            "allowed": self.legal,
            "score": self.score,
            "reason": self.reason,
            "reject_reason": self.reject_reason,
            "details": dict(self.details),
        }


class RuleEngine:
    """规则引擎入口。

    具体郴州字牌本地规则需要通过牌例逐步补齐。当前版本只提供保守动作枚举：
    可见按钮动作优先，轮到自己时允许出任意手牌。
    """

    def legal_actions(self, state: GameState) -> list[Action]:
        actions: list[Action] = []
        opponent_priority_pending = bool(state.metadata.get("opponent_priority_pending"))
        auto_quad_pending = bool(state.metadata.get("auto_quad_pending"))

        if opponent_priority_pending:
            return [Action(type=ActionType.PASS, reason="wait_opponent_priority")]

        if auto_quad_pending:
            return [Action(type=ActionType.PASS, reason="wait_auto_quad")]

        for button in state.buttons:
            if button.visible and button.action != ActionType.DISCARD:
                actions.append(Action(type=button.action, reason="visible_button"))

        if state.phase == Phase.AWAIT_ACTION:
            actions.extend(
                Action(type=ActionType.DISCARD, card=card, reason="card_in_hand")
                for card in state.hand
            )

        if not actions:
            actions.append(Action(type=ActionType.PASS, reason="fallback_no_action"))

        return sorted(actions, key=lambda action: ACTION_PRIORITY.get(action.type, 99))

    def is_legal(self, state: GameState, action: Action) -> bool:
        return any(_actions_match(candidate, action) for candidate in self.legal_actions(state))

    def score_action(self, state: GameState, action: Action) -> RuleScore:
        legal_actions = self.legal_actions(state)
        match = next((candidate for candidate in legal_actions if _actions_match(candidate, action)), None)
        if match is None:
            return RuleScore(
                action=action,
                legal=False,
                score=0.0,
                reason="action_not_legal",
                reject_reason="action_not_legal",
                details={"legal_actions": [item.to_dict() for item in legal_actions]},
            )
        score = HeuristicAgent()._score_action(state, match)
        return RuleScore(
            action=match,
            legal=True,
            score=score.score,
            reason=score.reason,
            details={"source": "HeuristicAgent"},
        )

    def is_terminal(self, state: GameState) -> bool:
        if state.phase == Phase.ROUND_OVER:
            return True
        return any(bool(state.metadata.get(key)) for key in ("terminal", "game_over", "round_over"))


def _actions_match(candidate: Action, requested: Action) -> bool:
    if candidate.type != requested.type:
        return False
    if requested.card is not None and candidate.card != requested.card:
        return False
    if requested.target is not None and candidate.target != requested.target:
        return False
    if requested.cards and candidate.cards != requested.cards:
        return False
    return True


__all__ = ["ACTION_PRIORITY", "RuleEngine", "RuleScore"]
