from __future__ import annotations

from dataclasses import dataclass

from aizipai.state.models import Action, ActionType, Card, Color, GameState


@dataclass(frozen=True)
class ActionScore:
    action: Action
    score: float
    reason: str


class HeuristicAgent:
    """首版可解释启发式 AI。"""

    def rank_actions(self, state: GameState, actions: list[Action]) -> list[ActionScore]:
        scored = [self._score_action(state, action) for action in actions]
        return sorted(scored, key=lambda item: item.score, reverse=True)

    def choose(self, state: GameState, actions: list[Action]) -> ActionScore:
        ranked = self.rank_actions(state, actions)
        return ranked[0]

    def _score_action(self, state: GameState, action: Action) -> ActionScore:
        if action.type == ActionType.HU:
            return ActionScore(action, 10_000, "胡牌优先")

        if action.type in {ActionType.TI, ActionType.PAO}:
            return ActionScore(action, 5_000, "提跑通常是高价值动作")

        if action.type == ActionType.PENG:
            return ActionScore(action, 1_000, "碰牌保留成组价值")

        if action.type == ActionType.CHI:
            return ActionScore(action, 800, "吃牌形成组合")

        if action.type == ActionType.DISCARD and action.card is not None:
            return self._score_discard(state, action, action.card)

        return ActionScore(action, 0, "默认动作")

    def _score_discard(self, state: GameState, action: Action, card: Card) -> ActionScore:
        score = 100.0
        reasons: list[str] = []

        same_rank_count = sum(1 for item in state.hand if item.rank == card.rank and item.suit == card.suit)
        if same_rank_count >= 2:
            score -= 80
            reasons.append("不轻易拆对子或坎")

        if card.color == Color.RED:
            score -= 30
            reasons.append("红牌暂时保留")

        if card.is_wildcard:
            score -= 200
            reasons.append("王牌癞子高价值保留")

        if not reasons:
            reasons.append("孤张优先考虑打出")

        return ActionScore(action, score, "；".join(reasons))

