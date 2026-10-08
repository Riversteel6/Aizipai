"""Game state and DecisionContext helpers."""

from __future__ import annotations

from ai.pro_brain import DecisionContext, GameStateBuilder, build_decision_context


def build_game_state(state_or_hand, *, rules=None, config_path: str = "config/rules.yaml") -> DecisionContext:
    """Normalize a vision state or raw hand into the unified local context."""

    return build_decision_context(state_or_hand, rules=rules, config_path=config_path)


__all__ = ["DecisionContext", "GameStateBuilder", "build_decision_context", "build_game_state"]
