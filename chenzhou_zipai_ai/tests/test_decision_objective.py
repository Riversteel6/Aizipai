from __future__ import annotations

import pytest

from ai.decision_objective import (
    DecisionAnchor,
    ObjectiveMode,
    OutcomeVector,
    scalar_sampling_reward,
)
from ai.full_game_simulator import GameResult


def _result(*, winner: int | None, score: float, xi: int) -> GameResult:
    return GameResult(
        seed=1,
        wildcard_enabled=True,
        winner=winner,
        dealer=0,
        turns=1,
        reason="hu" if winner is not None else "deck_exhausted",
        score=score,
        total_xi=xi,
        action_counts={},
        violations=(),
    )


def test_win_first_never_lets_score_compensate_for_a_loss() -> None:
    win = scalar_sampling_reward(
        _result(winner=0, score=0.0, xi=0),
        root_seat=0,
        mode=ObjectiveMode.WIN_FIRST,
    )
    high_score_loss = scalar_sampling_reward(
        _result(winner=1, score=10_000.0, xi=100),
        root_seat=0,
        mode=ObjectiveMode.WIN_FIRST,
    )

    assert win == 1.0
    assert high_score_loss == 0.0


def test_win_first_vector_uses_not_lose_then_score_as_tie_breakers() -> None:
    safer = OutcomeVector(0.5, 0.4, 0.1, -5.0, 0.0)
    riskier = OutcomeVector(0.5, 0.1, 0.4, 100.0, 30.0)

    assert safer.key(ObjectiveMode.WIN_FIRST) > riskier.key(ObjectiveMode.WIN_FIRST)


def test_score_first_remains_explicitly_available() -> None:
    low_score_win = scalar_sampling_reward(
        _result(winner=0, score=1.0, xi=9),
        root_seat=0,
        mode=ObjectiveMode.SCORE_FIRST,
    )
    high_score_win = scalar_sampling_reward(
        _result(winner=0, score=10.0, xi=30),
        root_seat=0,
        mode=ObjectiveMode.SCORE_FIRST,
    )

    assert high_score_win > low_score_win


def test_decision_anchor_is_immutable_and_production_only() -> None:
    anchor = DecisionAnchor("伍")

    assert anchor.label == "伍"
    assert anchor.source == "production"
    with pytest.raises(AttributeError):
        anchor.label = "八"
    with pytest.raises(ValueError, match="decision_anchor_must_be_production"):
        DecisionAnchor("伍", source="baseline")
