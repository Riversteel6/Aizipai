"""Explicit decision outcomes and lexicographic strategy objectives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


OBJECTIVE_VERSION = "win-first-v1"


class ObjectiveMode(str, Enum):
    WIN_FIRST = "win_first"
    SCORE_FIRST = "score_first"


class OutcomeKind(str, Enum):
    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"
    INVALID = "invalid"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class OutcomeVector:
    p_win: float
    p_draw: float
    p_loss: float
    expected_score: float
    expected_xi: float
    variance: float = 0.0

    def win_first_key(self) -> tuple[float, ...]:
        return (
            self.p_win,
            -self.p_loss,
            self.expected_score,
            self.expected_xi,
            -self.variance,
        )

    def score_first_key(self) -> tuple[float, ...]:
        return (
            self.expected_score,
            self.expected_xi,
            self.p_win,
            -self.p_loss,
            -self.variance,
        )

    def key(self, mode: ObjectiveMode) -> tuple[float, ...]:
        return (
            self.win_first_key()
            if mode is ObjectiveMode.WIN_FIRST
            else self.score_first_key()
        )


@dataclass(frozen=True)
class DecisionAnchor:
    label: str
    source: str = "production"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("decision_anchor_label_missing")
        if self.source != "production":
            raise ValueError("decision_anchor_must_be_production")


def outcome_kind(result: Any, *, root_seat: int) -> OutcomeKind:
    if getattr(result, "violations", ()):
        return OutcomeKind.INVALID
    if getattr(result, "coverage_failures", ()) or getattr(result, "reason", "") in {
        "time_budget",
        "unresolved",
    }:
        return OutcomeKind.UNRESOLVED
    winner = getattr(result, "winner", None)
    if winner is not None:
        return OutcomeKind.WIN if int(winner) == root_seat else OutcomeKind.LOSS
    if getattr(result, "reason", "") == "complete_hand_below_min_xi":
        stalled = getattr(result, "stalled_seat", None)
        if stalled is not None:
            return OutcomeKind.LOSS if int(stalled) == root_seat else OutcomeKind.WIN
    return OutcomeKind.DRAW


def outcome_vector(result: Any, *, root_seat: int) -> OutcomeVector:
    kind = outcome_kind(result, root_seat=root_seat)
    if kind in {OutcomeKind.INVALID, OutcomeKind.UNRESOLVED}:
        raise ValueError(f"non_decision_outcome:{kind.value}")
    won = kind is OutcomeKind.WIN
    lost = kind is OutcomeKind.LOSS
    score = float(getattr(result, "score", 0.0))
    xi = float(getattr(result, "total_xi", 0.0))
    return OutcomeVector(
        p_win=float(won),
        p_draw=float(kind is OutcomeKind.DRAW),
        p_loss=float(lost),
        expected_score=score if won else -score if lost else 0.0,
        expected_xi=xi if won else -xi if lost else 0.0,
    )


def scalar_sampling_reward(
    result: Any,
    *,
    root_seat: int,
    mode: ObjectiveMode,
) -> float:
    vector = outcome_vector(result, root_seat=root_seat)
    if mode is ObjectiveMode.WIN_FIRST:
        # This scalar is only the root-arm sampling signal. Final ranking uses
        # the full lexicographic outcome vector, never a weighted sum.
        return vector.p_win
    magnitude = (
        1.0
        + min(10.0, abs(vector.expected_score)) * 0.05
        + min(30.0, abs(vector.expected_xi)) * 0.01
    )
    if vector.p_win:
        return magnitude
    if vector.p_loss:
        return -magnitude
    return 0.0


__all__ = [
    "OBJECTIVE_VERSION",
    "DecisionAnchor",
    "ObjectiveMode",
    "OutcomeKind",
    "OutcomeVector",
    "outcome_kind",
    "outcome_vector",
    "scalar_sampling_reward",
]
