from __future__ import annotations

from dataclasses import asdict, dataclass, field
try:
    from enum import StrEnum
except ImportError:  # Python 3.10 used by the Android runtime.
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)
from typing import Any


class SerializableModel:
    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class Suit(StrEnum):
    SMALL = "small"
    BIG = "big"


class Color(StrEnum):
    RED = "red"
    BLACK = "black"


class Phase(StrEnum):
    WAITING = "waiting"
    AWAIT_ACTION = "await_action"
    ANIMATING = "animating"
    ROUND_OVER = "round_over"


class ActionType(StrEnum):
    DISCARD = "discard"
    CHI = "chi"
    PENG = "peng"
    PAO = "pao"
    TI = "ti"
    HU = "hu"
    PASS = "pass"


@dataclass(frozen=True)
class Card(SerializableModel):
    rank: int
    suit: Suit
    color: Color | None = None
    is_wildcard: bool = False

    def __post_init__(self) -> None:
        if not 1 <= int(self.rank) <= 10:
            raise ValueError("rank must be between 1 and 10")


@dataclass
class Meld(SerializableModel):
    type: ActionType
    cards: list[Card]
    source_seat: int | None = None


@dataclass
class ButtonState(SerializableModel):
    action: ActionType
    visible: bool = True
    confidence: float = 1.0
    bbox: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass
class ClickPoint(SerializableModel):
    x: int
    y: int
    delay_ms: int = 80


@dataclass
class Action(SerializableModel):
    type: ActionType
    card: Card | None = None
    cards: list[Card] = field(default_factory=list)
    target: Card | None = None
    click_plan: list[ClickPoint] = field(default_factory=list)
    reason: str | None = None


@dataclass
class GameState(SerializableModel):
    round_id: str
    seat: int
    turn: int
    phase: Phase
    hand: list[Card]
    melds: list[Meld] = field(default_factory=list)
    discards: list[Card] = field(default_factory=list)
    pending_action_card: Card | None = None
    opponent_pending_card: Card | None = None
    visible_cards: list[Card] = field(default_factory=list)
    buttons: list[ButtonState] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
