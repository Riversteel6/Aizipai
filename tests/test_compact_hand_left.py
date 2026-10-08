from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compact_hand_left import build_compact_plan
from vision.hand_recognizer import HandCard


def _card(name: str, x: int, y: int, *, clickable: bool = True) -> HandCard:
    return HandCard(name=name, confidence=0.9, x=x, y=y, w=90, h=130, template="fixture", clickable=clickable)


def test_build_compact_plan_moves_only_right_side_clickable_cards():
    cards = [
        _card("一", 520, 700),
        _card("二", 1600, 700),
        _card("三", 1720, 830),
        _card("四", 1810, 700, clickable=False),
    ]

    plan = build_compact_plan(cards, right_of=1500, target_x=620)

    assert [step["label"] for step in plan] == ["三", "二"]
    assert [(step["end_x"], step["end_y"]) for step in plan] == [(565, 765), (565, 765)]


def test_build_compact_plan_noops_when_hand_already_left_compact():
    cards = [_card("一", 520, 700), _card("二", 700, 700)]

    assert build_compact_plan(cards, right_of=1500) == []
