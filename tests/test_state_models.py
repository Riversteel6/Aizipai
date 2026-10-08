from aizipai.state.models import (
    Action,
    ActionType,
    ButtonState,
    Card,
    ClickPoint,
    Color,
    GameState,
    Meld,
    Phase,
    Suit,
)


def test_state_models_support_to_dict_contract():
    card = Card(rank=2, suit=Suit.SMALL, color=Color.RED)
    action = Action(
        type=ActionType.DISCARD,
        card=card,
        click_plan=[ClickPoint(x=10, y=20, delay_ms=30)],
        reason="card_in_hand",
    )
    state = GameState(
        round_id="round_1",
        seat=0,
        turn=3,
        phase=Phase.AWAIT_ACTION,
        hand=[card],
        melds=[Meld(type=ActionType.PENG, cards=[card, card, card], source_seat=1)],
        buttons=[ButtonState(action=ActionType.HU, bbox=(1, 2, 3, 4))],
        metadata={"opponent_priority_pending": False},
    )

    assert card.to_dict() == {
        "rank": 2,
        "suit": "small",
        "color": "red",
        "is_wildcard": False,
    }
    assert action.to_dict()["click_plan"][0]["delay_ms"] == 30
    payload = state.to_dict()
    assert payload["phase"] == "await_action"
    assert payload["hand"][0]["rank"] == 2
    assert payload["melds"][0]["type"] == "peng"
    assert payload["buttons"][0]["bbox"] == [1, 2, 3, 4]
