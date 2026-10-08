from aizipai.rules.engine import RuleEngine, RuleScore
from aizipai.state.models import Action, ActionType, ButtonState, Card, Color, GameState, Phase, Suit


def test_legal_actions_include_discards_for_hand_cards():
    state = GameState(
        round_id="test",
        seat=0,
        turn=0,
        phase=Phase.AWAIT_ACTION,
        hand=[
            Card(rank=1, suit=Suit.SMALL, color=Color.RED),
            Card(rank=2, suit=Suit.SMALL, color=Color.BLACK),
        ],
    )

    actions = RuleEngine().legal_actions(state)

    assert [action.type for action in actions] == [ActionType.DISCARD, ActionType.DISCARD]


def test_chi_is_not_legal_while_opponent_priority_is_pending():
    state = GameState(
        round_id="test",
        seat=0,
        turn=0,
        phase=Phase.AWAIT_ACTION,
        hand=[],
        buttons=[
            ButtonState(action=ActionType.CHI),
            ButtonState(action=ActionType.PASS),
        ],
        metadata={"opponent_priority_pending": True},
    )

    actions = RuleEngine().legal_actions(state)

    assert [(action.type, action.reason) for action in actions] == [
        (ActionType.PASS, "wait_opponent_priority")
    ]


def test_chi_is_legal_after_opponent_priority_clears():
    state = GameState(
        round_id="test",
        seat=0,
        turn=0,
        phase=Phase.AWAIT_ACTION,
        hand=[],
        buttons=[
            ButtonState(action=ActionType.CHI),
            ButtonState(action=ActionType.PASS),
        ],
        metadata={"opponent_priority_pending": False},
    )

    actions = RuleEngine().legal_actions(state)

    assert [action.type for action in actions] == [ActionType.CHI, ActionType.PASS]


def test_auto_quad_pending_suppresses_manual_actions():
    state = GameState(
        round_id="test",
        seat=0,
        turn=0,
        phase=Phase.AWAIT_ACTION,
        hand=[
            Card(rank=2, suit=Suit.SMALL, color=Color.BLACK),
            Card(rank=2, suit=Suit.SMALL, color=Color.BLACK),
        ],
        buttons=[
            ButtonState(action=ActionType.PENG),
            ButtonState(action=ActionType.PASS),
        ],
        metadata={"auto_quad_pending": True},
    )

    actions = RuleEngine().legal_actions(state)

    assert [(action.type, action.reason) for action in actions] == [
        (ActionType.PASS, "wait_auto_quad")
    ]


def test_hu_is_prioritized_over_chi_peng_and_pass():
    state = GameState(
        round_id="test",
        seat=0,
        turn=0,
        phase=Phase.WAITING,
        hand=[],
        buttons=[
            ButtonState(action=ActionType.CHI),
            ButtonState(action=ActionType.PASS),
            ButtonState(action=ActionType.PENG),
            ButtonState(action=ActionType.HU),
        ],
    )

    actions = RuleEngine().legal_actions(state)

    assert [action.type for action in actions] == [
        ActionType.HU,
        ActionType.PENG,
        ActionType.CHI,
        ActionType.PASS,
    ]


def test_rule_engine_is_legal_ignores_explanatory_reason():
    card = Card(rank=3, suit=Suit.SMALL, color=Color.BLACK)
    state = GameState(
        round_id="test",
        seat=0,
        turn=0,
        phase=Phase.AWAIT_ACTION,
        hand=[card],
    )

    assert RuleEngine().is_legal(state, Action(type=ActionType.DISCARD, card=card))


def test_rule_engine_score_action_returns_rule_score():
    red = Card(rank=2, suit=Suit.SMALL, color=Color.RED)
    black = Card(rank=3, suit=Suit.SMALL, color=Color.BLACK)
    state = GameState(
        round_id="test",
        seat=0,
        turn=0,
        phase=Phase.AWAIT_ACTION,
        hand=[red, black],
    )

    result = RuleEngine().score_action(state, Action(type=ActionType.DISCARD, card=black))

    assert isinstance(result, RuleScore)
    assert result.legal is True
    assert result.score > RuleEngine().score_action(state, Action(type=ActionType.DISCARD, card=red)).score
    assert result.to_dict()["details"]["source"] == "HeuristicAgent"


def test_rule_engine_score_action_reports_illegal_actions():
    state = GameState(
        round_id="test",
        seat=0,
        turn=0,
        phase=Phase.WAITING,
        hand=[],
    )

    result = RuleEngine().score_action(state, Action(type=ActionType.PENG))

    assert result.legal is False
    assert result.reject_reason == "action_not_legal"
    assert result.to_dict()["legal"] is False


def test_rule_engine_is_terminal_uses_phase_and_metadata():
    base = {
        "round_id": "test",
        "seat": 0,
        "turn": 0,
        "hand": [],
    }

    assert RuleEngine().is_terminal(GameState(**base, phase=Phase.ROUND_OVER))
    assert RuleEngine().is_terminal(GameState(**base, phase=Phase.WAITING, metadata={"game_over": True}))
    assert not RuleEngine().is_terminal(GameState(**base, phase=Phase.WAITING))
