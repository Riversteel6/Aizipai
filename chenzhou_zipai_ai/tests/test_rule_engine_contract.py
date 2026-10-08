"""Rule engine contract tests."""

from __future__ import annotations

from engine.legal_actions import (
    LegalActionGenerator,
    RuleScore,
    is_legal,
    is_terminal,
    legal_actions,
    score_action,
)


def _discard_state():
    return {
        "hand_details": [
            {"card_id": "h1", "label": "一", "clickable": True},
            {"card_id": "h2", "label": "二", "clickable": True},
            {"card_id": "h3", "label": "三", "clickable": True},
        ],
        "legal_actions": [{"type": "DISCARD"}],
        "flow_state": "play",
    }


def test_rule_engine_public_legal_actions_returns_action_list():
    actions = legal_actions(_discard_state())

    assert actions
    assert all(hasattr(action, "to_dict") for action in actions)
    assert {"DISCARD"} <= {action.type for action in actions}


def test_rule_engine_is_legal_matches_requested_action_fields():
    state = _discard_state()

    assert is_legal(state, {"type": "discard", "card_id": "h1", "label": "一"})
    assert not is_legal(state, {"type": "discard", "card_id": "missing", "label": "一"})
    assert not is_legal(state, "PENG")


def test_rule_engine_score_action_uses_ev_scorer_and_rule_score():
    result = score_action(_discard_state(), {"type": "discard", "label": "一"})

    assert isinstance(result, RuleScore)
    assert result.legal is True
    assert result.action.type == "DISCARD"
    assert result.action_eval is not None
    assert result.details["scorer"] == "EVScorer"
    assert result.to_dict()["action_eval"]["debug_details"]["scorer"] == "EVScorer"


def test_rule_engine_score_action_reports_illegal_with_reject_reason():
    result = score_action(_discard_state(), "PENG")

    assert result.legal is False
    assert result.score == 0
    assert result.reject_reason == "action_not_legal"


def test_rule_engine_is_terminal_understands_flow_states():
    assert is_terminal({"flow_state": "final_score"})
    assert is_terminal({"terminal": True})
    assert not is_terminal({"flow_state": "settlement_ready"})
    assert not is_terminal(_discard_state())


def test_legal_action_generator_exposes_rule_contract_methods():
    generator = LegalActionGenerator()
    state = _discard_state()

    assert generator.legal_actions(state)
    assert generator.is_legal(state, "DISCARD")
    assert generator.score_action(state, "DISCARD").legal
    assert generator.is_terminal({"flow_state": "final_score"})
