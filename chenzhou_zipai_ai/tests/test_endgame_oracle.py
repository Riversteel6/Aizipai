from __future__ import annotations

import ast
import inspect

from audit import endgame_oracle
from audit.endgame_oracle import IndependentEndgameOracle, OracleEndgameState
from engine.rules import rules_for_room


def _one_group_rules():
    rules = rules_for_room(wildcard_enabled=False, players=2)
    rules["rules"]["required_meld_groups"] = 1
    rules["rules"]["min_xi"] = 0
    return rules


def test_endgame_oracle_does_not_import_production_engine_or_ai_modules():
    tree = ast.parse(inspect.getsource(endgame_oracle))
    imported_roots = {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "engine" not in imported_roots
    assert "ai" not in imported_roots


def test_exact_draw_value_is_one_for_forced_self_draw_hu():
    state = OracleEndgameState(
        hands=(("一", "二"), ()),
        melds=((), ()),
        stock=("三",),
        current_seat=0,
        needs_draw=True,
    )

    result = IndependentEndgameOracle(
        _one_group_rules(),
        root_seat=0,
    ).solve(state)

    assert result.exact
    assert result.p_win == 1.0
    assert result.reason == "independent_exact_fully_observed_endgame"


def test_exact_draw_value_is_zero_when_opponent_has_forced_hu():
    state = OracleEndgameState(
        hands=((), ("一", "二")),
        melds=((), ()),
        stock=("三",),
        current_seat=1,
        needs_draw=True,
    )

    result = IndependentEndgameOracle(
        _one_group_rules(),
        root_seat=0,
    ).solve(state)

    assert result.exact
    assert result.p_win == 0.0


def test_large_information_set_is_refused_instead_of_called_exact():
    state = OracleEndgameState(
        hands=(("一", "二"), ()),
        melds=((), ()),
        stock=("三",) * 9,
        current_seat=0,
        needs_draw=True,
    )

    result = IndependentEndgameOracle(
        _one_group_rules(),
        root_seat=0,
        maximum_stock_cards=8,
    ).solve(state)

    assert not result.exact
    assert result.reason == "unsupported_stock_size:9>8"
