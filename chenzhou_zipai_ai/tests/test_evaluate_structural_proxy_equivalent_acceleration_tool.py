from ai.full_game_simulator import PublicView
from ai.opponent_proxy import FastRedBlackSearchProxyPolicy
from engine.rules import rules_for_room
from tools.evaluate_structural_proxy_equivalent_acceleration import (
    legacy_unbatched_selected,
)


def test_count_acceleration_preserves_red_black_selection() -> None:
    view = PublicView(
        seat=0,
        hand=("一", "二", "四", "六", "九", "壹", "贰"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(("一", 3), ("二", 3), ("三", 4), ("四", 3)),
        stock_count=35,
        hand_sizes=(7, 7),
    )
    rules = rules_for_room(wildcard_enabled=False, players=2)
    policy = FastRedBlackSearchProxyPolicy()

    assert policy.choose_discard(view, rules) == legacy_unbatched_selected(
        "red_black_search",
        policy,
        view,
        rules,
    )
