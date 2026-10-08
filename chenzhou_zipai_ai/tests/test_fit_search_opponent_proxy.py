from ai.ismcts import public_view_to_dict
from audit.independent_opponent import (
    IndependentFastRolloutPolicy,
    IndependentPolicyWeights,
)
from ai.full_game_simulator import PublicView
from engine.rules import rules_for_room
from tools.fit_search_opponent_proxy import (
    candidate_configurations,
    evaluate_weights,
)


def test_proxy_grid_is_unique_and_evaluable() -> None:
    configurations = candidate_configurations()
    assert len(configurations) == 432
    assert len(set(configurations)) == len(configurations)

    view = PublicView(
        seat=0,
        hand=("一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), (), ()),
        discards=((), (), ()),
        remaining_counts=(
            ("一", 3),
            ("二", 3),
            ("三", 4),
            ("四", 3),
            ("五", 4),
            ("六", 3),
            ("七", 4),
            ("八", 4),
            ("九", 3),
            ("十", 4),
        ),
        stock_count=35,
        hand_sizes=(5, 5, 5),
    )
    rules = rules_for_room(wildcard_enabled=False, players=3)
    target = IndependentFastRolloutPolicy().choose_discard(view, rules)
    result = evaluate_weights(
        [
            {
                "state_before_hash": "state",
                "target_label": target,
                "public_state": public_view_to_dict(view),
                "players": 3,
                "wildcard_enabled": False,
            }
        ],
        weights=IndependentPolicyWeights(),
    )

    assert result["agreements"] == 1
    assert result["agreement_rate"] == 1.0
