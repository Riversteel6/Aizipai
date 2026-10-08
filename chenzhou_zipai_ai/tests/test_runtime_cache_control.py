from ai.features import _quick_potential_cached
from ai.pro_brain import choose_action
from ai.runtime_cache_control import clear_round_strategy_caches
from engine.hu_checker import _solve_grouping_counts


def test_round_cache_cleanup_preserves_results_and_clears_entries():
    hand = ("一", "二", "三", "四", "五", "六")
    expected = _quick_potential_cached(hand)
    assert _quick_potential_cached.cache_info().currsize > 0

    outcome = clear_round_strategy_caches()

    assert outcome["cleared_cache_entries"] > 0
    assert _quick_potential_cached.cache_info().currsize == 0
    assert _solve_grouping_counts.cache_info().currsize == 0
    assert _quick_potential_cached(hand) == expected


def test_round_cache_cleanup_does_not_change_strategy_decision():
    state = {
        "hand": ["一", "二", "三", "四", "六", "七", "九", "贰", "柒", "拾"],
        "legal_actions": [{"type": "DISCARD"}],
    }
    before = choose_action(state)

    clear_round_strategy_caches()
    after = choose_action(state)

    assert after.selected_action == before.selected_action
    assert after.selected_label == before.selected_label
    assert after.ev == before.ev
    assert [
        (item.type, item.action.label, item.allowed, item.ev)
        for item in after.action_evals
    ] == [
        (item.type, item.action.label, item.allowed, item.ev)
        for item in before.action_evals
    ]
