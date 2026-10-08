"""Exact chi/compare rule tests reconstructed from the APK rule helper."""

from engine.chi_rules import enumerate_chi_plans


def test_mixed_same_rank_chi_is_multiset_based():
    plans = enumerate_chi_plans(["五", "伍"], "五")

    assert plans
    assert set(plans[0].initial_group) == {"五", "伍"}
    assert sorted(plans[0].consumed_from_hand) == sorted(["五", "伍"])


def test_chi_plan_includes_mandatory_compare_groups():
    plans = enumerate_chi_plans(["五", "四", "六", "伍", "伍"], "五")
    compare_plans = [plan for plan in plans if plan.compare_groups]

    assert compare_plans
    assert any(len(plan.consumed_from_hand) == 5 for plan in compare_plans)
    assert all("五" not in _remaining(["五", "四", "六", "伍", "伍"], plan.consumed_from_hand) for plan in plans)


def test_chi_is_rejected_when_leftover_same_card_cannot_complete_compare():
    plans = enumerate_chi_plans(["五", "四", "六", "伍"], "五")

    assert all(not plan.compare_groups for plan in plans)
    assert all(plan.initial_group.count("五") == 2 for plan in plans)


def test_three_matching_hand_cards_are_locked_against_chi():
    assert not enumerate_chi_plans(["二", "二", "二", "一"], "三")


def _remaining(hand: list[str], consumed: tuple[str, ...]) -> list[str]:
    result = list(hand)
    for label in consumed:
        result.remove(label)
    return result
