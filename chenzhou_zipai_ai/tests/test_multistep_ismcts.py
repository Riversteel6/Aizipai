from audit.independent_opponent import IndependentFastRolloutPolicy
from ai.full_game_simulator import PublicView
from ai.multistep_ismcts import BoundedMultiStepInformationSetPolicy
from engine.rules import rules_for_room


def test_multistep_continuation_decrements_self_search_depth():
    policy = BoundedMultiStepInformationSetPolicy(
        decision_depth=2,
        opponent_policy_factories=(IndependentFastRolloutPolicy,),
    )

    child = policy._next_self_policy()
    grandchild = child._next_self_policy()

    assert child.decision_depth == 1
    assert grandchild.decision_depth == 0
    assert child.name == "bounded_multistep_ismcts_v1"
    assert child.opponent_policy_factories == (IndependentFastRolloutPolicy,)


def test_depth_zero_uses_native_self_policy_without_recursive_search():
    policy = BoundedMultiStepInformationSetPolicy(
        decision_depth=0,
        opponent_policy_factories=(IndependentFastRolloutPolicy,),
    )
    view = PublicView(
        seat=0,
        hand=("一", "二", "三", "四"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(),
        stock_count=0,
        hand_sizes=(4, 0),
    )

    selected = policy.choose_discard(
        view,
        rules_for_room(wildcard_enabled=False, players=2),
    )

    assert selected in view.hand
    assert policy.last_search is None
