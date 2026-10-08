from types import SimpleNamespace

from tools.audit_policy_variant_roots import _should_bypass_variant


def test_no_wang_room_does_not_bypass_general_policy_candidate():
    policy = SimpleNamespace()
    view = SimpleNamespace(hand=("一", "二"))

    assert not _should_bypass_variant(policy, view)


def test_wang_only_candidate_bypasses_hand_without_wang():
    policy = SimpleNamespace(require_wildcard_in_hand=True)

    assert _should_bypass_variant(policy, SimpleNamespace(hand=("一", "二")))
    assert not _should_bypass_variant(policy, SimpleNamespace(hand=("一", "王")))
