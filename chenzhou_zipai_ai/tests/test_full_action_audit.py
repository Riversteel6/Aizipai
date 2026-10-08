"""Full-action audit helpers."""

from ai.full_action_audit import _legal_discard_labels


def test_full_action_legal_labels_exclude_wang_and_locked_triplets():
    hand = ("王", "八", "八", "八", "二", "二", "九")

    assert _legal_discard_labels(hand) == ("九", "二")
