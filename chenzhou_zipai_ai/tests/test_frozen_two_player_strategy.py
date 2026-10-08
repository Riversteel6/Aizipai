"""Product-entry regression tests for the frozen two-player releases."""

from __future__ import annotations

from ai.frozen_two_player_strategy import FROZEN_CANDIDATE, choose_action
from ai.full_game_simulator import PublicView, SimMeld, _production_state_from_public_view
from engine.rules import rules_for_room


def test_no_wang_product_entry_keeps_confirmed_early_mixed_chi_pass() -> None:
    view = PublicView(
        seat=0,
        hand=(
            "壹", "四", "五", "捌", "四", "七", "拾", "玖", "六", "五",
            "九", "贰", "贰", "伍", "九", "二", "柒", "九", "十", "捌",
        ),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(
            ("一", 4), ("七", 3), ("三", 4), ("九", 1), ("二", 3),
            ("五", 2), ("伍", 3), ("八", 4), ("六", 3), ("十", 3),
            ("叁", 4), ("四", 2), ("壹", 3), ("拾", 3), ("捌", 2),
            ("柒", 3), ("玖", 3), ("肆", 3), ("贰", 2), ("陆", 4),
        ),
        stock_count=39,
        hand_sizes=(20, 20),
        pending_card="肆",
        pending_source_seat=1,
    )
    state = _production_state_from_public_view(
        view,
        legal_actions=[{"type": "CHI"}, {"type": "PASS"}],
        pending_card="肆",
        chi_options=None,
    )
    rules = rules_for_room(wildcard_enabled=False, players=2)

    decision = choose_action(state, rules=rules)

    assert decision.selected_action == "PASS"
    assert decision.policy_version == "2p-no-wang-20260809-rc3"
    product = decision.context_snapshot["product_strategy"]
    assert product["candidate"] == FROZEN_CANDIDATE
    assert product["candidate_applied"] is True


def test_no_wang_product_entry_keeps_confirmed_expanded_wait_peng() -> None:
    own_melds = (
        SimMeld("mixed_same_rank_triplet", ("一", "壹", "壹")),
        SimMeld("special_123", ("一", "二", "三")),
        SimMeld("wei", ("四", "四", "四")),
    )
    view = PublicView(
        seat=0,
        hand=("陆", "陆", "伍", "五", "六", "九", "玖", "七", "七", "伍", "九"),
        own_melds=own_melds,
        all_melds=(
            own_melds,
            (
                SimMeld("ti", ("叁", "叁", "叁", "叁")),
                SimMeld("mixed_same_rank_triplet", ("八", "捌", "捌")),
                SimMeld("wei", ("拾", "拾", "拾")),
            ),
        ),
        discards=(("贰", "二", "肆", "玖"), ("肆", "玖", "玖", "一")),
        remaining_counts=(
            ("一", 1), ("七", 2), ("三", 3), ("九", 2), ("二", 2),
            ("五", 3), ("伍", 2), ("八", 3), ("六", 3), ("十", 4),
            ("四", 1), ("壹", 2), ("拾", 1), ("捌", 2), ("柒", 4),
            ("肆", 2), ("贰", 3), ("陆", 1),
        ),
        stock_count=31,
        hand_sizes=(11, 10),
        pending_card="陆",
        pending_source_seat=1,
    )
    state = _production_state_from_public_view(
        view,
        legal_actions=[{"type": "PENG"}, {"type": "PASS"}],
        pending_card="陆",
        chi_options=None,
    )

    decision = choose_action(
        state,
        rules=rules_for_room(wildcard_enabled=False, players=2),
    )

    assert decision.selected_action == "PENG"
    assert decision.selected_label == "陆"
    assert decision.policy_version == "2p-no-wang-20260809-rc3"


def test_no_wang_product_entry_keeps_confirmed_high_xi_123_chi() -> None:
    view = PublicView(
        seat=1,
        hand=(
            "一", "九", "八", "九", "捌", "三", "一", "三", "柒", "二",
            "陆", "八", "贰", "八", "肆", "贰", "肆", "玖", "拾", "叁",
        ),
        own_melds=(),
        all_melds=((), ()),
        discards=(("九", "捌"), ("六", "十", "六")),
        remaining_counts=(
            ("一", 2), ("七", 4), ("三", 2), ("九", 1), ("二", 3),
            ("五", 4), ("伍", 4), ("八", 1), ("六", 2), ("十", 3),
            ("叁", 3), ("四", 4), ("壹", 3), ("拾", 3), ("捌", 2),
            ("柒", 3), ("玖", 3), ("肆", 2), ("贰", 2), ("陆", 3),
        ),
        stock_count=34,
        hand_sizes=(20, 20),
        pending_card="壹",
        pending_source_seat=0,
    )
    state = _production_state_from_public_view(
        view,
        legal_actions=[{"type": "CHI"}, {"type": "PASS"}],
        pending_card="壹",
        chi_options=[
            {"option_id": "mixed", "labels": ["一", "一", "壹"], "confidence": 1.0},
            {"option_id": "big_123", "labels": ["壹", "贰", "叁"], "confidence": 1.0},
        ],
    )

    decision = choose_action(
        state,
        rules=rules_for_room(wildcard_enabled=False, players=2),
    )

    assert decision.selected_action == "CHI"
    assert decision.selected_option_id == "big_123"
    assert decision.policy_version == "2p-no-wang-20260809-rc3"


def test_wang_product_entry_is_stamped_with_its_own_frozen_release() -> None:
    rules = rules_for_room(wildcard_enabled=True, players=2)
    state = {
        "hand": ["一"],
        "legal_actions": [{"type": "DISCARD"}],
        "remaining_deck_count": 82,
        "memory": {
            "my_discards": [],
            "opponent_discards": [],
            "opponent_meld_groups": [],
        },
    }

    decision = choose_action(state, rules=rules)

    assert decision.selected_action == "DISCARD"
    assert decision.selected_label == "一"
    assert decision.policy_version == "v9"
    assert decision.context_snapshot["product_strategy"]["candidate_applied"] is True


def test_three_player_room_does_not_use_two_player_frozen_release() -> None:
    rules = rules_for_room(wildcard_enabled=False, players=3)
    decision = choose_action(
        {"hand": ["一"], "legal_actions": [{"type": "DISCARD"}]},
        rules=rules,
    )

    assert decision.policy_version not in {
        "2p-no-wang-20260808-rc2",
        "2p-wang-20260808-rc2",
    }
    assert "product_strategy" not in decision.context_snapshot
