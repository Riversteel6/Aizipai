"""Independent rules oracle tests.

The oracle must remain structurally separate from production rule helpers.
"""

from __future__ import annotations

import ast
import inspect

from audit import independent_rules
from audit.independent_rules import (
    apply_automatic_pao_oracle,
    apply_draw_auto_meld_oracle,
    apply_response_action_oracle,
    automatic_pao_seat_oracle,
    enumerate_chi_plans_oracle,
    evaluate_hu_oracle,
    legal_discard_labels_oracle,
    legal_response_actions_oracle,
    room_shape_oracle,
)
from engine.rules import rules_for_room


COMPLETE_HAND = [
    "贰", "柒", "拾",
    "壹", "贰", "叁",
    "九", "九", "九",
    "一", "二", "三",
    "四", "五", "六",
    "七", "八", "九",
    "肆", "伍", "陆",
]


def test_oracle_does_not_import_production_engine_or_ai_modules():
    tree = ast.parse(inspect.getsource(independent_rules))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert "engine" not in imported_roots
    assert "ai" not in imported_roots


def test_oracle_matches_confirmed_complete_hand_and_concealed_xi():
    result = evaluate_hu_oracle(
        COMPLETE_HAND,
        [],
        rules_for_room(wildcard_enabled=False),
    )

    assert result.can_hu
    assert result.partition_complete
    assert result.group_count == 7
    assert result.total_xi == 18


def test_oracle_independently_maps_two_wildcards_to_best_concealed_group():
    rules = rules_for_room(wildcard_enabled=True)
    rules["rules"]["required_meld_groups"] = 1
    rules["rules"]["min_xi"] = 3

    result = evaluate_hu_oracle(["四", "王", "王"], [], rules)

    assert result.can_hu
    assert result.total_xi == 3
    assert result.hand_groups[0].resolved_labels == ("四", "四", "四")


def test_oracle_accepts_wild_pair_after_multiple_four_card_melds():
    existing = [
        {"type": "ti", "labels": ["三"] * 4},
        {"type": "pao", "labels": ["四"] * 4},
        {"type": "normal_sequence", "labels": ["一", "二", "三"]},
        {"type": "normal_sequence", "labels": ["七", "八", "九"]},
        {"type": "normal_sequence", "labels": ["壹", "贰", "叁"]},
    ]

    result = evaluate_hu_oracle(
        ["伍", "伍", "伍", "王", "王"],
        existing,
        rules_for_room(wildcard_enabled=True),
    )

    assert result.can_hu
    assert result.partition_complete
    assert result.total_xi == 21
    assert result.group_count == 7
    pair = next(group for group in result.hand_groups if group.kind == "pair")
    assert pair.source_labels == ("王", "王")
    assert pair.wildcards_used == 2


def test_oracle_accepts_existing_mixed_same_rank_type_alias():
    existing = [
        {
            "type": "mixed_same_rank_triplet",
            "labels": ["三", "三", "叁"],
        },
        {
            "type": "normal_sequence",
            "labels": ["叁", "肆", "伍"],
        },
    ]
    hand = [
        "八", "二", "王", "五", "捌", "柒", "十", "捌",
        "四", "六", "拾", "拾", "贰", "七", "拾",
    ]

    result = evaluate_hu_oracle(
        hand,
        existing,
        rules_for_room(wildcard_enabled=True, players=2),
    )

    assert result.can_hu
    assert result.partition_complete
    assert result.total_xi == 15


def test_oracle_chi_requires_all_same_label_compare_cards_to_be_consumed():
    plans = enumerate_chi_plans_oracle(["五", "四", "六", "伍", "伍"], "五")

    assert plans
    assert any(plan.compare_groups for plan in plans)
    assert all(
        "五" not in _remaining(["五", "四", "六", "伍", "伍"], plan.consumed_from_hand)
        for plan in plans
    )


def test_oracle_rejects_chi_that_consumes_locked_triplet():
    assert not enumerate_chi_plans_oracle(["二", "二", "二", "一"], "三")


def test_oracle_wang_stays_in_hand_without_blocking_normal_chi():
    plans = enumerate_chi_plans_oracle(["伍", "伍", "王", "三"], "五")

    assert plans
    assert all("王" not in plan.consumed_from_hand for plan in plans)


def test_oracle_heads_up_room_shape_is_independent_and_uses_43_card_stock():
    shape = room_shape_oracle(players=2, wildcard_enabled=True)

    assert shape.deck_size == 84
    assert shape.initial_dealt == 41
    assert shape.initial_stock == 43


def test_oracle_discard_labels_exclude_wang_and_locked_triplets():
    assert legal_discard_labels_oracle(
        ["八", "八", "八", "捌", "王", "二", "二"]
    ) == ("二", "捌")


def test_oracle_response_actions_cover_hu_peng_chi_and_pass_together():
    hand = [
        "七", "八", "九",
        "四", "肆",
        "陆", "柒", "捌",
        "贰", "柒", "拾",
        "肆", "伍", "陆",
        "六", "六", "六",
        "二", "三", "四",
    ]

    response = legal_response_actions_oracle(
        hand,
        [],
        "肆",
        seat=0,
        source_seat=2,
        players=3,
        rules=rules_for_room(wildcard_enabled=False, players=3),
    )

    assert response.automatic_action is None
    assert {action.type for action in response.actions} == {
        "PASS",
        "HU",
        "PENG",
        "CHI",
    }


def test_oracle_automatic_pao_masks_normal_response_actions():
    melds = {
        0: [{"type": "peng", "labels": ["五", "五", "五"]}],
        1: [],
        2: [],
    }

    pao_seat = automatic_pao_seat_oracle(
        melds,
        "五",
        players=3,
        source_seat=2,
    )
    response = legal_response_actions_oracle(
        ["四", "六", "九"],
        melds[0],
        "五",
        seat=0,
        source_seat=2,
        players=3,
        rules=rules_for_room(wildcard_enabled=False, players=3),
        automatic_pao_seat=pao_seat,
    )

    assert pao_seat == 0
    assert response.automatic_action == "PAO"
    assert response.automatic_seat == 0
    assert response.actions == ()


def test_oracle_applies_peng_chi_and_automatic_pao_transitions():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    hand = ["五", "五", "四", "六", "伍", "八"]
    peng_response = legal_response_actions_oracle(
        hand,
        [],
        "五",
        seat=0,
        source_seat=2,
        players=3,
        rules=rules,
    )
    peng = next(
        action
        for action in peng_response.actions
        if action.type == "PENG"
    )
    peng_after = apply_response_action_oracle(
        hand,
        [],
        "五",
        peng,
    )
    chi = next(
        action
        for action in peng_response.actions
        if action.type == "CHI"
    )
    chi_after = apply_response_action_oracle(
        hand,
        [],
        "五",
        chi,
    )
    pao_after = apply_automatic_pao_oracle(
        ["四", "六"],
        [{"type": "peng", "labels": ["五", "五", "五"]}],
        "五",
    )

    assert peng_after.hand == ("四", "伍", "六", "八")
    assert peng_after.melds[-1].kind == "peng"
    assert len(chi_after.hand) < len(hand)
    assert all(meld.kind != "peng" for meld in chi_after.melds)
    assert pao_after.melds[-1].kind == "pao"
    assert pao_after.melds[-1].labels == ("五",) * 4


def test_oracle_draw_auto_meld_handles_wei_ti_pao_and_wang():
    wei = apply_draw_auto_meld_oracle(["五", "五", "九"], [], "五")
    ti = apply_draw_auto_meld_oracle(
        ["五", "五", "五", "九"],
        [],
        "五",
        quad_events=1,
    )
    pao = apply_draw_auto_meld_oracle(
        ["九"],
        [{"type": "peng", "labels": ["五", "五", "五"]}],
        "五",
    )
    wang = apply_draw_auto_meld_oracle(["王", "王", "九"], [], "王")

    assert wei.automatic_action == "WEI"
    assert wei.melds[-1].kind == "wei"
    assert ti.automatic_action == "TI"
    assert ti.quad_events == 2
    assert ti.skip_discard
    assert pao.automatic_action == "PAO"
    assert pao.melds[-1].kind == "pao"
    assert wang.automatic_action is None
    assert wang.hand.count("王") == 3


def _remaining(hand: list[str], consumed: tuple[str, ...]) -> list[str]:
    result = list(hand)
    for label in consumed:
        result.remove(label)
    return result
