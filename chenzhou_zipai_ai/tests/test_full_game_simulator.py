"""Full-game simulator invariants and room-profile tests."""

import random
from collections import Counter

import pytest

from ai.features import QUICK_POTENTIAL_LABELS
from ai.full_game_simulator import (
    _benchmark_assignment,
    _draw_shape_relevance,
    _draw_shape_relevance_values,
    _production_state_from_public_view,
    BaselinePolicy,
    FullGameSimulator,
    HuEvaluation,
    InformationSetSearchPolicy,
    ProfessionalBrainSimulationPolicy,
    PublicView,
    ChiPlan,
    SimMeld,
    SimPlayer,
    evaluate_hu,
)
from engine.deck import expanded_deck, full_deck_counts
from engine.rules import rules_for_room
from ai.simulation_trace import selected_action_is_legal


COMPLETE_HAND = [
    "贰", "柒", "拾",
    "壹", "贰", "叁",
    "九", "九", "九",
    "一", "二", "三",
    "四", "五", "六",
    "七", "八", "九",
    "肆", "伍", "陆",
]


def test_room_profile_keeps_9_xi_red_black_and_switches_wildcard():
    without_wild = rules_for_room(wildcard_enabled=False)
    with_wild = rules_for_room(wildcard_enabled=True)

    assert without_wild["rules"]["min_xi"] == 9
    assert without_wild["rules"]["red_black_mode"] == "red_black_mingtang"
    assert sum(full_deck_counts(rules=without_wild).values()) == 80
    assert sum(full_deck_counts(rules=with_wild).values()) == 84
    assert not any(
        with_wild["wildcard"][f"can_form_{action}"]
        for action in ("chi", "peng", "wei", "pao", "ti")
    )


def test_heads_up_room_uses_full_84_card_deck_and_43_card_stock():
    rules = rules_for_room(wildcard_enabled=True, players=2)
    deck_size = sum(full_deck_counts(rules=rules).values())

    assert rules["game"]["players"] == 2
    assert rules["room_profile"]["players"] == 2
    assert deck_size == 84
    assert deck_size - (21 + 20) == 43


@pytest.mark.parametrize(
    ("mode", "players", "wildcard_enabled", "deck_size", "stock"),
    [
        ("1v1-no-wang", 2, False, 80, 39),
        ("1v1-wang", 2, True, 84, 43),
        ("1v1v1-no-wang", 3, False, 80, 19),
        ("1v1v1-wang", 3, True, 84, 23),
    ],
)
def test_room_mode_presets_resolve_all_four_startup_choices(
    mode,
    players,
    wildcard_enabled,
    deck_size,
    stock,
):
    rules = rules_for_room(room_mode=mode)

    assert rules["room_profile"]["mode"] == mode
    assert rules["game"]["players"] == players
    assert rules["wildcard"]["enabled"] is wildcard_enabled
    assert sum(full_deck_counts(rules=rules).values()) == deck_size
    assert deck_size - (players * 20 + 1) == stock


def test_room_mode_rejects_conflicting_manual_switches():
    with pytest.raises(ValueError, match="conflicts with players"):
        rules_for_room(room_mode="1v1-wang", players=3)
    with pytest.raises(ValueError, match="conflicts with wildcard_enabled"):
        rules_for_room(room_mode="1v1-wang", wildcard_enabled=False)


def test_full_hu_requires_seven_groups_and_counts_concealed_triplet_as_wei():
    hu = evaluate_hu(COMPLETE_HAND, [], rules_for_room(wildcard_enabled=False))

    assert hu.can_hu
    assert len(hu.groups) == 7
    assert hu.total_xi == 18


def test_existing_peng_uses_exposed_xi_not_concealed_wei_xi():
    hand = COMPLETE_HAND[:-3]
    hu = evaluate_hu(
        hand,
        [SimMeld("peng", ("陆", "陆", "陆"))],
        rules_for_room(wildcard_enabled=False),
    )

    assert hu.can_hu
    assert hu.total_xi == 21


def test_wang_hu_partition_maximizes_xi_across_legal_resolutions():
    rules = rules_for_room(wildcard_enabled=True, players=3)
    hand = [
        "王", "九", "九", "王",
        "八", "三", "九", "八",
        "王", "三", "王", "三",
    ]
    melds = [
        SimMeld("special_123", ("壹", "贰", "叁")),
        SimMeld("mixed_same_rank", ("四", "肆", "肆")),
        SimMeld("mixed_same_rank", ("七", "七", "柒")),
    ]

    hu = evaluate_hu(hand, melds, rules)

    assert hu.can_hu
    assert hu.total_xi == 21
    assert hu.score == 5.0


def test_full_game_keeps_card_conservation_and_legal_melds():
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=False,
    )

    result = simulator.play(20260722)

    assert not result.violations
    assert result.reason in {"self_draw", "discard_hu", "stock_exhausted", "pao_hu"}


def test_heads_up_full_game_keeps_card_conservation_and_legal_melds():
    rules = rules_for_room(wildcard_enabled=True, players=2)
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=True,
        rules=rules,
    )

    result = simulator.play(20260726)

    assert not result.violations
    assert result.reason in {"self_draw", "discard_hu", "stock_exhausted", "pao_hu"}


def test_play_from_state_matches_normal_play_from_same_deal():
    seed = 20260724
    rules = rules_for_room(wildcard_enabled=False)
    direct = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=False,
        rules=rules,
    ).play(seed)

    deck = expanded_deck(full_deck_counts(rules=rules))
    random.Random(seed).shuffle(deck)
    players = [SimPlayer(seat=index) for index in range(3)]
    for _ in range(20):
        for player in players:
            player.hand.append(deck.pop())
    players[0].hand.append(deck.pop())
    resumed_simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=False,
        rules=rules,
    )
    resumed_simulator._lay_initial_quads(players, Counter())
    resumed = resumed_simulator.play_from_state(
        seed=seed,
        players=players,
        stock=deck,
        current=0,
    )

    assert not resumed.violations
    assert (resumed.winner, resumed.reason, resumed.turns, resumed.score) == (
        direct.winner,
        direct.reason,
        direct.turns,
        direct.score,
    )


def test_recorded_decision_trace_is_legal_and_deterministic():
    def run_once():
        simulator = FullGameSimulator(
            [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
            wildcard_enabled=False,
            record_decisions=True,
        )
        result = simulator.play(20260728)
        return result, simulator.decision_trace()

    first_result, first_trace = run_once()
    second_result, second_trace = run_once()

    assert not first_result.violations
    assert first_trace
    assert first_trace == second_trace
    assert first_result.to_dict() == second_result.to_dict()
    assert all(selected_action_is_legal(entry) for entry in first_trace)
    assert all(entry["state_before_hash"] for entry in first_trace)
    assert all(entry["state_after_hash"] for entry in first_trace)
    assert all(entry["public_state"]["hand"] for entry in first_trace)


def test_recorded_response_trace_contains_pass_and_every_chi_plan():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    players, stock = _state_with_pending_card(
        rules,
        hands=[["二", "十", "四"], [], []],
        pending="七",
    )
    simulator = FullGameSimulator(
        [_AlwaysChiPolicy(), _NoClaimPolicy(), _NoClaimPolicy()],
        wildcard_enabled=False,
        rules=rules,
        record_decisions=True,
    )

    result = simulator.play_from_pending_discard(
        seed=20260730,
        players=players,
        stock=stock,
        discarder=2,
        pending_card="七",
        max_turns=0,
    )
    trace = simulator.decision_trace()
    chi_entry = next(entry for entry in trace if entry["phase"] == "response_root")

    assert not result.violations
    assert selected_action_is_legal(chi_entry)
    assert chi_entry["selected_key"].startswith("CHI:")
    assert {action["type"] for action in chi_entry["legal_actions"]} == {"PASS", "CHI"}
    assert sum(
        action["type"] == "CHI"
        for action in chi_entry["legal_actions"]
    ) >= 1


def test_recorded_response_root_unifies_peng_chi_and_pass():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    players, stock = _state_with_pending_card(
        rules,
        hands=[["五", "五", "四", "六", "伍", "八"], [], []],
        pending="五",
    )
    simulator = FullGameSimulator(
        [_AlwaysChiPolicy(), _NoClaimPolicy(), _NoClaimPolicy()],
        wildcard_enabled=False,
        rules=rules,
        record_decisions=True,
    )

    result = simulator.play_from_pending_discard(
        seed=20260731,
        players=players,
        stock=stock,
        discarder=2,
        pending_card="五",
        max_turns=0,
    )
    response = next(
        entry for entry in simulator.decision_trace()
        if entry["phase"] == "response_root"
    )

    assert not result.violations
    assert selected_action_is_legal(response)
    assert response["selected_key"].startswith("CHI:")
    action_types = [action["type"] for action in response["legal_actions"]]
    assert action_types.count("PASS") == 1
    assert action_types.count("PENG") == 1
    assert action_types.count("CHI") >= 1
    assert len({action["key"] for action in response["legal_actions"]}) == len(
        response["legal_actions"]
    )


def test_recorded_response_root_unifies_hu_peng_chi_and_pass():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    waiting_hand = [
        "七", "八", "九",
        "四", "肆",
        "陆", "柒", "捌",
        "贰", "柒", "拾",
        "肆", "伍", "陆",
        "六", "六", "六",
        "二", "三", "四",
    ]
    players, stock = _state_with_pending_card(
        rules,
        hands=[waiting_hand, [], []],
        pending="肆",
    )
    simulator = FullGameSimulator(
        [_PassHuAlwaysChiPolicy(), _NoClaimPolicy(), _NoClaimPolicy()],
        wildcard_enabled=False,
        rules=rules,
        record_decisions=True,
    )

    result = simulator.play_from_pending_discard(
        seed=20260801,
        players=players,
        stock=stock,
        discarder=2,
        pending_card="肆",
        max_turns=0,
    )
    response = next(
        entry for entry in simulator.decision_trace()
        if entry["phase"] == "response_root"
    )

    assert not result.violations
    assert selected_action_is_legal(response)
    assert response["selected_key"].startswith("CHI:")
    assert {
        action["type"]
        for action in response["legal_actions"]
    } == {"PASS", "HU", "PENG", "CHI"}
    assert response["metadata"]["resolution"] == "chi"
    assert response["metadata"]["resolution_status"] == "executed"


def test_declined_hu_is_not_immediately_forced_again():
    rules = rules_for_room(wildcard_enabled=False, players=2)
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(COMPLETE_HAND)
    players = [
        SimPlayer(seat=0, hand=list(COMPLETE_HAND)),
        SimPlayer(seat=1),
    ]
    simulator = FullGameSimulator(
        [_PassHuPolicy(), _NoClaimPolicy()],
        wildcard_enabled=False,
        rules=rules,
        record_decisions=True,
    )

    result = simulator.play_from_state(
        seed=20260802,
        players=players,
        stock=expanded_deck(+remaining),
        current=0,
        needs_draw=False,
        max_turns=1,
    )
    hu_entries = [
        entry
        for entry in simulator.decision_trace()
        if entry["phase"] == "post_action_hu"
    ]

    assert result.winner is None
    assert result.reason == "max_turns"
    assert not result.violations
    assert len(hu_entries) == 1
    assert hu_entries[0]["selected_key"] == "PASS"
    assert selected_action_is_legal(hu_entries[0])


def test_pending_response_other_hu_overrides_root_non_hu_candidate():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    waiting_hand = list(COMPLETE_HAND)
    waiting_hand.remove("陆")
    players, stock = _state_with_pending_card(
        rules,
        hands=[["拾", "拾"], waiting_hand, []],
        pending="陆",
    )
    simulator = FullGameSimulator(
        [_NoClaimPolicy(), _NoClaimPolicy(), _NoClaimPolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_pending_discard(
        seed=20260726,
        players=players,
        stock=stock,
        discarder=2,
        pending_card="陆",
        blocked_auto_claim_seats=frozenset((0,)),
        max_turns=0,
    )

    assert result.winner == 1
    assert result.reason == "discard_hu"
    assert result.action_counts["discard_hu"] == 1
    assert result.action_counts["discard_hu:seat_1"] == 1
    assert not result.violations


def test_pending_response_other_peng_overrides_root_chi():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    players, stock = _state_with_pending_card(
        rules,
        hands=[["二", "十", "九"], ["七", "七", "四"], []],
        pending="七",
    )
    simulator = FullGameSimulator(
        [_AlwaysChiPolicy(), _AlwaysPengPolicy(), _NoClaimPolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_pending_discard(
        seed=20260727,
        players=players,
        stock=stock,
        discarder=2,
        pending_card="七",
        blocked_auto_claim_seats=frozenset((0,)),
        max_turns=0,
    )

    assert result.action_counts["peng"] == 1
    assert result.action_counts["peng:seat_1"] == 1
    assert result.action_counts.get("chi", 0) == 0
    assert players[1].melds[-1] == SimMeld("peng", ("七", "七", "七"))
    assert not result.violations


def test_pending_response_root_pass_still_allows_next_seat_chi():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    players, stock = _state_with_pending_card(
        rules,
        hands=[["一", "四"], [], ["二", "十", "九"]],
        pending="七",
    )
    simulator = FullGameSimulator(
        [_NoClaimPolicy(), _NoClaimPolicy(), _AlwaysChiPolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_pending_discard(
        seed=20260728,
        players=players,
        stock=stock,
        discarder=1,
        pending_card="七",
        blocked_auto_claim_seats=frozenset((0,)),
        max_turns=0,
    )

    assert result.action_counts["chi"] == 1
    assert result.action_counts["chi:seat_2"] == 1
    assert players[2].melds[-1].cards == ("二", "七", "十")
    assert not result.violations


def test_response_policies_receive_pending_card_and_source_seat():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    peng_policy = _RecordingNoClaimPolicy()
    players, stock = _state_with_pending_card(
        rules,
        hands=[["二", "二", "四"], [], []],
        pending="二",
    )
    simulator = FullGameSimulator(
        [peng_policy, _NoClaimPolicy(), _NoClaimPolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_pending_discard(
        seed=20260729,
        players=players,
        stock=stock,
        discarder=2,
        pending_card="二",
        max_turns=0,
    )

    assert not result.violations
    assert peng_policy.peng_view is not None
    assert peng_policy.peng_view.pending_card == "二"
    assert peng_policy.peng_view.pending_source_seat == 2
    assert sum(dict(peng_policy.peng_view.remaining_counts).values()) == (
        peng_policy.peng_view.stock_count
        + sum(
            size
            for seat, size in enumerate(peng_policy.peng_view.hand_sizes)
            if seat != peng_policy.peng_view.seat
        )
    )

    chi_policy = _RecordingNoClaimPolicy()
    players, stock = _state_with_pending_card(
        rules,
        hands=[["二", "十", "四"], [], []],
        pending="七",
    )
    simulator = FullGameSimulator(
        [chi_policy, _NoClaimPolicy(), _NoClaimPolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_pending_discard(
        seed=20260730,
        players=players,
        stock=stock,
        discarder=2,
        pending_card="七",
        max_turns=0,
    )

    assert not result.violations
    assert chi_policy.chi_view is not None
    assert chi_policy.chi_view.pending_card == "七"
    assert chi_policy.chi_view.pending_source_seat == 2
    assert sum(dict(chi_policy.chi_view.remaining_counts).values()) == (
        chi_policy.chi_view.stock_count
        + sum(
            size
            for seat, size in enumerate(chi_policy.chi_view.hand_sizes)
            if seat != chi_policy.chi_view.seat
        )
    )


def test_pending_response_rollout_honors_cooperative_deadline():
    rules = rules_for_room(wildcard_enabled=False, players=2)
    players, stock = _state_with_pending_card(
        rules,
        hands=[["一", "四"], []],
        pending="九",
    )
    simulator = FullGameSimulator(
        [_NoClaimPolicy(), _NoClaimPolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_pending_discard(
        seed=20260731,
        players=players,
        stock=stock,
        discarder=1,
        pending_card="九",
        deadline=0.0,
    )

    assert result.reason == "time_budget"
    assert not result.violations


def test_initial_ti_does_not_draw_a_replacement_card():
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=False,
    )
    players = [SimPlayer(0, ["一"] * 4), SimPlayer(1), SimPlayer(2)]
    actions: Counter[str] = Counter()

    simulator._lay_initial_quads(players, actions)

    assert players[0].hand == []
    assert players[0].melds == [SimMeld("ti", ("一",) * 4)]
    assert players[0].quad_events == 1
    assert actions == {"ti": 1}


def test_second_ti_skips_discard_instead_of_drawing_replacement():
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=False,
    )
    player = SimPlayer(0, ["二"] * 4, [SimMeld("ti", ("一",) * 4)], quad_events=1)
    actions: Counter[str] = Counter()

    resolution = simulator._apply_draw_auto_meld(player, "二", actions)

    assert resolution == "skip_discard"
    assert player.hand == []
    assert player.quad_events == 2
    assert actions == {"ti": 1}


def test_quad_compensation_wild_pair_is_a_verified_post_action_hu():
    rules = rules_for_room(wildcard_enabled=True, players=3)
    player = SimPlayer(
        0,
        ["王", "王"],
        [
            SimMeld("mixed_same_rank_triplet", ("三", "三", "叁")),
            SimMeld("pao", ("陆",) * 4),
            SimMeld("mixed_same_rank_triplet", ("四", "四", "肆")),
            SimMeld("normal_sequence", ("八", "九", "十")),
            SimMeld("normal_sequence", ("柒", "捌", "玖")),
            SimMeld("normal_sequence", ("二", "三", "四")),
        ],
        quad_events=1,
    )
    players = [player, SimPlayer(1), SimPlayer(2)]
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(player.hand)
    for meld in player.melds:
        remaining.subtract(meld.cards)
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=True,
        rules=rules,
    )

    result = simulator.play_from_state(
        seed=20260726,
        players=players,
        stock=expanded_deck(+remaining),
        current=0,
        max_turns=1,
    )

    assert result.reason == "post_action_hu"
    assert result.winner == 0
    assert result.total_xi >= 9
    assert not result.violations
    assert not result.coverage_failures


def test_quad_compensation_pair_below_nine_xi_is_a_non_winning_terminal():
    rules = rules_for_room(wildcard_enabled=True, players=3)
    player = SimPlayer(
        0,
        ["王", "王"],
        [
            SimMeld("mixed_same_rank_triplet", ("三", "三", "叁")),
            SimMeld("pao", ("六",) * 4),
            SimMeld("mixed_same_rank_triplet", ("四", "四", "肆")),
            SimMeld("normal_sequence", ("八", "九", "十")),
            SimMeld("normal_sequence", ("柒", "捌", "玖")),
            SimMeld("normal_sequence", ("二", "三", "四")),
        ],
        quad_events=1,
    )
    players = [player, SimPlayer(1), SimPlayer(2)]
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(player.hand)
    for meld in player.melds:
        remaining.subtract(meld.cards)
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=True,
        rules=rules,
    )

    result = simulator.play_from_state(
        seed=20260726,
        players=players,
        stock=expanded_deck(+remaining),
        current=0,
        max_turns=1,
    )

    assert result.reason == "complete_hand_below_min_xi"
    assert result.winner is None
    assert result.stalled_seat == 0
    assert result.total_xi < 9
    assert not result.violations
    assert not result.coverage_failures


def test_two_quad_wild_pair_terminal_is_a_verified_post_action_hu():
    rules = rules_for_room(wildcard_enabled=True, players=3)
    player = SimPlayer(
        0,
        ["叁", "王", "叁", "叁", "王"],
        [
            SimMeld("ti", ("三",) * 4),
            SimMeld("pao", ("四",) * 4),
            SimMeld("mixed_same_rank_triplet", ("五", "伍", "伍")),
            SimMeld("mixed_same_rank_triplet", ("二", "二", "贰")),
            SimMeld("peng", ("七",) * 3),
        ],
        quad_events=2,
    )
    players = [player, SimPlayer(1), SimPlayer(2)]
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(player.hand)
    for meld in player.melds:
        remaining.subtract(meld.cards)
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=True,
        rules=rules,
    )

    result = simulator.play_from_state(
        seed=20260726,
        players=players,
        stock=expanded_deck(+remaining),
        current=0,
        max_turns=1,
    )

    assert result.reason == "post_action_hu"
    assert result.winner == 0
    assert result.total_xi >= 9
    assert not result.violations
    assert not result.coverage_failures


def test_no_discardable_non_wildcard_incomplete_state_remains_an_invariant_failure():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    player = SimPlayer(
        0,
        ["叁", "叁", "叁"],
        [
            SimMeld("ti", ("三",) * 4),
            SimMeld("pao", ("四",) * 4),
            SimMeld("mixed_same_rank_triplet", ("五", "伍", "伍")),
            SimMeld("mixed_same_rank_triplet", ("二", "二", "贰")),
            SimMeld("peng", ("七",) * 3),
        ],
        quad_events=2,
    )
    players = [player, SimPlayer(1), SimPlayer(2)]
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(player.hand)
    for meld in player.melds:
        remaining.subtract(meld.cards)
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_state(
        seed=20260726,
        players=players,
        stock=expanded_deck(+remaining),
        current=0,
        max_turns=1,
    )

    assert result.reason == "invariant_failure"
    assert not result.coverage_failures
    assert "no_discardable_card" in result.violations[0]


def test_resumed_draw_auto_meld_skips_discard_when_only_locked_cards_remain():
    rules = rules_for_room(wildcard_enabled=False, players=2)
    player = SimPlayer(
        0,
        ["十", "十", "十", "八", "八", "八"],
        [
            SimMeld("special_123", ("壹", "贰", "叁")),
            SimMeld("special_2710", ("贰", "柒", "拾")),
            SimMeld("peng", ("五", "五", "五")),
            SimMeld("wei", ("玖", "玖", "玖")),
            SimMeld("mixed_same_rank_triplet", ("一", "壹", "壹")),
        ],
    )
    players = [player, SimPlayer(1)]
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(player.hand)
    for meld in player.melds:
        remaining.subtract(meld.cards)
    simulator = FullGameSimulator(
        [_PassHuPolicy(), BaselinePolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_state(
        seed=20261869,
        players=players,
        stock=expanded_deck(+remaining),
        current=0,
        initial_drawn_card="八",
        max_turns=1,
    )

    assert result.action_counts["wei"] == 1
    assert result.action_counts["auto_meld_no_discard"] == 1
    assert not result.violations
    assert not result.coverage_failures


def test_complete_non_wildcard_hand_below_nine_xi_is_a_non_winning_terminal():
    rules = rules_for_room(wildcard_enabled=False, players=2)
    player = SimPlayer(
        0,
        [],
        [
            SimMeld("mixed_same_rank_triplet", ("九", "玖", "玖")),
            SimMeld("normal_sequence", ("八", "九", "十")),
            SimMeld("normal_sequence", ("陆", "柒", "捌")),
            SimMeld("mixed_same_rank_triplet", ("三", "三", "叁")),
            SimMeld("mixed_same_rank_triplet", ("十", "十", "拾")),
            SimMeld("normal_sequence", ("捌", "玖", "拾")),
            SimMeld("wei", ("二", "二", "二")),
        ],
    )
    players = [player, SimPlayer(1)]
    remaining = full_deck_counts(rules=rules)
    for meld in player.melds:
        remaining.subtract(meld.cards)
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_state(
        seed=20260727,
        players=players,
        stock=expanded_deck(+remaining),
        current=0,
        max_turns=1,
        needs_draw=False,
    )

    assert result.reason == "complete_hand_below_min_xi"
    assert result.winner is None
    assert result.stalled_seat == 0
    assert result.total_xi < 9
    assert not result.coverage_failures
    assert not result.violations


def test_benchmark_rotates_search_seat_and_dealer_independently_on_shared_deals():
    assignments = [_benchmark_assignment(index, 1000) for index in range(9)]

    assert {(search, dealer) for search, dealer, _seed in assignments} == {
        (search, dealer) for search in range(3) for dealer in range(3)
    }
    assert [seed for _search, _dealer, seed in assignments[:3]] == [1000, 1000, 1000]


def test_heads_up_benchmark_rotates_both_seats_and_dealers():
    assignments = [
        _benchmark_assignment(index, 1000, players=2)
        for index in range(4)
    ]

    assert {(search, dealer) for search, dealer, _seed in assignments} == {
        (search, dealer) for search in range(2) for dealer in range(2)
    }
    assert [seed for _search, _dealer, seed in assignments[:2]] == [1000, 1000]


def test_information_set_search_reuses_hu_results_within_one_decision(
    monkeypatch,
):
    rules = rules_for_room(wildcard_enabled=True, players=2)
    hand = (
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "王",
    )
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(hand)
    view = PublicView(
        seat=0,
        hand=hand,
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=tuple(sorted((+remaining).items())),
        stock_count=sum((+remaining).values()),
    )
    calls: Counter[tuple[tuple[str, ...], tuple[SimMeld, ...]]] = Counter()
    uncached_evaluate_hu = evaluate_hu

    def counted_evaluate_hu(hand_arg, melds_arg, rules_arg):
        key = (tuple(sorted(hand_arg)), tuple(melds_arg))
        calls[key] += 1
        return uncached_evaluate_hu(hand_arg, melds_arg, rules_arg)

    monkeypatch.setattr(
        "ai.full_game_simulator.evaluate_hu",
        counted_evaluate_hu,
    )

    selected = InformationSetSearchPolicy().choose_discard(view, rules)

    assert selected in hand
    assert calls
    assert max(calls.values()) == 1
    labels = sorted(label for label in hand if label != "王")
    batched = InformationSetSearchPolicy._cheap_discard_values(
        view,
        rules,
        labels,
    )
    assert batched == {
        label: InformationSetSearchPolicy._cheap_discard_value(
            view,
            label,
            rules,
        )
        for label in labels
    }
    relevance = _draw_shape_relevance_values(
        tuple(QUICK_POTENTIAL_LABELS),
        hand,
    )
    assert relevance == {
        label: _draw_shape_relevance(label, hand)
        for label in QUICK_POTENTIAL_LABELS
    }


def test_draw_without_auto_meld_reuses_the_same_hu_evaluation(monkeypatch):
    rules = rules_for_room(wildcard_enabled=False, players=2)
    remaining = full_deck_counts(rules=rules)
    opponent_hand = ["十"]
    remaining.subtract(opponent_hand)
    stock = expanded_deck(+remaining)
    drawn = stock[-1]
    players = [SimPlayer(0), SimPlayer(1, opponent_hand)]
    calls: Counter[tuple[tuple[str, ...], tuple[SimMeld, ...]]] = Counter()
    uncached_evaluate_hu = evaluate_hu

    def counted_evaluate_hu(hand_arg, melds_arg, rules_arg):
        key = (tuple(sorted(hand_arg)), tuple(melds_arg))
        calls[key] += 1
        return uncached_evaluate_hu(hand_arg, melds_arg, rules_arg)

    monkeypatch.setattr(
        "ai.full_game_simulator.evaluate_hu",
        counted_evaluate_hu,
    )
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_state(
        seed=20260823,
        players=players,
        stock=stock,
        current=0,
        needs_draw=True,
        max_turns=1,
    )

    assert not result.violations
    assert calls[((drawn,), ())] == 1


def test_professional_brain_simulation_policy_uses_production_peng_and_chi():
    view = PublicView(
        seat=0,
        hand=("二", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), (), ()),
        discards=((), (), ()),
        remaining_counts=(),
        stock_count=19,
    )
    policy = ProfessionalBrainSimulationPolicy()
    rules = rules_for_room(wildcard_enabled=False)

    assert policy.choose_peng(view, "二", rules)

    chi_view = PublicView(
        seat=0,
        hand=("二", "十", "四", "六", "九"),
        own_melds=(),
        all_melds=((), (), ()),
        discards=((), (), ()),
        remaining_counts=(),
        stock_count=19,
    )
    plan = ChiPlan(("二", "七", "十"), (), ("二", "十"))

    assert policy.choose_chi(chi_view, [plan], rules) == plan


def test_professional_brain_simulation_policy_trusts_authoritative_hu(
    monkeypatch,
):
    class RejectedHuDecision:
        selected_action = "PASS"

        @staticmethod
        def to_dict():
            return {
                "selected_action": "PASS",
                "reason": "local_partition_false_negative",
            }

    view = PublicView(
        seat=0,
        hand=("二", "七", "十"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(),
        stock_count=39,
    )
    policy = ProfessionalBrainSimulationPolicy()
    monkeypatch.setattr(
        policy,
        "_choose",
        lambda *args, **kwargs: RejectedHuDecision(),
    )

    accepted = policy.choose_hu(
        view,
        HuEvaluation(True, 9, (), 1.0, 3),
        rules_for_room(wildcard_enabled=False, players=2),
    )

    assert accepted
    assert policy.last_decision_evidence["production_selected_action"] == "PASS"
    assert policy.last_decision_evidence["selected_action"] == "HU"
    assert (
        policy.last_decision_evidence["reason"]
        == "authoritative_simulator_hu_overrode_local_false_negative"
    )


def test_production_state_can_keep_opponents_in_turn_order():
    view = PublicView(
        seat=1,
        hand=("二", "四", "六", "九"),
        own_melds=(),
        all_melds=(
            (SimMeld("peng", ("壹", "壹", "壹")),),
            (),
            (SimMeld("chi", ("一", "二", "三")),),
        ),
        discards=(("伍",), ("五",), ("六",)),
        remaining_counts=(),
        stock_count=19,
        hand_sizes=(15, 16, 14),
    )

    state = _production_state_from_public_view(
        view,
        legal_actions=[{"type": "DISCARD"}],
        pending_card=None,
        chi_options=None,
        seat_aware_opponents=True,
    )
    opponents = state["memory"]["opponents"]

    assert [entry["seat"] for entry in opponents] == [2, 0]
    assert [entry["relative_offset"] for entry in opponents] == [1, 2]
    assert [entry["is_next_seat"] for entry in opponents] == [True, False]
    assert opponents[0]["discards"] == ["六"]
    assert opponents[1]["meld_groups"][0]["labels"] == ["壹", "壹", "壹"]


class _NoClaimPolicy(BaselinePolicy):
    def choose_peng(self, view, label, rules):
        return False

    def choose_chi(self, view, plans, rules):
        return None


class _AlwaysPengPolicy(_NoClaimPolicy):
    def choose_peng(self, view, label, rules):
        return True


class _AlwaysChiPolicy(_NoClaimPolicy):
    def choose_chi(self, view, plans, rules):
        return plans[0] if plans else None


class _PassHuPolicy(_NoClaimPolicy):
    def choose_hu(self, view, hu, rules):
        return False


class _PassHuAlwaysChiPolicy(_AlwaysChiPolicy):
    def choose_hu(self, view, hu, rules):
        return False


class _RecordingNoClaimPolicy(_NoClaimPolicy):
    def __init__(self):
        self.peng_view = None
        self.chi_view = None

    def choose_peng(self, view, label, rules):
        self.peng_view = view
        return False

    def choose_chi(self, view, plans, rules):
        self.chi_view = view
        return None


def _state_with_pending_card(rules, *, hands, pending):
    remaining = full_deck_counts(rules=rules)
    for hand in hands:
        remaining.subtract(hand)
    remaining.subtract((pending,))
    assert all(amount >= 0 for amount in remaining.values())
    players = [
        SimPlayer(seat=index, hand=list(hand))
        for index, hand in enumerate(hands)
    ]
    return players, expanded_deck(+remaining)
