"""Professional-brain contract tests from the implementation document."""

from copy import deepcopy
from collections import Counter
import gc
import random
import tracemalloc

import pytest

import ai.pro_brain as pro_brain
from ai.pro_brain import (
    EVScorer,
    TingEstimate,
    allocate_hand_structures,
    analyze_hand,
    build_decision_context,
    check_hu,
    choose_action,
    evaluate_action_ev,
    estimate_ting_distance,
    generate_legal_actions,
    simulate_action,
    validate_decision_consistency,
    _response_ting_reject_reason,
    _information_set_draw_value,
    _room_phase_thresholds,
    _choose_best_soft_allocation,
    _generate_soft_candidates,
    _scaled_discard_danger_loss,
    _soft_type_priority,
    _ting_value_delta,
    _context_after_ting_draw,
    _grouping_ting_hu_probe,
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
TING_HAND = [
    "贰", "柒",
    "壹", "贰", "叁",
    "九", "九", "九",
    "一", "二", "三",
    "四", "五", "六",
    "七", "八", "九",
    "肆", "伍", "陆",
]
WILD_CONTINUE_HAND = [
    "王", "王", "贰", "柒", "壹", "贰", "叁", "九", "九", "九", "二", "七",
    "四", "五", "六", "七", "八", "九", "肆", "伍", "陆",
]
ZERO_XI_COMPLETE_HAND = [
    "二", "三", "四",
    "五", "六", "七",
    "七", "八", "九",
    "贰", "叁", "肆",
    "肆", "伍", "陆",
    "陆", "柒", "捌",
    "柒", "捌", "玖",
]


def test_decision_context_preserves_passed_claim_state_in_fingerprint():
    rules = rules_for_room(wildcard_enabled=True, players=2)
    base = {
        "hand": ["一", "二"],
        "legal_actions": [{"type": "DISCARD"}],
    }
    plain = build_decision_context(base, rules=rules)
    passed = build_decision_context(
        {
            **base,
            "memory": {
                "my_passed_chi": ["七"],
                "my_passed_peng": ["陆"],
            },
        },
        rules=rules,
    )

    assert passed.passed_chi == ("七",)
    assert passed.passed_peng == ("陆",)
    assert passed.to_dict()["passed_chi"] == ["七"]
    assert passed.to_dict()["passed_peng"] == ["陆"]
    assert passed.state_fingerprint != plain.state_fingerprint


def _soft_meld_score(meld) -> float:
    return (
        meld.structure_value
        + meld.xi_value * 40
        + _soft_type_priority(meld.type) * 10.0
    )


def _soft_objective(melds) -> tuple[float, int]:
    return sum(_soft_meld_score(meld) for meld in melds), len(melds)


def _exhaustive_soft_objective(candidates) -> tuple[float, int]:
    best = (0.0, 0)

    def search(index: int, used: set[str], score: float, count: int) -> None:
        nonlocal best
        if index >= len(candidates):
            best = max(best, (score, count))
            return
        meld = candidates[index]
        ids = set(meld.card_ids)
        if not ids & used:
            search(
                index + 1,
                used | ids,
                score + _soft_meld_score(meld),
                count + 1,
            )
        search(index + 1, used, score, count)

    search(0, set(), 0.0, 0)
    return best


def test_instance_allocation_does_not_reuse_triplet_for_123():
    context = build_decision_context(["一", "二", "二", "二", "三"])
    allocation = allocate_hand_structures(context)
    decision = choose_action(["一", "二", "二", "二", "三"])

    assert allocation.locked_melds[0].type == "exact_triplet"
    assert allocation.locked_counts["二"] == 3
    assert allocation.free_counts_after_locked.get("二", 0) == 0
    assert not allocation.soft_melds
    assert decision.selected_label in {"一", "三"}
    assert all(item.label != "二" or not item.allowed for item in decision.action_evals)
    assert any("free_count is 0" in note for note in allocation.allocation_notes)


def test_mixed_triplet_and_sequence_use_different_card_instances_when_possible():
    context = build_decision_context(["伍", "伍", "五", "四", "五", "六", "九"])
    allocation = allocate_hand_structures(context)
    used = [card_id for meld in allocation.soft_melds for card_id in meld.card_ids]

    assert len(used) == len(set(used))
    assert {meld.type for meld in allocation.soft_melds} >= {"mixed_same_rank_triplet", "normal_sequence"}
    assert choose_action(context.state).selected_label == "九"


def test_single_five_cannot_feed_mixed_triplet_and_sequence_together():
    allocation = allocate_hand_structures(build_decision_context(["伍", "伍", "五", "四", "六", "九"]))
    types = [meld.type for meld in allocation.soft_melds]

    assert not ("mixed_same_rank_triplet" in types and "normal_sequence" in types)
    assert not allocation.allocation_conflicts


@pytest.mark.parametrize(
    "hand",
    [
        ["一", "二", "三", "四", "五", "六", "七", "九"],
        ["二", "七", "十", "贰", "柒", "拾", "五", "伍"],
        ["四", "五", "六", "五", "伍", "七", "八", "九"],
    ],
)
def test_soft_allocation_dynamic_program_matches_exhaustive_objective(hand):
    context = build_decision_context(hand)
    candidates = _generate_soft_candidates(context.card_instances, context.rules)

    selected = _choose_best_soft_allocation(candidates)
    exhaustive_score, exhaustive_count = _exhaustive_soft_objective(candidates)

    assert _soft_objective(selected) == (exhaustive_score, exhaustive_count)
    used_ids = [card_id for meld in selected for card_id in meld.card_ids]
    assert len(used_ids) == len(set(used_ids))


def test_soft_allocation_releases_per_call_recursive_cache_without_cyclic_gc():
    context = build_decision_context(
        [
            "一", "壹", "壹", "拾", "二", "二", "十", "十", "四", "五", "六",
            "九", "肆", "伍", "玖", "陆", "七", "柒", "柒", "八", "玖",
        ]
    )
    gc.collect()
    gc.disable()
    tracemalloc.start()
    try:
        for _ in range(300):
            allocate_hand_structures(context)
        current_bytes, _peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        gc.enable()
        gc.collect()

    assert current_bytes < 4_000_000


@pytest.mark.parametrize(
    ("hand", "expected"),
    [
        (["一", "二", "二", "二", "三"], {"hard": {"exact_triplet"}, "free": {"二": 0}, "missing_soft": {"special_123"}}),
        (["四", "四", "四", "四", "九"], {"hard": {"exact_quad"}, "free": {"四": 0}}),
        (["王", "二", "七", "九"], {"hard": {"wildcard_meld"}, "hard_labels": {"王"}}),
        (["伍", "伍", "五", "九"], {"soft": {"mixed_same_rank_triplet"}, "free_labels": {"九"}}),
        (["四", "五", "六", "九"], {"soft": {"normal_sequence"}, "free_labels": {"九"}}),
        (["二", "七", "十", "九"], {"soft": {"special_2710"}, "free_labels": {"九"}}),
        (["壹", "贰", "叁", "九"], {"soft": {"special_123"}, "free_labels": {"九"}}),
        (["伍", "伍", "五", "四", "六", "九"], {"not_both": {"mixed_same_rank_triplet", "normal_sequence"}}),
        (["伍", "伍", "五", "四", "五", "六", "九"], {"soft": {"mixed_same_rank_triplet", "normal_sequence"}, "free_labels": {"九"}}),
        (["八", "八", "九"], {"soft": {"pair"}, "free_labels": {"九"}}),
    ],
)
def test_structure_allocation_contract_cases(hand, expected):
    allocation = allocate_hand_structures(build_decision_context(hand))
    hard_types = {meld.type for meld in allocation.locked_melds}
    soft_types = {meld.type for meld in allocation.soft_melds}
    free_labels = {card.label for card in allocation.free_discard_instances}

    assert not allocation.allocation_conflicts
    if "hard" in expected:
        assert expected["hard"] <= hard_types
    if "soft" in expected:
        assert expected["soft"] <= soft_types
    if "missing_soft" in expected:
        assert not (expected["missing_soft"] & soft_types)
    if "not_both" in expected:
        assert not expected["not_both"] <= soft_types
    for label, count in expected.get("free", {}).items():
        assert allocation.free_counts_after_locked.get(label, 0) == count
    if "hard_labels" in expected:
        assert expected["hard_labels"] <= set(allocation.hard_protected_labels)
    if "free_labels" in expected:
        assert expected["free_labels"] <= free_labels


def test_hand_analysis_contains_xi_ting_wildcard_and_red_black_plans():
    context = build_decision_context(["王", "二", "七", "十", "九"])
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)
    payload = analysis.to_dict()

    assert payload["wildcard_value"]["wildcard_count"] == 1
    assert payload["red_black_plan"]["red_count"] == 3
    assert "ting_estimate" in payload
    assert payload["min_xi"] == 9


def test_room_phase_thresholds_preserve_three_player_and_scale_heads_up_stock():
    assert _room_phase_thresholds(
        rules_for_room(wildcard_enabled=False, players=3)
    ) == (15, 8)
    assert _room_phase_thresholds(
        rules_for_room(wildcard_enabled=True, players=3)
    ) == (15, 8)
    assert _room_phase_thresholds(
        rules_for_room(wildcard_enabled=False, players=2)
    ) == (31, 16)
    assert _room_phase_thresholds(
        rules_for_room(wildcard_enabled=True, players=2)
    ) == (28, 15)


def test_ting_estimator_enumerates_one_draw_hu_cards():
    context = build_decision_context(TING_HAND)
    allocation = allocate_hand_structures(context)
    ting = estimate_ting_distance(context, allocation)

    assert ting.shanten_like_distance == 1
    assert ting.is_ting
    assert "拾" in ting.hu_cards
    assert "拾" in ting.waiting_cards
    assert ting.expected_xi_if_hu >= ting.shanten_like_distance
    assert ting.expected_score_if_hu > 0
    assert "exact_hu_draw_probe=enabled" in ting.notes


def test_information_set_values_remaining_outs_not_only_wait_labels():
    open_context = build_decision_context(TING_HAND)
    thin_context = build_decision_context(
        {
            "hand": TING_HAND,
            "memory": {"opponent_discards": ["拾", "拾", "拾"]},
        }
    )
    open_analysis = analyze_hand(open_context, allocate_hand_structures(open_context))
    thin_analysis = analyze_hand(thin_context, allocate_hand_structures(thin_context))

    open_value = _information_set_draw_value(open_context, open_analysis)
    thin_value = _information_set_draw_value(thin_context, thin_analysis)

    assert open_value["hu_labels"] == thin_value["hu_labels"]
    assert "拾" in open_value["hu_labels"]
    assert open_value["hu_outs_by_label"]["拾"] == 4
    assert thin_value["hu_outs_by_label"]["拾"] == 1
    assert open_value["hu_outs"] - thin_value["hu_outs"] == 3
    assert open_value["value"] > thin_value["value"]


def test_ting_estimator_separates_improvement_from_hu_waits():
    context = build_decision_context(["一", "二", "四", "五", "九"])
    allocation = allocate_hand_structures(context)
    ting = estimate_ting_distance(context, allocation)

    assert not ting.is_ting
    assert "三" in ting.improving_cards
    assert not ting.hu_cards
    assert ting.shanten_like_distance > 1


def test_ting_value_delta_penalizes_losing_waits():
    before = TingEstimate(1, True, ["一", "二", "三"], [], ["一", "二", "三"], 9, 360.0, [])
    after = TingEstimate(1, True, ["一", "二"], [], ["一", "二"], 9, 360.0, [])

    assert _ting_value_delta(before, after, {}) < 0
    assert _ting_value_delta(before, after, {}, include_wait_breadth=False) == 0


def test_discard_ev_rewards_reducing_ting_distance_and_hu_waits():
    decision = choose_action([*TING_HAND, "十"])
    discard_ten = next(item for item in decision.action_evals if item.type == "DISCARD" and item.label == "十")
    payload = discard_ten.to_dict()

    assert discard_ten.allowed
    assert discard_ten.debug_details["ting_value_delta"] > 0
    assert "拾" in discard_ten.debug_details["ting_after"]["hu_cards"]
    assert payload["score_breakdown"]["ting_value_delta"] == discard_ten.debug_details["ting_value_delta"]


def test_discard_ev_weights_each_hu_out_by_its_own_score():
    state = {
        "hand": [
            "柒", "六", "壹", "拾", "八", "二", "八", "伍", "捌", "肆", "二",
            "陆", "二", "王", "贰", "捌", "拾", "三", "贰", "王", "一",
        ],
        "strategy_existing_melds": [],
        "legal_actions": [{"type": "DISCARD"}],
        "remaining_deck_count": 21,
        "memory": {
            "my_discards": [],
            "opponent_discards": ["七", "伍"],
            "opponent_meld_groups": [
                {"type": "normal_sequence", "labels": ["七", "八", "九"]},
                {"type": "mixed_same_rank_triplet", "labels": ["九", "九", "玖"]},
            ],
        },
    }

    decision = choose_action(state)
    discard_six = next(
        item for item in decision.action_evals if item.type == "DISCARD" and item.label == "六"
    )
    discard_big_one = next(
        item for item in decision.action_evals if item.type == "DISCARD" and item.label == "壹"
    )
    broad_info = discard_big_one.debug_details["information_set"]
    narrow_info = discard_six.debug_details["information_set"]

    assert broad_info["version"] == "information_set_search_v3"
    assert broad_info["hu_outs"] > narrow_info["hu_outs"]
    assert broad_info["expected_hu_score"] < narrow_info["expected_hu_score"]
    assert broad_info["expected_hu_score"] < max(broad_info["hu_score_by_label"].values())
    weighted_score = sum(
        broad_info["hu_outs_by_label"][label] * score
        for label, score in broad_info["hu_score_by_label"].items()
    ) / broad_info["hu_outs"]
    assert broad_info["expected_hu_score"] == pytest.approx(weighted_score, abs=1e-3)


def test_discard_does_not_use_ting_override_when_hand_is_not_really_ting():
    state = {
        "hand": ["壹", "二", "叁", "叁", "肆", "伍", "七", "七", "柒", "柒", "九", "九", "十"],
        "legal_actions": [{"type": "DISCARD"}],
        "remaining_deck_count": 20,
    }

    decision = choose_action(state)
    analysis = decision.context_snapshot["hand_analysis"]["ting_estimate"]
    selected = next(
        item
        for item in decision.action_evals
        if item.type == "DISCARD" and item.action.card_id == decision.selected_card_id
    )
    break_ting_labels = {
        item.label
        for item in decision.action_evals
        if item.type == "DISCARD" and item.reject_reason == "discard_breaks_ting"
    }

    assert decision.selected_action == "DISCARD"
    assert selected.allowed
    assert analysis["is_ting"] is False
    assert selected.debug_details["candidate_stage"] != "ting_preserve_override"
    assert break_ting_labels == set()


def test_discard_recognizes_quad_compensation_pair_ting_candidates():
    state = {
        "hand": list("玖一一捌叁八七陆九十二九六捌叁六三"),
        "strategy_existing_melds": [{"type": "pao", "labels": ["捌", "捌", "捌", "捌"]}],
        "legal_actions": [{"type": "DISCARD"}],
    }

    decision = choose_action(state)
    discard_evals = [item for item in decision.action_evals if item.type == "DISCARD"]

    assert decision.context_snapshot["hand_analysis"]["ting_estimate"]["is_ting"] is True
    assert decision.selected_action == "DISCARD"
    assert any(item.allowed for item in discard_evals)
    assert not any(item.reject_reason == "discard_breaks_ting" for item in discard_evals)
    assert all(not item.debug_details["would_break_ting"] for item in discard_evals if item.allowed)


def test_triplet_plus_three_pairs_is_not_ready_to_hu():
    state = {
        "hand": ["壹", "壹", "壹", "柒", "柒", "玖", "玖", "拾", "拾"],
        "legal_actions": [{"type": "WAIT"}],
    }
    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    ting = estimate_ting_distance(context, allocation)

    assert not check_hu(context, allocation).can_hu
    assert not ting.is_ting
    assert ting.hu_cards == []


def test_discard_tiebreak_preserves_pair_and_chooses_lonely_nine():
    state = {
        "hand": [
            "壹",
            "壹",
            "二",
            "二",
            "贰",
            "贰",
            "三",
            "叁",
            "四",
            "肆",
            "伍",
            "六",
            "陆",
            "七",
            "柒",
            "八",
            "八",
            "玖",
            "王",
            "王",
            "五",
        ],
        "legal_actions": [{"type": "DISCARD"}],
    }

    decision = choose_action(state)
    discard_one = next(item for item in decision.action_evals if item.type == "DISCARD" and item.label == "壹")
    discard_nine = next(item for item in decision.action_evals if item.type == "DISCARD" and item.label == "玖")

    assert decision.selected_label == "玖"
    assert discard_nine.ev > discard_one.ev
    assert discard_one.debug_details["shape_tiebreak"] < 0
    assert "pair kept for future peng" in discard_one.reason


def test_high_discard_danger_is_scaled_before_ev_penalty():
    assert _scaled_discard_danger_loss(100, {"high_danger_threshold": 250, "high_danger_multiplier": 1.6}) == 100
    assert _scaled_discard_danger_loss(335, {"high_danger_threshold": 250, "high_danger_multiplier": 1.6}) == 536


def test_soft_break_pair_uses_stronger_penalty():
    decision = choose_action({"hand": ["八", "八"], "legal_actions": [{"type": "DISCARD"}]})
    discard_evals = [item for item in decision.action_evals if item.type == "DISCARD" and item.allowed]

    assert discard_evals
    assert all(item.structure_loss >= 650 for item in discard_evals)
    assert all(item.debug_details["candidate_stage"] == "break_soft_protection" for item in discard_evals)


def test_policy_brain_scores_every_legal_action_with_ev_scorer_and_simulation():
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "DISCARD"}, {"type": "CHI"}, {"type": "PASS"}, {"type": "WAIT"}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 1.0}],
    }
    decision = choose_action(state)

    scored = [item for item in decision.action_evals if item.action.allowed and item.allowed]
    assert scored
    assert all(item.debug_details.get("scorer") == "EVScorer" for item in scored)
    assert all("simulation" in item.debug_details for item in scored)


def test_duplicate_discard_instances_share_semantic_hand_analysis(monkeypatch):
    import ai.pro_brain as pro_brain

    actual_analyze_hand = pro_brain.analyze_hand
    calls = 0

    def counted_analyze_hand(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual_analyze_hand(*args, **kwargs)

    monkeypatch.setattr(pro_brain, "analyze_hand", counted_analyze_hand)
    decision = pro_brain.choose_action(
        {
            "hand": ["一", "一", "二"],
            "legal_actions": [{"type": "DISCARD"}],
        }
    )

    discard_evals = [
        item for item in decision.action_evals if item.type == "DISCARD"
    ]
    assert len(discard_evals) == 3
    assert calls == 3


def test_wait_action_is_generated_simulated_and_scored_without_action_plan():
    state = {"hand": ["一", "二", "三"], "legal_actions": [{"type": "WAIT"}]}
    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)
    legal_actions, rejected = generate_legal_actions(context, allocation, analysis)
    wait = next(item for item in legal_actions if item.type == "WAIT")

    sim = simulate_action(context, wait, context, allocation)
    ev = evaluate_action_ev(context, wait, context, allocation, analysis)
    decision = choose_action(state)

    assert not rejected
    assert sim.hand_after == context.normalized_hand
    assert "wait_simulated" in sim.notes
    assert sim.debug_details["state_change"] == "none"
    assert ev.allowed
    assert ev.debug_details["candidate_stage"] == "wait"
    assert ev.debug_details["simulation"]["notes"] == sim.notes
    assert decision.selected_action == "WAIT"
    assert not decision.requires_action_plan


def test_action_simulator_models_chi_consumption_and_followup_discard():
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 1.0}],
    }
    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    legal_actions, _ = generate_legal_actions(context, allocation, analyze_hand(context, allocation))
    chi = next(item for item in legal_actions if item.type == "CHI")

    sim = simulate_action(context, chi, context, allocation)
    ev = EVScorer().evaluate(context, chi, allocation, analyze_hand(context, allocation))

    assert sim.consumed_card_ids
    assert sim.debug_details["consumed_from_hand"] == ["二", "十"]
    assert sim.debug_details["meld_type"] == "special_2710"
    assert sim.hand_after == ["四", "六", "九"]
    assert sim.followup_discard and sim.followup_discard["type"] == "DISCARD"
    assert ev.debug_details["simulation"]["consumed_card_ids"] == sim.consumed_card_ids


def test_response_evaluation_reuses_followup_discard_only_within_one_decision(monkeypatch):
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 1.0}],
    }
    original = pro_brain._best_followup_discard
    fingerprints: list[str] = []

    def counted(after_context):
        fingerprints.append(after_context.state_fingerprint)
        return original(after_context)

    monkeypatch.setattr(pro_brain, "_best_followup_discard", counted)

    first = choose_action(state)
    first_count = len(fingerprints)
    second = choose_action(state)

    assert first.to_dict() == second.to_dict()
    assert first_count == 1
    assert len(fingerprints) == 2


def test_ting_draw_context_matches_full_state_rebuild():
    state = {
        "context_id": "ting-equivalence",
        "frame_id": "frame-7",
        "phase": "play",
        "hand": ["一", "二", "三", "四"],
        "hand_details": [
            {"card_id": "h001", "name": "一", "label": "一", "confidence": 0.99, "x": 10, "y": 20},
            {"card_id": "h002", "name": "二", "label": "二", "confidence": 0.98, "x": 30, "y": 20},
            {"card_id": "h003", "name": "三", "label": "三", "confidence": 0.97, "x": 50, "y": 20},
            {"card_id": "h004", "name": "四", "label": "四", "confidence": 0.96, "x": 70, "y": 20},
        ],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "buttons": [{"name": "chi", "confidence": 0.9}],
        "pending_card": "伍",
        "chi_options": [{"option_id": "chi-1", "labels": ["三", "四", "五"]}],
        "recognition_warnings": ["transient-warning"],
    }
    context = build_decision_context(state)

    actual = _context_after_ting_draw(context, "六")
    drawn = actual.card_instances[-1]
    expected_state = {
        **context.state,
        "hand": [card.label for card in actual.card_instances],
        "raw_hand": [card.label for card in actual.card_instances],
        "hand_details": [card.to_dict() | {"name": card.label} for card in actual.card_instances],
        "recognition_warnings": [],
        "recognition_errors": [],
    }
    expected = build_decision_context(expected_state, rules=context.rules)

    assert drawn.label == "六"
    assert actual.state == expected.state
    assert actual.to_dict() == expected.to_dict()


def test_ting_hu_probe_cache_is_scoped_to_one_top_level_decision(monkeypatch):
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 1.0}],
    }
    original = pro_brain._grouping_ting_hu_probe
    calls: Counter[tuple[object, ...]] = Counter()

    def counted(context, rules):
        if isinstance(context, pro_brain.DecisionContext) and any(
            card.source == "ting_probe" for card in context.card_instances
        ):
            resolved_allocation = allocate_hand_structures(context)
            calls[pro_brain._ting_hu_probe_cache_key(
                context,
                resolved_allocation,
                rules,
            )] += 1
        return original(context, rules)

    monkeypatch.setattr(pro_brain, "_grouping_ting_hu_probe", counted)

    choose_action(state)
    first = calls.copy()
    choose_action(state)

    assert first
    assert max(first.values()) == 1
    assert calls == Counter({key: count * 2 for key, count in first.items()})


@pytest.mark.parametrize(
    "hand",
    [
        COMPLETE_HAND,
        [*COMPLETE_HAND[:-1], "王"],
        [*TING_HAND, "拾"],
        ZERO_XI_COMPLETE_HAND,
    ],
)
def test_ting_partition_prefilter_keeps_known_complete_shapes(hand):
    context = build_decision_context({"hand": hand, "legal_actions": [{"type": "DISCARD"}]})

    assert not pro_brain._ting_probe_partition_impossible(context)


def test_ting_partition_prefilter_preserves_full_response_decision(monkeypatch):
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 1.0}],
    }
    prefilter = pro_brain._ting_probe_partition_impossible
    monkeypatch.setattr(pro_brain, "_ting_probe_partition_impossible", lambda _context: False)
    baseline = choose_action(state).to_dict()
    monkeypatch.setattr(pro_brain, "_ting_probe_partition_impossible", prefilter)

    optimized = choose_action(state).to_dict()

    assert optimized == baseline


def test_ting_partition_prefilter_preserves_random_full_hu_results(monkeypatch):
    rng = random.Random(20260824)
    labels = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
              "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "拾", "王"]
    original_prefilter = pro_brain._ting_probe_partition_impossible

    for wildcard_enabled in (False, True):
        rules = rules_for_room(wildcard_enabled=wildcard_enabled, players=2)
        available = [label for label in labels if wildcard_enabled or label != "王"]
        deck = [label for label in available for _ in range(4)]
        for _ in range(24):
            hand = rng.sample(deck, 21)
            context = build_decision_context({"hand": hand}, rules=rules)
            allocation = allocate_hand_structures(context)
            monkeypatch.setattr(pro_brain, "_ting_probe_partition_impossible", lambda _context: False)
            baseline = check_hu(context, allocation, rules).to_dict()
            monkeypatch.setattr(pro_brain, "_ting_probe_partition_impossible", original_prefilter)
            optimized = check_hu(context, allocation, rules).to_dict()
            assert optimized == baseline


def test_ting_draw_features_reuse_only_identical_count_state(monkeypatch):
    rules = rules_for_room(wildcard_enabled=True, players=2)
    context = build_decision_context({"hand": [*TING_HAND, "王"]}, rules=rules)
    after = _context_after_ting_draw(context, "拾")
    original_allocate = pro_brain.allocate_hand_structures
    calls = 0

    def counted_allocate(value):
        nonlocal calls
        calls += 1
        return original_allocate(value)

    monkeypatch.setattr(pro_brain, "allocate_hand_structures", counted_allocate)
    uncached = pro_brain._ting_draw_features(
        after,
        rules,
        exact_probe_enabled=False,
    )
    token = pro_brain._TING_DRAW_FEATURE_CACHE.set({})
    try:
        first = pro_brain._ting_draw_features(after, rules, exact_probe_enabled=False)
        second = pro_brain._ting_draw_features(
            _context_after_ting_draw(context, "拾"),
            rules,
            exact_probe_enabled=False,
        )
    finally:
        pro_brain._TING_DRAW_FEATURE_CACHE.reset(token)

    assert first == second == uncached
    assert calls == 2


def test_action_simulator_models_peng_pair_consumption():
    state = {
        "hand": ["九", "九", "一", "二"],
        "legal_actions": [{"type": "PENG"}],
        "pending_card": "九",
    }
    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    legal_actions, _ = generate_legal_actions(context, allocation, analyze_hand(context, allocation))
    peng = next(item for item in legal_actions if item.type == "PENG")

    sim = simulate_action(context, peng, context, allocation)

    assert sim.consumed_card_ids
    assert sim.debug_details["consumed_from_hand"] == ["九", "九"]
    assert sim.debug_details["meld_type"] == "peng"
    assert sim.hand_after == ["一", "二"]
    assert "peng_simulated" in sim.notes


def test_conflict_guard_blocks_action_plan_reselected_card():
    state = {
        "hand": ["一", "二", "二", "二", "三"],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 0, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h002", "name": "二", "x": 10, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h003", "name": "二", "x": 20, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h004", "name": "二", "x": 30, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h005", "name": "三", "x": 40, "y": 0, "w": 10, "h": 10, "clickable": True},
        ],
        "legal_actions": [{"type": "discard"}],
    }
    context = build_decision_context(state)
    decision = choose_action(state)
    guard = validate_decision_consistency(
        context,
        decision,
        {
            "ready": True,
            "target_label": "二",
            "target_card_id": "h002",
            "reason": "bad",
        },
    )

    assert not guard.passed
    assert guard.safe_halt
    assert guard.reason == "conflict_action_plan_reselected_card"


def test_conflict_guard_blocks_state_mutation_between_policy_and_execution():
    state = {
        "context_id": "ctx_state_001",
        "frame_id": "frame_state_001",
        "hand": ["一", "四", "九"],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 0, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h002", "name": "四", "x": 10, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h003", "name": "九", "x": 20, "y": 0, "w": 10, "h": 10, "clickable": True},
        ],
        "legal_actions": [{"type": "discard"}],
    }
    decision = choose_action(state)
    mutated = deepcopy(state)
    mutated["hand"][0] = "二"
    mutated["hand_details"][0]["name"] = "二"

    guard = validate_decision_consistency(
        build_decision_context(mutated),
        decision,
        {
            "ready": True,
            "target_label": decision.selected_label,
            "target_card_id": decision.selected_card_id,
            "reason": "old action plan",
        },
    )

    assert not guard.passed
    assert guard.safe_halt
    assert guard.reason == "conflict_state_mutation"
    assert "expected_state_fingerprint" in guard.details
    assert "actual_state_fingerprint" in guard.details
    assert "changed_state_fields" in guard.details
    assert "card_instances" in guard.details["changed_state_fields"]


def test_professional_brain_halts_when_only_hard_cards_clickable_but_free_cards_exist():
    decision = choose_action(
        {
            "hand": ["叁", "叁", "叁", "四", "九"],
            "hand_details": [
                {"card_id": "h001", "name": "叁", "x": 0, "y": 0, "w": 10, "h": 10, "clickable": True},
                {"card_id": "h002", "name": "叁", "x": 10, "y": 0, "w": 10, "h": 10, "clickable": True},
                {"card_id": "h003", "name": "叁", "x": 20, "y": 0, "w": 10, "h": 10, "clickable": True},
                {"card_id": "h004", "name": "四", "x": 30, "y": 0, "w": 10, "h": 10, "clickable": False},
                {"card_id": "h005", "name": "九", "x": 40, "y": 0, "w": 10, "h": 10, "clickable": False},
            ],
            "legal_actions": [{"type": "discard"}],
        }
    )

    assert decision.selected_action == "SAFE_HALT"
    assert decision.reason == "only_hard_protected_clickable_but_unprotected_cards_exist_in_hand"


def _eval_for(decision, action_type):
    return next(item for item in decision.action_evals if item.type == action_type)


def test_chi_rejects_consuming_hard_protected_triplet_even_when_option_exists():
    state = {
        "hand": ["二", "二", "二", "一"],
        "legal_actions": [{"type": "CHI"}],
        "pending_card": "三",
        "chi_options": [{"option_id": "chi_123", "labels": ["一", "二", "三"], "confidence": 1.0}],
    }

    decision = choose_action(state)
    chi_eval = _eval_for(decision, "CHI")

    assert decision.selected_action == "PASS"
    assert not chi_eval.allowed
    assert chi_eval.reject_reason == "chi_consumes_hard_protected"
    assert chi_eval.debug_details["ev_delta_vs_pass"] < 0


def test_chi_response_does_not_require_consumed_hand_cards_to_be_clickable():
    state = {
        "hand": ["贰", "拾", "伍", "捌"],
        "hand_details": [
            {"card_id": "h001", "name": "贰", "x": 0, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h002", "name": "拾", "x": 10, "y": 0, "w": 10, "h": 10, "clickable": False},
            {"card_id": "h003", "name": "伍", "x": 20, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h004", "name": "捌", "x": 30, "y": 0, "w": 10, "h": 10, "clickable": True},
        ],
        "legal_actions": [{"type": "CHI"}],
        "buttons": [{"name": "chi", "x": 1, "y": 1, "w": 10, "h": 10}],
        "pending_card": "柒",
        "chi_options": [{"option_id": "chi_locked", "labels": ["贰", "柒", "拾"], "confidence": 1.0}],
    }

    decision = choose_action(state)
    chi_eval = _eval_for(decision, "CHI")

    assert chi_eval.reject_reason != "chi_consumes_nonclickable_locked_card"
    assert chi_eval.debug_details["consumed_from_hand"] == ["贰", "拾"]


def test_peng_response_does_not_require_consumed_pair_cards_to_be_clickable():
    state = {
        "hand": ["捌", "捌", "一", "四", "六", "九"],
        "hand_details": [
            {"card_id": "h001", "name": "捌", "x": 0, "y": 0, "w": 10, "h": 10, "clickable": False},
            {"card_id": "h002", "name": "捌", "x": 10, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h003", "name": "一", "x": 20, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h004", "name": "四", "x": 30, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h005", "name": "六", "x": 40, "y": 0, "w": 10, "h": 10, "clickable": True},
            {"card_id": "h006", "name": "九", "x": 50, "y": 0, "w": 10, "h": 10, "clickable": True},
        ],
        "legal_actions": [{"type": "PENG"}, {"type": "PASS"}],
        "buttons": [{"name": "peng", "x": 1, "y": 1, "w": 10, "h": 10}, {"name": "pass", "x": 12, "y": 1, "w": 10, "h": 10}],
        "pending_card": "捌",
    }

    decision = choose_action(state)
    peng_eval = _eval_for(decision, "PENG")

    assert decision.selected_action == "PENG"
    assert peng_eval.allowed
    assert peng_eval.debug_details["consumed_from_hand"] == ["捌", "捌"]


def test_chi_button_without_visible_options_expands_before_ev_choice():
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "七",
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "chi_options": [],
    }

    decision = choose_action(state)
    expand_eval = next(
        item
        for item in decision.action_evals
        if item.type == "EXPAND_CHI_OPTIONS"
    )

    assert decision.selected_action == "EXPAND_CHI_OPTIONS"
    assert decision.candidate_stage == "expand_chi_options"
    assert decision.reason == "EXPAND_CHI_OPTIONS reveal candidates; no chi meld selected"
    assert expand_eval.allowed is True
    assert expand_eval.debug_details["consumed_card_ids"] == []
    assert expand_eval.debug_details["followup_discard"] is None


def test_chi_visible_option_does_not_override_breaking_ting():
    state = {
        "hand": ["贰", "柒", "拾", "贰", "叁", "四", "五", "六", "七", "八", "九"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "buttons": [{"name": "chi", "x": 1, "y": 1, "w": 10, "h": 10}, {"name": "pass", "x": 20, "y": 1, "w": 10, "h": 10}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_breaks_ting", "labels": ["七", "七", "柒"], "confidence": 1.0}],
        "strategy_existing_melds": [
            {"type": "peng", "labels": ["玖", "玖", "玖"]},
            {"type": "special_123", "labels": ["一", "二", "三"]},
            {"type": "normal_sequence", "labels": ["肆", "伍", "陆"]},
        ],
    }

    decision = choose_action(state)
    chi_eval = _eval_for(decision, "CHI")

    assert decision.selected_action == "PASS"
    assert not chi_eval.allowed
    assert chi_eval.reject_reason == "chi_breaks_ting"
    assert chi_eval.debug_details["trusted_visible_response_override"] is False


def test_chi_accepts_high_value_2710_only_after_beating_pass_ev():
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 1.0}],
    }

    decision = choose_action(state)
    chi_eval = _eval_for(decision, "CHI")

    assert decision.selected_action == "CHI"
    assert decision.selected_option_id == "chi_2710"
    assert chi_eval.allowed
    assert chi_eval.ev > chi_eval.debug_details["pass_ev"]
    assert chi_eval.debug_details["consumed_from_hand"] == ["二", "十"]
    assert chi_eval.debug_details["followup_discard"]["type"] == "DISCARD"
    payload = chi_eval.to_dict()
    assert payload["pass_ev"] == chi_eval.debug_details["pass_ev"]
    assert payload["chi_ev"] == chi_eval.debug_details["chi_ev"]
    assert payload["ev_delta_vs_pass"] > 0
    assert "chi_meld_value" in payload["score_breakdown"]


def test_chi_2710_can_override_only_soft_sequences_when_it_creates_ting():
    state = {
        "hand": [
            "一", "柒", "六", "八", "叁", "七", "二", "捌", "三",
            "八", "五", "一", "一", "四", "叁", "壹", "陆",
        ],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "十",
        "chi_options": [
            {
                "option_id": "sim_chi_001",
                "labels": ["二", "七", "十"],
                "confidence": 1.0,
            }
        ],
        "strategy_existing_melds": [{"type": "wei", "labels": ["九", "九", "九"]}],
        "remaining_deck_count": 40,
    }

    decision = choose_action(state)
    chi_eval = _eval_for(decision, "CHI")

    assert decision.selected_action == "CHI"
    assert chi_eval.allowed
    assert chi_eval.debug_details["trusted_visible_response_override"] is True
    assert chi_eval.debug_details["ting_before"]["is_ting"] is False
    assert chi_eval.debug_details["ting_after"]["is_ting"] is True
    assert len(chi_eval.debug_details["ting_after"]["waiting_cards"]) == 4
    assert chi_eval.debug_details["consumption_impact"]["breaks_hard"] is False
    assert all(
        meld["type"] == "normal_sequence"
        for meld in chi_eval.debug_details["consumption_impact"]["breaks_soft_melds"]
    )


def test_chi_scores_misread_option_with_protocol_pending_card():
    state = {
        "hand": ["二", "七", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "十",
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "chi_options": [{"option_id": "chi_misread_2710", "labels": ["二", "七", "七"], "confidence": 1.0}],
    }

    decision = choose_action(state)
    chi_eval = _eval_for(decision, "CHI")

    assert decision.selected_action == "CHI"
    assert decision.selected_option_id == "chi_misread_2710"
    assert chi_eval.allowed
    assert chi_eval.debug_details["consumed_from_hand"] == ["二", "七"]
    assert chi_eval.debug_details["all_response_candidates"][0]["meld_type"] == "special_2710"
    assert chi_eval.debug_details["all_response_candidates"][0]["meld_cards"] == ["二", "七", "十"]
    assert chi_eval.debug_details["all_response_candidates"][0]["recognized_option_cards"] == ["二", "七", "七"]


def test_visible_response_buttons_infer_response_mode_without_protocol_actions():
    state = {
        "hand": ["壹", "叁", "二", "三", "四", "贰", "贰", "肆", "伍", "五", "六", "七", "柒", "八", "八", "捌", "捌", "玖", "十", "十"],
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "chi_options": [{"option_id": "chi_visible_001", "labels": ["九", "十"], "confidence": 1.0}],
    }

    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    actions, _ = generate_legal_actions(context, allocation, analyze_hand(context, allocation))

    assert context.legal_action_types == ["CHI", "PASS"]
    assert not any(action.type == "DISCARD" for action in actions)


def test_chi_rejects_visible_option_when_local_consumption_binding_is_missing():
    state = {
        "hand": ["壹", "叁", "二", "三", "四", "贰", "贰", "肆"],
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "chi_options": [{"option_id": "chi_visible_001", "labels": ["九", "十"], "confidence": 1.0}],
    }

    decision = choose_action(state)
    chi_eval = next(item for item in decision.action_evals if item.type == "CHI")

    assert decision.selected_action == "PASS"
    assert not chi_eval.allowed
    assert chi_eval.reject_reason == "chi_option_binding_missing"
    assert chi_eval.debug_details["selected_option_id"] == "chi_visible_001"
    assert chi_eval.debug_details["trusted_visible_response_override"] is False


def test_visible_chi_button_does_not_override_strategy_pass():
    state = {
        "hand": ["三", "五", "二", "九", "十"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "四",
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "chi_options": [{"option_id": "chi_345", "labels": ["三", "四", "五"], "confidence": 1.0}],
    }

    decision = choose_action(state)
    chi_eval = next(item for item in decision.action_evals if item.type == "CHI")

    assert decision.selected_action == "PASS"
    assert not chi_eval.allowed
    assert chi_eval.reject_reason == "chi_ev_not_enough"
    assert chi_eval.debug_details["trusted_visible_response_override"] is False


def test_chi_must_improve_existing_ting_outs():
    before = TingEstimate(
        shanten_like_distance=1,
        is_ting=True,
        waiting_cards=["一", "二", "三"],
        improving_cards=[],
        hu_cards=["一", "二", "三"],
        expected_xi_if_hu=15,
        expected_score_if_hu=30.0,
        notes=[],
    )
    same_outs = TingEstimate(
        shanten_like_distance=1,
        is_ting=True,
        waiting_cards=["一", "二", "三"],
        improving_cards=[],
        hu_cards=["一", "二", "三"],
        expected_xi_if_hu=15,
        expected_score_if_hu=30.0,
        notes=[],
    )
    more_outs = TingEstimate(
        shanten_like_distance=1,
        is_ting=True,
        waiting_cards=["一", "二", "三", "四"],
        improving_cards=[],
        hu_cards=["一", "二", "三", "四"],
        expected_xi_if_hu=15,
        expected_score_if_hu=30.0,
        notes=[],
    )

    reject, _reason = _response_ting_reject_reason("CHI", before, same_outs)
    allowed, _reason = _response_ting_reject_reason("CHI", before, more_outs)

    assert reject == "chi_does_not_improve_ting"
    assert allowed is None


def test_chi_visible_option_does_not_override_breaking_two_2710_sets():
    state = {
        "hand": ["二", "七", "十", "贰", "柒", "拾", "四"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "拾",
        "buttons": [{"name": "chi"}, {"name": "pass"}],
        "chi_options": [{"option_id": "chi_breaks_2710", "labels": ["拾", "拾", "十"], "confidence": 1.0}],
    }

    decision = choose_action(state)
    chi_eval = next(
        item
        for item in decision.action_evals
        if item.type == "CHI" and item.action.option_id == "chi_breaks_2710"
    )

    assert decision.selected_action == "PASS"
    assert not chi_eval.allowed
    assert chi_eval.debug_details["trusted_visible_response_override"] is False
    assert chi_eval.debug_details["consumption_impact"]["breaks_special_2710"] is True
    assert chi_eval.debug_details["breaks_melds"] == [["二", "七", "十"], ["贰", "柒", "拾"]]


def test_policy_decision_exports_selected_chi_option_cards():
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 1.0}],
    }

    payload = choose_action(state).to_dict()

    assert payload["selected_option_id"] == "chi_2710"
    assert payload["selected_option_cards"] == ["二", "七", "十"]
    assert payload["selected_action"]["option_cards"] == ["二", "七", "十"]


def test_context_keeps_compare_options_out_of_chi_legal_actions():
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}],
        "pending_card": "七",
        "option_details": [
            {"region_name": "compare_options", "labels": ["一", "二", "三"], "confidence": 1.0},
            {"region_name": "chi_options", "labels": ["二", "七", "十"], "confidence": 1.0},
        ],
    }
    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    actions, _ = generate_legal_actions(context, allocation, analyze_hand(context, allocation))

    chi_actions = [item for item in actions if item.type == "CHI"]

    assert len(chi_actions) == 1
    assert chi_actions[0].option_cards == ["二", "七", "十"]
    assert context.compare_options[0]["labels"] == ["一", "二", "三"]


def test_chi_passes_when_option_recognition_is_uncertain():
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 0.6}],
    }

    decision = choose_action(state)
    chi_eval = _eval_for(decision, "CHI")

    assert decision.selected_action == "PASS"
    assert not chi_eval.allowed
    assert chi_eval.reject_reason == "chi_option_uncertain"


def test_missing_visible_button_is_logged_as_rejected_action():
    state = {
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": [{"type": "CHI"}],
        "pending_card": "七",
        "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 1.0}],
        "buttons": [],
    }
    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)

    actions, rejected = generate_legal_actions(context, allocation, analysis)

    assert any(action.type == "PASS" for action in actions)
    assert any(action.type == "CHI" and action.reject_reason == "button_not_found" for action in rejected)


def test_peng_requires_pair_and_scores_against_pass_with_followup_discard():
    state = {
        "hand": ["五", "壹", "玖", "贰", "九", "捌", "六", "七", "拾", "玖", "壹"],
        "legal_actions": [{"type": "PENG"}],
        "pending_card": "壹",
        "strategy_existing_melds": [
            {"type": "normal_sequence", "labels": ["一", "二", "三"]},
            {"type": "normal_sequence", "labels": ["四", "五", "六"]},
            {"type": "normal_sequence", "labels": ["肆", "伍", "陆"]},
        ],
    }

    decision = choose_action(state)
    peng_eval = _eval_for(decision, "PENG")

    assert decision.selected_action == "PENG"
    assert peng_eval.allowed
    assert peng_eval.ev > peng_eval.debug_details["pass_ev"]
    assert peng_eval.debug_details["consumed_from_hand"] == ["壹", "壹"]
    assert peng_eval.debug_details["followup_discard"]["type"] == "DISCARD"
    payload = peng_eval.to_dict()
    assert payload["pass_ev"] == peng_eval.debug_details["pass_ev"]
    assert payload["peng_ev"] == peng_eval.debug_details["peng_ev"]
    assert payload["ev_delta_vs_pass"] > 0
    assert "peng_meld_value" in payload["score_breakdown"]


@pytest.mark.parametrize("pair_label", ["二", "贰"])
def test_peng_upgrades_pair_without_charging_pair_break_loss(pair_label):
    state = {
        "hand": [pair_label, pair_label, "四", "六", "九"],
        "legal_actions": [{"type": "PENG"}],
        "pending_card": pair_label,
    }

    decision = choose_action(state)
    peng_eval = _eval_for(decision, "PENG")
    impact = peng_eval.debug_details["consumption_impact"]

    assert decision.selected_action == "PENG"
    assert peng_eval.allowed
    assert peng_eval.score_gain == 120
    assert impact["breaks_pair"] is False
    assert impact["structure_loss"] == 0
    assert impact["breaks_melds"] == []
    assert [meld["type"] for meld in impact["preserved_melds"]] == ["pair"]


def test_peng_trusts_visible_button_when_local_pair_binding_is_missing():
    state = {
        "hand": ["二", "四", "六", "九"],
        "legal_actions": [{"type": "PENG"}, {"type": "PASS"}],
        "pending_card": "九",
        "buttons": [{"name": "peng"}, {"name": "pass"}],
    }

    decision = choose_action(state)
    peng_eval = _eval_for(decision, "PENG")

    assert decision.selected_action == "PENG"
    assert peng_eval.allowed
    assert peng_eval.reason == "peng_protocol_trusted_pair_unavailable_locally"
    assert peng_eval.debug_details["trusted_button"] is True
    assert peng_eval.debug_details["consumed_from_hand"] == []


def test_peng_does_not_break_soft_2710_when_response_ev_is_worse():
    state = {
        "hand": ["壹", "二", "贰", "贰", "三", "叁", "四", "肆", "五", "伍", "六", "柒", "八", "八", "捌", "捌", "玖", "十", "十", "七"],
        "legal_actions": [{"type": "PENG"}, {"type": "PASS"}],
        "pending_card": "十",
        "buttons": [{"name": "peng"}, {"name": "pass"}],
    }

    decision = choose_action(state)
    peng_eval = _eval_for(decision, "PENG")
    impact = peng_eval.debug_details["consumption_impact"]

    assert decision.selected_action == "PASS"
    assert not peng_eval.allowed
    assert peng_eval.reject_reason == "peng_ev_not_enough"
    assert impact["breaks_special_2710"] is True


def test_peng_should_not_break_double_sequence_556677():
    state = {
        "hand": ["五", "五", "六", "六", "七", "七"],
        "legal_actions": [{"type": "PENG"}, {"type": "PASS"}],
        "pending_card": "五",
    }

    decision = choose_action(state)
    peng_eval = _eval_for(decision, "PENG")
    impact = peng_eval.debug_details["consumption_impact"]

    assert decision.selected_action == "PASS"
    assert not peng_eval.allowed
    assert peng_eval.reject_reason == "peng_breaks_double_sequence"
    assert impact["breaks_double_sequence"] is True
    assert impact["breaks_melds"] == [["五", "六", "七"], ["五", "六", "七"]]
    assert "double sequence" in peng_eval.reason


def test_chi_should_not_break_existing_123_for_same_rank_triplet():
    state = {
        "hand": ["一", "二", "三", "叁"],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "pending_card": "三",
        "chi_options": [{"option_id": "chi_33big3", "labels": ["三", "三", "叁"], "confidence": 1.0}],
    }

    decision = choose_action(state)
    chi_eval = _eval_for(decision, "CHI")
    impact = chi_eval.debug_details["consumption_impact"]

    assert decision.selected_action == "PASS"
    assert not chi_eval.allowed
    assert chi_eval.reject_reason == "chi_breaks_existing_123"
    assert impact["breaks_special_123"] is True
    assert impact["breaks_melds"] == [["一", "二", "三"]]
    assert "一二三" in chi_eval.reason


def test_peng_rejects_when_pair_is_not_available():
    state = {
        "hand": ["二", "四", "六", "九"],
        "legal_actions": [{"type": "PENG"}],
        "pending_card": "二",
    }

    decision = choose_action(state)
    peng_eval = _eval_for(decision, "PENG")

    assert decision.selected_action == "PASS"
    assert not peng_eval.allowed
    assert peng_eval.reject_reason == "peng_pair_not_available"


@pytest.mark.parametrize("action_type", ["PAO", "TI", "MING_LONG", "AUTO_QUAD"])
def test_professional_brain_waits_for_auto_meld_actions(action_type):
    decision = choose_action(
        {
            "hand": ["一", "二", "三"],
            "legal_actions": [{"type": action_type}, {"type": "PASS"}],
        }
    )
    auto_eval = _eval_for(decision, "WAIT_AUTO_MELD")

    assert decision.selected_action == "WAIT_AUTO_MELD"
    assert decision.action == "wait_auto_meld"
    assert decision.candidate_stage == "wait_auto_meld"
    assert auto_eval.allowed
    assert auto_eval.debug_details["auto_meld_action"] == action_type


def test_legal_hu_has_priority_over_higher_scored_auto_ti():
    decision = choose_action(
        {
            "hand": COMPLETE_HAND,
            "legal_actions": [{"type": "HU"}, {"type": "TI"}],
            "buttons": [{"name": "hu"}],
        }
    )
    auto_eval = _eval_for(decision, "WAIT_AUTO_MELD")
    hu_eval = _eval_for(decision, "HU")

    assert auto_eval.ev > hu_eval.ev
    assert hu_eval.allowed
    assert decision.selected_action == "HU"


@pytest.mark.parametrize(
    ("action_type", "label", "expected_kind", "expected_xi"),
    [
        ("TI", "贰", "ti", 12),
        ("PAO", "二", "pao", 6),
        ("MING_LONG", "贰", "pao", 9),
        ("WEI", "二", "wei", 3),
    ],
)
def test_auto_meld_eval_records_xi_gain_cards_and_risk(action_type, label, expected_kind, expected_xi):
    decision = choose_action(
        {
            "hand": [label, label, label],
            "legal_actions": [{"type": action_type, "label": label}, {"type": "PASS"}],
            "remaining_deck_count": 10,
        }
    )
    auto_eval = _eval_for(decision, "WAIT_AUTO_MELD")

    assert auto_eval.allowed
    assert auto_eval.xi_gain == expected_xi
    assert auto_eval.score_gain > 0
    assert auto_eval.debug_details["auto_meld_kind"] == expected_kind
    assert auto_eval.debug_details["auto_meld_cards"]
    assert auto_eval.debug_details["xi_gain"] == expected_xi
    if action_type in {"PAO", "MING_LONG"}:
        assert auto_eval.opponent_gain_risk > 0


def test_hu_rejects_complete_partition_when_xi_is_below_minimum():
    context = build_decision_context(ZERO_XI_COMPLETE_HAND)
    allocation = allocate_hand_structures(context)
    hu = check_hu(context, allocation)

    assert not hu.can_hu
    assert hu.reject_reason == "xi_not_enough"
    assert hu.total_xi < hu.min_xi


def test_hu_accepts_complete_partition_when_xi_reaches_minimum():
    context = build_decision_context(COMPLETE_HAND)
    allocation = allocate_hand_structures(context)
    hu = check_hu(context, allocation)

    assert hu.can_hu
    assert hu.total_xi >= hu.min_xi
    assert {item["type"] for item in hu.partition} >= {"special_2710", "special_123", "exact_triplet"}


@pytest.mark.parametrize(
    "state,wildcard_enabled,players",
    [
        ({"hand": COMPLETE_HAND}, False, 2),
        ({"hand": ZERO_XI_COMPLETE_HAND}, False, 2),
        ({"hand": [*COMPLETE_HAND[:-1], "王"]}, True, 2),
        (
            {
                "hand": ["王", "王"],
                "strategy_existing_melds": [
                    {"type": "ti", "labels": ["三"] * 4},
                    {"type": "normal_sequence", "labels": ["一", "二", "三"]},
                    {"type": "normal_sequence", "labels": ["四", "五", "六"]},
                    {"type": "normal_sequence", "labels": ["七", "八", "九"]},
                    {"type": "special_123", "labels": ["壹", "贰", "叁"]},
                    {"type": "normal_sequence", "labels": ["肆", "伍", "陆"]},
                ],
            },
            True,
            2,
        ),
    ],
)
def test_count_grouping_ting_probe_matches_full_partition_result(
    state,
    wildcard_enabled,
    players,
):
    rules = rules_for_room(wildcard_enabled=wildcard_enabled, players=players)
    context = build_decision_context(state, rules=rules)
    full = check_hu(context, allocate_hand_structures(context), rules)

    projected = _grouping_ting_hu_probe(context, rules)

    assert projected == (full.can_hu, full.total_xi, full.score_now)


def test_hu_preserves_two_real_identical_existing_melds_across_mirrored_sources():
    existing_melds = [
        {"type": "wei", "labels": ["七", "七", "七"]},
        {"type": "special_2710", "labels": ["贰", "柒", "拾"]},
        {"type": "special_2710", "labels": ["贰", "柒", "拾"]},
    ]
    state = {
        "hand": ["伍", "五", "四", "叁", "三", "肆", "五", "肆", "三", "伍", "叁", "四"],
        "strategy_existing_melds": deepcopy(existing_melds),
        "meld_groups": {"my_melds": deepcopy(existing_melds)},
        "memory": {"my_meld_groups": deepcopy(existing_melds)},
        "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
    }

    context = build_decision_context(state)
    hu = check_hu(context, allocate_hand_structures(context))
    decision = choose_action(state)

    assert len(context.existing_melds) == 3
    assert sum(meld.type == "special_2710" for meld in context.existing_melds) == 2
    assert hu.can_hu
    assert hu.total_xi == 15
    assert len(hu.partition) == 7
    assert decision.selected_action == "HU"


def test_hu_accepts_wild_pair_as_the_seventh_group_after_a_quad():
    state = {
        "hand": ["王", "王"],
        "strategy_existing_melds": [
            {"type": "ti", "labels": ["三", "三", "三", "三"]},
            {"type": "normal_sequence", "labels": ["一", "二", "三"]},
            {"type": "normal_sequence", "labels": ["四", "五", "六"]},
            {"type": "normal_sequence", "labels": ["七", "八", "九"]},
            {"type": "special_123", "labels": ["壹", "贰", "叁"]},
            {"type": "normal_sequence", "labels": ["肆", "伍", "陆"]},
        ],
    }
    context = build_decision_context(state)

    hu = check_hu(context, allocate_hand_structures(context))

    assert hu.can_hu
    assert len(hu.partition) == 7
    assert sum(item["type"] == "pair" for item in hu.partition) == 1
    assert any(item["labels"] == ["王", "王"] for item in hu.partition)


def test_hu_does_not_promote_wild_pair_without_a_quad():
    state = {
        "hand": ["王", "王"],
        "strategy_existing_melds": [
            {"type": "normal_sequence", "labels": ["一", "二", "三"]},
            {"type": "normal_sequence", "labels": ["四", "五", "六"]},
            {"type": "normal_sequence", "labels": ["七", "八", "九"]},
            {"type": "special_123", "labels": ["壹", "贰", "叁"]},
            {"type": "normal_sequence", "labels": ["肆", "伍", "陆"]},
            {"type": "normal_sequence", "labels": ["柒", "捌", "玖"]},
        ],
    }
    context = build_decision_context(state)

    hu = check_hu(context, allocate_hand_structures(context))

    assert not hu.can_hu


def test_hu_accepts_pair_after_multiple_four_card_melds():
    state = {
        "hand": ["伍", "伍", "伍", "王", "王"],
        "strategy_existing_melds": [
            {"type": "ti", "labels": ["三", "三", "三", "三"]},
            {"type": "pao", "labels": ["四", "四", "四", "四"]},
            {"type": "normal_sequence", "labels": ["一", "二", "三"]},
            {"type": "normal_sequence", "labels": ["七", "八", "九"]},
            {"type": "special_123", "labels": ["壹", "贰", "叁"]},
        ],
    }
    context = build_decision_context(state)

    hu = check_hu(context, allocate_hand_structures(context))

    assert hu.can_hu
    assert hu.total_xi == 27
    assert len(hu.partition) == 7
    assert sum(item["type"] == "pair" for item in hu.partition) == 1


def test_hu_accepts_ordinary_pair_after_two_existing_quads():
    state = {
        "hand": [
            "五", "十", "伍", "五", "六", "陆",
            "拾", "十", "六", "拾", "拾",
        ],
        "strategy_existing_melds": [
            {"type": "ti", "labels": ["贰"] * 4},
            {
                "type": "mixed_same_rank_triplet",
                "labels": ["九", "玖", "玖"],
            },
            {"type": "ti", "labels": ["肆"] * 4},
        ],
        "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
    }
    rules = rules_for_room(wildcard_enabled=False, players=2)
    context = build_decision_context(state, rules=rules)

    hu = check_hu(context, allocate_hand_structures(context), rules)
    decision = choose_action(state, rules=rules)

    assert hu.can_hu
    assert hu.total_xi == 30
    assert sum(item["type"] == "pair" for item in hu.partition) == 1
    assert decision.selected_action == "HU"


def test_hu_partition_prioritizes_xi_over_structure_with_wildcard():
    state = {
        "hand": [
            "八", "二", "王", "五", "捌", "柒", "十", "捌",
            "四", "六", "拾", "拾", "贰", "七", "拾",
        ],
        "strategy_existing_melds": [
            {
                "type": "mixed_same_rank_triplet",
                "labels": ["三", "三", "叁"],
            },
            {
                "type": "normal_sequence",
                "labels": ["叁", "肆", "伍"],
            },
        ],
        "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
    }
    rules = rules_for_room(wildcard_enabled=True, players=2)
    context = build_decision_context(state, rules=rules)

    hu = check_hu(context, allocate_hand_structures(context), rules)
    decision = choose_action(state, rules=rules)

    assert hu.can_hu
    assert hu.total_xi == 15
    assert hu.decision == "hu_now"
    assert decision.selected_action == "HU"


def test_hu_partition_search_chooses_the_required_group_count_over_higher_xi_quads():
    state = {
        "hand": ["一", "二", "三"] * 4,
        "strategy_existing_melds": [
            {"type": "wei", "labels": ["壹", "壹", "壹"]},
            {"type": "wei", "labels": ["叁", "叁", "叁"]},
            {"type": "wei", "labels": ["伍", "伍", "伍"]},
        ],
    }
    context = build_decision_context(state)
    hu = check_hu(context, allocate_hand_structures(context))

    assert hu.can_hu
    assert len(hu.partition) == 7
    assert sum(item["type"] == "exact_quad" for item in hu.partition) == 0
    assert sum(item["type"] == "wei" for item in hu.partition) == 3
    assert sum(item["type"] in {"special_123", "exact_triplet"} for item in hu.partition) == 4


def test_hu_result_reports_red_black_bonus():
    black_hu_hand = [
        "壹", "壹", "壹",
        "一", "一", "一",
        "三", "三", "叁",
        "四", "四", "肆",
        "五", "五", "伍",
        "六", "六", "陆",
        "八", "八", "捌",
    ]
    context = build_decision_context(black_hu_hand)
    allocation = allocate_hand_structures(context)
    hu = check_hu(context, allocation)

    assert hu.can_hu
    assert hu.red_black_bonus > 0
    assert "red_black_bonus" in hu.reason


def test_hu_partition_search_can_use_wildcard_without_trusting_greedy_allocation():
    wildcard_hand = list(COMPLETE_HAND)
    wildcard_hand[wildcard_hand.index("拾")] = "王"
    context = build_decision_context(wildcard_hand)
    allocation = allocate_hand_structures(context)
    hu = check_hu(context, allocation)

    assert allocation.locked_melds[0].type == "wildcard_meld"
    assert hu.can_hu
    assert hu.wildcard_mapping
    assert "拾" in set(hu.wildcard_mapping.values())
    assert {item["type"] for item in hu.partition} >= {"wildcard_meld", "special_123", "exact_triplet"}
    assert hu.score_now > 0
    assert hu.continue_ev >= hu.score_now
    assert hu.decision == "hu_now"


def test_hu_wildcard_can_replace_a_label_that_exists_in_another_group():
    hand = [
        "六", "三", "王", "四", "贰", "四", "肆",
        "七", "叁", "肆", "八", "三", "拾", "八",
        "贰", "五", "五", "柒", "六", "三", "五",
    ]
    rules = rules_for_room(wildcard_enabled=True, players=3)
    context = build_decision_context(
        {"hand": hand, "legal_actions": [{"type": "HU"}, {"type": "PASS"}]},
        rules=rules,
    )

    hu = check_hu(context, allocate_hand_structures(context), rules)
    decision = choose_action(context.state, rules=rules)

    assert hu.can_hu
    assert hu.total_xi == 12
    assert decision.selected_action == "HU"
    assert "七" in hu.wildcard_mapping.values()


def test_response_hu_includes_pending_card_exactly_once():
    state = {
        "hand": [
            "王", "玖", "七", "五", "四", "壹", "捌",
            "贰", "九", "八", "七", "八", "王", "柒",
            "贰", "拾", "十", "叁", "三", "二",
        ],
        "pending_card": "十",
        "pending_source_seat": 0,
        "legal_actions": [
            {"type": "HU"},
            {"type": "CHI"},
            {"type": "PASS"},
        ],
    }
    rules = rules_for_room(wildcard_enabled=True, players=3)
    context = build_decision_context(state, rules=rules)

    hu = check_hu(context, allocate_hand_structures(context), rules)
    decision = choose_action(state, rules=rules)

    assert len(context.normalized_hand) == 20
    assert hu.can_hu
    assert hu.total_xi == 18
    assert decision.selected_action == "HU"
    hu_eval = _eval_for(decision, "HU")
    assert hu_eval.allowed
    assert hu_eval.debug_details["hu_result"]["total_xi"] == 18


def test_hu_all_wildcard_group_is_covered_by_partition_search():
    state = {
        "hand": ["王", "王", "王"],
        "strategy_existing_melds": [
            {"type": "wei", "labels": ["贰", "贰", "贰"]},
            {"type": "wei", "labels": ["肆", "肆", "肆"]},
            {"type": "normal_sequence", "labels": ["一", "二", "三"]},
            {"type": "normal_sequence", "labels": ["四", "五", "六"]},
            {"type": "normal_sequence", "labels": ["七", "八", "九"]},
            {"type": "normal_sequence", "labels": ["壹", "贰", "叁"]},
        ],
        "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
    }
    rules = rules_for_room(wildcard_enabled=True, players=3)
    context = build_decision_context(state, rules=rules)

    hu = check_hu(context, allocate_hand_structures(context), rules)

    assert hu.can_hu
    assert hu.total_xi >= 9
    assert len(hu.wildcard_mapping) == 3


def test_hu_partition_is_invariant_to_same_label_instance_ids():
    wildcard_hand = list(COMPLETE_HAND)
    wildcard_hand[wildcard_hand.index("拾")] = "王"
    first = build_decision_context(
        {
            "hand_details": [
                {"name": label, "card_id": f"first_{index:02d}"}
                for index, label in enumerate(wildcard_hand)
            ]
        }
    )
    second = build_decision_context(
        {
            "hand_details": [
                {"name": label, "card_id": f"second_{len(wildcard_hand) - index:02d}"}
                for index, label in enumerate(wildcard_hand)
            ]
        }
    )

    first_hu = check_hu(first, allocate_hand_structures(first))
    second_hu = check_hu(second, allocate_hand_structures(second))
    first_groups = sorted(
        (item["type"], tuple(sorted(item["labels"])))
        for item in first_hu.partition
    )
    second_groups = sorted(
        (item["type"], tuple(sorted(item["labels"])))
        for item in second_hu.partition
    )

    assert first_hu.can_hu == second_hu.can_hu
    assert first_hu.total_xi == second_hu.total_xi
    assert first_groups == second_groups
    assert sorted(first_hu.wildcard_mapping.values()) == sorted(
        second_hu.wildcard_mapping.values()
    )


def test_hu_strategy_does_not_reject_on_unvalidated_continue_heuristic():
    state = {
        "hand": WILD_CONTINUE_HAND,
        "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
        "remaining_deck_count": 30,
    }

    decision = choose_action(state)
    hu_eval = _eval_for(decision, "HU")
    hu_result = decision.context_snapshot["hand_analysis"]["hu_result"]

    assert decision.selected_action == "HU"
    assert hu_result["decision"] == "hu_now"
    assert hu_result["continue_ev"] > hu_result["score_now"]
    assert hu_eval.allowed
    assert hu_eval.reject_reason is None
    assert "未经整局配对反事实证明" in hu_result["reason"]


def test_hu_strategy_late_game_hu_now_even_when_continue_ev_is_high():
    state = {
        "hand": WILD_CONTINUE_HAND,
        "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
        "remaining_deck_count": 8,
    }

    decision = choose_action(state)
    hu_eval = _eval_for(decision, "HU")
    hu_result = decision.context_snapshot["hand_analysis"]["hu_result"]

    assert decision.selected_action == "HU"
    assert hu_eval.allowed
    assert hu_result["decision"] == "hu_now"
    assert "后盘有胡就胡" in hu_result["reason"]


def test_hu_strategy_opponent_danger_hu_now_even_when_continue_ev_is_high():
    state = {
        "hand": WILD_CONTINUE_HAND,
        "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
        "remaining_deck_count": 30,
        "memory": {
            "opponent_meld_groups": [
                ["一", "一", "一"],
                ["二", "二", "二"],
                ["三", "三", "三"],
            ],
            "opponent_discards": [],
        },
    }

    decision = choose_action(state)
    hu_eval = _eval_for(decision, "HU")
    hu_result = decision.context_snapshot["hand_analysis"]["hu_result"]

    assert decision.selected_action == "HU"
    assert hu_eval.allowed
    assert hu_result["decision"] == "hu_now"
    assert "对手危险时有胡就胡" in hu_result["reason"]


def test_must_hu_any_overrides_continue_ev():
    rules = {
        "rules": {"min_xi": 9, "must_hu": "any"},
        "xi": {
            "ti": {"small": 9, "big": 12},
            "pao": {"small": 6, "big": 9},
            "wei": {"small": 3, "big": 6},
            "peng": {"small": 1, "big": 3},
            "special_123": {"small": 3, "big": 6},
            "special_2710": {"small": 3, "big": 6},
        },
    }
    state = {
        "hand": WILD_CONTINUE_HAND,
        "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
        "remaining_deck_count": 30,
    }

    decision = choose_action(state, rules=rules)
    hu_result = decision.context_snapshot["hand_analysis"]["hu_result"]

    assert decision.selected_action == "HU"


def test_visible_hu_button_is_trusted_when_local_partition_is_incomplete():
    decision = choose_action(
        {
            "hand": ["王", "王", "五", "六", "陆", "陆", "柒", "七", "七", "八", "捌", "九", "玖", "玖", "玖", "拾", "拾"],
            "legal_actions": [{"type": "HU"}, {"type": "PASS"}],
            "buttons": [{"name": "hu", "confidence": 0.93}, {"name": "pass", "confidence": 1.0}],
        }
    )

    hu_eval = _eval_for(decision, "HU")

    assert decision.selected_action == "HU"
    assert hu_eval.allowed
    assert "visible_hu_button_trusted" in hu_eval.reason
