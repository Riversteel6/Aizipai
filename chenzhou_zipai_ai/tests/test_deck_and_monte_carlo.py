"""Tests for hidden-information helpers."""

import yaml

from ai.monte_carlo import simulate_discards
from ai.opponent_model import infer_opponent_profile, infer_opponent_profiles
from ai.pro_brain import build_decision_context, evaluate_discard_danger
from ai.risk_model import danger_score, opponent_meld_risk, visible_labels_from_memory
from ai.self_play import apply_weight_adjustments, run_self_play_calibration
from engine.deck import remaining_counts, visible_counts


def test_remaining_counts_subtract_visible_cards():
    memory = {
        "my_discards": ["一"],
        "opponent_discards": ["二", "二"],
        "my_meld_groups": [["三", "暗", "暗"]],
        "opponent_meld_groups": [],
    }

    visible = visible_counts(hand=["王", "一"], memory=memory)
    remaining = remaining_counts(hand=["王", "一"], memory=memory)

    assert visible["一"] == 2
    assert visible["二"] == 2
    assert visible["三"] == 1
    assert remaining["一"] == 2
    assert remaining["二"] == 2


def test_monte_carlo_returns_candidate_statistics():
    results = simulate_discards(["二", "七", "十", "九"], simulations=20, seed=1)

    assert results
    assert {item.label for item in results} == {"二", "七", "十", "九"}
    assert all(item.simulations == 20 for item in results)


def test_self_play_calibration_writes_long_term_ev_report(tmp_path):
    report = run_self_play_calibration(
        count=5,
        hand_size=8,
        simulations=5,
        seed=2,
        output_dir=tmp_path,
    )

    assert report["ok"]
    assert report["count"] == 5
    assert len(report["rows"]) == 5
    assert all("monte_carlo_best_label" in row for row in report["rows"])
    assert all("long_term_gap" in row for row in report["rows"])
    assert "calibration_adjustments" in report
    assert (tmp_path / "self_play_calibration.json").exists()
    assert (tmp_path / "self_play_calibration.md").exists()
    assert (tmp_path / "self_play_weight_adjustments.json").exists()


def test_self_play_weight_adjustments_can_write_tuned_config(tmp_path):
    config_path = tmp_path / "rules.yaml"
    tuned_path = tmp_path / "rules.tuned.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": {
                    "ting_improvement": 120,
                    "weak_potential_discard_bonus": 120,
                    "orphan_discard_bonus": 300,
                    "hard_protected_break_penalty": 10000,
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = {
        "count": 4,
        "label_mismatches": 2,
        "avg_long_term_gap": -80,
        "worst_long_term_gap": -220,
        "hard_violations": [],
    }

    result = apply_weight_adjustments(config_path=config_path, output_path=tuned_path, report=report)
    tuned = yaml.safe_load(tuned_path.read_text(encoding="utf-8"))

    assert result["applied_updates"]
    assert tuned["weights"]["ting_improvement"] > 120
    assert tuned["weights"]["weak_potential_discard_bonus"] > 120
    assert tuned["weights"]["orphan_discard_bonus"] > 300
    assert tuned["calibration"]["last_self_play_adjustment"]["metrics"]["label_mismatches"] == 2


def test_opponent_model_infers_fast_meld_profile():
    memory = {
        "opponent_discards": ["一", "三"],
        "opponent_meld_groups": [["二", "暗", "暗"], ["七", "暗", "暗"], ["十", "暗", "暗"]],
    }

    assert infer_opponent_profile(memory)["profile"] == "fast_meld"


def test_professional_risk_model_uses_memory_and_opponent_profile():
    memory = {
        "opponent_discards": ["一", "三"],
        "opponent_meld_groups": [["二", "暗", "暗"], ["七", "暗", "暗"], ["十", "暗", "暗"]],
    }
    context = build_decision_context(
        {
            "hand": ["二", "九"],
            "memory": memory,
            "remaining_deck_count": 10,
        }
    )
    red_card = next(card for card in context.card_instances if card.label == "二")
    plain_card = next(card for card in context.card_instances if card.label == "九")

    red_danger = evaluate_discard_danger(context, red_card)
    plain_danger = evaluate_discard_danger(context, plain_card)

    assert red_danger.danger_score > plain_danger.danger_score
    assert any("对手落地多" in reason for reason in red_danger.reasons)


def test_opponent_meld_risk_marks_possible_pao_and_late_bonus():
    memory = {"opponent_meld_groups": [["二", "二", "二"]]}

    score, reasons = opponent_meld_risk("二", memory, remaining_deck_count=8)

    assert score >= 500
    assert any("跑" in reason or "明龙" in reason for reason in reasons)
    assert any("后盘" in reason for reason in reasons)


def test_risk_model_uses_current_decision_rules_instead_of_reloading_defaults():
    score, reasons = danger_score(
        "二",
        remaining_deck_count=8,
        rules={
            "ai_weights": {"late_red_danger_penalty": -999},
            "weights": {"red_card_late_danger": 12},
        },
    )

    assert score == 77
    assert any("后盘红牌" in reason for reason in reasons)
    assert any("后盘未见牌" in reason for reason in reasons)


def test_risk_model_reads_multi_opponent_memory():
    memory = {
        "opponents": [
            {"seat": "left", "discards": ["一"], "meld_groups": [["二", "二", "二"]]},
            {"seat": "right", "discards": ["七"], "meld_groups": [["十", "十"]]},
        ]
    }

    visible = visible_labels_from_memory(memory)
    profile = infer_opponent_profile(memory)
    score, reasons = opponent_meld_risk("二", memory, remaining_deck_count=8)

    assert visible.count("二") == 3
    assert profile["opponent_count"] == 2
    assert profile["meld_count"] == 2
    assert score >= 500
    assert any("跑" in reason or "明龙" in reason for reason in reasons)


def test_seat_separated_memory_takes_precedence_over_legacy_aggregates():
    memory = {
        "opponent_discards": ["一"],
        "opponent_meld_groups": [["二", "二", "二"]],
        "opponents": [
            {
                "seat": 1,
                "discards": ["一"],
                "meld_groups": [["二", "二", "二"]],
            }
        ],
    }

    visible = visible_labels_from_memory(memory)
    profile = infer_opponent_profile(memory)

    assert visible.count("一") == 1
    assert visible.count("二") == 3
    assert profile["meld_count"] == 1


def test_opponent_profiles_keep_three_player_seats_independent():
    memory = {
        "opponents": [
            {
                "seat": 1,
                "relative_offset": 1,
                "is_next_seat": True,
                "discards": ["一"],
                "meld_groups": [["一", "三", "四"], ["五", "六", "八"]],
            },
            {
                "seat": 2,
                "relative_offset": 2,
                "is_next_seat": False,
                "discards": ["壹"],
                "meld_groups": [["壹", "叁", "肆"], ["伍", "陆", "捌"]],
            },
        ]
    }

    profiles = infer_opponent_profiles(memory)

    assert [profile["seat"] for profile in profiles] == [1, 2]
    assert [profile["profile"] for profile in profiles] == ["balanced", "balanced"]
    assert infer_opponent_profile(memory)["profile"] == "fast_meld"


def test_seat_aware_danger_does_not_invent_one_fast_three_player_opponent():
    groups_left = [["一", "三", "四"], ["五", "六", "八"]]
    groups_right = [["壹", "叁", "肆"], ["伍", "陆", "捌"]]
    aggregate_memory = {
        "opponent_meld_groups": groups_left + groups_right,
        "opponent_discards": ["一", "壹"],
    }
    seat_memory = {
        **aggregate_memory,
        "seat_aware_danger": True,
        "opponents": [
            {
                "seat": 1,
                "relative_offset": 1,
                "is_next_seat": True,
                "discards": ["一"],
                "meld_groups": groups_left,
            },
            {
                "seat": 2,
                "relative_offset": 2,
                "is_next_seat": False,
                "discards": ["壹"],
                "meld_groups": groups_right,
            },
        ],
    }

    aggregate_context = build_decision_context(
        {"hand": ["九", "玖"], "memory": aggregate_memory, "remaining_deck_count": 10}
    )
    seat_context = build_decision_context(
        {"hand": ["九", "玖"], "memory": seat_memory, "remaining_deck_count": 10}
    )
    aggregate_card = next(card for card in aggregate_context.card_instances if card.label == "九")
    seat_card = next(card for card in seat_context.card_instances if card.label == "九")

    aggregate_danger = evaluate_discard_danger(aggregate_context, aggregate_card)
    seat_danger = evaluate_discard_danger(seat_context, seat_card)

    assert aggregate_danger.danger_score == seat_danger.danger_score + 40
    assert any("落地多" in reason for reason in aggregate_danger.reasons)
    assert not any("落地多" in reason for reason in seat_danger.reasons)


def test_seat_aware_danger_is_identical_for_one_opponent():
    groups = [["一", "三", "四"], ["五", "六", "八"], ["壹", "叁", "肆"]]
    aggregate_memory = {
        "opponent_meld_groups": groups,
        "opponent_discards": ["一"],
    }
    seat_memory = {
        **aggregate_memory,
        "seat_aware_danger": True,
        "opponents": [
            {
                "seat": 1,
                "relative_offset": 1,
                "is_next_seat": True,
                "discards": ["一"],
                "meld_groups": groups,
            }
        ],
    }

    scores = []
    for memory in (aggregate_memory, seat_memory):
        context = build_decision_context(
            {"hand": ["九", "玖"], "memory": memory, "remaining_deck_count": 10}
        )
        card = next(item for item in context.card_instances if item.label == "九")
        scores.append(evaluate_discard_danger(context, card).danger_score)

    assert scores[0] == scores[1]


def test_professional_risk_model_applies_opponent_meld_risk_to_ev():
    memory = {"opponent_meld_groups": [["二", "二", "二"]]}
    context = build_decision_context(
        {
            "hand": ["二", "九"],
            "memory": memory,
            "remaining_deck_count": 8,
        }
    )
    risky_card = next(card for card in context.card_instances if card.label == "二")
    plain_card = next(card for card in context.card_instances if card.label == "九")

    risky_danger = evaluate_discard_danger(context, risky_card)
    plain_danger = evaluate_discard_danger(context, plain_card)

    assert risky_danger.risk_type == "opponent_pao_or_minglong"
    assert risky_danger.danger_score > plain_danger.danger_score
    assert any("跑" in reason or "明龙" in reason for reason in risky_danger.reasons)
