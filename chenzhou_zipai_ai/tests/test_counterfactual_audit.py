"""Decision-level paired counterfactual audit tests."""

from ai.counterfactual_audit import (
    CounterfactualAuditConfig,
    _confidently_better_advantages,
    _information_set_seed,
    audit_discard_trace,
    audit_hu_trace,
    audit_response_trace,
)
from ai.ismcts import (
    PairedAdvantageStats,
    PairedCandidateOutcome,
    PairedWorldOutcome,
    RootCandidateStats,
    RootISMCTSPolicy,
    RootResponseCandidateStats,
    RootResponseSearchResult,
    RootSearchResult,
    information_set_seed,
    public_view_from_dict,
)
from engine.deck import full_deck_counts
from engine.rules import rules_for_room


def test_information_set_seed_is_stable_across_trace_positions_and_choices():
    rules = rules_for_room(wildcard_enabled=True, players=3)
    trace = {
        "sequence": 4,
        "turn": 2,
        "phase": "discard",
        "selected_key": "DISCARD:二",
        "public_state": {
            "seat": 1,
            "hand": ["二", "三", "王"],
            "all_melds": [[], [], []],
            "discards": [[], ["一"], []],
            "remaining_counts": [["一", 3], ["二", 3], ["王", 3]],
            "stock_count": 9,
            "hand_sizes": [3, 3, 3],
            "pending_card": None,
            "pending_source_seat": None,
        },
    }
    moved_trace = {
        **trace,
        "sequence": 99,
        "turn": 40,
        "selected_key": "DISCARD:三",
    }
    changed_state = {
        **trace,
        "public_state": {
            **trace["public_state"],
            "stock_count": 8,
        },
    }

    seed = _information_set_seed(20260728, trace, rules)

    assert seed == _information_set_seed(20260728, moved_trace, rules)
    assert seed != _information_set_seed(20260728, changed_state, rules)


def test_confidently_suboptimal_accepts_any_significant_alternative():
    empirical_best = PairedAdvantageStats(
        candidate_key="五",
        preferred_key="九",
        samples=256,
        mean_delta=0.18,
        sample_stddev=1.0,
        standard_error=0.06,
        lower_confidence_bound=-0.001,
        upper_confidence_bound=0.36,
        positive_samples=68,
        tied_samples=147,
        negative_samples=41,
    )
    significant_runner_up = PairedAdvantageStats(
        candidate_key="八",
        preferred_key="九",
        samples=256,
        mean_delta=0.17,
        sample_stddev=0.8,
        standard_error=0.05,
        lower_confidence_bound=0.03,
        upper_confidence_bound=0.31,
        positive_samples=50,
        tied_samples=185,
        negative_samples=21,
    )

    better = _confidently_better_advantages(
        (empirical_best, significant_runner_up),
        min_confidence_pairs=256,
    )

    assert [item.candidate_key for item in better] == ["八"]


def test_discard_audit_covers_every_legal_action_and_reports_regret(monkeypatch):
    observed = {}

    def fake_search(self, view, **kwargs):
        observed["candidate_labels"] = tuple(kwargs["candidate_labels"])
        observed["preferred_label"] = kwargs["preferred_label"]
        observed["max_candidates"] = self.config.max_candidates
        observed["record_paired_worlds"] = self.config.record_paired_worlds
        observed["base_seed"] = self.config.seed
        worlds = tuple(
            PairedWorldOutcome(
                world_index=index,
                world_fingerprint=f"world-{index}",
                rollout_seed=100 + index,
                opponent_policy="baseline",
                outcomes=(
                    PairedCandidateOutcome("二", 0.0, None, 0.0, 0, 5, "stock_exhausted"),
                    PairedCandidateOutcome("三", 0.5, 0, 1.0, 9, 5, "self_draw"),
                ),
            )
            for index in range(2)
        )
        return RootSearchResult(
            selected_label="三",
            used_search=True,
            reason="root_discard_ismcts_completed",
            simulations=4,
            elapsed_ms=10.0,
            candidates=(
                RootCandidateStats("三", 2, 1.0, 0.5, 1.0, 20.0),
                RootCandidateStats("二", 2, 0.0, 0.0, 0.0, 10.0),
            ),
            paired_determinizations=2,
            empirical_best_label="三",
            paired_worlds=worlds,
        )

    monkeypatch.setattr(RootISMCTSPolicy, "search_discard", fake_search)
    trace = {
        "sequence": 7,
        "turn": 3,
        "phase": "discard",
        "seat": 0,
        "policy": "professional_brain_v2",
        "selected_key": "DISCARD:二",
        "public_state": {
            "seat": 0,
            "hand": ["二", "三"],
            "all_melds": [[], []],
            "discards": [[], []],
            "remaining_counts": [],
            "stock_count": 39,
            "hand_sizes": [2, 2],
            "pending_card": None,
            "pending_source_seat": None,
        },
        "legal_actions": [
            {"key": "DISCARD:二", "type": "DISCARD", "label": "二"},
            {"key": "DISCARD:三", "type": "DISCARD", "label": "三"},
        ],
    }

    audit = audit_discard_trace(
        trace,
        rules=rules_for_room(wildcard_enabled=False, players=2),
        config=CounterfactualAuditConfig(paired_worlds=2),
    )

    assert observed == {
        "candidate_labels": ("二", "三"),
        "preferred_label": "二",
        "max_candidates": 2,
        "record_paired_worlds": True,
        "base_seed": 20260728,
    }
    assert audit["audit_seed"] == information_set_seed(
        public_view_from_dict(trace["public_state"]),
        20260728,
    )
    assert audit["complete"]
    assert audit["best_key"] == "DISCARD:三"
    assert audit["selected_expected_reward"] == 0.0
    assert audit["best_expected_reward"] == 0.5
    assert audit["counterfactual_regret"] == 0.5
    assert not audit["selected_is_empirical_best"]
    assert not audit["confidently_suboptimal"]
    assert len(audit["paired_worlds"]) == 2


def test_discard_audit_can_filter_to_an_explicit_legal_pair(monkeypatch):
    observed = {}

    def fake_search(self, view, **kwargs):
        observed["candidate_labels"] = tuple(kwargs["candidate_labels"])
        worlds = (
            PairedWorldOutcome(
                world_index=0,
                world_fingerprint="pair-world",
                rollout_seed=100,
                opponent_policy="baseline",
                outcomes=(
                    PairedCandidateOutcome(
                        "二", 0.0, None, 0.0, 0, 1, "stock_exhausted"
                    ),
                    PairedCandidateOutcome(
                        "三", 0.5, 0, 1.0, 9, 1, "self_draw"
                    ),
                ),
            ),
        )
        return RootSearchResult(
            selected_label="三",
            used_search=True,
            reason="root_discard_ismcts_completed",
            simulations=2,
            elapsed_ms=5.0,
            candidates=(
                RootCandidateStats("三", 1, 0.5, 0.5, 1.0, 20.0),
                RootCandidateStats("二", 1, 0.0, 0.0, 0.0, 10.0),
            ),
            paired_determinizations=1,
            empirical_best_label="三",
            paired_worlds=worlds,
        )

    monkeypatch.setattr(RootISMCTSPolicy, "search_discard", fake_search)
    trace = {
        "sequence": 7,
        "turn": 3,
        "phase": "discard",
        "seat": 0,
        "policy": "professional_brain_v2",
        "selected_key": "DISCARD:二",
        "public_state": {
            "seat": 0,
            "hand": ["二", "三", "四"],
            "all_melds": [[], []],
            "discards": [[], []],
            "remaining_counts": [],
            "stock_count": 39,
            "hand_sizes": [3, 3],
            "pending_card": None,
            "pending_source_seat": None,
        },
        "legal_actions": [
            {"key": "DISCARD:二", "type": "DISCARD", "label": "二"},
            {"key": "DISCARD:三", "type": "DISCARD", "label": "三"},
            {"key": "DISCARD:四", "type": "DISCARD", "label": "四"},
        ],
    }

    audit = audit_discard_trace(
        trace,
        rules=rules_for_room(wildcard_enabled=False, players=2),
        config=CounterfactualAuditConfig(paired_worlds=1),
        candidate_labels=("二", "三"),
    )

    assert observed["candidate_labels"] == ("二", "三")
    assert audit["available_legal_action_count"] == 3
    assert audit["legal_action_count"] == 2
    assert audit["audited_action_count"] == 2
    assert audit["complete"]


def test_response_audit_covers_pass_peng_and_every_chi_plan(monkeypatch):
    observed = {}

    def fake_search(self, view, **kwargs):
        candidates = tuple(kwargs["candidates"])
        observed["keys"] = tuple(candidate.key for candidate in candidates)
        observed["types"] = tuple(candidate.action_type for candidate in candidates)
        observed["preferred_key"] = kwargs["preferred_key"]
        observed["max_candidates"] = self.config.max_candidates
        observed["base_seed"] = self.config.seed
        stats = tuple(
            RootResponseCandidateStats(
                candidate=candidate,
                visits=2,
                reward_sum=(
                    1.5 if candidate.key == "CHI:plan-a" else 0.0
                ),
                average_reward=(
                    0.75 if candidate.key == "CHI:plan-a" else 0.0
                ),
                win_rate=(
                    1.0 if candidate.key == "CHI:plan-a" else 0.0
                ),
            )
            for candidate in candidates
        )
        worlds = tuple(
            PairedWorldOutcome(
                world_index=index,
                world_fingerprint=f"response-world-{index}",
                rollout_seed=200 + index,
                opponent_policy="baseline",
                outcomes=tuple(
                    PairedCandidateOutcome(
                        candidate.key,
                        0.75 if candidate.key == "CHI:plan-a" else 0.0,
                        0 if candidate.key == "CHI:plan-a" else None,
                        1.0 if candidate.key == "CHI:plan-a" else 0.0,
                        9 if candidate.key == "CHI:plan-a" else 0,
                        5,
                        "self_draw" if candidate.key == "CHI:plan-a" else "stock_exhausted",
                    )
                    for candidate in candidates
                ),
            )
            for index in range(2)
        )
        return RootResponseSearchResult(
            selected_key="CHI:plan-a",
            used_search=True,
            reason="root_response_ismcts_completed",
            simulations=6,
            elapsed_ms=12.0,
            candidates=stats,
            paired_determinizations=2,
            empirical_best_key="CHI:plan-a",
            paired_worlds=worlds,
        )

    monkeypatch.setattr(RootISMCTSPolicy, "search_response", fake_search)
    trace = {
        "sequence": 8,
        "turn": 3,
        "phase": "response_root",
        "seat": 0,
        "policy": "professional_brain_v2",
        "selected_key": "PENG:五",
        "public_state": {
            "seat": 0,
            "hand": ["五", "五", "四", "六"],
            "all_melds": [[], []],
            "discards": [[], []],
            "remaining_counts": [],
            "stock_count": 39,
            "hand_sizes": [4, 2],
            "pending_card": "五",
            "pending_source_seat": 1,
        },
        "legal_actions": [
            {
                "key": "PASS",
                "type": "PASS",
                "consumed_from_hand": [],
                "meld_groups": [],
            },
            {
                "key": "PENG:五",
                "type": "PENG",
                "consumed_from_hand": ["五", "五"],
                "meld_groups": [["五", "五", "五"]],
            },
            {
                "key": "CHI:plan-a",
                "type": "CHI",
                "option_id": "plan-a",
                "consumed_from_hand": ["四", "六"],
                "meld_groups": [["四", "五", "六"]],
            },
        ],
    }

    audit = audit_response_trace(
        trace,
        rules=rules_for_room(wildcard_enabled=False, players=2),
        config=CounterfactualAuditConfig(paired_worlds=2),
    )

    assert observed == {
        "keys": ("PASS", "PENG:五", "CHI:plan-a"),
        "types": ("PASS", "PENG", "CHI"),
        "preferred_key": "PENG:五",
        "max_candidates": 3,
        "base_seed": 20260728,
    }
    assert audit["audit_seed"] == information_set_seed(
        public_view_from_dict(trace["public_state"]),
        20260728,
    )
    assert audit["complete"]
    assert audit["best_key"] == "CHI:plan-a"
    assert audit["selected_expected_reward"] == 0.0
    assert audit["best_expected_reward"] == 0.75
    assert audit["counterfactual_regret"] == 0.75
    assert not audit["selected_is_empirical_best"]
    assert not audit["confidently_suboptimal"]
    assert len(audit["paired_worlds"]) == 2


def test_hu_audit_compares_accepting_against_continuing_in_paired_worlds():
    hand = [
        "贰", "柒", "拾",
        "壹", "贰", "叁",
        "九", "九", "九",
        "一", "二", "三",
        "四", "五", "六",
        "七", "八", "九",
        "肆", "伍", "陆",
    ]
    rules = rules_for_room(wildcard_enabled=False, players=2)
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(hand)
    trace = {
        "sequence": 9,
        "turn": 4,
        "phase": "post_action_hu",
        "seat": 0,
        "policy": "professional_brain_v2",
        "selected_key": "PASS",
        "public_state": {
            "seat": 0,
            "hand": hand,
            "own_melds": [],
            "all_melds": [[], []],
            "discards": [[], []],
            "remaining_counts": [
                [label, amount]
                for label, amount in sorted((+remaining).items())
            ],
            "stock_count": sum(remaining.values()),
            "hand_sizes": [len(hand), 0],
            "pending_card": None,
            "pending_source_seat": None,
        },
        "legal_actions": [
            {"key": "PASS", "type": "PASS"},
            {"key": "HU", "type": "HU"},
        ],
    }

    audit = audit_hu_trace(
        trace,
        rules=rules,
        config=CounterfactualAuditConfig(
            paired_worlds=2,
            min_confidence_pairs=2,
            time_budget_ms=5_000,
            rollout_max_turns=4,
        ),
    )

    assert audit["complete"]
    assert audit["legal_action_count"] == 2
    assert audit["audited_action_count"] == 2
    assert audit["completed_paired_worlds"] == 2
    assert audit["best_key"] in {"PASS", "HU"}
    assert audit["counterfactual_regret"] >= 0.0
    assert audit["selected_expected_score"] is not None
    assert audit["selected_expected_signed_xi"] is not None
    assert audit["best_expected_score"] is not None
    assert audit["best_expected_signed_xi"] is not None
    for candidate in audit["candidate_stats"]:
        assert candidate["wins"] + candidate["losses"] + candidate["draws"] == 2
        assert candidate["win_rate"] + candidate["loss_rate"] + candidate["draw_rate"] == 1.0
        assert isinstance(candidate["mean_outcome_score"], float)
        assert isinstance(candidate["mean_signed_xi"], float)
    assert {
        outcome["candidate_key"]
        for world in audit["paired_worlds"]
        for outcome in world["outcomes"]
    } == {"PASS", "HU"}
    hu_outcomes = [
        outcome
        for world in audit["paired_worlds"]
        for outcome in world["outcomes"]
        if outcome["candidate_key"] == "HU"
    ]
    assert all(outcome["winner"] == 0 for outcome in hu_outcomes)
    assert all(outcome["reason"] == "post_action_hu" for outcome in hu_outcomes)


def test_self_hu_audit_replays_the_already_drawn_card_before_auto_melds():
    rules = rules_for_room(wildcard_enabled=False, players=2)
    hand = ["十", "十", "十", "八", "八", "八"]
    melds = [
        {"type": "special_123", "labels": ["壹", "贰", "叁"]},
        {"type": "special_2710", "labels": ["贰", "柒", "拾"]},
        {"type": "peng", "labels": ["五", "五", "五"]},
        {"type": "wei", "labels": ["玖", "玖", "玖"]},
        {
            "type": "mixed_same_rank_triplet",
            "labels": ["一", "壹", "壹"],
        },
    ]
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(hand)
    for meld in melds:
        remaining.subtract(meld["labels"])
    trace = {
        "sequence": 26,
        "turn": 17,
        "phase": "self_hu",
        "seat": 0,
        "policy": "professional_brain_v2",
        "selected_key": "HU",
        "metadata": {"drawn_card": "八"},
        "public_state": {
            "seat": 0,
            "hand": hand,
            "own_melds": melds,
            "all_melds": [melds, []],
            "discards": [[], []],
            "remaining_counts": [
                [label, amount]
                for label, amount in sorted((+remaining).items())
            ],
            "stock_count": sum(remaining.values()),
            "hand_sizes": [len(hand), 0],
            "pending_card": None,
            "pending_source_seat": None,
        },
        "legal_actions": [
            {"key": "PASS", "type": "PASS"},
            {"key": "HU", "type": "HU"},
        ],
    }

    audit = audit_hu_trace(
        trace,
        rules=rules,
        config=CounterfactualAuditConfig(
            paired_worlds=2,
            min_confidence_pairs=2,
            time_budget_ms=5_000,
            rollout_max_turns=4,
        ),
    )

    assert audit["complete"]
    assert audit["completed_paired_worlds"] == 2
    assert audit["search_health"]["rollout_invariant_violations"] == 0
