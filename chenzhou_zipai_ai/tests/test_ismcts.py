"""Information-set determinization and bounded root-search tests."""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

import ai.ismcts as ismcts_module
from ai.decision_objective import OBJECTIVE_VERSION, ObjectiveMode
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from ai.full_game_simulator import (
    BaselinePolicy,
    ChiPlan,
    FullGameSimulator,
    GameResult,
    ProfessionalBrainSimulationPolicy,
    PublicView,
    SimMeld,
    SimPlayer,
)
from ai.ismcts import (
    DiscardResponseRiskResult,
    ProfessionalConfidenceRootCandidatePolicy,
    ProfessionalConfidenceRootShadowPolicy,
    ProfessionalDiscardResponseRiskCandidatePolicy,
    ProfessionalDiscardResponseRiskShadowPolicy,
    ProfessionalDiscardShadowSimulationPolicy,
    ProfessionalFullActionTeacherPolicy,
    ProfessionalProgressiveAllActionCandidatePolicy,
    ProfessionalResponseShadowSimulationPolicy,
    ProgressiveRootISMCTSPolicy,
    PairedAdvantageStats,
    RootCandidateStats,
    RootISMCTSConfig,
    RootISMCTSPolicy,
    RootResponseCandidateStats,
    RootSearchResult,
    RootResponseCandidate,
    RootResponseSearchResult,
    _ForcedFirstDiscardPolicy,
    _ForcedRootResponsePolicy,
    _confidence_top_set,
    _minimum_regret_key,
    _progressive_discard_budget_fallback,
    build_response_candidates,
    _paired_advantage_stats,
    _root_reward,
    _sample_legal_hidden_assignment,
    determinize_public_view,
    public_view_from_dict,
    public_view_to_dict,
    shortlist_response_candidates,
)
from engine.deck import expanded_deck, full_deck_counts
from engine.rules import rules_for_room


def test_complete_hand_below_min_xi_is_a_loss_for_stalled_root_and_win_for_opponent():
    result = GameResult(
        seed=20260727,
        wildcard_enabled=True,
        winner=None,
        dealer=0,
        turns=10,
        reason="complete_hand_below_min_xi",
        score=0.0,
        total_xi=6,
        action_counts={},
        violations=(),
        stalled_seat=1,
    )

    assert _root_reward(result, root_seat=1) == 0.0
    assert _root_reward(result, root_seat=0) == 1.0


def test_progressive_response_threshold_applies_to_both_confirmation_paths():
    search = ProgressiveRootISMCTSPolicy(
        response_minimum_confident_advantage=0.25
    )

    assert (
        search.response_binary_confirmation.config
        .minimum_confident_advantage
        == 0.25
    )
    assert (
        search.response_confirmation.config.minimum_confident_advantage
        == 0.25
    )


def test_confidence_top_set_eliminates_only_statistically_dominated_arm():
    candidates = (
        RootCandidateStats("A", 100, 80.0, 0.8, 0.8, 0.0, wins=80),
        RootCandidateStats("B", 100, 40.0, 0.4, 0.4, 0.0, wins=40),
        RootCandidateStats("C", 100, 78.0, 0.78, 0.78, 0.0, wins=78),
    )

    assert _confidence_top_set(candidates) == {"A", "C"}


def test_deadline_fallback_selects_minimum_regret_arm_not_default_anchor():
    candidates = (
        RootCandidateStats("A", 100, 80.0, 0.8, 0.8, 0.0, wins=80),
        RootCandidateStats("B", 100, 40.0, 0.4, 0.4, 0.0, wins=40),
        RootCandidateStats("C", 100, 78.0, 0.78, 0.78, 0.0, wins=78),
    )
    coverage = RootSearchResult(
        selected_label="B",
        used_search=True,
        reason="coverage_complete",
        simulations=300,
        elapsed_ms=1.0,
        candidates=candidates,
        empirical_best_label="A",
    )

    assert _minimum_regret_key(candidates, preferred_key="B") == "A"
    fallback = _progressive_discard_budget_fallback(
        coverage,
        preferred_label="B",
    )
    assert fallback.selected_label == "A"
    assert fallback.confidence_override
    assert fallback.reason.endswith("overall_budget_minimum_regret")

def test_progressive_discard_search_covers_all_then_refines_shortlist(
    monkeypatch,
):
    search = ProgressiveRootISMCTSPolicy(refinement_candidates=3)
    labels = ["一", "二", "三", "四"]
    calls = []
    coverage = RootSearchResult(
        selected_label="四",
        used_search=True,
        reason="coverage",
        simulations=4,
        elapsed_ms=100.0,
        candidates=tuple(
            RootCandidateStats(
                label=label,
                visits=1,
                reward_sum=reward,
                average_reward=reward,
                win_rate=0.0,
                heuristic_value=0.0,
            )
            for label, reward in zip(labels, (0.0, 0.2, 0.4, 0.6))
        ),
        paired_determinizations=1,
    )
    refinement = RootSearchResult(
        selected_label="四",
        used_search=True,
        reason="refinement",
        simulations=24,
        elapsed_ms=200.0,
        candidates=(
            coverage.candidates[0],
            coverage.candidates[3],
            coverage.candidates[2],
        ),
        paired_determinizations=8,
        empirical_best_label="四",
        confidence_override=True,
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key="四",
                preferred_key="一",
                samples=8,
                mean_delta=0.8,
                sample_stddev=0.2,
                standard_error=0.1,
                lower_confidence_bound=0.5,
                upper_confidence_bound=1.1,
                positive_samples=8,
                tied_samples=0,
                negative_samples=0,
                familywise_comparisons=2,
            ),
        ),
    )
    selection = replace(
        refinement,
        reason="selection",
        simulations=60,
        elapsed_ms=250.0,
        paired_determinizations=20,
        confidence_override=False,
    )
    confirmation = RootSearchResult(
        selected_label="四",
        used_search=True,
        reason="confirmation",
        simulations=48,
        elapsed_ms=300.0,
        candidates=(
            coverage.candidates[0],
            coverage.candidates[3],
            coverage.candidates[2],
        ),
        paired_determinizations=24,
        empirical_best_label="四",
        confidence_override=True,
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key="四",
                preferred_key="一",
                samples=24,
                mean_delta=0.7,
                sample_stddev=0.2,
                standard_error=0.05,
                lower_confidence_bound=0.5,
                upper_confidence_bound=0.9,
                positive_samples=24,
                tied_samples=0,
                negative_samples=0,
            ),
        ),
    )

    def cover(*_args, **kwargs):
        calls.append(("coverage", list(kwargs["candidate_labels"])))
        return coverage

    def refine(*_args, **kwargs):
        calls.append(("refinement", list(kwargs["candidate_labels"])))
        return refinement

    def select(*_args, **kwargs):
        calls.append(("selection", list(kwargs["candidate_labels"])))
        return selection

    def confirm(*_args, **kwargs):
        calls.append(("confirmation", list(kwargs["candidate_labels"])))
        return confirmation

    monkeypatch.setattr(search.coverage, "search_discard", cover)
    monkeypatch.setattr(search.refinement, "search_discard", refine)
    monkeypatch.setattr(search.discard_selection, "search_discard", select)
    monkeypatch.setattr(search.confirmation, "search_discard", confirm)

    result = search.search_discard(
        SimpleNamespace(hand=labels),
        rules={},
        candidate_labels=labels,
        preferred_label="一",
    )

    assert calls[0] == ("coverage", labels)
    assert calls[1] == ("refinement", ["一", "四", "三"])
    assert calls[2] == ("selection", ["一", "四", "三"])
    assert calls[3] == ("confirmation", ["一", "四", "三"])
    assert [item.label for item in result.candidates] == labels
    assert result.selected_label == "四"
    assert result.confidence_override
    assert result.simulations == 136


def test_progressive_discard_reuses_priors_and_confirms_without_duplicate_selection(
    monkeypatch,
):
    search = ProgressiveRootISMCTSPolicy(
        refinement_candidates=3,
        discard_confirmation_alternatives=1,
        direct_discard_confirmation_from_refinement=True,
        reuse_candidate_priors_as_coverage=True,
    )
    labels = ["一", "二", "三", "四"]
    priors = {"一": 0.1, "二": 0.2, "三": 0.3, "四": 0.4}
    refinement_stats = tuple(
        RootCandidateStats(
            label=label,
            visits=8,
            reward_sum=reward * 8,
            average_reward=reward,
            win_rate=reward,
            heuristic_value=priors[label],
        )
        for label, reward in (("一", 0.2), ("四", 0.8), ("三", 0.5))
    )
    refinement = RootSearchResult(
        selected_label="一",
        used_search=True,
        reason="refinement",
        simulations=24,
        elapsed_ms=10.0,
        candidates=refinement_stats,
        paired_determinizations=8,
        empirical_best_label="四",
    )
    confirmation = RootSearchResult(
        selected_label="四",
        used_search=True,
        reason="confirmation",
        simulations=192,
        elapsed_ms=20.0,
        candidates=(refinement_stats[0], refinement_stats[1]),
        paired_determinizations=96,
        empirical_best_label="四",
        confidence_override=True,
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key="四",
                preferred_key="一",
                samples=96,
                mean_delta=0.6,
                sample_stddev=0.1,
                standard_error=0.02,
                lower_confidence_bound=0.5,
                upper_confidence_bound=0.7,
                positive_samples=90,
                tied_samples=0,
                negative_samples=6,
            ),
        ),
    )
    observed = {}
    monkeypatch.setattr(
        search.coverage,
        "search_discard",
        lambda *_args, **_kwargs: pytest.fail("production priors are coverage"),
    )

    def refine(*_args, **kwargs):
        observed["refinement"] = list(kwargs["candidate_labels"])
        return refinement

    def confirm(*_args, **kwargs):
        observed["confirmation"] = list(kwargs["candidate_labels"])
        return confirmation

    monkeypatch.setattr(search.refinement, "search_discard", refine)
    monkeypatch.setattr(
        search.discard_selection,
        "search_discard",
        lambda *_args, **_kwargs: pytest.fail("duplicate selection must not run"),
    )
    monkeypatch.setattr(search.confirmation, "search_discard", confirm)

    result = search.search_discard(
        SimpleNamespace(hand=labels),
        rules={},
        candidate_labels=labels,
        candidate_priors=priors,
        preferred_label="一",
    )

    assert observed == {
        "refinement": ["一", "四", "三"],
        "confirmation": ["一", "四"],
    }
    assert "production_prior_full_coverage" in result.reason
    assert len(result.candidates) == 4
    assert result.selected_label == "四"
    assert result.confidence_override


def test_progressive_discard_budget_fallback_keeps_preferred(monkeypatch):
    search = ProgressiveRootISMCTSPolicy(refinement_candidates=3)
    labels = ["一", "二", "三"]
    coverage = RootSearchResult(
        selected_label="三",
        used_search=True,
        reason="coverage",
        simulations=3,
        elapsed_ms=25.0,
        candidates=tuple(
            RootCandidateStats(
                label=label,
                visits=1,
                reward_sum=float(index),
                average_reward=float(index),
                win_rate=0.0,
                heuristic_value=0.0,
            )
            for index, label in enumerate(labels)
        ),
        paired_determinizations=1,
        empirical_best_label="三",
    )
    monkeypatch.setattr(
        search.coverage,
        "search_discard",
        lambda *_args, **_kwargs: coverage,
    )
    monkeypatch.setattr(
        search.refinement,
        "search_discard",
        lambda *_args, **_kwargs: pytest.fail(
            "refinement must not start after the shared deadline"
        ),
    )
    monkeypatch.setattr(
        ismcts_module.time,
        "perf_counter",
        lambda: 11.0,
    )

    result = search.search_discard(
        SimpleNamespace(hand=labels),
        rules={},
        candidate_labels=labels,
        preferred_label="一",
        absolute_deadline=10.0,
    )

    assert result.used_search
    assert result.selected_label == "一"
    assert not result.confidence_override
    assert result.reason.endswith("overall_budget_minimum_regret")
    assert result.simulations == 3
    assert {item.visits for item in result.candidates} == {1}


def test_external_deadline_is_hard_even_for_complete_first_batch():
    view, rules = _initial_public_view(20260730, players=2)
    labels = list(dict.fromkeys(view.hand))
    search = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=30_000,
            max_iterations=len(labels) * 24,
            max_candidates=len(labels),
            complete_first_paired_batch=True,
        ),
        rollout_policy_factories=(BaselinePolicy,),
    )

    result = search.search_discard(
        view,
        rules=rules,
        candidate_labels=labels,
        candidate_priors={
            label: float(len(labels) - index)
            for index, label in enumerate(labels)
        },
        force_search=True,
        paired_candidates=True,
        preferred_label=labels[0],
        absolute_deadline=0.0,
    )

    assert not result.used_search
    assert result.selected_label == labels[0]
    assert result.simulations == 0
    assert result.paired_determinizations == 0


def test_external_deadline_allows_mandatory_first_pair_past_local_budget(
    monkeypatch,
):
    view, rules = _initial_public_view(20260730, players=2)
    labels = list(dict.fromkeys(view.hand))[:2]
    observed_deadlines = []

    def fake_play(self, **kwargs):
        observed_deadlines.append(kwargs["deadline"])
        time.sleep(0.003)
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=0,
            turns=1,
            reason="stock_exhausted",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_play)
    search = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=1,
            max_iterations=2,
            max_candidates=2,
            complete_first_paired_batch=True,
        )
    )
    outer_deadline = time.perf_counter() + 1.0

    result = search.search_discard(
        view,
        rules=rules,
        candidate_labels=labels,
        candidate_priors={labels[0]: 2.0, labels[1]: 1.0},
        force_search=True,
        paired_candidates=True,
        preferred_label=labels[0],
        absolute_deadline=outer_deadline,
    )

    assert result.used_search
    assert result.paired_determinizations == 1
    assert {candidate.visits for candidate in result.candidates} == {1}
    assert observed_deadlines == [outer_deadline, outer_deadline]


def test_progressive_discard_shortlist_protects_prior_nominee(monkeypatch):
    search = ProgressiveRootISMCTSPolicy(refinement_candidates=3)
    labels = ["一", "二", "三", "四"]
    coverage = RootSearchResult(
        selected_label="二",
        used_search=True,
        reason="coverage",
        simulations=4,
        elapsed_ms=10.0,
        candidates=tuple(
            RootCandidateStats(
                label=label,
                visits=1,
                reward_sum=reward,
                average_reward=reward,
                win_rate=0.0,
                heuristic_value=prior,
            )
            for label, reward, prior in zip(
                labels,
                (0.0, 1.0, 0.9, -1.0),
                (0.0, 0.0, 0.0, 10.0),
            )
        ),
        paired_determinizations=1,
    )
    observed = {}

    monkeypatch.setattr(
        search.coverage,
        "search_discard",
        lambda *_args, **_kwargs: coverage,
    )

    def refine(*_args, **kwargs):
        observed["labels"] = list(kwargs["candidate_labels"])
        return RootSearchResult(
            selected_label="一",
            used_search=True,
            reason="refinement",
            simulations=3,
            elapsed_ms=10.0,
            candidates=tuple(
                item
                for item in coverage.candidates
                if item.label in kwargs["candidate_labels"]
            ),
            paired_determinizations=1,
            empirical_best_label="一",
        )

    monkeypatch.setattr(search.refinement, "search_discard", refine)

    search.search_discard(
        SimpleNamespace(hand=labels),
        rules={},
        candidate_labels=labels,
        preferred_label="一",
    )

    assert observed["labels"] == ["一", "二", "四"]


def test_progressive_confirmation_rejects_tentative_false_positive(
    monkeypatch,
):
    search = ProgressiveRootISMCTSPolicy(refinement_candidates=2)
    labels = ["二", "三"]
    stats = tuple(
        RootCandidateStats(
            label=label,
            visits=1,
            reward_sum=0.0,
            average_reward=0.0,
            win_rate=0.0,
            heuristic_value=0.0,
        )
        for label in labels
    )
    coverage = RootSearchResult(
        selected_label="三",
        used_search=True,
        reason="coverage",
        simulations=2,
        elapsed_ms=10.0,
        candidates=stats,
        paired_determinizations=1,
    )
    refinement = RootSearchResult(
        selected_label="三",
        used_search=True,
        reason="tentative_override",
        simulations=16,
        elapsed_ms=20.0,
        candidates=stats,
        paired_determinizations=8,
        empirical_best_label="三",
        confidence_override=True,
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key="三",
                preferred_key="二",
                samples=8,
                mean_delta=0.8,
                sample_stddev=0.2,
                standard_error=0.1,
                lower_confidence_bound=0.5,
                upper_confidence_bound=1.1,
                positive_samples=8,
                tied_samples=0,
                negative_samples=0,
            ),
        ),
    )
    confirmation = RootSearchResult(
        selected_label="二",
        used_search=True,
        reason="confirmation_kept_preferred",
        simulations=48,
        elapsed_ms=30.0,
        candidates=stats,
        paired_determinizations=24,
        empirical_best_label="二",
        confidence_override=False,
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key="三",
                preferred_key="二",
                samples=24,
                mean_delta=-0.8,
                sample_stddev=0.2,
                standard_error=0.05,
                lower_confidence_bound=-1.0,
                upper_confidence_bound=-0.6,
                positive_samples=0,
                tied_samples=0,
                negative_samples=24,
            ),
        ),
    )
    monkeypatch.setattr(
        search.coverage,
        "search_discard",
        lambda *_args, **_kwargs: coverage,
    )
    monkeypatch.setattr(
        search.refinement,
        "search_discard",
        lambda *_args, **_kwargs: refinement,
    )
    monkeypatch.setattr(
        search.confirmation,
        "search_discard",
        lambda *_args, **_kwargs: confirmation,
    )

    result = search.search_discard(
        SimpleNamespace(hand=labels),
        rules={},
        candidate_labels=labels,
        preferred_label="二",
    )

    assert result.used_search
    assert result.selected_label == "二"
    assert not result.confidence_override
    assert result.simulations == 66
    assert result.reason.endswith("confirmation_kept_preferred")
    assert result.paired_advantages[0].samples == 24


def test_progressive_discard_selection_keeps_second_alternative_for_confirmation(
    monkeypatch,
):
    search = ProgressiveRootISMCTSPolicy(refinement_candidates=3)
    labels = ["一", "二", "三"]
    coverage_stats = tuple(
        RootCandidateStats(
            label=label,
            visits=1,
            reward_sum=reward,
            average_reward=reward,
            win_rate=0.0,
            heuristic_value=0.0,
        )
        for label, reward in zip(labels, (0.0, 0.8, 0.6))
    )
    coverage = RootSearchResult(
        selected_label="二",
        used_search=True,
        reason="coverage",
        simulations=3,
        elapsed_ms=10.0,
        candidates=coverage_stats,
        paired_determinizations=1,
    )
    refinement = RootSearchResult(
        selected_label="一",
        used_search=True,
        reason="refinement",
        simulations=24,
        elapsed_ms=20.0,
        candidates=coverage_stats,
        paired_determinizations=8,
        empirical_best_label="二",
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key="二",
                preferred_key="一",
                samples=8,
                mean_delta=0.8,
                sample_stddev=0.1,
                standard_error=0.04,
                lower_confidence_bound=0.6,
                upper_confidence_bound=1.0,
                positive_samples=8,
                tied_samples=0,
                negative_samples=0,
            ),
            PairedAdvantageStats(
                candidate_key="三",
                preferred_key="一",
                samples=8,
                mean_delta=0.6,
                sample_stddev=0.1,
                standard_error=0.04,
                lower_confidence_bound=0.4,
                upper_confidence_bound=0.8,
                positive_samples=8,
                tied_samples=0,
                negative_samples=0,
            ),
        ),
    )
    selection_stats = tuple(
        replace(
            item,
            reward_sum=reward * 20,
            average_reward=reward,
            visits=20,
        )
        for item, reward in zip(coverage_stats, (0.0, 0.7, 0.9))
    )
    selection = RootSearchResult(
        selected_label="三",
        used_search=True,
        reason="selection",
        simulations=60,
        elapsed_ms=25.0,
        candidates=selection_stats,
        paired_determinizations=20,
        empirical_best_label="三",
    )
    confirmation = RootSearchResult(
        selected_label="三",
        used_search=True,
        reason="confirmation",
        simulations=72,
        elapsed_ms=30.0,
        candidates=selection_stats,
        paired_determinizations=24,
        empirical_best_label="三",
        confidence_override=True,
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key="三",
                preferred_key="一",
                samples=24,
                mean_delta=0.9,
                sample_stddev=0.1,
                standard_error=0.02,
                lower_confidence_bound=0.7,
                upper_confidence_bound=1.1,
                positive_samples=24,
                tied_samples=0,
                negative_samples=0,
            ),
        ),
    )
    calls = []
    monkeypatch.setattr(
        search.coverage,
        "search_discard",
        lambda *_args, **_kwargs: coverage,
    )
    monkeypatch.setattr(
        search.refinement,
        "search_discard",
        lambda *_args, **_kwargs: refinement,
    )

    def select(*_args, **kwargs):
        calls.append(("selection", list(kwargs["candidate_labels"])))
        return selection

    def confirm(*_args, **kwargs):
        calls.append(("confirmation", list(kwargs["candidate_labels"])))
        return confirmation

    monkeypatch.setattr(
        search.discard_selection,
        "search_discard",
        select,
    )
    monkeypatch.setattr(
        search.discard_confirmation,
        "search_discard",
        confirm,
    )

    result = search.search_discard(
        SimpleNamespace(hand=labels),
        rules={},
        candidate_labels=labels,
        preferred_label="一",
    )

    assert calls == [
        ("selection", ["一", "二", "三"]),
        ("confirmation", ["一", "三", "二"]),
    ]
    assert result.selected_label == "三"
    assert result.confidence_override
    assert [item.candidate_key for item in result.paired_advantages] == ["三"]
    assert all(item.samples == 24 for item in result.paired_advantages)


def test_progressive_response_search_keeps_all_candidates_in_result(
    monkeypatch,
):
    search = ProgressiveRootISMCTSPolicy(refinement_candidates=3)
    candidates = [
        RootResponseCandidate("PASS", "PASS", 0.0),
        RootResponseCandidate("PENG:五", "PENG", 1.0),
        RootResponseCandidate("CHI:test", "CHI", 2.0),
    ]
    coverage_stats = tuple(
        RootResponseCandidateStats(
            candidate=candidate,
            visits=1,
            reward_sum=reward,
            average_reward=reward,
            win_rate=0.0,
        )
        for candidate, reward in zip(candidates, (0.0, 0.2, 0.4))
    )
    coverage = RootResponseSearchResult(
        selected_key="CHI:test",
        used_search=True,
        reason="coverage",
        simulations=3,
        elapsed_ms=100.0,
        candidates=coverage_stats,
        paired_determinizations=1,
    )
    refinement = RootResponseSearchResult(
        selected_key="CHI:test",
        used_search=True,
        reason="selection",
        simulations=159,
        elapsed_ms=200.0,
        candidates=coverage_stats,
        paired_determinizations=53,
        empirical_best_key="CHI:test",
        confidence_override=False,
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key="CHI:test",
                preferred_key="PASS",
                samples=8,
                mean_delta=0.8,
                sample_stddev=0.2,
                standard_error=0.1,
                lower_confidence_bound=0.5,
                upper_confidence_bound=1.1,
                positive_samples=8,
                tied_samples=0,
                negative_samples=0,
            ),
        ),
    )
    activation = replace(
        refinement,
        reason="refinement",
        simulations=24,
        paired_determinizations=8,
    )
    confirmation = RootResponseSearchResult(
        selected_key="CHI:test",
        used_search=True,
        reason="confirmation",
        simulations=48,
        elapsed_ms=300.0,
        candidates=(coverage_stats[0], coverage_stats[2]),
        paired_determinizations=24,
        empirical_best_key="CHI:test",
        confidence_override=True,
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key="CHI:test",
                preferred_key="PASS",
                samples=24,
                mean_delta=0.7,
                sample_stddev=0.2,
                standard_error=0.05,
                lower_confidence_bound=0.5,
                upper_confidence_bound=0.9,
                positive_samples=24,
                tied_samples=0,
                negative_samples=0,
            ),
        ),
    )
    calls = []

    def cover(*_args, **kwargs):
        calls.append(("coverage", [item.key for item in kwargs["candidates"]]))
        return coverage

    def refine(*_args, **kwargs):
        calls.append(("selection", [item.key for item in kwargs["candidates"]]))
        return refinement

    def confirm(*_args, **kwargs):
        calls.append(("confirmation", [item.key for item in kwargs["candidates"]]))
        return confirmation

    monkeypatch.setattr(search.coverage, "search_response", cover)
    monkeypatch.setattr(
        search.refinement,
        "search_response",
        lambda *_args, **kwargs: (
            calls.append(
                (
                    "refinement",
                    [item.key for item in kwargs["candidates"]],
                )
            )
            or activation
        ),
    )
    monkeypatch.setattr(search.response_selection, "search_response", refine)
    monkeypatch.setattr(
        search.response_confirmation,
        "search_response",
        confirm,
    )

    result = search.search_response(
        SimpleNamespace(),
        rules={},
        candidates=candidates,
        preferred_key="PASS",
    )

    assert calls[0] == ("coverage", ["PASS", "PENG:五", "CHI:test"])
    assert calls[1] == (
        "refinement",
        ["PASS", "CHI:test", "PENG:五"],
    )
    assert calls[2] == (
        "selection",
        ["PASS", "CHI:test", "PENG:五"],
    )
    assert calls[3] == ("confirmation", ["PASS", "CHI:test"])
    assert [item.candidate.key for item in result.candidates] == [
        "PASS",
        "PENG:五",
        "CHI:test",
    ]
    assert result.selected_key == "CHI:test"
    assert result.confidence_override


def test_progressive_binary_response_skips_selection(monkeypatch):
    search = ProgressiveRootISMCTSPolicy()
    candidates = [
        RootResponseCandidate("PASS", "PASS", 0.0),
        RootResponseCandidate("PENG:五", "PENG", 1.0),
    ]
    stats = tuple(
        RootResponseCandidateStats(
            candidate=candidate,
            visits=1,
            reward_sum=float(index),
            average_reward=float(index),
            win_rate=0.0,
        )
        for index, candidate in enumerate(candidates)
    )
    coverage = RootResponseSearchResult(
        selected_key="PENG:五",
        used_search=True,
        reason="coverage",
        simulations=2,
        elapsed_ms=10.0,
        candidates=stats,
        paired_determinizations=1,
    )
    confirmation = RootResponseSearchResult(
        selected_key="PENG:五",
        used_search=True,
        reason="binary_confirmation",
        simulations=256,
        elapsed_ms=100.0,
        candidates=stats,
        paired_determinizations=128,
        empirical_best_key="PENG:五",
        confidence_override=True,
    )
    refinement = RootResponseSearchResult(
        selected_key="PENG:五",
        used_search=True,
        reason="refinement",
        simulations=16,
        elapsed_ms=20.0,
        candidates=stats,
        paired_determinizations=8,
        empirical_best_key="PENG:五",
    )
    observed = {}
    monkeypatch.setattr(
        search.coverage,
        "search_response",
        lambda *_args, **_kwargs: coverage,
    )
    monkeypatch.setattr(
        search.refinement,
        "search_response",
        lambda *_args, **_kwargs: refinement,
    )

    def confirm(*_args, **kwargs):
        observed["keys"] = [
            candidate.key for candidate in kwargs["candidates"]
        ]
        return confirmation

    monkeypatch.setattr(
        search.response_binary_confirmation,
        "search_response",
        confirm,
    )
    monkeypatch.setattr(
        search.response_selection,
        "search_response",
        lambda *_args, **_kwargs: pytest.fail(
            "binary response must skip selection"
        ),
    )

    result = search.search_response(
        SimpleNamespace(),
        rules={},
        candidates=candidates,
        preferred_key="PASS",
    )

    assert observed["keys"] == ["PASS", "PENG:五"]
    assert search.last_refinement is refinement
    assert search.last_confirmation is confirmation
    assert result.selected_key == "PENG:五"
    assert result.confidence_override
    assert result.simulations == 274


def test_progressive_hu_confirmation_merges_without_response_selection(
    monkeypatch,
):
    search = ProgressiveRootISMCTSPolicy()
    candidates = (
        RootResponseCandidate("HU", "HU", 1.0),
        RootResponseCandidate("PASS", "PASS", 0.0),
    )
    stats = tuple(
        RootResponseCandidateStats(
            candidate=candidate,
            visits=8,
            reward_sum=float(index),
            average_reward=float(index),
            win_rate=0.0,
        )
        for index, candidate in enumerate(candidates)
    )
    refinement = RootResponseSearchResult(
        selected_key="PASS",
        used_search=True,
        reason="hu_refinement",
        simulations=16,
        elapsed_ms=20.0,
        candidates=stats,
        paired_determinizations=8,
        empirical_best_key="PASS",
    )
    confirmation = RootResponseSearchResult(
        selected_key="PASS",
        used_search=True,
        reason="hu_confirmation",
        simulations=168,
        elapsed_ms=100.0,
        candidates=stats,
        paired_determinizations=84,
        empirical_best_key="PASS",
        confidence_override=True,
    )
    monkeypatch.setattr(
        search.refinement,
        "search_hu",
        lambda *_args, **_kwargs: refinement,
    )
    monkeypatch.setattr(
        search.response_confirmation,
        "search_hu",
        lambda *_args, **_kwargs: confirmation,
    )

    result = search.search_hu(
        SimpleNamespace(),
        rules={},
        preferred_key="HU",
    )

    assert search.last_response_selection is None
    assert search.last_confirmation is confirmation
    assert result.selected_key == "PASS"
    assert result.confidence_override
    assert result.simulations == 184
    assert result.paired_determinizations == 92


def test_progressive_incomplete_response_selection_cannot_validate(
    monkeypatch,
):
    search = ProgressiveRootISMCTSPolicy(refinement_candidates=3)
    candidates = [
        RootResponseCandidate("PASS", "PASS", 0.0),
        RootResponseCandidate("PENG:五", "PENG", 1.0),
        RootResponseCandidate("CHI:test", "CHI", 2.0),
    ]
    stats = tuple(
        RootResponseCandidateStats(
            candidate=candidate,
            visits=1,
            reward_sum=float(index),
            average_reward=float(index),
            win_rate=0.0,
        )
        for index, candidate in enumerate(candidates)
    )
    coverage = RootResponseSearchResult(
        selected_key="CHI:test",
        used_search=True,
        reason="coverage",
        simulations=3,
        elapsed_ms=10.0,
        candidates=stats,
        paired_determinizations=1,
    )
    incomplete = RootResponseSearchResult(
        selected_key="CHI:test",
        used_search=True,
        reason="selection",
        simulations=156,
        elapsed_ms=100.0,
        candidates=stats,
        paired_determinizations=52,
        deadline_interruptions=1,
        empirical_best_key="CHI:test",
    )
    refinement = replace(
        incomplete,
        reason="refinement",
        simulations=24,
        paired_determinizations=8,
        deadline_interruptions=0,
    )
    monkeypatch.setattr(
        search.coverage,
        "search_response",
        lambda *_args, **_kwargs: coverage,
    )
    monkeypatch.setattr(
        search.response_selection,
        "search_response",
        lambda *_args, **_kwargs: incomplete,
    )
    monkeypatch.setattr(
        search.refinement,
        "search_response",
        lambda *_args, **_kwargs: refinement,
    )
    monkeypatch.setattr(
        search.response_confirmation,
        "search_response",
        lambda *_args, **_kwargs: pytest.fail(
            "incomplete selection must not reach validation"
        ),
    )

    result = search.search_response(
        SimpleNamespace(),
        rules={},
        candidates=candidates,
        preferred_key="PASS",
    )

    assert result.selected_key == "PASS"
    assert not result.used_search
    assert not result.confidence_override
    assert result.reason.endswith("root_response_selection_budget_incomplete")
    assert search.last_confirmation is None


def test_progressive_response_shortlist_protects_prior_nominee(monkeypatch):
    search = ProgressiveRootISMCTSPolicy(refinement_candidates=3)
    candidates = [
        RootResponseCandidate("PASS", "PASS", 0.0),
        RootResponseCandidate("PENG:五", "PENG", 0.0),
        RootResponseCandidate("CHI:noise", "CHI", 0.0),
        RootResponseCandidate("CHI:prior", "CHI", 10.0),
    ]
    coverage = RootResponseSearchResult(
        selected_key="PENG:五",
        used_search=True,
        reason="coverage",
        simulations=4,
        elapsed_ms=10.0,
        candidates=tuple(
            RootResponseCandidateStats(
                candidate=candidate,
                visits=1,
                reward_sum=reward,
                average_reward=reward,
                win_rate=0.0,
            )
            for candidate, reward in zip(
                candidates,
                (0.0, 1.0, 0.9, -1.0),
            )
        ),
        paired_determinizations=1,
    )
    observed = {}

    monkeypatch.setattr(
        search.coverage,
        "search_response",
        lambda *_args, **_kwargs: coverage,
    )

    def refine(*_args, **kwargs):
        observed["keys"] = [
            candidate.key for candidate in kwargs["candidates"]
        ]
        return RootResponseSearchResult(
            selected_key="PASS",
            used_search=True,
            reason="refinement",
            simulations=24,
            elapsed_ms=10.0,
            candidates=tuple(
                item
                for item in coverage.candidates
                if item.candidate.key in observed["keys"]
            ),
            paired_determinizations=8,
            empirical_best_key="PASS",
        )

    monkeypatch.setattr(
        search.refinement,
        "search_response",
        refine,
    )

    search.search_response(
        SimpleNamespace(),
        rules={},
        candidates=candidates,
        preferred_key="PASS",
    )

    assert observed["keys"] == ["PASS", "PENG:五", "CHI:prior"]


def test_progressive_candidate_applies_all_action_search():
    policy = ProfessionalProgressiveAllActionCandidatePolicy()

    assert policy.apply_discard_selection
    assert policy.apply_response_selection
    assert policy.apply_hu_selection
    assert policy.discard_max_candidates is None
    assert policy.response_max_candidates is None
    assert isinstance(policy.search, ProgressiveRootISMCTSPolicy)
    assert policy.response_search is policy.search
    assert policy.search.refinement_candidates == 5
    assert policy.search.discard_confirmation_alternatives == 2
    assert policy.search.discard_selection.config.max_candidates == 5
    assert policy.search.discard_selection.config.max_iterations == 80
    assert policy.search.confirmation.config.max_candidates == 3
    assert policy.search.confirmation.config.max_iterations == 288
    assert policy.search.response_selection.config.max_candidates == 5
    assert policy.search.response_selection.config.max_iterations == 160
    assert (
        policy.search.response_binary_confirmation.config.max_iterations
        == 256
    )
    assert policy.search.response_confirmation.config.max_iterations == 168
    assert (
        policy.search.confirmation.config.minimum_confident_advantage
        == pytest.approx(0.02)
    )
    assert policy.search.confirmation.config.time_budget_ms == 12_000
    assert (
        policy.search.confirmation.config
        .require_complete_iteration_budget_for_override
    )
    assert len(
        {
            policy.search.coverage.config.seed,
            policy.search.refinement.config.seed,
            policy.search.discard_selection.config.seed,
            policy.search.confirmation.config.seed,
        }
    ) == 4


def test_progressive_discard_reselection_keeps_old_audit_state_as_regression_only():
    view = public_view_from_dict(
        {
            "seat": 0,
            "hand": [
                "七",
                "十",
                "三",
                "八",
                "拾",
                "捌",
                "贰",
                "柒",
                "捌",
                "四",
                "壹",
                "壹",
                "一",
                "八",
                "七",
                "二",
                "六",
                "九",
            ],
            "own_melds": [{"type": "wei", "labels": ["肆", "肆", "肆"]}],
            "all_melds": [
                [{"type": "wei", "labels": ["肆", "肆", "肆"]}],
                [
                    {"type": "peng", "labels": ["玖", "玖", "玖"]},
                    {"type": "normal_sequence", "labels": ["叁", "肆", "伍"]},
                    {"type": "normal_sequence", "labels": ["伍", "陆", "柒"]},
                    {
                        "type": "mixed_same_rank_triplet",
                        "labels": ["二", "二", "贰"],
                    },
                ],
            ],
            "discards": [["六", "九"], ["伍", "九", "九", "一", "六", "伍"]],
            "remaining_counts": [
                ["一", 2],
                ["七", 2],
                ["三", 3],
                ["二", 1],
                ["五", 4],
                ["八", 2],
                ["六", 1],
                ["十", 3],
                ["叁", 3],
                ["四", 3],
                ["壹", 2],
                ["拾", 3],
                ["捌", 2],
                ["柒", 2],
                ["玖", 1],
                ["贰", 2],
                ["陆", 3],
            ],
            "stock_count": 31,
            "hand_sizes": [18, 8],
            "pending_card": None,
            "pending_source_seat": None,
        }
    )
    policy = ProfessionalProgressiveAllActionCandidatePolicy(
        rollout_policy_factories=(
            IndependentFastRolloutPolicy,
            IndependentFastPressurePolicy,
            IndependentFastDenialPolicy,
        )
    )

    selected = policy.choose_discard(
        view,
        rules_for_room(wildcard_enabled=False, players=2),
    )
    assert selected in view.hand
    assert policy.search.last_discard_selection is not None
    assert policy.search.last_confirmation is not None
    assert policy.search.last_confirmation.selected_label == selected
    assert selected in {
        item.label for item in policy.search.last_confirmation.candidates
    }


def test_progressive_response_expands_coverage_beyond_legacy_24_candidate_cap(
    monkeypatch,
):
    search = ProgressiveRootISMCTSPolicy(refinement_candidates=2)
    candidates = [
        RootResponseCandidate(f"PASS:{index}", "PASS", float(index))
        for index in range(41)
    ]
    observed = {}

    def cover(*_args, **kwargs):
        observed["keys"] = tuple(item.key for item in kwargs["candidates"])
        observed["config"] = search.coverage.config
        return RootResponseSearchResult(
            selected_key=candidates[0].key,
            used_search=False,
            reason="test_coverage_only",
            simulations=41,
            elapsed_ms=1.0,
            candidates=tuple(
                RootResponseCandidateStats(
                    candidate=candidate,
                    visits=1,
                    reward_sum=0.0,
                    average_reward=0.0,
                    win_rate=0.0,
                )
                for candidate in candidates
            ),
            paired_determinizations=1,
        )

    monkeypatch.setattr(search.coverage, "search_response", cover)

    result = search.search_response(
        SimpleNamespace(),
        rules={},
        candidates=candidates,
        preferred_key=candidates[0].key,
    )

    assert len(observed["keys"]) == 41
    assert len(result.candidates) == 41
    assert observed["config"].max_candidates >= 41
    assert observed["config"].max_iterations >= 41
    assert observed["config"].complete_first_paired_batch


def test_determinization_preserves_every_card_and_hides_opponent_hands():
    view, rules = _initial_public_view(seed=20260726)

    determinized = determinize_public_view(
        view,
        rules=rules,
        rng=random.Random(10),
    )

    assert determinized is not None
    players, stock = determinized
    assert [len(player.hand) for player in players] == [21, 20, 20]
    assert len(stock) == 19
    observed = Counter(stock)
    for player in players:
        observed.update(player.hand)
        observed.update(player.discards)
        observed.update(card for meld in player.melds for card in meld.cards)
    assert observed == full_deck_counts(rules=rules)


def test_heads_up_determinization_assigns_one_opponent_and_43_card_stock():
    view, rules = _initial_public_view(
        seed=20260726,
        players=2,
        wildcard_enabled=True,
    )

    determinized = determinize_public_view(
        view,
        rules=rules,
        rng=random.Random(10),
    )

    assert determinized is not None
    players, stock = determinized
    assert [len(player.hand) for player in players] == [21, 20]
    assert len(stock) == 43
    observed = Counter(stock)
    for player in players:
        observed.update(player.hand)
        observed.update(player.discards)
        observed.update(card for meld in player.melds for card in meld.cards)
    assert observed == full_deck_counts(rules=rules)


def test_hidden_assignment_retries_when_opponent_hand_contains_unresolved_auto_ti():
    class TwoStageShuffle:
        def __init__(self) -> None:
            self.calls = 0

        def shuffle(self, cards: list[str]) -> None:
            self.calls += 1
            if self.calls == 1:
                cards[:] = ["一", "一", "一", "一", "二", "三"]
            else:
                cards[:] = ["一", "二", "一", "三", "一", "一"]

    rng = TwoStageShuffle()
    assignment = _sample_legal_hidden_assignment(
        ["一", "一", "一", "一", "二", "三"],
        opponent_seats=[1],
        sizes={1: 4},
        rng=rng,
    )

    assert assignment is not None
    hands, stock = assignment
    assert rng.calls == 2
    assert Counter(hands[1])["一"] == 2
    assert Counter((*hands[1], *stock)) == Counter(("一", "一", "一", "一", "二", "三"))


def test_root_ismcts_search_is_bounded_and_returns_allowed_candidate():
    view, rules = _initial_public_view(seed=20260727)
    counts = Counter(view.hand)
    candidates = sorted(label for label, amount in counts.items() if amount < 3)[:2]
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
            skip_search_gap=10_000,
            rollout_max_turns=80,
        )
    )

    result = policy.search_discard(
        view,
        rules=rules,
        candidate_labels=candidates,
        force_search=True,
    )

    assert result.used_search
    assert result.simulations == 2
    assert result.selected_label in candidates
    assert all(candidate.visits == 1 for candidate in result.candidates)
    assert result.determinization_failures == 0


def test_root_hu_search_covers_hu_and_pass_in_the_same_worlds():
    hand = (
        "贰", "柒", "拾",
        "壹", "贰", "叁",
        "九", "九", "九",
        "一", "二", "三",
        "四", "五", "六",
        "七", "八", "九",
        "肆", "伍", "陆",
    )
    rules = rules_for_room(wildcard_enabled=False, players=2)
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(hand)
    view = PublicView(
        seat=0,
        hand=hand,
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=tuple(sorted((+remaining).items())),
        stock_count=sum(remaining.values()),
        hand_sizes=(len(hand), 0),
    )
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=4,
            max_candidates=2,
            rollout_max_turns=4,
            record_paired_worlds=True,
        ),
        rollout_policy_factory=BaselinePolicy,
    )

    result = policy.search_hu(view, rules=rules, preferred_key="HU")

    assert result.used_search
    assert result.paired_determinizations == 2
    assert {item.candidate.key for item in result.candidates} == {"HU", "PASS"}
    assert all(item.visits == 2 for item in result.candidates)
    assert len(result.paired_worlds) == 2
    hu_outcomes = [
        outcome
        for world in result.paired_worlds
        for outcome in world.outcomes
        if outcome.candidate_key == "HU"
    ]
    assert all(outcome.winner == 0 for outcome in hu_outcomes)
    assert all(outcome.reason == "post_action_hu" for outcome in hu_outcomes)

    win_first = RootISMCTSPolicy(
        replace(policy.config, objective_mode=ObjectiveMode.WIN_FIRST),
        rollout_policy_factory=BaselinePolicy,
    ).search_hu(view, rules=rules, preferred_key="PASS")

    assert win_first.selected_key == "HU"
    assert win_first.used_search
    assert win_first.reason == "win_first_immediate_hu_dominance"
    assert win_first.simulations == 0
    assert win_first.confidence_override


def test_full_action_teacher_uses_all_action_capacity_and_independent_rollouts():
    policy = ProfessionalFullActionTeacherPolicy(
        rollout_policy_factories=(BaselinePolicy,),
    )

    assert policy.search.config.time_budget_ms == 12_000
    assert policy.search.config.max_candidates == 24
    assert policy.search.config.min_confidence_pairs == 8
    assert policy.search.config.complete_first_paired_batch
    assert policy.response_max_candidates is None
    assert policy.discard_max_candidates is None
    assert policy.include_policy_rejected_actions
    assert not policy.respect_response_safety_flags
    assert policy.apply_hu_selection


def test_full_action_teacher_does_not_drop_policy_rejected_legal_discard(
    monkeypatch,
):
    view, rules = _initial_public_view(seed=20260828, players=2)
    labels = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if label != "王" and amount < 3
    )[:3]
    action_evals = [
        SimpleNamespace(
            type="DISCARD",
            allowed=index != 1,
            ev=float(100 - index),
            action=SimpleNamespace(label=label),
        )
        for index, label in enumerate(labels)
    ]
    decision = SimpleNamespace(
        selected_action="DISCARD",
        selected_label=labels[0],
        action_evals=action_evals,
    )
    observed = {}
    policy = ProfessionalFullActionTeacherPolicy(
        rollout_policy_factories=(BaselinePolicy,),
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: decision)

    def fake_search(*_args, **kwargs):
        observed["labels"] = tuple(kwargs["candidate_labels"])
        return RootSearchResult(
            selected_label=labels[0],
            used_search=True,
            reason="test",
            simulations=3,
            elapsed_ms=1.0,
            candidates=(),
        )

    monkeypatch.setattr(policy.search, "search_discard", fake_search)

    assert policy.choose_discard(view, rules) == labels[0]
    assert set(observed["labels"]) == set(labels)
    event = policy.discard_events()[-1]
    assert event["search_attempted"]
    assert event["legal_candidate_count"] == len(labels)
    assert event["searched_candidate_count"] == len(labels)
    assert event["public_view"] == public_view_to_dict(view)


def test_full_action_teacher_excludes_rule_locked_triplet_from_discard_roots(
    monkeypatch,
):
    rules = rules_for_room(wildcard_enabled=False, players=2)
    view = PublicView(
        seat=0,
        hand=("二", "二", "二", "四", "六"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(),
        stock_count=10,
        hand_sizes=(5, 4),
    )
    decision = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="四",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                allowed=True,
                ev=300.0,
                action=SimpleNamespace(label="二"),
            ),
            SimpleNamespace(
                type="DISCARD",
                allowed=True,
                ev=200.0,
                action=SimpleNamespace(label="四"),
            ),
            SimpleNamespace(
                type="DISCARD",
                allowed=True,
                ev=100.0,
                action=SimpleNamespace(label="六"),
            ),
        ],
    )
    observed = {}
    policy = ProfessionalFullActionTeacherPolicy(
        rollout_policy_factories=(BaselinePolicy,),
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: decision)

    def fake_search(*_args, **kwargs):
        observed["labels"] = tuple(kwargs["candidate_labels"])
        return RootSearchResult(
            selected_label="四",
            used_search=True,
            reason="test",
            simulations=2,
            elapsed_ms=1.0,
            candidates=tuple(
                RootCandidateStats(
                    label=label,
                    visits=1,
                    reward_sum=0.0,
                    average_reward=0.0,
                    win_rate=0.0,
                    heuristic_value=0.0,
                )
                for label in kwargs["candidate_labels"]
            ),
        )

    monkeypatch.setattr(policy.search, "search_discard", fake_search)

    assert policy.choose_discard(view, rules) == "四"
    assert set(observed["labels"]) == {"四", "六"}
    event = policy.discard_events()[-1]
    assert event["legal_candidate_count"] == 2
    assert event["searched_candidate_count"] == 2


def test_full_action_response_builder_can_include_policy_rejected_candidate():
    rejected_pass = SimpleNamespace(
        type="PASS",
        allowed=False,
        reject_reason="production_policy_rejected",
        ev=-10.0,
        action=SimpleNamespace(option_id=None, action_id=None),
        debug_details={},
    )

    assert build_response_candidates("二", (rejected_pass,)) == []
    included = build_response_candidates(
        "二",
        (rejected_pass,),
        include_policy_rejected=True,
    )

    assert [candidate.key for candidate in included] == ["PASS"]


def test_discard_search_is_not_usable_until_every_candidate_is_visited():
    view, rules = _initial_public_view(seed=20260727)
    candidates = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=1,
            max_candidates=2,
            rollout_max_turns=8,
        )
    )

    result = policy.search_discard(
        view,
        rules=rules,
        candidate_labels=candidates,
        force_search=True,
    )

    assert not result.used_search
    assert result.reason == "candidate_coverage_incomplete"
    assert sum(candidate.visits for candidate in result.candidates) == 1
    assert any(candidate.visits == 0 for candidate in result.candidates)


def test_paired_discard_search_compares_candidates_on_same_determinization(monkeypatch):
    view, rules = _initial_public_view(seed=20260727)
    candidates = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]
    observed = []

    def fake_play(self, **kwargs):
        root_policy = self.policies[view.seat]
        observed.append((int(kwargs["seed"]), root_policy.first_label))
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=0,
            turns=1,
            reason="stock_exhausted",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_play)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=4,
            max_candidates=2,
            record_paired_worlds=True,
        )
    ).search_discard(
        view,
        rules=rules,
        candidate_labels=candidates,
        force_search=True,
        paired_candidates=True,
    )

    assert result.used_search
    assert result.paired_determinizations == 2
    assert result.simulations == 4
    assert observed[0][0] == observed[1][0]
    assert observed[2][0] == observed[3][0]
    assert {label for _seed, label in observed[:2]} == set(candidates)
    assert len(result.paired_worlds) == 2
    assert all(len(world.outcomes) == 2 for world in result.paired_worlds)
    assert all(
        world.root_continuation_policy == "root_self_continuation_v1"
        for world in result.paired_worlds
    )
    assert {
        outcome.candidate_key
        for outcome in result.paired_worlds[0].outcomes
    } == set(candidates)
    assert all(world.world_fingerprint for world in result.paired_worlds)
    assert all(candidate.wins == 0 for candidate in result.candidates)
    assert all(candidate.losses == 0 for candidate in result.candidates)
    assert all(candidate.draws == 2 for candidate in result.candidates)
    assert all(candidate.draw_rate == 1.0 for candidate in result.candidates)
    assert all(candidate.mean_outcome_score == 0.0 for candidate in result.candidates)
    assert all(candidate.mean_signed_xi == 0.0 for candidate in result.candidates)


def test_paired_discard_search_balances_mixed_opponents_without_breaking_pairs(
    monkeypatch,
):
    view, rules = _initial_public_view(seed=20260727)
    candidates = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]
    observed = []

    class StyleA(BaselinePolicy):
        name = "style_a"

    class StyleB(BaselinePolicy):
        name = "style_b"

    def fake_play(self, **kwargs):
        root_policy = self.policies[view.seat]
        opponent = self.policies[(view.seat + 1) % 3]
        observed.append(
            (
                opponent.name,
                root_policy.first_label,
                root_policy.continuation_policy.name,
            )
        )
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=0,
            turns=1,
            reason="stock_exhausted",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_play)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=4,
            max_candidates=2,
        ),
        rollout_policy_factories=(StyleA, StyleB),
    ).search_discard(
        view,
        rules=rules,
        candidate_labels=candidates,
        force_search=True,
        paired_candidates=True,
        preferred_label=candidates[0],
    )

    assert result.used_search
    assert [style for style, _label, _root_style in observed] == [
        "style_a",
        "style_a",
        "style_b",
        "style_b",
    ]
    assert {label for _style, label, _root_style in observed[:2]} == set(candidates)
    assert {label for _style, label, _root_style in observed[2:]} == set(candidates)
    assert all(
        root_style not in {"style_a", "style_b"}
        for _style, _label, root_style in observed
    )


def test_paired_advantage_uses_familywise_correction_for_multiple_alternatives():
    deltas = (1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
    single_batches = [
        {"production": 0.0, "alternative_a": delta}
        for delta in deltas
    ]
    multiple_batches = [
        {
            "production": 0.0,
            "alternative_a": delta,
            "alternative_b": delta,
        }
        for delta in deltas
    ]

    single = _paired_advantage_stats(
        single_batches,
        candidate_keys=("production", "alternative_a"),
        preferred_key="production",
    )
    multiple = _paired_advantage_stats(
        multiple_batches,
        candidate_keys=("production", "alternative_a", "alternative_b"),
        preferred_key="production",
    )
    frozen_familywise = _paired_advantage_stats(
        single_batches,
        candidate_keys=("production", "alternative_a"),
        preferred_key="production",
        familywise_comparisons=5,
    )

    assert single[0].familywise_comparisons == 1
    assert multiple[0].familywise_comparisons == 2
    assert frozen_familywise[0].familywise_comparisons == 5
    assert (
        multiple[0].lower_confidence_bound
        < single[0].lower_confidence_bound
    )
    assert (
        frozen_familywise[0].lower_confidence_bound
        < multiple[0].lower_confidence_bound
    )


@pytest.mark.parametrize(
    ("paired_samples", "expected_override"),
    ((2, False), (6, True)),
)
def test_paired_discard_confidence_gate_preserves_production_until_sample_floor(
    monkeypatch,
    paired_samples,
    expected_override,
):
    view, rules = _initial_public_view(seed=20260727)
    production, alternative = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]

    def fake_play(self, **kwargs):
        label = self.policies[view.seat].first_label
        winner = view.seat if label == alternative else (view.seat + 1) % 3
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=winner,
            dealer=0,
            turns=1,
            reason="hu",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_play)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=paired_samples * 2,
            max_candidates=2,
            require_confident_override=True,
            min_confidence_pairs=6,
        )
    ).search_discard(
        view,
        rules=rules,
        candidate_labels=(production, alternative),
        candidate_priors={production: 200.0, alternative: 100.0},
        force_search=True,
        paired_candidates=True,
        preferred_label=production,
    )

    assert result.used_search
    assert result.empirical_best_label == alternative
    assert result.confidence_override is expected_override
    assert result.selected_label == (
        alternative if expected_override else production
    )
    assert result.paired_advantages[0].samples == paired_samples
    assert result.paired_advantages[0].lower_confidence_bound == pytest.approx(2.0)


def test_paired_discard_incomplete_confirmation_budget_cannot_override(
    monkeypatch,
):
    view, rules = _initial_public_view(seed=20260727)
    production, alternative = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]
    calls = 0

    def fake_play(self, **kwargs):
        nonlocal calls
        calls += 1
        label = self.policies[view.seat].first_label
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=(
                None
                if calls == 5
                else (
                    view.seat
                    if label == alternative
                    else (view.seat + 1) % 3
                )
            ),
            dealer=0,
            turns=1,
            reason="time_budget" if calls == 5 else "hu",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_play)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=6,
            max_candidates=2,
            require_confident_override=True,
            min_confidence_pairs=2,
            require_complete_iteration_budget_for_override=True,
        )
    ).search_discard(
        view,
        rules=rules,
        candidate_labels=(production, alternative),
        force_search=True,
        paired_candidates=True,
        preferred_label=production,
    )

    assert result.used_search
    assert result.paired_determinizations == 2
    assert result.deadline_interruptions == 1
    assert result.empirical_best_label == alternative
    assert result.selected_label == production
    assert not result.confidence_override
    assert result.reason == "root_discard_confidence_budget_incomplete"


def test_paired_discard_search_rejects_rollout_invariant_violation(monkeypatch):
    view, rules = _initial_public_view(seed=20260727)
    production, alternative = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]

    def fake_invalid(self, **kwargs):
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=view.seat,
            dealer=0,
            turns=1,
            reason="hu",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=("card_conservation_failed",),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_invalid)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=12,
            max_candidates=2,
            require_confident_override=True,
            min_confidence_pairs=6,
        )
    ).search_discard(
        view,
        rules=rules,
        candidate_labels=(production, alternative),
        force_search=True,
        paired_candidates=True,
        preferred_label=production,
    )

    assert not result.used_search
    assert result.reason == "rollout_invariant_violation"
    assert result.selected_label == production
    assert result.rollout_invariant_violations == 1
    assert result.rollout_violations == ("card_conservation_failed",)


def test_paired_discard_search_rejects_unverified_rollout_coverage(monkeypatch):
    view, rules = _initial_public_view(seed=20260727, wildcard_enabled=True)
    candidates = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if label != "王" and amount < 3
    )[:2]

    def fake_unsupported(self, **kwargs):
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=True,
            winner=None,
            dealer=0,
            turns=1,
            reason="rollout_coverage_incomplete",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
            coverage_failures=("seat_0:unsupported_chenzhou_wildcard_terminal",),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_unsupported)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=4,
            max_candidates=2,
        )
    ).search_discard(
        view,
        rules=rules,
        candidate_labels=candidates,
        force_search=True,
        paired_candidates=True,
    )

    assert not result.used_search
    assert result.reason == "rollout_coverage_incomplete"
    assert result.simulations == 0
    assert result.paired_determinizations == 0
    assert result.rollout_coverage_failures == 1
    assert result.rollout_coverage_reasons == (
        "seat_0:unsupported_chenzhou_wildcard_terminal",
    )
    assert {candidate.visits for candidate in result.candidates} == {0}


def test_paired_discard_search_discards_incomplete_deadline_batch(monkeypatch):
    view, rules = _initial_public_view(seed=20260727)
    candidates = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]

    def fake_timeout(self, **kwargs):
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=0,
            turns=0,
            reason="time_budget",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_timeout)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
        )
    ).search_discard(
        view,
        rules=rules,
        candidate_labels=candidates,
        force_search=True,
        paired_candidates=True,
    )

    assert not result.used_search
    assert result.reason == "discard_deadline_before_complete_pair"
    assert result.simulations == 0
    assert result.paired_determinizations == 0
    assert result.deadline_interruptions == 1
    assert {candidate.visits for candidate in result.candidates} == {0}


def test_immediate_response_risk_search_pairs_candidates_and_records_next_seat_chi(
    monkeypatch,
):
    view, rules = _initial_public_view(seed=20260727)
    risky, safe = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]
    next_seat = (view.seat + 1) % 3

    def fake_play(self, **kwargs):
        label = self.policies[view.seat].first_label
        actions = {"discard": 1}
        if label == risky:
            actions.update({"chi": 1, f"chi:seat_{next_seat}": 1})
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=view.seat,
            turns=1,
            reason="max_turns",
            score=0.0,
            total_xi=0,
            action_counts=actions,
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_play)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=4,
            max_candidates=2,
        )
    ).search_discard_response_risk(
        view,
        rules=rules,
        candidate_labels=(risky, safe),
        candidate_priors={risky: 200.0, safe: 150.0},
    )

    assert result.used_search
    assert result.selected_label == safe
    assert result.simulations == 4
    assert result.paired_determinizations == 2
    by_label = {candidate.label: candidate.to_dict() for candidate in result.candidates}
    assert by_label[risky]["claim_rates"]["chi"] == 1.0
    assert by_label[risky]["claims_by_seat"] == [
        {
            "action": "chi",
            "seat": next_seat,
            "relative_offset": 1,
            "is_next_seat": True,
            "count": 2,
            "rate": 1.0,
        }
    ]
    assert by_label[safe]["claim_rate"] == 0.0


def test_immediate_response_risk_rejects_non_next_seat_chi(monkeypatch):
    view, rules = _initial_public_view(seed=20260727)
    labels = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]
    illegal_seat = (view.seat + 2) % 3

    def fake_play(self, **kwargs):
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=view.seat,
            turns=1,
            reason="max_turns",
            score=0.0,
            total_xi=0,
            action_counts={
                "discard": 1,
                "chi": 1,
                f"chi:seat_{illegal_seat}": 1,
            },
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_play)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
        )
    ).search_discard_response_risk(
        view,
        rules=rules,
        candidate_labels=labels,
        candidate_priors={label: 100.0 for label in labels},
    )

    assert not result.used_search
    assert result.reason == "rollout_invariant_failure"
    assert result.rollout_invariant_violations == 1
    assert result.rollout_violations[0].startswith("illegal_non_next_seat_chi")


def test_immediate_response_risk_tie_preserves_production_label(monkeypatch):
    view, rules = _initial_public_view(seed=20260727)
    labels = sorted(
        label
        for label, amount in Counter(view.hand).items()
        if amount < 3
    )[:2]

    def fake_play(self, **kwargs):
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=view.seat,
            turns=1,
            reason="max_turns",
            score=0.0,
            total_xi=0,
            action_counts={"discard": 1},
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_state", fake_play)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
        )
    ).search_discard_response_risk(
        view,
        rules=rules,
        candidate_labels=labels,
        candidate_priors={label: 100.0 for label in labels},
        preferred_label=labels[0],
    )

    assert result.used_search
    assert result.selected_label == labels[0]


def test_discard_response_risk_shadow_never_replaces_production(monkeypatch):
    view, rules = _initial_public_view(seed=20260727)
    production = ProfessionalBrainSimulationPolicy().choose_discard(view, rules)

    def fake_search(_view, *, candidate_labels, **_kwargs):
        alternate = next(label for label in candidate_labels if label != production)
        return DiscardResponseRiskResult(
            selected_label=alternate,
            used_search=True,
            reason="discard_response_risk_completed",
            simulations=8,
            elapsed_ms=5.0,
            candidates=(),
            paired_determinizations=4,
        )

    shadow = ProfessionalDiscardResponseRiskShadowPolicy(activation_ev_gap=10_000.0)
    monkeypatch.setattr(shadow.risk_search, "search_discard_response_risk", fake_search)
    candidate = ProfessionalDiscardResponseRiskCandidatePolicy(activation_ev_gap=10_000.0)
    monkeypatch.setattr(candidate.risk_search, "search_discard_response_risk", fake_search)

    assert shadow.choose_discard(view, rules) == production
    assert candidate.choose_discard(view, rules) != production
    event = shadow.discard_events()[0]
    assert event["disagreement"]
    assert not event["selection_applied"]


def test_discard_shadow_captures_root_without_replacing_production_discard():
    view, rules = _initial_public_view(seed=20260727)
    production = ProfessionalBrainSimulationPolicy().choose_discard(view, rules)
    policy = ProfessionalDiscardShadowSimulationPolicy(
        activation_ev_gap=10_000.0,
    )

    selected = policy.choose_discard(view, rules)

    assert selected == production
    captured = [event for event in policy.discard_events() if event["eligible"]]
    assert len(captured) == 1
    assert captured[0]["production_label"] == production
    assert captured[0]["public_view"] == public_view_to_dict(view)


def test_pending_response_card_is_public_but_unassigned_during_determinization():
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
    )

    determinized = determinize_public_view(view, rules=rules, rng=random.Random(10))

    assert determinized is not None
    players, stock = determinized
    observed = Counter(stock)
    observed.update((view.pending_card,))
    for player in players:
        observed.update(player.hand)
        observed.update(player.discards)
        observed.update(card for meld in player.melds for card in meld.cards)
    assert observed == full_deck_counts(rules=rules)


def test_forced_chi_rejects_wrong_source():
    view, rules = _response_public_view(
        hand=("二", "十", "四", "六", "九"),
        pending="七",
        source=2,
    )
    candidate = RootResponseCandidate(
        key="CHI:2710",
        action_type="CHI",
        heuristic_value=200.0,
        option_id="2710",
        consumed_from_hand=("二", "十"),
        meld_groups=(("二", "七", "十"),),
        followup_discard="九",
    )
    wrong_source = PublicView(
        **{
            **view.__dict__,
            "pending_source_seat": 1,
        }
    )
    with pytest.raises(ValueError, match="chi_candidate_wrong_source_seat"):
        RootISMCTSPolicy().search_response(
            wrong_source,
            rules=rules,
            candidates=(RootResponseCandidate("PASS", "PASS", 100.0), candidate),
            force_search=True,
        )


def test_root_response_search_is_bounded_and_compares_pass_with_peng():
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 100.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=200.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
            rollout_max_turns=8,
        )
    )

    result = policy.search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
    )

    assert result.used_search
    assert result.simulations == 2
    assert result.selected_key in {"PASS", "PENG:二"}
    assert all(candidate.visits == 1 for candidate in result.candidates)
    assert result.determinization_failures == 0
    assert result.paired_determinizations == 1


def test_response_search_separates_root_continuation_from_opponents(monkeypatch):
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 100.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=200.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )
    opponent_created = 0
    root_created = 0

    class MarkerPolicy(BaselinePolicy):
        name = "opponent_marker"

    class RootContinuationPolicy(BaselinePolicy):
        name = "root_continuation_marker"

    def marker_factory():
        nonlocal opponent_created
        opponent_created += 1
        return MarkerPolicy()

    def root_factory():
        nonlocal root_created
        root_created += 1
        return RootContinuationPolicy()

    observed_policy_names = []

    def fake_play(self, **kwargs):
        root_policy = self.policies[view.seat]
        observed_policy_names.append(
            (
                tuple(policy.name for policy in self.policies),
                root_policy.continuation_policy.name,
            )
        )
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=0,
            turns=1,
            reason="stock_exhausted",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_pending_discard", fake_play)
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
        ),
        rollout_policy_factory=marker_factory,
        root_continuation_policy_factory=root_factory,
    ).search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
    )

    assert result.used_search
    assert opponent_created == 4
    assert root_created == 2
    assert observed_policy_names == [
        (
            ("ismcts_response_rollout", "opponent_marker", "opponent_marker"),
            "root_continuation_marker",
        ),
        (
            ("ismcts_response_rollout", "opponent_marker", "opponent_marker"),
            "root_continuation_marker",
        ),
    ]


def test_forced_first_discard_delegates_every_later_decision():
    class ContinuationPolicy(BaselinePolicy):
        def choose_discard(self, view, rules):
            return "三"

        def choose_hu(self, view, hu, rules):
            return False

        def choose_peng(self, view, label, rules):
            return True

        def choose_chi(self, view, plans, rules):
            return plans[-1]

    policy = _ForcedFirstDiscardPolicy(
        "二",
        continuation_policy=ContinuationPolicy(),
    )
    view, rules = _response_public_view(
        hand=("二", "三"),
        pending="四",
        source=1,
    )
    plans = [
        ChiPlan(
            initial_group=("二", "三", "四"),
            compare_groups=(),
            consumed_from_hand=("二", "三"),
        )
    ]

    assert policy.choose_discard(view, rules) == "二"
    assert policy.choose_discard(view, rules) == "三"
    assert not policy.choose_hu(view, object(), rules)
    assert policy.choose_peng(view, "四", rules)
    assert policy.choose_chi(view, plans, rules) is plans[-1]


def test_forced_root_chi_does_not_leak_followup_when_other_peng_overrides_it():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    candidate = RootResponseCandidate(
        key="CHI:2710",
        action_type="CHI",
        heuristic_value=200.0,
        option_id="2710",
        consumed_from_hand=("二", "十"),
        meld_groups=(("二", "七", "十"),),
        followup_discard="九",
    )
    root_policy = _ForcedRootResponsePolicy(
        pending_card="七",
        candidate=candidate,
    )
    players, stock = _state_with_pending_card(
        rules,
        hands=[["二", "十", "九"], ["七", "七", "四"], []],
        pending="七",
    )
    simulator = FullGameSimulator(
        [root_policy, _AlwaysPengPolicy(), BaselinePolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_pending_discard(
        seed=20260729,
        players=players,
        stock=stock,
        discarder=2,
        pending_card="七",
        blocked_auto_claim_seats=frozenset((0,)),
        max_turns=0,
    )

    assert result.action_counts["peng"] == 1
    assert not root_policy.claim_committed
    assert root_policy.followup_used
    assert "九" in players[0].hand
    assert not result.violations


def test_forced_root_peng_commits_and_discards_followup_once():
    rules = rules_for_room(wildcard_enabled=False, players=3)
    candidate = RootResponseCandidate(
        key="PENG:二",
        action_type="PENG",
        heuristic_value=200.0,
        consumed_from_hand=("二", "二"),
        meld_groups=(("二", "二", "二"),),
        followup_discard="九",
    )
    root_policy = _ForcedRootResponsePolicy(
        pending_card="二",
        candidate=candidate,
    )
    players, stock = _state_with_pending_card(
        rules,
        hands=[["二", "二", "四", "六", "九"], ["一", "三"], []],
        pending="二",
    )
    simulator = FullGameSimulator(
        [root_policy, BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_pending_discard(
        seed=20260730,
        players=players,
        stock=stock,
        discarder=2,
        pending_card="二",
        blocked_auto_claim_seats=frozenset((0,)),
        max_turns=1,
    )

    assert result.action_counts["peng"] == 1
    assert result.action_counts["discard"] == 1
    assert root_policy.claim_committed
    assert root_policy.followup_used
    assert "九" not in players[0].hand
    assert players[0].melds[-1].cards == ("二", "二", "二")
    assert not result.violations


def test_simulator_uses_one_joint_response_selection_when_policy_supports_it():
    class JointPengPolicy(BaselinePolicy):
        name = "joint_peng"

        def choose_response(self, view, legal_actions, plans, hu, rules):
            assert {action.type for action in legal_actions} == {"PASS", "PENG"}
            assert not plans
            assert hu is None
            return f"PENG:{view.pending_card}"

        def choose_peng(self, view, label, rules):
            raise AssertionError("sequential response callback must not run")

    rules = rules_for_room(wildcard_enabled=False, players=2)
    players, stock = _state_with_pending_card(
        rules,
        hands=[["二", "二", "四", "六", "九"], ["一", "三"]],
        pending="二",
    )
    simulator = FullGameSimulator(
        [JointPengPolicy(), BaselinePolicy()],
        wildcard_enabled=False,
        rules=rules,
    )

    result = simulator.play_from_pending_discard(
        seed=20260730,
        players=players,
        stock=stock,
        discarder=1,
        pending_card="二",
        max_turns=0,
    )

    assert result.action_counts["peng"] == 1
    assert players[0].melds[-1].cards == ("二", "二", "二")
    assert not result.violations


def test_forced_response_delegates_after_the_root_response_finishes():
    class ContinuationPolicy(BaselinePolicy):
        def choose_discard(self, view, rules):
            return "三"

        def choose_hu(self, view, hu, rules):
            return False

        def choose_peng(self, view, label, rules):
            return True

    candidate = RootResponseCandidate(
        key="PASS",
        action_type="PASS",
        heuristic_value=0.0,
    )
    policy = _ForcedRootResponsePolicy(
        pending_card="四",
        candidate=candidate,
        continuation_policy=ContinuationPolicy(),
    )
    view, rules = _response_public_view(
        hand=("二", "三"),
        pending="四",
        source=1,
    )

    assert not policy.choose_hu(view, object(), rules)
    assert not policy.choose_peng(view, "四", rules)
    policy.finish_pending_response()
    assert not policy.choose_hu(view, object(), rules)
    assert policy.choose_peng(view, "四", rules)
    assert policy.choose_discard(view, rules) == "三"


def test_response_search_is_not_usable_until_every_candidate_is_visited():
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 100.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=200.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=1,
            max_candidates=2,
            rollout_max_turns=8,
        )
    )

    result = policy.search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
    )

    assert not result.used_search
    assert result.reason == "paired_candidate_budget_too_small"
    assert sum(candidate.visits for candidate in result.candidates) == 0
    assert result.paired_determinizations == 0


def test_response_search_discards_an_incomplete_deadline_batch(monkeypatch):
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
        players=2,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 200.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=100.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )
    calls = 0

    def fake_play_from_pending_discard(self, **kwargs):
        nonlocal calls
        calls += 1
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=0,
            turns=1,
            reason="time_budget" if calls == 3 else "stock_exhausted",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(
        FullGameSimulator,
        "play_from_pending_discard",
        fake_play_from_pending_discard,
    )
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=4,
            max_candidates=2,
        )
    )

    result = policy.search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
    )

    assert result.used_search
    assert result.reason == "root_response_ismcts_completed_at_deadline"
    assert result.simulations == 2
    assert result.paired_determinizations == 1
    assert result.deadline_interruptions == 1
    assert {candidate.visits for candidate in result.candidates} == {1}


def test_response_coverage_completes_first_pair_before_soft_deadline(
    monkeypatch,
):
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
        players=2,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 200.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=100.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )
    deadlines = []

    def fake_play_from_pending_discard(self, **kwargs):
        deadlines.append(kwargs["deadline"])
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=0,
            turns=1,
            reason="stock_exhausted",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(
        FullGameSimulator,
        "play_from_pending_discard",
        fake_play_from_pending_discard,
    )
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=1,
            max_iterations=2,
            max_candidates=2,
            complete_first_paired_batch=True,
        )
    )

    result = policy.search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
    )

    assert result.used_search
    assert result.paired_determinizations == 1
    assert {candidate.visits for candidate in result.candidates} == {1}
    assert deadlines == [None, None]


def test_response_confidence_gate_uses_paired_advantage_over_production(monkeypatch):
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
        players=2,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 200.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=100.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )

    def fake_play_from_pending_discard(self, **kwargs):
        root_policy = self.policies[view.seat]
        winner = view.seat if root_policy.candidate.key == "PENG:二" else 1
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=winner,
            dealer=0,
            turns=1,
            reason="hu",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(
        FullGameSimulator,
        "play_from_pending_discard",
        fake_play_from_pending_discard,
    )
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=12,
            max_candidates=2,
            require_confident_override=True,
            min_confidence_pairs=6,
        )
    ).search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
        preferred_key="PASS",
    )

    assert result.used_search
    assert result.empirical_best_key == "PENG:二"
    assert result.selected_key == "PENG:二"
    assert result.confidence_override
    assert result.paired_advantages[0].candidate_key == "PENG:二"
    assert result.paired_advantages[0].lower_confidence_bound == pytest.approx(2.0)


def test_response_incomplete_confirmation_budget_cannot_override(monkeypatch):
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
        players=2,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 200.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=100.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )
    calls = 0

    def fake_play_from_pending_discard(self, **kwargs):
        nonlocal calls
        calls += 1
        root_policy = self.policies[view.seat]
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=(
                None
                if calls == 5
                else (
                    view.seat
                    if root_policy.candidate.key == "PENG:二"
                    else 1
                )
            ),
            dealer=0,
            turns=1,
            reason="time_budget" if calls == 5 else "hu",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(
        FullGameSimulator,
        "play_from_pending_discard",
        fake_play_from_pending_discard,
    )
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=6,
            max_candidates=2,
            require_confident_override=True,
            min_confidence_pairs=2,
            require_complete_iteration_budget_for_override=True,
        )
    ).search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
        preferred_key="PASS",
    )

    assert result.used_search
    assert result.paired_determinizations == 2
    assert result.deadline_interruptions == 1
    assert result.empirical_best_key == "PENG:二"
    assert result.selected_key == "PASS"
    assert not result.confidence_override
    assert result.reason == "root_response_confidence_budget_incomplete"


def test_response_search_rejects_rollout_invariant_violation(monkeypatch):
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
        players=2,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 200.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=100.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )

    def fake_invalid(self, **kwargs):
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=view.seat,
            dealer=0,
            turns=1,
            reason="hu",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=("card_conservation_failed",),
        )

    monkeypatch.setattr(
        FullGameSimulator,
        "play_from_pending_discard",
        fake_invalid,
    )
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=12,
            max_candidates=2,
            require_confident_override=True,
            min_confidence_pairs=6,
        )
    ).search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
        preferred_key="PASS",
    )

    assert not result.used_search
    assert result.reason == "rollout_invariant_violation"
    assert result.selected_key == "PASS"
    assert result.rollout_invariant_violations == 1
    assert result.rollout_violations == ("card_conservation_failed",)


@pytest.mark.parametrize(
    ("policy_type", "expected_peng"),
    (
        (ProfessionalConfidenceRootShadowPolicy, False),
        (ProfessionalConfidenceRootCandidatePolicy, True),
    ),
)
def test_confidence_policy_keeps_shadow_and_candidate_response_authority_separate(
    monkeypatch,
    policy_type,
    expected_peng,
):
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
        players=2,
    )
    policy = policy_type(
        root_config=RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=12,
            require_confident_override=True,
            min_confidence_pairs=6,
        )
    )
    monkeypatch.setattr(
        policy,
        "_choose_peng_decision",
        lambda *_args, **_kwargs: SimpleNamespace(selected_action="PASS"),
    )

    def fake_observe(*_args, **_kwargs):
        policy.last_response_search = RootResponseSearchResult(
            selected_key="PENG:二",
            used_search=True,
            reason="root_response_confidence_override",
            simulations=12,
            elapsed_ms=1.0,
            candidates=(),
            confidence_override=True,
        )

    monkeypatch.setattr(policy, "_observe_response_search", fake_observe)

    assert policy.choose_peng(view, "二", rules) is expected_peng


@pytest.mark.parametrize(
    ("policy_type", "expected_peng"),
    (
        (ProfessionalConfidenceRootShadowPolicy, False),
        (ProfessionalConfidenceRootCandidatePolicy, True),
    ),
)
def test_joint_response_search_compares_peng_chi_and_pass_once(
    monkeypatch,
    policy_type,
    expected_peng,
):
    view, rules = _response_public_view(
        hand=("五", "五", "四", "六", "伍", "八"),
        pending="五",
        source=2,
        players=3,
    )
    policy = policy_type(
        root_config=RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=24,
            require_confident_override=True,
            min_confidence_pairs=6,
        )
    )
    plans = policy._joint_response_chi_plans(view, "五", rules)
    observed = []

    def force_peng_over_joint_candidates(*args, **kwargs):
        candidates = tuple(kwargs["candidates"])
        observed.append(
            (
                tuple(candidate.key for candidate in candidates),
                kwargs["preferred_key"],
            )
        )
        return RootResponseSearchResult(
            selected_key="PENG:五",
            used_search=True,
            reason="root_response_confidence_override",
            simulations=24,
            elapsed_ms=1.0,
            candidates=(),
            confidence_override=True,
        )

    monkeypatch.setattr(
        policy.response_search,
        "search_response",
        force_peng_over_joint_candidates,
    )

    assert policy.choose_peng(view, "五", rules) is expected_peng
    assert len(observed) == 1
    candidate_keys, production_key = observed[0]
    assert "PENG:五" in candidate_keys
    assert "PASS" in candidate_keys
    assert any(key.startswith("CHI:") for key in candidate_keys)
    assert production_key.startswith("CHI:")
    if not expected_peng:
        assert policy.choose_chi(view, plans, rules) is not None
        assert len(observed) == 1


def test_response_search_is_unusable_without_one_complete_deadline_pair(monkeypatch):
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九"),
        pending="二",
        source=1,
        players=2,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 200.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=100.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )

    def fake_timeout(self, **kwargs):
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=False,
            winner=None,
            dealer=0,
            turns=0,
            reason="time_budget",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
        )

    monkeypatch.setattr(FullGameSimulator, "play_from_pending_discard", fake_timeout)

    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
        )
    ).search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
    )

    assert not result.used_search
    assert result.reason == "response_deadline_before_complete_pair"
    assert result.simulations == 0
    assert result.paired_determinizations == 0
    assert result.deadline_interruptions == 1
    assert {candidate.visits for candidate in result.candidates} == {0}


def test_response_search_rejects_unverified_rollout_coverage(monkeypatch):
    view, rules = _response_public_view(
        hand=("二", "二", "四", "六", "九", "王"),
        pending="二",
        source=1,
        players=2,
        wildcard_enabled=True,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 200.0),
        RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=100.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        ),
    )

    def fake_unsupported(self, **kwargs):
        return GameResult(
            seed=int(kwargs["seed"]),
            wildcard_enabled=True,
            winner=None,
            dealer=0,
            turns=1,
            reason="rollout_coverage_incomplete",
            score=0.0,
            total_xi=0,
            action_counts={},
            violations=(),
            coverage_failures=("seat_0:unsupported_chenzhou_wildcard_terminal",),
        )

    monkeypatch.setattr(
        FullGameSimulator,
        "play_from_pending_discard",
        fake_unsupported,
    )
    result = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=4,
            max_candidates=2,
        )
    ).search_response(
        view,
        rules=rules,
        candidates=candidates,
        force_search=True,
    )

    assert not result.used_search
    assert result.reason == "rollout_coverage_incomplete"
    assert result.simulations == 0
    assert result.paired_determinizations == 0
    assert result.rollout_coverage_failures == 1
    assert result.rollout_coverage_reasons == (
        "seat_0:unsupported_chenzhou_wildcard_terminal",
    )
    assert {candidate.visits for candidate in result.candidates} == {0}


@pytest.mark.parametrize(
    ("players", "wildcard_enabled"),
    [
        (2, False),
        (2, True),
        (3, False),
        (3, True),
    ],
)
@pytest.mark.parametrize("action_type", ["CHI", "PENG"])
def test_response_search_preserves_invariants_in_all_room_profiles(
    players,
    wildcard_enabled,
    action_type,
):
    if action_type == "CHI":
        hand = ("二", "十", "四", "六", "九")
        pending = "七"
        source = players - 1
        claim = RootResponseCandidate(
            key="CHI:2710",
            action_type="CHI",
            heuristic_value=200.0,
            option_id="2710",
            consumed_from_hand=("二", "十"),
            meld_groups=(("二", "七", "十"),),
            followup_discard="九",
        )
    else:
        hand = ("二", "二", "四", "六", "九")
        pending = "二"
        source = 1
        claim = RootResponseCandidate(
            key="PENG:二",
            action_type="PENG",
            heuristic_value=200.0,
            consumed_from_hand=("二", "二"),
            meld_groups=(("二", "二", "二"),),
            followup_discard="九",
        )
    view, rules = _response_public_view(
        hand=hand,
        pending=pending,
        source=source,
        players=players,
        wildcard_enabled=wildcard_enabled,
    )
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
            rollout_max_turns=8,
        )
    )

    result = policy.search_response(
        view,
        rules=rules,
        candidates=(RootResponseCandidate("PASS", "PASS", 100.0), claim),
        force_search=True,
    )

    assert result.used_search
    assert result.simulations == 2
    assert result.determinization_failures == 0
    assert result.paired_determinizations == 1
    assert all(candidate.average_reward > -100.0 for candidate in result.candidates)


def test_response_search_can_compare_hu_against_passing():
    hand = (
        "七", "八", "九",
        "四", "肆",
        "陆", "柒", "捌",
        "贰", "柒", "拾",
        "肆", "伍", "陆",
        "六", "六", "六",
        "二", "三", "四",
    )
    view, rules = _response_public_view(
        hand=hand,
        pending="肆",
        source=2,
        players=3,
    )
    policy = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=5_000,
            max_iterations=2,
            max_candidates=2,
            rollout_max_turns=8,
            record_paired_worlds=True,
        )
    )

    result = policy.search_response(
        view,
        rules=rules,
        candidates=(
            RootResponseCandidate("PASS", "PASS", 0.0),
            RootResponseCandidate("HU", "HU", 0.0),
        ),
        force_search=True,
        preferred_key="PASS",
    )

    assert result.used_search
    assert result.simulations == 2
    assert result.paired_determinizations == 1
    assert {item.candidate.key: item.visits for item in result.candidates} == {
        "PASS": 1,
        "HU": 1,
    }
    assert len(result.paired_worlds) == 1
    hu_outcome = next(
        outcome
        for outcome in result.paired_worlds[0].outcomes
        if outcome.candidate_key == "HU"
    )
    assert hu_outcome.reason == "discard_hu"
    assert hu_outcome.winner == 0


def test_progressive_hu_response_excludes_chi_without_followup_discard():
    rules = rules_for_room(wildcard_enabled=False, players=2)
    hand = [
        "贰", "八", "十", "壹", "壹", "壹", "捌",
        "十", "十", "捌", "贰", "七", "捌", "贰",
    ]
    melds = [
        SimMeld("special_123", ("一", "二", "三")),
        SimMeld("mixed_same_rank_triplet", ("五", "伍", "伍")),
    ]
    players = [
        SimPlayer(seat=0, hand=list(hand), melds=list(melds)),
        SimPlayer(seat=1),
    ]
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(hand)
    for meld in melds:
        remaining.subtract(meld.cards)
    remaining.subtract(("九",))
    policy = ProfessionalProgressiveAllActionCandidatePolicy()
    simulator = FullGameSimulator(
        [policy, ProfessionalProgressiveAllActionCandidatePolicy()],
        wildcard_enabled=False,
        rules=rules,
        record_decisions=True,
    )

    result = simulator.play_from_pending_discard(
        seed=20370101,
        players=players,
        stock=expanded_deck(+remaining),
        discarder=1,
        pending_card="九",
        max_turns=0,
    )

    response = next(
        entry
        for entry in simulator.decision_trace()
        if entry["phase"] == "response_root"
    )
    assert {action["type"] for action in response["legal_actions"]} == {
        "PASS",
        "HU",
    }
    assert result.winner == 0
    assert result.reason == "discard_hu"
    assert not result.violations
    assert policy.response_search_contract_failures == 0
    assert policy.response_search_attempts == 1
    assert policy.response_search_usable == 1


def test_response_shadow_search_never_replaces_production_peng_or_chi(monkeypatch):
    policy = ProfessionalResponseShadowSimulationPolicy()

    def disagree_with_production(*args, **kwargs):
        return RootResponseSearchResult(
            selected_key="PASS",
            used_search=True,
            reason="test_forced_disagreement",
            simulations=4,
            elapsed_ms=12.5,
            candidates=(),
            paired_determinizations=2,
        )

    monkeypatch.setattr(policy.response_search, "search_response", disagree_with_production)
    rules = rules_for_room(wildcard_enabled=False, players=3)
    peng_view = PublicView(
        seat=0,
        hand=("二", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), (), ()),
        discards=((), (), ()),
        remaining_counts=(),
        stock_count=19,
        pending_card="二",
        pending_source_seat=2,
    )

    assert policy.choose_peng(peng_view, "二", rules)

    plan = ChiPlan(("二", "七", "十"), (), ("二", "十"))
    chi_view = PublicView(
        seat=0,
        hand=("二", "十", "四", "六", "九"),
        own_melds=(),
        all_melds=((), (), ()),
        discards=((), (), ()),
        remaining_counts=(),
        stock_count=19,
        pending_card="七",
        pending_source_seat=2,
    )

    assert policy.choose_chi(chi_view, [plan], rules) == plan
    assert policy.search_attempts == 0
    assert policy.response_search_attempts == 2
    assert policy.response_search_usable == 2
    assert policy.response_search_disagreements == 2
    assert all(event["disagreement"] for event in policy.response_events())


def test_response_shadow_captures_public_root_even_without_disagreement(monkeypatch):
    policy = ProfessionalResponseShadowSimulationPolicy()

    def agree_with_production(*args, **kwargs):
        return RootResponseSearchResult(
            selected_key="PENG:二",
            used_search=True,
            reason="test_agreement",
            simulations=4,
            elapsed_ms=12.5,
            candidates=(),
            paired_determinizations=2,
        )

    monkeypatch.setattr(policy.response_search, "search_response", agree_with_production)
    rules = rules_for_room(wildcard_enabled=False, players=3)
    view = PublicView(
        seat=0,
        hand=("二", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), (), ()),
        discards=((), (), ()),
        remaining_counts=(),
        stock_count=19,
        pending_card="二",
        pending_source_seat=2,
    )

    assert policy.choose_peng(view, "二", rules)

    event = policy.response_events()[-1]
    assert not event["disagreement"]
    assert event["public_view"] == public_view_to_dict(view)


def test_response_shadow_records_rollout_coverage_failure(monkeypatch):
    policy = ProfessionalResponseShadowSimulationPolicy()

    def unsupported_rollout(*args, **kwargs):
        return RootResponseSearchResult(
            selected_key="PENG:二",
            used_search=False,
            reason="rollout_coverage_incomplete",
            simulations=0,
            elapsed_ms=5.0,
            candidates=(),
            rollout_coverage_failures=1,
            rollout_coverage_reasons=(
                "seat_0:unsupported_chenzhou_wildcard_terminal",
            ),
        )

    monkeypatch.setattr(policy.response_search, "search_response", unsupported_rollout)
    rules = rules_for_room(wildcard_enabled=True, players=3)
    view = PublicView(
        seat=0,
        hand=("二", "二", "四", "六", "九", "王"),
        own_melds=(),
        all_melds=((), (), ()),
        discards=((), (), ()),
        remaining_counts=(),
        stock_count=23,
        pending_card="二",
        pending_source_seat=2,
    )

    assert policy.choose_peng(view, "二", rules)

    event = policy.response_events()[-1]
    assert not event["used_search"]
    assert event["reason"] == "rollout_coverage_incomplete"
    assert event["rollout_coverage_failures"] == 1
    assert event["rollout_coverage_reasons"] == [
        "seat_0:unsupported_chenzhou_wildcard_terminal",
    ]


def test_response_shortlist_always_keeps_production_candidate():
    candidates = [
        RootResponseCandidate(f"CHI:{index}", "CHI", 100.0 - index)
        for index in range(5)
    ]

    shortlist = shortlist_response_candidates(
        candidates,
        production_key="CHI:4",
        max_candidates=4,
    )

    assert len(shortlist) == 4
    assert {candidate.key for candidate in shortlist} == {
        "CHI:0",
        "CHI:1",
        "CHI:2",
        "CHI:4",
    }


def test_public_response_view_snapshot_round_trips_exactly():
    view, _rules = _response_public_view(
        hand=("二", "十", "四", "六", "九"),
        pending="七",
        source=2,
        players=3,
        wildcard_enabled=True,
    )
    view = replace(
        view,
        passed_chi=(("七",), (), ("二",)),
        passed_peng=((), ("陆",), ()),
    )

    restored = public_view_from_dict(public_view_to_dict(view))

    assert restored == view


def test_determinization_restores_public_passed_claim_state():
    view, rules = _response_public_view(
        hand=("二", "十", "四", "六", "九"),
        pending="七",
        source=1,
        players=2,
        wildcard_enabled=True,
    )
    view = replace(
        view,
        passed_chi=(("七",), ("二",)),
        passed_peng=(("陆",), ("玖",)),
    )

    sampled = determinize_public_view(view, rules=rules, rng=random.Random(17))

    assert sampled is not None
    players, _stock = sampled
    assert players[0].passed_chi == {"七"}
    assert players[0].passed_peng == {"陆"}
    assert players[1].passed_chi == {"二"}
    assert players[1].passed_peng == {"玖"}


def _initial_public_view(
    seed: int,
    *,
    players: int = 3,
    wildcard_enabled: bool = False,
) -> tuple[PublicView, dict]:
    rules = rules_for_room(
        wildcard_enabled=wildcard_enabled,
        players=players,
    )
    counts = full_deck_counts(rules=rules)
    deck = expanded_deck(counts)
    random.Random(seed).shuffle(deck)
    own_hand = tuple(deck[:21])
    remaining = counts.copy()
    remaining.subtract(own_hand)
    return (
        PublicView(
            seat=0,
            hand=own_hand,
            own_melds=(),
            all_melds=tuple(() for _ in range(players)),
            discards=tuple(() for _ in range(players)),
            remaining_counts=tuple(sorted((+remaining).items())),
            stock_count=len(deck) - (players * 20 + 1),
            hand_sizes=(21, *(20 for _ in range(players - 1))),
        ),
        rules,
    )


def _response_public_view(
    *,
    hand: tuple[str, ...],
    pending: str,
    source: int,
    players: int = 3,
    wildcard_enabled: bool = False,
) -> tuple[PublicView, dict]:
    rules = rules_for_room(
        wildcard_enabled=wildcard_enabled,
        players=players,
    )
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(hand)
    remaining.subtract((pending,))
    opponent_size = 5
    stock_count = sum(remaining.values()) - opponent_size * (players - 1)
    return (
        PublicView(
            seat=0,
            hand=hand,
            own_melds=(),
            all_melds=tuple(() for _ in range(players)),
            discards=tuple(() for _ in range(players)),
            remaining_counts=tuple(sorted((+remaining).items())),
            stock_count=stock_count,
            hand_sizes=(len(hand), *(opponent_size for _ in range(players - 1))),
            pending_card=pending,
            pending_source_seat=source,
        ),
        rules,
    )


class _AlwaysPengPolicy(BaselinePolicy):
    def choose_peng(self, view, label, rules):
        return True


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
