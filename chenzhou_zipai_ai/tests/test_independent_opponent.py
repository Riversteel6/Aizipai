"""Structural and behavioral checks for the independent validation opponent."""

from __future__ import annotations

import ast
import random
from collections import Counter
from pathlib import Path

import pytest

from audit.independent_opponent import (
    _shape_draw_gain,
    _shape_improving_draws,
    _shape_removal_loss,
    _shape_score,
)
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from audit.independent_rules import NORMAL_LABELS, WILD_LABEL
from ai.full_game_simulator import HuEvaluation, PublicView, SimMeld
from ai.ismcts import RootISMCTSConfig, RootISMCTSPolicy
from ai.opponent_league import create_policy, run_opponent_league
from engine.deck import expanded_deck, full_deck_counts
from engine.rules import rules_for_room


SOURCE = Path(__file__).resolve().parents[1] / "audit" / "independent_opponent.py"
FORBIDDEN_IMPORTS = {
    "ai.evaluator",
    "ai.features",
    "ai.ismcts",
    "ai.monte_carlo",
    "ai.pro_brain",
    "ai.search_brain",
}
FORBIDDEN_NAMES = {
    "InformationSetSearchPolicy",
    "ProfessionalBrainSimulationPolicy",
    "_position_value",
    "quick_potential",
}


def _view(*, wildcard_enabled: bool = False) -> tuple[PublicView, dict]:
    rules = rules_for_room(wildcard_enabled=wildcard_enabled, players=2)
    hand = (
        "一", "一", "二", "三", "四", "五", "六",
        "七", "八", "九", "十", "壹", "贰", "叁",
        "肆", "伍", "陆", "柒", "捌", "玖",
    )
    if wildcard_enabled:
        hand = (*hand[:-1], "王")
    remaining = full_deck_counts(rules=rules)
    for label in hand:
        remaining[label] -= 1
    return (
        PublicView(
            seat=0,
            hand=hand,
            own_melds=(),
            all_melds=((), ()),
            discards=((), ()),
            remaining_counts=tuple(sorted(remaining.items())),
            stock_count=sum(remaining.values()),
            hand_sizes=(len(hand), 20),
        ),
        rules,
    )


def test_independent_opponent_does_not_import_production_value_modules():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Name):
            names.add(node.id)

    assert not imports.intersection(FORBIDDEN_IMPORTS)
    assert not names.intersection(FORBIDDEN_NAMES)


def test_independent_variants_choose_legal_unlocked_discards_and_keep_wang():
    for wildcard_enabled in (False, True):
        view, rules = _view(wildcard_enabled=wildcard_enabled)
        for name in (
            "independent_balanced",
            "independent_pressure",
            "independent_denial",
            "independent_exploit_untrained",
        ):
            label = create_policy(name).choose_discard(view, rules)
            assert label in view.hand
            assert view.hand.count(label) < 3
            assert label != "王"


def test_independent_policy_accepts_a_legal_hu():
    policy = create_policy("independent_balanced")
    view, rules = _view()
    hu = HuEvaluation(True, 9, (), 1.0, 0)

    assert policy.choose_hu(view, hu, rules)


def test_fast_shape_draw_delta_matches_full_independent_shape_value():
    rng = random.Random(20260825)
    rules = rules_for_room(wildcard_enabled=True, players=2)
    labels = (*NORMAL_LABELS, WILD_LABEL)
    for _case in range(100):
        hand = tuple(rng.choice(labels) for _card in range(20))
        draw = rng.choice(labels)
        counts = Counter(label for label in hand if label != WILD_LABEL)
        expected = _shape_score((*hand, draw), rules) - _shape_score(hand, rules)

        actual = _shape_draw_gain(counts, draw, allow_1510=False)

        assert actual == expected


def test_fast_shape_improving_draws_matches_individual_draw_deltas():
    rng = random.Random(20270215)
    rules = rules_for_room(wildcard_enabled=True, players=2)
    labels = (*NORMAL_LABELS, WILD_LABEL)
    for _case in range(100):
        hand = tuple(rng.choice(labels) for _card in range(20))
        remaining = tuple(
            (label, rng.randrange(5))
            for label in labels
        )
        counts = Counter(label for label in hand if label != WILD_LABEL)
        expected_outs = 0
        expected_gain = 0.0
        for label, amount in remaining:
            gain = _shape_draw_gain(counts, label, allow_1510=False)
            if gain > 0:
                expected_outs += amount
                expected_gain += amount * gain

        assert _shape_improving_draws(hand, remaining, rules) == (
            expected_outs,
            expected_gain,
        )


def test_fast_shape_removal_delta_matches_full_shape_recalculation():
    rng = random.Random(20270216)
    rules = rules_for_room(wildcard_enabled=True, players=2)
    for _case in range(100):
        hand = tuple(rng.choice(NORMAL_LABELS) for _card in range(20))
        counts = tuple(hand.count(label) for label in NORMAL_LABELS)
        base_score = _shape_score(hand, rules)
        for label in set(hand):
            after = list(hand)
            after.remove(label)
            expected = base_score - _shape_score(after, rules)

            actual = _shape_removal_loss(
                counts,
                label,
                allow_1510=False,
            )

            assert actual == expected


def test_independent_policy_respects_locked_triplet_on_discard():
    view, rules = _view()
    view = PublicView(
        **{
            **view.__dict__,
            "hand": ("八", "八", "八", *view.hand[3:]),
        }
    )

    assert create_policy("independent_balanced").choose_discard(view, rules) != "八"


def test_exposed_independent_discard_scores_preserve_teacher_selection():
    view, rules = _view()
    for name in (
        "independent_balanced",
        "independent_pressure",
        "independent_denial",
    ):
        policy = create_policy(name)
        selected = policy.choose_discard(view, rules)
        scores = policy.discard_scores(view, rules)
        scored_selected = max(
            scores,
            key=lambda label: (scores[label][0], label),
        )
        all_scores = policy.discard_scores(
            view,
            rules,
            all_candidates=True,
        )

        assert scored_selected == selected
        assert set(scores).issubset(all_scores)
        assert selected in all_scores


def test_independent_opponent_league_smoke_is_legal():
    report = run_opponent_league(
        candidate="baseline",
        matchups=(("independent_balanced",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=20260822,
        workers=1,
        players=2,
    )

    assert report["games"] == 4
    assert report["invariant_violations"] == 0
    assert report["coverage_failures"] == 0


@pytest.mark.parametrize(
    ("players", "wildcard_enabled"),
    ((2, False), (2, True), (3, False), (3, True)),
)
def test_fast_independent_rollouts_finish_paired_root_search_within_budget(
    players,
    wildcard_enabled,
):
    rules = rules_for_room(
        wildcard_enabled=wildcard_enabled,
        players=players,
    )
    counts = full_deck_counts(rules=rules)
    deck = expanded_deck(counts)
    random.Random(20260827 + players + int(wildcard_enabled)).shuffle(deck)
    hand = tuple(deck[:21])
    remaining = counts.copy()
    remaining.subtract(hand)
    view = PublicView(
        seat=0,
        hand=hand,
        own_melds=(),
        all_melds=tuple(() for _ in range(players)),
        discards=tuple(() for _ in range(players)),
        remaining_counts=tuple(sorted((+remaining).items())),
        stock_count=len(deck) - (players * 20 + 1),
        hand_sizes=(21, *(20 for _ in range(players - 1))),
    )
    candidates = sorted(
        label
        for label, amount in Counter(hand).items()
        if label != WILD_LABEL and amount < 3
    )[:2]
    search = RootISMCTSPolicy(
        RootISMCTSConfig(
            time_budget_ms=2_500,
            max_iterations=4,
            max_candidates=2,
            skip_search_gap=10_000,
            rollout_max_turns=120,
        ),
        rollout_policy_factories=(
            IndependentFastRolloutPolicy,
            IndependentFastPressurePolicy,
            IndependentFastDenialPolicy,
        ),
    )

    result = search.search_discard(
        view,
        rules=rules,
        candidate_labels=candidates,
        force_search=True,
        paired_candidates=True,
        preferred_label=candidates[0],
    )

    assert result.used_search
    assert result.simulations == 4
    assert result.paired_determinizations == 2
    assert result.deadline_interruptions == 0
    assert result.rollout_invariant_violations == 0
    assert result.rollout_coverage_failures == 0
    assert result.elapsed_ms <= 2_500
