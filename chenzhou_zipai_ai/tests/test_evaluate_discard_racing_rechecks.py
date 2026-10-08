"""Selection-rule checks for offline discard racing evaluation."""

from unittest.mock import patch

from ai.ismcts import (
    PairedAdvantageStats,
    RootCandidateStats,
    RootSearchResult,
)
from ai.dual_discard_validator import _merge_parallel_coverage
from tools.evaluate_discard_racing_rechecks import (
    _apply_split_selection_rule,
    _split_dual_search,
    _dual_confirmation_override,
)


def _result(
    labels_and_rewards,
    *,
    preferred="一",
    challenger="二",
    mean_delta=0.05,
    lower_bound=-0.05,
):
    candidates = tuple(
        RootCandidateStats(
            label=label,
            visits=24,
            reward_sum=reward * 24,
            average_reward=reward,
            win_rate=0.0,
            heuristic_value=0.0,
        )
        for label, reward in labels_and_rewards
    )
    return RootSearchResult(
        selected_label=challenger,
        used_search=True,
        reason="test",
        simulations=48,
        elapsed_ms=1.0,
        candidates=candidates,
        paired_advantages=(
            PairedAdvantageStats(
                candidate_key=challenger,
                preferred_key=preferred,
                samples=384,
                mean_delta=mean_delta,
                sample_stddev=1.0,
                standard_error=0.05,
                lower_confidence_bound=lower_bound,
                upper_confidence_bound=mean_delta + 0.1,
                positive_samples=150,
                tied_samples=100,
                negative_samples=134,
            ),
        ),
    )


def _select(screening, confirmation, *, labels):
    return _apply_split_selection_rule(
        screening,
        confirmation,
        labels=labels,
        preferred="一",
        challenger="二",
        config={
            "split_selection_rule": "guarded_consensus",
            "minimum_confident_advantage": 0.02,
        },
    )


def test_guarded_split_overrides_on_strong_confirmation():
    labels = ["一", "二", "三", "四"]
    screening = _result(
        [("一", 0.4), ("二", 0.3), ("三", 0.2), ("四", 0.1)]
    )
    confirmation = _result(
        [("二", 0.6), ("一", 0.4)],
        lower_bound=0.03,
    )

    result = _select(screening, confirmation, labels=labels)

    assert result.selected_label == "二"
    assert result.confidence_override


def test_guarded_split_allows_independent_landslide_consensus():
    labels = ["一", "二", "三", "四", "五", "六"]
    screening = _result(
        [("二", 0.5), ("三", 0.4), ("四", 0.3), ("一", 0.2)]
    )
    confirmation = _result(
        [("二", 0.4), ("一", 0.35)],
        mean_delta=0.05,
        lower_bound=-0.05,
    )

    result = _select(screening, confirmation, labels=labels)

    assert result.selected_label == "二"
    assert not result.confidence_override


def test_guarded_split_keeps_preferred_in_screening_top_half():
    labels = ["一", "二", "三", "四", "五", "六"]
    screening = _result(
        [("二", 0.5), ("一", 0.2), ("三", 0.1), ("四", 0.0)]
    )
    confirmation = _result(
        [("二", 0.4), ("一", 0.35)],
        mean_delta=0.05,
        lower_bound=-0.05,
    )

    result = _select(screening, confirmation, labels=labels)

    assert result.selected_label == "一"
    assert not result.confidence_override


def test_dual_confirmation_requires_repeatable_and_pooled_confident_gain():
    first = _result(
        [("二", 0.4), ("一", 0.3)],
        mean_delta=0.1,
    )
    second = _result(
        [("二", 0.35), ("一", 0.3)],
        mean_delta=0.05,
    )
    pooled = _result(
        [("二", 0.375), ("一", 0.3)],
        mean_delta=0.075,
        lower_bound=0.01,
    )

    assert _dual_confirmation_override(
        first,
        second,
        pooled,
        challenger="二",
    )

    negative_second = _result(
        [("一", 0.35), ("二", 0.3)],
        mean_delta=-0.05,
    )
    assert not _dual_confirmation_override(
        first,
        negative_second,
        pooled,
        challenger="二",
    )

    unconfirmed_pool = RootSearchResult(
        **{
            **pooled.__dict__,
            "selected_label": "一",
        }
    )
    assert not _dual_confirmation_override(
        first,
        second,
        unconfirmed_pool,
        challenger="二",
    )


def test_dual_confirmation_stops_after_first_rejection():
    screening = _result(
        [("二", 0.4), ("一", 0.3)],
        mean_delta=0.1,
    )
    first = _result(
        [("一", 0.35), ("二", 0.3)],
        mean_delta=-0.05,
    )
    with patch(
        "tools.evaluate_discard_racing_rechecks._search",
        side_effect=(screening, first),
    ) as search:
        coverage, confirmation, combined = _split_dual_search(
            object(),
            rules={},
            labels=["一", "二"],
            priors={"一": 0.3, "二": 0.4},
            preferred="一",
            config={
                "coverage_worlds": 24,
                "dual_confirmation_worlds": 192,
                "time_budget_ms": 30_000,
            },
        )

    assert search.call_count == 2
    assert coverage is screening
    assert confirmation is first
    assert combined.selected_label == "一"
    assert combined.reason == "split_pair_dual_first_batch_rejected_challenger"


def test_parallel_screening_merge_preserves_all_candidate_statistics():
    first = _result([("一", 0.1), ("三", 0.4)])
    second = _result([("二", 0.3), ("四", 0.2)])

    merged = _merge_parallel_coverage(
        [first, second],
        preferred="一",
        elapsed_ms=12.5,
    )

    assert [item.label for item in merged.candidates] == [
        "三",
        "二",
        "四",
        "一",
    ]
    assert merged.selected_label == "三"
    assert merged.simulations == first.simulations + second.simulations
    assert merged.paired_determinizations == min(
        first.paired_determinizations,
        second.paired_determinizations,
    )
    assert merged.elapsed_ms == 12.5
