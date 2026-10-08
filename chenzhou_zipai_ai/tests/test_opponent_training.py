"""Counter-strategy scheduling, isolation, and selection tests."""

from ai.opponent_training import (
    DEFAULT_EXPLOIT_PROFILES,
    ExploitJob,
    _selection_key,
    _target_policy,
    build_exploit_jobs,
    summarize_exploit_rows,
)
from ai.opponent_league import ProductDecisionKernelSimulationPolicy


def test_exploit_schedule_balances_all_four_modes_seats_and_dealers():
    jobs = build_exploit_jobs(
        profiles=(DEFAULT_EXPLOIT_PROFILES[0],),
        target="professional_brain",
        deals_per_mode=1,
        seed=1000,
        split="train",
    )

    assert len(jobs) == 26
    by_mode = {}
    for job in jobs:
        by_mode.setdefault((job.players, job.wildcard_enabled), set()).add(
            (job.candidate_seat, job.dealer)
        )
    assert by_mode[(2, False)] == {
        (seat, dealer) for seat in range(2) for dealer in range(2)
    }
    assert by_mode[(2, True)] == by_mode[(2, False)]
    assert by_mode[(3, False)] == {
        (seat, dealer) for seat in range(3) for dealer in range(3)
    }
    assert by_mode[(3, True)] == by_mode[(3, False)]


def test_exploit_schedule_can_scope_to_two_player_wang_only():
    jobs = build_exploit_jobs(
        profiles=(DEFAULT_EXPLOIT_PROFILES[0],),
        target="professional_v81_two_player_exact_discard_sharded_research",
        deals_per_mode=1,
        seed=1000,
        split="train",
        modes=((2, True),),
    )

    assert len(jobs) == 4
    assert {(job.players, job.wildcard_enabled) for job in jobs} == {(2, True)}


def test_two_player_frozen_exploit_target_uses_product_kernel():
    job = ExploitJob(
        profile=DEFAULT_EXPLOIT_PROFILES[0],
        target="professional_v81_two_player_exact_discard_sharded_research",
        players=2,
        wildcard_enabled=True,
        seed=1,
        candidate_seat=0,
        dealer=0,
        split="test",
    )

    assert isinstance(_target_policy(job), ProductDecisionKernelSimulationPolicy)


def test_exploit_selection_prioritizes_worst_mode_before_aggregate():
    stable = {
        "worst_mode_win_share_all": 0.45,
        "candidate_win_share_all": 0.50,
        "worst_mode_mean_score": -0.2,
        "mean_candidate_outcome_score": 0.1,
    }
    brittle = {
        "worst_mode_win_share_all": 0.30,
        "candidate_win_share_all": 0.70,
        "worst_mode_mean_score": -2.0,
        "mean_candidate_outcome_score": 3.0,
    }

    assert _selection_key(stable) > _selection_key(brittle)


def test_exploit_summary_keeps_modes_separate():
    rows = [
        {
            "players": 2,
            "wildcard_enabled": False,
            "candidate_won": True,
            "draw": False,
            "candidate_outcome_score": 2.0,
            "invariant_violations": 0,
            "coverage_failures": 0,
        },
        {
            "players": 3,
            "wildcard_enabled": True,
            "candidate_won": False,
            "draw": False,
            "candidate_outcome_score": -1.0,
            "invariant_violations": 0,
            "coverage_failures": 0,
        },
    ]

    summary = summarize_exploit_rows(rows)

    assert summary["games"] == 2
    assert summary["candidate_win_share_all"] == 0.5
    assert summary["worst_mode_win_share_all"] == 0.0
    assert set(summary["by_mode"]) == {"2p_no_wang", "3p_wang"}
