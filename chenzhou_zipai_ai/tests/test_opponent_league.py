"""Opponent league scheduling and policy-contract tests."""

from types import SimpleNamespace

import pytest

import ai.opponent_league as opponent_league_module
from ai.full_game_simulator import PublicView, SimMeld
from ai.ismcts import RootCandidateStats, RootResponseCandidate, RootSearchResult
from ai.opponent_league import (
    LeagueJob,
    POLICY_FACTORIES,
    ProductDecisionKernelSimulationPolicy,
    _candidate_policy_for_job,
    _summarize_candidate_diagnostics,
    _summarize_discard_shadow,
    _summarize_response_shadow,
    _wildcard_utilization_quality,
    _wildcard_route_bonus,
    build_league_jobs,
    create_policy,
    run_opponent_league,
)
from engine.chi_rules import enumerate_chi_plans
from engine.rules import rules_for_room
from tools.run_opponent_league import (
    _LeagueCheckpoint,
    _league_checkpoint_metadata,
    build_dominated_matchup_early_stop_callback,
)


def _fake_league_row(job, *, won=True):
    return {
        "candidate": job.candidate,
        "opponents": sorted(job.opponents),
        "players": job.players,
        "candidate_seat": job.candidate_seat,
        "dealer": job.dealer,
        "seed": job.seed,
        "winner": job.candidate_seat if won else None,
        "winner_policy": job.candidate if won else None,
        "candidate_won": won,
        "draw": not won,
        "candidate_outcome_score": 1.0 if won else 0.0,
        "candidate_diagnostics": {},
        "candidate_response_events": [],
        "candidate_discard_events": [],
        "result": {
            "violations": [],
            "coverage_failures": [],
        },
    }


def test_formal_two_player_v81_job_uses_product_decision_kernel() -> None:
    job = LeagueJob(
        candidate="professional_v81_two_player_exact_discard_sharded_research",
        opponents=("independent_balanced",),
        candidate_seat=1,
        dealer=0,
        seed=17,
        wildcard_enabled=True,
        players=2,
    )

    policy = _candidate_policy_for_job(job)

    assert isinstance(policy, ProductDecisionKernelSimulationPolicy)
    assert policy.name == "product_decision_kernel_v1"


def test_non_product_league_job_keeps_direct_policy_factory() -> None:
    job = LeagueJob(
        candidate="professional_brain",
        opponents=("independent_balanced",),
        candidate_seat=0,
        dealer=0,
        seed=17,
        wildcard_enabled=True,
        players=2,
    )

    assert not isinstance(
        _candidate_policy_for_job(job),
        ProductDecisionKernelSimulationPolicy,
    )


def test_product_chi_mapping_accepts_search_override_of_production_gate() -> None:
    plan = SimpleNamespace(
        initial_group=("一", "二", "三"),
        compare_groups=(("一", "一", "壹"),),
        consumed_from_hand=("二", "三", "一", "一", "壹"),
        groups=(("一", "二", "三"), ("一", "一", "壹")),
    )
    decision = SimpleNamespace(
        selected_option_id="sim_chi_003",
        action_evals=[
            SimpleNamespace(
                action=SimpleNamespace(type="CHI", option_id="sim_chi_003"),
                allowed=False,
                reject_reason="chi_ev_not_enough",
                debug_details={
                    "consumed_from_hand": ["壹", "一", "三", "一", "二"],
                    "meld_cards": ["三", "一", "二"],
                    "compare_groups": [["壹", "一", "一"]],
                },
            )
        ],
    )

    selected = ProductDecisionKernelSimulationPolicy._product_chi_plan_for_decision(
        decision,
        [plan],
    )

    assert selected is plan


def test_candidate_diagnostics_sum_numbers_and_preserve_product_route() -> None:
    summary = _summarize_candidate_diagnostics(
        [
            {
                "candidate_diagnostics": {
                    "formal_route": "product_decision_kernel",
                    "product_discard_events": 2,
                }
            },
            {
                "candidate_diagnostics": {
                    "formal_route": "product_decision_kernel",
                    "product_discard_events": 3,
                }
            },
        ]
    )

    assert summary["formal_route"] == "product_decision_kernel"
    assert summary["product_discard_events"] == 5

def test_exact_sharded_v81_skips_only_complete_strong_agreement_confirmation():
    candidate = create_policy(
        "professional_v81_two_player_exact_discard_sharded_research"
    )
    frozen = create_policy(
        "professional_v81_two_player_wang_ting_guard_research"
    )
    baseline = RootSearchResult(
        selected_label="五",
        used_search=True,
        reason="complete",
        simulations=10,
        elapsed_ms=1.0,
        candidates=(),
    )
    coverage = RootSearchResult(
        selected_label="五",
        used_search=True,
        reason="complete",
        simulations=96,
        elapsed_ms=1.0,
        candidates=(
            RootCandidateStats("五", 48, 9.6, 0.2, 0.6, 10.0),
            RootCandidateStats("陆", 48, 4.8, 0.1, 0.5, 9.0),
            RootCandidateStats("壹", 48, 2.4, 0.05, 0.4, 8.0),
        ),
        paired_determinizations=48,
        empirical_best_label="五",
    )
    screened = SimpleNamespace(coverage=coverage)

    assert candidate._should_skip_discard_confirmation(
        baseline=baseline,
        screened=screened,
    )
    assert not frozen._should_skip_discard_confirmation(
        baseline=baseline,
        screened=screened,
    )
    weak = SimpleNamespace(
        coverage=RootSearchResult(
            selected_label="五",
            used_search=True,
            reason="complete",
            simulations=96,
            elapsed_ms=1.0,
            candidates=(
                RootCandidateStats("五", 48, 6.0, 0.125, 0.6, 10.0),
                RootCandidateStats("陆", 48, 4.8, 0.1, 0.5, 9.0),
            ),
            paired_determinizations=48,
            empirical_best_label="五",
        )
    )
    assert not candidate._should_skip_discard_confirmation(
        baseline=baseline,
        screened=weak,
    )


def test_strong_challenger_agreement_requires_production_anchored_confirmation():
    candidate = create_policy(
        "professional_v81_two_player_exact_discard_sharded_research"
    )
    baseline = RootSearchResult(
        selected_label="五",
        used_search=True,
        reason="complete",
        simulations=10,
        elapsed_ms=1.0,
        candidates=(),
    )
    coverage = RootSearchResult(
        selected_label="五",
        used_search=True,
        reason="complete",
        simulations=96,
        elapsed_ms=1.0,
        candidates=(
            RootCandidateStats("五", 48, 9.6, 0.2, 0.6, 10.0),
            RootCandidateStats("陆", 48, 4.8, 0.1, 0.5, 9.0),
            RootCandidateStats("壹", 48, 2.4, 0.05, 0.4, 8.0),
        ),
        paired_determinizations=48,
        empirical_best_label="五",
    )
    screened = SimpleNamespace(
        coverage=coverage,
        labels=("五", "陆", "壹"),
    )

    assert candidate._discard_confirmation_skip_reason(
        production_label="壹",
        baseline=baseline,
        screened=screened,
    ) is None
    assert candidate._discard_confirmation_skip_reason(
        production_label="五",
        baseline=baseline,
        screened=screened,
    ) == "strong_agreement_confirmation_skipped"


def test_league_reports_progress_for_each_completed_job(monkeypatch):
    def fake_play(job):
        return {
            "candidate": job.candidate,
            "opponents": sorted(job.opponents),
            "players": job.players,
            "candidate_seat": job.candidate_seat,
            "dealer": job.dealer,
            "seed": job.seed,
            "winner": job.candidate_seat,
            "winner_policy": job.candidate,
            "candidate_won": True,
            "draw": False,
            "candidate_outcome_score": 1.0,
            "candidate_diagnostics": {},
            "candidate_response_events": [],
            "candidate_discard_events": [],
            "result": {
                "violations": [],
                "coverage_failures": [],
            },
        }

    monkeypatch.setattr(
        opponent_league_module,
        "_play_league_job",
        fake_play,
    )
    progress = []

    report = run_opponent_league(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        workers=1,
        players=2,
        progress_callback=lambda completed, total, row: progress.append(
            (completed, total, row["seed"])
        ),
    )

    assert report["games"] == 4
    assert progress == [
        (1, 4, 500),
        (2, 4, 500),
        (3, 4, 500),
        (4, 4, 500),
    ]


def test_league_stops_after_completed_dominated_matchup(monkeypatch):
    matchups = (("baseline",), ("aggressive_meld",))
    jobs = build_league_jobs(
        candidate="baseline",
        matchups=matchups,
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        players=2,
    )
    baseline_rows = []
    for job in jobs:
        row = _fake_league_row(job)
        row["candidate_outcome_score"] = 2.0
        baseline_rows.append(row)
    baseline_report = {
        "players": 2,
        "wildcard_enabled": False,
        "seed": 500,
        "deals_per_matchup": 1,
        "balanced_rotation_only": False,
        "rows": baseline_rows,
    }
    stop_callback = build_dominated_matchup_early_stop_callback(
        baseline_report,
        players=2,
        wildcard_enabled=False,
        seed=500,
        deals_per_matchup=1,
        balanced_rotation_only=False,
        expected_games=8,
    )
    played = []

    def fake_play(job):
        played.append(job)
        row = _fake_league_row(job, won=False)
        row["draw"] = False
        row["candidate_outcome_score"] = -1.0
        return row

    monkeypatch.setattr(
        opponent_league_module,
        "_play_league_job",
        fake_play,
    )
    report = run_opponent_league(
        candidate="baseline",
        matchups=matchups,
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        workers=1,
        players=2,
        early_stop_callback=stop_callback,
    )

    assert len(played) == 4
    assert report["games"] == 4
    assert report["expected_games"] == 8
    assert report["early_stopped"]
    assert not report["ok"]
    assert report["early_stop"]["reason"] == "paired_matchup_dominated"
    assert report["early_stop"]["outcome_utility_delta"] == -8
    assert report["early_stop"]["total_score_delta"] == -12.0


def test_league_resume_runs_only_jobs_after_validated_prefix(monkeypatch):
    jobs = build_league_jobs(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        players=2,
    )
    played = []

    def fake_play(job):
        played.append((job.candidate_seat, job.dealer, job.seed))
        return _fake_league_row(job)

    monkeypatch.setattr(
        opponent_league_module,
        "_play_league_job",
        fake_play,
    )
    progress = []
    report = run_opponent_league(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        workers=1,
        players=2,
        completed_rows=[
            _fake_league_row(jobs[0]),
            _fake_league_row(jobs[1]),
        ],
        progress_callback=lambda completed, total, row: progress.append(
            (completed, total, row["candidate_seat"], row["dealer"])
        ),
    )

    assert report["games"] == 4
    assert report["resumed_games"] == 2
    assert played == [
        (jobs[2].candidate_seat, jobs[2].dealer, jobs[2].seed),
        (jobs[3].candidate_seat, jobs[3].dealer, jobs[3].seed),
    ]
    assert progress == [
        (3, 4, jobs[2].candidate_seat, jobs[2].dealer),
        (4, 4, jobs[3].candidate_seat, jobs[3].dealer),
    ]


def test_league_resume_rejects_row_that_is_not_exact_schedule_prefix():
    jobs = build_league_jobs(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        players=2,
    )

    with pytest.raises(
        ValueError,
        match="completed_league_row_mismatch",
    ):
        run_opponent_league(
            candidate="baseline",
            matchups=(("baseline",),),
            deals_per_matchup=1,
            wildcard_enabled=False,
            seed=500,
            workers=1,
            players=2,
            completed_rows=[_fake_league_row(jobs[1])],
        )


def test_league_checkpoint_recovers_torn_tail_and_continues(tmp_path):
    path = tmp_path / "league.checkpoint.jsonl"
    metadata = _league_checkpoint_metadata(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        players=2,
        record_decisions=False,
        balanced_rotation_only=False,
        total=4,
    )
    jobs = build_league_jobs(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        players=2,
    )
    checkpoint = _LeagueCheckpoint(path)
    checkpoint.create(metadata)
    checkpoint.append_row(1, _fake_league_row(jobs[0]))
    with path.open("ab") as handle:
        handle.write(b'{"record_type":"row","index":2')

    resumed = _LeagueCheckpoint(path)
    rows = resumed.load(metadata)
    resumed.append_row(2, _fake_league_row(jobs[1]))

    assert len(rows) == 1
    assert resumed.row_count == 2
    assert path.read_bytes().endswith(b"\n")


def test_league_checkpoint_compacts_traces_in_memory_but_streams_full_rows(
    tmp_path,
):
    path = tmp_path / "league.checkpoint.jsonl"
    metadata = _league_checkpoint_metadata(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        players=2,
        record_decisions=True,
        balanced_rotation_only=False,
        total=4,
    )
    jobs = build_league_jobs(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        players=2,
        record_decisions=True,
    )
    row = _fake_league_row(jobs[0])
    row["decision_trace"] = [{"sequence": 1, "phase": "discard"}]
    checkpoint = _LeagueCheckpoint(path)
    checkpoint.create(metadata)
    checkpoint.append_row(1, row)

    resumed = _LeagueCheckpoint(path)
    compact = resumed.load(metadata, compact_decision_traces=True)

    assert "decision_trace" not in compact[0]
    assert compact[0]["_decision_trace_checkpointed"] is True
    streamed = list(resumed.iter_rows())
    assert streamed[0]["decision_trace"] == [
        {"sequence": 1, "phase": "discard"}
    ]


def test_league_checkpoint_rejects_configuration_drift(tmp_path):
    path = tmp_path / "league.checkpoint.jsonl"
    metadata = _league_checkpoint_metadata(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
        players=2,
        record_decisions=False,
        balanced_rotation_only=False,
        total=4,
    )
    _LeagueCheckpoint(path).create(metadata)
    changed = dict(metadata)
    changed["seed"] = 501

    with pytest.raises(
        ValueError,
        match="checkpoint_configuration_mismatch:seed",
    ):
        _LeagueCheckpoint(path).load(changed)


def test_parallel_dual_candidate_uses_nondaemon_outer_executor(
    monkeypatch,
):
    calls = {}

    class FakeProcessPoolExecutor:
        def __init__(self, *, max_workers, mp_context):
            calls["max_workers"] = max_workers
            calls["start_method"] = mp_context.get_start_method()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, jobs, *, chunksize):
            calls["chunksize"] = chunksize
            return [function(job) for job in jobs]

        def shutdown(self, *, wait, cancel_futures):
            calls["shutdown"] = (wait, cancel_futures)

    monkeypatch.setattr(
        opponent_league_module,
        "ProcessPoolExecutor",
        FakeProcessPoolExecutor,
    )
    monkeypatch.setattr(
        opponent_league_module,
        "_play_league_job",
        _fake_league_row,
    )

    report = run_opponent_league(
        candidate="professional_parallel_dual_validated_candidate",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        workers=2,
        players=2,
    )

    assert report["ok"]
    assert report["games"] == 4
    assert report["effective_workers"] == 2
    assert report["worker_model"] == "spawn_process_executor"
    assert calls == {
        "max_workers": 2,
        "start_method": "spawn",
        "chunksize": 1,
        "shutdown": (True, False),
    }


def test_nested_search_outer_workers_are_cpu_limited(monkeypatch):
    monkeypatch.setattr(
        opponent_league_module.os,
        "cpu_count",
        lambda: 32,
    )

    assert opponent_league_module._effective_league_workers(
        candidate=(
            "professional_parallel_multi_opponent_robust_v8_research"
        ),
        requested=32,
        remaining_jobs=98,
    ) == 2
    assert opponent_league_module._effective_league_workers(
        candidate="professional_brain",
        requested=32,
        remaining_jobs=98,
    ) == 32
    assert opponent_league_module._effective_league_workers(
        candidate=(
            "professional_v81_two_player_exact_discard_sharded_research"
        ),
        requested=2,
        remaining_jobs=98,
    ) == 2


def test_league_marks_unexpected_validation_error_as_failed(
    monkeypatch,
):
    def fake_play(job):
        row = _fake_league_row(job)
        row["candidate_discard_events"] = [
            {
                "search_attempted": True,
                "used_search": False,
                "validation_complete": False,
                "validation_error": (
                    "AssertionError:daemonic processes are not allowed "
                    "to have children"
                ),
            }
        ]
        return row

    monkeypatch.setattr(
        opponent_league_module,
        "_play_league_job",
        fake_play,
    )

    report = run_opponent_league(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        workers=1,
        players=2,
    )

    assert not report["ok"]
    assert report["strategy_runtime_errors"] == 4
    assert report["strategy_runtime_error_reasons"] == {
        (
            "AssertionError:daemonic processes are not allowed "
            "to have children"
        ): 4
    }


def test_current_parallel_candidate_preserves_frozen_v32_factory():
    current = create_policy(
        "professional_parallel_multi_validated_candidate"
    )
    explicit_v4 = create_policy(
        "professional_parallel_multi_calibrated_candidate_v4"
    )
    frozen_v32 = create_policy(
        "professional_parallel_multi_validated_candidate_v3_2"
    )
    proxy_research = create_policy(
        "professional_parallel_multi_search_proxy_research"
    )
    sparse_proxy_research = create_policy(
        "professional_parallel_multi_sparse_proxy_research"
    )
    calibrated_v5_research = create_policy(
        "professional_parallel_multi_sparse_proxy_calibrated_v5_research"
    )
    calibrated_v6_research = create_policy(
        "professional_parallel_multi_sparse_proxy_calibrated_v6_research"
    )
    calibrated_v7_research = create_policy(
        "professional_parallel_multi_sparse_proxy_calibrated_v7_research"
    )
    opponent_robust_v8_research = create_policy(
        "professional_parallel_multi_opponent_robust_v8_research"
    )
    opponent_robust_v81_research = create_policy(
        "professional_parallel_multi_opponent_robust_v8_1_research"
    )
    two_player_response_gate = create_policy(
        "professional_v81_two_player_response_gate_014_research"
    )
    two_player_wang_ting_guard = create_policy(
        "professional_v81_two_player_wang_ting_guard_research"
    )
    opponent_robust_v82_research = create_policy(
        "professional_parallel_multi_opponent_robust_v8_2_research"
    )
    opponent_robust_v83_research = create_policy(
        "professional_parallel_multi_opponent_robust_v8_3_research"
    )

    assert current.name == (
        "professional_parallel_multi_calibrated_candidate_v4"
    )
    assert explicit_v4.name == current.name
    assert frozen_v32.name == (
        "professional_parallel_multi_calibrated_candidate_v3_2_budgeted"
    )
    assert proxy_research.name == (
        "professional_parallel_multi_search_proxy_research_v5"
    )
    assert sparse_proxy_research.name == (
        "professional_parallel_multi_sparse_proxy_research_v5"
    )
    assert sparse_proxy_research.confirmation_reserve_ms == 7_000
    assert calibrated_v5_research.name == (
        "professional_parallel_multi_sparse_proxy_calibrated_v5_research"
    )
    assert calibrated_v5_research.confirmation_reserve_ms == 7_000
    assert (
        calibrated_v5_research.discard_validator.config
        .evidence_calibrator.decision_margin
        == 0.03
    )
    assert calibrated_v6_research.name == (
        "professional_parallel_multi_sparse_proxy_calibrated_v6_research"
    )
    assert calibrated_v6_research.confirmation_reserve_ms == 7_000
    assert (
        calibrated_v6_research.discard_validator.config
        .evidence_calibrator.decision_margin
        == 0.04
    )
    assert calibrated_v7_research.name == (
        "professional_parallel_multi_sparse_proxy_calibrated_v7_research"
    )
    assert calibrated_v7_research.confirmation_reserve_ms == 7_000
    assert (
        calibrated_v7_research.discard_validator.config
        .evidence_calibrator.decision_margin
        == 0.05
    )
    assert opponent_robust_v8_research.name == (
        "professional_parallel_multi_opponent_robust_v8_research"
    )
    assert (
        opponent_robust_v8_research.discard_validator.config
        .evidence_calibrator.decision_margin
        == 0.14
    )
    assert (
        opponent_robust_v8_research.response_search
        .response_confirmation.config.minimum_confident_advantage
        == 0.175
    )
    assert opponent_robust_v81_research.name == (
        "professional_parallel_multi_opponent_robust_v8_1_research"
    )
    assert opponent_robust_v81_research.confirmation_reserve_ms == 7_750
    assert (
        opponent_robust_v81_research.discard_validator.config
        .evidence_calibrator.decision_margin
        == 0.14
    )
    assert len(
        opponent_robust_v81_research.discard_validator.config
        .evidence_calibrator.coefficients
    ) == 9
    assert (
        opponent_robust_v81_research.discard_validator.config
        .evidence_calibrator.minimum_confirmation_fraction
        == 1.0
    )
    assert (
        opponent_robust_v81_research.response_search
        .response_confirmation.config.minimum_confident_advantage
        == 0.175
    )
    assert two_player_response_gate.name == (
        "professional_v81_two_player_response_gate_014_research"
    )
    assert (
        two_player_response_gate.response_search
        .response_binary_confirmation.config.minimum_confident_advantage
        == 0.14
    )
    assert (
        two_player_response_gate.response_search
        .response_confirmation.config.minimum_confident_advantage
        == 0.14
    )
    two_player_response_gate._configure_response_margin(players=3)
    assert (
        two_player_response_gate.response_search
        .response_confirmation.config.minimum_confident_advantage
        == 0.175
    )
    two_player_response_gate._configure_response_margin(players=2)
    assert two_player_wang_ting_guard.name == (
        "professional_v81_two_player_wang_ting_guard_research"
    )
    assert (
        two_player_wang_ting_guard.response_search
        .response_confirmation.config.minimum_confident_advantage
        == 0.14
    )
    assert two_player_wang_ting_guard.confirmation_reserve_ms == 3_500
    assert (
        two_player_wang_ting_guard.discard_validator.config
        .adaptive_confirmation_guard_ms
        == 1_250
    )
    assert (
        two_player_wang_ting_guard.discard_validator.config
        .adaptive_confirmation_worlds_per_second
        == 15.0
    )
    assert two_player_wang_ting_guard.enable_late_full_coverage_fallback is True
    assert opponent_robust_v82_research.name == (
        "professional_parallel_multi_opponent_robust_v8_2_research"
    )
    assert opponent_robust_v82_research.confirmation_reserve_ms == 7_750
    assert (
        opponent_robust_v82_research.discard_validator.config
        .evidence_calibrator.decision_margin
        == 0.23
    )
    assert (
        opponent_robust_v82_research.discard_validator.config
        .evidence_calibrator.minimum_confirmation_fraction
        == 1.0
    )
    assert opponent_robust_v83_research.name == (
        "professional_parallel_multi_opponent_robust_v8_3_research"
    )
    assert opponent_robust_v83_research.confirmation_reserve_ms == 7_750
    assert (
        opponent_robust_v83_research.discard_validator.config
        .evidence_calibrator.decision_margin
        == 0.21
    )
    assert len(
        opponent_robust_v83_research.discard_validator.config
        .evidence_calibrator.coefficients
    ) == 25
    rollout_names = [
        factory().name
        for factory in (
            opponent_robust_v8_research.response_search.coverage
            .rollout_policy_factories
        )
    ]
    assert rollout_names.count("fast_information_set_proxy_v1") == 3
    assert len(rollout_names) == 21
    assert [
        factory().name
        for factory in (
            opponent_robust_v81_research.response_search.coverage
            .rollout_policy_factories
        )
    ] == rollout_names
    assert [
        factory().name
        for factory in (
            opponent_robust_v82_research.response_search.coverage
            .rollout_policy_factories
        )
    ] == rollout_names
    assert (
        current.discard_validator.config.evidence_calibrator.decision_margin
        == 0.01
    )
    assert (
        frozen_v32.discard_validator.config
        .evidence_calibrator.decision_margin
        == 0.0
    )


def test_league_schedule_rotates_candidate_seat_and_dealer_for_same_deal():
    jobs = build_league_jobs(
        candidate="information_set_search",
        matchups=(("baseline", "aggressive_meld"),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=500,
    )

    assert len(jobs) == 9
    assert {(job.candidate_seat, job.dealer) for job in jobs} == {
        (seat, dealer) for seat in range(3) for dealer in range(3)
    }
    assert {job.seed for job in jobs} == {500}


def test_heads_up_league_schedule_rotates_two_seats_and_dealers():
    jobs = build_league_jobs(
        candidate="information_set_search",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=True,
        seed=500,
        players=2,
    )

    assert len(jobs) == 4
    assert {(job.candidate_seat, job.dealer) for job in jobs} == {
        (seat, dealer) for seat in range(2) for dealer in range(2)
    }
    assert all(job.opponents == ("baseline",) for job in jobs)


def test_all_league_policies_produce_a_legal_discard_on_public_view():
    view = PublicView(
        seat=0,
        hand=("一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), (), ()),
        discards=((), (), ()),
        remaining_counts=(),
        stock_count=19,
    )
    rules = rules_for_room(wildcard_enabled=False)

    for name in POLICY_FACTORIES:
        if name == "root_ismcts":
            continue
        label = create_policy(name).choose_discard(view, rules)
        assert label in view.hand, name


def test_confidence_root_factories_use_the_audited_opponent_mixture():
    expected_names = (
        "independent_fast_rollout",
        "independent_fast_pressure",
        "independent_fast_denial",
    )

    for name in (
        "professional_confidence_root_shadow",
        "professional_confidence_root_candidate",
    ):
        policy = create_policy(name)
        observed_names = tuple(
            factory().name
            for factory in policy.search.rollout_policy_factories
        )
        assert observed_names == expected_names


def test_fast_league_smoke_has_no_invariant_violations():
    report = run_opponent_league(
        candidate="baseline",
        matchups=(("baseline", "aggressive_meld"),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=20260726,
        workers=1,
    )

    assert report["games"] == 9
    assert report["invariant_violations"] == 0
    assert report["coverage_failures"] == 0


def test_heads_up_fast_league_smoke_has_no_invariant_violations():
    report = run_opponent_league(
        candidate="baseline",
        matchups=(("aggressive_meld",),),
        deals_per_matchup=1,
        wildcard_enabled=True,
        seed=20260726,
        workers=1,
        players=2,
    )

    assert report["players"] == 2
    assert report["games"] == 4
    assert all(row["wildcard_enabled"] for row in report["rows"])
    assert report["invariant_violations"] == 0
    assert report["coverage_failures"] == 0


def test_two_player_wang_ting_guard_only_blocks_weak_structure_breaking_peng():
    policy = create_policy(
        "professional_v81_two_player_wang_ting_guard_research"
    )
    view = PublicView(
        seat=1,
        hand=("二", "二", "六", "陆", "柒", "柒", "王"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(("六", 2), ("陆", 4), ("柒", 2), ("拾", 2), ("王", 3)),
        stock_count=34,
        hand_sizes=(17, 20),
        pending_card="二",
        pending_source_seat=0,
    )
    selected_eval = SimpleNamespace(
        allowed=True,
        action=SimpleNamespace(type="PENG"),
        debug_details={
            "ting_before": {
                "is_ting": True,
                "waiting_cards": ["六", "陆", "柒", "王"],
                "expected_xi_if_hu": 11,
            },
            "ting_after": {
                "is_ting": True,
                "waiting_cards": ["六", "陆", "拾", "王"],
                "expected_xi_if_hu": 12,
            },
            "followup_discard": {
                "breaks_soft": True,
                "breaks_hard": False,
            },
        },
    )
    decision = SimpleNamespace(
        selected_action="PENG",
        action_evals=[selected_eval],
    )
    rules = rules_for_room(wildcard_enabled=True, players=2)

    assert policy._should_prefer_pass_for_weak_wang_ting_peng(
        decision,
        view=view,
        rules=rules,
    )

    selected_eval.debug_details["ting_after"]["expected_xi_if_hu"] = 20
    assert not policy._should_prefer_pass_for_weak_wang_ting_peng(
        decision,
        view=view,
        rules=rules,
    )

    selected_eval.debug_details["ting_after"]["expected_xi_if_hu"] = 12
    selected_eval.debug_details["followup_discard"]["breaks_soft"] = False
    assert not policy._should_prefer_pass_for_weak_wang_ting_peng(
        decision,
        view=view,
        rules=rules,
    )


def test_two_player_equal_wait_xi_peng_requests_selective_confirmation(
    monkeypatch,
):
    policy = create_policy(
        "professional_v81_two_player_wang_ting_guard_research"
    )
    view = PublicView(
        seat=1,
        hand=("玖", "玖", "九", "九", "一", "二", "三"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(("一", 1), ("四", 3), ("七", 3), ("壹", 0)),
        stock_count=31,
        hand_sizes=(17, 20),
        pending_card="玖",
        pending_source_seat=0,
    )
    impact = {
        "breaks_hard": False,
        "breaks_mixed_same_rank_triplet": True,
        "breaks_normal_sequence": False,
        "breaks_special_123": False,
        "breaks_special_2710": False,
        "breaks_wildcard_structure": False,
        "red_black_loss": 0,
        "breaks_soft_melds": [
            {"type": "mixed_same_rank_triplet", "xi_value": 0}
        ],
    }
    peng_eval = SimpleNamespace(
        action=SimpleNamespace(type="PENG"),
        debug_details={
            "ting_before": {
                "is_ting": True,
                "waiting_cards": ["一", "四", "七", "壹"],
                "expected_xi_if_hu": 10,
            },
            "ting_after": {
                "is_ting": True,
                "waiting_cards": ["一", "四", "七", "壹"],
                "expected_xi_if_hu": 14,
            },
            "consumption_impact": impact,
            "followup_discard": {
                "breaks_hard": False,
                "breaks_soft": False,
                "danger_score": 30,
            },
        },
    )
    monkeypatch.setattr(
        policy,
        "_choose_peng_decision",
        lambda *_args, **_kwargs: SimpleNamespace(
            action_evals=[peng_eval]
        ),
    )
    candidates = (
        SimpleNamespace(key="PASS"),
        SimpleNamespace(key="PENG:玖"),
    )

    assert policy._equal_wait_xi_peng_confirmation(
        view,
        rules_for_room(wildcard_enabled=False, players=2),
        candidates,
        "PASS",
    ) == ("PENG:玖", 0.02)

    peng_eval.debug_details["followup_discard"]["danger_score"] = 31
    assert policy._equal_wait_xi_peng_confirmation(
        view,
        rules_for_room(wildcard_enabled=False, players=2),
        candidates,
        "PASS",
    ) is None


def test_two_player_expanded_wait_peng_requests_selective_confirmation():
    policy = create_policy(
        "professional_v81_two_player_wang_ting_guard_research"
    )
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
    candidates = (
        RootResponseCandidate("PASS", "PASS", 0.0),
        RootResponseCandidate(
            "PENG:陆",
            "PENG",
            0.0,
            consumed_from_hand=("陆", "陆"),
            meld_groups=(("陆", "陆", "陆"),),
        ),
    )

    assert policy._equal_wait_xi_peng_confirmation(
        view,
        rules_for_room(wildcard_enabled=False, players=2),
        candidates,
        "PASS",
    ) == ("PENG:陆", 0.02)
    assert policy._choose_peng_decision(
        view,
        "陆",
        rules_for_room(wildcard_enabled=False, players=2),
    ).selected_action == "PENG"


def test_two_player_high_xi_chi_can_challenge_soft_mixed_structure_choice():
    policy = create_policy(
        "professional_v81_two_player_wang_ting_guard_research"
    )
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
    candidates = (
        RootResponseCandidate("PASS", "PASS", 0.0),
        RootResponseCandidate(
            "CHI:e08efcbf8ad2d892",
            "CHI",
            0.0,
            option_id="e08efcbf8ad2d892",
            consumed_from_hand=("一", "一"),
            meld_groups=(("一", "一", "壹"),),
        ),
        RootResponseCandidate(
            "CHI:f5e4d23ffa32c666",
            "CHI",
            0.0,
            option_id="f5e4d23ffa32c666",
            consumed_from_hand=("贰", "叁"),
            meld_groups=(("壹", "贰", "叁"),),
        ),
    )

    assert policy._no_wang_response_confirmation(
        view,
        rules_for_room(wildcard_enabled=False, players=2),
        candidates,
        "CHI:e08efcbf8ad2d892",
    ) == ("CHI:f5e4d23ffa32c666", 0.02)
    plans = enumerate_chi_plans(list(view.hand), "壹")
    decision = policy._choose_chi_decision(
        view,
        plans,
        rules_for_room(wildcard_enabled=False, players=2),
    )
    selected = policy._chi_plan_for_decision(decision, plans)
    assert selected is not None
    assert selected.initial_group == ("壹", "贰", "叁")


def test_two_player_early_zero_xi_chi_can_be_challenged_by_pass():
    policy = create_policy(
        "professional_v81_two_player_wang_ting_guard_research"
    )
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
    candidates = (
        RootResponseCandidate("PASS", "PASS", 0.0),
        RootResponseCandidate(
            "CHI:463dc8ecef33a701",
            "CHI",
            0.0,
            option_id="463dc8ecef33a701",
            consumed_from_hand=("四", "四"),
            meld_groups=(("四", "四", "肆"),),
        ),
    )

    assert policy._no_wang_response_confirmation(
        view,
        rules_for_room(wildcard_enabled=False, players=2),
        candidates,
        "CHI:463dc8ecef33a701",
    ) == ("PASS", 0.02)
    decision = policy._choose_chi_decision(
        view,
        enumerate_chi_plans(list(view.hand), "肆"),
        rules_for_room(wildcard_enabled=False, players=2),
    )
    assert decision.selected_action == "PASS"


def test_two_player_eight_out_early_mixed_chi_boundary_stays_unchanged():
    policy = create_policy(
        "professional_v81_two_player_wang_ting_guard_research"
    )
    view = PublicView(
        seat=1,
        hand=(
            "肆", "七", "九", "八", "七", "伍", "九", "陆", "拾", "叁",
            "柒", "拾", "拾", "陆", "三", "六", "七", "肆", "八", "叁",
        ),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ("二",)),
        remaining_counts=(
            ("一", 4), ("七", 1), ("三", 3), ("九", 2), ("二", 3),
            ("五", 4), ("伍", 3), ("八", 2), ("六", 3), ("十", 4),
            ("叁", 2), ("四", 4), ("壹", 4), ("拾", 1), ("捌", 3),
            ("柒", 3), ("玖", 4), ("肆", 2), ("贰", 4), ("陆", 2),
        ),
        stock_count=38,
        hand_sizes=(20, 20),
        pending_card="捌",
        pending_source_seat=0,
    )
    candidates = (
        RootResponseCandidate("PASS", "PASS", 0.0),
        RootResponseCandidate(
            "CHI:0631e9a59ab8d14e",
            "CHI",
            0.0,
            option_id="0631e9a59ab8d14e",
            consumed_from_hand=("陆", "柒"),
            meld_groups=(("陆", "柒", "捌"),),
        ),
        RootResponseCandidate(
            "CHI:387f1758d52beb4a",
            "CHI",
            0.0,
            option_id="387f1758d52beb4a",
            consumed_from_hand=("八", "八"),
            meld_groups=(("八", "八", "捌"),),
        ),
    )
    rules = rules_for_room(wildcard_enabled=False, players=2)
    plans = enumerate_chi_plans(list(view.hand), "捌")
    decision = policy._choose_chi_decision(view, plans, rules)
    selected = policy._chi_plan_for_decision(decision, plans)

    assert selected is not None
    assert selected.initial_group == ("八", "八", "捌")
    assert policy._no_wang_response_confirmation(
        view,
        rules,
        candidates,
        "CHI:387f1758d52beb4a",
    ) is None


def test_two_player_final_candidate_uses_coverage_safe_confirmation_reserve():
    anchor = create_policy(
        "professional_parallel_multi_opponent_robust_v8_1_research"
    )
    candidate = create_policy(
        "professional_v81_two_player_wang_ting_guard_research"
    )

    assert anchor.confirmation_reserve_ms == 7_750
    assert candidate.confirmation_reserve_ms == 3_500
    assert (
        anchor.discard_validator.config.adaptive_confirmation_guard_ms
        == 250
    )
    assert (
        candidate.discard_validator.config.adaptive_confirmation_guard_ms
        == 1_250
    )
    assert (
        anchor.discard_validator.config.adaptive_confirmation_worlds_per_second
        == 20.0
    )
    assert (
        candidate.discard_validator.config.adaptive_confirmation_worlds_per_second
        == 15.0
    )
    assert not anchor.enable_late_full_coverage_fallback
    assert candidate.enable_late_full_coverage_fallback
    assert anchor.late_full_coverage_minimum_window_ms == 0
    assert candidate.late_full_coverage_minimum_window_ms == 2_000


def test_training_league_can_keep_one_balanced_rotation_per_deal():
    report = run_opponent_league(
        candidate="baseline",
        matchups=(("aggressive_meld",),),
        deals_per_matchup=4,
        wildcard_enabled=False,
        seed=20260726,
        workers=1,
        players=2,
        balanced_rotation_only=True,
    )

    assert report["balanced_rotation_only"]
    assert report["games"] == 4
    assert len({row["seed"] for row in report["rows"]}) == 4
    assert {
        (row["candidate_seat"], row["dealer"])
        for row in report["rows"]
    } == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_league_can_capture_structured_decision_traces_on_demand():
    report = run_opponent_league(
        candidate="baseline",
        matchups=(("baseline",),),
        deals_per_matchup=1,
        wildcard_enabled=False,
        seed=20260728,
        workers=1,
        players=2,
        record_decisions=True,
    )

    assert report["decision_trace_recorded"]
    assert report["decision_trace_steps"] > 0
    assert all(row["decision_trace"] for row in report["rows"])
    assert all(
        entry["selected_key"]
        in {action["key"] for action in entry["legal_actions"]}
        for row in report["rows"]
        for entry in row["decision_trace"]
    )


def test_wang_route_variants_are_identical_to_production_when_wang_is_disabled():
    view = PublicView(
        seat=0,
        hand=("一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(),
        stock_count=43,
    )
    rules = rules_for_room(wildcard_enabled=False, players=2)
    expected = create_policy("professional_brain").choose_discard(view, rules)

    assert create_policy("professional_wang_route_light").choose_discard(view, rules) == expected
    assert create_policy("professional_wang_route_focus").choose_discard(view, rules) == expected


def test_wang_route_focus_is_identical_to_production_without_wang_in_hand():
    view = PublicView(
        seat=0,
        hand=("一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(),
        stock_count=43,
    )
    rules = rules_for_room(wildcard_enabled=True, players=2)
    expected = create_policy("professional_brain").choose_discard(view, rules)

    assert create_policy("professional_wang_route_focus").choose_discard(view, rules) == expected


def test_wang_route_bonus_only_rewards_reachable_red_black_thresholds():
    view = PublicView(
        seat=0,
        hand=("二", "七", "三", "四", "五", "六", "八", "九", "王"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(),
        stock_count=43,
    )
    rules = rules_for_room(wildcard_enabled=True, players=2)

    discard_red = _wildcard_route_bonus(view, "二", rules, per_step=30)
    discard_black = _wildcard_route_bonus(view, "三", rules, per_step=30)

    assert discard_red == 90
    assert discard_black == 60


def test_wang_utilization_variants_are_identical_without_wang_in_hand():
    view = PublicView(
        seat=0,
        hand=("一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(),
        stock_count=43,
    )
    rules = rules_for_room(wildcard_enabled=True, players=2)
    expected = create_policy("professional_brain").choose_discard(view, rules)

    assert create_policy("professional_wang_util_narrow").choose_discard(view, rules) == expected
    assert create_policy("professional_wang_util_balanced").choose_discard(view, rules) == expected


def test_seat_aware_opponent_candidate_is_identical_in_heads_up():
    view = PublicView(
        seat=1,
        hand=("一", "二", "四", "六", "九"),
        own_melds=(),
        all_melds=(
            (
                SimMeld("chi", ("壹", "贰", "叁")),
                SimMeld("chi", ("肆", "伍", "陆")),
                SimMeld("chi", ("柒", "捌", "玖")),
            ),
            (),
        ),
        discards=(("五", "八"), ("三",)),
        remaining_counts=(),
        stock_count=10,
        hand_sizes=(11, 12),
    )
    rules = rules_for_room(wildcard_enabled=False, players=2)

    production = create_policy("professional_brain").choose_discard(view, rules)
    candidate = create_policy("professional_seat_aware_opponents").choose_discard(view, rules)

    assert candidate == production


def test_wang_utilization_quality_deduplicates_options_and_uses_real_outs():
    weak = {
        "waiting_for": "十",
        "labels": ["二", "七"],
        "reason": "二七十 waiting_for 十",
    }
    action_eval = SimpleNamespace(
        debug_details={
            "simulation": {
                "allocation_after": {
                    "weak_potentials": [weak, dict(weak)],
                }
            },
            "ting_after": {
                "is_ting": True,
                "expected_xi_if_hu": 15,
            },
            "information_set": {
                "hu_outs": 6,
                "improving_outs": 8,
            },
        }
    )

    assert _wildcard_utilization_quality(action_eval) == (1, 6, 15, 1, 1, 1, 8)


def test_response_shadow_summary_reports_disagreements_and_latency_percentiles():
    report = _summarize_response_shadow(
        [
            {
                "response_type": "PENG",
                "search_attempted": True,
                "used_search": True,
                "disagreement": False,
                "reason": "root_response_ismcts_completed",
                "simulations": 4,
                "paired_determinizations": 2,
                "elapsed_ms": 10.0,
            },
            {
                "response_type": "CHI",
                "search_attempted": True,
                "used_search": True,
                "disagreement": True,
                "reason": "root_response_ismcts_completed",
                "simulations": 4,
                "paired_determinizations": 2,
                "elapsed_ms": 30.0,
            },
            {
                "response_type": "CHI",
                "search_attempted": False,
                "used_search": False,
                "reason": "production_safety_flag",
            },
        ]
    )

    assert report["opportunities"] == 3
    assert report["attempts"] == 2
    assert report["usable"] == 2
    assert report["disagreements"] == 1
    assert report["disagreement_rate"] == 0.5
    assert report["simulations"] == 8
    assert report["paired_determinizations"] == 4
    assert report["rollout_invariant_violations"] == 0
    assert report["rollout_coverage_failures"] == 0
    assert report["latency_ms"]["p50"] == 20.0
    assert report["latency_ms"]["max"] == 30.0


def test_discard_shadow_summary_reports_only_eligible_roots():
    report = _summarize_discard_shadow(
        [
            {"eligible": True, "reason": "captured_uncertain_discard_root"},
            {"eligible": False, "reason": "heuristic_gap_too_large"},
            {"eligible": False, "reason": "production_safety_flag"},
        ]
    )

    assert report["opportunities"] == 3
    assert report["captured_roots"] == 1
    assert report["capture_rate"] == 0.3333
    assert report["by_reason"] == {
        "captured_uncertain_discard_root": 1,
        "heuristic_gap_too_large": 1,
        "production_safety_flag": 1,
    }
    assert report["response_risk"]["attempts"] == 0
    assert report["root_search"]["attempts"] == 0


def test_discard_shadow_summary_reports_full_root_search_coverage():
    report = _summarize_discard_shadow(
        [
            {
                "search_attempted": True,
                "used_search": True,
                "disagreement": False,
                "reason": "progressive_all_action:completed:insufficient_confidence",
                "legal_candidate_count": 3,
                "searched_candidate_count": 3,
                "simulations": 9,
                "paired_determinizations": 3,
                "elapsed_ms": 1200.0,
                "validation_complete": False,
                "validation_reconfirmation_attempted": True,
                "validation_error": None,
                "validation_diagnostics": {
                    "coverage": {
                        "deadline_interruptions": 0,
                        "rollout_invariant_violations": 0,
                        "rollout_coverage_failures": 0,
                        "zero_visit_candidates": 0,
                    },
                    "first_confirmation": {
                        "deadline_interruptions": 1,
                        "rollout_invariant_violations": 0,
                        "rollout_coverage_failures": 0,
                        "zero_visit_candidates": 0,
                    },
                    "second_confirmation": {
                        "deadline_interruptions": 0,
                        "rollout_invariant_violations": 0,
                        "rollout_coverage_failures": 0,
                        "zero_visit_candidates": 0,
                    },
                },
                "candidates": [
                    {"label": "一", "visits": 1},
                    {"label": "二", "visits": 1},
                    {"label": "三", "visits": 1},
                ],
            }
        ]
    )

    root = report["root_search"]
    assert root["opportunities"] == 1
    assert root["attempts"] == 1
    assert root["usable"] == 1
    assert root["candidate_coverage_failures"] == 0
    assert root["simulations"] == 9
    assert root["latency_ms"]["p50"] == 1200.0
    assert root["dual_validation"] == {
        "attempts": 1,
        "complete": 0,
        "incomplete": 1,
        "errors": 0,
        "reconfirmation_attempts": 1,
        "decision_budget_fallbacks": 0,
        "deadline_interruptions": 1,
        "invariant_violations": 0,
        "coverage_failures": 0,
        "zero_visit_candidates": 0,
    }


def test_discard_shadow_summary_excludes_intentional_late_coverage_only():
    report = _summarize_discard_shadow(
        [
            {
                "search_attempted": True,
                "used_search": True,
                "reason": "parallel_dual_hybrid:late_full_coverage_only",
                "legal_candidate_count": 2,
                "searched_candidate_count": 2,
                "simulations": 6,
                "paired_determinizations": 3,
                "elapsed_ms": 9000.0,
                "validation_attempted": False,
                "validation_complete": False,
                "late_full_coverage_fallback": True,
                "candidates": [
                    {"label": "一", "visits": 1},
                    {"label": "二", "visits": 1},
                ],
            }
        ]
    )

    root = report["root_search"]
    assert root["attempts"] == 1
    assert root["usable"] == 1
    assert root["candidate_coverage_failures"] == 0
    assert root["dual_validation"]["attempts"] == 0
    assert root["dual_validation"]["incomplete"] == 0


def test_discard_shadow_summary_includes_immediate_response_risk_health():
    report = _summarize_discard_shadow(
        [
            {
                "eligible": True,
                "reason": "discard_response_risk_completed",
                "disagreement": True,
                "risk_search": {
                    "used_search": True,
                    "simulations": 8,
                    "paired_determinizations": 4,
                    "deadline_interruptions": 0,
                    "rollout_invariant_violations": 0,
                    "rollout_coverage_failures": 0,
                    "elapsed_ms": 12.0,
                },
            },
            {
                "eligible": False,
                "reason": "discard_response_deadline_before_complete_pair",
                "disagreement": False,
                "risk_search": {
                    "used_search": False,
                    "simulations": 0,
                    "paired_determinizations": 0,
                    "deadline_interruptions": 1,
                    "rollout_invariant_violations": 0,
                    "rollout_coverage_failures": 0,
                    "elapsed_ms": 20.0,
                },
            },
        ]
    )

    risk = report["response_risk"]
    assert risk["attempts"] == 2
    assert risk["usable"] == 1
    assert risk["disagreements"] == 1
    assert risk["simulations"] == 8
    assert risk["paired_determinizations"] == 4
    assert risk["deadline_interruptions"] == 1
    assert risk["latency_ms"]["p50"] == 16.0
    assert len(risk["disagreement_samples"]) == 1
