import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import replace
from types import SimpleNamespace

import ai.dual_discard_validator as validator_module
import pytest
from ai.dual_discard_validator import (
    DualBatchDiscardValidator,
    DualDiscardChallengerEvidence,
    DualDiscardEvidenceCalibratorConfig,
    DualDiscardCoverage,
    DualDiscardValidationResult,
    DualDiscardValidatorConfig,
    _adaptive_confirmation_worlds,
    _calibrate_challenger_evidence,
    _future_results_before_deadline,
    _select_challenger_labels,
)
from ai.dual_validated_candidate import (
    ProfessionalParallelDualValidatedCandidatePolicy,
)
from ai.full_game_simulator import BaselinePolicy, PublicView
from ai.ismcts import (
    PairedAdvantageStats,
    RootCandidateStats,
    RootSearchResult,
)


def test_future_collection_never_waits_past_absolute_deadline():
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(time.sleep, 0.25)
    started = time.perf_counter()

    with pytest.raises(FutureTimeoutError, match="future_deadline_exhausted"):
        _future_results_before_deadline(
            (future,),
            absolute_deadline=time.perf_counter() + 0.04,
        )

    elapsed = time.perf_counter() - started
    executor.shutdown(wait=False, cancel_futures=True)
    assert elapsed < 0.12


def test_future_collection_supports_android_remote_future_contract():
    class RemoteFuture:
        def __init__(self, value):
            self.value = value
            self.timeouts = []
            self.cancelled = False

        def result(self, timeout=None):
            self.timeouts.append(timeout)
            return self.value

        def cancel(self):
            self.cancelled = True
            return True

    futures = (RemoteFuture("a"), RemoteFuture("b"))

    results = _future_results_before_deadline(
        futures,
        absolute_deadline=time.perf_counter() + 1.0,
    )

    assert results == ["a", "b"]
    assert all(future.timeouts[0] > 0.0 for future in futures)


def test_confirmation_directly_compares_against_requested_preferred(monkeypatch):
    executor = _CapturingExecutor()
    coverage = _search_result(
        ("六", "拾", "壹"),
        worlds=24,
        selected="六",
    )
    screened = DualDiscardCoverage(
        view=_view(),
        rules={"players": 2},
        labels=("六", "拾", "壹"),
        candidate_priors={"六": 3.0, "拾": 2.0, "壹": 1.0},
        coverage=coverage,
        executor=executor,
        elapsed_ms=10.0,
    )
    combined = _search_result(
        ("拾", "六"),
        worlds=384,
        selected="六",
        preferred="拾",
        challenger="六",
    )
    monkeypatch.setattr(
        validator_module,
        "_combined_discard_confirmations",
        lambda *_args, **_kwargs: combined,
    )
    validator = DualBatchDiscardValidator(
        rollout_policy_factories=(BaselinePolicy,),
    )

    result = validator.confirm_discard(
        screened,
        preferred_label="拾",
        time_budget_ms=1_234,
    )

    assert result.preferred_label == "拾"
    assert result.challenger_label == "六"
    assert result.selected_label == "六"
    assert result.complete
    assert [payload["labels"] for payload in executor.payloads] == [
        ["拾", "六"],
        ["拾", "六"],
    ]
    assert all(payload["preferred"] == "拾" for payload in executor.payloads)
    assert all(
        payload["time_budget_ms"] == 1_234
        for payload in executor.payloads
    )
    assert not replace(
        result,
        expected_confirmation_worlds=193,
    ).complete


def test_confirmation_can_keep_larger_frozen_familywise_count(monkeypatch):
    executor = _CapturingExecutor()
    coverage = _search_result(("六", "拾"), worlds=24, selected="六")
    screened = DualDiscardCoverage(
        view=_view(),
        rules={"players": 2},
        labels=("六", "拾"),
        candidate_priors={"六": 2.0, "拾": 1.0},
        coverage=coverage,
        executor=executor,
        elapsed_ms=10.0,
    )
    captured = []
    original = validator_module._challenger_evidence

    def capture_familywise(**kwargs):
        captured.append(kwargs["familywise_comparisons"])
        return original(**kwargs)

    monkeypatch.setattr(
        validator_module,
        "_challenger_evidence",
        capture_familywise,
    )
    validator = DualBatchDiscardValidator(
        rollout_policy_factories=(BaselinePolicy,),
        config=DualDiscardValidatorConfig(
            confirmation_worlds=24,
            confirmation_challengers=1,
            confirmation_familywise_comparisons=5,
        ),
    )

    validator.confirm_discard(screened, preferred_label="拾")

    assert captured == [5]


def test_challenger_shortlist_combines_coverage_and_prior_ranks():
    candidates = (
        _candidate("A", reward=5.0, heuristic=1.0),
        _candidate("B", reward=4.0, heuristic=2.0),
        _candidate("C", reward=3.0, heuristic=100.0),
        _candidate("D", reward=2.0, heuristic=90.0),
        _candidate("E", reward=1.0, heuristic=80.0),
        _candidate("P", reward=0.0, heuristic=200.0),
    )

    selected = _select_challenger_labels(
        candidates,
        preferred_label="P",
        limit=4,
        prior_slots=2,
    )

    assert selected == ("A", "B", "C", "D")


def test_evidence_calibrator_can_override_without_raw_confidence():
    validation = _validation_result(
        preferred="壹",
        challenger="六",
        selected="壹",
        override=False,
    )
    evidence = DualDiscardChallengerEvidence(
        challenger_label="六",
        confidence_override=False,
        first_confirmation=validation.first_confirmation,
        second_confirmation=validation.second_confirmation,
        combined=validation.combined,
    )
    calibrated = _calibrate_challenger_evidence(
        evidence,
        coverage=validation.coverage,
        preferred_label="壹",
        expected_worlds=192,
        calibrator=DualDiscardEvidenceCalibratorConfig(
            feature_mean=(0.0,) * 8,
            feature_scale=(1.0,) * 8,
            intercept=1.0,
            coefficients=(0.0,) * 8,
            decision_margin=0.0,
        ),
    )

    assert calibrated.confidence_override
    assert calibrated.override_basis == "ridge_calibrated_expected_value"
    assert calibrated.calibrated_advantage == 1.0


def test_evidence_calibrator_keeps_strict_result_at_untrained_world_count():
    validation = _validation_result(
        preferred="壹",
        challenger="六",
        selected="壹",
        override=False,
    )
    evidence = DualDiscardChallengerEvidence(
        challenger_label="六",
        confidence_override=False,
        first_confirmation=validation.first_confirmation,
        second_confirmation=validation.second_confirmation,
        combined=validation.combined,
    )

    calibrated = _calibrate_challenger_evidence(
        evidence,
        coverage=validation.coverage,
        preferred_label="壹",
        expected_worlds=64,
        calibrator=DualDiscardEvidenceCalibratorConfig(
            feature_mean=(0.0,) * 8,
            feature_scale=(1.0,) * 8,
            intercept=1.0,
            coefficients=(0.0,) * 8,
            trained_confirmation_worlds=112,
        ),
    )

    assert calibrated is evidence


def test_evidence_calibrator_can_use_sufficient_partial_batches():
    validation = _validation_result(
        preferred="壹",
        challenger="六",
        selected="壹",
        override=False,
    )
    partial_stages = []
    for stage in (
        validation.first_confirmation,
        validation.second_confirmation,
    ):
        partial_stages.append(
            replace(
                stage,
                paired_determinizations=180,
                deadline_interruptions=1,
                candidates=tuple(
                    replace(candidate, visits=180)
                    for candidate in stage.candidates
                ),
            )
        )
    evidence = DualDiscardChallengerEvidence(
        challenger_label="六",
        confidence_override=False,
        first_confirmation=partial_stages[0],
        second_confirmation=partial_stages[1],
        combined=validation.combined,
    )

    calibrated = _calibrate_challenger_evidence(
        evidence,
        coverage=validation.coverage,
        preferred_label="壹",
        expected_worlds=192,
        calibrator=DualDiscardEvidenceCalibratorConfig(
            feature_mean=(0.0,) * 9,
            feature_scale=(1.0,) * 9,
            intercept=0.0,
            coefficients=(0.0,) * 8 + (1.0,),
            decision_margin=0.9,
            minimum_confirmation_fraction=0.9,
        ),
    )
    rejected = _calibrate_challenger_evidence(
        evidence,
        coverage=validation.coverage,
        preferred_label="壹",
        expected_worlds=192,
        calibrator=DualDiscardEvidenceCalibratorConfig(
            feature_mean=(0.0,) * 8,
            feature_scale=(1.0,) * 8,
            intercept=1.0,
            coefficients=(0.0,) * 8,
            decision_margin=0.0,
            minimum_confirmation_fraction=0.95,
        ),
    )

    assert calibrated.confidence_override
    assert rejected.calibrated_advantage is None


def test_adaptive_confirmation_worlds_respects_remaining_budget(monkeypatch):
    monkeypatch.setattr(
        validator_module.time,
        "perf_counter",
        lambda: 100.0,
    )

    assert (
        _adaptive_confirmation_worlds(
            maximum_worlds=112,
            minimum_worlds=32,
            worlds_per_second=14.0,
            guard_ms=250,
            absolute_deadline=105.0,
        )
        == 66
    )
    assert (
        _adaptive_confirmation_worlds(
            maximum_worlds=112,
            minimum_worlds=32,
            worlds_per_second=14.0,
            guard_ms=250,
            absolute_deadline=120.0,
        )
        == 112
    )


def test_familywise_confirmation_tolerates_one_neutral_split():
    preferred = "壹"
    challenger = "十"
    first = _search_result(
        (preferred, challenger),
        worlds=96,
        selected=preferred,
        preferred=preferred,
        challenger=challenger,
    )
    first = replace(
        first,
        paired_advantages=(
            replace(
                first.paired_advantages[0],
                mean_delta=-0.02,
                lower_confidence_bound=-0.20,
                upper_confidence_bound=0.16,
            ),
        ),
    )
    second = _search_result(
        (preferred, challenger),
        worlds=96,
        selected=challenger,
        preferred=preferred,
        challenger=challenger,
    )
    combined = _search_result(
        (preferred, challenger),
        worlds=192,
        selected=challenger,
        preferred=preferred,
        challenger=challenger,
    )

    basis = validator_module._dual_override_basis(
        first,
        second,
        combined,
        challenger=challenger,
        expected_worlds=96,
        repeatable_mean_override_threshold=None,
    )

    assert basis == "familywise_confident"
    contradicted_first = replace(
        first,
        paired_advantages=(
            replace(
                first.paired_advantages[0],
                upper_confidence_bound=-0.01,
            ),
        ),
    )
    assert (
        validator_module._dual_override_basis(
            contradicted_first,
            second,
            combined,
            challenger=challenger,
            expected_worlds=96,
            repeatable_mean_override_threshold=None,
        )
        is None
    )


def test_multi_confirmation_keeps_multiple_challengers_and_selects_strongest(
    monkeypatch,
):
    executor = _MultiCapturingExecutor()
    coverage = _search_result(
        ("六", "拾", "壹"),
        worlds=24,
        selected="六",
    )
    screened = DualDiscardCoverage(
        view=_view(),
        rules={"players": 2},
        labels=("六", "拾", "壹"),
        candidate_priors={"六": 3.0, "拾": 2.0, "壹": 1.0},
        coverage=coverage,
        executor=executor,
        elapsed_ms=10.0,
    )

    def combine(first, second, *, preferred_label, alternative_labels, **_kwargs):
        challenger = alternative_labels[0]
        strength = 0.4 if challenger == "拾" else 0.1
        result = _search_result(
            (preferred_label, challenger),
            worlds=192,
            selected=challenger,
            preferred=preferred_label,
            challenger=challenger,
        )
        advantage = replace(
            result.paired_advantages[0],
            mean_delta=strength,
            lower_confidence_bound=strength / 2.0,
        )
        return replace(result, paired_advantages=(advantage,))

    monkeypatch.setattr(
        validator_module,
        "_combined_discard_confirmations",
        combine,
    )
    validator = DualBatchDiscardValidator(
        rollout_policy_factories=(BaselinePolicy,),
        config=DualDiscardValidatorConfig(
            confirmation_worlds=96,
            confirmation_challengers=2,
        ),
    )

    result = validator.confirm_discard(
        screened,
        preferred_label="壹",
    )

    assert result.complete
    assert result.selected_label == "拾"
    assert result.challenger_label == "拾"
    assert [item.challenger_label for item in result.challenger_evidence] == [
        "六",
        "拾",
    ]
    assert [payload["labels"] for payload in executor.payloads] == [
        ["壹", "六"],
        ["壹", "六"],
        ["壹", "拾"],
        ["壹", "拾"],
    ]


def test_multi_confirmation_ranks_winners_on_pooled_direct_evidence(
    monkeypatch,
):
    executor = _MultiCapturingExecutor()
    coverage = _search_result(
        ("六", "拾", "壹"),
        worlds=24,
        selected="六",
    )
    screened = DualDiscardCoverage(
        view=_view(),
        rules={"players": 2},
        labels=("六", "拾", "壹"),
        candidate_priors={"六": 3.0, "拾": 2.0, "壹": 1.0},
        coverage=coverage,
        executor=executor,
        elapsed_ms=10.0,
    )

    def combine(first, second, *, preferred_label, alternative_labels, **_kwargs):
        challenger = alternative_labels[0]
        mean_delta = 0.28 if challenger == "六" else 0.29
        lower_bound = 0.03 if challenger == "六" else 0.04
        result = _search_result(
            (preferred_label, challenger),
            worlds=192,
            selected=challenger,
            preferred=preferred_label,
            challenger=challenger,
        )
        advantage = replace(
            result.paired_advantages[0],
            mean_delta=mean_delta,
            lower_confidence_bound=lower_bound,
        )
        return replace(result, paired_advantages=(advantage,))

    monkeypatch.setattr(
        validator_module,
        "_combined_discard_confirmations",
        combine,
    )
    validator = DualBatchDiscardValidator(
        rollout_policy_factories=(BaselinePolicy,),
        config=DualDiscardValidatorConfig(
            confirmation_worlds=96,
            confirmation_challengers=2,
        ),
    )

    result = validator.confirm_discard(
        screened,
        preferred_label="壹",
    )

    assert result.selected_label == "六"
    assert result.challenger_label == "六"


def test_multi_confirmation_is_incomplete_when_any_pair_stops_early():
    executor = _MultiCapturingExecutor(incomplete_label="拾")
    coverage = _search_result(
        ("六", "拾", "壹"),
        worlds=24,
        selected="六",
    )
    screened = DualDiscardCoverage(
        view=_view(),
        rules={"players": 2},
        labels=("六", "拾", "壹"),
        candidate_priors={"六": 3.0, "拾": 2.0, "壹": 1.0},
        coverage=coverage,
        executor=executor,
        elapsed_ms=10.0,
    )
    validator = DualBatchDiscardValidator(
        rollout_policy_factories=(BaselinePolicy,),
        config=DualDiscardValidatorConfig(
            confirmation_worlds=96,
            confirmation_challengers=2,
        ),
    )

    result = validator.confirm_discard(
        screened,
        preferred_label="壹",
    )

    assert not result.complete
    assert len(result.challenger_evidence) == 2


def test_incomplete_multi_result_can_use_complete_confident_evidence():
    complete = _validation_result(
        preferred="壹",
        challenger="六",
        selected="六",
        override=True,
    )
    incomplete_second = replace(
        complete.second_confirmation,
        paired_determinizations=191,
    )
    result = replace(
        complete,
        challenger_evidence=(
            DualDiscardChallengerEvidence(
                challenger_label="六",
                confidence_override=True,
                first_confirmation=complete.first_confirmation,
                second_confirmation=complete.second_confirmation,
                combined=complete.combined,
                override_basis="familywise_confident",
            ),
            DualDiscardChallengerEvidence(
                challenger_label="拾",
                confidence_override=False,
                first_confirmation=complete.first_confirmation,
                second_confirmation=incomplete_second,
                combined=complete.combined,
            ),
        ),
    )

    assert not result.complete
    assert result.usable_confident_override


def test_hybrid_uses_complete_override_when_other_challenger_is_incomplete(
    monkeypatch,
):
    complete = _validation_result(
        preferred="拾",
        challenger="六",
        selected="六",
        override=True,
    )
    validation = replace(
        complete,
        challenger_evidence=(
            DualDiscardChallengerEvidence(
                challenger_label="六",
                confidence_override=True,
                first_confirmation=complete.first_confirmation,
                second_confirmation=complete.second_confirmation,
                combined=complete.combined,
                override_basis="familywise_confident",
            ),
            DualDiscardChallengerEvidence(
                challenger_label="壹",
                confidence_override=False,
                first_confirmation=complete.first_confirmation,
                second_confirmation=replace(
                    complete.second_confirmation,
                    paired_determinizations=191,
                ),
                combined=complete.combined,
            ),
        ),
    )
    policy = ProfessionalParallelDualValidatedCandidatePolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=_FakeValidator(validation, validation),
    )
    policy.search = _FakeBaselineSearch(
        _search_result(
            ("拾", "六", "壹"),
            worlds=24,
            selected="拾",
        )
    )
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="拾",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("拾", 3.0), ("六", 2.0), ("壹", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    selected = policy.choose_discard(_view(), {"players": 2})

    assert selected == "六"
    assert policy.validation_partial_overrides == 1
    assert policy.validation_incomplete_fallbacks == 0
    event = policy.discard_events()[-1]
    assert not event["validation_complete"]
    assert event["confidence_override"]


def test_hybrid_validates_against_immutable_production_anchor(monkeypatch):
    initial_validation = _validation_result(
        preferred="壹",
        challenger="六",
        selected="六",
        override=True,
    )
    fake_validator = _FakeValidator(
        initial_validation,
        initial_validation,
    )
    policy = ProfessionalParallelDualValidatedCandidatePolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=fake_validator,
    )
    policy.search = _FakeBaselineSearch(
        _search_result(
            ("拾", "六", "壹"),
            worlds=24,
            selected="拾",
        )
    )
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="壹",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("壹", 3.0), ("六", 2.0), ("拾", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    selected = policy.choose_discard(_view(), {"players": 2})

    assert selected == "六"
    assert fake_validator.screened_preferred == production.selected_label
    assert fake_validator.confirmed_preferred == production.selected_label


def test_hybrid_reserves_confirmation_time_from_both_root_searches(monkeypatch):
    initial_validation = _validation_result(
        preferred="壹",
        challenger="六",
        selected="壹",
        override=False,
    )
    fake_validator = _FakeValidator(
        initial_validation,
        initial_validation,
    )
    policy = ProfessionalParallelDualValidatedCandidatePolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=fake_validator,
        decision_time_budget_ms=200,
        deadline_guard_ms=20,
        confirmation_reserve_ms=60,
    )
    fake_baseline = _FakeBaselineSearch(
        _search_result(
            ("拾", "六", "壹"),
            worlds=24,
            selected="拾",
        )
    )
    policy.search = fake_baseline
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="壹",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("壹", 3.0), ("六", 2.0), ("拾", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    started = time.perf_counter()
    policy.choose_discard(_view(), {"players": 2})

    assert fake_validator.screen_deadline is not None
    assert fake_baseline.absolute_deadline is not None
    assert fake_validator.screen_deadline == fake_baseline.absolute_deadline
    phase_budget_ms = (
        fake_validator.screen_deadline - started
    ) * 1000.0
    assert 90.0 <= phase_budget_ms <= 140.0
    event = policy.discard_events()[-1]
    assert event["confirmation_reserve_ms"] == 60


def test_completed_validation_separates_baseline_phase_deadline(monkeypatch):
    validation = _validation_result(
        preferred="拾",
        challenger="六",
        selected="六",
        override=True,
    )
    fake_validator = _FakeValidator(validation, validation)
    baseline = replace(
        _search_result(
            ("拾", "六", "壹"),
            worlds=24,
            selected="拾",
        ),
        used_search=False,
        deadline_interruptions=1,
    )
    policy = ProfessionalParallelDualValidatedCandidatePolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=fake_validator,
    )
    policy.search = _FakeBaselineSearch(baseline)
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="拾",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("拾", 3.0), ("六", 2.0), ("壹", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    assert policy.choose_discard(_view(), {"players": 2}) == "六"

    assert policy.last_search is not None
    assert policy.last_search.used_search
    assert policy.last_search.deadline_interruptions == 0
    event = policy.discard_events()[-1]
    assert event["baseline_phase_deadline_interruptions"] == 1
    assert event["baseline_diagnostics"]["deadline_interruptions"] == 1


def test_hybrid_stops_waiting_at_the_decision_budget(monkeypatch):
    fake_validator = _SlowValidator(delay_seconds=0.25)
    policy = ProfessionalParallelDualValidatedCandidatePolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=fake_validator,
        decision_time_budget_ms=100,
        deadline_guard_ms=20,
    )
    policy.search = _FakeBaselineSearch(
        _search_result(
            ("拾", "六", "壹"),
            worlds=24,
            selected="拾",
        )
    )
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="壹",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("壹", 3.0), ("六", 2.0), ("拾", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    started = time.perf_counter()
    selected = policy.choose_discard(_view(), {"players": 2})
    elapsed = time.perf_counter() - started

    assert selected == "拾"
    assert elapsed >= 0.06
    assert elapsed < 0.18
    assert policy.last_validation is None
    assert policy.last_validation_error == (
        "decision_budget_waiting_for_validation"
    )
    assert policy.validation_timeouts == 1
    assert policy.decision_budget_fallbacks == 1
    assert policy.last_search is not None
    assert policy.last_search.elapsed_ms >= 70.0
    event = policy.discard_events()[-1]
    assert event["decision_time_budget_ms"] == 100
    assert event["deadline_guard_ms"] == 20
    assert event["validation_diagnostics"] is None
    assert event["validation_worker_pending_at_cleanup"]
    assert not event["validation_worker_done_after_cleanup"]
    assert policy.validation_cleanup_waits == 1
    assert event["validation_cleanup_elapsed_ms"] < 20.0


def test_validation_timeout_without_proposed_override_is_not_a_fallback(
    monkeypatch,
):
    fake_validator = _SlowValidator(delay_seconds=0.25)
    policy = ProfessionalParallelDualValidatedCandidatePolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=fake_validator,
        decision_time_budget_ms=100,
        deadline_guard_ms=20,
    )
    policy.search = _FakeBaselineSearch(
        _search_result(
            ("拾", "六", "壹"),
            worlds=24,
            selected="拾",
        )
    )
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="拾",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("拾", 3.0), ("六", 2.0), ("壹", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    assert policy.choose_discard(_view(), {"players": 2}) == "拾"

    assert policy.last_validation_error == "decision_budget_waiting_for_validation"
    assert policy.validation_timeouts == 1
    assert policy.decision_budget_fallbacks == 0


def test_completed_validation_is_read_after_baseline_exhausts_wait_budget(
    monkeypatch,
):
    validation = _validation_result(
        preferred="拾",
        challenger="六",
        selected="拾",
        override=False,
    )
    fake_validator = _FakeValidator(validation, validation)
    policy = ProfessionalParallelDualValidatedCandidatePolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=fake_validator,
        decision_time_budget_ms=100,
        deadline_guard_ms=20,
        confirmation_reserve_ms=60,
    )
    baseline = _search_result(
        ("拾", "六", "壹"),
        worlds=24,
        selected="拾",
    )

    def slow_baseline(*_args, **_kwargs):
        time.sleep(0.09)
        return baseline

    monkeypatch.setattr(policy, "_search_discard_anchor", slow_baseline)
    monkeypatch.setattr(
        policy,
        "_discard_confirmation_skip_reason",
        lambda **_kwargs: "completed_coverage_requires_no_confirmation",
    )
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="拾",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("拾", 3.0), ("六", 2.0), ("壹", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    assert policy.choose_discard(_view(), {"players": 2}) == "拾"

    assert policy.last_validation_error is None
    assert policy.validation_timeouts == 0
    assert policy.decision_budget_fallbacks == 0
    event = policy.discard_events()[-1]
    assert event["validation_confirmation_skipped"]
    assert not event["validation_worker_pending_at_cleanup"]
    assert event["validation_worker_done_after_cleanup"]


def test_validation_coverage_can_replace_duplicate_provisional_baseline(
    monkeypatch,
):
    class CoverageBaselinePolicy(
        ProfessionalParallelDualValidatedCandidatePolicy
    ):
        reuse_validation_coverage_as_baseline = True

    validation = _validation_result(
        preferred="拾",
        challenger="六",
        selected="六",
        override=True,
    )
    fake_validator = _FakeValidator(validation, validation)
    policy = CoverageBaselinePolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=fake_validator,
    )

    def duplicate_baseline_must_not_run(*_args, **_kwargs):
        raise AssertionError("duplicate provisional baseline search ran")

    monkeypatch.setattr(
        policy,
        "_search_discard_anchor",
        duplicate_baseline_must_not_run,
    )
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="拾",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("拾", 3.0), ("六", 2.0), ("壹", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    assert policy.choose_discard(_view(), {"players": 2}) == "六"
    assert policy.last_baseline_search is validation.coverage
    assert policy.last_search is not None
    assert policy.last_search.simulations == (
        validation.coverage.simulations
        + validation.first_confirmation.simulations
        + validation.second_confirmation.simulations
    )
    event = policy.discard_events()[-1]
    assert event["baseline_diagnostics"]["candidate_count"] == 3
    assert event["baseline_diagnostics"]["zero_visit_candidates"] == 0


def test_single_pass_progressive_discard_never_starts_dual_validator(
    monkeypatch,
):
    class SinglePassPolicy(ProfessionalParallelDualValidatedCandidatePolicy):
        single_pass_progressive_discard_evidence = True

    class ForbiddenValidator:
        def screen_discard(self, *_args, **_kwargs):
            raise AssertionError("duplicate dual validator started")

    policy = SinglePassPolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=ForbiddenValidator(),
    )
    policy.search = _FakeBaselineSearch(
        _search_result(
            ("拾", "六", "壹"),
            worlds=8,
            selected="拾",
            preferred="拾",
            challenger="六",
        )
    )
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="拾",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("拾", 3.0), ("六", 2.0), ("壹", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    assert policy.choose_discard(_view(), {"players": 2}) == "拾"
    event = policy.discard_events()[-1]
    assert event["discard_evidence_route"] == "single_pass_progressive"
    assert not event["validation_attempted"]
    assert event["baseline_diagnostics"]["candidate_count"] == 3
    assert event["final_authorization_reason"] == "single_pass_production_kept"


def test_single_pass_progressive_discard_authorizes_only_clean_confidence(
    monkeypatch,
):
    class SinglePassPolicy(ProfessionalParallelDualValidatedCandidatePolicy):
        single_pass_progressive_discard_evidence = True

    class ForbiddenValidator:
        def screen_discard(self, *_args, **_kwargs):
            raise AssertionError("duplicate dual validator started")

    policy = SinglePassPolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=ForbiddenValidator(),
    )
    confirmed = _search_result(
        ("拾", "六", "壹"),
        worlds=32,
        selected="六",
        preferred="拾",
        challenger="六",
    )
    policy.search = _FakeBaselineSearch(confirmed)
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="拾",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("拾", 3.0), ("六", 2.0), ("壹", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    assert policy.choose_discard(_view(), {"players": 2}) == "六"
    event = policy.discard_events()[-1]
    assert event["final_authorization_reason"] == "single_pass_confidence_authorized"
    assert event["confidence_override"]

    policy.search.result = replace(
        confirmed,
        deadline_interruptions=1,
    )
    assert policy.choose_discard(_view(), {"players": 2}) == "拾"
    event = policy.discard_events()[-1]
    assert event["final_authorization_reason"] == "single_pass_evidence_incomplete"
    assert not event["confidence_override"]


def test_validator_propagates_absolute_deadline_to_process_tasks():
    executor = _CapturingExecutor()
    validator = DualBatchDiscardValidator(
        rollout_policy_factories=(BaselinePolicy,),
        executor=executor,
    )
    deadline = time.perf_counter() + 5.0

    screened = validator.screen_discard(
        _view(),
        rules={"players": 2},
        candidate_labels=("壹", "六", "拾"),
        candidate_priors={"壹": 3.0, "六": 2.0, "拾": 1.0},
        absolute_deadline=deadline,
    )
    result = validator.confirm_discard(
        screened,
        preferred_label="壹",
        absolute_deadline=deadline,
    )

    assert result.complete
    assert executor.payloads
    assert all(
        payload["absolute_deadline"] == deadline
        for payload in executor.payloads
    )
    assert all(
        1 <= payload["time_budget_ms"] <= 5_000
        for payload in executor.payloads
    )


def test_hybrid_records_incomplete_reconfirmation(monkeypatch):
    initial_validation = _validation_result(
        preferred="壹",
        challenger="六",
        selected="六",
        override=True,
    )
    incomplete_confirmation = replace(
        _validation_result(
            preferred="拾",
            challenger="六",
            selected="拾",
            override=False,
        ),
        expected_confirmation_worlds=193,
    )
    policy = ProfessionalParallelDualValidatedCandidatePolicy(
        rollout_policy_factories=(BaselinePolicy,),
        validator=_FakeValidator(
            initial_validation,
            incomplete_confirmation,
        ),
    )
    policy.search = _FakeBaselineSearch(
        _search_result(
            ("拾", "六", "壹"),
            worlds=24,
            selected="拾",
        )
    )
    production = SimpleNamespace(
        selected_action="DISCARD",
        selected_label="壹",
        action_evals=[
            SimpleNamespace(
                type="DISCARD",
                action=SimpleNamespace(label=label),
                ev=ev,
            )
            for label, ev in (("壹", 3.0), ("六", 2.0), ("拾", 1.0))
        ],
    )
    monkeypatch.setattr(policy, "_choose", lambda *_args, **_kwargs: production)

    selected = policy.choose_discard(_view(), {"players": 2})

    assert selected == "拾"
    assert policy.validation_incomplete_fallbacks == 1
    assert policy.reconfirmation_incomplete_fallbacks == 0
    event = policy.discard_events()[-1]
    assert not event["validation_reconfirmation_attempted"]
    assert not event["validation_complete"]
    assert not event["validation_diagnostics"]["complete"]
    assert (
        event["validation_diagnostics"]["expected_confirmation_worlds"]
        == 193
    )
    assert (
        event["validation_diagnostics"]["first_confirmation"][
            "paired_determinizations"
        ]
        == 192
    )


class _CapturingExecutor:
    def __init__(self):
        self.payloads = []

    def submit(self, _function, payload):
        self.payloads.append(payload)
        labels = tuple(payload["labels"])
        result = _search_result(
            labels,
            worlds=int(payload["worlds"]),
            selected=labels[1],
            preferred=labels[0],
            challenger=labels[1],
        )
        future = Future()
        future.set_result(result)
        return future


class _MultiCapturingExecutor:
    def __init__(self, *, incomplete_label=None):
        self.payloads = []
        self.incomplete_label = incomplete_label

    def submit(self, _function, payload):
        self.payloads.append(payload)
        labels = tuple(payload["labels"])
        worlds = int(payload["worlds"])
        result = _search_result(
            labels,
            worlds=worlds,
            selected=labels[1],
            preferred=labels[0],
            challenger=labels[1],
        )
        if (
            labels[1] == self.incomplete_label
            and int(payload["seed"]) == 20260733
        ):
            result = replace(
                result,
                paired_determinizations=worlds - 1,
            )
        future = Future()
        future.set_result(result)
        return future


class _FakeValidator:
    def __init__(self, initial_validation, baseline_validation):
        self.initial_validation = initial_validation
        self.baseline_validation = baseline_validation
        self.confirmed_preferred = None
        self.screened_preferred = None
        self.screen_deadline = None

    def screen_discard(self, *_args, **kwargs):
        self.screen_deadline = kwargs.get("absolute_deadline")
        self.screened_preferred = kwargs.get("preferred_label")
        return self.initial_validation

    def confirm_discard(self, _previous, *, preferred_label, **_kwargs):
        self.confirmed_preferred = preferred_label
        return self.baseline_validation


class _SlowValidator:
    def __init__(self, *, delay_seconds):
        self.delay_seconds = delay_seconds
        self.completed = False

    def screen_discard(
        self,
        *_args,
        absolute_deadline=None,
        **_kwargs,
    ):
        delay = self.delay_seconds
        if absolute_deadline is not None:
            delay = min(
                delay,
                max(0.0, absolute_deadline - time.perf_counter())
                + 0.02,
            )
        time.sleep(delay)
        self.completed = True
        return _validation_result(
            preferred="壹",
            challenger="六",
            selected="壹",
            override=False,
        )

    def confirm_discard(self, screened, **_kwargs):
        return screened


class _FakeBaselineSearch:
    def __init__(self, result):
        self.result = result
        self.absolute_deadline = None

    def search_discard(self, *_args, **kwargs):
        self.absolute_deadline = kwargs.get("absolute_deadline")
        return self.result


def _validation_result(*, preferred, challenger, selected, override):
    coverage = _search_result(
        (challenger, preferred, "壹"),
        worlds=24,
        selected=challenger,
    )
    first = _search_result(
        (preferred, challenger),
        worlds=192,
        selected=selected,
        preferred=preferred,
        challenger=challenger,
    )
    second = _search_result(
        (preferred, challenger),
        worlds=192,
        selected=selected,
        preferred=preferred,
        challenger=challenger,
    )
    combined = _search_result(
        (preferred, challenger),
        worlds=384,
        selected=selected,
        preferred=preferred,
        challenger=challenger,
    )
    return DualDiscardValidationResult(
        selected_label=selected,
        preferred_label=preferred,
        challenger_label=challenger,
        confidence_override=override,
        elapsed_ms=20.0,
        coverage=coverage,
        first_confirmation=first,
        second_confirmation=second,
        combined=combined,
        expected_coverage_worlds=24,
        expected_confirmation_worlds=192,
        expected_candidate_count=3,
    )


def _search_result(
    labels,
    *,
    worlds,
    selected,
    preferred=None,
    challenger=None,
):
    advantages = ()
    if preferred is not None and challenger is not None:
        advantages = (
            PairedAdvantageStats(
                candidate_key=challenger,
                preferred_key=preferred,
                samples=worlds,
                mean_delta=0.2,
                sample_stddev=0.1,
                standard_error=0.01,
                lower_confidence_bound=0.1,
                upper_confidence_bound=0.3,
                positive_samples=worlds,
                tied_samples=0,
                negative_samples=0,
            ),
        )
    return RootSearchResult(
        selected_label=selected,
        used_search=True,
        reason="test",
        simulations=worlds * len(labels),
        elapsed_ms=1.0,
        candidates=tuple(
            RootCandidateStats(
                label=label,
                visits=worlds,
                reward_sum=float(worlds),
                average_reward=float(len(labels) - index),
                win_rate=0.5,
                heuristic_value=float(len(labels) - index),
            )
            for index, label in enumerate(labels)
        ),
        paired_determinizations=worlds,
        empirical_best_label=selected,
        confidence_override=selected != preferred if preferred else False,
        paired_advantages=advantages,
    )


def _candidate(label, *, reward, heuristic):
    return RootCandidateStats(
        label=label,
        visits=24,
        reward_sum=reward * 24,
        average_reward=reward,
        win_rate=0.5,
        heuristic_value=heuristic,
    )


def _view():
    return PublicView(
        seat=0,
        hand=("壹", "六", "拾"),
        own_melds=(),
        all_melds=((), ()),
        discards=((), ()),
        remaining_counts=(("壹", 3), ("六", 3), ("拾", 3)),
        stock_count=12,
        hand_sizes=(3, 3),
    )
