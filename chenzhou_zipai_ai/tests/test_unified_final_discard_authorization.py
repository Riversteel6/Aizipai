from __future__ import annotations

from types import SimpleNamespace

from ai.dual_validated_candidate import (
    ProfessionalParallelMultiOpponentProductionAnchoredRank1ResearchPolicy,
    ProfessionalParallelMultiOpponentProductionAnchoredRank1ShardedResearchPolicy,
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy,
    ProfessionalParallelMultiOpponentUnifiedGateRank1ResearchPolicy,
    ProfessionalV81TwoPlayerExactDiscardShardedResearchPolicy,
    _authorize_final_discard,
)
from ai.decision_objective import OBJECTIVE_VERSION, ObjectiveMode
from ai.full_game_simulator import BaselinePolicy
from ai.opponent_league import create_policy
from engine.rules import rules_for_room


def validation(*, lower_bound: float, complete: bool = True):
    advantage = SimpleNamespace(
        candidate_key="challenger",
        preferred_key="baseline",
        lower_confidence_bound=lower_bound,
    )
    evidence = SimpleNamespace(
        confidence_override=True,
        challenger_label="challenger",
        combined=SimpleNamespace(paired_advantages=(advantage,)),
    )
    return SimpleNamespace(
        complete=complete,
        confidence_override=True,
        selected_label="challenger",
        preferred_label="baseline",
        selected_challenger_evidence=evidence,
    )


def authorize(candidate_validation):
    return _authorize_final_discard(
        production_label="production",
        proposed_label="challenger",
        validation=candidate_validation,
        minimum_familywise_lcb=0.30,
    )


def test_production_action_needs_no_override_evidence() -> None:
    assert _authorize_final_discard(
        production_label="production",
        proposed_label="production",
        validation=None,
        minimum_familywise_lcb=0.30,
    ) == ("production", "unified_gate_production_unchanged")


def test_missing_or_incomplete_validation_falls_back_to_production() -> None:
    assert authorize(None)[0] == "production"
    assert authorize(validation(lower_bound=0.50, complete=False))[0] == "production"


def test_familywise_lcb_below_frozen_minimum_falls_back() -> None:
    selected, reason = authorize(validation(lower_bound=0.299999))

    assert selected == "production"
    assert reason == "unified_gate_familywise_lcb_below_minimum"


def test_familywise_lcb_at_frozen_minimum_authorizes_challenger() -> None:
    selected, reason = authorize(validation(lower_bound=0.30))

    assert selected == "challenger"
    assert reason == "unified_gate_authorized"


def test_selected_label_mismatch_falls_back() -> None:
    candidate_validation = validation(lower_bound=0.50)
    candidate_validation.selected_label = "different"

    assert authorize(candidate_validation)[0] == "production"


def test_validation_without_confidence_override_falls_back() -> None:
    candidate_validation = validation(lower_bound=0.50)
    candidate_validation.confidence_override = False

    assert authorize(candidate_validation)[0] == "production"


def test_missing_or_unconfirmed_selected_evidence_falls_back() -> None:
    candidate_validation = validation(lower_bound=0.50)
    candidate_validation.selected_challenger_evidence = None
    assert authorize(candidate_validation)[0] == "production"

    candidate_validation = validation(lower_bound=0.50)
    candidate_validation.selected_challenger_evidence.confidence_override = False
    assert authorize(candidate_validation)[0] == "production"


def test_missing_familywise_lcb_falls_back() -> None:
    candidate_validation = validation(lower_bound=0.50)
    advantage = (
        candidate_validation.selected_challenger_evidence.combined
        .paired_advantages[0]
    )
    advantage.lower_confidence_bound = None

    assert authorize(candidate_validation)[0] == "production"


def test_frozen_v81_default_authorization_is_unchanged() -> None:
    policy = object.__new__(
        ProfessionalParallelMultiOpponentRobustV81ResearchPolicy
    )

    assert policy._authorize_final_discard(
        production_label="production",
        proposed_label="legacy-search",
        validation=None,
    ) == ("legacy-search", "legacy_final_authorization")


def test_exact_sharded_v81_confirmation_is_not_vetoed_by_coverage_gap() -> None:
    policy = object.__new__(
        ProfessionalV81TwoPlayerExactDiscardShardedResearchPolicy
    )
    candidate_validation = validation(lower_bound=0.01)
    candidate_validation.coverage = SimpleNamespace(
        candidates=(
            SimpleNamespace(
                label="challenger",
                average_reward=0.20,
                visits=48,
                heuristic_value=1.0,
            ),
            SimpleNamespace(
                label="other",
                average_reward=0.10,
                visits=48,
                heuristic_value=0.0,
            ),
        )
    )

    assert policy._authorize_final_discard(
        production_label="production",
        proposed_label="challenger",
        validation=candidate_validation,
    ) == ("challenger", "confirmation_gate_authorized")

    candidate_validation.coverage.candidates[0].average_reward = 0.05
    candidate_validation.coverage.candidates[1].average_reward = 0.30
    assert policy._authorize_final_discard(
        production_label="production",
        proposed_label="challenger",
        validation=candidate_validation,
    ) == ("challenger", "confirmation_gate_authorized")

    candidate_validation.coverage.candidates[0].average_reward = 0.17
    assert policy._authorize_final_discard(
        production_label="production",
        proposed_label="challenger",
        validation=candidate_validation,
    ) == ("challenger", "confirmation_gate_authorized")


def test_exact_sharded_v81_does_not_repeat_confirmed_rank1_in_full_batch() -> None:
    policy = object.__new__(
        ProfessionalV81TwoPlayerExactDiscardShardedResearchPolicy
    )
    confirmed = validation(lower_bound=0.01)
    calls = []

    def confirm_discard(_screened, **kwargs):
        calls.append(kwargs)
        return confirmed

    policy.discard_validator = SimpleNamespace(
        config=SimpleNamespace(
            confirmation_challengers=5,
            confirmation_familywise_comparisons=5,
            confirmation_parallel_shards=2,
            confirmation_worlds=112,
            parallel_workers=20,
        ),
        confirm_discard=confirm_discard,
    )
    screened = SimpleNamespace(
        coverage=SimpleNamespace(
            candidates=tuple(
                SimpleNamespace(label=label)
                for label in ("challenger", "other", "production")
            )
        )
    )

    result, route = policy._confirm_discard_validation(
        screened,
        preferred_label="production",
        time_budget_ms=3_500,
        absolute_deadline=None,
    )

    assert result is confirmed
    assert route == "rank1_prefilter_complete"
    assert len(calls) == 1
    assert calls[0]["challenger_limit"] == 1
    assert calls[0]["familywise_comparisons"] == 5


def test_exact_sharded_v81_wang_uses_win_first_without_legacy_calibrator() -> None:
    policy = ProfessionalV81TwoPlayerExactDiscardShardedResearchPolicy(
        rollout_policy_factories=(BaselinePolicy,),
    )

    assert policy.single_pass_progressive_discard_evidence
    assert all(
        getattr(getattr(policy.search, name), "parallel_shards", 0) == 20
        for name in (
            "refinement",
            "discard_selection",
        )
    )
    confirmation = policy.search.discard_confirmation.base.config
    assert policy.search.discard_confirmation_alternatives == 1
    assert policy.search.discard_confirmation.parallel_shards == 96
    assert policy.response_search.parallel_shards == 20
    assert policy.response_search.parallel_workers == 12
    assert confirmation.max_candidates == 2
    assert confirmation.max_iterations // confirmation.max_candidates == 96
    assert confirmation.confidence_familywise_comparisons == 5

    policy._configure_objective(
        rules_for_room(wildcard_enabled=True, players=2)
    )

    assert policy.objective_mode is ObjectiveMode.WIN_FIRST
    assert policy.objective_version == OBJECTIVE_VERSION
    assert policy.discard_validator.config.objective_mode is ObjectiveMode.WIN_FIRST
    assert policy.discard_validator.config.evidence_calibrator is None
    configured = []
    for search in (policy.search, policy.response_search):
        progressive = getattr(search, "base", search)
        for attribute in (
            "coverage",
            "refinement",
            "discard_selection",
            "discard_confirmation",
            "response_selection",
            "response_binary_confirmation",
            "response_confirmation",
        ):
            nested = getattr(progressive, attribute, None)
            nested = getattr(nested, "base", nested)
            if nested is not None and hasattr(nested, "config"):
                configured.append(nested.config)
    assert configured
    assert all(
        config.objective_mode is ObjectiveMode.WIN_FIRST
        and config.objective_version == OBJECTIVE_VERSION
        for config in configured
    )

    policy._configure_objective(
        rules_for_room(wildcard_enabled=False, players=2)
    )
    assert policy.objective_mode is ObjectiveMode.SCORE_FIRST
    assert policy.discard_validator.config.evidence_calibrator is not None
    assert policy.two_player_response_margin == 0.14


def test_rank1_candidate_keeps_worlds_and_five_comparison_correction() -> None:
    policy = ProfessionalParallelMultiOpponentUnifiedGateRank1ResearchPolicy(
        rollout_policy_factories=(BaselinePolicy,),
    )
    config = policy.discard_validator.config

    assert config.coverage_worlds == 48
    assert config.confirmation_worlds == 112
    assert config.confirmation_challengers == 1
    assert config.confirmation_familywise_comparisons == 5


def test_production_anchored_candidate_skips_provisional_search() -> None:
    policy = ProfessionalParallelMultiOpponentProductionAnchoredRank1ResearchPolicy(
        rollout_policy_factories=(BaselinePolicy,),
    )
    anchor = policy._search_discard_anchor(
        object(),
        rules={"players": 2},
        candidate_labels=("壹", "贰"),
        candidate_priors={"壹": 2.0, "贰": 1.0},
        preferred_label="壹",
        absolute_deadline=None,
    )

    assert anchor.selected_label == "壹"
    assert anchor.empirical_best_label == "壹"
    assert anchor.simulations == 0
    assert anchor.candidates == ()
    assert not anchor.used_search
    assert anchor.reason == "direct_production_anchor_without_provisional_search"


def test_production_anchored_candidate_is_registered_with_frozen_evidence() -> None:
    policy = create_policy(
        "professional_parallel_multi_opponent_"
        "production_anchored_rank1_research"
    )
    config = policy.discard_validator.config

    assert isinstance(
        policy,
        ProfessionalParallelMultiOpponentProductionAnchoredRank1ResearchPolicy,
    )
    assert config.coverage_worlds == 48
    assert config.confirmation_worlds == 112
    assert config.confirmation_challengers == 1
    assert config.confirmation_familywise_comparisons == 5


def test_sharded_candidate_changes_only_confirmation_scheduling() -> None:
    policy = create_policy(
        "professional_parallel_multi_opponent_"
        "production_anchored_rank1_sharded_research"
    )
    config = policy.discard_validator.config

    assert isinstance(
        policy,
        ProfessionalParallelMultiOpponentProductionAnchoredRank1ShardedResearchPolicy,
    )
    assert config.coverage_worlds == 48
    assert config.confirmation_worlds == 112
    assert config.confirmation_challengers == 1
    assert config.confirmation_familywise_comparisons == 5
    assert config.confirmation_parallel_shards == 5
