from tools.evaluate_strategy_acceptance import (
    _audit_failures,
    _rotation_balance_failures,
    _wilson_lower_bound,
    evaluate_acceptance,
)


def test_audit_requires_every_world_and_candidate_visit():
    audit = {
        "complete": True,
        "requested_paired_worlds": 2,
        "completed_paired_worlds": 2,
        "search_health": {
            "deadline_interruptions": 0,
            "determinization_failures": 0,
            "rollout_invariant_violations": 0,
            "rollout_coverage_failures": 0,
        },
        "candidate_stats": [
            {"key": "DISCARD:A", "visits": 2},
            {"key": "DISCARD:B", "visits": 2},
        ],
        "paired_worlds": [
            {
                "world_index": index,
                "outcomes": [
                    {"candidate_key": "A", "reward": 1.0},
                    {"candidate_key": "B", "reward": 0.0},
                ],
            }
            for index in range(2)
        ],
    }

    assert not _audit_failures(audit, expected_worlds=2, prefix="audit")
    audit["candidate_stats"][0]["visits"] = 1
    assert "audit:candidate_visits" in _audit_failures(
        audit,
        expected_worlds=2,
        prefix="audit",
    )


def test_wilson_gate_does_not_treat_raw_rate_as_lower_bound():
    assert _wilson_lower_bound(65, 100) < 0.65
    assert _wilson_lower_bound(75, 100) > 0.65


def test_rotation_gate_checks_observed_seat_and_dealer_balance():
    rows = [
        {
            "opponents": ["strong"],
            "candidate_seat": seat,
            "dealer": dealer,
        }
        for seat in range(2)
        for dealer in range(2)
    ]

    assert not _rotation_balance_failures(
        {"rows": rows},
        mode={"players": 2},
    )
    assert _rotation_balance_failures(
        {"rows": rows[:-1]},
        mode={"players": 2},
    ) == ["rotation_coverage:strong"]


def test_missing_mode_evidence_cannot_pass(tmp_path):
    contract = _contract()
    evidence = {
        "schema_version": "aizipai-strategy-evidence-v1",
        "candidate_id": "candidate",
        "anchor_id": "anchor",
        "mode_evidence": {"2p_off": {}},
    }

    report = evaluate_acceptance(contract, evidence, evidence_base=tmp_path)

    assert not report["strategy_promotion_pass"]
    assert not report["overall_pass"]
    assert any("formal_dataset_not_sealed" in item for item in report["failures"])
    assert any("root:evidence_missing" in item for item in report["failures"])


def _contract():
    return {
        "schema_version": "aizipai-strategy-acceptance-v1",
        "required_modes": [
            {"id": "2p_off", "players": 2, "wildcard_enabled": False}
        ],
        "root_counterfactual": {
            "required_batches_per_state": 5,
            "required_worlds_per_batch": 128,
            "minimum_changed_action_coverage": 1.0,
            "maximum_harmful_overrides": 0,
        },
        "formal_league": {
            "minimum_games": 500,
            "minimum_paired_score_lower_bound": 0.0,
            "minimum_paired_outcome_lower_bound": 0.0,
            "maximum_outcome_sign_test_p": 0.05,
            "minimum_strong_pool_win_lower_bound": 0.5,
            "maximum_decision_budget_fallback_fraction": 0.01,
            "maximum_discard_deadline_fraction": 0.01,
            "maximum_response_deadline_fraction": 0.01,
            "maximum_discard_p95_ms": 10000.0,
            "maximum_response_p95_ms": 10000.0,
        },
        "ordinary_league": {
            "minimum_games": 500,
            "minimum_win_lower_bound": 0.65,
        },
        "device_soak": {
            "minimum_games_per_mode": 20,
            "minimum_primary_mode_games": 100,
            "maximum_cycle_p95_ms": 15000.0,
            "maximum_critical_failures": 0,
            "maximum_strategy_mismatches": 0,
        },
        "external_calibration": {
            "minimum_games": 100,
            "minimum_reviewed_decisions": 500,
            "require_professional_level_supported": True,
        },
    }
