from tools.evaluate_public_belief_validator import (
    focused_gate_failures,
    validation_evidence_summary,
    validation_health_details,
)


def test_focused_gate_requires_same_candidate_to_preserve_and_improve() -> None:
    gate = {
        "required_states": 286,
        "maximum_health_errors": 0,
        "require_all_validations_complete": True,
        "maximum_harmful_overrides": 8,
        "maximum_missed_beneficial_actions": 17,
        "minimum_confidently_positive_overrides": 34,
        "require_fewer_harmful_overrides_than_anchor": True,
        "require_lower_mean_regret_than_anchor": True,
        "minimum_mean_improvement_lower_95": 0.0,
    }
    candidate = {
        "harmful_overrides": 8,
        "missed_overrides": 17,
        "confidently_positive_overrides": 34,
        "mean_regret_to_heldout_best": 0.05,
        "improvement_confidence": {"lower_95": 0.001},
    }
    anchor = {
        "harmful_overrides": 20,
        "mean_regret_to_heldout_best": 0.063,
    }
    assert focused_gate_failures(
        candidate,
        anchor=anchor,
        gate=gate,
        health_errors=0,
        incomplete=0,
        states=286,
    ) == []

    candidate["confidently_positive_overrides"] = 33
    assert focused_gate_failures(
        candidate,
        anchor=anchor,
        gate=gate,
        health_errors=0,
        incomplete=0,
        states=286,
    ) == ["confidently_positive_overrides"]


def test_health_details_reports_incomplete_validation_without_stage_error() -> None:
    class Validation:
        complete = False
        coverage = type(
            "Stage",
            (),
            {
                "paired_determinizations": 1,
                "candidates": (type("Candidate", (), {"visits": 1})(),),
                "deadline_interruptions": 0,
                "rollout_invariant_violations": 0,
                "rollout_coverage_failures": 0,
                "determinization_failures": 0,
            },
        )()
        expected_coverage_worlds = 1
        expected_confirmation_worlds = 1
        challenger_evidence = ()

    assert validation_health_details(Validation()) == [
        {"stage": "validation", "incomplete_without_stage_error": True}
    ]


def test_validation_evidence_summary_keeps_calibrated_challenger_chain() -> None:
    candidate = type("Candidate", (), {"to_dict": lambda self: {"label": "一"}})()
    advantage = type(
        "Advantage",
        (),
        {"to_dict": lambda self: {"mean_delta": 0.2}},
    )()
    stage = type("Stage", (), {"paired_advantages": (advantage,)})()
    evidence = type(
        "Evidence",
        (),
        {
            "challenger_label": "二",
            "confidence_override": True,
            "override_basis": "calibrated",
            "calibrated_advantage": 0.3,
            "first_confirmation": stage,
            "second_confirmation": stage,
            "combined": stage,
        },
    )()
    validation = type(
        "Validation",
        (),
        {
            "coverage": type("Coverage", (), {"candidates": (candidate,)})(),
            "challenger_evidence": (evidence,),
        },
    )()

    summary = validation_evidence_summary(validation)

    assert summary["coverage_candidates"] == [{"label": "一"}]
    assert summary["challengers"][0]["calibrated_advantage"] == 0.3
    assert summary["challengers"][0]["combined_paired_advantages"] == [
        {"mean_delta": 0.2}
    ]
