from ai.simulation_trace import canonical_public_state, public_state_identity
from tools.extract_candidate_override_traces import (
    extract_discard_development_manifest,
    extract_override_traces,
)


def test_public_state_identity_ignores_redundant_own_melds():
    compact = _public_state()
    traced = {
        **compact,
        "own_melds": [],
    }

    assert canonical_public_state(compact) == canonical_public_state(traced)
    assert public_state_identity(compact) == public_state_identity(traced)


def test_public_state_identity_includes_passed_claim_state():
    before = _public_state()
    explicit_empty = {
        **before,
        "passed_chi": [],
        "passed_peng": [],
    }
    after = {
        **before,
        "passed_chi": [["七"], []],
        "passed_peng": [[], ["陆"]],
    }

    assert public_state_identity(before) == public_state_identity(explicit_empty)
    assert public_state_identity(before) != public_state_identity(after)


def test_extract_override_traces_strictly_joins_event_and_trace():
    public_state = _public_state()
    identity = {
        "seed": 101,
        "candidate_seat": 0,
        "dealer": 1,
        "opponents": ["independent_balanced"],
    }
    event = {
        "public_view": public_state,
        "public_view_id": public_state_identity(public_state),
        "production_label": "二",
        "baseline_selected_label": "二",
        "validation_selected_label": "三",
        "search_selected_label": "三",
        "confidence_override": True,
    }
    report = {
        "rows": [
            {
                **identity,
                "candidate_discard_events": [event],
                "candidate_won": False,
                "draw": False,
                "candidate_outcome_score": -1.0,
            }
        ]
    }
    baseline = {
        "rows": [
            {
                **identity,
                "candidate_won": True,
                "draw": False,
                "candidate_outcome_score": 2.0,
            }
        ]
    }
    traced_public_state = {**public_state, "own_melds": []}
    traces = [
        {
            **identity,
            "decision_trace": [
                {
                    "phase": "discard",
                    "seat": 0,
                    "sequence": 3,
                    "turn": 2,
                    "public_state": traced_public_state,
                    "public_state_id": public_state_identity(
                        traced_public_state
                    ),
                    "selected_key": "DISCARD:三",
                    "state_before_hash": "full-state-1",
                }
            ],
        }
    ]

    manifest = extract_override_traces(
        report,
        traces=traces,
        baseline_report=baseline,
    )

    assert manifest["alignment_errors"] == 0
    assert manifest["discard_events"] == 1
    assert manifest["override_states"] == 1
    assert manifest["negative_games"] == 1
    assert manifest["score_regressed_games"] == 1
    assert manifest["rows"][0]["state_before_hash"] == "full-state-1"
    assert manifest["rows"][0]["outcome_transition"] == "win_to_loss"
    assert manifest["rows"][0]["baseline_outcome_score"] == 2.0
    assert manifest["rows"][0]["candidate_outcome_score"] == -1.0
    assert manifest["rows"][0]["score_delta"] == -3.0
    assert manifest["rows"][0]["score_direction"] == "regressed"


def test_extract_override_traces_includes_response_disagreements():
    public_state = {
        **_public_state(),
        "pending_card": "四",
        "pending_source_seat": 1,
    }
    identity = {
        "seed": 202,
        "candidate_seat": 0,
        "dealer": 0,
        "opponents": ["independent_pressure"],
    }
    report = {
        "rows": [
            {
                **identity,
                "candidate_discard_events": [],
                "candidate_response_events": [
                    {
                        "public_view": public_state,
                        "response_type": "JOINT",
                        "pending_card": "四",
                        "production_key": "PASS",
                        "search_selected_key": "CHI:option",
                        "empirical_best_key": "CHI:option",
                        "candidate_count": 2,
                        "paired_advantages": [
                            {
                                "candidate_key": "CHI:option",
                                "mean_delta": 0.3,
                                "standard_error": 0.05,
                                "lower_confidence_bound": 0.2,
                                "samples": 128,
                            }
                        ],
                        "confidence_override": True,
                        "disagreement": True,
                    }
                ],
                "candidate_won": True,
                "draw": False,
                "candidate_outcome_score": 2.0,
            }
        ]
    }
    baseline = {
        "rows": [
            {
                **identity,
                "candidate_won": False,
                "draw": False,
            }
        ]
    }
    traces = [
        {
            **identity,
            "decision_trace": [
                {
                    "phase": "response_root",
                    "seat": 0,
                    "sequence": 8,
                    "turn": 5,
                    "public_state": public_state,
                    "public_state_id": public_state_identity(public_state),
                    "selected_key": "CHI:option",
                    "state_before_hash": "full-response-state",
                }
            ],
        }
    ]

    manifest = extract_override_traces(
        report,
        traces=traces,
        baseline_report=baseline,
    )

    assert manifest["discard_events"] == 0
    assert manifest["response_events"] == 1
    assert manifest["discard_override_states"] == 0
    assert manifest["response_override_states"] == 1
    assert manifest["response_empirical_challenger_states"] == 1
    assert manifest["rows"][0]["kind"] == "response"
    assert manifest["rows"][0]["production_key"] == "PASS"
    assert manifest["rows"][0]["selected_key"] == "CHI:option"
    assert (
        manifest["rows"][0]["state_before_hash"]
        == "full-response-state"
    )
    challenger = manifest["response_challenger_rows"][0]
    assert challenger["empirical_best_key"] == "CHI:option"
    assert challenger["online_lower_confidence_bound"] == 0.2
    assert challenger["applied_override"]


def test_development_manifest_keeps_all_changes_and_matched_controls():
    public_states = [
        {**_public_state(), "stock_count": 10 - index}
        for index in range(4)
    ]
    identity = {
        "seed": 303,
        "candidate_seat": 0,
        "dealer": 1,
        "opponents": ["independent_balanced"],
    }
    events = []
    steps = []
    for index, public_state in enumerate(public_states):
        changed = index in {0, 2}
        selected = "三" if changed else "二"
        events.append(
            {
                "public_view": public_state,
                "public_view_id": public_state_identity(public_state),
                "production_label": "二",
                "search_selected_label": selected,
                "confidence_override": index == 0,
                "validation_complete": index != 3,
                "legal_candidate_count": 3 + index,
                "elapsed_ms": 1000 + index * 100,
            }
        )
        steps.append(
            {
                "phase": "discard",
                "seat": 0,
                "sequence": index + 1,
                "turn": index + 1,
                "public_state": public_state,
                "public_state_id": public_state_identity(public_state),
                "selected_key": f"DISCARD:{selected}",
                "state_before_hash": f"state-{index}",
            }
        )
    report = {
        "rows": [
            {
                **identity,
                "candidate_discard_events": events,
                "candidate_won": True,
                "draw": False,
                "candidate_outcome_score": 2.0,
            }
        ]
    }
    baseline = {
        "rows": [
            {
                **identity,
                "candidate_won": False,
                "draw": False,
                "candidate_outcome_score": -1.0,
            }
        ]
    }
    traces = [{**identity, "decision_trace": steps}]

    manifest = extract_discard_development_manifest(
        report,
        traces=traces,
        baseline_report=baseline,
    )

    assert manifest["ok"]
    assert manifest["discard_events"] == 4
    assert manifest["changed_states"] == 2
    assert manifest["confidence_changed_states"] == 1
    assert manifest["nonconfidence_changed_states"] == 1
    assert manifest["control_states"] == 2
    assert manifest["selected_states"] == 4
    assert {row["cohort"] for row in manifest["rows"]} == {
        "changed",
        "matched_unchanged",
    }


def _public_state():
    return {
        "seat": 0,
        "hand": ["二", "三"],
        "all_melds": [[], []],
        "discards": [[], ["九"]],
        "remaining_counts": [["二", 3], ["三", 3], ["九", 2]],
        "stock_count": 10,
        "hand_sizes": [2, 2],
        "pending_card": None,
        "pending_source_seat": None,
    }
