from tools.benchmark_candidate_roots import (
    _discard_trace_cases,
    _strategy_runtime_error_reasons,
)


def test_discard_trace_cases_reads_candidate_public_roots() -> None:
    public_state = {
        "seat": 1,
        "hand": ["一", "二", "三"],
        "melds": [],
        "discards": [],
        "current_card": None,
        "source_seat": None,
        "turn": 2,
        "remaining_cards": 50,
    }
    cases = _discard_trace_cases(
        iter(
            [
                {
                    "seed": 17,
                    "players": 2,
                    "wildcard_enabled": False,
                    "candidate_seat": 1,
                    "dealer": 0,
                    "opponents": ["pressure"],
                    "decision_trace": [
                        {
                            "phase": "discard",
                            "seat": 1,
                            "policy": "professional_brain_v2",
                            "selected_key": "DISCARD:一",
                            "public_state": public_state,
                        },
                        {
                            "phase": "discard",
                            "seat": 0,
                            "policy": "pressure",
                            "selected_key": "DISCARD:二",
                            "public_state": public_state,
                        },
                    ],
                }
            ]
        ),
        source_policy="professional_brain_v2",
    )

    assert len(cases) == 1
    assert cases[0]["source_selected_label"] == "一"
    assert cases[0]["candidate_seat"] == 1
    assert cases[0]["matchup"] == ["pressure"]


def test_strategy_runtime_errors_ignore_budget_fallbacks() -> None:
    reasons = _strategy_runtime_error_reasons(
        [
            {"validation_error": None},
            {
                "validation_error": (
                    "decision_budget_validation_incomplete"
                )
            },
            {"validation_error": "AttributeError:missing_field"},
        ]
    )

    assert reasons == ["AttributeError:missing_field"]
