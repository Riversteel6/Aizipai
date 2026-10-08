from tools.benchmark_candidate_responses import response_cases


def test_response_cases_strictly_join_report_events_to_trace_steps():
    view = {
        "seat": 0,
        "hand": ["一", "一", "二"],
        "all_melds": [[], []],
        "discards": [[], []],
        "remaining_counts": [["一", 1], ["二", 3]],
        "stock_count": 10,
        "hand_sizes": [3, 2],
        "pending_card": "一",
        "pending_source_seat": 1,
    }
    identity = {
        "seed": 10,
        "candidate_seat": 0,
        "dealer": 0,
        "players": 2,
        "opponents": ["independent_pressure"],
    }
    report = {
        "rows": [
            {
                **identity,
                "wildcard_enabled": False,
                "candidate_response_events": [
                    {
                        "public_view": view,
                        "production_key": "PASS",
                        "search_selected_key": "PENG:一",
                    }
                ],
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
                    "public_state": {**view, "own_melds": []},
                    "legal_actions": [
                        {"key": "PASS", "type": "PASS"},
                        {
                            "key": "PENG:一",
                            "type": "PENG",
                            "label": "一",
                            "consumed_from_hand": ["一", "一"],
                        },
                    ],
                    "selected_key": "PENG:一",
                }
            ],
        }
    ]

    cases = response_cases(report, traces=traces)

    assert len(cases) == 1
    assert cases[0]["production_key"] == "PASS"
    assert cases[0]["source_selected_key"] == "PENG:一"


def test_response_cases_inherit_wildcard_mode_from_legacy_report():
    view = {
        "seat": 0,
        "hand": ["王", "一", "一"],
        "all_melds": [[], []],
        "discards": [[], []],
        "remaining_counts": [["王", 3], ["一", 2]],
        "stock_count": 10,
        "hand_sizes": [3, 2],
        "pending_card": "一",
        "pending_source_seat": 1,
    }
    identity = {
        "seed": 11,
        "candidate_seat": 0,
        "dealer": 0,
        "players": 2,
        "opponents": ["independent_pressure"],
    }
    report = {
        "wildcard_enabled": True,
        "rows": [
            {
                **identity,
                "candidate_response_events": [
                    {
                        "public_view": view,
                        "production_key": "PASS",
                    }
                ],
            }
        ],
    }
    traces = [
        {
            **identity,
            "wildcard_enabled": True,
            "decision_trace": [
                {
                    "phase": "response_root",
                    "seat": 0,
                    "public_state": {**view, "own_melds": []},
                    "legal_actions": [
                        {"key": "PASS", "type": "PASS"},
                        {
                            "key": "PENG:一",
                            "type": "PENG",
                            "label": "一",
                            "consumed_from_hand": ["一", "一"],
                        },
                    ],
                    "selected_key": "PASS",
                }
            ],
        }
    ]

    cases = response_cases(report, traces=traces)

    assert cases[0]["wildcard_enabled"] is True
