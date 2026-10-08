from tools.benchmark_response_roots import response_cases


def test_response_cases_inherit_wildcard_mode_from_legacy_report():
    report = {
        "wildcard_enabled": True,
        "rows": [
            {
                "seed": 7,
                "candidate_seat": 1,
                "dealer": 0,
                "players": 2,
                "opponents": ["baseline"],
                "candidate_response_events": [
                    {
                        "production_key": "PASS",
                        "public_view": {"seat": 1, "hand": ["王"]},
                        "candidates": [
                            {
                                "key": "PASS",
                                "action_type": "PASS",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    cases = response_cases(report)

    assert len(cases) == 1
    assert cases[0]["wildcard_enabled"] is True


def test_response_cases_prefer_event_wildcard_mode():
    report = {
        "wildcard_enabled": False,
        "rows": [
            {
                "seed": 7,
                "candidate_seat": 1,
                "dealer": 0,
                "players": 2,
                "wildcard_enabled": False,
                "opponents": ["baseline"],
                "candidate_response_events": [
                    {
                        "wildcard_enabled": True,
                        "production_key": "PASS",
                        "public_view": {"seat": 1, "hand": ["王"]},
                        "candidates": [
                            {
                                "key": "PASS",
                                "action_type": "PASS",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    cases = response_cases(report)

    assert cases[0]["wildcard_enabled"] is True
