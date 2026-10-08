from tools.compare_paired_decision_traces import (
    compare_paired_decision_traces,
)


def _report(*, won: bool, score: float) -> dict:
    return {
        "rows": [
            {
                "opponents": ["independent_balanced"],
                "candidate_seat": 0,
                "dealer": 1,
                "seed": 7,
                "candidate_won": won,
                "draw": False,
                "candidate_outcome_score": score,
            }
        ]
    }


def _trace(selected_key: str) -> list[dict]:
    return [
        {
            "opponents": ["independent_balanced"],
            "candidate_seat": 0,
            "dealer": 1,
            "seed": 7,
            "decision_trace": [
                {
                    "state_before_hash": "state-1",
                    "public_state_id": "public-1",
                    "seat": 0,
                    "phase": "discard",
                    "turn": 3,
                    "selected_key": selected_key,
                    "reason": "test",
                    "public_state": {"hand": ["一", "二"]},
                    "legal_actions": [
                        {"key": "DISCARD:一"},
                        {"key": "DISCARD:二"},
                    ],
                }
            ],
        }
    ]


def test_compare_paired_decision_traces_finds_first_action_change():
    report = compare_paired_decision_traces(
        _report(won=True, score=2.0),
        _trace("DISCARD:一"),
        _report(won=False, score=-3.0),
        _trace("DISCARD:二"),
    )

    assert report["ok"]
    assert report["games"] == 1
    assert report["selected_action_divergences"] == 1
    assert report["candidate_seat_first_divergences"] == 1
    assert report["outcome_regressed"] == 1
    assert report["rows"][0]["baseline_selected_key"] == "DISCARD:一"
    assert report["rows"][0]["candidate_selected_key"] == "DISCARD:二"
    assert report["rows"][0]["score_delta"] == -5.0


def test_compare_paired_decision_traces_reports_identical_path():
    report = compare_paired_decision_traces(
        _report(won=True, score=2.0),
        _trace("DISCARD:一"),
        _report(won=True, score=2.0),
        _trace("DISCARD:一"),
    )

    assert report["ok"]
    assert report["identical_action_paths"] == 1
    assert report["rows"][0]["divergence_type"] == "none"
