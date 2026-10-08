"""Cross-opponent disagreement audit aggregation tests."""

import json

from tools.recheck_response_disagreements import (
    DEFAULT_OPPONENT_PROFILES,
    _load_cases,
    _stable_selection,
)


def test_deep_recheck_allows_independent_fast_opponent_profiles():
    assert {
        "independent_fast_rollout",
        "independent_fast_pressure",
        "independent_fast_denial",
    }.issubset(DEFAULT_OPPONENT_PROFILES)


def test_load_cases_can_expand_from_disagreements_to_every_eligible_root(tmp_path):
    candidates = [
        {
            "key": "PASS",
            "action_type": "PASS",
            "heuristic_value": 100.0,
            "average_reward": 0.5,
        },
        {
            "key": "PENG:二",
            "action_type": "PENG",
            "heuristic_value": 90.0,
            "average_reward": 0.25,
        },
    ]
    base_event = {
        "players": 2,
        "wildcard_enabled": False,
        "response_type": "PENG",
        "pending_card": "二",
        "production_key": "PASS",
        "search_selected_key": "PASS",
        "public_view": {"seat": 0, "hand": ["二", "二", "九"]},
        "candidates": candidates,
    }
    report_path = tmp_path / "response_roots.json"
    report_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "candidate_response_events": [
                            {**base_event, "disagreement": False},
                            {
                                **base_event,
                                "disagreement": True,
                                "search_selected_key": "PENG:二",
                                "public_view": {
                                    "seat": 0,
                                    "hand": ["二", "二", "八"],
                                },
                            },
                        ]
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    disagreements = _load_cases([report_path])
    all_eligible = _load_cases([report_path], include_all_eligible=True)

    assert len(disagreements) == 1
    assert len(all_eligible) == 2


def test_stable_selection_requires_every_profile_to_be_seed_stable():
    profiles = [
        {
            "usable_runs": 5,
            "stable_selected_key": "PENG:肆",
        },
        {
            "usable_runs": 5,
            "stable_selected_key": None,
        },
    ]

    assert _stable_selection(profiles, repeats=5) is None


def test_stable_selection_requires_same_action_across_profiles():
    profiles = [
        {
            "usable_runs": 5,
            "stable_selected_key": "PASS",
        },
        {
            "usable_runs": 5,
            "stable_selected_key": "PENG:肆",
        },
    ]

    assert _stable_selection(profiles, repeats=5) is None


def test_stable_selection_accepts_one_fully_stable_action():
    profiles = [
        {
            "usable_runs": 5,
            "stable_selected_key": "CHI:sim_chi_001",
        },
        {
            "usable_runs": 5,
            "stable_selected_key": "CHI:sim_chi_001",
        },
    ]

    assert _stable_selection(profiles, repeats=5) == "CHI:sim_chi_001"
