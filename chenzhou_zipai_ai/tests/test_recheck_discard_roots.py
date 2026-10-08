import json
from types import SimpleNamespace

from tools.recheck_discard_roots import (
    _apply_compare_pairs,
    _load_cases,
    _parse_compare_pairs,
    _search_run_complete,
    _selected_vs_production_adjusted_margin,
    _selected_vs_production_confidence_lower_bound,
    _selected_vs_production_reward_margin,
    _strict_stable_override_label,
)


def test_load_cases_accepts_progressive_all_action_discard_event(tmp_path):
    report_path = tmp_path / "progressive.json"
    report_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "candidate_discard_events": [
                            {
                                "search_attempted": True,
                                "search_selected_label": "三",
                                "disagreement": True,
                                "public_view": {"seat": 0, "hand": ["二", "三"]},
                                "players": 2,
                                "wildcard_enabled": False,
                                "seed": 10,
                                "candidate_seat": 0,
                                "dealer": 1,
                                "production_label": "二",
                                "candidates": [
                                    {
                                        "label": "二",
                                        "heuristic_value": 100.0,
                                    },
                                    {
                                        "label": "三",
                                        "heuristic_value": 80.0,
                                    },
                                ],
                            }
                        ]
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = _load_cases([report_path])

    assert len(cases) == 1
    assert cases[0]["production_label"] == "二"
    assert cases[0]["source_search_selected_label"] == "三"
    assert cases[0]["source_disagreement"]
    assert cases[0]["heuristic_gap"] == 20.0


def test_compare_pair_selects_one_root_and_exactly_two_labels(tmp_path):
    report_path = tmp_path / "pair.json"
    report_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "candidate_discard_events": [
                            {
                                "search_attempted": True,
                                "public_view_id": "view-a",
                                "public_view": {"seat": 0, "hand": ["二", "三", "四"]},
                                "players": 2,
                                "wildcard_enabled": False,
                                "seed": 10,
                                "candidate_seat": 0,
                                "dealer": 1,
                                "production_label": "二",
                                "candidates": [
                                    {"label": "二", "heuristic_value": 100.0},
                                    {"label": "三", "heuristic_value": 90.0},
                                    {"label": "四", "heuristic_value": 80.0},
                                ],
                            }
                        ]
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pairs = _parse_compare_pairs(["view-a=二,四"])
    selected = _apply_compare_pairs(_load_cases([report_path]), pairs)

    assert len(selected) == 1
    assert selected[0]["public_view_id"] == "view-a"
    assert selected[0]["comparison_labels"] == ("二", "四")


def test_compare_pair_rejects_an_old_label_that_is_not_production(tmp_path):
    report_path = tmp_path / "pair.json"
    report_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "candidate_discard_events": [
                            {
                                "eligible": True,
                                "public_view_id": "view-a",
                                "public_view": {"seat": 0, "hand": ["二", "三"]},
                                "players": 2,
                                "wildcard_enabled": False,
                                "seed": 10,
                                "candidate_seat": 0,
                                "dealer": 1,
                                "production_label": "二",
                                "candidates": [
                                    {"label": "二", "heuristic_value": 100.0},
                                    {"label": "三", "heuristic_value": 90.0},
                                ],
                            }
                        ]
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pairs = _parse_compare_pairs(["view-a=三,二"])

    try:
        _apply_compare_pairs(_load_cases([report_path]), pairs)
    except ValueError as exc:
        assert str(exc) == "compare_pair_old_label_mismatch:view-a"
    else:
        raise AssertionError("old label mismatch should fail")


def test_reward_margin_reports_a_tie_as_zero():
    candidates = [
        {"label": "二", "visits": 4, "average_reward": 1.62},
        {"label": "十", "visits": 4, "average_reward": 1.62},
    ]

    margin = _selected_vs_production_reward_margin(
        candidates,
        selected_label="十",
        production_label="二",
    )

    assert margin == 0.0


def test_immediate_response_margin_uses_sampled_adjusted_value():
    candidates = [
        {"label": "二", "samples": 16, "adjusted_value": 900.0},
        {"label": "十", "samples": 16, "adjusted_value": 1025.0},
    ]

    margin = _selected_vs_production_adjusted_margin(
        candidates,
        selected_label="十",
        production_label="二",
    )

    assert margin == 125.0


def test_confidence_margin_uses_paired_candidate_lower_bound():
    lower_bound = _selected_vs_production_confidence_lower_bound(
        [
            {
                "candidate_key": "十",
                "preferred_key": "二",
                "lower_confidence_bound": 0.375,
            }
        ],
        selected_label="十",
        production_label="二",
    )

    assert lower_bound == 0.375


def test_confidence_margin_is_absent_when_production_is_kept():
    assert (
        _selected_vs_production_confidence_lower_bound(
            [],
            selected_label="二",
            production_label="二",
        )
        is None
    )


def test_stable_tie_cannot_be_promoted_as_an_override():
    runs = [
        {"strictly_beats_production": False},
        {"strictly_beats_production": False},
        {"strictly_beats_production": False},
    ]

    assert (
        _strict_stable_override_label(
            runs,
            stable_selected_label="十",
            production_label="二",
        )
        is None
    )


def test_stable_strict_reward_advantage_can_reach_the_override_gate():
    runs = [
        {"strictly_beats_production": True},
        {"strictly_beats_production": True},
        {"strictly_beats_production": True},
    ]

    assert _strict_stable_override_label(
        runs,
        stable_selected_label="十",
        production_label="二",
    ) == "十"


def test_full_rollout_run_requires_every_requested_paired_world():
    search = SimpleNamespace(
        used_search=True,
        deadline_interruptions=1,
        rollout_invariant_violations=0,
        rollout_coverage_failures=0,
        paired_determinizations=31,
    )

    assert not _search_run_complete(
        search,
        candidate_rows=[
            {"label": "二", "visits": 31},
            {"label": "十", "visits": 31},
        ],
        search_mode="full_rollout",
        expected_paired_worlds=32,
    )


def test_full_rollout_run_accepts_exact_complete_candidate_coverage():
    search = SimpleNamespace(
        used_search=True,
        deadline_interruptions=0,
        rollout_invariant_violations=0,
        rollout_coverage_failures=0,
        paired_determinizations=32,
    )

    assert _search_run_complete(
        search,
        candidate_rows=[
            {"label": "二", "visits": 32},
            {"label": "十", "visits": 32},
        ],
        search_mode="full_rollout",
        expected_paired_worlds=32,
    )
