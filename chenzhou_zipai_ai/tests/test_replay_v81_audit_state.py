from __future__ import annotations

import json

import pytest

from ai.ismcts import public_view_from_dict
from engine.rules import rules_for_room
from tools.replay_v81_audit_state import _response_types, _state_for_case, load_case


def test_load_case_selects_exact_seed_opponent_and_event(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    rows = [
        {"seed": 7, "opponent": "a", "event_index": 1},
        {"seed": 7, "opponent": "b", "event_index": 2},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    selected = load_case(path, seed=7, opponent="b", event_index=2)

    assert selected["opponent"] == "b"
    assert selected["_source_line"] == 2


def test_load_case_rejects_ambiguous_seed(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"seed":7,"opponent":"a"}\n{"seed":7,"opponent":"b"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="audit_case_ambiguous"):
        load_case(path, seed=7)


def test_response_state_uses_logged_root_action_types() -> None:
    case = {
        "category": "response_internal_better_alternative_ignored",
        "pending_card": "一",
        "production_key": "PENG:一",
        "actual_selected_key": "PENG:一",
        "empirical_best_key": "PASS",
    }
    view = public_view_from_dict(
        {
            "seat": 0,
            "hand": ["一", "一", "二"],
            "all_melds": [[], []],
            "discards": [[], []],
            "remaining_counts": [["三", 1]],
            "stock_count": 1,
            "hand_sizes": [3, 1],
            "pending_card": "一",
            "pending_source_seat": 1,
        }
    )

    assert _response_types(case) == {"PASS", "PENG"}
    state, root_kind = _state_for_case(case, view, rules_for_room(players=2))

    assert root_kind == "response"
    assert {item["type"] for item in state["legal_actions"]} == {"PASS", "PENG"}
    assert state["pending_card"] == "一"


def test_legal_hu_replay_state_is_not_converted_to_discard() -> None:
    case = {"category": "legal_hu_passed", "pending_card": "一"}
    view = public_view_from_dict(
        {
            "seat": 0,
            "hand": ["二"],
            "all_melds": [[], []],
            "discards": [[], []],
            "remaining_counts": [],
            "stock_count": 0,
            "hand_sizes": [1, 0],
            "pending_card": "一",
            "pending_source_seat": 1,
        }
    )

    state, root_kind = _state_for_case(case, view, rules_for_room(players=2))

    assert root_kind == "response"
    assert {item["type"] for item in state["legal_actions"]} == {"HU", "PASS"}
