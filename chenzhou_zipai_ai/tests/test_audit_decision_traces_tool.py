import json

from tools.audit_decision_traces import (
    _AuditCheckpoint,
    _is_balanced_rotation,
    _load_discard_comparisons,
    _load_state_opponents,
    _read_jsonl,
    _sample_game_traces,
    _sort_tasks_by_state_order,
)


def test_sample_game_traces_keeps_phase_families_and_trace_order():
    traces = [
        {"sequence": 1, "phase": "discard"},
        {"sequence": 2, "phase": "discard"},
        {"sequence": 3, "phase": "response_root"},
        {"sequence": 4, "phase": "self_hu"},
        {"sequence": 5, "phase": "discard"},
    ]

    sampled = _sample_game_traces(
        traces,
        maximum=3,
        seed=20260728,
    )

    assert [item["sequence"] for item in sampled] == sorted(
        item["sequence"] for item in sampled
    )
    assert {item["phase"] for item in sampled} == {
        "discard",
        "response_root",
        "self_hu",
    }


def test_sample_game_traces_zero_limit_preserves_all_traces():
    traces = [
        {"sequence": 1, "phase": "discard"},
        {"sequence": 2, "phase": "response_root"},
    ]

    assert _sample_game_traces(traces, maximum=0, seed=1) == traces


def test_balanced_rotation_is_deterministic_from_deal_seed():
    game = {
        "players": 3,
        "seed": 20260728,
        "candidate_seat": 20260728 % 3,
        "dealer": (20260728 // 3) % 3,
    }

    assert _is_balanced_rotation(game)
    game["dealer"] = (game["dealer"] + 1) % 3
    assert not _is_balanced_rotation(game)


def test_load_discard_comparisons_extracts_action_pairs(tmp_path):
    path = tmp_path / "comparison.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "state_before_hash": "state-1",
                        "baseline_selected_key": "DISCARD:二",
                        "candidate_selected_key": "DISCARD:三",
                    },
                    {
                        "state_before_hash": "same-action",
                        "baseline_selected_key": "DISCARD:四",
                        "candidate_selected_key": "DISCARD:四",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert _load_discard_comparisons(path) == {
        "state-1": ("二", "三")
    }


def test_load_state_opponents_extracts_manifest_assignments(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "state_before_hash": "state-a",
                        "opponent_stratum": "aggressive_meld",
                    },
                    {
                        "state_before_hash": "state-b",
                        "opponent_stratum": "defensive_search",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _load_state_opponents(path) == {
        "state-a": ("aggressive_meld",),
        "state-b": ("defensive_search",),
    }


def test_read_jsonl_skips_parsing_nonmatching_lines(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(
            (
                '{"state_before_hash":"state-a","value":1}',
                '{"state_before_hash":"state-b","value":2}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    rows = list(
        _read_jsonl(
            path,
            required_substrings={"state-b"},
        )
    )

    assert rows == [None, {"state_before_hash": "state-b", "value": 2}]


def test_audit_checkpoint_round_trips_out_of_order_results(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    metadata = {"task_count": 2, "fingerprint": "fixed"}
    checkpoint = _AuditCheckpoint(path)
    checkpoint.create(metadata)
    checkpoint.append_result({"task_index": 1, "audit": {"complete": True}})
    checkpoint.append_result({"task_index": 0, "audit": {"complete": True}})
    checkpoint.mark_complete(2)

    results, complete = checkpoint.load(metadata)

    assert complete
    assert sorted(results) == [0, 1]


def test_audit_checkpoint_repairs_only_a_trailing_partial_record(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    metadata = {"task_count": 1, "fingerprint": "fixed"}
    checkpoint = _AuditCheckpoint(path)
    checkpoint.create(metadata)
    checkpoint.append_result({"task_index": 0, "audit": {"complete": True}})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"record_type":"complete"')

    results, complete = checkpoint.load(metadata)

    assert not complete
    assert sorted(results) == [0]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_sort_tasks_by_state_order_uses_manifest_order():
    tasks = [
        {
            "trace": {"state_before_hash": "fast", "sequence": 2},
            "game": {"seed": 20},
        },
        {
            "trace": {"state_before_hash": "slow", "sequence": 1},
            "game": {"seed": 10},
        },
    ]

    _sort_tasks_by_state_order(
        tasks,
        state_order=("slow", "fast"),
    )

    assert [task["trace"]["state_before_hash"] for task in tasks] == [
        "slow",
        "fast",
    ]
