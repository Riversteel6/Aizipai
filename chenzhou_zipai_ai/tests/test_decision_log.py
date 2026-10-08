"""Tests for decision logging."""

import json

from ai.decision_log import append_decision


def test_append_decision_writes_jsonl_record(tmp_path):
    path = tmp_path / "decisions.jsonl"
    append_decision(
        path,
        {
            "screenshot": "a.png",
            "hand": ["一"],
            "decision": {"action": "discard", "label": "一"},
            "action_plan": {"ready": True},
        },
    )

    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["screenshot"] == "a.png"
    assert record["decision"]["label"] == "一"
