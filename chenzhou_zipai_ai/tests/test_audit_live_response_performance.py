import json

from tools.audit_live_response_performance import analyze_event_log


def _event(timestamp, event_type, frame_id, data, *, decision_id=None):
    row = {
        "timestamp": timestamp,
        "event_type": event_type,
        "frame_id": frame_id,
        "data": data,
    }
    if decision_id is not None:
        row["decision_id"] = decision_id
    return row


def test_live_response_audit_reports_latency_and_missing_peng_evaluation(tmp_path):
    rows = [
        _event("2026-08-14T10:00:00+00:00", "FRAME_CAPTURED", "frame_1", {}),
        _event(
            "2026-08-14T10:00:00.200000+00:00",
            "FRAME_RECOGNIZED",
            "frame_1",
            {"buttons": [{"name": "peng"}, {"name": "pass"}]},
        ),
        _event(
            "2026-08-14T10:00:00.300000+00:00",
            "LEGAL_ACTIONS_GENERATED",
            "frame_1",
            {"legal_actions": [{"type": "PASS", "allowed": True}, {"type": "PENG", "allowed": True}]},
            decision_id="decision_1",
        ),
        _event(
            "2026-08-14T10:00:00.350000+00:00",
            "ACTION_EVALUATED",
            "frame_1",
            {"action": {"type": "PENG"}, "allowed": True, "ev": 10},
            decision_id="decision_1",
        ),
        _event(
            "2026-08-14T10:00:00.500000+00:00",
            "DECISION_SELECTED",
            "frame_1",
            {"selected_action": {"type": "PENG"}},
            decision_id="decision_1",
        ),
        _event(
            "2026-08-14T10:00:01+00:00",
            "TAP_EXECUTED",
            "frame_1",
            {"action_plan": {"action": "peng"}},
            decision_id="decision_1",
        ),
        _event("2026-08-14T10:00:02+00:00", "FRAME_CAPTURED", "frame_2", {}),
        _event(
            "2026-08-14T10:00:02.100000+00:00",
            "FRAME_RECOGNIZED",
            "frame_2",
            {"buttons": [{"name": "chi"}, {"name": "peng"}, {"name": "pass"}]},
        ),
        _event(
            "2026-08-14T10:00:02.200000+00:00",
            "LEGAL_ACTIONS_GENERATED",
            "frame_2",
            {"legal_actions": [{"type": "PENG", "allowed": True}, {"type": "CHI", "allowed": True}]},
            decision_id="decision_2",
        ),
        _event(
            "2026-08-14T10:00:02.400000+00:00",
            "DECISION_SELECTED",
            "frame_2",
            {"selected_action": {"type": "CHI"}},
            decision_id="decision_2",
        ),
    ]
    path = tmp_path / "session_events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = analyze_event_log(path)

    assert report["actions"]["peng"]["count"] == 1
    assert report["actions"]["peng"]["capture_to_tap_log_ms"]["p95"] == 1000.0
    assert report["peng_audit"]["legal_frames"] == 2
    assert report["peng_audit"]["selected_frames"] == 1
    assert report["peng_audit"]["missing_evaluation_frames"] == ["frame_2"]
