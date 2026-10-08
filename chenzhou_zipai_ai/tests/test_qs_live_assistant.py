import time

import pytest

from tools.qs_live_assistant import run_protocol_vision_live


def test_qs_live_assistant_runs_capture_and_live_loop_together(monkeypatch, tmp_path):
    seen_capture = {}
    seen_loop = {}

    def fake_capture_game_packets(**kwargs):
        seen_capture.update(kwargs)
        kwargs["ready_event"].set()
        kwargs["output_jsonl"].write_text('{"type":"qs_packet"}\n', encoding="utf-8")
        kwargs["output_pcap"].write_bytes(b"pcap")
        return {"qs_packet_count": 1, "live_qs_packet_count": 1}

    def fake_run_loop(**kwargs):
        seen_loop.update(kwargs)
        return {
            "action_plan": {"action": "wait", "ready": False, "clicks": []},
            "executed": False,
        }

    monkeypatch.setattr("tools.qs_live_assistant.capture_game_packets", fake_capture_game_packets)
    monkeypatch.setattr("tools.qs_live_assistant.run_loop", fake_run_loop)

    output_pcap = tmp_path / "live.pcap"
    output_jsonl = tmp_path / "live.jsonl"
    result = run_protocol_vision_live(
        duration_seconds=1,
        interval_seconds=0.2,
        device_id="dev",
        output_pcap=output_pcap,
        output_jsonl=output_jsonl,
        launch_game=False,
        execute_play_actions=False,
        max_steps=1,
    )

    assert seen_capture["output_pcap"] == output_pcap
    assert seen_capture["output_jsonl"] == output_jsonl
    assert seen_capture["launch_game"] is False
    assert seen_loop["protocol_payload"] == output_jsonl
    assert seen_loop["seat_role"] == "auto"
    assert seen_loop["execute_play_actions"] is False
    assert result["capture_done"] is True
    assert result["capture_result"]["live_qs_packet_count"] == 1


def test_qs_live_assistant_stops_capture_when_live_loop_returns(monkeypatch, tmp_path):
    seen_capture = {}

    def fake_capture_game_packets(**kwargs):
        kwargs["ready_event"].set()
        stop_event = kwargs["stop_event"]
        deadline = time.monotonic() + 2.0
        while not stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        seen_capture["stopped"] = stop_event.is_set()
        kwargs["output_jsonl"].write_text('{"type":"qs_packet"}\n', encoding="utf-8")
        kwargs["output_pcap"].write_bytes(b"pcap")
        return {"qs_packet_count": 1, "live_qs_packet_count": 1}

    def fake_run_loop(**kwargs):
        return {
            "action_plan": {"action": "safe_halt", "ready": False, "clicks": []},
            "executed": False,
        }

    monkeypatch.setattr("tools.qs_live_assistant.capture_game_packets", fake_capture_game_packets)
    monkeypatch.setattr("tools.qs_live_assistant.run_loop", fake_run_loop)

    result = run_protocol_vision_live(
        duration_seconds=7200,
        interval_seconds=0.2,
        device_id="dev",
        output_pcap=tmp_path / "live.pcap",
        output_jsonl=tmp_path / "live.jsonl",
        launch_game=False,
        execute_play_actions=True,
        max_steps=1,
    )

    assert seen_capture["stopped"] is True
    assert result["capture_done"] is True


def test_qs_live_assistant_defaults_to_capture_until_live_loop_exits(monkeypatch, tmp_path):
    seen_capture = {}
    seen_loop = {}

    def fake_capture_game_packets(**kwargs):
        seen_capture.update(kwargs)
        kwargs["ready_event"].set()
        kwargs["output_jsonl"].write_text("", encoding="utf-8")
        kwargs["output_pcap"].write_bytes(b"")
        return {}

    def fake_run_loop(**kwargs):
        seen_loop.update(kwargs)
        return {"action_plan": {"action": "safe_halt", "ready": False, "clicks": []}}

    monkeypatch.setattr("tools.qs_live_assistant.capture_game_packets", fake_capture_game_packets)
    monkeypatch.setattr("tools.qs_live_assistant.run_loop", fake_run_loop)

    run_protocol_vision_live(
        device_id="dev",
        output_pcap=tmp_path / "live.pcap",
        output_jsonl=tmp_path / "live.jsonl",
        launch_game=False,
    )

    assert seen_capture["duration_seconds"] == 0
    assert seen_loop["max_steps"] is None


def test_qs_live_assistant_does_not_start_live_loop_when_capture_start_fails(monkeypatch, tmp_path):
    loop_called = []

    def fake_capture_game_packets(**kwargs):
        raise RuntimeError("capture unavailable")

    monkeypatch.setattr("tools.qs_live_assistant.capture_game_packets", fake_capture_game_packets)
    monkeypatch.setattr("tools.qs_live_assistant.run_loop", lambda **kwargs: loop_called.append(True))

    with pytest.raises(RuntimeError, match="Packet capture failed before live loop"):
        run_protocol_vision_live(
            device_id="dev",
            output_pcap=tmp_path / "live.pcap",
            output_jsonl=tmp_path / "live.jsonl",
            launch_game=False,
        )

    assert loop_called == []
