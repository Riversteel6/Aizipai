"""Run PCAPdroid QS capture and the vision live assistant together."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.live_assistant import DEFAULT_DEVICE, run_loop
from tools.pcapdroid_live_capture import GAME_PACKAGE, capture_game_packets
from chenzhou_zipai_ai.game_logging import GameLogger, LoggerConfig
from engine.rules import ROOM_MODE_PRESETS, rules_for_room

def run_protocol_vision_live(
    *,
    duration_seconds: int = 0,
    interval_seconds: float = 1.5,
    device_id: str = DEFAULT_DEVICE,
    port: int = 5123,
    output_pcap: str | Path | None = None,
    output_jsonl: str | Path | None = None,
    app_filter: str = GAME_PACKAGE,
    launch_game: bool = True,
    expected_total: int = 20,
    auto_seat: bool = True,
    execute_play_actions: bool = False,
    execute_settlement_ready: bool = False,
    capture_after_action: bool = False,
    memory_file: str | Path = "logs/alphadog_live_memory.json",
    guard_file: str | Path = "logs/live_action_guard.json",
    session_log_dir: str | Path | None = None,
    reset_memory: bool = False,
    simulations: int = 0,
    max_steps: int | None = None,
    continue_on_halt: bool = False,
    players: int | None = None,
    room_mode: str | None = None,
) -> dict[str, Any]:
    pcap_path, jsonl_path = _default_outputs(output_pcap=output_pcap, output_jsonl=output_jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("", encoding="utf-8")

    capture_result: dict[str, Any] = {}
    capture_error: str | None = None
    stop_capture_event = threading.Event()
    capture_ready_event = threading.Event()

    def capture_worker() -> None:
        nonlocal capture_result, capture_error
        try:
            capture_result = capture_game_packets(
                duration_seconds=duration_seconds,
                port=port,
                output_pcap=pcap_path,
                output_jsonl=jsonl_path,
                app_filter=app_filter,
                launch_game=launch_game,
                device_id=device_id,
                stop_event=stop_capture_event,
                ready_event=capture_ready_event,
            )
        except Exception as exc:  # pragma: no cover - reported to caller.
            capture_error = f"{type(exc).__name__}: {exc}"
            capture_ready_event.set()

    capture_thread = threading.Thread(target=capture_worker, daemon=True)
    capture_thread.start()
    _wait_for_capture_ready(capture_ready_event, capture_thread, timeout_seconds=25.0)
    if not capture_ready_event.is_set():
        stop_capture_event.set()
        capture_thread.join(timeout=5.0)
        raise TimeoutError("Packet capture did not become ready within 25 seconds")
    if capture_error is not None:
        stop_capture_event.set()
        capture_thread.join(timeout=5.0)
        raise RuntimeError(f"Packet capture failed before live loop: {capture_error}")
    logger = None
    if session_log_dir is not None:
        logger = GameLogger(
            Path(session_log_dir),
            config=LoggerConfig(save_before_after_action=True, save_decision_frames=True, save_error_frames=True),
        )

    resolved_max_steps = max_steps
    if resolved_max_steps is None and duration_seconds > 0:
        resolved_max_steps = max(1, math.ceil(duration_seconds / max(interval_seconds, 0.1)) + 2)

    def capture_health_check() -> str | None:
        if capture_error is not None:
            return f"packet_capture_failed:{capture_error}"
        if not capture_thread.is_alive():
            return "packet_capture_stopped"
        return None

    live_result = run_loop(
        device_id=device_id,
        expected_total=expected_total,
        memory_file=memory_file,
        guard_file=guard_file,
        reset_memory=reset_memory,
        execute_settlement_ready=execute_settlement_ready,
        execute_play_actions=execute_play_actions,
        simulations=simulations,
        interval_seconds=interval_seconds,
        max_steps=resolved_max_steps,
        seat_role="auto" if auto_seat else "manual",
        capture_after_action=capture_after_action,
        protocol_payload=jsonl_path,
        logger=logger,
        continue_on_halt=continue_on_halt,
        external_health_check=capture_health_check,
        players=players,
        room_mode=room_mode,
    )

    stop_capture_event.set()
    capture_thread.join(timeout=15.0)
    return {
        "pcap": str(pcap_path),
        "jsonl": str(jsonl_path),
        "capture_done": not capture_thread.is_alive(),
        "capture_error": capture_error,
        "capture_result": capture_result,
        "live_result": live_result,
        "session_log_dir": str(session_log_dir) if session_log_dir is not None else None,
        "session_id": getattr(logger, "session_id", None) if logger is not None else None,
    }


def _default_outputs(*, output_pcap: str | Path | None, output_jsonl: str | Path | None) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcap_path = Path(output_pcap) if output_pcap else WORKSPACE / "dist" / "captures" / f"pcapdroid_qs_live_{stamp}.pcap"
    jsonl_path = Path(output_jsonl) if output_jsonl else ROOT / "logs" / f"qs_packets_live_{stamp}.jsonl"
    return pcap_path, jsonl_path


def _wait_for_protocol_log(path: Path, *, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.1)


def _wait_for_capture_ready(ready_event: threading.Event, capture_thread: threading.Thread, *, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if ready_event.is_set() or not capture_thread.is_alive():
            return
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QS packet capture and protocol+vision live assistant together.")
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=0,
        help="Capture duration; 0 keeps capture active until the live loop exits.",
    )
    parser.add_argument("--interval-seconds", type=float, default=1.5)
    parser.add_argument("--device-id", default=DEFAULT_DEVICE)
    parser.add_argument("--port", type=int, default=5123)
    parser.add_argument("--output-pcap", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--app-filter", default=GAME_PACKAGE)
    parser.add_argument("--no-launch-game", action="store_true")
    parser.add_argument("--expected-total", type=int, default=20)
    parser.add_argument("--manual-seat", action="store_true")
    parser.add_argument("--execute-play-actions", action="store_true")
    parser.add_argument("--execute-settlement-ready", action="store_true")
    parser.add_argument("--capture-after-action", action="store_true")
    parser.add_argument("--memory-file", default="logs/alphadog_live_memory.json")
    parser.add_argument("--guard-file", default="logs/live_action_guard.json")
    parser.add_argument("--session-log-dir", default=None)
    parser.add_argument("--reset-memory", action="store_true")
    parser.add_argument("--simulate", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--continue-on-halt", action="store_true")
    parser.add_argument("--players", type=int, choices=(2, 3), default=None)
    parser.add_argument("--room-mode", choices=tuple(ROOM_MODE_PRESETS), default=None)
    args = parser.parse_args()
    rules_for_room(players=args.players, room_mode=args.room_mode)
    memory_file = args.memory_file
    guard_file = args.guard_file
    if args.room_mode is not None:
        mode_key = args.room_mode.replace("-", "_")
        if memory_file == "logs/alphadog_live_memory.json":
            memory_file = f"logs/alphadog_live_memory_{mode_key}.json"
        if guard_file == "logs/live_action_guard.json":
            guard_file = f"logs/live_action_guard_{mode_key}.json"

    result = run_protocol_vision_live(
        duration_seconds=args.duration_seconds,
        interval_seconds=args.interval_seconds,
        device_id=args.device_id,
        port=args.port,
        output_pcap=args.output_pcap,
        output_jsonl=args.output_jsonl,
        app_filter=args.app_filter,
        launch_game=not args.no_launch_game,
        expected_total=args.expected_total,
        auto_seat=not args.manual_seat,
        execute_play_actions=args.execute_play_actions,
        execute_settlement_ready=args.execute_settlement_ready,
        capture_after_action=args.capture_after_action,
        memory_file=memory_file,
        guard_file=guard_file,
        session_log_dir=args.session_log_dir,
        reset_memory=args.reset_memory,
        simulations=args.simulate,
        max_steps=args.max_steps,
        continue_on_halt=args.continue_on_halt,
        players=args.players,
        room_mode=args.room_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
