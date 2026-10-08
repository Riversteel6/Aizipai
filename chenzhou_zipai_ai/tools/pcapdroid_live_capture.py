"""Drive PCAPdroid TCP exporter and save a local PCAP.

This controls only the installed PCAPdroid app and Android system permission
dialogs. It does not install, uninstall, modify, or tap inside the game app.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture.adb_capture import run_adb
from tools.qs_packet_extract import QSStreamExtractor, append_qs_jsonl, extract_qs_packets, write_qs_jsonl

PCAPDROID_COMPONENT = "com.emanuelef.remote_capture/.activities.CaptureCtrl"
GAME_PACKAGE = "com.jiahe.kaixinpaohuzi"


@dataclass
class ReceiverStats:
    bytes_received: int = 0
    connections: int = 0
    live_qs_packets: int = 0
    error: str | None = None


def capture_game_packets(
    *,
    duration_seconds: int = 45,
    port: int = 5123,
    output_pcap: str | Path | None = None,
    output_jsonl: str | Path | None = None,
    app_filter: str = GAME_PACKAGE,
    launch_game: bool = True,
    device_id: str | None = None,
    stop_event: threading.Event | None = None,
    ready_event: threading.Event | None = None,
) -> dict[str, Any]:
    pcap_path = Path(output_pcap) if output_pcap else _default_pcap_path()
    jsonl_path = Path(output_jsonl) if output_jsonl else pcap_path.with_suffix(".jsonl")
    pcap_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    archived_outputs = [
        str(path)
        for path in (
            _archive_existing_output(pcap_path),
            _archive_existing_output(jsonl_path),
        )
        if path is not None
    ]
    jsonl_path.write_text("", encoding="utf-8")

    receiver_stop_event = threading.Event()
    stats = ReceiverStats()
    receiver = threading.Thread(
        target=_tcp_receiver,
        args=("127.0.0.1", port, pcap_path, jsonl_path, receiver_stop_event, stats),
        daemon=True,
    )
    receiver.start()
    time.sleep(0.2)

    run_adb(["reverse", f"tcp:{port}", f"tcp:{port}"], device_id=device_id, timeout=15)
    prompts: list[dict[str, Any]] = []
    try:
        _stop_pcapdroid(device_id=device_id)
        prompts.extend(_tap_allowed_prompts(device_id=device_id, max_seconds=5))

        _start_pcapdroid_tcp_exporter(port=port, app_filter=app_filter, device_id=device_id)
        prompts.extend(_tap_allowed_prompts(device_id=device_id, max_seconds=12))

        if launch_game:
            _launch_game(device_id=device_id)

        if ready_event is not None:
            ready_event.set()

        deadline = None if duration_seconds <= 0 and stop_event is not None else time.time() + max(0, duration_seconds)
        while deadline is None or time.time() < deadline:
            if stop_event is not None and stop_event.is_set():
                break
            remaining = 0.2 if deadline is None else max(0.0, deadline - time.time())
            time.sleep(min(0.2, remaining))
    finally:
        if ready_event is not None:
            ready_event.set()
        _stop_pcapdroid(device_id=device_id)
        prompts.extend(_tap_allowed_prompts(device_id=device_id, max_seconds=5))
        if launch_game:
            _launch_game(device_id=device_id)
        receiver_stop_event.set()
        receiver.join(timeout=5)

    packets = extract_qs_packets(pcap_path.read_bytes(), source=str(pcap_path))
    write_qs_jsonl(packets, jsonl_path)
    return {
        "pcap": str(pcap_path),
        "jsonl": str(jsonl_path),
        "bytes_received": stats.bytes_received,
        "tcp_connections": stats.connections,
        "receiver_error": stats.error,
        "live_qs_packet_count": stats.live_qs_packets,
        "qs_packet_count": len(packets),
        "permission_taps": prompts,
        "archived_outputs": archived_outputs,
    }


def _archive_existing_output(path: Path, *, timestamp: str | None = None) -> Path | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, archive_path)
    return archive_path


def _tcp_receiver(host: str, port: int, output: Path, live_jsonl: Path, stop_event: threading.Event, stats: ReceiverStats) -> None:
    try:
        stream_extractor = QSStreamExtractor(source="pcapdroid_tcp_exporter_live")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(1)
            server.settimeout(1)
            with output.open("wb") as handle:
                conn: socket.socket | None = None
                while not stop_event.is_set():
                    if conn is None:
                        try:
                            conn, _ = server.accept()
                            conn.settimeout(1)
                            stats.connections += 1
                        except TimeoutError:
                            continue
                    try:
                        chunk = conn.recv(65536)
                    except TimeoutError:
                        continue
                    except OSError:
                        conn = None
                        continue
                    if not chunk:
                        conn.close()
                        conn = None
                        continue
                    handle.write(chunk)
                    handle.flush()
                    stats.bytes_received += len(chunk)
                    packets = stream_extractor.feed(chunk)
                    if packets:
                        written = append_qs_jsonl(
                            packets,
                            live_jsonl,
                            start_index=stats.live_qs_packets + 1,
                        )
                        stats.live_qs_packets += written
                if conn is not None:
                    conn.close()
    except Exception as exc:  # pragma: no cover - reported to the caller.
        stats.error = str(exc)


def _start_pcapdroid_tcp_exporter(*, port: int, app_filter: str, device_id: str | None) -> None:
    run_adb(
        [
            "shell",
            "am",
            "start",
            "-e",
            "action",
            "start",
            "-e",
            "pcap_dump_mode",
            "tcp_exporter",
            "-e",
            "collector_ip_address",
            "127.0.0.1",
            "--ei",
            "collector_port",
            str(port),
            "-e",
            "app_filter",
            app_filter,
            "--ez",
            "full_payload",
            "true",
            "--ez",
            "root_capture",
            "false",
            "-n",
            PCAPDROID_COMPONENT,
        ],
        device_id=device_id,
        timeout=15,
    )


def _stop_pcapdroid(*, device_id: str | None) -> None:
    try:
        run_adb(["shell", "am", "start", "-e", "action", "stop", "-n", PCAPDROID_COMPONENT], device_id=device_id, timeout=15)
    except Exception:
        pass


def _launch_game(*, device_id: str | None) -> None:
    run_adb(
        ["shell", "monkey", "-p", GAME_PACKAGE, "-c", "android.intent.category.LAUNCHER", "1"],
        device_id=device_id,
        timeout=15,
    )


def _tap_allowed_prompts(*, device_id: str | None, max_seconds: float) -> list[dict[str, Any]]:
    tapped: list[dict[str, Any]] = []
    deadline = time.time() + max_seconds
    allowed_packages = {"com.emanuelef.remote_capture", "com.android.permissioncontroller", "com.android.vpndialogs"}
    allow_texts = {"允许", "确定", "始终允许", "ALLOW", "OK", "同意"}
    while time.time() < deadline:
        nodes = _dump_ui_nodes(device_id=device_id)
        candidate = None
        for node in nodes:
            if node.get("package") not in allowed_packages:
                continue
            text = str(node.get("text") or "").strip()
            resource_id = str(node.get("resource-id") or "")
            if text in allow_texts or resource_id.endswith(("allow_btn", "permission_allow_button", "button1")):
                candidate = node
        if candidate is None:
            time.sleep(0.5)
            continue
        x, y = _bounds_center(str(candidate["bounds"]))
        run_adb(["shell", "input", "tap", str(x), str(y)], device_id=device_id, timeout=10)
        tapped.append({"text": candidate.get("text"), "resource_id": candidate.get("resource-id"), "x": x, "y": y})
        time.sleep(1)
    return tapped


def _dump_ui_nodes(*, device_id: str | None) -> list[dict[str, str]]:
    try:
        run_adb(["shell", "uiautomator", "dump", "/sdcard/pcap_ui.xml"], device_id=device_id, timeout=10)
        xml_text = run_adb(["exec-out", "cat", "/sdcard/pcap_ui.xml"], device_id=device_id, timeout=10)
    except Exception:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    return [dict(node.attrib) for node in root.iter("node")]


def _bounds_center(bounds: str) -> tuple[int, int]:
    left_top, right_bottom = bounds.strip("[]").split("][")
    left, top = [int(value) for value in left_top.split(",")]
    right, bottom = [int(value) for value in right_bottom.split(",")]
    return (left + right) // 2, (top + bottom) // 2


def _default_pcap_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return WORKSPACE / "dist" / "captures" / f"pcapdroid_qs_{stamp}.pcap"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=int, default=45)
    parser.add_argument("--port", type=int, default=5123)
    parser.add_argument("--output-pcap", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--app-filter", default=GAME_PACKAGE)
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--no-launch-game", action="store_true")
    args = parser.parse_args()
    result = capture_game_packets(
        duration_seconds=args.duration_seconds,
        port=args.port,
        output_pcap=args.output_pcap,
        output_jsonl=args.output_jsonl,
        app_filter=args.app_filter,
        launch_game=not args.no_launch_game,
        device_id=args.device_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
