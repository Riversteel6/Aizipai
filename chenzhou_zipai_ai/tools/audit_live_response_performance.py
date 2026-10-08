"""Summarize live CHI/PENG response latency and PENG coverage from event logs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


def analyze_event_log(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    frames: dict[str, dict[str, Any]] = {}
    malformed_rows = 0
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                malformed_rows += 1
                continue
            frame_id = str(row.get("frame_id") or "")
            if not frame_id:
                continue
            frame = frames.setdefault(frame_id, {"evaluated": set()})
            event_type = str(row.get("event_type") or "")
            data = row.get("data") or {}
            timestamp = _timestamp(row.get("timestamp"))
            if event_type == "FRAME_CAPTURED":
                frame.setdefault("captured_at", timestamp)
            elif event_type == "FRAME_RECOGNIZED" and "buttons" in data:
                frame["recognized_at"] = timestamp
                frame["buttons"] = [
                    str(item.get("name") if isinstance(item, dict) else item).lower()
                    for item in data.get("buttons") or []
                ]
            elif event_type == "LEGAL_ACTIONS_GENERATED":
                frame["legal"] = [
                    str(item.get("type") or "").upper()
                    for item in data.get("legal_actions") or []
                    if item.get("allowed", True)
                ]
            elif event_type == "ACTION_EVALUATED":
                action = data.get("action") or {}
                frame["evaluated"].add(str(action.get("type") or "").upper())
            elif event_type == "DECISION_SELECTED":
                selected = data.get("selected_action") or {}
                frame["selected"] = str(selected.get("type") or "").upper()
                frame["decision_at"] = timestamp
            elif event_type == "TAP_EXECUTED":
                plan = data.get("action_plan") or {}
                frame["tap_action"] = str(plan.get("action") or "").lower()
                frame["tap_logged_at"] = timestamp

    action_rows: dict[str, list[dict[str, float]]] = {}
    for frame in frames.values():
        action = frame.get("tap_action")
        required = ("captured_at", "recognized_at", "decision_at", "tap_logged_at")
        if not action or not all(key in frame for key in required):
            continue
        action_rows.setdefault(action, []).append(
            {
                "capture_to_recognize_ms": _elapsed(frame, "captured_at", "recognized_at"),
                "recognize_to_decision_ms": _elapsed(frame, "recognized_at", "decision_at"),
                "decision_to_tap_log_ms": _elapsed(frame, "decision_at", "tap_logged_at"),
                "capture_to_tap_log_ms": _elapsed(frame, "captured_at", "tap_logged_at"),
            }
        )

    peng_frames = {
        frame_id: frame
        for frame_id, frame in frames.items()
        if "PENG" in frame.get("legal", [])
    }
    return {
        "schema_version": "live-response-performance-v1",
        "source": str(source),
        "frame_count": len(frames),
        "malformed_rows": malformed_rows,
        "actions": {
            action: {
                "count": len(rows),
                **{
                    metric: _summary([row[metric] for row in rows])
                    for metric in rows[0]
                },
            }
            for action, rows in sorted(action_rows.items())
        },
        "peng_audit": {
            "legal_frames": len(peng_frames),
            "evaluated_frames": sum("PENG" in frame["evaluated"] for frame in peng_frames.values()),
            "selected_frames": sum(frame.get("selected") == "PENG" for frame in peng_frames.values()),
            "missing_evaluation_frames": sorted(
                frame_id
                for frame_id, frame in peng_frames.items()
                if "PENG" not in frame["evaluated"]
            ),
        },
    }


def _timestamp(value: Any) -> float:
    return datetime.fromisoformat(str(value)).timestamp()


def _elapsed(frame: dict[str, Any], start: str, end: str) -> float:
    return round((float(frame[end]) - float(frame[start])) * 1000.0, 3)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    rank = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "average": round(mean(ordered), 3),
        "median": round(median(ordered), 3),
        "p95": round(ordered[rank], 3),
        "max": round(ordered[-1], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_event_log(args.events)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if not report["peng_audit"]["missing_evaluation_frames"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
