"""ADB tap helpers."""

from __future__ import annotations

import argparse
import random

from capture.adb_capture import run_adb


def jitter_value(value: int, radius: int) -> int:
    if radius <= 0:
        return value
    return value + random.randint(-radius, radius)


def tap(x: int, y: int, device_id: str | None = None, jitter_px: int = 0) -> None:
    x = jitter_value(x, jitter_px)
    y = jitter_value(y, jitter_px)
    run_adb(["shell", "input", "tap", str(x), str(y)], device_id=device_id)


def drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: int = 450,
    device_id: str | None = None,
    start_jitter_px: int = 0,
    end_jitter_x_px: int = 0,
    end_jitter_y_px: int = 0,
    duration_jitter_ms: int = 0,
) -> None:
    start_x = jitter_value(start_x, start_jitter_px)
    start_y = jitter_value(start_y, start_jitter_px)
    end_x = jitter_value(end_x, end_jitter_x_px)
    end_y = jitter_value(end_y, end_jitter_y_px)
    duration_ms = max(1, jitter_value(duration_ms, duration_jitter_ms))

    run_adb(
        [
            "shell",
            "input",
            "swipe",
            str(start_x),
            str(start_y),
            str(end_x),
            str(end_y),
            str(duration_ms),
        ],
        device_id=device_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tap or drag on an Android device screen.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tap_parser = subparsers.add_parser("tap", help="Tap a coordinate.")
    tap_parser.add_argument("x", type=int)
    tap_parser.add_argument("y", type=int)
    tap_parser.add_argument("--jitter-px", type=int, default=0)
    tap_parser.add_argument("--device-id")

    drag_parser = subparsers.add_parser("drag", help="Drag from one coordinate to another.")
    drag_parser.add_argument("start_x", type=int)
    drag_parser.add_argument("start_y", type=int)
    drag_parser.add_argument("end_x", type=int)
    drag_parser.add_argument("end_y", type=int)
    drag_parser.add_argument("--duration-ms", type=int, default=450)
    drag_parser.add_argument("--start-jitter-px", type=int, default=0)
    drag_parser.add_argument("--end-jitter-x-px", type=int, default=0)
    drag_parser.add_argument("--end-jitter-y-px", type=int, default=0)
    drag_parser.add_argument("--duration-jitter-ms", type=int, default=0)
    drag_parser.add_argument("--device-id")

    parser.add_argument("--device-id")
    args = parser.parse_args()

    if args.command == "tap":
        tap(args.x, args.y, args.device_id, args.jitter_px)
        print(f"Tapped {args.x},{args.y}")
    elif args.command == "drag":
        drag(
            args.start_x,
            args.start_y,
            args.end_x,
            args.end_y,
            args.duration_ms,
            args.device_id,
            args.start_jitter_px,
            args.end_jitter_x_px,
            args.end_jitter_y_px,
            args.duration_jitter_ms,
        )
        print(f"Dragged {args.start_x},{args.start_y} -> {args.end_x},{args.end_y}")


if __name__ == "__main__":
    main()
