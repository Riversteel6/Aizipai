"""Read-only device readiness check for live play."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture.adb_capture import run_adb, save_screen
from capture.device import get_screen_size, list_devices


def focused_window(device_id: str) -> str:
    try:
        output = run_adb(["shell", "dumpsys", "window"], device_id=device_id, timeout=10)
    except Exception:
        return ""
    for line in output.splitlines():
        if "mCurrentFocus" in line or "mFocusedApp" in line:
            return line.strip()
    return ""


def check_device(device_id: str | None = None, *, capture: bool = True) -> dict:
    devices = [device for device in list_devices() if device.status == "device"]
    selected = next((device for device in devices if device.serial == device_id), None) if device_id else (devices[0] if len(devices) == 1 else None)
    result = {
        "ok": selected is not None,
        "device_count": len(devices),
        "device_id": selected.serial if selected else device_id,
        "devices": [device.__dict__ for device in devices],
        "screen_size": None,
        "focused_window": "",
        "screenshot": None,
        "screenshot_size": None,
        "errors": [],
    }
    if selected is None:
        result["errors"].append("no_unique_authorized_device")
        return result
    try:
        result["screen_size"] = get_screen_size(selected.serial)
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"screen_size_failed:{type(exc).__name__}")
    result["focused_window"] = focused_window(selected.serial)
    if capture:
        try:
            screenshot = save_screen(device_id=selected.serial)
            result["screenshot"] = str(screenshot)
            image = cv2.imread(str(screenshot))
            if image is not None:
                result["screenshot_size"] = (int(image.shape[1]), int(image.shape[0]))
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"screenshot_failed:{type(exc).__name__}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check_device(args.device_id, capture=not args.no_capture)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"ok={result['ok']} device_id={result['device_id']} device_count={result['device_count']}")
    print(f"screen_size={result['screen_size']} screenshot_size={result['screenshot_size']}")
    print(f"focused_window={result['focused_window'] or 'unknown'}")
    print(f"screenshot={result['screenshot']}")
    if result["errors"]:
        print("errors=" + ",".join(result["errors"]))


if __name__ == "__main__":
    main()
