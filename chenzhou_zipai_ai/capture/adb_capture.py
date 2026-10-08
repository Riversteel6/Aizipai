"""ADB screenshot capture."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns, sleep

import cv2
import numpy as np


class ADBError(RuntimeError):
    """Raised when an ADB command fails."""


def find_adb() -> str:
    adb = shutil.which("adb")
    if adb:
        return adb

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        matches = sorted(packages.glob("Google.PlatformTools_*/*/adb.exe"))
        if matches:
            return str(matches[-1])

    raise ADBError("adb not found. Install Android SDK Platform-Tools and reopen the terminal.")


def run_adb(
    args: list[str],
    *,
    device_id: str | None = None,
    binary: bool = False,
    timeout: int = 30,
) -> str | bytes:
    command = [find_adb()]
    if device_id:
        command.extend(["-s", device_id])
    command.extend(args)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        serial = f" -s {device_id}" if device_id else ""
        raise ADBError(f"adb{serial} {' '.join(args)} timed out after {timeout}s") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        serial = f" -s {device_id}" if device_id else ""
        raise ADBError(f"adb{serial} {' '.join(args)} failed: {detail}")

    if binary:
        return result.stdout
    return result.stdout.decode("utf-8", errors="replace")


def capture_png(device_id: str | None = None) -> bytes:
    data = run_adb(["exec-out", "screencap", "-p"], device_id=device_id, binary=True)
    if not isinstance(data, bytes) or not data:
        raise ADBError("ADB screenshot returned empty data.")
    return data


def capture_screen(device_id: str | None = None) -> np.ndarray:
    """Capture current Android screen and return an OpenCV BGR image."""

    data = np.frombuffer(capture_png(device_id), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ADBError("Failed to decode ADB screenshot as PNG.")
    return image


def _default_screenshot_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"screenshot_{stamp}_{perf_counter_ns() % 1_000_000:06d}.png"
    return Path("data/screenshots") / filename


def save_screen(path: str | Path | None = None, device_id: str | None = None) -> Path:
    output_path = Path(path) if path is not None else _default_screenshot_path()
    if output_path.suffix.lower() != ".png":
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / _default_screenshot_path().name
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    image = capture_screen(device_id)
    temp_path = output_path.with_name(f"{output_path.stem}.{perf_counter_ns()}.tmp{output_path.suffix}")
    ok = cv2.imwrite(str(temp_path), image)
    if not ok:
        raise ADBError(f"Failed to write screenshot to {temp_path}")
    last_error: PermissionError | None = None
    for _ in range(3):
        try:
            temp_path.replace(output_path)
            break
        except PermissionError as exc:
            last_error = exc
            sleep(0.02)
    else:
        temp_path.unlink(missing_ok=True)
        raise ADBError(f"Failed to replace screenshot {output_path} after 3 attempts") from last_error
    return output_path


def save_screenshot(output_dir: str | Path = "data/screenshots", device_id: str | None = None) -> Path:
    """Backward-compatible directory-based screenshot save helper."""

    return save_screen(output_dir, device_id=device_id)
