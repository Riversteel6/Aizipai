"""Tests for ADB capture save helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import subprocess

from capture import adb_capture


def test_run_adb_wraps_subprocess_timeout_as_adb_error(monkeypatch):
    monkeypatch.setattr(adb_capture, "find_adb", lambda: "adb")
    monkeypatch.setattr(
        adb_capture.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("adb", 3)),
    )

    with pytest.raises(adb_capture.ADBError, match="timed out after 3s"):
        adb_capture.run_adb(["devices"], timeout=3)


def test_save_screen_writes_bgr_image_to_explicit_file(monkeypatch, tmp_path):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :] = (7, 8, 9)
    seen: dict[str, str | None] = {}

    def fake_capture_screen(device_id=None):
        seen["device_id"] = device_id
        return image

    monkeypatch.setattr(adb_capture, "capture_screen", fake_capture_screen)
    output = tmp_path / "shot.png"

    result = adb_capture.save_screen(output, device_id="device-1")

    assert result == output
    assert seen["device_id"] == "device-1"
    loaded = cv2.imread(str(output))
    assert loaded.shape == image.shape
    assert loaded[0, 0].tolist() == [7, 8, 9]


def test_save_screenshot_uses_save_screen_directory_semantics(monkeypatch, tmp_path):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    seen: dict[str, str | None] = {}

    def fake_capture_screen(device_id=None):
        seen["device_id"] = device_id
        return image

    monkeypatch.setattr(adb_capture, "capture_screen", fake_capture_screen)

    result = adb_capture.save_screenshot(tmp_path, device_id="device-2")

    assert result.parent == tmp_path
    assert result.suffix == ".png"
    assert seen["device_id"] == "device-2"
    assert result.exists()
