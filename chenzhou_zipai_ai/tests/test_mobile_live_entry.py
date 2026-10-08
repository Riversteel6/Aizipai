from __future__ import annotations

from pathlib import Path

import pytest

from tools.live_assistant import _capture_runtime_screen


def test_mobile_entry_reuses_existing_screenshot_without_adb(tmp_path: Path, monkeypatch) -> None:
    screenshot = tmp_path / "frame.png"
    screenshot.write_bytes(b"frame")
    monkeypatch.setattr("tools.live_assistant.save_screen", lambda **kwargs: pytest.fail("ADB called"))

    actual = _capture_runtime_screen(
        "mobile",
        screenshot,
        reuse_runtime_screenshot=True,
    )

    assert actual == screenshot


def test_mobile_entry_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="runtime_screenshot_not_found"):
        _capture_runtime_screen(
            "mobile",
            tmp_path / "missing.png",
            reuse_runtime_screenshot=True,
        )
