from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import cv2
import numpy as np
import pytest

from tools.inspect_state import VisionPipelineTimeout, _await_parallel_vision, inspect_screenshot
from tools.live_assistant import _capture_runtime_screen
from vision.image_source import (
    is_registered_frame,
    read_bgr,
    register_rgba_frame,
    unregister_frame,
)


def _rgba_bytes(image: np.ndarray, *, padding: int = 0) -> tuple[bytes, int]:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    row_bytes = image.shape[1] * 4 + padding
    rows = np.zeros((image.shape[0], row_bytes), dtype=np.uint8)
    rows[:, : image.shape[1] * 4] = rgba.reshape(image.shape[0], -1)
    return rows.tobytes(), row_bytes


def test_registered_android_frame_is_pixel_equal_to_lossless_png(tmp_path: Path) -> None:
    image = np.array(
        [
            [[0, 0, 255], [0, 255, 0], [255, 0, 0]],
            [[3, 7, 11], [127, 128, 129], [250, 251, 252]],
        ],
        dtype=np.uint8,
    )
    png = tmp_path / "reference.png"
    assert cv2.imwrite(str(png), image)
    pixels, row_bytes = _rgba_bytes(image, padding=8)
    token = register_rgba_frame(
        pixels,
        width=image.shape[1],
        height=image.shape[0],
        row_bytes=row_bytes,
        token_root=tmp_path,
    )
    try:
        assert is_registered_frame(token)
        assert _capture_runtime_screen("mobile", token, reuse_runtime_screenshot=True) == token
        registered = read_bgr(token)
        assert not registered.flags.writeable
        assert read_bgr(token) is registered
        assert np.array_equal(registered, read_bgr(png))
    finally:
        unregister_frame(token)
    assert not is_registered_frame(token)


def test_registered_frame_rejects_wrong_buffer_length(tmp_path: Path) -> None:
    with np.testing.assert_raises_regex(ValueError, "invalid_rgba_frame_length"):
        register_rgba_frame(
            b"\x00" * 15,
            width=2,
            height=2,
            row_bytes=8,
            token_root=tmp_path,
        )


def _stable_state(state: dict) -> dict:
    stable = deepcopy(state)
    stable.pop("screenshot", None)
    stable.get("metadata", {}).pop("screenshot_path", None)
    stable.get("hand_recognition", {}).pop("neural_elapsed_ms", None)
    stable.get("metadata", {}).get("hand_recognition", {}).pop("neural_elapsed_ms", None)
    return stable


@pytest.mark.parametrize(
    "screenshot",
    [
        Path("data/screenshots/screenshot_20260527_142258.png"),
        Path("data/screenshots/screenshot_20260527_160227.png"),
        Path("data/screenshots/screenshot_20260527_170608.png"),
    ],
)
def test_historical_semantics_match_between_png_and_memory_frame(
    screenshot: Path,
    tmp_path: Path,
) -> None:
    if not screenshot.is_file():
        pytest.skip(f"missing historical frame: {screenshot}")
    image = read_bgr(screenshot)
    pixels, row_bytes = _rgba_bytes(image, padding=16)
    token = register_rgba_frame(
        pixels,
        width=image.shape[1],
        height=image.shape[0],
        row_bytes=row_bytes,
        token_root=tmp_path,
    )
    try:
        from_png = inspect_screenshot(screenshot)
        from_memory = inspect_screenshot(token)
    finally:
        unregister_frame(token)
    assert _stable_state(from_memory) == _stable_state(from_png)


@pytest.mark.parametrize(
    "screenshot",
    [
        Path("data/screenshots/screenshot_20260527_142258.png"),
        Path("data/screenshots/screenshot_20260527_160227.png"),
        Path("data/screenshots/screenshot_20260527_170608.png"),
    ],
)
def test_parallel_android_vision_is_semantically_identical(
    screenshot: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not screenshot.is_file():
        pytest.skip(f"missing historical frame: {screenshot}")
    monkeypatch.delenv("AIZIPAI_PARALLEL_VISION", raising=False)
    sequential = inspect_screenshot(screenshot)
    monkeypatch.setenv("AIZIPAI_PARALLEL_VISION", "1")
    parallel = inspect_screenshot(screenshot)
    assert _stable_state(parallel) == _stable_state(sequential)


def test_inspect_reuses_buttons_from_the_exact_current_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.inspect_state as inspect_state

    screenshot = Path("data/screenshots/screenshot_20260527_142258.png")
    if not screenshot.is_file():
        pytest.skip(f"missing historical frame: {screenshot}")
    expected = inspect_screenshot(screenshot)
    image = inspect_state.read_image(screenshot)
    buttons = inspect_state.detect_buttons(image)

    def unexpected_button_scan(*_args, **_kwargs):
        raise AssertionError("same-frame buttons were detected twice")

    monkeypatch.setattr(inspect_state, "detect_buttons", unexpected_button_scan)
    state = inspect_screenshot(screenshot, precomputed_buttons=buttons)

    assert _stable_state(state) == _stable_state(expected)
    assert state["buttons"] == [button.name for button in buttons]


def test_inspect_reuses_confirmed_response_hand_without_running_hand_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.inspect_state as inspect_state

    screenshot = Path("data/screenshots/screenshot_20260527_142258.png")
    if not screenshot.is_file():
        pytest.skip(f"missing historical frame: {screenshot}")
    labels = ["二", "二", "七", "十"]

    def unexpected_hand_scan(*_args, **_kwargs):
        raise AssertionError("confirmed response hand must bypass hand OCR")

    monkeypatch.setattr(inspect_state, "recognize_hand", unexpected_hand_scan)
    state = inspect_screenshot(
        screenshot,
        precomputed_hand_labels=labels,
    )

    assert state["hand"] == labels
    assert state["hand_recognition"]["ledger_reuse"] is True
    assert all(item["clickable"] for item in state["hand_details"])


def test_parallel_vision_timeout_is_bounded_and_cancels_pending_work() -> None:
    release = Event()
    executor = ThreadPoolExecutor(max_workers=1)
    blocked = executor.submit(release.wait, 1.0)
    queued = executor.submit(lambda: "must-not-run")
    try:
        with pytest.raises(VisionPipelineTimeout, match="branch=hand"):
            _await_parallel_vision(
                (("hand", blocked), ("options", queued)),
                timeout_seconds=0.05,
            )
        assert queued.cancelled()
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)
