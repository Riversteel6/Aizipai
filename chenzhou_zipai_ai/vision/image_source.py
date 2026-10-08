"""Lossless image sources shared by desktop files and Android memory frames."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from threading import RLock

import cv2
import numpy as np


_FRAMES: dict[str, np.ndarray] = {}
_LOCK = RLock()


def _key(source: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(source)))


def register_rgba_frame(
    pixels: object,
    *,
    width: int,
    height: int,
    row_bytes: int,
    token_root: str | Path,
) -> Path:
    """Register Android Bitmap RGBA bytes as an exact OpenCV BGR frame."""
    width = int(width)
    height = int(height)
    row_bytes = int(row_bytes)
    if width <= 0 or height <= 0 or row_bytes < width * 4:
        raise ValueError("invalid_rgba_frame_shape")
    expected = row_bytes * height
    actual = len(pixels)  # type: ignore[arg-type]
    if actual != expected:
        raise ValueError(f"invalid_rgba_frame_length:{actual}!={expected}")
    rows = np.frombuffer(pixels, dtype=np.uint8).reshape(height, row_bytes)
    rgba = rows[:, : width * 4].reshape(height, width, 4)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    bgr.setflags(write=False)
    token = Path(token_root).resolve() / f".aizipai_memory_frame_{uuid.uuid4().hex}.png"
    with _LOCK:
        _FRAMES[_key(token)] = bgr
    return token


def unregister_frame(source: str | Path) -> None:
    with _LOCK:
        _FRAMES.pop(_key(source), None)


def is_registered_frame(source: str | Path) -> bool:
    with _LOCK:
        return _key(source) in _FRAMES


def read_bgr(source: str | Path) -> np.ndarray:
    key = _key(source)
    with _LOCK:
        registered = _FRAMES.get(key)
        if registered is not None:
            return registered
    data = np.fromfile(os.fspath(source), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read screenshot: {source}")
    return image
