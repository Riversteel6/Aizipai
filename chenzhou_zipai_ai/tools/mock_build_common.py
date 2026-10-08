"""Shared helpers for local mock-table build analysis tools."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DIST = WORKSPACE / "dist"
MOCK_BUILD_LOGS = WORKSPACE / "logs" / "mock_build"


def ensure_mock_build_logs() -> Path:
    MOCK_BUILD_LOGS.mkdir(parents=True, exist_ok=True)
    return MOCK_BUILD_LOGS


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": rel(path),
        "bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def iter_files(root: Path, suffixes: tuple[str, ...] | None = None) -> Iterable[Path]:
    if not root.exists():
        return
    lowered = tuple(s.lower() for s in suffixes or ())
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if lowered and path.suffix.lower() not in lowered:
                continue
            yield path
