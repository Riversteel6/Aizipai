"""Load persisted round and session logs for replay tools."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplayBundle:
    session_id: str
    round_id: str
    session_meta: dict[str, Any]
    round_meta: dict[str, Any]
    events: list[dict[str, Any]]
    states: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_round_bundle(session_round_path: Path) -> ReplayBundle:
    session_dir = session_round_path.parents[1] if session_round_path.parent.name == "rounds" else session_round_path.parent
    session_meta = _read_json(session_dir / "session_meta.json")
    round_meta = _read_json(session_round_path / "round_meta.json")
    return ReplayBundle(
        session_id=session_meta.get("session_id", ""),
        round_id=round_meta.get("round_id", ""),
        session_meta=session_meta,
        round_meta=round_meta,
        events=load_jsonl(session_round_path / "events.jsonl"),
        states=load_jsonl(session_round_path / "states.jsonl"),
        decisions=load_jsonl(session_round_path / "decisions.jsonl"),
        actions=load_jsonl(session_round_path / "actions.jsonl"),
        errors=load_jsonl(session_round_path / "errors.jsonl"),
    )


def load_round_bundle_from_path(session_round_path: Path | str) -> ReplayBundle:
    path = Path(session_round_path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
            if "round_replay.json" not in names:
                raise FileNotFoundError(f"round_replay.json not found in {path}")
            payload = json.loads(archive.read("round_replay.json").decode("utf-8"))
            return ReplayBundle(
                session_id=payload.get("session_meta", {}).get("session_id", ""),
                round_id=payload.get("round_meta", {}).get("round_id", ""),
                session_meta=payload.get("session_meta", {}),
                round_meta=payload.get("round_meta", {}),
                events=payload.get("events", []),
                states=payload.get("states", []),
                decisions=payload.get("decisions", []),
                actions=payload.get("actions", []),
                errors=payload.get("errors", []),
            )
    return load_round_bundle(path)


def find_round_bundle(root: Path, session_id: str, round_id: str | None = None) -> Path:
    session_dir = root / "sessions" / session_id
    rounds_dir = session_dir / "rounds"
    if round_id is not None:
        return rounds_dir / round_id
    rounds = sorted([item for item in rounds_dir.iterdir() if item.is_dir() and item.name.startswith("round_")])
    if not rounds:
        raise FileNotFoundError(f"No round under {session_dir}")
    return rounds[-1]


def _latest_dir(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No matching directory found")
    return sorted(paths, key=lambda item: (item.stat().st_mtime, item.name))[-1]


def latest_round_bundle(root: Path) -> tuple[Path, Path, str, str]:
    """Return the latest session/round under a logs root or session directory."""

    root = Path(root)
    if (root / "round_meta.json").exists():
        session_dir = root.parents[1] if root.parent.name == "rounds" else root.parent
        return session_dir, root, session_dir.name, root.name
    if (root / "session_meta.json").exists() and (root / "rounds").exists():
        session_dir = root
    else:
        sessions_root = root / "sessions"
        sessions = [
            path
            for path in sessions_root.iterdir()
            if path.is_dir() and path.name.startswith("session_")
        ] if sessions_root.exists() else []
        if not sessions:
            raise FileNotFoundError(f"no session found under {root}")
        session_dir = _latest_dir(sessions)
    rounds_root = session_dir / "rounds"
    rounds = [
        path
        for path in rounds_root.iterdir()
        if path.is_dir() and path.name.startswith("round_")
    ] if rounds_root.exists() else []
    if not rounds:
        raise FileNotFoundError(f"no round found under {session_dir}")
    round_dir = _latest_dir(rounds)
    return session_dir, round_dir, session_dir.name, round_dir.name


def all_sessions(root: Path) -> list[Path]:
    sessions_root = root / "sessions"
    if not sessions_root.exists():
        return []
    return sorted(
        [item for item in sessions_root.iterdir() if item.is_dir() and item.name.startswith("session_")],
        key=lambda item: item.name,
    )
