"""Filesystem layout helpers for game session and round logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LogPaths:
    root: Path = Path("chenzhou_zipai_ai/logs")

    def sessions_root(self) -> Path:
        return self.root / "sessions"

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_root() / session_id

    def session_meta_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session_meta.json"

    def session_events_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session_events.jsonl"

    def round_dir(self, session_id: str, round_id: str) -> Path:
        return self.session_dir(session_id) / "rounds" / round_id

    def round_meta_path(self, session_id: str, round_id: str) -> Path:
        return self.round_dir(session_id, round_id) / "round_meta.json"

    def round_events_path(self, session_id: str, round_id: str) -> Path:
        return self.round_dir(session_id, round_id) / "events.jsonl"

    def round_states_path(self, session_id: str, round_id: str) -> Path:
        return self.round_dir(session_id, round_id) / "states.jsonl"

    def round_decisions_path(self, session_id: str, round_id: str) -> Path:
        return self.round_dir(session_id, round_id) / "decisions.jsonl"

    def round_actions_path(self, session_id: str, round_id: str) -> Path:
        return self.round_dir(session_id, round_id) / "actions.jsonl"

    def round_errors_path(self, session_id: str, round_id: str) -> Path:
        return self.round_dir(session_id, round_id) / "errors.jsonl"

    def round_screenshot_dir(self, session_id: str, round_id: str) -> Path:
        return self.round_dir(session_id, round_id) / "screenshots"

    def round_crops_root(self, session_id: str, round_id: str) -> Path:
        return self.round_dir(session_id, round_id) / "crops"

    def round_export_dir(self, session_id: str, round_id: str) -> Path:
        return self.round_dir(session_id, round_id) / "exports"
