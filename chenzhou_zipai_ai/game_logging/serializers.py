"""Lightweight JSON serializers and helpers for game logs."""

from __future__ import annotations

from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any


def to_plain_object(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_plain_object(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_plain_object(v) for key, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain_object(item) for item in value]
    return value


def to_jsonl_line(payload: dict[str, Any]) -> str:
    import json
    return json.dumps(to_plain_object(payload), ensure_ascii=False)
