"""Decision logging for replay and future training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_decision(path: str | Path, result: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screenshot": result.get("screenshot"),
        "raw_hand": result.get("raw_hand", result.get("hand", [])),
        "hand": result.get("hand", []),
        "remaining_deck_count": result.get("remaining_deck_count"),
        "action_plan_target": result.get("action_plan_target"),
        "action_plan_ready": result.get("action_plan_ready"),
        "sanity_checks": result.get("sanity_checks", {}),
        "opponent_profile": result.get("opponent_profile", {}),
        "decision": result.get("decision", {}),
        "action_plan": result.get("action_plan", {}),
        "simulations": result.get("simulations", []),
        "memory": result.get("memory"),
    }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output
