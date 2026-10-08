"""Shadow ledger updated only after an action-specific confirmation."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from time import time


def record_confirmed_action(
    path: str | Path,
    *,
    mode: str,
    action: str,
    context: dict,
) -> dict:
    ledger_path = Path(path)
    previous = load_confirmed_ledger(ledger_path)
    observed = _counts_from_signature(context.get("hand_signature"))
    if not previous or previous.get("mode") != mode:
        previous = _new_ledger(mode, observed)

    before = Counter(previous.get("hand_counts") or {})
    if observed and before != observed:
        previous["observation_mismatch"] = {
            "ledger": _sorted_counts(before),
            "observed": _sorted_counts(observed),
        }

    action = str(action or "").lower()
    next_counts = Counter(observed or before)
    trusted = bool(next_counts)
    pending_reason = None
    if action == "discard":
        target = context.get("planned_hand_target") or {}
        label = str(target.get("label") or "")
        if not label or next_counts[label] <= 0:
            trusted = False
            pending_reason = "confirmed_discard_target_missing_from_ledger"
        else:
            next_counts[label] -= 1
            if next_counts[label] <= 0:
                next_counts.pop(label, None)
    elif action == "peng":
        label = str(context.get("opponent_pending_card") or context.get("pending_action_card") or "")
        if not label or next_counts[label] < 2:
            trusted = False
            pending_reason = "confirmed_peng_requires_reconciliation"
        else:
            next_counts[label] -= 2
            if next_counts[label] <= 0:
                next_counts.pop(label, None)
    elif action in {"chi_option", "compare_option", "chi"}:
        trusted = False
        pending_reason = "confirmed_combination_requires_reconciliation"
    elif action == "settlement_ready":
        next_counts.clear()
        trusted = False
        pending_reason = "awaiting_next_initial_hand"

    updated = {
        "schema_version": 1,
        "mode": mode,
        "state_version": int(previous.get("state_version", 0)) + 1,
        "trusted": trusted,
        "shadow_only": True,
        "hand_counts": _sorted_counts(next_counts),
        "last_confirmed_action": action,
        "pending_reconciliation_reason": pending_reason,
        "updated_at": time(),
    }
    if previous.get("observation_mismatch"):
        updated["observation_mismatch"] = previous["observation_mismatch"]
    _atomic_write_json(ledger_path, updated)
    return updated


def load_confirmed_ledger(path: str | Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _new_ledger(mode: str, counts: Counter) -> dict:
    return {
        "schema_version": 1,
        "mode": mode,
        "state_version": 0,
        "trusted": bool(counts),
        "shadow_only": True,
        "hand_counts": _sorted_counts(counts),
    }


def _counts_from_signature(raw: object) -> Counter:
    counts: Counter = Counter()
    if not isinstance(raw, list):
        return counts
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            count = int(item[1])
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[str(item[0])] += count
    return counts


def _sorted_counts(counts: Counter) -> dict[str, int]:
    return {label: int(count) for label, count in sorted(counts.items()) if count > 0}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["load_confirmed_ledger", "record_confirmed_action"]
