"""Structured, replay-oriented evidence for offline simulator decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


TRACE_SCHEMA_VERSION = "simulation-decision-trace-v3"
PUBLIC_STATE_ID_KEYS = (
    "seat",
    "hand",
    "all_melds",
    "discards",
    "remaining_counts",
    "stock_count",
    "hand_sizes",
    "pending_card",
    "pending_source_seat",
    "passed_chi",
    "passed_peng",
)


def canonical_public_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the non-redundant public information used to identify a root."""
    return {
        key: (
            payload.get(key) or []
            if key in {"passed_chi", "passed_peng"}
            else payload.get(key)
        )
        for key in PUBLIC_STATE_ID_KEYS
    }


def public_state_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_public_state(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TraceAction:
    key: str
    type: str
    label: str | None = None
    option_id: str | None = None
    consumed_from_hand: tuple[str, ...] = ()
    meld_groups: tuple[tuple[str, ...], ...] = ()
    priority: int = 0
    automatic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "type": self.type,
            "label": self.label,
            "option_id": self.option_id,
            "consumed_from_hand": list(self.consumed_from_hand),
            "meld_groups": [list(group) for group in self.meld_groups],
            "priority": self.priority,
            "automatic": self.automatic,
        }


@dataclass(frozen=True)
class SimulationDecisionTrace:
    sequence: int
    turn: int
    phase: str
    seat: int
    policy: str
    public_state: Mapping[str, Any]
    legal_actions: tuple[TraceAction, ...]
    selected_key: str
    reason: str
    state_before_hash: str
    state_after_hash: str
    policy_evidence: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "sequence": self.sequence,
            "turn": self.turn,
            "phase": self.phase,
            "seat": self.seat,
            "policy": self.policy,
            "public_state": dict(self.public_state),
            "public_state_id": public_state_identity(self.public_state),
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "selected_key": self.selected_key,
            "reason": self.reason,
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after_hash,
            "policy_evidence": dict(self.policy_evidence),
            "metadata": dict(self.metadata),
        }


def selected_action_is_legal(trace: Mapping[str, Any]) -> bool:
    selected = str(trace.get("selected_key") or "")
    return any(
        str(action.get("key") or "") == selected
        for action in trace.get("legal_actions") or ()
        if isinstance(action, Mapping)
    )


def trace_action_keys(actions: Sequence[TraceAction]) -> tuple[str, ...]:
    return tuple(action.key for action in actions)
