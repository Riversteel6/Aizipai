"""Schema definitions for the game logging stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    ROUND_START = "ROUND_START"
    ROUND_END = "ROUND_END"
    FRAME_CAPTURED = "FRAME_CAPTURED"
    FRAME_RECOGNIZED = "FRAME_RECOGNIZED"
    STATE_BUILT = "STATE_BUILT"
    STRUCTURE_ALLOCATED = "STRUCTURE_ALLOCATED"
    HAND_ANALYZED = "HAND_ANALYZED"
    LEGAL_ACTIONS_GENERATED = "LEGAL_ACTIONS_GENERATED"
    ACTION_EVALUATION_STARTED = "ACTION_EVALUATION_STARTED"
    ACTION_EVALUATED = "ACTION_EVALUATED"
    DECISION_SELECTED = "DECISION_SELECTED"
    ACTION_PLAN_CREATED = "ACTION_PLAN_CREATED"
    ACTION_PLAN_VALIDATED = "ACTION_PLAN_VALIDATED"
    TAP_EXECUTED = "TAP_EXECUTED"
    SCREEN_AFTER_ACTION = "SCREEN_AFTER_ACTION"
    SAFE_HALT = "SAFE_HALT"
    ERROR = "ERROR"
    WARNING = "WARNING"
    REPLAY_MARK = "REPLAY_MARK"
    MANUAL_NOTE = "MANUAL_NOTE"


class Severity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


SAFE_HALT_REASONS = {
    "recognition_uncertain",
    "no_clickable_cards",
    "selected_card_not_clickable",
    "blocked_discard_hard_protected",
    "only_hard_protected_clickable_but_unprotected_cards_exist_in_hand",
    "button_not_found",
    "option_not_clear",
    "sanity_check_failed",
    "opponent_priority_pending",
    "wait_auto_meld",
    "screen_not_stable",
    "action_plan_policy_mismatch",
    "action_plan_contract_error",
    "duplicate_card_allocation",
    "chi_ev_not_enough",
    "peng_ev_not_enough",
    "hu_xi_not_enough",
    "logger_write_failed",
    "action_guard_write_failed",
    "unexpected_runtime_error",
    "unknown_flow_state",
    "unknown_error",
    "hu_button_not_visible",
    "conflict_duplicate_card_allocation",
    "conflict_state_mutation",
    "conflict_policy_action_not_legal",
    "conflict_hard_protected_discard",
    "conflict_only_hard_protected_clickable_but_free_cards_exist",
    "conflict_uncertain_recognition",
    "conflict_unknown",
    "conflict_action_plan_reselected_card",
    "conflict_action_plan_targets_hard_protected_while_free_exists",
    "packet_capture_failed",
    "packet_capture_stopped",
    "external_health_check_failed",
}


@dataclass(frozen=True)
class LogEvent:
    timestamp: str
    session_id: str
    event_type: str
    severity: str
    data: dict[str, Any] = field(default_factory=dict)
    round_id: str | None = None
    frame_id: str | None = None
    decision_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "data": self.data,
        }
        if self.round_id is not None:
            payload["round_id"] = self.round_id
        if self.frame_id is not None:
            payload["frame_id"] = self.frame_id
        if self.decision_id is not None:
            payload["decision_id"] = self.decision_id
        return payload
