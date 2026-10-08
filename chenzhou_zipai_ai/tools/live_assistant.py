"""One-step live assistant for ADB play sessions."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SRC = WORKSPACE / "src"
for path in (WORKSPACE, ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture.adb_capture import ADBError, save_screen
from control.adb_tap import drag, tap
from control.action_plan import action_plan_contract_error, build_action_plan
from control.action_executor import execute_discard_drag
from chenzhou_zipai_ai.game_logging import (
    EventType,
    GameLogger,
    LoggerConfig,
    SAFE_HALT_REASONS,
    Severity,
)
from engine.rules import ROOM_MODE_PRESETS, rules_for_room
from tools.compact_hand_left import run_compact
from tools.inspect_state import inspect_screenshot
from tools.protocol_log_to_state import ProtocolLogTail
from tools.recommend_action import build_compare_option_plan, recommend_from_screenshot
from vision.flow_detector import detect_flow_state_from_path
from vision.history_memory import VisionMemory, save_memory
from vision.image_source import is_registered_frame
from vision.button_detector import ButtonDetection
from vision.hand_recognizer import detect_hand_slots, read_image
from vision.option_recognizer import recognize_options
from vision.pending_card_recognizer import recognize_pending_cards


DEFAULT_DEVICE = "3B1F5WEA9BBUX9ZQ"
DEFAULT_CONFIG_PATH = ROOT / "config" / "screen_1080x2400.yaml"
STOP_ACTIONS = {
    "hu",
    "peng",
    "chi",
    "expand_chi_options",
    "pass",
    "discard",
    "chi_option",
    "compare_option",
    "safe_halt",
}
CHI_EXPAND_WAIT_SECONDS = 4.0
CHI_EXPAND_MAX_ATTEMPTS = 2
PENDING_CARD_RECHECK_MAX_FRAMES = 3
DEFAULT_RUNTIME_SCREENSHOT = Path("data/screenshots/live_current.png")
MAX_TRANSIENT_BACKOFF_SECONDS = 15.0
MIN_FLOW_CLICK_CONFIDENCE = 0.75
DEALER_OPENING_RECOGNITION_RETRIES = 0
COMPACT_INFLIGHT_TTL_SECONDS = 8.0
OPENING_RETRY_TTL_SECONDS = 10.0
OPENING_RECOVERY_TTL_SECONDS = 30.0
OPENING_COMPACT_MAX_ATTEMPTS = 4
OPENING_COMPACT_MAX_NO_PROGRESS = 2
OPENING_COMPACT_RIGHT_OF = 1500
OPENING_COMPACT_BATCH_SIZE = 2
OPENING_HAND_MIN_CONFIDENCE = 0.80
OPENING_UNTRUSTED_HYBRID_REASONS = {
    "confident_disagreement",
    "insufficient_independent_evidence",
    "template_missing",
}


def infer_seat_role(result: dict) -> dict:
    detected = result.get("seat_role") or (result.get("metadata") or {}).get("seat_role")
    checks = result.get("sanity_checks", {}) or {}
    controlled = checks.get("controlled_card_count")
    try:
        controlled_count = int(controlled)
    except (TypeError, ValueError):
        controlled_count = None
    if isinstance(detected, dict) and detected.get("role") in {"dealer", "player"}:
        role = str(detected["role"])
        reason = str(detected.get("reason") or "dealer_marker_detector")
        if (
            role == "dealer"
            and controlled_count is not None
            and controlled_count < 21
            and _has_dealer_opening_evidence(result)
        ):
            reason = f"dealer_marker_opening_shortage_{controlled_count}:{reason}"
        return {
            "role": role,
            "expected_total": 21 if role == "dealer" else 20,
            "confidence": detected.get("confidence", "vision"),
            "reason": reason,
        }
    discard_button_visible = result.get("discard_button") is not None
    if controlled_count == 21:
        return {
            "role": "dealer",
            "expected_total": 21,
            "confidence": "high",
            "reason": "controlled_card_count_is_21",
        }
    if controlled_count == 20 and discard_button_visible:
        return {
            "role": "unknown",
            "expected_total": None,
            "confidence": "low",
            "reason": "20_cards_with_discard_button_visible_dealer_or_player_ambiguous",
        }
    if controlled_count == 20:
        return {
            "role": "player",
            "expected_total": 20,
            "confidence": "medium",
            "reason": "controlled_card_count_is_20_without_own_discard_button",
        }
    return {
        "role": "unknown",
        "expected_total": None,
        "confidence": "low",
        "reason": f"unsupported_controlled_card_count:{controlled_count}",
    }


def _expected_total_for_current_phase(result: dict) -> int:
    """Return controlled-card total for the current UI phase, independent of seat."""
    button_names = {
        str(button.get("name") or button.get("type") or "").lower()
        for button in result.get("buttons") or []
        if isinstance(button, dict)
    }
    response_visible = bool(button_names & {"chi", "peng", "pao", "hu", "pass"})
    option_visible = bool(result.get("option_details") or result.get("options") or result.get("option_stage"))
    own_discard_turn = result.get("discard_button") is not None and not response_visible and not option_visible
    return 21 if own_discard_turn else 20


def _recover_current_phase_count_shortage(
    result: dict,
    *,
    screenshot: Path,
    expected_total: int,
    opponent_priority_pending: bool,
) -> dict:
    checks = result.get("sanity_checks") or {}
    current_count = _safe_int(checks.get("controlled_card_count"))
    if current_count is None or current_count >= expected_total:
        return result
    recovered = inspect_screenshot(
        screenshot,
        expected_total=expected_total,
        opponent_priority_pending=opponent_priority_pending,
    )
    recovered_count = _safe_int(
        (recovered.get("sanity_checks") or {}).get("controlled_card_count")
    )
    if recovered_count is None or recovered_count <= current_count:
        return result
    return recovered


def _has_dealer_opening_evidence(result: dict) -> bool:
    if result.get("discard_button") is None:
        return False
    button_names = {
        str(button.get("name") or button.get("type") or "").lower()
        for button in result.get("buttons") or []
        if isinstance(button, dict)
    }
    if button_names & {"chi", "peng", "pao", "hu", "pass"}:
        return False
    if result.get("option_details") or result.get("options") or result.get("option_stage"):
        return False

    checks = result.get("sanity_checks") or {}
    try:
        meld_cells = int(checks.get("my_meld_cell_count") or 0)
    except (TypeError, ValueError):
        meld_cells = 0
    if meld_cells > 0:
        return False

    discards = result.get("discards") or {}
    if isinstance(discards, dict):
        for values in discards.values():
            if isinstance(values, list) and values:
                return False
    return True


def _has_midgame_seat_evidence(result: dict) -> bool:
    checks = result.get("sanity_checks") or {}
    try:
        meld_cells = int(checks.get("my_meld_cell_count") or 0)
    except (TypeError, ValueError):
        meld_cells = 0
    if meld_cells > 0:
        return True

    try:
        remaining = int(result.get("remaining_deck_count") or 0)
    except (TypeError, ValueError):
        remaining = 0
    if 0 < remaining < 44:
        return True

    discards = result.get("discards") or {}
    if isinstance(discards, dict):
        for key in ("my_discards", "protocol_out_cards", "played_cards"):
            values = discards.get(key)
            if isinstance(values, list) and values:
                return True
    return False


def _seat_role_safe_halt(result: dict, seat_role: dict) -> dict:
    result = dict(result)
    decision = dict(result.get("decision") or {})
    decision.update(
        {
            "action": "safe_halt",
            "label": None,
            "reason": f"seat_role_uncertain:{seat_role.get('reason')}",
        }
    )
    result["decision"] = decision
    result["seat_role"] = seat_role
    result["executed"] = False
    result["action_plan"] = {
        "action": "safe_halt",
        "ready": False,
        "reason": f"seat_role_uncertain:{seat_role.get('reason')}",
        "clicks": [],
        "validation": {
            "passed": False,
            "reason_code": "seat_role_uncertain",
            "seat_role": seat_role,
            "checks": ["seat_role_inference"],
        },
    }
    return result


def _can_continue_response_window_without_seat(result: dict) -> bool:
    plan = result.get("action_plan") or {}
    decision = result.get("decision") or {}
    action = str(plan.get("action") or decision.get("action") or "").lower()
    if action == "discard":
        sanity = result.get("sanity_checks") or {}
        button_names = {
            str(button.get("name") or button.get("type") or "").lower()
            for button in result.get("buttons", []) or []
            if isinstance(button, dict)
        }
        if button_names & {"chi", "peng", "hu", "pao", "pass"}:
            return False
        if result.get("option_details") or result.get("options") or result.get("option_stage"):
            return False
        return (
            bool(plan.get("ready"))
            and plan.get("target_type") == "hand_card"
            and bool(sanity.get("ok", True))
            and _has_midgame_seat_evidence(result)
        )
    if action not in {"chi", "expand_chi_options", "peng", "hu", "pass", "chi_option", "compare_option"}:
        return False
    if result.get("option_details") or result.get("options") or result.get("option_stage"):
        return True
    button_names = {
        str(button.get("name") or button.get("type") or "").lower()
        for button in result.get("buttons", []) or []
        if isinstance(button, dict)
    }
    return bool(button_names & {"chi", "peng", "hu", "pass"})


def _make_live_logger(
    logger: GameLogger | None,
    *,
    device_id: str,
    execute_enabled: bool,
    logs_root: Path,
    logger_config: LoggerConfig | None,
) -> GameLogger | None:
    if logger is None:
        return None
    if logger_config is not None:
        logger.config = logger_config
    if not logger.session_active:
        logger.start_session(
            device_id=device_id,
            execute_enabled=execute_enabled,
            screen_size=(1080, 2400),
        )
    return logger


def _sync_round_by_flow(
    logger: GameLogger,
    flow_state: str,
    *,
    execute_enabled: bool,
) -> None:
    if flow_state == "play":
        if not logger.round_active:
            logger.start_round()
        return
    if flow_state in {"final_score"}:
        if logger.round_active:
            logger.end_round(result=flow_state)
        return


def _logger_failure_count(logger: GameLogger | None) -> int:
    if logger is None:
        return 0
    value = getattr(logger, "log_write_failures", getattr(logger, "_log_write_failures", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _runtime_health_issue(check: Callable[[], str | None] | None) -> str | None:
    if check is None:
        return None
    try:
        issue = check()
    except Exception as exc:
        return f"external_health_check_failed:{type(exc).__name__}:{exc}"
    return str(issue) if issue else None


def _capture_runtime_screen(
    device_id: str,
    runtime_screenshot_path: str | Path | None,
    *,
    reuse_runtime_screenshot: bool = False,
) -> Path:
    if reuse_runtime_screenshot:
        if runtime_screenshot_path is None:
            raise ValueError("reuse_runtime_screenshot_requires_path")
        screenshot = Path(runtime_screenshot_path)
        if not screenshot.is_file() and not is_registered_frame(screenshot):
            raise FileNotFoundError(f"runtime_screenshot_not_found:{screenshot}")
        return screenshot
    if runtime_screenshot_path is None:
        return save_screen(device_id=device_id)
    return save_screen(path=runtime_screenshot_path, device_id=device_id)


def _runtime_exception_result(exc: Exception, *, transient: bool, consecutive_failures: int) -> dict:
    reason_code = "transient_runtime_error" if transient else "unexpected_runtime_error"
    action = "wait_runtime_recovery" if transient else "safe_halt"
    return {
        "executed": False,
        "fatal": not transient,
        "runtime_error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "consecutive_failures": consecutive_failures,
            "transient": transient,
        },
        "action_plan": {
            "action": action,
            "ready": False,
            "reason": f"{reason_code}:{type(exc).__name__}:{exc}",
            "clicks": [],
            "validation": {
                "passed": False,
                "reason_code": reason_code,
                "checks": ["runtime_exception_boundary"],
            },
        },
    }


def _external_health_reason_code(issue: str) -> str:
    reason_code = issue.split(":", 1)[0]
    return reason_code if reason_code in SAFE_HALT_REASONS else "external_health_check_failed"


def _halt_for_external_runtime(
    result: dict,
    issue: str,
    *,
    logger: GameLogger | None,
    frame_id: str,
    screenshot_path: str,
) -> dict:
    reason_code = _external_health_reason_code(issue)
    result = dict(result)
    result["executed"] = False
    result["action_plan"] = {
        "action": "safe_halt",
        "ready": False,
        "reason": issue,
        "clicks": [],
        "validation": {
            "passed": False,
            "reason_code": reason_code,
            "checks": ["external_health_check"],
        },
    }
    if logger is not None and logger.round_active:
        logger.log_safe_halt(
            frame_id=frame_id,
            reason=reason_code,
            decision_id=result.get("decision_id"),
            raw_hand=result.get("raw_hand") or result.get("hand") or [],
            hard_protected=result.get("decision", {}).get("hard_protected", []),
            screenshot_path=screenshot_path,
        )
    return result


def _halt_for_logger_failure(
    result: dict,
    *,
    logger: GameLogger | None,
    frame_id: str,
    decision_id: str | None = None,
) -> dict:
    failure_count = _logger_failure_count(logger)
    result["executed"] = False
    result["critical_error"] = {
        "type": "LOGGER_WRITE_FAILED",
        "failure_count": failure_count,
        "message": "日志写入失败，停止执行点击，避免无日志运行。",
    }
    result["action_plan"] = {
        **(result.get("action_plan") or {}),
        "action": "safe_halt",
        "ready": False,
        "reason": "logger_write_failed",
        "clicks": [],
        "validation": {
            "passed": False,
            "reason_code": "logger_write_failed",
            "checks": ["logger_write_failed"],
        },
    }
    if logger is not None and getattr(logger, "round_active", False):
        logger.log_error(
            frame_id=frame_id,
            message="CRITICAL_LOGGER_WRITE_FAILED",
            context={
                "failure_count": failure_count,
                "action_plan": result.get("action_plan"),
                "decision": result.get("decision"),
            },
            severity=Severity.CRITICAL,
            decision_id=decision_id,
        )
        logger.log_safe_halt(
            frame_id=frame_id,
            reason="logger_write_failed",
            decision_id=decision_id,
            raw_hand=result.get("raw_hand") or result.get("hand") or [],
            hard_protected=result.get("decision", {}).get("hard_protected", []),
            screenshot_path=result.get("screenshot"),
        )
    return result


def _halt_for_guard_persistence(
    result: dict,
    *,
    phase: str,
    action_was_attempted: bool,
) -> dict:
    halted = dict(result)
    halted["fatal"] = True
    halted["executed"] = bool(result.get("executed")) if action_was_attempted else False
    halted["critical_error"] = {
        "type": "ACTION_GUARD_WRITE_FAILED",
        "phase": phase,
        "action_was_attempted": action_was_attempted,
    }
    halted["action_plan"] = {
        "action": "safe_halt",
        "ready": False,
        "reason": f"action_guard_write_failed:{phase}",
        "clicks": [],
        "validation": {
            "passed": False,
            "reason_code": "action_guard_write_failed",
            "checks": ["write_ahead_action_guard"],
        },
    }
    return halted


def _arm_action_guard(
    path: str | Path,
    guard_state: dict,
    signature: tuple,
    context_signature: dict,
) -> bool:
    return _save_guard(
        path,
        {
            **guard_state,
            "inflight_action": {
                "signature": _signature_to_json(signature),
                "context": context_signature,
                "started_at": time.time(),
            },
        },
    )


def prepare_external_action_guard(
    guard_file: str | Path,
    *,
    signature: tuple,
    context_signature: dict,
) -> str:
    """Atomically reserve one externally executed action through the shared guard."""
    guard_state = _load_guard(guard_file)
    if _should_block_repeated_execution(guard_state, signature, context_signature):
        return "duplicate_execution_same_context"
    if not _arm_action_guard(guard_file, guard_state, signature, context_signature):
        return "guard_write_failed"
    return "armed"


def abort_external_action_guard(
    guard_file: str | Path,
    *,
    signature: tuple,
    context_signature: dict,
) -> bool:
    """Release a reservation only when the external gesture was not dispatched."""
    guard_state = _load_guard(guard_file)
    inflight = guard_state.get("inflight_action")
    if not isinstance(inflight, dict):
        return True
    if (
        inflight.get("signature") != _signature_to_json(signature)
        or inflight.get("context") != context_signature
    ):
        return True
    cleaned = dict(guard_state)
    cleaned.pop("inflight_action", None)
    return _save_guard(guard_file, cleaned)


def commit_executed_action_guard(
    guard_file: str | Path,
    *,
    signature: tuple,
    context_signature: dict,
    action_plan: dict,
    executed_at: float | None = None,
) -> bool:
    """Commit a confirmed external or built-in action using one guard contract."""
    now = time.time() if executed_at is None else float(executed_at)
    guard_payload = {
        **_load_guard(guard_file),
        "last_executed_signature": _signature_to_json(signature),
        "last_context": context_signature,
        "last_executed_at": now,
    }
    guard_payload.pop("inflight_action", None)
    action_name = action_plan.get("action")
    if _is_chi_expand_plan(action_plan):
        old_pending = _pending_response_any(guard_payload)
        old_context = old_pending.get("context") if old_pending else None
        try:
            old_attempts = int(old_pending.get("attempts") or 0) if old_pending else 0
        except (TypeError, ValueError):
            old_attempts = 0
        attempts = old_attempts + 1 if old_context == context_signature else 1
        pending_response = {
            "action": "chi",
            "created_at": now,
            "expires_at": now + CHI_EXPAND_WAIT_SECONDS,
            "attempts": attempts,
            "context": context_signature,
            "last_signature": _signature_to_json(signature),
        }
        intended_cards = list(action_plan.get("target_option_cards") or [])
        if intended_cards:
            pending_response["intended_option_cards"] = intended_cards
        if action_plan.get("target_option_id"):
            pending_response["intended_option_id"] = action_plan["target_option_id"]
        guard_payload["pending_response"] = pending_response
    elif action_name == "chi":
        guard_payload.pop("pending_response", None)
    elif action_name == "chi_option":
        pending_response = {
            "action": "compare",
            "created_at": now,
            "expires_at": now + CHI_EXPAND_WAIT_SECONDS,
            "attempts": 1,
            "context": context_signature,
            "last_signature": _signature_to_json(signature),
        }
        selected_cards = list(action_plan.get("target_option_cards") or [])
        if selected_cards:
            pending_response["selected_chi_option_cards"] = selected_cards
        if action_plan.get("target_option_id"):
            pending_response["selected_chi_option_id"] = action_plan["target_option_id"]
        guard_payload["pending_response"] = pending_response
    elif action_name in {"compare_option", "pass", "discard", "hu", "peng"}:
        guard_payload.pop("pending_response", None)
    return _save_guard(guard_file, guard_payload)


def run_once(
    *,
    device_id: str = DEFAULT_DEVICE,
    expected_total: int = 20,
    memory_file: str | Path = "logs/alphadog_live_memory.json",
    guard_file: str | Path = "logs/live_action_guard.json",
    reset_memory: bool = False,
    opponent_priority_pending: bool = False,
    execute_settlement_ready: bool = False,
    execute_play_actions: bool = False,
    require_action: str | None = None,
    require_target: str | None = None,
    simulations: int = 0,
    logger: GameLogger | None = None,
    logs_root: str | Path = "chenzhou_zipai_ai/logs",
    seat_role: str = "manual",
    hidden_cards: list[str] | None = None,
    auto_compact_dealer_hand: bool = True,
    allow_opening_hand_compact: bool = True,
    capture_after_action: bool = False,
    protocol_payload: str | Path | dict | None = None,
    external_health_check: Callable[[], str | None] | None = None,
    runtime_screenshot_path: str | Path | None = None,
    reuse_runtime_screenshot: bool = False,
    wildcard_enabled: bool | None = None,
    players: int | None = None,
    room_mode: str | None = None,
    decision_time_budget_seconds: float | None = None,
    precomputed_buttons: list[ButtonDetection] | None = None,
    pending_surface_buttons: list[ButtonDetection] | None = None,
    precomputed_hand_labels: list[str] | None = None,
) -> dict:
    logger = _make_live_logger(
        logger,
        device_id=device_id,
        execute_enabled=execute_play_actions,
        logs_root=Path(logs_root),
        logger_config=None,
    )
    log_failures_before_turn = _logger_failure_count(logger)
    if reset_memory:
        if not save_memory(VisionMemory(), memory_file):
            raise OSError(f"vision_memory_write_failed:{memory_file}")

    screenshot = _capture_runtime_screen(
        device_id,
        runtime_screenshot_path,
        reuse_runtime_screenshot=reuse_runtime_screenshot,
    )
    flow = detect_flow_state_from_path(screenshot)
    if flow.state != "settlement_ready":
        _clear_flow_ready_click_guard(guard_file)

    if logger is not None:
        if not logger.session_active:
            logger.start_session(
                device_id=device_id,
                execute_enabled=execute_play_actions,
                screen_size=(1080, 2400),
            )
        _sync_round_by_flow(logger, flow.state, execute_enabled=execute_play_actions)
        frame_id = logger.next_frame_id()
        logger.log_event(
            EventType.FRAME_CAPTURED,
            {
                "flow_state": flow.state,
                "confidence": flow.confidence,
                "frame_capture_ok": True,
            },
            frame_id=frame_id,
            severity=Severity.DEBUG if flow.state != "play" else Severity.INFO,
        )
    else:
        frame_id = "frame_unknown"

    if flow.state == "settlement_ready":
        plan = _flow_click_plan(flow, "settlement_ready", "结算界面准备按钮")
        guard_state = _load_guard(guard_file)
        if execute_settlement_ready and plan["ready"] and _should_skip_recent_flow_ready_click(guard_state, flow):
            return {
                "screenshot": str(screenshot),
                "flow": flow.to_dict(),
                "action_plan": {
                    "action": "wait_next_round",
                    "ready": False,
                    "reason": "已点击准备，等待牌局开始",
                    "clicks": [],
                },
                "executed": False,
            }
        if execute_settlement_ready and plan["ready"]:
            if not save_memory(VisionMemory(), memory_file):
                raise OSError(f"round_memory_reset_failed:{memory_file}")
        if logger is not None and logger.round_active:
            logger.log_event(
                EventType.ACTION_PLAN_CREATED,
                {
                    "action_plan": plan,
                    "source": "flow_state",
                },
                frame_id=frame_id,
            )
        if execute_settlement_ready and _logger_failure_count(logger) > 0:
            return _halt_for_logger_failure(
                {
                    "screenshot": str(screenshot),
                    "flow": flow.to_dict(),
                    "action_plan": plan,
                },
                logger=logger,
                frame_id=frame_id,
            )
        executed = False
        if execute_settlement_ready and plan["ready"]:
            health_issue = _runtime_health_issue(external_health_check)
            if health_issue:
                return _halt_for_external_runtime(
                    {
                        "screenshot": str(screenshot),
                        "flow": flow.to_dict(),
                        "action_plan": plan,
                    },
                    health_issue,
                    logger=logger,
                    frame_id=frame_id,
                    screenshot_path=str(screenshot),
                )
            flow_context = _result_context_signature(
                {"flow": flow.to_dict(), "hand": [], "decision": {"action": "settlement_ready"}}
            )
            flow_signature = _plan_signature(plan)
            if flow_signature is None or not _arm_action_guard(
                guard_file,
                guard_state,
                flow_signature,
                flow_context,
            ):
                return _halt_for_guard_persistence(
                    {
                        "screenshot": str(screenshot),
                        "flow": flow.to_dict(),
                        "action_plan": plan,
                        "executed": False,
                    },
                    phase="before_settlement_ready",
                    action_was_attempted=False,
                )
            executed = execute_action_plan(plan, device_id=device_id)
        if executed:
            now = time.time()
            commit_guard = {
                **_load_guard(guard_file),
                "last_flow_ready_click": {
                    "action": "settlement_ready",
                    "center": list(flow.center) if flow.center is not None else None,
                    "created_at": now,
                },
            }
            commit_guard.pop("inflight_action", None)
            commit_guard.pop("pending_opening_recognition_retry", None)
            if not _save_guard(
                guard_file,
                commit_guard,
            ):
                return _halt_for_guard_persistence(
                    {
                        "screenshot": str(screenshot),
                        "flow": flow.to_dict(),
                        "action_plan": plan,
                        "executed": True,
                    },
                    phase="after_settlement_ready",
                    action_was_attempted=True,
                )
        return {
            "screenshot": str(screenshot),
            "flow": flow.to_dict(),
            "action_plan": plan,
            "executed": executed,
        }
    if flow.state == "final_score":
        return {
            "screenshot": str(screenshot),
            "flow": flow.to_dict(),
            "action_plan": {
                "action": "round_finished",
                "ready": False,
                "reason": "整桌结算/战绩页，等待用户重新开局",
                "clicks": [],
            },
            "executed": False,
        }
    if flow.state == "already_ready":
        return {
            "screenshot": str(screenshot),
            "flow": flow.to_dict(),
            "action_plan": {
                "action": "wait_next_round",
                "ready": False,
                "reason": "已准备，等待下一局开局",
                "clicks": [],
            },
            "executed": False,
        }

    guard_state_at_frame = _load_guard(guard_file)
    # A mobile caller may cheaply pre-scan only the highest-priority HU button.
    # That partial result is sufficient to prove HU is absent for an already
    # committed CHI transaction, but it must never be reused as the complete
    # button surface by the normal recognition path.
    surface_guard_buttons = (
        precomputed_buttons
        if precomputed_buttons is not None
        else pending_surface_buttons
    )
    pending_surface_result = (
        _pending_chi_option_plan_from_surface(
            screenshot,
            guard_state_at_frame,
            flow=flow,
            buttons=surface_guard_buttons,
        )
        if surface_guard_buttons is not None
        else None
    )
    if pending_surface_result is None and surface_guard_buttons is not None:
        pending_surface_result = _pending_compare_option_plan_from_surface(
            screenshot,
            guard_state_at_frame,
            flow=flow,
            buttons=surface_guard_buttons,
        )
    recognized_state = pending_surface_result
    preflight_inferred_seat = None
    preflight_expected_total = None
    if recognized_state is None and auto_compact_dealer_hand and (seat_role == "auto" or expected_total == 21):
        recognized_state = inspect_screenshot(
            screenshot,
            expected_total=None,
            infer_expected_total_from_phase=True,
            opponent_priority_pending=opponent_priority_pending,
            precomputed_buttons=precomputed_buttons,
            precomputed_hand_labels=precomputed_hand_labels,
        )
        preflight_inferred_seat = (
            infer_seat_role(recognized_state)
            if seat_role == "auto"
            else {
                "role": "dealer",
                "expected_total": 21,
                "confidence": 1.0,
                "reason": "manual_dealer_expected_total",
            }
        )
        preflight_expected_total = _expected_total_for_current_phase(recognized_state)
        if seat_role == "auto":
            preflight_inferred_seat = infer_seat_role(recognized_state)
        recognized_state["seat_role"] = preflight_inferred_seat
        recognized_state["flow"] = flow.to_dict()
        opening_guard_state = _load_guard(guard_file)
        unchanged_wait = _unchanged_executed_frame_wait(
            recognized_state,
            opening_guard_state,
        )
        if unchanged_wait is not None:
            return unchanged_wait
        if _should_auto_compact_dealer_hand(recognized_state):
            if not allow_opening_hand_compact:
                return _wait_for_stable_opening_layout(recognized_state)
            retry_result = _defer_opening_compact_for_recognition_retry(
                recognized_state,
                guard_file=guard_file,
                guard_state=opening_guard_state,
            )
            if retry_result is not None:
                return retry_result
            return _handle_opening_compact(
                recognized_state,
                screenshot=screenshot,
                device_id=device_id,
                guard_file=guard_file,
                guard_state=_load_guard(guard_file),
                execute_play_actions=execute_play_actions,
                logger=logger,
                frame_id=frame_id,
                external_health_check=external_health_check,
            )
        _clear_opening_recovery(guard_file, opening_guard_state)

    initial_expected_total = (
        int(preflight_expected_total)
        if preflight_expected_total is not None
        else None if seat_role == "auto" else expected_total
    )
    initial_update_memory = seat_role != "auto" or preflight_expected_total is not None
    pending_chi_result = pending_surface_result or (
        _reuse_pending_chi_option_plan(recognized_state, _load_guard(guard_file))
        if logger is None and isinstance(recognized_state, dict)
        else None
    )
    result = pending_chi_result or recommend_from_screenshot(
        screenshot,
        memory_file=memory_file,
        expected_total=initial_expected_total,
        reset_memory=False,
        simulations=simulations,
        opponent_priority_pending=opponent_priority_pending,
        logger=logger,
        frame_id=frame_id,
        hidden_cards=hidden_cards,
        fast=True,
        protocol_payload=protocol_payload,
        precomputed_state=recognized_state,
        update_memory=initial_update_memory,
        include_recognized_state=recognized_state is None,
        log_precomputed_state=recognized_state is not None,
        wildcard_enabled=wildcard_enabled,
        players=players,
        room_mode=room_mode,
        decision_time_budget_seconds=decision_time_budget_seconds,
        precomputed_hand_labels=precomputed_hand_labels,
    )
    if recognized_state is None:
        recognized_state = result.pop("_recognized_state", None)
    else:
        result.pop("_recognized_state", None)
    memory_update_pending = not initial_update_memory

    def rerun_policy(*, corrected_expected_total: int | None) -> dict:
        nonlocal memory_update_pending
        kwargs = {
            "memory_file": memory_file,
            "expected_total": corrected_expected_total,
            "reset_memory": False,
            "simulations": simulations,
            "opponent_priority_pending": opponent_priority_pending,
            "logger": logger,
            "frame_id": frame_id,
            "hidden_cards": hidden_cards,
            "fast": True,
            "protocol_payload": protocol_payload,
            "wildcard_enabled": wildcard_enabled,
            "players": players,
            "room_mode": room_mode,
            "decision_time_budget_seconds": decision_time_budget_seconds,
        }
        if isinstance(recognized_state, dict):
            kwargs.update(
                {
                    "precomputed_state": recognized_state,
                    "update_memory": memory_update_pending,
                    "hidden_cards": None,
                    "protocol_payload": None,
                }
            )
        rerun = recommend_from_screenshot(screenshot, **kwargs)
        memory_update_pending = False
        rerun.pop("_recognized_state", None)
        return rerun
    inferred_seat = preflight_inferred_seat
    if seat_role == "auto":
        inferred_seat = inferred_seat or infer_seat_role(result)
        if inferred_seat.get("expected_total") is None:
            result["flow"] = flow.to_dict()
            if not _can_continue_response_window_without_seat(result):
                return _seat_role_safe_halt(result, inferred_seat)
            result["seat_role"] = inferred_seat
        elif preflight_expected_total is None and not _can_reuse_auto_seat_result(result, inferred_seat):
            result = rerun_policy(corrected_expected_total=int(inferred_seat["expected_total"]))
        result["seat_role"] = inferred_seat
    if logger is not None and logger.round_active:
        decision_id = result.get("decision_id")
        if decision_id:
            logger.log_event(
                EventType.FRAME_RECOGNIZED,
                result.get("sanity_checks", {}),
                frame_id=frame_id,
                decision_id=str(decision_id),
            )
    effective_expected_total = _expected_total_for_current_phase(result)
    if _should_retry_dealer_after_first_discard(result, effective_expected_total):
        result = rerun_policy(corrected_expected_total=20)
        if seat_role == "auto":
            result["seat_role"] = infer_seat_role(result)
    if _should_relax_visible_option_count(result):
        result = rerun_policy(corrected_expected_total=None)
        result = _mark_midgame_count_relaxed(result, inferred_seat if seat_role == "auto" else result.get("seat_role"))
    result["flow"] = flow.to_dict()
    result["executed"] = False
    guard_state = _load_guard(guard_file)
    result, bounded_guard, bounded_guard_changed = _bounded_untrusted_pending_card_recheck(
        result,
        guard_state,
    )
    if bounded_guard_changed:
        if not _save_guard(guard_file, bounded_guard):
            return _halt_for_guard_persistence(
                result,
                phase="pending_card_recheck",
                action_was_attempted=False,
            )
        guard_state = bounded_guard
    candidate_recovery, recovered_guard = _stabilize_pending_chi_candidate(
        result,
        guard_state,
    )
    if candidate_recovery is not None:
        if not _save_guard(guard_file, recovered_guard):
            return _halt_for_guard_persistence(
                result,
                phase="pending_chi_candidate_recovery",
                action_was_attempted=False,
            )
        guard_state = recovered_guard
        if candidate_recovery == "recheck":
            result["action_plan"] = {
                "action": "wait_chi_candidate_recheck",
                "ready": False,
                "reason": "吃牌候选与预判不一致，正在用下一帧复核当前真实候选",
                "clicks": [],
            }
            return result
        result["pending_chi_candidate_recovered"] = True
    if _should_confirm_discard_warning(result, guard_state):
        result["action_plan"] = _confirm_discard_warning_plan(result)
        if execute_play_actions:
            health_issue = _runtime_health_issue(external_health_check)
            if health_issue:
                return _halt_for_external_runtime(
                    result,
                    health_issue,
                    logger=logger,
                    frame_id=frame_id,
                    screenshot_path=str(screenshot),
                )
            warning_signature = _plan_signature(result["action_plan"])
            warning_context = _result_context_signature(result)
            if warning_signature is None or _should_block_repeated_execution(
                guard_state,
                warning_signature,
                warning_context,
            ):
                result["action_plan"] = {
                    "action": "wait_duplicate_execution",
                    "ready": False,
                    "reason": "弃牌确认动作仍在处理中，禁止重复点击",
                    "clicks": [],
                }
                return result
            if not _arm_action_guard(guard_file, guard_state, warning_signature, warning_context):
                return _halt_for_guard_persistence(
                    result,
                    phase="before_discard_warning_confirm",
                    action_was_attempted=False,
                )
            result["executed"] = execute_action_plan(result["action_plan"], device_id=device_id)
            if result["executed"]:
                warning_guard = {
                    **_load_guard(guard_file),
                    "last_executed_signature": _signature_to_json(warning_signature),
                    "last_context": warning_context,
                    "last_executed_at": time.time(),
                    "last_confirmed_discard_warning_at": time.time(),
                    "last_confirmed_discard_warning_for": guard_state.get("last_executed_signature"),
                }
                warning_guard.pop("inflight_action", None)
                if not _save_guard(guard_file, warning_guard):
                    return _halt_for_guard_persistence(
                        result,
                        phase="after_discard_warning_confirm",
                        action_was_attempted=True,
                    )
        return result
    pending_response = _pending_response_lock(guard_state)
    if pending_response and _should_hold_for_response_options(result, pending_response):
        result["action_plan"] = {
            "action": "wait_response_options",
            "ready": False,
            "reason": f"已点击{pending_response.get('action')}，等待候选展开，禁止立刻过",
            "clicks": [],
        }
        return result
    if _should_hold_after_chi_click(result, guard_state):
        result["action_plan"] = {
            "action": "wait_response_options",
            "ready": False,
            "reason": "上一步已点吃且吃/过仍同屏，禁止立刻过",
            "clicks": [],
        }
        return result
    if _should_halt_for_failed_chi_expand(result, guard_state):
        pending = _pending_response_any(guard_state) or {}
        attempts = int(pending.get("attempts") or CHI_EXPAND_MAX_ATTEMPTS)
        result["action_plan"] = {
            "action": "safe_halt",
            "ready": False,
            "reason": f"已点吃 {attempts} 次但候选未展开，暂停等待人工复核",
            "clicks": [],
            "validation": {
                "passed": False,
                "reason_code": "chi_options_expand_failed",
                "checks": ["pending_response_options_missing"],
            },
        }
        if logger is not None and logger.round_active:
            logger.log_safe_halt(
                frame_id=frame_id,
                reason="chi_options_expand_failed",
                decision_id=result.get("decision_id"),
                raw_hand=result.get("raw_hand") or result.get("hand") or [],
                hard_protected=result.get("decision", {}).get("hard_protected", []),
                screenshot_path=str(screenshot),
            )
        return result
    _clear_opening_recovery(guard_file, guard_state)
    if _should_relax_midgame_count(result):
        result = rerun_policy(corrected_expected_total=None)
        result = _mark_midgame_count_relaxed(result, inferred_seat if seat_role == "auto" else result.get("seat_role"))
    guard_result = result.get("conflict_guard_result") or {}
    if guard_result and not guard_result.get("passed", True):
        guard_reason = str(guard_result.get("reason") or "conflict_unknown")
        result["action_plan"] = {
            "action": "safe_halt",
            "ready": False,
            "reason": guard_reason,
            "clicks": [],
            "validation": {"passed": False, "guard_result": guard_result},
        }
        if logger is not None and logger.round_active:
            logger.log_safe_halt(
                frame_id=frame_id,
                reason=guard_reason,
                decision_id=result.get("decision_id"),
                raw_hand=result.get("raw_hand", []),
                hard_protected=result.get("decision", {}).get("hard_protected", []),
                screenshot_path=str(screenshot),
            )
        return result
    result["action_plan"] = _normalized_action_plan_for_execution(result.get("action_plan", {}))
    contract_error = action_plan_contract_error(result.get("action_plan", {}))
    if execute_play_actions and result.get("action_plan", {}).get("ready") and contract_error is not None:
        result["fatal"] = True
        result["action_plan"] = {
            "action": "safe_halt",
            "ready": False,
            "reason": f"action_plan_contract_error:{contract_error}",
            "clicks": [],
            "validation": {
                "passed": False,
                "reason_code": "action_plan_contract_error",
                "checks": [contract_error],
            },
        }
        if logger is not None and logger.round_active:
            logger.log_safe_halt(
                frame_id=frame_id,
                reason="action_plan_contract_error",
                decision_id=result.get("decision_id"),
                raw_hand=result.get("raw_hand") or result.get("hand") or [],
                hard_protected=result.get("decision", {}).get("hard_protected", []),
                screenshot_path=str(screenshot),
            )
        return result
    if execute_play_actions and not _matches_required_plan(result.get("action_plan", {}), require_action, require_target):
        result["action_plan"] = {
            "action": "plan_changed",
            "ready": False,
            "reason": f"当前动作与执行前确认不一致，暂停：require_action={require_action} require_target={require_target}",
            "clicks": [],
        }
        if logger is not None and logger.round_active:
            logger.log_safe_halt(
                frame_id=frame_id,
                reason="action_plan_policy_mismatch",
                decision_id=result.get("decision_id"),
                raw_hand=result.get("raw_hand", []),
                hard_protected=result.get("decision", {}).get("hard_protected", []),
                screenshot_path=str(screenshot),
            )
        return result
    if execute_play_actions and _is_play_action_plan(result.get("action_plan", {})):
        if _logger_failure_count(logger) > log_failures_before_turn:
            return _halt_for_logger_failure(
                result,
                logger=logger,
                frame_id=frame_id,
                decision_id=result.get("decision_id"),
            )
        health_issue = _runtime_health_issue(external_health_check)
        if health_issue:
            return _halt_for_external_runtime(
                result,
                health_issue,
                logger=logger,
                frame_id=frame_id,
                screenshot_path=str(screenshot),
            )
        before_screenshot = str(screenshot)
        signature = _plan_signature(result["action_plan"])
        context_signature = _result_context_signature(result)
        if _should_block_repeated_execution(guard_state, signature, context_signature):
            result["executed"] = False
            result["action_plan"] = {
                **result.get("action_plan", {}),
                "action": "wait_duplicate_execution",
                "ready": False,
                "reason": "上一次执行后画面未变化，禁止重复点击同一目标，等待下一帧",
                "clicks": [],
                "validation": {
                    "passed": False,
                    "reason_code": "duplicate_execution_same_context",
                    "checks": ["duplicate_execution_same_context"],
                },
            }
            if logger is not None and logger.round_active:
                logger.log_safe_halt(
                    frame_id=frame_id,
                    reason="duplicate_execution_same_context",
                    decision_id=result.get("decision_id"),
                    raw_hand=result.get("raw_hand") or result.get("hand") or [],
                    hard_protected=result.get("decision", {}).get("hard_protected", []),
                    screenshot_path=str(screenshot),
                )
            return result
        if signature is None or not _arm_action_guard(guard_file, guard_state, signature, context_signature):
            return _halt_for_guard_persistence(
                result,
                phase="before_execution",
                action_was_attempted=False,
            )
        try:
            result["executed"] = execute_action_plan(result["action_plan"], device_id=device_id)
        except Exception as exc:
            result["executed"] = False
            result["fatal"] = True
            result["action_plan"] = {
                **result.get("action_plan", {}),
                "ready": False,
                "action": "safe_halt",
                "reason": "execution_exception",
                "clicks": [],
                "validation": {
                    "passed": False,
                    "reason_code": "unknown_error",
                    "checks": ["execution_exception"],
                },
            }
            if logger is not None and logger.round_active:
                logger.log_error(
                    frame_id=frame_id,
                    message="CRITICAL_EXECUTION_ERROR",
                    context={
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "stacktrace": traceback.format_exc(),
                        "current_frame": frame_id,
                        "current_decision": result.get("decision"),
                        "action_plan": result.get("action_plan"),
                        "before_screenshot": before_screenshot,
                    },
                    severity=Severity.CRITICAL,
                    decision_id=result.get("decision_id"),
                )
            return result
        capture_this_action = _should_capture_after_action(
            result.get("action_plan", {}),
            capture_after_action=capture_after_action,
        )
        after_screenshot = (
            str(save_screen(device_id=device_id))
            if result["executed"] and capture_this_action
            else None
        )
        if result["executed"] and capture_after_action and not capture_this_action:
            result["after_action_capture"] = "deferred_to_next_frame"
        if logger is not None and logger.round_active:
            logger.log_tap(
                frame_id=frame_id,
                decision_id=result.get("decision_id"),
                plan=result["action_plan"],
                execute_enabled=True,
                dry_run=False,
                tap_executed=result["executed"],
                tap_x=result["action_plan"].get("clicks", [{}])[0].get("x") if result["action_plan"].get("clicks") else None,
                tap_y=result["action_plan"].get("clicks", [{}])[0].get("y") if result["action_plan"].get("clicks") else None,
                target_label=result["action_plan"].get("target_label"),
                target_card_id=result["action_plan"].get("target_card_id"),
                adb_result="ok" if result["executed"] else "not_executed",
                before_screenshot=before_screenshot,
                after_screenshot=after_screenshot,
            )
        if result["executed"] and signature is not None:
            if not commit_executed_action_guard(
                guard_file,
                signature=signature,
                context_signature=context_signature,
                action_plan=result.get("action_plan", {}),
            ):
                return _halt_for_guard_persistence(
                    result,
                    phase="after_execution",
                    action_was_attempted=True,
                )
    return result


def _should_retry_dealer_after_first_discard(result: dict, expected_total: int) -> bool:
    if expected_total != 21:
        return False
    checks = result.get("sanity_checks", {})
    if checks.get("ok"):
        return False
    if checks.get("controlled_card_count") != 20:
        return False
    if result.get("discard_button") is not None:
        return False
    # Dealer starts with 21. Once the first discard has left the hand, normal
    # controlled-card accounting drops to 20 for the rest of the round. Do not
    # downshift while the current screen still looks like our discard turn.
    frames_seen = int((result.get("memory") or {}).get("frames_seen", 0))
    if result.get("decision", {}).get("action") == "discard":
        return False
    return True


def _wait_for_stable_opening_layout(result: dict) -> dict:
    waiting = dict(result)
    sanity = result.get("sanity_checks") or {}
    controlled_count = _safe_int(sanity.get("controlled_card_count"))
    waiting["executed"] = False
    waiting["action_plan"] = {
        "action": "wait_initial_layout",
        "ready": False,
        "reason": (
            f"开局当前确认到 {controlled_count} 张，等待发牌或自动落地动画稳定；"
            "单击出牌模式不自动拖动手牌"
        ),
        "clicks": [],
        "validation": {
            "passed": False,
            "reason_code": "opening_layout_not_stable",
            "checks": [
                "dealer_marker",
                "discard_button_visible",
                "hand_plus_meld_count",
                "no_automatic_hand_compact",
            ],
        },
    }
    return waiting


def _should_relax_midgame_count(result: dict) -> bool:
    checks = result.get("sanity_checks", {}) or {}
    if checks.get("ok", True):
        return False
    expected_total = checks.get("expected_total")
    controlled_count = checks.get("controlled_card_count")
    if expected_total is None or controlled_count is None:
        return False
    try:
        expected = int(expected_total)
        controlled = int(controlled_count)
    except (TypeError, ValueError):
        return False
    if controlled >= expected or controlled < 10:
        return False
    if result.get("discard_button") is None:
        return False
    try:
        remaining_count = int(result.get("remaining_deck_count"))
    except (TypeError, ValueError):
        return False
    if remaining_count >= 44:
        return False
    action = (result.get("action_plan") or {}).get("action") or (result.get("decision") or {}).get("action")
    return action == "discard"


def _should_relax_visible_option_count(result: dict) -> bool:
    if not result.get("option_details"):
        return False
    checks = result.get("sanity_checks", {}) or {}
    return not bool(checks.get("ok", True))


def _mark_midgame_count_relaxed(result: dict, seat_role_info: dict | None) -> dict:
    result = dict(result)
    metadata = dict(result.get("metadata") or {})
    metadata["count_validation_mode"] = "midgame_relaxed"
    metadata["count_validation_reason"] = "remaining_deck_below_opening_count"
    result["metadata"] = metadata
    checks = dict(result.get("sanity_checks") or {})
    checks["mode"] = "midgame_relaxed"
    checks["warnings"] = list(checks.get("warnings") or []) + [
        "midgame_count_validation_relaxed: visible hand count is below opening total"
    ]
    result["sanity_checks"] = checks
    if seat_role_info:
        result["seat_role"] = seat_role_info
    return result


def _flow_click_plan(flow, action: str, reason: str) -> dict:
    center = flow.center
    if center is None:
        return {
            "action": action,
            "ready": False,
            "reason": f"未定位到{reason}",
            "clicks": [],
            "validation": {"passed": False, "reason_code": "target_not_found", "checks": []},
        }
    if float(flow.confidence) < MIN_FLOW_CLICK_CONFIDENCE:
        return {
            "action": action,
            "ready": False,
            "reason": f"{reason}识别置信度不足，等待下一帧",
            "clicks": [],
            "validation": {
                "passed": False,
                "reason_code": "recognition_uncertain",
                "checks": ["flow_click_confidence"],
            },
        }
    x, y = center
    return {
        "action": action,
        "ready": True,
        "reason": reason,
        "clicks": [{"target": action, "x": x, "y": y, "delay_ms": 80}],
        "validation": {"passed": True, "checks": ["target_clickable"]},
    }


def _should_skip_recent_flow_ready_click(guard_state: dict, flow) -> bool:
    inflight = guard_state.get("inflight_action")
    if isinstance(inflight, dict):
        signature = inflight.get("signature")
        if isinstance(signature, list) and signature and signature[0] == "settlement_ready":
            return True
    recent = guard_state.get("last_flow_ready_click")
    if not isinstance(recent, dict):
        return False
    if recent.get("action") != "settlement_ready":
        return False
    if flow.center is None:
        return True
    try:
        old_x, old_y = recent.get("center") or [None, None]
        old_x = int(old_x)
        old_y = int(old_y)
    except (TypeError, ValueError):
        return True
    x, y = flow.center
    return abs(int(x) - old_x) <= 80 and abs(int(y) - old_y) <= 80


def _clear_flow_ready_click_guard(path: str | Path) -> None:
    guard = _load_guard(path)
    if "last_flow_ready_click" not in guard:
        return
    guard.pop("last_flow_ready_click", None)
    _save_guard(path, guard)


def execute_action_plan(action_plan: dict, *, device_id: str = DEFAULT_DEVICE) -> bool:
    if action_plan_contract_error(action_plan) is not None:
        return False
    clicks = action_plan["clicks"]
    if action_plan.get("execution_mode") == "drag_sequence":
        for click in clicks:
            drag(
                int(click["from_x"]),
                int(click["from_y"]),
                int(click["x"]),
                int(click["y"]),
                duration_ms=int(click.get("duration_ms", 520)),
                device_id=device_id,
            )
        return True
    if action_plan.get("action") == "discard":
        if action_plan.get("execution_mode") == "discard_drag":
            click = clicks[0]
            execute_discard_drag(
                int(click["x"]),
                int(click["y"]),
                device_id=device_id,
                config_path=DEFAULT_CONFIG_PATH,
            )
            return True
        for click in clicks:
            tap(int(click["x"]), int(click["y"]), device_id=device_id)
            time.sleep(float(click.get("delay_ms", 80)) / 1000.0)
        return True
    for click in clicks:
        tap(int(click["x"]), int(click["y"]), device_id=device_id)
        time.sleep(float(click.get("delay_ms", 80)) / 1000.0)
    return True


def _confirm_discard_warning_plan(result: dict) -> dict:
    center = _discard_warning_confirm_center(result)
    if center is None:
        return {
            "action": "confirm_discard_warning",
            "ready": False,
            "reason": "未确认弃牌警告弹窗，禁止固定坐标盲点",
            "clicks": [],
            "validation": {
                "passed": False,
                "reason_code": "discard_warning_not_confirmed",
                "checks": ["discard_warning_anchor"],
            },
        }
    x, y = center
    return {
        "action": "confirm_discard_warning",
        "ready": True,
        "reason": "上一手弃牌触发已有玩家偎/不能进牌确认，按原策略确认打出",
        "execution_mode": "tap_sequence",
        "target_type": "button",
        "target_label": "confirm",
        "clicks": [
            {
                "target": "button:confirm_discard_warning",
                "x": x,
                "y": y,
                "delay_ms": 80,
            }
        ],
        "validation": {
            "passed": True,
            "checks": ["previous_discard_triggered_warning"],
        },
    }


def _discard_warning_confirm_center(result: dict) -> tuple[int, int] | None:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    warning = metadata.get("discard_warning") if isinstance(metadata, dict) else None
    if not isinstance(warning, dict) or warning.get("detected") is not True:
        return None
    try:
        confidence = float(warning.get("confidence") or 0.0)
        x, y = warning.get("confirm_center") or [None, None]
        x, y = int(x), int(y)
    except (TypeError, ValueError):
        return None
    if confidence < 0.9 or not (0 <= x < 2400 and 0 <= y < 1080):
        return None
    return x, y


def _should_confirm_discard_warning(result: dict, guard_state: dict) -> bool:
    last_signature = guard_state.get("last_executed_signature")
    if not isinstance(last_signature, list) or not last_signature or last_signature[0] != "discard":
        return False
    try:
        last_executed_at = float(guard_state.get("last_executed_at") or 0)
    except (TypeError, ValueError):
        return False
    if time.time() - last_executed_at > 60.0:
        return False
    if guard_state.get("last_confirmed_discard_warning_for") == last_signature:
        return False

    plan = result.get("action_plan") or {}
    if plan.get("ready"):
        return False
    if result.get("flow", {}).get("state") != "play":
        return False
    if result.get("buttons") or result.get("discard_button") or result.get("option_details"):
        return False
    if result.get("hand"):
        return False

    reason = str(plan.get("reason") or "")
    return "未找到 pass 按钮" in reason and _discard_warning_confirm_center(result) is not None


def _is_play_action_plan(action_plan: dict) -> bool:
    action_plan = _normalized_action_plan_for_execution(action_plan)
    return action_plan.get("action") in STOP_ACTIONS and bool(action_plan.get("ready"))


def _normalized_action_plan_for_execution(action_plan: dict) -> dict:
    return action_plan


def _matches_required_plan(action_plan: dict, require_action: str | None, require_target: str | None) -> bool:
    action_plan = _normalized_action_plan_for_execution(action_plan)
    if require_action is not None and action_plan.get("action") != require_action:
        return False
    if require_target is not None:
        clicks = action_plan.get("clicks", [])
        if not clicks or clicks[0].get("target") != require_target:
            return False
    return True


def should_stop_for_user(result: dict, *, execute_play_actions: bool = False) -> bool:
    """Return True when a human should confirm the next play action."""

    action_plan = result.get("action_plan", {})
    action = action_plan.get("action")
    plan_ready = bool(action_plan.get("ready"))
    if action in STOP_ACTIONS:
        return not (execute_play_actions and plan_ready)
    if result.get("flow", {}).get("state") == "final_score":
        return True
    return False


def _continue_nonfatal_halts(
    *,
    loop: bool,
    execute_play_actions: bool,
    explicitly_continue: bool,
    explicitly_stop: bool,
) -> bool:
    if explicitly_stop:
        return False
    return explicitly_continue or (loop and execute_play_actions)


def run_loop(
    *,
    device_id: str = DEFAULT_DEVICE,
    expected_total: int = 20,
    memory_file: str | Path = "logs/alphadog_live_memory.json",
    guard_file: str | Path = "logs/live_action_guard.json",
    reset_memory: bool = False,
    opponent_priority_pending: bool = False,
    execute_settlement_ready: bool = False,
    execute_play_actions: bool = False,
    require_action: str | None = None,
    require_target: str | None = None,
    simulations: int = 0,
    interval_seconds: float = 1.5,
    max_steps: int | None = None,
    seat_role: str = "manual",
    hidden_cards: list[str] | None = None,
    auto_compact_dealer_hand: bool = True,
    capture_after_action: bool = False,
    protocol_payload: str | Path | dict | None = None,
    logger: GameLogger | None = None,
    continue_on_halt: bool = False,
    external_health_check: Callable[[], str | None] | None = None,
    runtime_screenshot_path: str | Path = DEFAULT_RUNTIME_SCREENSHOT,
    wildcard_enabled: bool | None = None,
    players: int | None = None,
    room_mode: str | None = None,
) -> dict:
    last_result: dict | None = None
    last_executed_signature: tuple | None = None
    last_executed_context: tuple | dict | None = None
    unchanged_after_execute_count = 0
    consecutive_transient_failures = 0
    step = 0
    protocol_feed = _prepare_protocol_payload_for_loop(protocol_payload)
    while max_steps is None or step < max_steps:
        external_halt_reason = _runtime_health_issue(external_health_check)
        if external_halt_reason:
            last_result = {
                "executed": False,
                "action_plan": {
                    "action": "safe_halt",
                    "ready": False,
                    "reason": external_halt_reason,
                    "clicks": [],
                    "validation": {
                        "passed": False,
                        "reason_code": "external_runtime_unhealthy",
                        "checks": ["external_health_check"],
                    },
                },
            }
            break
        try:
            last_result = run_once(
                device_id=device_id,
                expected_total=expected_total,
                memory_file=memory_file,
                guard_file=guard_file,
                reset_memory=reset_memory and step == 0,
                opponent_priority_pending=opponent_priority_pending,
                execute_settlement_ready=execute_settlement_ready,
                execute_play_actions=execute_play_actions,
                require_action=require_action,
                require_target=require_target,
                simulations=simulations,
                seat_role=seat_role,
                hidden_cards=hidden_cards,
                auto_compact_dealer_hand=auto_compact_dealer_hand,
                capture_after_action=capture_after_action,
                protocol_payload=protocol_feed,
                logger=logger,
                external_health_check=external_health_check,
                runtime_screenshot_path=runtime_screenshot_path,
                wildcard_enabled=wildcard_enabled,
                players=players,
                room_mode=room_mode,
            )
        except (ADBError, OSError, TimeoutError) as exc:
            consecutive_transient_failures += 1
            step += 1
            last_result = _runtime_exception_result(
                exc,
                transient=True,
                consecutive_failures=consecutive_transient_failures,
            )
            backoff = min(
                max(interval_seconds, 0.1) * (2 ** min(consecutive_transient_failures - 1, 4)),
                MAX_TRANSIENT_BACKOFF_SECONDS,
            )
            time.sleep(backoff)
            continue
        except Exception as exc:
            step += 1
            last_result = _runtime_exception_result(
                exc,
                transient=False,
                consecutive_failures=1,
            )
            break
        consecutive_transient_failures = 0
        step += 1
        if last_result.get("fatal"):
            break
        current_signature = _plan_signature(last_result.get("action_plan", {}))
        current_context = _result_context_signature(last_result)
        if (
            execute_play_actions
            and last_result.get("executed")
            and current_signature is not None
            and current_signature == last_executed_signature
            and current_context == last_executed_context
        ):
            if current_signature[0] == "settlement_ready":
                last_result["action_plan"] = {
                    "action": "wait_next_round",
                    "ready": False,
                    "reason": "已点击准备，等待牌局开始",
                    "clicks": [],
                }
                last_result["executed"] = False
                time.sleep(interval_seconds)
                continue
            unchanged_after_execute_count += 1
            last_result["action_plan"] = {
                "action": "wait_opponent_priority",
                "ready": False,
                "reason": "执行后画面未变化，疑似对手优先权或当前 UI 不可点击，等待下一帧",
                "clicks": [],
            }
            last_result["executed"] = False
            if unchanged_after_execute_count >= 5:
                last_result["action_plan"]["reason"] = "连续执行后画面未变化，疑似 UI 不可点击，暂停等待"
                if not continue_on_halt:
                    break
            time.sleep(interval_seconds)
            continue
        if last_result.get("executed"):
            last_executed_signature = current_signature
            last_executed_context = current_context
            unchanged_after_execute_count = 0
            if (last_result.get("action_plan") or {}).get("action") == "compact_hand":
                time.sleep(0.15)
                continue
        if _should_wait_for_transient_frame(last_result):
            time.sleep(interval_seconds)
            continue
        if should_stop_for_user(last_result, execute_play_actions=execute_play_actions):
            if continue_on_halt:
                time.sleep(interval_seconds)
                continue
            break
        time.sleep(_loop_interval_after_result(last_result, interval_seconds))
    return last_result or {"action_plan": {"ready": False, "reason": "loop did not run", "clicks": []}}


def _loop_interval_after_result(result: dict, default_interval: float) -> float:
    if not result.get("executed"):
        return default_interval
    action = str((result.get("action_plan") or {}).get("action") or "").lower()
    if action not in {"expand_chi_options", "chi", "chi_option", "compare_option"}:
        return default_interval
    return max(0.0, min(float(default_interval), 0.15))


def _prepare_protocol_payload_for_loop(protocol_payload: str | Path | dict | None) -> str | Path | dict | ProtocolLogTail | None:
    if protocol_payload is None or hasattr(protocol_payload, "read_latest"):
        return protocol_payload
    if isinstance(protocol_payload, dict):
        return protocol_payload
    payload_path = Path(protocol_payload)
    if payload_path.exists() and payload_path.is_file():
        return ProtocolLogTail(payload_path)
    return protocol_payload


def _should_wait_for_transient_frame(result: dict) -> bool:
    plan = result.get("action_plan") or {}
    if plan.get("ready"):
        return False
    buttons = result.get("buttons") or []
    hand = result.get("hand") or []
    reason = str(plan.get("reason") or "")
    no_actionable_ui = not buttons and not result.get("discard_button")
    empty_or_bad_hand = not hand or not (result.get("sanity_checks") or {}).get("ok", True)
    return no_actionable_ui and empty_or_bad_hand and (
        "未找到 pass 按钮" in reason
        or "手牌数量校验未通过" in reason
        or result.get("flow", {}).get("state") == "play"
    )


def _opening_untrusted_hand_cards(result: dict) -> list[dict]:
    untrusted: list[dict] = []
    for card in result.get("hand_details") or []:
        if not isinstance(card, dict):
            continue
        confidence = _safe_float(card.get("confidence"))
        hybrid_reason = str(card.get("hybrid_reason") or "")
        if (
            (confidence is not None and confidence < OPENING_HAND_MIN_CONFIDENCE)
            or hybrid_reason in OPENING_UNTRUSTED_HYBRID_REASONS
        ):
            untrusted.append(card)
    return untrusted


def _should_auto_compact_dealer_hand(result: dict) -> bool:
    seat = result.get("seat_role") or (result.get("metadata") or {}).get("seat_role") or {}
    if not isinstance(seat, dict) or seat.get("role") != "dealer":
        return False
    try:
        seat_confidence = float(seat.get("confidence") or 0.0)
    except (TypeError, ValueError):
        seat_confidence = 0.0
    if seat_confidence < 0.9:
        return False
    if not str(seat.get("reason") or "").startswith(("dealer_marker", "manual_dealer")):
        return False
    opening_shortage = _is_dealer_opening_shortage(result)
    if result.get("discard_button") is None:
        return False
    if not opening_shortage:
        return False

    button_names = {
        str(button.get("name") or button.get("type") or "").lower()
        for button in result.get("buttons", []) or []
        if isinstance(button, dict)
    }
    if button_names & {"chi", "peng", "hu", "pao", "pass"}:
        return False
    if result.get("option_details") or result.get("options") or result.get("option_stage"):
        return False

    sanity = result.get("sanity_checks") or {}
    try:
        meld_cells = int(sanity.get("my_meld_cell_count") or 0)
    except (TypeError, ValueError):
        meld_cells = 0
    if meld_cells > 0:
        return False

    hand = result.get("hand") or []
    try:
        hand_count = int((result.get("metadata") or {}).get("hand_count") or len(hand))
    except (TypeError, ValueError):
        hand_count = len(hand) if isinstance(hand, list) else 0
    if not (18 <= hand_count < 21):
        binding = (result.get("metadata") or {}).get("protocol_hand_binding") or {}
        visible_count = _safe_int(binding.get("visible_slot_count")) if isinstance(binding, dict) else 0
        if not (18 <= visible_count < 21):
            return False
    return True


def _unchanged_executed_frame_wait(result: dict, guard_state: dict) -> dict | None:
    signature = guard_state.get("last_executed_signature")
    previous_context = guard_state.get("last_context")
    if (
        not isinstance(signature, list)
        or not signature
        or signature[0] in {"compact_hand", "settlement_ready"}
        or _signature_is_chi_expand(signature)
        or not isinstance(previous_context, dict)
    ):
        return None
    current_context = _result_context_signature(result)
    visual_keys = (
        "flow_state",
        "remaining_deck_count",
        "controlled_card_count",
        "hand_signature",
        "button_names",
        "discard_button_visible",
        "pending_action_card",
        "opponent_pending_card",
        "option_stage",
    )
    if any(current_context.get(key) != previous_context.get(key) for key in visual_keys):
        return None
    waiting = dict(result)
    waiting["executed"] = False
    waiting["action_plan"] = {
        "action": "wait_unchanged_after_execution",
        "ready": False,
        "reason": "上一个动作执行后画面尚未变化，等待新画面，禁止重复思考和点击",
        "clicks": [],
        "validation": {
            "passed": False,
            "reason_code": "unchanged_after_execution",
            "checks": ["visual_context_changed_before_redecision"],
        },
    }
    return waiting


def _handle_opening_compact(
    result: dict,
    *,
    screenshot: str | Path,
    device_id: str,
    guard_file: str | Path,
    guard_state: dict,
    execute_play_actions: bool,
    logger: GameLogger | None,
    frame_id: str,
    external_health_check: Callable[[], str | None] | None,
) -> dict:
    recovery = guard_state.get("opening_compact_recovery")
    recovery = recovery if isinstance(recovery, dict) else {}
    attempts = _safe_int(recovery.get("attempts"))
    no_progress = _safe_int(recovery.get("no_progress"))
    layout_signature = _opening_layout_signature(result)
    previous_no_movable_signature = recovery.get("no_movable_signature")
    no_movable_rechecks = _safe_int(recovery.get("no_movable_rechecks"))
    if attempts >= OPENING_COMPACT_MAX_ATTEMPTS or no_progress >= OPENING_COMPACT_MAX_NO_PROGRESS:
        return _opening_compact_halt(
            result,
            attempts=attempts,
            no_progress=no_progress,
            reason="整理手牌连续没有改善，已停止坐标重试",
        )
    if no_movable_rechecks >= 1 and previous_no_movable_signature == layout_signature:
        return _opening_compact_halt(
            result,
            attempts=attempts,
            no_progress=no_progress,
            reason="开局仍未确认满21张，且右侧已没有可安全移动的牌；保持轻量监控，不会继续重复整理",
            reason_code="opening_compact_exhausted",
        )

    compact = run_compact(
        screenshot=screenshot,
        device_id=device_id,
        expected_total=21,
        force=True,
        execute=False,
    )
    result["compact_hand"] = compact
    planned_drags = list(compact.get("planned_drags") or [])
    if not planned_drags:
        if no_movable_rechecks >= 1:
            return _opening_compact_halt(
                result,
                attempts=attempts,
                no_progress=no_progress,
                reason="开局仍未确认满21张，且右侧已没有可安全移动的牌；保持轻量监控，不会继续重复整理",
                reason_code="opening_compact_exhausted",
            )
        next_recovery = {
            "attempts": attempts,
            "no_progress": no_progress,
            "no_movable_rechecks": 1,
            "no_movable_signature": layout_signature,
            "last_visible_count": _safe_int(
                (result.get("sanity_checks") or {}).get("controlled_card_count")
            ),
            "updated_at": time.time(),
        }
        next_guard = {
            **_load_guard(guard_file),
            "opening_compact_recovery": next_recovery,
        }
        if not _save_guard(guard_file, next_guard):
            return _halt_for_guard_persistence(
                result,
                phase="after_opening_no_movable_cards",
                action_was_attempted=False,
            )
        waiting = dict(result)
        waiting["executed"] = False
        waiting["opening_compact_recovery"] = next_recovery
        waiting["action_plan"] = {
            "action": "wait_opening_locked_recheck",
            "ready": False,
            "reason": "右侧已无可移动牌，停止整理；下一帧只复核锁牌和21张总数",
            "clicks": [],
            "validation": {
                "passed": False,
                "reason_code": "opening_locked_recheck",
                "checks": ["no_more_drags", "one_fresh_frame_recheck"],
            },
        }
        return waiting

    visible_count = _safe_int((result.get("sanity_checks") or {}).get("controlled_card_count"))
    untrusted_cards = _opening_untrusted_hand_cards(result)
    steps = _select_opening_compact_batch(planned_drags)
    issue = (
        f"识别到 {visible_count} 张且有 {len(untrusted_cards)} 张低可信牌"
        if untrusted_cards
        else f"只识别到 {visible_count}/21 张"
    )
    compact_plan = {
        "action": "compact_hand",
        "ready": True,
        "reason": f"庄家开局{issue}，从最右侧列起连续移动最多2张可移动牌后统一复查",
        "execution_mode": "drag_sequence",
        "clicks": [
            {
                "target": "hand:compact",
                "x": step["end_x"],
                "y": step["end_y"],
                "from_x": step["start_x"],
                "from_y": step["start_y"],
                "label": step["label"],
                "duration_ms": 520,
            }
            for step in steps
        ],
        "validation": {
            "passed": True,
            "checks": [
                "dealer_marker_precedes_hand_count",
                "right_to_left_clickable_sources_only",
                "locked_source_columns_skipped",
                "max_four_cards_per_column",
                "two_card_batch_then_single_recapture",
            ],
        },
    }
    result["action_plan"] = compact_plan
    result["executed"] = False
    if not execute_play_actions:
        return result

    health_issue = _runtime_health_issue(external_health_check)
    if health_issue:
        return _halt_for_external_runtime(
            result,
            health_issue,
            logger=logger,
            frame_id=frame_id,
            screenshot_path=str(screenshot),
        )
    compact_signature = _plan_signature(compact_plan)
    compact_context = _result_context_signature(result)
    if compact_signature is None:
        result["action_plan"] = {
            "action": "wait_duplicate_execution",
            "ready": False,
            "reason": "整理手牌动作仍在处理中，等待新画面后再规划",
            "clicks": [],
        }
        return result
    if _should_block_repeated_execution(
        guard_state,
        compact_signature,
        compact_context,
    ):
        if _verified_compact_no_progress(
            guard_state,
            compact_signature,
            compact_context,
            no_progress=no_progress,
        ):
            return _opening_compact_halt(
                result,
                attempts=attempts,
                no_progress=no_progress,
                reason="上一次整理已执行但画面没有改善，已停止重复同一坐标",
            )
        result["action_plan"] = {
            "action": "wait_duplicate_execution",
            "ready": False,
            "reason": "整理手牌动作仍在处理中，等待新画面后再规划",
            "clicks": [],
        }
        return result
    if not _arm_action_guard(guard_file, guard_state, compact_signature, compact_context):
        return _halt_for_guard_persistence(
            result,
            phase="before_compact_hand",
            action_was_attempted=False,
        )

    before_metrics = _opening_layout_metrics(result)
    result["executed"] = execute_action_plan(compact_plan, device_id=device_id)
    if not result["executed"]:
        failed_recovery = {
            "attempts": attempts + 1,
            "no_progress": no_progress + 1,
            "last_visible_count": before_metrics["visible_count"],
            "last_right_zone_count": before_metrics["right_zone_count"],
            "last_untrusted_count": before_metrics["untrusted_count"],
            "updated_at": time.time(),
        }
        failed_guard = {
            **_load_guard(guard_file),
            "opening_compact_recovery": failed_recovery,
        }
        failed_guard.pop("inflight_action", None)
        if not _save_guard(guard_file, failed_guard):
            return _halt_for_guard_persistence(
                result,
                phase="after_failed_compact_hand",
                action_was_attempted=True,
            )
        result["compact_verification"] = {
            "screenshot": None,
            "before": before_metrics,
            "after": before_metrics,
            "progress": False,
            "recovery": failed_recovery,
        }
        return result
    after_screenshot = str(save_screen(device_id=device_id)) if result["executed"] else None
    after_state = inspect_screenshot(after_screenshot, expected_total=None) if after_screenshot else None
    after_metrics = _opening_layout_metrics(after_state or {})
    progress = bool(
        after_state
        and (
            after_metrics["visible_count"] > before_metrics["visible_count"]
            or after_metrics["right_zone_count"] < before_metrics["right_zone_count"]
            or after_metrics["untrusted_count"] < before_metrics["untrusted_count"]
        )
    )
    next_recovery = {
        "attempts": attempts + 1,
        "no_progress": 0 if progress else no_progress + 1,
        "last_visible_count": after_metrics["visible_count"],
        "last_right_zone_count": after_metrics["right_zone_count"],
        "last_untrusted_count": after_metrics["untrusted_count"],
        "updated_at": time.time(),
    }
    compact_guard = {
        **_load_guard(guard_file),
        "last_executed_signature": _signature_to_json(compact_signature),
        "last_context": compact_context,
        "last_executed_at": time.time(),
        "opening_compact_recovery": next_recovery,
    }
    compact_guard.pop("inflight_action", None)
    if after_metrics["visible_count"] >= 21 and after_metrics["untrusted_count"] == 0:
        compact_guard.pop("opening_compact_recovery", None)
        compact_guard.pop("pending_opening_recognition_retry", None)
    if not _save_guard(guard_file, compact_guard):
        return _halt_for_guard_persistence(
            result,
            phase="after_compact_hand",
            action_was_attempted=True,
        )
    result["compact_verification"] = {
        "screenshot": after_screenshot,
        "before": before_metrics,
        "after": after_metrics,
        "progress": progress,
        "recovery": next_recovery,
    }
    if logger is not None and logger.round_active:
        last_click = compact_plan["clicks"][-1]
        logger.log_tap(
            frame_id=frame_id,
            decision_id=result.get("decision_id"),
            plan=compact_plan,
            execute_enabled=True,
            dry_run=False,
            tap_executed=result["executed"],
            tap_x=int(last_click["x"]),
            tap_y=int(last_click["y"]),
            target_label=str(last_click.get("label") or "") or None,
            adb_result="ok" if result["executed"] else "not_executed",
            before_screenshot=str(screenshot),
            after_screenshot=after_screenshot,
            executed_clicks=compact_plan["clicks"],
        )
    return result


def _select_opening_compact_batch(planned_drags: list[dict]) -> list[dict]:
    """Select at most one drag per destination column before the next recapture."""

    selected: list[dict] = []
    target_columns: set[int] = set()
    for step in planned_drags:
        try:
            target_x = int(step["end_x"])
        except (KeyError, TypeError, ValueError):
            continue
        if target_x in target_columns:
            continue
        selected.append(step)
        target_columns.add(target_x)
        if len(selected) >= OPENING_COMPACT_BATCH_SIZE:
            break
    return selected


def _opening_layout_metrics(result: dict) -> dict[str, int]:
    cards = [item for item in result.get("hand_details") or [] if isinstance(item, dict)]
    right_zone_count = 0
    for card in cards:
        center = card.get("center")
        try:
            center_x = int(center[0]) if center else int(card.get("x")) + int(card.get("w", 0)) // 2
        except (TypeError, ValueError, IndexError):
            continue
        if center_x >= OPENING_COMPACT_RIGHT_OF:
            right_zone_count += 1
    return {
        "visible_count": len(cards),
        "right_zone_count": right_zone_count,
        "untrusted_count": len(_opening_untrusted_hand_cards(result)),
    }


def _opening_layout_signature(result: dict) -> list[list[object]]:
    signature: list[list[object]] = []
    for card in result.get("hand_details") or []:
        if not isinstance(card, dict):
            continue
        center = card.get("center")
        try:
            center_x = int(center[0])
            center_y = int(center[1])
        except (TypeError, ValueError, IndexError):
            continue
        signature.append(
            [
                int(round(center_x / 12.0) * 12),
                int(round(center_y / 12.0) * 12),
                bool(card.get("clickable", True)),
            ]
        )
    signature.sort(key=lambda item: (int(item[0]), int(item[1]), bool(item[2])))
    return signature


def _verified_compact_no_progress(
    guard_state: dict,
    signature: tuple,
    context_signature: dict,
    *,
    no_progress: int,
) -> bool:
    if no_progress <= 0:
        return False
    return (
        not isinstance(guard_state.get("inflight_action"), dict)
        and guard_state.get("last_executed_signature") == _signature_to_json(signature)
        and guard_state.get("last_context") == context_signature
    )


def _opening_compact_halt(
    result: dict,
    *,
    attempts: int,
    no_progress: int,
    reason: str,
    reason_code: str = "opening_compact_no_progress",
) -> dict:
    halted = dict(result)
    halted["executed"] = False
    halted["action_plan"] = {
        "action": "safe_halt",
        "ready": False,
        "reason": reason,
        "clicks": [],
        "validation": {
            "passed": False,
            "reason_code": reason_code,
            "checks": ["bounded_attempts", "fresh_frame_progress"],
        },
    }
    halted["opening_compact_recovery"] = {
        "attempts": attempts,
        "no_progress": no_progress,
    }
    return halted


def _defer_opening_compact_for_recognition_retry(
    result: dict,
    *,
    guard_file: str | Path,
    guard_state: dict,
) -> dict | None:
    pending = guard_state.get("pending_opening_recognition_retry")
    try:
        attempts = int(pending.get("attempts") or 0) if isinstance(pending, dict) else 0
    except (TypeError, ValueError):
        attempts = 0
    if attempts >= DEALER_OPENING_RECOGNITION_RETRIES:
        _clear_opening_recognition_retry(guard_file, guard_state)
        return None

    next_attempt = attempts + 1
    updated_guard = {
        **guard_state,
        "pending_opening_recognition_retry": {
            "attempts": next_attempt,
            "controlled_card_count": _safe_int(
                (result.get("sanity_checks") or {}).get("controlled_card_count")
            ),
            "created_at": time.time(),
        },
    }
    if not _save_guard(guard_file, updated_guard):
        return _halt_for_guard_persistence(
            result,
            phase="before_opening_recognition_retry",
            action_was_attempted=False,
        )

    deferred = dict(result)
    deferred["executed"] = False
    deferred["action_plan"] = {
        "action": "wait_recognition_retry",
        "ready": False,
        "reason": (
            f"庄家首出只识别到 {updated_guard['pending_opening_recognition_retry']['controlled_card_count']} 张，"
            "等待下一帧避开手指动画；仍不足再整理手牌"
        ),
        "clicks": [],
        "validation": {
            "passed": False,
            "reason_code": "dealer_opening_recognition_retry",
            "checks": ["dealer_marker", "discard_button_visible", "opening_hand_shortage"],
        },
    }
    deferred["recognition_retry"] = updated_guard["pending_opening_recognition_retry"]
    return deferred


def _clear_opening_recognition_retry(guard_file: str | Path, guard_state: dict) -> None:
    if "pending_opening_recognition_retry" not in guard_state:
        return
    updated = dict(guard_state)
    updated.pop("pending_opening_recognition_retry", None)
    _save_guard(guard_file, updated)


def _clear_opening_recovery(guard_file: str | Path, guard_state: dict) -> None:
    keys = {"pending_opening_recognition_retry", "opening_compact_recovery"}
    if not keys.intersection(guard_state):
        return
    updated = dict(guard_state)
    for key in keys:
        updated.pop(key, None)
    _save_guard(guard_file, updated)


def _discard_target_is_already_safe(result: dict, plan: dict) -> bool:
    if plan.get("action") != "discard" or not plan.get("ready"):
        return False
    if plan.get("execution_mode") not in {"discard_drag", "select_then_discard_button"}:
        return False
    target_card_id = str(plan.get("target_card_id") or "")
    clicks = plan.get("clicks") or []
    click = clicks[0] if clicks and isinstance(clicks[0], dict) else {}
    try:
        click_x = int(click.get("x"))
    except (TypeError, ValueError):
        return False
    if click_x >= 1500:
        return False
    for card in result.get("hand_details") or []:
        if str(card.get("card_id") or card.get("id") or "") != target_card_id:
            continue
        if not card.get("clickable", True):
            return False
        return card.get("center") is not None or (card.get("x") is not None and card.get("y") is not None)
    return bool(target_card_id and click.get("target"))


def _is_dealer_opening_shortage(result: dict) -> bool:
    protocol_binding = (result.get("metadata") or {}).get("protocol_hand_binding") or {}
    if isinstance(protocol_binding, dict):
        protocol_count = _safe_int(protocol_binding.get("protocol_count"))
        visible_count = _safe_int(protocol_binding.get("visible_slot_count"))
        missing_count = _safe_int(protocol_binding.get("missing_coordinate_count"))
        if protocol_count == 21 and 18 <= visible_count < protocol_count and missing_count > 0:
            try:
                meld_cells = int((result.get("sanity_checks") or {}).get("my_meld_cell_count") or 0)
            except (TypeError, ValueError):
                meld_cells = 0
            my_discards = ((result.get("discards") or {}).get("my_discards") or [])
            if meld_cells == 0 and not my_discards:
                return True

    sanity = result.get("sanity_checks") or {}
    try:
        controlled = int(sanity.get("controlled_card_count") or 0)
    except (TypeError, ValueError):
        controlled = 0
    if not (18 <= controlled < 21):
        return False
    try:
        meld_cells = int(sanity.get("my_meld_cell_count") or 0)
    except (TypeError, ValueError):
        meld_cells = 0
    if meld_cells > 0:
        return False
    my_discards = ((result.get("discards") or {}).get("my_discards") or [])
    if my_discards:
        return False
    return True


def _pending_response_any(guard_state: dict) -> dict | None:
    pending = guard_state.get("pending_response")
    if not isinstance(pending, dict):
        return None
    return pending


def _reuse_pending_chi_option_plan(result: dict, guard_state: dict) -> dict | None:
    pending = _pending_response_any(guard_state)
    if pending is None or str(pending.get("action")) != "chi":
        return None
    intended_cards = [str(label) for label in pending.get("intended_option_cards") or []]
    if not intended_cards:
        return None
    previous_context = pending.get("context")
    if not isinstance(previous_context, dict):
        return None
    current_context = _result_context_signature(result)
    if current_context.get("hand_signature") != previous_context.get("hand_signature"):
        return None
    if current_context.get("controlled_card_count") != previous_context.get("controlled_card_count"):
        return None

    visible_options = [
        option
        for option in result.get("option_details") or []
        if isinstance(option, dict) and option.get("region_name") == "chi_options"
    ]
    matching_index = next(
        (
            index
            for index, option in enumerate(visible_options, start=1)
            if Counter(str(label) for label in option.get("labels") or [])
            == Counter(intended_cards)
            and option.get("center") is not None
        ),
        None,
    )
    matching_option = visible_options[matching_index - 1] if matching_index is not None else None
    if matching_option is None:
        return None

    decision = {
        "action": "chi",
        "candidate_stage": "cached_semantic_chi",
        "selected_option_id": matching_option.get("option_id") or f"chi_{matching_index:03d}",
        "selected_option_cards": [str(label) for label in matching_option.get("labels") or []],
        "reason": "复用上一帧已完成的吃牌策略，只映射当前候选坐标",
    }
    action_plan = build_action_plan(result, decision)
    if not action_plan.get("ready") or action_plan.get("action") != "chi_option":
        return None
    reused = dict(result)
    reused["decision"] = decision
    reused["action_plan"] = action_plan
    reused["conflict_guard_result"] = {
        "passed": True,
        "safe_halt": False,
        "reason": None,
        "checks": [
            "pending_chi_hand_signature_unchanged",
            "pending_chi_controlled_count_unchanged",
            "pending_chi_option_exact_match",
        ],
    }
    reused["pending_chi_plan_cache_hit"] = True
    return reused


def _pending_chi_option_plan_from_surface(
    screenshot: str | Path,
    guard_state: dict,
    *,
    flow,
    buttons: list[ButtonDetection],
) -> dict | None:
    """Map one confirmed CHI intent on its immediate candidate frame."""
    pending = _pending_response_lock(guard_state)
    if pending is None:
        return None
    if str(pending.get("action")) != "chi":
        return None
    if str(getattr(flow, "state", "")) != "play":
        return None
    button_dicts = [button.to_dict() for button in buttons]
    if any(str(button.get("name") or "").lower() == "hu" for button in button_dicts):
        return None

    intended_cards = [str(label) for label in pending.get("intended_option_cards") or []]
    previous_context = pending.get("context")
    if not intended_cards or not isinstance(previous_context, dict):
        return None
    if previous_context.get("flow_state") not in {None, "play"}:
        return None

    hand: list[str] = []
    for item in previous_context.get("hand_signature") or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            count = int(item[1])
        except (TypeError, ValueError):
            return None
        if count < 0:
            return None
        hand.extend([str(item[0])] * count)
    try:
        controlled_count = int(previous_context.get("controlled_card_count"))
    except (TypeError, ValueError):
        return None
    if controlled_count < len(hand):
        return None

    image = read_image(Path(screenshot))
    visible_hand_slots = detect_hand_slots(image, config_path=DEFAULT_CONFIG_PATH)
    visible_contour_count = sum(
        getattr(slot, "source", None) == "contour"
        for slot in visible_hand_slots
    )
    if visible_contour_count != len(hand):
        return None
    current_pending = recognize_pending_cards(image, config_path=DEFAULT_CONFIG_PATH)
    source_key = (
        "opponent_pending_card"
        if previous_context.get("opponent_pending_card")
        else "pending_action_card"
    )
    expected_incoming = previous_context.get(source_key)
    observed_incoming = current_pending.get(source_key)
    if (
        not expected_incoming
        or observed_incoming is None
        or str(observed_incoming.name) != str(expected_incoming)
    ):
        return None
    option_details = [
        option.to_dict()
        for option in recognize_options(image, config_path=DEFAULT_CONFIG_PATH)
    ]
    if not option_details:
        return None

    result = {
        "screenshot": str(screenshot),
        "hand": hand,
        "hand_count": len(hand),
        "buttons": button_dicts,
        "option_details": option_details,
        "options": [option.get("labels") or [] for option in option_details],
        "option_stage": "chi",
        "remaining_deck_count": previous_context.get("remaining_deck_count"),
        "sanity_checks": {
            "ok": True,
            "controlled_card_count": controlled_count,
            "expected_total": controlled_count,
        },
        "flow": flow.to_dict(),
        "pending_action_card": (
            current_pending["pending_action_card"].to_dict()
            if current_pending.get("pending_action_card") is not None
            else None
        ),
        "opponent_pending_card": (
            current_pending["opponent_pending_card"].to_dict()
            if current_pending.get("opponent_pending_card") is not None
            else None
        ),
        # Preserve the source-specific incoming card proven by the transaction
        # guard. A low-confidence detection in the other pending-card region must
        # not rewrite the selected CHI option during action-plan construction.
        "pending_card": expected_incoming,
        "metadata": {
            "hand_count": len(hand),
            "pending_chi_surface_only": True,
        },
    }
    reused = _reuse_pending_chi_option_plan(result, guard_state)
    if reused is None or action_plan_contract_error(reused.get("action_plan") or {}) is not None:
        return None
    reused["pending_chi_surface_fast_path"] = True
    reused["conflict_guard_result"]["checks"].extend(
        [
            "pending_chi_transaction_unexpired",
            "current_play_surface",
            "visible_hu_absent",
            "current_hand_slot_count_unchanged",
            "current_incoming_card_and_source_unchanged",
        ]
    )
    return reused


def _pending_compare_option_plan_from_surface(
    screenshot: str | Path,
    guard_state: dict,
    *,
    flow,
    buttons: list[ButtonDetection],
) -> dict | None:
    """Map the third, physically required compare click without a second policy search."""
    pending = _pending_response_lock(guard_state)
    if pending is None or str(pending.get("action")) != "compare":
        return None
    if str(getattr(flow, "state", "")) != "play":
        return None
    button_dicts = [button.to_dict() for button in buttons]
    if any(str(button.get("name") or "").lower() == "hu" for button in button_dicts):
        return None
    previous_context = pending.get("context")
    if not isinstance(previous_context, dict):
        return None

    image = read_image(Path(screenshot))
    option_details = [
        option.to_dict()
        for option in recognize_options(image, config_path=DEFAULT_CONFIG_PATH)
    ]
    compare_options = [
        option for option in option_details if option.get("region_name") == "compare_options"
    ]
    if not compare_options:
        return None

    hand: list[str] = []
    for item in previous_context.get("hand_signature") or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            hand.extend([str(item[0])] * max(0, int(item[1])))
        except (TypeError, ValueError):
            continue
    try:
        controlled_count = int(previous_context.get("controlled_card_count") or len(hand))
    except (TypeError, ValueError):
        controlled_count = len(hand)
    result = {
        "screenshot": str(screenshot),
        "raw_hand": hand,
        "hand": hand,
        "hand_count": len(hand),
        "buttons": button_dicts,
        "option_details": option_details,
        "options": [option.get("labels") or [] for option in option_details],
        "option_stage": "compare",
        "remaining_deck_count": previous_context.get("remaining_deck_count"),
        "sanity_checks": {
            "ok": True,
            "controlled_card_count": controlled_count,
            "expected_total": controlled_count,
        },
        "flow": flow.to_dict(),
        "decision": {
            "action": "compare_option",
            "candidate_stage": "pending_compare_surface",
            "reason": "复用已确认的吃牌事务，只评估当前比牌候选",
        },
        "metadata": {"pending_compare_surface_only": True},
    }
    action_plan = build_compare_option_plan(result)
    if action_plan is None or action_plan_contract_error(action_plan) is not None:
        return None
    result["action_plan"] = action_plan
    result["conflict_guard_result"] = {
        "passed": True,
        "safe_halt": False,
        "reason": None,
        "checks": [
            "pending_compare_transaction_unexpired",
            "current_play_surface",
            "visible_hu_absent",
            "visible_compare_options_evaluated",
            "compare_target_clickable",
        ],
    }
    result["pending_compare_surface_fast_path"] = True
    return result


def _pending_response_lock(guard_state: dict) -> dict | None:
    pending = _pending_response_any(guard_state)
    if pending is None:
        return None
    try:
        expires_at = float(pending.get("expires_at") or 0)
    except (TypeError, ValueError):
        return None
    if expires_at <= time.time():
        return None
    return pending


def _bounded_untrusted_pending_card_recheck(
    result: dict,
    guard_state: dict,
) -> tuple[dict, dict, bool]:
    """Bound an unreadable incoming-card retry without ever probing CHI blindly."""

    plan = result.get("action_plan") or {}
    if str(plan.get("action") or "") != "wait_chi_candidate_recheck":
        if "pending_card_recheck" not in guard_state:
            return result, guard_state, False
        cleaned = dict(guard_state)
        cleaned.pop("pending_card_recheck", None)
        return result, cleaned, True

    observation_key = _result_context_signature(result)
    previous = guard_state.get("pending_card_recheck")
    stable_frames = 1
    if isinstance(previous, dict) and previous.get("observation_key") == observation_key:
        stable_frames = int(previous.get("stable_frames") or 0) + 1

    updated_guard = dict(guard_state)
    updated_guard["pending_card_recheck"] = {
        "observation_key": observation_key,
        "stable_frames": stable_frames,
        "updated_at": time.time(),
    }
    if stable_frames < PENDING_CARD_RECHECK_MAX_FRAMES:
        return result, updated_guard, True

    pass_plan = build_action_plan(result, {"action": "pass", "label": None})
    if not pass_plan.get("ready"):
        return result, updated_guard, True

    pass_plan["reason"] = "进牌连续三帧无法可靠识别，为避免盲点吃和界面死循环，保守点过"
    pass_plan["policy_selected_action"] = {"type": "pass", "label": None}
    pass_plan.setdefault("validation", {}).update(
        {
            "passed": True,
            "reason_code": "pending_card_untrusted_bounded_pass",
            "checks": [
                "pending_card_rechecked_three_stable_frames",
                "pass_button_visible",
                "blind_chi_forbidden",
            ],
        }
    )
    patched = dict(result)
    decision = dict(patched.get("decision") or {})
    decision.update(
        {
            "action": "pass",
            "policy_action": "pass",
            "selected_label": None,
            "selected_option_id": None,
            "selected_option_cards": [],
            "candidate_stage": "pending_card_untrusted_bounded_pass",
            "reason": pass_plan["reason"],
        }
    )
    patched["decision"] = decision
    patched["action_plan"] = pass_plan
    patched["pending_card_untrusted_bounded_pass"] = True
    updated_guard.pop("pending_card_recheck", None)
    return patched, updated_guard, True


def _stabilize_pending_chi_candidate(
    result: dict,
    guard_state: dict,
) -> tuple[str | None, dict]:
    """Retarget a stale semantic CHI only after two identical real surfaces.

    The pre-click strategy intent remains the zero-delay path. This recovery
    exists only when the game's visible candidate set does not contain that
    cached intent. A candidate must be both selected by the normal policy and
    present at a clickable visible column on two consecutive frames; PASS and
    non-visible targets can never be released here.
    """

    pending = _pending_response_any(guard_state)
    if pending is None or str(pending.get("action")) != "chi":
        return None, guard_state
    intended_cards = [str(label) for label in pending.get("intended_option_cards") or []]
    if not intended_cards:
        return None, guard_state

    plan = result.get("action_plan") or {}
    if (
        not plan.get("ready")
        or str(plan.get("action") or "") != "chi_option"
        or plan.get("target_type") != "option_column"
    ):
        return None, guard_state
    selected_cards = [str(label) for label in plan.get("target_option_cards") or []]
    if not selected_cards:
        return None, guard_state
    if Counter(selected_cards) == Counter(intended_cards):
        return None, guard_state

    visible_options = [
        option
        for option in result.get("option_details") or []
        if (
            isinstance(option, dict)
            and option.get("region_name") == "chi_options"
            and option.get("center") is not None
            and option.get("labels")
        )
    ]
    if not any(
        Counter(str(label) for label in option.get("labels") or [])
        == Counter(selected_cards)
        for option in visible_options
    ):
        return None, guard_state

    visible_signature = sorted(
        sorted(str(label) for label in option.get("labels") or [])
        for option in visible_options
    )
    observation_key = {
        "selected_cards": sorted(selected_cards),
        "visible_options": visible_signature,
    }
    previous = pending.get("candidate_recovery")
    stable_frames = 1
    if (
        isinstance(previous, dict)
        and previous.get("observation_key") == observation_key
        and not previous.get("released")
    ):
        stable_frames = int(previous.get("stable_frames") or 0) + 1

    updated_guard = dict(guard_state)
    updated_pending = dict(pending)
    updated_guard["pending_response"] = updated_pending
    if stable_frames < 2:
        updated_pending["candidate_recovery"] = {
            "released": False,
            "stable_frames": stable_frames,
            "observation_key": observation_key,
        }
        return "recheck", updated_guard

    updated_pending["intended_option_cards"] = selected_cards
    updated_pending["candidate_recovery"] = {
        "released": True,
        "stable_frames": stable_frames,
        "selected_cards": selected_cards,
    }
    return "release", updated_guard


def _should_hold_for_response_options(result: dict, pending_response: dict) -> bool:
    if str(pending_response.get("action")) != "chi":
        return False
    plan = result.get("action_plan") or {}
    action = str(plan.get("action") or "")
    if action == "hu":
        return False
    intended_cards = [str(label) for label in pending_response.get("intended_option_cards") or []]
    selected_cards = [str(label) for label in plan.get("target_option_cards") or []]
    if intended_cards:
        if action in {"chi_option", "chi"} and plan.get("target_type") == "option_column":
            return Counter(selected_cards) != Counter(intended_cards)
        if result.get("option_details") or result.get("options") or result.get("option_stage"):
            return True
    elif action in {"chi_option", "compare_option"}:
        return False
    elif action == "chi" and plan.get("target_type") == "option_column":
        return False
    elif result.get("option_details") or result.get("options") or result.get("option_stage"):
        return False
    return action == "pass" or _is_chi_expand_plan(plan) or not plan.get("ready")


def _should_halt_for_failed_chi_expand(result: dict, guard_state: dict) -> bool:
    pending = _pending_response_any(guard_state)
    if not pending or str(pending.get("action")) != "chi":
        return False
    options_visible = bool(
        result.get("option_details") or result.get("options") or result.get("option_stage")
    )
    intended_cards = [str(label) for label in pending.get("intended_option_cards") or []]
    if options_visible and not intended_cards:
        return False
    try:
        expires_at = float(pending.get("expires_at") or 0)
    except (TypeError, ValueError):
        return False
    if expires_at > time.time():
        return False
    try:
        attempts = int(pending.get("attempts") or 1)
    except (TypeError, ValueError):
        attempts = 1
    if not options_visible and attempts < CHI_EXPAND_MAX_ATTEMPTS:
        return False
    plan = result.get("action_plan") or {}
    action = str(plan.get("action") or "")
    if options_visible and intended_cards:
        selected_cards = [str(label) for label in plan.get("target_option_cards") or []]
        return not (
            action in {"chi_option", "chi"}
            and plan.get("target_type") == "option_column"
            and Counter(selected_cards) == Counter(intended_cards)
        )
    return action == "pass" or _is_chi_expand_plan(plan) or not plan.get("ready")


def _should_hold_after_chi_click(result: dict, guard_state: dict) -> bool:
    try:
        last_executed_at = float(guard_state.get("last_executed_at") or 0)
    except (TypeError, ValueError):
        last_executed_at = 0.0
    if time.time() - last_executed_at > 5.0:
        return False
    last = guard_state.get("last_executed_signature")
    if not isinstance(last, list) or not last or last[0] not in {"chi", "expand_chi_options"}:
        return False
    plan = result.get("action_plan") or {}
    if plan.get("action") != "pass":
        return False
    button_names = {
        str(button.get("name") or button.get("type") or "").lower()
        for button in result.get("buttons", []) or []
        if isinstance(button, dict)
    }
    return {"chi", "pass"} <= button_names and not (result.get("option_details") or result.get("options"))


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _can_reuse_auto_seat_result(result: dict, inferred_seat: dict) -> bool:
    expected = inferred_seat.get("expected_total")
    if expected is None:
        return False
    try:
        expected_count = int(expected)
    except (TypeError, ValueError):
        return False

    sanity = result.get("sanity_checks") or {}
    if sanity.get("expected_total") is None:
        return False
    count = sanity.get("controlled_card_count")
    if count is None:
        count = (result.get("metadata") or {}).get("hand_count")
    try:
        controlled_count = int(count)
    except (TypeError, ValueError):
        controlled_count = len(result.get("hand") or [])
    if controlled_count != expected_count:
        return False

    plan = result.get("action_plan") or {}
    reason = str(plan.get("reason") or "")
    return "手牌数量校验未通过" not in reason


def _result_context_signature(result: dict) -> dict:
    sanity = result.get("sanity_checks") or {}
    hand = result.get("hand", [])
    if not isinstance(hand, list):
        hand = list(hand)
    hand_counts = [[str(label), count] for label, count in sorted(Counter(map(str, hand)).items())]
    button_names = sorted(
        str(button.get("name") or button.get("type") or "").lower()
        for button in result.get("buttons") or []
        if isinstance(button, dict) and (button.get("name") or button.get("type"))
    )

    def card_label(key: str) -> str | None:
        card = result.get(key)
        if not isinstance(card, dict):
            return None
        value = card.get("name") or card.get("label")
        return str(value) if value else None

    return {
        "flow_state": result.get("flow", {}).get("state"),
        "remaining_deck_count": result.get("remaining_deck_count"),
        "controlled_card_count": sanity.get("controlled_card_count"),
        "hand_signature": hand_counts,
        "button_names": button_names,
        "discard_button_visible": result.get("discard_button") is not None,
        "pending_action_card": card_label("pending_action_card"),
        "opponent_pending_card": card_label("opponent_pending_card"),
        "option_stage": result.get("option_stage"),
        "decision_action": result.get("decision", {}).get("action"),
    }


def _should_block_repeated_execution(guard_state: dict, signature: tuple | None, context_signature: dict) -> bool:
    if signature is None:
        return False
    action = signature[0]
    if action == "hu":
        return False
    if _should_allow_chi_expand_retry(guard_state, signature, context_signature):
        return False
    inflight = guard_state.get("inflight_action")
    if isinstance(inflight, dict):
        if (
            inflight.get("signature") == _signature_to_json(signature)
            and inflight.get("context") == context_signature
        ):
            return True
    return (
        guard_state.get("last_executed_signature") == _signature_to_json(signature)
        and guard_state.get("last_context") == context_signature
    )


def _should_allow_chi_expand_retry(guard_state: dict, signature: tuple, context_signature: dict) -> bool:
    if not _signature_is_chi_expand(signature):
        return False
    pending = _pending_response_any(guard_state)
    if not pending or str(pending.get("action")) != "chi":
        return False
    if pending.get("context") != context_signature:
        return False
    try:
        expires_at = float(pending.get("expires_at") or 0)
    except (TypeError, ValueError):
        return False
    if expires_at > time.time():
        return False
    try:
        attempts = int(pending.get("attempts") or 1)
    except (TypeError, ValueError):
        attempts = 1
    return attempts < CHI_EXPAND_MAX_ATTEMPTS


def _is_chi_expand_plan(action_plan: dict) -> bool:
    if action_plan.get("action") not in {"chi", "expand_chi_options"}:
        return False
    if action_plan.get("target_type") not in {None, "button"}:
        return False
    clicks = action_plan.get("clicks") or []
    targets = [str(click.get("target") or "") for click in clicks if isinstance(click, dict)]
    return targets == ["button:chi"]


def _should_capture_after_action(
    action_plan: dict,
    *,
    capture_after_action: bool,
) -> bool:
    if not capture_after_action:
        return False
    return not _is_chi_expand_plan(action_plan)


def _signature_is_chi_expand(signature: tuple | None) -> bool:
    if signature is None or signature[0] not in {"chi", "expand_chi_options"}:
        return False
    clicks = signature[1] if len(signature) > 1 else ()
    return tuple(click[0] for click in clicks) == ("button:chi",)


def _plan_signature(action_plan: dict) -> tuple | None:
    if not action_plan.get("ready"):
        return None
    clicks = tuple((click.get("target"), int(click.get("x", 0)), int(click.get("y", 0))) for click in action_plan.get("clicks", []))
    return (action_plan.get("action"), clicks)


def _signature_to_json(signature: tuple) -> list:
    action, clicks = signature
    return [action, [list(click) for click in clicks]]


def _load_guard(path: str | Path) -> dict:
    guard_path = Path(path)
    if not guard_path.exists():
        return {}
    try:
        loaded = json.loads(guard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return _expire_stale_guard_state(loaded)


def _expire_stale_guard_state(guard_state: dict, *, now: float | None = None) -> dict:
    current_time = time.time() if now is None else now
    cleaned = dict(guard_state)
    inflight = cleaned.get("inflight_action")
    if isinstance(inflight, dict):
        signature = inflight.get("signature")
        action = signature[0] if isinstance(signature, list) and signature else None
        started_at = _safe_float(inflight.get("started_at"))
        if (
            action == "compact_hand"
            and started_at is not None
            and current_time - started_at > COMPACT_INFLIGHT_TTL_SECONDS
        ):
            cleaned.pop("inflight_action", None)

    pending_retry = cleaned.get("pending_opening_recognition_retry")
    if isinstance(pending_retry, dict):
        created_at = _safe_float(pending_retry.get("created_at"))
        if created_at is not None and current_time - created_at > OPENING_RETRY_TTL_SECONDS:
            cleaned.pop("pending_opening_recognition_retry", None)

    recovery = cleaned.get("opening_compact_recovery")
    if isinstance(recovery, dict):
        updated_at = _safe_float(recovery.get("updated_at"))
        if updated_at is not None and current_time - updated_at > OPENING_RECOVERY_TTL_SECONDS:
            cleaned.pop("opening_compact_recovery", None)
    return cleaned


def _save_guard(path: str | Path, data: dict) -> bool:
    guard_path = Path(path)
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = guard_path.with_name(f"{guard_path.stem}.{id(data)}.tmp{guard_path.suffix}")
    try:
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(guard_path)
        return True
    except OSError:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def reset_transient_external_action_guard(path: str | Path) -> bool:
    guard_state = _load_guard(path)
    cleaned = dict(guard_state)
    cleaned.pop("inflight_action", None)
    cleaned.pop("pending_response", None)
    if _signature_is_chi_expand(cleaned.get("last_executed_signature")):
        cleaned.pop("last_executed_signature", None)
        cleaned.pop("last_context", None)
        cleaned.pop("last_executed_at", None)
    return _save_guard(path, cleaned)


def print_summary(result: dict) -> None:
    screenshot = result.get("screenshot") or "unavailable"
    flow = result.get("flow") or {}
    print(f"screenshot={screenshot}")
    print(f"flow={flow.get('state', 'unknown')} confidence={flow.get('confidence', 0.0)}")
    if "hand" in result:
        print("hand=" + " ".join(result["hand"]))
        checks = result.get("sanity_checks") or {}
        print(
            "sanity="
            + f"controlled={checks.get('controlled_card_count')} "
            + f"expected={checks.get('expected_total')} ok={checks.get('ok')}"
        )
        decision = result.get("decision") or {}
        print(f"recommend={decision.get('action', 'unknown')} {decision.get('label') or ''}".strip())
    plan = result.get("action_plan") or {}
    print(
        "action_plan="
        + f"ready={plan.get('ready', False)} reason={plan.get('reason', 'unknown')} "
        + f"clicks={plan.get('clicks', [])}"
    )
    print(f"executed={bool(result.get('executed'))}")
    if result.get("runtime_error"):
        print(f"runtime_error={result['runtime_error'].get('message', 'unknown')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one live screen and produce the next action plan.")
    parser.add_argument("--device-id", default=DEFAULT_DEVICE)
    parser.add_argument("--expected-total", type=int, default=20)
    parser.add_argument("--dealer", action="store_true", help="Shortcut for --expected-total 21.")
    parser.add_argument("--player", action="store_true", help="Shortcut for --expected-total 20.")
    parser.add_argument("--auto-seat", action="store_true", help="Infer dealer/player from the current screen.")
    parser.add_argument("--no-auto-compact-dealer-hand", action="store_true")
    parser.add_argument("--capture-after-action", action="store_true")
    parser.add_argument("--hidden-card", action="append", default=[], help="Add a known but visually hidden hand card, e.g. --hidden-card 拾.")
    parser.add_argument("--protocol-payload", default=None, help="JSON/key=value protocol fields from hook or packet capture.")
    parser.add_argument("--memory-file", default="logs/alphadog_live_memory.json")
    parser.add_argument("--guard-file", default="logs/live_action_guard.json")
    parser.add_argument("--session-log-dir", default=None)
    parser.add_argument("--reset-memory", action="store_true")
    parser.add_argument("--opponent-priority-pending", action="store_true")
    parser.add_argument("--execute-settlement-ready", action="store_true")
    parser.add_argument("--execute-play-actions", action="store_true")
    parser.add_argument("--require-action", default=None)
    parser.add_argument("--require-target", default=None)
    parser.add_argument("--simulate", type=int, default=0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=1.5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--continue-on-halt", action="store_true")
    parser.add_argument("--stop-on-halt", action="store_true")
    parser.add_argument("--runtime-screenshot", default=str(DEFAULT_RUNTIME_SCREENSHOT))
    parser.add_argument("--wildcard-mode", choices=("config", "on", "off"), default="config")
    parser.add_argument("--players", type=int, choices=(2, 3), default=None)
    parser.add_argument("--room-mode", choices=tuple(ROOM_MODE_PRESETS), default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    expected_total = 21 if args.dealer else 20 if args.player else args.expected_total
    seat_role = "auto" if args.auto_seat else "manual"
    wildcard_enabled = None if args.wildcard_mode == "config" else args.wildcard_mode == "on"
    rules_for_room(
        wildcard_enabled=wildcard_enabled,
        players=args.players,
        room_mode=args.room_mode,
    )
    memory_file = args.memory_file
    guard_file = args.guard_file
    if args.room_mode is not None:
        mode_key = args.room_mode.replace("-", "_")
        if memory_file == "logs/alphadog_live_memory.json":
            memory_file = f"logs/alphadog_live_memory_{mode_key}.json"
        if guard_file == "logs/live_action_guard.json":
            guard_file = f"logs/live_action_guard_{mode_key}.json"

    runner = run_loop if args.loop else run_once
    logger = None
    if args.session_log_dir:
        logger = GameLogger(
            Path(args.session_log_dir),
            config=LoggerConfig(save_before_after_action=True, save_decision_frames=True, save_error_frames=True),
        )
    kwargs = {
        "device_id": args.device_id,
        "expected_total": expected_total,
        "memory_file": memory_file,
        "guard_file": guard_file,
        "reset_memory": args.reset_memory,
        "opponent_priority_pending": args.opponent_priority_pending,
        "execute_settlement_ready": args.execute_settlement_ready,
        "execute_play_actions": args.execute_play_actions,
        "require_action": args.require_action,
        "require_target": args.require_target,
        "simulations": args.simulate,
        "seat_role": seat_role,
        "hidden_cards": args.hidden_card,
        "auto_compact_dealer_hand": not args.no_auto_compact_dealer_hand,
        "capture_after_action": args.capture_after_action,
        "protocol_payload": args.protocol_payload,
        "logger": logger,
        "runtime_screenshot_path": args.runtime_screenshot,
        "wildcard_enabled": wildcard_enabled,
        "players": args.players,
        "room_mode": args.room_mode,
    }
    if args.loop:
        kwargs["interval_seconds"] = args.interval_seconds
        kwargs["max_steps"] = args.max_steps
        kwargs["continue_on_halt"] = _continue_nonfatal_halts(
            loop=True,
            execute_play_actions=args.execute_play_actions,
            explicitly_continue=args.continue_on_halt,
            explicitly_stop=args.stop_on_halt,
        )
    result = runner(**kwargs)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print_summary(result)


if __name__ == "__main__":
    main()
