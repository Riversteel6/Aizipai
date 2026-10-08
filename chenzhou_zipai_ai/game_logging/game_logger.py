"""Replay-friendly logging helpers for each game session and round."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import EventType, SAFE_HALT_REASONS, LogEvent, Severity
from .serializers import to_jsonl_line
from .log_paths import LogPaths


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_git_commit(path: Path | None = None) -> str | None:
    repo = path or Path(".")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        value = (result.stdout or "").strip()
        return value or None
    except Exception:
        return None


def _normalise_clicks(clicks: Any) -> list[dict[str, Any]]:
    if not isinstance(clicks, list):
        return []
    normalised: list[dict[str, Any]] = []
    for index, click in enumerate(clicks):
        if not isinstance(click, dict):
            continue
        item = {
            "index": index,
            "target": click.get("target"),
            "x": click.get("x"),
            "y": click.get("y"),
        }
        for key in ("delay_ms", "label", "card_id", "option_id", "option_cards"):
            if key in click:
                item[key] = click.get(key)
        normalised.append(item)
    return normalised


def _clicks_from_plan(plan: dict[str, Any], tap_x: int | None = None, tap_y: int | None = None) -> list[dict[str, Any]]:
    clicks = _normalise_clicks(plan.get("clicks"))
    if clicks:
        return clicks
    if tap_x is None or tap_y is None:
        return []
    return [{"index": 0, "target": plan.get("target"), "x": tap_x, "y": tap_y}]


@dataclass
class LoggerConfig:
    save_raw_screenshot: bool = False
    save_debug_screenshot: bool = False
    save_every_frame: bool = False
    save_decision_frames: bool = True
    save_error_frames: bool = True
    save_before_after_action: bool = True
    max_sessions_keep: int | None = None
    compress_round_on_end: bool = False


@dataclass
class GameLogger:
    base_dir: Path
    config: LoggerConfig = field(default_factory=LoggerConfig)
    enabled: bool = True
    runtime_local_only: bool = True
    session_id: str | None = None
    round_id: str | None = None
    dry_run: bool = False
    execute_enabled: bool = False
    device_id: str = ""
    screen_size: tuple[int, int] | None = None
    project_version: str | None = None
    git_commit: str | None = None
    paths: LogPaths = field(default_factory=lambda: LogPaths())

    _session_started: bool = field(default=False, init=False)
    _frame_seq: int = field(default=0, init=False)
    _decision_seq: int = field(default=0, init=False)
    _round_seq: int = field(default=0, init=False)
    _round_paths: dict[str, Any] = field(default_factory=dict, init=False)
    _log_write_failures: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.paths = LogPaths(root=self.base_dir)

    @property
    def session_active(self) -> bool:
        return self._session_started and self.session_id is not None

    @property
    def round_active(self) -> bool:
        return self.round_id is not None

    @property
    def log_write_failures(self) -> int:
        return self._log_write_failures

    @property
    def has_log_write_failure(self) -> bool:
        return self._log_write_failures > 0

    @property
    def round_counters(self) -> dict[str, Any]:
        if not self.round_active:
            return {}
        return self._round_paths.setdefault(
            self.round_id,
            {
                "events": 0,
                "decisions": 0,
                "actions": 0,
                "errors": 0,
                "states": 0,
                "total_safe_halts": 0,
            },
        )

    def start_session(
        self,
        *,
        device_id: str | None = None,
        screen_size: tuple[int, int] | None = None,
        rules_config_path: str = "config/rules.yaml",
        screen_config_path: str = "config/screen_1080x2400.yaml",
        thresholds_config_path: str = "config/thresholds.yaml",
        execute_enabled: bool | None = None,
        notes: str = "",
    ) -> str:
        if not self.enabled:
            return "disabled"

        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid():05d}"
        self.device_id = device_id or self.device_id
        self.screen_size = screen_size
        if execute_enabled is not None:
            self.execute_enabled = execute_enabled

        self._ensure(self.paths.session_dir(self.session_id))
        self._session_started = True
        self.git_commit = _safe_git_commit(Path("."))
        self.project_version = self.project_version or "0.1.0"

        session_meta = {
            "session_id": self.session_id,
            "started_at": _utc_now(),
            "ended_at": None,
            "device_id": self.device_id,
            "screen_size": list(screen_size) if screen_size is not None else None,
            "project_version": self.project_version,
            "git_commit": self.git_commit,
            "dry_run": self.dry_run,
            "execute_enabled": self.execute_enabled,
            "rules_config_path": rules_config_path,
            "screen_config_path": screen_config_path,
            "thresholds_config_path": thresholds_config_path,
            "runtime_local_only": self.runtime_local_only,
            "notes": notes,
            "logging_config": self.config.__dict__,
        }
        self._write_json(self.paths.session_meta_path(self.session_id), session_meta)
        self._append_jsonl(self.paths.session_events_path(self.session_id), self._event_payload(
            EventType.SESSION_START,
            Severity.INFO,
            {"message": "session started", "rules_config_path": rules_config_path, "notes": notes},
        ))
        self._sweep_old_sessions()
        return self.session_id

    def start_round(
        self,
        *,
        round_id: str | None = None,
        dealer: str = "unknown",
        initial_hand_count: int | None = None,
        wildcard_enabled: bool = True,
        red_black_mode: str = "",
        min_xi: int = 9,
        rules: dict[str, Any] | None = None,
        result: str | None = None,
    ) -> str:
        if not self.enabled:
            return "disabled"
        if not self.session_active:
            raise RuntimeError("session not started")
        if round_id is None:
            self._round_seq += 1
            round_id = f"round_{self._round_seq:04d}"
        self.round_id = round_id

        rule_payload = rules or {}
        self._ensure(self.paths.round_dir(self.session_id, self.round_id))
        self._ensure(self.paths.round_screenshot_dir(self.session_id, self.round_id))
        self._ensure(self.paths.round_crops_root(self.session_id, self.round_id))
        self._ensure(self.paths.round_export_dir(self.session_id, self.round_id))

        self._round_paths[self.round_id] = {
            "events": 0,
            "decisions": 0,
            "actions": 0,
            "errors": 0,
            "states": 0,
            "total_safe_halts": 0,
        }

        round_meta = {
            "session_id": self.session_id,
            "round_id": self.round_id,
            "started_at": _utc_now(),
            "ended_at": None,
            "dealer": dealer,
            "initial_hand_count": initial_hand_count,
            "rules": rule_payload,
            "wildcard_enabled": wildcard_enabled,
            "red_black_mode": red_black_mode,
            "min_xi": min_xi,
            "result": result,
            "summary": {
                "total_frames": 0,
                "total_decisions": 0,
                "total_actions": 0,
                "total_safe_halts": 0,
            },
        }
        self._write_json(self.paths.round_meta_path(self.session_id, self.round_id), round_meta)
        self._append_jsonl(
            self.paths.round_events_path(self.session_id, self.round_id),
            self._event_payload(
                EventType.ROUND_START,
                Severity.INFO,
                {
                    "round_id": self.round_id,
                    "dealer": dealer,
                    "initial_hand_count": initial_hand_count,
                    "wildcard_enabled": wildcard_enabled,
                    "red_black_mode": red_black_mode,
                    "min_xi": min_xi,
                },
            ),
        )
        return self.round_id

    def end_round(self, result: str | None = None) -> None:
        if not self.round_active or not self.enabled:
            return
        round_meta_path = self.paths.round_meta_path(self.session_id, self.round_id)
        if round_meta_path.exists():
            data = json.loads(round_meta_path.read_text(encoding="utf-8"))
            data["ended_at"] = _utc_now()
            data["result"] = result
            if self.round_id in self._round_paths:
                summary = data.setdefault("summary", {})
                summary.update(self._round_paths[self.round_id])
            self._write_json(round_meta_path, data)
        self._append_jsonl(
            self.paths.round_events_path(self.session_id, self.round_id),
            self._event_payload(
                EventType.ROUND_END,
                Severity.INFO,
                {"round_id": self.round_id, "result": result},
            ),
        )
        self.round_id = None

    def end_session(self, notes: str | None = None) -> None:
        if not self.enabled or not self.session_active:
            return
        if self.round_active:
            self.end_round()
        session_meta = self.paths.session_meta_path(self.session_id)
        if session_meta.exists():
            payload = json.loads(session_meta.read_text(encoding="utf-8"))
            payload["ended_at"] = _utc_now()
            if notes is not None:
                payload["notes"] = notes
            self._write_json(session_meta, payload)
        self._append_jsonl(
            self.paths.session_events_path(self.session_id),
            self._event_payload(
                EventType.SESSION_END,
                Severity.INFO,
                {"message": "session ended", "notes": notes},
            ),
        )
        self._session_started = False

    def next_frame_id(self) -> str:
        self._frame_seq += 1
        return f"frame_{self._frame_seq:06d}"

    def next_decision_id(self) -> str:
        self._decision_seq += 1
        return f"decision_{self._decision_seq:06d}"

    def capture_frame_snapshot(
        self,
        source_screenshot: Path | str,
        frame_id: str,
        *,
        debug_overlay: Path | str | None = None,
        label: str = "raw",
        is_error: bool = False,
    ) -> str:
        if not self.enabled or not self.round_active:
            return str(source_screenshot)
        source = Path(source_screenshot)
        suffix = "raw" if label == "raw" else "debug"
        target_name = f"{frame_id}_{suffix}.png"
        target = self.paths.round_screenshot_dir(self.session_id, self.round_id) / target_name
        if source.is_file():
            self._ensure(target.parent)
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        elif label == "raw":
            raise FileNotFoundError(f"capture source not found: {source}")

        if debug_overlay is not None:
            debug_path = Path(debug_overlay)
            if debug_path.is_file():
                crop_target = self.paths.round_crops_root(self.session_id, self.round_id) / f"{frame_id}_debug.jpg"
                self._ensure(crop_target.parent)
                if debug_path.resolve() != crop_target.resolve():
                    shutil.copy2(debug_path, crop_target)
                return target.relative_to(self.paths.round_dir(self.session_id, self.round_id)).as_posix()
        return target.relative_to(self.paths.round_dir(self.session_id, self.round_id)).as_posix()

    def capture_action_snapshot(self, source_screenshot: Path | str, frame_id: str, label: str) -> str:
        if not self.enabled or not self.round_active:
            return str(source_screenshot)
        source = Path(source_screenshot)
        if not source.is_file():
            return str(source_screenshot)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "action"
        target_name = f"{frame_id}_{safe_label}.png"
        target = self.paths.round_screenshot_dir(self.session_id, self.round_id) / target_name
        self._ensure(target.parent)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target.relative_to(self.paths.round_dir(self.session_id, self.round_id)).as_posix()

    def log_frame_captured(
        self,
        frame_id: str,
        screenshot_path: Path | str,
        *,
        adb_device: str | None = None,
        capture_ok: bool = True,
        capture_error: str | None = None,
    ) -> str:
        relative_path = self.capture_frame_snapshot(
            screenshot_path,
            frame_id,
            label="raw",
            is_error=not capture_ok,
        ) if self.config.save_raw_screenshot or self.config.save_every_frame else str(screenshot_path)
        if self.config.save_raw_screenshot or self.config.save_every_frame:
            self.round_counters["events"] += 1
            self.log_event(
                EventType.FRAME_CAPTURED,
                {
                    "frame_id": frame_id,
                    "screenshot_path": str(relative_path),
                    "captured_at": _utc_now(),
                    "screen_size": list(self.screen_size or []),
                    "adb_device": adb_device or self.device_id,
                    "capture_ok": capture_ok,
                    "capture_error": capture_error,
                },
                frame_id=frame_id,
                severity=Severity.WARNING if not capture_ok else Severity.INFO,
            )
        return str(relative_path)

    def log_frame_recognized(
        self,
        frame_id: str,
        state: dict[str, Any],
        *,
        debug_image: Path | str | None = None,
    ) -> None:
        if not self._round_logging_enabled():
            return
        self.log_event(
            EventType.FRAME_RECOGNIZED,
            {
                "frame_id": frame_id,
                **state,
                "debug_image_path": str(debug_image) if debug_image else None,
            },
            frame_id=frame_id,
        )
        self.log_event(
            EventType.STATE_BUILT,
            {
                "frame_id": frame_id,
                "phase": str(state.get("phase")),
                "flow_state": state.get("flow_state") or state.get("phase"),
                "hand_count": len(state.get("hand", [])),
                "raw_hand": state.get("raw_hand", []),
                "normalized_hand": state.get("hand", []),
                "phase_confidence": state.get("vision_confidence", state.get("confidence", 0.0)),
                "raw_state": state,
            },
            frame_id=frame_id,
            severity=Severity.DEBUG,
        )
        self.log_state(frame_id=frame_id, state=state)

    def log_structure_allocation(self, frame_id: str, decision_id: str | None, allocation: dict[str, Any]) -> None:
        if not self._round_logging_enabled():
            return
        self.round_counters["events"] += 1
        self.log_event(
            EventType.STRUCTURE_ALLOCATED,
            {
                "frame_id": frame_id,
                "decision_id": decision_id,
                **allocation,
            },
            frame_id=frame_id,
            decision_id=decision_id,
            severity=Severity.INFO,
        )

    def log_hand_analysis(self, frame_id: str, decision_id: str | None, analysis: dict[str, Any]) -> None:
        if not self._round_logging_enabled():
            return
        self.log_event(
            EventType.HAND_ANALYZED,
            {
                "frame_id": frame_id,
                "decision_id": decision_id,
                **analysis,
            },
            frame_id=frame_id,
            decision_id=decision_id,
            severity=Severity.INFO,
        )

    def log_legal_actions(
        self,
        frame_id: str,
        decision_id: str | None,
        legal_actions: list[dict[str, Any]],
        rejected_actions: list[dict[str, Any]],
    ) -> None:
        if not self._round_logging_enabled():
            return
        self.round_counters["events"] += 1
        self.log_event(
            EventType.LEGAL_ACTIONS_GENERATED,
            {
                "frame_id": frame_id,
                "decision_id": decision_id,
                "legal_actions": legal_actions,
                "rejected_actions": rejected_actions,
            },
            frame_id=frame_id,
            decision_id=decision_id,
            severity=Severity.INFO,
        )

    def log_action_evaluated(
        self,
        frame_id: str,
        decision_id: str,
        eval_payload: dict[str, Any],
        *,
        is_rejected: bool = False,
    ) -> None:
        if not self._round_logging_enabled():
            return
        self.round_counters["events"] += 1
        self.log_event(
            EventType.ACTION_EVALUATED,
            {"frame_id": frame_id, "decision_id": decision_id, **eval_payload},
            frame_id=frame_id,
            decision_id=decision_id,
            severity=Severity.INFO if not is_rejected else Severity.WARNING,
        )

    def log_action_evaluation_started(
        self,
        frame_id: str,
        decision_id: str,
        *,
        candidate_count: int,
        source: str = "policy_brain",
    ) -> None:
        if not self._round_logging_enabled():
            return
        self.round_counters["events"] += 1
        self.log_event(
            EventType.ACTION_EVALUATION_STARTED,
            {
                "frame_id": frame_id,
                "decision_id": decision_id,
                "candidate_count": candidate_count,
                "source": source,
            },
            frame_id=frame_id,
            decision_id=decision_id,
            severity=Severity.INFO,
        )

    def save_crop(self, source: Path | str, frame_id: str, subdir: str, filename: str | None = None) -> str:
        if not self.enabled or not self.round_active:
            return str(source)
        source_path = Path(source)
        if not source_path.is_file():
            return str(source)
        filename = filename or source_path.name
        target = self.paths.round_crops_root(self.session_id, self.round_id) / subdir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return str(target.relative_to(self.paths.round_dir(self.session_id, self.round_id)))

    def log_event(
        self,
        event_type: EventType,
        data: dict[str, Any],
        *,
        severity: Severity = Severity.INFO,
        frame_id: str | None = None,
        decision_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        payload = self._event_payload(event_type, severity, data, frame_id=frame_id, decision_id=decision_id)
        if self.round_active:
            self.round_counters["events"] += 1
        if self.session_active:
            self._append_jsonl(self.paths.session_events_path(self.session_id), payload)
        if self.round_active:
            self._append_jsonl(self.paths.round_events_path(self.session_id, self.round_id), payload)
        return payload

    def log_state(self, frame_id: str, state: dict[str, Any]) -> None:
        if not self._round_logging_enabled():
            return
        if self._round_paths.setdefault(self.round_id, {}).get("states") is not None:
            self.round_counters["states"] += 1
            stats = self.round_counters
            round_meta = self.paths.round_meta_path(self.session_id, self.round_id)
            if round_meta.exists():
                data = json.loads(round_meta.read_text(encoding="utf-8"))
                data.setdefault("summary", {})["total_frames"] = stats["states"]
                self._write_json(round_meta, data)
        self._append_jsonl(
            self.paths.round_states_path(self.session_id, self.round_id),
            {
                "timestamp": _utc_now(),
                "session_id": self.session_id,
                "round_id": self.round_id,
                "frame_id": frame_id,
                "state": state,
            },
        )

    def log_decision(self, decision_id: str, frame_id: str, decision: dict[str, Any], reason: str | None = None) -> None:
        if not self._round_logging_enabled():
            return
        self.round_counters["decisions"] += 1
        self._append_jsonl(
            self.paths.round_decisions_path(self.session_id, self.round_id),
            {
                "timestamp": _utc_now(),
                "session_id": self.session_id,
                "round_id": self.round_id,
                "frame_id": frame_id,
                "decision_id": decision_id,
                "reason": reason,
                "decision": decision,
            },
        )
        self.log_event(
            EventType.DECISION_SELECTED,
            {
                "decision_id": decision_id,
                "selected_action": decision.get("selected_action")
                or {
                    "type": str(decision.get("action", "")).upper(),
                    "card_id": decision.get("selected_card_id"),
                    "label": decision.get("label") or decision.get("selected_label"),
                    "ev": decision.get("ev") or decision.get("score"),
                },
                "selected_reason": decision.get("selected_reason") or decision.get("reason"),
                "reason": reason,
                "candidate_stage": decision.get("candidate_stage"),
                "all_action_evals_count": len(decision.get("action_evals", decision.get("evaluations", []))),
                "best_alternatives": decision.get("best_alternatives", []),
                "policy_version": decision.get("policy_version"),
            },
            severity=Severity.INFO,
            frame_id=frame_id,
            decision_id=decision_id,
        )

    def log_action_plan(self, decision_id: str, frame_id: str, plan: dict[str, Any]) -> None:
        if not self._round_logging_enabled():
            return
        planned_clicks = _clicks_from_plan(plan, plan.get("tap_x"), plan.get("tap_y"))
        self.round_counters["actions"] += 1
        self._append_jsonl(
            self.paths.round_actions_path(self.session_id, self.round_id),
            {
                "timestamp": _utc_now(),
                "session_id": self.session_id,
                "round_id": self.round_id,
                "frame_id": frame_id,
                "decision_id": decision_id,
                "action_plan": plan,
            },
        )
        self.log_event(
            EventType.ACTION_PLAN_CREATED,
            {
                "decision_id": decision_id,
                "ready": plan.get("ready"),
                "reason": plan.get("reason"),
                "policy_selected_action": plan.get("policy_selected_action"),
                "action_plan": {
                    "policy_selected_action": plan.get("policy_selected_action"),
                    "target_type": plan.get("target_type"),
                    "target_card_id": plan.get("target_card_id"),
                    "target_label": plan.get("target_label"),
                    "reason": plan.get("reason"),
                    "tap_x": plan.get("tap_x"),
                    "tap_y": plan.get("tap_y"),
                    "click_count": len(planned_clicks),
                    "clicks": planned_clicks,
                },
                "policy_action_plan_match": plan.get("validation", {}).get("target_matches_policy"),
                "validation": plan.get("validation", {}),
            },
            severity=Severity.INFO if plan.get("ready") else Severity.WARNING,
            frame_id=frame_id,
            decision_id=decision_id,
        )
        self.log_event(
            EventType.ACTION_PLAN_VALIDATED,
            {
                "decision_id": decision_id,
                "passed": plan.get("validation", {}).get("passed"),
                "reason_code": plan.get("validation", {}).get("reason_code"),
                "checks": plan.get("validation", {}).get("checks", []),
                "target_matches_policy": plan.get("validation", {}).get("target_matches_policy"),
                "guard_result": plan.get("validation", {}).get("guard_result"),
            },
            severity=Severity.INFO if plan.get("validation", {}).get("passed") else Severity.WARNING,
            frame_id=frame_id,
            decision_id=decision_id,
        )

    def log_safe_halt(
        self,
        frame_id: str,
        reason: str,
        decision_id: str | None = None,
        raw_hand: list[str] | None = None,
        hard_protected: list[str] | None = None,
        free_cards: list[str] | None = None,
        clickable_cards: list[str] | None = None,
        screenshots: list[str] | None = None,
        screenshot_path: str | None = None,
        debug_image_path: str | None = None,
    ) -> None:
        screenshot_list = screenshots or []
        primary_screenshot = screenshot_path or (screenshot_list[0] if screenshot_list else None)
        payload = {
            "frame_id": frame_id,
            "halt_reason": reason,
            "raw_hand": raw_hand or [],
            "hard_protected": hard_protected or [],
            "free_cards": free_cards or [],
            "clickable_cards": clickable_cards or [],
            "screenshot_path": primary_screenshot,
            "screenshots": screenshot_list,
            "debug_image_path": debug_image_path,
        }
        if reason not in SAFE_HALT_REASONS:
            payload["halt_reason"] = "unknown_error"
            payload["raw_reason"] = reason
        if self.round_active:
            self._round_paths[self.round_id]["total_safe_halts"] += 1
        self.round_counters["events"] += 1
        self.log_event(
            EventType.SAFE_HALT,
            {
                "frame_id": frame_id,
                "decision_id": decision_id,
                **payload,
                "explanation": payload.get("halt_reason"),
                "recommended_fix": "检查当前手牌识别、按钮状态或执行后画面",
            },
            frame_id=frame_id,
            decision_id=decision_id,
            severity=Severity.WARNING,
        )
        self.log_error(
            frame_id=frame_id,
            message="SAFE_HALT",
            context=payload,
            severity=Severity.WARNING,
            decision_id=decision_id,
        )

    def log_tap(
        self,
        frame_id: str,
        decision_id: str | None,
        plan: dict[str, Any],
        *,
        execute_enabled: bool,
        dry_run: bool,
        tap_executed: bool,
        tap_x: int | None,
        tap_y: int | None,
        target_label: str | None = None,
        target_card_id: str | None = None,
        adb_result: str = "ok",
        before_screenshot: str | None = None,
        after_screenshot: str | None = None,
        executed_clicks: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._round_logging_enabled():
            return
        if self.config.save_before_after_action:
            if before_screenshot:
                before_screenshot = self.capture_action_snapshot(before_screenshot, frame_id, "before_action")
            if after_screenshot:
                after_screenshot = self.capture_action_snapshot(after_screenshot, frame_id, "after_action")
        executed_at = _utc_now() if tap_executed else None
        reason = "dry_run" if dry_run and not tap_executed else None
        planned_clicks = _clicks_from_plan(plan, tap_x, tap_y)
        logged_executed_clicks = _normalise_clicks(executed_clicks) if executed_clicks is not None else planned_clicks
        if not tap_executed:
            logged_executed_clicks = []
        self.round_counters["events"] += 1
        self.log_event(
            EventType.TAP_EXECUTED,
            {
                "frame_id": frame_id,
                "decision_id": decision_id,
                "execute_enabled": execute_enabled,
                "dry_run": dry_run,
                "tap_executed": tap_executed,
                "tap_x": tap_x,
                "tap_y": tap_y,
                "would_tap_x": tap_x if dry_run and not tap_executed else None,
                "would_tap_y": tap_y if dry_run and not tap_executed else None,
                "target_label": target_label,
                "target_card_id": target_card_id,
                "plan_target": plan.get("target"),
                "action_plan": plan,
                "click_count": len(planned_clicks),
                "planned_click_count": len(planned_clicks),
                "planned_clicks": planned_clicks,
                "executed_click_count": len(logged_executed_clicks),
                "executed_clicks": logged_executed_clicks,
                "would_clicks": planned_clicks if dry_run and not tap_executed else [],
                "adb_result": adb_result,
                "before_screenshot": before_screenshot,
                "after_screenshot": after_screenshot,
                "executed_at": executed_at,
                "reason": reason,
            },
            frame_id=frame_id,
            decision_id=decision_id,
        )
        if after_screenshot:
            self.log_event(
                EventType.SCREEN_AFTER_ACTION,
                {
                    "frame_id": frame_id,
                    "decision_id": decision_id,
                    "after_screenshot": after_screenshot,
                },
                frame_id=frame_id,
                decision_id=decision_id,
                severity=Severity.DEBUG,
            )

    def log_error(
        self,
        *,
        frame_id: str | None,
        message: str,
        context: dict[str, Any] | None = None,
        severity: Severity = Severity.ERROR,
        decision_id: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        context = context or {}
        if self.round_active:
            self.round_counters["errors"] += 1
            self._append_jsonl(
                self.paths.round_errors_path(self.session_id, self.round_id),
                {
                    "timestamp": _utc_now(),
                    "session_id": self.session_id,
                    "round_id": self.round_id,
                    "frame_id": frame_id,
                    "decision_id": decision_id,
                    "message": message,
                    **context,
                },
            )
        self.log_event(
            EventType.ERROR,
            {"message": message, **context},
            severity=severity,
            frame_id=frame_id,
            decision_id=decision_id,
        )

    def _round_logging_enabled(self) -> bool:
        return bool(self.enabled and self.round_active)

    def _event_payload(
        self,
        event_type: EventType,
        severity: Severity,
        data: dict[str, Any],
        *,
        frame_id: str | None = None,
        decision_id: str | None = None,
    ) -> dict[str, Any]:
        return LogEvent(
            timestamp=_utc_now(),
            session_id=self.session_id or "disabled",
            event_type=event_type.value,
            severity=severity.value,
            data=data,
            round_id=self.round_id,
            frame_id=frame_id,
            decision_id=decision_id,
        ).to_dict()

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            self._ensure(path.parent)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(to_jsonl_line(payload) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:
            self._log_write_failures += 1
            self._append_emergency_jsonl(path, payload, exc)

    def _append_emergency_jsonl(self, path: Path, payload: dict[str, Any], exc: Exception) -> None:
        fallback_dir = self.base_dir / "_logger_failures"
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            fallback_dir = Path(tempfile.gettempdir()) / "chenzhou_zipai_ai_logger_failures"
            fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / "logger_failures.jsonl"
        emergency_payload = {
            "timestamp": _utc_now(),
            "target_path": str(path),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "payload": payload,
        }
        with open(fallback_path, "a", encoding="utf-8") as handle:
            handle.write(to_jsonl_line(emergency_payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _ensure(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _sweep_old_sessions(self) -> None:
        keep = self.config.max_sessions_keep
        if not keep:
            return
        sessions_root = self.paths.sessions_root()
        if not sessions_root.exists():
            return
        entries = [path for path in sessions_root.iterdir() if path.is_dir() and re.match(r"^session_", path.name)]
        if len(entries) <= keep:
            return
        sorted_entries = sorted(entries, key=lambda item: item.stat().st_mtime)
        for entry in sorted_entries[: max(0, len(sorted_entries) - keep)]:
            shutil.rmtree(entry, ignore_errors=True)
