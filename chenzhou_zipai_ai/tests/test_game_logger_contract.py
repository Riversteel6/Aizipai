"""Logging/export/replay contract tests from the implementation document."""

import errno
import json
import os
import shutil
import zipfile
from pathlib import Path

from ai.pro_brain import allocate_hand_structures, analyze_hand, build_decision_context, choose_action
import chenzhou_zipai_ai.game_logging.game_logger as game_logger_module
from chenzhou_zipai_ai.game_logging import (
    EventType,
    GameLogger,
    LoggerConfig,
    replay_loader,
)
from chenzhou_zipai_ai.game_logging.export_bundle import export_round_bundle
from chenzhou_zipai_ai.game_logging.validate_logs import validate_round_bundle
from control.action_plan import build_action_plan
from tools.export_latest_round import main as export_latest_round_main
from tools.print_round_as_markdown import print_round_as_markdown
from tools.replay_decision import _replay_one_decision
from tools.replay_policy_fixtures import run_fixture_replay
from tools.verify_real_replays import verify_replay_logs


def _read_events(round_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (round_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample_state(screenshot_path: str) -> dict:
    return {
        "screenshot": screenshot_path,
        "screenshot_path": screenshot_path,
        "raw_hand": ["一", "二", "二", "二", "三"],
        "hand": ["一", "二", "二", "二", "三"],
        "normalized_hand": ["一", "二", "二", "二", "三"],
        "phase": "play",
        "buttons": [],
        "legal_actions": [{"type": "discard"}],
        "hand_details": [
            {"card_id": "h001", "name": "一", "x": 0, "y": 0, "w": 10, "h": 10, "confidence": 0.99, "clickable": True},
            {"card_id": "h002", "name": "二", "x": 10, "y": 0, "w": 10, "h": 10, "confidence": 0.99, "clickable": True},
            {"card_id": "h003", "name": "二", "x": 20, "y": 0, "w": 10, "h": 10, "confidence": 0.99, "clickable": True},
            {"card_id": "h004", "name": "二", "x": 30, "y": 0, "w": 10, "h": 10, "confidence": 0.99, "clickable": True},
            {"card_id": "h005", "name": "三", "x": 40, "y": 0, "w": 10, "h": 10, "confidence": 0.99, "clickable": True},
        ],
    }


def _write_complete_round(tmp_path, *, execute_tap: bool = False, after_screenshot: str | None = None):
    source = tmp_path / "source.png"
    source.write_text("fake image placeholder", encoding="utf-8")
    logger = GameLogger(
        tmp_path / "logs",
        config=LoggerConfig(save_raw_screenshot=True, save_debug_screenshot=False),
    )
    session_id = logger.start_session(device_id="dev", screen_size=(1080, 2344))
    round_id = logger.start_round()
    frame_id = logger.next_frame_id()
    rel_screenshot = logger.log_frame_captured(frame_id, source)
    state = _sample_state(rel_screenshot)
    logger.log_frame_recognized(frame_id, state)

    decision_id = logger.next_decision_id()
    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)
    decision = choose_action(state)
    logger.log_structure_allocation(frame_id, decision_id, allocation.to_dict())
    logger.log_hand_analysis(frame_id, decision_id, analysis.to_dict())
    legal_actions = decision.context_snapshot["legal_actions"]
    rejected_actions = decision.context_snapshot["rejected_actions"]
    logger.log_legal_actions(frame_id, decision_id, legal_actions, rejected_actions)
    logger.log_action_evaluation_started(
        frame_id,
        decision_id,
        candidate_count=len(decision.action_evals),
        source="test",
    )
    for item in decision.action_evals:
        logger.log_action_evaluated(frame_id, decision_id, item.to_dict(), is_rejected=not item.allowed)
    plan = {
        "action": "discard",
        "ready": True,
        "reason": "target card found and clickable",
        "target_type": "hand_card",
        "target_card_id": decision.selected_card_id,
        "target_label": decision.selected_label,
        "policy_selected_action": {
            "type": "discard",
            "label": decision.selected_label,
        },
        "validation": {
            "passed": True,
            "target_matches_policy": True,
            "checks": ["target_matches_policy", "target_clickable", "not_hard_protected"],
        },
    }
    logger.log_decision(decision_id, frame_id, decision.to_dict(), reason=decision.reason)
    logger.log_action_plan(decision_id, frame_id, plan)
    logger.log_tap(
        frame_id,
        decision_id,
        plan,
        execute_enabled=execute_tap,
        dry_run=not execute_tap,
        tap_executed=execute_tap,
        tap_x=5,
        tap_y=5,
        target_label=decision.selected_label,
        target_card_id=decision.selected_card_id,
        adb_result="ok" if execute_tap else "dry_run",
        before_screenshot=rel_screenshot,
        after_screenshot=after_screenshot if execute_tap else None,
    )
    logger.end_round(result="test")
    return logger.paths.round_dir(session_id, round_id)


def test_game_logger_round_validates_exports_and_markdown(tmp_path):
    round_path = _write_complete_round(tmp_path)

    result = validate_round_bundle(round_path)
    assert result["ok"], result["errors"]

    bundle = export_round_bundle(round_path)
    assert bundle.round_bundle_path.exists()
    with zipfile.ZipFile(bundle.round_bundle_path) as archive:
        names = set(archive.namelist())
        assert "round_summary.md" in names
        assert "round_replay.json" in names
        assert "events.jsonl" in names
        assert "bundle_manifest.json" in names
        assert "version_info.json" in names
        assert "crops/unknown/manifest.json" in names
        manifest = json.loads(archive.read("bundle_manifest.json").decode("utf-8"))
        assert manifest["unknown_crops"]["count"] == 0
        assert any(path.endswith("rules.yaml") for path in manifest["config_snapshots"])

    markdown = print_round_as_markdown(round_path)
    content = markdown.read_text(encoding="utf-8")
    assert "Candidates" in content
    assert "SAFE_HALT" not in content or "## SAFE_HALT" in content

    events = [
        json.loads(line)
        for line in (round_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["event_type"] == EventType.STRUCTURE_ALLOCATED.value for event in events)
    assert any(event["event_type"] == EventType.ACTION_EVALUATION_STARTED.value for event in events)
    assert any(event["event_type"] == EventType.ACTION_EVALUATED.value for event in events)
    assert any(event["event_type"] == EventType.ACTION_PLAN_VALIDATED.value for event in events)
    tap = next(event for event in events if event["event_type"] == EventType.TAP_EXECUTED.value)
    assert tap["data"]["reason"] == "dry_run"


def test_export_latest_round_cli_prints_review_and_bundle(monkeypatch, capsys, tmp_path):
    round_path = _write_complete_round(tmp_path)
    session_dir = round_path.parents[1]

    assert replay_loader.latest_round_bundle(tmp_path / "logs")[1] == round_path
    assert replay_loader.latest_round_bundle(session_dir)[1] == round_path
    assert replay_loader.latest_round_bundle(round_path)[1] == round_path

    monkeypatch.setattr("sys.argv", ["export_latest_round", "--logs-root", str(tmp_path / "logs")])
    export_latest_round_main()
    output = capsys.readouterr().out.splitlines()
    paths = {
        key: Path(value)
        for line in output
        if "=" in line
        for key, value in [line.split("=", 1)]
        if key in {"ROUND_REVIEW_MD", "ROUND_BUNDLE_ZIP"}
    }

    assert paths["ROUND_REVIEW_MD"].exists()
    assert paths["ROUND_REVIEW_MD"].name == "round_for_review.md"
    assert paths["ROUND_BUNDLE_ZIP"].exists()
    assert paths["ROUND_BUNDLE_ZIP"].suffix == ".zip"


def test_safe_halt_logs_screenshot_path(tmp_path):
    logger = GameLogger(tmp_path / "logs")
    session_id = logger.start_session(device_id="dev")
    round_id = logger.start_round()
    frame_id = logger.next_frame_id()

    logger.log_safe_halt(
        frame_id=frame_id,
        decision_id=None,
        reason="recognition_uncertain",
        raw_hand=["一", "二"],
        screenshot_path="screenshots/frame_000001_raw.png",
    )

    round_path = logger.paths.round_dir(session_id, round_id)
    result = validate_round_bundle(round_path)
    events = [
        json.loads(line)
        for line in (round_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    halt = next(event for event in events if event["event_type"] == EventType.SAFE_HALT.value)

    assert result["ok"], result["errors"]
    assert halt["data"]["screenshot_path"] == "screenshots/frame_000001_raw.png"


def test_safe_halt_preserves_conflict_guard_reason(tmp_path):
    logger = GameLogger(tmp_path / "logs")
    logger.start_session(device_id="dev")
    logger.start_round()
    frame_id = logger.next_frame_id()

    logger.log_safe_halt(
        frame_id=frame_id,
        reason="conflict_state_mutation",
        screenshot_path="screenshots/frame_000001_raw.png",
    )

    events_path = logger.paths.round_events_path(logger.session_id, logger.round_id)
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    halt = next(event for event in events if event["event_type"] == EventType.SAFE_HALT.value)

    assert halt["data"]["halt_reason"] == "conflict_state_mutation"
    assert "raw_reason" not in halt["data"]


def test_expand_chi_options_round_validates_as_probe_not_chi(tmp_path):
    source = tmp_path / "source.png"
    source.write_text("fake image placeholder", encoding="utf-8")
    logger = GameLogger(tmp_path / "logs", config=LoggerConfig(save_raw_screenshot=True))
    session_id = logger.start_session(device_id="dev", screen_size=(1080, 2344))
    round_id = logger.start_round()
    frame_id = logger.next_frame_id()
    rel_screenshot = logger.log_frame_captured(frame_id, source)
    state = {
        "screenshot": rel_screenshot,
        "screenshot_path": rel_screenshot,
        "raw_hand": ["二", "十", "四", "六", "九"],
        "hand": ["二", "十", "四", "六", "九"],
        "normalized_hand": ["二", "十", "四", "六", "九"],
        "phase": "play",
        "pending_card": "七",
        "buttons": [
            {"name": "chi", "x": 100, "y": 200, "w": 80, "h": 40},
            {"name": "pass", "x": 200, "y": 200, "w": 80, "h": 40},
        ],
        "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
        "chi_options": [],
        "option_details": [],
        "hand_details": [
            {"card_id": f"h{index:03d}", "name": label, "x": index * 10, "y": 0, "w": 10, "h": 10, "confidence": 0.99, "clickable": True}
            for index, label in enumerate(["二", "十", "四", "六", "九"], start=1)
        ],
    }
    logger.log_frame_recognized(frame_id, state)
    decision_id = logger.next_decision_id()
    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)
    decision = choose_action(state)
    plan = build_action_plan(state, decision.to_dict())

    assert decision.selected_action == "EXPAND_CHI_OPTIONS"
    assert plan["action"] == "expand_chi_options"

    logger.log_structure_allocation(frame_id, decision_id, allocation.to_dict())
    logger.log_hand_analysis(frame_id, decision_id, analysis.to_dict())
    logger.log_legal_actions(
        frame_id,
        decision_id,
        decision.context_snapshot["legal_actions"],
        decision.context_snapshot["rejected_actions"],
    )
    logger.log_action_evaluation_started(frame_id, decision_id, candidate_count=len(decision.action_evals), source="test")
    for item in decision.action_evals:
        logger.log_action_evaluated(frame_id, decision_id, item.to_dict(), is_rejected=not item.allowed)
    logger.log_decision(decision_id, frame_id, decision.to_dict(), reason=decision.reason)
    logger.log_action_plan(decision_id, frame_id, plan)
    logger.log_tap(
        frame_id,
        decision_id,
        plan,
        execute_enabled=False,
        dry_run=True,
        tap_executed=False,
        tap_x=None,
        tap_y=None,
        before_screenshot=rel_screenshot,
    )

    round_path = logger.paths.round_dir(session_id, round_id)
    result = validate_round_bundle(round_path)

    assert result["ok"], result["errors"]


def test_executed_tap_logs_before_after_and_screen_after_action(tmp_path):
    after = "screenshots/frame_000124_raw.png"
    round_path = _write_complete_round(tmp_path, execute_tap=True, after_screenshot=after)
    result = validate_round_bundle(round_path)
    events = _read_events(round_path)
    tap = next(event for event in events if event["event_type"] == EventType.TAP_EXECUTED.value)
    screen_after = next(event for event in events if event["event_type"] == EventType.SCREEN_AFTER_ACTION.value)

    assert result["ok"], result["errors"]
    assert tap["data"]["before_screenshot"]
    assert tap["data"]["after_screenshot"] == after
    assert tap["data"]["executed_at"]
    assert screen_after["data"]["after_screenshot"] == after
    assert screen_after["decision_id"] == tap["decision_id"]


def test_multi_click_tap_logs_full_click_sequence_and_validates(tmp_path):
    logger = GameLogger(tmp_path / "logs")
    session_id = logger.start_session(device_id="dev", execute_enabled=True)
    round_id = logger.start_round()
    frame_id = logger.next_frame_id()
    plan = {
        "action": "chi",
        "ready": True,
        "reason": "selected chi option with compare follow-up",
        "clicks": [
            {"target": "chi", "x": 100, "y": 800, "delay_ms": 80},
            {"target": "chi_option", "x": 320, "y": 910, "delay_ms": 120, "option_id": "chi_二二贰"},
            {"target": "compare_option", "x": 500, "y": 1080, "delay_ms": 80, "option_id": "compare_一二三"},
        ],
        "validation": {"passed": True, "checks": ["target_matches_policy", "target_clickable"]},
    }

    logger.log_tap(
        frame_id,
        None,
        plan,
        execute_enabled=True,
        dry_run=False,
        tap_executed=True,
        tap_x=100,
        tap_y=800,
        target_label="二",
        adb_result="ok",
        before_screenshot="before.png",
        after_screenshot="after.png",
    )
    logger.end_round(result="test")

    round_path = logger.paths.round_dir(session_id, round_id)
    result = validate_round_bundle(round_path)
    events = _read_events(round_path)
    tap = next(event for event in events if event["event_type"] == EventType.TAP_EXECUTED.value)

    assert result["ok"], result["errors"]
    assert tap["data"]["click_count"] == 3
    assert tap["data"]["planned_click_count"] == 3
    assert tap["data"]["executed_click_count"] == 3
    assert tap["data"]["executed_clicks"] == tap["data"]["planned_clicks"]
    assert [click["target"] for click in tap["data"]["executed_clicks"]] == ["chi", "chi_option", "compare_option"]
    assert tap["data"]["action_plan"]["clicks"][2]["option_id"] == "compare_一二三"


def test_validate_rejects_multiclick_tap_without_executed_clicks(tmp_path):
    logger = GameLogger(tmp_path / "logs")
    session_id = logger.start_session(device_id="dev", execute_enabled=True)
    round_id = logger.start_round()
    frame_id = logger.next_frame_id()
    planned_clicks = [
        {"target": "chi", "x": 100, "y": 800},
        {"target": "chi_option", "x": 320, "y": 910},
    ]

    logger.log_event(
        EventType.TAP_EXECUTED,
        {
            "frame_id": frame_id,
            "execute_enabled": True,
            "dry_run": False,
            "tap_executed": True,
            "tap_x": 100,
            "tap_y": 800,
            "click_count": 2,
            "planned_click_count": 2,
            "planned_clicks": planned_clicks,
            "executed_click_count": 0,
            "adb_result": "ok",
            "before_screenshot": "before.png",
            "after_screenshot": "after.png",
            "executed_at": "2026-05-31T00:00:00+00:00",
        },
        frame_id=frame_id,
    )
    logger.log_event(
        EventType.SCREEN_AFTER_ACTION,
        {"frame_id": frame_id, "after_screenshot": "after.png"},
        frame_id=frame_id,
    )
    logger.end_round(result="test")

    round_path = logger.paths.round_dir(session_id, round_id)
    result = validate_round_bundle(round_path)

    assert not result["ok"]
    assert any("executed_clicks" in error for error in result["errors"])


def test_executed_tap_copies_before_after_screenshots_into_round(tmp_path):
    before_source = tmp_path / "before.png"
    after_source = tmp_path / "after.png"
    before_source.write_bytes(b"before")
    after_source.write_bytes(b"after")
    logger = GameLogger(
        tmp_path / "logs",
        config=LoggerConfig(save_before_after_action=True),
    )
    session_id = logger.start_session(device_id="dev", execute_enabled=True)
    round_id = logger.start_round()
    frame_id = logger.next_frame_id()

    logger.log_tap(
        frame_id,
        None,
        {"target": "hand_card"},
        execute_enabled=True,
        dry_run=False,
        tap_executed=True,
        tap_x=153,
        tap_y=1938,
        target_label="一",
        target_card_id="h001",
        adb_result="ok",
        before_screenshot=str(before_source),
        after_screenshot=str(after_source),
    )
    logger.end_round(result="test")

    round_path = logger.paths.round_dir(session_id, round_id)
    result = validate_round_bundle(round_path)
    events = _read_events(round_path)
    tap = next(event for event in events if event["event_type"] == EventType.TAP_EXECUTED.value)
    screen_after = next(event for event in events if event["event_type"] == EventType.SCREEN_AFTER_ACTION.value)

    assert result["ok"], result["errors"]
    assert tap["data"]["before_screenshot"] == f"screenshots/{frame_id}_before_action.png"
    assert tap["data"]["after_screenshot"] == f"screenshots/{frame_id}_after_action.png"
    assert screen_after["data"]["after_screenshot"] == tap["data"]["after_screenshot"]
    assert (round_path / tap["data"]["before_screenshot"]).exists()
    assert (round_path / tap["data"]["after_screenshot"]).exists()


def test_validate_rejects_executed_tap_without_after_screenshot(tmp_path):
    round_path = _write_complete_round(tmp_path, execute_tap=True, after_screenshot=None)
    result = validate_round_bundle(round_path)

    assert not result["ok"]
    assert any("after_screenshot" in error for error in result["errors"])


def test_logger_spools_jsonl_payload_when_primary_write_fails(monkeypatch, tmp_path):
    logger = GameLogger(tmp_path / "logs")
    logger.start_session(device_id="dev")
    original_open = Path.open

    def flaky_open(path, *args, **kwargs):
        if path.name == "session_events.jsonl":
            raise OSError("simulated write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)

    logger.log_event(EventType.MANUAL_NOTE, {"note": "must survive"})

    fallback = tmp_path / "logs" / "_logger_failures" / "logger_failures.jsonl"
    assert fallback.exists()
    rows = [json.loads(line) for line in fallback.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[-1]["exception_type"] == "OSError"
    assert rows[-1]["payload"]["data"]["note"] == "must survive"
    assert logger._log_write_failures == 1


def test_logger_spools_to_temp_when_local_failure_dir_is_unavailable(monkeypatch, tmp_path):
    logger = GameLogger(tmp_path / "logs")
    logger.start_session(device_id="dev")
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    original_open = Path.open
    original_mkdir = Path.mkdir

    def flaky_open(path, *args, **kwargs):
        if path.name == "session_events.jsonl":
            raise OSError("simulated primary write failure")
        return original_open(path, *args, **kwargs)

    def flaky_mkdir(path, *args, **kwargs):
        if path.name == "_logger_failures":
            raise PermissionError("simulated local fallback permission failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)
    monkeypatch.setattr(game_logger_module.tempfile, "gettempdir", lambda: str(temp_root))

    logger.log_event(EventType.MANUAL_NOTE, {"note": "survive in temp"})

    fallback = temp_root / "chenzhou_zipai_ai_logger_failures" / "logger_failures.jsonl"
    assert fallback.exists()
    rows = [json.loads(line) for line in fallback.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[-1]["exception_type"] == "OSError"
    assert rows[-1]["payload"]["data"]["note"] == "survive in temp"
    assert logger._log_write_failures == 1


def test_logger_spools_jsonl_payload_when_fsync_reports_disk_full(monkeypatch, tmp_path):
    logger = GameLogger(tmp_path / "logs")
    logger.start_session(device_id="dev")
    original_fsync = game_logger_module.os.fsync
    calls = {"count": 0}

    def flaky_fsync(fd):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return original_fsync(fd)

    monkeypatch.setattr(game_logger_module.os, "fsync", flaky_fsync)

    logger.log_event(EventType.MANUAL_NOTE, {"note": "survive disk full"})

    fallback = tmp_path / "logs" / "_logger_failures" / "logger_failures.jsonl"
    assert fallback.exists()
    rows = [json.loads(line) for line in fallback.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[-1]["exception_type"] == "OSError"
    assert "No space left on device" in rows[-1]["exception"]
    assert rows[-1]["payload"]["data"]["note"] == "survive disk full"
    assert logger._log_write_failures == 1


def test_replay_policy_fixtures_cover_twenty_typical_logs(tmp_path):
    result = run_fixture_replay(tmp_path / "fixture_logs")

    assert result["ok"], result
    assert result["replayed"] >= 20
    assert result["matched"] == result["replayed"]

    verification = verify_replay_logs(tmp_path / "fixture_logs", min_decisions=20, include_fixtures=True)
    assert verification["ok"], verification
    assert verification["decisions_checked"] >= 20


def test_verify_real_replays_stops_after_newest_valid_round(tmp_path):
    logs_root = tmp_path / "logs"
    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    bad_round = _write_complete_round(bad_root, execute_tap=True, after_screenshot=None)
    target_session = logs_root / "sessions" / "session_20000101_000000_old_bad"
    target_round = target_session / "rounds" / bad_round.name
    shutil.copytree(bad_round, target_round)
    old_timestamp = 1_700_000_000
    for path in (target_round, target_round.parent, target_session):
        os.utime(path, (old_timestamp, old_timestamp))

    result = run_fixture_replay(logs_root)
    assert result["ok"], result

    verification = verify_replay_logs(logs_root, min_decisions=20, include_fixtures=True)
    assert verification["ok"], verification
    assert verification["errors"] == []


def test_replay_decision_accepts_real_state_without_legacy_hand(tmp_path):
    logger = GameLogger(tmp_path / "logs")
    session_id = logger.start_session(device_id="dev")
    round_id = logger.start_round()
    frame_id = logger.next_frame_id()
    state_for_decision = {
        "raw_hand": ["一", "二", "三", "四", "五"],
        "normalized_hand": ["一", "二", "三", "四", "五"],
        "hand_details": [
            {"card_id": f"h{index:03d}", "name": label, "confidence": 0.99, "clickable": True}
            for index, label in enumerate(["一", "二", "三", "四", "五"], start=1)
        ],
        "phase": "play",
        "buttons": [],
        "recognition_warnings": ["low_confidence_card:h001:一:0.50"],
        "remaining_cards_estimate": 34,
    }
    logged_state = dict(state_for_decision)
    logged_state.pop("recognition_warnings")
    logger.log_frame_recognized(frame_id, logged_state)
    decision_id = logger.next_decision_id()
    decision = choose_action(state_for_decision)

    assert decision.action == "safe_halt"

    logger.log_decision(decision_id, frame_id, decision.to_dict(), reason=decision.reason)
    logger.end_round(result="test")

    round_path = logger.paths.round_dir(session_id, round_id)
    bundle = replay_loader.load_round_bundle_from_path(round_path)
    replayed = _replay_one_decision(bundle, decision_id)

    assert replayed["replayed"]["action"] == "safe_halt"
    assert replayed["same_action"]
    assert replayed["same_label"]
