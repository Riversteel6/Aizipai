"""Build a per-round zip package for review and replay."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .replay_loader import ReplayBundle, load_round_bundle


@dataclass(frozen=True)
class RoundExport:
    round_bundle_path: Path
    session_id: str
    round_id: str


def _write_round_summary(bundle: ReplayBundle, output: Path) -> None:
    initial_hand = ""
    if bundle.states:
        first_state = bundle.states[0].get("state", {})
        hand = first_state.get("normalized_hand") or first_state.get("hand") or []
        if isinstance(hand, list):
            initial_hand = " ".join(str(item) for item in hand)
    safe_halts = [row for row in bundle.events if row.get("event_type") == "SAFE_HALT"]
    lines = [
        f"# Round {bundle.round_id} Summary",
        "",
        "## 基本信息",
        "",
        f"- session_id: {bundle.session_id}",
        f"- started_at: {bundle.round_meta.get('started_at')}",
        f"- ended_at: {bundle.round_meta.get('ended_at')}",
        f"- result: {bundle.round_meta.get('result')}",
        "",
        "## 回合统计",
        "",
        f"- total_frames: {len(bundle.states)}",
        f"- total_decisions: {len(bundle.decisions)}",
        f"- total_actions: {len(bundle.actions)}",
        f"- total_safe_halts: {bundle.round_meta.get('summary', {}).get('total_safe_halts', 0)}",
        "",
        "## 起手牌",
        "",
        initial_hand or "(unknown)",
        "",
        "## 每手决策摘要",
        "",
        "| turn | phase | hand | action | reason | screenshot |",
        "|---|---|---|---|---|---|",
    ]
    for index, item in enumerate(bundle.decisions, 1):
        decision = item.get("decision", {})
        hand = " ".join(decision.get("hand", [])) if isinstance(decision.get("hand"), list) else ""
        action = decision.get("action", "")
        reason = (item.get("reason") or decision.get("reason") or "")
        frame = item.get("frame_id", "")
        lines.append(f"| {index} | {decision.get('candidate_stage', '')} | {hand} | {action} | {reason} | {frame} |")
    lines.extend(["", "## 关键决策", ""])
    for row in bundle.events:
        if row.get("event_type") in {"ACTION_EVALUATED", "SAFE_HALT", "ACTION_PLAN_CREATED"}:
            data = row.get("data", {})
            reason = data.get("reason") or data.get("halt_reason") or data.get("reject_reason") or ""
            lines.append(f"- {row.get('event_type')} {row.get('decision_id') or ''}: {reason}")
    lines.extend(["", "## 错误/警告", ""])
    if safe_halts:
        for row in safe_halts:
            data = row.get("data", {})
            lines.append(f"- SAFE_HALT {row.get('frame_id')}: {data.get('halt_reason')}")
    elif bundle.errors:
        for row in bundle.errors:
            lines.append(f"- {row.get('message')}")
    else:
        lines.append("(none)")
    lines.extend(
        [
            "",
            "## 可复盘文件",
            "",
            "- decisions.jsonl",
            "- states.jsonl",
            "- events.jsonl",
            "- screenshots/",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _copy_round_log_files(round_path: Path, staging: Path) -> None:
    for filename in ("round_meta.json", "events.jsonl", "states.jsonl", "decisions.jsonl", "actions.jsonl", "errors.jsonl"):
        source = round_path / filename
        if source.exists():
            shutil.copy2(source, staging / filename)


def _copy_directory(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for path in source.glob("**/*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _copy_config_snapshots(staging: Path) -> list[str]:
    copied: list[str] = []
    for config_path in (
        Path("config/rules.yaml"),
        Path("config/thresholds.yaml"),
        Path("config/screen_1080x2400.yaml"),
        Path("chenzhou_zipai_ai/config/rules.yaml"),
        Path("chenzhou_zipai_ai/config/thresholds.yaml"),
        Path("chenzhou_zipai_ai/config/screen_1080x2400.yaml"),
        Path("pyproject.toml"),
    ):
        if not config_path.exists():
            continue
        target = staging / "config" / config_path.name
        if config_path.parts[0] == "chenzhou_zipai_ai":
            target = staging / "config" / "chenzhou_zipai_ai" / config_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, target)
        copied.append(str(target.relative_to(staging)))
    return copied


def _write_unknown_crops_manifest(staging: Path) -> dict[str, Any]:
    unknown_dir = staging / "crops" / "unknown"
    unknown_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        str(path.relative_to(staging)).replace("\\", "/")
        for path in unknown_dir.glob("**/*")
        if path.is_file() and path.name != "manifest.json"
    )
    payload = {
        "description": "Unknown recognition crops copied from this round, if any.",
        "count": len(files),
        "files": files,
    }
    (unknown_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _write_bundle_manifests(
    staging: Path,
    bundle: ReplayBundle,
    *,
    copied_configs: list[str],
    unknown_crops: dict[str, Any],
) -> None:
    screenshots = sorted(
        str(path.relative_to(staging)).replace("\\", "/")
        for path in (staging / "screenshots").glob("**/*")
        if path.is_file()
    )
    payload = {
        "session_id": bundle.session_id,
        "round_id": bundle.round_id,
        "files": {
            "round_meta": "round_meta.json",
            "events": "events.jsonl",
            "states": "states.jsonl",
            "decisions": "decisions.jsonl",
            "actions": "actions.jsonl",
            "errors": "errors.jsonl",
            "summary": "round_summary.md",
            "replay": "round_replay.json",
        },
        "screenshots": screenshots,
        "screenshot_count": len(screenshots),
        "config_snapshots": copied_configs,
        "unknown_crops": unknown_crops,
        "version_info": "version_info.json",
    }
    (staging / "bundle_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    version = {
        "project_version": bundle.session_meta.get("project_version") or "0.1.0",
        "git_commit": bundle.session_meta.get("git_commit"),
        "runtime_local_only": bundle.session_meta.get("runtime_local_only", True),
        "rules_config_path": bundle.session_meta.get("rules_config_path"),
        "screen_config_path": bundle.session_meta.get("screen_config_path"),
        "thresholds_config_path": bundle.session_meta.get("thresholds_config_path"),
    }
    (staging / "version_info.json").write_text(
        json.dumps(version, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_round_bundle(session_round_path: Path, output_dir: Path | None = None) -> RoundExport:
    session_round_path = Path(session_round_path)
    if not session_round_path.is_dir():
        raise FileNotFoundError(f"round directory not found: {session_round_path}")

    bundle = load_round_bundle(session_round_path)
    export_root = session_round_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    if output_dir is None:
        output_dir = session_round_path
    output_zip = Path(output_dir) / f"{bundle.round_id}_bundle.zip"

    staging = session_round_path / ".bundle_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    _copy_round_log_files(session_round_path, staging)
    _copy_directory(session_round_path / "crops", staging / "crops")
    _write_round_summary(bundle, staging / "round_summary.md")

    replay_rows = {
        "session_meta": bundle.session_meta,
        "round_meta": bundle.round_meta,
        "events": bundle.events,
        "states": bundle.states,
        "decisions": bundle.decisions,
        "actions": bundle.actions,
    }
    (staging / "round_replay.json").write_text(
        json.dumps(replay_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for path in (session_round_path / "screenshots").glob("*.*"):
        if path.is_file():
            target = staging / "screenshots" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for path in (session_round_path / "exports").glob("*.*"):
        if path.is_file():
            target = staging / "exports" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    copied_configs = _copy_config_snapshots(staging)
    unknown_crops = _write_unknown_crops_manifest(staging)
    _write_bundle_manifests(staging, bundle, copied_configs=copied_configs, unknown_crops=unknown_crops)

    (staging / "version.txt").write_text("0.1.0", encoding="utf-8")
    if output_zip.exists():
        output_zip.unlink()
    shutil.make_archive(str(output_zip.with_suffix("")), "zip", staging)
    return RoundExport(output_zip, bundle.session_id, bundle.round_id)


def latest_round_bundle(root: Path) -> tuple[Path, Path, str, str]:
    sessions = [path for path in root.glob("sessions/session_*") if path.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"no session found under {root}")
    latest_session = sorted(sessions, key=lambda item: item.name)[-1]
    rounds = sorted((item for item in (latest_session / "rounds").iterdir() if item.is_dir()))
    if not rounds:
        raise FileNotFoundError(f"no round found under {latest_session}")
    latest_round = rounds[-1]
    return latest_session, latest_round, latest_session.name, latest_round.name
