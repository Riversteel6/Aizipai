"""Inventory local assets that can inform the offline mock table."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

from chenzhou_zipai_ai.tools.mock_build_common import (
    DIST,
    ROOT,
    WORKSPACE,
    ensure_mock_build_logs,
    file_record,
    iter_files,
    rel,
    write_json,
    write_markdown,
)

CARD_LABELS = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "拾", "王"]
BUTTON_LABELS = ["准备", "胡", "吃", "碰", "过", "确定", "取消"]
BACKGROUND_WORDS = ["bg", "background", "table", "desk", "room", "桌", "背景"]
PANEL_WORDS = ["hand", "chi", "discard", "settle", "ready", "panel", "手牌", "吃", "弃牌", "结算", "准备"]
LUA_GROUPS = {
    "possible_table_ui": ["table", "room", "game", "paohuzi", "zipai", "牌桌", "房间"],
    "possible_buttons": ["button", "btn", "click", "准备", "胡", "吃", "碰", "过"],
    "possible_protocol_cmd": ["cmd", "proto", "protocol", "socket", "msg", "1001", "1012", "1035"],
    "possible_card_actions": ["chi", "peng", "hu", "discard", "跑", "提", "偎", "龙"],
    "possible_settlement": ["settle", "result", "score", "结算"],
}


def image_dimensions(path: Path) -> dict[str, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
    except OSError:
        return None
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        return {"width": int.from_bytes(header[16:20], "big"), "height": int.from_bytes(header[20:24], "big")}
    return None


def classify_by_label(paths: list[Path], labels: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in labels:
        matches = [path for path in paths if label in path.stem]
        result[label] = {
            "status": "confirmed" if matches else "unknown",
            "count": len(matches),
            "examples": [rel(path) for path in matches[:8]],
        }
    return result


def classify_by_words(paths: list[Path], words: list[str]) -> list[dict[str, Any]]:
    rows = []
    lowered_words = [word.lower() for word in words]
    for path in paths:
        probe = path.as_posix().lower()
        if any(word.lower() in probe for word in lowered_words):
            row = file_record(path)
            dims = image_dimensions(path)
            if dims:
                row.update(dims)
            rows.append(row)
    return rows[:80]


def screenshot_summary(root: Path, sample_limit: int) -> dict[str, Any]:
    if not root.exists():
        return {"root": rel(root), "exists": False}
    count = 0
    total_bytes = 0
    samples: list[dict[str, Any]] = []
    newest: dict[str, Any] | None = None
    for entry in os.scandir(root):
        if not entry.is_file():
            continue
        path = Path(entry.path)
        count += 1
        stat = entry.stat()
        total_bytes += stat.st_size
        record = {
            "path": rel(path),
            "bytes": stat.st_size,
            "modified": stat.st_mtime,
        }
        if newest is None or stat.st_mtime > newest["modified"]:
            newest = record
        if len(samples) < sample_limit:
            dims = image_dimensions(path)
            if dims:
                record.update(dims)
            samples.append(record)
    for record in samples:
        record["modified"] = _format_ts(record["modified"])
    if newest:
        newest["modified"] = _format_ts(newest["modified"])
    return {
        "root": rel(root),
        "exists": True,
        "file_count": count,
        "total_bytes": total_bytes,
        "sample_files": samples,
        "newest_file": newest,
    }


def _format_ts(value: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def find_lua_files(lua_root: Path) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {key: [] for key in LUA_GROUPS}
    total = 0
    for path in iter_files(lua_root, (".lua",)):
        total += 1
        probe = path.as_posix().lower()
        for group, words in LUA_GROUPS.items():
            if len(grouped[group]) >= 30:
                continue
            if any(word.lower() in probe for word in words):
                grouped[group].append(rel(path))
    return {"root": rel(lua_root), "file_count": total, "groups": grouped}


def scan_assets(args: argparse.Namespace) -> dict[str, Any]:
    template_paths = list(iter_files(args.templates, (".png", ".jpg", ".jpeg", ".webp")))
    suffix_counts = Counter(path.suffix.lower() for path in template_paths)
    capture_files = [file_record(path) for path in iter_files(args.captures, (".pcap", ".jsonl"))]
    apk_dirs = [
        rel(path)
        for path in [WORKSPACE / "data" / "apk_extract", DIST / "apk_extract", DIST / "base_apk"]
        if path.exists()
    ]
    apk_files = [file_record(path) for path in DIST.glob("*.apk")]

    return {
        "templates_root": rel(args.templates),
        "templates": {
            "file_count": len(template_paths),
            "suffix_counts": dict(suffix_counts),
            "cards": classify_by_label(template_paths, CARD_LABELS),
            "buttons": classify_by_label(template_paths, BUTTON_LABELS),
            "background_candidates": classify_by_words(template_paths, BACKGROUND_WORDS),
            "panel_candidates": classify_by_words(template_paths, PANEL_WORDS),
        },
        "screenshots": screenshot_summary(args.screenshots, args.screenshot_samples),
        "captures": {
            "root": rel(args.captures),
            "files": capture_files[:80],
            "file_count": len(capture_files),
        },
        "apk": {"apk_files": apk_files, "extract_dirs": apk_dirs},
        "lua": find_lua_files(args.lua_root),
        "missing": {
            "cards": [label for label, row in classify_by_label(template_paths, CARD_LABELS).items() if row["status"] == "unknown"],
            "buttons": [label for label, row in classify_by_label(template_paths, BUTTON_LABELS).items() if row["status"] == "unknown"],
            "layout": ["ready_button", "settlement_button"],
        },
    }


def markdown_report(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Asset Inventory",
        "",
        "## Templates",
        "",
        f"- Root: `{payload['templates_root']}`",
        f"- Files: `{payload['templates']['file_count']}`",
        f"- Suffixes: `{payload['templates']['suffix_counts']}`",
        "",
        "## Card Assets",
        "",
    ]
    for label, row in payload["templates"]["cards"].items():
        examples = ", ".join(f"`{item}`" for item in row["examples"][:3]) or "none"
        lines.append(f"- `{label}`: {row['status']} ({row['count']}) {examples}")
    lines.extend(["", "## Button Assets", ""])
    for label, row in payload["templates"]["buttons"].items():
        examples = ", ".join(f"`{item}`" for item in row["examples"][:3]) or "none"
        lines.append(f"- `{label}`: {row['status']} ({row['count']}) {examples}")
    lines.extend(["", "## Background / Panel Candidates", ""])
    lines.append(f"- Background candidates: `{len(payload['templates']['background_candidates'])}`")
    lines.append(f"- Panel candidates: `{len(payload['templates']['panel_candidates'])}`")
    lines.extend(["", "## Screenshots", ""])
    screenshots = payload["screenshots"]
    lines.append(f"- Root: `{screenshots['root']}`")
    lines.append(f"- Exists: `{screenshots['exists']}`")
    if screenshots.get("exists"):
        lines.append(f"- File count: `{screenshots['file_count']}`")
        lines.append(f"- Total bytes: `{screenshots['total_bytes']}`")
        if screenshots.get("newest_file"):
            lines.append(f"- Newest: `{screenshots['newest_file']['path']}`")
    lines.extend(["", "## Captures", ""])
    lines.append(f"- Root: `{payload['captures']['root']}`")
    lines.append(f"- Files: `{payload['captures']['file_count']}`")
    for row in payload["captures"]["files"][:20]:
        lines.append(f"- `{row['path']}` ({row['bytes']} bytes)")
    lines.extend(["", "## Lua File Clues", ""])
    for group, files in payload["lua"]["groups"].items():
        lines.append(f"- `{group}`: {len(files)}")
        for file in files[:8]:
            lines.append(f"  - `{file}`")
    lines.extend(["", "## Missing / Unknown", ""])
    lines.append(f"- Cards: `{payload['missing']['cards']}`")
    lines.append(f"- Buttons: `{payload['missing']['buttons']}`")
    lines.append(f"- Layout fields needing confirmation: `{payload['missing']['layout']}`")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lua-root", type=Path, default=DIST / "lua_all_decompiled")
    parser.add_argument("--templates", type=Path, default=ROOT / "data" / "templates")
    parser.add_argument("--screenshots", type=Path, default=ROOT / "data" / "screenshots")
    parser.add_argument("--captures", type=Path, default=DIST / "captures")
    parser.add_argument("--screenshot-samples", type=int, default=20)
    args = parser.parse_args()

    payload = scan_assets(args)
    out_dir = ensure_mock_build_logs()
    write_json(out_dir / "asset_inventory.json", payload)
    write_markdown(out_dir / "asset_inventory.md", markdown_report(payload))
    print(f"ASSET_INVENTORY_MD={out_dir / 'asset_inventory.md'}")
    print(f"ASSET_INVENTORY_JSON={out_dir / 'asset_inventory.json'}")


if __name__ == "__main__":
    main()
