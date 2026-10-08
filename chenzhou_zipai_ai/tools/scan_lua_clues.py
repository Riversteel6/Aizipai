"""Scan decompiled Lua files for local mock-table UI/protocol clues."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from chenzhou_zipai_ai.tools.mock_build_common import (
    DIST,
    ensure_mock_build_logs,
    iter_files,
    rel,
    write_json,
    write_markdown,
)

KEYWORDS = [
    "准备",
    "胡",
    "吃",
    "碰",
    "过",
    "跑",
    "提",
    "偎",
    "龙",
    "手牌",
    "出牌",
    "弃牌",
    "结算",
    "room",
    "table",
    "card",
    "cards",
    "hand",
    "discard",
    "chi",
    "peng",
    "hu",
    "pass",
    "ready",
    "cmd",
    "proto",
    "protocol",
    "msg",
    "socket",
    "1001",
    "1003",
    "1012",
    "1013",
    "1014",
    "1086",
    "1087",
    "1027",
    "1031",
    "1035",
]

BUTTON_WORDS = {"准备", "胡", "吃", "碰", "过", "pass", "ready", "chi", "peng", "hu"}
PROTOCOL_WORDS = {"cmd", "proto", "protocol", "msg", "socket", "1001", "1003", "1012", "1013", "1014", "1086", "1087", "1027", "1031", "1035"}
LAYOUT_WORDS = {"room", "table", "手牌", "出牌", "弃牌", "结算", "card", "cards", "hand", "discard"}
ACTION_WORDS = {"跑", "提", "偎", "龙"}


def guess_kind(path: Path, keyword: str, context: str) -> str:
    probe = f"{path.as_posix()} {keyword} {context}".lower()
    if keyword in PROTOCOL_WORDS or any(word in probe for word in ("socket", "protocol", "cmd", "parse")):
        return "possible_protocol_handler"
    if keyword in BUTTON_WORDS or any(word in probe for word in ("button", "btn", "click")):
        return "possible_button_logic"
    if keyword in LAYOUT_WORDS or any(word in probe for word in ("view", "layer", "panel", "layout")):
        return "possible_ui_layout"
    if keyword in ACTION_WORDS or any(word in probe for word in ("action", "operate", "op")):
        return "possible_action_flow"
    return "unknown"


def scan_lua_clues(lua_root: Path, max_matches: int, max_per_file: int) -> dict[str, Any]:
    clues: list[dict[str, Any]] = []
    keyword_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    lower_keywords = [(keyword, _keyword_pattern(keyword)) for keyword in KEYWORDS]

    for path in iter_files(lua_root, (".lua",)):
        matches_in_file = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, start=1):
                    lowered = line.lower()
                    for keyword, pattern in lower_keywords:
                        if not pattern.search(lowered):
                            continue
                        context = " ".join(line.strip().split())[:220]
                        item = {
                            "file": rel(path),
                            "line": line_no,
                            "keyword": keyword,
                            "context": context,
                            "guess": guess_kind(path, keyword, context),
                        }
                        clues.append(item)
                        keyword_counts[keyword] += 1
                        file_counts[rel(path)] += 1
                        matches_in_file += 1
                        break
                    if len(clues) >= max_matches or matches_in_file >= max_per_file:
                        break
            if len(clues) >= max_matches:
                break
        except OSError:
            continue

    return {
        "lua_root": rel(lua_root),
        "max_matches": max_matches,
        "max_per_file": max_per_file,
        "matches": clues,
        "keyword_counts": dict(keyword_counts.most_common()),
        "top_files": dict(file_counts.most_common(30)),
    }


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    lowered = keyword.lower()
    if lowered.isascii() and lowered.replace("_", "").isalnum():
        return re.compile(rf"(?<![a-z0-9_]){re.escape(lowered)}(?![a-z0-9_])")
    return re.compile(re.escape(lowered))


def markdown_report(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Lua Clues",
        "",
        f"- Lua root: `{payload['lua_root']}`",
        f"- Matches recorded: `{len(payload['matches'])}`",
        f"- Match cap: `{payload['max_matches']}`",
        f"- Per-file cap: `{payload['max_per_file']}`",
        "",
        "## Keyword Counts",
        "",
    ]
    for keyword, count in payload["keyword_counts"].items():
        lines.append(f"- `{keyword}`: {count}")
    lines.extend(["", "## Top Files", ""])
    for file, count in payload["top_files"].items():
        lines.append(f"- `{file}`: {count}")
    lines.extend(["", "## Sample Clues", ""])
    for item in payload["matches"][:120]:
        lines.append(
            f"- `{item['file']}:{item['line']}` `{item['keyword']}` "
            f"`{item['guess']}` - {item['context']}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lua-root", type=Path, default=DIST / "lua_all_decompiled")
    parser.add_argument("--max-matches", type=int, default=2000)
    parser.add_argument("--max-per-file", type=int, default=25)
    args = parser.parse_args()

    payload = scan_lua_clues(args.lua_root, args.max_matches, args.max_per_file)
    out_dir = ensure_mock_build_logs()
    write_json(out_dir / "lua_clues.json", payload)
    write_markdown(out_dir / "lua_clues.md", markdown_report(payload))
    print(f"LUA_CLUES_MD={out_dir / 'lua_clues.md'}")
    print(f"LUA_CLUES_JSON={out_dir / 'lua_clues.json'}")


if __name__ == "__main__":
    main()
