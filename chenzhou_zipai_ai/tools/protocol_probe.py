"""Read-only logcat probe for APK protocol field strings."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture.adb_capture import run_adb
from vision.protocol_state import PROTOCOL_FIELD_NAMES, protocol_state_from_payload

DEFAULT_PATTERNS = (
    "playerholdcards",
    "curr_card",
    "out_cards",
    "canpeng",
    "canchi",
    "canhu",
    "showGuo",
    "zhuang",
    "zuozhuang",
    "chairId",
    "douzhuangChairId",
)


def filter_protocol_lines(text: str, patterns: tuple[str, ...] = DEFAULT_PATTERNS, *, limit: int = 80) -> list[str]:
    regex = re.compile("|".join(re.escape(item) for item in patterns), re.IGNORECASE)
    lines: list[str] = []
    for line in text.splitlines():
        if regex.search(line):
            lines.append(line.strip())
            if len(lines) >= limit:
                break
    return lines


def probe_logcat(device_id: str | None = None, *, tail: int = 3000, limit: int = 80) -> dict:
    output = run_adb(["shell", "logcat", "-d", "-t", str(tail)], device_id=device_id, timeout=20)
    lines = filter_protocol_lines(output, limit=limit)
    parsed = [protocol_state_from_payload(line) for line in lines[:5]]
    return {
        "device_id": device_id,
        "tail": tail,
        "patterns": list(DEFAULT_PATTERNS),
        "match_count": len(lines),
        "matches": lines,
        "parsed_samples": parsed,
        "known_protocol_fields": sorted(PROTOCOL_FIELD_NAMES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--tail", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = probe_logcat(args.device_id, tail=args.tail, limit=args.limit)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"match_count={result['match_count']} tail={result['tail']}")
    for line in result["matches"][: args.limit]:
        print(line)


if __name__ == "__main__":
    main()
