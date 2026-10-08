"""Summarize SAFE_HALT reasons in logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chenzhou_zipai_ai.game_logging.replay_loader import (
    load_round_bundle_from_path,
)


def analyze_halts(path: Path) -> dict:
    bundle = load_round_bundle_from_path(path)
    counter: Counter[str] = Counter()
    rows = []
    for event in bundle.events:
        if event.get("event_type") != "SAFE_HALT":
            continue
        reason = event.get("data", {}).get("halt_reason") or "unknown_error"
        counter[str(reason)] += 1
        rows.append(
            {
                "frame_id": event.get("frame_id"),
                "decision_id": event.get("decision_id"),
                "reason": reason,
                "explanation": event.get("data", {}).get("explanation"),
            }
        )
    return {
        "session_id": bundle.session_id,
        "round_id": bundle.round_id,
        "total_safe_halts": sum(counter.values()),
        "by_reason": dict(counter),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAFE_HALT reasons.")
    parser.add_argument("round_path", help="round directory or bundle zip")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_halts(Path(args.round_path))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"SAFE_HALT total={result['total_safe_halts']}")
        for reason, count in result["by_reason"].items():
            print(f"{reason}\t{count}")


if __name__ == "__main__":
    main()
