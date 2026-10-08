"""Package one failed decision or SAFE_HALT case for review."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chenzhou_zipai_ai.game_logging.replay_loader import (
    load_round_bundle_from_path,
)


def collect_failure_case(round_path: Path, decision_id: str | None = None, output_dir: Path | None = None) -> Path:
    bundle = load_round_bundle_from_path(round_path)
    output_dir = output_dir or Path(round_path).with_name(f"{Path(round_path).name}_failure_case")
    output_dir.mkdir(parents=True, exist_ok=True)
    if decision_id is None:
        safe_halts = [row for row in bundle.events if row.get("event_type") == "SAFE_HALT"]
        if safe_halts:
            decision_id = str(safe_halts[-1].get("decision_id") or "")
        elif bundle.decisions:
            decision_id = str(bundle.decisions[-1].get("decision_id") or "")
    selected_events = [
        row for row in bundle.events if not decision_id or str(row.get("decision_id")) == decision_id
    ]
    selected_decisions = [
        row for row in bundle.decisions if not decision_id or str(row.get("decision_id")) == decision_id
    ]
    selected_actions = [
        row for row in bundle.actions if not decision_id or str(row.get("decision_id")) == decision_id
    ]
    payload = {
        "session_meta": bundle.session_meta,
        "round_meta": bundle.round_meta,
        "decision_id": decision_id,
        "events": selected_events,
        "decisions": selected_decisions,
        "actions": selected_actions,
    }
    (output_dir / "failure_case.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if Path(round_path).is_dir():
        for event in selected_events:
            data = event.get("data", {})
            for key in ("screenshot_path", "debug_image_path"):
                value = data.get(key)
                if not value:
                    continue
                source = Path(round_path) / str(value)
                if source.exists() and source.is_file():
                    target = output_dir / source.name
                    shutil.copy2(source, target)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one failure case.")
    parser.add_argument("round_path", help="round directory or bundle zip")
    parser.add_argument("--decision-id", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = collect_failure_case(
        Path(args.round_path),
        decision_id=args.decision_id,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(out)


if __name__ == "__main__":
    main()
