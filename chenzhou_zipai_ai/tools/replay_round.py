"""Replay one full round as human-readable text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chenzhou_zipai_ai.game_logging import replay_loader


def _state_by_frame(bundle, frame_id: str) -> dict:
    for row in bundle.states:
        if row.get("frame_id") == frame_id:
            return row
    return {}


def _safe_get_hand(state_row: dict) -> str:
    state = state_row.get("state", {})
    hand = state.get("normalized_hand") or state.get("hand") or []
    if isinstance(hand, list):
        return " ".join(hand)
    return ""


def _find_events(bundle, frame_id: str, event_type: str) -> list[dict]:
    return [row for row in bundle.events if row.get("frame_id") == frame_id and row.get("event_type") == event_type]


def _action_detail(decision_row: dict) -> str:
    decision = decision_row.get("decision", {})
    action = decision.get("action") or decision_row.get("action") or ""
    label = decision.get("label") or decision_row.get("label") or ""
    reason = decision.get("reason") or decision.get("selected_reason") or decision_row.get("reason") or ""
    return f"{action}\t{label}\t{reason}"


def _decision_event_reason(bundle, decision_id: str) -> str:
    for row in bundle.events:
        if row.get("event_type") == "DECISION_SELECTED" and row.get("decision_id") == decision_id:
            return row.get("data", {}).get("selected_reason") or ""
    return ""


def _safe_halt_rows(bundle) -> list[dict]:
    return [row for row in bundle.events if row.get("event_type") == "SAFE_HALT"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print replay summary for a round.")
    parser.add_argument("bundle", help="round directory or exported round_bundle.zip")
    args = parser.parse_args()
    bundle = replay_loader.load_round_bundle_from_path(Path(args.bundle))
    print(f"session_id={bundle.session_id}")
    print(f"round_id={bundle.round_id}")
    print("turn\tframe\taction\tlabel\treason\tscreen\thand")
    for index, decision_row in enumerate(bundle.decisions, start=1):
        decision_id = str(decision_row.get("decision_id", ""))
        frame_id = str(decision_row.get("frame_id", ""))
        state = _state_by_frame(bundle, frame_id)
        screenshot = state.get("state", {}).get("screenshot", "")
        _ = _find_events(bundle, frame_id, "ACTION_PLAN_CREATED")
        print(f"{index}\t{frame_id}\t{_action_detail(decision_row)}\t{screenshot}\t{_safe_get_hand(state)}")
        detail = _decision_event_reason(bundle, decision_id)
        if detail:
            print(f"  detail={detail}")
    print("safe_halt_count=", len(_safe_halt_rows(bundle)))
    for row in _safe_halt_rows(bundle):
        payload = row.get("data", {})
        print(
            "  "
            + "\t".join(
                str(item)
                for item in (
                    row.get("frame_id", ""),
                    payload.get("halt_reason", ""),
                    payload.get("explanation", ""),
                )
            )
        )


if __name__ == "__main__":
    main()
