"""Export one round as markdown for human review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chenzhou_zipai_ai.game_logging import replay_loader


def _format_cards(values: object) -> str:
    if not isinstance(values, list):
        return ""
    return " ".join(str(item) for item in values)


def _find_state_by_frame(bundle, frame_id: str) -> dict:
    for row in bundle.states:
        if row.get("frame_id") == frame_id:
            return row
    return {}


def _action_row(index: int, decision: dict, state: dict) -> str:
    item = decision.get("decision", {})
    action = item.get("action") or ""
    label = item.get("label") or ""
    reason = item.get("reason") or item.get("selected_reason") or ""
    hand = _format_cards(item.get("hand") or state.get("state", {}).get("normalized_hand", []))
    buttons = _format_cards([button.get("type") or button.get("name") for button in state.get("state", {}).get("buttons", [])])
    frame_id = str(decision.get("frame_id", ""))
    screenshot = str(state.get("state", {}).get("screenshot_path") or state.get("state", {}).get("screenshot") or "")
    return (
        f"| {index} | {frame_id} | {hand} | {buttons} | {action} | {label} | {reason} "
        f"| {screenshot} |\n"
    )


def print_round_as_markdown(bundle_path: Path, output: Path | None = None) -> Path:
    bundle = replay_loader.load_round_bundle_from_path(bundle_path)
    output = output or bundle_path.parent / "exports" / "round_for_review.md"
    lines = [
        f"# Round {bundle.round_id} Review",
        "",
        "## 复盘摘要",
        "",
        f"- session_id: {bundle.session_id}",
        f"- round_id: {bundle.round_id}",
        f"- started_at: {bundle.round_meta.get('started_at', '')}",
        f"- ended_at: {bundle.round_meta.get('ended_at', '')}",
        "",
        "| turn | frame | hand | buttons | action | label | reason | screenshot |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    evals_by_decision: dict[str, list[dict]] = {}
    for row in bundle.events:
        if row.get("event_type") == "ACTION_EVALUATED":
            evals_by_decision.setdefault(str(row.get("decision_id", "")), []).append(row.get("data", {}))
    for index, decision in enumerate(bundle.decisions, start=1):
        frame_id = str(decision.get("frame_id", ""))
        state = _find_state_by_frame(bundle, frame_id)
        lines.append(_action_row(index, decision, state))
        decision_id = str(decision.get("decision_id", ""))
        if evals_by_decision.get(decision_id):
            lines.extend(["", f"### Decision {decision_id} Candidates", ""])
            lines.append("| type | label | ev | allowed | reject_reason | reason |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in evals_by_decision[decision_id][:8]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(item.get("type") or item.get("action", {}).get("type", "")),
                            str(item.get("label") or item.get("action", {}).get("label", "")),
                            str(item.get("ev", "")),
                            str(item.get("allowed", "")),
                            str(item.get("reject_reason", "")),
                            str(item.get("reason", "")),
                        ]
                    )
                    + " |"
                )

    safe_halts = [row for row in bundle.events if row.get("event_type") == "SAFE_HALT"]
    if safe_halts:
        lines.extend(["", "## SAFE_HALT", "", "| frame | reason | explanation |", "| --- | --- | --- |"])
        for row in safe_halts:
            payload = row.get("data", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("frame_id", "")),
                        str(payload.get("halt_reason", "")),
                        str(payload.get("explanation", "")),
                    ]
                )
                + " |"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown review file for one round.")
    parser.add_argument("bundle", help="round directory or exported round_bundle.zip")
    parser.add_argument("--output", default=None, help="Output markdown file path.")
    args = parser.parse_args()
    out = print_round_as_markdown(Path(args.bundle), Path(args.output) if args.output else None)
    print(out)


if __name__ == "__main__":
    main()
