"""Explain the latest decision from the latest logged round."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chenzhou_zipai_ai.game_logging import replay_loader


def latest_round_path(logs_root: Path) -> Path:
    _, round_path, _, _ = replay_loader.latest_round_bundle(logs_root)
    return round_path


def explain_latest_decision(logs_root: Path = Path("chenzhou_zipai_ai/logs")) -> str:
    round_path = latest_round_path(logs_root)
    bundle = replay_loader.load_round_bundle_from_path(round_path)
    if not bundle.decisions:
        return "本局未检测到决策记录。"
    latest = bundle.decisions[-1]
    decision = latest.get("decision", {})
    frame_id = str(latest.get("frame_id", ""))
    state = next(
        (item.get("state", {}) for item in bundle.states if item.get("frame_id") == frame_id),
        {},
    )
    action = decision.get("action", "")
    label = decision.get("label") or ""
    reason = decision.get("reason") or decision.get("selected_reason") or ""
    hand = " ".join(state.get("normalized_hand", []))
    button_list = ",".join(item.get("type", "") for item in state.get("buttons", []))
    summary = [
        f"当前手牌：{hand}",
        f"可见按钮：{button_list}",
        f"结构保留：{state.get('orphan_cards', []) or []}",
        f"候选评分（前 3）：",
    ]
    evals = decision.get("evaluations", [])
    for item in evals[:3]:
        summary.append(f"  - {item.get('label')}：{item.get('score', '')}，理由：{item.get('reason', item.get('reasons', ''))}")
    summary.append(f"最终选择：{action}{(' ' + label) if label else ''}")
    summary.append(f"选择原因：{reason}")
    summary.append(f"frame_id：{frame_id}")
    return "\n".join(summary)


def main() -> None:
    print(explain_latest_decision())


if __name__ == "__main__":
    main()
