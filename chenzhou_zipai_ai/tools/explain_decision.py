"""Explain a single hand/state with the professional local brain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.pro_brain import allocate_hand_structures, analyze_hand, build_decision_context, choose_action


def explain_state(state: dict | list[str]) -> str:
    context = build_decision_context(state)
    allocation = allocate_hand_structures(context)
    analysis = analyze_hand(context, allocation)
    decision = choose_action(state)
    lines = [
        "当前手牌：",
        " ".join(context.normalized_hand),
        "",
        "结构分析：",
    ]
    for meld in allocation.locked_melds:
        lines.append(f"- {''.join(meld.labels)} 已锁定：{meld.reason}，protect={meld.protect_level}")
    for meld in allocation.soft_melds:
        lines.append(f"- {''.join(meld.labels)} 软保护：{meld.reason}")
    for note in allocation.allocation_notes:
        lines.append(f"- {note}")
    if allocation.orphan_cards:
        lines.append("可优先处理：" + " ".join(allocation.orphan_cards))
    lines.extend(
        [
            "",
            f"胡息：confirmed={analysis.confirmed_xi} potential={analysis.potential_xi} min={analysis.min_xi}",
            f"方向：{analysis.hand_direction} / {analysis.attack_defense_mode}",
            "",
            f"最终选择：{decision.selected_action} {decision.selected_label or ''}".strip(),
            f"理由：{decision.reason}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain one decision.")
    parser.add_argument("--state-json", default=None, help="State JSON string or path.")
    parser.add_argument("cards", nargs="*", help="Hand labels if --state-json is not provided.")
    args = parser.parse_args()
    if args.state_json:
        candidate = Path(args.state_json)
        if candidate.exists():
            state = json.loads(candidate.read_text(encoding="utf-8"))
        else:
            state = json.loads(args.state_json)
    else:
        state = args.cards
    print(explain_state(state))


if __name__ == "__main__":
    main()
