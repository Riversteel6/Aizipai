"""Run deterministic no-device dry-run decisions and write a review report."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai.pro_brain import (
    allocate_hand_structures,
    build_decision_context,
    choose_action,
    run_decision_self_check,
)
from control.action_plan import build_action_plan
from engine.cards import BIG_LABELS, SMALL_LABELS, WILD_LABEL


DECK = [*SMALL_LABELS, *BIG_LABELS] * 4 + [WILD_LABEL] * 4


SCRIPTED_CASES = [
    {"name": "triplet_overlap", "hand": ["一", "二", "二", "二", "三"], "legal_actions": ["discard"]},
    {"name": "wildcard_keep", "hand": ["王", "二", "七", "四", "九"], "legal_actions": ["discard"]},
    {"name": "chi_2710", "hand": ["二", "十", "四", "六", "九"], "legal_actions": ["chi", "pass"], "pending_card": "七", "chi_options": [["二", "七", "十"]]},
    {"name": "peng_pair", "hand": ["二", "二", "四", "六", "九"], "legal_actions": ["peng", "pass"], "pending_card": "二"},
    {"name": "hu_complete", "hand": ["贰", "柒", "拾", "壹", "贰", "叁", "九", "九", "九"], "legal_actions": ["hu", "pass"]},
    {"name": "wait_auto_ti", "hand": ["一", "二", "三"], "legal_actions": ["ti", "pass"]},
]


def _hand_details(hand: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "card_id": f"h{index:03d}",
            "name": label,
            "x": 20 + index * 18,
            "y": 100,
            "w": 14,
            "h": 20,
            "confidence": 0.99,
            "clickable": True,
        }
        for index, label in enumerate(hand, start=1)
    ]


def _state_for_case(case: dict[str, Any]) -> dict[str, Any]:
    actions = [str(item).upper() for item in case.get("legal_actions", ["discard"])]
    buttons = [
        {"name": action.lower(), "type": action.lower(), "x": 100 + idx * 70, "y": 40, "w": 60, "h": 32}
        for idx, action in enumerate(actions)
        if action not in {"DISCARD", "TI", "PAO", "MING_LONG", "AUTO_QUAD"}
    ]
    chi_options = [
        {"option_id": f"chi_{idx:03d}", "labels": labels, "confidence": 1.0}
        for idx, labels in enumerate(case.get("chi_options", []), start=1)
    ]
    return {
        "context_id": f"dry_run_{case['name']}",
        "hand": list(case["hand"]),
        "raw_hand": list(case["hand"]),
        "hand_details": _hand_details(case["hand"]),
        "legal_actions": [{"type": action} for action in actions],
        "buttons": buttons,
        "chi_options": chi_options,
        "option_details": chi_options,
        "pending_card": case.get("pending_card"),
        "remaining_deck_count": case.get("remaining_deck_count", 30),
        "discard_button": {"x": 50, "y": 50, "w": 80, "h": 30},
        "sanity_checks": {"ok": True},
        "phase": "dry_run",
    }


def _random_case(rng: random.Random, index: int, hand_size: int) -> dict[str, Any]:
    return {
        "name": f"random_discard_{index:03d}",
        "hand": rng.sample(DECK, hand_size),
        "legal_actions": ["discard"],
        "remaining_deck_count": rng.randint(8, 40),
    }


def _case_rows(count: int, hand_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        if index < len(SCRIPTED_CASES):
            rows.append(SCRIPTED_CASES[index])
        else:
            rows.append(_random_case(rng, index, hand_size))
    return rows


def run_dry_run(
    *,
    count: int = 50,
    hand_size: int = 14,
    seed: int = 20260531,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    cases = _case_rows(count, hand_size, seed)
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        state = _state_for_case(case)
        context = build_decision_context(state)
        allocation = allocate_hand_structures(context)
        decision = choose_action(state)
        action_plan = build_action_plan(state, decision.to_dict())
        guard = run_decision_self_check(context, decision, action_plan)
        hard_ids = set(allocation.hard_protected_instances)
        violation_reasons: list[str] = []
        if allocation.allocation_conflicts:
            violation_reasons.append("duplicate_card_allocation")
        if decision.selected_action == "DISCARD" and decision.selected_card_id in hard_ids and decision.candidate_stage != "forced_break_hard_protection":
            violation_reasons.append("discarded_hard_protected")
        if action_plan.get("ready") and not guard.passed:
            violation_reasons.append("ready_plan_failed_self_check")
        if action_plan.get("ready") and action_plan.get("validation", {}).get("passed") is False:
            violation_reasons.append("ready_plan_validation_failed")
        row = {
            "index": index,
            "name": case["name"],
            "hand": list(case["hand"]),
            "action": decision.action,
            "selected_label": decision.selected_label,
            "selected_card_id": decision.selected_card_id,
            "candidate_stage": decision.candidate_stage,
            "ev": decision.ev,
            "reason": decision.reason,
            "action_plan_ready": action_plan.get("ready"),
            "self_check_passed": guard.passed,
            "self_check_reason": guard.reason,
            "hard_protected_labels": allocation.hard_protected_labels,
            "soft_protected_labels": allocation.soft_protected_labels,
            "violations": violation_reasons,
        }
        rows.append(row)
        if violation_reasons:
            violations.append(row)
    result = {
        "ok": not violations,
        "count": count,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
        "violations": violations,
        "rows": rows,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dry_run_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_markdown_report(result, output_dir / "dry_run_report.md")
    return result


def _write_markdown_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Dry Run Report",
        "",
        f"- ok: {result['ok']}",
        f"- decisions: {result['count']}",
        f"- seed: {result['seed']}",
        f"- violations: {len(result['violations'])}",
        "",
        "| # | case | hand | action | stage | ready | self_check | reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        hand = " ".join(row["hand"])
        reason = str(row["reason"]).replace("|", "/")
        lines.append(
            f"| {row['index']} | {row['name']} | {hand} | {row['action']} {row.get('selected_label') or ''} | "
            f"{row['candidate_stage']} | {row['action_plan_ready']} | {row['self_check_passed']} | {reason} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic 50-decision no-device dry-run.")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--hand-size", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_dry_run(
        count=args.count,
        hand_size=args.hand_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"dry_run ok={result['ok']} decisions={result['count']} violations={len(result['violations'])}")
        if args.output_dir:
            print(f"report={args.output_dir}")
        for row in result["violations"][:10]:
            print(f"  {row['index']}: {','.join(row['violations'])} hand={' '.join(row['hand'])}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
