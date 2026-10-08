"""Run fixed professional-brain policy cases from the specification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.pro_brain import allocate_hand_structures, build_decision_context, choose_action


CASES = [
    {
        "name": "peng_556677_on_5_should_pass",
        "state": {
            "hand": ["五", "五", "六", "六", "七", "七"],
            "legal_actions": [{"type": "PENG"}, {"type": "PASS"}],
            "pending_card": "五",
        },
        "selected_action": "PASS",
        "eval_type": "PENG",
        "eval_allowed": False,
        "eval_reject": "peng_breaks_double_sequence",
    },
    {
        "name": "chi_123_plus_叁_on_三_should_pass",
        "state": {
            "hand": ["一", "二", "三", "叁"],
            "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
            "pending_card": "三",
            "chi_options": [{"option_id": "chi_same_rank", "labels": ["三", "三", "叁"], "confidence": 1.0}],
        },
        "selected_action": "PASS",
        "eval_type": "CHI",
        "eval_allowed": False,
        "eval_reject": "chi_breaks_existing_123",
    },
    {
        "name": "chi_2710_clean_should_chi",
        "state": {
            "hand": ["二", "十", "九"],
            "legal_actions": [{"type": "CHI"}, {"type": "PASS"}],
            "pending_card": "七",
            "chi_options": [{"option_id": "chi_2710_clean", "labels": ["二", "七", "十"], "confidence": 1.0}],
        },
        "selected_action": "CHI",
        "selected_option_id": "chi_2710_clean",
        "eval_type": "CHI",
        "eval_allowed": True,
        "ev_beats_pass": True,
    },
    {
        "name": "auto_pao_ti_should_wait_not_decide",
        "state": {
            "hand": ["一", "二", "三"],
            "legal_actions": [{"type": "TI"}, {"type": "PASS"}],
        },
        "selected_action": "WAIT_AUTO_MELD",
    },
    {
        "name": "triplet_blocks_123_reuse",
        "hand": ["一", "二", "二", "二", "三"],
        "not_label": "二",
        "locked_type": "exact_triplet",
        "free_count": ("二", 0),
    },
    {
        "name": "triplet_blocks_2710_reuse",
        "hand": ["二", "七", "七", "七", "十"],
        "not_label": "七",
        "locked_type": "exact_triplet",
        "free_count": ("七", 0),
    },
    {
        "name": "mixed_triplet_vs_sequence_single_overlap",
        "hand": ["伍", "伍", "五", "四", "六", "九"],
        "selected": "九",
    },
    {
        "name": "mixed_triplet_and_sequence_two_fives",
        "hand": ["伍", "伍", "五", "四", "五", "六", "九"],
        "selected": "九",
    },
    {
        "name": "wildcard_not_discarded",
        "hand": ["王", "二", "七", "四", "九"],
        "not_label": "王",
    },
    {
        "name": "sequence_protected",
        "hand": ["四", "五", "六", "八", "九"],
        "selected_in": ["八", "九"],
    },
    {
        "name": "chi_rejects_hard_triplet_123",
        "state": {
            "hand": ["二", "二", "二", "一"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "三",
            "chi_options": [{"option_id": "chi_123", "labels": ["一", "二", "三"], "confidence": 1.0}],
        },
        "selected_action": "PASS",
        "eval_type": "CHI",
        "eval_allowed": False,
        "eval_reject": "chi_consumes_hard_protected",
    },
    {
        "name": "chi_accepts_small_2710",
        "state": {
            "hand": ["二", "十", "四", "六", "九"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "七",
            "chi_options": [{"option_id": "chi_2710", "labels": ["二", "七", "十"], "confidence": 1.0}],
        },
        "selected_action": "CHI",
        "selected_option_id": "chi_2710",
        "eval_type": "CHI",
        "eval_allowed": True,
        "ev_beats_pass": True,
    },
    {
        "name": "chi_accepts_big_123",
        "state": {
            "hand": ["壹", "贰", "四", "六", "九"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "叁",
            "chi_options": [{"option_id": "chi_123_big", "labels": ["壹", "贰", "叁"], "confidence": 1.0}],
        },
        "selected_action": "CHI",
        "selected_option_id": "chi_123_big",
        "eval_type": "CHI",
        "eval_allowed": True,
        "ev_beats_pass": True,
    },
    {
        "name": "chi_accepts_normal_456",
        "state": {
            "hand": ["五", "六", "二", "九", "十"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "四",
            "chi_options": [{"option_id": "chi_456", "labels": ["四", "五", "六"], "confidence": 1.0}],
        },
        "selected_action": "CHI",
        "selected_option_id": "chi_456",
        "eval_type": "CHI",
        "eval_allowed": True,
        "ev_beats_pass": True,
    },
    {
        "name": "chi_accepts_normal_345",
        "state": {
            "hand": ["三", "四", "八", "九", "十"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "五",
            "chi_options": [{"option_id": "chi_345", "labels": ["三", "四", "五"], "confidence": 1.0}],
        },
        "selected_action": "CHI",
        "selected_option_id": "chi_345",
        "eval_type": "CHI",
        "eval_allowed": True,
        "ev_beats_pass": True,
    },
    {
        "name": "chi_rejects_missing_hand_cards",
        "state": {
            "hand": ["一", "四", "九"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "三",
            "chi_options": [{"option_id": "chi_missing", "labels": ["一", "二", "三"], "confidence": 1.0}],
        },
        "selected_action": "PASS",
        "eval_type": "CHI",
        "eval_allowed": False,
        "eval_reject": "chi_option_not_in_hand",
    },
    {
        "name": "chi_rejects_uncertain_option",
        "state": {
            "hand": ["二", "十", "四", "六", "九"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "七",
            "chi_options": [{"option_id": "chi_uncertain", "labels": ["二", "七", "十"], "confidence": 0.6}],
        },
        "selected_action": "PASS",
        "eval_type": "CHI",
        "eval_allowed": False,
        "eval_reject": "chi_option_uncertain",
    },
    {
        "name": "chi_rejects_wildcard_without_big_gain",
        "state": {
            "hand": ["四", "五", "九"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "王",
            "chi_options": [{"option_id": "chi_wild", "labels": ["四", "五", "王"], "confidence": 1.0}],
        },
        "selected_action": "PASS",
        "eval_type": "CHI",
        "eval_allowed": False,
        "eval_reject": "chi_consumes_wildcard_without_big_gain",
    },
    {
        "name": "chi_rejects_soft_break_when_ev_not_enough",
        "state": {
            "hand": ["五", "六", "七", "八", "九"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "四",
            "chi_options": [{"option_id": "chi_soft_break", "labels": ["四", "五", "六"], "confidence": 1.0}],
        },
        "selected_action": "PASS",
        "eval_type": "CHI",
        "eval_allowed": False,
        "eval_reject": "chi_breaks_complete_meld",
    },
    {
        "name": "chi_accepts_8910",
        "state": {
            "hand": ["八", "九", "二", "四", "五"],
            "legal_actions": [{"type": "CHI"}],
            "pending_card": "十",
            "chi_options": [{"option_id": "chi_8910", "labels": ["八", "九", "十"], "confidence": 1.0}],
        },
        "selected_action": "CHI",
        "selected_option_id": "chi_8910",
        "eval_type": "CHI",
        "eval_allowed": True,
        "ev_beats_pass": True,
    },
    {
        "name": "peng_accepts_small_pair",
        "state": {
            "hand": ["二", "二", "四", "六", "九"],
            "legal_actions": [{"type": "PENG"}],
            "pending_card": "二",
        },
        "selected_action": "PENG",
        "eval_type": "PENG",
        "eval_allowed": True,
        "ev_beats_pass": True,
    },
    {
        "name": "peng_accepts_big_pair",
        "state": {
            "hand": ["贰", "贰", "四", "六", "九"],
            "legal_actions": [{"type": "PENG"}],
            "pending_card": "贰",
        },
        "selected_action": "PENG",
        "eval_type": "PENG",
        "eval_allowed": True,
        "ev_beats_pass": True,
    },
    {
        "name": "peng_rejects_missing_pair",
        "state": {
            "hand": ["二", "四", "六", "九"],
            "legal_actions": [{"type": "PENG"}],
            "pending_card": "二",
        },
        "selected_action": "PASS",
        "eval_type": "PENG",
        "eval_allowed": False,
        "eval_reject": "peng_pair_not_available",
    },
    {
        "name": "peng_rejects_hard_triplet_consumption",
        "state": {
            "hand": ["二", "二", "二", "四", "九"],
            "legal_actions": [{"type": "PENG"}],
            "pending_card": "二",
        },
        "selected_action": "PASS",
        "eval_type": "PENG",
        "eval_allowed": False,
        "eval_reject": "peng_consumes_hard_protected",
    },
    {
        "name": "peng_rejects_followup_forced_hard_break",
        "state": {
            "hand": ["二", "二", "三", "三", "三"],
            "legal_actions": [{"type": "PENG"}],
            "pending_card": "二",
        },
        "selected_action": "PASS",
        "eval_type": "PENG",
        "eval_allowed": False,
        "eval_reject": "peng_followup_forces_hard_break",
    },
]


def run_case(case: dict) -> dict:
    state_or_hand = case.get("state") or case["hand"]
    context = build_decision_context(state_or_hand)
    allocation = allocate_hand_structures(context)
    decision = choose_action(state_or_hand)
    errors: list[str] = []
    selected = decision.selected_label
    if case.get("selected_action") is not None and decision.selected_action != case["selected_action"]:
        errors.append(f"expected action {case['selected_action']}, got {decision.selected_action}")
    if case.get("selected_option_id") is not None and decision.selected_option_id != case["selected_option_id"]:
        errors.append(f"expected option {case['selected_option_id']}, got {decision.selected_option_id}")
    if case.get("selected") is not None and selected != case["selected"]:
        errors.append(f"expected selected {case['selected']}, got {selected}")
    if case.get("not_label") is not None and selected == case["not_label"]:
        errors.append(f"forbidden discard selected: {selected}")
    if case.get("selected_in") is not None and selected not in set(case["selected_in"]):
        errors.append(f"expected one of {case['selected_in']}, got {selected}")
    if case.get("locked_type") is not None and not any(
        meld.type == case["locked_type"] for meld in allocation.locked_melds
    ):
        errors.append(f"missing locked meld type {case['locked_type']}")
    if case.get("free_count") is not None:
        label, expected = case["free_count"]
        got = allocation.free_counts_after_locked.get(label, 0)
        if got != expected:
            errors.append(f"free_count {label} expected {expected}, got {got}")
    eval_payload = None
    if case.get("eval_type") is not None:
        action_eval = next((item for item in decision.action_evals if item.type == case["eval_type"]), None)
        if action_eval is None:
            errors.append(f"missing action eval {case['eval_type']}")
        else:
            eval_payload = action_eval.to_dict()
            if case.get("eval_allowed") is not None and action_eval.allowed != case["eval_allowed"]:
                errors.append(f"expected eval allowed {case['eval_allowed']}, got {action_eval.allowed}")
            if case.get("eval_reject") is not None and action_eval.reject_reason != case["eval_reject"]:
                errors.append(f"expected reject {case['eval_reject']}, got {action_eval.reject_reason}")
            if case.get("ev_beats_pass"):
                pass_ev = action_eval.debug_details.get("pass_ev")
                if pass_ev is None or action_eval.ev <= pass_ev:
                    errors.append(f"expected {case['eval_type']} ev {action_eval.ev} to beat pass {pass_ev}")
    return {
        "name": case["name"],
        "ok": not errors,
        "errors": errors,
        "selected": selected,
        "selected_action": decision.selected_action,
        "selected_option_id": decision.selected_option_id,
        "candidate_stage": decision.candidate_stage,
        "response_eval": eval_payload,
        "allocation": allocation.to_dict(),
    }


def run_cases() -> dict:
    rows = [run_case(case) for case in CASES]
    return {
        "ok": all(row["ok"] for row in rows),
        "total": len(rows),
        "passed": sum(1 for row in rows if row["ok"]),
        "failed": [row for row in rows if not row["ok"]],
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed professional policy cases.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_cases()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for row in result["cases"]:
            status = "ok" if row["ok"] else "FAIL"
            print(f"{status}\t{row['name']}\tselected={row['selected']}\tstage={row['candidate_stage']}")
            for error in row["errors"]:
                print(f"  ERROR {error}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
