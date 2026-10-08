"""Generate and replay deterministic policy log fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai.pro_brain import allocate_hand_structures, analyze_hand, build_decision_context, choose_action
from chenzhou_zipai_ai.game_logging import GameLogger, LoggerConfig
from chenzhou_zipai_ai.game_logging.validate_logs import validate_round_bundle
from tools.replay_decision import _replay_one_decision
from chenzhou_zipai_ai.game_logging import replay_loader


FIXTURE_CASES = [
    {"name": "discard_triplet_safe", "hand": ["一", "二", "二", "二", "三"], "legal_actions": ["discard"]},
    {"name": "discard_wildcard_safe", "hand": ["王", "二", "七", "四", "九"], "legal_actions": ["discard"]},
    {"name": "discard_mixed_triplet", "hand": ["伍", "伍", "五", "九", "八"], "legal_actions": ["discard"]},
    {"name": "discard_sequence", "hand": ["四", "五", "六", "九", "八"], "legal_actions": ["discard"]},
    {
        "name": "chi_2710",
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": ["chi", "pass"],
        "option_details": [{"region_name": "chi_options", "labels": ["二", "七", "十"], "confidence": 1.0}],
        "pending_card": "七",
    },
    {
        "name": "chi_uncertain_pass",
        "hand": ["二", "十", "四", "六", "九"],
        "legal_actions": ["chi", "pass"],
        "option_details": [{"region_name": "chi_options", "labels": ["二", "七", "十"], "confidence": 0.6}],
        "pending_card": "七",
    },
    {"name": "peng_pair", "hand": ["二", "二", "四", "六", "九"], "legal_actions": ["peng", "pass"], "pending_card": "二"},
    {"name": "peng_missing_pair_pass", "hand": ["二", "四", "六", "九"], "legal_actions": ["peng", "pass"], "pending_card": "二"},
    {"name": "hu_complete", "hand": ["贰", "柒", "拾", "壹", "贰", "叁", "九", "九", "九"], "legal_actions": ["hu", "pass"]},
    {"name": "hu_xi_not_enough_pass", "hand": ["二", "七", "十"], "legal_actions": ["hu", "pass"]},
    {"name": "discard_2710_protected", "hand": ["二", "七", "十", "九", "八"], "legal_actions": ["discard"]},
    {"name": "discard_123_protected", "hand": ["一", "二", "三", "九", "八"], "legal_actions": ["discard"]},
    {"name": "discard_big_sequence_protected", "hand": ["肆", "伍", "陆", "九", "八"], "legal_actions": ["discard"]},
    {
        "name": "chi_missing_cards_pass",
        "hand": ["一", "四", "九"],
        "legal_actions": ["chi", "pass"],
        "option_details": [{"region_name": "chi_options", "labels": ["一", "二", "三"], "confidence": 1.0}],
        "pending_card": "三",
    },
    {
        "name": "chi_consumes_hard_triplet_pass",
        "hand": ["二", "二", "二", "一"],
        "legal_actions": ["chi", "pass"],
        "option_details": [{"region_name": "chi_options", "labels": ["一", "二", "三"], "confidence": 1.0}],
        "pending_card": "三",
    },
    {
        "name": "chi_soft_break_ev_not_enough",
        "hand": ["五", "六", "七", "八", "九"],
        "legal_actions": ["chi", "pass"],
        "option_details": [{"region_name": "chi_options", "labels": ["四", "五", "六"], "confidence": 1.0}],
        "pending_card": "四",
    },
    {"name": "peng_hard_triplet_pass", "hand": ["二", "二", "二", "四", "九"], "legal_actions": ["peng", "pass"], "pending_card": "二"},
    {"name": "peng_followup_hard_break_pass", "hand": ["二", "二", "三", "三", "三"], "legal_actions": ["peng", "pass"], "pending_card": "二"},
    {"name": "hu_wildcard_partition", "hand": ["王", "贰", "柒", "壹", "贰", "叁", "九", "九", "九"], "legal_actions": ["hu", "pass"]},
    {
        "name": "safe_halt_low_confidence",
        "hand": ["一", "二", "三", "九", "八"],
        "legal_actions": ["discard"],
        "low_confidence": True,
    },
    {"name": "wait_auto_pao", "hand": ["一", "二", "三"], "legal_actions": ["pao", "pass"]},
    {"name": "wait_auto_ti", "hand": ["一", "二", "三"], "legal_actions": ["ti", "pass"]},
]


def _hand_details(hand: list[str], *, low_confidence: bool = False) -> list[dict]:
    return [
        {
            "card_id": f"h{index:03d}",
            "name": label,
            "x": index * 12,
            "y": 20,
            "w": 10,
            "h": 16,
            "confidence": 0.40 if low_confidence and index == 1 else 0.99,
            "clickable": True,
        }
        for index, label in enumerate(hand, start=1)
    ]


def _state_for_case(case: dict, screenshot_path: str) -> dict:
    option_details = list(case.get("option_details") or [])
    state = {
        "screenshot": screenshot_path,
        "screenshot_path": screenshot_path,
        "raw_hand": list(case["hand"]),
        "hand": list(case["hand"]),
        "normalized_hand": list(case["hand"]),
        "phase": "play",
        "buttons": [{"name": action, "x": 100, "y": 100, "w": 60, "h": 36} for action in case["legal_actions"] if action != "discard"],
        "legal_actions": [{"type": action} for action in case["legal_actions"]],
        "hand_details": _hand_details(case["hand"], low_confidence=bool(case.get("low_confidence"))),
        "option_details": option_details,
        "chi_options": option_details,
        "pending_card": case.get("pending_card"),
        "remaining_deck_count": 30,
        "fixture_name": case["name"],
    }
    if case.get("low_confidence"):
        state["recognition_warnings"] = ["fixture_low_confidence_card"]
    return state


def _plan_for_decision(decision) -> dict:
    action = decision.action
    ready = action in {"discard", "hu", "chi", "peng"}
    target_type = "hand_card" if action == "discard" else "button" if ready else None
    return {
        "action": action,
        "ready": ready,
        "reason": decision.reason,
        "target_type": target_type,
        "target_card_id": decision.selected_card_id,
        "target_label": decision.selected_label,
        "policy_selected_action": {"type": action, "label": decision.selected_label},
        "validation": {
            "passed": action != "safe_halt",
            "target_matches_policy": True,
            "checks": ["fixture_replay", "target_matches_policy"],
        },
    }


def build_fixture_round(logs_root: Path) -> tuple[Path, list[str]]:
    logs_root.mkdir(parents=True, exist_ok=True)
    source = logs_root / "fixture_source.png"
    source.write_text("fixture screenshot placeholder", encoding="utf-8")
    logger = GameLogger(
        logs_root,
        config=LoggerConfig(save_raw_screenshot=True, save_debug_screenshot=False),
    )
    session_id = logger.start_session(device_id="fixture", screen_size=(1080, 2344))
    round_id = logger.start_round()
    decision_ids: list[str] = []
    for case in FIXTURE_CASES:
        frame_id = logger.next_frame_id()
        rel_screenshot = logger.log_frame_captured(frame_id, source)
        state = _state_for_case(case, rel_screenshot)
        logger.log_frame_recognized(frame_id, state)
        decision_id = logger.next_decision_id()
        decision_ids.append(decision_id)
        context = build_decision_context(state)
        allocation = allocate_hand_structures(context)
        analysis = analyze_hand(context, allocation)
        decision = choose_action(state)
        logger.log_structure_allocation(frame_id, decision_id, allocation.to_dict())
        logger.log_hand_analysis(frame_id, decision_id, analysis.to_dict())
        logger.log_legal_actions(
            frame_id,
            decision_id,
            decision.context_snapshot["legal_actions"],
            decision.context_snapshot["rejected_actions"],
        )
        logger.log_action_evaluation_started(
            frame_id,
            decision_id,
            candidate_count=len(decision.action_evals),
            source="fixture",
        )
        for item in decision.action_evals:
            logger.log_action_evaluated(frame_id, decision_id, item.to_dict(), is_rejected=not item.allowed)
        plan = _plan_for_decision(decision)
        logger.log_decision(decision_id, frame_id, decision.to_dict(), reason=decision.reason)
        logger.log_action_plan(decision_id, frame_id, plan)
        if decision.action == "safe_halt":
            logger.log_safe_halt(
                frame_id=frame_id,
                decision_id=decision_id,
                reason=decision.reason,
                raw_hand=state.get("raw_hand") or state.get("hand") or [],
                hard_protected=decision.hard_protected,
                screenshot_path=rel_screenshot,
            )
        logger.log_tap(
            frame_id,
            decision_id,
            plan,
            execute_enabled=False,
            dry_run=True,
            tap_executed=False,
            tap_x=None,
            tap_y=None,
            target_label=decision.selected_label,
            target_card_id=decision.selected_card_id,
            adb_result="fixture_dry_run",
            before_screenshot=rel_screenshot,
        )
    logger.end_round(result="fixture")
    return logger.paths.round_dir(session_id, round_id), decision_ids


def run_fixture_replay(logs_root: Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="zipai_replay_fixtures_") as tmp:
        root = logs_root or Path(tmp)
        round_path, decision_ids = build_fixture_round(root)
        validation = validate_round_bundle(round_path)
        bundle = replay_loader.load_round_bundle_from_path(round_path)
        replay_results = [_replay_one_decision(bundle, decision_id) for decision_id in decision_ids]
        matches = [row["same_action"] and row["same_label"] for row in replay_results]
        return {
            "ok": validation["ok"] and all(matches),
            "round_path": str(round_path),
            "validation": validation,
            "replayed": len(replay_results),
            "matched": sum(1 for item in matches if item),
            "mismatches": [
                row for row, matched in zip(replay_results, matches) if not matched
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and replay deterministic policy log fixtures.")
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_fixture_replay(args.logs_root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = "ok" if result["ok"] else "FAIL"
        print(f"{status}\treplayed={result['replayed']}\tmatched={result['matched']}\tround={result['round_path']}")
        for row in result["mismatches"]:
            print(f"  mismatch {row['decision_id']}: logged={row['logged']} replayed={row['replayed']}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
