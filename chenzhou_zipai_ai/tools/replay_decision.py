"""Replay a single decision from an exported round bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai.policy import choose_action, choose_discard
from ai.pro_brain import choose_action as choose_professional_action
from chenzhou_zipai_ai.game_logging import replay_loader


def _build_state_hand(bundle, decision):
    frame_id = decision.get("frame_id")
    if not frame_id:
        return None, None
    state_rows = [row for row in bundle.states if row.get("frame_id") == frame_id]
    if not state_rows:
        return None, None
    return state_rows[-1].get("state", {}), frame_id


def _state_hand(state: dict[str, Any]) -> list[str]:
    for key in ("hand", "normalized_hand", "raw_hand"):
        value = state.get(key)
        if isinstance(value, list) and value:
            return [str(item) for item in value]
    details = state.get("hand_details") or []
    return [
        str(item.get("name") or item.get("label") or "")
        for item in details
        if isinstance(item, dict) and (item.get("name") or item.get("label"))
    ]


def _normalise_legal_actions(actions: Any) -> list[dict[str, Any]]:
    if not isinstance(actions, list):
        return []
    result: list[dict[str, Any]] = []
    for item in actions:
        if isinstance(item, str):
            result.append({"type": item})
        elif isinstance(item, dict):
            action_type = item.get("type") or item.get("action")
            if action_type:
                normalised = dict(item)
                normalised["type"] = action_type
                result.append(normalised)
    return result


def _decision_legal_actions(decision_row: dict[str, Any]) -> list[dict[str, Any]]:
    decision = decision_row.get("decision") or {}
    for payload in (
        decision,
        decision.get("context_snapshot") or {},
        (decision.get("professional_decision") or {}).get("context_snapshot") or {},
    ):
        legal_actions = _normalise_legal_actions(payload.get("legal_actions"))
        if legal_actions:
            return legal_actions
    return []


def _decision_context(decision_row: dict[str, Any]) -> dict[str, Any]:
    decision = decision_row.get("decision") or {}
    for payload in (decision.get("professional_decision") or {}, decision):
        context = (payload.get("context_snapshot") or {}).get("context") or {}
        if context.get("raw_hand") or context.get("normalized_hand"):
            return context
    return {}


def _hand_details_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for item in context.get("card_instances") or []:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not label:
            continue
        details.append(
            {
                "card_id": item.get("card_id") or item.get("id"),
                "name": label,
                "label": label,
                "center_x": item.get("x"),
                "center_y": item.get("y"),
                "confidence": item.get("confidence"),
                "clickable": item.get("clickable", True),
            }
        )
    return details


def _apply_decision_context(replay_state: dict[str, Any], context: dict[str, Any]) -> None:
    raw_hand = context.get("raw_hand")
    normalized_hand = context.get("normalized_hand")
    if isinstance(raw_hand, list) and raw_hand:
        replay_state["raw_hand"] = [str(item) for item in raw_hand]
    if isinstance(normalized_hand, list) and normalized_hand:
        replay_state["normalized_hand"] = [str(item) for item in normalized_hand]
        replay_state["hand"] = [str(item) for item in normalized_hand]
    elif isinstance(raw_hand, list) and raw_hand:
        replay_state["hand"] = [str(item) for item in raw_hand]
    hand_details = _hand_details_from_context(context)
    if hand_details:
        replay_state["hand_details"] = hand_details
    for key in ("phase", "frame_id", "buttons", "chi_options", "compare_options", "recognition_warnings", "recognition_errors"):
        if key in context:
            replay_state[key] = context[key]
    if context.get("remaining_deck_count") is not None:
        replay_state["remaining_deck_count"] = context.get("remaining_deck_count")
    legal_types = context.get("legal_action_types")
    if isinstance(legal_types, list) and legal_types:
        replay_state["legal_actions"] = [{"type": str(item)} for item in legal_types]


def _build_replay_state(state: dict[str, Any], decision_row: dict[str, Any]) -> dict[str, Any]:
    replay_state = dict(state)
    decision_context = _decision_context(decision_row)
    if decision_context:
        _apply_decision_context(replay_state, decision_context)
    hand = _state_hand(replay_state)
    if hand:
        replay_state.setdefault("hand", hand)
        replay_state.setdefault("raw_hand", hand)
        replay_state.setdefault("normalized_hand", hand)
    if not replay_state.get("legal_actions"):
        legal_actions = _decision_legal_actions(decision_row)
        if legal_actions:
            replay_state["legal_actions"] = legal_actions
    if replay_state.get("remaining_deck_count") is None and replay_state.get("remaining_cards_estimate") is not None:
        replay_state["remaining_deck_count"] = replay_state.get("remaining_cards_estimate")
    return replay_state


def _replay_one_decision(bundle, decision_id: str) -> dict:
    found = None
    for item in bundle.decisions:
        if str(item.get("decision_id")) == decision_id:
            found = item
            break
    if found is None:
        raise SystemExit(f"decision {decision_id} not found")

    state, _ = _build_state_hand(bundle, found)
    if not state:
        raise SystemExit(f"no state for decision {decision_id}")
    replay_state = _build_replay_state(state, found)
    hand = _state_hand(replay_state)
    legal = [item.get("type") for item in replay_state.get("legal_actions", [])]
    option_details = replay_state.get("option_details", []) or []
    option_count = 0
    if option_details is not None:
        option_count = len(option_details)
    memory = replay_state.get("memory")
    remaining_deck_count = replay_state.get("remaining_deck_count")
    option_stage = "compare" if any(item.get("region_name") == "compare_options" for item in option_details) else "chi"

    replayed = None
    professional = None
    if hand:
        try:
            professional = choose_professional_action(replay_state)
        except Exception:
            professional = None
        if legal:
            if "discard" in legal and not ("hu" in legal or "chi" in legal or "peng" in legal or "pao" in legal):
                replayed = choose_discard(
                    hand,
                    legal_actions=legal,
                    remaining_deck_count=remaining_deck_count,
                    memory=memory,
                )
            else:
                replayed = choose_action(
                    hand,
                    legal_actions=legal,
                    option_count=option_count,
                    option_details=option_details,
                    remaining_deck_count=remaining_deck_count,
                    memory=memory,
                )
    if replayed is None and professional is None:
        raise SystemExit(f"unable to replay decision {decision_id}")

    replayed_action = replayed.action if replayed is not None else None
    replayed_label = replayed.label if replayed is not None else None
    explanation = replayed.reason if replayed is not None else ""
    if professional is not None:
        pro_payload = professional.to_dict()
        replayed_action = pro_payload.get("action")
        replayed_label = pro_payload.get("label")
        explanation = pro_payload.get("reason") or explanation

    return {
        "decision_id": decision_id,
        "logged": {
            "action": found.get("decision", {}).get("action"),
            "label": found.get("decision", {}).get("label"),
        },
        "replayed": {"action": replayed_action, "label": replayed_label},
        "professional_replayed": professional.to_dict() if professional is not None else None,
        "same_action": replayed_action == found.get("decision", {}).get("action"),
        "same_label": replayed_label == found.get("decision", {}).get("label"),
        "explanation": explanation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one decision from exported bundle.")
    parser.add_argument("bundle", help="round directory or exported round_bundle.zip")
    parser.add_argument("--decision-id", required=True)
    args = parser.parse_args()
    bundle = replay_loader.load_round_bundle_from_path(Path(args.bundle))
    result = _replay_one_decision(bundle, args.decision_id)
    print(f"logged={result['logged']}")
    print(f"replayed={result['replayed']}")
    if result.get("professional_replayed"):
        pro = result["professional_replayed"]
        print(
            "professional="
            + str(
                {
                    "action": pro.get("action"),
                    "label": pro.get("label"),
                    "reason": pro.get("reason"),
                }
            )
        )
    print(f"explanation={result['explanation']}")
    print(f"match={result['same_action'] and result['same_label']}")
    if not (result["same_action"] and result["same_label"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
