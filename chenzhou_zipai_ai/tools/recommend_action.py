"""Recommend a first-generation AlphaDog action from a screenshot."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SRC = WORKSPACE / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai.policy import choose_discard  # Compatibility hook for older tests/tools; not used for live policy.
from ai.pro_brain import (
    allocate_hand_structures as allocate_professional_structures,
    analyze_hand as analyze_professional_hand,
    build_decision_context as build_professional_context,
    choose_action as choose_professional_action,
    run_decision_self_check,
)
from ai.frozen_two_player_strategy import choose_action
from ai.decision_log import append_decision
from ai.monte_carlo import simulate_discards
from ai.option_policy import choose_option, evaluate_option
from ai.opponent_model import infer_opponent_profile
from chenzhou_zipai_ai.game_logging import GameLogger
from control.action_plan import (
    action_plan_contract_error,
    build_action_plan,
    chi_option_guard_reason,
    describe_chi_option_guard,
    repair_chi_option_labels,
)
from engine.hu_checker import explain_hu
from engine.xi_calculator import total_xi
from tools.inspect_state import inspect_screenshot, write_preview
from tools.protocol_log_to_state import latest_protocol_state_from_log
from tools.qs_protocol_parser import QSProtocolError, parse_packet_to_state
from vision.history_memory import (
    VisionMemory,
    build_sanity_checks,
    load_memory,
    recover_temporally_hidden_hand,
    save_memory,
)
from vision.protocol_state import protocol_state_from_payload
from engine.cards import normalize_card_label
from engine.rules import ROOM_MODE_PRESETS, rules_for_room
from engine.chi_rules import enumerate_chi_plans


CHI_OPTION_MIN_CONFIDENCE = 0.65
RESPONSE_ACTION_NAMES = {"chi", "peng", "pao", "hu", "pass"}


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for base in (ROOT, WORKSPACE):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return candidate


def _apply_hidden_cards(state: dict, hidden_cards: list[str], *, expected_total: int | None) -> dict:
    patched = dict(state)
    raw_hand = list(patched.get("raw_hand") or patched.get("hand") or [])
    hand = list(patched.get("hand") or raw_hand)
    details = list(patched.get("hand_details") or [])
    metadata = dict(patched.get("metadata") or {})
    normalized_hidden = [normalize_card_label(label) or str(label) for label in hidden_cards]
    for index, label in enumerate(normalized_hidden, start=1):
        card_id = f"hidden_{index:03d}"
        raw_hand.append(label)
        hand.append(label)
        details.append(
            {
                "card_id": card_id,
                "id": card_id,
                "name": label,
                "label": label,
                "x": None,
                "y": None,
                "w": None,
                "h": None,
                "center": None,
                "confidence": 1.0,
                "clickable": False,
                "source": "hidden_manual_override",
            }
        )
    metadata["hidden_cards"] = normalized_hidden
    metadata["hidden_card_count"] = len(normalized_hidden)
    patched["raw_hand"] = raw_hand
    patched["hand"] = hand
    patched["hand_details"] = details
    patched["hand_count"] = len(hand)
    patched["metadata"] = metadata
    patched["sanity_checks"] = build_sanity_checks(patched, expected_total=expected_total)
    return patched


def _apply_protocol_state(state: dict, protocol_payload: str | Path | dict, *, expected_total: int | None) -> dict:
    protocol = _load_protocol_state(protocol_payload)
    patched = dict(state)
    metadata = dict(patched.get("metadata") or {})
    protocol_metadata = protocol.get("metadata", {})
    metadata["protocol_state"] = protocol_metadata
    if protocol_metadata.get("opponent_priority_pending"):
        metadata["opponent_priority_pending"] = True
        metadata["opponent_priority_source"] = protocol_metadata.get("opponent_priority_source") or "protocol_state"
    metadata["recognition_source"] = "protocol_with_screenshot_coordinates"
    patched["metadata"] = metadata

    if not _has_visual_game_anchor(patched):
        metadata["protocol_overlay_skipped"] = {
            "reason": "no_visual_game_anchor",
            "protocol_hand_count": len(protocol.get("hand") or []),
            "protocol_legal_actions": protocol.get("legal_actions") or [],
        }
        patched["metadata"] = metadata
        patched["legal_actions"] = [{"type": "WAIT", "source": "no_visual_game_anchor"}]
        return patched

    hand = list(protocol.get("hand") or [])
    if hand:
        if _should_skip_stale_protocol_hand_overlay(patched, hand):
            metadata["protocol_hand_binding"] = {
                "skipped": True,
                "reason": "protocol_hand_count_mismatch_in_discard_phase",
                "protocol_count": len(hand),
                "visible_count": len(patched.get("hand") or patched.get("raw_hand") or []),
            }
            patched["metadata"] = metadata
        else:
            patched["raw_hand"] = hand
            patched["hand"] = hand
            patched["hand_count"] = len(hand)
            patched["hand_details"], binding = _bind_protocol_hand_to_screenshot_slots(patched.get("hand_details") or [], hand)
            metadata["protocol_hand_binding"] = binding
            patched["metadata"] = metadata
            patched["sanity_checks"] = build_sanity_checks(patched, expected_total=expected_total)
    ignore_protocol_actions = _should_ignore_unanchored_protocol_pass_prompt(patched, protocol)
    if ignore_protocol_actions:
        metadata["ignored_protocol_legal_actions"] = {
            "reason": "empty_pending_pass_without_clickable_button",
            "legal_actions": protocol.get("legal_actions"),
        }
        patched["metadata"] = metadata
    if (
        protocol.get("legal_actions")
        and not ignore_protocol_actions
        and not _should_preserve_visible_discard_phase(patched, protocol)
    ):
        patched["legal_actions"] = protocol["legal_actions"]
        patched["buttons"] = _merge_protocol_legal_actions_with_visible_buttons(
            patched.get("buttons") or [],
            protocol["legal_actions"],
            protocol.get("buttons") or [],
        )
    for key in ("pending_card", "remaining_deck_count", "option_details", "discards"):
        value = protocol.get(key)
        if value not in (None, [], {}):
            patched[key] = value
    if protocol.get("option_details"):
        patched["options"] = list(protocol["option_details"])
    return patched


def _has_visual_game_anchor(state: dict) -> bool:
    if state.get("hand") or state.get("raw_hand") or state.get("hand_details"):
        return True
    if state.get("discard_button") or state.get("option_details") or state.get("options"):
        return True
    for button in state.get("buttons") or []:
        if not isinstance(button, dict):
            continue
        name = str(button.get("name") or button.get("type") or "").lower()
        if name in RESPONSE_ACTION_NAMES:
            return True
    for groups in (state.get("meld_groups") or {}).values():
        if groups:
            return True
    for cells in (state.get("melds") or {}).values():
        if cells:
            return True
    for cards in (state.get("discards") or {}).values():
        if cards:
            return True
    return bool(state.get("pending_action_card") or state.get("opponent_pending_card"))


def _should_ignore_unanchored_protocol_pass_prompt(state: dict, protocol: dict) -> bool:
    action_types = {
        str(action.get("type") if isinstance(action, dict) else action).lower()
        for action in protocol.get("legal_actions") or []
    }
    if action_types != {"pass"}:
        return False
    if protocol.get("pending_card"):
        return False
    metadata = protocol.get("metadata") or {}
    payload = metadata.get("qs_payload") or {}
    if payload and payload.get("cmd") != 1014:
        return False
    buttons = [*(state.get("buttons") or []), *(protocol.get("buttons") or [])]
    return not any(
        isinstance(button, dict)
        and str(button.get("name") or button.get("type") or "").lower() == "pass"
        and _has_click_box(button)
        for button in buttons
    )


def _should_skip_stale_protocol_hand_overlay(state: dict, protocol_hand: list[str]) -> bool:
    if not protocol_hand:
        return False
    phase = str(state.get("phase") or state.get("metadata", {}).get("phase") or "")
    legal_actions = state.get("legal_actions") or []
    discard_phase = phase == "await_action" or bool(state.get("discard_button"))
    if legal_actions:
        action_types = {
            str(action.get("type") if isinstance(action, dict) else action).lower()
            for action in legal_actions
        }
        discard_phase = discard_phase or (action_types and action_types <= {"discard"})
    if not discard_phase:
        return False
    visible_hand = state.get("hand") or state.get("raw_hand") or []
    if not visible_hand:
        return False
    return len(protocol_hand) != len(visible_hand)


def _load_protocol_state(protocol_payload: str | Path | dict) -> dict:
    if hasattr(protocol_payload, "read_latest"):
        return protocol_payload.read_latest()

    if isinstance(protocol_payload, dict) and ("packet_hex" in protocol_payload or "packet_base64" in protocol_payload):
        try:
            return parse_packet_to_state(protocol_payload)
        except QSProtocolError:
            return protocol_state_from_payload(protocol_payload)

    if isinstance(protocol_payload, (str, Path)):
        payload_path = Path(protocol_payload)
        try:
            if payload_path.exists():
                return latest_protocol_state_from_log(payload_path)
        except OSError:
            pass

    text = str(protocol_payload)
    if "packet_hex" in text or "packet_base64" in text:
        try:
            return parse_packet_to_state(text)
        except QSProtocolError:
            pass
    return protocol_state_from_payload(protocol_payload)


def _overlay_protocol_hand_details(details: list[dict], labels: list[str]) -> list[dict]:
    return _bind_protocol_hand_to_screenshot_slots(details, labels)[0]


def _merge_protocol_legal_actions_with_visible_buttons(
    visible_buttons: list[dict],
    legal_actions: list[dict],
    protocol_buttons: list[dict],
) -> list[dict]:
    legal_names = {
        str(action.get("type") if isinstance(action, dict) else action)
        for action in legal_actions
        if str(action.get("type") if isinstance(action, dict) else action) != "discard"
    }
    merged: list[dict] = [
        dict(button) | {"source": "protocol_allowed_visible_button"}
        for button in visible_buttons
        if str(button.get("name") or button.get("type")) in legal_names and _has_click_box(button)
    ]
    for button in protocol_buttons:
        if not isinstance(button, dict) or not _has_click_box(button):
            continue
        name = str(button.get("name") or button.get("type"))
        if name in legal_names and not any(str(item.get("name") or item.get("type")) == name for item in merged):
            merged.append(dict(button) | {"source": "protocol_button"})
    inferred = _infer_missing_protocol_response_buttons(merged, legal_names)
    for button in inferred:
        name = str(button.get("name") or button.get("type"))
        if not any(str(item.get("name") or item.get("type")) == name for item in merged):
            merged.append(button)
    return merged


def _infer_missing_protocol_response_buttons(visible_buttons: list[dict], legal_names: set[str]) -> list[dict]:
    response_order = ["hu", "chi", "peng", "pass"]
    known: dict[str, tuple[int, int]] = {}
    boxes: dict[str, tuple[int, int]] = {}
    for button in visible_buttons:
        name = str(button.get("name") or button.get("type") or "")
        if name not in response_order or not _has_click_box(button):
            continue
        center = _button_center(button)
        if center is None:
            continue
        known[name] = center
        boxes[name] = (int(button.get("w") or 150), int(button.get("h") or 170))
    if len(known) < 2:
        return []

    indexed = sorted((response_order.index(name), center) for name, center in known.items())
    gaps = [
        abs(right[1][0] - left[1][0]) / max(1, right[0] - left[0])
        for left, right in zip(indexed, indexed[1:])
    ]
    if not gaps:
        return []
    slot = int(round(sorted(gaps)[len(gaps) // 2]))
    y = int(round(sum(center[1] for center in known.values()) / len(known)))
    inferred: list[dict] = []
    for name in response_order:
        if name not in legal_names or name in known:
            continue
        target_index = response_order.index(name)
        nearest_index, nearest_center = min(
            indexed,
            key=lambda item: abs(item[0] - target_index),
        )
        x = int(round(nearest_center[0] + (target_index - nearest_index) * slot))
        width, height = boxes.get(name) or _typical_response_button_box(boxes)
        inferred.append(
            {
                "name": name,
                "source": "protocol_inferred_visible_button",
                "confidence": 1.0,
                "x": max(0, int(x - width / 2)),
                "y": max(0, int(y - height / 2)),
                "w": width,
                "h": height,
            }
        )
    return inferred


def _button_center(button: dict) -> tuple[int, int] | None:
    if button.get("center") is not None:
        center = button["center"]
        return int(center[0]), int(center[1])
    if all(key in button for key in ("x", "y", "w", "h")):
        return int(button["x"] + button["w"] / 2), int(button["y"] + button["h"] / 2)
    return None


def _typical_response_button_box(boxes: dict[str, tuple[int, int]]) -> tuple[int, int]:
    widths = [item[0] for item in boxes.values()]
    heights = [item[1] for item in boxes.values()]
    if not widths or not heights:
        return 150, 170
    return sorted(widths)[len(widths) // 2], sorted(heights)[len(heights) // 2]


def _should_preserve_visible_discard_phase(state: dict, protocol: dict) -> bool:
    if not state.get("discard_button"):
        return False
    if state.get("buttons"):
        return False
    visible_actions = state.get("legal_actions") or []
    if visible_actions and not any((action.get("type") if isinstance(action, dict) else action) == "discard" for action in visible_actions):
        return False
    protocol_actions = protocol.get("legal_actions") or []
    action_types = {
        str(action.get("type") if isinstance(action, dict) else action)
        for action in protocol_actions
    }
    if action_types - {"pass", "discard"}:
        return False
    return True


def _bind_protocol_hand_to_screenshot_slots(details: list[dict], labels: list[str]) -> tuple[list[dict], dict]:
    ordered_details = _ordered_hand_slots(details)
    visual_counts = Counter(_detail_label(item) for item in ordered_details)
    protocol_counts = Counter(normalize_card_label(label) or str(label) for label in labels)
    slots_by_label: dict[str, list[dict]] = defaultdict(list)
    for item in ordered_details:
        label = _detail_label(item)
        if label:
            slots_by_label[label].append(item)
    used_slot_ids: set[int] = set()
    result: list[dict] = []
    clickable_bound = 0
    missing_coordinate = 0
    exact_label_bound = 0
    positional_fallback_bound = 0
    for index, label in enumerate(labels):
        normalized_label = normalize_card_label(label) or str(label)
        base, binding_mode = _select_protocol_slot(
            normalized_label,
            index,
            ordered_details,
            slots_by_label,
            used_slot_ids,
            visual_counts,
            protocol_counts,
        )
        has_coordinate = _has_hand_coordinate(base)
        clickable = bool(base.get("clickable", has_coordinate)) and has_coordinate
        if clickable:
            clickable_bound += 1
        if not has_coordinate:
            missing_coordinate += 1
        if binding_mode == "exact_label":
            exact_label_bound += 1
        elif binding_mode == "positional_ocr_override":
            positional_fallback_bound += 1
        card_id = base.get("card_id") or base.get("id") or f"protocol_{index + 1:03d}"
        base.update(
            {
                "name": normalized_label,
                "label": normalized_label,
                "source": "protocol_overlay",
                "confidence": 1.0,
                "card_id": card_id,
                "id": base.get("id") or card_id,
                "clickable": clickable,
                "protocol_index": index + 1,
                "protocol_binding": binding_mode,
            }
        )
        result.append(base)
    binding = {
        "mode": "protocol_labels_screenshot_coordinates",
        "protocol_count": len(labels),
        "visible_slot_count": len(ordered_details),
        "clickable_bound_count": clickable_bound,
        "missing_coordinate_count": missing_coordinate,
        "extra_visible_slot_count": max(0, len(ordered_details) - len(labels)),
        "exact_label_bound_count": exact_label_bound,
        "positional_fallback_bound_count": positional_fallback_bound,
    }
    return result, binding


def _select_protocol_slot(
    label: str,
    index: int,
    ordered_details: list[dict],
    slots_by_label: dict[str, list[dict]],
    used_slot_ids: set[int],
    visual_counts: Counter[str],
    protocol_counts: Counter[str],
) -> tuple[dict, str]:
    for slot in slots_by_label.get(label, []):
        slot_id = id(slot)
        if slot_id in used_slot_ids:
            continue
        used_slot_ids.add(slot_id)
        return dict(slot), "exact_label"
    if index < len(ordered_details):
        candidate = ordered_details[index]
        candidate_id = id(candidate)
        candidate_label = _detail_label(candidate)
        if candidate_id not in used_slot_ids and visual_counts[candidate_label] > protocol_counts[candidate_label]:
            used_slot_ids.add(candidate_id)
            return dict(candidate) | {"visual_label_before_protocol_overlay": candidate_label}, "positional_ocr_override"
    return {}, "missing_coordinate"


def _detail_label(item: dict) -> str:
    return normalize_card_label(item.get("name") or item.get("label") or "")


def _ordered_hand_slots(details: list[dict]) -> list[dict]:
    indexed = [(index, item) for index, item in enumerate(details) if isinstance(item, dict)]
    indexed.sort(key=lambda item: _hand_slot_sort_key(item[1], item[0]))
    return [dict(item) for _, item in indexed]


def _hand_slot_sort_key(item: dict, index: int) -> tuple[float, float, int]:
    center = item.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return (float(center[0]), float(center[1]), index)
    x = item.get("x")
    y = item.get("y")
    w = item.get("w") or 0
    h = item.get("h") or 0
    if x is None:
        return (float(index), 0.0, index)
    return (float(x) + float(w) / 2, float(y or 0) + float(h) / 2, index)


def _has_hand_coordinate(item: dict) -> bool:
    if item.get("center") is not None:
        return True
    return item.get("x") is not None and item.get("y") is not None


def _has_click_box(item: dict) -> bool:
    if item.get("center") is not None:
        return True
    return all(item.get(key) is not None for key in ("x", "y", "w", "h"))


def _clear_stale_options_when_discard_button_visible(state: dict) -> dict:
    if not state.get("discard_button"):
        return state
    buttons = state.get("buttons") or []
    response_buttons = {
        str(button.get("name") or button.get("type") or "").lower()
        for button in buttons
        if isinstance(button, dict)
    }
    if response_buttons & {"chi", "peng", "hu", "pass"}:
        return state
    if not state.get("option_details") and not state.get("options"):
        return state
    patched = dict(state)
    metadata = dict(patched.get("metadata") or {})
    metadata["stale_options_cleared"] = {
        "reason": "discard_button_visible_without_response_buttons",
        "option_detail_count": len(patched.get("option_details") or []),
        "option_count": len(patched.get("options") or []),
    }
    patched["metadata"] = metadata
    patched["option_details"] = []
    patched["options"] = []
    return patched


def _repair_chi_options_before_policy(state: dict) -> dict:
    patched = dict(state)

    def repair_item(item: dict) -> dict:
        if not isinstance(item, dict) or item.get("region_name") != "chi_options":
            return item
        original = list(item.get("labels") or item.get("cards") or item.get("option_cards") or [])
        repaired = repair_chi_option_labels(original, patched)
        if repaired == original:
            return item
        updated = dict(item)
        updated["labels"] = repaired
        updated["cards"] = repaired
        updated["option_cards"] = repaired
        updated["raw_labels"] = original
        updated["label_repair"] = {
            "from": original,
            "to": repaired,
            "reason": "policy_pre_repair",
        }
        return updated

    option_details = [repair_item(item) for item in patched.get("option_details") or []]
    if option_details:
        patched["option_details"] = option_details
        patched["options"] = [repair_item(item) for item in patched.get("options") or option_details]
        patched["chi_options"] = [item for item in option_details if item.get("region_name") == "chi_options"]
    elif patched.get("chi_options"):
        repaired_chi_options = []
        for item in patched.get("chi_options") or []:
            if isinstance(item, dict):
                repaired_chi_options.append(repair_item({**item, "region_name": "chi_options"}))
            else:
                repaired_chi_options.append(item)
        patched["chi_options"] = repaired_chi_options
    return patched


def _inject_semantic_chi_options_before_policy(state: dict, rules: dict) -> dict:
    if state.get("chi_options") or state.get("option_details") or state.get("options"):
        return state
    button_names = {
        str(item.get("name") or item.get("type") or "").lower()
        for item in state.get("buttons") or []
        if isinstance(item, dict)
    }
    if "chi" not in button_names:
        return state
    # A protocol card is an exact game fact and must outrank an approximate
    # screenshot match.  Without protocol evidence, do not turn a weak visual
    # guess into a committed CHI probe: the candidate screen can then disagree
    # with the cached intent and leave the UI transaction locked.
    pending = state.get("pending_card")
    pending_source = "protocol" if isinstance(pending, str) and pending else None
    if not pending_source:
        visual_pending = state.get("pending_action_card")
        if isinstance(visual_pending, dict):
            try:
                visual_confidence = float(visual_pending.get("confidence") or 0.0)
            except (TypeError, ValueError):
                visual_confidence = 0.0
            if visual_confidence >= 0.60:
                pending = visual_pending.get("name") or visual_pending.get("label")
                pending_source = "visual_pending_surface"
            else:
                pending = None
    if not isinstance(pending, str) or not pending:
        patched = dict(state)
        metadata = dict(patched.get("metadata") or {})
        metadata["semantic_chi_pending_card_untrusted"] = {
            "reason": "no_exact_protocol_or_confident_visual_pending_card",
        }
        patched["metadata"] = metadata
        return patched
    hand = [str(label) for label in state.get("hand") or []]
    if not hand:
        return state
    plans = enumerate_chi_plans(
        hand,
        pending,
        allow_1510=bool(rules.get("rules", {}).get("allow_1510", False)),
    )
    if not plans:
        return state
    unique_initial_groups: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for plan in plans:
        key = tuple(plan.initial_group)
        if key in seen:
            continue
        seen.add(key)
        unique_initial_groups.append(key)
    patched = dict(state)
    patched["chi_options"] = [
        {
            "option_id": f"semantic_chi_{index:03d}",
            "labels": list(group),
            "cards": list(group),
            "option_cards": list(group),
            "confidence": 1.0,
            "semantic_only": True,
        }
        for index, group in enumerate(unique_initial_groups, start=1)
    ]
    metadata = dict(patched.get("metadata") or {})
    metadata["semantic_chi_plans"] = {
        "pending_card": pending,
        "pending_card_source": pending_source,
        "plan_count": len(plans),
        "initial_group_count": len(unique_initial_groups),
    }
    patched["metadata"] = metadata
    return patched


def recommend_from_screenshot(
    screenshot: str | Path,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    rules_path: str | Path = "config/rules.yaml",
    wildcard_enabled: bool | None = None,
    players: int | None = None,
    room_mode: str | None = None,
    memory_file: str | Path | None = None,
    reset_memory: bool = False,
    expected_total: int | None = None,
    simulations: int = 0,
    opponent_priority_pending: bool = False,
    logger: GameLogger | None = None,
    frame_id: str | None = None,
    decision_id: str | None = None,
    hidden_cards: list[str] | None = None,
    fast: bool = False,
    protocol_payload: str | Path | dict | None = None,
    precomputed_state: dict | None = None,
    update_memory: bool = True,
    include_recognized_state: bool = False,
    log_precomputed_state: bool = False,
    decision_time_budget_seconds: float | None = None,
    precomputed_hand_labels: list[str] | None = None,
) -> dict:
    config_path = _resolve_project_path(config_path)
    rules_path = _resolve_project_path(rules_path)
    if precomputed_state is None:
        state = inspect_screenshot(
            screenshot,
            config_path=config_path,
            expected_total=expected_total,
            opponent_priority_pending=opponent_priority_pending,
            precomputed_hand_labels=precomputed_hand_labels,
        )
    else:
        state = deepcopy(precomputed_state)
        state["sanity_checks"] = build_sanity_checks(state, expected_total=expected_total)
    if protocol_payload is not None:
        state = _apply_protocol_state(state, protocol_payload, expected_total=expected_total)
    if hidden_cards:
        state = _apply_hidden_cards(state, hidden_cards, expected_total=expected_total)
    state = _clear_stale_options_when_discard_button_visible(state)
    state = _repair_chi_options_before_policy(state)
    rules = rules_for_room(
        rules_path,
        wildcard_enabled=wildcard_enabled,
        players=players,
        room_mode=room_mode,
    )
    state = _inject_semantic_chi_options_before_policy(state, rules)

    frame_id = frame_id or "frame_unknown"
    memory = None
    if memory_file:
        memory = VisionMemory() if reset_memory else load_memory(memory_file)
        state = recover_temporally_hidden_hand(
            state,
            memory,
            expected_total=expected_total,
        )
        if update_memory:
            memory.update_from_snapshot(state)
            if not save_memory(memory, memory_file):
                raise OSError(f"vision_memory_write_failed:{memory_file}")

    if (
        logger is not None
        and logger.round_active
        and (precomputed_state is None or log_precomputed_state)
    ):
        source_screenshot = state.get("screenshot") or str(screenshot)
        logged_screenshot = logger.log_frame_captured(frame_id, source_screenshot)
        state = dict(state)
        state["screenshot"] = logged_screenshot
        state["screenshot_path"] = logged_screenshot
        _save_debug_overlay(logger, frame_id, state, config_path=config_path, screenshot_path=source_screenshot)
        logger.log_frame_recognized(frame_id, _frame_log_payload(state), debug_image=f"{frame_id}_debug.jpg")

    memory_dict = memory.to_dict() if memory is not None else None

    legal_actions = [item["type"] for item in state["legal_actions"]]
    my_existing_groups = [
        [cell["name"] for cell in group]
        for group in state["meld_groups"].get("my_melds", [])
    ]
    if fast:
        existing_xi = 0
        hu_breakdown_dict = {
            "can_hu": False,
            "total_xi": 0,
            "min_xi": 0,
            "hand_groups": [],
            "existing_groups": my_existing_groups,
            "reasons": ["fast_mode_skipped_hu_breakdown"],
        }
    else:
        existing_xi = total_xi(my_existing_groups, config_path=rules_path)
        hu_breakdown_dict = explain_hu(
            state["hand"],
            existing_groups=my_existing_groups,
            config_path=rules_path,
        ).to_dict()

    state_for_policy = dict(state)
    state_for_policy["memory"] = memory_dict
    state_for_policy["room_players"] = int(rules["game"]["players"])
    decision_deadline = (
        None
        if decision_time_budget_seconds is None
        else time.perf_counter() + max(0.1, float(decision_time_budget_seconds))
    )
    decision = choose_action(
        state_for_policy,
        rules=rules,
        config_path=rules_path,
        absolute_deadline=decision_deadline,
    )

    # PolicyDecision.to_dict() deep-copies the complete evaluation tree. Keep
    # one immutable snapshot for the plan, PASS guard, and final result.
    decision_dict = decision.to_dict()
    action_plan = build_action_plan(state, decision_dict)
    if decision.action == "discard" and not state.get("sanity_checks", {}).get("ok", True):
        action_plan = _patch_unsure_action_plan(
            action_plan,
            "手牌数量校验未通过，疑似遮挡或漏识别，暂停出牌",
        )
    if decision.action == "pass":
        pass_guard = _validate_pass_response_context(state, decision_dict)
        if pass_guard is not None:
            action_plan = _patch_action_plan(action_plan, pass_guard, ready=False)
            action_plan["clicks"] = []

    decision_dict["decision_id"] = decision_id
    decision_dict["selected_discard"] = decision_dict.get("selected_discard")
    policy_selected_option_present = bool(
        decision_dict.get("selected_option_id")
        or decision_dict.get("selected_option_cards")
        or decision_dict.get("option_cards")
    )
    action_plan_target = None
    if decision.action == "discard":
        action_plan_target = decision_dict.get("selected_discard")

    if opponent_priority_pending and decision.action == "pass":
        action_plan = {
            "action": "wait_opponent_priority",
            "ready": False,
            "reason": "对手优先权未结束，当前可见吃/过 UI 可能不可点击，等待对手吃或过",
            "clicks": [],
        }

    option_evaluations = [
        evaluate_option(
            item["labels"],
            action="compare" if item["region_name"] == "compare_options" else "chi",
            config_path=rules_path,
        ).to_dict()
        | {"center": item["center"], "region_name": item["region_name"], "index": item["index"]}
        for item in state.get("option_details", [])
    ]

    chi_options = [item for item in state.get("option_details", []) if item["region_name"] == "chi_options"]
    compare_options = [item for item in state.get("option_details", []) if item["region_name"] == "compare_options"]
    option_stage = "compare" if compare_options else "chi"
    stage_options = compare_options if compare_options else chi_options
    stage_region_name = f"{option_stage}_options"
    chosen_option = choose_option(
        [item["labels"] for item in stage_options],
        action=option_stage,
        config_path=rules_path,
    )
    stage_option_evaluations = [
        item for item in option_evaluations if item["region_name"] == stage_region_name
    ]
    all_unclear_stage_options = bool(stage_option_evaluations) and all(
        item["type"] in {"unknown", "complete"} or "不明确" in item.get("reason", "")
        for item in stage_option_evaluations
    )
    if option_stage == "compare" and chosen_option is not None:
        compare_plan = build_compare_option_plan(state, config_path=rules_path)
        if compare_plan is not None:
            action_plan = compare_plan
    elif decision.action == "chi" and chosen_option is not None and not policy_selected_option_present:
        if all_unclear_stage_options:
            chosen_detail = sorted(stage_options, key=lambda item: (item["x"], item["y"]))[0]
            chosen_labels = "".join(chosen_detail["labels"])
            action_plan = _patch_action_plan(
                action_plan,
                f"吃牌候选全部不明确，禁止自动选择候选 {chosen_labels}",
                ready=False,
                target=f"chi:{chosen_labels}",
            )
            action_plan["target"] = f"chi:{chosen_labels}"
            action_plan["target_type"] = "option_column"
            action_plan["target_option_id"] = _option_detail_id(chosen_detail, option_stage)
            action_plan["target_option_cards"] = list(chosen_detail["labels"])
            action_plan["option_center"] = chosen_detail["center"]
            action_plan["clicks"] = []
            action_plan.setdefault("validation", {}).update(
                {
                    "passed": False,
                    "reason_code": "chi_option_recognition_uncertain",
                    "checks": ["option_stage_evaluated", "chi_option_recognition_uncertain"],
                }
            )
        else:
            chosen_detail = next(
                item for item in stage_options if item["labels"] == chosen_option.labels
            )
            chosen_detail, confidence_guard_reason, confidence_guard_ready = _guard_low_confidence_chi_option(
                chosen_detail,
                stage_options,
                option_stage=option_stage,
            )
            original_labels = list(chosen_detail["labels"])
            chosen_label_list = repair_chi_option_labels(original_labels, state)
            chosen_labels = "".join(chosen_label_list)
            if confidence_guard_reason is not None:
                reason = confidence_guard_reason
                ready = confidence_guard_ready
            else:
                reason = (
                    f"选择候选 {''.join(chosen_option.labels)}：{chosen_option.reason}"
                    if chosen_option.type != "unknown"
                    else f"候选 {''.join(chosen_option.labels)} 组合价值不明确，暂停自动点击"
                )
                ready = chosen_option.type != "unknown"
            if chosen_label_list != original_labels:
                reason = f"{reason}；候选识别 {''.join(original_labels)} 按手牌纠正为 {chosen_labels}"
            guard_reason = chi_option_guard_reason(chosen_label_list, state)
            if guard_reason is not None:
                reason = describe_chi_option_guard(guard_reason, chosen_label_list, state)
                ready = False
            action_plan = _patch_action_plan(
                action_plan,
                reason,
                ready=ready,
            )
            action_plan["target"] = f"{option_stage}:{chosen_labels}"
            action_plan["target_type"] = "option_column"
            action_plan["target_option_id"] = _option_detail_id(chosen_detail, option_stage)
            action_plan["target_option_cards"] = chosen_label_list
            action_plan["option_center"] = chosen_detail["center"]
            if action_plan.get("ready"):
                _append_option_click(action_plan, chosen_detail["center"], action_plan["target"])
            else:
                action_plan["clicks"] = []
                if guard_reason is not None:
                    action_plan.setdefault("validation", {}).update(
                        {"passed": False, "reason_code": guard_reason}
                    )
    decision_id = decision_id or (logger.next_decision_id() if logger is not None and logger.round_active else None)
    if logger is not None and logger.round_active and decision_id is None:
        decision_id = logger.next_decision_id()

    primary_is_professional = hasattr(decision, "context_snapshot")
    professional_snapshot = (
        _build_fast_professional_snapshot(state_for_policy, rules, action_plan, decision)
        if fast and logger is None
        else _build_professional_snapshot(state_for_policy, rules, action_plan, primary_decision=decision)
    )
    guard_result = professional_snapshot.get("guard_result") or {}
    action_plan = _attach_self_check_validation(action_plan, guard_result)
    if primary_is_professional and guard_result and not guard_result.get("passed", True):
        action_plan = _patch_action_plan(
            action_plan,
            guard_result.get("reason") or "conflict_unknown",
            ready=False,
        )
        action_plan.setdefault("validation", {}).update(
            {
                "passed": False,
                "reason_code": guard_result.get("reason") or "conflict_unknown",
                "guard_result": guard_result,
            }
        )
        professional_snapshot = (
            _build_fast_professional_snapshot(state_for_policy, rules, action_plan, decision)
            if fast and logger is None
            else _build_professional_snapshot(state_for_policy, rules, action_plan, primary_decision=decision)
        )
        guard_result = professional_snapshot.get("guard_result") or guard_result
        action_plan = _attach_self_check_validation(action_plan, guard_result)
    option_override = _visible_option_override_plan(
        state,
        option_stage=option_stage,
        stage_options=stage_options,
        chosen_option=chosen_option,
        all_unclear=all_unclear_stage_options,
        current_action=decision.action,
    )
    if option_override is not None:
        action_plan = option_override
        professional_snapshot = dict(professional_snapshot)
        professional_snapshot["guard_result"] = {
            "passed": True,
            "safe_halt": False,
            "reason": None,
            "checks": ["visible_option_override", "target_clickable"],
            "details": {},
        }
    action_plan, professional_snapshot = _force_visible_hu_priority(
        state,
        action_plan,
        professional_snapshot,
    )
    action_plan, professional_snapshot = _hold_for_opponent_priority(
        state,
        action_plan,
        professional_snapshot,
    )
    if (
        (state.get("metadata") or {}).get("semantic_chi_pending_card_untrusted")
        and decision.action in {"chi", "expand_chi_options"}
        and any(
            isinstance(click, dict) and click.get("target") == "button:chi"
            for click in action_plan.get("clicks") or []
        )
    ):
        action_plan = {
            "action": "wait_chi_candidate_recheck",
            "ready": False,
            "reason": "进牌尚未可靠识别，复核下一帧后再决定吃或过",
            "clicks": [],
            "validation": {
                "passed": False,
                "reason_code": "pending_card_untrusted",
                "checks": ["pending_card_surface_required_before_chi"],
            },
        }

    if logger is not None and logger.round_active:
        card_pool = _build_hand_card_pool(state.get("hand_details", []))
        _log_structure_analysis(logger, frame_id, decision_id, state, decision, rules, professional_snapshot)
        _log_legal_actions(logger, frame_id, decision_id, state, decision, card_pool, professional_snapshot)
        _log_decision_evaluations(
            logger,
            frame_id=frame_id,
            decision_id=decision_id,
            decision=decision,
            card_pool=card_pool,
            professional_snapshot=professional_snapshot,
        )
        logger.log_decision(
            decision_id=decision_id,
            frame_id=frame_id,
            decision={
                "action": decision.action,
                "label": decision.label,
                "selected_card_id": action_plan.get("target_card_id"),
                "candidate_stage": decision.candidate_stage,
                "hand": state["hand"],
                "selected_reason": decision.selected_reason,
                "reason": decision.reason,
                "evaluations": [item.to_dict() for item in decision.evaluations],
                "response_evaluations": decision.response_evaluations,
                "professional_decision": professional_snapshot.get("decision"),
                "conflict_guard_result": professional_snapshot.get("guard_result"),
            },
            reason=decision.reason,
        )
        logger.log_action_plan(str(decision_id), frame_id, action_plan)
        if decision.action == "safe_halt" or (
            decision.action == "pass" and decision.reason.startswith("SAFE_HALT")
        ):
            reason = decision.reason.replace("SAFE_HALT:", "").strip() or "safe_halt"
            logger.log_safe_halt(
                frame_id=frame_id,
                decision_id=decision_id,
                reason=reason,
                raw_hand=state.get("raw_hand", []),
                hard_protected=decision.hard_protected,
                free_cards=state.get("hand", []),
                clickable_cards=_clickable_cards_from_state(state),
                screenshot_path=str(screenshot),
                debug_image_path=state.get("debug_image_path"),
            )

    simulation_results = []
    if simulations > 0:
        simulation_results = [
            item.to_dict()
            for item in simulate_discards(
                state["hand"],
                memory=memory_dict,
                meld_groups=state["meld_groups"],
                simulations=simulations,
                config_path=rules_path,
                rules=rules,
            )
        ]

    result = {
        "screenshot": state["screenshot"],
        "raw_hand": state.get("raw_hand", state["hand"]),
        "hand": state["hand"],
        "room_profile": rules["room_profile"],
        "decision_id": decision_id,
        "frame_id": frame_id,
        "discard_button": state.get("discard_button"),
        "buttons": state.get("buttons", []),
        "pending_action_card": state.get("pending_action_card"),
        "opponent_pending_card": state.get("opponent_pending_card"),
        "remaining_deck_count": state["remaining_deck_count"],
        "seat_role": state.get("seat_role"),
        "metadata": state.get("metadata", {}),
        "discards": state.get("discards", {}),
        "sanity_checks": state["sanity_checks"],
        "existing_xi": existing_xi,
        "hu_breakdown": hu_breakdown_dict,
        "opponent_profile": infer_opponent_profile(memory_dict),
        "decision": decision_dict,
        "action_plan_target": action_plan_target,
        "action_plan_ready": action_plan.get("ready"),
        "action_plan": action_plan,
        "professional_brain": professional_snapshot,
        "conflict_guard_result": professional_snapshot.get("guard_result"),
        "option_evaluations": option_evaluations,
        "option_details": state.get("option_details", []),
        "option_stage": option_stage if stage_options else None,
        "chosen_option": chosen_option.to_dict() if chosen_option else None,
        "simulations": simulation_results,
        "memory": memory_dict,
    }
    if include_recognized_state:
        result["_recognized_state"] = deepcopy(state)
    return result


def _build_professional_snapshot(
    state: dict,
    rules: dict,
    action_plan: dict,
    *,
    primary_decision=None,
) -> dict:
    try:
        context = build_professional_context(state, rules=rules)
        allocation = allocate_professional_structures(context)
        analysis = analyze_professional_hand(context, allocation)
        decision = (
            primary_decision
            if primary_decision is not None and hasattr(primary_decision, "context_snapshot")
            else choose_professional_action(state, rules=rules)
        )
        guard = run_decision_self_check(context, decision, action_plan)
        return {
            "context": context.to_dict(),
            "structure_allocation": allocation.to_dict(),
            "hand_analysis": analysis.to_dict(),
            "decision": decision.to_dict(),
            "guard_result": guard.to_dict(),
            "live_decision_source": "professional_policy_brain",
        }
    except Exception as exc:
        return {
            "error": type(exc).__name__,
            "message": str(exc),
            "guard_result": {
                "passed": False,
                "safe_halt": True,
                "reason": "conflict_unknown",
                "checks": [],
            },
        }


def _build_fast_professional_snapshot(state: dict, rules: dict, action_plan: dict, decision) -> dict:
    try:
        context = build_professional_context(state, rules=rules)
        guard = run_decision_self_check(context, decision, action_plan)
        return {
            "guard_result": guard.to_dict(),
            "live_decision_source": "professional_policy_brain_fast",
        }
    except Exception as exc:
        return {
            "error": type(exc).__name__,
            "message": str(exc),
            "guard_result": {
                "passed": False,
                "safe_halt": True,
                "reason": "conflict_unknown",
                "checks": [],
            },
        }


def _append_option_click(action_plan: dict, option_center: Any, option_target: str) -> None:
    if not isinstance(action_plan, dict):
        return
    clicks = list(action_plan.get("clicks", []))
    try:
        x = int(option_center[0])
        y = int(option_center[1])
    except Exception:
        return
    new_click = {"target": option_target, "x": x, "y": y, "delay_ms": 80}
    if new_click not in clicks:
        clicks.append(new_click)
        action_plan["clicks"] = clicks


def build_compare_option_plan(
    state: dict,
    *,
    config_path: str | Path = "config/rules.yaml",
) -> dict | None:
    """Map a visible compare surface without rerunning the full play policy."""
    rules_path = _resolve_project_path(config_path)
    stage_options = [
        item
        for item in state.get("option_details", [])
        if item.get("region_name") == "compare_options"
    ]
    if not stage_options:
        return None
    evaluations = [
        evaluate_option(item["labels"], action="compare", config_path=rules_path).to_dict()
        for item in stage_options
    ]
    chosen_option = choose_option(
        [item["labels"] for item in stage_options],
        action="compare",
        config_path=rules_path,
    )
    if chosen_option is None:
        return None
    all_unclear = bool(evaluations) and all(
        item["type"] in {"unknown", "complete"} or "不明确" in item.get("reason", "")
        for item in evaluations
    )
    if all_unclear:
        chosen_detail = sorted(stage_options, key=lambda item: (item["x"], item["y"]))[0]
        chosen_labels = list(chosen_detail["labels"])
        reason = f"比牌/下火候选全部不明确，采用左侧兜底候选 {''.join(chosen_labels)}"
        ready = True
    else:
        chosen_detail = next(
            (item for item in stage_options if item["labels"] == chosen_option.labels),
            None,
        )
        if chosen_detail is None:
            return None
        chosen_labels = list(chosen_option.labels)
        reason = (
            f"选择比牌/下火候选 {''.join(chosen_option.labels)}：{chosen_option.reason}"
            if chosen_option.type != "unknown"
            else f"比牌/下火候选 {''.join(chosen_option.labels)} 组合价值不明确，暂停自动点击"
        )
        ready = chosen_option.type != "unknown"
    action_plan = build_action_plan(
        state,
        {
            "action": "compare_option",
            "selected_option_id": _option_detail_id(chosen_detail, "compare"),
            "selected_option_cards": chosen_labels,
        },
    )
    action_plan = _patch_action_plan(
        action_plan,
        reason,
        ready=ready and bool(action_plan.get("ready")),
        target=f"compare:{''.join(chosen_labels)}",
    )
    if not action_plan.get("ready"):
        action_plan["clicks"] = []
    return action_plan


def _visible_option_override_plan(
    state: dict,
    *,
    option_stage: str,
    stage_options: list[dict],
    chosen_option,
    all_unclear: bool,
    current_action: str,
) -> dict | None:
    if not stage_options:
        return None
    if current_action not in {"chi_option", "compare_option"}:
        return None
    if option_stage == "chi" and current_action != "chi_option":
        return None
    if option_stage == "compare" and current_action != "compare_option":
        return None
    if all_unclear:
        return {
            "action": f"{option_stage}_option",
            "ready": False,
            "reason": "候选区已展开但组合识别不明确，禁止普通出牌",
            "target": None,
            "target_type": "option_column",
            "target_label": None,
            "target_card_id": None,
            "target_option_id": None,
            "target_option_cards": [],
            "clicks": [],
            "policy_selected_action": {"type": f"{option_stage}_option", "label": None},
            "validation": {
                "passed": False,
                "reason_code": "option_recognition_uncertain",
                "checks": ["visible_option_override", "option_recognition_uncertain"],
            },
        }
    if chosen_option is None:
        return {
            "action": f"{option_stage}_option",
            "ready": False,
            "reason": "候选区已展开，但策略没有给出可精确映射的组合，暂停自动点击",
            "target": None,
            "target_type": "option_column",
            "target_label": None,
            "target_card_id": None,
            "target_option_id": None,
            "target_option_cards": [],
            "clicks": [],
            "policy_selected_action": {"type": f"{option_stage}_option", "label": None},
            "validation": {
                "passed": False,
                "reason_code": "policy_option_missing",
                "checks": ["visible_option_override", "policy_option_missing"],
            },
        }
    chosen_detail = next((item for item in stage_options if item["labels"] == chosen_option.labels), None)
    reason = f"候选区已展开，选择 {''.join(chosen_option.labels)}：{chosen_option.reason}"
    if chosen_detail is None:
        return None
    chosen_detail, confidence_guard_reason, confidence_guard_ready = _guard_low_confidence_chi_option(
        chosen_detail,
        stage_options,
        option_stage=option_stage,
    )
    if confidence_guard_reason is not None:
        reason = f"候选区已展开，{confidence_guard_reason}"
    original_labels = list(chosen_detail["labels"])
    labels = repair_chi_option_labels(original_labels, state) if option_stage == "chi" else original_labels
    if labels != original_labels:
        reason = f"{reason}；候选识别 {''.join(original_labels)} 按手牌纠正为 {''.join(labels)}"
    guard_reason = chi_option_guard_reason(labels, state) if option_stage == "chi" else None
    if guard_reason is not None:
        reason = f"候选区已展开，{describe_chi_option_guard(guard_reason, labels, state)}"
        confidence_guard_ready = False
    target = f"{option_stage}:{''.join(labels)}"
    return {
        "action": f"{option_stage}_option",
        "ready": confidence_guard_ready,
        "reason": reason,
        "target": target,
        "target_type": "option_column",
        "target_label": "".join(labels),
        "target_card_id": None,
        "target_option_id": _option_detail_id(chosen_detail, option_stage),
        "target_option_cards": labels,
        "option_center": chosen_detail["center"],
        "clicks": [
            {
                "target": target,
                "x": int(chosen_detail["center"][0]),
                "y": int(chosen_detail["center"][1]),
                "delay_ms": 80,
            }
        ] if confidence_guard_ready else [],
        "policy_selected_action": {"type": f"{option_stage}_option", "label": None, "option_cards": labels},
        "validation": {
            "passed": confidence_guard_ready,
            **({"reason_code": guard_reason} if guard_reason is not None else {}),
            "checks": [
                "visible_option_override",
                "target_clickable" if confidence_guard_ready else (guard_reason or "option_recognition_uncertain"),
            ],
            "target_matches_policy": True,
        },
    }


def _guard_low_confidence_chi_option(
    chosen_detail: dict,
    stage_options: list[dict],
    *,
    option_stage: str | None,
) -> tuple[dict, str | None, bool]:
    if option_stage != "chi":
        return chosen_detail, None, True
    confidence = _option_confidence(chosen_detail)
    if confidence >= CHI_OPTION_MIN_CONFIDENCE:
        return chosen_detail, None, True
    del stage_options
    return (
        chosen_detail,
        f"候选 {''.join(chosen_detail.get('labels') or [])} 置信度 {confidence:.3f} 过低，暂停自动点击",
        False,
    )


def _option_confidence(detail: dict) -> float:
    try:
        return float(detail.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _center_click_from_box(item: dict, *, target: str) -> dict:
    if item.get("center") is not None:
        x, y = item["center"]
        return {"target": target, "x": int(x), "y": int(y), "delay_ms": 80}
    return {
        "target": target,
        "x": int(item["x"] + item["w"] / 2),
        "y": int(item["y"] + item["h"] / 2),
        "delay_ms": 80,
    }


def _option_detail_id(option: dict, prefix: str) -> str:
    return str(option.get("option_id") or f"{prefix}_{int(option.get('index') or 1):03d}")


def _attach_self_check_validation(action_plan: dict, guard_result: dict) -> dict:
    if not isinstance(action_plan, dict) or not guard_result:
        return action_plan
    plan = dict(action_plan)
    validation = dict(plan.get("validation") or {})
    guard_passed = bool(guard_result.get("passed", False))
    validation["passed"] = bool(validation.get("passed", True) and guard_passed)
    validation["self_check"] = "run_decision_self_check"
    validation["guard_result"] = guard_result
    checks = list(validation.get("checks") or [])
    if "run_decision_self_check" not in checks:
        checks.append("run_decision_self_check")
    for check in guard_result.get("checks", []) or []:
        if check not in checks:
            checks.append(check)
    validation["checks"] = checks
    if not guard_passed:
        validation["reason_code"] = guard_result.get("reason") or validation.get("reason_code") or "conflict_unknown"
        plan["ready"] = False
        plan["reason"] = guard_result.get("reason") or plan.get("reason")
    plan["validation"] = validation
    return plan


def _validate_pass_response_context(state: dict, decision: dict) -> str | None:
    buttons = state.get("buttons") or []
    button_names = {
        str(button.get("name") or button.get("type") or "").lower()
        for button in buttons
        if isinstance(button, dict)
    }
    if "pass" not in button_names:
        return None
    response_buttons = button_names & {"chi", "peng", "hu", "pao"}
    if not response_buttons:
        has_option_ui = bool(state.get("option_details") or state.get("options") or state.get("option_stage"))
        if has_option_ui:
            return None
        return "只识别到过按钮，疑似吃碰胡按钮漏识别，暂停自动过"
    if "hu" in response_buttons:
        return "胡按钮可见，禁止自动过"

    eval_types = {
        str(item.get("type") or "").lower()
        for item in decision.get("action_evals", []) or []
        if isinstance(item, dict)
    }
    missing = sorted(name for name in response_buttons if name not in eval_types)
    if missing:
        return "响应按钮未完整进入策略评估，暂停自动过：" + ",".join(missing)
    return None


def _visible_hu_override_plan(state: dict, decision: dict) -> dict | None:
    action = str(decision.get("action") or "").lower()
    if action == "hu":
        return None
    buttons = state.get("buttons") or []
    has_hu_button = any(
        isinstance(button, dict)
        and str(button.get("name") or button.get("type") or "").lower() == "hu"
        for button in buttons
    )
    has_hu_legal_action = any(
        str(action.get("type") if isinstance(action, dict) else action).lower() == "hu"
        for action in state.get("legal_actions") or []
    )
    if not has_hu_button and not has_hu_legal_action:
        return None
    plan = build_action_plan(state, {"action": "hu", "label": None})
    if plan.get("ready"):
        plan["reason"] = "胡按钮可见，强制优先胡"
    else:
        plan["reason"] = "规则允许胡，但当前画面未识别到可点击的胡按钮，暂停执行"
        plan.setdefault("validation", {}).update(
            {
                "passed": False,
                "reason_code": "hu_button_not_visible",
                "checks": ["visible_hu_override", "hu_button_not_visible"],
            }
        )
    return plan


def _force_visible_hu_priority(
    state: dict,
    action_plan: dict,
    professional_snapshot: dict,
) -> tuple[dict, dict]:
    current_action = str(action_plan.get("action") or "").lower()
    if current_action == "hu" and not action_plan.get("ready"):
        forced_hu_plan = {
            **action_plan,
            "reason": "规则允许胡，但当前画面未识别到可点击的胡按钮，暂停执行",
            "validation": {
                **(action_plan.get("validation") or {}),
                "passed": False,
                "reason_code": "hu_button_not_visible",
                "checks": ["visible_hu_override", "hu_button_not_visible"],
            },
        }
    else:
        forced_hu_plan = _visible_hu_override_plan(state, action_plan)
    if forced_hu_plan is None:
        return action_plan, professional_snapshot
    contract_error = action_plan_contract_error(forced_hu_plan)
    if not forced_hu_plan.get("ready") or contract_error is not None:
        reason_code = str(
            (forced_hu_plan.get("validation") or {}).get("reason_code")
            or contract_error
            or "hu_button_not_visible"
        )
        forced_hu_plan.setdefault("validation", {}).update(
            {
                "passed": False,
                "reason_code": reason_code,
            }
        )
        snapshot = dict(professional_snapshot)
        snapshot["guard_result"] = {
            "passed": False,
            "safe_halt": True,
            "reason": reason_code,
            "checks": ["visible_hu_override", reason_code],
            "details": {"action_plan_contract_error": contract_error},
        }
        return forced_hu_plan, snapshot
    forced_hu_plan.setdefault("validation", {}).update(
        {
            "passed": True,
            "checks": ["visible_hu_override", "target_clickable"],
            "reason_code": "visible_hu_priority",
        }
    )
    snapshot = dict(professional_snapshot)
    snapshot["guard_result"] = {
        "passed": True,
        "safe_halt": False,
        "reason": None,
        "checks": ["visible_hu_override", "target_clickable"],
        "details": {},
    }
    return forced_hu_plan, snapshot


def _hold_for_opponent_priority(
    state: dict,
    action_plan: dict,
    professional_snapshot: dict,
) -> tuple[dict, dict]:
    if not _is_opponent_priority_state(state):
        return action_plan, professional_snapshot
    if _plan_targets_visible_response_button(action_plan, state):
        return action_plan, professional_snapshot
    plan = {
        "action": "wait_opponent_priority",
        "ready": False,
        "reason": "对手抓牌/待操作区优先权未结束，禁止点击吃碰胡过或候选",
        "target": None,
        "clicks": [],
        "policy_selected_action": {"type": "wait_opponent_priority", "label": None},
        "validation": {
            "passed": False,
            "reason_code": "opponent_priority_pending",
            "checks": ["opponent_priority_guard"],
        },
    }
    snapshot = dict(professional_snapshot)
    snapshot["guard_result"] = {
        "passed": True,
        "safe_halt": False,
        "reason": None,
        "checks": ["opponent_priority_guard"],
        "details": {},
    }
    return plan, snapshot


def _is_opponent_priority_state(state: dict) -> bool:
    if state.get("discard_button") is not None:
        return False
    metadata = state.get("metadata") or {}
    if not metadata.get("opponent_priority_pending"):
        return False
    # The card may remain in opponent_pending_card after the opponent passes and
    # our UI becomes truly actionable. Do not infer priority from that visual
    # region alone; require protocol/manual metadata to set this flag.
    button_names = {
        str(button.get("name") or button.get("type") or "").lower()
        for button in state.get("buttons") or []
        if isinstance(button, dict)
    }
    has_response_ui = bool(button_names & RESPONSE_ACTION_NAMES)
    has_option_ui = bool(state.get("option_details") or state.get("options") or state.get("option_stage"))
    return has_response_ui or has_option_ui


def _plan_targets_visible_response_button(action_plan: dict, state: dict) -> bool:
    action = str(action_plan.get("action") or "").lower()
    if action not in RESPONSE_ACTION_NAMES:
        return False
    if not action_plan.get("ready"):
        return False
    button_names = {
        str(button.get("name") or button.get("type") or "").lower()
        for button in state.get("buttons") or []
        if isinstance(button, dict)
    }
    return action in button_names


def _patch_unsure_action_plan(action_plan: dict, reason: str) -> dict:
    plan = dict(action_plan)
    if isinstance(action_plan, dict):
        plan["ready"] = False
        plan["reason"] = reason
        plan["clicks"] = []
        plan.setdefault("validation", {}).update({"passed": False, "reason_code": "decision_sanity_failed"})
        plan["target_type"] = plan.get("target_type", "unknown")
    return plan


def _patch_action_plan(action_plan: dict, reason: str, ready: bool = True, target: str | None = None) -> dict:
    plan = dict(action_plan)
    if isinstance(action_plan, dict):
        plan["ready"] = ready
        plan["reason"] = reason
        if target is not None:
            plan["target"] = target
        plan.setdefault("validation", {}).update({"passed": ready, "reason_code": "option_stage_evaluated"})
    return plan


def _save_debug_overlay(
    logger: GameLogger,
    frame_id: str,
    state: dict,
    *,
    config_path: str | Path,
    screenshot_path: str | Path,
) -> None:
    if not logger.config.save_debug_screenshot or not logger.round_active:
        return
    output = logger.paths.round_crops_root(logger.session_id, logger.round_id) / f"{frame_id}_debug.jpg"
    try:
        write_preview(
            screenshot_path,
            state,
            config_path=config_path,
            output=output,
            max_width=900,
        )
    except Exception:
        # Debug overlay is best-effort; do not block decision flow.
        pass


def _frame_log_payload(state: dict) -> dict:
    return {
        "phase": str(state["phase"]),
        "flow_state": "play",
        "screenshot_path": state.get("screenshot"),
        "raw_hand": state.get("raw_hand", []),
        "normalized_hand": state.get("hand", []),
        "hand_count": len(state.get("hand", [])),
        "hand_details": state.get("hand_details", []),
        "buttons": state.get("buttons", []),
        "chi_options": [item for item in state.get("options", []) if item.get("region_name") == "chi_options"],
        "compare_options": [item for item in state.get("options", []) if item.get("region_name") == "compare_options"],
        "my_melds": state.get("melds", {}).get("my_melds", []),
        "left_melds": state.get("melds", {}).get("left_melds", []),
        "discards": state.get("discards", {}),
        "remaining_cards_estimate": state.get("remaining_deck_count"),
        "opponent_priority_pending": state.get("metadata", {}).get("opponent_priority_pending"),
        "auto_quad_pending": state.get("metadata", {}).get("auto_quad_pending"),
        "hand_recognition": state.get("hand_recognition")
        or state.get("metadata", {}).get("hand_recognition", {}),
        "recognition_warnings": state.get("recognition_warnings", []),
        "recognition_errors": state.get("recognition_errors", []),
        "debug_image_path": None,
    }


def _build_hand_card_pool(hand_details: list[dict[str, Any]]) -> dict[str, list[str]]:
    pool = defaultdict(list)
    for item in hand_details:
        label = normalize_card_label(item.get("name", ""))
        if not label:
            continue
        card_id = item.get("card_id")
        if card_id:
            pool[label].append(card_id)
    return pool


def _consume_card_id(pool: dict[str, list[str]], label: str) -> str | None:
    items = pool.get(label, [])
    if not items:
        return None
    return items.pop(0)


def _clickable_cards_from_state(state: dict) -> list[str]:
    return [
        normalize_card_label(card["name"])
        for card in state.get("hand_details", [])
        if card.get("clickable", True)
    ]


def _log_structure_analysis(
    logger: GameLogger,
    frame_id: str,
    decision_id: str,
    state: dict,
    decision,
    rules: dict,
    professional_snapshot: dict | None = None,
) -> None:
    professional_snapshot = professional_snapshot or {}
    allocation_payload = professional_snapshot.get("structure_allocation")
    analysis_payload = professional_snapshot.get("hand_analysis")
    if not allocation_payload:
        context = build_professional_context(state, rules=rules)
        allocation_payload = allocate_professional_structures(context).to_dict()
    if not analysis_payload:
        context = build_professional_context(state, rules=rules)
        allocation = allocate_professional_structures(context)
        analysis_payload = analyze_professional_hand(context, allocation).to_dict()
    allocation_payload["decision_id"] = decision_id
    logger.log_structure_allocation(
        frame_id=frame_id,
        decision_id=decision_id,
        allocation=allocation_payload,
    )
    logger.log_hand_analysis(
        frame_id=frame_id,
        decision_id=decision_id,
        analysis=analysis_payload,
    )


def _log_legal_actions(
    logger: GameLogger,
    frame_id: str,
    decision_id: str,
    state: dict,
    decision,
    card_pool: dict[str, list[str]],
    professional_snapshot: dict | None = None,
) -> None:
    professional_snapshot = professional_snapshot or {}
    pro_context = professional_snapshot.get("decision", {}).get("context_snapshot", {})
    if pro_context.get("legal_actions") is not None:
        logger.log_legal_actions(
            frame_id=frame_id,
            decision_id=decision_id,
            legal_actions=pro_context.get("legal_actions", []),
            rejected_actions=pro_context.get("rejected_actions", []),
        )
        return

    legal_actions: list[dict[str, Any]] = []
    for index, action in enumerate(state.get("legal_actions", []), start=1):
        action_type = action.get("type")
        label = None
        if action_type == "discard":
            label = _consume_label_candidate(card_pool, decision)
        legal_actions.append(
            {
                "action_id": f"a{index:03d}",
                "type": action_type,
                "label": label,
                "source": "hand" if action_type == "discard" else "button",
                "requires_button": action_type != "discard",
                "requires_option": action_type in {"chi", "peng"},
                "allowed_by_rules": True,
            }
        )
    rejected_actions = _collect_rejected_actions(decision)
    logger.log_legal_actions(
        frame_id=frame_id,
        decision_id=decision_id,
        legal_actions=legal_actions,
        rejected_actions=rejected_actions,
    )


def _consume_label_candidate(pool: dict[str, list[str]], decision) -> str | None:
    if decision.action != "discard" or not getattr(decision, "label", None):
        return None
    return _consume_card_id(pool, normalize_card_label(decision.label))


def _collect_rejected_actions(decision) -> list[dict[str, Any]]:
    if decision.action != "discard":
        return []
    reject_reason = (
        "hard_protected_card" if decision.candidate_stage == "forced_break_hard_protection"
        else "candidate_pool_empty_after_filter" if decision.candidate_stage == "safe_halt"
        else "candidate_not_selected"
    )
    return [
        {
            "type": "DISCARD",
            "label": label,
            "reason": reject_reason,
        }
        for label in decision.discard_candidates_after_hard_filter or []
    ]


def _log_decision_evaluations(
    logger: GameLogger,
    *,
    frame_id: str,
    decision_id: str,
    decision,
    card_pool: dict[str, list[str]],
    professional_snapshot: dict | None = None,
) -> None:
    professional_snapshot = professional_snapshot or {}
    pro_evals = professional_snapshot.get("decision", {}).get("action_evals", [])
    if pro_evals:
        logger.log_action_evaluation_started(
            frame_id=frame_id,
            decision_id=decision_id,
            candidate_count=len(pro_evals),
            source="professional_policy_brain",
        )
        for payload in pro_evals:
            logger.log_action_evaluated(
                frame_id=frame_id,
                decision_id=decision_id,
                eval_payload=payload,
                is_rejected=not payload.get("allowed", True),
            )
        return

    eval_payloads: list[dict[str, Any]] = []
    action_stage = decision.candidate_stage
    for item in decision.evaluations:
        card_label = normalize_card_label(item.label)
        card_id = _consume_card_id(card_pool, card_label)
        is_reject = False
        if decision.action == "discard":
            allowed = decision.label == item.label
        else:
            allowed = True
        eval_payloads.append(
            {
                "action_id": f"e{item.label}",
                "type": "DISCARD",
                "card_id": card_id,
                "label": item.label,
                "allowed": allowed,
                "candidate_stage": action_stage,
                "ev": item.score,
                "score_breakdown": {
                    "hand_value_after": item.after_xi_potential * 20,
                    "orphan_score": item.orphan_score,
                    "xi_gain": 0,
                    "xi_loss": 0,
                    "red_black_gain": 0,
                    "wildcard_loss": 0,
                    "structure_loss": item.structure_loss,
                    "danger_loss": item.danger,
                    "opponent_gain_risk": 0,
                    "uncertainty_penalty": 0,
                },
                "breaks_melds": item.breaks_melds,
                "danger_reasons": item.penalties,
                "reason": "；".join(item.reasons) if item.reasons else item.reason if hasattr(item, "reason") else "评分依据",
                "reject_reason": None if is_reject else None,
            }
        )
    eval_payloads.sort(key=lambda item: item["ev"], reverse=True)
    logger.log_action_evaluation_started(
        frame_id=frame_id,
        decision_id=decision_id,
        candidate_count=len(eval_payloads),
        source="legacy_compatibility",
    )
    for payload in eval_payloads:
        logger.log_action_evaluated(frame_id=frame_id, decision_id=decision_id, eval_payload=payload, is_rejected=not payload["allowed"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend an AlphaDog action from screenshot vision.")
    parser.add_argument("screenshot")
    parser.add_argument("--config", default="config/screen_1080x2400.yaml")
    parser.add_argument("--rules", default="config/rules.yaml")
    parser.add_argument("--wildcard-mode", choices=("config", "on", "off"), default="config")
    parser.add_argument("--players", type=int, choices=(2, 3), default=None)
    parser.add_argument("--room-mode", choices=tuple(ROOM_MODE_PRESETS), default=None)
    parser.add_argument("--memory-file", default=None)
    parser.add_argument("--reset-memory", action="store_true")
    parser.add_argument("--expected-total", type=int, default=None)
    parser.add_argument("--simulate", type=int, default=0)
    parser.add_argument("--decision-log", default=None)
    parser.add_argument("--opponent-priority-pending", action="store_true")
    parser.add_argument("--protocol-payload", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    wildcard_enabled = None if args.wildcard_mode == "config" else args.wildcard_mode == "on"
    result = recommend_from_screenshot(
        args.screenshot,
        config_path=args.config,
        rules_path=args.rules,
        wildcard_enabled=wildcard_enabled,
        players=args.players,
        room_mode=args.room_mode,
        memory_file=args.memory_file,
        reset_memory=args.reset_memory,
        expected_total=args.expected_total,
        simulations=args.simulate,
        opponent_priority_pending=args.opponent_priority_pending,
        protocol_payload=args.protocol_payload,
    )
    if args.json:
        if args.decision_log:
            append_decision(args.decision_log, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    decision = result["decision"]
    print("hand=" + " ".join(result["hand"]))
    print(f"remaining_deck_count={result['remaining_deck_count']}")
    print(f"existing_xi={result['existing_xi']}")
    hu = result["hu_breakdown"]
    print(
        "hu_breakdown="
        + f"can_hu={hu['can_hu']} total_xi={hu['total_xi']} min_xi={hu['min_xi']} "
        + f"groups={' | '.join(' '.join(group) for group in hu['hand_groups'])}"
    )
    print(
        "sanity="
        + f"controlled={result['sanity_checks']['controlled_card_count']} "
        + f"ok={result['sanity_checks']['ok']}"
    )
    print(f"recommend={decision['action']} {decision['label'] or ''}".strip())
    print(f"score={decision['score']}")
    print(f"reason={decision['reason']}")
    plan = result["action_plan"]
    print(f"action_plan=ready={plan['ready']} reason={plan['reason']} clicks={plan['clicks']}")
    for item in decision["evaluations"][:8]:
        print(
            f"candidate={item['label']} score={item['score']} xi={item['after_xi_potential']} "
            + f"danger={item['danger']} reasons={';'.join(item['reasons'])} "
            + f"penalties={';'.join(item['penalties'])}"
        )
    for item in decision["response_evaluations"] or []:
        print(f"response_candidate={item['action']} score={item['score']} reason={item['reason']}")
    for item in result["option_evaluations"]:
        print(
            f"option_candidate={''.join(item['labels'])} score={item['score']} "
            + f"type={item['type']} xi={item['xi']} center={item['center']} reason={item['reason']}"
        )
    if result["chosen_option"]:
        option = result["chosen_option"]
        print(
            f"chosen_option_stage={result['option_stage']} chosen_option={''.join(option['labels'])} "
            + f"score={option['score']} type={option['type']} reason={option['reason']}"
        )
    if result["simulations"]:
        print("simulations=")
        for item in result["simulations"][:8]:
            print(
                f"  candidate={item['label']} avg_xi_after_draw={item['avg_xi_after_draw']} "
                + f"improve_rate={item['improve_rate']} avg_score={item['avg_score']}"
            )
    profile = result["opponent_profile"]
    print(
        "opponent_profile="
        + f"{profile['profile']} red_pressure={profile['red_pressure']} meld_count={profile['meld_count']}"
    )
    if args.decision_log:
        output = append_decision(args.decision_log, result)
        print(f"decision_log={output}")


if __name__ == "__main__":
    main()
