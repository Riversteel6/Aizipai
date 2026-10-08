"""Normalize runtime protocol fields from the APK into AI state."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from engine.cards import BIG_LABELS, SMALL_LABELS, normalize_card_label

CARD_CODE_MAP: dict[str, str] = {
    **{f"{index}s": label for index, label in zip(range(1, 10), SMALL_LABELS[:9])},
    "Ts": "十",
    **{f"{index}b": label for index, label in zip(range(1, 10), BIG_LABELS[:9])},
    "Tb": "拾",
    "Qh": "王",
}
PROTOCOL_FIELD_NAMES = {
    "playerholdcards",
    "curr_card",
    "out_cards",
    "canpeng",
    "canchi",
    "canhu",
    "showGuo",
    "huxi",
    "hupai",
    "precardval",
    "isoutcard",
    "huzi_cardval",
    "card_num",
    "gamestate",
    "dealer",
    "zhuang",
    "zuozhuang",
    "douzhuangChairId",
    "srcdouzhuangChairId",
    "chairId",
    "chair_id",
    "seatid",
    "drawChairId",
    "draw_chair_id",
    "selfChairId",
    "self_chair_id",
    "myChairId",
    "localChairId",
}


def load_protocol_payload(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        nested = _payload_text_from_hook_message(source)
        if nested:
            parsed = _parse_text_payload(nested)
            return {**source, **parsed}
        return dict(source)
    text = _read_text_or_value(source)
    return _parse_text_payload(text)


def _parse_text_payload(text: str) -> dict[str, Any]:
    if "\n" in text:
        latest: dict[str, Any] | None = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = _parse_text_payload(line)
            if any(key in parsed for key in PROTOCOL_FIELD_NAMES):
                latest = parsed
        if latest is not None:
            return latest
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            nested = _payload_text_from_hook_message(payload)
            if nested:
                return {**payload, **_parse_text_payload(nested)}
            return payload
        return {"payload": payload}
    except json.JSONDecodeError:
        return _parse_key_value_protocol_text(text)


def protocol_state_from_payload(payload: str | Path | dict[str, Any]) -> dict[str, Any]:
    data = load_protocol_payload(payload)
    hand = normalize_protocol_cards(_resolve_hand_cards(data))
    pending_card = normalize_protocol_card(_first_present(data, "curr_card", "precardval", "huzi_cardval"))
    out_cards = normalize_protocol_cards(_first_present(data, "out_cards", "outcards", "discardcards"))
    legal_actions = _legal_actions_from_payload(data)
    seat_fields = _seat_fields_from_payload(data)
    state = {
        "hand": hand,
        "raw_hand": hand,
        "hand_count": len(hand),
        "pending_card": pending_card,
        "discards": {"protocol_out_cards": out_cards},
        "legal_actions": legal_actions,
        "buttons": _buttons_from_legal_actions(legal_actions),
        "option_details": _option_details_from_payload(data),
        "remaining_deck_count": _as_int(_first_present(data, "card_num", "remaincards", "leftcards")),
        "expected_count": _as_int(data.get("expected_count")),
        "controlled_count": _as_int(data.get("controlled_count")),
        "metadata": {
            "source": "protocol_fields",
            "protocol_fields_present": sorted(key for key in PROTOCOL_FIELD_NAMES if key in data),
            "screenshot_fallback_required": not bool(hand),
            "seat_fields": seat_fields,
            "opponent_priority_pending": _opponent_priority_pending_from_payload(data, legal_actions),
        },
    }
    return state


def normalize_protocol_card(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("card", "cardval", "card_val", "val", "value", "name", "label"):
            label = normalize_protocol_card(value.get(key))
            if label:
                return label
        return ""
    if isinstance(value, (list, tuple)):
        return normalize_protocol_card(value[0]) if value else ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return _numeric_card_label(value)
    text = str(value).strip()
    if not text:
        return ""
    if text in CARD_CODE_MAP:
        return CARD_CODE_MAP[text]
    if re.fullmatch(r"-?\d+", text):
        return _numeric_card_label(int(text))
    return normalize_card_label(text)


def normalize_protocol_cards(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = re.split(r"[\s,|;]+", text)
        return normalize_protocol_cards(parsed)
    if isinstance(value, dict):
        for key in ("cards", "list", "values", "playerholdcards"):
            if key in value:
                return normalize_protocol_cards(value[key])
        single = normalize_protocol_card(value)
        return [single] if single else []
    if not isinstance(value, (list, tuple)):
        single = normalize_protocol_card(value)
        return [single] if single else []
    cards: list[str] = []
    for item in value:
        label = normalize_protocol_card(item)
        if label:
            cards.append(label)
    return cards


def _numeric_card_label(value: int) -> str:
    if 1 <= value <= 10:
        return SMALL_LABELS[value - 1]
    if 11 <= value <= 20:
        return BIG_LABELS[value - 11]
    if 101 <= value <= 110:
        return BIG_LABELS[value - 101]
    if 201 <= value <= 210:
        return SMALL_LABELS[value - 201]
    return str(value)


def _legal_actions_from_payload(data: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for field, action in (("canhu", "hu"), ("canpeng", "peng"), ("canchi", "chi"), ("showGuo", "pass")):
        if _truthy(data.get(field)):
            actions.append({"type": action, "source": "protocol", "field": field})
    return actions


def _buttons_from_legal_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": str(action.get("type")), "source": "protocol"} for action in actions if action.get("type") != "discard"]


def _seat_fields_from_payload(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "dealer",
        "zhuang",
        "zuozhuang",
        "douzhuangChairId",
        "srcdouzhuangChairId",
        "chairId",
        "chair_id",
        "seatid",
        "drawChairId",
        "draw_chair_id",
        "selfChairId",
        "self_chair_id",
        "myChairId",
        "localChairId",
    )
    return {key: data[key] for key in keys if key in data}


def _opponent_priority_pending_from_payload(data: dict[str, Any], legal_actions: list[dict[str, Any]]) -> bool:
    action_types = {str(action.get("type")) for action in legal_actions if isinstance(action, dict)}
    if not action_types & {"hu", "peng", "chi", "pass"}:
        return False
    self_chair = _first_int_present(data, "selfChairId", "self_chair_id", "myChairId", "localChairId")
    prompt_chair = _first_int_present(data, "chairId", "chair_id", "seatid")
    if self_chair is not None and prompt_chair is not None and self_chair == prompt_chair:
        return False
    source_chair = _first_int_present(data, "drawChairId", "draw_chair_id", "chairId", "chair_id", "seatid")
    if self_chair is None or source_chair is None or self_chair == source_chair:
        return False
    return True


def _first_int_present(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in data:
            continue
        try:
            return int(data[key])
        except (TypeError, ValueError):
            continue
    return None


def _option_details_from_payload(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_options = _first_present(data, "chi_options", "canchi_cards", "chiCards")
    options = raw_options if isinstance(raw_options, list) else []
    details: list[dict[str, Any]] = []
    for index, option in enumerate(options, start=1):
        labels = normalize_protocol_cards(option)
        if labels:
            details.append(
                {
                    "region_name": "chi_options",
                    "index": index,
                    "labels": labels,
                    "confidence": 1.0,
                    "source": "protocol",
                }
            )
    return details


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _resolve_hand_cards(data: dict[str, Any]) -> Any:
    cards = _first_present(data, "playerholdcards", "holdcards", "hand")
    if not isinstance(cards, dict):
        return cards
    for key in ("selfChairId", "self_chair_id", "chairId", "chair_id", "seatid", "winsite"):
        seat = data.get(key)
        if seat in cards:
            return cards[seat]
        if str(seat) in cards:
            return cards[str(seat)]
    for value in cards.values():
        if isinstance(value, list):
            return value
    return []


def _payload_text_from_hook_message(data: dict[str, Any]) -> str:
    for key in ("text", "message", "payload"):
        value = data.get(key)
        if isinstance(value, str) and any(field in value for field in PROTOCOL_FIELD_NAMES):
            return value
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_text_or_value(source: str | Path) -> str:
    path = Path(source)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if exists:
        return path.read_text(encoding="utf-8", errors="ignore")
    return str(source)


def _parse_key_value_protocol_text(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pattern = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(?P<value>\[[^\]]*\]|true|false|-?\d+|\"[^\"]*\"|'[^']*'|[^\s,;]+)")
    for match in pattern.finditer(text):
        key = match.group("key")
        if key not in PROTOCOL_FIELD_NAMES:
            continue
        value = match.group("value").strip()
        try:
            result[key] = json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            result[key] = value.strip("\"'")
    return result
