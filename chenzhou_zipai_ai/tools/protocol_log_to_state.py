"""Convert a Frida/logcat protocol probe log into the latest normalized state."""

from __future__ import annotations

import argparse
import copy
import ipaddress
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vision.protocol_state import protocol_state_from_payload
from vision.protocol_state import normalize_protocol_cards
from tools.qs_protocol_parser import QSProtocolError, parse_packet_to_state


class ProtocolLogStateTracker:
    def __init__(self) -> None:
        self.latest: dict | None = None
        self.latest_hand: dict | None = None
        self.tracked_hand: list[str] | None = None
        self.tracked_warnings: list[dict] = []
        self.self_seat: int | None = None
        self.last_draw_seat: int | None = None
        self.last_turn_event: dict | None = None

    def update_line(self, line: str) -> dict | None:
        if _is_likely_outbound_qs_row(line):
            return None
        state = _state_from_line(line)
        if not _has_meaningful_protocol_state(state):
            return None
        payload = (state.get("metadata") or {}).get("qs_payload") or {}
        self.self_seat = _updated_self_seat(self.self_seat, payload)
        self.last_draw_seat = _updated_last_draw_seat(self.last_draw_seat, payload)
        self.last_turn_event = _updated_last_turn_event(self.last_turn_event, payload)
        state = _annotate_priority_from_qs(state, self.self_seat, self.last_draw_seat, self.last_turn_event)
        state = _annotate_discard_turn_from_qs(state, payload, self.self_seat)
        self.tracked_hand = _update_tracked_hand(
            self.tracked_hand,
            payload,
            self.self_seat,
            self.tracked_warnings,
        )
        if state.get("hand"):
            self.latest_hand = state
        self.latest = _merge_latest_hand(self.latest_hand, state)
        if self.tracked_hand is not None:
            self.latest = _override_with_tracked_hand(self.latest, self.tracked_hand, self.tracked_warnings)
        return copy.deepcopy(self.latest)

    def current_state(self) -> dict:
        return copy.deepcopy(self.latest) if self.latest is not None else protocol_state_from_payload({})


class ProtocolLogTail:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.position = 0
        self.tracker = ProtocolLogStateTracker()
        self._prefix: bytes | None = None

    def read_latest(self) -> dict:
        if not self.path.exists():
            state = self.tracker.current_state()
            metadata = dict(state.get("metadata") or {})
            metadata["protocol_tail"] = {
                "path": str(self.path),
                "offset": self.position,
                "rows_read": 0,
                "state_updated": False,
                "error": "file_not_found",
            }
            state["metadata"] = metadata
            return state
        size = self.path.stat().st_size
        current_prefix = self._read_prefix()
        prefix_changed = (
            self.position > 0
            and self._prefix not in (None, b"")
            and not current_prefix.startswith(self._prefix)
        )
        if size < self.position or prefix_changed:
            self.position = 0
            self.tracker = ProtocolLogStateTracker()
            self._prefix = current_prefix
        elif self._prefix is None or (self._prefix == b"" and current_prefix):
            self._prefix = current_prefix
        updated = False
        rows_read = 0
        with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(self.position)
            for line in handle:
                rows_read += 1
                if self.tracker.update_line(line) is not None:
                    updated = True
            self.position = handle.tell()
        state = self.tracker.current_state()
        metadata = dict(state.get("metadata") or {})
        metadata["protocol_tail"] = {
            "path": str(self.path),
            "offset": self.position,
            "rows_read": rows_read,
            "state_updated": updated,
        }
        state["metadata"] = metadata
        return state

    def _read_prefix(self, limit: int = 256) -> bytes:
        try:
            with self.path.open("rb") as handle:
                return handle.read(limit)
        except OSError:
            return b""


def latest_protocol_state_from_log(path: str | Path) -> dict:
    log_path = Path(path)
    tracker = ProtocolLogStateTracker()
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tracker.update_line(line)
    return tracker.current_state()


def _state_from_line(line: str) -> dict:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and ("packet_hex" in payload or "packet_base64" in payload):
        try:
            return parse_packet_to_state(payload)
        except QSProtocolError:
            return protocol_state_from_payload(line)
    return protocol_state_from_payload(line)


def _has_meaningful_protocol_state(state: dict) -> bool:
    metadata = state.get("metadata") or {}
    fields = metadata.get("protocol_fields_present") or []
    if fields:
        return True
    return bool(state.get("hand") or state.get("pending_card"))


def _merge_latest_hand(hand_state: dict | None, state: dict) -> dict:
    if state.get("hand") or not hand_state or not hand_state.get("hand"):
        return state
    merged = copy.deepcopy(state)
    hand = list(hand_state.get("hand") or [])
    merged["hand"] = hand
    merged["raw_hand"] = list(hand_state.get("raw_hand") or hand)
    merged["hand_count"] = len(hand)
    if not merged.get("expected_count"):
        merged["expected_count"] = hand_state.get("expected_count")
    if not merged.get("controlled_count"):
        merged["controlled_count"] = hand_state.get("controlled_count")
    metadata = dict(merged.get("metadata") or {})
    hand_metadata = hand_state.get("metadata") or {}
    metadata["merged_hand_from_qs_cmd"] = hand_metadata.get("qs_cmd")
    metadata["merged_hand_seat_fields"] = hand_metadata.get("seat_fields") or {}
    merged["metadata"] = metadata
    return merged


def _updated_self_seat(current: int | None, payload: dict) -> int | None:
    cmd = payload.get("cmd")
    if cmd == 1001 or current is None and cmd == 1014:
        value = payload.get("chairId", payload.get("seatid"))
        try:
            return int(value)
        except (TypeError, ValueError):
            return current
    return current


def _updated_last_draw_seat(current: int | None, payload: dict) -> int | None:
    if payload.get("cmd") != 1013:
        return current
    value = payload.get("chairId", payload.get("seatid"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return current


def _updated_last_turn_event(current: dict | None, payload: dict) -> dict | None:
    cmd = payload.get("cmd")
    if cmd == 1013:
        return {"type": "draw", "seat": _payload_seat(payload), "card": payload.get("val")}
    if cmd == 1012 and payload.get("err") == 0 and _as_int(payload.get("action_type")) == 11:
        return {"type": "discard", "seat": _payload_seat(payload), "cards": list(payload.get("cards") or [])}
    return current


def _annotate_priority_from_qs(
    state: dict,
    self_seat: int | None,
    last_draw_seat: int | None,
    last_turn_event: dict | None,
) -> dict:
    if self_seat is None and last_draw_seat is None and last_turn_event is None:
        return state
    merged = copy.deepcopy(state)
    metadata = dict(merged.get("metadata") or {})
    seat_fields = dict(metadata.get("seat_fields") or {})
    if self_seat is not None:
        metadata["self_chair_id"] = self_seat
        seat_fields.setdefault("selfChairId", self_seat)
    if last_draw_seat is not None:
        metadata["last_draw_chair_id"] = last_draw_seat
        seat_fields.setdefault("drawChairId", last_draw_seat)
    if last_turn_event is not None:
        metadata["last_turn_event"] = dict(last_turn_event)
    metadata["seat_fields"] = seat_fields
    event_type = (last_turn_event or {}).get("type")
    event_seat = (last_turn_event or {}).get("seat")
    payload = metadata.get("qs_payload") or {}
    prompt_seat = _payload_seat(payload)
    if (
        _has_response_action(merged)
        and self_seat is not None
        and event_type == "discard"
        and event_seat is not None
        and self_seat != event_seat
    ):
        metadata["opponent_priority_pending"] = False
        metadata.pop("opponent_priority_source", None)
        metadata["response_source"] = "qs_opponent_discard"
        metadata["response_source_chair_id"] = event_seat
    elif (
        _has_response_action(merged)
        and payload.get("cmd") == 1014
        and self_seat is not None
        and prompt_seat == self_seat
    ):
        metadata["opponent_priority_pending"] = False
        metadata.pop("opponent_priority_source", None)
        metadata["response_source"] = "qs_self_prompt"
        if event_seat is not None and event_seat != self_seat:
            metadata["response_source_chair_id"] = event_seat
        elif last_draw_seat is not None and last_draw_seat != self_seat:
            metadata["response_source_chair_id"] = last_draw_seat
    elif (
        _has_response_action(merged)
        and self_seat is not None
        and event_type == "draw"
        and event_seat is not None
        and self_seat != event_seat
    ):
        metadata["opponent_priority_pending"] = True
        metadata["opponent_priority_source"] = "qs_last_turn_draw"
        metadata["response_source_chair_id"] = event_seat
    merged["metadata"] = metadata
    return merged


def _has_response_action(state: dict) -> bool:
    return any(
        str(action.get("type") if isinstance(action, dict) else action) in {"hu", "peng", "chi", "pass"}
        for action in state.get("legal_actions") or []
    )


def _annotate_discard_turn_from_qs(state: dict, payload: dict, self_seat: int | None) -> dict:
    if payload.get("cmd") != 1013:
        return state
    if not _truthy(payload.get("jionhold")):
        return state
    if not _payload_matches_self_seat(payload, self_seat):
        return state
    merged = copy.deepcopy(state)
    merged["legal_actions"] = [{"type": "discard", "source": "qs_own_draw"}]
    merged["buttons"] = []
    metadata = dict(merged.get("metadata") or {})
    metadata["discard_turn_source"] = "qs_own_draw"
    merged["metadata"] = metadata
    return merged


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _payload_seat(payload: dict) -> int | None:
    try:
        return int(payload.get("chairId", payload.get("seatid")))
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _update_tracked_hand(
    hand: list[str] | None,
    payload: dict,
    self_seat: int | None,
    warnings: list[dict],
) -> list[str] | None:
    cmd = payload.get("cmd")
    if cmd in {1011, 1035} and isinstance(payload.get("playerholdcards"), list):
        return [str(card) for card in payload["playerholdcards"]]
    if hand is None:
        return None
    if cmd == 1013 and _payload_matches_self_seat(payload, self_seat) and payload.get("jionhold"):
        card = payload.get("val")
        return [*hand, str(card)] if card else hand
    if cmd == 1012 and payload.get("err") == 0 and _payload_matches_self_seat(payload, self_seat):
        updated = list(hand)
        for card in _cards_removed_by_action(payload):
            if not _remove_first(updated, card):
                warnings.append({"cmd": cmd, "action_type": payload.get("action_type"), "missing_card": card})
        return updated
    return hand


def _payload_matches_self_seat(payload: dict, self_seat: int | None) -> bool:
    if self_seat is None:
        return False
    try:
        return int(payload.get("chairId", payload.get("seatid"))) == self_seat
    except (TypeError, ValueError):
        return False


def _cards_removed_by_action(payload: dict) -> list[str]:
    cards = [str(card) for card in payload.get("cards") or []]
    try:
        action_type = int(payload.get("action_type"))
    except (TypeError, ValueError):
        return []
    if action_type == 11:
        return cards
    if action_type in {101, 102}:
        return ["Qh"]
    if action_type in {103, 105}:
        return ["Qh", "Qh"]
    if action_type in {104, 106}:
        return ["Qh", "Qh", "Qh"]
    if action_type == 1:
        return cards[1:]
    if action_type in {7, 9, 10, 17, 18, 19}:
        return []
    return cards


def _remove_first(cards: list[str], card: str) -> bool:
    try:
        cards.remove(card)
        return True
    except ValueError:
        return False


def _override_with_tracked_hand(state: dict, raw_cards: list[str], warnings: list[dict]) -> dict:
    merged = copy.deepcopy(state)
    hand = normalize_protocol_cards(raw_cards)
    merged["hand"] = hand
    merged["raw_hand"] = hand
    merged["hand_count"] = len(hand)
    merged["expected_count"] = len(hand)
    merged["controlled_count"] = len(hand)
    metadata = dict(merged.get("metadata") or {})
    metadata["hand_source"] = "folded_qs_log"
    metadata["raw_hand_codes"] = list(raw_cards)
    if warnings:
        metadata["hand_update_warnings"] = list(warnings[-10:])
    merged["metadata"] = metadata
    return merged


def _is_likely_outbound_qs_row(line: str) -> bool:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    tcp = payload.get("tcp")
    if not isinstance(tcp, dict):
        return False
    src_host = _host_from_endpoint(str(tcp.get("src") or ""))
    dst_host = _host_from_endpoint(str(tcp.get("dst") or ""))
    return _is_private_ip(src_host) and bool(dst_host) and not _is_private_ip(dst_host)


def _host_from_endpoint(endpoint: str) -> str:
    if endpoint.startswith("["):
        return endpoint.split("]", 1)[0].lstrip("[")
    if ":" not in endpoint:
        return endpoint
    return endpoint.rsplit(":", 1)[0]


def _is_private_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_private
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    args = parser.parse_args()
    print(json.dumps(latest_protocol_state_from_log(args.log), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
