"""Parse local QS binary packets into normalized protocol state.

The format was recovered from ``dist/lua_all_decompiled/app.net.MyByteArray.lua``
and ``app.net.ParseSocket.lua``. It is intentionally local-only: no device,
socket, proxy, or Frida access happens here.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vision.protocol_state import protocol_state_from_payload


class QSProtocolError(ValueError):
    """Raised when a QS packet is malformed or cannot be decoded."""


@dataclass(frozen=True)
class QSPacket:
    cmd: int
    length: int
    body: bytes


class QSReader:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    @property
    def available(self) -> int:
        return len(self._data) - self._pos

    def read_int(self) -> int:
        raw = self._read(4)
        return int.from_bytes(raw, "little", signed=True)

    def read_uint(self) -> int:
        raw = self._read(4)
        return int.from_bytes(raw, "little", signed=False)

    def read_long_int(self) -> int:
        low = self.read_uint()
        high = self.read_uint()
        value = (high << 32) | low
        if high & 0x80000000:
            value -= 1 << 64
        return value

    def read_short(self) -> int:
        raw = self._read(2)
        return int.from_bytes(raw, "little", signed=True)

    def read_string(self) -> str:
        length = self.read_int()
        if length < 0:
            raise QSProtocolError(f"negative string length: {length}")
        raw = self._read(length)
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        return raw.decode("utf-8", errors="replace")

    def _read(self, size: int) -> bytes:
        if size > self.available:
            raise QSProtocolError(f"packet ended early: need {size}, available {self.available}")
        start = self._pos
        self._pos += size
        return self._data[start : start + size]


def parse_qs_header(packet: bytes) -> QSPacket:
    if len(packet) < 6:
        raise QSProtocolError("packet too short for QS header")
    if packet[:2] != b"QS":
        raise QSProtocolError("missing QS magic header")
    length = int.from_bytes(packet[2:4], "little", signed=True)
    cmd = int.from_bytes(packet[4:6], "little", signed=True)
    if length < 0:
        raise QSProtocolError(f"negative body length: {length}")
    actual = len(packet) - 6
    if actual != length:
        raise QSProtocolError(f"body length mismatch: header={length}, actual={actual}")
    return QSPacket(cmd=cmd, length=length, body=packet[6:])


def decode_packet_input(value: str | bytes | dict[str, Any]) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        if "packet_hex" in value:
            return decode_packet_input(str(value["packet_hex"]))
        if "packet_base64" in value:
            return base64.b64decode(str(value["packet_base64"]), validate=True)
        raise QSProtocolError("JSON object needs packet_hex or packet_base64")

    text = str(value).strip()
    if not text:
        raise QSProtocolError("empty packet input")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return decode_packet_input(payload)

    cleaned = "".join(text.split())
    if cleaned.lower().startswith("0x"):
        cleaned = cleaned[2:]
    if cleaned and len(cleaned) % 2 == 0:
        try:
            return bytes.fromhex(cleaned)
        except ValueError:
            pass
    try:
        return base64.b64decode(text, validate=True)
    except binascii.Error as exc:
        raise QSProtocolError("input is not valid hex, base64, or JSON packet text") from exc


def parse_packet(value: str | bytes | dict[str, Any]) -> dict[str, Any]:
    packet = parse_qs_header(decode_packet_input(value))
    reader = QSReader(packet.body)
    if packet.cmd == 1001:
        fields = _parse_fun1001(reader)
    elif packet.cmd == 1003:
        fields = _parse_fun1003(reader)
    elif packet.cmd == 1011:
        fields = _parse_hand_card_list(reader)
    elif packet.cmd == 1012:
        fields = _parse_fun1012(reader)
    elif packet.cmd == 1013:
        fields = _parse_fun1013(reader)
    elif packet.cmd == 1014:
        fields = _parse_fun1014(reader)
    elif packet.cmd == 1017:
        fields = {"gamecount": reader.read_int()}
    elif packet.cmd == 1026:
        fields = _parse_fun1026(reader)
    elif packet.cmd == 1027:
        fields = {"err": reader.read_int()}
    elif packet.cmd == 1035:
        fields = _parse_hand_card_list(reader)
    else:
        fields = {"unparsed_bytes": reader.available}
    fields.update({"cmd": packet.cmd, "qs_length": packet.length})
    return fields


def parse_packet_to_state(value: str | bytes | dict[str, Any]) -> dict[str, Any]:
    payload = parse_packet(value)
    state = protocol_state_from_payload(payload)
    controlled_count = _controlled_count(state)
    expected_count = _expected_count(payload, controlled_count)
    state["expected_count"] = expected_count
    state["controlled_count"] = controlled_count
    state["metadata"]["qs_cmd"] = payload.get("cmd")
    state["metadata"]["qs_payload"] = payload
    return state


def _parse_fun1001(reader: QSReader) -> dict[str, Any]:
    data: dict[str, Any] = {"game_status_type": reader.read_int()}
    if data["game_status_type"] == 3:
        data.update(_parse_fun1001_paohuzi(reader))
    else:
        data["unparsed_bytes"] = reader.available
    return data


def _parse_fun1001_paohuzi(reader: QSReader) -> dict[str, Any]:
    data: dict[str, Any] = {"err": reader.read_int()}
    if data["err"] != 0:
        return data
    data["chairId"] = reader.read_int()
    data["seatid"] = data["chairId"]
    data["gametype"] = reader.read_string()
    data["peoplenum"] = reader.read_int()
    if reader.available > 10:
        data["room_cfg_marker"] = reader.read_int()
        for key in (
            "mingtang",
            "fangxing",
            "mingtang2",
            "minscore",
            "maxhuxi",
            "Zimo",
            "Bihu",
            "MaoHu",
            "Minhuxi",
            "Tunxi",
            "lianzhuan",
            "datuo",
            "m1_5_10",
            "zhuaniao",
            "wangnum",
            "xianhu",
            "piaotype",
            "QhCost",
            "paytype",
        ):
            if reader.available < 4:
                return data
            data[key] = reader.read_int()
        if reader.available >= 8:
            data["aapay"] = reader.read_long_int()
        for key in ("passWord", "wanfa_index", "minglongguize", "zhuang"):
            if reader.available < 4:
                return data
            data[key] = reader.read_int()
    return data


def _parse_fun1003(reader: QSReader) -> dict[str, Any]:
    data: dict[str, Any] = {
        "room_id": reader.read_int(),
        "gametype": reader.read_string(),
        "gameNum": reader.read_int(),
        "dealer": reader.read_int(),
        "zhuang": None,
        "curr_seatid": reader.read_int(),
        "chairId": None,
        "curr_card": reader.read_string(),
        "pre_outseatid": reader.read_int(),
        "pre_outtype": reader.read_int(),
        "seats": {},
    }
    data["zhuang"] = data["dealer"]
    data["chairId"] = data["curr_seatid"]
    player_num = reader.read_int()
    for _ in range(player_num):
        seatid = reader.read_int()
        seat = {
            "seatid": seatid,
            "mid": reader.read_int(),
            "leave": reader.read_int(),
            "leaveTime": reader.read_int(),
            "ready": reader.read_int(),
            "piaotype": reader.read_int(),
            "mustache": reader.read_int(),
            "out_cards": _read_string_list(reader),
        }
        mustache_cards = []
        for _ in range(reader.read_int()):
            mustache_cards.append({"_type": reader.read_int(), "cards": _read_string_list(reader)})
        seat["mustache_cards"] = mustache_cards
        data["seats"][seatid] = seat
    if reader.available >= 4:
        data["room_owner"] = reader.read_int()
    if reader.available >= 4:
        data["gamestate"] = reader.read_int()
    return data


def _parse_fun1014(reader: QSReader) -> dict[str, Any]:
    seatid = reader.read_int()
    precardval = reader.read_string()
    isoutcard = reader.read_int()
    canpeng = reader.read_int()
    canchi = reader.read_int()
    canhu_raw = reader.read_int()
    data = {
        "seatid": seatid,
        "chairId": None,
        "precardval": precardval,
        "isoutcard": isoutcard,
        "canpeng": canpeng,
        "canchi": canchi,
        "canhu": canhu_raw,
        "canhu_raw": canhu_raw,
        "actionid": reader.read_string(),
        "hutype": reader.read_int(),
        "alarm": reader.read_int(),
        "showGuo": reader.read_int(),
    }
    data["chairId"] = data["seatid"]
    if data["canhu"] == 1:
        data["canhu"] = data["canhu"] + data["hutype"]
    data["hu_action_code"] = data["canhu"]
    return data


def _parse_hand_card_list(reader: QSReader) -> dict[str, Any]:
    cards = _read_string_list(reader)
    return {"playerholdcards": cards, "cards": cards}


def _parse_fun1012(reader: QSReader) -> dict[str, Any]:
    data: dict[str, Any] = {"err": reader.read_int()}
    if data["err"] != 0:
        data["unparsed_bytes"] = reader.available
        return data
    data.update(
        {
            "seatid": reader.read_int(),
            "chairId": None,
            "mustache": reader.read_int(),
            "action_type": reader.read_int(),
        }
    )
    data["chairId"] = data["seatid"]
    data["cards"] = _read_string_list(reader)
    data["peoplescore"] = []
    for _ in range(reader.read_int()):
        data["peoplescore"].append([reader.read_int(), reader.read_int(), reader.read_int()])
    if reader.available >= 4:
        data["addscoretype"] = reader.read_int()
    if reader.available >= 4:
        data["tail_ints"] = [reader.read_int() for _ in range(reader.available // 4)]
    return data


def _parse_fun1013(reader: QSReader) -> dict[str, Any]:
    data = {
        "seatid": reader.read_int(),
        "chairId": None,
        "_size": reader.read_int(),
        "val": reader.read_string(),
        "curr_card": None,
        "isshow": reader.read_int(),
        "jionhold": reader.read_int(),
        "isfanxing": reader.read_int(),
        "ishucard": reader.read_int(),
    }
    data["chairId"] = data["seatid"]
    data["curr_card"] = data["val"]
    return data


def _parse_fun1026(reader: QSReader) -> dict[str, Any]:
    data: dict[str, Any] = {"over_type": reader.read_int(), "cards": [], "playerholdcards": {}}
    seats = []
    for index in range(reader.read_int()):
        seat = {
            "seatid": reader.read_int(),
            "score": reader.read_long_int(),
            "piaotype": reader.read_int(),
            "piao": reader.read_long_int(),
        }
        if data["over_type"] == 1 and index == 0:
            data["winsite"] = seat["seatid"]
        seats.append(seat)
    data["seats"] = seats
    data["dunshu"] = reader.read_int()
    data["fanxingdunshu"] = reader.read_int()
    data["fanxingcard"] = reader.read_string()
    data["kingtable"] = _read_string_list(reader)
    win_count = reader.read_int()
    data["win_type"] = []
    data["win_type_mustache"] = []
    for _ in range(win_count):
        data["win_type"].append(reader.read_int())
        data["win_type_mustache"].append(reader.read_int())
    data["cards"] = _read_string_list(reader)
    for _ in range(reader.read_int()):
        seatid = reader.read_int()
        data["playerholdcards"][seatid] = _read_string_list(reader)
    data["mustache_cards"] = _read_mustache_cards(reader)
    data["mustache"] = reader.read_int()
    data["huzi_index"] = reader.read_int()
    data["huzi_cardval"] = reader.read_string()
    data["total_score"] = []
    for _ in range(reader.read_int()):
        data["total_score"].append({"seatid": reader.read_int(), "score": reader.read_int()})
    data["dealer"] = reader.read_int()
    data["zhuang"] = data["dealer"]
    data["roomid"] = reader.read_int()
    data["xingtype"] = reader.read_int()
    return data


def _read_string_list(reader: QSReader) -> list[str]:
    return [reader.read_string() for _ in range(reader.read_int())]


def _read_mustache_cards(reader: QSReader) -> list[dict[str, Any]]:
    cards = []
    for _ in range(reader.read_int()):
        cards.append({"_type": reader.read_int(), "mustache": reader.read_int(), "cards": _read_string_list(reader)})
    return cards


def _controlled_count(state: dict[str, Any]) -> int:
    return int(state.get("hand_count") or len(state.get("hand") or []))


def _expected_count(payload: dict[str, Any], fallback: int) -> int:
    value = payload.get("expected_count")
    if value is not None:
        return int(value)
    hand = payload.get("playerholdcards")
    if isinstance(hand, dict) and hand:
        return max(len(cards) for cards in hand.values() if isinstance(cards, list))
    if isinstance(hand, list):
        return len(hand)
    return fallback


def _load_cli_packet(text_or_path: str) -> str:
    path = Path(text_or_path)
    try:
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line:
                    return line
            raise QSProtocolError(f"no packet lines in {path}")
    except OSError:
        pass
    return text_or_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", help="hex, base64, JSON text, or path to a JSONL/text packet file")
    parser.add_argument("--raw", action="store_true", help="print decoded fields instead of normalized AI state")
    args = parser.parse_args()
    packet = _load_cli_packet(args.packet)
    result = parse_packet(packet) if args.raw else parse_packet_to_state(packet)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
