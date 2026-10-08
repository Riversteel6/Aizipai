import base64
import json

import pytest

from tools.protocol_log_to_state import latest_protocol_state_from_log
from tools.qs_protocol_parser import QSProtocolError, parse_packet, parse_packet_to_state, parse_qs_header
from vision.protocol_state import protocol_state_from_payload


def test_qs_header_parses_length_and_cmd():
    packet = _packet(1014, _i(1) + _s("7s") + _i(0) + _i(1) + _i(0) + _i(0) + _s("a1") + _i(0) + _i(0) + _i(1))

    header = parse_qs_header(packet)

    assert header.cmd == 1014
    assert header.length == len(packet) - 6


def test_qs_cmd_parses_from_hex():
    packet_hex = _packet(1014, _i(1) + _s("7s") + _i(0) + _i(1) + _i(1) + _i(0) + _s("a1") + _i(0) + _i(0) + _i(0)).hex()

    payload = parse_packet(packet_hex)

    assert payload["cmd"] == 1014
    assert payload["precardval"] == "7s"


def test_qs_length_mismatch_is_rejected():
    packet = b"QS" + (99).to_bytes(2, "little", signed=True) + (1014).to_bytes(2, "little", signed=True)

    with pytest.raises(QSProtocolError, match="body length mismatch"):
        parse_qs_header(packet)


def test_fun1001_paohuzi_synthetic_parses_seat_and_dealer_fields():
    body = (
        _i(3)
        + _i(0)
        + _i(2)
        + _s("3")
        + _i(3)
        + _i(0)
        + b"".join(_i(value) for value in range(1, 20))
        + _long(0)
        + _i(1234)
        + _i(8)
        + _i(0)
        + _i(2)
    )

    state = parse_packet_to_state(_packet(1001, body))

    assert state["metadata"]["qs_cmd"] == 1001
    assert state["metadata"]["seat_fields"]["chairId"] == 2
    assert state["metadata"]["seat_fields"]["zhuang"] == 2


def test_fun1014_synthetic_parses_action_prompt_state():
    body = _i(2) + _s("7s") + _i(1) + _i(1) + _i(1) + _i(1) + _s("act-7") + _i(2) + _i(0) + _i(1)

    state = parse_packet_to_state(_packet(1014, body))

    assert state["pending_card"] == "七"
    assert [item["type"] for item in state["legal_actions"]] == ["hu", "peng", "chi", "pass"]
    assert state["metadata"]["seat_fields"]["chairId"] == 2


def test_fun1026_synthetic_parses_hands_and_dealer():
    body = (
        _i(1)
        + _i(1)
        + _i(2)
        + _long(12)
        + _i(0)
        + _long(0)
        + _i(4)
        + _i(0)
        + _s("")
        + _list()
        + _i(0)
        + _list("1s", "2s")
        + _i(1)
        + _i(2)
        + _list("3s", "4s", "5s")
        + _i(0)
        + _i(6)
        + _i(2)
        + _s("8s")
        + _i(1)
        + _i(2)
        + _i(99)
        + _i(2)
        + _i(456)
        + _i(0)
    )

    state = parse_packet_to_state(_packet(1026, body))

    assert state["hand"] == ["三", "四", "五"]
    assert state["pending_card"] == "八"
    assert state["expected_count"] == 3
    assert state["controlled_count"] == 3
    assert state["metadata"]["seat_fields"]["dealer"] == 2


def test_fun1011_synthetic_parses_full_hand_snapshot():
    state = parse_packet_to_state(_packet(1011, _list("1s", "2b", "Qh")))

    assert state["hand"] == ["一", "贰", "王"]
    assert state["expected_count"] == 3
    assert state["controlled_count"] == 3


def test_fun1013_synthetic_parses_turn_card():
    body = _i(2) + _i(40) + _s("Qh") + _i(1) + _i(1) + _i(0) + _i(0)

    state = parse_packet_to_state(_packet(1013, body))

    assert state["pending_card"] == "王"
    assert state["metadata"]["seat_fields"]["chairId"] == 2


def test_jsonl_packet_hex_connects_to_protocol_log_to_state(tmp_path):
    body = _i(2) + _s("7s") + _i(1) + _i(0) + _i(1) + _i(0) + _s("act-7") + _i(0) + _i(0) + _i(0)
    log = tmp_path / "packets.jsonl"
    log.write_text(json.dumps({"packet_hex": _packet(1014, body).hex()}) + "\n", encoding="utf-8")

    state = latest_protocol_state_from_log(log)

    assert state["pending_card"] == "七"
    assert [item["type"] for item in state["legal_actions"]] == ["chi"]


def test_live_wildcard_code_qh_normalizes_to_wang():
    body = _i(2) + _s("Qh") + _i(0) + _i(0) + _i(1) + _i(0) + _s("act-wang") + _i(0) + _i(0) + _i(1)

    state = parse_packet_to_state(_packet(1014, body))

    assert state["pending_card"] == "王"
    assert [item["type"] for item in state["legal_actions"]] == ["chi", "pass"]


def test_packet_base64_and_protocol_state_seat_map():
    packet = _packet(1026, _minimal_1026_body())
    state = parse_packet_to_state({"packet_base64": base64.b64encode(packet).decode("ascii")})

    assert state["hand"] == ["三", "四", "五"]
    assert protocol_state_from_payload({"playerholdcards": {2: ["3s"]}, "chairId": 2})["hand"] == ["三"]


def _minimal_1026_body() -> bytes:
    return (
        _i(1)
        + _i(1)
        + _i(2)
        + _long(12)
        + _i(0)
        + _long(0)
        + _i(4)
        + _i(0)
        + _s("")
        + _list()
        + _i(0)
        + _list("1s", "2s")
        + _i(1)
        + _i(2)
        + _list("3s", "4s", "5s")
        + _i(0)
        + _i(6)
        + _i(2)
        + _s("8s")
        + _i(0)
        + _i(2)
        + _i(456)
        + _i(0)
    )


def _packet(cmd: int, body: bytes) -> bytes:
    return b"QS" + len(body).to_bytes(2, "little", signed=True) + cmd.to_bytes(2, "little", signed=True) + body


def _i(value: int) -> bytes:
    return value.to_bytes(4, "little", signed=True)


def _long(value: int) -> bytes:
    return value.to_bytes(8, "little", signed=True)


def _s(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return _i(len(raw)) + raw


def _list(*values: str) -> bytes:
    return _i(len(values)) + b"".join(_s(value) for value in values)
