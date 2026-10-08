import json
import struct

from tools.protocol_log_to_state import latest_protocol_state_from_log
from tools.qs_packet_extract import (
    ExtractedQS,
    QSStreamExtractor,
    append_qs_jsonl,
    collect_terminal_evidence,
    extract_qs_packets,
    write_qs_jsonl,
)


def test_extract_qs_packets_from_raw_stream(tmp_path):
    packet = _qs_packet(1014, _i(2) + _s("7s") + _i(1) + _i(1) + _i(0) + _i(0) + _s("a") + _i(0) + _i(0) + _i(0))

    packets = extract_qs_packets(b"noise" + packet + b"tail")
    output = write_qs_jsonl(packets, tmp_path / "packets.jsonl")

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["cmd"] == 1014
    assert rows[0]["state_summary"]["pending_card"] == "七"
    assert rows[0]["protocol_fields"]["canhu_raw"] == 0


def test_qs_jsonl_connects_to_latest_protocol_state(tmp_path):
    packet = _qs_packet(1014, _i(2) + _s("7s") + _i(1) + _i(0) + _i(1) + _i(0) + _s("a") + _i(0) + _i(0) + _i(0))
    output = write_qs_jsonl(extract_qs_packets(packet), tmp_path / "packets.jsonl")

    state = latest_protocol_state_from_log(output)

    assert state["pending_card"] == "七"
    assert [item["type"] for item in state["legal_actions"]] == ["chi"]


def test_extract_qs_packets_from_split_tcp_pcap_stream():
    packet = _qs_packet(1014, _i(2) + _s("7s") + _i(1) + _i(0) + _i(0) + _i(1) + _s("a") + _i(2) + _i(0) + _i(1))
    first, second = packet[:5], packet[5:]
    pcap = _pcap_raw_ip(
        [
            _ipv4_tcp_packet(first, seq=100),
            _ipv4_tcp_packet(second, seq=100 + len(first)),
        ]
    )

    packets = extract_qs_packets(pcap)

    assert len(packets) == 1
    assert packets[0].packet == packet


def test_qs_jsonl_preserves_raw_and_display_hu_codes(tmp_path):
    packet = _qs_packet(
        1014,
        _i(2) + _s("7s") + _i(1) + _i(0) + _i(0) + _i(1) + _s("a") + _i(2) + _i(0) + _i(1),
    )
    output = write_qs_jsonl(extract_qs_packets(packet), tmp_path / "packets.jsonl")

    row = json.loads(output.read_text(encoding="utf-8"))

    assert row["protocol_fields"]["canhu_raw"] == 1
    assert row["protocol_fields"]["hutype"] == 2
    assert row["protocol_fields"]["canhu"] == 3
    assert row["protocol_fields"]["hu_action_code"] == 3


def test_collect_terminal_evidence_links_room_prompt_action_and_pair_shape(monkeypatch):
    decoded = iter(
        [
            {"cmd": 1001, "gametype": "30", "wangnum": 2, "peoplenum": 2, "seatid": 2},
            {"cmd": 1014, "seatid": 2, "canhu_raw": 1},
            {"cmd": 1012, "err": 0, "seatid": 2, "action_type": 9},
            {
                "cmd": 1026,
                "winsite": 2,
                "over_type": 1,
                "roomid": 434112,
                "mustache": 17,
                "huzi_cardval": "6b",
                "win_type": [],
                "playerholdcards": {2: ["Tb", "Qh"]},
                "mustache_cards": [
                    {"cards": ["6b", "6b", "6b", "6b"]},
                    {"cards": ["Tb", "Qh"]},
                ],
            },
        ]
    )
    monkeypatch.setattr("tools.qs_packet_extract.parse_packet", lambda _packet: next(decoded))
    packets = [ExtractedQS(packet=bytes([index]), source="test", offset=index) for index in range(4)]

    evidence = collect_terminal_evidence(packets)

    assert len(evidence) == 1
    assert evidence[0]["gametype"] == "30"
    assert evidence[0]["wangnum"] == 2
    assert evidence[0]["terminal_path"] == "prompted_hu"
    assert evidence[0]["pair_groups"] == [["Tb", "Qh"]]
    assert evidence[0]["quad_groups"] == [["6b", "6b", "6b", "6b"]]
    assert evidence[0]["winning_hand_wang_count"] == 1


def test_stream_extractor_appends_complete_qs_rows(tmp_path):
    packet = _qs_packet(1014, _i(2) + _s("7s") + _i(1) + _i(0) + _i(1) + _i(0) + _s("a") + _i(0) + _i(0) + _i(0))
    extractor = QSStreamExtractor(source="test_live")

    assert extractor.feed(b"noise" + packet[:5]) == []
    packets = extractor.feed(packet[5:] + b"tail")
    assert len(packets) == 1
    assert packets[0].packet == packet

    output = tmp_path / "live_qs.jsonl"
    written = append_qs_jsonl(packets, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert written == 1
    assert rows[0]["source"] == "test_live"
    assert rows[0]["cmd"] == 1014
    assert latest_protocol_state_from_log(output)["pending_card"] == "七"


def _qs_packet(cmd: int, body: bytes) -> bytes:
    return b"QS" + len(body).to_bytes(2, "little", signed=True) + cmd.to_bytes(2, "little", signed=True) + body


def _i(value: int) -> bytes:
    return value.to_bytes(4, "little", signed=True)


def _s(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return _i(len(raw)) + raw


def _pcap_raw_ip(packets: list[bytes]) -> bytes:
    data = bytearray(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 101))
    for index, packet in enumerate(packets, start=1):
        data.extend(struct.pack("<IIII", index, 0, len(packet), len(packet)))
        data.extend(packet)
    return bytes(data)


def _ipv4_tcp_packet(payload: bytes, *, seq: int) -> bytes:
    src = bytes([10, 255, 0, 1])
    dst = bytes([47, 114, 97, 95])
    tcp = (
        (46150).to_bytes(2, "big")
        + (24000).to_bytes(2, "big")
        + seq.to_bytes(4, "big")
        + (0).to_bytes(4, "big")
        + bytes([0x50, 0x18])
        + (8192).to_bytes(2, "big")
        + (0).to_bytes(2, "big")
        + (0).to_bytes(2, "big")
        + payload
    )
    total_len = 20 + len(tcp)
    return (
        bytes([0x45, 0])
        + total_len.to_bytes(2, "big")
        + (0).to_bytes(2, "big")
        + (0).to_bytes(2, "big")
        + bytes([64, 6])
        + (0).to_bytes(2, "big")
        + src
        + dst
        + tcp
    )
