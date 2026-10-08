"""Extract QS packets from local raw streams or packet captures.

This tool is deliberately offline/local. It does not touch ADB, Frida, proxy
settings, APK installs, or the phone. Feed it a raw TCP stream, pcap, or pcapng
export and it writes JSONL rows with ``packet_hex`` that the protocol state
tools already understand.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.qs_protocol_parser import QSProtocolError, parse_packet, parse_packet_to_state, parse_qs_header

MAX_QS_BODY = 16_384 - 6


@dataclass(frozen=True)
class TCPPayload:
    payload: bytes
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    seq: int
    timestamp: float | None = None

    @property
    def flow_key(self) -> tuple[str, int, str, int]:
        return (self.src_ip, self.src_port, self.dst_ip, self.dst_port)


@dataclass(frozen=True)
class ExtractedQS:
    packet: bytes
    source: str
    offset: int
    tcp: TCPPayload | None = None


@dataclass
class QSStreamExtractor:
    source: str = "tcp_exporter_live"
    max_buffer: int = MAX_QS_BODY + 4096
    buffer: bytes = b""
    base_offset: int = 0
    seen_packets: set[bytes] = field(default_factory=set)

    def feed(self, chunk: bytes) -> list[ExtractedQS]:
        if not chunk:
            return []
        self.buffer += chunk
        packets: list[ExtractedQS] = []
        pos = 0
        keep_from = 0
        while True:
            start = self.buffer.find(b"QS", pos)
            if start < 0:
                keep_from = max(0, len(self.buffer) - 2)
                break
            if start + 6 > len(self.buffer):
                keep_from = start
                break
            length = int.from_bytes(self.buffer[start + 2 : start + 4], "little", signed=True)
            cmd = int.from_bytes(self.buffer[start + 4 : start + 6], "little", signed=True)
            end = start + 6 + length
            if not (0 <= length <= MAX_QS_BODY and 0 <= cmd <= 32767):
                pos = start + 2
                keep_from = pos
                continue
            if end > len(self.buffer):
                keep_from = start
                break
            packet = self.buffer[start:end]
            try:
                parse_qs_header(packet)
            except QSProtocolError:
                pos = start + 2
                keep_from = pos
                continue
            if packet not in self.seen_packets:
                self.seen_packets.add(packet)
                packets.append(ExtractedQS(packet=packet, source=self.source, offset=self.base_offset + start))
            pos = end
            keep_from = pos

        if keep_from:
            self.buffer = self.buffer[keep_from:]
            self.base_offset += keep_from
        if len(self.buffer) > self.max_buffer:
            drop = len(self.buffer) - self.max_buffer
            self.buffer = self.buffer[drop:]
            self.base_offset += drop
        return packets


def extract_qs_packets(data: bytes, *, source: str = "bytes") -> list[ExtractedQS]:
    """Extract complete QS packets from raw bytes or supported packet captures."""

    tcp_payloads = list(_iter_tcp_payloads(data))
    packets: list[ExtractedQS] = []
    for payload in tcp_payloads:
        packets.extend(_scan_qs_frames(payload.payload, source="tcp_payload", tcp=payload))

    for stream_key, stream in _reassemble_streams(tcp_payloads).items():
        src_ip, src_port, dst_ip, dst_port = stream_key
        stream_tcp = TCPPayload(
            payload=stream,
            src_ip=src_ip,
            src_port=src_port,
            dst_ip=dst_ip,
            dst_port=dst_port,
            seq=0,
            timestamp=None,
        )
        packets.extend(_scan_qs_frames(stream, source="tcp_stream", tcp=stream_tcp))

    if not packets:
        packets.extend(_scan_qs_frames(data, source=source, tcp=None))

    return _dedupe_packets(packets)


def write_qs_jsonl(packets: Iterable[ExtractedQS], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(packets, start=1):
            handle.write(json.dumps(qs_jsonl_row(item, index=index), ensure_ascii=False, sort_keys=True) + "\n")
    return output


def append_qs_jsonl(packets: Iterable[ExtractedQS], path: str | Path, *, start_index: int = 1) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("a", encoding="utf-8") as handle:
        for count, item in enumerate(packets, start=1):
            row = qs_jsonl_row(item, index=start_index + count - 1)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    return count


def qs_jsonl_row(item: ExtractedQS, *, index: int) -> dict[str, Any]:
    try:
        decoded = parse_packet(item.packet)
        state = parse_packet_to_state(item.packet)
        error = None
    except QSProtocolError as exc:
        decoded = {}
        state = {}
        error = str(exc)
    row: dict[str, Any] = {
        "type": "qs_packet",
        "index": index,
        "source": item.source,
        "offset": item.offset,
        "packet_hex": item.packet.hex(),
        "cmd": decoded.get("cmd"),
        "qs_length": decoded.get("qs_length"),
        "protocol_fields": {
            key: value
            for key, value in decoded.items()
            if key not in {"cmd", "qs_length"}
        },
        "state_summary": _state_summary(state),
    }
    if item.tcp is not None:
        row["tcp"] = {
            "src": f"{item.tcp.src_ip}:{item.tcp.src_port}",
            "dst": f"{item.tcp.dst_ip}:{item.tcp.dst_port}",
            "seq": item.tcp.seq,
            "timestamp": item.tcp.timestamp,
        }
    if error:
        row["error"] = error
    return row


def _scan_qs_frames(data: bytes, *, source: str, tcp: TCPPayload | None) -> list[ExtractedQS]:
    packets: list[ExtractedQS] = []
    pos = 0
    while True:
        start = data.find(b"QS", pos)
        if start < 0:
            break
        if start + 6 > len(data):
            break
        length = int.from_bytes(data[start + 2 : start + 4], "little", signed=True)
        cmd = int.from_bytes(data[start + 4 : start + 6], "little", signed=True)
        end = start + 6 + length
        if 0 <= length <= MAX_QS_BODY and 0 <= cmd <= 32767 and end <= len(data):
            packet = data[start:end]
            try:
                parse_qs_header(packet)
            except QSProtocolError:
                pos = start + 2
                continue
            packets.append(ExtractedQS(packet=packet, source=source, offset=start, tcp=tcp))
            pos = end
            continue
        pos = start + 2
    return packets


def _dedupe_packets(packets: list[ExtractedQS]) -> list[ExtractedQS]:
    seen: set[bytes] = set()
    unique: list[ExtractedQS] = []
    for item in packets:
        if item.packet in seen:
            continue
        seen.add(item.packet)
        unique.append(item)
    return unique


def _iter_tcp_payloads(data: bytes) -> Iterable[TCPPayload]:
    if data.startswith(b"\x0a\x0d\x0d\x0a"):
        yield from _iter_pcapng_tcp_payloads(data)
        return
    if len(data) >= 24 and data[:4] in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"M<\xb2\xa1", b"\xa1\xb2<M"):
        yield from _iter_pcap_tcp_payloads(data)


def _iter_pcap_tcp_payloads(data: bytes) -> Iterable[TCPPayload]:
    endian = "<" if data[:4] in (b"\xd4\xc3\xb2\xa1", b"M<\xb2\xa1") else ">"
    if len(data) < 24:
        return
    _, _, _, _, _, _, linktype = struct.unpack(endian + "IHHIIII", data[:24])
    offset = 24
    while offset + 16 <= len(data):
        ts_sec, ts_frac, incl_len, _ = struct.unpack(endian + "IIII", data[offset : offset + 16])
        offset += 16
        packet = data[offset : offset + incl_len]
        offset += incl_len
        timestamp = ts_sec + (ts_frac / 1_000_000)
        payload = _tcp_payload_from_link_packet(packet, linktype=linktype, timestamp=timestamp)
        if payload is not None:
            yield payload


def _iter_pcapng_tcp_payloads(data: bytes) -> Iterable[TCPPayload]:
    endian = "<"
    linktypes: dict[int, int] = {}
    offset = 0
    while offset + 12 <= len(data):
        block_type = int.from_bytes(data[offset : offset + 4], endian == ">" and "big" or "little")
        total_length = int.from_bytes(data[offset + 4 : offset + 8], endian == ">" and "big" or "little")
        if total_length < 12 or offset + total_length > len(data):
            break
        block = data[offset : offset + total_length]
        if block_type == 0x0A0D0D0A and len(block) >= 16:
            magic = block[8:12]
            if magic == b"\x1a\x2b\x3c\x4d":
                endian = ">"
            elif magic == b"\x4d\x3c\x2b\x1a":
                endian = "<"
        elif block_type == 1 and len(block) >= 20:
            linktype = struct.unpack(endian + "H", block[8:10])[0]
            linktypes[len(linktypes)] = linktype
        elif block_type == 6 and len(block) >= 28:
            interface_id, ts_high, ts_low, captured_len, _ = struct.unpack(endian + "IIIII", block[8:28])
            packet = block[28 : 28 + captured_len]
            timestamp = ((ts_high << 32) | ts_low) / 1_000_000
            payload = _tcp_payload_from_link_packet(packet, linktype=linktypes.get(interface_id, 101), timestamp=timestamp)
            if payload is not None:
                yield payload
        elif block_type == 3 and len(block) >= 16:
            original_len = struct.unpack(endian + "I", block[8:12])[0]
            packet = block[12 : 12 + original_len]
            payload = _tcp_payload_from_link_packet(packet, linktype=linktypes.get(0, 101), timestamp=None)
            if payload is not None:
                yield payload
        offset += total_length


def _tcp_payload_from_link_packet(packet: bytes, *, linktype: int, timestamp: float | None) -> TCPPayload | None:
    ip_packet: bytes
    if linktype == 1:
        if len(packet) < 14:
            return None
        ether_type = int.from_bytes(packet[12:14], "big")
        cursor = 14
        while ether_type == 0x8100 and len(packet) >= cursor + 4:
            ether_type = int.from_bytes(packet[cursor + 2 : cursor + 4], "big")
            cursor += 4
        if ether_type not in (0x0800, 0x86DD):
            return None
        ip_packet = packet[cursor:]
    elif linktype == 101:
        ip_packet = packet
    elif linktype == 113:
        if len(packet) < 16:
            return None
        protocol = int.from_bytes(packet[14:16], "big")
        if protocol not in (0x0800, 0x86DD):
            return None
        ip_packet = packet[16:]
    else:
        return None
    return _tcp_payload_from_ip_packet(ip_packet, timestamp=timestamp)


def _tcp_payload_from_ip_packet(packet: bytes, *, timestamp: float | None) -> TCPPayload | None:
    if not packet:
        return None
    version = packet[0] >> 4
    if version == 4:
        return _tcp_payload_from_ipv4(packet, timestamp=timestamp)
    if version == 6:
        return _tcp_payload_from_ipv6(packet, timestamp=timestamp)
    return None


def _tcp_payload_from_ipv4(packet: bytes, *, timestamp: float | None) -> TCPPayload | None:
    if len(packet) < 20:
        return None
    ihl = (packet[0] & 0x0F) * 4
    total_len = int.from_bytes(packet[2:4], "big")
    if ihl < 20 or total_len < ihl or len(packet) < total_len or packet[9] != 6:
        return None
    src_ip = ".".join(str(part) for part in packet[12:16])
    dst_ip = ".".join(str(part) for part in packet[16:20])
    return _tcp_payload_from_tcp_segment(packet[ihl:total_len], src_ip=src_ip, dst_ip=dst_ip, timestamp=timestamp)


def _tcp_payload_from_ipv6(packet: bytes, *, timestamp: float | None) -> TCPPayload | None:
    if len(packet) < 40 or packet[6] != 6:
        return None
    payload_len = int.from_bytes(packet[4:6], "big")
    end = 40 + payload_len
    if len(packet) < end:
        return None
    src_ip = _format_ipv6(packet[8:24])
    dst_ip = _format_ipv6(packet[24:40])
    return _tcp_payload_from_tcp_segment(packet[40:end], src_ip=src_ip, dst_ip=dst_ip, timestamp=timestamp)


def _tcp_payload_from_tcp_segment(segment: bytes, *, src_ip: str, dst_ip: str, timestamp: float | None) -> TCPPayload | None:
    if len(segment) < 20:
        return None
    src_port = int.from_bytes(segment[0:2], "big")
    dst_port = int.from_bytes(segment[2:4], "big")
    seq = int.from_bytes(segment[4:8], "big")
    data_offset = (segment[12] >> 4) * 4
    if data_offset < 20 or len(segment) < data_offset:
        return None
    payload = segment[data_offset:]
    if not payload:
        return None
    return TCPPayload(payload=payload, src_ip=src_ip, src_port=src_port, dst_ip=dst_ip, dst_port=dst_port, seq=seq, timestamp=timestamp)


def _reassemble_streams(payloads: Iterable[TCPPayload]) -> dict[tuple[str, int, str, int], bytes]:
    grouped: dict[tuple[str, int, str, int], list[TCPPayload]] = {}
    for payload in payloads:
        grouped.setdefault(payload.flow_key, []).append(payload)

    streams: dict[tuple[str, int, str, int], bytes] = {}
    for key, items in grouped.items():
        chunks = sorted(items, key=lambda item: item.seq)
        stream = bytearray()
        expected_seq: int | None = None
        for item in chunks:
            if expected_seq is None:
                stream.extend(item.payload)
                expected_seq = item.seq + len(item.payload)
                continue
            if item.seq < expected_seq:
                overlap = expected_seq - item.seq
                if overlap < len(item.payload):
                    stream.extend(item.payload[overlap:])
                    expected_seq += len(item.payload) - overlap
                continue
            stream.extend(item.payload)
            expected_seq = item.seq + len(item.payload)
        streams[key] = bytes(stream)
    return streams


def _format_ipv6(raw: bytes) -> str:
    parts = [raw[index : index + 2].hex() for index in range(0, 16, 2)]
    return ":".join(parts)


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return {}
    return {
        "hand": state.get("hand") or [],
        "pending_card": state.get("pending_card") or "",
        "legal_actions": [item.get("type") for item in state.get("legal_actions") or []],
        "expected_count": state.get("expected_count"),
        "controlled_count": state.get("controlled_count"),
        "seat_fields": (state.get("metadata") or {}).get("seat_fields") or {},
        "qs_cmd": (state.get("metadata") or {}).get("qs_cmd"),
    }


def collect_terminal_evidence(packets: Iterable[ExtractedQS]) -> list[dict[str, Any]]:
    """Collect room-linked round settlements without inferring hidden server rules."""

    room: dict[str, Any] = {}
    round_prompts: list[dict[str, Any]] = []
    round_hu_actions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(packets, start=1):
        try:
            decoded = parse_packet(item.packet)
        except QSProtocolError:
            continue
        cmd = decoded.get("cmd")
        if cmd == 1001 and decoded.get("gametype") is not None:
            room = {
                "gametype": str(decoded.get("gametype")),
                "wangnum": decoded.get("wangnum"),
                "players": decoded.get("peoplenum"),
                "self_seat": decoded.get("seatid"),
            }
        elif cmd == 1014:
            round_prompts.append(decoded)
        elif cmd == 1012 and decoded.get("err") == 0 and decoded.get("action_type") == 9:
            round_hu_actions.append(decoded)
        elif cmd == 1026:
            winner = decoded.get("winsite")
            prompts = [
                prompt
                for prompt in round_prompts
                if prompt.get("seatid") == winner and int(prompt.get("canhu_raw", 0) or 0) > 0
            ]
            hu_actions = [
                action
                for action in round_hu_actions
                if action.get("seatid") == winner
            ]
            groups = [
                list(group.get("cards") or [])
                for group in decoded.get("mustache_cards") or []
            ]
            pair_groups = [group for group in groups if len(group) == 2]
            quad_groups = [group for group in groups if len(group) == 4]
            player_hands = decoded.get("playerholdcards") or {}
            winning_hand = player_hands.get(winner) or player_hands.get(str(winner)) or []
            if prompts:
                terminal_path = "prompted_hu"
            elif hu_actions:
                terminal_path = "server_hu_action_without_local_prompt"
            else:
                terminal_path = "settlement_without_observed_hu_action"
            evidence.append(
                {
                    "packet_index": index,
                    **room,
                    "room_id": decoded.get("roomid"),
                    "winner": winner,
                    "self_won": winner == room.get("self_seat"),
                    "over_type": decoded.get("over_type"),
                    "total_xi": decoded.get("mustache"),
                    "huzi_cardval": decoded.get("huzi_cardval"),
                    "win_type": list(decoded.get("win_type") or []),
                    "group_lengths": [len(group) for group in groups],
                    "pair_groups": pair_groups,
                    "quad_groups": quad_groups,
                    "wang_groups": [group for group in groups if "Qh" in group],
                    "winning_hand_wang_count": list(winning_hand).count("Qh"),
                    "hu_prompt_seen_for_winner": bool(prompts),
                    "server_hu_action_seen": bool(hu_actions),
                    "terminal_path": terminal_path,
                }
            )
            round_prompts.clear()
            round_hu_actions.clear()
    return evidence


def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "logs" / f"qs_packets_{stamp}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", help="raw TCP stream, pcap, or pcapng file")
    parser.add_argument("--output", default=None, help="JSONL output path; defaults under chenzhou_zipai_ai/logs")
    parser.add_argument("--json", action="store_true", help="print compact JSON summary")
    args = parser.parse_args()

    source = Path(args.capture)
    data = source.read_bytes()
    packets = extract_qs_packets(data, source=str(source))
    output = write_qs_jsonl(packets, args.output or _default_output_path())
    terminal_evidence = collect_terminal_evidence(packets)
    summary = {
        "source": str(source),
        "output": str(output),
        "packet_count": len(packets),
        "terminal_evidence_count": len(terminal_evidence),
        "chenzhou_multi_wang_terminal_count": sum(
            item.get("gametype") == "30" and int(item.get("wangnum") or 0) > 0
            for item in terminal_evidence
        ),
        "terminal_evidence": terminal_evidence,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"QS_PACKET_JSONL={output}")
        print(f"QS_PACKET_COUNT={len(packets)}")


if __name__ == "__main__":
    main()
