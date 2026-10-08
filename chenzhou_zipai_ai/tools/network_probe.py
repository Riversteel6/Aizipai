"""Read-only network socket probe for the running game process."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture.adb_capture import run_adb

TCP_STATES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "03": "SYN_RECV",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
}


def package_uid(package: str, *, device_id: str | None = None) -> int | None:
    output = run_adb(["shell", "dumpsys", "package", package], device_id=device_id, timeout=10)
    match = re.search(r"\buid=(\d+)\b", output)
    return int(match.group(1)) if match else None


def probe_network(package: str = "com.jiahe.kaixinpaohuzi", *, device_id: str | None = None) -> dict[str, Any]:
    uid = package_uid(package, device_id=device_id)
    rows: list[dict[str, Any]] = []
    for table in ("tcp", "tcp6"):
        output = run_adb(["shell", "cat", f"/proc/net/{table}"], device_id=device_id, timeout=10)
        rows.extend(_parse_tcp_table(output, table=table, uid=uid))
    rows.sort(key=lambda item: (item["state"] != "ESTABLISHED", item["remote_ip"], item["remote_port"]))
    return {"package": package, "uid": uid, "connections": rows}


def _parse_tcp_table(text: str, *, table: str, uid: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            row_uid = int(parts[7])
        except ValueError:
            continue
        if uid is not None and row_uid != uid:
            continue
        local_ip, local_port = _decode_address(parts[1], table=table)
        remote_ip, remote_port = _decode_address(parts[2], table=table)
        rows.append(
            {
                "table": table,
                "uid": row_uid,
                "state": TCP_STATES.get(parts[3], parts[3]),
                "local_ip": local_ip,
                "local_port": local_port,
                "remote_ip": remote_ip,
                "remote_port": remote_port,
                "inode": parts[9] if len(parts) > 9 else None,
            }
        )
    return rows


def _decode_address(value: str, *, table: str) -> tuple[str, int]:
    host_hex, port_hex = value.split(":", 1)
    port = int(port_hex, 16)
    if table == "tcp":
        octets = [str(int(host_hex[index : index + 2], 16)) for index in range(6, -1, -2)]
        return ".".join(octets), port
    if host_hex.startswith("0000000000000000FFFF0000"):
        ipv4 = host_hex[-8:]
        octets = [str(int(ipv4[index : index + 2], 16)) for index in range(6, -1, -2)]
        return ".".join(octets), port
    chunks = [host_hex[index : index + 4] for index in range(0, len(host_hex), 4)]
    return ":".join(chunks), port


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--package", default="com.jiahe.kaixinpaohuzi")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = probe_network(args.package, device_id=args.device_id)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"package={result['package']} uid={result['uid']} connections={len(result['connections'])}")
    for row in result["connections"]:
        print(
            f"{row['state']}\t{row['local_ip']}:{row['local_port']}\t"
            f"{row['remote_ip']}:{row['remote_port']}\t{row['table']}"
        )


if __name__ == "__main__":
    main()
