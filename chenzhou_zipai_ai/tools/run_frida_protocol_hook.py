"""Launch the read-only Frida protocol hook if Frida is installed."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "tools/frida_protocol_hook.js"
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture.adb_capture import run_adb


def resolve_pid(package: str, *, device_id: str | None = None) -> str:
    output = run_adb(["shell", "pidof", package], device_id=device_id, timeout=10).strip()
    return output.split()[0] if output else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--package", default="com.jiahe.kaixinpaohuzi")
    parser.add_argument("--pid", default=None)
    args = parser.parse_args()
    frida = shutil.which("frida")
    if not frida:
        print("FRIDA_AVAILABLE=False")
        print("reason=frida CLI not found on this PC")
        sys.exit(2)
    pid = args.pid or resolve_pid(args.package, device_id=args.device_id)
    target_args = ["-p", pid] if pid else ["-n", args.package]
    command = [frida, "-U", *target_args, "-l", str(HOOK)]
    if args.device_id:
        command = [frida, "-D", args.device_id, *target_args, "-l", str(HOOK)]
    print("FRIDA_AVAILABLE=True")
    print(f"target_pid={pid or 'not_found'}")
    print("command=" + " ".join(command))
    subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
