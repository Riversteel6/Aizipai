"""Summarize Frida direct-attach and Gadget readiness without changing the phone."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture.adb_capture import run_adb


def frida_readiness(
    *,
    device_id: str,
    package: str = "com.jiahe.kaixinpaohuzi",
    gadget_apk: str | Path = WORKSPACE / "dist/base_frida_gadget_unsigned.apk",
) -> dict:
    frida = shutil.which("frida")
    pid = ""
    try:
        pid = run_adb(["shell", "pidof", package], device_id=device_id, timeout=10).strip().split()[0]
    except Exception:
        pid = ""
    result = {
        "device_id": device_id,
        "package": package,
        "pid": pid,
        "frida_cli": frida,
        "direct_attach_ok": False,
        "direct_attach_error": "",
        "gadget_apk": str(gadget_apk),
        "gadget_apk_exists": Path(gadget_apk).exists(),
        "requires_reinstall_for_gadget": True,
        "safe_install_note": "Do not install over the current game unless account/data risk is accepted.",
    }
    if frida and pid:
        command = [frida, "-D", device_id, "-p", pid, "-e", "console.log('probe');", "--runtime=v8"]
        try:
            probe = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
            result["direct_attach_ok"] = probe.returncode == 0 and "Failed to attach" not in (probe.stdout + probe.stderr)
            result["direct_attach_error"] = (probe.stdout + probe.stderr).strip()[-500:]
        except Exception as exc:
            result["direct_attach_error"] = f"{type(exc).__name__}:{exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--package", default="com.jiahe.kaixinpaohuzi")
    parser.add_argument("--gadget-apk", default=str(WORKSPACE / "dist/base_frida_gadget_unsigned.apk"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = frida_readiness(device_id=args.device_id, package=args.package, gadget_apk=args.gadget_apk)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"pid={result['pid']} frida_cli={result['frida_cli']}")
    print(f"direct_attach_ok={result['direct_attach_ok']}")
    if result["direct_attach_error"]:
        print(f"direct_attach_error={result['direct_attach_error']}")
    print(f"gadget_apk_exists={result['gadget_apk_exists']} gadget_apk={result['gadget_apk']}")
    print(result["safe_install_note"])


if __name__ == "__main__":
    main()
