"""Print or execute the install command for a Gadget APK."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", default=r"D:\Codex Work\aizipai\dist\base_frida_gadget_unsigned.apk")
    parser.add_argument("--device-id", default="3B1F5WEA9BBUX9ZQ")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    command = ["adb", "-s", args.device_id, "install", "-r", "-t", str(Path(args.apk))]
    print("INSTALL_COMMAND=" + " ".join(command))
    print("WARNING=This may fail due to signature mismatch or replace the current game install if accepted.")
    if args.execute:
        subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
