"""Collect screenshots from Android device."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture.adb_capture import save_screen


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a PNG screenshot from Android via ADB.")
    parser.add_argument("--out", default="data/screenshots", help="Output file or directory.")
    parser.add_argument("--device-id", help="ADB device serial.")
    args = parser.parse_args()

    path = save_screen(args.out, device_id=args.device_id)
    print(path)


if __name__ == "__main__":
    main()
