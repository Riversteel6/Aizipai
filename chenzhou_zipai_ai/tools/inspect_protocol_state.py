"""Inspect normalized APK protocol fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vision.protocol_state import protocol_state_from_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", help="JSON text, key=value text, or a file path.")
    args = parser.parse_args()
    state = protocol_state_from_payload(args.payload)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
