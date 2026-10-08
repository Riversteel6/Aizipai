"""Production entry point for the screenshot-only live runtime."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from tools.live_assistant import main as live_main

    live_main()


if __name__ == "__main__":
    main()
