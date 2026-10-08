"""Validate a round log folder or bundle zip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chenzhou_zipai_ai.game_logging.validate_logs import validate_round_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a round log bundle or round directory.")
    parser.add_argument("round_path", help="round directory or exported round bundle zip")
    args = parser.parse_args()
    validate_round_path(Path(args.round_path))


if __name__ == "__main__":
    main()
