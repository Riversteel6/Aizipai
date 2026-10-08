"""Export latest logged round as zip bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chenzhou_zipai_ai.game_logging import export_round_bundle
from chenzhou_zipai_ai.game_logging import replay_loader
from tools.print_round_as_markdown import print_round_as_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Export latest logged round for review.")
    parser.add_argument("--logs-root", type=Path, default=Path("chenzhou_zipai_ai/logs"))
    args = parser.parse_args()
    _, round_path, session_id, round_id = replay_loader.latest_round_bundle(args.logs_root)
    review_path = print_round_as_markdown(round_path)
    result = export_round_bundle(round_path)
    print(f"session={session_id} round={round_id}")
    print(f"ROUND_REVIEW_MD={review_path.resolve()}")
    print(f"ROUND_BUNDLE_ZIP={result.round_bundle_path.resolve()}")


if __name__ == "__main__":
    main()
