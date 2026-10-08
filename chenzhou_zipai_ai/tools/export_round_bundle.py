"""CLI wrapper to export a single round log folder as zip bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from chenzhou_zipai_ai.game_logging import export_round_bundle
from chenzhou_zipai_ai.game_logging import replay_loader


def resolve_round_path(root: Path, session_id: str | None, round_id: str | None) -> Path:
    if session_id is None and round_id is None:
        session_dir, round_dir, _, _ = replay_loader.latest_round_bundle(root)
        return round_dir
    sessions_root = root / "sessions"
    target_session = sessions_root / session_id if session_id else None
    if not target_session or not target_session.exists():
        raise FileNotFoundError(f"session not found: {session_id}")
    if round_id is None:
        _, target_round, _, _ = replay_loader.latest_round_bundle(target_session)
        return target_round
    target_round = target_session / "rounds" / round_id
    if not target_round.exists():
        raise FileNotFoundError(f"round not found: {target_round}")
    return target_round


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a single round as zip bundle.")
    parser.add_argument("round_path", nargs="?", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--round-id", default=None)
    parser.add_argument("--logs-root", default="chenzhou_zipai_ai/logs")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.logs_root)
    if args.round_path:
        round_path = Path(args.round_path)
    else:
        round_path = resolve_round_path(root, args.session_id, args.round_id)

    out = export_round_bundle(round_path, Path(args.output_dir) if args.output_dir else None)
    print(out.round_bundle_path)


if __name__ == "__main__":
    main()
