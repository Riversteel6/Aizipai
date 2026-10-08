"""Run seat-balanced full-game strategy benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.full_game_simulator import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full two- or three-player Chenzhou Zipai simulations.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--wildcard", choices=("on", "off"), default="off")
    parser.add_argument("--players", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--candidate-policy",
        choices=("information_set_search", "professional_brain"),
        default="information_set_search",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_benchmark(
        args.games,
        wildcard_enabled=args.wildcard == "on",
        seed=args.seed,
        workers=max(1, args.workers),
        candidate_policy=args.candidate_policy,
        players=args.players,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        summary = {key: value for key, value in report.items() if key != "results"}
        summary["output"] = str(args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(payload)


if __name__ == "__main__":
    main()
