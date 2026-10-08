"""Benchmark production strategy latency across deterministic deck samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.performance_benchmark import run_strategy_latency_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--wildcard", choices=("off", "on"), required=True)
    parser.add_argument("--players", type=int, choices=(2, 3), default=3)
    parser.add_argument("--max-ms", type=float, default=8_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_strategy_latency_benchmark(
        samples=max(1, args.samples),
        seed=args.seed,
        wildcard_enabled=args.wildcard == "on",
        max_allowed_ms=args.max_ms,
        players=args.players,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "ok",
                    "policy_version",
                    "wildcard_enabled",
                    "players",
                    "samples",
                    "mean_ms",
                    "median_ms",
                    "p95_ms",
                    "max_ms",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
