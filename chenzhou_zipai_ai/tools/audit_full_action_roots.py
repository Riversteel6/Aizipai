"""Write a four-mode full-action root-search coverage report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.full_action_audit import audit_full_action_roots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots-per-mode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_full_action_roots(
        roots_per_mode=max(1, args.roots_per_mode),
        seed=args.seed,
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
                    "roots",
                    "all_legal_covered_roots",
                    "every_candidate_visited_roots",
                    "minimum_paired_determinizations",
                    "deadline_interrupted_roots",
                    "max_wall_ms",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
