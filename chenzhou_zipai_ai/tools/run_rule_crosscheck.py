"""Run the independent-vs-production rule crosscheck."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from audit.differential_rules import run_rule_crosscheck


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hu-cases", type=int, default=500)
    parser.add_argument("--chi-cases", type=int, default=500)
    parser.add_argument("--discard-cases", type=int, default=500)
    parser.add_argument("--response-cases", type=int, default=500)
    parser.add_argument("--draw-auto-cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--wildcard", choices=("off", "on"), required=True)
    parser.add_argument("--players", type=int, choices=(2, 3), default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_rule_crosscheck(
        hu_cases=max(1, args.hu_cases),
        chi_cases=max(1, args.chi_cases),
        discard_cases=max(1, args.discard_cases),
        response_cases=max(1, args.response_cases),
        draw_auto_cases=max(1, args.draw_auto_cases),
        seed=args.seed,
        wildcard_enabled=args.wildcard == "on",
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
                    "wildcard_enabled",
                    "players",
                    "room_shape",
                    "hu_cases",
                    "hu_complete_cases",
                    "chi_cases",
                    "chi_positive_cases",
                    "discard_cases",
                    "discard_locked_cases",
                    "response_cases",
                    "response_positive_cases",
                    "response_joint_cases",
                    "response_auto_pao_cases",
                    "response_transition_cases",
                    "draw_auto_cases",
                    "draw_auto_positive_cases",
                    "draw_auto_skip_cases",
                    "mismatch_count",
                    "mismatches_by_domain",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
