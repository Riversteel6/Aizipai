"""Run offline self-play calibration for policy versus Monte Carlo EV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai.self_play import apply_weight_adjustments, run_self_play_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline self-play calibration.")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--hand-size", type=int, default=14)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=Path("config/rules.yaml"))
    parser.add_argument("--write-tuned-config", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_self_play_calibration(
        count=args.count,
        hand_size=args.hand_size,
        simulations=args.simulations,
        seed=args.seed,
        output_dir=args.output_dir,
        config_path=str(args.config_path),
    )
    tuning_result = None
    if args.write_tuned_config:
        tuning_result = apply_weight_adjustments(
            config_path=args.config_path,
            output_path=args.write_tuned_config,
            report=report,
        )
    if args.json:
        payload = {"calibration": report, "tuning_result": tuning_result}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"self_play ok={report['ok']} decisions={report['count']} "
            f"mismatches={report['label_mismatches']} worst_gap={report['worst_long_term_gap']}"
        )
        if args.output_dir:
            print(f"report={args.output_dir}")
        if tuning_result:
            print(
                f"tuned_config={tuning_result['output_path']} "
                f"updates={len(tuning_result['applied_updates'])}"
            )
        for row in report["hard_violations"][:10]:
            print(f"  hard_violation {row['index']}: {' '.join(row['hand'])}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
