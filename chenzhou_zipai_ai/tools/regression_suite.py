"""One-command local regression runner for the professional brain."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> dict:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
    }


def run_regression_suite(*, include_pytest: bool = True) -> dict:
    commands = []
    if include_pytest:
        commands.append([sys.executable, "-m", "pytest", "-q"])
    commands.extend(
        [
            [sys.executable, "chenzhou_zipai_ai/tools/eval_policy_cases.py"],
            [sys.executable, "chenzhou_zipai_ai/tools/simulate_random_hands.py", "--count", "50"],
            [sys.executable, "chenzhou_zipai_ai/tools/dry_run_50_games.py", "--count", "50"],
            [sys.executable, "chenzhou_zipai_ai/tools/run_self_play_calibration.py", "--count", "20", "--simulations", "12"],
            [sys.executable, "chenzhou_zipai_ai/tools/replay_policy_fixtures.py"],
        ]
    )
    rows = [_run(command) for command in commands]
    return {
        "ok": all(row["ok"] for row in rows),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local regression suite.")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()
    result = run_regression_suite(include_pytest=not args.skip_pytest)
    for row in result["results"]:
        status = "ok" if row["ok"] else "FAIL"
        print(f"{status}\t{row['command']}")
        if not row["ok"]:
            if row["stdout"]:
                print(row["stdout"])
            if row["stderr"]:
                print(row["stderr"])
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
