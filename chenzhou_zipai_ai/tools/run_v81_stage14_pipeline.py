"""Run the long v8.1 WIN_FIRST validation pipeline unattended.

The league is checkpointed per game and resumes only when its fixed
configuration matches.  This driver does not relax any acceptance threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parent
REPORTS = WORKSPACE_ROOT / "reports"
CHECKPOINTS = REPORTS / "checkpoints"
STATUS_PATH = REPORTS / "v81_win_first_stage14_pipeline_status_20260822.json"
LOG_PATH = REPORTS / "v81_win_first_stage14_pipeline_20260822.log"
LEAGUE_OUTPUT = REPORTS / "v81_win_first_stage14_final504_2p_on_seed60260822.json"
LEAGUE_TRACE = REPORTS / "v81_win_first_stage14_final504_2p_on_seed60260822.jsonl.gz"
LEAGUE_CHECKPOINT = (
    CHECKPOINTS / "v81_win_first_stage14_final504_2p_on_seed60260822.jsonl"
)
EXPLOIT_OUTPUT = REPORTS / "v81_win_first_stage14_best_response_2p_wang_20260822.json"
PYTEST_OUTPUT = REPORTS / "v81_win_first_stage14_full_pytest_20260822.txt"
CANDIDATE = "professional_v81_two_player_exact_discard_sharded_research"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(*, phase: str, state: str, **details: object) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v81-win-first-stage14-pipeline-status-v1",
        "updated_at": _utc_now(),
        "pipeline_pid": os.getpid(),
        "phase": phase,
        "state": state,
        **details,
    }
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATUS_PATH)


def _completed_league_exists() -> bool:
    if not LEAGUE_OUTPUT.exists():
        return False
    try:
        report = json.loads(LEAGUE_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        report.get("ok")
        and not report.get("early_stopped")
        and int(report.get("games") or 0) == 504
        and int(report.get("expected_games") or 0) == 504
    )


def _completed_exploit_exists() -> bool:
    if not EXPLOIT_OUTPUT.exists():
        return False
    try:
        report = json.loads(EXPLOIT_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        report.get("ok")
        and int(report.get("train_deals_per_mode") or 0) == 20
        and int(report.get("holdout_deals_per_mode") or 0) == 20
        and int(report.get("train_holdout_seed_overlap") or -1) == 0
    )


def _run_step(name: str, command: Sequence[str], *, output: Path | None = None) -> None:
    _write_status(
        phase=name,
        state="running",
        command=list(command),
        log=str(LOG_PATH),
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] START {name}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=APP_ROOT,
            stdout=subprocess.PIPE if output is not None else log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        _write_status(
            phase=name,
            state="running",
            child_pid=process.pid,
            command=list(command),
            log=str(LOG_PATH),
        )
        captured, _ = process.communicate()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(captured or "", encoding="utf-8")
            log.write(captured or "")
        log.write(f"\n[{_utc_now()}] END {name} exit={process.returncode}\n")
        log.flush()
    if process.returncode:
        _write_status(
            phase=name,
            state="failed",
            exit_code=process.returncode,
            log=str(LOG_PATH),
        )
        raise SystemExit(process.returncode)


def _commands() -> list[tuple[str, list[str], Path | None]]:
    league = [
        sys.executable,
        str(APP_ROOT / "tools" / "run_opponent_league.py"),
        "--candidate",
        CANDIDATE,
        "--deals-per-matchup",
        "18",
        "--workers",
        "1",
        "--seed",
        "60260822",
        "--wildcard",
        "on",
        "--players",
        "2",
        "--output",
        str(LEAGUE_OUTPUT),
        "--trace-output",
        str(LEAGUE_TRACE),
        "--checkpoint-output",
        str(LEAGUE_CHECKPOINT),
        "--progress-every",
        "4",
        "--heartbeat-seconds",
        "30",
    ]
    if LEAGUE_CHECKPOINT.exists():
        league.append("--resume")
    exploit = [
        sys.executable,
        str(APP_ROOT / "tools" / "train_exploit_opponent.py"),
        "--target",
        CANDIDATE,
        "--train-deals-per-mode",
        "20",
        "--holdout-deals-per-mode",
        "20",
        "--train-seed",
        "70260822",
        "--holdout-seed",
        "80260822",
        "--workers",
        "1",
        "--two-player-wang-only",
        "--output",
        str(EXPLOIT_OUTPUT),
    ]
    tests = [sys.executable, "-m", "pytest", "-q"]
    return [
        ("formal_league", league, None),
        ("best_response", exploit, None),
        ("full_pytest", tests, PYTEST_OUTPUT),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commands = _commands()
    if args.dry_run:
        print(json.dumps(commands, ensure_ascii=False, indent=2, default=str))
        return 0

    _write_status(phase="startup", state="running", log=str(LOG_PATH))
    for name, command, output in commands:
        if name == "formal_league" and _completed_league_exists():
            continue
        if name == "best_response" and _completed_exploit_exists():
            continue
        _run_step(name, command, output=output)
    _write_status(
        phase="complete",
        state="awaiting_acceptance_analysis",
        league_report=str(LEAGUE_OUTPUT),
        exploit_report=str(EXPLOIT_OUTPUT),
        pytest_report=str(PYTEST_OUTPUT),
        log=str(LOG_PATH),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
