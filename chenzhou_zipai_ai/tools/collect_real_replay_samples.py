"""Collect real no-click replay samples from the connected training phone."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chenzhou_zipai_ai.game_logging import GameLogger, LoggerConfig
from tools.live_assistant import DEFAULT_DEVICE, run_once
from tools.verify_real_replays import verify_replay_logs


Runner = Callable[..., dict[str, Any]]
Verifier = Callable[..., dict[str, Any]]


def collect_real_replay_samples(
    *,
    device_id: str = DEFAULT_DEVICE,
    logs_root: Path = ROOT / "logs",
    target_decisions: int = 10,
    max_steps: int = 40,
    delay_seconds: float = 1.0,
    expected_total: int = 20,
    memory_file: Path | None = None,
    reset_memory: bool = False,
    simulations: int = 0,
    runner: Runner = run_once,
    verifier: Verifier = verify_replay_logs,
) -> dict[str, Any]:
    logs_root = Path(logs_root)
    memory_file = memory_file or (logs_root / "real_sample_memory.json")
    logger = GameLogger(
        logs_root,
        config=LoggerConfig(
            save_raw_screenshot=True,
            save_debug_screenshot=True,
            save_every_frame=False,
            save_decision_frames=True,
            save_error_frames=True,
            save_before_after_action=True,
        ),
        dry_run=True,
        execute_enabled=False,
    )
    collected: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    try:
        logger.start_session(
            device_id=device_id,
            execute_enabled=False,
            notes="collect_real_replay_samples no-click dry-run capture",
        )
        for step in range(1, max_steps + 1):
            result = runner(
                device_id=device_id,
                expected_total=expected_total,
                memory_file=memory_file,
                reset_memory=reset_memory and step == 1,
                execute_settlement_ready=False,
                execute_play_actions=False,
                simulations=simulations,
                logger=logger,
                logs_root=logs_root,
            )
            row = {
                "step": step,
                "flow_state": (result.get("flow") or {}).get("state"),
                "decision_id": result.get("decision_id"),
                "action": (result.get("decision") or {}).get("action"),
                "label": (result.get("decision") or {}).get("label")
                or (result.get("decision") or {}).get("selected_label"),
                "action_plan_ready": (result.get("action_plan") or {}).get("ready"),
                "executed": result.get("executed"),
            }
            step_rows.append(row)
            if row["decision_id"]:
                collected.append(row)
            if len(collected) >= target_decisions:
                break
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    finally:
        if logger.round_active:
            logger.end_round(result="real_sample_dry_run")
        if logger.session_active:
            logger.end_session(notes="real sample collection finished")

    verification = verifier(logs_root, min_decisions=target_decisions, include_fixtures=False)
    report = {
        "ok": len(collected) >= target_decisions and bool(verification.get("ok")),
        "device_id": device_id,
        "logs_root": str(logs_root),
        "target_decisions": target_decisions,
        "decisions_collected": len(collected),
        "steps": len(step_rows),
        "max_steps": max_steps,
        "execute_play_actions": False,
        "tap_executed": False,
        "collected": collected,
        "step_rows": step_rows,
        "verification": verification,
    }
    logs_root.mkdir(parents=True, exist_ok=True)
    (logs_root / "real_replay_sample_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect no-click real replay samples from ADB screenshots.")
    parser.add_argument("--device-id", default=DEFAULT_DEVICE)
    parser.add_argument("--logs-root", type=Path, default=ROOT / "logs")
    parser.add_argument("--target-decisions", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--expected-total", type=int, default=20)
    parser.add_argument("--memory-file", type=Path, default=None)
    parser.add_argument("--reset-memory", action="store_true")
    parser.add_argument("--simulations", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = collect_real_replay_samples(
        device_id=args.device_id,
        logs_root=args.logs_root,
        target_decisions=args.target_decisions,
        max_steps=args.max_steps,
        delay_seconds=args.delay_seconds,
        expected_total=args.expected_total,
        memory_file=args.memory_file,
        reset_memory=args.reset_memory,
        simulations=args.simulations,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        status = "ok" if report["ok"] else "FAIL"
        verification = report["verification"]
        print(
            f"{status}\tdecisions={report['decisions_collected']}/{report['target_decisions']}\t"
            f"steps={report['steps']}\treplay={verification.get('decisions_matched', 0)}/"
            f"{verification.get('min_decisions', report['target_decisions'])}\tlogs={report['logs_root']}"
        )
        if not report["ok"]:
            for item in verification.get("errors", [])[:10]:
                print(f"  ERROR {item}")
            for item in verification.get("mismatches", [])[:10]:
                print(f"  MISMATCH {item}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
