"""Verify replay consistency across persisted round logs."""

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

from chenzhou_zipai_ai.game_logging import replay_loader
from chenzhou_zipai_ai.game_logging.validate_logs import validate_round_bundle
from tools.replay_decision import _replay_one_decision


def _round_dirs(logs_root: Path) -> list[Path]:
    sessions_root = logs_root / "sessions"
    if not sessions_root.exists():
        return []
    rounds: list[Path] = []
    sessions = sorted(
        (path for path in sessions_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for session in sessions:
        rounds_root = session / "rounds"
        if not rounds_root.exists():
            continue
        rounds.extend(
            sorted(
                (path for path in rounds_root.iterdir() if path.is_dir() and path.name.startswith("round_")),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
    return rounds


def _is_fixture_round(bundle) -> bool:
    return bundle.session_meta.get("device_id") == "fixture" or bundle.round_meta.get("result") == "fixture"


def verify_replay_logs(
    logs_root: Path,
    *,
    min_decisions: int = 10,
    include_fixtures: bool = False,
) -> dict:
    checked_decisions = 0
    matched_decisions = 0
    round_count = 0
    errors: list[str] = []
    mismatches: list[dict] = []
    for round_path in _round_dirs(logs_root):
        try:
            bundle = replay_loader.load_round_bundle_from_path(round_path)
        except Exception as exc:
            errors.append(f"{round_path}: load failed: {type(exc).__name__}: {exc}")
            continue
        if _is_fixture_round(bundle) and not include_fixtures:
            continue
        validation = validate_round_bundle(round_path)
        if not validation["ok"]:
            errors.extend(f"{round_path}: {item}" for item in validation["errors"])
            continue
        round_count += 1
        for decision in bundle.decisions:
            decision_id = str(decision.get("decision_id") or "")
            if not decision_id:
                continue
            try:
                replayed = _replay_one_decision(bundle, decision_id)
            except Exception as exc:
                errors.append(f"{round_path} {decision_id}: replay failed: {type(exc).__name__}: {exc}")
                continue
            checked_decisions += 1
            if replayed["same_action"] and replayed["same_label"]:
                matched_decisions += 1
            else:
                mismatches.append(
                    {
                        "round": str(round_path),
                        "decision_id": decision_id,
                        "logged": replayed["logged"],
                        "replayed": replayed["replayed"],
                    }
                )
        if checked_decisions >= min_decisions and not mismatches and not errors:
            break
    ok = checked_decisions >= min_decisions and matched_decisions == checked_decisions and not mismatches and not errors
    return {
        "ok": ok,
        "logs_root": str(logs_root),
        "rounds_checked": round_count,
        "decisions_checked": checked_decisions,
        "decisions_matched": matched_decisions,
        "min_decisions": min_decisions,
        "include_fixtures": include_fixtures,
        "mismatches": mismatches,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify replay consistency over persisted round logs.")
    parser.add_argument("--logs-root", type=Path, default=Path("chenzhou_zipai_ai/logs"))
    parser.add_argument("--min-decisions", type=int, default=10)
    parser.add_argument("--include-fixtures", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify_replay_logs(
        args.logs_root,
        min_decisions=args.min_decisions,
        include_fixtures=args.include_fixtures,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = "ok" if result["ok"] else "FAIL"
        print(
            f"{status}\trounds={result['rounds_checked']}\t"
            f"decisions={result['decisions_checked']}/{result['min_decisions']}\t"
            f"matched={result['decisions_matched']}"
        )
        for item in result["mismatches"]:
            print(f"  mismatch {item['decision_id']}: logged={item['logged']} replayed={item['replayed']}")
        for item in result["errors"][:20]:
            print(f"  ERROR {item}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
