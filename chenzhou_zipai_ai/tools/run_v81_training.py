"""Run an exact number of frozen v8.1 two-player replay games."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.frozen_two_player_strategy import FROZEN_CANDIDATE, FROZEN_RELEASES


HEADS_UP_OPPONENTS = (
    "independent_balanced",
    "independent_pressure",
    "independent_denial",
    "information_set_search",
    "aggressive_meld",
    "defensive_search",
    "red_black_search",
)


def allocate_games(total: int) -> list[tuple[str, int]]:
    if total < 1:
        raise ValueError("games must be positive")
    base, remainder = divmod(total, len(HEADS_UP_OPPONENTS))
    return [
        (opponent, base + int(index < remainder))
        for index, opponent in enumerate(HEADS_UP_OPPONENTS)
        if base + int(index < remainder) > 0
    ]


def build_chunk_command(
    *,
    opponent: str,
    games: int,
    wildcard: str,
    seed: int,
    workers: int,
    summary_path: Path,
    trace_path: Path,
    checkpoint_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "tools.run_opponent_league",
        "--candidate",
        FROZEN_CANDIDATE,
        "--deals-per-matchup",
        str(games),
        "--workers",
        str(max(1, workers)),
        "--seed",
        str(seed),
        "--wildcard",
        wildcard,
        "--players",
        "2",
        "--matchup",
        opponent,
        "--output",
        str(summary_path),
        "--trace-output",
        str(trace_path),
        "--checkpoint-output",
        str(checkpoint_path),
        "--balanced-rotation-only",
        "--progress-every",
        str(max(1, games // 20)),
        "--heartbeat-seconds",
        "5",
    ]


def _package_run(
    run_dir: Path,
    manifest_path: Path,
    chunk_rows: list[dict[str, Any]],
    *,
    runner_log_path: Path,
) -> Path:
    archive_path = run_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, manifest_path.name)
        archive.write(runner_log_path, runner_log_path.name)
        for row in chunk_rows:
            for key in ("summary", "trace"):
                path = Path(row[key])
                archive.write(path, path.name)
    return archive_path


def _duration_text(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "暂不可估算"
    whole = int(round(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def format_child_output(
    line: str,
    *,
    opponent: str,
    chunk_games: int,
    completed_before: int,
    requested_games: int,
    run_started: float,
    now: float | None = None,
) -> str:
    """Turn the league JSON stream into a compact human-readable status line."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return f"[运行信息] {line}"

    event = str(payload.get("event") or "")
    if event in {"league_progress", "league_heartbeat"}:
        chunk_completed = max(0, int(payload.get("completed") or 0))
        overall_completed = min(requested_games, completed_before + chunk_completed)
        percent = 100.0 * overall_completed / max(1, requested_games)
        elapsed = max(0.0, (time.perf_counter() if now is None else now) - run_started)
        eta = None
        if overall_completed > 0:
            eta = elapsed * (requested_games - overall_completed) / overall_completed
        if chunk_completed == 0:
            detail = "正在计算本组第1局"
        else:
            detail = (
                f"本组 {chunk_completed}/{chunk_games}，"
                f"胜/负/和 {int(payload.get('wins') or 0)}/"
                f"{int(payload.get('losses') or 0)}/"
                f"{int(payload.get('draws') or 0)}"
            )
        return (
            f"进度 {overall_completed}/{requested_games} ({percent:.1f}%) | "
            f"对手 {opponent} | {detail} | "
            f"已用 {_duration_text(elapsed)} | 剩余约 {_duration_text(eta)}"
        )

    if "ok" in payload and "games" in payload:
        status = "本组通过" if bool(payload.get("ok")) else "本组跑完但未通过"
        runtime_errors = int(payload.get("strategy_runtime_errors") or 0)
        return (
            f"{status}：{opponent}，{int(payload.get('games') or 0)}局，"
            f"胜/负/和 {int(payload.get('candidate_wins') or 0)}/"
            f"{int(payload.get('losses') or 0)}/"
            f"{int(payload.get('draws') or 0)}，策略运行异常 {runtime_errors} 次"
        )
    return f"[运行信息] {line}"


def _run_chunk_with_progress(
    command: list[str],
    *,
    log_path: Path,
    opponent: str,
    chunk_games: int,
    completed_before: int,
    requested_games: int,
    run_started: float,
) -> int:
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=APP_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            log_handle.write(f"{datetime.now().isoformat()} {line}\n")
            print(
                format_child_output(
                    line,
                    opponent=opponent,
                    chunk_games=chunk_games,
                    completed_before=completed_before,
                    requested_games=requested_games,
                    run_started=run_started,
                ),
                flush=True,
            )
        return process.wait()


def main(
    *,
    product_label: str = "v8.1",
    file_label: str = "v81",
    schema_version: str = "v81-custom-training-v1",
    fixed_wildcard: str | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"Run frozen {product_label} 1v1 simulations and package complete "
            "decision traces."
        )
    )
    parser.add_argument("--games", type=int, required=True)
    if fixed_wildcard is None:
        parser.add_argument("--wildcard", choices=("on", "off"), required=True)
    else:
        parser.set_defaults(wildcard=fixed_wildcard)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(f"logs/{file_label}_training"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.games < 1:
        parser.error("--games must be positive")
    mode = "wang" if args.wildcard == "on" else "no_wang"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root if args.output_root.is_absolute() else APP_ROOT / args.output_root
    run_dir = output_root / f"{file_label}_1v1_{mode}_{args.games}games_{stamp}"
    allocation = allocate_games(args.games)

    plan = {
        "schema_version": schema_version,
        "candidate": FROZEN_CANDIDATE,
        "release_id": FROZEN_RELEASES[args.wildcard == "on"],
        "players": 2,
        "wildcard_enabled": args.wildcard == "on",
        "requested_games": args.games,
        "seed": args.seed,
        "workers": max(1, args.workers),
        "allocation": [
            {"opponent": opponent, "games": games}
            for opponent, games in allocation
        ],
        "run_dir": str(run_dir),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    runner_log_path = run_dir / "runner.log"
    runner_log_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mode_name = "有王" if args.wildcard == "on" else "无王"
    run_started = time.perf_counter()
    print("=" * 66, flush=True)
    print(f"{product_label} 两人{mode_name}模式离线复盘测试", flush=True)
    print(f"总局数：{args.games} | 对手类型：{len(allocation)}组", flush=True)
    print(f"完整记录目录：{run_dir}", flush=True)
    print("每局都会保存完整决策轨迹；下方每5秒更新运行状态。", flush=True)
    print("=" * 66, flush=True)
    chunk_rows: list[dict[str, Any]] = []
    for index, (opponent, games) in enumerate(allocation, start=1):
        prefix = f"{index:02d}_{opponent}"
        summary_path = run_dir / f"{prefix}_summary.json"
        trace_path = run_dir / f"{prefix}_decisions.jsonl.gz"
        checkpoint_path = run_dir / f"{prefix}_checkpoint.jsonl"
        command = build_chunk_command(
            opponent=opponent,
            games=games,
            wildcard=args.wildcard,
            seed=args.seed + (index - 1) * 1_000_000,
            workers=args.workers,
            summary_path=summary_path,
            trace_path=trace_path,
            checkpoint_path=checkpoint_path,
        )
        completed_before = sum(row["games"] for row in chunk_rows)
        print(
            f"开始第 {index}/{len(allocation)} 组：{opponent}，"
            f"本组 {games} 局，总进度 {completed_before}/{args.games}",
            flush=True,
        )
        returncode = _run_chunk_with_progress(
            command,
            log_path=runner_log_path,
            opponent=opponent,
            chunk_games=games,
            completed_before=completed_before,
            requested_games=args.games,
            run_started=run_started,
        )
        if returncode != 0:
            print("", file=sys.stderr)
            print(f"训练中止：对手组 {opponent} 返回错误码 {returncode}。", file=sys.stderr)
            print(f"已完成内容保留在：{checkpoint_path}", file=sys.stderr)
            print(f"详细错误日志：{runner_log_path}", file=sys.stderr)
            return returncode
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        chunk_rows.append(
            {
                "opponent": opponent,
                "games": int(report["games"]),
                "summary": str(summary_path),
                "trace": str(trace_path),
                "checkpoint": str(checkpoint_path),
                "ok": bool(report.get("ok")),
            }
        )

    completed_games = sum(row["games"] for row in chunk_rows)
    manifest = {
        **plan,
        "completed_games": completed_games,
        "complete": completed_games == args.games and all(row["ok"] for row in chunk_rows),
        "chunks": chunk_rows,
        "runner_log": str(runner_log_path),
        "created_at": datetime.now().isoformat(),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    archive_path = _package_run(
        run_dir,
        manifest_path,
        chunk_rows,
        runner_log_path=runner_log_path,
    )
    elapsed = time.perf_counter() - run_started
    print("=" * 66, flush=True)
    print(f"全部完成：{completed_games}/{args.games} 局，用时 {_duration_text(elapsed)}", flush=True)
    print(f"复盘压缩包：{archive_path}", flush=True)
    print("把这个ZIP发来即可逐局、逐决策复盘。", flush=True)
    print("=" * 66, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
