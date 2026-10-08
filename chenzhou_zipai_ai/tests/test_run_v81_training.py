import json
import sys
from pathlib import Path

from ai.frozen_two_player_strategy import FROZEN_CANDIDATE
from tools.run_v81_training import (
    HEADS_UP_OPPONENTS,
    _run_chunk_with_progress,
    allocate_games,
    build_chunk_command,
    format_child_output,
)
from tools.v81_training_launcher import build_training_command, parse_game_count


def test_allocate_games_is_exact_and_balanced():
    allocation = allocate_games(500)
    counts = [games for _opponent, games in allocation]

    assert sum(counts) == 500
    assert len(allocation) == len(HEADS_UP_OPPONENTS)
    assert max(counts) - min(counts) <= 1


def test_build_chunk_command_freezes_v81_two_player_trace_inputs(tmp_path):
    command = build_chunk_command(
        opponent="independent_balanced",
        games=17,
        wildcard="on",
        seed=123,
        workers=1,
        summary_path=tmp_path / "summary.json",
        trace_path=tmp_path / "trace.jsonl.gz",
        checkpoint_path=tmp_path / "checkpoint.jsonl",
    )

    joined = " ".join(command)
    assert FROZEN_CANDIDATE in command
    assert "--players 2" in joined
    assert "--wildcard on" in joined
    assert "--deals-per-matchup 17" in joined
    assert "--balanced-rotation-only" in command
    assert "--trace-output" in command
    assert "--checkpoint-output" in command
    assert command[command.index("--heartbeat-seconds") + 1] == "5"


def test_human_progress_includes_total_progress_and_eta():
    message = format_child_output(
        json.dumps(
            {
                "event": "league_progress",
                "completed": 5,
                "total": 20,
                "wins": 3,
                "losses": 1,
                "draws": 1,
            }
        ),
        opponent="independent_balanced",
        chunk_games=20,
        completed_before=10,
        requested_games=100,
        run_started=100.0,
        now=110.0,
    )

    assert "进度 15/100 (15.0%)" in message
    assert "本组 5/20" in message
    assert "胜/负/和 3/1/1" in message
    assert "剩余约" in message


def test_failed_chunk_summary_is_not_presented_as_passed():
    message = format_child_output(
        json.dumps(
            {
                "ok": False,
                "games": 72,
                "candidate_wins": 47,
                "losses": 25,
                "draws": 0,
                "strategy_runtime_errors": 1,
            }
        ),
        opponent="independent_balanced",
        chunk_games=72,
        completed_before=0,
        requested_games=500,
        run_started=0.0,
        now=1.0,
    )

    assert "本组跑完但未通过" in message
    assert "策略运行异常 1 次" in message


def test_chunk_output_is_tee_logged_and_humanized(tmp_path, capsys):
    payload = {
        "event": "league_heartbeat",
        "completed": 0,
        "total": 1,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }
    command = [
        sys.executable,
        "-c",
        f"import json; print(json.dumps({payload!r}), flush=True)",
    ]
    log_path = tmp_path / "runner.log"

    returncode = _run_chunk_with_progress(
        command,
        log_path=log_path,
        opponent="independent_balanced",
        chunk_games=1,
        completed_before=0,
        requested_games=1,
        run_started=0.0,
    )

    assert returncode == 0
    assert '"event": "league_heartbeat"' in log_path.read_text(encoding="utf-8")
    assert "正在计算本组第1局" in capsys.readouterr().out


def test_python_launcher_validates_count_and_keeps_modes_separate():
    assert parse_game_count(" 1000 ") == 1000
    assert "--wildcard" in build_training_command(games=10, wildcard="on")
    assert build_training_command(games=10, wildcard="on")[-1] == "on"
    assert build_training_command(games=10, wildcard="off")[-1] == "off"
