"""Friendly double-click launcher for frozen v8.1 offline simulations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


def parse_game_count(value: str) -> int:
    text = value.strip()
    if not text.isdigit() or int(text) < 1:
        raise ValueError("局数必须是大于0的整数")
    return int(text)


def build_training_command(*, games: int, wildcard: str) -> list[str]:
    if wildcard not in {"on", "off"}:
        raise ValueError(f"unsupported wildcard mode: {wildcard}")
    return [
        sys.executable,
        "-u",
        "-m",
        "tools.run_v81_training",
        "--games",
        str(games),
        "--wildcard",
        wildcard,
    ]


def main(wildcard: str) -> int:
    mode_name = "有王" if wildcard == "on" else "无王"
    interactive = len(sys.argv) < 2
    print(f"v8.1 两人{mode_name}模式离线复盘测试")
    print("说明：每局都会保存完整决策轨迹；有王和无王必须一次只跑一个。")
    try:
        raw_games = sys.argv[1] if not interactive else input("请输入测试局数：")
        games = parse_game_count(raw_games)
        print(f"即将运行 {games} 局，运行期间会持续显示人话进度。")
        completed = subprocess.run(
            build_training_command(games=games, wildcard=wildcard),
            cwd=APP_ROOT,
            check=False,
        )
        returncode = completed.returncode
    except ValueError as exc:
        print(f"输入错误：{exc}")
        returncode = 2
    except KeyboardInterrupt:
        print("\n已停止。已经完成的逐局断点仍保留在 logs\\v81_training。")
        returncode = 130
    except Exception as exc:
        print(f"启动失败：{type(exc).__name__}: {exc}")
        returncode = 1

    if interactive:
        input("按回车关闭窗口……")
    return returncode
