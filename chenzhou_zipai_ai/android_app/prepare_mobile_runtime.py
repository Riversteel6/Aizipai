"""Build the minimal Python source tree embedded in the Android app."""

from __future__ import annotations

import shutil
from pathlib import Path


ANDROID_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ANDROID_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
TARGET = ANDROID_ROOT / "app" / "src" / "main" / "python"
MOBILE_ENTRYPOINTS = ("mobile_bridge.py", "mobile_worker.py")


def _reset_target() -> None:
    resolved = TARGET.resolve()
    expected_parent = (ANDROID_ROOT / "app" / "src" / "main").resolve()
    if resolved.parent != expected_parent or resolved.name != "python":
        raise RuntimeError(f"unsafe_mobile_runtime_target:{resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _copy_python_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def main() -> None:
    _reset_target()
    package = TARGET / "chenzhou_zipai_ai"
    package.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "__init__.py", package / "__init__.py")

    for name in ("ai", "audit", "engine", "vision", "control", "capture", "game_logging"):
        _copy_python_tree(PROJECT_ROOT / name, package / name)
    _copy_tree(PROJECT_ROOT / "vision" / "assets", package / "vision" / "assets")

    tools = package / "tools"
    tools.mkdir(parents=True)
    (tools / "__init__.py").write_text("", encoding="ascii")
    for name in (
        "compact_hand_left.py",
        "inspect_state.py",
        "live_assistant.py",
        "confirmed_hand_ledger.py",
        "mobile_action_verifier.py",
        "mobile_freshness.py",
        "mobile_priority.py",
        "protocol_log_to_state.py",
        "qs_protocol_parser.py",
        "recommend_action.py",
    ):
        shutil.copy2(PROJECT_ROOT / "tools" / name, tools / name)

    config = package / "config"
    config.mkdir()
    for name in ("rules.yaml", "screen_1080x2400.yaml", "thresholds.yaml"):
        shutil.copy2(PROJECT_ROOT / "config" / name, config / name)
    _copy_tree(config, TARGET / "config")

    models = package / "models"
    models.mkdir()
    for name in ("card_classifier.onnx", "card_classifier.json", "opponent_belief.json"):
        shutil.copy2(PROJECT_ROOT / "models" / name, models / name)
    _copy_tree(PROJECT_ROOT / "data" / "templates", package / "data" / "templates")

    _copy_python_tree(WORKSPACE_ROOT / "src" / "aizipai", TARGET / "aizipai")
    for name in MOBILE_ENTRYPOINTS:
        shutil.copy2(ANDROID_ROOT / "mobile_runtime" / name, TARGET / name)


if __name__ == "__main__":
    main()
