from __future__ import annotations

from pathlib import Path

from tools import compact_hand_left, inspect_state, recognize_hand


def test_runtime_paths_fall_back_when_android_cwd_denies_stat(monkeypatch) -> None:
    original_exists = Path.exists

    def android_exists(path: Path) -> bool:
        if not path.is_absolute():
            raise PermissionError("Android working directory is not readable")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", android_exists)

    screen = "config/screen_1080x2400.yaml"
    for module in (compact_hand_left, inspect_state, recognize_hand):
        resolved = module._resolve_project_path(screen)
        assert resolved.is_absolute()
        assert original_exists(resolved)
