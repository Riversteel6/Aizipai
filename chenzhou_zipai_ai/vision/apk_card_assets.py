"""Helpers for card glyphs extracted from the game APK."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent

DEFAULT_MANIFEST = WORKSPACE / "data/apk_extract/base_apk_selected/zipai_card_glyph_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/templates/apk_card_glyphs"


def sync_apk_card_glyph_templates(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Copy APK glyph PNGs into normal template names like 三_apk_001.png."""

    manifest_path = _resolve_workspace_path(manifest_path)
    output_dir = _resolve_workspace_path(output_dir)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for row in rows:
        if not row.get("exists"):
            continue
        label = str(row.get("label") or "").strip()
        source = _resolve_workspace_path(row.get("file") or "")
        if not label or not source.exists():
            continue
        target = output_dir / f"{label}_001.png"
        shutil.copyfile(source, target)
        written.append(target)
    return written


def _resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    try:
        if candidate.exists():
            return candidate
    except OSError:
        pass
    return WORKSPACE / candidate
