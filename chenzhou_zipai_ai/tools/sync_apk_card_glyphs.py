"""Sync card glyph templates extracted from the APK into vision templates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vision.apk_card_assets import DEFAULT_MANIFEST, DEFAULT_OUTPUT_DIR, sync_apk_card_glyph_templates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    written = sync_apk_card_glyph_templates(args.manifest, args.output_dir)
    print(f"APK_CARD_GLYPH_TEMPLATES={len(written)}")
    print(f"APK_CARD_GLYPH_DIR={Path(args.output_dir)}")


if __name__ == "__main__":
    main()
