"""Build a mock-table layout JSON from the current screen calibration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from chenzhou_zipai_ai.tools.mock_build_common import (
    ROOT,
    ensure_mock_build_logs,
    rel,
    write_json,
    write_markdown,
)

REGION_ALIASES = {
    "my_hand": ["my_hand"],
    "buttons": ["buttons"],
    "chi_options": ["chi_options"],
    "center_discards": ["my_discards", "opponent_discards"],
    "my_melds": ["my_melds"],
    "left_melds": ["my_melds"],
    "right_melds": ["opponent_melds"],
    "ready_button": ["buttons"],
    "settlement_button": ["buttons"],
}


def load_layout(config_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    screen = data.get("screen") or {}
    source_regions = data.get("regions") or {}
    regions: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for target, aliases in REGION_ALIASES.items():
        for alias in aliases:
            if alias in source_regions:
                regions[target] = dict(source_regions[alias])
                provenance[target] = {"source": alias, "confidence": "confirmed"}
                break
        else:
            regions[target] = {"x": 0, "y": 0, "w": 0, "h": 0}
            provenance[target] = {"source": None, "confidence": "unknown"}

    card = infer_card_metrics(regions.get("my_hand", {}))
    return {
        "screen": {
            "width": int(screen.get("width") or 0),
            "height": int(screen.get("height") or 0),
        },
        "regions": regions,
        "card": card,
        "provenance": {
            "config": rel(config_path),
            "regions": provenance,
            "notes": [
                "Generated from current calibration YAML.",
                "ready_button and settlement_button reuse buttons region until a dedicated region is confirmed.",
            ],
        },
    }


def infer_card_metrics(my_hand: dict[str, Any]) -> dict[str, Any]:
    width = int(my_hand.get("w") or 0)
    height = int(my_hand.get("h") or 0)
    if not width or not height:
        return {"w": 0, "h": 0, "overlap": 0, "confidence": "unknown"}
    card_h = min(360, max(1, int(height * 0.86)))
    card_w = min(145, max(1, int(card_h * 0.42)))
    overlap = max(1, int(card_w * 0.58))
    return {"w": card_w, "h": card_h, "overlap": overlap, "confidence": "estimated_from_my_hand"}


def choose_sample_screenshot(screenshot_root: Path) -> Path | None:
    if not screenshot_root.exists():
        return None
    newest: tuple[float, Path] | None = None
    for entry in os.scandir(screenshot_root):
        if not entry.is_file():
            continue
        path = Path(entry.path)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        modified = entry.stat().st_mtime
        if newest is None or modified > newest[0]:
            newest = (modified, path)
    return newest[1] if newest else None


def draw_preview(layout: dict[str, Any], screenshot: Path | None, output: Path) -> dict[str, Any]:
    screen = layout["screen"]
    width = int(screen.get("width") or 1280)
    height = int(screen.get("height") or 720)
    image = None
    source = None
    if screenshot:
        image = cv2.imread(str(screenshot))
        if image is not None:
            source = rel(screenshot)
    if image is None:
        image = np.full((height, width, 3), 255, dtype=np.uint8)
    scale = min(800 / max(image.shape[1], 1), 800 / max(image.shape[0], 1), 1.0)
    preview = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

    colors = {
        "my_hand": (40, 180, 40),
        "buttons": (40, 80, 230),
        "chi_options": (230, 120, 20),
        "center_discards": (180, 40, 180),
        "my_melds": (0, 160, 180),
        "left_melds": (120, 120, 40),
        "right_melds": (180, 90, 90),
        "ready_button": (40, 80, 230),
        "settlement_button": (40, 80, 230),
    }
    for name, region in layout["regions"].items():
        x = int(region.get("x") or 0)
        y = int(region.get("y") or 0)
        w = int(region.get("w") or 0)
        h = int(region.get("h") or 0)
        if not w or not h:
            continue
        pt1 = (int(x * scale), int(y * scale))
        pt2 = (int((x + w) * scale), int((y + h) * scale))
        color = colors.get(name, (0, 0, 0))
        cv2.rectangle(preview, pt1, pt2, color, 2)
        cv2.putText(preview, name, (pt1[0] + 4, max(16, pt1[1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), preview)
    return {"preview": rel(output), "source_screenshot": source, "scale": scale}


def markdown_report(layout: dict[str, Any], preview: dict[str, Any]) -> list[str]:
    lines = [
        "# Mock Layout Report",
        "",
        f"- Config source: `{layout['provenance']['config']}`",
        f"- Screen: `{layout['screen']['width']}x{layout['screen']['height']}`",
        f"- Preview: `{preview['preview']}`",
        f"- Preview source screenshot: `{preview.get('source_screenshot') or 'none'}`",
        "",
        "## Regions",
        "",
    ]
    for name, region in layout["regions"].items():
        provenance = layout["provenance"]["regions"][name]
        lines.append(
            f"- `{name}`: x={region['x']} y={region['y']} w={region['w']} h={region['h']} "
            f"source=`{provenance['source']}` confidence=`{provenance['confidence']}`"
        )
    lines.extend(["", "## Card Metrics", ""])
    card = layout["card"]
    lines.append(f"- w={card['w']} h={card['h']} overlap={card['overlap']} confidence=`{card['confidence']}`")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "screen_1080x2400.yaml")
    parser.add_argument("--screenshots", type=Path, default=ROOT / "data" / "screenshots")
    parser.add_argument("--sample-screenshot", type=Path)
    args = parser.parse_args()

    layout = load_layout(args.config)
    out_dir = ensure_mock_build_logs()
    screenshot = args.sample_screenshot or choose_sample_screenshot(args.screenshots)
    preview = draw_preview(layout, screenshot, out_dir / "mock_layout_preview.png")
    write_json(out_dir / "mock_layout.generated.json", layout)
    write_markdown(out_dir / "mock_layout_report.md", markdown_report(layout, preview))
    print(f"MOCK_LAYOUT_JSON={out_dir / 'mock_layout.generated.json'}")
    print(f"MOCK_LAYOUT_PREVIEW={out_dir / 'mock_layout_preview.png'}")
    print(f"MOCK_LAYOUT_REPORT={out_dir / 'mock_layout_report.md'}")


if __name__ == "__main__":
    main()
