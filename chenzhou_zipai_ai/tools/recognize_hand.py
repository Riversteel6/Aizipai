"""Recognize own-hand cards from a screenshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vision.hand_recognizer import recognize_hand_from_path

DEFAULT_TEMPLATE_DIRS = ("data/templates/hand_auto",)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    try:
        if candidate.exists():
            return candidate
    except OSError:
        pass
    for base in (ROOT, WORKSPACE):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return candidate


def _grid_lines(cards) -> list[str]:
    columns = []
    for card in sorted(cards, key=lambda item: item.x):
        for column in columns:
            if abs(column[0].x - card.x) <= 35:
                column.append(card)
                break
        else:
            columns.append([card])

    lines = [f"当前识别到：{len(cards)}张"]
    for column_index, column in enumerate(columns, start=1):
        lines.append(f"第{column_index}排：")
        for layer_index, card in enumerate(sorted(column, key=lambda item: item.y), start=1):
            lines.append(f"  第{layer_index}层：{card.name}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize own-hand cards from a screenshot.")
    parser.add_argument("screenshot")
    parser.add_argument("--config", default="config/screen_1080x2400.yaml")
    parser.add_argument("--template-dir", action="append", default=None)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--search-padding-left", type=int, default=180)
    parser.add_argument("--search-padding-top", type=int, default=180)
    parser.add_argument("--search-padding-right", type=int, default=260)
    parser.add_argument("--unknown-crop-dir", default="data/crops/unknown")
    parser.add_argument("--debug-out", default="data/crops/debug_hand.png")
    parser.add_argument("--details", action="store_true", help="Print coordinates and scores.")
    args = parser.parse_args()

    cards = recognize_hand_from_path(
        _resolve_project_path(args.screenshot),
        config_path=_resolve_project_path(args.config),
        template_dir=tuple(args.template_dir) if args.template_dir else DEFAULT_TEMPLATE_DIRS,
        threshold=args.threshold,
        search_padding_left=args.search_padding_left,
        search_padding_top=args.search_padding_top,
        search_padding_right=args.search_padding_right,
        unknown_crop_dir=args.unknown_crop_dir,
        debug_path=args.debug_out,
    )
    print("\n".join(_grid_lines(cards)))
    if args.details:
        for index, card in enumerate(cards, 1):
            print(
                f"#{index} {card.name} score={card.confidence:.3f} "
                f"x={card.x} y={card.y} w={card.w} h={card.h} template={card.template}"
            )


if __name__ == "__main__":
    main()
