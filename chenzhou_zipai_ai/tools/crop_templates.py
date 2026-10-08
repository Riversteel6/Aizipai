"""Crop card and button templates from screenshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image) -> bool:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
    return True


def _next_template_path(template_dir: Path, card_name: str, appearance: str | None = None) -> Path:
    template_dir.mkdir(parents=True, exist_ok=True)
    stem = card_name if not appearance or appearance == "normal" else f"{card_name}_{appearance}"
    index = 1
    while True:
        path = template_dir / f"{stem}_{index:03d}.png"
        if not path.exists():
            return path
        index += 1


def _validate_box(image, x: int, y: int, w: int, h: int) -> None:
    height, width = image.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("Crop width and height must be positive.")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(
            f"Crop box x={x}, y={y}, w={w}, h={h} is outside image size {width}x{height}."
        )


def crop_template(
    screenshot: str | Path,
    *,
    card_name: str,
    x: int,
    y: int,
    w: int,
    h: int,
    template_group: str = "hand",
    appearance: str | None = None,
    output_root: str | Path = "data/templates",
    preview_root: str | Path = "data/crops/previews",
) -> tuple[Path, Path]:
    image_path = Path(screenshot)
    image = _read_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read screenshot: {image_path}")

    _validate_box(image, x, y, w, h)
    crop = image[y : y + h, x : x + w]

    template_dir = _project_path(output_root) / template_group
    output_path = _next_template_path(template_dir, card_name, appearance)
    if not _write_image(output_path, crop):
        raise RuntimeError(f"Failed to write template: {output_path}")

    preview_dir = _project_path(preview_root) / template_group
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{output_path.stem}_preview.png"
    preview = image.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 3)
    cv2.putText(
        preview,
        output_path.stem,
        (x, max(24, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    scale = min(1.0, 1000 / preview.shape[1])
    if scale < 1.0:
        preview = cv2.resize(preview, (round(preview.shape[1] * scale), round(preview.shape[0] * scale)))
    if not _write_image(preview_path, preview):
        raise RuntimeError(f"Failed to write preview: {preview_path}")

    return output_path, preview_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop a card or button template from a screenshot.")
    parser.add_argument("screenshot", help="Source screenshot path.")
    parser.add_argument("--card-name", required=True, help="Template label, for example 二, 贰, 王, chi.")
    parser.add_argument("--template-group", default="hand", help="Template group under data/templates.")
    parser.add_argument(
        "--appearance",
        choices=("normal", "selected", "triple_stack"),
        default=None,
        help="Optional hand-card appearance state; normal keeps the historical 牌名_001.png naming.",
    )
    parser.add_argument("--x", required=True, type=int)
    parser.add_argument("--y", required=True, type=int)
    parser.add_argument("--w", required=True, type=int)
    parser.add_argument("--h", required=True, type=int)
    parser.add_argument("--output-root", default="data/templates")
    parser.add_argument("--preview-root", default="data/crops/previews")
    args = parser.parse_args()

    output_path, preview_path = crop_template(
        args.screenshot,
        card_name=args.card_name,
        x=args.x,
        y=args.y,
        w=args.w,
        h=args.h,
        template_group=args.template_group,
        appearance=args.appearance,
        output_root=args.output_root,
        preview_root=args.preview_root,
    )
    print(f"template={output_path}")
    print(f"preview={preview_path}")


if __name__ == "__main__":
    main()
