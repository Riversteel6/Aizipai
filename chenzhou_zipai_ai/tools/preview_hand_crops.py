"""Preview candidate own-hand card crops with index labels."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.regions import crop_region, load_regions_for_size


@dataclass(frozen=True)
class CropBox:
    index: int
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


def _read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _find_my_hand_region(config: str | Path, width: int, height: int):
    for region in load_regions_for_size(config, width, height):
        if region.name == "my_hand":
            return region
    raise ValueError(f"Region config has no 'my_hand' region: {config}")


def detect_hand_crop_boxes(image, config: str | Path) -> list[CropBox]:
    height, width = image.shape[:2]
    region = _find_my_hand_region(config, width, height)
    hand = crop_region(image, region)
    gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 170, 255)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 38 <= w <= 150 and 70 <= h <= 230 and area >= 2500:
            raw.append((x + region.x, y + region.y, w, h))

    # Sort by row then x, so stacked cards stay readable in the contact sheet.
    raw.sort(key=lambda box: (round(box[1] / 35), box[0]))
    return [CropBox(index=index, x=x, y=y, w=w, h=h) for index, (x, y, w, h) in enumerate(raw, 1)]


def make_contact_sheet(image, boxes: list[CropBox], output: str | Path) -> Path:
    tiles: list[Image.Image] = []
    for box in boxes:
        crop = image[box.y : box.y + box.h, box.x : box.x + box.w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tile_img = Image.fromarray(rgb)
        tile_img.thumbnail((96, 150))
        canvas = Image.new("RGB", (120, 190), "white")
        canvas.paste(tile_img, ((120 - tile_img.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 160), f"#{box.index}", fill=(0, 0, 0))
        draw.text((8, 176), f"{box.x},{box.y}", fill=(80, 80, 80))
        tiles.append(canvas)

    cols = 8
    rows = max(1, (len(tiles) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * 120, rows * 190), "white")
    for offset, tile in enumerate(tiles):
        sheet.paste(tile, ((offset % cols) * 120, (offset // cols) * 190))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def preview_hand_crops(
    screenshot: str | Path,
    *,
    config: str | Path = "config/screen_1080x2400.yaml",
    output: str | Path = "data/crops/previews/hand_crop_candidates.png",
) -> tuple[list[CropBox], Path]:
    image_path = Path(screenshot)
    image = _read_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read screenshot: {image_path}")
    boxes = detect_hand_crop_boxes(image, config)
    output_path = make_contact_sheet(image, boxes, output)
    return boxes, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview own-hand card crop candidates.")
    parser.add_argument("screenshot")
    parser.add_argument("--config", default="config/screen_1080x2400.yaml")
    parser.add_argument("--out", default="data/crops/previews/hand_crop_candidates.png")
    args = parser.parse_args()

    boxes, output = preview_hand_crops(args.screenshot, config=args.config, output=args.out)
    for box in boxes:
        print(f"#{box.index}: x={box.x} y={box.y} w={box.w} h={box.h} center={box.center}")
    print(output)


if __name__ == "__main__":
    main()
