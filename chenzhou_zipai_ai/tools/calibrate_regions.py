"""Calibrate screen regions for a target device."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.regions import Region, load_regions, load_regions_for_size


COLORS = {
    "my_hand": (0, 255, 0),
    "opponent_discards": (255, 128, 0),
    "my_discards": (60, 220, 255),
    "pending_action_card": (255, 255, 255),
    "remaining_deck_count": (70, 180, 255),
    "opponent_pending_card": (180, 180, 255),
    "my_melds": (255, 0, 255),
    "my_dealer_marker": (40, 210, 160),
    "opponent_melds": (255, 255, 0),
    "buttons": (0, 0, 255),
    "discard_button": (80, 80, 255),
    "chi_options": (30, 30, 30),
    "compare_options": (120, 120, 120),
}

LABELS = {
    "my_hand": "我的手牌区",
    "opponent_discards": "对方历史弃牌区",
    "my_discards": "我的历史弃牌区",
    "pending_action_card": "我方本轮摸牌区",
    "remaining_deck_count": "剩余牌数区",
    "opponent_pending_card": "对手打出的响应牌区",
    "my_melds": "我吃碰的牌区",
    "my_dealer_marker": "我的庄家标记区",
    "opponent_melds": "对家吃碰的牌区",
    "buttons": "吃碰胡过按钮区",
    "discard_button": "出牌按钮区",
    "chi_options": "吃牌候选框区",
    "compare_options": "比牌候选框区",
}


def load_chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_region(image, region: Region) -> None:
    color = COLORS.get(region.name, (255, 255, 255))
    cv2.rectangle(image, (region.x, region.y), (region.x2, region.y2), color, 3)


def draw_labels(image, regions: list[Region]) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    font = load_chinese_font(28)

    for region in regions:
        color = COLORS.get(region.name, (255, 255, 255))
        text = LABELS.get(region.name, region.name)
        label_y = max(region.y - 34, 8)
        fill = (color[2], color[1], color[0])
        draw.text((region.x, label_y), text, font=font, fill=fill)

    image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def calibrate_regions(
    screenshot: str | Path,
    config: str | Path,
    output: str | Path,
    *,
    scale_to_image: bool = True,
) -> Path:
    image = cv2.imread(str(screenshot))
    if image is None:
        raise FileNotFoundError(f"Could not read screenshot: {screenshot}")

    height, width = image.shape[:2]
    regions = load_regions_for_size(config, width, height) if scale_to_image else load_regions(config)
    for region in regions:
        draw_region(image, region)
    draw_labels(image, regions)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Failed to write debug image: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw configured regions on a screenshot.")
    parser.add_argument("screenshot", help="Screenshot image path.")
    parser.add_argument("--config", default="config/screen_1080x2400.yaml")
    parser.add_argument("--out", default="data/screenshots/debug_regions.png")
    parser.add_argument("--no-scale", action="store_true", help="Use raw config pixels.")
    args = parser.parse_args()

    path = calibrate_regions(args.screenshot, args.config, args.out, scale_to_image=not args.no_scale)
    print(path)


if __name__ == "__main__":
    main()
