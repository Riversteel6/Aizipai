"""Preview chi and compare/xiahu candidate option click points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.option_detector import detect_option_candidates
from vision.regions import load_regions_for_size


COLORS = {
    "compare_options": (255, 0, 255),
    "chi_options": (0, 255, 255),
}


def preview_options(
    screenshot: str | Path,
    config: str | Path,
    output: str | Path,
    *,
    max_preview_width: int = 1000,
) -> Path:
    image = cv2.imread(str(screenshot))
    if image is None:
        raise FileNotFoundError(f"Could not read screenshot: {screenshot}")

    candidates = detect_option_candidates(image, config_path=config)
    height, width = image.shape[:2]
    for region in load_regions_for_size(config, width, height):
        if region.name not in COLORS:
            continue
        color = COLORS[region.name]
        cv2.rectangle(image, (region.x, region.y), (region.x2, region.y2), color, 3)
        cv2.putText(
            image,
            region.name,
            (region.x, max(24, region.y + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    for candidate in candidates:
        color = COLORS.get(candidate.region_name, (255, 255, 255))
        cv2.rectangle(
            image,
            (candidate.x, candidate.y),
            (candidate.x + candidate.w, candidate.y + candidate.h),
            color,
            4,
        )
        cx, cy = candidate.center
        cv2.circle(image, (cx, cy), 14, color, -1)
        cv2.putText(
            image,
            f"{candidate.region_name}:{candidate.index}",
            (candidate.x, max(24, candidate.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        print(candidate.to_dict())

    scale = min(1.0, max_preview_width / image.shape[1])
    if scale < 1.0:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write preview: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview chi and compare/xiahu option points.")
    parser.add_argument("screenshot")
    parser.add_argument("--config", default="config/screen_1080x2400.yaml")
    parser.add_argument("--out", default="data/screenshots/debug_options_preview.png")
    args = parser.parse_args()

    print(preview_options(args.screenshot, args.config, args.out))


if __name__ == "__main__":
    main()
