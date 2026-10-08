from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vision.option_detector import detect_option_candidates


def _draw_column(image: np.ndarray, x: int, y: int, card_count: int) -> None:
    for index in range(card_count):
        top = y + index * 70
        cv2.rectangle(image, (x, top), (x + 56, top + 58), (245, 245, 245), -1)


def test_fixed_region_rollback_detects_compare_and_chi_option_columns():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    config_path = root / "config" / "screen_1080x2400.yaml"
    image = np.zeros((1080, 2344, 3), dtype=np.uint8)

    _draw_column(image, 1428, 30, 3)
    _draw_column(image, 1640, 30, 3)
    _draw_column(image, 1768, 30, 3)
    _draw_column(image, 1896, 30, 3)

    candidates = detect_option_candidates(image, config_path=config_path, anchor_mode=False)
    by_region = {}
    for candidate in candidates:
        by_region.setdefault(candidate.region_name, []).append(candidate)

    assert len(by_region["compare_options"]) == 1
    assert len(by_region["chi_options"]) == 3
    assert [candidate.index for candidate in by_region["chi_options"]] == [1, 2, 3]
