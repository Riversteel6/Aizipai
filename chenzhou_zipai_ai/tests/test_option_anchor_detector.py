"""Regression coverage for text-anchored chi/compare option detection."""

from pathlib import Path

import cv2

from vision.option_anchor_detector import detect_option_anchors
from vision.option_detector import detect_option_candidates


OVERFLOW_SCREENSHOT = Path(
    "logs/device_soak_rc3_1v1_wang_hu_priority_20260812_151547/debug/overflow_stuck_20260812.png"
)


def _read_optional(path: Path):
    if not path.exists():
        return None
    image = cv2.imread(str(path))
    assert image is not None
    return image


def test_eight_column_layout_is_split_by_visible_text_anchors():
    image = _read_optional(OVERFLOW_SCREENSHOT)
    if image is None:
        return

    anchors = detect_option_anchors(image)
    candidates = detect_option_candidates(image)

    assert [anchor.region_name for anchor in anchors] == ["compare_options", "chi_options"]
    assert [candidate.region_name for candidate in candidates] == [
        "compare_options",
        "compare_options",
        "compare_options",
        "chi_options",
        "chi_options",
        "chi_options",
        "chi_options",
        "chi_options",
    ]
    assert [candidate.center[0] for candidate in candidates] == [1079, 1207, 1334, 1552, 1680, 1808, 1935, 2063]


def test_single_chi_anchor_keeps_all_visible_chi_columns():
    image = _read_optional(Path("data/screenshots/screenshot_20260527_160227.png"))
    if image is None:
        return

    anchors = detect_option_anchors(image)
    candidates = detect_option_candidates(image)

    assert [anchor.region_name for anchor in anchors] == ["chi_options"]
    assert [candidate.region_name for candidate in candidates] == [
        "chi_options",
        "chi_options",
        "chi_options",
    ]


def test_compare_and_chi_anchor_positions_can_move_between_screenshots():
    image = _read_optional(Path("data/screenshots/screenshot_20260527_170608.png"))
    if image is None:
        return

    anchors = detect_option_anchors(image)
    candidates = detect_option_candidates(image)

    assert [(anchor.region_name, anchor.x) for anchor in anchors] == [
        ("compare_options", 1376),
        ("chi_options", 1594),
    ]
    assert [candidate.region_name for candidate in candidates] == [
        "compare_options",
        "chi_options",
        "chi_options",
    ]


def test_white_meld_columns_without_text_anchor_are_not_options():
    image = _read_optional(Path("data/screenshots/screenshot_20260527_104127.png"))
    if image is None:
        return

    assert detect_option_anchors(image) == []
    assert detect_option_candidates(image) == []
    assert len(detect_option_candidates(image, anchor_mode=False)) == 5


def test_chi_anchor_survives_overlapping_red_prompt_text():
    image = _read_optional(Path("data/screenshots/screenshot_20260812_151807_724035_052700.png"))
    if image is None:
        return

    anchors = detect_option_anchors(image)
    candidates = detect_option_candidates(image)

    assert [(anchor.region_name, anchor.x) for anchor in anchors] == [("chi_options", 1593)]
    assert [candidate.region_name for candidate in candidates] == [
        "chi_options",
        "chi_options",
        "chi_options",
    ]
