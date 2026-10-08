from __future__ import annotations

from pathlib import Path

from vision.regions import load_regions_for_size
from vision.viewport import ViewportTransform


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_width_identity() -> None:
    transform = ViewportTransform(2344, 1080, 2344, 1080)

    assert transform.map_point(430, 660) == (430, 660)
    assert transform.map_box(430, 660, 1680, 420) == (430, 660, 1680, 420)


def test_fixed_width_centres_vertical_calibration_on_16_by_9() -> None:
    transform = ViewportTransform(2344, 1080, 1920, 1080)

    assert transform.scale_x == transform.scale_y
    assert transform.offset_y > 0
    assert transform.map_y(540) == 540
    assert transform.map_point(1172, 540) == (960, 540)


def test_fixed_width_round_trip() -> None:
    transform = ViewportTransform(2344, 1080, 2560, 1440)
    mapped = transform.map_point(1731, 713)
    original = transform.unmap_point(*mapped)

    assert abs(original[0] - 1731) < 0.6
    assert abs(original[1] - 713) < 0.6


def test_boxes_are_clipped_on_wider_screen() -> None:
    transform = ViewportTransform(2344, 1080, 2400, 1080)
    x, y, width, height = transform.map_box(0, 0, 2344, 1080)

    assert (x, width) == (0, 2400)
    assert y == 0
    assert height == 1080


def test_all_configured_regions_stay_in_target_bounds() -> None:
    config = PROJECT_ROOT / "config" / "screen_1080x2400.yaml"
    for target_width, target_height in ((1920, 1080), (2160, 1080), (2400, 1080), (2560, 1440)):
        for region in load_regions_for_size(config, target_width, target_height):
            assert 0 <= region.x <= target_width
            assert 0 <= region.y <= target_height
            assert 0 <= region.x2 <= target_width
            assert 0 <= region.y2 <= target_height


def test_response_button_region_covers_leftmost_hu_in_four_button_layout() -> None:
    config = PROJECT_ROOT / "config" / "screen_1080x2400.yaml"
    regions = {
        region.name: region
        for region in load_regions_for_size(config, 2344, 1080)
    }

    buttons = regions["buttons"]
    assert (buttons.x, buttons.x2) == (1400, 2340)


def test_runtime_overlay_area_does_not_overlap_any_recognition_region() -> None:
    config = PROJECT_ROOT / "config" / "screen_1080x2400.yaml"
    overlay = (331, 3, 681, 89)

    for region in load_regions_for_size(config, 2344, 1080):
        overlaps = not (
            overlay[2] <= region.x
            or region.x2 <= overlay[0]
            or overlay[3] <= region.y
            or region.y2 <= overlay[1]
        )
        assert not overlaps, f"overlay overlaps recognition region {region.name}"
