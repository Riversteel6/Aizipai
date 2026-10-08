"""Candidate option detection for chi and compare/xiahu popups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from vision.regions import Region, crop_region, load_regions_for_size
from vision.option_anchor_detector import OptionAnchor, detect_option_anchors
from vision.image_source import read_bgr


@dataclass(frozen=True)
class OptionCandidate:
    region_name: str
    index: int
    x: int
    y: int
    w: int
    h: int
    card_count: int
    card_boxes: tuple[tuple[int, int, int, int], ...] = ()

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> dict[str, object]:
        return {
            "region_name": self.region_name,
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "card_count": self.card_count,
            "card_boxes": [list(box) for box in self.card_boxes],
            "center": self.center,
        }


def _target_regions(
    config_path: str | Path,
    width: int,
    height: int,
    names: Iterable[str],
) -> list[Region]:
    wanted = set(names)
    return [region for region in load_regions_for_size(config_path, width, height) if region.name in wanted]


def _excluded_regions(config_path: str | Path, width: int, height: int) -> list[Region]:
    return [
        region
        for region in load_regions_for_size(config_path, width, height)
        if region.name in {"buttons", "discard_button"}
    ]


def _inside_any_region(x: int, y: int, regions: list[Region]) -> bool:
    return any(region.x <= x <= region.x2 and region.y <= y <= region.y2 for region in regions)


def _overlap_ratio(
    box: tuple[int, int, int, int],
    region: Region,
) -> float:
    x1, y1, x2, y2 = box
    ix1 = max(x1, region.x)
    iy1 = max(y1, region.y)
    ix2 = min(x2, region.x2)
    iy2 = min(y2, region.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    box_area = max(1, (x2 - x1) * (y2 - y1))
    return ((ix2 - ix1) * (iy2 - iy1)) / box_area


def _white_card_boxes(crop: np.ndarray, offset_x: int, offset_y: int) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mask = cv2.inRange(gray, 170, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 20 <= w <= 180 and 40 <= h <= 210 and area >= 900:
            boxes.append((x + offset_x, y + offset_y, w, h))
    return boxes


def _merge_column_boxes(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    groups: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda item: item[0] + item[2] / 2):
        x, _, w, _ = box
        center_x = x + w / 2
        if box[1] > 430:
            continue
        if not groups:
            groups.append([box])
            continue

        last_group = groups[-1]
        gx1 = min(item[0] for item in last_group)
        gx2 = max(item[0] + item[2] for item in last_group)
        group_center = (gx1 + gx2) / 2
        if abs(center_x - group_center) <= 35:
            last_group.append(box)
        else:
            groups.append([box])
    return groups


def _candidate_cards(group: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    cards = [box for box in group if box[1] <= 430]
    cards.sort(key=lambda item: item[1])
    return cards[:3]


def _expanded_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Expand a tight white-card box to a more useful tappable option box."""

    left_pad = 6
    right_pad = 22
    top_pad = 8
    bottom_pad = 10
    x1 = max(0, x1 - left_pad)
    y1 = max(0, y1 - top_pad)
    x2 = min(image_width, x2 + right_pad)
    y2 = min(image_height, y2 + bottom_pad)
    return x1, y1, x2, y2


def detect_option_candidates(
    image: np.ndarray,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    region_names: Iterable[str] = ("compare_options", "chi_options"),
    anchor_mode: bool = True,
) -> list[OptionCandidate]:
    """Detect option columns and classify them from visible 比牌/吃牌 labels."""

    if anchor_mode:
        return _detect_anchor_classified_candidates(
            image,
            config_path=config_path,
            region_names=region_names,
        )
    return _detect_fixed_region_candidates(
        image,
        config_path=config_path,
        region_names=region_names,
    )


def _detect_anchor_classified_candidates(
    image: np.ndarray,
    *,
    config_path: str | Path,
    region_names: Iterable[str],
) -> list[OptionCandidate]:
    height, width = image.shape[:2]
    wanted = set(region_names)
    anchors = [anchor for anchor in detect_option_anchors(image) if anchor.region_name in wanted]
    if not anchors:
        return []

    scan_height = min(height, round(540 * height / 1080))
    boxes = _white_card_boxes(image[:scan_height, :], 0, 0)
    groups = _merge_column_boxes(boxes)
    excluded_regions = _excluded_regions(config_path, width, height)
    candidates: list[OptionCandidate] = []
    counts: dict[str, int] = {name: 0 for name in wanted}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    for group in groups:
        cards = _candidate_cards(group)
        if len(cards) < 2 or not _on_dark_option_panel(gray, cards):
            continue
        tight_x1 = min(item[0] for item in cards)
        tight_y1 = min(item[1] for item in cards)
        tight_x2 = max(item[0] + item[2] for item in cards)
        tight_y2 = max(item[1] + item[3] for item in cards)
        center_x = (tight_x1 + tight_x2) // 2
        anchor = _nearest_preceding_anchor(center_x, anchors)
        if anchor is None:
            continue
        x1, y1, x2, y2 = _expanded_box(
            tight_x1,
            tight_y1,
            tight_x2,
            tight_y2,
            image_width=width,
            image_height=height,
        )
        center_y = (y1 + y2) // 2
        if _inside_any_region(center_x, center_y, excluded_regions):
            continue
        if any(_overlap_ratio((x1, y1, x2, y2), region) >= 0.25 for region in excluded_regions):
            continue
        counts[anchor.region_name] += 1
        candidates.append(
            OptionCandidate(
                region_name=anchor.region_name,
                index=counts[anchor.region_name],
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                card_count=len(cards),
                card_boxes=tuple(cards),
            )
        )
    return sorted(candidates, key=lambda item: item.x)


def _nearest_preceding_anchor(center_x: int, anchors: list[OptionAnchor]) -> OptionAnchor | None:
    preceding = [anchor for anchor in anchors if anchor.x2 < center_x]
    return max(preceding, key=lambda anchor: anchor.x) if preceding else None


def _on_dark_option_panel(
    gray: np.ndarray,
    cards: list[tuple[int, int, int, int]],
) -> bool:
    x1 = min(item[0] for item in cards)
    y1 = min(item[1] for item in cards)
    x2 = max(item[0] + item[2] for item in cards)
    y2 = max(item[1] + item[3] for item in cards)
    left = gray[y1:y2, max(0, x1 - 12) : max(0, x1 - 3)]
    right = gray[y1:y2, min(gray.shape[1], x2 + 3) : min(gray.shape[1], x2 + 12)]
    ratios = [float((strip < 90).mean()) for strip in (left, right) if strip.size]
    return bool(ratios) and min(ratios) >= 0.45


def _detect_fixed_region_candidates(
    image: np.ndarray,
    *,
    config_path: str | Path,
    region_names: Iterable[str],
) -> list[OptionCandidate]:
    """Retain the pre-anchor detector as an explicit rollback path."""

    height, width = image.shape[:2]
    candidates: list[OptionCandidate] = []
    excluded_regions = _excluded_regions(config_path, width, height)
    for region in _target_regions(config_path, width, height, region_names):
        boxes = _white_card_boxes(crop_region(image, region), region.x, region.y)
        groups = _merge_column_boxes(boxes)
        for index, group in enumerate(groups, start=1):
            group = _candidate_cards(group)
            if len(group) < 2:
                continue
            x1 = min(item[0] for item in group)
            y1 = min(item[1] for item in group)
            x2 = max(item[0] + item[2] for item in group)
            y2 = max(item[1] + item[3] for item in group)
            x1, y1, x2, y2 = _expanded_box(
                x1,
                y1,
                x2,
                y2,
                image_width=width,
                image_height=height,
            )
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if _inside_any_region(center_x, center_y, excluded_regions):
                continue
            if any(_overlap_ratio((x1, y1, x2, y2), region) >= 0.25 for region in excluded_regions):
                continue
            candidates.append(
                OptionCandidate(
                    region_name=region.name,
                    index=index,
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    card_count=len(group),
                    card_boxes=tuple(group),
                )
            )
    return candidates


def detect_option_candidates_from_path(
    screenshot: str | Path,
    **kwargs,
) -> list[OptionCandidate]:
    image = read_bgr(screenshot)
    return detect_option_candidates(image, **kwargs)
