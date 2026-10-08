"""Recognize left-side meld/history cards, including red-backed hidden cards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from vision.hand_recognizer import classify_hand_slot
from vision.regions import Region, crop_region, load_regions_for_size
from vision.template_loader import TemplateImage, load_templates_from_dirs


@dataclass(frozen=True)
class MeldCell:
    region_name: str
    name: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    template: str
    hidden: bool = False

    def to_dict(self) -> dict[str, bool | float | int | str]:
        return {
            "region_name": self.region_name,
            "name": self.name,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "template": self.template,
            "hidden": self.hidden,
        }


def _target_regions(config_path: str | Path, width: int, height: int) -> list[Region]:
    wanted = {"opponent_melds", "my_melds"}
    return [region for region in load_regions_for_size(config_path, width, height) if region.name in wanted]


def _my_hand_region(config_path: str | Path, width: int, height: int) -> Region | None:
    return next(
        (region for region in load_regions_for_size(config_path, width, height) if region.name == "my_hand"),
        None,
    )


def _load_templates(template_dirs: tuple[str | Path, ...]) -> list[TemplateImage]:
    return load_templates_from_dirs(template_dirs, grayscale=False)


def _red_mask(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return (((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60))


def _component_cell_groups(image, region: Region) -> list[list[tuple[int, int, int, int]]]:
    crop = crop_region(image, region)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(gray, 140, 255)
    red = _red_mask(crop).astype("uint8") * 255
    mask = cv2.bitwise_or(white, red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 45 <= w <= 240 and 45 <= h <= 320 and area >= 1500:
            components.append((x + region.x, y + region.y, w, h, area))

    components.sort(key=lambda item: item[4], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for x, y, w, h, _area in components:
        if any(kx <= x and ky <= y and kx + kw >= x + w and ky + kh >= y + h for kx, ky, kw, kh in kept):
            continue
        kept.append((x, y, w, h))

    groups: list[list[tuple[int, int, int, int]]] = []
    for x, y, w, h in kept:
        cols = max(1, round(w / 70))
        rows = max(1, round(h / 70))
        cell_w = w / cols
        cell_h = h / rows
        for col in range(cols):
            group: list[tuple[int, int, int, int]] = []
            for row in range(rows):
                x1 = round(x + col * cell_w)
                y1 = round(y + row * cell_h)
                x2 = round(x + (col + 1) * cell_w)
                y2 = round(y + (row + 1) * cell_h)
                group.append((x1, y1, x2 - x1, y2 - y1))
            groups.append(group)
    groups.sort(key=lambda group: (group[0][0], group[0][1]))
    return groups


def _merge_vertical_groups(groups: list[list[tuple[int, int, int, int]]]) -> list[list[tuple[int, int, int, int]]]:
    merged: list[list[tuple[int, int, int, int]]] = []
    for group in sorted(groups, key=lambda item: (item[0][0], item[0][1])):
        center_x = sum(box[0] + box[2] / 2 for box in group) / len(group)
        target = None
        for existing in merged:
            existing_center_x = sum(box[0] + box[2] / 2 for box in existing) / len(existing)
            if abs(center_x - existing_center_x) <= 16:
                target = existing
                break
        if target is None:
            merged.append(list(group))
        else:
            target.extend(group)
            target.sort(key=lambda box: box[1])
    return merged


def _recognize_cell(image, region_name: str, box: tuple[int, int, int, int], templates, threshold: float) -> MeldCell | None:
    x, y, w, h = box
    cell_image = image[y : y + h, x : x + w]
    if float(_red_mask(cell_image).mean()) > 0.35:
        return MeldCell(region_name, "暗", 1.0, x, y, w, h, "red_back", hidden=True)
    name, confidence, template = classify_hand_slot(cell_image, templates)
    if confidence < threshold:
        return None
    return MeldCell(region_name, name, confidence, x, y, w, h, template)


def recognize_melds(
    image,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    template_dirs: tuple[str | Path, ...] = (
        "data/templates/meld_auto",
        "data/templates/discard_auto",
        "data/templates/hand_auto",
    ),
    threshold: float = 0.42,
) -> dict[str, list[MeldCell]]:
    grouped = recognize_meld_groups(
        image,
        config_path=config_path,
        template_dirs=template_dirs,
        threshold=threshold,
    )
    return {region_name: [cell for group in groups for cell in group] for region_name, groups in grouped.items()}


def recognize_meld_groups(
    image,
    *,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    template_dirs: tuple[str | Path, ...] = (
        "data/templates/meld_auto",
        "data/templates/discard_auto",
        "data/templates/hand_auto",
    ),
    threshold: float = 0.42,
) -> dict[str, list[list[MeldCell]]]:
    height, width = image.shape[:2]
    templates = _load_templates(template_dirs)
    hand_region = _my_hand_region(config_path, width, height)
    result: dict[str, list[list[MeldCell]]] = {}
    for region in _target_regions(config_path, width, height):
        groups: list[list[MeldCell]] = []
        for boxes in _merge_vertical_groups(_component_cell_groups(image, region)):
            cells = [
                cell
                for box in boxes
                if not _inside_region_center(box, hand_region)
                if (cell := _recognize_cell(image, region.name, box, templates, threshold)) is not None
            ]
            if len(cells) >= 3:
                groups.append(cells)
        result[region.name] = groups
    return result


def _inside_region_center(box: tuple[int, int, int, int], region: Region | None) -> bool:
    if region is None:
        return False
    x, y, w, h = box
    center_x = x + w // 2
    center_y = y + h // 2
    if not (region.x <= center_x <= region.x2 and region.y <= center_y <= region.y2):
        return False
    # Left-side meld columns can extend slightly into the configured hand region.
    # Only suppress boxes that are low enough to be real bottom hand cards.
    return center_y >= region.y + 180
