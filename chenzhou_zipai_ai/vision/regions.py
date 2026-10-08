"""Screen region definitions and calibration loading."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from vision.viewport import transform_from_config


@dataclass(frozen=True)
class Region:
    name: str
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def to_dict(self) -> dict[str, int | str]:
        return {"name": self.name, "x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def scaled(self, scale_x: float, scale_y: float) -> "Region":
        return Region(
            name=self.name,
            x=round(self.x * scale_x),
            y=round(self.y * scale_y),
            w=round(self.w * scale_x),
            h=round(self.h * scale_y),
        )


def load_region_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Region config not found: {config_path}")
    return deepcopy(_load_region_config_cached(str(config_path), config_path.stat().st_mtime_ns))


@lru_cache(maxsize=16)
def _load_region_config_cached(path: str, modified_ns: int) -> dict:
    del modified_ns
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if "regions" not in data:
        raise ValueError(f"Region config has no 'regions' section: {config_path}")
    return data


def config_screen_size(config: dict[str, Any]) -> tuple[int, int]:
    screen = config.get("screen", {})
    width = int(screen.get("width", 0))
    height = int(screen.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Region config must include positive screen.width and screen.height.")
    return width, height


def parse_regions(config: dict[str, Any]) -> list[Region]:
    regions: list[Region] = []
    for name, item in config["regions"].items():
        missing = {"x", "y", "w", "h"} - set(item)
        if missing:
            raise ValueError(f"Region {name!r} missing fields: {sorted(missing)}")
        regions.append(
            Region(
                name=name,
                x=int(item["x"]),
                y=int(item["y"]),
                w=int(item["w"]),
                h=int(item["h"]),
            )
        )
    return regions


def load_regions(path: str | Path) -> list[Region]:
    return parse_regions(load_region_config(path))


def load_regions_for_size(path: str | Path, target_width: int, target_height: int) -> list[Region]:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Region config not found: {config_path}")
    return list(
        _load_regions_for_size_cached(
            str(config_path),
            config_path.stat().st_mtime_ns,
            int(target_width),
            int(target_height),
        )
    )


@lru_cache(maxsize=64)
def _load_regions_for_size_cached(
    path: str,
    modified_ns: int,
    target_width: int,
    target_height: int,
) -> tuple[Region, ...]:
    config = _load_region_config_cached(path, modified_ns)
    config_screen_size(config)
    transform = transform_from_config(config, target_width, target_height)
    return tuple(
        Region(region.name, *transform.map_box(region.x, region.y, region.w, region.h))
        for region in parse_regions(config)
    )


def crop_region(image, region: Region):
    return image[region.y : region.y2, region.x : region.x2]
