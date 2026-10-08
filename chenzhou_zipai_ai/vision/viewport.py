"""Coordinate transforms between the calibrated and current game viewport."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewportTransform:
    """Map calibrated pixels using the game's Cocos FIXED_WIDTH policy.

    Cocos scales both axes from the screen width and vertically centres the
    design viewport. The calibration may therefore be cropped on wider screens
    or surrounded by additional vertical content on narrower screens.
    """

    base_width: int
    base_height: int
    target_width: int
    target_height: int
    mode: str = "fixed_width"

    def __post_init__(self) -> None:
        for name in ("base_width", "base_height", "target_width", "target_height"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mode not in {"fixed_width", "stretch"}:
            raise ValueError(f"Unsupported viewport scale mode: {self.mode}")

    @property
    def scale_x(self) -> float:
        return self.target_width / self.base_width

    @property
    def scale_y(self) -> float:
        if self.mode == "stretch":
            return self.target_height / self.base_height
        return self.scale_x

    @property
    def offset_y(self) -> float:
        if self.mode == "stretch":
            return 0.0
        return (self.target_height - self.base_height * self.scale_y) / 2.0

    def map_x(self, value: float) -> int:
        return round(value * self.scale_x)

    def map_y(self, value: float) -> int:
        return round(value * self.scale_y + self.offset_y)

    def map_point(self, x: float, y: float, *, clip: bool = False) -> tuple[int, int]:
        mapped_x = self.map_x(x)
        mapped_y = self.map_y(y)
        if clip:
            mapped_x = min(max(mapped_x, 0), self.target_width - 1)
            mapped_y = min(max(mapped_y, 0), self.target_height - 1)
        return mapped_x, mapped_y

    def map_box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        clip: bool = True,
    ) -> tuple[int, int, int, int]:
        left = self.map_x(x)
        top = self.map_y(y)
        right = self.map_x(x + width)
        bottom = self.map_y(y + height)
        if clip:
            left = min(max(left, 0), self.target_width)
            right = min(max(right, 0), self.target_width)
            top = min(max(top, 0), self.target_height)
            bottom = min(max(bottom, 0), self.target_height)
        return left, top, max(0, right - left), max(0, bottom - top)

    def unmap_point(self, x: float, y: float) -> tuple[float, float]:
        return x / self.scale_x, (y - self.offset_y) / self.scale_y


def transform_from_config(
    config: dict,
    target_width: int,
    target_height: int,
) -> ViewportTransform:
    screen = config.get("screen", {})
    return ViewportTransform(
        base_width=int(screen.get("width", 0)),
        base_height=int(screen.get("height", 0)),
        target_width=int(target_width),
        target_height=int(target_height),
        mode=str(screen.get("scale_mode", "stretch")),
    )


__all__ = ["ViewportTransform", "transform_from_config"]
