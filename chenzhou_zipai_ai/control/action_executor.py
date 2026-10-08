"""Convert engine actions to tap plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import yaml

from control.adb_tap import drag


@dataclass(frozen=True)
class DragPlan:
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    duration_ms: int

    def to_dict(self) -> dict[str, int]:
        return {
            "start_x": self.start_x,
            "start_y": self.start_y,
            "end_x": self.end_x,
            "end_y": self.end_y,
            "duration_ms": self.duration_ms,
        }


def load_play_config(path: str | Path = "config/screen_1080x2400.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data.get("play", {})


def build_discard_drag_plan(
    card_center_x: int,
    card_center_y: int,
    config_path: str | Path = "config/screen_1080x2400.yaml",
) -> DragPlan:
    play = load_play_config(config_path)
    drop_y = int(play.get("discard_drop_y", 470))
    duration_ms = int(play.get("drag_duration_ms", 450))
    start_jitter = int(play.get("drag_start_jitter_px", 0))
    end_jitter_x = int(play.get("drag_end_jitter_x_px", 0))
    end_jitter_y = int(play.get("drag_end_jitter_y_px", 0))
    duration_jitter = int(play.get("drag_duration_jitter_ms", 0))

    start_x = card_center_x + random.randint(-start_jitter, start_jitter)
    start_y = card_center_y + random.randint(-start_jitter, start_jitter)
    end_x = card_center_x + random.randint(-end_jitter_x, end_jitter_x)
    end_y = drop_y + random.randint(-end_jitter_y, end_jitter_y)
    duration_ms = max(1, duration_ms + random.randint(-duration_jitter, duration_jitter))

    return DragPlan(
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        duration_ms=duration_ms,
    )


def execute_discard_drag(
    card_center_x: int,
    card_center_y: int,
    *,
    device_id: str | None = None,
    config_path: str | Path = "config/screen_1080x2400.yaml",
) -> DragPlan:
    plan = build_discard_drag_plan(card_center_x, card_center_y, config_path)
    drag(
        plan.start_x,
        plan.start_y,
        plan.end_x,
        plan.end_y,
        duration_ms=plan.duration_ms,
        device_id=device_id,
        start_jitter_px=0,
        end_jitter_x_px=0,
        end_jitter_y_px=0,
        duration_jitter_ms=0,
    )
    return plan
