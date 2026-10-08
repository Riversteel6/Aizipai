"""Plan or execute ADB drags that compact visible hand cards to the left."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT, WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture.adb_capture import save_screen
from control.adb_tap import drag
from vision.hand_recognizer import (
    HandCard,
    _is_clickable_card,
    detect_hand_slots,
    read_image,
    recognize_hand_from_path,
)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    try:
        if candidate.exists():
            return candidate
    except OSError:
        pass
    nested = ROOT / candidate
    if nested.exists():
        return nested
    return WORKSPACE / candidate


def build_compact_plan(
    cards: list[HandCard],
    *,
    occupancy_cards: list[HandCard] | None = None,
    right_of: int = 1500,
    target_x: int | None = None,
    x_step: int = 8,
    column_gap: int = 145,
    column_tolerance: int = 55,
    max_cards_per_column: int = 4,
) -> list[dict[str, int | str]]:
    """Return right-to-left movable-card drags into detected left-side capacity."""

    movable = _right_to_left_movable_cards(
        cards,
        right_of=right_of,
        column_tolerance=column_tolerance,
    )
    if not movable:
        return []

    occupancy_evidence = occupancy_cards if occupancy_cards is not None else cards
    layout_cards = _merge_layout_evidence(
        cards,
        occupancy_evidence,
        column_tolerance=column_tolerance,
    )
    left_cards = [card for card in layout_cards if card.center[0] < right_of]
    if target_x is None:
        target_x = min((card.center[0] for card in left_cards), default=620)
    blocked_target_xs = [
        card.center[0]
        for card in [*cards, *occupancy_evidence]
        if card.center[0] < right_of and not card.clickable
    ]
    columns = _build_columns(
        left_cards,
        target_x=target_x,
        column_gap=column_gap,
        tolerance=column_tolerance,
        blocked_xs=blocked_target_xs,
    )
    y_slots = _infer_y_slots(layout_cards, max_cards_per_column=max_cards_per_column)

    plan: list[dict[str, int | str]] = []
    for card in movable:
        start_x, start_y = card.center
        target = _allocate_drop_slot(
            columns,
            fallback_y=start_y,
            y_slots=y_slots,
            column_gap=column_gap,
            max_cards_per_column=max_cards_per_column,
        )
        if target is None:
            break
        end_x, end_y = target
        plan.append(
            {
                "label": card.name,
                "card_id": card.card_id or "",
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            }
        )
    return plan


def _merge_layout_evidence(
    recognized_cards: list[HandCard],
    occupancy_cards: list[HandCard],
    *,
    column_tolerance: int,
) -> list[HandCard]:
    columns: list[dict[str, object]] = []
    for source, source_cards in (
        ("recognized", recognized_cards),
        ("occupancy", occupancy_cards),
    ):
        for card in sorted(source_cards, key=lambda item: (item.center[0], item.center[1])):
            center_x = card.center[0]
            column = next(
                (
                    item
                    for item in columns
                    if abs(int(item["x"]) - center_x) <= column_tolerance
                ),
                None,
            )
            if column is None:
                column = {"x": center_x, "recognized": [], "occupancy": []}
                columns.append(column)
            column[source].append(card)

    merged: list[HandCard] = []
    for column in columns:
        recognized = list(column["recognized"])
        occupancy = list(column["occupancy"])
        merged.extend(recognized if len(recognized) >= len(occupancy) else occupancy)
    return merged


def _right_to_left_movable_cards(
    cards: list[HandCard],
    *,
    right_of: int,
    column_tolerance: int,
) -> list[HandCard]:
    """Scan source columns from right to left and skip locked cards or columns."""

    source_columns: list[dict[str, object]] = []
    for card in sorted(
        (item for item in cards if item.center[0] >= right_of),
        key=lambda item: (item.center[0], item.center[1]),
        reverse=True,
    ):
        center_x = card.center[0]
        matched = next(
            (
                column
                for column in source_columns
                if abs(int(column["x"]) - center_x) <= column_tolerance
            ),
            None,
        )
        if matched is None:
            matched = {"x": center_x, "cards": []}
            source_columns.append(matched)
        matched["cards"].append(card)

    source_columns.sort(key=lambda item: int(item["x"]), reverse=True)
    movable: list[HandCard] = []
    for column in source_columns:
        column_cards = sorted(
            (card for card in column["cards"] if card.clickable),
            key=lambda item: item.center[1],
        )
        movable.extend(column_cards)
    return movable


def _build_columns(
    cards: list[HandCard],
    *,
    target_x: int,
    column_gap: int,
    tolerance: int,
    blocked_xs: list[int] | None = None,
) -> list[dict[str, object]]:
    columns: list[dict[str, object]] = []
    for card in sorted(cards, key=lambda item: (item.center[0], item.center[1])):
        center_x, center_y = card.center
        matched = None
        for column in columns:
            if abs(int(column["x"]) - center_x) <= tolerance:
                matched = column
                break
        if matched is None:
            matched = {"x": center_x, "ys": []}
            columns.append(matched)
        matched["ys"].append(center_y)
    if not columns:
        columns.append({"x": target_x, "ys": []})
    columns.sort(key=lambda item: int(item["x"]))

    first_x = int(columns[0]["x"])
    if target_x < first_x - tolerance:
        columns.insert(0, {"x": target_x, "ys": []})
    for column in columns:
        column["blocked"] = any(
            abs(int(column["x"]) - blocked_x) <= tolerance
            for blocked_x in (blocked_xs or [])
        )
    return columns


def _infer_y_slots(cards: list[HandCard], *, max_cards_per_column: int) -> list[int]:
    ys = sorted({card.center[1] for card in cards})
    if not ys:
        return [740, 865, 1005]

    clustered: list[int] = []
    for y in ys:
        if not clustered or abs(clustered[-1] - y) > 45:
            clustered.append(y)
        else:
            clustered[-1] = int(round((clustered[-1] + y) / 2))

    if len(clustered) >= max_cards_per_column:
        return clustered[:max_cards_per_column]

    gaps = [b - a for a, b in zip(clustered, clustered[1:]) if b > a]
    gap = int(round(sum(gaps) / len(gaps))) if gaps else 125
    while len(clustered) < max_cards_per_column:
        candidate = clustered[0] - gap
        if candidate < 90:
            candidate = clustered[-1] + gap
        clustered.insert(0, candidate)
    return sorted(clustered)


def _allocate_drop_slot(
    columns: list[dict[str, object]],
    *,
    fallback_y: int,
    y_slots: list[int],
    column_gap: int,
    max_cards_per_column: int,
) -> tuple[int, int] | None:
    for column in columns:
        if bool(column.get("blocked")):
            continue
        occupied = list(column["ys"])
        if len(occupied) >= max_cards_per_column:
            continue
        actual_y = _first_free_y(y_slots, occupied, fallback_y=fallback_y)
        end_y = _drop_anchor_y(actual_y, occupied, y_slots, fallback_y=fallback_y)
        occupied.append(actual_y)
        column["ys"] = occupied
        return int(column["x"]), int(end_y)
    return None


def _first_free_y(y_slots: list[int], occupied: list[int], *, fallback_y: int) -> int:
    preferred = y_slots[1:] or y_slots
    for slot_y in preferred:
        if all(abs(slot_y - used_y) > 45 for used_y in occupied):
            return slot_y
    for slot_y in y_slots:
        if all(abs(slot_y - used_y) > 45 for used_y in occupied):
            return slot_y
    return fallback_y


def _drop_anchor_y(actual_y: int, occupied: list[int], y_slots: list[int], *, fallback_y: int) -> int:
    if occupied:
        return max(occupied)
    return actual_y


def run_compact(
    *,
    screenshot: str | Path | None = None,
    device_id: str | None = None,
    config_path: str | Path = "config/screen_1080x2400.yaml",
    template_dir: str | Path = "data/templates/hand_auto",
    right_of: int = 1500,
    target_x: int | None = None,
    expected_total: int | None = 21,
    force: bool = False,
    duration_ms: int = 520,
    execute: bool = False,
) -> dict[str, object]:
    source = Path(screenshot) if screenshot else save_screen(device_id=device_id)
    cards = recognize_hand_from_path(
        source,
        config_path=_resolve_project_path(config_path),
        template_dir=_resolve_project_path(template_dir),
    )
    image = read_image(source)
    contour_slots = [
        slot
        for slot in detect_hand_slots(image, config_path=_resolve_project_path(config_path))
        if slot.source == "contour"
    ]
    occupancy_cards = [
        HandCard(
            name="",
            confidence=1.0,
            x=slot.x,
            y=slot.y,
            w=slot.w,
            h=slot.h,
            template="layout_occupancy",
            clickable=_is_clickable_card(
                image[slot.y : slot.y + slot.h, slot.x : slot.x + slot.w]
            ),
        )
        for slot in contour_slots
    ]
    if expected_total is not None and len(cards) >= expected_total and not force:
        plan: list[dict[str, int | str]] = []
        reason = f"hand_count_ok:{len(cards)}>=expected_total:{expected_total}"
    else:
        plan = build_compact_plan(
            cards,
            occupancy_cards=occupancy_cards,
            right_of=right_of,
            target_x=target_x,
        )
        reason = "planned" if plan else "no_right_side_cards_to_compact"

    if execute and plan:
        for step in plan:
            drag(
                int(step["start_x"]),
                int(step["start_y"]),
                int(step["end_x"]),
                int(step["end_y"]),
                duration_ms=duration_ms,
                device_id=device_id,
            )

    return {
        "screenshot": str(source),
        "hand_count": len(cards),
        "occupied_slot_count": len(occupancy_cards),
        "planned_drags": plan,
        "executed": bool(execute and plan),
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact right-side hand cards toward the left.")
    parser.add_argument("--screenshot")
    parser.add_argument("--device-id")
    parser.add_argument("--config-path", default="config/screen_1080x2400.yaml")
    parser.add_argument("--template-dir", default="data/templates/hand_auto")
    parser.add_argument("--right-of", type=int, default=1500)
    parser.add_argument("--target-x", type=int)
    parser.add_argument("--expected-total", type=int, default=21)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--duration-ms", type=int, default=520)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    result = run_compact(
        screenshot=args.screenshot,
        device_id=args.device_id,
        config_path=args.config_path,
        template_dir=args.template_dir,
        right_of=args.right_of,
        target_x=args.target_x,
        expected_total=args.expected_total,
        force=args.force,
        duration_ms=args.duration_ms,
        execute=args.execute,
    )
    print(f"hand_count={result['hand_count']} executed={result['executed']} reason={result['reason']}")
    for step in result["planned_drags"]:
        print(
            f"{step['card_id']} {step['label']}: "
            f"{step['start_x']},{step['start_y']} -> {step['end_x']},{step['end_y']}"
        )


if __name__ == "__main__":
    main()
