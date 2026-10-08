from tools.compact_hand_left import build_compact_plan
from vision.hand_recognizer import HandCard


def _card(label: str, x: int, y: int, *, clickable: bool = True) -> HandCard:
    return HandCard(
        name=label,
        confidence=1.0,
        x=x - 60,
        y=y - 60,
        w=120,
        h=120,
        template="test",
        clickable=clickable,
    )


def test_compact_plan_counts_locked_slots_but_never_moves_them():
    first_full = [_card(str(index), 500, 610 + index * 125) for index in range(4)]
    second_column = [_card(str(index), 650, 735 + index * 125) for index in range(3)]
    locked = _card("八", 1700, 735, clickable=False)
    rightmost = _card("九", 1850, 985)
    cards = [*first_full, *second_column, locked, rightmost]

    plan = build_compact_plan(cards, occupancy_cards=cards)

    assert len(plan) == 1
    assert plan[0]["label"] == "九"
    assert plan[0]["start_x"] == 1850
    assert plan[0]["end_x"] == 650


def test_compact_plan_skips_locked_column_and_orders_source_top_to_bottom():
    destinations = [
        *[_card(str(index), 500, 610 + index * 125) for index in range(3)],
        *[_card(str(index), 650, 735 + index * 125) for index in range(2)],
    ]
    second_from_right = [
        _card("柒", 1700, 860),
        _card("捌", 1700, 985),
    ]
    locked_rightmost = [
        _card("九", 1850, 735, clickable=False),
        _card("十", 1850, 860, clickable=False),
        _card("壹", 1850, 985, clickable=False),
    ]
    cards = [*destinations, *second_from_right, *locked_rightmost]

    plan = build_compact_plan(cards, occupancy_cards=cards)

    assert [step["label"] for step in plan] == ["柒", "捌"]
    assert [step["start_y"] for step in plan] == [860, 985]
    assert all(step["start_x"] == 1700 for step in plan)


def test_compact_plan_fills_detected_columns_left_to_right_up_to_four_cards():
    first_column = [_card(str(index), 500, 610 + index * 125) for index in range(3)]
    second_column = [_card(str(index), 650, 735 + index * 125) for index in range(2)]
    sources = [
        _card("柒", 1600, 985),
        _card("捌", 1750, 985),
        _card("玖", 1900, 985),
    ]
    cards = [*first_column, *second_column, *sources]

    plan = build_compact_plan(cards, occupancy_cards=cards)

    assert [step["end_x"] for step in plan[:3]] == [500, 650, 650]


def test_compact_plan_never_drops_into_locked_left_side_columns():
    first_full = [_card(str(index), 500, 610 + index * 125) for index in range(4)]
    locked_second = [
        _card("二", 650, 735 + index * 125, clickable=False)
        for index in range(3)
    ]
    third_full = [_card(str(index), 800, 610 + index * 125) for index in range(4)]
    open_fourth = [_card("五", 950, 860), _card("伍", 950, 985)]
    sources = [_card("九", 1700, 985), _card("十", 1850, 985)]
    cards = [*first_full, *locked_second, *third_full, *open_fourth, *sources]

    plan = build_compact_plan(cards, occupancy_cards=cards)

    assert [step["end_x"] for step in plan] == [950, 950]
    assert all(step["end_x"] != 650 for step in plan)


def test_compact_plan_uses_occupancy_to_block_an_unrecognized_locked_column():
    first_full = [_card(str(index), 500, 610 + index * 125) for index in range(4)]
    third_full = [_card(str(index), 800, 610 + index * 125) for index in range(4)]
    open_fourth = [_card("五", 950, 860), _card("伍", 950, 985)]
    sources = [_card("九", 1700, 985)]
    recognized = [*first_full, *third_full, *open_fourth, *sources]
    locked_occupancy = [
        _card("", 650, 735 + index * 125, clickable=False)
        for index in range(3)
    ]

    plan = build_compact_plan(
        recognized,
        occupancy_cards=[*recognized, *locked_occupancy],
    )

    assert [step["end_x"] for step in plan] == [950]


def test_compact_plan_merges_recognized_columns_missing_from_contours():
    first_full = [_card(str(index), 500, 610 + index * 125) for index in range(4)]
    second = [_card("二", 650, 735 + index * 125) for index in range(3)]
    third = [_card("三", 800, 735 + index * 125) for index in range(3)]
    fourth = [_card("四", 950, 735 + index * 125) for index in range(3)]
    sources = [
        _card("柒", 1600, 985),
        _card("捌", 1750, 985),
        _card("玖", 1900, 985),
    ]
    recognized = [*first_full, *second, *third, *fourth, *sources]
    contour_occupancy = [*first_full, *second, *fourth, *sources]

    plan = build_compact_plan(recognized, occupancy_cards=contour_occupancy)

    assert [step["end_x"] for step in plan[:3]] == [650, 800, 950]


def test_compact_plan_does_not_invent_a_target_when_known_columns_are_full():
    destinations = [
        *[_card(str(index), 500, 610 + index * 125) for index in range(4)],
        *[_card(str(index), 650, 610 + index * 125) for index in range(4)],
    ]
    source = _card("玖", 1700, 985)

    plan = build_compact_plan([*destinations, source], occupancy_cards=[*destinations, source])

    assert plan == []
