from vision.history_memory import build_sanity_checks


def _meld_cells(labels):
    return [{"name": label} for label in labels]


def test_opening_total_counts_three_automatic_landed_cards():
    snapshot = {
        "hand_count": 18,
        "meld_groups": {"my_melds": [_meld_cells(["七", "七", "七"])]},
    }

    checks = build_sanity_checks(snapshot, expected_total=21)

    assert checks["ok"] is True
    assert checks["controlled_card_count"] == 21
    assert checks["my_meld_cell_count"] == 3


def test_opening_total_counts_four_automatic_landed_cards():
    snapshot = {
        "hand_count": 17,
        "meld_groups": {"my_melds": [_meld_cells(["八", "八", "八", "八"])]},
    }

    checks = build_sanity_checks(snapshot, expected_total=21)

    assert checks["ok"] is True
    assert checks["controlled_card_count"] == 21
    assert checks["my_meld_cell_count"] == 4
