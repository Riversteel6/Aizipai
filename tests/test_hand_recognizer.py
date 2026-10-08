from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vision import hand_recognizer as hand_recognizer_module
from vision.hand_recognizer import (
    CardSlot,
    HandCard,
    _infer_occluded_column_cards,
    _remove_contained_duplicate_cards,
    _with_recovered_top_row,
    clear_classification_cache,
    classify_hand_slot,
    detect_hand_slots,
    recognize_hand,
)
from vision.template_loader import TemplateImage


def _read_template(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def test_recognize_hand_saves_low_confidence_crops_and_debug(monkeypatch, tmp_path):
    image = np.zeros((90, 90, 3), dtype=np.uint8)
    image[10:30, 10:30] = (30, 40, 50)
    image[40:60, 10:30] = (180, 180, 180)
    slots = [CardSlot(10, 10, 20, 20), CardSlot(10, 40, 20, 20)]
    classifications = iter(
        [
            ("候选/牌", 0.65, "low.png"),
            ("二", 0.93, "二_001.png"),
        ]
    )

    monkeypatch.setattr(hand_recognizer_module, "detect_hand_slots", lambda image, **kwargs: slots)
    monkeypatch.setattr(hand_recognizer_module, "load_templates", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        hand_recognizer_module,
        "classify_hand_slot",
        lambda slot_image, templates: next(classifications),
    )

    unknown_dir = tmp_path / "unknown"
    debug_path = tmp_path / "debug_hand.png"
    cards = recognize_hand(
        image,
        config_path=tmp_path / "missing.yaml",
        template_dir=tmp_path / "templates",
        threshold=0.72,
        unknown_crop_dir=unknown_dir,
        debug_path=debug_path,
    )

    assert [card.name for card in cards] == ["二"]
    assert cards[0].card_id == "h001"
    unknown_files = list(unknown_dir.glob("unknown_001_*_650.png"))
    assert len(unknown_files) == 1
    assert unknown_files[0].stat().st_size > 0
    assert debug_path.stat().st_size > 0


def test_infers_occluded_card_from_same_column_pair():
    cards = [
        HandCard("拾", 0.98, 1800, 800, 120, 130, "拾_001.png"),
        HandCard("拾", 0.97, 1800, 930, 120, 140, "拾_001.png"),
    ]
    rejected = [
        {
            "slot": CardSlot(1800, 670, 120, 130),
            "name": "拾",
            "confidence": 0.44,
            "runner_up_name": "七",
            "runner_up_confidence": 0.36,
        }
    ]

    inferred = _infer_occluded_column_cards(cards, rejected)

    assert len(inferred) == 1
    assert inferred[0].name == "拾"
    assert inferred[0].clickable is False
    assert inferred[0].template == "occlusion_inferred_same_column"


def test_does_not_infer_blank_grid_fallback_from_same_column_pair():
    cards = [
        HandCard("拾", 0.98, 1800, 800, 120, 130, "拾_001.png"),
        HandCard("拾", 0.97, 1800, 930, 120, 140, "拾_001.png"),
    ]
    rejected = [
        {
            "slot": CardSlot(1800, 670, 120, 130, source="grid_fallback"),
            "name": "四",
            "confidence": 0.41,
            "runner_up_confidence": 0.40,
        }
    ]

    assert _infer_occluded_column_cards(cards, rejected) == []


def test_recovers_missing_top_row_from_stable_hand_spacing():
    rows = [
        (673, 145, 128),
        (801, 145, 127),
        (928, 145, 152),
    ]

    recovered = _with_recovered_top_row(
        rows,
        SimpleNamespace(y=545),
        search_padding_top=0,
    )

    assert recovered == [
        (545, 145, 128),
        *rows,
    ]


def test_infers_bracketed_runner_up_when_pointer_hides_middle_card():
    cards = [
        HandCard("拾", 0.95, 1760, 545, 145, 128, "拾_top.png"),
        HandCard("拾", 0.94, 1760, 801, 145, 127, "拾_bottom.png"),
    ]
    rejected = [
        {
            "slot": CardSlot(1760, 673, 145, 128, source="grid_fallback"),
            "name": "十",
            "confidence": 0.443,
            "runner_up_name": "拾",
            "runner_up_confidence": 0.412,
        }
    ]

    inferred = _infer_occluded_column_cards(cards, rejected)

    assert [card.name for card in inferred] == ["拾"]
    assert inferred[0].recognition_source == "stack_bracket_consensus"
    assert inferred[0].clickable is False


def test_recognize_hand_recovers_low_confidence_contour_with_clear_margin(monkeypatch, tmp_path):
    image = np.full((80, 80, 3), 230, dtype=np.uint8)
    monkeypatch.setattr(
        hand_recognizer_module,
        "detect_hand_slots",
        lambda image, **kwargs: [CardSlot(10, 10, 30, 40)],
    )
    monkeypatch.setattr(hand_recognizer_module, "load_templates", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        hand_recognizer_module,
        "classify_hand_slot",
        lambda slot_image, templates: ("陆", 0.66, "陆_low.png"),
    )
    monkeypatch.setattr(
        hand_recognizer_module,
        "_runner_up_score",
        lambda slot_image, templates, best_name: ("柒", 0.55),
    )

    cards = recognize_hand(
        image,
        config_path=tmp_path / "missing.yaml",
        template_dir=tmp_path / "templates",
    )

    assert [card.name for card in cards] == ["陆"]
    assert cards[0].confidence == 0.70
    assert cards[0].raw_confidence == 0.66
    assert cards[0].recognition_source == "contour_margin_recovery"


def test_recognize_hand_applies_same_column_occlusion_recovery(monkeypatch, tmp_path):
    image = np.full((160, 60, 3), 230, dtype=np.uint8)
    slots = [
        CardSlot(10, 0, 30, 40),
        CardSlot(10, 50, 30, 40),
        CardSlot(10, 100, 30, 40),
    ]
    classifications = iter(
        [
            ("肆", 0.91, "肆_a.png"),
            ("肆", 0.90, "肆_b.png"),
            ("肆", 0.43, "肆_c.png"),
        ]
    )
    monkeypatch.setattr(hand_recognizer_module, "detect_hand_slots", lambda image, **kwargs: slots)
    monkeypatch.setattr(hand_recognizer_module, "load_templates", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        hand_recognizer_module,
        "classify_hand_slot",
        lambda slot_image, templates: next(classifications),
    )
    monkeypatch.setattr(
        hand_recognizer_module,
        "_runner_up_score",
        lambda slot_image, templates, best_name: ("拾", 0.36),
    )
    monkeypatch.setattr(
        hand_recognizer_module,
        "_secondary_low_confidence_acceptance",
        lambda *args, **kwargs: None,
    )

    diagnostics = {}
    cards = recognize_hand(
        image,
        config_path=tmp_path / "missing.yaml",
        template_dir=tmp_path / "templates",
        diagnostics=diagnostics,
    )

    assert [card.name for card in cards] == ["肆", "肆", "肆"]
    assert cards[-1].recognition_source == "stack_consensus"
    assert not cards[-1].clickable
    assert diagnostics["accepted_count"] == 3
    assert diagnostics["inferred_count"] == 1
    assert diagnostics["recognition_sources"]["stack_consensus"] == 1


def test_classification_cache_reuses_identical_slot(monkeypatch, tmp_path):
    clear_classification_cache()
    templates = [
        TemplateImage("一", tmp_path / "一.png", np.zeros((20, 20, 3), dtype=np.uint8), label="一"),
        TemplateImage("二", tmp_path / "二.png", np.zeros((20, 20, 3), dtype=np.uint8), label="二"),
    ]
    calls = []

    def fake_match(slot_image, slot_gray, slot_mask, template, **_precomputed_scores):
        calls.append(template.label)
        return 0.9 if template.label == "一" else 0.4

    monkeypatch.setattr(hand_recognizer_module, "_match_slot", fake_match)
    crop = np.full((40, 30, 3), 220, dtype=np.uint8)

    assert classify_hand_slot(crop, templates)[0] == "一"
    assert classify_hand_slot(crop.copy(), templates)[0] == "一"
    # The first uncached call uses the fast precomputed score and exact-refines
    # both close candidates; the second identical crop must add no calls.
    assert calls == ["一", "二", "一", "二"]
    clear_classification_cache()


def test_recognizes_synthetic_own_hand_cards():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    template_dir = root / "data" / "templates" / "hand_auto"
    config_path = root / "config" / "screen_1080x2400.yaml"
    canvas = np.zeros((1080, 2344, 3), dtype=np.uint8)

    placements = [
        ("一_001.png", 510, 928),
        ("三_001.png", 660, 928),
        ("贰_003.png", 810, 928),
    ]
    for filename, x, y in placements:
        template = _read_template(template_dir / filename)
        assert template is not None
        h, w = template.shape[:2]
        canvas[y : y + h, x : x + w] = template

    cards = recognize_hand(
        canvas,
        config_path=config_path,
        template_dir=template_dir,
        threshold=0.99,
    )

    assert [card.name for card in cards] == ["一", "三", "贰"]
    assert all(card.confidence > 0.99 for card in cards)


def test_splits_stacked_hand_component_into_visible_slots():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    config_path = root / "config" / "screen_1080x2400.yaml"
    canvas = np.zeros((1080, 2344, 3), dtype=np.uint8)
    canvas[673:1080, 805:950] = (240, 240, 240)

    slots = detect_hand_slots(canvas, config_path=config_path)

    assert [(slot.x, slot.y, slot.w, slot.h) for slot in slots] == [
        (805, 673, 145, 128),
        (805, 801, 145, 127),
        (805, 928, 145, 152),
    ]


def test_splits_four_stacked_hand_component_into_visible_slots():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    config_path = root / "config" / "screen_1080x2400.yaml"
    canvas = np.zeros((1080, 2344, 3), dtype=np.uint8)
    canvas[545:1080, 805:950] = (240, 240, 240)

    slots = detect_hand_slots(canvas, config_path=config_path)

    assert [(slot.x, slot.y, slot.w, slot.h) for slot in slots] == [
        (805, 545, 145, 128),
        (805, 673, 145, 127),
        (805, 800, 145, 128),
        (805, 928, 145, 152),
    ]


def test_exact_expected_contours_skip_empty_grid_expansion():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    config_path = root / "config" / "screen_1080x2400.yaml"
    canvas = np.zeros((1080, 2344, 3), dtype=np.uint8)
    canvas[673:1080, 805:950] = (240, 240, 240)
    canvas[673:1080, 970:1115] = (240, 240, 240)

    slots = detect_hand_slots(
        canvas,
        config_path=config_path,
        expected_slot_count=6,
    )

    assert len(slots) == 6
    assert all(slot.source == "contour" for slot in slots)


def test_grid_fallback_surface_filter_keeps_partial_card_but_drops_background():
    background = np.zeros((128, 145, 3), dtype=np.uint8)
    partial_card = background.copy()
    partial_card[:, :32] = (240, 240, 240)

    assert not hand_recognizer_module._grid_fallback_has_card_surface(background)
    assert hand_recognizer_module._grid_fallback_has_card_surface(partial_card)


def test_ignores_large_floating_action_card_above_hand_region():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    config_path = root / "config" / "screen_1080x2400.yaml"
    canvas = np.zeros((1080, 2344, 3), dtype=np.uint8)
    canvas[480:667, 1124:1276] = (240, 240, 240)

    assert detect_hand_slots(canvas, config_path=config_path) == []


def test_removes_equal_size_overlapping_duplicate_card():
    cards = [
        HandCard("壹", 0.99, 805, 928, 145, 152, "壹_001.png"),
        HandCard(
            "壹",
            0.73,
            838,
            928,
            144,
            152,
            "occlusion_inferred_same_column",
            clickable=False,
            recognition_source="stack_consensus",
        ),
    ]

    result = _remove_contained_duplicate_cards(cards)

    assert len(result) == 1
    assert result[0].x == 805


def test_grid_fallback_recovers_right_hand_column_under_pointer():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = root / "data" / "screenshots" / "screenshot_20260527_155542.png"
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    cards = recognize_hand(
        image,
        config_path=root / "config" / "screen_1080x2400.yaml",
        template_dir=root / "data" / "templates" / "hand_auto",
    )

    assert [card.name for card in cards] == [
        "王",
        "壹",
        "三",
        "三",
        "三",
        "四",
        "肆",
        "五",
        "伍",
        "伍",
        "陆",
        "六",
        "六",
        "七",
        "七",
        "捌",
        "玖",
        "九",
        "九",
        "十",
        "拾",
    ]


def test_pointer_overlay_does_not_duplicate_rightmost_hand_card():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = root / "data" / "screenshots" / "screenshot_20260527_164534.png"
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    cards = recognize_hand(
        image,
        config_path=root / "config" / "screen_1080x2400.yaml",
        template_dir=root / "data" / "templates" / "hand_auto",
    )

    assert len(cards) == 21
    assert [card.name for card in cards][-1:] == ["十"]


def test_selected_pink_card_is_clickable_while_dark_triplets_stay_locked():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = (
        root
        / "data"
        / "debug_previews"
        / "selected_card_20260723"
        / "phone_current.png"
    )
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    cards = recognize_hand(
        image,
        config_path=root / "config" / "screen_1080x2400.yaml",
        template_dir=root / "data" / "templates" / "hand_auto",
    )

    selected_eights = [card for card in cards if card.name == "捌" and card.y < 900]
    assert len(selected_eights) == 1
    assert selected_eights[0].clickable is True
    assert all(not card.clickable for card in cards if card.name in {"九", "拾"})
