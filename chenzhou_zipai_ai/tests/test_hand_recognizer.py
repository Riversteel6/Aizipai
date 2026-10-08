"""Tests for own-hand visual state helpers."""

from pathlib import Path

import cv2
import numpy as np

from vision.hand_recognizer import (
    _batch_glyph_template_scores,
    _batch_resized_template_scores,
    _glyph_mask,
    _is_clickable_card,
    _is_selected_card,
    _with_recovered_top_row,
)
from vision.regions import Region
from vision.template_loader import TemplateImage


def _card_like_image(background: int) -> np.ndarray:
    image = np.full((152, 144, 3), background, dtype=np.uint8)
    cv2.line(image, (72, 28), (72, 120), (20, 20, 20), thickness=8)
    cv2.line(image, (48, 64), (96, 64), (20, 20, 20), thickness=8)
    cv2.line(image, (52, 100), (92, 122), (20, 20, 20), thickness=8)
    return image


def test_batched_resized_scores_match_opencv_template_scores():
    rng = np.random.default_rng(20260824)
    slot = rng.integers(0, 256, size=(73, 61), dtype=np.uint8)
    templates = [
        TemplateImage(
            name=f"template_{index}",
            path=Path(f"template_{index}.png"),
            image=rng.integers(
                0,
                256,
                size=(41 + index * 3, 37 + index * 2),
                dtype=np.uint8,
            ),
        )
        for index in range(7)
    ]

    batched = _batch_resized_template_scores(slot, templates)
    expected = np.asarray(
        [
            cv2.matchTemplate(
                slot,
                cv2.resize(template.image, (slot.shape[1], slot.shape[0])),
                cv2.TM_CCOEFF_NORMED,
            )[0, 0]
            for template in templates
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(batched, expected, rtol=0.0, atol=2e-6)


def test_batched_glyph_scores_match_opencv_template_scores():
    rng = np.random.default_rng(20260825)
    slot_mask = (rng.integers(0, 2, size=(80, 80), dtype=np.uint8) * 255)
    templates = [
        TemplateImage(
            name=f"glyph_{index}",
            path=Path(f"glyph_{index}.png"),
            image=np.repeat(
                (rng.integers(0, 2, size=(80, 80), dtype=np.uint8) * 255)[..., None],
                3,
                axis=2,
            ),
        )
        for index in range(7)
    ]

    batched = _batch_glyph_template_scores(slot_mask, templates)
    expected = np.asarray(
        [
            cv2.matchTemplate(
                slot_mask,
                _glyph_mask(template.image),
                cv2.TM_CCOEFF_NORMED,
            )[0, 0]
            for template in templates
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(batched, expected, rtol=0.0, atol=2e-6)


def test_clickable_card_keeps_white_background_even_with_dark_glyph():
    assert _is_clickable_card(_card_like_image(225)) is True


def test_dark_locked_card_rejects_grey_background():
    assert _is_clickable_card(_card_like_image(179)) is False


def test_pink_selected_card_remains_clickable_even_with_dark_center():
    image = np.full((152, 144, 3), (154, 171, 213), dtype=np.uint8)
    cv2.rectangle(image, (38, 24), (106, 128), (20, 20, 20), thickness=-1)

    assert _is_selected_card(image) is True
    assert _is_clickable_card(image) is True


def test_locked_grey_card_with_red_glyph_is_not_selected():
    image = np.full((152, 144, 3), 179, dtype=np.uint8)
    cv2.rectangle(image, (38, 24), (106, 128), (40, 40, 190), thickness=-1)

    assert _is_selected_card(image) is False
    assert _is_clickable_card(image) is False


def test_recovered_fourth_row_uses_configured_top_search_padding():
    region = Region(name="my_hand", x=430, y=660, w=1680, h=420)
    rows = [
        (673, 144, 128),
        (801, 144, 127),
        (928, 144, 152),
    ]

    recovered = _with_recovered_top_row(rows, region, search_padding_top=180)

    assert recovered[0] == (545, 144, 128)


def test_glyph_mask_ignores_neighbor_column_and_stacked_card_edges():
    clean = np.full((127, 170, 3), 225, dtype=np.uint8)
    red = (30, 30, 190)
    cv2.line(clean, (52, 18), (109, 75), red, thickness=10)
    cv2.line(clean, (101, 18), (54, 103), red, thickness=10)
    cv2.line(clean, (34, 68), (124, 68), red, thickness=10)
    cv2.circle(clean, (52, 34), 8, red, thickness=-1)

    contaminated = clean.copy()
    cv2.rectangle(contaminated, (143, 0), (169, 126), (25, 25, 25), thickness=-1)
    cv2.line(contaminated, (14, 126), (130, 126), (25, 25, 25), thickness=1)
    cv2.rectangle(contaminated, (140, 4), (141, 5), (25, 25, 25), thickness=-1)

    clean_mask = _glyph_mask(clean)
    contaminated_mask = _glyph_mask(contaminated, remove_edge_artifacts=True)

    assert clean_mask is not None
    assert contaminated_mask is not None
    similarity = cv2.matchTemplate(
        clean_mask,
        contaminated_mask,
        cv2.TM_CCOEFF_NORMED,
    )[0, 0]
    assert similarity >= 0.95
