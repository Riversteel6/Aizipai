from types import SimpleNamespace

import numpy as np

import vision.hand_target_locator as hand_target_locator
from vision.hand_target_locator import TargetCandidate, choose_target_candidate


def _candidate(
    label: str,
    x: int,
    *,
    clickable: bool = True,
    accepted: bool = True,
    selected: bool = False,
):
    return TargetCandidate(
        label=label,
        confidence=0.92 if accepted else 0.60,
        x=x,
        y=800,
        w=100,
        h=140,
        clickable=clickable,
        accepted=accepted,
        reason="test",
        selected=selected,
    )


def test_target_stays_at_original_coordinate():
    result = choose_target_candidate(
        [_candidate("一", 450)],
        expected_label="一",
        original_x=500,
        original_y=870,
    )

    assert result["status"] == "matched"
    assert (result["x"], result["y"]) == (500, 870)


def test_target_relocates_and_ignores_locked_copy():
    result = choose_target_candidate(
        [_candidate("一", 450, clickable=False), _candidate("一", 750)],
        expected_label="一",
        original_x=500,
        original_y=870,
    )

    assert result["status"] == "relocated"
    assert (result["x"], result["y"]) == (800, 870)


def test_different_card_at_old_coordinate_is_not_accepted():
    result = choose_target_candidate(
        [_candidate("九", 450)],
        expected_label="一",
        original_x=500,
        original_y=870,
    )

    assert result["status"] == "missing"


def test_low_confidence_target_is_uncertain_not_executable():
    result = choose_target_candidate(
        [_candidate("一", 450, accepted=False)],
        expected_label="一",
        original_x=500,
        original_y=870,
    )

    assert result["status"] == "uncertain"


def test_semantic_target_preserves_selected_appearance():
    result = choose_target_candidate(
        [_candidate("一", 450, selected=True)],
        expected_label="一",
        original_x=500,
        original_y=870,
    )

    assert result["status"] == "matched"
    assert result["selected"] is True


def test_selection_verification_prefers_selected_duplicate_over_old_coordinate():
    result = choose_target_candidate(
        [
            _candidate("一", 450, selected=False),
            _candidate("一", 750, selected=True),
        ],
        expected_label="一",
        original_x=500,
        original_y=870,
        prefer_selected=True,
    )

    assert result["status"] == "relocated"
    assert (result["x"], result["y"]) == (800, 870)
    assert result["selected"] is True


def test_selected_target_is_classified_without_the_pink_overlay(monkeypatch):
    slot = SimpleNamespace(x=0, y=0, w=10, h=10)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 80
    image[:, :, 1] = 120
    image[:, :, 2] = 220

    def classify(crops, **_kwargs):
        assert len(crops) == 1
        assert np.array_equal(crops[0][:, :, 0], crops[0][:, :, 1])
        assert np.array_equal(crops[0][:, :, 1], crops[0][:, :, 2])
        return [("柒", 0.90, True, "selected_overlay_normalized")]

    monkeypatch.setattr(hand_target_locator, "detect_hand_slots", lambda *_args, **_kwargs: [slot])
    monkeypatch.setattr(hand_target_locator, "_is_clickable_card", lambda _crop: True)
    monkeypatch.setattr(hand_target_locator, "_is_selected_card", lambda _crop: True)
    monkeypatch.setattr(hand_target_locator, "_classify_slot_crops", classify)

    result = hand_target_locator.locate_hand_target(
        image,
        expected_label="柒",
        original_x=5,
        original_y=5,
        config_path="unused.yaml",
        template_dir="unused",
        prefer_selected=True,
    )

    assert result["status"] == "matched"
    assert result["selected"] is True


def test_selected_target_retries_original_color_only_after_grayscale_rejection(monkeypatch):
    slot = SimpleNamespace(x=0, y=0, w=10, h=10)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 80
    image[:, :, 1] = 120
    image[:, :, 2] = 220
    classify_calls = []

    def classify(crops, **_kwargs):
        classify_calls.append(crops[0].copy())
        if len(classify_calls) == 1:
            assert np.array_equal(crops[0][:, :, 0], crops[0][:, :, 1])
            return [("", 0.0, False, "confident_disagreement")]
        assert np.array_equal(crops[0], image)
        return [("二", 0.89, True, "independent_agreement")]

    monkeypatch.setattr(hand_target_locator, "detect_hand_slots", lambda *_args, **_kwargs: [slot])
    monkeypatch.setattr(hand_target_locator, "_is_clickable_card", lambda _crop: True)
    monkeypatch.setattr(hand_target_locator, "_is_selected_card", lambda _crop: True)
    monkeypatch.setattr(hand_target_locator, "_classify_slot_crops", classify)

    result = hand_target_locator.locate_hand_target(
        image,
        expected_label="二",
        original_x=5,
        original_y=5,
        config_path="unused.yaml",
        template_dir="unused",
        prefer_selected=True,
    )

    assert result["status"] == "matched"
    assert result["selected"] is True
    assert result["reason"] == "selected_color_fallback:independent_agreement"
    assert len(classify_calls) == 2


def test_selected_color_retry_cannot_override_accepted_wrong_card(monkeypatch):
    slot = SimpleNamespace(x=0, y=0, w=10, h=10)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    classify_batch_sizes = []

    def classify(crops, **_kwargs):
        classify_batch_sizes.append(len(crops))
        return [("九", 0.94, True, "independent_agreement") for _crop in crops]

    monkeypatch.setattr(hand_target_locator, "detect_hand_slots", lambda *_args, **_kwargs: [slot])
    monkeypatch.setattr(hand_target_locator, "_is_clickable_card", lambda _crop: True)
    monkeypatch.setattr(hand_target_locator, "_is_selected_card", lambda _crop: True)
    monkeypatch.setattr(hand_target_locator, "_classify_slot_crops", classify)

    result = hand_target_locator.locate_hand_target(
        image,
        expected_label="二",
        original_x=5,
        original_y=5,
        config_path="unused.yaml",
        template_dir="unused",
        prefer_selected=True,
    )

    assert result["status"] == "missing"
    assert classify_batch_sizes == [1, 0]


def test_near_original_selected_relocation_does_not_scan_the_whole_hand(monkeypatch):
    slots = [
        SimpleNamespace(x=0, y=0, w=10, h=10),
        SimpleNamespace(x=20, y=0, w=10, h=10),
    ]
    classify_batch_sizes = []

    def classify(crops, **_kwargs):
        classify_batch_sizes.append(len(crops))
        return [("二", 0.93, True, "independent_agreement") for _crop in crops]

    monkeypatch.setattr(hand_target_locator, "detect_hand_slots", lambda *_args, **_kwargs: slots)
    monkeypatch.setattr(hand_target_locator, "_is_clickable_card", lambda _crop: True)
    monkeypatch.setattr(
        hand_target_locator,
        "_is_selected_card",
        lambda crop: bool(np.shares_memory(crop, image[:, :10])),
    )
    monkeypatch.setattr(hand_target_locator, "_classify_slot_crops", classify)
    image = np.zeros((10, 30, 3), dtype=np.uint8)

    result = hand_target_locator.locate_hand_target(
        image,
        expected_label="二",
        original_x=5,
        original_y=24,
        config_path="unused.yaml",
        template_dir="unused",
        prefer_selected=True,
    )

    assert result["status"] == "relocated"
    assert result["selected"] is True
    assert result["classified_slot_count"] == 1
    assert classify_batch_sizes == [1]


def test_unselected_target_keeps_original_color_crop():
    crop = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)

    normalized = hand_target_locator._classification_crop(crop, selected=False)

    assert normalized is crop


def test_locate_hand_target_scans_other_slots_when_original_card_changed(monkeypatch):
    slots = [
        SimpleNamespace(x=0, y=0, w=10, h=10),
        SimpleNamespace(x=20, y=0, w=10, h=10),
    ]
    classify_calls = 0

    def classify(crops, **_kwargs):
        nonlocal classify_calls
        classify_calls += 1
        if classify_calls == 1:
            return [("九", 0.95, True, "test_original_changed")]
        return [("一", 0.94, True, "test_relocated") for _crop in crops]

    monkeypatch.setattr(hand_target_locator, "detect_hand_slots", lambda *_args, **_kwargs: slots)
    monkeypatch.setattr(hand_target_locator, "_is_clickable_card", lambda _crop: True)
    monkeypatch.setattr(hand_target_locator, "_is_selected_card", lambda _crop: False)
    monkeypatch.setattr(hand_target_locator, "_classify_slot_crops", classify)

    result = hand_target_locator.locate_hand_target(
        np.zeros((10, 30, 3), dtype=np.uint8),
        expected_label="一",
        original_x=5,
        original_y=5,
        config_path="unused.yaml",
        template_dir="unused",
    )

    assert result["status"] == "relocated"
    assert (result["x"], result["y"]) == (25, 5)
    assert result["classified_slot_count"] == 2


def test_slot_classifier_falls_back_when_neural_result_count_is_wrong(monkeypatch, tmp_path):
    model = tmp_path / "card_classifier.onnx"
    model.write_bytes(b"test")

    class BrokenClassifier:
        def classify_batch(self, _crops):
            return []

    monkeypatch.setattr(hand_target_locator, "load_templates", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(
        hand_target_locator,
        "_candidate_scores",
        lambda _crop, _templates: [(0.95, "一", "template")],
    )
    monkeypatch.setattr(hand_target_locator, "_load_neural_classifier", lambda *_args: BrokenClassifier())

    predictions = hand_target_locator._classify_slot_crops(
        [np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)],
        template_dir="unused",
        neural_model_path=model,
    )

    assert predictions == [
        ("一", 0.95, True, "strong_template"),
        ("一", 0.95, True, "strong_template"),
    ]
