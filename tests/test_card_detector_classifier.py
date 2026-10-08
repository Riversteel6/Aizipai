from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vision import card_classifier, card_detector
from vision.hand_recognizer import CardSlot


def test_card_detector_wraps_hand_slot_detection_and_crops(monkeypatch):
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[5:15, 10:25] = (12, 34, 56)
    slots = [CardSlot(x=10, y=5, w=15, h=10)]
    captured = {}

    def fake_detect_hand_slots(image_arg, **kwargs):
        captured["shape"] = image_arg.shape
        captured["kwargs"] = kwargs
        return slots

    monkeypatch.setattr(card_detector, "detect_hand_slots", fake_detect_hand_slots)

    detected = card_detector.detect_card_slots(
        image,
        config_path="config.yaml",
        search_padding_left=1,
        search_padding_top=2,
        search_padding_right=3,
    )
    crops = card_detector.crop_card_slots(image, detected)

    assert detected == slots
    assert captured["shape"] == image.shape
    assert captured["kwargs"] == {
        "config_path": "config.yaml",
        "search_padding_left": 1,
        "search_padding_top": 2,
        "search_padding_right": 3,
    }
    assert crops[0].shape == (10, 15, 3)
    assert tuple(crops[0][0, 0]) == (12, 34, 56)
    crops[0][0, 0] = (0, 0, 0)
    assert tuple(image[5, 10]) == (12, 34, 56)


def test_card_classifier_loads_templates_once_and_reports_threshold(monkeypatch, tmp_path):
    crops = [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)]
    loaded_templates = [object()]
    calls = {"load": 0, "classify": 0}

    def fake_load_templates(template_dir, grayscale=False):
        calls["load"] += 1
        assert Path(template_dir) == tmp_path
        assert grayscale is False
        return loaded_templates

    def fake_classify_hand_slot(crop, templates):
        calls["classify"] += 1
        assert templates == loaded_templates
        if calls["classify"] == 1:
            return "二", 0.91, "二_001.png"
        return "三", 0.61, "三_001.png"

    monkeypatch.setattr(card_classifier, "load_templates", fake_load_templates)
    monkeypatch.setattr(card_classifier, "classify_hand_slot", fake_classify_hand_slot)

    results = card_classifier.classify_card_crops(crops, template_dir=tmp_path, threshold=0.72)

    assert calls == {"load": 1, "classify": 2}
    assert [item.name for item in results] == ["二", "三"]
    assert [item.accepted for item in results] == [True, False]
    assert results[0].to_dict() == {
        "name": "二",
        "confidence": 0.91,
        "template": "二_001.png",
        "accepted": True,
    }
