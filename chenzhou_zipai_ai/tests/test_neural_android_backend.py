from __future__ import annotations

import sys

import numpy as np

from vision.neural_card_classifier import OnnxCardClassifier


def test_opencv_dnn_fallback_uses_same_model_labels(monkeypatch) -> None:
    crop = np.full((160, 90, 3), 255, dtype=np.uint8)
    expected = OnnxCardClassifier(cache_size=0).classify_batch([crop])[0]
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    actual = OnnxCardClassifier(cache_size=0).classify_batch([crop])[0]

    assert actual.label == expected.label
    assert abs(actual.confidence - expected.confidence) < 0.02
