"""Optional ONNX Runtime card classifier used beside template matching."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "card_classifier.onnx"


@dataclass(frozen=True)
class NeuralCardPrediction:
    label: str
    confidence: float
    runner_up_label: str
    runner_up_confidence: float

    @property
    def margin(self) -> float:
        return self.confidence - self.runner_up_confidence


class OnnxCardClassifier:
    """Small batch classifier with lazy runtime loading and crop caching."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        metadata_path: str | Path | None = None,
        cache_size: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        metadata_file = (
            Path(metadata_path)
            if metadata_path is not None
            else self.model_path.with_suffix(".json")
        )
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        self.labels = tuple(str(label) for label in metadata["labels"])
        self.input_size = int(metadata.get("input_size", 160))
        self.mean = np.asarray(
            metadata.get("mean", [0.485, 0.456, 0.406]),
            dtype=np.float32,
        ).reshape(1, 1, 3)
        self.std = np.asarray(
            metadata.get("std", [0.229, 0.224, 0.225]),
            dtype=np.float32,
        ).reshape(1, 1, 3)
        self.temperature = max(0.05, float(metadata.get("temperature", 1.0)))
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[bytes, NeuralCardPrediction] = OrderedDict()
        self._cache_lock = Lock()

        self._session = None
        self._net = None
        self._android_bridge = None
        self._net_lock = Lock()
        try:
            import onnxruntime as ort
        except ImportError:
            if os.environ.get("ANDROID_ARGUMENT"):
                from java import jclass

                self._android_bridge = jclass(
                    "com.example.ai.AndroidOnnxBridge"
                )
            else:
                self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        else:
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        self._input_name = self._session.get_inputs()[0].name if self._session is not None else None

    def classify_batch(
        self,
        crops: Sequence[np.ndarray],
    ) -> list[NeuralCardPrediction]:
        results: list[NeuralCardPrediction | None] = [None] * len(crops)
        missing_indices: list[int] = []
        missing_inputs: list[np.ndarray] = []
        missing_keys: list[bytes] = []
        for index, crop in enumerate(crops):
            key = self._cache_key(crop)
            cached = self._cached(key)
            if cached is not None:
                results[index] = cached
                continue
            missing_indices.append(index)
            missing_inputs.append(self._preprocess(crop))
            missing_keys.append(key)

        if missing_inputs:
            batch = np.stack(missing_inputs).astype(np.float32, copy=False)
            if self._session is not None:
                logits = np.asarray(
                    self._session.run(None, {self._input_name: batch})[0],
                    dtype=np.float32,
                )
            elif self._android_bridge is not None:
                little_endian = batch.astype("<f4", copy=False)
                flat_logits = self._android_bridge.runBytes(
                    str(self.model_path),
                    little_endian.tobytes(order="C"),
                    list(batch.shape),
                )
                logits = np.asarray(flat_logits, dtype=np.float32).reshape(
                    len(batch),
                    len(self.labels),
                )
            else:
                with self._net_lock:
                    self._net.setInput(batch)
                    logits = np.asarray(self._net.forward(), dtype=np.float32)
            predictions = self._predictions(logits)
            for index, key, prediction in zip(
                missing_indices,
                missing_keys,
                predictions,
            ):
                results[index] = prediction
                self._store(key, prediction)
        return [
            prediction
            if prediction is not None
            else NeuralCardPrediction("", 0.0, "", 0.0)
            for prediction in results
        ]

    @property
    def backend_name(self) -> str:
        if self._session is not None:
            return "onnxruntime"
        if self._android_bridge is not None:
            return "onnxruntime_android"
        return "opencv_dnn"

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            rgb = np.full(
                (self.input_size, self.input_size, 3),
                255,
                dtype=np.uint8,
            )
        else:
            rgb = (
                cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
                if crop.ndim == 2
                else cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2RGB)
            )
            rgb = cv2.resize(
                rgb,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_AREA,
            )
        normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return np.transpose(normalized, (2, 0, 1))

    def _predictions(self, logits: np.ndarray) -> list[NeuralCardPrediction]:
        scaled = logits / self.temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predictions: list[NeuralCardPrediction] = []
        for row in probabilities:
            order = np.argsort(row)[::-1]
            best = int(order[0])
            second = int(order[1]) if len(order) > 1 else best
            predictions.append(
                NeuralCardPrediction(
                    label=self.labels[best],
                    confidence=float(row[best]),
                    runner_up_label=self.labels[second],
                    runner_up_confidence=float(row[second]),
                )
            )
        return predictions

    def _cache_key(self, crop: np.ndarray) -> bytes:
        if crop.size == 0:
            return b"empty"
        small = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
        return hashlib.blake2b(
            np.ascontiguousarray(small // 4).tobytes(),
            digest_size=16,
        ).digest()

    def _cached(self, key: bytes) -> NeuralCardPrediction | None:
        with self._cache_lock:
            prediction = self._cache.get(key)
            if prediction is not None:
                self._cache.move_to_end(key)
            return prediction

    def _store(self, key: bytes, prediction: NeuralCardPrediction) -> None:
        if self.cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[key] = prediction
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)


__all__ = [
    "DEFAULT_MODEL_PATH",
    "NeuralCardPrediction",
    "OnnxCardClassifier",
]
