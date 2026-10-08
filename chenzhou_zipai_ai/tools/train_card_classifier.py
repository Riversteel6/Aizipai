"""Train and export a compact 21-class card recognizer for hybrid vision."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cards import BIG_LABELS, SMALL_LABELS, WILD_LABEL
from vision.neural_card_classifier import OnnxCardClassifier
from vision.template_loader import load_templates_from_dirs


LABELS = (*SMALL_LABELS, *BIG_LABELS, WILD_LABEL)
DEFAULT_TEMPLATE_DIRS = (
    ROOT / "data/templates/hand_auto",
    ROOT / "data/templates/hand",
    ROOT / "data/templates/hand_glyph_clean",
    ROOT / "data/templates/apk_card_glyphs",
    ROOT / "data/templates/wildcard",
)
DEFAULT_REAL_CORPUS = ROOT / "data/card_classifier_corpus"


@dataclass(frozen=True)
class CardSample:
    label: str
    image: np.ndarray
    source: str


def collect_samples(
    directories: Sequence[str | Path] = DEFAULT_TEMPLATE_DIRS,
    *,
    real_corpus: str | Path | None = DEFAULT_REAL_CORPUS,
) -> list[CardSample]:
    templates = load_templates_from_dirs(
        tuple(directories),
        grayscale=False,
        require_any=True,
    )
    samples = [
        CardSample(template.label, template.image, str(template.path))
        for template in templates
        if template.label in LABELS
    ]
    corpus = Path(real_corpus) if real_corpus is not None else None
    if corpus is not None and corpus.exists():
        for label in LABELS:
            for path in sorted((corpus / label).glob("*.png")):
                data = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is not None:
                    samples.append(CardSample(label, image, str(path)))
    return samples


def split_samples(
    samples: Sequence[CardSample],
) -> tuple[list[CardSample], list[CardSample]]:
    grouped: dict[str, list[CardSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    missing = [label for label in LABELS if not grouped[label]]
    if missing:
        raise ValueError("missing_training_labels:" + ",".join(missing))

    train: list[CardSample] = []
    validation: list[CardSample] = []
    for label in LABELS:
        ordered = sorted(grouped[label], key=lambda item: item.source)
        preferred = [
            item
            for item in ordered
            if Path(item.source).parent.name == "hand_auto"
        ]
        holdout = preferred[-1] if preferred else ordered[-1]
        validation.append(holdout)
        train.extend(item for item in ordered if item.source != holdout.source)
        if not any(item.label == label for item in train):
            train.append(holdout)
    return train, validation


def appearance_augmentation(
    image: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = image[:, :, :3].copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    spread = result.max(axis=2).astype(np.int16) - result.min(axis=2).astype(np.int16)
    background = (gray >= 165) & (spread <= 55)
    appearance = seed % 4
    if appearance == 1:
        target = np.array([154, 171, 213], dtype=np.float32)
        result[background] = (
            0.30 * result[background].astype(np.float32) + 0.70 * target
        ).astype(np.uint8)
    elif appearance == 2:
        target = np.array([179, 179, 179], dtype=np.float32)
        result[background] = (
            0.25 * result[background].astype(np.float32) + 0.75 * target
        ).astype(np.uint8)

    if seed % 5 == 0:
        height, width = result.shape[:2]
        occlusion_width = max(2, int(width * rng.uniform(0.05, 0.13)))
        x0 = 0 if seed % 10 == 0 else width - occlusion_width
        color = tuple(int(value) for value in (176, 190, 224))
        cv2.rectangle(
            result,
            (x0, int(height * 0.18)),
            (x0 + occlusion_width, int(height * 0.82)),
            color,
            thickness=-1,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/card_classifier.onnx")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--variants-per-label", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    from torchvision.transforms import v2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))

    samples = collect_samples()
    train_samples, validation_samples = split_samples(samples)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((args.input_size, args.input_size), antialias=True),
            v2.RandomPerspective(distortion_scale=0.18, p=0.35),
            v2.RandomAffine(
                degrees=6,
                translate=(0.04, 0.04),
                scale=(0.90, 1.10),
                shear=3,
            ),
            v2.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
            v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.18),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomErasing(p=0.12, scale=(0.01, 0.04), ratio=(0.3, 3.0)),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    validation_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((args.input_size, args.input_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    class BalancedDataset(Dataset):
        def __init__(self) -> None:
            self.grouped = {
                label: [item for item in train_samples if item.label == label]
                for label in LABELS
            }
            self.length = len(LABELS) * max(1, args.variants_per_label)

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int):
            label_index = index % len(LABELS)
            variant = index // len(LABELS)
            candidates = self.grouped[LABELS[label_index]]
            sample = candidates[variant % len(candidates)]
            image = appearance_augmentation(
                sample.image,
                seed=args.seed + index * 1009,
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return train_transform(image), label_index

    class ValidationDataset(Dataset):
        def __len__(self) -> int:
            return len(validation_samples)

        def __getitem__(self, index: int):
            sample = validation_samples[index]
            image = cv2.cvtColor(sample.image[:, :, :3], cv2.COLOR_BGR2RGB)
            return validation_transform(image), LABELS.index(sample.label)

    train_loader = DataLoader(
        BalancedDataset(),
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=max(0, args.workers),
    )
    validation_loader = DataLoader(
        ValidationDataset(),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for block in model.features[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LABELS))
    model.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": 1e-3},
            {
                "params": (
                    parameter
                    for block in model.features[-3:]
                    for parameter in block.parameters()
                ),
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.04)
    history: list[dict[str, float]] = []
    best_validation_accuracy = -1.0
    best_state: dict[str, object] | None = None
    started = perf_counter()
    for epoch in range(max(1, args.epochs)):
        total_loss = 0.0
        total = 0
        correct = 0
        for batch, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            total += len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum())
        validation_accuracy = _torch_accuracy(model, validation_loader)
        row = {
            "epoch": float(epoch + 1),
            "loss": round(total_loss / max(1, total), 6),
            "train_accuracy": round(correct / max(1, total), 6),
            "validation_accuracy": round(validation_accuracy, 6),
        }
        history.append(row)
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(json.dumps(row, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    example = torch.zeros(1, 3, args.input_size, args.input_size)
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    metadata = {
        "schema_version": "aizipai-card-classifier-v1",
        "architecture": "mobilenet_v3_small_fp32",
        "labels": list(LABELS),
        "input_size": args.input_size,
        "mean": mean,
        "std": std,
        "temperature": 1.0,
        "training_seed": args.seed,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    onnx_metrics = _onnx_validation_metrics(args.output, validation_samples)
    report = {
        **metadata,
        "source_samples": len(samples),
        "train_originals": len(train_samples),
        "validation_originals": len(validation_samples),
        "epochs": max(1, args.epochs),
        "variants_per_label": max(1, args.variants_per_label),
        "training_elapsed_s": round(perf_counter() - started, 3),
        "history": history,
        "best_validation_accuracy": round(best_validation_accuracy, 6),
        "onnx_validation": onnx_metrics,
        "model_bytes": args.output.stat().st_size,
    }
    report_path = args.report or args.output.with_name(args.output.stem + "_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _torch_accuracy(model, loader) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch, targets in loader:
            logits = model(batch)
            correct += int((logits.argmax(dim=1) == targets).sum())
            total += len(targets)
    model.train()
    return correct / max(1, total)


def _onnx_validation_metrics(
    model_path: Path,
    samples: Sequence[CardSample],
) -> dict[str, float | int]:
    classifier = OnnxCardClassifier(model_path)
    started = perf_counter()
    predictions = classifier.classify_batch([sample.image for sample in samples])
    elapsed_ms = (perf_counter() - started) * 1000.0
    correct = sum(
        prediction.label == sample.label
        for prediction, sample in zip(predictions, samples)
    )
    return {
        "samples": len(samples),
        "correct": correct,
        "accuracy": round(correct / max(1, len(samples)), 6),
        "batch_elapsed_ms": round(elapsed_ms, 3),
        "per_card_ms": round(elapsed_ms / max(1, len(samples)), 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
