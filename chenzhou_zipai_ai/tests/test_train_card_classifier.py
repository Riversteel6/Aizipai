import numpy as np

from tools.train_card_classifier import (
    LABELS,
    CardSample,
    appearance_augmentation,
    collect_samples,
    split_samples,
)


def test_split_keeps_all_labels_in_training_and_validation() -> None:
    samples = [
        CardSample(label, np.full((20, 20, 3), 220, dtype=np.uint8), f"{label}-{index}")
        for label in LABELS
        for index in range(2)
    ]

    train, validation = split_samples(samples)

    assert {sample.label for sample in train} == set(LABELS)
    assert {sample.label for sample in validation} == set(LABELS)
    assert len(validation) == len(LABELS)


def test_selected_appearance_changes_bright_background() -> None:
    image = np.full((40, 30, 3), 225, dtype=np.uint8)
    image[10:30, 12:18] = 20

    selected = appearance_augmentation(image, seed=1)

    assert tuple(selected[0, 0]) != tuple(image[0, 0])
    assert tuple(selected[15, 15]) == tuple(image[15, 15])


def test_collect_samples_adds_labelled_real_crops(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    label_dir = corpus / "一"
    label_dir.mkdir(parents=True)
    image = np.full((20, 20, 3), 225, dtype=np.uint8)
    import cv2

    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(label_dir / "sample.png"))

    samples = collect_samples(real_corpus=corpus)

    assert any(sample.source.endswith("sample.png") for sample in samples)
