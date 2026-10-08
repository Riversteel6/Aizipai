"""Live option-recognition regressions."""

from pathlib import Path

import cv2

from vision.neural_card_classifier import NeuralCardPrediction
from vision.option_recognizer import (
    _correct_option_labels,
    _resolve_option_cell,
    load_option_templates,
    recognize_options,
)


def test_load_option_templates_resolves_nested_default_dirs_from_workspace_root():
    templates = load_option_templates(("data/templates/option_auto", "data/templates/hand_auto"))

    assert templates
    assert all(template.path.exists() for template in templates)


def test_recognizes_live_sequence_chi_options():
    screenshot = Path("data/screenshots/screenshot_20260527_160227.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    options = recognize_options(image)

    assert [option.labels for option in options[:2]] == [
        ["六", "七", "八"],
        ["七", "八", "九"],
    ]


def test_recognizes_live_same_rank_chi_options():
    screenshot = Path("data/screenshots/screenshot_20260527_160538.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    options = recognize_options(image)

    assert [option.labels for option in options[:2]] == [
        ["玖", "玖", "九"],
        ["九", "九", "玖"],
    ]


def test_recognizes_live_six_chi_options_without_sandwich_false_positive():
    screenshot = Path("data/screenshots/screenshot_20260527_161344.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    options = recognize_options(image)

    assert [option.labels for option in options[:3]] == [
        ["陆", "陆", "六"],
        ["六", "六", "陆"],
        ["肆", "伍", "陆"],
    ]


def test_recognizes_live_big_345_option():
    screenshot = Path("data/screenshots/screenshot_20260527_161846.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    options = recognize_options(image)

    assert [option.labels for option in options[:1]] == [["叁", "肆", "伍"]]


def test_recognizes_live_small_456_and_567_options():
    screenshot = Path("data/screenshots/screenshot_20260527_162924.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    options = recognize_options(image)

    assert [option.labels for option in options[:2]] == [
        ["四", "五", "六"],
        ["五", "六", "七"],
    ]


def test_recognizes_live_big_2710_option_under_close_match():
    screenshot = Path("data/screenshots/screenshot_20260527_164926.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    options = recognize_options(image)

    assert [option.labels for option in options[:1]] == [["贰", "柒", "拾"]]


def test_recognizes_live_small_345_option_under_close_match():
    screenshot = Path("data/screenshots/screenshot_20260527_170217.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    options = recognize_options(image)

    assert [option.labels for option in options[:1]] == [["三", "四", "五"]]


def test_recognizes_live_compare_8910_option_under_close_match():
    screenshot = Path("data/screenshots/screenshot_20260527_170608.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    options = recognize_options(image)

    compare = [option for option in options if option.region_name == "compare_options"]
    assert [option.labels for option in compare[:1]] == [["捌", "玖", "拾"]]


def test_recognizes_incident_same_rank_option_without_size_guessing():
    screenshot = Path(
        "logs/auto_script_only_1v1_wang_chi_speed_resume_20260814_104531/"
        "sessions/session_20260814_104531_41536/rounds/round_0001/"
        "screenshots/frame_000001_before_action.png"
    )
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    options = recognize_options(image)

    assert options[0].labels == ["叁", "叁", "三"]


def test_option_cell_uses_medium_probability_neural_result_with_clear_margin():
    prediction = NeuralCardPrediction(
        label="叁",
        confidence=0.468,
        runner_up_label="八",
        runner_up_confidence=0.147,
    )

    resolved = _resolve_option_cell(
        [(0.741, "八", "八.png"), (0.727, "六", "六.png")],
        prediction,
    )

    assert resolved is not None
    assert resolved[0] == "叁"
    assert resolved[2] == "neural_clear_margin"


def test_option_cell_keeps_strong_template_when_neural_is_ambiguous():
    prediction = NeuralCardPrediction(
        label="九",
        confidence=0.32,
        runner_up_label="玖",
        runner_up_confidence=0.28,
    )

    resolved = _resolve_option_cell(
        [(1.0, "玖", "玖.png"), (0.748, "八", "八.png")],
        prediction,
    )

    assert resolved is not None
    assert resolved[0] == "玖"
    assert resolved[2] == "template_clear_margin"


def test_option_cell_rejects_two_ambiguous_branches():
    prediction = NeuralCardPrediction(
        label="九",
        confidence=0.32,
        runner_up_label="玖",
        runner_up_confidence=0.28,
    )

    assert _resolve_option_cell(
        [(0.77, "七", "七.png"), (0.769, "拾", "拾.png")],
        prediction,
    ) is None


def test_option_labels_are_not_speculatively_rewritten_without_context():
    for labels in (
        ["十", "十", "七"],
        ["七", "七", "陆"],
        ["六", "六", "七"],
        ["五", "伍", "五"],
        ["伍", "五", "伍"],
        ["五", "五", "伍"],
        ["叁", "八", "八"],
    ):
        assert _correct_option_labels(labels) == labels
