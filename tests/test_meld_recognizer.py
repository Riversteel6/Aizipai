from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vision.meld_recognizer import _load_templates, recognize_meld_groups, recognize_melds


def _read_template(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def test_meld_templates_resolve_nested_default_dirs_from_workspace_root():
    templates = _load_templates(("data/templates/meld_auto", "data/templates/discard_auto", "data/templates/hand_auto"))

    assert templates
    assert all(template.path.exists() for template in templates)


def test_recognizes_synthetic_left_side_meld_cells_and_hidden_cards():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    template_dir = root / "data" / "templates" / "meld_auto"
    config_path = root / "config" / "screen_1080x2400.yaml"
    canvas = np.zeros((1080, 2344, 3), dtype=np.uint8)

    top = _read_template(template_dir / "一_001.png")
    assert top is not None
    canvas[622:692, 105:175] = top
    canvas[692:762, 105:175] = (35, 35, 150)
    canvas[762:832, 105:175] = (35, 35, 150)

    melds = recognize_melds(
        canvas,
        config_path=config_path,
        template_dirs=(template_dir,),
        threshold=0.75,
    )

    assert [cell.name for cell in melds["my_melds"]] == ["一", "暗", "暗"]
    assert [cell.hidden for cell in melds["my_melds"]] == [False, True, True]


def test_recognizes_meld_cells_column_first_for_vertical_groups():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    template_dir = root / "data" / "templates" / "meld_auto"
    config_path = root / "config" / "screen_1080x2400.yaml"
    canvas = np.zeros((1080, 2344, 3), dtype=np.uint8)

    left_top = _read_template(template_dir / "柒_001.png")
    right_top = _read_template(template_dir / "壹_001.png")
    right_mid = _read_template(template_dir / "壹_002.png")
    right_bottom = _read_template(template_dir / "一_001.png")
    assert left_top is not None
    assert right_top is not None
    assert right_mid is not None
    assert right_bottom is not None

    canvas[309:379, 105:175] = left_top
    canvas[379:449, 105:175] = (35, 35, 150)
    canvas[449:519, 105:175] = (35, 35, 150)
    canvas[309:379, 175:245] = right_top
    canvas[379:449, 175:245] = right_mid
    canvas[449:519, 175:245] = right_bottom

    melds = recognize_melds(
        canvas,
        config_path=config_path,
        template_dirs=(template_dir,),
        threshold=0.75,
    )
    groups = recognize_meld_groups(
        canvas,
        config_path=config_path,
        template_dirs=(template_dir,),
        threshold=0.75,
    )

    assert [cell.name for cell in melds["opponent_melds"]] == ["柒", "暗", "暗", "壹", "壹", "一"]
    assert [[cell.name for cell in group] for group in groups["opponent_melds"]] == [
        ["柒", "暗", "暗"],
        ["壹", "壹", "一"],
    ]


def test_recognizes_dense_live_meld_columns():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = root / "data" / "screenshots" / "screenshot_20260527_153938.png"
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    groups = recognize_meld_groups(
        image,
        config_path=root / "config" / "screen_1080x2400.yaml",
        template_dirs=(
            root / "data" / "templates" / "meld_auto",
            root / "data" / "templates" / "discard_auto",
            root / "data" / "templates" / "hand_auto",
        ),
        threshold=0.42,
    )

    labels = [[cell.name for cell in group] for group in groups["my_melds"]]
    assert len(labels) >= 4
    assert labels[0] == ["玖", "玖", "玖"]


def test_keeps_red_back_meld_cells_that_overlap_hand_region():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = root / "data" / "screenshots" / "screenshot_20260527_154719.png"
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    groups = recognize_meld_groups(
        image,
        config_path=root / "config" / "screen_1080x2400.yaml",
        template_dirs=(
            root / "data" / "templates" / "meld_auto",
            root / "data" / "templates" / "discard_auto",
            root / "data" / "templates" / "hand_auto",
        ),
        threshold=0.42,
    )

    labels = [[cell.name for cell in group] for group in groups["my_melds"]]
    assert ["柒", "暗", "暗"] in labels


def test_recognizes_live_meld_seven_not_ten():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = root / "data" / "screenshots" / "screenshot_20260527_161846.png"
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    groups = recognize_meld_groups(
        image,
        config_path=root / "config" / "screen_1080x2400.yaml",
        template_dirs=(
            root / "data" / "templates" / "meld_auto",
            root / "data" / "templates" / "discard_auto",
            root / "data" / "templates" / "hand_auto",
        ),
        threshold=0.42,
    )

    labels = [[cell.name for cell in group] for group in groups["my_melds"]]
    assert ["八", "六", "七"] in labels
