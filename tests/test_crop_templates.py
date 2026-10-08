from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.crop_templates import crop_template


def test_crop_template_saves_incrementing_template_and_preview(tmp_path):
    screenshot = tmp_path / "screen.png"
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[20:50, 30:70] = (10, 120, 240)
    assert cv2.imwrite(str(screenshot), image)

    first, first_preview = crop_template(
        screenshot,
        card_name="王",
        x=30,
        y=20,
        w=40,
        h=30,
        template_group="hand",
        output_root=tmp_path / "templates",
        preview_root=tmp_path / "previews",
    )
    second, _ = crop_template(
        screenshot,
        card_name="王",
        x=30,
        y=20,
        w=40,
        h=30,
        template_group="hand",
        output_root=tmp_path / "templates",
        preview_root=tmp_path / "previews",
    )

    assert first.name == "王_001.png"
    assert second.name == "王_002.png"
    assert first_preview.exists()
    cropped = cv2.imdecode(np.fromfile(str(first), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert cropped.shape[:2] == (30, 40)


def test_crop_template_can_save_appearance_variants(tmp_path):
    screenshot = tmp_path / "screen.png"
    image = np.zeros((80, 90, 3), dtype=np.uint8)
    image[10:40, 15:45] = (30, 160, 220)
    assert cv2.imwrite(str(screenshot), image)

    selected, _ = crop_template(
        screenshot,
        card_name="二",
        appearance="selected",
        x=15,
        y=10,
        w=30,
        h=30,
        template_group="hand",
        output_root=tmp_path / "templates",
        preview_root=tmp_path / "previews",
    )
    triple, _ = crop_template(
        screenshot,
        card_name="二",
        appearance="triple_stack",
        x=15,
        y=10,
        w=30,
        h=30,
        template_group="hand",
        output_root=tmp_path / "templates",
        preview_root=tmp_path / "previews",
    )

    assert selected.name == "二_selected_001.png"
    assert triple.name == "二_triple_stack_001.png"
