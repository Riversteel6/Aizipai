from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vision.discard_recognizer import _load_discard_templates, recognize_discards


def _read_template(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def test_discard_templates_resolve_nested_default_dirs_from_workspace_root():
    templates = _load_discard_templates(("data/templates/discard_auto", "data/templates/hand_auto"))

    assert templates
    assert all(template.path.exists() for template in templates)


def test_recognizes_synthetic_right_side_discard_cards():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    template_dir = root / "data" / "templates" / "hand_auto"
    config_path = root / "config" / "screen_1080x2400.yaml"
    canvas = np.zeros((1080, 2344, 3), dtype=np.uint8)

    placements = [
        ("十_001.png", 1661, 92),
        ("九_001.png", 1718, 92),
        ("拾_004.png", 1830, 92),
    ]
    for filename, x, y in placements:
        template = _read_template(template_dir / filename)
        assert template is not None
        small = cv2.resize(template, (55, 55), interpolation=cv2.INTER_AREA)
        canvas[y : y + 55, x : x + 55] = small

    discards = recognize_discards(
        canvas,
        config_path=config_path,
        template_dirs=(template_dir,),
        threshold=0.75,
    )

    assert [card.name for card in discards["opponent_discards"]] == ["十", "九", "拾"]
