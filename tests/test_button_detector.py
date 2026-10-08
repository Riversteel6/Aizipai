from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vision.button_detector import detect_buttons


def test_detects_action_button_templates_on_synthetic_buttons_region():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    template_dir = root / "data" / "templates" / "buttons"
    config_path = root / "config" / "screen_1080x2400.yaml"

    for name in ("chi", "peng", "pass", "hu"):
        canvas = np.zeros((1080, 2344, 3), dtype=np.uint8)
        template = cv2.imread(str(template_dir / f"{name}_reference.png"), cv2.IMREAD_COLOR)
        assert template is not None
        h, w = template.shape[:2]
        x, y = 1970, 455
        canvas[y : y + h, x : x + w] = template

        detections = detect_buttons(
            canvas,
            config_path=config_path,
            template_dir=template_dir,
            threshold=0.99,
            names=[name],
        )

        assert [item.name for item in detections] == [name]
        assert detections[0].confidence > 0.99


def test_detects_live_peng_button_at_current_threshold():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = root / "data" / "screenshots" / "screenshot_20260527_154343.png"
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    detections = detect_buttons(
        image,
        config_path=root / "config" / "screen_1080x2400.yaml",
        template_dir=root / "data" / "templates" / "buttons",
        threshold=0.75,
    )

    assert "peng" in [button.name for button in detections]


def test_response_button_row_fallback_keeps_lower_confidence_pass_and_peng():
    root = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
    screenshot = Path("data/screenshots/screenshot_20260601_102440.png")
    if not screenshot.exists():
        return
    image = cv2.imread(str(screenshot))

    detections = detect_buttons(
        image,
        config_path=root / "config" / "screen_1080x2400.yaml",
        template_dir=root / "data" / "templates" / "buttons",
        threshold=0.82,
    )

    names = {button.name for button in detections}
    assert {"chi", "peng", "pass"}.issubset(names)
