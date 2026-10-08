from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vision.seat_role_detector import detect_seat_role


def test_detect_seat_role_marks_gold_badge_as_dealer(tmp_path):
    config = tmp_path / "screen.yaml"
    config.write_text(
        """
screen:
  width: 200
  height: 120
regions:
  my_dealer_marker:
    x: 50
    y: 30
    w: 60
    h: 50
""",
        encoding="utf-8",
    )
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.circle(image, (80, 55), 18, (0, 190, 255), -1)

    result = detect_seat_role(image, config_path=config)

    assert result.role == "dealer"
    assert result.expected_total == 21


def test_detect_seat_role_marks_missing_badge_as_player(tmp_path):
    config = tmp_path / "screen.yaml"
    config.write_text(
        """
screen:
  width: 200
  height: 120
regions:
  my_dealer_marker:
    x: 50
    y: 30
    w: 60
    h: 50
""",
        encoding="utf-8",
    )
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    result = detect_seat_role(image, config_path=config)

    assert result.role == "player"
    assert result.expected_total == 20
