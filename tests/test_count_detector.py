from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vision.count_detector import _classify_digit


def test_classify_digit_treats_narrow_remaining_count_glyph_as_one():
    mask = np.zeros((70, 70), dtype=np.uint8)
    mask[14:52, 29:40] = 255

    assert _classify_digit(mask, (29, 14, 11, 38)) == "1"
