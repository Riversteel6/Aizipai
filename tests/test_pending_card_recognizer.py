from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "chenzhou_zipai_ai"
SRC = Path(__file__).resolve().parents[1] / "src"
for path in (PROJECT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vision import pending_card_recognizer
from vision.card_classifier import CardClassification
from vision.hand_recognizer import HandCard
from vision.pending_card_recognizer import PendingCard, _load_pending_templates, recognize_pending_cards
from tools import inspect_state as inspect_state_module


def _write_pending_config(path: Path) -> None:
    path.write_text(
        """
screen:
  width: 100
  height: 100
regions:
  pending_action_card:
    x: 10
    y: 20
    w: 20
    h: 30
  opponent_pending_card:
    x: 60
    y: 20
    w: 20
    h: 30
""".strip(),
        encoding="utf-8",
    )


def test_recognize_pending_cards_keeps_interactive_cards_out_of_discards(monkeypatch, tmp_path):
    config_path = tmp_path / "screen.yaml"
    _write_pending_config(config_path)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:50, 10:30] = 255
    image[20:50, 60:80] = 255
    calls = {"classify": 0}
    loaded_templates = [object()]

    monkeypatch.setattr(pending_card_recognizer, "load_templates", lambda *args, **kwargs: loaded_templates)

    def fake_classify_card_crop(crop, *, threshold, templates):
        calls["classify"] += 1
        assert templates == loaded_templates
        if calls["classify"] == 1:
            return CardClassification("二", 0.88, "二_001.png", True)
        return CardClassification("九", 0.40, "九_001.png", False)

    monkeypatch.setattr(pending_card_recognizer, "classify_card_crop", fake_classify_card_crop)

    result = recognize_pending_cards(
        image,
        config_path=config_path,
        template_dirs=(tmp_path,),
        threshold=0.72,
    )

    assert result["pending_action_card"].name == "二"
    assert result["pending_action_card"].center == (20, 35)
    assert result["opponent_pending_card"] is None


def test_recognize_pending_tall_surface_classifies_only_upright_top_face(monkeypatch, tmp_path):
    config_path = tmp_path / "screen.yaml"
    config_path.write_text(
        """
screen:
  width: 240
  height: 120
regions:
  opponent_pending_card:
    x: 10
    y: 5
    w: 60
    h: 110
""".strip(),
        encoding="utf-8",
    )
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    image[8:113, 18:58] = 255
    seen = {}

    monkeypatch.setattr(pending_card_recognizer, "load_templates", lambda *args, **kwargs: [object()])

    def fake_classify_card_crop(crop, *, threshold, templates):
        seen.setdefault("shapes", []).append(crop.shape[:2])
        seen.setdefault("thresholds", []).append(threshold)
        return CardClassification("玖", 0.76, "玖_001.png", True)

    monkeypatch.setattr(pending_card_recognizer, "classify_card_crop", fake_classify_card_crop)

    result = recognize_pending_cards(
        image,
        config_path=config_path,
        template_dirs=(tmp_path,),
        threshold=0.45,
        region_names=("opponent_pending_card",),
    )

    assert seen == {
        "shapes": [(42, 40), (46, 40)],
        "thresholds": [0.0, 0.0],
    }
    assert result["opponent_pending_card"].name == "玖"
    assert result["opponent_pending_card"].center == (38, 60)


def test_recognize_pending_tall_surface_rejects_disagreeing_crop_scales(monkeypatch, tmp_path):
    config_path = tmp_path / "screen.yaml"
    config_path.write_text(
        """
screen:
  width: 240
  height: 120
regions:
  opponent_pending_card:
    x: 10
    y: 5
    w: 60
    h: 110
""".strip(),
        encoding="utf-8",
    )
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    image[8:113, 18:58] = 255
    calls = 0

    monkeypatch.setattr(pending_card_recognizer, "load_templates", lambda *args, **kwargs: [object()])

    def fake_classify_card_crop(_crop, *, threshold, templates):
        nonlocal calls
        calls += 1
        label = "玖" if calls == 1 else "五"
        return CardClassification(label, 0.90, f"{label}_001.png", True)

    monkeypatch.setattr(pending_card_recognizer, "classify_card_crop", fake_classify_card_crop)

    result = recognize_pending_cards(
        image,
        config_path=config_path,
        template_dirs=(tmp_path,),
        threshold=0.45,
        region_names=("opponent_pending_card",),
    )

    assert result["opponent_pending_card"] is None


def test_recognize_pending_card_rejects_region_background_without_a_card_surface(monkeypatch, tmp_path):
    config_path = tmp_path / "screen.yaml"
    _write_pending_config(config_path)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    monkeypatch.setattr(pending_card_recognizer, "load_templates", lambda *args, **kwargs: [object()])

    def unexpected_classification(*_args, **_kwargs):
        raise AssertionError("background-only pending region reached card classification")

    monkeypatch.setattr(pending_card_recognizer, "classify_card_crop", unexpected_classification)

    result = recognize_pending_cards(
        image,
        config_path=config_path,
        template_dirs=(tmp_path,),
    )

    assert result == {"pending_action_card": None, "opponent_pending_card": None}


def test_pending_card_templates_resolve_nested_default_dirs_from_workspace_root():
    templates = _load_pending_templates(("data/templates/discard_auto", "data/templates/hand_auto"))

    assert templates
    assert all(template.path.exists() for template in templates)


def test_inspect_state_exports_pending_cards_into_state(monkeypatch, tmp_path):
    pending = PendingCard("pending_action_card", "二", 0.9, 10, 20, 20, 30, "二_001.png")
    opponent = PendingCard("opponent_pending_card", "七", 0.86, 60, 20, 20, 30, "七_001.png")

    monkeypatch.setattr(inspect_state_module, "read_image", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(
        inspect_state_module,
        "recognize_hand",
        lambda *args, **kwargs: [HandCard("一", 0.99, 1, 2, 10, 12, "一_001.png")],
    )
    monkeypatch.setattr(inspect_state_module, "detect_buttons", lambda *args, **kwargs: [])
    monkeypatch.setattr(inspect_state_module, "detect_discard_button", lambda *args, **kwargs: None)
    monkeypatch.setattr(inspect_state_module, "detect_option_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(inspect_state_module, "recognize_options", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        inspect_state_module,
        "recognize_pending_cards",
        lambda *args, **kwargs: {
            "pending_action_card": pending,
            "opponent_pending_card": opponent,
        },
    )
    monkeypatch.setattr(inspect_state_module, "detect_remaining_deck_count", lambda *args, **kwargs: 16)
    monkeypatch.setattr(
        inspect_state_module,
        "recognize_discards",
        lambda *args, **kwargs: {"opponent_discards": [], "my_discards": []},
    )
    monkeypatch.setattr(inspect_state_module, "recognize_meld_groups", lambda *args, **kwargs: {"my_melds": []})

    result = inspect_state_module.inspect_screenshot(tmp_path / "frame.png")

    assert result["pending_action_card"]["name"] == "二"
    assert result["opponent_pending_card"]["name"] == "七"
    assert result["metadata"]["pending_action_card"]["name"] == "二"
    assert result["metadata"]["opponent_pending_card"]["name"] == "七"
