"""Tests for template loading contracts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision.hand_recognizer import _card_name
from vision.template_loader import load_templates, load_templates_from_dirs, parse_template_stem, resolve_template_dir, template_name


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 10, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_parse_template_stem_groups_appearance_variants_by_label():
    assert parse_template_stem("二_001") == ("二", "normal")
    assert parse_template_stem("二_selected_001") == ("二", "selected")
    assert parse_template_stem("二_triple_stack_002") == ("二", "triple_stack")
    assert parse_template_stem("chi_reference") == ("chi", "normal")


def test_template_name_preserves_appearance_for_audit_but_card_name_groups_label():
    path = Path("card_selected_001.png")

    assert template_name(path) == "card_selected"


def test_load_templates_exposes_label_and_appearance(tmp_path):
    _write_png(tmp_path / "card_001.png")
    _write_png(tmp_path / "card_selected_001.png")
    _write_png(tmp_path / "card_triple_stack_001.png")

    templates = load_templates(tmp_path, grayscale=False)

    assert [item.label for item in templates] == ["card", "card", "card"]
    assert [item.appearance for item in templates] == ["normal", "selected", "triple_stack"]
    assert [_card_name(item) for item in templates] == ["card", "card", "card"]


def test_load_templates_from_dirs_skips_missing_dirs(tmp_path):
    _write_png(tmp_path / "card_001.png")

    templates = load_templates_from_dirs((tmp_path / "missing", tmp_path), grayscale=False)

    assert [item.label for item in templates] == ["card"]


def test_template_loader_resolves_nested_default_template_dir_from_workspace_root():
    resolved = resolve_template_dir("data/templates/hand_auto")

    assert resolved.exists()
    assert resolved.name == "hand_auto"
