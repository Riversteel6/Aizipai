"""Template image loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


APPEARANCE_SUFFIXES = {"normal", "selected", "triple_stack"}
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


@dataclass(frozen=True)
class TemplateImage:
    name: str
    path: Path
    image: np.ndarray
    label: str = ""
    appearance: str = "normal"

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def parse_template_stem(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    if parts and parts[-1].isdigit() and len(parts[-1]) == 3:
        parts = parts[:-1]
    appearance = "normal"
    if len(parts) >= 3 and "_".join(parts[-2:]) in APPEARANCE_SUFFIXES:
        appearance = "_".join(parts[-2:])
        parts = parts[:-2]
    elif len(parts) >= 2 and parts[-1] in APPEARANCE_SUFFIXES:
        appearance = parts[-1]
        parts = parts[:-1]
    for suffix in ("reference", "template"):
        if len(parts) >= 2 and parts[-1] == suffix:
            parts = parts[:-1]
            break
    label = "_".join(parts) if parts else stem
    return label, appearance


def template_name(path: Path) -> str:
    stem = path.stem
    label, appearance = parse_template_stem(stem)
    return label if appearance == "normal" else f"{label}_{appearance}"


def resolve_template_dir(directory: str | Path) -> Path:
    template_dir = Path(directory)
    if template_dir.is_absolute() or template_dir.exists():
        return template_dir
    for base in (ROOT, WORKSPACE):
        candidate = base / template_dir
        if candidate.exists():
            return candidate
    return template_dir


def load_templates(directory: str | Path, *, grayscale: bool = True) -> list[TemplateImage]:
    template_dir = resolve_template_dir(directory)
    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")
    return list(_load_templates_cached(str(template_dir.resolve()), grayscale))


def load_templates_from_dirs(
    directories: str | Path | tuple[str | Path, ...] | list[str | Path],
    *,
    grayscale: bool = True,
    require_any: bool = True,
) -> list[TemplateImage]:
    if isinstance(directories, (str, Path)):
        directories = (directories,)
    templates: list[TemplateImage] = []
    resolved_dirs: list[Path] = []
    for directory in directories:
        path = resolve_template_dir(directory)
        resolved_dirs.append(path)
        if path.exists():
            templates.extend(load_templates(path, grayscale=grayscale))
    if require_any and not templates:
        raise FileNotFoundError(f"No templates found in: {resolved_dirs}")
    return templates


@lru_cache(maxsize=32)
def _load_templates_cached(directory: str, grayscale: bool) -> tuple[TemplateImage, ...]:
    template_dir = Path(directory)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    templates: list[TemplateImage] = []
    for path in sorted(template_dir.glob("*.png")):
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, flag)
        if image is None:
            raise ValueError(f"Could not read template image: {path}")
        label, appearance = parse_template_stem(path.stem)
        templates.append(
            TemplateImage(
                name=template_name(path),
                label=label,
                appearance=appearance,
                path=path,
                image=image,
            )
        )
    return tuple(templates)
