"""Chenzhou Zipai application package."""

from __future__ import annotations

import sys

from . import game_logging


# Preserve the old qualified import without leaving a top-level ``logging``
# package that can shadow Python's standard library when the app root is on
# sys.path.
sys.modules.setdefault(f"{__name__}.logging", game_logging)
