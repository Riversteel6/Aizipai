"""Run frozen V9 Wang-mode data collection without changing the strategy."""

from __future__ import annotations

from tools.run_v81_training import main as run_frozen_training


def main() -> int:
    return run_frozen_training(
        product_label="V9",
        file_label="v9",
        schema_version="v9-custom-training-v1",
        fixed_wildcard="on",
    )


if __name__ == "__main__":
    raise SystemExit(main())
