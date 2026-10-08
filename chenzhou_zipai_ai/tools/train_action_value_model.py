"""Train and evaluate the lightweight full-game action-value model."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.action_value_model import train_action_value_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    rows = _read_rows(args.input)
    started = time.perf_counter()
    model, report = train_action_value_model(rows, seed=args.seed)
    training_seconds = time.perf_counter() - started
    benchmark_rows = rows[: min(2_000, len(rows))]
    benchmark_started = time.perf_counter()
    repetitions = 10
    for _ in range(repetitions):
        model.predict_rows(benchmark_rows)
    elapsed = time.perf_counter() - benchmark_started
    report["training_seconds"] = round(training_seconds, 6)
    report["inference"] = {
        "rows": len(benchmark_rows) * repetitions,
        "microseconds_per_action": round(
            elapsed * 1_000_000 / max(1, len(benchmark_rows) * repetitions),
            3,
        ),
    }
    model.save(
        args.model_output,
        metadata={
            "schema_version": report["schema_version"],
            "selected_config": report["selected_config"],
            "source": str(args.input),
        },
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "model_output": str(args.model_output),
                "selected_config": report["selected_config"],
                "accepted_as_search_prior": report[
                    "accepted_as_search_prior"
                ],
                "validation": report["metrics"]["validation"],
                "test": report["metrics"]["test"],
                "training_seconds": report["training_seconds"],
                "inference": report["inference"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


if __name__ == "__main__":
    raise SystemExit(main())
