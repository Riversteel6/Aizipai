"""Flatten counterfactual evidence into a leakage-safe value corpus."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Iterator


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.value_corpus import build_value_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--split-salt", default="aizipai-value-v1")
    args = parser.parse_args()
    report = build_value_corpus(
        (
            row
            for path in args.input
            for row in _read_jsonl(path)
        ),
        split_salt=args.split_salt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        for row in report.pop("rows"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


def _read_jsonl(path: Path) -> Iterator[dict]:
    if path.suffix.lower() == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
