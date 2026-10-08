"""Merge disjoint discard-calibration report shards safely."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.calibrate_discard_validation import build_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-states", type=int, default=0)
    args = parser.parse_args()

    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.input
    ]
    rows, evidence_inputs, benchmark_report = collect_calibration_rows(
        reports,
        expected_states=args.expected_states,
    )
    report = build_report(
        rows,
        evidence_inputs=[Path(path) for path in evidence_inputs],
        benchmark_report=Path(benchmark_report),
    )
    report["source_reports"] = [str(path) for path in args.input]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "ok",
                    "states",
                    "complete_online_validations",
                    "incomplete_online_validations",
                    "harmful_overrides",
                    "missed_overrides",
                    "mean_improvement_over_preferred",
                    "mean_regret_to_pooled_best",
                    "confidently_negative_overrides",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def collect_calibration_rows(
    reports: Sequence[Mapping[str, Any]],
    *,
    expected_states: int = 0,
) -> tuple[list[dict[str, Any]], list[str], str]:
    if not reports:
        raise ValueError("discard_calibration_merge_requires_reports")

    benchmark_reports = {
        str(report.get("benchmark_report") or "")
        for report in reports
    }
    if len(benchmark_reports) != 1 or not next(iter(benchmark_reports)):
        raise ValueError("discard_calibration_merge_benchmark_mismatch")
    benchmark_report = next(iter(benchmark_reports))

    rows_by_view: dict[str, dict[str, Any]] = {}
    evidence_inputs: list[str] = []
    for report in reports:
        if report.get("schema_version") != "discard-validator-calibration-v1":
            raise ValueError("discard_calibration_merge_schema_mismatch")
        if not report.get("ok"):
            raise ValueError("discard_calibration_merge_source_not_ok")
        evidence_inputs.extend(
            str(path)
            for path in report.get("evidence_inputs") or ()
        )
        for row in report.get("rows") or ():
            public_view_id = str(row.get("public_view_id") or "")
            if not public_view_id:
                raise ValueError("discard_calibration_merge_view_missing")
            if public_view_id in rows_by_view:
                raise ValueError(
                    "discard_calibration_merge_duplicate_view:"
                    f"{public_view_id}"
                )
            rows_by_view[public_view_id] = dict(row)

    if expected_states and len(rows_by_view) != expected_states:
        raise ValueError(
            "discard_calibration_merge_state_count:"
            f"{len(rows_by_view)}:{expected_states}"
        )
    return (
        [rows_by_view[key] for key in sorted(rows_by_view)],
        evidence_inputs,
        benchmark_report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
