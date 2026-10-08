import pytest

from tools.merge_discard_calibration_reports import (
    collect_calibration_rows,
)


def _report(view_id: str, *, benchmark: str = "benchmark.json") -> dict:
    return {
        "ok": True,
        "schema_version": "discard-validator-calibration-v1",
        "benchmark_report": benchmark,
        "evidence_inputs": [f"{view_id}.jsonl.gz"],
        "rows": [{"public_view_id": view_id}],
    }


def test_collect_calibration_rows_merges_disjoint_shards() -> None:
    rows, evidence, benchmark = collect_calibration_rows(
        [_report("b"), _report("a")],
        expected_states=2,
    )

    assert [row["public_view_id"] for row in rows] == ["a", "b"]
    assert evidence == ["b.jsonl.gz", "a.jsonl.gz"]
    assert benchmark == "benchmark.json"


def test_collect_calibration_rows_rejects_duplicate_views() -> None:
    with pytest.raises(
        ValueError,
        match="discard_calibration_merge_duplicate_view",
    ):
        collect_calibration_rows([_report("a"), _report("a")])


def test_collect_calibration_rows_rejects_benchmark_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="discard_calibration_merge_benchmark_mismatch",
    ):
        collect_calibration_rows(
            [
                _report("a", benchmark="one.json"),
                _report("b", benchmark="two.json"),
            ]
        )
