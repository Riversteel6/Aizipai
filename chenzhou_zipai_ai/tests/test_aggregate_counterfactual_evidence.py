import json

import pytest

from tools.aggregate_counterfactual_evidence import (
    _resolve_action_keys,
)


def test_action_keys_can_be_derived_without_command_line_card_labels(
    tmp_path,
):
    report = tmp_path / "benchmark.json"
    report.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "public_view_id": "state-1",
                        "source_selected_label": "\u4e00",
                        "selected_label": "\u56db",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert _resolve_action_keys(
        preferred_key=None,
        candidate_key=None,
        benchmark_report=report,
        public_view_id="state-1",
    ) == ("\u4e00", "\u56db")


def test_action_key_derivation_rejects_ambiguous_input(tmp_path):
    report = tmp_path / "benchmark.json"
    report.write_text(
        json.dumps({"rows": []}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="expected_one_benchmark_row",
    ):
        _resolve_action_keys(
            preferred_key=None,
            candidate_key=None,
            benchmark_report=report,
            public_view_id="missing",
        )
