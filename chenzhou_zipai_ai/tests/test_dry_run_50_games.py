"""Tests for no-device dry-run validation."""

import json

from tools.dry_run_50_games import run_dry_run


def test_dry_run_generates_report_and_has_no_violations(tmp_path):
    result = run_dry_run(count=12, hand_size=10, seed=1, output_dir=tmp_path)

    assert result["ok"], result["violations"]
    assert result["count"] == 12
    assert all(row["self_check_passed"] for row in result["rows"])
    assert (tmp_path / "dry_run_report.json").exists()
    assert (tmp_path / "dry_run_report.md").exists()
    payload = json.loads((tmp_path / "dry_run_report.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
