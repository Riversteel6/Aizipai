import json
import subprocess
import sys


def test_v9_training_dry_run_is_frozen_wang_mode(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.run_v9_training",
            "--games",
            "14",
            "--workers",
            "2",
            "--seed",
            "90260823",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    plan = json.loads(completed.stdout)
    assert plan["schema_version"] == "v9-custom-training-v1"
    assert plan["release_id"] == "v9"
    assert plan["wildcard_enabled"] is True
    assert plan["requested_games"] == 14
    assert plan["workers"] == 2
    assert "v9_1v1_wang_14games_" in plan["run_dir"]
