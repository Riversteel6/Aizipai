"""Tests for no-click real replay sample collection."""

from pathlib import Path

from tools.collect_real_replay_samples import collect_real_replay_samples


def test_collect_real_replay_samples_never_enables_execution(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("tools.collect_real_replay_samples.time.sleep", lambda _: None)

    def fake_runner(**kwargs):
        calls.append(kwargs)
        index = len(calls)
        return {
            "flow": {"state": "play"},
            "decision_id": f"decision_{index:06d}",
            "decision": {"action": "discard", "label": "九"},
            "action_plan": {"ready": True},
            "executed": False,
        }

    def fake_verifier(logs_root: Path, *, min_decisions: int, include_fixtures: bool):
        return {
            "ok": True,
            "logs_root": str(logs_root),
            "decisions_checked": min_decisions,
            "decisions_matched": min_decisions,
            "min_decisions": min_decisions,
            "include_fixtures": include_fixtures,
            "errors": [],
            "mismatches": [],
        }

    report = collect_real_replay_samples(
        logs_root=tmp_path / "logs",
        target_decisions=3,
        max_steps=5,
        delay_seconds=0,
        runner=fake_runner,
        verifier=fake_verifier,
    )

    assert report["ok"]
    assert report["decisions_collected"] == 3
    assert report["execute_play_actions"] is False
    assert report["tap_executed"] is False
    assert all(call["execute_play_actions"] is False for call in calls)
    assert all(call["execute_settlement_ready"] is False for call in calls)
    assert (tmp_path / "logs" / "real_replay_sample_report.json").exists()


def test_collect_real_replay_samples_reports_failed_replay_verification(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.collect_real_replay_samples.time.sleep", lambda _: None)

    def fake_runner(**kwargs):
        return {
            "flow": {"state": "play"},
            "decision_id": "decision_000001",
            "decision": {"action": "discard", "label": "九"},
            "action_plan": {"ready": True},
            "executed": False,
        }

    def fake_verifier(logs_root: Path, *, min_decisions: int, include_fixtures: bool):
        return {
            "ok": False,
            "logs_root": str(logs_root),
            "decisions_checked": 1,
            "decisions_matched": 0,
            "min_decisions": min_decisions,
            "include_fixtures": include_fixtures,
            "errors": ["sample mismatch"],
            "mismatches": [],
        }

    report = collect_real_replay_samples(
        logs_root=tmp_path / "logs",
        target_decisions=2,
        max_steps=2,
        delay_seconds=0,
        runner=fake_runner,
        verifier=fake_verifier,
    )

    assert not report["ok"]
    assert report["decisions_collected"] == 2
    assert report["verification"]["errors"] == ["sample mismatch"]
