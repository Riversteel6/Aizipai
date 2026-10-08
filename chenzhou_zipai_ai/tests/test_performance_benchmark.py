"""Latency benchmark contract tests."""

from ai.performance_benchmark import run_strategy_latency_benchmark


def test_latency_benchmark_reports_version_and_rows():
    report = run_strategy_latency_benchmark(
        samples=1,
        seed=20260729,
        wildcard_enabled=False,
        max_allowed_ms=30_000,
    )

    assert report["ok"]
    assert report["policy_version"] == "v2.2.0"
    assert report["samples"] == 1
    assert len(report["rows"]) == 1
    assert report["max_ms"] > 0
