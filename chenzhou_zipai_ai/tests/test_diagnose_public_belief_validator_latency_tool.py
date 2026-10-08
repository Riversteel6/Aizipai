from tools.diagnose_public_belief_validator_latency import (
    _forced_factory_map,
    _profile_summaries,
)


class _Policy:
    pass


def test_forced_factory_map_reuses_one_factory_for_every_profile() -> None:
    mapping = _forced_factory_map(("a", "b", "c"), _Policy)

    assert mapping == {"a": _Policy, "b": _Policy, "c": _Policy}


def test_profile_summaries_keep_runtime_and_health_separate() -> None:
    rows = [
        {
            "profile": "a",
            "elapsed_ms": 10.0,
            "validation_complete": True,
            "health_errors": 0,
        },
        {
            "profile": "a",
            "elapsed_ms": 30.0,
            "validation_complete": False,
            "health_errors": 2,
        },
    ]

    summaries = _profile_summaries(rows, ("a", "b"))

    assert summaries["a"] == {
        "runs": 2,
        "complete_runs": 1,
        "health_errors": 2,
        "minimum_elapsed_ms": 10.0,
        "median_elapsed_ms": 30.0,
        "maximum_elapsed_ms": 30.0,
    }
    assert summaries["b"]["runs"] == 0
