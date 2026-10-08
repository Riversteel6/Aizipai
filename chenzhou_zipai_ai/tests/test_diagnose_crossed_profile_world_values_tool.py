from ai.ismcts import PairedCandidateOutcome, PairedWorldOutcome
from tools.diagnose_crossed_profile_world_values import (
    crossed_reward_batches,
    crossed_world_identity_mismatches,
    integrity_gate_failures,
)


def _world(fingerprint: str, rewards: dict[str, float]) -> PairedWorldOutcome:
    return PairedWorldOutcome(
        world_index=0,
        world_fingerprint=fingerprint,
        rollout_seed=7,
        opponent_policy="test",
        outcomes=tuple(
            PairedCandidateOutcome(
                candidate_key=label,
                reward=reward,
                winner=None,
                score=0.0,
                total_xi=0,
                turns=1,
                reason="draw",
            )
            for label, reward in rewards.items()
        ),
    )


def test_crossed_reward_batches_weights_same_world_profiles() -> None:
    worlds = {
        "a": (_world("same", {"x": 1.0, "y": 3.0}),),
        "b": (_world("same", {"x": 5.0, "y": -1.0}),),
    }

    batches = crossed_reward_batches(
        worlds,
        weights={"a": 0.25, "b": 0.75},
        labels=("x", "y"),
    )

    assert batches == [{"x": 4.0, "y": 0.0}]
    assert crossed_world_identity_mismatches(worlds) == 0


def test_crossed_world_identity_mismatches_detects_reassignment() -> None:
    worlds = {
        "a": (_world("first", {"x": 1.0}),),
        "b": (_world("second", {"x": 1.0}),),
    }

    assert crossed_world_identity_mismatches(worlds) == 1


def test_integrity_gate_failures_checks_frozen_dimensions() -> None:
    failures = integrity_gate_failures(
        [{"legal_labels": ["x"]}],
        profiles=("a",),
        stages=({"id": "coverage"},),
        health_errors=1,
        world_identity_mismatches=1,
        gate={
            "required_states": 1,
            "required_profiles": 1,
            "required_stages": 1,
            "maximum_health_errors": 0,
            "maximum_world_identity_mismatches": 0,
            "require_all_legal_actions": True,
            "require_all_crossed_worlds": True,
        },
    )

    assert failures == ["health_errors", "world_identity_mismatches"]
