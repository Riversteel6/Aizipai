from __future__ import annotations

import pytest

from tools.fit_anchored_sequential_belief_fusion import (
    choose_power,
    fused_posterior,
)


PROFILES = ("alpha", "beta")


def test_zero_power_reproduces_anchor_posterior() -> None:
    anchor = {"alpha": 0.73, "beta": 0.27}
    posterior = fused_posterior(
        anchor,
        {"alpha": -9.0, "beta": 11.0},
        power=0.0,
        profiles=PROFILES,
    )

    assert posterior == pytest.approx(anchor, abs=1e-15)


def test_choose_power_obeys_low_evidence_cap() -> None:
    records = [
        {
            "game_id": "game-1",
            "profile": "alpha",
            "evidence_count": 1,
            "baseline_posterior": {"alpha": 0.5, "beta": 0.5},
            "sequence_scores": {"alpha": 2.0, "beta": 0.0},
        }
    ]

    power, calibration = choose_power(
        records,
        profiles=PROFILES,
        grid=(0.0, 0.1, 1.0),
        maximum_low_evidence_posterior=0.56,
    )

    assert power == 0.1
    assert calibration["maximum_feasible_power"] == 0.1
    assert calibration["selected_low_evidence_maximum"] <= 0.56


def test_choose_power_uses_lower_power_as_exact_tie_break() -> None:
    records = [
        {
            "game_id": "game-1",
            "profile": "alpha",
            "evidence_count": 3,
            "baseline_posterior": {"alpha": 0.6, "beta": 0.4},
            "sequence_scores": {"alpha": 0.0, "beta": 0.0},
        }
    ]

    power, _ = choose_power(
        records,
        profiles=PROFILES,
        grid=(0.0, 0.5, 1.0),
        maximum_low_evidence_posterior=0.35,
    )

    assert power == 0.0

