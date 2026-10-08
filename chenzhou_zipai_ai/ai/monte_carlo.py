"""Monte Carlo simulation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ai.evaluator import evaluate_discard
from ai.features import quick_potential
from engine.deck import expanded_deck, remaining_counts


@dataclass(frozen=True)
class SimulationResult:
    label: str
    simulations: int
    avg_xi_after_draw: float
    improve_rate: float
    avg_score: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "simulations": self.simulations,
            "avg_xi_after_draw": round(self.avg_xi_after_draw, 3),
            "improve_rate": round(self.improve_rate, 3),
            "avg_score": round(self.avg_score, 3),
        }


def simulate_discards(
    hand: list[str],
    *,
    memory: dict | None = None,
    meld_groups: dict | None = None,
    simulations: int = 200,
    seed: int = 7,
    config_path: str = "config/rules.yaml",
    rules: dict[str, Any] | None = None,
) -> list[SimulationResult]:
    rng = random.Random(seed)
    remaining = remaining_counts(
        hand=hand,
        memory=memory,
        meld_groups=meld_groups,
        config_path=config_path,
        rules=rules,
    )
    deck = expanded_deck(remaining)
    labels = list(dict.fromkeys(hand))
    results: list[SimulationResult] = []
    for label in labels:
        if label not in hand:
            continue
        after = list(hand)
        after.remove(label)
        base_xi = quick_potential(after)
        xi_total = 0
        improvements = 0
        for _ in range(max(1, simulations)):
            draw = rng.choice(deck) if deck else None
            sampled = after + ([draw] if draw else [])
            xi = quick_potential(sampled)
            xi_total += xi
            if xi > base_xi:
                improvements += 1
        heuristic = evaluate_discard(hand, label, memory=memory, config_path=config_path)
        avg_xi = xi_total / max(1, simulations)
        results.append(
            SimulationResult(
                label=label,
                simulations=max(1, simulations),
                avg_xi_after_draw=avg_xi,
                improve_rate=improvements / max(1, simulations),
                avg_score=heuristic.score + avg_xi * 5,
            )
        )
    return sorted(results, key=lambda item: item.avg_score, reverse=True)
