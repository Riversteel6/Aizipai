"""Repeatable strategy latency benchmarks over legal deck samples."""

from __future__ import annotations

import random
import time
from statistics import mean, median
from typing import Any

from ai.pro_brain import POLICY_VERSION, choose_action
from engine.deck import expanded_deck, full_deck_counts
from engine.rules import rules_for_room


def run_strategy_latency_benchmark(
    *,
    samples: int = 20,
    seed: int = 20260726,
    wildcard_enabled: bool,
    max_allowed_ms: float = 8_000.0,
    players: int = 3,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if players not in {2, 3}:
        raise ValueError("players must be 2 or 3")
    rules = rules_for_room(
        wildcard_enabled=wildcard_enabled,
        players=players,
    )
    full_deck = expanded_deck(full_deck_counts(rules=rules))
    rows: list[dict[str, Any]] = []
    for index in range(samples):
        deck = list(full_deck)
        random.Random(seed + index).shuffle(deck)
        hand = deck[:21]
        state = {
            "context_id": f"latency_{seed}_{index:04d}",
            "hand": hand,
            "legal_actions": [{"type": "DISCARD"}],
            "remaining_deck_count": max(0, len(deck) - (players * 20 + 1)),
            "room_players": players,
            "memory": {
                "my_discards": [],
                "opponent_discards": [],
                "opponent_meld_groups": [],
            },
        }
        started = time.perf_counter()
        decision = choose_action(state, rules=rules)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "index": index,
                "seed": seed + index,
                "elapsed_ms": round(elapsed_ms, 3),
                "action": decision.selected_action,
                "label": decision.selected_label,
                "candidate_stage": decision.candidate_stage,
                "safety_flags": list(decision.safety_flags),
            }
        )
    latencies = sorted(float(row["elapsed_ms"]) for row in rows)
    p95 = _nearest_rank(latencies, 0.95)
    maximum = max(latencies)
    return {
        "ok": maximum <= max_allowed_ms and not any(row["safety_flags"] for row in rows),
        "policy_version": POLICY_VERSION,
        "wildcard_enabled": wildcard_enabled,
        "players": players,
        "samples": samples,
        "seed": seed,
        "max_allowed_ms": max_allowed_ms,
        "mean_ms": round(mean(latencies), 3),
        "median_ms": round(median(latencies), 3),
        "p95_ms": round(p95, 3),
        "max_ms": round(maximum, 3),
        "rows": rows,
    }


def _nearest_rank(values: list[float], percentile: float) -> float:
    index = max(0, min(len(values) - 1, int((len(values) * percentile) + 0.999999) - 1))
    return values[index]


__all__ = ["run_strategy_latency_benchmark"]
