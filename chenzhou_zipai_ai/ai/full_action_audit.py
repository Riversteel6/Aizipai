"""Four-mode coverage audit for the full-action information-set teacher."""

from __future__ import annotations

import random
import time
from collections import Counter
from typing import Any

from ai.full_game_simulator import PublicView
from ai.opponent_league import create_policy
from engine.cards import WILD_LABEL
from engine.deck import expanded_deck, full_deck_counts
from engine.rules import rules_for_room


def audit_full_action_roots(
    *,
    roots_per_mode: int,
    seed: int,
) -> dict[str, Any]:
    if roots_per_mode < 1:
        raise ValueError("roots_per_mode_must_be_positive")
    rows: list[dict[str, Any]] = []
    for mode_index, (players, wildcard_enabled) in enumerate(
        ((2, False), (2, True), (3, False), (3, True))
    ):
        for root_index in range(roots_per_mode):
            root_seed = seed + mode_index * 100_000 + root_index
            view, rules = _initial_public_view(
                root_seed,
                players=players,
                wildcard_enabled=wildcard_enabled,
            )
            legal_labels = _legal_discard_labels(view.hand)
            policy = create_policy("professional_full_action_teacher")
            started = time.perf_counter()
            selected = policy.choose_discard(view, rules)
            wall_ms = (time.perf_counter() - started) * 1000.0
            search = policy.last_search
            if search is None:
                rows.append(
                    {
                        "players": players,
                        "wildcard_enabled": wildcard_enabled,
                        "seed": root_seed,
                        "legal_labels": legal_labels,
                        "selected_label": selected,
                        "search_missing": True,
                        "all_legal_covered": len(legal_labels) <= 1,
                        "every_candidate_visited": len(legal_labels) <= 1,
                        "wall_ms": round(wall_ms, 3),
                    }
                )
                continue
            evaluated = tuple(candidate.label for candidate in search.candidates)
            visited = {
                candidate.label
                for candidate in search.candidates
                if candidate.visits > 0
            }
            rows.append(
                {
                    "players": players,
                    "wildcard_enabled": wildcard_enabled,
                    "seed": root_seed,
                    "legal_labels": legal_labels,
                    "evaluated_labels": evaluated,
                    "missing_labels": sorted(set(legal_labels) - set(evaluated)),
                    "selected_label": selected,
                    "search_missing": False,
                    "all_legal_covered": set(evaluated) == set(legal_labels),
                    "every_candidate_visited": set(visited) == set(legal_labels),
                    "wall_ms": round(wall_ms, 3),
                    "search": search.to_dict(),
                }
            )
    failures = [
        row
        for row in rows
        if (
            not row["all_legal_covered"]
            or not row["every_candidate_visited"]
            or row.get("search", {}).get("rollout_invariant_violations", 0)
            or row.get("search", {}).get("rollout_coverage_failures", 0)
        )
    ]
    return {
        "ok": not failures,
        "seed": seed,
        "roots_per_mode": roots_per_mode,
        "roots": len(rows),
        "all_legal_covered_roots": sum(
            int(row["all_legal_covered"])
            for row in rows
        ),
        "every_candidate_visited_roots": sum(
            int(row["every_candidate_visited"])
            for row in rows
        ),
        "minimum_paired_determinizations": min(
            (
                int(row.get("search", {}).get("paired_determinizations", 0))
                for row in rows
            ),
            default=0,
        ),
        "deadline_interrupted_roots": sum(
            int(
                bool(
                    row.get("search", {}).get(
                        "deadline_interruptions",
                        0,
                    )
                )
            )
            for row in rows
        ),
        "max_wall_ms": round(
            max((float(row["wall_ms"]) for row in rows), default=0.0),
            3,
        ),
        "failures": failures,
        "rows": rows,
    }


def _initial_public_view(
    seed: int,
    *,
    players: int,
    wildcard_enabled: bool,
) -> tuple[PublicView, dict[str, Any]]:
    rules = rules_for_room(
        wildcard_enabled=wildcard_enabled,
        players=players,
    )
    counts = full_deck_counts(rules=rules)
    deck = expanded_deck(counts)
    random.Random(seed).shuffle(deck)
    hand = tuple(deck[:21])
    remaining = counts.copy()
    remaining.subtract(hand)
    return (
        PublicView(
            seat=0,
            hand=hand,
            own_melds=(),
            all_melds=tuple(() for _seat in range(players)),
            discards=tuple(() for _seat in range(players)),
            remaining_counts=tuple(sorted((+remaining).items())),
            stock_count=len(deck) - (players * 20 + 1),
            hand_sizes=(21, *(20 for _seat in range(players - 1))),
        ),
        rules,
    )


def _legal_discard_labels(hand: tuple[str, ...]) -> tuple[str, ...]:
    counts = Counter(hand)
    return tuple(
        sorted(
            label
            for label, amount in counts.items()
            if label != WILD_LABEL and amount < 3
        )
    )


__all__ = ["audit_full_action_roots"]
