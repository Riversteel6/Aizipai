"""Balanced train/holdout selection for independent counter-strategy opponents."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from statistics import mean
from typing import Any, Iterable, Sequence

from ai.full_game_simulator import FullGameSimulator, SimulationPolicy
from ai.opponent_league import (
    ProductDecisionKernelSimulationPolicy,
    _FROZEN_PRODUCT_CANDIDATE,
    create_policy,
)
from audit.independent_opponent import (
    IndependentBalancedPolicy,
    IndependentPolicyWeights,
)
from engine.rules import rules_for_room


@dataclass(frozen=True)
class ExploitProfile:
    name: str
    weights: IndependentPolicyWeights

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": asdict(self.weights),
        }


@dataclass(frozen=True)
class ExploitJob:
    profile: ExploitProfile
    target: str
    players: int
    wildcard_enabled: bool
    seed: int
    candidate_seat: int
    dealer: int
    split: str


class TunableIndependentPolicy(IndependentBalancedPolicy):
    def __init__(self, profile: ExploitProfile) -> None:
        self.profile = profile
        self.weights = profile.weights
        self.name = f"independent_exploit_{profile.name}"


DEFAULT_EXPLOIT_PROFILES: tuple[ExploitProfile, ...] = (
    ExploitProfile("balanced", IndependentPolicyWeights()),
    ExploitProfile(
        "claim_pressure",
        replace(
            IndependentPolicyWeights(),
            shape=10.5,
            discard_danger=1.3,
            peng_margin=-6.0,
            chi_margin=0.0,
        ),
    ),
    ExploitProfile(
        "concealed_denial",
        replace(
            IndependentPolicyWeights(),
            hu_out=160.0,
            hu_score=30.0,
            route=10.0,
            discard_danger=3.0,
            peng_margin=42.0,
            chi_margin=58.0,
        ),
    ),
    ExploitProfile(
        "score_pressure",
        replace(
            IndependentPolicyWeights(),
            hu_out=135.0,
            hu_score=38.0,
            exposed_xi=62.0,
            route=11.0,
            peng_margin=4.0,
            chi_margin=8.0,
        ),
    ),
    ExploitProfile(
        "shape_mobility",
        replace(
            IndependentPolicyWeights(),
            partition_out=46.0,
            shape=13.0,
            discard_danger=1.4,
            peng_margin=10.0,
            chi_margin=12.0,
        ),
    ),
    ExploitProfile(
        "late_defense",
        replace(
            IndependentPolicyWeights(),
            hu_out=170.0,
            partition_out=44.0,
            discard_danger=4.0,
            peng_margin=24.0,
            chi_margin=34.0,
        ),
    ),
)


def build_exploit_jobs(
    *,
    profiles: Sequence[ExploitProfile],
    target: str,
    deals_per_mode: int,
    seed: int,
    split: str,
    modes: Sequence[tuple[int, bool]] | None = None,
) -> list[ExploitJob]:
    if deals_per_mode < 1:
        raise ValueError("deals_per_mode_must_be_positive")
    jobs: list[ExploitJob] = []
    resolved_modes = tuple(modes or ((2, False), (2, True), (3, False), (3, True)))
    for players, _wildcard_enabled in resolved_modes:
        if players not in {2, 3}:
            raise ValueError(f"unsupported_exploit_mode_players:{players}")
    for mode_index, (players, wildcard_enabled) in enumerate(resolved_modes):
        for deal_index in range(deals_per_mode):
            game_seed = seed + mode_index * 100_000 + deal_index
            for candidate_seat in range(players):
                for dealer in range(players):
                    for profile in profiles:
                        jobs.append(
                            ExploitJob(
                                profile=profile,
                                target=target,
                                players=players,
                                wildcard_enabled=wildcard_enabled,
                                seed=game_seed,
                                candidate_seat=candidate_seat,
                                dealer=dealer,
                                split=split,
                            )
                        )
    return jobs


def train_exploit_opponent(
    *,
    profiles: Sequence[ExploitProfile] = DEFAULT_EXPLOIT_PROFILES,
    target: str = "professional_brain",
    train_deals_per_mode: int,
    holdout_deals_per_mode: int,
    train_seed: int,
    holdout_seed: int,
    workers: int = 1,
    modes: Sequence[tuple[int, bool]] | None = None,
) -> dict[str, Any]:
    resolved_profiles = tuple(profiles)
    if not resolved_profiles:
        raise ValueError("no_exploit_profiles")
    if len({profile.name for profile in resolved_profiles}) != len(resolved_profiles):
        raise ValueError("duplicate_exploit_profile_name")
    train_jobs = build_exploit_jobs(
        profiles=resolved_profiles,
        target=target,
        deals_per_mode=train_deals_per_mode,
        seed=train_seed,
        split="train",
        modes=modes,
    )
    train_seeds = _job_seed_keys(train_jobs)
    holdout_probe = build_exploit_jobs(
        profiles=(resolved_profiles[0],),
        target=target,
        deals_per_mode=holdout_deals_per_mode,
        seed=holdout_seed,
        split="holdout",
        modes=modes,
    )
    holdout_seeds = _job_seed_keys(holdout_probe)
    overlap = train_seeds.intersection(holdout_seeds)
    if overlap:
        raise ValueError(f"train_holdout_seed_overlap:{sorted(overlap)[:5]}")

    train_rows = _run_jobs(train_jobs, workers=workers)
    training_by_profile = {
        profile.name: summarize_exploit_rows(
            [row for row in train_rows if row["profile"] == profile.name]
        )
        for profile in resolved_profiles
    }
    selected = max(
        resolved_profiles,
        key=lambda profile: _selection_key(training_by_profile[profile.name]),
    )
    holdout_jobs = build_exploit_jobs(
        profiles=(selected,),
        target=target,
        deals_per_mode=holdout_deals_per_mode,
        seed=holdout_seed,
        split="holdout",
        modes=modes,
    )
    holdout_rows = _run_jobs(holdout_jobs, workers=workers)
    return {
        "ok": all(
            row["invariant_violations"] == 0 and row["coverage_failures"] == 0
            for row in (*train_rows, *holdout_rows)
        ),
        "target": target,
        "selection_rule": [
            "maximize_worst_mode_win_share_all",
            "maximize_aggregate_win_share_all",
            "maximize_worst_mode_mean_score",
            "maximize_aggregate_mean_score",
        ],
        "train_seed": train_seed,
        "holdout_seed": holdout_seed,
        "train_holdout_seed_overlap": 0,
        "train_deals_per_mode": train_deals_per_mode,
        "holdout_deals_per_mode": holdout_deals_per_mode,
        "modes": [
            {"players": players, "wildcard_enabled": wildcard_enabled}
            for players, wildcard_enabled in tuple(
                modes or ((2, False), (2, True), (3, False), (3, True))
            )
        ],
        "profiles": [profile.to_dict() for profile in resolved_profiles],
        "training_by_profile": training_by_profile,
        "selected_profile": selected.to_dict(),
        "holdout": summarize_exploit_rows(holdout_rows),
        "training_rows": train_rows,
        "holdout_rows": holdout_rows,
    }


def summarize_exploit_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_mode_key(row), []).append(row)
    by_mode = {
        mode: _summarize_group(group)
        for mode, group in sorted(grouped.items())
    }
    aggregate = _summarize_group(list(rows))
    return {
        **aggregate,
        "worst_mode_win_share_all": min(
            (summary["candidate_win_share_all"] for summary in by_mode.values()),
            default=0.0,
        ),
        "worst_mode_mean_score": min(
            (summary["mean_candidate_outcome_score"] for summary in by_mode.values()),
            default=0.0,
        ),
        "by_mode": by_mode,
    }


def _run_jobs(jobs: Sequence[ExploitJob], *, workers: int) -> list[dict[str, Any]]:
    if workers > 1:
        with get_context("spawn").Pool(
            processes=min(max(1, workers), len(jobs), os.cpu_count() or 1)
        ) as pool:
            return pool.map(_play_exploit_job, jobs, chunksize=1)
    return [_play_exploit_job(job) for job in jobs]


def _play_exploit_job(job: ExploitJob) -> dict[str, Any]:
    policies: list[SimulationPolicy] = [
        _target_policy(job)
        for _seat in range(job.players)
    ]
    policies[job.candidate_seat] = TunableIndependentPolicy(job.profile)
    simulator = FullGameSimulator(
        policies,
        wildcard_enabled=job.wildcard_enabled,
        dealer=job.dealer,
        rules=rules_for_room(
            wildcard_enabled=job.wildcard_enabled,
            players=job.players,
        ),
    )
    result = simulator.play(job.seed)
    candidate_won = result.winner == job.candidate_seat
    draw = result.winner is None
    outcome_score = (
        result.score
        if candidate_won
        else -result.score
        if result.winner is not None
        else 0.0
    )
    return {
        "profile": job.profile.name,
        "target": job.target,
        "split": job.split,
        "players": job.players,
        "wildcard_enabled": job.wildcard_enabled,
        "seed": job.seed,
        "candidate_seat": job.candidate_seat,
        "dealer": job.dealer,
        "candidate_won": candidate_won,
        "draw": draw,
        "candidate_outcome_score": outcome_score,
        "invariant_violations": len(result.violations),
        "coverage_failures": len(result.coverage_failures),
        "result": result.to_dict(),
    }


def _target_policy(job: ExploitJob) -> SimulationPolicy:
    if job.players == 2 and job.target == _FROZEN_PRODUCT_CANDIDATE:
        return ProductDecisionKernelSimulationPolicy()
    return create_policy(job.target)


def _selection_key(summary: MappingLike) -> tuple[float, float, float, float]:
    return (
        float(summary["worst_mode_win_share_all"]),
        float(summary["candidate_win_share_all"]),
        float(summary["worst_mode_mean_score"]),
        float(summary["mean_candidate_outcome_score"]),
    )


def _mode_key(row: dict[str, Any]) -> str:
    players = int(row["players"])
    wildcard = "wang" if row["wildcard_enabled"] else "no_wang"
    return f"{players}p_{wildcard}"


def _summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(int(row["candidate_won"]) for row in rows)
    draws = sum(int(row["draw"]) for row in rows)
    losses = len(rows) - wins - draws
    low, high = _wilson_interval(wins, len(rows))
    return {
        "games": len(rows),
        "candidate_wins": wins,
        "losses": losses,
        "draws": draws,
        "candidate_win_share_all": round(wins / len(rows), 4) if rows else 0.0,
        "wilson_95_all": [round(low, 4), round(high, 4)],
        "mean_candidate_outcome_score": round(
            mean(float(row["candidate_outcome_score"]) for row in rows),
            4,
        )
        if rows
        else 0.0,
        "invariant_violations": sum(
            int(row["invariant_violations"])
            for row in rows
        ),
        "coverage_failures": sum(
            int(row["coverage_failures"])
            for row in rows
        ),
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _job_seed_keys(jobs: Iterable[ExploitJob]) -> set[tuple[int, bool, int]]:
    return {
        (job.players, job.wildcard_enabled, job.seed)
        for job in jobs
    }


MappingLike = dict[str, Any]


__all__ = [
    "DEFAULT_EXPLOIT_PROFILES",
    "ExploitJob",
    "ExploitProfile",
    "TunableIndependentPolicy",
    "build_exploit_jobs",
    "summarize_exploit_rows",
    "train_exploit_opponent",
]
