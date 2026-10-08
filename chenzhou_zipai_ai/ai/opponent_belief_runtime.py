"""Runtime adapter from public opponent evidence to rollout-policy factories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from ai.full_game_simulator import PublicView, SimulationPolicy
from ai.opponent_belief import (
    PublicOpponentBeliefModel,
    public_opponent_features,
    smooth_weighted_profile_schedule,
)
from ai.opponent_proxy import (
    FastAggressiveMeldProxyPolicy,
    FastDefensiveSearchProxyPolicy,
    FastInformationSetProxyPolicy,
    FastRedBlackSearchProxyPolicy,
)


DEFAULT_MODEL_REPORT = Path(__file__).resolve().parents[1] / "models" / "opponent_belief.json"

PUBLIC_PROFILE_FACTORIES: Mapping[
    str,
    Callable[[], SimulationPolicy],
] = {
    "aggressive_meld": FastAggressiveMeldProxyPolicy,
    "defensive_search": FastDefensiveSearchProxyPolicy,
    "independent_balanced": IndependentFastRolloutPolicy,
    "independent_denial": IndependentFastDenialPolicy,
    "independent_pressure": IndependentFastPressurePolicy,
    "information_set_search": FastInformationSetProxyPolicy,
    "red_black_search": FastRedBlackSearchProxyPolicy,
}


@dataclass(frozen=True)
class RuntimeOpponentBelief:
    opponent_seat: int
    public_actions: int
    posterior: dict[str, float]
    schedule: tuple[str, ...]
    model_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_seat": self.opponent_seat,
            "public_actions": self.public_actions,
            "posterior": {
                name: round(value, 8)
                for name, value in self.posterior.items()
            },
            "schedule": list(self.schedule),
            "model_source": self.model_source,
            "hidden_information_used": False,
        }


@lru_cache(maxsize=4)
def load_runtime_opponent_belief_model(
    path: str | Path = DEFAULT_MODEL_REPORT,
) -> PublicOpponentBeliefModel:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    model_payload = payload.get("final_model", payload)
    return PublicOpponentBeliefModel.from_dict(model_payload)


def runtime_opponent_rollout_schedule(
    view: PublicView,
    rules: Mapping[str, Any],
    *,
    slots: int,
    model_path: str | Path = DEFAULT_MODEL_REPORT,
) -> tuple[tuple[Callable[[], SimulationPolicy], ...], RuntimeOpponentBelief]:
    evidence = public_opponent_features(view, rules)
    model = load_runtime_opponent_belief_model(model_path)
    posterior = model.posterior(
        evidence.features,
        public_actions=evidence.public_actions,
    )
    unknown = set(posterior) - set(PUBLIC_PROFILE_FACTORIES)
    if unknown:
        raise ValueError(
            "unmapped_public_opponent_profiles:"
            + ",".join(sorted(unknown))
        )
    schedule = smooth_weighted_profile_schedule(
        posterior,
        slots=max(1, int(slots)),
    )
    factories = tuple(PUBLIC_PROFILE_FACTORIES[name] for name in schedule)
    return factories, RuntimeOpponentBelief(
        opponent_seat=evidence.opponent_seat,
        public_actions=evidence.public_actions,
        posterior=posterior,
        schedule=schedule,
        model_source=str(Path(model_path).resolve()),
    )


__all__ = [
    "DEFAULT_MODEL_REPORT",
    "PUBLIC_PROFILE_FACTORIES",
    "RuntimeOpponentBelief",
    "load_runtime_opponent_belief_model",
    "runtime_opponent_rollout_schedule",
]
