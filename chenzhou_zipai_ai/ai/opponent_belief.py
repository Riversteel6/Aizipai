"""Public-information opponent-style features and belief inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np

from engine.cards import BIG_LABELS, RED_LABELS, SMALL_LABELS
from engine.xi_calculator import meld_xi


PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES = (
    "stock_active_ratio",
    "discard_count_scaled",
    "meld_count_scaled",
    "claim_share",
    "chi_share",
    "peng_share",
    "concealed_share",
    "special_chi_share",
    "mixed_triplet_share",
    "exposed_xi_scaled",
    "meld_red_share",
    "discard_red_share",
    "discard_2710_share",
    "discard_big_share",
    "discard_rank_mean_scaled",
    "discard_rank_std_scaled",
    "discard_unique_share",
    "recent_discard_red_share",
    "opponent_hand_size_scaled",
    "opponent_meld_cards_scaled",
    "public_evidence_count_scaled",
)

_CHI_KINDS = {
    "normal_sequence",
    "special_123",
    "special_2710",
    "mixed_same_rank_triplet",
}
_CONCEALED_KINDS = {"wei", "ti", "pao"}
_SPECIAL_CHI_KINDS = {"special_123", "special_2710"}


@dataclass(frozen=True)
class PublicOpponentEvidence:
    opponent_seat: int
    public_actions: int
    features: tuple[float, ...]


@dataclass(frozen=True)
class PublicOpponentBeliefModel:
    profile_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    temperature: float = 1.0
    full_reliability_public_actions: int = 6

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicOpponentBeliefModel":
        return cls(
            profile_names=tuple(str(item) for item in payload["profile_names"]),
            feature_names=tuple(str(item) for item in payload["feature_names"]),
            coefficients=tuple(
                tuple(float(value) for value in row)
                for row in payload["coefficients"]
            ),
            intercepts=tuple(float(value) for value in payload["intercepts"]),
            feature_means=tuple(float(value) for value in payload["feature_means"]),
            feature_scales=tuple(float(value) for value in payload["feature_scales"]),
            temperature=float(payload.get("temperature", 1.0)),
            full_reliability_public_actions=int(
                payload.get("full_reliability_public_actions", 6)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_names": list(self.profile_names),
            "feature_names": list(self.feature_names),
            "coefficients": [list(row) for row in self.coefficients],
            "intercepts": list(self.intercepts),
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "temperature": self.temperature,
            "full_reliability_public_actions": self.full_reliability_public_actions,
        }

    def posterior(
        self,
        features: Sequence[float],
        *,
        public_actions: int,
    ) -> dict[str, float]:
        feature_count = len(self.feature_names)
        profile_count = len(self.profile_names)
        if self.feature_names != PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES:
            raise ValueError("public_opponent_belief_feature_schema_mismatch")
        if len(features) != feature_count:
            raise ValueError("public_opponent_belief_feature_count_mismatch")
        if (
            len(self.coefficients) != feature_count
            or any(len(row) != profile_count for row in self.coefficients)
            or len(self.intercepts) != profile_count
            or len(self.feature_means) != feature_count
            or len(self.feature_scales) != feature_count
            or self.temperature <= 0.0
            or self.full_reliability_public_actions <= 0
        ):
            raise ValueError("invalid_public_opponent_belief_model")

        values = np.asarray(features, dtype=float)
        means = np.asarray(self.feature_means, dtype=float)
        scales = np.asarray(self.feature_scales, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(scales <= 0.0):
            raise ValueError("invalid_public_opponent_belief_input")
        logits = (
            ((values - means) / scales)
            @ np.asarray(self.coefficients, dtype=float)
            + np.asarray(self.intercepts, dtype=float)
        ) / self.temperature
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))

        reliability = min(
            1.0,
            max(0, int(public_actions))
            / float(self.full_reliability_public_actions),
        )
        uniform = 1.0 / max(1, profile_count)
        probabilities = reliability * probabilities + (1.0 - reliability) * uniform
        return {
            name: float(probabilities[index])
            for index, name in enumerate(self.profile_names)
        }


def smooth_weighted_profile_schedule(
    posterior: Mapping[str, float],
    *,
    slots: int,
) -> tuple[str, ...]:
    """Apportion a posterior into a deterministic, prefix-balanced schedule."""
    resolved_slots = max(1, int(slots))
    names = tuple(sorted(str(name) for name in posterior))
    if not names:
        raise ValueError("public_opponent_schedule_empty")
    probabilities = {
        name: max(0.0, float(posterior[name]))
        for name in names
    }
    total = sum(probabilities.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("public_opponent_schedule_invalid_probabilities")
    normalized = {name: probabilities[name] / total for name in names}
    quotas = {name: normalized[name] * resolved_slots for name in names}
    counts = {name: int(math.floor(quotas[name])) for name in names}
    remainder = resolved_slots - sum(counts.values())
    for name in sorted(
        names,
        key=lambda item: (quotas[item] - counts[item], item),
        reverse=True,
    )[:remainder]:
        counts[name] += 1

    current = {name: 0 for name in names}
    schedule: list[str] = []
    for _ in range(resolved_slots):
        available = [name for name in names if schedule.count(name) < counts[name]]
        for name in available:
            current[name] += counts[name]
        selected = max(available, key=lambda name: (current[name], name))
        current[selected] -= resolved_slots
        schedule.append(selected)
    if {name: schedule.count(name) for name in names} != counts:
        raise AssertionError("public_opponent_schedule_count_mismatch")
    return tuple(schedule)


def public_opponent_features(
    view: Any,
    rules: Mapping[str, Any],
    *,
    opponent_seat: int | None = None,
) -> PublicOpponentEvidence:
    seat = int(_field(view, "seat", 0) or 0)
    all_melds = tuple(_field(view, "all_melds", ()) or ())
    discards = tuple(_field(view, "discards", ()) or ())
    hand_sizes = tuple(_field(view, "hand_sizes", ()) or ())
    player_count = max(len(all_melds), len(discards), len(hand_sizes))
    opponent_seats = [index for index in range(player_count) if index != seat]
    if opponent_seat is None:
        if len(opponent_seats) != 1:
            raise ValueError("public_opponent_seat_required")
        opponent_seat = opponent_seats[0]
    if opponent_seat == seat or opponent_seat not in opponent_seats:
        raise ValueError("invalid_public_opponent_seat")

    melds = (
        tuple(all_melds[opponent_seat])
        if opponent_seat < len(all_melds)
        else ()
    )
    opponent_discards = (
        tuple(_label(item) for item in discards[opponent_seat])
        if opponent_seat < len(discards)
        else ()
    )
    opponent_discards = tuple(label for label in opponent_discards if label)
    kinds = tuple(_meld_kind(meld) for meld in melds)
    meld_labels = tuple(label for meld in melds for label in _meld_labels(meld))
    meld_count = len(melds)
    discard_count = len(opponent_discards)
    meld_cards = len(meld_labels)
    public_actions = meld_count + discard_count
    active_cards = max(0, int(_field(view, "stock_count", 0) or 0)) + sum(
        max(0, int(size)) for size in hand_sizes
    )
    ranks = [_rank(label) for label in opponent_discards]
    numeric_ranks = [rank for rank in ranks if rank is not None]
    exposed_xi = sum(
        meld_xi(
            list(_meld_labels(meld)),
            kind=_meld_kind(meld) or None,
            rules=dict(rules),
        )
        for meld in melds
    )
    recent = opponent_discards[-3:]
    opponent_hand_size = (
        max(0, int(hand_sizes[opponent_seat]))
        if opponent_seat < len(hand_sizes)
        else 0
    )
    features = (
        max(0, int(_field(view, "stock_count", 0) or 0)) / max(1, active_cards),
        discard_count / 20.0,
        meld_count / 6.0,
        meld_count / max(1, public_actions),
        sum(kind in _CHI_KINDS for kind in kinds) / max(1, meld_count),
        sum(kind == "peng" for kind in kinds) / max(1, meld_count),
        sum(kind in _CONCEALED_KINDS for kind in kinds) / max(1, meld_count),
        sum(kind in _SPECIAL_CHI_KINDS for kind in kinds) / max(1, meld_count),
        sum(kind == "mixed_same_rank_triplet" for kind in kinds) / max(1, meld_count),
        exposed_xi / 30.0,
        sum(label in RED_LABELS for label in meld_labels) / max(1, meld_cards),
        sum(label in RED_LABELS for label in opponent_discards) / max(1, discard_count),
        sum(_rank(label) in {2, 7, 10} for label in opponent_discards)
        / max(1, discard_count),
        sum(label in BIG_LABELS for label in opponent_discards) / max(1, discard_count),
        (fmean(numeric_ranks) / 10.0) if numeric_ranks else 0.0,
        (pstdev(numeric_ranks) / 10.0) if len(numeric_ranks) >= 2 else 0.0,
        len(set(opponent_discards)) / max(1, discard_count),
        sum(label in RED_LABELS for label in recent) / max(1, len(recent)),
        opponent_hand_size / 20.0,
        meld_cards / 20.0,
        min(20, public_actions) / 20.0,
    )
    if len(features) != len(PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES):
        raise AssertionError("public_opponent_feature_schema_internal_error")
    if not all(math.isfinite(value) for value in features):
        raise ValueError("nonfinite_public_opponent_feature")
    return PublicOpponentEvidence(
        opponent_seat=opponent_seat,
        public_actions=public_actions,
        features=features,
    )


def _field(view: Any, name: str, default: Any) -> Any:
    if isinstance(view, Mapping):
        return view.get(name, default)
    return getattr(view, name, default)


def _meld_kind(meld: Any) -> str:
    if isinstance(meld, Mapping):
        return str(meld.get("type") or meld.get("kind") or "")
    return str(getattr(meld, "kind", ""))


def _meld_labels(meld: Any) -> tuple[str, ...]:
    if isinstance(meld, Mapping):
        cards: Sequence[Any] = meld.get("labels") or meld.get("cards") or ()
    else:
        cards = getattr(meld, "cards", ())
    return tuple(label for item in cards if (label := _label(item)))


def _label(card: Any) -> str:
    if isinstance(card, str):
        return card
    if isinstance(card, Mapping):
        return str(card.get("label") or "")
    return str(getattr(card, "label", ""))


def _rank(label: str) -> int | None:
    if label in SMALL_LABELS:
        return SMALL_LABELS.index(label) + 1
    if label in BIG_LABELS:
        return BIG_LABELS.index(label) + 1
    return None


__all__ = [
    "PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES",
    "PublicOpponentBeliefModel",
    "PublicOpponentEvidence",
    "public_opponent_features",
    "smooth_weighted_profile_schedule",
]
