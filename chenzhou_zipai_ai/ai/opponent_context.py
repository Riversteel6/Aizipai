"""Public opponent-style features shared by runtime and calibration."""

from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Sequence


OPPONENT_CONTEXT_FEATURE_NAMES = (
    "stock_active_ratio",
    "opponent_discard_mean_scaled",
    "opponent_meld_mean_scaled",
    "opponent_chi_share",
    "opponent_peng_share",
    "opponent_concealed_share",
    "opponent_hand_size_mean_scaled",
    "opponent_meld_cards_mean_scaled",
)

OPPONENT_CONTEXT_INTERACTION_FEATURE_NAMES = tuple(
    f"combined_mean_delta_x_{name}"
    for name in OPPONENT_CONTEXT_FEATURE_NAMES
)


def opponent_context_features(view: Any) -> tuple[float, ...]:
    seat = int(_field(view, "seat", 0) or 0)
    all_melds = tuple(_field(view, "all_melds", ()) or ())
    discards = tuple(_field(view, "discards", ()) or ())
    hand_sizes = tuple(_field(view, "hand_sizes", ()) or ())
    player_count = max(len(all_melds), len(discards), len(hand_sizes))
    opponent_seats = [
        index
        for index in range(player_count)
        if index != seat
    ]
    if not opponent_seats:
        return (0.0,) * len(OPPONENT_CONTEXT_FEATURE_NAMES)

    opponent_melds = [
        tuple(all_melds[index])
        if index < len(all_melds)
        else ()
        for index in opponent_seats
    ]
    flat_melds = [
        meld
        for melds in opponent_melds
        for meld in melds
    ]
    meld_kinds = [_meld_kind(meld) for meld in flat_melds]
    meld_count = len(flat_melds)
    chi_count = sum(
        kind not in {"peng", "wei", "ti", "pao"}
        for kind in meld_kinds
    )
    peng_count = sum(kind == "peng" for kind in meld_kinds)
    concealed_count = sum(
        kind in {"wei", "ti", "pao"}
        for kind in meld_kinds
    )
    opponent_discard_counts = [
        len(discards[index]) if index < len(discards) else 0
        for index in opponent_seats
    ]
    opponent_hand_sizes = [
        int(hand_sizes[index])
        if index < len(hand_sizes)
        else 0
        for index in opponent_seats
    ]
    opponent_meld_cards = [
        sum(_meld_size(meld) for meld in melds)
        for melds in opponent_melds
    ]
    stock_count = max(0, int(_field(view, "stock_count", 0) or 0))
    active_cards = stock_count + sum(
        max(0, int(size))
        for size in hand_sizes
    )
    return (
        stock_count / max(1, active_cards),
        mean(opponent_discard_counts) / 20.0,
        mean(len(melds) for melds in opponent_melds) / 6.0,
        chi_count / max(1, meld_count),
        peng_count / max(1, meld_count),
        concealed_count / max(1, meld_count),
        mean(opponent_hand_sizes) / 20.0,
        mean(opponent_meld_cards) / 20.0,
    )


def opponent_context_interactions(
    combined_mean_delta: float,
    context: Sequence[float],
) -> tuple[float, ...]:
    if len(context) != len(OPPONENT_CONTEXT_FEATURE_NAMES):
        raise ValueError("opponent_context_interaction_feature_mismatch")
    return tuple(float(combined_mean_delta) * float(value) for value in context)


def _field(view: Any, name: str, default: Any) -> Any:
    if isinstance(view, Mapping):
        return view.get(name, default)
    return getattr(view, name, default)


def _meld_kind(meld: Any) -> str:
    if isinstance(meld, Mapping):
        return str(meld.get("type") or meld.get("kind") or "")
    return str(getattr(meld, "kind", ""))


def _meld_size(meld: Any) -> int:
    if isinstance(meld, Mapping):
        cards: Sequence[Any] = (
            meld.get("labels")
            or meld.get("cards")
            or ()
        )
    else:
        cards = getattr(meld, "cards", ())
    return len(cards)
