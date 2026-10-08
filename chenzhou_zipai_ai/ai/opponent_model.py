"""Simple opponent modelling from visible history."""

from __future__ import annotations

from collections import Counter

from engine.cards import RED_LABELS, normalize_card_label


def _normalize_labels(raw_labels) -> list[str]:
    if not isinstance(raw_labels, list):
        return []
    labels: list[str] = []
    for raw in raw_labels:
        label = normalize_card_label(str(raw))
        if label and label != "暗":
            labels.append(label)
    return labels


def _opponent_entries(memory: dict | None) -> list[dict]:
    if not memory:
        return []
    entries: list[dict] = []
    for key in ("opponents", "opponent_states", "other_players"):
        raw = memory.get(key, [])
        if isinstance(raw, dict):
            raw = raw.values()
        if not isinstance(raw, list) and not hasattr(raw, "__iter__"):
            continue
        for item in raw:
            if isinstance(item, dict):
                entries.append(item)
    return entries


def opponent_entries(memory: dict | None) -> list[dict]:
    """Return seat-separated public opponent records when available."""
    return _opponent_entries(memory)


def _opponent_groups(memory: dict | None) -> list[list[str]]:
    if not memory:
        return []
    groups: list[list[str]] = []
    opponents = _opponent_entries(memory)
    if not opponents:
        for group in memory.get("opponent_meld_groups") or []:
            groups.append(
                _normalize_labels(group.get("cards") or group.get("labels") or [])
                if isinstance(group, dict)
                else _normalize_labels(group)
            )
    for opponent in opponents:
        for key in ("meld_groups", "opponent_meld_groups", "exposed_melds"):
            for group in opponent.get(key, []) or []:
                groups.append(
                    _normalize_labels(group.get("cards") or group.get("labels") or [])
                    if isinstance(group, dict)
                    else _normalize_labels(group)
                )
    return [group for group in groups if group]


def _opponent_discards(memory: dict | None) -> list[str]:
    if not memory:
        return []
    opponents = _opponent_entries(memory)
    discards = [] if opponents else _normalize_labels(memory.get("opponent_discards", []))
    for opponent in opponents:
        discards.extend(_normalize_labels(opponent.get("discards", [])))
        discards.extend(_normalize_labels(opponent.get("opponent_discards", [])))
    return discards


def _infer_profile(opponent_discards: list[str], opponent_melds: list[list[str]]) -> dict:
    discarded_red = sum(1 for label in opponent_discards if label in RED_LABELS)
    meld_count = len(opponent_melds)
    red_in_melds = sum(1 for group in opponent_melds for label in group if label in RED_LABELS)
    counts = Counter(label for group in opponent_melds for label in group if label != "暗")
    if meld_count >= 3:
        profile = "fast_meld"
    elif red_in_melds >= discarded_red + 2:
        profile = "red_value"
    elif any(amount >= 2 for amount in counts.values()):
        profile = "pair_triplet_value"
    else:
        profile = "balanced"
    return {
        "profile": profile,
        "red_pressure": red_in_melds - discarded_red,
        "meld_count": meld_count,
    }


def infer_opponent_profiles(memory: dict | None) -> list[dict]:
    """Infer one profile per opponent without merging different seats."""
    profiles: list[dict] = []
    for index, opponent in enumerate(_opponent_entries(memory)):
        discards: list[str] = []
        for key in ("discards", "opponent_discards"):
            discards.extend(_normalize_labels(opponent.get(key, [])))
        groups: list[list[str]] = []
        for key in ("meld_groups", "opponent_meld_groups", "exposed_melds"):
            for group in opponent.get(key, []) or []:
                labels = (
                    _normalize_labels(group.get("cards") or group.get("labels") or [])
                    if isinstance(group, dict)
                    else _normalize_labels(group)
                )
                if labels:
                    groups.append(labels)
        profile = _infer_profile(discards, groups)
        profile.update(
            {
                "seat": opponent.get("seat", index + 1),
                "relative_offset": opponent.get("relative_offset"),
                "is_next_seat": bool(opponent.get("is_next_seat", False)),
            }
        )
        profiles.append(profile)
    return profiles


def infer_opponent_profile(memory: dict | None) -> dict:
    if not memory:
        return {"profile": "unknown", "red_pressure": 0, "meld_count": 0}
    opponent_discards = _opponent_discards(memory)
    opponent_melds = _opponent_groups(memory)
    profile = _infer_profile(opponent_discards, opponent_melds)
    profile["opponent_count"] = max(
        1,
        len(_opponent_entries(memory)) or int(bool(opponent_melds or opponent_discards)),
    )
    return profile
