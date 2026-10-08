"""Persistent vision memory across screenshots."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SMALL_LABELS = "一二三四五六七八九十"
BIG_LABELS = "壹贰叁肆伍陆柒捌玖拾"


@dataclass
class VisionMemory:
    frames_seen: int = 0
    last_screenshot: str | None = None
    remaining_deck_count: int | None = None
    opponent_discards: list[str] = field(default_factory=list)
    my_discards: list[str] = field(default_factory=list)
    opponent_meld_groups: list[list[str]] = field(default_factory=list)
    my_meld_groups: list[list[str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    last_hand: list[str] = field(default_factory=list)
    last_hand_details: list[dict[str, Any]] = field(default_factory=list)
    last_hand_expected_total: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionMemory":
        return cls(
            frames_seen=int(data.get("frames_seen", 0)),
            last_screenshot=data.get("last_screenshot"),
            remaining_deck_count=data.get("remaining_deck_count"),
            opponent_discards=list(data.get("opponent_discards", [])),
            my_discards=list(data.get("my_discards", [])),
            opponent_meld_groups=[list(group) for group in data.get("opponent_meld_groups", [])],
            my_meld_groups=[list(group) for group in data.get("my_meld_groups", [])],
            warnings=list(data.get("warnings", [])),
            last_hand=list(data.get("last_hand", [])),
            last_hand_details=[dict(item) for item in data.get("last_hand_details", [])],
            last_hand_expected_total=data.get("last_hand_expected_total"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_seen": self.frames_seen,
            "last_screenshot": self.last_screenshot,
            "remaining_deck_count": self.remaining_deck_count,
            "opponent_discards": self.opponent_discards,
            "my_discards": self.my_discards,
            "opponent_meld_groups": self.opponent_meld_groups,
            "my_meld_groups": self.my_meld_groups,
            "warnings": self.warnings,
            "last_hand": self.last_hand,
            "last_hand_details": self.last_hand_details,
            "last_hand_expected_total": self.last_hand_expected_total,
        }

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.frames_seen += 1
        self.last_screenshot = snapshot.get("screenshot")
        if snapshot.get("remaining_deck_count") is not None:
            self.remaining_deck_count = snapshot["remaining_deck_count"]

        discards = snapshot.get("discards", {})
        self.opponent_discards = _merge_sequence(
            self.opponent_discards,
            _labels_from_items(discards.get("opponent_discards", [])),
        )
        self.my_discards = _merge_sequence(
            self.my_discards,
            _labels_from_items(discards.get("my_discards", [])),
        )

        meld_groups = snapshot.get("meld_groups", {})
        self.opponent_meld_groups = _merge_groups(
            self.opponent_meld_groups,
            _labels_from_groups(meld_groups.get("opponent_melds", [])),
        )
        self.my_meld_groups = _merge_groups(
            self.my_meld_groups,
            _labels_from_groups(meld_groups.get("my_melds", [])),
        )

        sanity = snapshot.get("sanity_checks", {})
        self.warnings = list(sanity.get("warnings", []))
        hand = list(snapshot.get("hand") or [])
        if hand and sanity.get("ok", True):
            self.last_hand = hand
            self.last_hand_details = [dict(item) for item in snapshot.get("hand_details", [])]
            self.last_hand_expected_total = sanity.get("expected_total")


def recover_temporally_hidden_hand(
    snapshot: dict[str, Any],
    memory: VisionMemory,
    *,
    expected_total: int | None,
    max_recovered: int = 2,
) -> dict[str, Any]:
    """Restore fully occluded cards only from the last validated, unchanged hand."""

    if expected_total is None or not memory.last_hand:
        return snapshot
    metadata = snapshot.get("metadata") or {}
    if metadata.get("recognition_source") == "protocol_with_screenshot_coordinates":
        return snapshot
    if memory.last_hand_expected_total not in (None, expected_total):
        return snapshot
    if not _is_decision_window(snapshot):
        return snapshot

    sanity = build_sanity_checks(snapshot, expected_total=expected_total)
    controlled_count = int(sanity.get("controlled_card_count") or 0)
    recovery_limit = max_recovered
    if _is_dealer_opening_occlusion_window(snapshot, memory, expected_total):
        recovery_limit = max(recovery_limit, 3)
    deficit = expected_total - controlled_count
    if deficit <= 0 or deficit > recovery_limit:
        return snapshot
    if _has_new_visible_history(snapshot, memory):
        return snapshot

    current_hand = list(snapshot.get("hand") or [])
    previous_counts = Counter(memory.last_hand)
    current_counts = Counter(current_hand)
    if current_counts - previous_counts:
        return snapshot
    missing = list((previous_counts - current_counts).elements())
    if len(missing) != deficit:
        return snapshot

    patched = dict(snapshot)
    patched_hand = list(current_hand)
    patched_raw_hand = list(snapshot.get("raw_hand") or current_hand)
    patched_details = [dict(item) for item in snapshot.get("hand_details", [])]
    detail_pool: dict[str, list[dict[str, Any]]] = {}
    for item in memory.last_hand_details:
        label = str(item.get("name") or item.get("label") or "")
        if label:
            detail_pool.setdefault(label, []).append(dict(item))

    recovered_details: list[dict[str, Any]] = []
    for index, label in enumerate(missing, start=1):
        base = detail_pool.get(label, []).pop(0) if detail_pool.get(label) else {}
        card_id = f"temporal_hidden_{index:03d}"
        base.update(
            {
                "name": label,
                "label": label,
                "card_id": card_id,
                "id": card_id,
                "confidence": 0.76,
                "raw_confidence": base.get("raw_confidence", base.get("confidence", 0.0)),
                "clickable": False,
                "recognition_source": "temporal_validated_hand",
                "source": "temporal_validated_hand",
            }
        )
        recovered_details.append(base)
        patched_hand.append(label)
        patched_raw_hand.append(label)

    patched["hand"] = patched_hand
    patched["raw_hand"] = patched_raw_hand
    patched["hand_details"] = patched_details + recovered_details
    patched["hand_count"] = len(patched_hand)
    patched_metadata = dict(metadata)
    patched_metadata["temporal_hand_recovery"] = {
        "labels": missing,
        "count": len(missing),
        "source_screenshot": memory.last_screenshot,
    }
    patched["metadata"] = patched_metadata
    patched["sanity_checks"] = build_sanity_checks(patched, expected_total=expected_total)
    return patched


def _is_dealer_opening_occlusion_window(
    snapshot: dict[str, Any],
    memory: VisionMemory,
    expected_total: int,
) -> bool:
    if expected_total != 21 or snapshot.get("discard_button") is None:
        return False
    if memory.my_discards or memory.my_meld_groups:
        return False
    if (snapshot.get("discards") or {}).get("my_discards"):
        return False
    if (snapshot.get("meld_groups") or {}).get("my_melds"):
        return False
    response_names = {"chi", "peng", "pao", "hu", "pass"}
    return not any(
        str(button.get("name") or button.get("type") or "").lower() in response_names
        for button in snapshot.get("buttons") or []
        if isinstance(button, dict)
    )


def _is_decision_window(snapshot: dict[str, Any]) -> bool:
    if snapshot.get("discard_button"):
        return True
    response_names = {"chi", "peng", "pao", "hu", "pass"}
    return any(
        str(button.get("name") or button.get("type") or "").lower() in response_names
        for button in snapshot.get("buttons") or []
        if isinstance(button, dict)
    )


def _has_new_visible_history(snapshot: dict[str, Any], memory: VisionMemory) -> bool:
    current_discards = _labels_from_items((snapshot.get("discards") or {}).get("my_discards", []))
    if _is_subsequence(memory.my_discards, current_discards) and len(current_discards) > len(memory.my_discards):
        return True
    current_groups = _labels_from_groups((snapshot.get("meld_groups") or {}).get("my_melds", []))
    return any(group not in memory.my_meld_groups for group in current_groups)


def load_memory(path: str | Path) -> VisionMemory:
    memory_path = Path(path)
    if not memory_path.exists():
        return VisionMemory()
    try:
        return VisionMemory.from_dict(json.loads(memory_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return VisionMemory()


def save_memory(memory: VisionMemory, path: str | Path) -> bool:
    memory_path = Path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = memory_path.with_name(f"{memory_path.stem}.{id(memory)}.tmp{memory_path.suffix}")
    try:
        temp_path.write_text(json.dumps(memory.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(memory_path)
        return True
    except OSError:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def infer_meld_type(labels: list[str]) -> str:
    visible = [label for label in labels if label != "暗"]
    hidden_count = len(labels) - len(visible)
    if not labels:
        return "empty"
    if hidden_count:
        if len(labels) >= 4:
            return "hidden_quad"
        if len(labels) == 3 and len(visible) == 1:
            return "hidden_triplet"
        return "hidden_meld"
    if len(set(labels)) == 1:
        return "peng" if len(labels) == 3 else "quad"
    parsed = [_parse_label(label) for label in labels]
    if all(item is not None for item in parsed):
        suits = {item[0] for item in parsed if item is not None}
        ranks = sorted(item[1] for item in parsed if item is not None)
        if len(suits) == 1 and len(ranks) >= 3 and ranks == list(range(ranks[0], ranks[0] + len(ranks))):
            return "chi_sequence"
        if len(set(ranks)) == 1:
            return "chi_same_rank"
    return "unknown"


def build_sanity_checks(snapshot: dict[str, Any], expected_total: int | None = None) -> dict[str, Any]:
    my_groups = _labels_from_groups(snapshot.get("meld_groups", {}).get("my_melds", []))
    my_meld_cell_count = sum(len(group) for group in my_groups)
    hand_count = int(snapshot.get("hand_count", 0))
    controlled_count = hand_count + my_meld_cell_count
    accepted_counts = [expected_total] if expected_total is not None else []
    transient_source_card = False
    warnings: list[str] = []

    if (
        expected_total is not None
        and controlled_count == expected_total + 1
        and _is_awaiting_own_discard(snapshot)
    ):
        accepted_counts.append(expected_total + 1)
        transient_source_card = True

    if expected_total is not None and controlled_count not in accepted_counts:
        warnings.append(
            f"controlled_card_count_mismatch: hand({hand_count}) + my_melds({my_meld_cell_count}) = "
            f"{controlled_count}, expected one of {accepted_counts}"
        )

    return {
        "hand_count": hand_count,
        "my_meld_cell_count": my_meld_cell_count,
        "controlled_card_count": controlled_count,
        "expected_total": expected_total,
        "accepted_counts": accepted_counts,
        "transient_source_card": transient_source_card,
        "ok": not warnings,
        "warnings": warnings,
    }


def _is_awaiting_own_discard(snapshot: dict[str, Any]) -> bool:
    phase = str(snapshot.get("phase", ""))
    if "await_action" not in phase.lower():
        return False
    return snapshot.get("discard_button") is not None


def enrich_meld_groups(meld_groups: dict[str, list[list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    enriched: dict[str, list[dict[str, Any]]] = {}
    for region_name, groups in meld_groups.items():
        enriched[region_name] = []
        for group in groups:
            labels = [item["name"] for item in group]
            enriched[region_name].append(
                {
                    "labels": labels,
                    "type": infer_meld_type(labels),
                    "hidden_count": sum(1 for label in labels if label == "暗"),
                    "cells": group,
                }
            )
    return enriched


def _parse_label(label: str) -> tuple[str, int] | None:
    if label in SMALL_LABELS:
        return ("small", SMALL_LABELS.index(label) + 1)
    if label in BIG_LABELS:
        return ("big", BIG_LABELS.index(label) + 1)
    return None


def _labels_from_items(items: list[dict[str, Any]]) -> list[str]:
    return [item["name"] for item in items]


def _labels_from_groups(groups: list[list[dict[str, Any]]]) -> list[list[str]]:
    return [[item["name"] for item in group] for group in groups]


def _merge_sequence(previous: list[str], current: list[str]) -> list[str]:
    if not current:
        return previous
    if current == previous or _is_subsequence(current, previous):
        return previous
    if _is_subsequence(previous, current):
        return current
    max_overlap = min(len(previous), len(current))
    for overlap in range(max_overlap, 0, -1):
        if previous[-overlap:] == current[:overlap]:
            return previous + current[overlap:]
    return current if len(current) > len(previous) else previous


def _merge_groups(previous: list[list[str]], current: list[list[str]]) -> list[list[str]]:
    result = [list(group) for group in previous]
    for group in current:
        if group not in result:
            result.append(list(group))
    return result


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return True
    index = 0
    for item in haystack:
        if item == needle[index]:
            index += 1
            if index == len(needle):
                return True
    return False
