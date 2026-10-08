"""Build leakage-safe action-value samples from counterfactual audits."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from engine.cards import BIG_LABELS, SMALL_LABELS, WILD_LABEL


ALL_LABELS = (*SMALL_LABELS, *BIG_LABELS, WILD_LABEL)
MINIMUM_DEALS_PER_MODE_SPLIT = 10


def build_value_corpus(
    audits: Iterable[Mapping[str, Any]],
    *,
    split_salt: str = "aizipai-value-v1",
) -> dict[str, Any]:
    audit_list = [dict(audit) for audit in audits]
    preferred_by_group: dict[str, tuple[int, int]] = {}
    superseded_audits = 0
    for audit_index, audit in enumerate(audit_list):
        if not audit.get("complete"):
            continue
        game = dict(audit.get("game") or {})
        public_state = dict(audit.get("public_state") or {})
        if not game or not public_state:
            continue
        group_id = _group_id(audit, game)
        quality = int(audit.get("completed_paired_worlds") or 0)
        previous = preferred_by_group.get(group_id)
        if previous is None:
            preferred_by_group[group_id] = (quality, audit_index)
        elif quality > previous[0]:
            preferred_by_group[group_id] = (quality, audit_index)
            superseded_audits += 1
        else:
            superseded_audits += 1
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_actions: set[tuple[str, str]] = set()
    split_by_deal: dict[str, str] = {}
    for audit_index, audit in enumerate(audit_list):
        if not audit.get("complete"):
            failures.append(
                {
                    "audit_index": audit_index,
                    "reason": "incomplete_counterfactual_audit",
                }
            )
            continue
        game = dict(audit.get("game") or {})
        public_state = dict(audit.get("public_state") or {})
        if not game or not public_state:
            failures.append(
                {
                    "audit_index": audit_index,
                    "reason": "missing_game_or_public_state",
                }
            )
            continue
        group_id = _group_id(audit, game)
        if preferred_by_group[group_id][1] != audit_index:
            continue
        deal_key = _deal_key(game)
        split = _split_for_deal(deal_key, salt=split_salt)
        previous_split = split_by_deal.setdefault(deal_key, split)
        if previous_split != split:
            failures.append(
                {
                    "audit_index": audit_index,
                    "reason": "deal_split_leakage",
                    "deal_key": deal_key,
                }
            )
            continue
        for candidate in audit.get("candidate_stats") or ():
            candidate_payload = dict(candidate)
            action_key = _candidate_key(candidate_payload)
            if not action_key:
                failures.append(
                    {
                        "audit_index": audit_index,
                        "reason": "candidate_key_missing",
                    }
                )
                continue
            dedupe_key = (group_id, action_key)
            if dedupe_key in seen_actions:
                failures.append(
                    {
                        "audit_index": audit_index,
                        "reason": "duplicate_group_action",
                        "group_id": group_id,
                        "action_key": action_key,
                    }
                )
                continue
            seen_actions.add(dedupe_key)
            rows.append(
                {
                    "schema_version": "action-value-sample-v1",
                    "split": split,
                    "deal_key": deal_key,
                    "group_id": group_id,
                    "state_before_hash": str(
                        audit.get("state_before_hash") or ""
                    ),
                    "game": game,
                    "phase": str(audit.get("phase") or ""),
                    "turn": int(audit.get("turn") or 0),
                    "seat": int(audit.get("seat") or 0),
                    "action": _action_payload(
                        action_key,
                        candidate_payload,
                    ),
                    "selected": action_key == str(
                        audit.get("selected_key") or ""
                    ),
                    "empirical_best": action_key == str(
                        audit.get("best_key") or ""
                    ),
                    "state_features": _state_features(public_state),
                    "targets": _target_payload(candidate_payload),
                    "audit_health": {
                        "paired_worlds": int(
                            audit.get("completed_paired_worlds") or 0
                        ),
                        "confidently_suboptimal": bool(
                            audit.get("confidently_suboptimal")
                        ),
                    },
                }
            )
    split_counts = Counter(row["split"] for row in rows)
    phase_counts = Counter(row["phase"] for row in rows)
    mode_counts = Counter(
        _mode_key(row["game"])
        for row in rows
    )
    group_counts = Counter(row["group_id"] for row in rows)
    group_mode = {
        row["group_id"]: _mode_key(row["game"])
        for row in rows
    }
    group_split = {
        row["group_id"]: row["split"]
        for row in rows
    }
    group_mode_split = {
        row["group_id"]: (_mode_key(row["game"]), row["split"])
        for row in rows
    }
    deal_mode_split = {
        (row["deal_key"], _mode_key(row["game"]), row["split"])
        for row in rows
    }
    decision_groups_by_mode = Counter(group_mode.values())
    decision_groups_by_split = Counter(group_split.values())
    decision_groups_by_mode_split = Counter(group_mode_split.values())
    deals_by_mode_split = Counter(
        (mode, split)
        for _deal_key_value, mode, split in deal_mode_split
    )
    readiness_failures: list[str] = []
    if len(group_counts) < 2_000:
        readiness_failures.append("decision_groups_below_2000")
    for mode in ("2p_no_wang", "2p_wang", "3p_no_wang", "3p_wang"):
        if decision_groups_by_mode[mode] < 400:
            readiness_failures.append(
                f"{mode}_decision_groups_below_400"
            )
    for split in ("train", "validation", "test"):
        if decision_groups_by_split[split] < 100:
            readiness_failures.append(
                f"{split}_decision_groups_below_100"
            )
    minimum_groups = {
        "train": 150,
        "validation": 40,
        "test": 40,
    }
    for mode in ("2p_no_wang", "2p_wang", "3p_no_wang", "3p_wang"):
        for split, minimum in minimum_groups.items():
            if decision_groups_by_mode_split[(mode, split)] < minimum:
                readiness_failures.append(
                    f"{mode}_{split}_decision_groups_below_{minimum}"
                )
            if (
                deals_by_mode_split[(mode, split)]
                < MINIMUM_DEALS_PER_MODE_SPLIT
            ):
                readiness_failures.append(
                    f"{mode}_{split}_deals_below_"
                    f"{MINIMUM_DEALS_PER_MODE_SPLIT}"
                )
    return {
        "ok": not failures,
        "training_ready": not failures and not readiness_failures,
        "schema_version": "action-value-corpus-summary-v1",
        "samples": len(rows),
        "decision_groups": len(group_counts),
        "minimum_actions_per_group": min(group_counts.values(), default=0),
        "maximum_actions_per_group": max(group_counts.values(), default=0),
        "superseded_audits": superseded_audits,
        "by_split": dict(sorted(split_counts.items())),
        "by_phase": dict(sorted(phase_counts.items())),
        "by_mode": dict(sorted(mode_counts.items())),
        "decision_groups_by_mode": dict(
            sorted(decision_groups_by_mode.items())
        ),
        "decision_groups_by_split": dict(
            sorted(decision_groups_by_split.items())
        ),
        "decision_groups_by_mode_split": {
            f"{mode}:{split}": count
            for (mode, split), count in sorted(
                decision_groups_by_mode_split.items()
            )
        },
        "deals_by_mode_split": {
            f"{mode}:{split}": count
            for (mode, split), count in sorted(deals_by_mode_split.items())
        },
        "split_deals": dict(
            sorted(Counter(split_by_deal.values()).items())
        ),
        "split_leakage": 0
        if all(
            split_by_deal[key] == split
            for key, split in split_by_deal.items()
        )
        else 1,
        "failures": failures,
        "readiness_failures": readiness_failures,
        "rows": rows,
    }


def _target_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "visits": int(candidate.get("visits") or 0),
        "mean_reward": float(candidate.get("average_reward") or 0.0),
        "win_rate": float(candidate.get("win_rate") or 0.0),
        "loss_rate": float(candidate.get("loss_rate") or 0.0),
        "draw_rate": float(candidate.get("draw_rate") or 0.0),
        "mean_outcome_score": float(
            candidate.get("mean_outcome_score") or 0.0
        ),
        "mean_signed_xi": float(
            candidate.get("mean_signed_xi") or 0.0
        ),
    }


def _state_features(public_state: Mapping[str, Any]) -> dict[str, Any]:
    hand = tuple(str(label) for label in public_state.get("hand") or ())
    own_melds = tuple(public_state.get("own_melds") or ())
    all_melds = tuple(public_state.get("all_melds") or ())
    discards = tuple(public_state.get("discards") or ())
    return {
        "hand_counts": {
            label: hand.count(label)
            for label in ALL_LABELS
        },
        "own_meld_count": len(own_melds),
        "own_melds": [
            _public_meld_payload(meld)
            for meld in own_melds
        ],
        "all_melds": [
            [
                _public_meld_payload(meld)
                for meld in seat_melds or ()
            ]
            for seat_melds in all_melds
        ],
        "discards": [
            [str(label) for label in labels or ()]
            for labels in discards
        ],
        "opponent_meld_counts": [
            len(melds)
            for seat, melds in enumerate(all_melds)
            if seat != int(public_state.get("seat") or 0)
        ],
        "discard_counts": [
            len(labels)
            for labels in discards
        ],
        "remaining_counts": {
            str(label): int(amount)
            for label, amount in public_state.get("remaining_counts") or ()
        },
        "stock_count": int(public_state.get("stock_count") or 0),
        "hand_sizes": [
            int(size)
            for size in public_state.get("hand_sizes") or ()
        ],
        "pending_card": public_state.get("pending_card"),
        "pending_source_seat": public_state.get("pending_source_seat"),
    }


def _public_meld_payload(meld: Any) -> dict[str, Any]:
    if isinstance(meld, Mapping):
        return {
            "type": str(meld.get("type") or meld.get("kind") or ""),
            "labels": [
                str(label)
                for label in meld.get("labels") or meld.get("cards") or ()
            ],
        }
    return {
        "type": "",
        "labels": [str(label) for label in meld or ()],
    }


def _action_payload(
    action_key: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    nested = dict(candidate.get("candidate") or {})
    action_type, _separator, key_suffix = action_key.partition(":")
    action_type = str(
        candidate.get("action_type")
        or nested.get("action_type")
        or action_type
    )
    label = candidate.get("label", nested.get("label"))
    if label is None and action_type in {"DISCARD", "PENG"}:
        label = key_suffix or None
    return {
        "key": action_key,
        "type": action_type,
        "label": label,
        "option_id": candidate.get(
            "option_id",
            nested.get("option_id"),
        ),
        "consumed_from_hand": list(
            candidate.get(
                "consumed_from_hand",
                nested.get("consumed_from_hand"),
            )
            or ()
        ),
        "meld_groups": list(
            candidate.get(
                "meld_groups",
                nested.get("meld_groups"),
            )
            or ()
        ),
        "followup_discard": candidate.get(
            "followup_discard",
            nested.get("followup_discard"),
        ),
        "heuristic_value": float(
            candidate.get(
                "heuristic_value",
                nested.get("heuristic_value", 0.0),
            )
            or 0.0
        ),
    }


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    direct = str(candidate.get("key") or "")
    if direct:
        return direct
    nested = candidate.get("candidate")
    if isinstance(nested, Mapping):
        return str(nested.get("key") or "")
    return ""


def _deal_key(game: Mapping[str, Any]) -> str:
    return ":".join(
        (
            str(int(game.get("players") or 0)),
            str(int(bool(game.get("wildcard_enabled")))),
            str(int(game.get("seed") or 0)),
        )
    )


def _group_id(
    audit: Mapping[str, Any],
    game: Mapping[str, Any],
) -> str:
    payload = "|".join(
        (
            _deal_key(game),
            str(int(game.get("candidate_seat") or 0)),
            str(int(game.get("dealer") or 0)),
            str(audit.get("state_before_hash") or ""),
            str(audit.get("phase") or ""),
            str(int(audit.get("trace_sequence") or 0)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_for_deal(deal_key: str, *, salt: str) -> str:
    digest = hashlib.sha256(
        f"{salt}|{deal_key}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def _mode_key(game: Mapping[str, Any]) -> str:
    players = int(game.get("players") or 0)
    wildcard = "wang" if game.get("wildcard_enabled") else "no_wang"
    return f"{players}p_{wildcard}"


__all__ = [
    "MINIMUM_DEALS_PER_MODE_SPLIT",
    "build_value_corpus",
]
