"""Independent Chenzhou Zipai rules oracle.

This module intentionally does not import anything from ``engine`` or ``ai``.
It exists to catch common-mode bugs in the production rule implementation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


SMALL_LABELS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
BIG_LABELS = ("壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "拾")
NORMAL_LABELS = (*SMALL_LABELS, *BIG_LABELS)
WILD_LABEL = "王"
RED_LABELS = frozenset({"二", "七", "十", "贰", "柒", "拾"})
LABEL_INDEX = {label: index for index, label in enumerate((*NORMAL_LABELS, WILD_LABEL))}


@dataclass(frozen=True)
class OracleRoomShape:
    players: int
    wildcard_enabled: bool
    deck_size: int
    initial_dealt: int
    initial_stock: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "players": self.players,
            "wildcard_enabled": self.wildcard_enabled,
            "deck_size": self.deck_size,
            "initial_dealt": self.initial_dealt,
            "initial_stock": self.initial_stock,
        }


@dataclass(frozen=True)
class _Template:
    labels: tuple[str, str, str]
    kind: str
    suit: str | None


@dataclass(frozen=True)
class OracleGroup:
    source_labels: tuple[str, ...]
    resolved_labels: tuple[str, ...]
    kind: str
    suit: str | None
    xi: int
    wildcards_used: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_labels": list(self.source_labels),
            "resolved_labels": list(self.resolved_labels),
            "kind": self.kind,
            "suit": self.suit,
            "xi": self.xi,
            "wildcards_used": self.wildcards_used,
        }


@dataclass(frozen=True)
class OracleHuResult:
    can_hu: bool
    partition_complete: bool
    total_xi: int
    min_xi: int
    group_count: int
    hand_groups: tuple[OracleGroup, ...]
    existing_xi: int
    red_count: int
    red_black_kind: str | None
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_hu": self.can_hu,
            "partition_complete": self.partition_complete,
            "total_xi": self.total_xi,
            "min_xi": self.min_xi,
            "group_count": self.group_count,
            "hand_groups": [group.to_dict() for group in self.hand_groups],
            "existing_xi": self.existing_xi,
            "red_count": self.red_count,
            "red_black_kind": self.red_black_kind,
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class OracleChiPlan:
    initial_group: tuple[str, str, str]
    compare_groups: tuple[tuple[str, str, str], ...]
    consumed_from_hand: tuple[str, ...]

    @property
    def groups(self) -> tuple[tuple[str, str, str], ...]:
        return (self.initial_group, *self.compare_groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_group": list(self.initial_group),
            "compare_groups": [list(group) for group in self.compare_groups],
            "consumed_from_hand": list(self.consumed_from_hand),
        }


@dataclass(frozen=True)
class OracleAction:
    type: str
    label: str | None = None
    consumed_from_hand: tuple[str, ...] = ()
    meld_groups: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "label": self.label,
            "consumed_from_hand": list(self.consumed_from_hand),
            "meld_groups": [list(group) for group in self.meld_groups],
        }


@dataclass(frozen=True)
class OracleResponseActions:
    actions: tuple[OracleAction, ...]
    automatic_action: str | None = None
    automatic_seat: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "automatic_action": self.automatic_action,
            "automatic_seat": self.automatic_seat,
        }


@dataclass(frozen=True)
class OracleMeldState:
    kind: str
    labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class OracleResponseTransition:
    hand: tuple[str, ...]
    melds: tuple[OracleMeldState, ...]
    claim_type: str
    terminal: bool
    passed_chi: tuple[str, ...] = ()
    passed_peng: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand": list(self.hand),
            "melds": [meld.to_dict() for meld in self.melds],
            "claim_type": self.claim_type,
            "terminal": self.terminal,
            "passed_chi": list(self.passed_chi),
            "passed_peng": list(self.passed_peng),
        }


@dataclass(frozen=True)
class OracleDrawTransition:
    hand: tuple[str, ...]
    melds: tuple[OracleMeldState, ...]
    automatic_action: str | None
    quad_events: int
    skip_discard: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand": list(self.hand),
            "melds": [meld.to_dict() for meld in self.melds],
            "automatic_action": self.automatic_action,
            "quad_events": self.quad_events,
            "skip_discard": self.skip_discard,
        }


def room_shape_oracle(
    *,
    players: int,
    wildcard_enabled: bool,
    deck_copies: int = 4,
    wildcard_copies: int = 4,
) -> OracleRoomShape:
    """Compute room deck/deal shape without importing production helpers."""

    if players not in {2, 3}:
        raise ValueError("players must be 2 or 3")
    if deck_copies < 1 or wildcard_copies < 0:
        raise ValueError("deck copy counts must be non-negative")
    deck_size = len(NORMAL_LABELS) * deck_copies
    if wildcard_enabled:
        deck_size += wildcard_copies
    initial_dealt = players * 20 + 1
    if initial_dealt > deck_size:
        raise ValueError("initial deal exceeds deck size")
    return OracleRoomShape(
        players=players,
        wildcard_enabled=wildcard_enabled,
        deck_size=deck_size,
        initial_dealt=initial_dealt,
        initial_stock=deck_size - initial_dealt,
    )


def evaluate_hu_oracle(
    hand: Sequence[str],
    existing_melds: Sequence[Any] | None,
    rules: Mapping[str, Any],
) -> OracleHuResult:
    """Evaluate Hu through independent exact-cover recursion."""

    labels = tuple(str(label) for label in hand)
    invalid = [label for label in labels if label not in LABEL_INDEX]
    minimum = int(_section(rules, "rules").get("min_xi", 9))
    required_groups = int(_section(rules, "rules").get("required_meld_groups", 7))
    if invalid:
        return OracleHuResult(
            can_hu=False,
            partition_complete=False,
            total_xi=0,
            min_xi=minimum,
            group_count=0,
            hand_groups=(),
            existing_xi=0,
            red_count=0,
            red_black_kind=None,
            score=0.0,
            reasons=(f"invalid_labels:{','.join(invalid)}",),
        )

    existing = tuple(existing_melds or ())
    existing_xi, existing_valid = _existing_meld_xi(existing, rules)
    groups_needed = required_groups - len(existing)
    existing_quad_count = sum(
        len(_meld_fields(meld)[0]) == 4
        for meld in existing
    )
    pair_needed = int(existing_quad_count >= 1)
    triple_groups_needed = groups_needed - pair_needed
    expected_hand_cards = triple_groups_needed * 3 + pair_needed * 2
    if (
        groups_needed < 0
        or triple_groups_needed < 0
        or len(labels) != expected_hand_cards
    ):
        return OracleHuResult(
            can_hu=False,
            partition_complete=False,
            total_xi=existing_xi,
            min_xi=minimum,
            group_count=len(existing),
            hand_groups=(),
            existing_xi=existing_xi,
            red_count=_red_count(labels, existing),
            red_black_kind=None,
            score=0.0,
            reasons=(
                f"hand_card_count={len(labels)}",
                f"groups_needed={groups_needed}",
                f"pair_needed={pair_needed}",
                f"existing_valid={existing_valid}",
            ),
        )

    state = _state_from_labels(labels)
    allow_1510 = bool(_section(rules, "rules").get("allow_1510", False))
    wildcard_enabled = bool(_section(rules, "wildcard").get("enabled", False))
    if state[-1] and not wildcard_enabled:
        partition = None
    else:
        if pair_needed:
            partition = _solve_hu_partition_with_pair(
                state,
                triple_groups_needed,
                _xi_signature(rules),
                allow_1510,
            )
        else:
            partition = _solve_hu_partition(
                state,
                triple_groups_needed,
                _xi_signature(rules),
                allow_1510,
            )
    hand_groups = partition[1] if partition is not None else ()
    hand_xi = partition[0] if partition is not None else 0
    total_xi = existing_xi + hand_xi
    complete = partition is not None and len(hand_groups) == groups_needed and existing_valid
    can_hu = complete and total_xi >= minimum
    red_count = _red_count(labels, existing)
    red_black_kind, special_points = _red_black_score(red_count, rules)
    score = _final_score(total_xi, minimum, special_points, rules) if can_hu else 0.0
    reasons = [
        f"hand_xi={hand_xi}",
        f"existing_xi={existing_xi}",
        f"group_count={len(hand_groups) + len(existing)}",
    ]
    if not complete:
        reasons.append("no_complete_independent_partition")
    elif total_xi < minimum:
        reasons.append(f"xi_below_minimum:{total_xi}<{minimum}")
    else:
        reasons.append("independent_oracle_hu")
    return OracleHuResult(
        can_hu=can_hu,
        partition_complete=complete,
        total_xi=total_xi,
        min_xi=minimum,
        group_count=len(hand_groups) + len(existing),
        hand_groups=hand_groups,
        existing_xi=existing_xi,
        red_count=red_count,
        red_black_kind=red_black_kind if can_hu else None,
        score=score,
        reasons=tuple(reasons),
    )


def enumerate_chi_plans_oracle(
    hand: Sequence[str],
    external_label: str,
    *,
    allow_1510: bool = False,
) -> list[OracleChiPlan]:
    """Enumerate CHI and mandatory compare groups without production helpers."""

    normalized = [str(label) for label in hand]
    external = str(external_label)
    if (
        external not in NORMAL_LABELS
        or any(label not in LABEL_INDEX for label in normalized)
        or normalized.count(external) >= 3
    ):
        return []

    locked = {label for label, amount in Counter(normalized).items() if label != external and amount >= 3}
    pool = Counter(normalized)
    pool[external] += 1
    templates = _chi_templates(allow_1510)
    initial_options = _chi_groups_available(pool, external, templates, locked, max_external_count=3)
    plans: list[OracleChiPlan] = []
    seen: set[tuple[Any, ...]] = set()
    for initial in initial_options:
        remaining = _subtract_counter(pool, initial)
        if remaining is None:
            continue
        for compares in _complete_compare_groups_oracle(
            remaining,
            external,
            templates,
            locked,
        ):
            groups = (initial, *compares)
            consumed = [label for group in groups for label in group]
            consumed.remove(external)
            plan = OracleChiPlan(
                initial_group=initial,
                compare_groups=compares,
                consumed_from_hand=tuple(sorted(consumed, key=_label_sort_key)),
            )
            key = _chi_plan_key(plan)
            if key in seen:
                continue
            seen.add(key)
            plans.append(plan)
    return sorted(plans, key=lambda plan: (len(plan.compare_groups), _chi_plan_key(plan)))


def legal_discard_labels_oracle(hand: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(str(label) for label in hand)
    if any(label not in LABEL_INDEX for label in labels):
        raise ValueError("invalid_discard_hand_label")
    counts = Counter(labels)
    return tuple(
        sorted(
            (
                label
                for label, amount in counts.items()
                if label != WILD_LABEL and amount < 3
            ),
            key=_label_sort_key,
        )
    )


def legal_response_actions_oracle(
    hand: Sequence[str],
    existing_melds: Sequence[Any] | None,
    pending_label: str,
    *,
    seat: int,
    source_seat: int,
    players: int,
    rules: Mapping[str, Any],
    passed_chi: Iterable[str] = (),
    passed_peng: Iterable[str] = (),
    blocked_auto_claim: bool = False,
    automatic_pao_seat: int | None = None,
) -> OracleResponseActions:
    if players not in {2, 3}:
        raise ValueError("players must be 2 or 3")
    if not 0 <= seat < players or not 0 <= source_seat < players:
        raise ValueError("response seat out of range")
    if seat == source_seat:
        raise ValueError("response seat cannot be source seat")
    pending = str(pending_label)
    normalized_hand = tuple(str(label) for label in hand)
    if pending not in LABEL_INDEX or any(
        label not in LABEL_INDEX
        for label in normalized_hand
    ):
        raise ValueError("invalid_response_label")
    melds = tuple(existing_melds or ())
    automatic_pao = (
        automatic_pao_seat
        if automatic_pao_seat is not None
        else _automatic_pao_seat_oracle(
            melds_by_seat={seat: melds},
            pending=pending,
            players=players,
            source_seat=source_seat,
            blocked_seats={seat} if blocked_auto_claim else set(),
        )
    )
    actions: list[OracleAction] = []
    hu = None
    if not blocked_auto_claim:
        hu = evaluate_hu_oracle(
            (*normalized_hand, pending),
            melds,
            rules,
        )
        if hu.can_hu:
            actions.append(OracleAction(type="HU", label=pending))
    if automatic_pao is None:
        if (
            pending != WILD_LABEL
            and normalized_hand.count(pending) == 2
            and pending not in set(passed_peng)
        ):
            actions.append(
                OracleAction(
                    type="PENG",
                    label=pending,
                    consumed_from_hand=(pending, pending),
                    meld_groups=((pending, pending, pending),),
                )
            )
        if (
            pending != WILD_LABEL
            and seat == (source_seat + 1) % players
            and pending not in set(passed_chi)
        ):
            actions.extend(
                OracleAction(
                    type="CHI",
                    label=pending,
                    consumed_from_hand=plan.consumed_from_hand,
                    meld_groups=plan.groups,
                )
                for plan in enumerate_chi_plans_oracle(
                    normalized_hand,
                    pending,
                    allow_1510=bool(
                        _section(rules, "rules").get(
                            "allow_1510",
                            False,
                        )
                    ),
                )
            )
    if actions:
        actions.insert(0, OracleAction(type="PASS"))
    return OracleResponseActions(
        actions=tuple(actions),
        automatic_action="PAO" if automatic_pao is not None else None,
        automatic_seat=automatic_pao,
    )


def automatic_pao_seat_oracle(
    melds_by_seat: Mapping[int, Sequence[Any]],
    pending_label: str,
    *,
    players: int,
    source_seat: int,
    blocked_seats: Iterable[int] = (),
) -> int | None:
    return _automatic_pao_seat_oracle(
        melds_by_seat=melds_by_seat,
        pending=str(pending_label),
        players=players,
        source_seat=source_seat,
        blocked_seats=set(blocked_seats),
    )


def apply_response_action_oracle(
    hand: Sequence[str],
    existing_melds: Sequence[Any] | None,
    pending_label: str,
    action: OracleAction,
    *,
    legal_actions: Sequence[OracleAction] | None = None,
    passed_chi: Iterable[str] = (),
    passed_peng: Iterable[str] = (),
) -> OracleResponseTransition:
    pending = str(pending_label)
    normalized_hand = [str(label) for label in hand]
    melds = [
        OracleMeldState(
            kind=_normalized_meld_kind(kind),
            labels=tuple(labels),
        )
        for labels, kind in (
            _meld_fields(meld)
            for meld in (existing_melds or ())
        )
    ]
    action_type = action.type.upper()
    legal_types = {
        item.type.upper()
        for item in (legal_actions or (action,))
    }
    next_passed_chi = set(str(label) for label in passed_chi)
    next_passed_peng = set(str(label) for label in passed_peng)
    if action_type != "HU":
        if "PENG" in legal_types and action_type != "PENG":
            next_passed_peng.add(pending)
        if "CHI" in legal_types and action_type != "CHI":
            next_passed_chi.add(pending)
    if action_type in {"PASS", "HU"}:
        return OracleResponseTransition(
            hand=tuple(sorted(normalized_hand, key=_label_sort_key)),
            melds=tuple(melds),
            claim_type=action_type.lower(),
            terminal=action_type == "HU",
            passed_chi=tuple(sorted(next_passed_chi, key=_label_sort_key)),
            passed_peng=tuple(sorted(next_passed_peng, key=_label_sort_key)),
        )
    if action_type not in {"PENG", "CHI"}:
        raise ValueError(f"unsupported_oracle_response_action:{action_type}")
    for label in action.consumed_from_hand:
        if label not in normalized_hand:
            raise ValueError("oracle_response_consumption_unavailable")
        normalized_hand.remove(label)
    if action_type == "PENG":
        if (
            action.consumed_from_hand != (pending, pending)
            or action.meld_groups != ((pending, pending, pending),)
        ):
            raise ValueError("oracle_peng_payload_invalid")
        melds.append(
            OracleMeldState(
                kind="peng",
                labels=(pending, pending, pending),
            )
        )
    else:
        if not action.meld_groups:
            raise ValueError("oracle_chi_groups_missing")
        melds.extend(
            OracleMeldState(
                kind=_chi_group_kind_oracle(group),
                labels=tuple(group),
            )
            for group in action.meld_groups
        )
    return OracleResponseTransition(
        hand=tuple(sorted(normalized_hand, key=_label_sort_key)),
        melds=tuple(melds),
        claim_type=action_type.lower(),
        terminal=False,
        passed_chi=tuple(sorted(next_passed_chi, key=_label_sort_key)),
        passed_peng=tuple(sorted(next_passed_peng, key=_label_sort_key)),
    )


def apply_automatic_pao_oracle(
    hand: Sequence[str],
    existing_melds: Sequence[Any],
    pending_label: str,
) -> OracleResponseTransition:
    pending = str(pending_label)
    melds: list[OracleMeldState] = []
    upgraded = False
    for meld in existing_melds:
        labels, kind = _meld_fields(meld)
        normalized_kind = _normalized_meld_kind(kind)
        if (
            not upgraded
            and normalized_kind in {"peng", "wei"}
            and len(labels) == 3
            and len(set(labels)) == 1
            and labels[0] == pending
        ):
            melds.append(
                OracleMeldState(
                    kind="pao",
                    labels=(pending,) * 4,
                )
            )
            upgraded = True
        else:
            melds.append(
                OracleMeldState(
                    kind=normalized_kind,
                    labels=tuple(labels),
                )
            )
    if not upgraded:
        raise ValueError("oracle_pao_source_meld_missing")
    return OracleResponseTransition(
        hand=tuple(sorted((str(label) for label in hand), key=_label_sort_key)),
        melds=tuple(melds),
        claim_type="pao",
        terminal=False,
    )


def apply_draw_auto_meld_oracle(
    hand_before_draw: Sequence[str],
    existing_melds: Sequence[Any],
    drawn_label: str,
    *,
    quad_events: int = 0,
) -> OracleDrawTransition:
    drawn = str(drawn_label)
    hand = [str(label) for label in hand_before_draw]
    if drawn not in LABEL_INDEX or any(label not in LABEL_INDEX for label in hand):
        raise ValueError("invalid_draw_transition_label")
    hand.append(drawn)
    melds = [
        OracleMeldState(
            kind=_normalized_meld_kind(kind),
            labels=tuple(labels),
        )
        for labels, kind in (
            _meld_fields(meld)
            for meld in existing_melds
        )
    ]
    action = None
    next_quad_events = int(quad_events)
    if drawn != WILD_LABEL:
        for index, meld in enumerate(melds):
            if (
                meld.kind in {"peng", "wei"}
                and len(meld.labels) == 3
                and len(set(meld.labels)) == 1
                and meld.labels[0] == drawn
            ):
                hand.remove(drawn)
                melds[index] = OracleMeldState(
                    kind="pao",
                    labels=(drawn,) * 4,
                )
                next_quad_events += 1
                action = "PAO"
                break
        if action is None and hand.count(drawn) >= 4:
            for _ in range(4):
                hand.remove(drawn)
            melds.append(
                OracleMeldState(
                    kind="ti",
                    labels=(drawn,) * 4,
                )
            )
            next_quad_events += 1
            action = "TI"
        elif action is None and hand.count(drawn) == 3:
            for _ in range(3):
                hand.remove(drawn)
            melds.append(
                OracleMeldState(
                    kind="wei",
                    labels=(drawn,) * 3,
                )
            )
            action = "WEI"
    return OracleDrawTransition(
        hand=tuple(sorted(hand, key=_label_sort_key)),
        melds=tuple(melds),
        automatic_action=action,
        quad_events=next_quad_events,
        skip_discard=(
            action in {"PAO", "TI"}
            and next_quad_events >= 2
        ),
    )


def hu_templates(*, allow_1510: bool = False) -> tuple[tuple[str, str, str], ...]:
    """Expose resolved HU templates for deterministic audit-case generation."""

    return tuple(template.labels for template in _hu_templates(allow_1510))


def _section(rules: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = rules.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _automatic_pao_seat_oracle(
    *,
    melds_by_seat: Mapping[int, Sequence[Any]],
    pending: str,
    players: int,
    source_seat: int,
    blocked_seats: set[int],
) -> int | None:
    if pending == WILD_LABEL:
        return None
    for offset in range(1, players):
        seat = (source_seat + offset) % players
        if seat in blocked_seats:
            continue
        for meld in melds_by_seat.get(seat, ()):
            labels, kind = _meld_fields(meld)
            normalized_kind = {
                "hidden_triplet": "wei",
                "exact_triplet": "wei",
            }.get(kind, kind)
            if (
                normalized_kind in {"peng", "wei"}
                and len(labels) == 3
                and len(set(labels)) == 1
                and labels[0] == pending
            ):
                return seat
    return None


def _normalized_meld_kind(kind: str) -> str:
    return {
        "hidden_quad": "ti",
        "quad": "pao",
        "hidden_triplet": "wei",
        "exact_triplet": "wei",
        "sequence": "normal_sequence",
    }.get(str(kind), str(kind))


def _chi_group_kind_oracle(group: Sequence[str]) -> str:
    labels = tuple(str(label) for label in group)
    if len(labels) != 3 or any(label not in NORMAL_LABELS for label in labels):
        raise ValueError("oracle_chi_group_invalid")
    suits_and_ranks = [_suit_rank(label) for label in labels]
    suits = {item[0] for item in suits_and_ranks}
    ranks = sorted(item[1] for item in suits_and_ranks)
    if len(suits) == 2 and len(set(ranks)) == 1:
        return "mixed_same_rank_triplet"
    if len(suits) != 1:
        raise ValueError("oracle_chi_group_mixed_suit_invalid")
    if ranks == [1, 2, 3]:
        return "special_123"
    if ranks == [2, 7, 10]:
        return "special_2710"
    if ranks == [1, 5, 10]:
        return "special_1510"
    if ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1:
        return "normal_sequence"
    raise ValueError("oracle_chi_group_shape_invalid")


def _state_from_labels(labels: Iterable[str]) -> tuple[int, ...]:
    counts = Counter(labels)
    return tuple(counts.get(label, 0) for label in (*NORMAL_LABELS, WILD_LABEL))


def _xi_signature(rules: Mapping[str, Any]) -> tuple[int, ...]:
    xi = _section(rules, "xi")
    values: list[int] = []
    for kind in ("ti", "pao", "wei", "peng", "special_123", "special_2710"):
        section = xi.get(kind, {})
        if not isinstance(section, Mapping):
            section = {}
        values.extend((int(section.get("small", 0)), int(section.get("big", 0))))
    wildcard_counts = int(bool(_section(rules, "wildcard").get("wildcard_groups_count_xi", True)))
    return (*values, wildcard_counts)


def _xi_from_signature(kind: str, suit: str | None, signature: tuple[int, ...]) -> int:
    if suit not in {"small", "big"}:
        return 0
    index_by_kind = {
        "ti": 0,
        "pao": 2,
        "wei": 4,
        "peng": 6,
        "special_123": 8,
        "special_2710": 10,
    }
    offset = index_by_kind.get(kind)
    if offset is None:
        return 0
    return int(signature[offset + int(suit == "big")])


@lru_cache(maxsize=300_000)
def _solve_hu_partition(
    state: tuple[int, ...],
    groups_needed: int,
    xi_signature: tuple[int, ...],
    allow_1510: bool,
) -> tuple[int, tuple[OracleGroup, ...]] | None:
    if groups_needed == 0:
        return (0, ()) if not any(state) else None
    if sum(state) != groups_needed * 3:
        return None

    pivot_index = next((index for index, amount in enumerate(state[:-1]) if amount), None)
    if pivot_index is None:
        if state[-1] != groups_needed * 3:
            return None
        template = max(
            _hu_templates(allow_1510),
            key=lambda item: _template_xi(item, xi_signature, wildcards_used=3),
        )
        xi = _template_xi(template, xi_signature, wildcards_used=3)
        group = OracleGroup(
            source_labels=(WILD_LABEL, WILD_LABEL, WILD_LABEL),
            resolved_labels=template.labels,
            kind=template.kind,
            suit=template.suit,
            xi=xi,
            wildcards_used=3,
        )
        return xi * groups_needed, (group,) * groups_needed

    pivot = NORMAL_LABELS[pivot_index]
    best: tuple[int, tuple[OracleGroup, ...]] | None = None
    for template in _hu_templates(allow_1510):
        pattern = Counter(template.labels)
        if pivot not in pattern:
            continue
        for real_use in _real_consumption_options(pattern, state, pivot):
            real_total = sum(real_use.values())
            wildcards_used = 3 - real_total
            if wildcards_used < 0 or wildcards_used > state[-1]:
                continue
            next_state = list(state)
            for label, amount in real_use.items():
                next_state[LABEL_INDEX[label]] -= amount
            next_state[-1] -= wildcards_used
            child = _solve_hu_partition(
                tuple(next_state),
                groups_needed - 1,
                xi_signature,
                allow_1510,
            )
            if child is None:
                continue
            source = [
                label
                for label in NORMAL_LABELS
                for _ in range(real_use.get(label, 0))
            ]
            source.extend([WILD_LABEL] * wildcards_used)
            xi = _template_xi(template, xi_signature, wildcards_used=wildcards_used)
            group = OracleGroup(
                source_labels=tuple(sorted(source, key=_label_sort_key)),
                resolved_labels=template.labels,
                kind=template.kind,
                suit=template.suit,
                xi=xi,
                wildcards_used=wildcards_used,
            )
            candidate = (xi + child[0], (group, *child[1]))
            if best is None or _partition_key(candidate) > _partition_key(best):
                best = candidate
    return best


def _solve_hu_partition_with_pair(
    state: tuple[int, ...],
    triple_groups_needed: int,
    xi_signature: tuple[int, ...],
    allow_1510: bool,
) -> tuple[int, tuple[OracleGroup, ...]] | None:
    if sum(state) != triple_groups_needed * 3 + 2:
        return None
    pair_options: list[
        tuple[tuple[str, ...], tuple[str, ...], str | None, int]
    ] = []
    wildcard_count = state[-1]
    for index, label in enumerate(NORMAL_LABELS):
        amount = state[index]
        suit = _suit_rank(label)[0]
        if amount >= 2:
            pair_options.append(((label, label), (label, label), suit, 0))
        if amount >= 1 and wildcard_count >= 1:
            pair_options.append(((label, WILD_LABEL), (label, label), suit, 1))
    if wildcard_count >= 2:
        pair_options.append(
            (
                (WILD_LABEL, WILD_LABEL),
                (WILD_LABEL, WILD_LABEL),
                None,
                2,
            )
        )
    best: tuple[int, tuple[OracleGroup, ...]] | None = None
    for source, resolved, suit, wildcards_used in pair_options:
        next_state = list(state)
        for label in source:
            next_state[LABEL_INDEX[label]] -= 1
        child = _solve_hu_partition(
            tuple(next_state),
            triple_groups_needed,
            xi_signature,
            allow_1510,
        )
        if child is None:
            continue
        pair = OracleGroup(
            source_labels=source,
            resolved_labels=resolved,
            kind="pair",
            suit=suit,
            xi=0,
            wildcards_used=wildcards_used,
        )
        candidate = (child[0], (pair, *child[1]))
        if best is None or _partition_key(candidate) > _partition_key(best):
            best = candidate
    return best


def _real_consumption_options(
    pattern: Counter[str],
    state: tuple[int, ...],
    pivot: str,
) -> Iterable[Counter[str]]:
    labels = tuple(pattern)
    ranges = [
        range(0, min(pattern[label], state[LABEL_INDEX[label]]) + 1)
        for label in labels
    ]
    for amounts in product(*ranges):
        use = Counter(dict(zip(labels, amounts)))
        if use[pivot] <= 0:
            continue
        if sum(use.values()) > 3:
            continue
        yield +use


def _partition_key(
    value: tuple[int, tuple[OracleGroup, ...]],
) -> tuple[int, tuple[tuple[Any, ...], ...]]:
    return (
        value[0],
        tuple(
            (
                group.xi,
                group.kind,
                tuple(_label_sort_key(label) for label in group.resolved_labels),
            )
            for group in value[1]
        ),
    )


def _template_xi(
    template: _Template,
    signature: tuple[int, ...],
    *,
    wildcards_used: int,
) -> int:
    if wildcards_used and not signature[-1]:
        return 0
    kind = "wei" if template.kind == "exact_triplet" else template.kind
    return _xi_from_signature(kind, template.suit, signature)


@lru_cache(maxsize=2)
def _hu_templates(allow_1510: bool) -> tuple[_Template, ...]:
    result: list[_Template] = []
    for suit, labels in (("small", SMALL_LABELS), ("big", BIG_LABELS)):
        for start in range(0, 8):
            group = tuple(labels[start : start + 3])
            kind = "special_123" if start == 0 else "normal_sequence"
            result.append(_Template(group, kind, suit))
        result.append(_Template((labels[1], labels[6], labels[9]), "special_2710", suit))
        if allow_1510:
            result.append(_Template((labels[0], labels[4], labels[9]), "special_1510", suit))
        for label in labels:
            result.append(_Template((label, label, label), "exact_triplet", suit))
    for rank in range(10):
        small = SMALL_LABELS[rank]
        big = BIG_LABELS[rank]
        result.append(_Template((small, small, big), "mixed_same_rank", None))
        result.append(_Template((small, big, big), "mixed_same_rank", None))
    return tuple(_dedupe_templates(result))


@lru_cache(maxsize=2)
def _chi_templates(allow_1510: bool) -> tuple[tuple[str, str, str], ...]:
    allowed = {
        "normal_sequence",
        "special_123",
        "special_2710",
        "mixed_same_rank",
    }
    if allow_1510:
        allowed.add("special_1510")
    groups = [
        template.labels
        for template in _hu_templates(allow_1510)
        if template.kind in allowed
    ]
    return tuple(dict.fromkeys(tuple(sorted(group, key=_label_sort_key)) for group in groups))


def _dedupe_templates(templates: Iterable[_Template]) -> list[_Template]:
    result: list[_Template] = []
    seen: set[tuple[Any, ...]] = set()
    for template in templates:
        normalized = tuple(sorted(template.labels, key=_label_sort_key))
        key = (normalized, template.kind, template.suit)
        if key in seen:
            continue
        seen.add(key)
        result.append(_Template(normalized, template.kind, template.suit))
    return result


def _existing_meld_xi(
    melds: Sequence[Any],
    rules: Mapping[str, Any],
) -> tuple[int, bool]:
    total = 0
    valid = True
    signature = _xi_signature(rules)
    for meld in melds:
        labels, kind = _meld_fields(meld)
        suit = _common_suit(labels)
        normalized_kind = {
            "hidden_quad": "ti",
            "quad": "pao",
            "hidden_triplet": "wei",
            "exact_triplet": "wei",
            "sequence": "normal_sequence",
            "mixed_same_rank_triplet": "mixed_same_rank",
        }.get(kind, kind)
        legal = _existing_meld_is_legal(labels, normalized_kind)
        valid = valid and legal
        total += _xi_from_signature(normalized_kind, suit, signature) if legal else 0
    return total, valid


def _red_count(hand: Sequence[str], melds: Sequence[Any]) -> int:
    count = sum(label in RED_LABELS for label in hand)
    for meld in melds:
        labels, _kind = _meld_fields(meld)
        count += sum(label in RED_LABELS for label in labels)
    return count


def _red_black_score(
    red_count: int,
    rules: Mapping[str, Any],
) -> tuple[str | None, float]:
    rule_section = _section(rules, "rules")
    scoring = _section(rules, "scoring")
    mode = str(rule_section.get("red_black_mode", "red_black_mingtang")).strip().lower()
    if mode in {"none", "off", "disabled"}:
        return None, 0.0
    if red_count == int(scoring.get("black_hu_red_count", 0)):
        kind = "black_hu"
    elif red_count == int(scoring.get("one_red_hu_red_count", 1)):
        kind = "one_red_hu"
    elif red_count >= int(scoring.get("red_hu_min_red", 13)):
        kind = "red_hu"
    else:
        return None, 0.0
    by_kind = scoring.get("red_black_special_points", {})
    if not isinstance(by_kind, Mapping):
        by_kind = {}
    points = float(by_kind.get(kind, scoring.get("red_black_special_point", 1)))
    multiplier = float(scoring.get("red_black_point_multiplier", 1))
    if mode in {"red_black_point_double", "red_black_point_2x", "red_black_double", "double"}:
        multiplier = float(scoring.get("red_black_double_multiplier", 2))
    return kind, points * multiplier


def _final_score(
    total_xi: int,
    minimum: int,
    special_points: float,
    rules: Mapping[str, Any],
) -> float:
    rule_section = _section(rules, "rules")
    base_tun = int(rule_section.get("base_tun_at_9_xi", 1))
    conversion = str(rule_section.get("xi_to_tun", "3_to_1"))
    step = 3 if conversion == "3_to_1" else 1
    tun = base_tun + max(0, total_xi - minimum) // step
    return float(tun + special_points)


def _meld_fields(meld: Any) -> tuple[tuple[str, ...], str]:
    if isinstance(meld, Mapping):
        labels = meld.get("labels") or meld.get("cards") or ()
        kind = meld.get("kind") or meld.get("type") or ""
    else:
        labels = getattr(meld, "cards", getattr(meld, "labels", ()))
        kind = getattr(meld, "kind", getattr(meld, "type", ""))
    return tuple(str(label) for label in labels), str(kind)


def _existing_meld_is_legal(labels: tuple[str, ...], kind: str) -> bool:
    if any(label not in NORMAL_LABELS for label in labels):
        return False
    counts = Counter(labels)
    if kind in {"ti", "pao"}:
        return len(labels) == 4 and len(counts) == 1
    if kind in {"wei", "peng"}:
        return len(labels) == 3 and len(counts) == 1
    template_kinds = {
        template.labels: template.kind
        for template in _hu_templates(False)
    }
    normalized = tuple(sorted(labels, key=_label_sort_key))
    actual = template_kinds.get(normalized)
    return actual == kind or (
        kind == "normal_sequence"
        and actual in {"normal_sequence", "special_123"}
    )


def _common_suit(labels: Sequence[str]) -> str | None:
    suits = {_suit_rank(label)[0] for label in labels if label in NORMAL_LABELS}
    return next(iter(suits)) if len(suits) == 1 else None


def _complete_compare_groups_oracle(
    counts: Counter[str],
    external: str,
    templates: tuple[tuple[str, str, str], ...],
    locked: set[str],
) -> list[tuple[tuple[str, str, str], ...]]:
    if counts.get(external, 0) <= 0:
        return [()]
    groups = _chi_groups_available(
        counts,
        external,
        templates,
        locked,
        max_external_count=2,
    )
    result: list[tuple[tuple[str, str, str], ...]] = []
    for group in groups:
        remaining = _subtract_counter(counts, group)
        if remaining is None:
            continue
        for suffix in _complete_compare_groups_oracle(
            remaining,
            external,
            templates,
            locked,
        ):
            result.append((group, *suffix))
    return result


def _chi_groups_available(
    counts: Counter[str],
    external: str,
    templates: tuple[tuple[str, str, str], ...],
    locked: set[str],
    *,
    max_external_count: int,
) -> list[tuple[str, str, str]]:
    if counts.get(external, 0) <= 0 or counts.get(external, 0) > max_external_count:
        return []
    result: list[tuple[str, str, str]] = []
    for group in templates:
        needed = Counter(group)
        if external not in needed:
            continue
        if any(label in locked and needed[label] for label in needed if label != external):
            continue
        if any(counts.get(label, 0) < amount for label, amount in needed.items()):
            continue
        result.append(group)
    return result


def _subtract_counter(
    counts: Counter[str],
    labels: Sequence[str],
) -> Counter[str] | None:
    needed = Counter(labels)
    if any(counts.get(label, 0) < amount for label, amount in needed.items()):
        return None
    remaining = counts.copy()
    remaining.subtract(needed)
    return +remaining


def _chi_plan_key(plan: OracleChiPlan) -> tuple[Any, ...]:
    return (
        tuple(sorted(plan.initial_group, key=_label_sort_key)),
        tuple(
            sorted(
                (tuple(sorted(group, key=_label_sort_key)) for group in plan.compare_groups),
                key=lambda group: tuple(_label_sort_key(label) for label in group),
            )
        ),
        tuple(sorted(plan.consumed_from_hand, key=_label_sort_key)),
    )


def _suit_rank(label: str) -> tuple[str, int]:
    if label in SMALL_LABELS:
        return "small", SMALL_LABELS.index(label) + 1
    if label in BIG_LABELS:
        return "big", BIG_LABELS.index(label) + 1
    raise ValueError(f"unknown normal label: {label}")


def _label_sort_key(label: str) -> tuple[int, int, str]:
    if label == WILD_LABEL:
        return (99, 99, label)
    suit, rank = _suit_rank(label)
    return (rank, 0 if suit == "small" else 1, label)


__all__ = [
    "BIG_LABELS",
    "NORMAL_LABELS",
    "OracleAction",
    "OracleChiPlan",
    "OracleDrawTransition",
    "OracleGroup",
    "OracleHuResult",
    "OracleMeldState",
    "OracleRoomShape",
    "OracleResponseActions",
    "OracleResponseTransition",
    "RED_LABELS",
    "SMALL_LABELS",
    "WILD_LABEL",
    "apply_automatic_pao_oracle",
    "apply_draw_auto_meld_oracle",
    "apply_response_action_oracle",
    "automatic_pao_seat_oracle",
    "enumerate_chi_plans_oracle",
    "evaluate_hu_oracle",
    "hu_templates",
    "legal_discard_labels_oracle",
    "legal_response_actions_oracle",
    "room_shape_oracle",
]
