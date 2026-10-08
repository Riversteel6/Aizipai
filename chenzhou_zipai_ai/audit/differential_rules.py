"""Differential checks between the independent oracle and production rules."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ai.full_game_simulator import (
    BaselinePolicy,
    FullGameSimulator,
    SimMeld,
    SimPlayer,
    _discardable_labels,
    evaluate_hu,
)
from audit.independent_rules import (
    NORMAL_LABELS,
    WILD_LABEL,
    apply_automatic_pao_oracle,
    apply_draw_auto_meld_oracle,
    apply_response_action_oracle,
    automatic_pao_seat_oracle,
    enumerate_chi_plans_oracle,
    evaluate_hu_oracle,
    hu_templates,
    legal_discard_labels_oracle,
    legal_response_actions_oracle,
    room_shape_oracle,
)
from engine.chi_rules import enumerate_chi_plans
from engine.deck import expanded_deck, full_deck_counts
from engine.rules import rules_for_room


@dataclass(frozen=True)
class RuleMismatch:
    domain: str
    index: int
    payload: dict[str, Any]
    oracle: dict[str, Any]
    production: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "index": self.index,
            "payload": self.payload,
            "oracle": self.oracle,
            "production": self.production,
        }


def run_rule_crosscheck(
    *,
    hu_cases: int = 500,
    chi_cases: int = 500,
    discard_cases: int = 500,
    response_cases: int = 500,
    draw_auto_cases: int = 500,
    seed: int = 20260726,
    wildcard_enabled: bool,
    players: int = 3,
) -> dict[str, Any]:
    rules = rules_for_room(
        wildcard_enabled=wildcard_enabled,
        players=players,
    )
    rng = random.Random(seed)
    mismatches: list[RuleMismatch] = []
    oracle_room = room_shape_oracle(
        players=players,
        wildcard_enabled=wildcard_enabled,
        deck_copies=int(rules.get("game", {}).get("deck_copies", 4)),
        wildcard_copies=int(rules.get("wildcard", {}).get("copies", 4)),
    )
    production_deck_size = sum(full_deck_counts(rules=rules).values())
    production_room = {
        "players": int(rules.get("game", {}).get("players", 0)),
        "wildcard_enabled": bool(rules.get("wildcard", {}).get("enabled", False)),
        "deck_size": production_deck_size,
        "initial_dealt": players * 20 + 1,
        "initial_stock": production_deck_size - (players * 20 + 1),
    }
    if oracle_room.to_dict() != production_room:
        mismatches.append(
            RuleMismatch(
                domain="room",
                index=0,
                payload={"players": players, "wildcard_enabled": wildcard_enabled},
                oracle=oracle_room.to_dict(),
                production=production_room,
            )
        )
    hu_complete_cases = 0
    hu_positive_cases = 0
    for index in range(1, hu_cases + 1):
        hand, existing_melds, source = _generate_hu_case(
            rng,
            wildcard_enabled=wildcard_enabled,
        )
        oracle = evaluate_hu_oracle(hand, existing_melds, rules)
        production = evaluate_hu(hand, existing_melds, rules)
        production_complete = (
            len(production.groups) + len(existing_melds)
            == int(rules["rules"]["required_meld_groups"])
        )
        hu_complete_cases += int(oracle.partition_complete or production_complete)
        hu_positive_cases += int(oracle.can_hu or production.can_hu)
        differs = (
            oracle.partition_complete != production_complete
            or oracle.can_hu != production.can_hu
            or (
                oracle.partition_complete
                and production_complete
                and oracle.total_xi != production.total_xi
            )
            or (
                oracle.can_hu
                and production.can_hu
                and (
                    oracle.red_count != production.red_count
                    or abs(oracle.score - production.score) > 1e-9
                )
            )
        )
        if differs:
            mismatches.append(
                RuleMismatch(
                    domain="hu",
                    index=index,
                    payload={
                        "hand": hand,
                        "existing_melds": [
                            {"kind": meld.kind, "cards": list(meld.cards)}
                            for meld in existing_melds
                        ],
                        "source": source,
                    },
                    oracle=oracle.to_dict(),
                    production={
                        "can_hu": production.can_hu,
                        "partition_complete": production_complete,
                        "total_xi": production.total_xi,
                        "groups": [list(group) for group in production.groups],
                        "red_count": production.red_count,
                        "score": production.score,
                    },
                )
            )

    chi_positive_cases = 0
    for index in range(1, chi_cases + 1):
        hand, external, source = _generate_chi_case(rng)
        oracle_plans = enumerate_chi_plans_oracle(
            hand,
            external,
            allow_1510=bool(rules["rules"].get("allow_1510", False)),
        )
        production_plans = enumerate_chi_plans(
            hand,
            external,
            allow_1510=bool(rules["rules"].get("allow_1510", False)),
        )
        chi_positive_cases += int(bool(oracle_plans or production_plans))
        oracle_keys = {_plan_key(plan) for plan in oracle_plans}
        production_keys = {_plan_key(plan) for plan in production_plans}
        if oracle_keys != production_keys:
            mismatches.append(
                RuleMismatch(
                    domain="chi",
                    index=index,
                    payload={
                        "hand": hand,
                        "external": external,
                        "source": source,
                    },
                    oracle={
                        "plan_count": len(oracle_plans),
                        "plans": [plan.to_dict() for plan in oracle_plans],
                        "only_signatures": sorted(repr(key) for key in oracle_keys - production_keys),
                    },
                    production={
                        "plan_count": len(production_plans),
                        "plans": [plan.to_dict() for plan in production_plans],
                        "only_signatures": sorted(repr(key) for key in production_keys - oracle_keys),
                    },
                )
            )

    discard_locked_cases = 0
    for index in range(1, discard_cases + 1):
        hand, source = _generate_discard_case(
            rng,
            wildcard_enabled=wildcard_enabled,
        )
        oracle_labels = legal_discard_labels_oracle(hand)
        production_labels = tuple(
            sorted(
                _discardable_labels(hand),
                key=_sort_key,
            )
        )
        counts = Counter(hand)
        discard_locked_cases += int(
            any(amount >= 3 for amount in counts.values())
            or WILD_LABEL in counts
        )
        if oracle_labels != production_labels:
            mismatches.append(
                RuleMismatch(
                    domain="discard",
                    index=index,
                    payload={"hand": hand, "source": source},
                    oracle={"labels": list(oracle_labels)},
                    production={"labels": list(production_labels)},
                )
            )

    response_positive_cases = 0
    response_joint_cases = 0
    response_auto_pao_cases = 0
    response_transition_cases = 0
    for index in range(1, response_cases + 1):
        hand, melds, pending, source = _generate_response_case(
            rng,
            wildcard_enabled=wildcard_enabled,
        )
        source_seat = players - 1
        seat = 0
        melds_by_seat = {
            player_seat: (
                melds
                if player_seat == seat
                else []
            )
            for player_seat in range(players)
        }
        oracle_pao_seat = automatic_pao_seat_oracle(
            melds_by_seat,
            pending,
            players=players,
            source_seat=source_seat,
        )
        oracle_response = legal_response_actions_oracle(
            hand,
            melds,
            pending,
            seat=seat,
            source_seat=source_seat,
            players=players,
            rules=rules,
            automatic_pao_seat=oracle_pao_seat,
        )
        production_response = _production_response_actions(
            hand=hand,
            melds=melds,
            pending=pending,
            seat=seat,
            source_seat=source_seat,
            players=players,
            rules=rules,
        )
        oracle_signatures = {
            _action_signature(action)
            for action in oracle_response.actions
        }
        production_signatures = {
            _action_signature(action)
            for action in production_response["actions"]
        }
        action_types = {
            signature[0]
            for signature in oracle_signatures | production_signatures
        }
        response_positive_cases += int(bool(action_types))
        response_joint_cases += int(
            len(action_types & {"HU", "PENG", "CHI"}) >= 2
        )
        response_auto_pao_cases += int(
            oracle_response.automatic_action == "PAO"
            or production_response["automatic_action"] == "PAO"
        )
        if (
            oracle_response.automatic_action
            != production_response["automatic_action"]
            or oracle_response.automatic_seat
            != production_response["automatic_seat"]
            or oracle_signatures != production_signatures
        ):
            mismatches.append(
                RuleMismatch(
                    domain="response",
                    index=index,
                    payload={
                        "hand": hand,
                        "existing_melds": [
                            {
                                "kind": meld.kind,
                                "cards": list(meld.cards),
                            }
                            for meld in melds
                        ],
                        "pending": pending,
                        "source": source,
                    },
                    oracle={
                        **oracle_response.to_dict(),
                        "signatures": sorted(
                            repr(item)
                            for item in oracle_signatures
                        ),
                    },
                    production={
                        **production_response,
                        "signatures": sorted(
                            repr(item)
                            for item in production_signatures
                        ),
                    },
                )
            )
            continue
        for action in oracle_response.actions:
            if (
                oracle_response.automatic_action == "PAO"
                and action.type == "PASS"
            ):
                continue
            oracle_transition = apply_response_action_oracle(
                hand,
                melds,
                pending,
                action,
                legal_actions=oracle_response.actions,
            )
            production_transition = _production_response_transition(
                hand=hand,
                melds=melds,
                pending=pending,
                action=action,
                source_seat=source_seat,
                players=players,
                rules=rules,
            )
            response_transition_cases += 1
            if _canonical_transition(
                oracle_transition.to_dict()
            ) != _canonical_transition(production_transition):
                mismatches.append(
                    RuleMismatch(
                        domain="response_transition",
                        index=index,
                        payload={
                            "hand": hand,
                            "existing_melds": [
                                {
                                    "kind": meld.kind,
                                    "cards": list(meld.cards),
                                }
                                for meld in melds
                            ],
                            "pending": pending,
                            "source": source,
                            "action": action.to_dict(),
                        },
                        oracle=oracle_transition.to_dict(),
                        production=production_transition,
                    )
                )
        if oracle_response.automatic_action == "PAO":
            oracle_transition = apply_automatic_pao_oracle(
                hand,
                melds,
                pending,
            )
            production_transition = _production_automatic_pao_transition(
                hand=hand,
                melds=melds,
                pending=pending,
                source_seat=source_seat,
                players=players,
                rules=rules,
            )
            response_transition_cases += 1
            if _canonical_transition(
                oracle_transition.to_dict()
            ) != _canonical_transition(production_transition):
                mismatches.append(
                    RuleMismatch(
                        domain="response_transition",
                        index=index,
                        payload={
                            "hand": hand,
                            "existing_melds": [
                                {
                                    "kind": meld.kind,
                                    "cards": list(meld.cards),
                                }
                                for meld in melds
                            ],
                            "pending": pending,
                            "source": source,
                            "action": {"type": "PAO", "automatic": True},
                        },
                        oracle=oracle_transition.to_dict(),
                        production=production_transition,
                    )
                )

    draw_auto_positive_cases = 0
    draw_auto_skip_cases = 0
    for index in range(1, draw_auto_cases + 1):
        (
            hand,
            melds,
            drawn,
            quad_events,
            source,
        ) = _generate_draw_auto_case(
            rng,
            wildcard_enabled=wildcard_enabled,
        )
        oracle_transition = apply_draw_auto_meld_oracle(
            hand,
            melds,
            drawn,
            quad_events=quad_events,
        )
        production_transition = _production_draw_auto_transition(
            hand=hand,
            melds=melds,
            drawn=drawn,
            quad_events=quad_events,
            rules=rules,
        )
        draw_auto_positive_cases += int(
            oracle_transition.automatic_action is not None
            or production_transition["automatic_action"] is not None
        )
        draw_auto_skip_cases += int(
            oracle_transition.skip_discard
            or production_transition["skip_discard"]
        )
        if _canonical_draw_transition(
            oracle_transition.to_dict()
        ) != _canonical_draw_transition(production_transition):
            mismatches.append(
                RuleMismatch(
                    domain="draw_auto_transition",
                    index=index,
                    payload={
                        "hand_before_draw": hand,
                        "existing_melds": [
                            {
                                "kind": meld.kind,
                                "cards": list(meld.cards),
                            }
                            for meld in melds
                        ],
                        "drawn": drawn,
                        "quad_events": quad_events,
                        "source": source,
                    },
                    oracle=oracle_transition.to_dict(),
                    production=production_transition,
                )
            )

    by_domain = Counter(item.domain for item in mismatches)
    return {
        "ok": not mismatches,
        "oracle_independent": True,
        "seed": seed,
        "players": players,
        "wildcard_enabled": wildcard_enabled,
        "room_shape": production_room,
        "hu_cases": hu_cases,
        "hu_complete_cases": hu_complete_cases,
        "hu_positive_cases": hu_positive_cases,
        "chi_cases": chi_cases,
        "chi_positive_cases": chi_positive_cases,
        "discard_cases": discard_cases,
        "discard_locked_cases": discard_locked_cases,
        "response_cases": response_cases,
        "response_positive_cases": response_positive_cases,
        "response_joint_cases": response_joint_cases,
        "response_auto_pao_cases": response_auto_pao_cases,
        "response_transition_cases": response_transition_cases,
        "draw_auto_cases": draw_auto_cases,
        "draw_auto_positive_cases": draw_auto_positive_cases,
        "draw_auto_skip_cases": draw_auto_skip_cases,
        "mismatch_count": len(mismatches),
        "mismatches_by_domain": dict(sorted(by_domain.items())),
        "mismatches": [item.to_dict() for item in mismatches],
    }


def _generate_hu_case(
    rng: random.Random,
    *,
    wildcard_enabled: bool,
) -> tuple[list[str], list[SimMeld], str]:
    if rng.random() < 0.7:
        groups = _structured_groups(rng)
        existing_count = rng.randint(0, 3)
        existing_melds = [_sim_meld_for_group(group, rng) for group in groups[:existing_count]]
        hand = [label for group in groups[existing_count:] for label in group]
        source = f"structured_existing_{existing_count}"
        if wildcard_enabled:
            replacements = rng.randint(0, min(4, len(hand)))
            for index in rng.sample(range(len(hand)), replacements):
                hand[index] = WILD_LABEL
            source += "_wild"
        if rng.random() < 0.35:
            _mutate_one_card(rng, hand, wildcard_enabled=wildcard_enabled)
            source += "_mutated"
        rng.shuffle(hand)
        return hand, existing_melds, source

    deck = [label for label in NORMAL_LABELS for _ in range(4)]
    if wildcard_enabled:
        deck.extend([WILD_LABEL] * 4)
    return rng.sample(deck, 21), [], "random_deck"


def _structured_groups(rng: random.Random) -> list[tuple[str, str, str]]:
    templates = hu_templates()
    for _ in range(2_000):
        groups = [rng.choice(templates) for _ in range(7)]
        hand = [label for group in groups for label in group]
        counts = Counter(hand)
        if all(amount <= 4 for amount in counts.values()):
            return groups
    raise RuntimeError("could_not_generate_structured_hand")


def _sim_meld_for_group(
    group: tuple[str, str, str],
    rng: random.Random,
) -> SimMeld:
    counts = Counter(group)
    if len(counts) == 1:
        return SimMeld(rng.choice(("wei", "peng")), group)
    ranks = [_sort_key(label)[0] for label in group]
    suits = {_sort_key(label)[1] for label in group}
    if len(suits) == 1 and sorted(ranks) == [2, 7, 10]:
        kind = "special_2710"
    elif len(suits) == 1 and sorted(ranks) == [1, 2, 3]:
        kind = "special_123"
    elif len(suits) == 1:
        kind = "normal_sequence"
    else:
        kind = "mixed_same_rank"
    return SimMeld(kind, group)


def _mutate_one_card(
    rng: random.Random,
    hand: list[str],
    *,
    wildcard_enabled: bool,
) -> None:
    labels = list(NORMAL_LABELS)
    if wildcard_enabled:
        labels.append(WILD_LABEL)
    counts = Counter(hand)
    position = rng.randrange(len(hand))
    old = hand[position]
    candidates = [
        label
        for label in labels
        if label != old and counts[label] < 4
    ]
    if candidates:
        hand[position] = rng.choice(candidates)


def _generate_chi_case(rng: random.Random) -> tuple[list[str], str, str]:
    external = rng.choice(NORMAL_LABELS)
    if rng.random() < 0.7:
        candidates = [
            group
            for group in hu_templates()
            if external in group and len(set(group)) > 1
        ]
        group = list(rng.choice(candidates))
        group.remove(external)
        deck = [label for label in NORMAL_LABELS for _ in range(4)]
        for label in group:
            deck.remove(label)
        extra_count = rng.randint(2, 16)
        hand = [*group, *rng.sample(deck, extra_count)]
        rng.shuffle(hand)
        return hand, external, "structured"

    deck = [label for label in NORMAL_LABELS for _ in range(4)]
    hand = rng.sample(deck, rng.randint(4, 20))
    return hand, external, "random_deck"


def _generate_discard_case(
    rng: random.Random,
    *,
    wildcard_enabled: bool,
) -> tuple[list[str], str]:
    deck = [label for label in NORMAL_LABELS for _ in range(4)]
    if wildcard_enabled:
        deck.extend([WILD_LABEL] * 4)
    size = rng.randint(1, 21)
    hand = rng.sample(deck, size)
    source = "random"
    if rng.random() < 0.6:
        label = rng.choice(NORMAL_LABELS)
        hand = [card for card in hand if card != label]
        hand.extend([label] * rng.choice((3, 4)))
        hand = hand[:21]
        source = "locked_triplet"
    if wildcard_enabled and rng.random() < 0.4 and WILD_LABEL not in hand:
        hand.append(WILD_LABEL)
        hand = hand[:21]
        source += "_wang"
    return hand, source


def _generate_response_case(
    rng: random.Random,
    *,
    wildcard_enabled: bool,
) -> tuple[list[str], list[SimMeld], str, str]:
    category = rng.randrange(6)
    if category == 0:
        return (
            [
                "七", "八", "九",
                "四", "肆",
                "陆", "柒", "捌",
                "贰", "柒", "拾",
                "肆", "伍", "陆",
                "六", "六", "六",
                "二", "三", "四",
            ],
            [],
            "肆",
            "joint_hu_peng_chi",
        )
    if category == 1:
        pending = rng.choice(NORMAL_LABELS)
        deck = _deck_without((pending, pending, pending, pending))
        return [pending, pending, *rng.sample(deck, rng.randint(2, 12))], [], pending, "peng"
    if category == 2:
        for _ in range(100):
            hand, pending, _ = _generate_chi_case(rng)
            counts = Counter(hand)
            counts[pending] += 1
            if all(amount <= 4 for amount in counts.values()):
                return hand, [], pending, "chi"
        raise RuntimeError("could_not_generate_valid_chi_response")
    if category == 3:
        groups = _structured_groups(rng)
        complete = [label for group in groups for label in group]
        pending = rng.choice(complete)
        complete.remove(pending)
        return complete, [], pending, "hu_wait"
    if category == 4:
        pending = rng.choice(NORMAL_LABELS)
        deck = _deck_without((pending, pending, pending, pending))
        hand = rng.sample(deck, rng.randint(2, 12))
        return (
            hand,
            [SimMeld("peng", (pending, pending, pending))],
            pending,
            "automatic_pao",
        )
    deck = [label for label in NORMAL_LABELS for _ in range(4)]
    if wildcard_enabled:
        deck.extend([WILD_LABEL] * 4)
    pending = rng.choice(deck)
    deck.remove(pending)
    return rng.sample(deck, rng.randint(2, 20)), [], pending, "random"


def _deck_without(labels: tuple[str, ...]) -> list[str]:
    deck = [label for label in NORMAL_LABELS for _ in range(4)]
    for label in labels:
        deck.remove(label)
    return deck


def _generate_draw_auto_case(
    rng: random.Random,
    *,
    wildcard_enabled: bool,
) -> tuple[list[str], list[SimMeld], str, int, str]:
    category = rng.randrange(6 if wildcard_enabled else 5)
    normal = rng.choice(NORMAL_LABELS)
    quad_events = rng.randint(0, 1)
    if category == 0:
        deck = _deck_without((normal, normal, normal, normal))
        hand = [normal, *rng.sample(deck, rng.randint(1, 12))]
        return hand, [], normal, quad_events, "no_auto"
    if category == 1:
        deck = _deck_without((normal, normal, normal, normal))
        hand = [normal, normal, *rng.sample(deck, rng.randint(1, 10))]
        return hand, [], normal, quad_events, "wei"
    if category == 2:
        deck = _deck_without((normal, normal, normal, normal))
        hand = [normal, normal, normal, *rng.sample(deck, rng.randint(1, 10))]
        return hand, [], normal, quad_events, "ti"
    if category in {3, 4}:
        deck = _deck_without((normal, normal, normal, normal))
        hand = rng.sample(deck, rng.randint(1, 10))
        kind = "peng" if category == 3 else "wei"
        return (
            hand,
            [SimMeld(kind, (normal, normal, normal))],
            normal,
            quad_events,
            f"pao_from_{kind}",
        )
    wang_count = rng.randint(0, 3)
    deck = [label for label in NORMAL_LABELS for _ in range(4)]
    hand = [WILD_LABEL] * wang_count
    hand.extend(rng.sample(deck, rng.randint(1, 10)))
    return hand, [], WILD_LABEL, quad_events, "wang_no_auto"


class _PassResponsePolicy(BaselinePolicy):
    def choose_hu(self, view, hu, rules):
        return False

    def choose_peng(self, view, label, rules):
        return False

    def choose_chi(self, view, plans, rules):
        return None


def _production_response_actions(
    *,
    hand: list[str],
    melds: list[SimMeld],
    pending: str,
    seat: int,
    source_seat: int,
    players: int,
    rules: dict[str, Any],
) -> dict[str, Any]:
    policies = [_PassResponsePolicy() for _ in range(players)]
    simulator = FullGameSimulator(
        policies,
        wildcard_enabled=bool(
            rules.get("wildcard", {}).get("enabled", False)
        ),
        rules=rules,
    )
    simulated_players = [
        SimPlayer(
            seat=player_seat,
            hand=list(hand) if player_seat == seat else [],
            melds=list(melds) if player_seat == seat else [],
        )
        for player_seat in range(players)
    ]
    pao_seat = simulator._first_pao_claim(
        simulated_players,
        source_seat,
        pending,
    )
    intents = simulator._collect_response_intents(
        simulated_players,
        discarder=source_seat,
        discard=pending,
        stock=[],
        initial_counts=full_deck_counts(rules=rules),
        blocked_auto_claim_seats=frozenset(),
        automatic_pao_pending=pao_seat is not None,
    )
    intent = next(
        (
            item
            for item in intents
            if item.seat == seat
        ),
        None,
    )
    return {
        "actions": (
            [
                action.to_dict()
                for action in intent.legal_actions
            ]
            if intent is not None
            else []
        ),
        "automatic_action": "PAO" if pao_seat is not None else None,
        "automatic_seat": pao_seat,
    }


class _ForcedResponsePolicy(_PassResponsePolicy):
    def __init__(self, action: Any) -> None:
        self.action = action

    def choose_hu(self, view, hu, rules):
        return str(self.action.type).upper() == "HU"

    def choose_peng(self, view, label, rules):
        return str(self.action.type).upper() == "PENG"

    def choose_chi(self, view, plans, rules):
        if str(self.action.type).upper() != "CHI":
            return None
        target = _action_signature(self.action)
        return next(
            (
                plan
                for plan in plans
                if _action_signature(
                    {
                        "type": "CHI",
                        "consumed_from_hand": plan.consumed_from_hand,
                        "meld_groups": plan.groups,
                    }
                )
                == target
            ),
            None,
        )


def _production_response_transition(
    *,
    hand: list[str],
    melds: list[SimMeld],
    pending: str,
    action: Any,
    source_seat: int,
    players: int,
    rules: dict[str, Any],
) -> dict[str, Any]:
    return _run_production_response_transition(
        hand=hand,
        melds=melds,
        pending=pending,
        root_policy=_ForcedResponsePolicy(action),
        source_seat=source_seat,
        players=players,
        rules=rules,
        blocked_auto_claim=(
            str(action.type).upper() != "HU"
        ),
        expected_claim=str(action.type).lower(),
    )


def _production_automatic_pao_transition(
    *,
    hand: list[str],
    melds: list[SimMeld],
    pending: str,
    source_seat: int,
    players: int,
    rules: dict[str, Any],
) -> dict[str, Any]:
    return _run_production_response_transition(
        hand=hand,
        melds=melds,
        pending=pending,
        root_policy=_PassResponsePolicy(),
        source_seat=source_seat,
        players=players,
        rules=rules,
        blocked_auto_claim=False,
        expected_claim="pao",
    )


def _run_production_response_transition(
    *,
    hand: list[str],
    melds: list[SimMeld],
    pending: str,
    root_policy: BaselinePolicy,
    source_seat: int,
    players: int,
    rules: dict[str, Any],
    blocked_auto_claim: bool,
    expected_claim: str,
) -> dict[str, Any]:
    remaining = full_deck_counts(rules=rules)
    remaining.subtract(hand)
    remaining.subtract(
        card
        for meld in melds
        for card in meld.cards
    )
    remaining.subtract((pending,))
    if any(amount < 0 for amount in remaining.values()):
        raise ValueError("generated_response_state_exceeds_deck")
    simulated_players = [
        SimPlayer(
            seat=seat,
            hand=list(hand) if seat == 0 else [],
            melds=list(melds) if seat == 0 else [],
        )
        for seat in range(players)
    ]
    policies = [_PassResponsePolicy() for _ in range(players)]
    policies[0] = root_policy
    simulator = FullGameSimulator(
        policies,
        wildcard_enabled=bool(
            rules.get("wildcard", {}).get("enabled", False)
        ),
        rules=rules,
    )
    result = simulator.play_from_pending_discard(
        seed=20260803,
        players=simulated_players,
        stock=expanded_deck(+remaining),
        discarder=source_seat,
        pending_card=pending,
        blocked_auto_claim_seats=(
            frozenset((0,))
            if blocked_auto_claim
            else frozenset()
        ),
        max_turns=0,
    )
    player = simulated_players[0]
    terminal = expected_claim == "hu"
    if result.violations or result.coverage_failures:
        raise ValueError(
            "production_response_transition_invalid:"
            f"{result.violations}:{result.coverage_failures}"
        )
    if terminal and result.winner != 0:
        raise ValueError("forced_hu_did_not_win")
    if not terminal and result.action_counts.get(expected_claim, 0) != (
        0 if expected_claim == "pass" else 1
    ):
        raise ValueError(f"forced_response_claim_missing:{expected_claim}")
    return {
        "hand": sorted(player.hand, key=_sort_key),
        "melds": [
            {
                "kind": meld.kind,
                "labels": list(meld.cards),
            }
            for meld in player.melds
        ],
        "claim_type": expected_claim,
        "terminal": terminal,
        "passed_chi": sorted(player.passed_chi, key=_sort_key),
        "passed_peng": sorted(player.passed_peng, key=_sort_key),
    }


def _production_draw_auto_transition(
    *,
    hand: list[str],
    melds: list[SimMeld],
    drawn: str,
    quad_events: int,
    rules: dict[str, Any],
) -> dict[str, Any]:
    simulator = FullGameSimulator(
        [BaselinePolicy(), BaselinePolicy()],
        wildcard_enabled=bool(
            rules.get("wildcard", {}).get("enabled", False)
        ),
        rules=rules_for_room(
            wildcard_enabled=bool(
                rules.get("wildcard", {}).get("enabled", False)
            ),
            players=2,
        ),
    )
    player = SimPlayer(
        seat=0,
        hand=[*hand, drawn],
        melds=list(melds),
        quad_events=quad_events,
    )
    actions: Counter[str] = Counter()
    resolution = simulator._apply_draw_auto_meld(
        player,
        drawn,
        actions,
    )
    automatic = next(
        (
            action.upper()
            for action in ("pao", "ti", "wei")
            if actions.get(action)
        ),
        None,
    )
    return {
        "hand": sorted(player.hand, key=_sort_key),
        "melds": [
            {
                "kind": meld.kind,
                "labels": list(meld.cards),
            }
            for meld in player.melds
        ],
        "automatic_action": automatic,
        "quad_events": player.quad_events,
        "skip_discard": resolution == "skip_discard",
    }


def _action_signature(action: Any) -> tuple[Any, ...]:
    if isinstance(action, dict):
        action_type = str(action.get("type") or "").upper()
        label = action.get("label")
        consumed = action.get("consumed_from_hand") or ()
        groups = action.get("meld_groups") or ()
    else:
        action_type = str(getattr(action, "type", "")).upper()
        label = getattr(action, "label", None)
        consumed = getattr(action, "consumed_from_hand", ())
        groups = getattr(action, "meld_groups", ())
    return (
        action_type,
        (
            str(label)
            if label is not None and action_type in {"HU", "PENG"}
            else None
        ),
        tuple(sorted((str(item) for item in consumed), key=_sort_key)),
        tuple(
            sorted(
                (
                    tuple(
                        sorted(
                            (str(item) for item in group),
                            key=_sort_key,
                        )
                    )
                    for group in groups
                ),
                key=lambda group: tuple(_sort_key(item) for item in group),
            )
        ),
    )


def _canonical_transition(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    normalized["hand"] = sorted(
        (str(label) for label in value.get("hand") or ()),
        key=_sort_key,
    )
    normalized["melds"] = sorted(
        (
            {
                "kind": str(meld.get("kind") or ""),
                "labels": sorted(
                    (str(label) for label in meld.get("labels") or ()),
                    key=_sort_key,
                ),
            }
            for meld in value.get("melds") or ()
        ),
        key=lambda meld: (
            meld["kind"],
            tuple(_sort_key(label) for label in meld["labels"]),
        ),
    )
    normalized["passed_chi"] = sorted(
        (str(label) for label in value.get("passed_chi") or ()),
        key=_sort_key,
    )
    normalized["passed_peng"] = sorted(
        (str(label) for label in value.get("passed_peng") or ()),
        key=_sort_key,
    )
    return normalized


def _canonical_draw_transition(value: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        key: value.get(key)
        for key in (
            "automatic_action",
            "quad_events",
            "skip_discard",
        )
    }
    normalized["hand"] = sorted(
        (str(label) for label in value.get("hand") or ()),
        key=_sort_key,
    )
    normalized["melds"] = _canonical_transition(
        {"melds": value.get("melds") or ()}
    )["melds"]
    return normalized


def _plan_key(plan: Any) -> tuple[Any, ...]:
    initial = tuple(sorted(plan.initial_group, key=_sort_key))
    compares = tuple(
        sorted(
            (tuple(sorted(group, key=_sort_key)) for group in plan.compare_groups),
            key=lambda group: tuple(_sort_key(label) for label in group),
        )
    )
    consumed = tuple(sorted(plan.consumed_from_hand, key=_sort_key))
    return initial, compares, consumed


def _sort_key(label: str) -> tuple[int, int, str]:
    if label == WILD_LABEL:
        return 99, 99, label
    if label in NORMAL_LABELS[:10]:
        return NORMAL_LABELS.index(label) + 1, 0, label
    return NORMAL_LABELS.index(label) - 9, 1, label


__all__ = ["RuleMismatch", "run_rule_crosscheck"]
