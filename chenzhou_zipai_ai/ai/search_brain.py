"""Shadow-first production wrapper for selective root ISMCTS."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ai.full_game_simulator import PublicView, SimMeld
from ai.ismcts import (
    RootISMCTSConfig,
    RootISMCTSPolicy,
    RootResponseCandidate,
    RootResponseSearchResult,
    RootSearchResult,
    build_response_candidates,
    shortlist_response_candidates,
)
from ai.pro_brain import ActionEval, PolicyDecision, choose_action
from engine.cards import normalize_card_label
from engine.deck import full_deck_counts
from engine.rules import load_rules


SEARCH_POLICY_VERSION = "v3.0.0-shadow"


@dataclass(frozen=True)
class SearchAugmentationConfig:
    enabled: bool = True
    shadow_only: bool = True
    activation_ev_gap: float = 120.0
    max_candidates: int = 2
    max_response_candidates: int = 4
    root: RootISMCTSConfig = RootISMCTSConfig()


@dataclass(frozen=True)
class SearchAugmentedDecision:
    decision: PolicyDecision
    production_decision: PolicyDecision
    search_result: RootSearchResult | RootResponseSearchResult | None
    search_attempted: bool
    search_applied: bool
    reason: str
    policy_version: str = SEARCH_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "search_attempted": self.search_attempted,
            "search_applied": self.search_applied,
            "reason": self.reason,
            "production_decision": self.production_decision.to_dict(),
            "decision": self.decision.to_dict(),
            "search_result": self.search_result.to_dict() if self.search_result else None,
        }


def choose_action_with_search(
    state_or_hand: dict[str, Any] | list[str],
    *,
    rules: dict[str, Any] | None = None,
    config_path: str = "config/rules.yaml",
    search_config: SearchAugmentationConfig | None = None,
) -> SearchAugmentedDecision:
    """Evaluate search in shadow mode unless promotion is explicitly enabled."""

    config = search_config or SearchAugmentationConfig()
    resolved_rules = rules or load_rules(config_path)
    production = choose_action(state_or_hand, rules=resolved_rules, config_path=config_path)
    if not config.enabled:
        return _unchanged(production, "search_disabled")
    if not isinstance(state_or_hand, dict):
        return _unchanged(production, "public_state_required")
    if production.selected_action == "HU":
        return _unchanged(production, "terminal_hu_keeps_hard_priority")
    if production.selected_action in {"CHI", "PENG", "PASS"}:
        return _search_response_action(
            state_or_hand,
            production,
            rules=resolved_rules,
            config=config,
        )
    if production.selected_action != "DISCARD":
        return _unchanged(production, "search_action_not_supported")
    if production.safety_flags:
        return _unchanged(production, "production_safety_flag")

    candidate_evals = _candidate_discard_evals(production.action_evals)
    if len(candidate_evals) < 2:
        return _unchanged(production, "fewer_than_two_safe_candidates")
    ev_gap = candidate_evals[0].ev - candidate_evals[1].ev
    if ev_gap > config.activation_ev_gap:
        return _unchanged(production, f"production_ev_gap={round(ev_gap, 3)}")

    view = public_view_from_production_state(
        state_or_hand,
        production,
        rules=resolved_rules,
    )
    if view is None:
        return _unchanged(production, "insufficient_public_state_for_determinization")
    labels = _ordered_unique(
        item.action.label
        for item in candidate_evals[: max(2, config.max_candidates)]
        if item.action.label
    )
    search_policy = RootISMCTSPolicy(replace(config.root, max_candidates=config.max_candidates))
    search = search_policy.search_discard(
        view,
        rules=resolved_rules,
        candidate_labels=labels,
        candidate_priors={
            item.action.label: item.ev
            for item in candidate_evals
            if item.action.label in labels
        },
        force_search=True,
    )
    if not search.used_search:
        return SearchAugmentedDecision(
            decision=production,
            production_decision=production,
            search_result=search,
            search_attempted=True,
            search_applied=False,
            reason=f"search_not_usable:{search.reason}",
        )
    if config.shadow_only:
        return SearchAugmentedDecision(
            decision=production,
            production_decision=production,
            search_result=search,
            search_attempted=True,
            search_applied=False,
            reason="shadow_only",
        )
    selected_eval = next(
        (
            item
            for item in candidate_evals
            if item.action.label == search.selected_label
        ),
        None,
    )
    if selected_eval is None:
        return SearchAugmentedDecision(
            decision=production,
            production_decision=production,
            search_result=search,
            search_attempted=True,
            search_applied=False,
            reason="search_label_not_in_safe_production_candidates",
        )
    promoted = replace(
        production,
        selected_card_id=selected_eval.action.card_id,
        selected_label=selected_eval.action.label,
        candidate_stage="root_ismcts",
        ev=selected_eval.ev,
        reason=(
            f"root_ismcts selected {selected_eval.action.label}; "
            f"production selected {production.selected_label}; "
            f"simulations={search.simulations}"
        ),
    )
    return SearchAugmentedDecision(
        decision=promoted,
        production_decision=production,
        search_result=search,
        search_attempted=True,
        search_applied=True,
        reason="search_promoted",
    )


def _search_response_action(
    state: Mapping[str, Any],
    production: PolicyDecision,
    *,
    rules: dict[str, Any],
    config: SearchAugmentationConfig,
) -> SearchAugmentedDecision:
    if production.safety_flags:
        return _unchanged(production, "production_safety_flag")
    candidates = _response_candidates(state, production.action_evals)
    candidates = shortlist_response_candidates(
        candidates,
        production_key=_production_response_key(state, production),
        max_candidates=config.max_response_candidates,
    )
    if len(candidates) < 2:
        return _unchanged(production, "fewer_than_two_safe_response_candidates")
    view = public_view_from_production_state(
        state,
        production,
        rules=rules,
        response_root=True,
    )
    if view is None:
        return _unchanged(production, "insufficient_response_public_state")
    search_policy = RootISMCTSPolicy(
        replace(
            config.root,
            max_candidates=max(2, config.max_response_candidates),
        )
    )
    try:
        search = search_policy.search_response(
            view,
            rules=rules,
            candidates=candidates,
            force_search=True,
        )
    except ValueError as exc:
        return _unchanged(production, f"invalid_response_search_contract:{exc}")
    if not search.used_search:
        return SearchAugmentedDecision(
            decision=production,
            production_decision=production,
            search_result=search,
            search_attempted=True,
            search_applied=False,
            reason=f"response_search_not_usable:{search.reason}",
        )
    return SearchAugmentedDecision(
        decision=production,
        production_decision=production,
        search_result=search,
        search_attempted=True,
        search_applied=False,
        reason=(
            "response_shadow_only"
            if config.shadow_only
            else "response_promotion_not_validated"
        ),
    )


def _response_candidates(
    state: Mapping[str, Any],
    evals: Sequence[ActionEval],
) -> list[RootResponseCandidate]:
    pending = normalize_card_label(
        str(state.get("pending_card") or state.get("external_card") or "")
    )
    return build_response_candidates(pending, evals)


def _production_response_key(
    state: Mapping[str, Any],
    decision: PolicyDecision,
) -> str:
    action_type = decision.selected_action.upper()
    if action_type == "PASS":
        return "PASS"
    if action_type == "PENG":
        pending = normalize_card_label(
            str(state.get("pending_card") or state.get("external_card") or "")
        )
        return f"PENG:{pending}" if pending else ""
    if action_type == "CHI" and decision.selected_option_id:
        return f"CHI:{decision.selected_option_id}"
    return ""


def public_view_from_production_state(
    state: Mapping[str, Any],
    decision: PolicyDecision,
    *,
    rules: Mapping[str, Any],
    response_root: bool = False,
) -> PublicView | None:
    context = decision.context_snapshot.get("context") or {}
    hand = tuple(str(label) for label in context.get("normalized_hand") or state.get("hand") or [])
    if not hand:
        return None
    own_melds = _parse_melds(context.get("existing_melds") or state.get("strategy_existing_melds") or [])
    if own_melds is None:
        return None
    player_count = int(rules.get("game", {}).get("players", 3))
    if player_count not in {2, 3}:
        return None
    opponent_count = player_count - 1
    memory = state.get("memory") if isinstance(state.get("memory"), Mapping) else {}
    opponent_entries = _opponent_entries(memory)
    opponent_melds: list[tuple[SimMeld, ...]] = []
    opponent_discards: list[tuple[str, ...]] = []
    opponent_sizes: list[int | None] = []
    opponent_passed_chi: list[tuple[str, ...]] = []
    opponent_passed_peng: list[tuple[str, ...]] = []
    for entry in opponent_entries[:opponent_count]:
        melds = _parse_melds(
            entry.get("meld_groups")
            or entry.get("opponent_meld_groups")
            or entry.get("exposed_melds")
            or []
        )
        if melds is None:
            return None
        opponent_melds.append(melds)
        opponent_discards.append(tuple(_labels(entry.get("discards") or entry.get("opponent_discards") or [])))
        opponent_passed_chi.append(
            tuple(_labels(entry.get("passed_chi") or entry.get("opponent_passed_chi") or []))
        )
        opponent_passed_peng.append(
            tuple(_labels(entry.get("passed_peng") or entry.get("opponent_passed_peng") or []))
        )
        raw_size = entry.get("hand_size")
        opponent_sizes.append(int(raw_size) if isinstance(raw_size, int) else None)

    if not opponent_entries:
        melds = _parse_melds(memory.get("opponent_meld_groups") or [])
        if melds is None:
            return None
        opponent_melds = [melds]
        opponent_discards = [tuple(_labels(memory.get("opponent_discards") or []))]
        opponent_passed_chi = [
            tuple(_labels(memory.get("opponent_passed_chi") or []))
        ]
        opponent_passed_peng = [
            tuple(_labels(memory.get("opponent_passed_peng") or []))
        ]
        opponent_sizes = [None]
    while len(opponent_melds) < opponent_count:
        opponent_melds.append(())
        opponent_discards.append(())
        opponent_passed_chi.append(())
        opponent_passed_peng.append(())
        opponent_sizes.append(None)

    own_discards = tuple(_labels(memory.get("my_discards") or state.get("my_discards") or []))
    all_melds = (own_melds, *opponent_melds)
    discards = (own_discards, *opponent_discards)
    own_passed_chi = tuple(
        _labels(
            state.get("passed_chi")
            or memory.get("my_passed_chi")
            or memory.get("passed_chi")
            or []
        )
    )
    own_passed_peng = tuple(
        _labels(
            state.get("passed_peng")
            or memory.get("my_passed_peng")
            or memory.get("passed_peng")
            or []
        )
    )
    pending_card: str | None = None
    pending_source_seat: int | None = None
    if response_root:
        pending_card = normalize_card_label(
            str(state.get("pending_card") or state.get("external_card") or "")
        )
        pending_source_seat = _response_source_seat(state, player_count)
        if not pending_card or pending_source_seat is None:
            return None
    remaining = full_deck_counts(rules=dict(rules))
    remaining.subtract(hand)
    for melds in all_melds:
        remaining.subtract(card for meld in melds for card in meld.cards)
    for seat_discards in discards:
        remaining.subtract(seat_discards)
    if pending_card:
        remaining.subtract((pending_card,))
    if any(amount < 0 for amount in remaining.values()):
        return None
    remaining = +remaining
    stock_count = int(
        state.get("remaining_deck_count")
        or state.get("remaining_cards_estimate")
        or context.get("remaining_deck_count")
        or 0
    )
    hidden_total = sum(remaining.values()) - stock_count
    if hidden_total < 0:
        return None
    if all(size is not None for size in opponent_sizes):
        sizes = [int(size) for size in opponent_sizes]
        if sum(sizes) != hidden_total:
            return None
    else:
        base, remainder = divmod(hidden_total, opponent_count)
        sizes = [
            base + int(index < remainder)
            for index in range(opponent_count)
        ]
    return PublicView(
        seat=0,
        hand=hand,
        own_melds=own_melds,
        all_melds=all_melds,
        discards=discards,
        remaining_counts=tuple(sorted(remaining.items())),
        stock_count=stock_count,
        hand_sizes=(len(hand), *sizes),
        pending_card=pending_card,
        pending_source_seat=pending_source_seat,
        passed_chi=(own_passed_chi, *opponent_passed_chi),
        passed_peng=(own_passed_peng, *opponent_passed_peng),
    )


def _response_source_seat(
    state: Mapping[str, Any],
    player_count: int,
) -> int | None:
    metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
    explicit = state.get("pending_source_seat", metadata.get("pending_source_seat"))
    if isinstance(explicit, int) and 0 <= explicit < player_count and explicit != 0:
        return explicit
    if player_count == 2:
        return 1
    action_types = {
        str(action.get("type") if isinstance(action, Mapping) else action).upper()
        for action in state.get("legal_actions") or []
    }
    if "CHI" in action_types:
        return player_count - 1

    pending = normalize_card_label(
        str(state.get("pending_card") or state.get("external_card") or "")
    )
    event = metadata.get("last_turn_event")
    source_chair: int | None = None
    if isinstance(event, Mapping) and _event_matches_pending(event, pending):
        source_chair = _as_int(event.get("seat"))
    if source_chair is None:
        source_chair = _as_int(metadata.get("response_source_chair_id"))
    seat_fields = metadata.get("seat_fields") if isinstance(metadata.get("seat_fields"), Mapping) else {}
    self_chair = _as_int(
        metadata.get(
            "self_chair_id",
            seat_fields.get("selfChairId", seat_fields.get("self_chair_id")),
        )
    )
    turn_step = _as_int(metadata.get("chair_turn_step"))
    if source_chair is None or self_chair is None or turn_step not in {-1, 1}:
        return None
    logical = ((source_chair - self_chair) * turn_step) % player_count
    return logical if logical != 0 else None


def _event_matches_pending(event: Mapping[str, Any], pending: str) -> bool:
    if not pending:
        return False
    raw_cards = event.get("cards")
    if isinstance(raw_cards, Sequence) and not isinstance(raw_cards, (str, bytes)):
        return pending in _labels(raw_cards)
    return normalize_card_label(str(event.get("card") or "")) == pending


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_discard_evals(evals: Sequence[ActionEval]) -> list[ActionEval]:
    best_by_label: dict[str, ActionEval] = {}
    for item in evals:
        label = item.action.label
        if item.type != "DISCARD" or not item.allowed or not label:
            continue
        previous = best_by_label.get(label)
        if previous is None or item.ev > previous.ev:
            best_by_label[label] = item
    return sorted(best_by_label.values(), key=lambda item: item.ev, reverse=True)


def _parse_melds(raw_melds: Any) -> tuple[SimMeld, ...] | None:
    if not isinstance(raw_melds, Sequence) or isinstance(raw_melds, (str, bytes)):
        return ()
    result: list[SimMeld] = []
    for raw in raw_melds:
        if not isinstance(raw, Mapping):
            return None
        labels = tuple(_labels(raw.get("labels") or raw.get("cards") or []))
        if not labels or "暗" in labels:
            return None
        kind = str(raw.get("kind") or raw.get("type") or "unknown")
        result.append(SimMeld(kind, labels))
    return tuple(result)


def _opponent_entries(memory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("opponents", "opponent_states", "other_players"):
        raw = memory.get(key)
        if isinstance(raw, Mapping):
            return [value for value in raw.values() if isinstance(value, Mapping)]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [value for value in raw if isinstance(value, Mapping)]
    return []


def _labels(raw: Any) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    result: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            if item.get("cards") or item.get("labels"):
                result.extend(_labels(item.get("cards") or item.get("labels")))
                continue
            item = item.get("label") or item.get("name") or item.get("card")
        label = normalize_card_label(str(item))
        if label:
            result.append(label)
    return result


def _ordered_unique(values: Sequence[str] | Any) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _unchanged(decision: PolicyDecision, reason: str) -> SearchAugmentedDecision:
    return SearchAugmentedDecision(
        decision=decision,
        production_decision=decision,
        search_result=None,
        search_attempted=False,
        search_applied=False,
        reason=reason,
    )


__all__ = [
    "SEARCH_POLICY_VERSION",
    "SearchAugmentationConfig",
    "SearchAugmentedDecision",
    "choose_action_with_search",
    "public_view_from_production_state",
]
