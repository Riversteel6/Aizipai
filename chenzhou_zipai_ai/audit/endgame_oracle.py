"""Independent exact two-player endgame solver for fully observed small states.

The solver intentionally imports neither ``engine`` nor ``ai``.  It is an
acceptance oracle for states whose two hands and remaining stock multiset are
fully known; it must not be used to pretend a large information set is exact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from audit.independent_rules import (
    OracleAction,
    OracleMeldState,
    apply_draw_auto_meld_oracle,
    apply_response_action_oracle,
    evaluate_hu_oracle,
    legal_discard_labels_oracle,
    legal_response_actions_oracle,
)


@dataclass(frozen=True)
class OracleEndgameState:
    hands: tuple[tuple[str, ...], tuple[str, ...]]
    melds: tuple[tuple[OracleMeldState, ...], tuple[OracleMeldState, ...]]
    stock: tuple[str, ...]
    current_seat: int
    needs_draw: bool = True
    passed_chi: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    passed_peng: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())


@dataclass(frozen=True)
class OracleActionValue:
    label: str
    p_win: float
    regret: float


@dataclass(frozen=True)
class OracleEndgameResult:
    selected_label: str | None
    p_win: float
    action_values: tuple[OracleActionValue, ...]
    exact: bool
    nodes: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_label": self.selected_label,
            "p_win": self.p_win,
            "action_values": [
                {
                    "label": item.label,
                    "p_win": item.p_win,
                    "confidence_interval": [item.p_win, item.p_win],
                    "regret": item.regret,
                }
                for item in self.action_values
            ],
            "exact": self.exact,
            "nodes": self.nodes,
            "reason": self.reason,
        }


class IndependentEndgameOracle:
    def __init__(
        self,
        rules: Mapping[str, Any],
        *,
        root_seat: int,
        maximum_stock_cards: int = 8,
        maximum_nodes: int = 250_000,
    ) -> None:
        if root_seat not in {0, 1}:
            raise ValueError("oracle_root_seat_must_be_zero_or_one")
        self.rules = rules
        self.root_seat = root_seat
        self.maximum_stock_cards = max(0, int(maximum_stock_cards))
        self.maximum_nodes = max(1, int(maximum_nodes))
        self._memo: dict[OracleEndgameState, float] = {}
        self._nodes = 0

    def solve(self, state: OracleEndgameState) -> OracleEndgameResult:
        self._validate_state(state)
        if len(state.stock) > self.maximum_stock_cards:
            return OracleEndgameResult(
                selected_label=None,
                p_win=0.0,
                action_values=(),
                exact=False,
                nodes=0,
                reason=(
                    "unsupported_stock_size:"
                    f"{len(state.stock)}>{self.maximum_stock_cards}"
                ),
            )
        self._memo.clear()
        self._nodes = 0
        if state.current_seat == self.root_seat and not state.needs_draw:
            action_values = self._discard_values(state)
            selected = max(
                action_values,
                key=lambda item: (item.p_win, item.label),
                default=None,
            )
            return OracleEndgameResult(
                selected_label=selected.label if selected else None,
                p_win=selected.p_win if selected else 0.0,
                action_values=action_values,
                exact=True,
                nodes=self._nodes,
                reason="independent_exact_fully_observed_endgame",
            )
        value = self._value(state)
        return OracleEndgameResult(
            selected_label=None,
            p_win=value,
            action_values=(),
            exact=True,
            nodes=self._nodes,
            reason="independent_exact_fully_observed_endgame",
        )

    def _value(self, state: OracleEndgameState) -> float:
        cached = self._memo.get(state)
        if cached is not None:
            return cached
        self._nodes += 1
        if self._nodes > self.maximum_nodes:
            raise RuntimeError("oracle_endgame_node_budget_exhausted")
        if state.needs_draw:
            value = self._draw_value(state)
        else:
            values = self._discard_values(state)
            if not values:
                value = 0.0
            elif state.current_seat == self.root_seat:
                value = max(item.p_win for item in values)
            else:
                value = min(item.p_win for item in values)
        self._memo[state] = value
        return value

    def _draw_value(self, state: OracleEndgameState) -> float:
        if not state.stock:
            return 0.0
        counts = Counter(state.stock)
        total = len(state.stock)
        expected = 0.0
        seat = state.current_seat
        for label, amount in counts.items():
            stock = list(state.stock)
            stock.remove(label)
            transition = apply_draw_auto_meld_oracle(
                state.hands[seat],
                state.melds[seat],
                label,
            )
            hu = evaluate_hu_oracle(
                transition.hand,
                transition.melds,
                self.rules,
            )
            if hu.can_hu:
                child = 1.0 if seat == self.root_seat else 0.0
            else:
                hands = list(state.hands)
                melds = list(state.melds)
                hands[seat] = transition.hand
                melds[seat] = transition.melds
                child = self._value(
                    OracleEndgameState(
                        hands=tuple(hands),
                        melds=tuple(melds),
                        stock=tuple(sorted(stock)),
                        current_seat=seat,
                        needs_draw=transition.skip_discard,
                        passed_chi=state.passed_chi,
                        passed_peng=state.passed_peng,
                    )
                )
            expected += amount / total * child
        return expected

    def _discard_values(
        self,
        state: OracleEndgameState,
    ) -> tuple[OracleActionValue, ...]:
        seat = state.current_seat
        labels = legal_discard_labels_oracle(state.hands[seat])
        raw = [
            (label, self._after_discard(state, label))
            for label in labels
        ]
        if not raw:
            return ()
        best = max(value for _label, value in raw)
        return tuple(
            OracleActionValue(
                label=label,
                p_win=value,
                regret=(best - value if seat == self.root_seat else value - min(v for _, v in raw)),
            )
            for label, value in raw
        )

    def _after_discard(
        self,
        state: OracleEndgameState,
        label: str,
    ) -> float:
        source = state.current_seat
        responder = 1 - source
        hands = [list(hand) for hand in state.hands]
        hands[source].remove(label)
        hands_tuple = (tuple(sorted(hands[0])), tuple(sorted(hands[1])))
        response = legal_response_actions_oracle(
            hands_tuple[responder],
            state.melds[responder],
            label,
            seat=responder,
            source_seat=source,
            players=2,
            rules=self.rules,
            passed_chi=state.passed_chi[responder],
            passed_peng=state.passed_peng[responder],
        )
        actions = response.actions or (OracleAction(type="PASS"),)
        values = [
            self._response_value(
                state,
                hands_tuple=hands_tuple,
                source=source,
                responder=responder,
                pending=label,
                action=action,
                legal_actions=actions,
            )
            for action in actions
        ]
        return max(values) if responder == self.root_seat else min(values)

    def _response_value(
        self,
        state: OracleEndgameState,
        *,
        hands_tuple: tuple[tuple[str, ...], tuple[str, ...]],
        source: int,
        responder: int,
        pending: str,
        action: OracleAction,
        legal_actions: Sequence[OracleAction],
    ) -> float:
        if action.type.upper() == "HU":
            return 1.0 if responder == self.root_seat else 0.0
        transition = apply_response_action_oracle(
            hands_tuple[responder],
            state.melds[responder],
            pending,
            action,
            legal_actions=legal_actions,
            passed_chi=state.passed_chi[responder],
            passed_peng=state.passed_peng[responder],
        )
        hands = list(hands_tuple)
        melds = list(state.melds)
        passed_chi = list(state.passed_chi)
        passed_peng = list(state.passed_peng)
        hands[responder] = transition.hand
        melds[responder] = transition.melds
        passed_chi[responder] = transition.passed_chi
        passed_peng[responder] = transition.passed_peng
        claimed = action.type.upper() in {"CHI", "PENG"}
        return self._value(
            OracleEndgameState(
                hands=tuple(hands),
                melds=tuple(melds),
                stock=state.stock,
                current_seat=responder,
                needs_draw=not claimed,
                passed_chi=tuple(passed_chi),
                passed_peng=tuple(passed_peng),
            )
        )

    @staticmethod
    def _validate_state(state: OracleEndgameState) -> None:
        if len(state.hands) != 2 or len(state.melds) != 2:
            raise ValueError("oracle_endgame_requires_two_players")
        if state.current_seat not in {0, 1}:
            raise ValueError("oracle_endgame_current_seat_invalid")


__all__ = [
    "IndependentEndgameOracle",
    "OracleActionValue",
    "OracleEndgameResult",
    "OracleEndgameState",
]
