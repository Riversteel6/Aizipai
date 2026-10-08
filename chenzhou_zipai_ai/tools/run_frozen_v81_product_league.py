"""Run a frozen v8.1 product release through its native league stack.

This verification-only entry point adapts the frozen product ``choose_action``
API to the joint-response simulator policy API.  It deliberately lives outside
the production policy registry so an old release cannot be selected at runtime.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _bootstrap_source_root(argv: list[str]) -> tuple[Path, str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--expected-release",
        default="2p-wang-20260809-rc3",
    )
    args, remaining = parser.parse_known_args(argv)
    source_root = args.source_root.resolve()
    required = source_root / "ai" / "frozen_two_player_strategy.py"
    if not required.is_file():
        parser.error(f"frozen source root is incomplete: {required}")
    sys.path.insert(0, str(source_root))
    return source_root, str(args.expected_release), remaining


def main(argv: list[str] | None = None) -> int:
    _source_root, expected_release, league_args = _bootstrap_source_root(
        list(sys.argv[1:] if argv is None else argv)
    )

    from ai import frozen_two_player_strategy as frozen
    from ai import opponent_league as league
    from ai.full_game_simulator import (
        ChiPlan,
        ProfessionalBrainSimulationPolicy,
        PublicView,
        _production_state_from_public_view,
    )

    observed_release = str(frozen.FROZEN_RELEASES.get(True) or "")
    if observed_release != expected_release:
        raise SystemExit(
            "frozen release mismatch:"
            f"expected={expected_release}:observed={observed_release}"
        )

    candidate = str(frozen.FROZEN_CANDIDATE)
    native_factory = league.POLICY_FACTORIES.get(candidate)
    if native_factory is None:
        raise SystemExit(f"frozen candidate factory missing: {candidate}")
    raw_policy = native_factory()

    class FrozenV81ProductSimulationPolicy(ProfessionalBrainSimulationPolicy):
        """Adapter around the exact frozen product route, for audit only."""

        name = "frozen_v81_product_rc3_verification"
        seat_aware_opponents = True

        def __init__(self) -> None:
            super().__init__()
            self._product_discard_events: list[dict[str, Any]] = []
            self._product_response_events: list[dict[str, Any]] = []

        def _choose(
            self,
            view: PublicView,
            rules: dict[str, Any],
            *,
            legal_actions: list[dict[str, str]],
            pending_card: str | None = None,
            chi_options: list[dict[str, object]] | None = None,
        ) -> Any:
            before_discard = len(raw_policy.discard_events())
            before_response = len(raw_policy.response_events())
            state = _production_state_from_public_view(
                view,
                legal_actions=legal_actions,
                pending_card=pending_card,
                chi_options=chi_options,
                seat_aware_opponents=True,
            )
            decision = frozen.choose_action(state, rules=rules)
            self._product_discard_events.extend(
                raw_policy.discard_events()[before_discard:]
            )
            self._product_response_events.extend(
                raw_policy.response_events()[before_response:]
            )
            return decision

        def choose_response(
            self,
            view: PublicView,
            legal_actions: tuple[Any, ...],
            plans: tuple[ChiPlan, ...],
            hu: Any | None,
            rules: dict[str, Any],
        ) -> str:
            legal_types = tuple(
                dict.fromkeys(str(action.type).upper() for action in legal_actions)
            )
            option_groups: list[tuple[str, str, str]] = []
            for plan in plans:
                if plan.initial_group not in option_groups:
                    option_groups.append(plan.initial_group)
            chi_options = [
                {
                    "option_id": f"sim_chi_{index:03d}",
                    "labels": list(group),
                    "confidence": 1.0,
                }
                for index, group in enumerate(option_groups, start=1)
            ]
            decision = self._choose(
                view,
                rules,
                legal_actions=[{"type": action_type} for action_type in legal_types],
                pending_card=view.pending_card,
                chi_options=chi_options or None,
            )
            self._remember_decision(decision)
            selected_type = str(decision.selected_action).upper()
            if selected_type == "CHI":
                selected_plan = self._product_chi_plan_for_decision(
                    decision,
                    list(plans),
                )
                for action in legal_actions:
                    if (
                        str(action.type).upper() == "CHI"
                        and tuple(action.consumed_from_hand)
                        == tuple(selected_plan.consumed_from_hand)
                        and tuple(action.meld_groups) == tuple(selected_plan.groups)
                    ):
                        return str(action.key)
                raise ValueError("frozen_product_chi_plan_not_legal")
            for action in legal_actions:
                if str(action.type).upper() == selected_type:
                    return str(action.key)
            raise ValueError(
                f"frozen_product_response_not_legal:{selected_type}"
            )

        @staticmethod
        def _product_chi_plan_for_decision(
            decision: Any,
            plans: list[ChiPlan],
        ) -> ChiPlan:
            selected_eval = next(
                (
                    item
                    for item in decision.action_evals
                    if item.type == "CHI"
                    and item.action.option_id == decision.selected_option_id
                ),
                None,
            )
            if selected_eval is None:
                raise ValueError("frozen_product_chi_eval_missing")
            consumed = Counter(
                selected_eval.debug_details.get("consumed_from_hand") or ()
            )
            initial = Counter(selected_eval.debug_details.get("meld_cards") or ())
            compares = tuple(
                sorted(
                    tuple(sorted(group))
                    for group in selected_eval.debug_details.get("compare_groups")
                    or ()
                )
            )
            for plan in plans:
                plan_compares = tuple(
                    sorted(tuple(sorted(group)) for group in plan.compare_groups)
                )
                if (
                    Counter(plan.initial_group) == initial
                    and Counter(plan.consumed_from_hand) == consumed
                    and plan_compares == compares
                ):
                    return plan
            raise ValueError("frozen_product_chi_plan_mapping_failed")

        def discard_events(self) -> tuple[dict[str, Any], ...]:
            return tuple(dict(event) for event in self._product_discard_events)

        def response_events(self) -> tuple[dict[str, Any], ...]:
            return tuple(dict(event) for event in self._product_response_events)

        def diagnostics(self) -> dict[str, Any]:
            return {
                "frozen_product_route_events": (
                    len(self._product_discard_events)
                    + len(self._product_response_events)
                ),
                "frozen_release_verified": 1,
                "product_discard_events": len(self._product_discard_events),
                "product_response_events": len(self._product_response_events),
            }

    frozen._frozen_policy = lambda: raw_policy
    league.POLICY_FACTORIES[candidate] = FrozenV81ProductSimulationPolicy

    from tools import run_opponent_league as runner

    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(runner.__file__).resolve()), *league_args]
        return int(runner.main())
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
