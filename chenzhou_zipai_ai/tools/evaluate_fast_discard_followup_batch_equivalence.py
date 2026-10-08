"""Verify exact proxy equivalence for batched follow-up discard features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.features import (
    QUICK_POTENTIAL_LABELS,
    quick_potential_after_each_removal_from_counts,
    quick_potential_from_counts,
)
from ai.full_game_simulator import (
    _discardable_labels,
    _draw_shape_relevance,
    _public_discard_danger,
    _remove_one,
)
from ai.ismcts import public_view_from_dict
from ai.opponent_proxy import (
    _QUICK_INDEX,
    _fast_discard_values,
    _fast_followup_value,
)
from engine.cards import RED_LABELS
from engine.rules import rules_for_room
from engine.xi_calculator import meld_xi
from tools.evaluate_anchored_structural_discard_proxies import (
    PROXY_TYPES,
    sha256_file,
)
from tools.fit_anchored_teacher_discard_proxies import collect_decisions


_AFFECTED_PROFILES = frozenset(
    {
        "aggressive_meld",
        "defensive_search",
        "information_set_search",
        "red_black_search",
    }
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_equivalence(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "focused_gate_pass": report["focused_gate_pass"],
                "states": report["states"],
                "quick_potential_values_checked": report[
                    "quick_potential_values_checked"
                ],
                "quick_potential_value_mismatches": report[
                    "quick_potential_value_mismatches"
                ],
                "proxy_value_mismatches": report["proxy_value_mismatches"],
                "proxy_action_mismatches": report["proxy_action_mismatches"],
                "p95_inference_ms": report["p95_inference_ms"],
                "failures": report["focused_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] and report["focused_gate_pass"] else 1


def run_equivalence(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    _verify_input_hash(inputs, "trace")
    for key in sorted(inputs):
        if key.endswith("_report") and f"{key}_sha256" in inputs:
            _verify_input_hash(inputs, key)
    decisions, integrity_errors = collect_decisions(
        Path(inputs["trace"]),
        profiles=tuple(PROXY_TYPES),
        folds=5,
    )

    states = 0
    quick_checked = 0
    quick_mismatches = 0
    value_mismatches = Counter()
    action_mismatches = Counter()
    inference_times: list[float] = []
    for profile, policy_type in PROXY_TYPES.items():
        policy = policy_type()
        for decision in decisions[profile]:
            rules = rules_for_room(
                wildcard_enabled=decision.wildcard_enabled,
                players=decision.players,
            )
            view = public_view_from_dict(decision.public_state)
            started = time.perf_counter()
            selected = policy.choose_discard(view, rules)
            inference_times.append((time.perf_counter() - started) * 1000.0)
            states += 1
            if profile not in _AFFECTED_PROFILES:
                continue
            labels = _scored_labels(profile, policy, view, rules)
            old_values, checked, mismatches = _reference_fast_discard_values(
                view,
                rules,
                labels,
            )
            new_values = _fast_discard_values(view, rules, labels)
            quick_checked += checked
            quick_mismatches += mismatches
            if old_values != new_values:
                value_mismatches[profile] += 1
            old_selected = _select_from_values(
                profile,
                view,
                labels,
                old_values,
            )
            if selected != old_selected:
                action_mismatches[profile] += 1

    elapsed = sorted(inference_times)
    p95 = elapsed[min(len(elapsed) - 1, int((len(elapsed) - 1) * 0.95))]
    gate = dict(preregistration["focused_gate"])
    failures = _focused_gate_failures(
        gate,
        states=states,
        quick_mismatches=quick_mismatches,
        value_mismatches=sum(value_mismatches.values()),
        action_mismatches=sum(action_mismatches.values()),
        p95_inference_ms=p95,
    )
    if integrity_errors:
        failures.append("integrity_errors")
    return {
        "ok": not integrity_errors and states > 0,
        "schema_version": "fast-discard-followup-batch-equivalence-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "trace": str(inputs["trace"]),
        "trace_sha256": sha256_file(Path(inputs["trace"])),
        "states": states,
        "affected_profiles": sorted(_AFFECTED_PROFILES),
        "quick_potential_values_checked": quick_checked,
        "quick_potential_value_mismatches": quick_mismatches,
        "proxy_value_mismatches": sum(value_mismatches.values()),
        "proxy_value_mismatches_by_profile": dict(value_mismatches),
        "proxy_action_mismatches": sum(action_mismatches.values()),
        "proxy_action_mismatches_by_profile": dict(action_mismatches),
        "p95_inference_ms": round(p95, 6),
        "maximum_inference_ms": round(max(elapsed, default=0.0), 6),
        "integrity_errors": integrity_errors,
        "focused_gate": gate,
        "focused_gate_pass": not failures,
        "focused_gate_failures": failures,
    }


def _scored_labels(
    profile: str,
    policy: Any,
    view: Any,
    rules: dict[str, Any],
) -> list[str]:
    labels = _discardable_labels(view.hand)
    if profile in {"aggressive_meld", "information_set_search"}:
        shortlist_size = 8 if rules.get("wildcard", {}).get("enabled", False) else 4
        return sorted(
            labels,
            key=lambda label: policy._cheap_discard_value(view, label, rules),
            reverse=True,
        )[:shortlist_size]
    return labels


def _select_from_values(
    profile: str,
    view: Any,
    labels: list[str],
    values: Mapping[str, float],
) -> str:
    if profile in {"aggressive_meld", "information_set_search"}:
        return max(labels, key=values.__getitem__)
    if profile == "defensive_search":
        return max(
            labels,
            key=lambda label: (
                values[label] - _public_discard_danger(label, view) * 2.5,
                label,
            ),
        )
    current_red = sum(label in RED_LABELS for label in view.hand)
    chase_red = current_red >= 7

    def route_value(label: str) -> float:
        after_red = current_red - int(label in RED_LABELS)
        route_bonus = (
            after_red * 18.0
            if chase_red
            else -min(abs(after_red), abs(after_red - 1)) * 22.0
        )
        return values[label] + route_bonus

    return max(labels, key=lambda label: (route_value(label), label))


def _reference_fast_discard_values(
    view: Any,
    rules: dict[str, Any],
    labels: list[str],
) -> tuple[dict[str, float], int, int]:
    base_counts = [0] * len(QUICK_POTENTIAL_LABELS)
    for card in view.hand:
        index = _QUICK_INDEX.get(card)
        if index is not None:
            base_counts[index] += 1
    existing_xi = sum(
        meld_xi(list(meld.cards), kind=meld.kind, rules=rules)
        for meld in view.own_melds
    )
    unseen = Counter(dict(view.remaining_counts))
    draw_sample_size = 12 if rules.get("wildcard", {}).get("enabled", False) else 8
    checked = 0
    mismatches = 0
    values: dict[str, float] = {}
    for label in labels:
        discard_index = _QUICK_INDEX[label]
        after_counts = list(base_counts)
        after_counts[discard_index] -= 1
        after = tuple(_remove_one(view.hand, label))
        current = (
            quick_potential_from_counts(tuple(after_counts)) * 12.0
            + existing_xi * 45.0
            + sum(card in RED_LABELS for card in after) * 3.0
        )
        sampled_draws = sorted(
            unseen.items(),
            key=lambda item: (item[1], _draw_shape_relevance(item[0], after)),
            reverse=True,
        )[:draw_sample_size]
        sampled_total = sum(amount for _draw, amount in sampled_draws)
        expected = 0.0
        for draw, amount in sampled_draws:
            if amount <= 0:
                continue
            draw_index = _QUICK_INDEX.get(draw)
            if draw_index is None:
                future = _fast_followup_value((*after, draw), view.own_melds, rules)
            else:
                drawn_counts = list(after_counts)
                drawn_counts[draw_index] += 1
                batched = quick_potential_after_each_removal_from_counts(
                    tuple(drawn_counts)
                )
                followup_values: list[int] = []
                for followup_index, followup_amount in enumerate(drawn_counts[:-1]):
                    if followup_amount <= 0:
                        continue
                    direct_counts = list(drawn_counts)
                    direct_counts[followup_index] -= 1
                    direct = quick_potential_from_counts(tuple(direct_counts))
                    checked += 1
                    mismatches += int(batched[followup_index] != direct)
                    if followup_amount < 3:
                        followup_values.append(direct)
                future = (
                    max(followup_values) * 12.0 + existing_xi * 45.0
                    if followup_values
                    else -500.0
                )
            expected += amount / max(1, sampled_total) * future
        values[label] = round(
            current + expected * 0.35 - _public_discard_danger(label, view),
            9,
        )
    return values, checked, mismatches


def _focused_gate_failures(
    gate: Mapping[str, Any],
    *,
    states: int,
    quick_mismatches: int,
    value_mismatches: int,
    action_mismatches: int,
    p95_inference_ms: float,
) -> list[str]:
    failures: list[str] = []
    if states < int(gate["minimum_trace_states"]):
        failures.append("minimum_trace_states")
    if gate["require_zero_quick_potential_value_mismatches"] and quick_mismatches:
        failures.append("quick_potential_value_mismatches")
    if gate["require_zero_proxy_value_mismatches"] and value_mismatches:
        failures.append("proxy_value_mismatches")
    if gate["require_zero_proxy_action_mismatches"] and action_mismatches:
        failures.append("proxy_action_mismatches")
    if p95_inference_ms > float(gate["maximum_trace_p95_inference_ms"]):
        failures.append("p95_inference_ms")
    return failures


def _verify_input_hash(inputs: Mapping[str, Any], key: str) -> None:
    path = Path(str(inputs[key]))
    if sha256_file(path) != str(inputs[f"{key}_sha256"]):
        raise ValueError(f"fast_followup_batch_{key}_hash_mismatch")


if __name__ == "__main__":
    raise SystemExit(main())
