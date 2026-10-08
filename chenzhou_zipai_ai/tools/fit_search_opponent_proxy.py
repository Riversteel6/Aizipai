"""Fit a fast rollout proxy to frozen information-set-search traces."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.ismcts import public_view_from_dict
from ai.opponent_proxy import FastInformationSetProxyPolicy
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
    IndependentPolicyWeights,
)
from engine.rules import rules_for_room


TARGET_POLICY = "information_set_search_v2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-trace",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--holdout-trace",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--target-policy", default=TARGET_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = collect_cases(
        args.train_trace,
        target_policy=str(args.target_policy),
    )
    holdout = collect_cases(
        args.holdout_trace,
        target_policy=str(args.target_policy),
    )
    overlap = {
        str(case["state_before_hash"])
        for case in train
    } & {
        str(case["state_before_hash"])
        for case in holdout
    }
    if overlap:
        raise ValueError(
            f"search_proxy_train_holdout_overlap:{len(overlap)}"
        )
    configurations = candidate_configurations()
    training_results = [
        evaluate_weights(train, weights=weights)
        for weights in configurations
    ]
    selected = max(
        training_results,
        key=lambda row: (
            float(row["agreement_rate"]),
            -int(row["disagreements"]),
            -_configuration_complexity(row["weights"]),
            _weights_key(row["weights"]),
        ),
    )
    selected_weights = IndependentPolicyWeights(
        **selected["weights"]
    )
    holdout_result = evaluate_weights(
        holdout,
        weights=selected_weights,
    )
    references = {
        policy.name: evaluate_policy(holdout, policy=policy)
        for policy in (
            FastInformationSetProxyPolicy(),
            IndependentFastRolloutPolicy(),
            IndependentFastPressurePolicy(),
            IndependentFastDenialPolicy(),
        )
    }
    report = {
        "ok": bool(train) and bool(holdout) and not overlap,
        "schema_version": "search-opponent-proxy-fit-v1",
        "target_policy": str(args.target_policy),
        "train_traces": [str(path) for path in args.train_trace],
        "holdout_traces": [
            str(path)
            for path in args.holdout_trace
        ],
        "train_states": len(train),
        "holdout_states": len(holdout),
        "state_overlap": len(overlap),
        "configurations": len(configurations),
        "selected_training_result": selected,
        "selected_holdout_result": holdout_result,
        "holdout_references": references,
        "top_training_results": sorted(
            training_results,
            key=lambda row: (
                float(row["agreement_rate"]),
                -_configuration_complexity(row["weights"]),
                _weights_key(row["weights"]),
            ),
            reverse=True,
        )[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "train_states": report["train_states"],
                "holdout_states": report["holdout_states"],
                "configurations": report["configurations"],
                "train_agreement": selected["agreement_rate"],
                "holdout_agreement": holdout_result[
                    "agreement_rate"
                ],
                "holdout_reference_agreement": {
                    name: row["agreement_rate"]
                    for name, row in references.items()
                },
                "selected_weights": selected["weights"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def collect_cases(
    paths: Sequence[Path],
    *,
    target_policy: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for game in _read_jsonl(path):
            for trace in game.get("decision_trace") or ():
                if (
                    str(trace.get("phase") or "") != "discard"
                    or str(trace.get("policy") or "")
                    != target_policy
                ):
                    continue
                state_hash = str(
                    trace.get("state_before_hash") or ""
                )
                public_state = trace.get("public_state")
                selected_key = str(trace.get("selected_key") or "")
                if (
                    not state_hash
                    or state_hash in seen
                    or not isinstance(public_state, Mapping)
                    or not selected_key.startswith("DISCARD:")
                ):
                    continue
                seen.add(state_hash)
                cases.append(
                    {
                        "state_before_hash": state_hash,
                        "target_label": selected_key.split(":", 1)[1],
                        "public_state": dict(public_state),
                        "players": int(game.get("players") or 3),
                        "wildcard_enabled": bool(
                            game.get("wildcard_enabled")
                        ),
                    }
                )
    return cases


def candidate_configurations() -> list[IndependentPolicyWeights]:
    base = IndependentPolicyWeights()
    return [
        replace(
            base,
            exposed_xi=exposed_xi,
            shape=shape,
            route=route,
            discard_danger=discard_danger,
            peng_margin=15.0,
            chi_margin=20.0,
            discard_shortlist=shortlist,
        )
        for (
            exposed_xi,
            shape,
            route,
            discard_danger,
            shortlist,
        ) in itertools.product(
            (40.0, 45.0, 52.0),
            (9.0, 12.0, 15.0),
            (0.0, 3.0, 6.0, 8.0),
            (0.5, 1.0, 1.5, 2.0),
            (4, 6, 8),
        )
    ]


def evaluate_weights(
    cases: Sequence[Mapping[str, Any]],
    *,
    weights: IndependentPolicyWeights,
) -> dict[str, Any]:
    policy = IndependentFastRolloutPolicy()
    policy.weights = weights
    result = evaluate_policy(cases, policy=policy)
    return {
        **result,
        "weights": {
            field: getattr(weights, field)
            for field in weights.__dataclass_fields__
        },
    }


def evaluate_policy(
    cases: Sequence[Mapping[str, Any]],
    *,
    policy: Any,
) -> dict[str, Any]:
    agreements = 0
    disagreements: Counter[str] = Counter()
    started = time.perf_counter()
    for case in cases:
        view = public_view_from_dict(case["public_state"])
        rules = rules_for_room(
            wildcard_enabled=bool(case["wildcard_enabled"]),
            players=int(case["players"]),
        )
        selected = str(policy.choose_discard(view, rules))
        target = str(case["target_label"])
        agreements += int(selected == target)
        if selected != target:
            disagreements[f"{target}->{selected}"] += 1
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "states": len(cases),
        "agreements": agreements,
        "disagreements": len(cases) - agreements,
        "agreement_rate": round(
            agreements / len(cases),
            6,
        )
        if cases
        else 0.0,
        "mean_latency_ms": round(
            elapsed_ms / len(cases),
            6,
        )
        if cases
        else 0.0,
        "disagreement_pairs": dict(
            disagreements.most_common(20)
        ),
    }


def _configuration_complexity(
    weights: Mapping[str, Any],
) -> float:
    baseline = IndependentPolicyWeights()
    return sum(
        abs(float(weights[name]) - float(getattr(baseline, name)))
        for name in (
            "exposed_xi",
            "shape",
            "route",
            "discard_danger",
            "discard_shortlist",
        )
    )


def _weights_key(weights: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(
        float(weights[name])
        for name in sorted(weights)
    )


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
