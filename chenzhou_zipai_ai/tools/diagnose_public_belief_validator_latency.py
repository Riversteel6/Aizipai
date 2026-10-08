"""Attribute a public-belief validator deadline to individual proxy profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.dual_discard_validator import close_shared_dual_discard_executors
from ai.dual_validated_candidate import (
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy,
)
from ai.full_game_simulator import SimulationPolicy
from ai.opponent_belief import PublicOpponentBeliefModel
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from tools.evaluate_hybrid_response_proxies import HYBRID_RESPONSE_PROXY_TYPES
from tools.evaluate_opponent_distribution_split import _load_audits
from tools.evaluate_public_belief_validator import evaluate_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose_latency(
        preregistration_path=args.preregistration,
        state_id=str(args.state),
        repeats=max(1, int(args.repeats)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "state": report["state_before_hash"],
                "repeats": report["repeats"],
                "profiles": report["profile_summaries"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def diagnose_latency(
    *,
    preregistration_path: Path,
    state_id: str,
    repeats: int,
) -> dict[str, Any]:
    raw = preregistration_path.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    inputs = dict(preregistration["inputs"])
    belief_report = _read_json(Path(inputs["belief_report"]))
    belief_model = PublicOpponentBeliefModel.from_dict(
        belief_report["final_model"]
    )
    benchmark = _read_json(Path(inputs["benchmark"]))
    benchmark_rows = {
        str(row["state_before_hash"]): row for row in benchmark["rows"]
    }
    batch_4 = _load_audits(Path(inputs["heldout_batch_4"]))
    batch_5 = _load_audits(Path(inputs["heldout_batch_5"]))
    if state_id not in benchmark_rows or state_id not in batch_4 or state_id not in batch_5:
        raise ValueError(f"latency_diagnostic_unknown_state:{state_id}")

    anchor = ProfessionalParallelMultiOpponentRobustV81ResearchPolicy(
        rollout_policy_factories=(
            IndependentFastRolloutPolicy,
            IndependentFastPressurePolicy,
            IndependentFastDenialPolicy,
        )
    )
    validator_config = anchor.discard_validator.config
    slots = int(preregistration["distribution"]["slots"])
    profile_names = tuple(
        str(name) for name in preregistration["distribution"]["profile_order"]
    )
    if set(profile_names) != set(HYBRID_RESPONSE_PROXY_TYPES):
        raise ValueError("latency_diagnostic_profile_coverage_mismatch")

    rows: list[dict[str, Any]] = []
    try:
        for profile in profile_names:
            factory = HYBRID_RESPONSE_PROXY_TYPES[profile]
            forced_factories = _forced_factory_map(profile_names, factory)
            for repeat in range(1, repeats + 1):
                row = evaluate_state(
                    benchmark_rows[state_id],
                    batch_4[state_id],
                    batch_5[state_id],
                    belief_model=belief_model,
                    validator_config=validator_config,
                    slots=slots,
                    profile_factories=forced_factories,
                )
                diagnostic = {
                    "profile": profile,
                    "factory": factory.__name__,
                    "repeat": repeat,
                    "elapsed_ms": row["elapsed_ms"],
                    "validation_complete": row["validation_complete"],
                    "health_errors": row["health_errors"],
                    "health_details": row["health_details"],
                    "selected_label": row["candidate_selected_label"],
                }
                rows.append(diagnostic)
                print(json.dumps(diagnostic, ensure_ascii=False), flush=True)
                close_shared_dual_discard_executors()
    finally:
        close_shared_dual_discard_executors()

    return {
        "schema_version": "public-belief-validator-latency-diagnostic-v1",
        "promotion_effect": "none",
        "preregistration": str(preregistration_path),
        "preregistration_sha256": hashlib.sha256(raw).hexdigest(),
        "state_before_hash": state_id,
        "repeats": repeats,
        "isolation": (
            "The public posterior schedule, candidates, seeds, and frozen validator "
            "budget are retained, while every schedule slot is forced to one proxy "
            "factory for runtime attribution only."
        ),
        "rows": rows,
        "profile_summaries": _profile_summaries(rows, profile_names),
    }


def _forced_factory_map(
    profile_names: tuple[str, ...],
    factory: Callable[[], SimulationPolicy],
) -> dict[str, Callable[[], SimulationPolicy]]:
    return {name: factory for name in profile_names}


def _profile_summaries(
    rows: list[Mapping[str, Any]],
    profile_names: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for profile in profile_names:
        selected = [row for row in rows if row["profile"] == profile]
        elapsed = sorted(float(row["elapsed_ms"]) for row in selected)
        summaries[profile] = {
            "runs": len(selected),
            "complete_runs": sum(bool(row["validation_complete"]) for row in selected),
            "health_errors": sum(int(row["health_errors"]) for row in selected),
            "minimum_elapsed_ms": round(min(elapsed, default=0.0), 3),
            "median_elapsed_ms": round(
                elapsed[len(elapsed) // 2] if elapsed else 0.0,
                3,
            ),
            "maximum_elapsed_ms": round(max(elapsed, default=0.0), 3),
        }
    return summaries


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
