"""Diagnose a bounded state range of the frozen full-proxy validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.dual_discard_validator import close_shared_dual_discard_executors
from ai.dual_validated_candidate import (
    ProfessionalParallelMultiOpponentRobustV81ResearchPolicy,
)
from ai.opponent_belief import PublicOpponentBeliefModel
from audit.independent_opponent import (
    IndependentFastDenialPolicy,
    IndependentFastPressurePolicy,
    IndependentFastRolloutPolicy,
)
from tools.evaluate_hybrid_response_proxies import HYBRID_RESPONSE_PROXY_TYPES
from tools.evaluate_opponent_distribution_split import _load_audits
from tools.evaluate_proxy_public_belief_validator import verify_capability_reports
from tools.evaluate_public_belief_validator import evaluate_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = diagnose_range(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(raw).hexdigest(),
        start_index=int(args.start_index),
        end_index=int(args.end_index),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "range": [report["start_index"], report["end_index"]],
                "states": report["states"],
                "health_errors": report["health_errors"],
                "incomplete_states": report["incomplete_states"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def diagnose_range(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    inputs = dict(preregistration["inputs"])
    verify_capability_reports(inputs)
    belief_report = _read_json(Path(inputs["belief_report"]))
    belief_model = PublicOpponentBeliefModel.from_dict(
        belief_report["final_model"]
    )
    benchmark = _read_json(Path(inputs["benchmark"]))
    benchmark_rows = {
        str(row["state_before_hash"]): row for row in benchmark["rows"]
    }
    state_ids = select_state_ids(
        tuple(sorted(benchmark_rows)),
        start_index=start_index,
        end_index=end_index,
    )
    batch_4 = _load_audits(Path(inputs["heldout_batch_4"]))
    batch_5 = _load_audits(Path(inputs["heldout_batch_5"]))
    anchor = ProfessionalParallelMultiOpponentRobustV81ResearchPolicy(
        rollout_policy_factories=(
            IndependentFastRolloutPolicy,
            IndependentFastPressurePolicy,
            IndependentFastDenialPolicy,
        )
    )
    validator_config = anchor.discard_validator.config
    slots = int(preregistration["distribution"]["slots"])
    rows: list[dict[str, Any]] = []
    try:
        for index, state_id in enumerate(state_ids, start=start_index):
            row = evaluate_state(
                benchmark_rows[state_id],
                batch_4[state_id],
                batch_5[state_id],
                belief_model=belief_model,
                validator_config=validator_config,
                slots=slots,
                profile_factories=HYBRID_RESPONSE_PROXY_TYPES,
            )
            diagnostic = {
                "index": index,
                "state_before_hash": state_id,
                "elapsed_ms": row["elapsed_ms"],
                "validation_complete": row["validation_complete"],
                "health_errors": row["health_errors"],
                "health_details": row["health_details"],
                "posterior": row["posterior"],
                "schedule_counts": row["schedule_counts"],
                "selected_label": row["candidate_selected_label"],
            }
            rows.append(diagnostic)
            print(json.dumps(diagnostic, ensure_ascii=False), flush=True)
    finally:
        close_shared_dual_discard_executors()
    return {
        "schema_version": "proxy-public-belief-validator-range-diagnostic-v1",
        "promotion_effect": "none",
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "start_index": start_index,
        "end_index": end_index,
        "states": len(rows),
        "health_errors": sum(int(row["health_errors"]) for row in rows),
        "incomplete_states": sum(
            not bool(row["validation_complete"]) for row in rows
        ),
        "rows": rows,
    }


def select_state_ids(
    state_ids: tuple[str, ...],
    *,
    start_index: int,
    end_index: int,
) -> tuple[str, ...]:
    if start_index < 1 or end_index < start_index or end_index > len(state_ids):
        raise ValueError("validator_range_bounds")
    return state_ids[start_index - 1 : end_index]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
