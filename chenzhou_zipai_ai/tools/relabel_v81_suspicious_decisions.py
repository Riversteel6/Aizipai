"""Re-evaluate the 179 external v8.1 suspicious states without trusting old labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from audit.independent_rules import evaluate_hu_oracle
from ai.dual_discard_validator import close_shared_dual_discard_executors
from ai.frozen_two_player_strategy import stop_two_player_strategy_runtime
from engine.rules import rules_for_room
from tools.replay_v81_audit_state import replay_case


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    WORKSPACE
    / "debug"
    / "external_v81_audit_20260822"
    / "v81_code_audit_master_pack_20260822"
    / "v81_suspicious_decisions_20260822.jsonl"
)
DEFAULT_OUTPUT = (
    WORKSPACE / "reports" / "v81_suspicious_decisions_relabel_20260822.jsonl"
)
DEFAULT_SUMMARY = (
    WORKSPACE / "reports" / "v81_suspicious_decisions_relabel_20260822.json"
)


def relabel_row(row: dict[str, Any], *, run_product: bool) -> dict[str, Any]:
    category = str(row.get("category") or "")
    before_selected = (
        row.get("actual_selected_label")
        or row.get("actual_selected_key")
    )
    result: dict[str, Any] = {
        "schema_version": "v81-suspicious-relabel-v1",
        "source_index": row.get("_source_index"),
        "seed": row.get("seed"),
        "opponent": row.get("opponent"),
        "event_index": row.get("event_index"),
        "category": category,
        "public_view_id": row.get("public_view_id"),
        "before_selected": before_selected,
        "before_action_type": _action_type(before_selected),
        "before_production": (
            row.get("production_label") or row.get("production_key")
        ),
        "legacy_search_suggestion": (
            row.get("empirical_best_label")
            or row.get("proposed_search_selected_label")
        ),
        "after_selected": None,
        "delta_pwin": None,
        "confidence_interval": None,
        "action_regret": None,
        "oracle_status": "UNRESOLVED",
        "oracle_reason": None,
        "acceptance": "FAIL",
    }
    if category == "legal_hu_passed":
        view = row["public_view"]
        seat = int(view["seat"])
        pending = str(row.get("pending_card") or view.get("pending_card"))
        own_melds = (view.get("all_melds") or [[], []])[seat]
        oracle = evaluate_hu_oracle(
            [*view["hand"], pending],
            own_melds,
            rules_for_room(wildcard_enabled=True, players=2),
        )
        result.update(
            {
                "after_selected": "HU" if oracle.can_hu else None,
                "after_pwin": 1.0 if oracle.can_hu else None,
                "oracle_status": "EXACT_ACTION" if oracle.can_hu else "FAIL",
                "oracle_reason": list(oracle.reasons),
                "acceptance": "PASS" if oracle.can_hu else "FAIL",
            }
        )
    elif int((row.get("public_view") or {}).get("stock_count") or 0) > 8:
        result["oracle_reason"] = (
            "public_information_set_not_exactly_solvable:"
            f"stock_count={row['public_view']['stock_count']};"
            "opponent_hand_hidden"
        )
        result["acceptance"] = "UNRESOLVED"

    if run_product:
        try:
            replay = replay_case(row, run_direct=False, run_product=True)
            product = replay.get("current_rc3", {}).get("product_path") or {}
            event = product.get("event") or {}
            result["after_selected"] = (
                product.get("selected_label")
                if category.startswith("discard_")
                else (
                    event.get("search_selected_key")
                    or product.get("selected_action")
                )
            )
            result["after_action_type"] = _action_type(result["after_selected"])
            result["product_release"] = product.get("policy_version")
            result["product_final_authorization"] = event.get(
                "final_authorization_reason"
            ) or event.get("reason") or (
                product.get("strategy_context") or {}
            ).get("route")
            result["product_fallback_reason"] = (
                product.get("strategy_context") or {}
            ).get("fallback_reason")
            result["product_elapsed_ms"] = (
                product.get("strategy_context") or {}
            ).get("elapsed_ms")
            paired = event.get("paired_advantages") or []
            selected = result["after_selected"]
            selected_candidate = next(
                (
                    item
                    for item in event.get("candidates") or ()
                    if item.get("key") == selected
                ),
                None,
            )
            if selected_candidate is not None:
                result["after_action_details"] = {
                    key: selected_candidate.get(key)
                    for key in (
                        "key",
                        "action_type",
                        "consumed_from_hand",
                        "meld_groups",
                    )
                }
            result["semantic_action_changed"] = _semantic_action_changed(
                row,
                result,
            )
            evidence = next(
                (
                    item
                    for item in paired
                    if item.get("candidate_key") == selected
                ),
                None,
            )
            if evidence is not None:
                result["delta_pwin"] = evidence.get(
                    "mean_delta_p_win",
                    evidence.get("mean_delta"),
                )
                result["confidence_interval"] = [
                    evidence.get("lower_confidence_bound"),
                    evidence.get("upper_confidence_bound"),
                ]
                result["action_regret"] = max(
                    0.0,
                    -float(result["delta_pwin"] or 0.0),
                )
                result["estimate_source"] = (
                    "current_paired_multistep_rollout_not_exact_oracle"
                )
            elif (
                result["after_selected"] == result["before_production"]
                and result["before_selected"] == result["before_production"]
            ):
                result["delta_pwin"] = 0.0
                result["confidence_interval"] = [0.0, 0.0]
                result["estimate_source"] = "same_action_as_production_anchor"
            if result["oracle_status"] != "EXACT_ACTION":
                result["oracle_status"] = "ROLLOUT_ONLY"
                result["acceptance"] = "UNRESOLVED"
        except Exception as exc:
            result["product_error"] = f"{type(exc).__name__}:{exc}"
    return result


def _action_type(value: object) -> str | None:
    text = str(value or "").upper()
    return text.split(":", 1)[0] if text else None


def _plan_signature(details: Any) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    if not isinstance(details, dict):
        return None
    consumed = tuple(sorted(str(value) for value in details.get("consumed_from_hand") or ()))
    groups = tuple(
        sorted(
            tuple(sorted(str(value) for value in group))
            for group in details.get("meld_groups") or ()
        )
    )
    return consumed, groups


def _semantic_action_changed(row: dict[str, Any], result: dict[str, Any]) -> bool | None:
    before_type = result.get("before_action_type")
    after_type = result.get("after_action_type")
    if before_type is None or after_type is None:
        return None
    if before_type != after_type:
        return True
    if before_type != "CHI":
        return False
    before_plan = _plan_signature(row.get("actual_candidate_stats"))
    after_plan = _plan_signature(result.get("after_action_details"))
    if before_plan is None or after_plan is None:
        return None
    return before_plan != after_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run-product", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source_rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed_source_rows = list(enumerate(source_rows, start=1))
    indexed_source_rows = [
        (index, row)
        for index, row in indexed_source_rows
        if index >= max(1, args.start_index)
        and (args.end_index is None or index <= args.end_index)
    ]
    if args.limit is not None:
        indexed_source_rows = indexed_source_rows[: max(0, args.limit)]
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing_rows = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.run_product:
            existing_rows = [
                row
                for row in existing_rows
                if "product_release" in row or "product_error" in row
            ]
    completed_keys = {row.get("source_index") for row in existing_rows}
    rows: list[dict[str, Any]] = list(existing_rows)
    file_mode = "a" if existing_rows else "w"
    try:
        with args.output.open(file_mode, encoding="utf-8") as output_handle:
            for source_index, row in indexed_source_rows:
                key = source_index
                if key in completed_keys:
                    continue
                row["_source_index"] = source_index
                relabeled = relabel_row(row, run_product=args.run_product)
                rows.append(relabeled)
                completed_keys.add(key)
                output_handle.write(
                    json.dumps(relabeled, ensure_ascii=False) + "\n"
                )
                output_handle.flush()
                if args.run_product:
                    print(
                        json.dumps(
                            {
                                "completed": len(rows),
                                "total": len(indexed_source_rows),
                            }
                        ),
                        flush=True,
                    )
    finally:
        stop_two_player_strategy_runtime(timeout_seconds=1.0)
        close_shared_dual_discard_executors(wait=False)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["oracle_status"])
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "schema_version": "v81-suspicious-relabel-summary-v1",
        "source_rows": len(indexed_source_rows),
        "output_rows": len(rows),
        "run_product": args.run_product,
        "oracle_status_counts": counts,
        "exact_oracle_actions": sum(
            row["oracle_status"] == "EXACT_ACTION" for row in rows
        ),
        "unresolved": sum(
            row["acceptance"] == "UNRESOLVED" for row in rows
        ),
        "fail": sum(row["acceptance"] == "FAIL" for row in rows),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "final_acceptance": (
            "PASS" if all(row["acceptance"] == "PASS" for row in rows) else "FAIL"
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
