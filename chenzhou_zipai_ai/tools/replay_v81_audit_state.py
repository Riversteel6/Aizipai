"""Replay one external v8.1 audit state through direct and product paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from ai.dual_discard_validator import close_shared_dual_discard_executors
from ai.frozen_two_player_strategy import (
    FROZEN_CANDIDATE,
    FROZEN_RELEASES,
    _frozen_policy,
    choose_action as choose_product_action,
    stop_two_player_strategy_runtime,
)
from ai.full_game_simulator import _production_state_from_public_view
from ai.ismcts import public_view_from_dict
from ai.opponent_league import create_policy
from ai.pro_brain import choose_action as choose_production_action
from engine.chi_rules import enumerate_chi_plans
from engine.rules import rules_for_room


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_CASES = (
    WORKSPACE
    / "debug"
    / "external_v81_audit_20260822"
    / "v81_code_audit_master_pack_20260822"
    / "v81_suspicious_decisions_20260822.jsonl"
)
FINGERPRINT_PATHS = (
    "chenzhou_zipai_ai/ai/frozen_two_player_strategy.py",
    "chenzhou_zipai_ai/ai/dual_validated_candidate.py",
    "chenzhou_zipai_ai/ai/dual_discard_validator.py",
    "chenzhou_zipai_ai/ai/ismcts.py",
    "chenzhou_zipai_ai/ai/full_game_simulator.py",
    "chenzhou_zipai_ai/engine/rules.py",
    "config/rules.yaml",
    "config/two_player_product_freeze_20260808.json",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_fingerprint(paths: Iterable[str]) -> str:
    records = []
    for relative in sorted(paths):
        path = WORKSPACE / relative
        records.append(f"{relative}:{_file_sha256(path)}")
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _git_status_fingerprint() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = result.stdout if result.returncode == 0 else result.stderr
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_case(
    path: str | Path,
    *,
    seed: int,
    opponent: str | None = None,
    event_index: int | None = None,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if int(row.get("seed", -1)) != seed:
                continue
            if opponent is not None and row.get("opponent") != opponent:
                continue
            if event_index is not None and int(row.get("event_index", -1)) != event_index:
                continue
            row["_source_line"] = line_number
            matches.append(row)
    if not matches:
        raise ValueError("audit_case_not_found")
    if len(matches) != 1:
        choices = [
            {
                "opponent": row.get("opponent"),
                "event_index": row.get("event_index"),
                "category": row.get("category"),
            }
            for row in matches
        ]
        raise ValueError(f"audit_case_ambiguous:{choices}")
    return matches[0]


def _last_discard_event(policy: Any) -> dict[str, Any] | None:
    events = policy.discard_events()
    return dict(events[-1]) if events else None


def _last_response_event(policy: Any) -> dict[str, Any] | None:
    events = policy.response_events()
    return dict(events[-1]) if events else None


def _response_types(case: dict[str, Any]) -> set[str]:
    keys: list[object] = [
        case.get("production_key"),
        case.get("actual_selected_key"),
        case.get("empirical_best_key"),
    ]
    keys.extend(
        item.get("candidate_key")
        for item in case.get("challengers") or ()
        if isinstance(item, dict)
    )
    types = {"PASS"}
    for value in keys:
        key = str(value or "").upper()
        if key == "HU" or key.startswith("HU:"):
            types.add("HU")
        elif key.startswith("PENG:"):
            types.add("PENG")
        elif key.startswith("CHI:"):
            types.add("CHI")
    return types


def _state_for_case(
    case: dict[str, Any],
    view: Any,
    rules: dict[str, Any],
) -> tuple[dict[str, object], str]:
    category = str(case.get("category") or "")
    if category.startswith("discard_"):
        return (
            _production_state_from_public_view(
                view,
                legal_actions=[{"type": "DISCARD"}],
                pending_card=None,
                chi_options=None,
            ),
            "discard",
        )
    pending = str(case.get("pending_card") or view.pending_card or "")
    legal_types = (
        {"HU", "PASS"}
        if category == "legal_hu_passed"
        else _response_types(case)
    )
    chi_options = None
    if "CHI" in legal_types:
        plans = enumerate_chi_plans(
            list(view.hand),
            pending,
            allow_1510=bool(rules.get("rules", {}).get("allow_1510", False)),
        )
        unique_groups: list[tuple[str, ...]] = []
        for plan in plans:
            group = tuple(plan.initial_group)
            if group not in unique_groups:
                unique_groups.append(group)
        chi_options = [
            {
                "option_id": f"audit_chi_{index:03d}",
                "labels": list(group),
                "confidence": 1.0,
            }
            for index, group in enumerate(unique_groups, start=1)
        ]
    return (
        _production_state_from_public_view(
            view,
            legal_actions=[{"type": action_type} for action_type in sorted(legal_types)],
            pending_card=pending,
            chi_options=chi_options,
        ),
        "response",
    )


def replay_case(case: dict[str, Any], *, run_direct: bool, run_product: bool) -> dict[str, Any]:
    started = time.perf_counter()
    view = public_view_from_dict(case["public_view"])
    player_count = len(view.all_melds)
    rules = rules_for_room(wildcard_enabled=True, players=player_count)
    state, root_kind = _state_for_case(case, view, rules)
    production = choose_production_action(
        state,
        rules=rules,
        parallel_evaluation=True,
    )
    legal_labels = sorted(set(view.hand)) if root_kind == "discard" else []
    report: dict[str, Any] = {
        "ok": True,
        "schema_version": "v81-product-state-replay-v1",
        "provenance": {
            "git_head": _git_head(),
            "git_status_sha256": _git_status_fingerprint(),
            "source_fingerprint": _combined_fingerprint(FINGERPRINT_PATHS),
            "rules_fingerprint": _file_sha256(WORKSPACE / "config" / "rules.yaml"),
            "config_fingerprint": _file_sha256(
                WORKSPACE / "config" / "two_player_product_freeze_20260808.json"
            ),
            "candidate": FROZEN_CANDIDATE,
            "release": FROZEN_RELEASES[True],
            "objective_version": "legacy_scalar_reward",
            "python": platform.python_version(),
        },
        "case": {
            "source_line": case.get("_source_line"),
            "source_archive_sha256": case.get("source_archive_sha256"),
            "category": case.get("category"),
            "opponent": case.get("opponent"),
            "seed": case.get("seed"),
            "event_index": case.get("event_index"),
            "public_view_id": case.get("public_view_id"),
        },
        "logged_rc2": {
            "production_label": case.get("production_label"),
            "baseline_selected_label": case.get("baseline_selected_label"),
            "validation_selected_label": case.get("validation_selected_label"),
            "proposed_search_selected_label": case.get("proposed_search_selected_label"),
            "actual_selected_label": case.get("actual_selected_label"),
            "empirical_best_label": case.get("empirical_best_label"),
            "final_authorization_reason": case.get("final_authorization_reason"),
        },
        "current_rc3": {
            "engine_legal_discard_labels": legal_labels,
            "executable_discard_labels": legal_labels,
            "production_anchor": production.selected_label,
            "production_matches_logged_rc2": (
                production.selected_label == case.get("production_label")
            ),
        },
    }
    direct_event: dict[str, Any] | None = None
    product_event: dict[str, Any] | None = None
    if run_direct:
        direct_policy = create_policy(FROZEN_CANDIDATE)
        if root_kind == "discard":
            direct_label = direct_policy.choose_discard(
                view,
                rules,
                production_decision=production,
            )
            direct_event = _last_discard_event(direct_policy)
        else:
            raise ValueError("direct_response_replay_not_supported_use_product_path")
        report["current_rc3"]["direct_candidate"] = {
            "selected_label": direct_label,
            "event": direct_event,
        }
    if run_product:
        product_policy = _frozen_policy()
        product_event_count = len(
            product_policy.discard_events()
            if root_kind == "discard"
            else product_policy.response_events()
        )
        product_decision = choose_product_action(
            state,
            rules=rules,
            baseline_decision=production,
        )
        current_events = (
            product_policy.discard_events()
            if root_kind == "discard"
            else product_policy.response_events()
        )
        product_event = (
            dict(current_events[-1])
            if len(current_events) > product_event_count
            else None
        )
        report["current_rc3"]["product_path"] = {
            "selected_action": product_decision.selected_action,
            "selected_label": product_decision.selected_label,
            "policy_version": product_decision.policy_version,
            "strategy_context": product_decision.context_snapshot.get("product_strategy"),
            "event": product_event,
        }
    direct = report["current_rc3"].get("direct_candidate")
    product = report["current_rc3"].get("product_path")
    action_parity = (
        direct["selected_label"] == product["selected_label"]
        if direct is not None and product is not None
        else None
    )
    state_parity = (
        direct_event.get("public_view_id") == product_event.get("public_view_id")
        if direct_event is not None and product_event is not None
        else None
    )
    report["current_rc3"]["product_direct_action_parity"] = action_parity
    report["current_rc3"]["product_direct_state_parity"] = state_parity
    report["current_rc3"]["product_direct_parity"] = (
        action_parity and state_parity
        if action_parity is not None and state_parity is not None
        else None
    )
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--opponent")
    parser.add_argument("--event-index", type=int)
    parser.add_argument("--path", choices=("both", "direct", "product"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        case = load_case(
            args.input,
            seed=args.seed,
            opponent=args.opponent,
            event_index=args.event_index,
        )
        report = replay_case(
            case,
            run_direct=args.path in {"both", "direct"},
            run_product=args.path in {"both", "product"},
        )
        exit_code = 0
    except Exception as exc:
        report = {
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
        exit_code = 1
    finally:
        stop_two_player_strategy_runtime(timeout_seconds=1.0)
        close_shared_dual_discard_executors(wait=False)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        summary = {
            "ok": report.get("ok"),
            "output": str(args.output.resolve()),
            "error": report.get("error"),
            "current_rc3": {
                key: report.get("current_rc3", {}).get(key)
                for key in (
                    "production_anchor",
                    "production_matches_logged_rc2",
                    "product_direct_action_parity",
                    "product_direct_state_parity",
                    "product_direct_parity",
                )
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
