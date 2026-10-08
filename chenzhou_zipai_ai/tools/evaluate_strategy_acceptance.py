"""Evaluate strategy evidence against the frozen product acceptance contract."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    report = evaluate_acceptance(
        contract,
        evidence,
        evidence_base=args.evidence.parent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "strategy_promotion_pass": report[
                    "strategy_promotion_pass"
                ],
                "passed_modes": report["passed_modes"],
                "required_modes": report["required_modes"],
                "failure_count": len(report["failures"]),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["overall_pass"] else 1


def evaluate_acceptance(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    evidence_base: Path,
) -> dict[str, Any]:
    _require_schema(
        contract,
        "aizipai-strategy-acceptance-v1",
        "contract",
    )
    _require_schema(
        evidence,
        "aizipai-strategy-evidence-v1",
        "evidence",
    )
    mode_evidence = evidence.get("mode_evidence") or {}
    mode_results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for mode in contract.get("required_modes") or ():
        mode_id = str(mode["id"])
        result = _evaluate_mode(
            mode,
            mode_evidence.get(mode_id) or {},
            contract=contract,
            evidence_base=evidence_base,
        )
        mode_results[mode_id] = result
        failures.extend(
            f"{mode_id}:{failure}" for failure in result["failures"]
        )

    strategy_promotion_pass = bool(mode_results) and all(
        result["passed"] for result in mode_results.values()
    )
    device = _evaluate_device_soak(
        evidence.get("device_soak_report"),
        contract.get("device_soak") or {},
        mode_ids=tuple(mode_results),
        evidence_base=evidence_base,
    )
    external = _evaluate_external_calibration(
        evidence.get("external_calibration_report"),
        contract.get("external_calibration") or {},
        evidence_base=evidence_base,
    )
    failures.extend(f"device_soak:{item}" for item in device["failures"])
    failures.extend(
        f"external_calibration:{item}" for item in external["failures"]
    )
    overall_pass = (
        strategy_promotion_pass and device["passed"] and external["passed"]
    )
    return {
        "schema_version": "aizipai-strategy-acceptance-report-v1",
        "candidate_id": evidence.get("candidate_id"),
        "anchor_id": evidence.get("anchor_id"),
        "overall_pass": overall_pass,
        "strategy_promotion_pass": strategy_promotion_pass,
        "professional_level_claim_pass": overall_pass,
        "required_modes": len(mode_results),
        "passed_modes": sum(result["passed"] for result in mode_results.values()),
        "mode_results": mode_results,
        "device_soak": device,
        "external_calibration": external,
        "failures": failures,
    }


def _evaluate_mode(
    mode: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    evidence_base: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    role = str(evidence.get("dataset_role") or "missing")
    if role != "sealed":
        failures.append(f"formal_dataset_not_sealed:{role}")

    root = _evaluate_root_counterfactual(
        evidence.get("root_calibration_report"),
        contract.get("root_counterfactual") or {},
        evidence_base=evidence_base,
    )
    failures.extend(f"root:{item}" for item in root["failures"])

    formal = _evaluate_formal_league(
        evidence.get("formal_league_report"),
        evidence.get("paired_anchor_report"),
        mode=mode,
        thresholds=contract.get("formal_league") or {},
        root_states=int(root.get("states") or 0),
        minimum_root_coverage=float(
            (contract.get("root_counterfactual") or {}).get(
                "minimum_changed_action_coverage",
                1.0,
            )
        ),
        evidence_base=evidence_base,
    )
    failures.extend(f"formal:{item}" for item in formal["failures"])

    ordinary = _evaluate_ordinary_league(
        evidence.get("ordinary_league_report"),
        mode=mode,
        thresholds=contract.get("ordinary_league") or {},
        evidence_base=evidence_base,
    )
    failures.extend(f"ordinary:{item}" for item in ordinary["failures"])
    return {
        "passed": not failures,
        "dataset_role": role,
        "root_counterfactual": root,
        "formal_league": formal,
        "ordinary_league": ordinary,
        "failures": failures,
    }


def _evaluate_root_counterfactual(
    raw_path: Any,
    thresholds: Mapping[str, Any],
    *,
    evidence_base: Path,
) -> dict[str, Any]:
    path = _resolve_optional_path(raw_path, evidence_base)
    if path is None:
        return _failed("evidence_missing")
    if not path.exists():
        return _failed(f"file_missing:{path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "discard-validator-calibration-v1":
        return _failed("schema_mismatch")

    batches_required = int(thresholds["required_batches_per_state"])
    worlds_required = int(thresholds["required_worlds_per_batch"])
    audits_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[str] = []
    evidence_inputs = report.get("evidence_inputs") or ()
    for raw_input in evidence_inputs:
        input_path = _resolve_path(raw_input, path.parent)
        if not input_path.exists():
            failures.append(f"input_missing:{input_path}")
            continue
        seen_states: set[str] = set()
        for audit in _read_jsonl(input_path):
            state_hash = str(audit.get("state_before_hash") or "")
            if not state_hash or state_hash in seen_states:
                failures.append(f"invalid_state_identity:{input_path.name}")
                continue
            seen_states.add(state_hash)
            failures.extend(
                _audit_failures(
                    audit,
                    expected_worlds=worlds_required,
                    prefix=f"audit:{state_hash[:12]}",
                )
            )
            audits_by_state[state_hash].append(audit)

    rows = {
        str(row.get("state_before_hash") or ""): row
        for row in report.get("rows") or ()
    }
    if not rows or set(rows) != set(audits_by_state):
        failures.append("state_set_mismatch")
    for state_hash, audits in audits_by_state.items():
        if len(audits) != batches_required:
            failures.append(
                f"batch_count:{state_hash[:12]}:{len(audits)}"
            )
        seeds = [int(audit.get("audit_seed") or 0) for audit in audits]
        if len(seeds) != len(set(seeds)):
            failures.append(f"duplicate_audit_seed:{state_hash[:12]}")
        row = rows.get(state_hash) or {}
        if sorted(seeds) != sorted(int(seed) for seed in row.get("audit_seeds") or ()):
            failures.append(f"reported_seed_mismatch:{state_hash[:12]}")
        expected_pool = batches_required * worlds_required
        if int(row.get("pooled_worlds") or 0) != expected_pool:
            failures.append(f"pooled_world_count:{state_hash[:12]}")

    harmful = int(report.get("harmful_overrides") or 0)
    maximum_harmful = int(thresholds["maximum_harmful_overrides"])
    if harmful > maximum_harmful:
        failures.append(f"harmful_overrides:{harmful}>{maximum_harmful}")
    return {
        "passed": not failures,
        "path": str(path),
        "states": len(audits_by_state),
        "audit_batches": sum(len(items) for items in audits_by_state.values()),
        "paired_worlds": sum(
            len(audit.get("paired_worlds") or ())
            for items in audits_by_state.values()
            for audit in items
        ),
        "harmful_overrides": harmful,
        "failures": failures,
    }


def _audit_failures(
    audit: Mapping[str, Any],
    *,
    expected_worlds: int,
    prefix: str,
) -> list[str]:
    failures: list[str] = []
    worlds = list(audit.get("paired_worlds") or ())
    health = audit.get("search_health") or {}
    candidate_stats = list(audit.get("candidate_stats") or ())
    labels = {
        str(item.get("key") or "").rsplit(":", 1)[-1]
        for item in candidate_stats
    }
    checks = {
        "incomplete": bool(audit.get("complete")),
        "requested_worlds": int(audit.get("requested_paired_worlds") or 0)
        == expected_worlds,
        "completed_worlds": int(audit.get("completed_paired_worlds") or 0)
        == expected_worlds,
        "world_rows": len(worlds) == expected_worlds,
        "deadline_interruptions": int(health.get("deadline_interruptions") or 0)
        == 0,
        "determinization_failures": int(
            health.get("determinization_failures") or 0
        )
        == 0,
        "invariant_violations": int(
            health.get("rollout_invariant_violations") or 0
        )
        == 0,
        "coverage_failures": int(health.get("rollout_coverage_failures") or 0)
        == 0,
        "world_indexes": [world.get("world_index") for world in worlds]
        == list(range(expected_worlds)),
        "candidate_count": len(labels) >= 2,
        "candidate_visits": all(
            int(item.get("visits") or 0) == expected_worlds
            for item in candidate_stats
        ),
        "world_candidate_coverage": all(
            {
                str(outcome.get("candidate_key") or "").rsplit(":", 1)[-1]
                for outcome in world.get("outcomes") or ()
            }
            == labels
            for world in worlds
        ),
    }
    return [f"{prefix}:{name}" for name, passed in checks.items() if not passed]


def _evaluate_formal_league(
    raw_league_path: Any,
    raw_paired_path: Any,
    *,
    mode: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    root_states: int,
    minimum_root_coverage: float,
    evidence_base: Path,
) -> dict[str, Any]:
    league_path = _resolve_optional_path(raw_league_path, evidence_base)
    paired_path = _resolve_optional_path(raw_paired_path, evidence_base)
    if league_path is None or paired_path is None:
        return _failed("evidence_missing")
    if not league_path.exists() or not paired_path.exists():
        return _failed("file_missing")
    league = json.loads(league_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    _check_league_integrity(
        league,
        mode=mode,
        minimum_games=int(thresholds["minimum_games"]),
        failures=failures,
    )
    failures.extend(_rotation_balance_failures(league, mode=mode))
    if not paired.get("ok"):
        failures.append("paired_report_not_ok")
    if int(paired.get("games") or 0) < int(thresholds["minimum_games"]):
        failures.append("paired_games_below_minimum")
    if int(paired.get("players") or 0) != int(mode["players"]):
        failures.append("paired_players_mismatch")
    if bool(paired.get("wildcard_enabled")) != bool(mode["wildcard_enabled"]):
        failures.append("paired_wildcard_mismatch")

    summary = paired.get("paired_summary") or {}
    score_interval = summary.get("mean_score_delta_95") or (0.0, 0.0)
    outcome_interval = summary.get("mean_outcome_utility_delta_95") or (0.0, 0.0)
    if float(score_interval[0]) <= float(
        thresholds["minimum_paired_score_lower_bound"]
    ):
        failures.append(f"paired_score_lower_bound:{score_interval[0]}")
    if float(outcome_interval[0]) <= float(
        thresholds["minimum_paired_outcome_lower_bound"]
    ):
        failures.append(f"paired_outcome_lower_bound:{outcome_interval[0]}")
    if float(summary.get("outcome_sign_test_p") or 1.0) >= float(
        thresholds["maximum_outcome_sign_test_p"]
    ):
        failures.append("paired_outcome_sign_test")
    for profile, profile_summary in (paired.get("by_matchup") or {}).items():
        profile_outcome = profile_summary.get("mean_outcome_utility_delta_95") or (
            0.0,
            0.0,
        )
        profile_score = profile_summary.get("mean_score_delta_95") or (0.0, 0.0)
        if float(profile_outcome[1]) < 0.0 or float(profile_score[1]) < 0.0:
            failures.append(f"confidently_worse_matchup:{profile}")

    games = max(1, int(league.get("games") or 0))
    wins = int(league.get("candidate_wins") or 0)
    win_lower = _wilson_lower_bound(wins, games)
    if win_lower <= float(thresholds["minimum_strong_pool_win_lower_bound"]):
        failures.append(f"strong_pool_win_lower_bound:{win_lower}")

    diagnostics = league.get("candidate_diagnostics") or {}
    attempts = max(1, int(diagnostics.get("search_attempts") or 0))
    response_attempts = max(
        1,
        int(diagnostics.get("response_search_attempts") or 0),
    )
    fallback_fraction = float(
        diagnostics.get("decision_budget_fallbacks") or 0
    ) / attempts
    discard_deadline_fraction = float(
        diagnostics.get("search_deadline_interruptions") or 0
    ) / attempts
    response_deadline_fraction = float(
        diagnostics.get("response_deadline_interruptions") or 0
    ) / response_attempts
    if fallback_fraction > float(
        thresholds["maximum_decision_budget_fallback_fraction"]
    ):
        failures.append(f"decision_budget_fallback_fraction:{fallback_fraction:.6f}")
    if discard_deadline_fraction > float(
        thresholds["maximum_discard_deadline_fraction"]
    ):
        failures.append(f"discard_deadline_fraction:{discard_deadline_fraction:.6f}")
    if response_deadline_fraction > float(
        thresholds["maximum_response_deadline_fraction"]
    ):
        failures.append(f"response_deadline_fraction:{response_deadline_fraction:.6f}")

    discard_latencies = _event_latencies(league, "candidate_discard_events")
    response_latencies = _event_latencies(league, "candidate_response_events")
    discard_p95 = _percentile(discard_latencies, 0.95)
    response_p95 = _percentile(response_latencies, 0.95)
    if discard_p95 > float(thresholds["maximum_discard_p95_ms"]):
        failures.append(f"discard_p95_ms:{discard_p95:.3f}")
    if response_p95 > float(thresholds["maximum_response_p95_ms"]):
        failures.append(f"response_p95_ms:{response_p95:.3f}")

    changed_actions = int(diagnostics.get("search_overrides") or 0)
    root_coverage = root_states / changed_actions if changed_actions else 1.0
    if root_coverage < minimum_root_coverage:
        failures.append(f"root_coverage:{root_coverage:.6f}")
    return {
        "passed": not failures,
        "league_path": str(league_path),
        "paired_path": str(paired_path),
        "games": int(league.get("games") or 0),
        "wins": wins,
        "strong_pool_win_lower_bound": round(win_lower, 6),
        "paired_score_lower_bound": float(score_interval[0]),
        "paired_outcome_lower_bound": float(outcome_interval[0]),
        "decision_budget_fallback_fraction": round(fallback_fraction, 6),
        "discard_deadline_fraction": round(discard_deadline_fraction, 6),
        "response_deadline_fraction": round(response_deadline_fraction, 6),
        "discard_p95_ms": round(discard_p95, 3),
        "response_p95_ms": round(response_p95, 3),
        "changed_actions": changed_actions,
        "root_coverage": round(root_coverage, 6),
        "failures": failures,
    }


def _check_league_integrity(
    league: Mapping[str, Any],
    *,
    mode: Mapping[str, Any],
    minimum_games: int,
    failures: list[str],
) -> None:
    if not league.get("ok"):
        failures.append("league_report_not_ok")
    if int(league.get("games") or 0) < minimum_games:
        failures.append("games_below_minimum")
    if int(league.get("players") or 0) != int(mode["players"]):
        failures.append("players_mismatch")
    if bool(league.get("wildcard_enabled")) != bool(mode["wildcard_enabled"]):
        failures.append("wildcard_mismatch")
    for key in (
        "invariant_violations",
        "coverage_failures",
        "strategy_runtime_errors",
    ):
        if int(league.get(key) or 0) != 0:
            failures.append(f"{key}:{league.get(key)}")
    if not league.get("decision_trace_recorded"):
        failures.append("decision_trace_missing")


def _rotation_balance_failures(
    league: Mapping[str, Any],
    *,
    mode: Mapping[str, Any],
) -> list[str]:
    players = int(mode["players"])
    expected_rotations = {
        (candidate_seat, dealer)
        for candidate_seat in range(players)
        for dealer in range(players)
    }
    counts_by_matchup: dict[tuple[str, ...], Counter[tuple[int, int]]] = (
        defaultdict(Counter)
    )
    for row in league.get("rows") or ():
        matchup = tuple(sorted(str(name) for name in row.get("opponents") or ()))
        rotation = (int(row.get("candidate_seat") or 0), int(row.get("dealer") or 0))
        counts_by_matchup[matchup][rotation] += 1
    failures: list[str] = []
    if not counts_by_matchup:
        return ["rotation_rows_missing"]
    for matchup, counts in counts_by_matchup.items():
        if set(counts) != expected_rotations:
            failures.append(f"rotation_coverage:{'+'.join(matchup)}")
            continue
        values = list(counts.values())
        if max(values) != min(values):
            failures.append(f"rotation_count_imbalance:{'+'.join(matchup)}")
    return failures


def _evaluate_ordinary_league(
    raw_path: Any,
    *,
    mode: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    evidence_base: Path,
) -> dict[str, Any]:
    path = _resolve_optional_path(raw_path, evidence_base)
    if path is None:
        return _failed("evidence_missing")
    if not path.exists():
        return _failed(f"file_missing:{path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    minimum_games = int(thresholds["minimum_games"])
    _check_league_integrity(
        report,
        mode=mode,
        minimum_games=minimum_games,
        failures=failures,
    )
    games = max(1, int(report.get("games") or 0))
    wins = int(report.get("candidate_wins") or 0)
    lower = _wilson_lower_bound(wins, games)
    if lower < float(thresholds["minimum_win_lower_bound"]):
        failures.append(f"win_lower_bound:{lower}")
    return {
        "passed": not failures,
        "path": str(path),
        "games": games,
        "wins": wins,
        "win_lower_bound": round(lower, 6),
        "failures": failures,
    }


def _evaluate_device_soak(
    raw_path: Any,
    thresholds: Mapping[str, Any],
    *,
    mode_ids: tuple[str, ...],
    evidence_base: Path,
) -> dict[str, Any]:
    path = _resolve_optional_path(raw_path, evidence_base)
    if path is None:
        return _failed("evidence_missing")
    if not path.exists():
        return _failed(f"file_missing:{path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if report.get("schema_version") != "aizipai-device-soak-v1":
        failures.append("schema_mismatch")
    mode_results = report.get("mode_results") or {}
    for mode_id in mode_ids:
        games = int((mode_results.get(mode_id) or {}).get("games") or 0)
        if games < int(thresholds["minimum_games_per_mode"]):
            failures.append(f"mode_games:{mode_id}:{games}")
    primary_mode = str(report.get("primary_mode") or "")
    primary_games = int((mode_results.get(primary_mode) or {}).get("games") or 0)
    if primary_games < int(thresholds["minimum_primary_mode_games"]):
        failures.append(f"primary_mode_games:{primary_games}")
    if float(report.get("cycle_p95_ms") or math.inf) > float(
        thresholds["maximum_cycle_p95_ms"]
    ):
        failures.append("cycle_p95_ms")
    if int(report.get("critical_failures") or 0) > int(
        thresholds["maximum_critical_failures"]
    ):
        failures.append("critical_failures")
    if int(report.get("strategy_mismatches") or 0) > int(
        thresholds["maximum_strategy_mismatches"]
    ):
        failures.append("strategy_mismatches")
    return {"passed": not failures, "path": str(path), "failures": failures}


def _evaluate_external_calibration(
    raw_path: Any,
    thresholds: Mapping[str, Any],
    *,
    evidence_base: Path,
) -> dict[str, Any]:
    path = _resolve_optional_path(raw_path, evidence_base)
    if path is None:
        return _failed("evidence_missing")
    if not path.exists():
        return _failed(f"file_missing:{path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if report.get("schema_version") != "aizipai-external-calibration-v1":
        failures.append("schema_mismatch")
    if int(report.get("games") or 0) < int(thresholds["minimum_games"]):
        failures.append("games_below_minimum")
    if int(report.get("reviewed_decisions") or 0) < int(
        thresholds["minimum_reviewed_decisions"]
    ):
        failures.append("reviewed_decisions_below_minimum")
    if thresholds.get("require_professional_level_supported") and not report.get(
        "professional_level_supported"
    ):
        failures.append("professional_level_not_supported")
    return {"passed": not failures, "path": str(path), "failures": failures}


def _event_latencies(report: Mapping[str, Any], key: str) -> list[float]:
    return [
        float(event.get("elapsed_ms") or 0.0)
        for row in report.get("rows") or ()
        for event in row.get(key) or ()
    ]


def _percentile(values: Iterable[float], quantile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return math.inf
    index = (len(rows) - 1) * quantile
    lower = int(index)
    upper = min(len(rows) - 1, lower + 1)
    return rows[lower] + (rows[upper] - rows[lower]) * (index - lower)


def _wilson_lower_bound(successes: int, samples: int) -> float:
    if samples <= 0:
        return 0.0
    z = 1.959963984540054
    ratio = successes / samples
    denominator = 1.0 + z * z / samples
    center = ratio + z * z / (2.0 * samples)
    margin = z * math.sqrt(
        ratio * (1.0 - ratio) / samples + z * z / (4.0 * samples * samples)
    )
    return (center - margin) / denominator


def _resolve_optional_path(raw_path: Any, base: Path) -> Path | None:
    if not raw_path:
        return None
    return _resolve_path(raw_path, base)


def _resolve_path(raw_path: Any, base: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base / path


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _failed(reason: str) -> dict[str, Any]:
    return {"passed": False, "failures": [reason]}


def _require_schema(
    value: Mapping[str, Any],
    expected: str,
    label: str,
) -> None:
    if value.get("schema_version") != expected:
        raise ValueError(f"{label}_schema_mismatch")


if __name__ == "__main__":
    raise SystemExit(main())
