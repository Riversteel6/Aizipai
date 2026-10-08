"""Compare two opponent-league reports job by job."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare_reports(
        baseline,
        candidate,
        baseline_path=str(args.baseline),
        candidate_path=str(args.candidate),
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
                "games": report["games"],
                "baseline": report["baseline_candidate"],
                "candidate": report["candidate"],
                **report["paired_summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_path: str | None = None,
    candidate_path: str | None = None,
) -> dict[str, Any]:
    shared_fields = ("players", "wildcard_enabled", "seed", "deals_per_matchup", "games")
    mismatches = [
        field
        for field in shared_fields
        if baseline.get(field) != candidate.get(field)
    ]
    baseline_rows = {_job_key(row): row for row in baseline.get("rows") or []}
    candidate_rows = {_job_key(row): row for row in candidate.get("rows") or []}
    baseline_keys = set(baseline_rows)
    candidate_keys = set(candidate_rows)
    missing_candidate = sorted(baseline_keys - candidate_keys)
    missing_baseline = sorted(candidate_keys - baseline_keys)
    pairs = [
        _paired_row(baseline_rows[key], candidate_rows[key])
        for key in sorted(baseline_keys & candidate_keys)
    ]
    by_matchup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_matchup[row["matchup"]].append(row)
    ok = (
        not mismatches
        and not missing_candidate
        and not missing_baseline
        and bool(baseline.get("ok"))
        and bool(candidate.get("ok"))
    )
    return {
        "ok": ok,
        "baseline_report": baseline_path,
        "candidate_report": candidate_path,
        "baseline_candidate": baseline.get("candidate"),
        "candidate": candidate.get("candidate"),
        "players": candidate.get("players"),
        "wildcard_enabled": candidate.get("wildcard_enabled"),
        "seed": candidate.get("seed"),
        "deals_per_matchup": candidate.get("deals_per_matchup"),
        "games": len(pairs),
        "field_mismatches": mismatches,
        "missing_candidate_jobs": [list(key) for key in missing_candidate],
        "missing_baseline_jobs": [list(key) for key in missing_baseline],
        "baseline_summary": _league_summary(baseline),
        "candidate_summary": _league_summary(candidate),
        "paired_summary": _summarize_pairs(pairs),
        "by_matchup": {
            matchup: _summarize_pairs(rows)
            for matchup, rows in sorted(by_matchup.items())
        },
        "pairs": pairs,
    }


def _job_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    matchup = " + ".join(sorted(str(item) for item in row.get("opponents") or []))
    return (
        matchup,
        int(row["candidate_seat"]),
        int(row["dealer"]),
        int(row["seed"]),
    )


def _paired_row(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_score = float(baseline["candidate_outcome_score"])
    candidate_score = float(candidate["candidate_outcome_score"])
    delta = candidate_score - baseline_score
    baseline_outcome_utility = _outcome_utility(baseline)
    candidate_outcome_utility = _outcome_utility(candidate)
    outcome_delta = candidate_outcome_utility - baseline_outcome_utility
    return {
        "matchup": " + ".join(baseline["opponents"]),
        "candidate_seat": int(baseline["candidate_seat"]),
        "dealer": int(baseline["dealer"]),
        "seed": int(baseline["seed"]),
        "baseline_won": bool(baseline["candidate_won"]),
        "candidate_won": bool(candidate["candidate_won"]),
        "baseline_draw": bool(baseline["draw"]),
        "candidate_draw": bool(candidate["draw"]),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "score_delta": round(delta, 6),
        "direction": "improved" if delta > 0 else "regressed" if delta < 0 else "equal",
        "baseline_outcome_utility": baseline_outcome_utility,
        "candidate_outcome_utility": candidate_outcome_utility,
        "outcome_delta": outcome_delta,
        "outcome_direction": (
            "improved"
            if outcome_delta > 0
            else "regressed"
            if outcome_delta < 0
            else "equal"
        ),
    }


def _league_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "games",
            "candidate_wins",
            "losses",
            "draws",
            "candidate_win_share_all",
            "candidate_win_share_decisive",
            "mean_candidate_outcome_score",
            "invariant_violations",
            "coverage_failures",
        )
    }


def _summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["score_delta"]) for row in rows]
    average, score_interval = _mean_interval(deltas)
    outcome_deltas = [float(row["outcome_delta"]) for row in rows]
    outcome_average, outcome_interval = _mean_interval(outcome_deltas)
    outcome_improved = sum(
        row["outcome_direction"] == "improved" for row in rows
    )
    outcome_regressed = sum(
        row["outcome_direction"] == "regressed" for row in rows
    )
    outcome_sign_test_p = _two_sided_sign_test(
        outcome_improved,
        outcome_regressed,
    )
    return {
        "games": len(rows),
        "improved": sum(row["direction"] == "improved" for row in rows),
        "regressed": sum(row["direction"] == "regressed" for row in rows),
        "equal": sum(row["direction"] == "equal" for row in rows),
        "candidate_win_delta": sum(
            int(row["candidate_won"]) - int(row["baseline_won"])
            for row in rows
        ),
        "mean_score_delta": round(average, 6),
        "mean_score_delta_95": score_interval,
        "total_score_delta": round(sum(deltas), 6),
        "outcome_improved": outcome_improved,
        "outcome_regressed": outcome_regressed,
        "outcome_equal": sum(
            row["outcome_direction"] == "equal" for row in rows
        ),
        "mean_outcome_utility_delta": round(outcome_average, 6),
        "mean_outcome_utility_delta_95": outcome_interval,
        "outcome_sign_test_p": round(outcome_sign_test_p, 8),
        "outcome_significant_positive": bool(
            outcome_interval[0] > 0.0
            and outcome_sign_test_p < 0.05
        ),
        "outcome_transitions": dict(
            sorted(
                _transition_counts(rows).items()
            )
        ),
    }


def _outcome_utility(row: dict[str, Any]) -> int:
    if row.get("draw"):
        return 0
    return 1 if row.get("candidate_won") else -1


def _mean_interval(values: list[float]) -> tuple[float, list[float]]:
    average = mean(values) if values else 0.0
    if len(values) >= 2:
        variance = sum(
            (value - average) ** 2 for value in values
        ) / (len(values) - 1)
        margin = 1.959963984540054 * math.sqrt(
            variance / len(values)
        )
    else:
        margin = 0.0
    return average, [
        round(average - margin, 6),
        round(average + margin, 6),
    ]


def _two_sided_sign_test(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant <= 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(improved, regressed) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _transition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        transition = (
            f"{_utility_name(int(row['baseline_outcome_utility']))}"
            f"_to_{_utility_name(int(row['candidate_outcome_utility']))}"
        )
        counts[transition] = counts.get(transition, 0) + 1
    return counts


def _utility_name(value: int) -> str:
    return "win" if value > 0 else "loss" if value < 0 else "draw"


if __name__ == "__main__":
    raise SystemExit(main())
