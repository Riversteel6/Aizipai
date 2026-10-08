"""Locate the first action divergence in paired full-game traces."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = compare_paired_decision_traces(
        json.loads(args.baseline_report.read_text(encoding="utf-8")),
        _read_jsonl(args.baseline_trace),
        json.loads(args.candidate_report.read_text(encoding="utf-8")),
        _read_jsonl(args.candidate_trace),
        baseline_report_path=str(args.baseline_report),
        baseline_trace_path=str(args.baseline_trace),
        candidate_report_path=str(args.candidate_report),
        candidate_trace_path=str(args.candidate_trace),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "ok",
                    "games",
                    "identical_action_paths",
                    "selected_action_divergences",
                    "state_path_divergences",
                    "trace_length_divergences",
                    "candidate_seat_first_divergences",
                    "outcome_improved",
                    "outcome_regressed",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def compare_paired_decision_traces(
    baseline_report: Mapping[str, Any],
    baseline_traces: Iterable[Mapping[str, Any]],
    candidate_report: Mapping[str, Any],
    candidate_traces: Iterable[Mapping[str, Any]],
    *,
    baseline_report_path: str = "",
    baseline_trace_path: str = "",
    candidate_report_path: str = "",
    candidate_trace_path: str = "",
) -> dict[str, Any]:
    baseline_rows = _rows_by_identity(baseline_report.get("rows") or ())
    candidate_rows = _rows_by_identity(candidate_report.get("rows") or ())
    baseline_games = _traces_by_identity(baseline_traces)
    candidate_games = _traces_by_identity(candidate_traces)
    identities = (
        set(baseline_rows)
        & set(candidate_rows)
        & set(baseline_games)
        & set(candidate_games)
    )
    all_identity_sets = (
        set(baseline_rows),
        set(candidate_rows),
        set(baseline_games),
        set(candidate_games),
    )
    aligned = all(items == identities for items in all_identity_sets)

    rows: list[dict[str, Any]] = []
    by_matchup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    divergence_types: Counter[str] = Counter()
    outcome_transitions: Counter[str] = Counter()
    for identity in sorted(identities):
        baseline_row = baseline_rows[identity]
        candidate_row = candidate_rows[identity]
        baseline_game = baseline_games[identity]
        candidate_game = candidate_games[identity]
        divergence = _first_divergence(
            baseline_game.get("decision_trace") or (),
            candidate_game.get("decision_trace") or (),
            candidate_seat=int(candidate_game["candidate_seat"]),
        )
        baseline_outcome = _outcome(baseline_row)
        candidate_outcome = _outcome(candidate_row)
        transition = f"{baseline_outcome}_to_{candidate_outcome}"
        outcome_transitions[transition] += 1
        matchup = " + ".join(sorted(str(name) for name in identity[0]))
        row = {
            "matchup": matchup,
            "candidate_seat": int(identity[1]),
            "dealer": int(identity[2]),
            "seed": int(identity[3]),
            "baseline_outcome": baseline_outcome,
            "candidate_outcome": candidate_outcome,
            "outcome_transition": transition,
            "baseline_outcome_score": float(
                baseline_row["candidate_outcome_score"]
            ),
            "candidate_outcome_score": float(
                candidate_row["candidate_outcome_score"]
            ),
            "score_delta": round(
                float(candidate_row["candidate_outcome_score"])
                - float(baseline_row["candidate_outcome_score"]),
                6,
            ),
            **divergence,
        }
        rows.append(row)
        by_matchup[matchup].append(row)
        divergence_types[row["divergence_type"]] += 1

    return {
        "ok": bool(identities) and aligned,
        "schema_version": "paired-decision-trace-comparison-v1",
        "baseline_report": baseline_report_path,
        "baseline_trace": baseline_trace_path,
        "candidate_report": candidate_report_path,
        "candidate_trace": candidate_trace_path,
        "games": len(rows),
        "identity_alignment_complete": aligned,
        "missing_identity_counts": {
            "baseline_report": len(identities ^ all_identity_sets[0]),
            "candidate_report": len(identities ^ all_identity_sets[1]),
            "baseline_trace": len(identities ^ all_identity_sets[2]),
            "candidate_trace": len(identities ^ all_identity_sets[3]),
        },
        "identical_action_paths": divergence_types["none"],
        "selected_action_divergences": divergence_types["selected_action"],
        "state_path_divergences": divergence_types["state_path"],
        "trace_length_divergences": divergence_types["trace_length"],
        "candidate_seat_first_divergences": sum(
            int(row["candidate_seat_divergence"])
            for row in rows
        ),
        "outcome_improved": sum(
            int(
                _outcome_utility(row["candidate_outcome"])
                > _outcome_utility(row["baseline_outcome"])
            )
            for row in rows
        ),
        "outcome_regressed": sum(
            int(
                _outcome_utility(row["candidate_outcome"])
                < _outcome_utility(row["baseline_outcome"])
            )
            for row in rows
        ),
        "outcome_transitions": dict(sorted(outcome_transitions.items())),
        "by_matchup": {
            matchup: _summarize_rows(group)
            for matchup, group in sorted(by_matchup.items())
        },
        "rows": rows,
    }


def _first_divergence(
    baseline_trace: Iterable[Mapping[str, Any]],
    candidate_trace: Iterable[Mapping[str, Any]],
    *,
    candidate_seat: int,
) -> dict[str, Any]:
    baseline_steps = list(baseline_trace)
    candidate_steps = list(candidate_trace)
    for index, (baseline, candidate) in enumerate(
        zip(baseline_steps, candidate_steps, strict=False)
    ):
        baseline_signature = _step_signature(baseline)
        candidate_signature = _step_signature(candidate)
        if baseline_signature != candidate_signature:
            return _divergence_payload(
                "state_path",
                index,
                baseline,
                candidate,
                candidate_seat=candidate_seat,
            )
        if str(baseline.get("selected_key")) != str(
            candidate.get("selected_key")
        ):
            return _divergence_payload(
                "selected_action",
                index,
                baseline,
                candidate,
                candidate_seat=candidate_seat,
            )
    if len(baseline_steps) != len(candidate_steps):
        index = min(len(baseline_steps), len(candidate_steps))
        baseline = baseline_steps[index] if index < len(baseline_steps) else {}
        candidate = (
            candidate_steps[index] if index < len(candidate_steps) else {}
        )
        return _divergence_payload(
            "trace_length",
            index,
            baseline,
            candidate,
            candidate_seat=candidate_seat,
        )
    return {
        "divergence_type": "none",
        "divergence_index": None,
        "candidate_seat_divergence": False,
    }


def _divergence_payload(
    divergence_type: str,
    index: int,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    candidate_seat: int,
) -> dict[str, Any]:
    seat = candidate.get("seat", baseline.get("seat"))
    return {
        "divergence_type": divergence_type,
        "divergence_index": index,
        "candidate_seat_divergence": (
            seat is not None and int(seat) == candidate_seat
        ),
        "seat": int(seat) if seat is not None else None,
        "phase": candidate.get("phase", baseline.get("phase")),
        "turn": candidate.get("turn", baseline.get("turn")),
        "state_before_hash": candidate.get(
            "state_before_hash",
            baseline.get("state_before_hash"),
        ),
        "public_state_id": candidate.get(
            "public_state_id",
            baseline.get("public_state_id"),
        ),
        "baseline_selected_key": baseline.get("selected_key"),
        "candidate_selected_key": candidate.get("selected_key"),
        "baseline_reason": baseline.get("reason"),
        "candidate_reason": candidate.get("reason"),
        "baseline_policy_evidence": baseline.get("policy_evidence"),
        "candidate_policy_evidence": candidate.get("policy_evidence"),
        "public_state": candidate.get(
            "public_state",
            baseline.get("public_state"),
        ),
        "legal_actions": candidate.get(
            "legal_actions",
            baseline.get("legal_actions"),
        ),
    }


def _summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(rows),
        "identical_action_paths": sum(
            int(row["divergence_type"] == "none") for row in rows
        ),
        "selected_action_divergences": sum(
            int(row["divergence_type"] == "selected_action")
            for row in rows
        ),
        "candidate_seat_first_divergences": sum(
            int(row["candidate_seat_divergence"]) for row in rows
        ),
        "outcome_improved": sum(
            int(
                _outcome_utility(row["candidate_outcome"])
                > _outcome_utility(row["baseline_outcome"])
            )
            for row in rows
        ),
        "outcome_regressed": sum(
            int(
                _outcome_utility(row["candidate_outcome"])
                < _outcome_utility(row["baseline_outcome"])
            )
            for row in rows
        ),
        "mean_score_delta": round(
            sum(float(row["score_delta"]) for row in rows) / len(rows),
            6,
        ),
    }


def _step_signature(step: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        step.get("state_before_hash"),
        step.get("seat"),
        step.get("phase"),
    )


def _rows_by_identity(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {_row_identity(row): row for row in rows}


def _traces_by_identity(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {_row_identity(row): row for row in rows}


def _row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(sorted(str(name) for name in row["opponents"])),
        int(row["candidate_seat"]),
        int(row["dealer"]),
        int(row["seed"]),
    )


def _outcome(row: Mapping[str, Any]) -> str:
    if bool(row.get("draw")):
        return "draw"
    return "win" if bool(row.get("candidate_won")) else "loss"


def _outcome_utility(outcome: str) -> int:
    return {"loss": -1, "draw": 0, "win": 1}[outcome]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
