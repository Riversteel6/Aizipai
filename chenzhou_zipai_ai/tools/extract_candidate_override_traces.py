"""Join candidate action changes to full traces without hidden-state guesses."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.simulation_trace import canonical_public_state, public_state_identity


NEGATIVE_TRANSITIONS = frozenset(
    {"win_to_draw", "win_to_loss", "draw_to_loss"}
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-report", type=Path, required=True)
    parser.add_argument("--trace-input", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-output", type=Path, required=True)
    parser.add_argument("--score-regressed-output", type=Path)
    parser.add_argument("--development-output", type=Path)
    parser.add_argument("--controls-per-change", type=int, default=1)
    args = parser.parse_args()

    report = json.loads(args.league_report.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    traces = list(_read_jsonl(args.trace_input))
    manifest = extract_override_traces(
        report,
        traces=traces,
        baseline_report=baseline,
        source_report=str(args.league_report),
        source_trace=str(args.trace_input),
        baseline_source=str(args.baseline_report),
    )
    negative_manifest = select_override_manifest(
        manifest,
        selection="negative_outcome_transition",
        predicate=lambda row: (
            row["outcome_transition"] in NEGATIVE_TRANSITIONS
        ),
    )
    score_regressed_manifest = select_override_manifest(
        manifest,
        selection="negative_score_delta",
        predicate=lambda row: (
            row.get("score_delta") is not None
            and float(row["score_delta"]) < 0.0
        ),
    )
    _write_json(args.output, manifest)
    _write_json(args.negative_output, negative_manifest)
    if args.score_regressed_output is not None:
        _write_json(args.score_regressed_output, score_regressed_manifest)
    development_manifest = None
    if args.development_output is not None:
        development_manifest = extract_discard_development_manifest(
            report,
            traces=traces,
            baseline_report=baseline,
            controls_per_change=max(1, args.controls_per_change),
            source_report=str(args.league_report),
            source_trace=str(args.trace_input),
            baseline_source=str(args.baseline_report),
        )
        _write_json(args.development_output, development_manifest)
    print(
        json.dumps(
            {
                "games": manifest["games"],
                "discard_events": manifest["discard_events"],
                "response_events": manifest["response_events"],
                "override_states": manifest["override_states"],
                "discard_override_states": manifest[
                    "discard_override_states"
                ],
                "response_override_states": manifest[
                    "response_override_states"
                ],
                "response_empirical_challenger_states": manifest[
                    "response_empirical_challenger_states"
                ],
                "unique_state_hashes": manifest["unique_state_hashes"],
                "negative_games": manifest["negative_games"],
                "negative_override_states": negative_manifest["override_states"],
                "score_regressed_games": manifest["score_regressed_games"],
                "score_regressed_override_states": score_regressed_manifest[
                    "override_states"
                ],
                "alignment_errors": manifest["alignment_errors"],
                "development_changed_states": (
                    development_manifest["changed_states"]
                    if development_manifest is not None
                    else None
                ),
                "development_control_states": (
                    development_manifest["control_states"]
                    if development_manifest is not None
                    else None
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def extract_discard_development_manifest(
    report: Mapping[str, Any],
    *,
    traces: Iterable[Mapping[str, Any]],
    baseline_report: Mapping[str, Any],
    controls_per_change: int = 1,
    source_report: str = "",
    source_trace: str = "",
    baseline_source: str = "",
) -> dict[str, Any]:
    """Keep every changed discard and deterministic matched keep controls."""

    report_rows = list(report.get("rows") or ())
    baseline_rows = {
        _game_identity(row): row
        for row in baseline_report.get("rows") or ()
    }
    trace_rows = {
        _game_identity(row): row
        for row in traces
    }
    identities = {_game_identity(row) for row in report_rows}
    if set(trace_rows) != identities:
        raise ValueError("development_trace_game_identity_mismatch")
    if set(baseline_rows) != identities:
        raise ValueError("development_baseline_game_identity_mismatch")

    discard_rows: list[dict[str, Any]] = []
    for game in report_rows:
        identity = _game_identity(game)
        events = list(game.get("candidate_discard_events") or ())
        steps = [
            step
            for step in trace_rows[identity].get("decision_trace") or ()
            if (
                step.get("phase") == "discard"
                and int(step.get("seat", -1)) == int(game["candidate_seat"])
            )
        ]
        if len(events) != len(steps):
            raise ValueError(
                "development_discard_event_count_mismatch:"
                f"{_identity_text(identity)}:{len(events)}!={len(steps)}"
            )
        baseline_game = baseline_rows[identity]
        baseline_score = _optional_float(
            baseline_game.get("candidate_outcome_score")
        )
        candidate_score = _optional_float(game.get("candidate_outcome_score"))
        score_delta = (
            candidate_score - baseline_score
            if baseline_score is not None and candidate_score is not None
            else None
        )
        transition = f"{_outcome(baseline_game)}_to_{_outcome(game)}"
        for event_index, (event, step) in enumerate(zip(events, steps)):
            _verify_event_alignment(
                identity,
                event_index=event_index,
                event=event,
                step=step,
            )
            production_label = str(event.get("production_label") or "")
            selected_label = str(
                event.get("search_selected_label")
                or production_label
            )
            discard_rows.append(
                {
                    "state_before_hash": str(step["state_before_hash"]),
                    "public_state_id": public_state_identity(
                        step.get("public_state") or {}
                    ),
                    "seed": int(game["seed"]),
                    "candidate_seat": int(game["candidate_seat"]),
                    "dealer": int(game["dealer"]),
                    "opponents": list(game.get("opponents") or ()),
                    "opponent_stratum": ",".join(
                        str(name) for name in game.get("opponents") or ()
                    ),
                    "event_index": event_index,
                    "trace_sequence": int(step["sequence"]),
                    "trace_turn": int(step["turn"]),
                    "production_label": production_label,
                    "selected_label": selected_label,
                    "changed": selected_label != production_label,
                    "confidence_override": bool(
                        event.get("confidence_override")
                    ),
                    "validation_complete": bool(
                        event.get("validation_complete")
                    ),
                    "validation_error": event.get("validation_error"),
                    "legal_candidate_count": int(
                        event.get("legal_candidate_count") or 0
                    ),
                    "elapsed_ms": round(
                        float(event.get("elapsed_ms") or 0.0),
                        3,
                    ),
                    "outcome_transition": transition,
                    "score_delta": score_delta,
                }
            )

    changed = sorted(
        (row for row in discard_rows if row["changed"]),
        key=_development_row_key,
    )
    available_controls = {
        str(row["state_before_hash"]): row
        for row in discard_rows
        if not row["changed"]
    }
    selected_controls: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for changed_row in changed:
        for _ in range(max(1, controls_per_change)):
            candidates = [
                row
                for row in available_controls.values()
                if row["opponent_stratum"]
                == changed_row["opponent_stratum"]
            ]
            if not candidates:
                raise ValueError(
                    "development_control_stratum_exhausted:"
                    f"{changed_row['opponent_stratum']}"
                )
            control = min(
                candidates,
                key=lambda row: _control_distance(changed_row, row),
            )
            available_controls.pop(str(control["state_before_hash"]))
            selected_controls.append(control)
            paired_rows.extend(
                (
                    {**changed_row, "cohort": "changed"},
                    {
                        **control,
                        "cohort": "matched_unchanged",
                        "matched_changed_state_hash": changed_row[
                            "state_before_hash"
                        ],
                    },
                )
            )

    fingerprint_rows = [
        f"{row['cohort']}:{row['state_before_hash']}"
        for row in paired_rows
    ]
    selection_sha256 = hashlib.sha256(
        "\n".join(fingerprint_rows).encode("utf-8")
    ).hexdigest()
    return {
        "ok": bool(changed)
        and len(selected_controls) == len(changed) * max(1, controls_per_change),
        "schema_version": "discard-development-manifest-v1",
        "dataset_role": "development_exposed",
        "source_report": source_report,
        "source_trace": source_trace,
        "baseline_source": baseline_source,
        "games": len(report_rows),
        "discard_events": len(discard_rows),
        "changed_states": len(changed),
        "confidence_changed_states": sum(
            row["confidence_override"] for row in changed
        ),
        "nonconfidence_changed_states": sum(
            not row["confidence_override"] for row in changed
        ),
        "controls_per_change": max(1, controls_per_change),
        "control_states": len(selected_controls),
        "selected_states": len(paired_rows),
        "selection_sha256": selection_sha256,
        "rows": paired_rows,
    }


def _development_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["opponent_stratum"]),
        int(row["seed"]),
        int(row["trace_sequence"]),
        str(row["state_before_hash"]),
    )


def _control_distance(
    changed: Mapping[str, Any],
    control: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        int(bool(changed["validation_complete"]) != bool(control["validation_complete"])),
        abs(
            int(changed["legal_candidate_count"])
            - int(control["legal_candidate_count"])
        ),
        abs(int(changed["trace_turn"]) - int(control["trace_turn"])),
        abs(float(changed["elapsed_ms"]) - float(control["elapsed_ms"])),
        str(control["state_before_hash"]),
    )


def extract_override_traces(
    report: Mapping[str, Any],
    *,
    traces: Iterable[Mapping[str, Any]],
    baseline_report: Mapping[str, Any],
    source_report: str = "",
    source_trace: str = "",
    baseline_source: str = "",
) -> dict[str, Any]:
    report_rows = list(report.get("rows") or ())
    baseline_rows = {
        _game_identity(row): row
        for row in baseline_report.get("rows") or ()
    }
    trace_rows = {_game_identity(row): row for row in traces}
    report_identities = {_game_identity(row) for row in report_rows}
    if set(trace_rows) != report_identities:
        missing = sorted(report_identities - set(trace_rows))
        extra = sorted(set(trace_rows) - report_identities)
        raise ValueError(
            f"trace_game_identity_mismatch:missing={len(missing)}:extra={len(extra)}"
        )
    if set(baseline_rows) != report_identities:
        missing = sorted(report_identities - set(baseline_rows))
        extra = sorted(set(baseline_rows) - report_identities)
        raise ValueError(
            f"baseline_game_identity_mismatch:missing={len(missing)}:extra={len(extra)}"
        )

    overrides: list[dict[str, Any]] = []
    response_challengers: list[dict[str, Any]] = []
    discard_events = 0
    response_events = 0
    transitions: dict[str, int] = {}
    negative_games = 0
    score_regressed_games = 0
    for game in report_rows:
        identity = _game_identity(game)
        trace_game = trace_rows[identity]
        baseline_score = _optional_float(
            baseline_rows[identity].get("candidate_outcome_score")
        )
        candidate_score = _optional_float(
            game.get("candidate_outcome_score")
        )
        score_delta = (
            candidate_score - baseline_score
            if baseline_score is not None and candidate_score is not None
            else None
        )
        score_direction = (
            "improved"
            if score_delta is not None and score_delta > 0.0
            else "regressed"
            if score_delta is not None and score_delta < 0.0
            else "equal"
            if score_delta is not None
            else "unknown"
        )
        score_regressed_games += int(score_direction == "regressed")
        events = list(game.get("candidate_discard_events") or ())
        steps = [
            step
            for step in trace_game.get("decision_trace") or ()
            if (
                step.get("phase") == "discard"
                and int(step.get("seat", -1)) == int(game["candidate_seat"])
            )
        ]
        if len(events) != len(steps):
            raise ValueError(
                "discard_event_count_mismatch:"
                f"{_identity_text(identity)}:{len(events)}!={len(steps)}"
            )
        transition = (
            f"{_outcome(baseline_rows[identity])}_to_{_outcome(game)}"
        )
        transitions[transition] = transitions.get(transition, 0) + 1
        negative_games += int(transition in NEGATIVE_TRANSITIONS)
        for event_index, (event, step) in enumerate(zip(events, steps)):
            discard_events += 1
            _verify_event_alignment(
                identity,
                event_index=event_index,
                event=event,
                step=step,
            )
            if not event.get("confidence_override"):
                continue
            selected_label = str(event["search_selected_label"])
            overrides.append(
                {
                    "kind": "discard",
                    "phase": "discard",
                    "seed": int(game["seed"]),
                    "candidate_seat": int(game["candidate_seat"]),
                    "dealer": int(game["dealer"]),
                    "opponents": list(game.get("opponents") or ()),
                    "event_index": event_index,
                    "trace_sequence": int(step["sequence"]),
                    "trace_turn": int(step["turn"]),
                    "state_before_hash": str(step["state_before_hash"]),
                    "public_state_id": public_state_identity(
                        step.get("public_state") or {}
                    ),
                    "production_label": str(event["production_label"]),
                    "baseline_selected_label": str(
                        event["baseline_selected_label"]
                    ),
                    "validation_selected_label": str(
                        event["validation_selected_label"]
                    ),
                    "selected_label": selected_label,
                    "outcome_transition": transition,
                    "baseline_outcome_score": baseline_score,
                    "candidate_outcome_score": candidate_score,
                    "score_delta": score_delta,
                    "score_direction": score_direction,
                }
            )

        joint_events = list(game.get("candidate_response_events") or ())
        joint_steps = [
            step
            for step in trace_game.get("decision_trace") or ()
            if (
                step.get("phase") == "response_root"
                and int(step.get("seat", -1))
                == int(game["candidate_seat"])
            )
        ]
        if len(joint_events) != len(joint_steps):
            raise ValueError(
                "response_event_count_mismatch:"
                f"{_identity_text(identity)}:"
                f"{len(joint_events)}!={len(joint_steps)}"
            )
        for event_index, (event, step) in enumerate(
            zip(joint_events, joint_steps)
        ):
            response_events += 1
            _verify_response_event_alignment(
                identity,
                event_index=event_index,
                event=event,
                step=step,
            )
            empirical_key = str(event.get("empirical_best_key") or "")
            production_key = str(event.get("production_key") or "")
            if empirical_key and empirical_key != production_key:
                advantage = next(
                    (
                        row
                        for row in event.get("paired_advantages") or ()
                        if str(row.get("candidate_key") or "")
                        == empirical_key
                    ),
                    {},
                )
                response_challengers.append(
                    {
                        "kind": "response_challenger",
                        "phase": "response_root",
                        "seed": int(game["seed"]),
                        "candidate_seat": int(game["candidate_seat"]),
                        "dealer": int(game["dealer"]),
                        "opponents": list(game.get("opponents") or ()),
                        "event_index": event_index,
                        "trace_sequence": int(step["sequence"]),
                        "trace_turn": int(step["turn"]),
                        "state_before_hash": str(
                            step["state_before_hash"]
                        ),
                        "public_state_id": public_state_identity(
                            step.get("public_state") or {}
                        ),
                        "response_type": str(
                            event.get("response_type") or ""
                        ),
                        "pending_card": event.get("pending_card"),
                        "production_key": production_key,
                        "selected_key": str(
                            event.get("search_selected_key") or ""
                        ),
                        "empirical_best_key": empirical_key,
                        "applied_override": bool(event.get("disagreement")),
                        "confidence_override": bool(
                            event.get("confidence_override")
                        ),
                        "candidate_count": int(
                            event.get("candidate_count") or 0
                        ),
                        "online_mean_delta": _optional_float(
                            advantage.get("mean_delta")
                        ),
                        "online_standard_error": _optional_float(
                            advantage.get("standard_error")
                        ),
                        "online_lower_confidence_bound": _optional_float(
                            advantage.get("lower_confidence_bound")
                        ),
                        "online_samples": int(
                            advantage.get("samples") or 0
                        ),
                        "outcome_transition": transition,
                        "baseline_outcome_score": baseline_score,
                        "candidate_outcome_score": candidate_score,
                        "score_delta": score_delta,
                        "score_direction": score_direction,
                    }
                )
            if not event.get("disagreement"):
                continue
            overrides.append(
                {
                    "kind": "response",
                    "phase": "response_root",
                    "seed": int(game["seed"]),
                    "candidate_seat": int(game["candidate_seat"]),
                    "dealer": int(game["dealer"]),
                    "opponents": list(game.get("opponents") or ()),
                    "event_index": event_index,
                    "trace_sequence": int(step["sequence"]),
                    "trace_turn": int(step["turn"]),
                    "state_before_hash": str(step["state_before_hash"]),
                    "public_state_id": public_state_identity(
                        step.get("public_state") or {}
                    ),
                    "response_type": str(event.get("response_type") or ""),
                    "pending_card": event.get("pending_card"),
                    "production_key": str(
                        event.get("production_key") or ""
                    ),
                    "selected_key": str(
                        event.get("search_selected_key") or ""
                    ),
                    "confidence_override": bool(
                        event.get("confidence_override")
                    ),
                    "outcome_transition": transition,
                    "baseline_outcome_score": baseline_score,
                    "candidate_outcome_score": candidate_score,
                    "score_delta": score_delta,
                    "score_direction": score_direction,
                }
            )

    state_hashes = {row["state_before_hash"] for row in overrides}
    discard_overrides = sum(row["kind"] == "discard" for row in overrides)
    response_overrides = sum(row["kind"] == "response" for row in overrides)
    return {
        "schema_version": "candidate-override-trace-manifest-v3",
        "source_report": source_report,
        "source_trace": source_trace,
        "baseline_source": baseline_source,
        "games": len(report_rows),
        "discard_events": discard_events,
        "response_events": response_events,
        "override_states": len(overrides),
        "discard_override_states": discard_overrides,
        "response_override_states": response_overrides,
        "response_empirical_challenger_states": len(
            response_challengers
        ),
        "unique_state_hashes": len(state_hashes),
        "negative_games": negative_games,
        "score_regressed_games": score_regressed_games,
        "outcome_transitions": dict(sorted(transitions.items())),
        "alignment_errors": 0,
        "response_challenger_rows": response_challengers,
        "rows": overrides,
    }


def select_override_manifest(
    manifest: Mapping[str, Any],
    *,
    selection: str,
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    rows = [
        row
        for row in manifest.get("rows") or ()
        if predicate(row)
    ]
    challenger_rows = [
        row
        for row in manifest.get("response_challenger_rows") or ()
        if predicate(row)
    ]
    return {
        **{
            key: value
            for key, value in manifest.items()
            if key not in {"rows", "response_challenger_rows"}
        },
        "selection": selection,
        "override_states": len(rows),
        "discard_override_states": sum(
            row["kind"] == "discard" for row in rows
        ),
        "response_override_states": sum(
            row["kind"] == "response" for row in rows
        ),
        "response_empirical_challenger_states": len(challenger_rows),
        "unique_state_hashes": len(
            {row["state_before_hash"] for row in rows}
        ),
        "response_challenger_rows": challenger_rows,
        "rows": rows,
    }


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _verify_event_alignment(
    identity: tuple[int, int, int, tuple[str, ...]],
    *,
    event_index: int,
    event: Mapping[str, Any],
    step: Mapping[str, Any],
) -> None:
    event_view = event.get("public_view") or {}
    trace_view = step.get("public_state") or {}
    if canonical_public_state(event_view) != canonical_public_state(trace_view):
        raise ValueError(
            "discard_public_state_mismatch:"
            f"{_identity_text(identity)}:{event_index}"
        )
    event_id = str(event.get("public_view_id") or "")
    trace_id = str(step.get("public_state_id") or "")
    calculated_id = public_state_identity(event_view)
    if event_id and event_id != calculated_id:
        raise ValueError(
            "discard_event_public_state_id_mismatch:"
            f"{_identity_text(identity)}:{event_index}"
        )
    if trace_id and trace_id != calculated_id:
        raise ValueError(
            "discard_trace_public_state_id_mismatch:"
            f"{_identity_text(identity)}:{event_index}"
        )
    selected = str(
        event.get("search_selected_label")
        or event.get("production_label")
        or ""
    )
    if str(step.get("selected_key") or "") != f"DISCARD:{selected}":
        raise ValueError(
            "discard_selected_action_mismatch:"
            f"{_identity_text(identity)}:{event_index}"
        )


def _verify_response_event_alignment(
    identity: tuple[int, int, int, tuple[str, ...]],
    *,
    event_index: int,
    event: Mapping[str, Any],
    step: Mapping[str, Any],
) -> None:
    event_view = event.get("public_view") or {}
    trace_view = step.get("public_state") or {}
    if canonical_public_state(event_view) != canonical_public_state(trace_view):
        raise ValueError(
            "response_public_state_mismatch:"
            f"{_identity_text(identity)}:{event_index}"
        )
    trace_id = str(step.get("public_state_id") or "")
    calculated_id = public_state_identity(event_view)
    if trace_id and trace_id != calculated_id:
        raise ValueError(
            "response_trace_public_state_id_mismatch:"
            f"{_identity_text(identity)}:{event_index}"
        )
    selected = str(
        event.get("search_selected_key")
        or event.get("production_key")
        or ""
    )
    if str(step.get("selected_key") or "") != selected:
        raise ValueError(
            "response_selected_action_mismatch:"
            f"{_identity_text(identity)}:{event_index}"
        )


def _game_identity(
    row: Mapping[str, Any],
) -> tuple[int, int, int, tuple[str, ...]]:
    return (
        int(row["seed"]),
        int(row["candidate_seat"]),
        int(row["dealer"]),
        tuple(sorted(str(value) for value in row.get("opponents") or ())),
    )


def _identity_text(identity: tuple[int, int, int, tuple[str, ...]]) -> str:
    seed, seat, dealer, opponents = identity
    return f"{seed}:{seat}:{dealer}:{','.join(opponents)}"


def _outcome(row: Mapping[str, Any]) -> str:
    if row.get("draw"):
        return "draw"
    return "win" if row.get("candidate_won") else "loss"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("rt", encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
