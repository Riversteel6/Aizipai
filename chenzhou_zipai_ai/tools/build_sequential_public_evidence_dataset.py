"""Build and audit an observer-safe chronological opponent evidence dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.opponent_belief import (
    PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES,
    public_opponent_features,
)
from engine.rules import rules_for_room
from tools.evaluate_anchored_structural_discard_proxies import sha256_file


PUBLIC_STATE_KEYS = ("all_melds", "discards", "hand_sizes", "stock_count")
FORBIDDEN_OUTPUT_KEYS = {
    "hand",
    "remaining_counts",
    "legal_actions",
    "policy_evidence",
    "outcome",
    "result",
    "winner",
    "candidate_won",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.preregistration.read_bytes()
    preregistration = json.loads(raw.decode("utf-8"))
    report = run_audit(
        preregistration,
        preregistration_path=args.preregistration,
        preregistration_sha256=hashlib.sha256(raw).hexdigest(),
        dataset_output=args.dataset_output,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "integrity_gate_pass": report["integrity_gate_pass"],
                "games": report["games"],
                "events": report["events"],
                "event_kinds": report["event_kinds"],
                "reconstructed_context_mismatches": report[
                    "reconstructed_context_mismatches"
                ],
                "forbidden_output_keys": report["forbidden_output_keys"],
                "failures": report["integrity_gate_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] and report["integrity_gate_pass"] else 1


def run_audit(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path | str,
    preregistration_sha256: str,
    dataset_output: Path,
) -> dict[str, Any]:
    trace = Path(str(preregistration["inputs"]["trace"]))
    if sha256_file(trace) != str(preregistration["inputs"]["trace_sha256"]):
        raise ValueError("sequential_public_trace_hash_mismatch")
    profiles = tuple(str(item) for item in preregistration["profiles"])
    folds = int(preregistration["folds"])
    fold_by_source, game_ids, profile_by_source = assign_game_folds(
        trace,
        profiles=profiles,
        folds=folds,
    )

    events = []
    diagnostics = Counter()
    games_by_profile = Counter()
    profiles_by_fold: dict[int, set[str]] = defaultdict(set)
    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            game = json.loads(line)
            profile = profile_by_source[source_index]
            fold = fold_by_source[source_index]
            game_events, game_diagnostics = extract_game_events(
                game,
                game_id=game_ids[source_index],
                profile=profile,
                fold=fold,
            )
            events.extend(game_events)
            diagnostics.update(game_diagnostics)
            games_by_profile[profile] += 1
            profiles_by_fold[fold].add(profile)

    sequence_errors = sequence_error_count(events)
    forbidden_keys = sum(count_forbidden_keys(event) for event in events)
    event_kinds = Counter(str(event["action_kind"]) for event in events)
    events_by_profile = Counter(str(event["profile"]) for event in events)
    group_leakage = grouped_fold_leakage(events)
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_gzip(dataset_output, events)
    dataset_sha256 = sha256_file(dataset_output)

    gate = dict(preregistration["integrity_gate"])
    failures = integrity_gate_failures(
        games=sum(games_by_profile.values()),
        games_by_profile=games_by_profile,
        profiles=profiles,
        folds=folds,
        events=len(events),
        event_kinds=event_kinds,
        response_observations=sum(
            count for kind, count in event_kinds.items() if kind != "DISCARD"
        ),
        group_leakage=group_leakage,
        sequence_errors=sequence_errors,
        private_projection_errors=int(diagnostics["private_projection_errors"]),
        reconstructed_context_mismatches=int(
            diagnostics["reconstructed_context_mismatches"]
        ),
        forbidden_output_keys=forbidden_keys,
        profiles_by_fold=profiles_by_fold,
        gate=gate,
    )
    return {
        "ok": bool(events),
        "schema_version": "sequential-public-evidence-dataset-audit-v1",
        "experiment_id": str(preregistration["experiment_id"]),
        "dataset_role": str(preregistration["dataset_role"]),
        "promotion_effect": str(preregistration["promotion_effect"]),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": preregistration_sha256,
        "trace": str(trace),
        "trace_sha256": sha256_file(trace),
        "dataset": str(dataset_output),
        "dataset_sha256": dataset_sha256,
        "games": sum(games_by_profile.values()),
        "games_by_profile": dict(games_by_profile),
        "events": len(events),
        "events_by_profile": dict(events_by_profile),
        "event_kinds": dict(event_kinds),
        "response_observations": sum(
            count for kind, count in event_kinds.items() if kind != "DISCARD"
        ),
        "explicit_private_passes_mapped_to_no_claim": int(
            diagnostics["explicit_pass"]
        ),
        "implicit_no_claim_observations": int(diagnostics["implicit_no_claim"]),
        "visible_claim_observations": int(diagnostics["visible_claim"]),
        "terminal_hu_events_excluded": int(diagnostics["terminal_hu_excluded"]),
        "group_leakage": group_leakage,
        "sequence_errors": sequence_errors,
        "private_projection_errors": int(
            diagnostics["private_projection_errors"]
        ),
        "reconstructed_context_mismatches": int(
            diagnostics["reconstructed_context_mismatches"]
        ),
        "forbidden_output_keys": forbidden_keys,
        "profiles_by_fold": {
            str(fold): sorted(values) for fold, values in profiles_by_fold.items()
        },
        "feature_names": list(PUBLIC_OPPONENT_BELIEF_FEATURE_NAMES),
        "integrity_gate": gate,
        "integrity_gate_pass": not failures,
        "integrity_gate_failures": failures,
    }


def assign_game_folds(
    trace: Path,
    *,
    profiles: Sequence[str],
    folds: int,
) -> tuple[dict[int, int], dict[int, str], dict[int, str]]:
    grouped: dict[str, list[tuple[tuple[int, int, int, int], int]]] = defaultdict(list)
    profile_by_source = {}
    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            game = json.loads(line)
            opponents = tuple(str(item) for item in game.get("opponents") or ())
            if len(opponents) != 1 or opponents[0] not in profiles:
                raise ValueError("sequential_public_opponent_profile_contract")
            profile = opponents[0]
            key = (
                int(game.get("seed") or 0),
                int(game.get("candidate_seat") or 0),
                int(game.get("dealer") or 0),
                source_index,
            )
            grouped[profile].append((key, source_index))
            profile_by_source[source_index] = profile
    fold_by_source = {}
    game_ids = {}
    for profile in sorted(grouped):
        for profile_index, (key, source_index) in enumerate(sorted(grouped[profile])):
            fold_by_source[source_index] = profile_index % folds
            game_ids[source_index] = (
                f"{profile}:{key[0]}:{key[1]}:{key[2]}:{source_index}"
            )
    return fold_by_source, game_ids, profile_by_source


def extract_game_events(
    game: Mapping[str, Any],
    *,
    game_id: str,
    profile: str,
    fold: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    players = int(game.get("players") or 0)
    candidate_seat = int(game.get("candidate_seat") or 0)
    if players != 2 or candidate_seat not in (0, 1):
        raise ValueError("sequential_public_requires_two_player_trace")
    opponent_seat = 1 - candidate_seat
    rules = rules_for_room(
        wildcard_enabled=bool(game.get("wildcard_enabled")),
        players=players,
    )
    traces = sorted(
        list(game.get("decision_trace") or ()),
        key=lambda item: int(item.get("sequence") or 0),
    )
    events = []
    diagnostics: Counter[str] = Counter()
    for index, trace in enumerate(traces):
        seat = int(trace.get("seat") or 0)
        phase = str(trace.get("phase") or "")
        selected_key = str(trace.get("selected_key") or "")
        sequence = int(trace.get("sequence") or 0)
        if seat == opponent_seat and phase == "discard":
            label = _discard_label(selected_key)
            context = observer_context(
                trace["public_state"],
                candidate_seat=candidate_seat,
                opponent_seat=opponent_seat,
                rules=rules,
            )
            events.append(
                event_record(
                    game_id=game_id,
                    profile=profile,
                    fold=fold,
                    sequence=sequence,
                    event_order=len(events),
                    action_kind="DISCARD",
                    action_token=f"DISCARD:{label}",
                    trigger_label=None,
                    public_meld_groups=(),
                    context=context,
                )
            )
        if seat == opponent_seat and phase in {
            "self_hu",
            "post_auto_hu",
            "post_action_hu",
        }:
            diagnostics["terminal_hu_excluded"] += 1
        if seat != candidate_seat or phase != "discard":
            continue

        trigger_label = _discard_label(selected_key)
        reconstructed = public_state_after_discard(
            trace["public_state"],
            seat=candidate_seat,
            label=trigger_label,
        )
        response = next_opponent_response(
            traces,
            start=index + 1,
            opponent_seat=opponent_seat,
        )
        if response is None:
            action_kind = "NO_CLAIM"
            action_token = "NO_CLAIM"
            meld_groups: tuple[tuple[str, ...], ...] = ()
            response_state = reconstructed
            diagnostics["implicit_no_claim"] += 1
        else:
            response_type = str(response.get("selected_key") or "").split(":", 1)[0]
            action_kind = "NO_CLAIM" if response_type == "PASS" else response_type
            action_token = "NO_CLAIM" if action_kind == "NO_CLAIM" else response_type
            selected_action = selected_public_action(response)
            meld_groups = tuple(
                tuple(str(label) for label in group)
                for group in selected_action.get("meld_groups") or ()
            )
            response_state = response["public_state"]
            if action_kind == "NO_CLAIM":
                diagnostics["explicit_pass"] += 1
            else:
                diagnostics["visible_claim"] += 1
        reconstructed_context = observer_context(
            reconstructed,
            candidate_seat=candidate_seat,
            opponent_seat=opponent_seat,
            rules=rules,
        )
        context = observer_context(
            response_state,
            candidate_seat=candidate_seat,
            opponent_seat=opponent_seat,
            rules=rules,
        )
        diagnostics["reconstructed_context_mismatches"] += int(
            not same_context(reconstructed_context, context)
        )
        events.append(
            event_record(
                game_id=game_id,
                profile=profile,
                fold=fold,
                sequence=sequence,
                event_order=len(events),
                action_kind=action_kind,
                action_token=action_token,
                trigger_label=trigger_label,
                public_meld_groups=meld_groups,
                context=context,
            )
        )

    for event in events:
        diagnostics["private_projection_errors"] += int(
            not all(math.isfinite(float(value)) for value in event["context_features"])
        )
    return events, diagnostics


def next_opponent_response(
    traces: Sequence[Mapping[str, Any]],
    *,
    start: int,
    opponent_seat: int,
) -> Mapping[str, Any] | None:
    for trace in traces[start:]:
        if str(trace.get("phase") or "") == "discard":
            return None
        if (
            int(trace.get("seat") or 0) == opponent_seat
            and str(trace.get("phase") or "") == "response_root"
        ):
            return trace
    return None


def observer_context(
    state: Mapping[str, Any],
    *,
    candidate_seat: int,
    opponent_seat: int,
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    projected = public_projection(state, observer_seat=candidate_seat)
    evidence = public_opponent_features(
        projected,
        rules,
        opponent_seat=opponent_seat,
    )
    return {
        "public_actions_before": evidence.public_actions,
        "features": tuple(float(value) for value in evidence.features),
    }


def public_projection(
    state: Mapping[str, Any],
    *,
    observer_seat: int,
) -> dict[str, Any]:
    return {
        "seat": observer_seat,
        **{key: deepcopy(state.get(key)) for key in PUBLIC_STATE_KEYS},
    }


def public_state_after_discard(
    state: Mapping[str, Any],
    *,
    seat: int,
    label: str,
) -> dict[str, Any]:
    projected = public_projection(state, observer_seat=seat)
    hand_sizes = list(projected.get("hand_sizes") or ())
    discards = [list(items) for items in projected.get("discards") or ()]
    if seat >= len(hand_sizes) or seat >= len(discards):
        raise ValueError("sequential_public_discard_projection_seat")
    hand_sizes[seat] = max(0, int(hand_sizes[seat]) - 1)
    discards[seat].append(label)
    projected["hand_sizes"] = hand_sizes
    projected["discards"] = discards
    return projected


def event_record(
    *,
    game_id: str,
    profile: str,
    fold: int,
    sequence: int,
    event_order: int,
    action_kind: str,
    action_token: str,
    trigger_label: str | None,
    public_meld_groups: Sequence[Sequence[str]],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "sequential-public-opponent-event-v1",
        "game_id": game_id,
        "profile": profile,
        "fold": fold,
        "source_sequence": sequence,
        "event_order": event_order,
        "action_kind": action_kind,
        "action_token": action_token,
        "trigger_label": trigger_label,
        "public_meld_groups": [list(group) for group in public_meld_groups],
        "public_actions_before": int(context["public_actions_before"]),
        "context_features": [float(value) for value in context["features"]],
    }


def selected_public_action(response: Mapping[str, Any]) -> Mapping[str, Any]:
    selected_key = str(response.get("selected_key") or "")
    return next(
        (
            action
            for action in response.get("legal_actions") or ()
            if str(action.get("key") or "") == selected_key
        ),
        {},
    )


def same_context(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return int(first["public_actions_before"]) == int(second["public_actions_before"]) and all(
        abs(float(left) - float(right)) <= 1e-12
        for left, right in zip(first["features"], second["features"])
    )


def sequence_error_count(events: Sequence[Mapping[str, Any]]) -> int:
    by_game: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        by_game[str(event["game_id"])].append(
            (int(event["source_sequence"]), int(event["event_order"]))
        )
    return sum(
        pair <= previous
        for rows in by_game.values()
        for previous, pair in zip(rows, rows[1:])
    )


def grouped_fold_leakage(events: Sequence[Mapping[str, Any]]) -> int:
    folds_by_game: dict[str, set[int]] = defaultdict(set)
    for event in events:
        folds_by_game[str(event["game_id"])].add(int(event["fold"]))
    return sum(len(folds) != 1 for folds in folds_by_game.values())


def count_forbidden_keys(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(
            int(str(key) in FORBIDDEN_OUTPUT_KEYS) + count_forbidden_keys(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sum(count_forbidden_keys(item) for item in value)
    return 0


def integrity_gate_failures(
    *,
    games: int,
    games_by_profile: Mapping[str, int],
    profiles: Sequence[str],
    folds: int,
    events: int,
    event_kinds: Mapping[str, int],
    response_observations: int,
    group_leakage: int,
    sequence_errors: int,
    private_projection_errors: int,
    reconstructed_context_mismatches: int,
    forbidden_output_keys: int,
    profiles_by_fold: Mapping[int, set[str]],
    gate: Mapping[str, Any],
) -> list[str]:
    failures = []
    if games != int(gate["required_games"]):
        failures.append("required_games")
    if len(profiles) != int(gate["required_profiles"]):
        failures.append("required_profiles")
    if any(
        games_by_profile.get(profile, 0)
        != int(gate["required_games_per_profile"])
        for profile in profiles
    ):
        failures.append("required_games_per_profile")
    if folds != int(gate["required_folds"]):
        failures.append("required_folds")
    if events < int(gate["minimum_events"]):
        failures.append("minimum_events")
    if int(event_kinds.get("DISCARD", 0)) < int(gate["minimum_discard_events"]):
        failures.append("minimum_discard_events")
    if response_observations < int(gate["minimum_response_observations"]):
        failures.append("minimum_response_observations")
    if group_leakage > int(gate["maximum_group_leakage"]):
        failures.append("group_leakage")
    if sequence_errors > int(gate["maximum_sequence_errors"]):
        failures.append("sequence_errors")
    if private_projection_errors > int(gate["maximum_private_projection_errors"]):
        failures.append("private_projection_errors")
    if reconstructed_context_mismatches > int(
        gate["maximum_reconstructed_context_mismatches"]
    ):
        failures.append("reconstructed_context_mismatches")
    if forbidden_output_keys > int(gate["maximum_forbidden_output_keys"]):
        failures.append("forbidden_output_keys")
    if gate["require_all_public_action_kinds"] and set(event_kinds) != {
        "DISCARD",
        "NO_CLAIM",
        "CHI",
        "PENG",
        "HU",
    }:
        failures.append("public_action_kinds")
    if gate["require_all_profiles_in_every_fold"] and any(
        set(profiles_by_fold.get(fold, set())) != set(profiles)
        for fold in range(folds)
    ):
        failures.append("profiles_in_every_fold")
    return failures


def write_jsonl_gzip(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    text.write("\n")


def _discard_label(selected_key: str) -> str:
    if not selected_key.startswith("DISCARD:"):
        raise ValueError("sequential_public_discard_key")
    return selected_key.split(":", 1)[1]


if __name__ == "__main__":
    raise SystemExit(main())
