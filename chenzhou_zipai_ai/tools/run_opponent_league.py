"""Run the reproducible multi-style strategy league."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
while str(APP_ROOT) in sys.path:
    sys.path.remove(str(APP_ROOT))

import logging as _stdlib_logging


sys.path.insert(0, str(APP_ROOT))

from ai.opponent_league import (
    DEFAULT_HEADS_UP_MATCHUPS,
    DEFAULT_MATCHUPS,
    run_opponent_league,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="professional_brain")
    parser.add_argument("--deals-per-matchup", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--wildcard", choices=("off", "on"), required=True)
    parser.add_argument("--players", type=int, choices=(2, 3), default=3)
    parser.add_argument("--matchup", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="Optional gzip JSONL path for complete per-game decision traces.",
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        help="Optional durable JSONL checkpoint written after every game.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --checkpoint-output after strict configuration checks.",
    )
    parser.add_argument("--balanced-rotation-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--paired-baseline-report",
        type=Path,
        help="Frozen paired league used for formal early rejection.",
    )
    parser.add_argument(
        "--early-reject-dominated-matchup",
        action="store_true",
        help=(
            "Stop after a complete matchup when both paired outcome utility "
            "and score are below the frozen baseline."
        ),
    )
    args = parser.parse_args()
    default_matchups = DEFAULT_HEADS_UP_MATCHUPS if args.players == 2 else DEFAULT_MATCHUPS
    matchups = (
        tuple(_parse_matchup(value, opponents=args.players - 1) for value in args.matchup)
        or default_matchups
    )
    total = (
        len(matchups)
        * max(1, args.deals_per_matchup)
        * (
            1
            if args.balanced_rotation_only
            else args.players * args.players
        )
    )
    if args.resume and args.checkpoint_output is None:
        parser.error("--resume requires --checkpoint-output")
    if (
        args.early_reject_dominated_matchup
        and args.paired_baseline_report is None
    ):
        parser.error(
            "--early-reject-dominated-matchup requires "
            "--paired-baseline-report"
        )
    if args.checkpoint_output is not None:
        checkpoint_path = args.checkpoint_output.resolve()
        conflicting_paths = {
            args.output.resolve(),
            *(
                [args.trace_output.resolve()]
                if args.trace_output is not None
                else []
            ),
        }
        if checkpoint_path in conflicting_paths:
            parser.error(
                "--checkpoint-output must differ from --output and --trace-output"
            )
    checkpoint = (
        _LeagueCheckpoint(args.checkpoint_output)
        if args.checkpoint_output is not None
        else None
    )
    checkpoint_metadata = _league_checkpoint_metadata(
        candidate=args.candidate,
        matchups=matchups,
        deals_per_matchup=max(1, args.deals_per_matchup),
        wildcard_enabled=args.wildcard == "on",
        seed=args.seed,
        players=args.players,
        record_decisions=args.trace_output is not None,
        balanced_rotation_only=args.balanced_rotation_only,
        total=total,
    )
    completed_rows: list[dict] = []
    compact_checkpointed_traces = bool(
        checkpoint is not None and args.trace_output is not None
    )
    if checkpoint is not None:
        try:
            if args.resume:
                completed_rows = checkpoint.load(
                    checkpoint_metadata,
                    compact_decision_traces=compact_checkpointed_traces,
                )
            else:
                checkpoint.create(checkpoint_metadata)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    progress = _LeagueProgressReporter(
        every=max(0, int(args.progress_every)),
        heartbeat_seconds=max(1.0, float(args.heartbeat_seconds)),
        total=total,
        initial_rows=completed_rows,
    )
    early_stop_callback = None
    if args.early_reject_dominated_matchup:
        try:
            baseline_report = json.loads(
                args.paired_baseline_report.read_text(encoding="utf-8")
            )
            early_stop_callback = (
                build_dominated_matchup_early_stop_callback(
                    baseline_report,
                    players=args.players,
                    wildcard_enabled=args.wildcard == "on",
                    seed=args.seed,
                    deals_per_matchup=max(1, args.deals_per_matchup),
                    balanced_rotation_only=args.balanced_rotation_only,
                    expected_games=total,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))

    def record_progress(completed: int, expected_total: int, row: dict) -> None:
        if checkpoint is not None:
            checkpoint.append_row(completed, row)
        if compact_checkpointed_traces:
            _compact_checkpointed_decision_trace(row)
        progress.update(completed, expected_total, row)

    progress.start()
    try:
        report = run_opponent_league(
            candidate=args.candidate,
            matchups=matchups,
            deals_per_matchup=max(1, args.deals_per_matchup),
            wildcard_enabled=args.wildcard == "on",
            seed=args.seed,
            workers=max(1, args.workers),
            players=args.players,
            record_decisions=args.trace_output is not None,
            balanced_rotation_only=args.balanced_rotation_only,
            completed_rows=completed_rows,
            progress_callback=record_progress,
            early_stop_callback=early_stop_callback,
        )
    finally:
        progress.stop()
    if args.trace_output is not None:
        trace_games = 0
        trace_steps = 0
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_rows = (
            checkpoint.iter_rows()
            if checkpoint is not None
            else iter(report["rows"])
        )
        with gzip.open(args.trace_output, "wt", encoding="utf-8") as handle:
            for row in trace_rows:
                trace = row.get("decision_trace", [])
                trace_games += 1
                trace_steps += len(trace)
                handle.write(
                    json.dumps(
                        {
                            "candidate": row["candidate"],
                            "opponents": row["opponents"],
                            "players": row["players"],
                            "candidate_seat": row["candidate_seat"],
                            "dealer": row["dealer"],
                            "seed": row["seed"],
                            "wildcard_enabled": report["wildcard_enabled"],
                            "result": row["result"],
                            "decision_trace": trace,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        for row in report["rows"]:
            row.pop("decision_trace", None)
            row.pop("_decision_trace_checkpointed", None)
        report["decision_trace_manifest"] = {
            "format": "gzip-jsonl",
            "path": str(args.trace_output),
            "games": trace_games,
            "steps": trace_steps,
        }
    if checkpoint is not None:
        report["checkpoint_manifest"] = {
            "schema_version": _LeagueCheckpoint.SCHEMA_VERSION,
            "path": str(args.checkpoint_output),
            "resumed_games": len(completed_rows),
            "games": report["games"],
        }
        if not report["early_stopped"]:
            checkpoint.mark_complete(report["games"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    console_summary = {
        key: report[key]
        for key in (
            "ok",
            "candidate",
            "wildcard_enabled",
            "players",
            "games",
            "candidate_wins",
            "losses",
            "draws",
            "candidate_win_share_all",
            "candidate_win_share_decisive",
            "mean_candidate_outcome_score",
            "invariant_violations",
            "candidate_diagnostics",
            "worst_matchup",
            "resumed_games",
            "expected_games",
            "early_stopped",
            "early_stop",
        )
    }
    response_shadow = dict(report["response_shadow"])
    response_shadow.pop("disagreement_samples", None)
    console_summary["response_shadow"] = response_shadow
    discard_shadow = dict(report["discard_shadow"])
    discard_shadow.pop("samples", None)
    console_summary["discard_shadow"] = discard_shadow
    print(json.dumps(console_summary, ensure_ascii=False))
    return 0 if report["ok"] else 1


def build_dominated_matchup_early_stop_callback(
    baseline_report: Mapping[str, Any],
    *,
    players: int,
    wildcard_enabled: bool,
    seed: int,
    deals_per_matchup: int,
    balanced_rotation_only: bool,
    expected_games: int,
) -> Callable[
    [list[dict[str, Any]], int],
    Mapping[str, Any] | None,
]:
    expected_config = {
        "players": players,
        "wildcard_enabled": wildcard_enabled,
        "seed": seed,
        "deals_per_matchup": deals_per_matchup,
        "balanced_rotation_only": balanced_rotation_only,
    }
    mismatches = {
        key: {
            "expected": value,
            "observed": baseline_report.get(key),
        }
        for key, value in expected_config.items()
        if baseline_report.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "paired_baseline_configuration_mismatch:"
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    baseline_rows = {
        _paired_row_identity(row): row
        for row in baseline_report.get("rows") or ()
    }
    if len(baseline_rows) != expected_games:
        raise ValueError(
            "paired_baseline_game_count_mismatch:"
            f"expected={expected_games}:observed={len(baseline_rows)}"
        )
    matchup_games = deals_per_matchup * (
        1 if balanced_rotation_only else players * players
    )

    def evaluate(
        rows: list[dict[str, Any]],
        total_games: int,
    ) -> Mapping[str, Any] | None:
        if (
            not rows
            or total_games != expected_games
            or len(rows) % matchup_games
        ):
            return None
        candidate_group = rows[-matchup_games:]
        identities = [_paired_row_identity(row) for row in candidate_group]
        missing = [identity for identity in identities if identity not in baseline_rows]
        if missing:
            raise ValueError(
                f"paired_baseline_rows_missing:{len(missing)}"
            )
        baseline_group = [baseline_rows[identity] for identity in identities]
        matchups = {
            tuple(sorted(str(name) for name in row["opponents"]))
            for row in candidate_group
        }
        if len(matchups) != 1:
            raise ValueError("candidate_matchup_boundary_not_aligned")
        candidate_utility = sum(
            _row_outcome_utility(row) for row in candidate_group
        )
        baseline_utility = sum(
            _row_outcome_utility(row) for row in baseline_group
        )
        candidate_score = sum(
            float(row["candidate_outcome_score"])
            for row in candidate_group
        )
        baseline_score = sum(
            float(row["candidate_outcome_score"])
            for row in baseline_group
        )
        if (
            candidate_utility >= baseline_utility
            or candidate_score >= baseline_score
        ):
            return None
        matchup = next(iter(matchups))
        return {
            "reason": "paired_matchup_dominated",
            "matchup": " + ".join(matchup),
            "completed_games": len(rows),
            "expected_games": expected_games,
            "matchup_games": matchup_games,
            "candidate_outcome_utility": candidate_utility,
            "baseline_outcome_utility": baseline_utility,
            "outcome_utility_delta": (
                candidate_utility - baseline_utility
            ),
            "candidate_total_score": round(candidate_score, 6),
            "baseline_total_score": round(baseline_score, 6),
            "total_score_delta": round(
                candidate_score - baseline_score,
                6,
            ),
        }

    return evaluate


def _paired_row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(sorted(str(name) for name in row["opponents"])),
        int(row["candidate_seat"]),
        int(row["dealer"]),
        int(row["seed"]),
    )


def _row_outcome_utility(row: Mapping[str, Any]) -> int:
    if bool(row.get("draw")):
        return 0
    return 1 if bool(row.get("candidate_won")) else -1


def _parse_matchup(value: str, *, opponents: int) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(parts) != opponents:
        raise argparse.ArgumentTypeError(
            f"matchup must contain exactly {opponents} opponent policy name(s)"
        )
    return parts


def _league_checkpoint_metadata(
    *,
    candidate: str,
    matchups: tuple[tuple[str, ...], ...],
    deals_per_matchup: int,
    wildcard_enabled: bool,
    seed: int,
    players: int,
    record_decisions: bool,
    balanced_rotation_only: bool,
    total: int,
) -> dict:
    return {
        "candidate": candidate,
        "matchups": [list(matchup) for matchup in matchups],
        "deals_per_matchup": deals_per_matchup,
        "wildcard_enabled": wildcard_enabled,
        "seed": seed,
        "players": players,
        "record_decisions": record_decisions,
        "balanced_rotation_only": balanced_rotation_only,
        "total": total,
    }


def _compact_checkpointed_decision_trace(row: dict) -> None:
    """Keep full trace on disk while retaining only a presence marker in RAM."""

    if "decision_trace" in row:
        row.pop("decision_trace")
        row["_decision_trace_checkpointed"] = True


class _LeagueCheckpoint:
    SCHEMA_VERSION = "opponent-league-checkpoint-v1"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.metadata: dict | None = None
        self.row_count = 0
        self.complete = False

    def create(self, metadata: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "record_type": "header",
            "schema_version": self.SCHEMA_VERSION,
            "metadata": metadata,
        }
        try:
            with self.path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise ValueError(
                f"checkpoint_exists_use_resume:{self.path}"
            ) from exc
        self.metadata = dict(metadata)

    def load(
        self,
        expected_metadata: dict,
        *,
        compact_decision_traces: bool = False,
    ) -> list[dict]:
        if not self.path.is_file():
            raise ValueError(f"checkpoint_not_found:{self.path}")
        self._repair_torn_tail()
        rows: list[dict] = []
        header_seen = False
        complete_seen = False
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid_checkpoint_json:line={line_number}"
                    ) from exc
                record_type = payload.get("record_type")
                if not header_seen:
                    if (
                        record_type != "header"
                        or payload.get("schema_version")
                        != self.SCHEMA_VERSION
                    ):
                        raise ValueError("invalid_checkpoint_header")
                    observed_metadata = payload.get("metadata")
                    if observed_metadata != expected_metadata:
                        differences = sorted(
                            key
                            for key in set(expected_metadata)
                            | set(observed_metadata or {})
                            if (observed_metadata or {}).get(key)
                            != expected_metadata.get(key)
                        )
                        raise ValueError(
                            "checkpoint_configuration_mismatch:"
                            + ",".join(differences)
                        )
                    header_seen = True
                    self.metadata = dict(expected_metadata)
                    continue
                if record_type == "row":
                    if complete_seen:
                        raise ValueError("checkpoint_row_after_complete")
                    expected_index = len(rows) + 1
                    if payload.get("index") != expected_index:
                        raise ValueError(
                            "checkpoint_row_index_mismatch:"
                            f"expected={expected_index}:"
                            f"observed={payload.get('index')}"
                        )
                    row = payload.get("row")
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"invalid_checkpoint_row:index={expected_index}"
                        )
                    if compact_decision_traces:
                        _compact_checkpointed_decision_trace(row)
                    rows.append(row)
                    continue
                if record_type == "complete":
                    if int(payload.get("games") or -1) != len(rows):
                        raise ValueError("checkpoint_complete_count_mismatch")
                    complete_seen = True
                    continue
                raise ValueError(
                    f"unknown_checkpoint_record:line={line_number}"
                )
        if not header_seen:
            raise ValueError("checkpoint_missing_header")
        if len(rows) > int(expected_metadata["total"]):
            raise ValueError("checkpoint_rows_exceed_schedule")
        if complete_seen and len(rows) != int(expected_metadata["total"]):
            raise ValueError("checkpoint_complete_before_schedule_end")
        self.row_count = len(rows)
        self.complete = complete_seen
        return rows

    def iter_rows(self):
        """Stream full checkpoint rows without retaining their traces in RAM."""

        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("record_type") == "row":
                    yield payload["row"]

    def append_row(self, completed: int, row: dict) -> None:
        if self.complete:
            raise ValueError("checkpoint_already_complete")
        expected = self.row_count + 1
        if completed != expected:
            raise ValueError(
                "checkpoint_append_index_mismatch:"
                f"expected={expected}:observed={completed}"
            )
        self._append(
            {
                "record_type": "row",
                "index": completed,
                "row": row,
            }
        )
        self.row_count = completed

    def mark_complete(self, games: int) -> None:
        if self.complete:
            if games != self.row_count:
                raise ValueError("checkpoint_complete_count_mismatch")
            return
        if games != self.row_count:
            raise ValueError(
                "checkpoint_incomplete:"
                f"rows={self.row_count}:games={games}"
            )
        self._append(
            {
                "record_type": "complete",
                "games": games,
            }
        )
        self.complete = True

    def _append(self, payload: dict) -> None:
        encoded = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with self.path.open("ab", buffering=0) as handle:
            handle.write(encoded)
            os.fsync(handle.fileno())

    def _repair_torn_tail(self) -> None:
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        last_newline = raw.rfind(b"\n")
        tail = raw[last_newline + 1 :]
        try:
            json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            with self.path.open("r+b") as handle:
                handle.truncate(last_newline + 1)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with self.path.open("ab", buffering=0) as handle:
                handle.write(b"\n")
                os.fsync(handle.fileno())


class _LeagueProgressReporter:
    def __init__(
        self,
        *,
        every: int,
        heartbeat_seconds: float,
        total: int,
        initial_rows: list[dict] | None = None,
    ) -> None:
        rows = list(initial_rows or [])
        self.every = every
        self.heartbeat_seconds = heartbeat_seconds
        self.started = time.perf_counter()
        self.completed = len(rows)
        self.total = max(0, int(total))
        self.wins = sum(int(bool(row.get("candidate_won"))) for row in rows)
        self.draws = sum(int(bool(row.get("draw"))) for row in rows)
        self.losses = self.completed - self.wins - self.draws
        self.last_seed = int(rows[-1]["seed"]) if rows else None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="league-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def update(
        self,
        completed: int,
        total: int,
        row: dict,
    ) -> None:
        with self._lock:
            self.completed = completed
            self.total = total
            self.wins += int(bool(row.get("candidate_won")))
            self.draws += int(bool(row.get("draw")))
            self.losses = completed - self.wins - self.draws
            self.last_seed = int(row["seed"])
            should_print = bool(
                self.every
                and (
                    completed % self.every == 0
                    or completed == total
                )
            )
            snapshot = self._snapshot("league_progress")
        if should_print:
            self._emit(snapshot)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.heartbeat_seconds + 1.0)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            with self._lock:
                snapshot = self._snapshot("league_heartbeat")
            self._emit(snapshot)

    def _snapshot(self, event: str) -> dict:
        return {
            "event": event,
            "completed": self.completed,
            "total": self.total,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "last_seed": self.last_seed,
            "elapsed_seconds": round(
                time.perf_counter() - self.started,
                1,
            ),
        }

    @staticmethod
    def _emit(payload: dict) -> None:
        print(
            json.dumps(payload, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
