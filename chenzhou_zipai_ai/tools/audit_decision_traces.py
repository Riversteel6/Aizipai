"""Audit traced decisions against every legal action in paired hidden worlds."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
while str(APP_ROOT) in sys.path:
    sys.path.remove(str(APP_ROOT))

from multiprocessing import get_context
from statistics import mean
from typing import Any, Iterator


sys.path.insert(0, str(APP_ROOT))

from ai.counterfactual_audit import (
    CounterfactualAuditConfig,
    audit_discard_trace,
    audit_hu_trace,
    audit_response_trace,
)
from ai.opponent_league import create_policy
from engine.rules import rules_for_room


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--policy", default="professional_brain_v2")
    parser.add_argument("--paired-worlds", type=int, default=8)
    parser.add_argument("--min-confidence-pairs", type=int, default=8)
    parser.add_argument("--time-budget-ms", type=int, default=60_000)
    parser.add_argument("--rollout-max-turns", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--max-decisions-per-game", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260728)
    parser.add_argument("--balanced-rotation-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        help="Optional durable JSONL checkpoint written after every state.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a strictly matching --checkpoint-output.",
    )
    parser.add_argument(
        "--state-hash",
        action="append",
        default=[],
        help="Audit only matching state_before_hash values; repeat for multiple states.",
    )
    parser.add_argument(
        "--state-opponent-manifest",
        type=Path,
        help=(
            "Optional manifest whose rows assign opponent_stratum to each "
            "state_before_hash. Its states are added to --state-hash and "
            "share one global worker pool."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("all", "discard", "response", "hu"),
        default="all",
    )
    parser.add_argument("--opponent-policy", action="append", default=[])
    parser.add_argument(
        "--discard-comparison-report",
        type=Path,
        help=(
            "Optional paired trace-comparison report. For matching discard "
            "states, audit only its baseline and candidate actions."
        ),
    )
    args = parser.parse_args()
    if args.resume and args.checkpoint_output is None:
        parser.error("--resume requires --checkpoint-output")
    if args.checkpoint_output is not None:
        checkpoint_path = args.checkpoint_output.resolve()
        conflicting_paths = {
            args.output.resolve(),
            args.evidence_output.resolve(),
        }
        if checkpoint_path in conflicting_paths:
            parser.error(
                "--checkpoint-output must differ from --output and "
                "--evidence-output"
            )

    opponent_names = tuple(
        args.opponent_policy
        or (
            "independent_fast_rollout",
            "independent_fast_pressure",
            "independent_fast_denial",
        )
    )
    state_opponents = (
        _load_state_opponents(args.state_opponent_manifest)
        if args.state_opponent_manifest is not None
        else {}
    )
    state_hashes = {
        str(value)
        for value in args.state_hash
        if str(value)
    } | set(state_opponents)
    discard_comparisons = (
        _load_discard_comparisons(args.discard_comparison_report)
        if args.discard_comparison_report is not None
        else {}
    )
    if discard_comparisons and args.phase not in {"all", "discard"}:
        parser.error(
            "--discard-comparison-report requires discard phase"
        )
    tasks: list[dict[str, Any]] = []
    seen_games = 0
    eligible = 0
    eligible_phase_counts: Counter[str] = Counter()
    sampled_phase_counts: Counter[str] = Counter()
    sampled_games = 0
    stop = False
    for game_index, game in enumerate(
        _read_jsonl(
            args.trace_input,
            required_substrings=state_hashes,
        )
    ):
        seen_games += 1
        if game is None:
            continue
        if (
            args.balanced_rotation_only
            and not _is_balanced_rotation(game)
        ):
            continue
        eligible_traces: list[dict[str, Any]] = []
        for trace in game.get("decision_trace") or ():
            if (
                state_hashes
                and str(trace.get("state_before_hash") or "") not in state_hashes
            ):
                continue
            phase = str(trace.get("phase") or "")
            if phase not in {
                "discard",
                "response_root",
                "self_hu",
                "post_auto_hu",
                "post_action_hu",
            }:
                continue
            if args.phase == "discard" and phase != "discard":
                continue
            if args.phase == "response" and phase != "response_root":
                continue
            if args.phase == "hu" and phase not in {
                "self_hu",
                "post_auto_hu",
                "post_action_hu",
            }:
                continue
            if args.policy and str(trace.get("policy") or "") != args.policy:
                continue
            eligible += 1
            eligible_phase_counts[phase] += 1
            eligible_traces.append(trace)
        selected_traces = _sample_game_traces(
            eligible_traces,
            maximum=max(0, args.max_decisions_per_game),
            seed=(
                int(args.sample_seed)
                + int(game.get("seed") or 0) * 31
                + game_index
            ),
        )
        if args.max_decisions > 0:
            remaining = max(0, args.max_decisions - len(tasks))
            selected_traces = selected_traces[:remaining]
        if selected_traces:
            sampled_games += 1
        for trace in selected_traces:
            phase = str(trace.get("phase") or "")
            state_hash = str(trace.get("state_before_hash") or "")
            sampled_phase_counts[phase] += 1
            tasks.append(
                {
                    "game_index": game_index,
                    "game": {
                        "seed": int(game["seed"]),
                        "players": int(game["players"]),
                        "wildcard_enabled": bool(game["wildcard_enabled"]),
                        "candidate_seat": int(game["candidate_seat"]),
                        "dealer": int(game["dealer"]),
                        "opponents": list(game.get("opponents") or []),
                    },
                    "trace": trace,
                    "opponent_names": state_opponents.get(
                        state_hash,
                        opponent_names,
                    ),
                    "paired_worlds": max(1, args.paired_worlds),
                    "min_confidence_pairs": max(2, args.min_confidence_pairs),
                    "time_budget_ms": max(1, args.time_budget_ms),
                    "rollout_max_turns": max(1, args.rollout_max_turns),
                    "seed": args.seed,
                    "discard_candidate_labels": (
                        discard_comparisons.get(
                            str(trace.get("state_before_hash") or "")
                        )
                    ),
                }
            )
            if args.max_decisions > 0 and len(tasks) >= args.max_decisions:
                stop = True
                break
        if stop:
            break

    if state_opponents:
        _sort_tasks_by_state_order(
            tasks,
            state_order=tuple(state_opponents),
        )
    for task_index, task in enumerate(tasks):
        task["task_index"] = task_index

    checkpoint = (
        _AuditCheckpoint(args.checkpoint_output)
        if args.checkpoint_output is not None
        else None
    )
    checkpoint_metadata = _checkpoint_metadata(args, tasks)
    resumed_results: dict[int, dict[str, Any]] = {}
    checkpoint_complete = False
    if checkpoint is not None:
        if args.resume:
            resumed_results, checkpoint_complete = checkpoint.load(
                checkpoint_metadata
            )
        else:
            checkpoint.create(checkpoint_metadata)

    pending_tasks = [
        task
        for task in tasks
        if int(task["task_index"]) not in resumed_results
    ]
    results_by_index = dict(resumed_results)
    workers = min(
        max(1, args.workers),
        max(1, len(pending_tasks)),
        os.cpu_count() or 1,
    )
    if workers > 1:
        with get_context("spawn").Pool(processes=workers) as pool:
            result_stream = pool.imap_unordered(
                _audit_task,
                pending_tasks,
                chunksize=1,
            )
            _collect_results(
                result_stream,
                results_by_index=results_by_index,
                checkpoint=checkpoint,
                total=len(tasks),
            )
    else:
        _collect_results(
            (_audit_task(task) for task in pending_tasks),
            results_by_index=results_by_index,
            checkpoint=checkpoint,
            total=len(tasks),
        )
    if set(results_by_index) != set(range(len(tasks))):
        raise ValueError("counterfactual_checkpoint_result_set_incomplete")
    results = [results_by_index[index] for index in range(len(tasks))]
    if checkpoint is not None and not checkpoint_complete:
        checkpoint.mark_complete(len(tasks))
    audited_opponent_names = sorted(
        {
            name
            for task in tasks
            for name in task["opponent_names"]
        }
    )
    audits = [
        result["audit"]
        for result in results
        if "audit" in result
    ]
    failures = [
        result["failure"]
        for result in results
        if "failure" in result
    ]
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.evidence_output, "wt", encoding="utf-8") as evidence:
        for result in results:
            evidence.write(
                json.dumps(
                    result.get("audit") or result["failure"],
                    ensure_ascii=False,
                )
                + "\n"
            )

    regrets = [
        float(audit["counterfactual_regret"])
        for audit in audits
        if audit.get("counterfactual_regret") is not None
    ]
    complete = [audit for audit in audits if audit.get("complete")]
    suboptimal = [
        audit
        for audit in complete
        if not audit.get("selected_is_empirical_best")
    ]
    confident_suboptimal = [
        audit
        for audit in complete
        if audit.get("confidently_suboptimal")
    ]
    complete_by_phase = Counter(
        str(audit.get("phase") or "")
        for audit in complete
    )
    suboptimal_by_phase = Counter(
        str(audit.get("phase") or "")
        for audit in suboptimal
    )
    confident_suboptimal_by_phase = Counter(
        str(audit.get("phase") or "")
        for audit in confident_suboptimal
    )
    summary = {
        "ok": not failures and len(complete) == len(tasks),
        "schema_version": "counterfactual-trace-audit-summary-v2",
        "trace_input": str(args.trace_input),
        "evidence_output": str(args.evidence_output),
        "policy": args.policy,
        "phase": args.phase,
        "opponent_policies": audited_opponent_names,
        "state_opponent_manifest": (
            str(args.state_opponent_manifest)
            if args.state_opponent_manifest is not None
            else None
        ),
        "state_opponent_assignments": len(state_opponents),
        "games_read": seen_games,
        "sampled_games": sampled_games,
        "eligible_decisions": eligible,
        "sampled_decisions": len(tasks),
        "audited_decisions": len(audits),
        "complete_decisions": len(complete),
        "failed_decisions": len(failures),
        "empirically_suboptimal_decisions": len(suboptimal),
        "confidently_suboptimal_decisions": len(confident_suboptimal),
        "eligible_by_phase": dict(sorted(eligible_phase_counts.items())),
        "sampled_by_phase": dict(sorted(sampled_phase_counts.items())),
        "sampling": {
            "balanced_rotation_only": bool(args.balanced_rotation_only),
            "max_decisions_per_game": max(
                0,
                int(args.max_decisions_per_game),
            ),
            "sample_seed": int(args.sample_seed),
        },
        "complete_by_phase": dict(sorted(complete_by_phase.items())),
        "empirically_suboptimal_by_phase": dict(
            sorted(suboptimal_by_phase.items())
        ),
        "confidently_suboptimal_by_phase": dict(
            sorted(confident_suboptimal_by_phase.items())
        ),
        "suboptimal_rate": (
            round(len(suboptimal) / len(complete), 6)
            if complete
            else None
        ),
        "mean_counterfactual_regret": (
            round(mean(regrets), 6)
            if regrets
            else None
        ),
        "max_counterfactual_regret": (
            round(max(regrets), 6)
            if regrets
            else None
        ),
        "mean_selected_win_rate": _mean_field(
            complete,
            "selected_win_rate",
        ),
        "mean_selected_loss_rate": _mean_field(
            complete,
            "selected_loss_rate",
        ),
        "mean_selected_draw_rate": _mean_field(
            complete,
            "selected_draw_rate",
        ),
        "mean_selected_expected_score": _mean_field(
            complete,
            "selected_expected_score",
        ),
        "mean_best_win_rate": _mean_field(
            complete,
            "best_win_rate",
        ),
        "mean_best_expected_score": _mean_field(
            complete,
            "best_expected_score",
        ),
        "requested_paired_worlds": max(1, args.paired_worlds),
        "checkpoint_manifest": (
            {
                "path": str(args.checkpoint_output),
                "resumed_results": len(resumed_results),
                "new_results": len(pending_tasks),
                "complete": True,
            }
            if args.checkpoint_output is not None
            else None
        ),
        "failures": failures[:100],
        "largest_regrets": [
            {
                "game": audit["game"],
                "trace_sequence": audit["trace_sequence"],
                "turn": audit["turn"],
                "phase": audit["phase"],
                "selected_key": audit["selected_key"],
                "best_key": audit["best_key"],
                "counterfactual_regret": audit["counterfactual_regret"],
                "confidently_suboptimal": audit["confidently_suboptimal"],
                "selected_win_rate": audit.get("selected_win_rate"),
                "selected_expected_score": audit.get("selected_expected_score"),
                "best_win_rate": audit.get("best_win_rate"),
                "best_expected_score": audit.get("best_expected_score"),
            }
            for audit in sorted(
                complete,
                key=lambda item: float(item.get("counterfactual_regret") or 0.0),
                reverse=True,
            )[:100]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"failures", "largest_regrets"}}, ensure_ascii=False))
    return 0 if summary["ok"] else 1


def _is_balanced_rotation(game: dict[str, Any]) -> bool:
    players = int(game.get("players") or 0)
    if players not in {2, 3}:
        return False
    seed = abs(int(game.get("seed") or 0))
    return (
        int(game.get("candidate_seat") or 0) == seed % players
        and int(game.get("dealer") or 0) == (seed // players) % players
    )


def _sample_game_traces(
    traces: list[dict[str, Any]],
    *,
    maximum: int,
    seed: int,
) -> list[dict[str, Any]]:
    if maximum <= 0 or len(traces) <= maximum:
        return list(traces)
    rng = random.Random(seed)
    indexed = list(enumerate(traces))
    by_family: dict[str, list[tuple[int, dict[str, Any]]]] = {
        "discard": [],
        "response": [],
        "hu": [],
    }
    for item in indexed:
        phase = str(item[1].get("phase") or "")
        family = (
            "discard"
            if phase == "discard"
            else "response"
            if phase == "response_root"
            else "hu"
        )
        by_family[family].append(item)
    selected: list[tuple[int, dict[str, Any]]] = []
    selected_indices: set[int] = set()
    for family in ("response", "hu", "discard"):
        if len(selected) >= maximum or not by_family[family]:
            continue
        item = rng.choice(by_family[family])
        selected.append(item)
        selected_indices.add(item[0])
    remaining = [
        item
        for item in indexed
        if item[0] not in selected_indices
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, maximum - len(selected))])
    return [
        trace
        for _index, trace in sorted(selected, key=lambda item: item[0])
    ]


def _read_jsonl(
    path: Path,
    *,
    required_substrings: set[str] | None = None,
) -> Iterator[dict[str, Any] | None]:
    if path.suffix.lower() == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    required = tuple(required_substrings or ())
    with handle:
        for line in handle:
            if not line.strip():
                continue
            if required and not any(value in line for value in required):
                yield None
                continue
            yield json.loads(line)


def _audit_task(task: dict[str, Any]) -> dict[str, Any]:
    task_index = int(task["task_index"])
    trace = task["trace"]
    game = task["game"]
    phase = str(trace.get("phase") or "")
    rules = rules_for_room(
        wildcard_enabled=bool(game["wildcard_enabled"]),
        players=int(game["players"]),
    )
    factories = tuple(
        (lambda name=name: create_policy(name))
        for name in task["opponent_names"]
    )
    try:
        config = CounterfactualAuditConfig(
            paired_worlds=int(task["paired_worlds"]),
            min_confidence_pairs=int(task["min_confidence_pairs"]),
            time_budget_ms=int(task["time_budget_ms"]),
            rollout_max_turns=int(task["rollout_max_turns"]),
            seed=int(task["seed"]),
        )
        if phase == "discard":
            audit = audit_discard_trace(
                trace,
                rules=rules,
                config=config,
                rollout_policy_factories=factories,
                candidate_labels=task.get("discard_candidate_labels"),
            )
        elif phase == "response_root":
            audit = audit_response_trace(
                trace,
                rules=rules,
                config=config,
                rollout_policy_factories=factories,
            )
        else:
            audit = audit_hu_trace(
                trace,
                rules=rules,
                config=config,
                rollout_policy_factories=factories,
            )
        audit["game"] = game
        return {"task_index": task_index, "audit": audit}
    except (TypeError, ValueError) as exc:
        return {
            "task_index": task_index,
            "failure": {
                "game_seed": game.get("seed"),
                "trace_sequence": trace.get("sequence"),
                "phase": phase,
                "error": f"{type(exc).__name__}:{exc}",
            }
        }


def _collect_results(
    result_stream: Iterator[dict[str, Any]],
    *,
    results_by_index: dict[int, dict[str, Any]],
    checkpoint: "_AuditCheckpoint | None",
    total: int,
) -> None:
    report_every = max(1, total // 20)
    for result in result_stream:
        task_index = int(result["task_index"])
        if task_index in results_by_index:
            raise ValueError(
                f"counterfactual_duplicate_task_result:{task_index}"
            )
        results_by_index[task_index] = result
        if checkpoint is not None:
            checkpoint.append_result(result)
        completed = len(results_by_index)
        if completed == total or completed % report_every == 0:
            print(
                json.dumps(
                    {
                        "checkpoint_progress": completed,
                        "total": total,
                    }
                ),
                flush=True,
            )


def _checkpoint_metadata(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    task_identities = [
        {
            "task_index": int(task["task_index"]),
            "state_before_hash": str(
                task["trace"].get("state_before_hash") or ""
            ),
            "trace_sequence": int(task["trace"].get("sequence") or 0),
            "game_seed": int(task["game"]["seed"]),
            "opponent_names": list(task["opponent_names"]),
        }
        for task in tasks
    ]
    fingerprint = hashlib.sha256(
        json.dumps(
            task_identities,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "trace_input": str(args.trace_input.resolve()),
        "policy": str(args.policy),
        "phase": str(args.phase),
        "paired_worlds": max(1, int(args.paired_worlds)),
        "min_confidence_pairs": max(2, int(args.min_confidence_pairs)),
        "time_budget_ms": max(1, int(args.time_budget_ms)),
        "rollout_max_turns": max(1, int(args.rollout_max_turns)),
        "seed": int(args.seed),
        "sample_seed": int(args.sample_seed),
        "max_decisions": max(0, int(args.max_decisions)),
        "max_decisions_per_game": max(
            0,
            int(args.max_decisions_per_game),
        ),
        "balanced_rotation_only": bool(args.balanced_rotation_only),
        "task_count": len(tasks),
        "task_identity_sha256": fingerprint,
    }


class _AuditCheckpoint:
    SCHEMA_VERSION = "counterfactual-trace-audit-checkpoint-v1"

    def __init__(self, path: Path) -> None:
        self.path = path

    def create(self, metadata: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ValueError(
                f"checkpoint_exists_use_resume:{self.path}"
            )
        self._append(
            {
                "record_type": "header",
                "schema_version": self.SCHEMA_VERSION,
                "metadata": metadata,
            }
        )

    def load(
        self,
        expected_metadata: dict[str, Any],
    ) -> tuple[dict[int, dict[str, Any]], bool]:
        if not self.path.exists():
            raise ValueError(f"checkpoint_not_found:{self.path}")
        records = self._read_and_repair_trailing_record()
        if not records:
            raise ValueError("checkpoint_missing_header")
        header = records[0]
        if (
            header.get("record_type") != "header"
            or header.get("schema_version") != self.SCHEMA_VERSION
        ):
            raise ValueError("invalid_checkpoint_header")
        if header.get("metadata") != expected_metadata:
            raise ValueError("checkpoint_configuration_mismatch")

        task_count = int(expected_metadata["task_count"])
        results: dict[int, dict[str, Any]] = {}
        complete = False
        for record in records[1:]:
            record_type = record.get("record_type")
            if record_type == "result":
                if complete:
                    raise ValueError("checkpoint_result_after_complete")
                result = record.get("result")
                if not isinstance(result, dict):
                    raise ValueError("invalid_checkpoint_result")
                task_index = int(result.get("task_index", -1))
                if task_index < 0 or task_index >= task_count:
                    raise ValueError(
                        f"checkpoint_task_index_out_of_range:{task_index}"
                    )
                if task_index in results:
                    raise ValueError(
                        f"checkpoint_duplicate_task_index:{task_index}"
                    )
                results[task_index] = result
            elif record_type == "complete":
                if int(record.get("count", -1)) != task_count:
                    raise ValueError("checkpoint_complete_count_mismatch")
                complete = True
            else:
                raise ValueError(
                    f"unknown_checkpoint_record:{record_type}"
                )
        if complete and len(results) != task_count:
            raise ValueError("checkpoint_complete_before_all_results")
        return results, complete

    def append_result(self, result: dict[str, Any]) -> None:
        self._append({"record_type": "result", "result": result})

    def mark_complete(self, count: int) -> None:
        self._append({"record_type": "complete", "count": int(count)})

    def _append(self, record: dict[str, Any]) -> None:
        payload = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def _read_and_repair_trailing_record(self) -> list[dict[str, Any]]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError(
                        f"invalid_checkpoint_json:line={index + 1}"
                    )
                valid_text = "\n".join(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for record in records
                )
                self.path.write_text(
                    valid_text + ("\n" if valid_text else ""),
                    encoding="utf-8",
                )
        return records


def _load_discard_comparisons(
    path: Path,
) -> dict[str, tuple[str, str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    comparisons: dict[str, tuple[str, str]] = {}
    for row in report.get("rows") or ():
        state_hash = str(row.get("state_before_hash") or "")
        baseline = str(row.get("baseline_selected_key") or "")
        candidate = str(row.get("candidate_selected_key") or "")
        if (
            not state_hash
            or not baseline.startswith("DISCARD:")
            or not candidate.startswith("DISCARD:")
            or baseline == candidate
        ):
            continue
        labels = (
            baseline.split(":", 1)[1],
            candidate.split(":", 1)[1],
        )
        previous = comparisons.setdefault(state_hash, labels)
        if previous != labels:
            raise ValueError(
                f"discard_comparison_conflict:{state_hash}"
            )
    if not comparisons:
        raise ValueError("discard_comparison_report_has_no_actions")
    return comparisons


def _load_state_opponents(path: Path) -> dict[str, tuple[str, ...]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    assignments: dict[str, tuple[str, ...]] = {}
    for row in report.get("rows") or ():
        state_hash = str(row.get("state_before_hash") or "")
        opponent = str(row.get("opponent_stratum") or "")
        if not state_hash or not opponent:
            raise ValueError("state_opponent_manifest_row_incomplete")
        names = (opponent,)
        previous = assignments.setdefault(state_hash, names)
        if previous != names:
            raise ValueError(
                f"state_opponent_manifest_conflict:{state_hash}"
            )
    if not assignments:
        raise ValueError("state_opponent_manifest_has_no_assignments")
    return assignments


def _sort_tasks_by_state_order(
    tasks: list[dict[str, Any]],
    *,
    state_order: tuple[str, ...],
) -> None:
    positions = {
        state_hash: index
        for index, state_hash in enumerate(state_order)
    }
    fallback = len(positions)
    tasks.sort(
        key=lambda task: (
            positions.get(
                str(task["trace"].get("state_before_hash") or ""),
                fallback,
            ),
            int(task["game"]["seed"]),
            int(task["trace"].get("sequence") or 0),
        )
    )


def _mean_field(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
    ]
    return round(mean(values), 6) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
