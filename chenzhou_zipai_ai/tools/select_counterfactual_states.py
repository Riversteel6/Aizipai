"""Select deterministic opponent-stratified states for counterfactual calibration."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ai.simulation_trace import public_state_identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        default="professional_parallel_dual_validated_candidate_v1",
    )
    parser.add_argument("--per-opponent", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--exclude-state-hash", action="append", default=[])
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
    )
    args = parser.parse_args()

    excluded_hashes = {
        str(value)
        for value in args.exclude_state_hash
        if str(value)
    }
    excluded_hashes.update(
        load_excluded_state_hashes(args.exclude_report)
    )
    candidates = collect_discard_states(
        _read_jsonl(args.trace_input),
        policy=args.policy,
        excluded_hashes=excluded_hashes,
    )
    selected = select_stratified_states(
        candidates,
        per_stratum=max(1, args.per_opponent),
        seed=int(args.seed),
    )
    counts: dict[str, int] = defaultdict(int)
    for row in selected:
        counts[str(row["opponent_stratum"])] += 1
    report = {
        "ok": bool(selected)
        and all(
            count == max(1, args.per_opponent)
            for count in counts.values()
        ),
        "schema_version": "counterfactual-state-selection-v1",
        "trace_input": str(args.trace_input),
        "policy": args.policy,
        "seed": int(args.seed),
        "per_opponent": max(1, args.per_opponent),
        "excluded_state_hashes": len(excluded_hashes),
        "available_states": len(candidates),
        "selected_states": len(selected),
        "selected_by_opponent": dict(sorted(counts.items())),
        "rows": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "available_states": report["available_states"],
                "selected_states": report["selected_states"],
                "selected_by_opponent": report[
                    "selected_by_opponent"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


def load_excluded_state_hashes(paths: Sequence[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report.get("rows") or ()
        for row in rows:
            state_hash = str(row.get("state_before_hash") or "")
            if state_hash:
                hashes.add(state_hash)
    return hashes


def collect_discard_states(
    games: Sequence[Mapping[str, Any]] | Iterator[Mapping[str, Any]],
    *,
    policy: str,
    excluded_hashes: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for game in games:
        opponents = tuple(str(value) for value in game.get("opponents") or ())
        stratum = ",".join(opponents)
        for trace in game.get("decision_trace") or ():
            if str(trace.get("phase") or "") != "discard":
                continue
            if policy and str(trace.get("policy") or "") != policy:
                continue
            state_hash = str(trace.get("state_before_hash") or "")
            public_state = trace.get("public_state")
            if (
                not state_hash
                or state_hash in excluded_hashes
                or state_hash in seen
                or not isinstance(public_state, Mapping)
            ):
                continue
            seen.add(state_hash)
            rows.append(
                {
                    "state_before_hash": state_hash,
                    "public_view_id": public_state_identity(public_state),
                    "opponent_stratum": stratum,
                    "game_seed": int(game.get("seed") or 0),
                    "candidate_seat": int(game.get("candidate_seat") or 0),
                    "dealer": int(game.get("dealer") or 0),
                    "trace_sequence": int(
                        trace.get("trace_sequence") or 0
                    ),
                    "turn": int(trace.get("turn") or 0),
                    "selected_key": str(trace.get("selected_key") or ""),
                    "legal_action_count": int(
                        trace.get("legal_action_count") or 0
                    ),
                }
            )
    return rows


def select_stratified_states(
    candidates: Sequence[Mapping[str, Any]],
    *,
    per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["opponent_stratum"])].append(dict(candidate))
    selected: list[dict[str, Any]] = []
    for stratum, rows in sorted(grouped.items()):
        if len(rows) < per_stratum:
            raise ValueError(
                f"counterfactual_stratum_too_small:{stratum}:{len(rows)}"
            )
        stable_rows = sorted(
            rows,
            key=lambda row: str(row["state_before_hash"]),
        )
        digest = hashlib.sha256(stratum.encode("utf-8")).digest()
        stratum_seed = seed + int.from_bytes(digest[:8], "big")
        sampled = random.Random(stratum_seed).sample(
            stable_rows,
            per_stratum,
        )
        selected.extend(
            sorted(
                sampled,
                key=lambda row: (
                    int(row["game_seed"]),
                    int(row["trace_sequence"]),
                    str(row["state_before_hash"]),
                ),
            )
        )
    return selected


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
