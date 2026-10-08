"""Audit text-anchored option detection across the curated screenshot corpus."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from vision.option_anchor_detector import detect_option_anchors
from vision.option_detector import detect_option_candidates


DEFAULT_ROOTS = (Path("data/screenshots"), Path("../data/screenshots"))
DEFAULT_EXTRAS = (
    Path("logs/device_soak_rc3_1v1_wang_hu_priority_20260812_151547/debug/overflow_stuck_20260812.png"),
)


def _counts(candidates) -> dict[str, int]:
    return {
        "compare_options": sum(item.region_name == "compare_options" for item in candidates),
        "chi_options": sum(item.region_name == "chi_options" for item in candidates),
    }


def _audit_one(path: Path) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        return {"path": str(path), "error": "image_read_failed"}
    anchors = detect_option_anchors(image)
    anchored = detect_option_candidates(image)
    legacy = detect_option_candidates(image, anchor_mode=False)
    return {
        "path": str(path),
        "anchors": [anchor.to_dict() for anchor in anchors],
        "anchored_counts": _counts(anchored),
        "legacy_counts": _counts(legacy),
    }


def audit_corpus(paths: list[Path], *, workers: int = 8) -> dict:
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(_audit_one, paths))

    readable = [row for row in rows if "error" not in row]
    errors = [row for row in rows if "error" in row]
    anchor_frames = [row for row in readable if row["anchors"]]
    anchor_without_columns = [
        row
        for row in anchor_frames
        if sum(row["anchored_counts"].values()) == 0
    ]
    legacy_only = [
        row
        for row in readable
        if sum(row["legacy_counts"].values()) > 0 and not row["anchors"]
    ]
    changed = [
        row
        for row in readable
        if row["anchored_counts"] != row["legacy_counts"]
    ]
    distributions: dict[str, int] = {}
    for row in anchor_frames:
        key = (
            f"compare={row['anchored_counts']['compare_options']},"
            f"chi={row['anchored_counts']['chi_options']}"
        )
        distributions[key] = distributions.get(key, 0) + 1
    return {
        "summary": {
            "files_scanned": len(paths),
            "files_read": len(readable),
            "read_errors": len(errors),
            "frames_with_text_anchors": len(anchor_frames),
            "anchor_frames_without_columns": len(anchor_without_columns),
            "legacy_candidate_frames_without_anchors": len(legacy_only),
            "frames_changed_from_legacy": len(changed),
            "anchored_layout_distribution": dict(sorted(distributions.items())),
        },
        "anchor_frames": anchor_frames,
        "anchor_frames_without_columns": anchor_without_columns,
        "legacy_candidate_frames_without_anchors": legacy_only,
        "read_errors": errors,
    }


def _collect_paths(roots: list[Path], extras: list[Path]) -> list[Path]:
    paths = [path for root in roots if root.exists() for path in root.glob("*.png")]
    paths.extend(path for path in extras if path.exists())
    return sorted(set(path.resolve() for path in paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    parser.add_argument("--extra", action="append", type=Path, dest="extras")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/option_anchor_corpus_audit.json"),
    )
    args = parser.parse_args()
    roots = args.roots or list(DEFAULT_ROOTS)
    extras = list(DEFAULT_EXTRAS) + (args.extras or [])
    paths = _collect_paths(roots, extras)
    report = audit_corpus(paths, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
