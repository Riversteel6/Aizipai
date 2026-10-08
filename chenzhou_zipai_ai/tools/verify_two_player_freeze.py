"""Verify that the frozen two-player strategy and its evidence have not drifted."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = WORKSPACE / "config" / "two_player_product_freeze_20260808.json"


def verify_freeze_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = WORKSPACE / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    protected = manifest["protected_scope"]
    files = _protected_files(protected, errors)
    records = [
        f"{file.relative_to(WORKSPACE).as_posix()}:{_file_hash(file)}"
        for file in files
    ]
    digest = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    if len(files) != int(protected["file_count"]):
        errors.append(
            f"protected_file_count_changed:{len(files)}!={protected['file_count']}"
        )
    if digest != protected["sha256"]:
        errors.append(f"protected_tree_hash_changed:{digest}!={protected['sha256']}")

    for relative, expected in manifest.get("entry_hashes", {}).items():
        file = _workspace_file(relative, errors)
        if file is not None:
            actual = _file_hash(file)
            if actual != expected:
                errors.append(f"entry_hash_changed:{relative}:{actual}!={expected}")

    for mode_files in manifest.get("evidence", {}).values():
        for relative in mode_files:
            _workspace_file(relative, errors)

    modes = {item["mode"]: item for item in manifest.get("modes", [])}
    for mode in ("1v1-no-wang", "1v1-wang"):
        if mode not in modes:
            errors.append(f"missing_frozen_mode:{mode}")

    return {
        "ok": not errors,
        "freeze_id": manifest.get("freeze_id"),
        "status": manifest.get("status"),
        "protected_file_count": len(files),
        "protected_tree_sha256": digest,
        "errors": errors,
    }


def _protected_files(scope: dict[str, Any], errors: list[str]) -> list[Path]:
    files: set[Path] = set()
    for item in scope.get("directories", []):
        directory = _workspace_file(item["path"], errors, require_file=False)
        if directory is None:
            continue
        iterator = (
            directory.rglob(item["pattern"])
            if item.get("recursive")
            else directory.glob(item["pattern"])
        )
        files.update(path for path in iterator if path.is_file())
    for relative in scope.get("files", []):
        file = _workspace_file(relative, errors)
        if file is not None:
            files.add(file)
    return sorted(files, key=lambda file: file.relative_to(WORKSPACE).as_posix())


def _workspace_file(
    relative: str,
    errors: list[str],
    *,
    require_file: bool = True,
) -> Path | None:
    resolved = (WORKSPACE / relative).resolve()
    try:
        resolved.relative_to(WORKSPACE.resolve())
    except ValueError:
        errors.append(f"path_outside_workspace:{relative}")
        return None
    expected = resolved.is_file() if require_file else resolved.is_dir()
    if not expected:
        errors.append(f"missing_path:{relative}")
        return None
    return resolved


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    result = verify_freeze_manifest(args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
