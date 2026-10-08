"""Freeze-manifest contract tests."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from ai.frozen_two_player_strategy import FROZEN_CANDIDATE, FROZEN_RELEASES
from tools.verify_two_player_freeze import DEFAULT_MANIFEST, verify_freeze_manifest


def test_two_player_product_freeze_has_no_code_or_evidence_drift() -> None:
    result = verify_freeze_manifest()

    assert result["ok"], result["errors"]
    assert result["status"] == "v9_r73_pending_card_multiscale_installed_device_gate_pass"


def test_manifest_and_runtime_release_ids_are_the_same() -> None:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    modes = {item["mode"]: item for item in manifest["modes"]}

    assert manifest["candidate"] == FROZEN_CANDIDATE
    assert modes["1v1-no-wang"]["release_id"] == FROZEN_RELEASES[False]
    assert modes["1v1-wang"]["release_id"] == FROZEN_RELEASES[True]
    assert manifest["device_gate"]["status"] == "r73_pending_card_multiscale_installed_device_gate_pass"
    assert manifest["packaging"]["authorized"] is True
    assert manifest["packaging"]["installation_authorized"] is True
    assert manifest["packaging"]["version_name"] == "1.0-r73-v9-pending-card-multiscale"
    assert manifest["packaging"]["version_code"] == 73


def test_android_version_label_matches_packaged_version_name() -> None:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    strings_path = (
        DEFAULT_MANIFEST.parents[1]
        / "chenzhou_zipai_ai"
        / "android_app"
        / "app"
        / "src"
        / "main"
        / "res"
        / "values"
        / "strings.xml"
    )
    resources = ET.parse(strings_path).getroot()
    labels = {item.attrib["name"]: item.text for item in resources.findall("string")}

    assert labels["app_version_label"] == (
        f"版本：{manifest['packaging']['version_name']}"
    )
