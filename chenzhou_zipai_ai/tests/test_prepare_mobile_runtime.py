from pathlib import Path


def test_mobile_bridge_local_imports_are_packaged() -> None:
    android_root = Path(__file__).resolve().parents[1] / "android_app"
    bridge = (android_root / "mobile_runtime" / "mobile_bridge.py").read_text(encoding="utf-8")
    prepare = (android_root / "prepare_mobile_runtime.py").read_text(encoding="utf-8")

    required = {
        "tools.mobile_action_verifier": "mobile_action_verifier.py",
        "tools.confirmed_hand_ledger": "confirmed_hand_ledger.py",
    }
    for module, filename in required.items():
        assert f"from {module} import" in bridge
        assert f'"{filename}"' in prepare


def test_runtime_opponent_belief_model_is_packaged() -> None:
    project_root = Path(__file__).resolve().parents[1]
    prepare = (project_root / "android_app" / "prepare_mobile_runtime.py").read_text(
        encoding="utf-8"
    )
    model_name = "opponent_belief.json"

    assert (project_root / "models" / model_name).is_file()
    assert f'"card_classifier.json", "{model_name}"' in prepare
