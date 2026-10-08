from pathlib import Path

from tools.build_frida_gadget_apk import build_gadget_apk


def test_build_gadget_apk_raises_without_frida_apk(monkeypatch, tmp_path):
    original_exists = Path.exists

    def fake_exists(self):
        if str(self).endswith("frida-apk.exe"):
            return False
        return original_exists(self)

    monkeypatch.setattr("tools.build_frida_gadget_apk.shutil.which", lambda _: None)
    monkeypatch.setattr(Path, "exists", fake_exists)

    try:
        build_gadget_apk(base_apk=tmp_path / "base.apk", output_apk=tmp_path / "out.apk", work_dir=tmp_path)
    except RuntimeError as exc:
        assert "frida-apk not found" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
