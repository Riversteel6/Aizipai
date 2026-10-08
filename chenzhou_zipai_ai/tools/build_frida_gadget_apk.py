"""Build a Frida Gadget-injected APK without installing it."""

from __future__ import annotations

import argparse
import lzma
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def build_gadget_apk(
    *,
    base_apk: str | Path,
    output_apk: str | Path,
    version: str = "17.10.0",
    work_dir: str | Path = WORKSPACE / "dist/frida_gadget",
) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    frida_apk = shutil.which("frida-apk")
    if not frida_apk:
        scripts = Path.home() / "AppData/Roaming/Python/Python314/Scripts/frida-apk.exe"
        frida_apk = str(scripts) if scripts.exists() else ""
    if not frida_apk:
        raise RuntimeError("frida-apk not found; run: python -m pip install --user frida-tools")

    gadget_so = work_dir / "libfrida-gadget.so"
    if not gadget_so.exists():
        archive = work_dir / f"frida-gadget-{version}-android-arm64.so.xz"
        if not archive.exists():
            url = f"https://github.com/frida/frida/releases/download/{version}/frida-gadget-{version}-android-arm64.so.xz"
            urllib.request.urlretrieve(url, archive)
        with lzma.open(archive, "rb") as src, gadget_so.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    output_apk = Path(output_apk)
    output_apk.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([frida_apk, "-g", str(gadget_so), "-o", str(output_apk), str(base_apk)], check=True)
    return output_apk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-apk", default=r"D:\Weixin\xwechat_files\wxid_17pkya5ctd4h22_77c7\msg\file\2026-06\base.apk")
    parser.add_argument("--output-apk", default=str(WORKSPACE / "dist/base_frida_gadget_unsigned.apk"))
    parser.add_argument("--version", default="17.10.0")
    args = parser.parse_args()
    output = build_gadget_apk(base_apk=args.base_apk, output_apk=args.output_apk, version=args.version)
    print(f"GADGET_APK={output}")


if __name__ == "__main__":
    main()
