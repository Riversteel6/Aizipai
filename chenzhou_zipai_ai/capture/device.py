"""Device connection and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from capture.adb_capture import run_adb
except ModuleNotFoundError:
    from chenzhou_zipai_ai.capture.adb_capture import run_adb


@dataclass(frozen=True)
class Device:
    serial: str
    status: str


def list_devices() -> list[Device]:
    output = run_adb(["devices"])
    devices: list[Device] = []
    for line in output.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            devices.append(Device(serial=parts[0], status=parts[1]))
    return devices


def get_connected_device() -> Device:
    devices = [device for device in list_devices() if device.status == "device"]
    if not devices:
        raise RuntimeError("No authorized Android device found. Check USB debugging authorization.")
    if len(devices) > 1:
        raise RuntimeError("Multiple devices connected. Disconnect extras or add serial support.")
    return devices[0]


def get_screen_size(device_id: str | None = None) -> tuple[int, int]:
    output = run_adb(["shell", "wm", "size"], device_id=device_id)
    marker = "Physical size:"
    for line in output.splitlines():
        if marker in line:
            width, height = line.split(marker, 1)[1].strip().split("x", 1)
            return int(width), int(height)
    raise RuntimeError(f"Could not parse screen size from: {output!r}")


def main() -> None:
    device = get_connected_device()
    width, height = get_screen_size(device.serial)
    print(f"{device.serial}\t{device.status}\t{width}x{height}")


if __name__ == "__main__":
    main()
