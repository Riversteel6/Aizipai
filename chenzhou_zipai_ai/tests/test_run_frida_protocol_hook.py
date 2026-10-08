from tools.run_frida_protocol_hook import resolve_pid


def test_resolve_pid_uses_first_pid(monkeypatch):
    monkeypatch.setattr("tools.run_frida_protocol_hook.run_adb", lambda *args, **kwargs: "123 456\n")

    assert resolve_pid("pkg", device_id="dev") == "123"
