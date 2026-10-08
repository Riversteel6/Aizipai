from tools.pcapdroid_live_capture import _archive_existing_output


def test_archive_existing_output_keeps_previous_capture(tmp_path):
    output = tmp_path / "qs_packets_live_current.jsonl"
    output.write_text("old packet\n", encoding="utf-8")

    archived = _archive_existing_output(output, timestamp="20260601_221500")

    assert archived is not None
    assert archived.name == "qs_packets_live_current_20260601_221500.jsonl"
    assert archived.read_text(encoding="utf-8") == "old packet\n"
    assert output.read_text(encoding="utf-8") == "old packet\n"


def test_archive_existing_output_skips_empty_file(tmp_path):
    output = tmp_path / "empty.jsonl"
    output.write_text("", encoding="utf-8")

    assert _archive_existing_output(output, timestamp="20260601_221500") is None
