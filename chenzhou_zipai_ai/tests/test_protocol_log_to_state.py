from tools.protocol_log_to_state import ProtocolLogTail, latest_protocol_state_from_log


def test_latest_protocol_state_from_log(tmp_path):
    log = tmp_path / "hook.log"
    log.write_text(
        "noise\n"
        "{\"type\":\"recv\",\"text\":\"playerholdcards=[1,2] canchi=true\"}\n"
        "{\"type\":\"recv\",\"text\":\"playerholdcards=[3,4] canhu=true\"}\n",
        encoding="utf-8",
    )

    state = latest_protocol_state_from_log(log)

    assert state["hand"] == ["三", "四"]
    assert [item["type"] for item in state["legal_actions"]] == ["hu"]


def test_latest_qs_log_merges_hand_snapshot_with_latest_action_prompt(tmp_path):
    log = tmp_path / "packets.jsonl"
    hand = _packet(1011, _list("1s", "2b", "Qh"))
    prompt = _packet(1014, _i(2) + _s("7s") + _i(1) + _i(0) + _i(1) + _i(0) + _s("a") + _i(0) + _i(0) + _i(1))
    log.write_text(
        _row(hand, src="47.114.97.95:24000", dst="10.215.173.1:51626") + "\n"
        + _row(prompt, src="47.114.97.95:24000", dst="10.215.173.1:51626") + "\n",
        encoding="utf-8",
    )

    state = latest_protocol_state_from_log(log)

    assert state["hand"] == ["一", "贰", "王"]
    assert state["pending_card"] == "七"
    assert [item["type"] for item in state["legal_actions"]] == ["chi", "pass"]
    assert state["metadata"]["merged_hand_from_qs_cmd"] == 1011


def test_latest_qs_log_ignores_outbound_phone_requests(tmp_path):
    log = tmp_path / "packets.jsonl"
    hand = _packet(1011, _list("1s", "2b"))
    outbound_prompt = _packet(1014, _i(2) + _s("9s") + _i(1) + _i(0) + _i(0) + _i(1) + _s("bad") + _i(0) + _i(0) + _i(0))
    log.write_text(
        _row(hand, src="47.114.97.95:24000", dst="10.215.173.1:51626") + "\n"
        + _row(outbound_prompt, src="10.215.173.1:51626", dst="47.114.97.95:24000") + "\n",
        encoding="utf-8",
    )

    state = latest_protocol_state_from_log(log)

    assert state["hand"] == ["一", "贰"]
    assert state["pending_card"] == ""
    assert state["legal_actions"] == []


def test_latest_qs_log_folds_self_draw_and_discard_into_hand(tmp_path):
    log = tmp_path / "packets.jsonl"
    hand = _packet(1011, _list("1s", "2s"))
    prompt = _packet(1014, _i(2) + _s("Qh") + _i(1) + _i(0) + _i(0) + _i(0) + _s("a") + _i(0) + _i(0) + _i(1))
    draw = _packet(1013, _i(2) + _i(40) + _s("Qh") + _i(1) + _i(1) + _i(0) + _i(0))
    discard = _packet(1012, _i(0) + _i(2) + _i(0) + _i(11) + _list("1s") + _i(0) + _i(0))
    log.write_text(
        "\n".join(
            [
                _row(hand, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(prompt, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(draw, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(discard, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = latest_protocol_state_from_log(log)

    assert state["hand"] == ["二", "王"]
    assert state["expected_count"] == 2
    assert state["metadata"]["hand_source"] == "folded_qs_log"


def test_latest_qs_log_marks_self_draw_as_discard_turn(tmp_path):
    log = tmp_path / "packets.jsonl"
    login = _packet(1001, _i(3) + _i(0) + _i(2) + _s("3") + _i(3))
    hand = _packet(1011, _list("1s", "2s"))
    draw = _packet(1013, _i(2) + _i(40) + _s("Qh") + _i(1) + _i(1) + _i(0) + _i(0))
    log.write_text(
        "\n".join(
            [
                _row(login, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(hand, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(draw, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = latest_protocol_state_from_log(log)

    assert state["hand"] == ["一", "二", "王"]
    assert state["legal_actions"] == [{"type": "discard", "source": "qs_own_draw"}]
    assert state["metadata"]["discard_turn_source"] == "qs_own_draw"


def test_latest_qs_log_allows_self_prompt_after_opponent_draw(tmp_path):
    log = tmp_path / "packets.jsonl"
    login = _packet(1001, _i(3) + _i(0) + _i(1) + _s("3") + _i(3))
    hand = _packet(1011, _list("1s", "2s"))
    opponent_draw = _packet(1013, _i(2) + _i(40) + _s("Ts") + _i(1) + _i(1) + _i(0) + _i(0))
    prompt = _packet(1014, _i(1) + _s("Ts") + _i(1) + _i(0) + _i(1) + _i(0) + _s("a") + _i(0) + _i(0) + _i(1))
    log.write_text(
        "\n".join(
            [
                _row(login, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(hand, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(opponent_draw, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(prompt, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = latest_protocol_state_from_log(log)

    assert state["hand"] == ["一", "二"]
    assert state["pending_card"] == "十"
    assert [item["type"] for item in state["legal_actions"]] == ["chi", "pass"]
    assert state["metadata"]["self_chair_id"] == 1
    assert state["metadata"]["last_draw_chair_id"] == 2
    assert state["metadata"]["opponent_priority_pending"] is False
    assert state["metadata"]["response_source"] == "qs_self_prompt"
    assert state["metadata"]["response_source_chair_id"] == 2


def test_latest_qs_log_allows_response_after_opponent_discard(tmp_path):
    log = tmp_path / "packets.jsonl"
    login = _packet(1001, _i(3) + _i(0) + _i(2) + _s("3") + _i(3))
    hand = _packet(1011, _list("6b", "8b"))
    opponent_draw = _packet(1013, _i(1) + _i(40) + _s("8b") + _i(1) + _i(1) + _i(0) + _i(0))
    opponent_discard = _packet(1012, _i(0) + _i(1) + _i(0) + _i(11) + _list("7b") + _i(0) + _i(0))
    prompt = _packet(1014, _i(2) + _s("7b") + _i(0) + _i(0) + _i(1) + _i(0) + _s("a") + _i(0) + _i(0) + _i(1))
    log.write_text(
        "\n".join(
            [
                _row(login, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(hand, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(opponent_draw, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(opponent_discard, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
                _row(prompt, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = latest_protocol_state_from_log(log)

    assert state["pending_card"] == "柒"
    assert [item["type"] for item in state["legal_actions"]] == ["chi", "pass"]
    assert state["metadata"]["opponent_priority_pending"] is False
    assert state["metadata"]["response_source"] == "qs_opponent_discard"
    assert state["metadata"]["response_source_chair_id"] == 1


def test_protocol_log_tail_reads_only_appended_rows(tmp_path):
    log = tmp_path / "packets.jsonl"
    log.write_text("", encoding="utf-8")
    tail = ProtocolLogTail(log)

    empty = tail.read_latest()
    assert empty["metadata"]["protocol_tail"]["rows_read"] == 0

    hand = _packet(1011, _list("1s", "2b", "Qh"))
    log.write_text(_row(hand, src="47.114.97.95:24000", dst="10.215.173.1:51626") + "\n", encoding="utf-8")
    state = tail.read_latest()

    assert state["hand"] == ["一", "贰", "王"]
    assert state["metadata"]["protocol_tail"]["rows_read"] == 1
    assert state["metadata"]["protocol_tail"]["state_updated"] is True

    prompt = _packet(1014, _i(2) + _s("7s") + _i(1) + _i(0) + _i(1) + _i(0) + _s("a") + _i(0) + _i(0) + _i(1))
    with log.open("a", encoding="utf-8") as handle:
        handle.write(_row(prompt, src="47.114.97.95:24000", dst="10.215.173.1:51626") + "\n")
    state = tail.read_latest()

    assert state["hand"] == ["一", "贰", "王"]
    assert state["pending_card"] == "七"
    assert [item["type"] for item in state["legal_actions"]] == ["chi", "pass"]
    assert state["metadata"]["protocol_tail"]["rows_read"] == 1

    stale = tail.read_latest()
    assert stale["pending_card"] == "七"
    assert stale["metadata"]["protocol_tail"]["rows_read"] == 0
    assert stale["metadata"]["protocol_tail"]["state_updated"] is False


def test_protocol_log_tail_resets_when_capture_file_is_rewritten(tmp_path):
    log = tmp_path / "packets.jsonl"
    hand = _packet(1011, _list("1s", "2b"))
    log.write_text(_row(hand, src="47.114.97.95:24000", dst="10.215.173.1:51626") + "\n", encoding="utf-8")
    tail = ProtocolLogTail(log)

    assert tail.read_latest()["hand"] == ["一", "贰"]

    prompt = _packet(1014, _i(2) + _s("7s") + _i(1) + _i(0) + _i(1) + _i(0) + _s("a") + _i(0) + _i(0) + _i(1))
    rewritten = "\n".join(
        [
            _row(prompt, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
            _row(prompt, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
            _row(prompt, src="47.114.97.95:24000", dst="10.215.173.1:51626"),
        ]
    )
    log.write_text(rewritten + "\n", encoding="utf-8")
    state = tail.read_latest()

    assert state["pending_card"] == "七"
    assert state["metadata"]["protocol_tail"]["rows_read"] == 3


def _row(packet: bytes, *, src: str, dst: str) -> str:
    return '{"type":"qs_packet","packet_hex":"' + packet.hex() + '","tcp":{"src":"' + src + '","dst":"' + dst + '"}}'


def _packet(cmd: int, body: bytes) -> bytes:
    return b"QS" + len(body).to_bytes(2, "little", signed=True) + cmd.to_bytes(2, "little", signed=True) + body


def _i(value: int) -> bytes:
    return value.to_bytes(4, "little", signed=True)


def _s(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return _i(len(raw)) + raw


def _list(*values: str) -> bytes:
    return _i(len(values)) + b"".join(_s(value) for value in values)
