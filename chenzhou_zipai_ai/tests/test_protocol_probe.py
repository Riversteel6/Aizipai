from tools.protocol_probe import filter_protocol_lines


def test_filter_protocol_lines_finds_apk_fields():
    lines = filter_protocol_lines(
        "noise\nParseSocket playerholdcards=[1,2] canchi=true\nother zhuang=1 chairId=2\n",
        limit=10,
    )

    assert lines == [
        "ParseSocket playerholdcards=[1,2] canchi=true",
        "other zhuang=1 chairId=2",
    ]
