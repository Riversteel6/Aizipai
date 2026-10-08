from tools.network_probe import _decode_address, _parse_tcp_table


def test_decode_ipv4_tcp_address():
    assert _decode_address("1390602F:1E61", table="tcp") == ("47.96.144.19", 7777)


def test_parse_tcp_table_filters_uid():
    text = """  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
  0: 0100FF0A:A778 1390602F:1E61 01 00000000:00000000 00:00000000 00000000 10443 0 2749851
  1: 0100FF0A:A779 1390602F:01BB 01 00000000:00000000 00:00000000 00000000 10000 0 2749852
"""

    rows = _parse_tcp_table(text, table="tcp", uid=10443)

    assert rows == [
        {
            "table": "tcp",
            "uid": 10443,
            "state": "ESTABLISHED",
            "local_ip": "10.255.0.1",
            "local_port": 42872,
            "remote_ip": "47.96.144.19",
            "remote_port": 7777,
            "inode": "2749851",
        }
    ]
