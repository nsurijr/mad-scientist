import http.client
import json
import shutil
import socket
import sys
from http.server import BaseHTTPRequestHandler

import pytest

from harness import socket_bridge
from harness.uds import ThreadingUnixHTTPServer, serve_in_thread, short_socket_dir


class Echo(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        data = json.dumps({"path": self.path, "got": json.loads(body)}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def unix_server():
    folder = short_socket_dir("t-")
    server = serve_in_thread(ThreadingUnixHTTPServer(folder / "s.sock", Echo))
    yield server
    server.shutdown()
    server.server_close()
    shutil.rmtree(folder, ignore_errors=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_socket_is_private_and_removed_on_close(unix_server):
    path = unix_server.path
    assert path.is_socket() and oct(path.stat().st_mode)[-3:] == "600"
    unix_server.shutdown()
    unix_server.server_close()
    assert not path.exists()


def test_a_loopback_port_reaches_the_unix_socket_with_keep_alive(unix_server):
    port = free_port()
    listener = socket_bridge.forward(port, str(unix_server.path))
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        for i in range(3):  # one connection, several requests, as AoD's HTTP clients do
            conn.request("POST", "/v1/tools/lookup", body=json.dumps({"i": i}),
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            assert r.status == 200 and json.loads(r.read()) == {"path": "/v1/tools/lookup", "got": {"i": i}}
        conn.close()
    finally:
        listener.close()


def test_the_bridge_runs_the_session_and_returns_its_status(unix_server):
    port = free_port()
    probe = ("import json, sys, urllib.request\n"
             f"r = urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:{port}/x', data=b'{{\"a\": 1}}'))\n"
             "sys.exit(7 if json.loads(r.read())['got'] == {'a': 1} else 1)\n")
    assert socket_bridge.main(["--forward", f"{port}={unix_server.path}", "--", sys.executable, "-c", probe]) == 7


def test_a_missing_socket_closes_the_connection(tmp_path):
    port = free_port()
    listener = socket_bridge.forward(port, str(tmp_path / "absent.sock"))
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        with pytest.raises((http.client.HTTPException, ConnectionError, OSError)):
            conn.request("GET", "/health")
            conn.getresponse()
    finally:
        listener.close()


def test_socket_paths_must_stay_short(tmp_path):
    with pytest.raises(ValueError, match="too long"):
        ThreadingUnixHTTPServer(tmp_path / ("x" * 120) / "s.sock", Echo)
