"""Inside a stage container with no network: loopback ports forwarded to bind-mounted Unix sockets,
then the AoD session as a child process.

    python /opt/harness_bridge/socket_bridge.py --forward 18080=/run/harness/gw/gateway.sock \
        [--forward 18081=/run/harness/hooks/hooks.sock] -- python -m projects.default ...

The container's network namespace has only its own loopback, so AoD and the code its agents run can
reach the gateway and the hook relay, and nothing else. AoD keeps its HTTP clients and only its URLs
change (http://127.0.0.1:18080/v1), which are not part of any cache key. The exit status is AoD's.
Standard library only: the bridge runs in the pipeline image, which has no harness package.
"""

from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import sys
import threading

CHUNK = 1 << 16


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while data := src.recv(CHUNK):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _connection(client: socket.socket, path: str) -> None:
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(path)
    except OSError:
        client.close()
        upstream.close()
        return
    back = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
    back.start()
    _pump(client, upstream)
    back.join()
    client.close()
    upstream.close()


def forward(port: int, path: str, host: str = "127.0.0.1") -> socket.socket:
    """Listens on host:port now; each connection is relayed to the Unix socket at path."""
    server = socket.create_server((host, port))

    def accept() -> None:
        while True:
            try:
                client, _ = server.accept()
            except OSError:
                return
            threading.Thread(target=_connection, args=(client, path), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        print("usage: socket_bridge.py --forward PORT=SOCKET ... -- COMMAND ...", file=sys.stderr)
        return 2
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--forward", action="append", default=[], metavar="PORT=SOCKET")
    args = ap.parse_args(argv[:split])
    command = argv[split + 1:]
    servers = []
    for spec in args.forward:
        port, _, path = spec.partition("=")
        servers.append(forward(int(port), path))  # bound before the session starts
    child = subprocess.Popen(command)
    for sig in (signal.SIGTERM, signal.SIGINT):  # the stage time limit reaches the session too
        signal.signal(sig, lambda signum, frame: child.send_signal(signum))
    try:
        return child.wait()
    finally:
        for s in servers:
            s.close()


if __name__ == "__main__":
    sys.exit(main())
