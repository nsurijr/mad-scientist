"""HTTP servers on Unix sockets, for stage containers with no network.

A stage container runs with --network none. The gateway and the runner's hook relay
also listen on Unix sockets, whose folders are bind-mounted into the container; inside it,
harness/socket_bridge.py makes them reachable on the container's own loopback.
"""

from __future__ import annotations

import os
import socketserver
import tempfile
import threading
from pathlib import Path

MAX_PATH = 100  # sun_path holds 108 bytes on Linux, 104 on macOS


class ThreadingUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """http.server's handlers on a Unix socket, readable and writable by this user only."""

    daemon_threads = True

    def __init__(self, path: Path, handler: type):
        path = Path(path)
        if len(str(path)) > MAX_PATH:
            raise ValueError(f"Unix socket path too long ({len(str(path))} > {MAX_PATH} characters): {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_socket():
            path.unlink()
        self.path = path
        super().__init__(str(path), handler)
        os.chmod(path, 0o600)

    def get_request(self):
        conn, _ = self.socket.accept()
        return conn, ("unix", 0)  # http.server expects a (host, port) client address

    def server_close(self) -> None:
        super().server_close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def serve_in_thread(server: socketserver.BaseServer) -> socketserver.BaseServer:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def short_socket_dir(prefix: str) -> Path:
    """A fresh folder for a socket, under /tmp where paths stay short (node-local on Bouchet)."""
    return Path(tempfile.mkdtemp(prefix=prefix, dir="/tmp" if os.path.isdir("/tmp") else None))
