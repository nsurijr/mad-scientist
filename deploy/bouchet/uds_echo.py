"""Answer one 'ping' with 'pong' on a Unix socket; used by sandbox_check.sbatch."""

import os
import socket
import sys

path = sys.argv[1]
server = socket.socket(socket.AF_UNIX)
server.bind(path)
os.chmod(path, 0o666)
server.listen(1)
conn, _ = server.accept()
conn.sendall(b"pong" if conn.recv(16) == b"ping" else b"?")
conn.close()
