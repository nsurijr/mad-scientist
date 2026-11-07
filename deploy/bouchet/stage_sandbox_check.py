"""Checks a stage container with no network, built exactly as harness/runner.py builds it.

    python deploy/bouchet/stage_sandbox_check.py <pipeline.sif>

A gateway (replay mode, empty store) listens on a Unix socket and on a loopback port of the node.
The container runs a probe instead of the AoD session. It must reach the gateway through the bridge,
and nothing else: not the node's loopback port, not the internet. It also reports whether nested
user namespaces work inside the container, which could cut agent code off from the gateway too.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from harness.cas import Policy, Store
from harness.gateway import Gateway, Upstream, make_server, make_unix_server
from harness.runner import STAGES, Settings, stage_command
from harness.uds import serve_in_thread, short_socket_dir

PROBE = r"""
import json, os, shutil, socket, subprocess, urllib.request
out = {}
try:
    with urllib.request.urlopen(os.environ["OPENAI_BASE_URL"].rsplit("/v1", 1)[0] + "/health", timeout=10) as r:
        out["gateway_through_bridge"] = json.loads(r.read()).get("status") == "ok"
except Exception as e:
    out["gateway_through_bridge"] = f"failed: {e}"
def reach(host, port):
    try:
        socket.create_connection((host, port), timeout=5).close()
        return True
    except OSError:
        return False
out["node_loopback_port"] = reach("127.0.0.1", int(os.environ["PROBE_NODE_PORT"]))
out["internet_443"] = reach("140.82.114.4", 443)
out["interfaces"] = sorted(os.listdir("/sys/class/net")) if os.path.isdir("/sys/class/net") else None
unshare = shutil.which("unshare")
out["unshare_binary"] = unshare
if unshare:
    r = subprocess.run([unshare, "--user", "--map-root-user", "--mount", "--net", "true"], capture_output=True, text=True)
    out["nested_userns"] = r.returncode == 0 or r.stderr.strip()[:200]
out["os_unshare"] = hasattr(os, "unshare")
print("PROBE " + json.dumps(out))
"""


def main() -> int:
    image = Path(sys.argv[1])
    work = Path(tempfile.mkdtemp(prefix="sbx-"))
    sockets = short_socket_dir("sbx-gw-")
    gw = Gateway(Store(work / "cas", Policy.replay_only("empty")), "replay", upstreams={"*": Upstream(None, "r", {})},
                 sampling={}, container_digest="c" * 64, harness_seed=0)
    tcp = make_server(gw)
    threading.Thread(target=tcp.serve_forever, daemon=True).start()
    unix = serve_in_thread(make_unix_server(gw, sockets / "gateway.sock"))
    stage_dir = work / "S1"
    for sub in ("out", "trace", "task", "in"):
        (stage_dir / sub).mkdir(parents=True)
    (work / "data").mkdir()
    settings = Settings(base_url=f"http://127.0.0.1:{tcp.server_address[1]}/v1", image=image,
                        gateway_socket=sockets / "gateway.sock", stage_timeout_s=300)
    cmd, env = stage_command(STAGES[0], stage_dir, work / "data", settings, program=["python", "-c", PROBE])
    env["APPTAINERENV_PROBE_NODE_PORT"] = str(tcp.server_address[1])
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE ")), None)
    unix.shutdown()
    unix.server_close()
    tcp.shutdown()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(sockets, ignore_errors=True)
    if line is None:
        print(f"FAIL: no probe output (exit {r.returncode})\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return 1
    out = json.loads(line[6:])
    checks = [("the gateway answers through the bridge", out["gateway_through_bridge"] is True),
              ("the node's loopback port is unreachable", out["node_loopback_port"] is False),
              ("the internet is unreachable", out["internet_443"] is False)]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    print(f"info: interfaces {out['interfaces']}; unshare {out['unshare_binary']}; "
          f"nested user namespace {out.get('nested_userns')}; os.unshare {out['os_unshare']}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
