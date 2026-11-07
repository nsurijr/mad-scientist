"""The hook host: a team's attack code in its own sandboxed process.

    python -m harness.hookhost serve /submission      # inside the sandbox, started by the runner

Attacker code never runs in the runner. The runner starts one host per eval, in the pipeline image
with no network and only the submission folder mounted (read-only; apptainer_command), and talks
to it over the host's stdin and stdout: each message is a 4-byte big-endian length and a JSON
object, with bytes as {"$b64": ...}. No pickle crosses the boundary.

    runner -> host   {"op": "load", "seed", "harness_seed", "shared"}
    host -> runner   {"op": "loaded", "hooks": [the hooks the attack overrides]}
    runner -> host   {"op": "call", "hook", "stage", "call_id", "args": {...}, "agent"}
                     (replace_agent's result is AgentPolicy.to_json() or null)
    host -> runner   {"op": "read", "name"}            (ctx.read during a call, any number of times)
    runner -> host   {"op": "data", "data": bytes} or {"op": "error", "error"}
    host -> runner   {"op": "result", "value"} or {"op": "error", "error": traceback}
    runner -> host   {"op": "exit"}

The host keeps the protocol on private copies of its stdin and stdout, so a hook's print() goes to
stderr (the hook log) instead of corrupting the stream.
"""

from __future__ import annotations

import base64
import importlib.util
import inspect
import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import IO, Any, Callable

from harness.attack import HOOKS, AgentPolicy, Attack, Ctx, call_rng

ATTACK_FILE = "attack.py"
CALL_TIMEOUT_S = 120  # per hook call, ctx.read included


class HookError(Exception):
    """The attack failed: it raised, timed out, returned the wrong type or broke the protocol."""


# -- framing -------------------------------------------------------------------------------------

def _encode(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return {"$b64": base64.b64encode(obj).decode()}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {"$b64"}:
            return base64.b64decode(obj["$b64"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def send(stream: IO[bytes], msg: dict[str, Any]) -> None:
    data = json.dumps(_encode(msg)).encode()
    stream.write(len(data).to_bytes(4, "big") + data)
    stream.flush()


def recv(stream: IO[bytes]) -> dict[str, Any] | None:
    """The next message, or None at the end of the stream."""
    header = stream.read(4)
    if len(header) < 4:
        return None
    n = int.from_bytes(header, "big")
    data = stream.read(n)
    if len(data) < n:
        return None
    return _decode(json.loads(data))


# -- inside the sandbox ----------------------------------------------------------------------------

def load_attack(submission: Path) -> Attack:
    """The one Attack subclass defined in the submission's attack.py, instantiated."""
    sys.path.insert(0, str(submission))  # so attack.py can import its own helper modules
    spec = importlib.util.spec_from_file_location("team_attack", submission / ATTACK_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [c for _, c in inspect.getmembers(module, inspect.isclass)
               if issubclass(c, Attack) and c is not Attack and c.__module__ == module.__name__]
    if len(classes) != 1:
        raise HookError(f"{ATTACK_FILE} must define exactly one Attack subclass, found {[c.__name__ for c in classes]}")
    return classes[0]()


def _check_result(hook: str, args: dict[str, Any], value: Any) -> None:
    if hook == "replace_agent" and value is not None and not isinstance(value, AgentPolicy):
        raise HookError(f"replace_agent must return an AgentPolicy or None, not {type(value).__name__}")
    if hook == "on_input" and not isinstance(value, bytes):
        raise HookError(f"on_input must return bytes, not {type(value).__name__}")
    if hook == "on_tool_result" and not isinstance(value, str):
        raise HookError(f"on_tool_result must return str, not {type(value).__name__}")
    if hook == "on_context":
        if not isinstance(value, dict) or set(value) != set(args["docs"]) or not all(isinstance(v, str) for v in value.values()):
            raise HookError(f"on_context must return a dict of str with the same names {sorted(args['docs'])}")


def serve(submission: Path) -> int:
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(2, 1)  # a hook's print() lands in the hook log
    sys.stdout = sys.stderr
    os.environ["HARNESS_SUBMISSION"] = str(submission)  # for read_asset

    hello = recv(proto_in)
    try:
        attack = load_attack(submission)
    except Exception:
        send(proto_out, {"op": "error", "error": traceback.format_exc(limit=5)})
        return 1
    send(proto_out, {"op": "loaded", "hooks": type(attack).implemented()})

    def reader(name: str) -> bytes:
        send(proto_out, {"op": "read", "name": name})
        reply = recv(proto_in)
        if reply is None or reply["op"] != "data":
            raise FileNotFoundError((reply or {}).get("error", f"{name} is not readable here"))
        return reply["data"]

    while (msg := recv(proto_in)) is not None and msg["op"] != "exit":
        ctx = Ctx(seed=hello["seed"], stage=msg["stage"], agent=msg.get("agent"), shared=hello.get("shared") or {},
                  rng=call_rng(hello["seed"], msg["stage"], msg["hook"], msg["call_id"], hello.get("harness_seed", 0)),
                  _reader=reader)
        args = msg["args"]
        try:
            if msg["hook"] == "replace_agent":
                value = attack.replace_agent(msg["stage"], ctx)
            elif msg["hook"] == "on_input":
                value = attack.on_input(args["name"], args["data"], ctx)
            elif msg["hook"] == "on_context":
                value = attack.on_context(msg["stage"], dict(args["docs"]), ctx)
            else:
                value = attack.on_tool_result(msg["stage"], args["tool"], args["args"], args["result"], ctx)
            _check_result(msg["hook"], args, value)
            if isinstance(value, AgentPolicy):
                value = value.to_json()
            send(proto_out, {"op": "result", "value": value})
        except Exception:
            send(proto_out, {"op": "error", "error": traceback.format_exc(limit=5)})
    return 0


# -- the runner's side -----------------------------------------------------------------------------

# The only harness files a sandbox sees: the SDK, never the verifier, scorer or gateway.
SDK_FILES = ("__init__.py", "attack.py", "hookhost.py", "replacement.py")


def sdk_binds(harness_dir: Path) -> list[str]:
    """apptainer --bind arguments mounting the SDK files read-only at /opt/sdk/harness."""
    return [arg for name in SDK_FILES for arg in ("--bind", f"{harness_dir / name}:/opt/sdk/harness/{name}:ro")]


def apptainer_command(image: Path, submission: Path, harness_dir: Path) -> list[str]:
    """The sandbox: no network, a fresh home, only the submission (read-only) and the SDK mounted."""
    return ["apptainer", "exec", "--containall", "--net", "--network", "none",
            "--bind", f"{submission}:/submission:ro", *sdk_binds(harness_dir),
            "--env", "PYTHONPATH=/opt/sdk", str(image),
            "python", "-m", "harness.hookhost", "serve", "/submission"]


class HookHost:
    """A running hook host. reader(stage, name) answers ctx.read; it decides what a stage may read."""

    def __init__(self, command: list[str], *, seed: str, reader: Callable[[str, str], bytes],
                 shared: dict[str, Any] | None = None, harness_seed: int = 0, log_path: Path | None = None,
                 timeout_s: float = CALL_TIMEOUT_S, env: dict[str, str] | None = None):
        self.reader, self.timeout_s = reader, timeout_s
        self._log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, env=env)
        self._messages: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._lock = threading.Lock()
        send(self.proc.stdin, {"op": "load", "seed": seed, "harness_seed": harness_seed, "shared": shared or {}})
        reply = self._next()
        if reply["op"] != "loaded":
            self.close()
            raise HookError(f"the attack did not load:\n{reply.get('error', reply)}")
        self.hooks: list[str] = [h for h in reply["hooks"] if h in HOOKS]

    def _pump(self) -> None:
        while (msg := recv(self.proc.stdout)) is not None:
            self._messages.put(msg)
        self._messages.put(None)

    def _next(self) -> dict[str, Any]:
        try:
            msg = self._messages.get(timeout=self.timeout_s)
        except queue.Empty:
            self.close()
            raise HookError(f"the hook host did not answer within {self.timeout_s:.0f} s") from None
        if msg is None:
            raise HookError(f"the hook host exited (status {self.proc.poll()})")
        return msg

    def call(self, hook: str, stage: str, call_id: str, args: dict[str, Any], agent: str | None = None) -> Any:
        with self._lock:
            send(self.proc.stdin, {"op": "call", "hook": hook, "stage": stage, "call_id": call_id, "args": args,
                                   "agent": agent})
            while True:
                msg = self._next()
                if msg["op"] == "read":
                    try:
                        send(self.proc.stdin, {"op": "data", "data": self.reader(stage, msg["name"])})
                    except (KeyError, FileNotFoundError) as e:
                        send(self.proc.stdin, {"op": "error", "error": f"not readable: {e}"})
                elif msg["op"] == "result":
                    return msg["value"]
                elif msg["op"] == "error":
                    raise HookError(f"{hook} at {stage} failed:\n{msg['error']}")
                else:
                    raise HookError(f"unexpected message from the hook host: {msg['op']}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                send(self.proc.stdin, {"op": "exit"})
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        if self._log is not subprocess.DEVNULL:
            self._log.close()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "serve":
        print("usage: python -m harness.hookhost serve <submission folder>", file=sys.stderr)
        return 2
    return serve(Path(argv[1]))


if __name__ == "__main__":
    sys.exit(main())
