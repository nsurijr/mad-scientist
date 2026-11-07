"""Record-and-replay gateway between the pipeline's agents and the model servers.

    python -m harness.gateway serve --cas cas --namespace golden --mode record \
        --upstream http://127.0.0.1:8000/v1 --port 9000 --log runs/<batch>/gateway.jsonl ...
    python -m harness.gateway serve --cas cas --namespace golden --mode replay --port 9000 ...
    python -m harness.gateway serve --cas cas --namespace golden --mode eval --secret <key file> \
        --upstreams upstreams.json --port 9000 ...
    python -m harness.gateway seal --cas cas --namespace golden
    python -m harness.gateway summary runs/<batch>/gateway.jsonl

Agents call it like an OpenAI server (OPENAI_BASE_URL=http://127.0.0.1:9000/v1), and AoD's
execute_python asks it before running code (AOD_TOOL_CACHE_URL=http://127.0.0.1:9000/v1/tools,
patch P9). With --socket it also listens on a Unix socket, the only way in for stage containers
with no network (harness/socket_bridge.py); the runner's eval control still uses the port.

    POST /v1/chat/completions      the recorded reply, byte for byte, or a live one
    POST /v1/tools/lookup          {tool, tool_version, args, manifest}: recorded result and files
    POST /v1/tools/record          {tool, tool_version, args, manifest, result, files, deleted}
    POST /v1/evals/<id>/diverge    {"at": ...}: the eval's first hook change; strict mode ends (control)
    GET  /v1/evals/<id>            the eval's state: strict, calls, live tokens, first live call (control)
    GET  /health

Modes:
- record: look up the namespace first. On a miss, call the upstream (LLM) or accept the client's
  live execution (tools), and store the result. Re-recording an existing namespace only adds.
- replay: look up only, with no upstream. A miss is HTTP 409 with error type "replay_miss" and a
  diff against the nearest recorded request; AoD turns it into HarnessHalt, which ends the session.
- eval: every request carries an eval token (harness/evaltoken.py) as its bearer token. The token
  names the eval, its namespace, its budget and its models; --namespace names golden.
  - Lookup is golden, then the eval's namespace, then live; writes go to the eval's namespace only.
  - Each eval starts strict: a miss is HTTP 409 "replay_miss" with a diff against the nearest
    golden request, because before the attack changes anything every call must match golden. The
    runner ends strict mode at the first hook change (POST .../diverge, control token only), so
    neither agents nor attackers can switch it.
  - After that, misses run live. A preview token (live false) stops at the first one instead,
    with HTTP 409 "preview_stop" and the same diff.
  - Budgets: past the token's wall time or live-token cap, every call is HTTP 429
    "budget_exceeded", which halts AoD and makes the run invalid. A model outside the token's
    allowed_models is HTTP 403, which keeps the monitor's model away from teams.

Upstreams: each model name maps to a base URL, the name the upstream serves it under, an API key
from the environment, and what goes into the key about it: the model revision and what the server
adds to the prompt itself (gpt-oss's pinned system date). So recordings from Bouchet's vLLM never
stand in for the event API's replies.

Keys (harness/keys.py): the gateway enforces the sampling values (temperature, top_p, seed,
max_tokens, reasoning effort), so the key holds what the server got. Every call is logged as one
JSON line: eval tag (X-Eval header, "<seed>/<stage>"), kind, key, outcome (hit, live, miss), tokens,
and in eval mode the eval id, team and role.

Standard library only for the server, rather than FastAPI.
"""

from __future__ import annotations

import argparse
import base64
import faulthandler
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from harness.cas import Policy, Store
from harness.evaltoken import TokenError, read_secret, verify
from harness.keys import canonical_json, llm_key, manifest_hash, tool_key
from harness.missdiff import miss_report
from harness.uds import ThreadingUnixHTTPServer, serve_in_thread

MODES = ("record", "replay", "eval")
ANY_MODEL = "*"
_EVAL_PATH = re.compile(r"^/v1/evals/([A-Za-z0-9_.-]{1,128})(/diverge)?$")


def _error(kind: str, message: str, **extra: Any) -> bytes:
    return json.dumps({"error": {"type": kind, "message": message, **extra}}).encode()


@dataclass(frozen=True)
class Upstream:
    """Where one model is served, and what about it goes into the key."""

    base_url: str | None  # None: nothing to forward to (replay)
    model_rev: str
    server: dict[str, Any]  # what the server adds to the prompt itself: gpt-oss's pinned system date
    served_name: str | None = None  # the upstream's own name for the model, if it differs
    api_key: str | None = field(default=None, repr=False)
    # A shared API (iLaaS) rate-limits and fails now and then; AoD itself never retries.
    retries: int = 0  # extra attempts after HTTP 429 or 5xx, or no connection, with growing waits
    max_concurrency: int | None = None  # requests in flight to this upstream at once
    drop_params: tuple[str, ...] = ()  # request fields it rejects; still in the key, never sent


RETRY_STATUS = (429, 500, 502, 503, 504)
RETRY_WAIT_S, RETRY_MAX_WAIT_S = 2.0, 120.0


def load_upstreams(path: Path) -> dict[str, Upstream]:
    """{model name: Upstream} from a JSON file such as

        {"gpt-oss-120b": {"base_url": "http://127.0.0.1:8000/v1", "model_rev": "b5c939de",
                          "server": {"system_start_date": "2026-11-23"},
                          "served_name": "openai/gpt-oss-120b", "api_key_env": "EVENT_API_KEY",
                          "retries": 6, "max_concurrency": 16, "drop_params": ["seed"]}}

    The name "*" serves every model not listed. API keys are read from the environment here, so a
    missing one fails at start-up rather than at the first live call."""
    upstreams = {}
    for name, spec in json.loads(Path(path).read_text(encoding="utf-8")).items():
        key_env = spec.get("api_key_env")
        if key_env and key_env not in os.environ:
            raise SystemExit(f"{path}: {name}: environment variable {key_env} is not set")
        upstreams[name] = Upstream(base_url=spec.get("base_url"), model_rev=spec["model_rev"],
                                   server=spec.get("server", {}), served_name=spec.get("served_name"),
                                   api_key=os.environ[key_env] if key_env else None,
                                   retries=int(spec.get("retries", 0)), max_concurrency=spec.get("max_concurrency"),
                                   drop_params=tuple(spec.get("drop_params", ())))
    return upstreams


@dataclass
class EvalState:
    """What the gateway knows about one eval, shared by all of its tokens (pipeline and replacement)."""

    eval_id: str
    team: str
    seed: str
    namespace: str
    strict: bool = True
    diverged_at: Any = None
    live_tokens: int = 0
    calls: Counter = field(default_factory=Counter)
    first_live_call: dict[str, Any] | None = None
    stopped: dict[str, Any] | None = None  # the strict miss or preview stop that ended it

    def summary(self) -> dict[str, Any]:
        return {"eval_id": self.eval_id, "team": self.team, "seed": self.seed, "namespace": self.namespace,
                "strict": self.strict, "diverged_at": self.diverged_at, "calls": dict(self.calls),
                "live_tokens": self.live_tokens, "first_live_call": self.first_live_call, "stopped": self.stopped}


@dataclass
class Call:
    """One request's context: its eval tag and, in eval mode, its token's claims and eval."""

    tag: str
    claims: dict[str, Any] | None = None
    ev: EvalState | None = None

    def who(self) -> dict[str, Any]:
        if self.claims is None:
            return {}
        return {"eval_id": self.claims["eval_id"], "team": self.claims["team"], "role": self.claims["role"]}


class Gateway:
    def __init__(self, store: Store, mode: str, *, upstreams: dict[str, Upstream], sampling: dict[str, Any],
                 container_digest: str, harness_seed: int, log_path: Path | None = None,
                 timeout_s: float = 1800, secret: bytes | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "record" and not any(u.base_url for u in upstreams.values()):
            raise ValueError("record mode needs an upstream")
        if mode == "eval" and not secret:
            raise ValueError("eval mode needs the eval-token secret")
        self.store, self.mode, self.upstreams, self.secret = store, mode, dict(upstreams), secret
        # In eval mode the store reads golden only; each eval gets a store over golden and its namespace.
        self.golden = store.policy.read[0]
        self.sampling = dict(sampling)
        self.container_digest, self.harness_seed = container_digest, harness_seed
        self.log_path, self.timeout_s = log_path, timeout_s
        self.counts: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._seq = 0
        self._evals: dict[str, EvalState] = {}
        self._stores: dict[str, Store] = {}
        self._slots = {id(u): threading.BoundedSemaphore(u.max_concurrency)
                       for u in self.upstreams.values() if u.max_concurrency}
        self.sleep = time.sleep  # tests replace it

    def _log(self, call: Call, **fields: Any) -> None:
        with self._lock:
            self._seq += 1
            name = f"{fields['kind']}_{fields['outcome']}"
            self.counts[name] += 1
            if call.ev is not None:
                call.ev.calls[name] += 1
            if self.log_path is not None:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"seq": self._seq, "eval": call.tag, **call.who(), **fields}) + "\n")

    # -- evals -----------------------------------------------------------------------------------
    def _eval(self, claims: dict[str, Any]) -> EvalState:
        with self._lock:
            ev = self._evals.get(claims["eval_id"])
            if ev is None:
                ev = self._evals[claims["eval_id"]] = EvalState(claims["eval_id"], claims["team"], claims["seed"],
                                                                 claims["namespace"])
            return ev

    def _claims(self, token: str | None, role: str | None = None) -> tuple[dict | None, tuple[int, bytes] | None]:
        try:
            claims = verify(self.secret, token or "")
        except TokenError as e:
            return None, (401, _error("unauthorized", str(e)))
        if (role == "control") != (claims["role"] == "control"):
            return None, (403, _error("forbidden", f"a {claims['role']} token cannot do this"))
        if claims["namespace"] == self.golden:
            return None, (403, _error("forbidden", "an eval cannot write the golden namespace"))
        return claims, None

    def _admit(self, token: str | None, tag: str, model: str | None = None) -> tuple[Call, tuple[int, bytes] | None]:
        """The call's context if its token allows it, else the refusal (HTTP status, body)."""
        if self.mode != "eval":
            return Call(tag), None
        claims, refused = self._claims(token)
        if refused:
            return Call(tag), refused
        call = Call(tag, claims, self._eval(claims))
        if time.time() > claims["exp"]:
            return call, (429, _error("budget_exceeded", "the eval is past its wall-time budget", budget="wall"))
        if call.ev.live_tokens >= claims["token_cap"]:
            return call, (429, _error("budget_exceeded", "the eval has used its live tokens", budget="tokens"))
        if model is not None and model not in claims["allowed_models"]:
            return call, (403, _error("model_not_allowed", f"{model} is not available to this eval"))
        return call, None

    def _store(self, call: Call) -> Store:
        if call.claims is None:
            return self.store
        ns = call.claims["namespace"]
        with self._lock:
            if ns not in self._stores:
                self._stores[ns] = Store(self.store.root, Policy(read=(self.golden, ns), write=ns))
            return self._stores[ns]

    def _miss(self, call: Call, kind: str, key: str,
              make_report: Callable[[], dict[str, Any] | None]) -> tuple[int, bytes] | None:
        """The refusal for a cache miss, or None when the call may run live.

        Strict (replay mode, or an eval before its first hook change): a harness bug, HTTP 409
        replay_miss. A preview after divergence: the first call that would need inference, HTTP 409
        preview_stop. Either way the report says where the request departs from the nearest one;
        it is only made for a refusal, because finding the nearest request reads the namespace."""
        preview = call.claims is not None and not call.claims["live"]
        if self.mode == "replay" or (call.ev and call.ev.strict):
            outcome, error, message = "miss", "replay_miss", "strict replay: this call is not recorded"
        elif preview:
            outcome, error, message = "preview_stop", "preview_stop", "preview: this call would run live"
        else:
            return None
        report = make_report()
        if call.ev is not None:
            with self._lock:
                call.ev.stopped = call.ev.stopped or {"kind": kind, "eval": call.tag, "error": error, "key": key,
                                                      "report": report}
                if preview and call.ev.first_live_call is None:
                    call.ev.first_live_call = {"call": sum(call.ev.calls.values()) + 1, "eval": call.tag, "kind": kind}
        self._log(call, kind=kind, key=key, outcome=outcome, report=report)
        return 409, _error(error, message, key=key, report=report)

    def _note_live(self, call: Call, kind: str, tokens: int = 0) -> None:
        if call.ev is None:
            return
        with self._lock:
            call.ev.live_tokens += tokens
            if call.ev.first_live_call is None:
                call.ev.first_live_call = {"call": sum(call.ev.calls.values()) + 1, "eval": call.tag, "kind": kind}

    def eval_state(self, eval_id: str, token: str | None) -> tuple[int, bytes]:
        claims, refused = self._claims(token, role="control")
        if refused:
            return refused
        if claims["eval_id"] != eval_id:
            return 403, _error("forbidden", "this control token is for another eval")
        return 200, json.dumps(self._eval(claims).summary()).encode()

    def diverge(self, eval_id: str, body: dict[str, Any], token: str | None) -> tuple[int, bytes]:
        """The runner saw the eval's first hook change: from now on a miss is the attack's effect."""
        claims, refused = self._claims(token, role="control")
        if refused:
            return refused
        if claims["eval_id"] != eval_id:
            return 403, _error("forbidden", "this control token is for another eval")
        ev = self._eval(claims)
        with self._lock:
            if ev.strict:
                ev.strict, ev.diverged_at = False, body.get("at")
        self._log(Call("", claims), kind="control", key="", outcome="diverged", at=body.get("at"))
        return 200, json.dumps(ev.summary()).encode()

    # -- LLM calls -------------------------------------------------------------------------------
    def upstream_for(self, model: str) -> Upstream | None:
        return self.upstreams.get(model) or self.upstreams.get(ANY_MODEL)

    def llm_request_key(self, body: dict[str, Any]) -> tuple[dict[str, Any], str, list]:
        """The request as forwarded (sampling enforced) and its key."""
        upstream = self.upstream_for(body["model"])
        if upstream is None:
            raise KeyError(f"no upstream serves {body['model']}")
        request = {**body, **self.sampling}
        extras = {k: v for k, v in request.items() if k not in ("model", "messages", "tools", *self.sampling)}
        key, norm = llm_key(model_id=request["model"], model_rev=upstream.model_rev,
                            sampling={**self.sampling, **extras}, messages=request["messages"],
                            tools=request.get("tools"), server=upstream.server)
        return request, key, norm.replacements

    def chat(self, body: dict[str, Any], tag: str, token: str | None = None) -> tuple[int, bytes]:
        if body.get("stream"):
            return 400, _error("invalid_request", "streaming is not supported")
        call, refused = self._admit(token, tag, body["model"])
        if refused:
            return refused
        if self.upstream_for(body["model"]) is None:
            return 404, _error("unknown_model", f"no upstream serves {body['model']}")
        request, key, replaced = self.llm_request_key(body)
        start = time.time()
        store = self._store(call)
        entry = store.get(key)
        if entry is not None:
            self._log(call, kind="llm", key=key, outcome="hit", ns=entry.namespace, normalized=replaced)
            return 200, entry.value
        refused = self._miss(call, "llm", key, lambda: self._miss_report(request, tag))
        if refused:
            return refused
        status, data, retried = self._forward(request, self.upstream_for(request["model"]))
        usage = {}
        if status == 200:
            usage = json.loads(data).get("usage") or {}
            store.put(key, data, meta={"model": request["model"], "usage": usage, "eval": tag},
                      files={"request.json": canonical_json(request)})
            self._note_live(call, "llm", (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))
        self._log(call, kind="llm", key=key, outcome="live" if status == 200 else f"upstream_{status}",
                  normalized=replaced, latency_s=round(time.time() - start, 3), retries=retried,
                  prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
        return status, data

    def _miss_report(self, request: dict[str, Any], tag: str) -> dict[str, Any] | None:
        """Where a missing request departs from the nearest recorded one: golden in eval mode, the
        namespace itself in replay mode. Only computed when the miss will be refused."""
        if self.mode == "record":
            return None
        try:
            return miss_report(self.store, self.golden, request, tag)
        except (OSError, ValueError) as e:  # never let the report hide the miss itself
            return {"error": f"{type(e).__name__}: {e}"}

    def _forward(self, request: dict[str, Any], upstream: Upstream) -> tuple[int, bytes, int]:
        """(status, body, retries): the upstream's reply, after up to upstream.retries more attempts
        on HTTP 429 or 5xx or no connection. Waits double from RETRY_WAIT_S, or follow Retry-After."""
        if not upstream.base_url:
            return 503, _error("no_upstream", f"no upstream URL for {request['model']}"), 0
        headers = {"Content-Type": "application/json"}
        if upstream.api_key:
            headers["Authorization"] = f"Bearer {upstream.api_key}"
        body = {k: v for k, v in request.items() if k not in upstream.drop_params}
        data = json.dumps({**body, "model": upstream.served_name or request["model"]}).encode()
        slot = self._slots.get(id(upstream))
        for attempt in range(upstream.retries + 1):
            retry_after = None
            if slot:
                slot.acquire()
            try:
                req = urllib.request.Request(upstream.base_url.rstrip("/") + "/chat/completions", data=data,
                                             headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return r.status, r.read(), attempt
            except urllib.error.HTTPError as e:
                status, reply, retry_after = e.code, e.read(), e.headers.get("Retry-After")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                status, reply = 502, _error("upstream_unreachable", str(e))
            finally:
                if slot:
                    slot.release()
            if status not in RETRY_STATUS or attempt == upstream.retries:
                return status, reply, attempt
            try:
                wait = float(retry_after)
            except (TypeError, ValueError):
                wait = RETRY_WAIT_S * 2 ** attempt
            self.sleep(min(max(wait, 0.0), RETRY_MAX_WAIT_S))
        return status, reply, upstream.retries  # not reached

    # -- tool calls ------------------------------------------------------------------------------
    def tool_request_key(self, body: dict[str, Any]) -> str:
        return tool_key(tool_name=body["tool"], tool_version=body["tool_version"], args=body["args"],
                        workspace_manifest_hash=manifest_hash(body["manifest"]),
                        container_digest=self.container_digest, harness_seed=self.harness_seed)

    def tool_lookup(self, body: dict[str, Any], tag: str, token: str | None = None) -> tuple[int, bytes]:
        call, refused = self._admit(token, tag)
        if refused:
            return refused
        key = self.tool_request_key(body)
        store = self._store(call)
        entry = store.get(key)
        if entry is not None:
            files = {name: base64.b64encode(store.get_file(entry, name)).decode() for name in entry.files}
            self._log(call, kind="tool", tool=body["tool"], key=key, outcome="hit", ns=entry.namespace)
            return 200, json.dumps({"hit": True, "key": key, "result": entry.value.decode(), "files": files,
                                    "deleted": entry.meta.get("deleted", [])}).encode()
        refused = self._miss(call, "tool", key, lambda: {"tool": body["tool"], "args": body["args"]})
        if refused:
            return refused
        self._note_live(call, "tool")
        return 200, json.dumps({"hit": False, "key": key}).encode()

    def tool_record(self, body: dict[str, Any], tag: str, token: str | None = None) -> tuple[int, bytes]:
        if self.mode == "replay":
            return 403, _error("read_only", "this gateway does not record")
        call, refused = self._admit(token, tag)
        if refused:
            return refused
        key = self.tool_request_key(body)
        files = {name: base64.b64decode(data) for name, data in body.get("files", {}).items()}
        self._store(call).put(key, body["result"].encode(), files=files,
                              meta={"tool": body["tool"], "args": body["args"], "deleted": body.get("deleted", []),
                                    "eval": tag})
        self._log(call, kind="tool", tool=body["tool"], key=key, outcome="live",
                  files=len(files), bytes=sum(len(d) for d in files.values()))
        return 200, json.dumps({"stored": True, "key": key}).encode()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    gateway: Gateway  # set by serve()

    def _send(self, status: int, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _token(self) -> str | None:
        auth = self.headers.get("Authorization", "")
        return auth[len("Bearer "):] if auth.startswith("Bearer ") else None

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        path = self.path.rstrip("/")
        match = _EVAL_PATH.match(path)
        if path == "/health":
            self._send(200, json.dumps({"status": "ok", "mode": self.gateway.mode}).encode())
        elif match and not match.group(2) and self.gateway.mode == "eval":
            self._send(*self.gateway.eval_state(match.group(1), self._token()))
        else:
            self._send(404, _error("not_found", self.path))

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        routes = {"/v1/chat/completions": self.gateway.chat, "/v1/tools/lookup": self.gateway.tool_lookup,
                  "/v1/tools/record": self.gateway.tool_record}
        match = _EVAL_PATH.match(path)
        if path not in routes and not (match and match.group(2) and self.gateway.mode == "eval"):
            self._send(404, _error("not_found", self.path))
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, _error("invalid_json", str(e)))
            return
        try:
            if match:
                self._send(*self.gateway.diverge(match.group(1), body, self._token()))
            else:
                self._send(*routes[path](body, self.headers.get("X-Eval", ""), self._token()))
        except (KeyError, TypeError) as e:
            self._send(400, _error("invalid_request", f"{type(e).__name__}: {e}"))

    def log_message(self, format: str, *args: Any) -> None:  # every call is in the gateway log instead
        pass


def make_server(gateway: Gateway, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"gateway": gateway})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def make_unix_server(gateway: Gateway, path: Path) -> ThreadingUnixHTTPServer:
    """The same gateway on a Unix socket, for stage containers with no network (harness/uds.py)."""
    return ThreadingUnixHTTPServer(path, type("Handler", (_Handler,), {"gateway": gateway}))


def summarize(log_path: Path) -> dict[str, Any]:
    """Counts per outcome, overall and per eval tag, from a gateway log."""
    total: Counter[str] = Counter()
    per_eval: dict[str, Counter[str]] = {}
    tokens = 0
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["kind"] == "control":
            continue
        name = f"{rec['kind']}_{rec['outcome']}"
        total[name] += 1
        per_eval.setdefault(rec.get("eval", ""), Counter())[name] += 1
        tokens += (rec.get("prompt_tokens") or 0) + (rec.get("completion_tokens") or 0)
    return {"total": dict(total), "per_eval": {k: dict(v) for k, v in sorted(per_eval.items())}, "live_tokens": tokens}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--cas", type=Path, required=True)
    s.add_argument("--namespace", required=True, help="record/replay: the namespace; eval: the golden namespace")
    s.add_argument("--mode", choices=MODES, required=True)
    s.add_argument("--upstreams", type=Path, help="JSON map of model name to upstream (see load_upstreams)")
    s.add_argument("--upstream", help="without --upstreams: the one OpenAI-compatible base URL, for every model")
    s.add_argument("--model-rev", help="without --upstreams: the model revision in every key")
    s.add_argument("--system-start-date", help="without --upstreams: gpt-oss's pinned system date (SYSTEM_START_DATE)")
    s.add_argument("--secret", type=Path, help="eval mode: the eval-token secret (harness/evaltoken.py)")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--socket", type=Path, help="also listen on this Unix socket (stage containers with no network)")
    s.add_argument("--log", type=Path)
    s.add_argument("--container-digest", required=True, help="sha256 of the image tool calls run in")
    s.add_argument("--harness-seed", type=int, default=0)
    s.add_argument("--llm-seed", type=int, default=0)
    s.add_argument("--max-tokens", type=int, default=16384)
    s.add_argument("--reasoning-effort", default="medium")
    seal = sub.add_parser("seal")
    seal.add_argument("--cas", type=Path, required=True)
    seal.add_argument("--namespace", required=True)
    summ = sub.add_parser("summary")
    summ.add_argument("log", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "seal":
        Store(args.cas, Policy.recording(args.namespace)).seal(args.namespace)
        print(f"sealed {args.namespace}")
        return 0
    if args.cmd == "summary":
        print(json.dumps(summarize(args.log), indent=2))
        return 0

    if args.upstreams:
        upstreams = load_upstreams(args.upstreams)
    elif args.model_rev and args.system_start_date:
        upstreams = {ANY_MODEL: Upstream(base_url=args.upstream, model_rev=args.model_rev,
                                         server={"system_start_date": args.system_start_date})}
    else:
        ap.error("give --upstreams, or --model-rev and --system-start-date")
    faulthandler.enable()  # a crash in a C extension still prints every thread's traceback
    policy = Policy.recording(args.namespace) if args.mode == "record" else Policy.replay_only(args.namespace)
    # The same values AoD's chat_backend.sampling() sends; enforced whatever a client sends.
    sampling = {"temperature": 0.0, "top_p": 1.0, "seed": args.llm_seed, "max_tokens": args.max_tokens,
                "reasoning_effort": args.reasoning_effort}
    gateway = Gateway(Store(args.cas, policy), args.mode, upstreams=upstreams, sampling=sampling,
                      container_digest=args.container_digest, harness_seed=args.harness_seed, log_path=args.log,
                      secret=read_secret(args.secret) if args.secret else None)
    server = make_server(gateway, port=args.port)
    unix = serve_in_thread(make_unix_server(gateway, args.socket)) if args.socket else None

    def stop(signum, frame):  # the Slurm job stops the gateway with SIGTERM
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    print(f"gateway {args.mode} namespace={args.namespace} on 127.0.0.1:{args.port}"
          + (f" and {args.socket}" if unix else ""), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if unix is not None:
            unix.shutdown()
            unix.server_close()
        print(f"gateway counts: {dict(gateway.counts)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
