"""Memoized code execution through the harness gateway.

When AOD_TOOL_CACHE_URL is set, execute_python asks the gateway before running anything. The key
covers the call's arguments and the hash of every file the code could read: the working folder
(minus AoD's own logs, which hold timings) and the read-only /in and /data. On a hit the recorded
output files are written back into the working folder and the recorded result is returned, so no
code runs. On a miss the code runs, and its result and changed files are recorded. A strict-replay
miss or a gateway failure raises HarnessHalt. Standard library only.
"""
import base64
import fnmatch
import hashlib
import json
import os
import urllib.error
import urllib.request

from utility.harness_halt import HarnessHalt

TOOL_VERSION = "1"
# label=path pairs of the read-only folders code can read; tests point them elsewhere.
READ_ONLY_ROOTS_ENV = "AOD_TOOL_CACHE_READ_ROOTS"
DEFAULT_READ_ONLY_ROOTS = "in=/in,data=/data"
# AoD's own logs in the working folder: rewritten on every call, metrics hold timings.
LOG_PATTERNS = ("conversation*.md", "metrics_*.json", "logic_review_*", "task_*.md")

_read_only_cache = {}  # (path, size, mtime_ns) -> sha256; /in and /data never change in a session


def enabled() -> bool:
    return bool(os.environ.get("AOD_TOOL_CACHE_URL"))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _walk(root: str):
    """Relative paths of the regular files under root, sorted."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.isfile(path) and not os.path.islink(path):
                out.append(os.path.relpath(path, root).replace(os.sep, "/"))
    return sorted(out)


def _is_log(rel: str) -> bool:
    return "/" not in rel and any(fnmatch.fnmatch(rel, p) for p in LOG_PATTERNS)


def snapshot(work_dir: str) -> dict:
    """{relative path: sha256} of the working folder, without AoD's logs."""
    return {rel: _sha256(os.path.join(work_dir, rel)) for rel in _walk(work_dir) if not _is_log(rel)}


def manifest(work_snapshot: dict) -> list:
    """Every file the code could read, as sorted [label/path, sha256] pairs."""
    entries = [[f"work/{rel}", digest] for rel, digest in sorted(work_snapshot.items())]
    roots = os.environ.get(READ_ONLY_ROOTS_ENV, DEFAULT_READ_ONLY_ROOTS)
    for label, root in (pair.split("=", 1) for pair in roots.split(",") if pair):
        if not os.path.isdir(root):
            continue
        for rel in _walk(root):
            path = os.path.join(root, rel)
            st = os.stat(path)
            sig = (path, st.st_size, st.st_mtime_ns)
            if sig not in _read_only_cache:
                _read_only_cache[sig] = _sha256(path)
            entries.append([f"{label}/{rel}", _read_only_cache[sig]])
    return entries


def _post(endpoint: str, payload: dict) -> dict:
    url = os.environ["AOD_TOOL_CACHE_URL"].rstrip("/") + "/" + endpoint
    headers = {"Content-Type": "application/json", "X-Eval": os.environ.get("AOD_EVAL_TAG", "")}
    if os.environ.get("OPENAI_API_KEY", "EMPTY") != "EMPTY":  # the eval token, as the chat client sends it
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 409:
            raise HarnessHalt(f"strict replay: tool call not recorded ({body})") from None
        raise HarnessHalt(f"tool cache {endpoint}: HTTP {e.code} {body}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise HarnessHalt(f"tool cache {endpoint} unreachable: {e}") from None


def lookup(tool: str, args: dict, files: list):
    """The recorded {result, files, deleted}, or None on a miss (record mode)."""
    reply = _post("lookup", {"tool": tool, "tool_version": TOOL_VERSION, "args": args, "manifest": files})
    return reply if reply.get("hit") else None


def record(tool: str, args: dict, files: list, result: str, work_dir: str, before: dict) -> None:
    """Store a live execution: its result and every file it created, changed or deleted."""
    after = snapshot(work_dir)
    changed = sorted(rel for rel, digest in after.items() if before.get(rel) != digest)
    outputs = {}
    for rel in changed:
        with open(os.path.join(work_dir, rel), "rb") as f:
            outputs[rel] = base64.b64encode(f.read()).decode()
    _post("record", {"tool": tool, "tool_version": TOOL_VERSION, "args": args, "manifest": files,
                     "result": result, "files": outputs, "deleted": sorted(set(before) - set(after))})


def materialize(work_dir: str, hit: dict) -> None:
    """Write a recorded execution's output files into the working folder, as the code would have."""
    root = os.path.realpath(work_dir)
    for rel, data in hit.get("files", {}).items():
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(root + os.sep):
            raise HarnessHalt(f"recorded output {rel!r} lies outside the working folder")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))
    for rel in hit.get("deleted", []):
        path = os.path.join(root, rel)
        if os.path.isfile(path):
            os.remove(path)
