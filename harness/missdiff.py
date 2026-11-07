"""Where a request that missed the cache departs from the nearest recorded one.

A strict-mode miss upstream of an attack's first change is a harness bug, and a preview stops at
the first call that would need inference. Either way, the useful answer is which recorded request
the call came closest to, and the first message where the two differ.

Nearest means the longest run of identical leading messages; ties go to the longer common text in
the first message that differs. Entries recorded under the same eval tag ("<seed>/<stage>") are
searched first, so a miss is compared with the same stage of the same seed when it was recorded.
"""

from __future__ import annotations

import difflib
import json
import os
from typing import Any

from harness.cas import Entry, Store

MAX_DIFF_LINES = 80


def _render(message: dict[str, Any]) -> list[str]:
    """A message as diffable lines: role and other fields first, then the content's own lines."""
    lines = [f"role: {message.get('role')}"]
    lines += [f"{k}: {json.dumps(message[k], sort_keys=True)}" for k in sorted(message) if k not in ("role", "content")]
    content = message.get("content")
    if isinstance(content, str):
        lines += content.splitlines()
    elif content is not None:
        lines.append(json.dumps(content, sort_keys=True))
    return lines


def _leading_match(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _closeness(request: dict[str, Any], recorded: dict[str, Any]) -> tuple[int, int]:
    a, b = recorded.get("messages", []), request.get("messages", [])
    n = _leading_match(a, b)
    if n < len(a) and n < len(b):
        common = len(os.path.commonprefix([json.dumps(a[n], sort_keys=True), json.dumps(b[n], sort_keys=True)]))
    else:
        common = 0
    return n, common


def nearest(store: Store, ns: str, request: dict[str, Any], tag: str = "") -> tuple[Entry, dict[str, Any]] | None:
    """The recorded LLM request in ns closest to request, with the entry it belongs to."""
    recorded = [e for e in store.scan(ns) if "request.json" in e.files]
    same_tag = [e for e in recorded if tag and e.meta.get("eval") == tag]
    best = None
    for entry in same_tag or recorded:
        candidate = json.loads(store.get_file(entry, "request.json"))
        score = _closeness(request, candidate)
        if best is None or score > best[0]:
            best = (score, entry, candidate)
    return None if best is None else (best[1], best[2])


def describe(request: dict[str, Any], recorded: dict[str, Any], recorded_key: str) -> dict[str, Any]:
    """How request differs from a recorded one: the first differing message as a unified diff, or
    the other fields that differ when the messages are identical."""
    a, b = recorded.get("messages", []), request.get("messages", [])
    i = _leading_match(a, b)
    report: dict[str, Any] = {"nearest_key": recorded_key, "messages": [len(a), len(b)], "first_difference": i}
    if i == len(a) == len(b):
        report["other_fields"] = sorted(k for k in set(request) | set(recorded)
                                        if k != "messages" and request.get(k) != recorded.get(k))
        return report
    old = _render(a[i]) if i < len(a) else []
    new = _render(b[i]) if i < len(b) else []
    lines = list(difflib.unified_diff(old, new, "recorded", "this request", n=2, lineterm=""))
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"... {len(lines) - MAX_DIFF_LINES} more lines"]
    report["diff"] = "\n".join(lines)
    return report


def miss_report(store: Store, ns: str, request: dict[str, Any], tag: str = "") -> dict[str, Any]:
    """describe() against the nearest recorded request in ns, or a note that ns holds none."""
    found = nearest(store, ns, request, tag)
    if found is None:
        return {"nearest_key": None, "note": f"namespace {ns!r} holds no recorded requests"}
    entry, recorded = found
    return {"namespace": ns, **describe(request, recorded, entry.key)}
