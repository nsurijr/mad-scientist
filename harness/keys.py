"""Cache keys for LLM calls and tool calls.

A key is the SHA-256 of a canonical JSON encoding of everything that can change a call's
result. Two calls share a key exactly when their inputs are byte-identical after the narrow
normalization below, which is what lets replay detect where an attack starts to bite.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness import HARNESS_VERSION

_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
_TIMESTAMP = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8, NaN/Infinity refused."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class Normalization:
    """What normalize() replaced. Logged beside every request; empty on clean golden runs."""

    replacements: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.replacements)


class Normalizer:
    """Safety net for values that vary without meaning (timestamps, UUIDs, scratch paths).

    The harness should keep these out of prompts in the first place; anything this catches is a
    determinism leak worth fixing at the source. Path prefixes are explicit so normalization never
    rewrites content an attacker controls beyond these narrow patterns.
    """

    def __init__(self, path_prefixes: tuple[str, ...] = ()):
        # Longest first, so a nested prefix wins over its parent.
        self.path_prefixes = tuple(sorted(path_prefixes, key=len, reverse=True))

    def _text(self, s: str, report: Normalization) -> str:
        def sub(pattern: re.Pattern[str], placeholder: str, text: str) -> str:
            def repl(m: re.Match[str]) -> str:
                report.replacements.append((m.group(0), placeholder))
                return placeholder

            return pattern.sub(repl, text)

        s = sub(_TIMESTAMP, "<TIMESTAMP>", s)
        s = sub(_UUID, "<UUID>", s)
        for prefix in self.path_prefixes:
            if prefix in s:
                report.replacements.append((prefix, "<WORKDIR>"))
                s = s.replace(prefix, "<WORKDIR>")
        return s

    def __call__(self, obj: Any, report: Normalization | None = None) -> tuple[Any, Normalization]:
        report = report if report is not None else Normalization()
        if isinstance(obj, str):
            return self._text(obj, report), report
        if isinstance(obj, list):
            return [self(v, report)[0] for v in obj], report
        if isinstance(obj, dict):
            return {k: self(v, report)[0] for k, v in obj.items()}, report
        return obj, report


def llm_key(
    *,
    model_id: str,
    model_rev: str,
    sampling: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    server: dict[str, Any] | None = None,
    normalizer: Normalizer | None = None,
    harness_version: str = HARNESS_VERSION,
) -> tuple[str, Normalization]:
    """Key for one chat-completion request. `sampling` must be the values the gateway enforces;
    `server` holds what the model server adds to the prompt itself (gpt-oss's pinned system date),
    which the request bytes do not carry."""
    normalizer = normalizer or Normalizer()
    norm_messages, report = normalizer(messages)
    payload = {
        "kind": "llm",
        "harness_version": harness_version,
        "model_id": model_id,
        "model_rev": model_rev,
        "sampling": sampling,
        "server": server or {},
        "messages": norm_messages,
        "tools_sha256": sha256_bytes(canonical_json(tools or [])),
    }
    return sha256_bytes(canonical_json(payload)), report


def workspace_manifest(root: Path) -> list[tuple[str, str]]:
    """Sorted (relative path, sha256) for every regular file a tool could read under root."""
    root = Path(root)
    entries = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            entries.append((p.relative_to(root).as_posix(), sha256_file(p)))
    return entries


def manifest_hash(manifest: list[tuple[str, str]]) -> str:
    return sha256_bytes(canonical_json(manifest))


def tool_key(
    *,
    tool_name: str,
    tool_version: str,
    args: dict[str, Any],
    workspace_manifest_hash: str,
    container_digest: str,
    harness_seed: int,
    harness_version: str = HARNESS_VERSION,
) -> str:
    """Key for one tool execution: same tool, args, readable files, image and seed -> same result."""
    payload = {
        "kind": "tool",
        "harness_version": harness_version,
        "tool_name": tool_name,
        "tool_version": tool_version,
        "args": args,
        "workspace_manifest": workspace_manifest_hash,
        "container_digest": container_digest,
        "harness_seed": harness_seed,
    }
    return sha256_bytes(canonical_json(payload))
