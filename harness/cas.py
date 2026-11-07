"""Content-addressed store for recorded calls, split into namespaces.

Layout, one folder per namespace (plain files, so it is safe on NFS where SQLite locking is not):

    <root>/<ns>/objects/<aa>/<sha256>        zstd-compressed blob, named by the sha256 of its bytes
    <root>/<ns>/index/<aa>/<key>.json        {"key", "blob", "meta", "files": {name: sha256}}
    <root>/<ns>/SEALED                       present once the namespace is read-only

A Policy says which namespaces a process may read (in lookup order) and the single one it may
write. Teams read golden then their own namespace and write only their own; nothing but an explicit
recording policy can write golden, and a sealed namespace refuses all writes.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import zstandard

from harness.keys import sha256_bytes

GOLDEN = "golden"
_NS = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_KEY = re.compile(r"^[0-9a-f]{64}$")


class CASError(Exception):
    pass


class WriteNotAllowed(CASError):
    pass


class NamespaceSealed(CASError):
    pass


class IntegrityError(CASError):
    pass


def _check_ns(ns: str) -> str:
    if not _NS.match(ns):
        raise CASError(f"invalid namespace name: {ns!r}")
    return ns


def _check_key(key: str) -> str:
    if not _KEY.match(key):
        raise CASError(f"invalid key (want 64 lowercase hex chars): {key!r}")
    return key


@dataclass(frozen=True)
class Policy:
    read: tuple[str, ...]
    write: str | None = None

    def __post_init__(self) -> None:
        for ns in self.read:
            _check_ns(ns)
        if self.write is not None:
            _check_ns(self.write)

    @classmethod
    def for_team(cls, team: str) -> Policy:
        if team == GOLDEN:
            raise CASError("a team namespace cannot be called 'golden'")
        return cls(read=(GOLDEN, team), write=team)

    @classmethod
    def recording(cls, ns: str = GOLDEN) -> Policy:
        return cls(read=(ns,), write=ns)

    @classmethod
    def replay_only(cls, *namespaces: str) -> Policy:
        return cls(read=tuple(namespaces) or (GOLDEN,), write=None)


@dataclass(frozen=True)
class Entry:
    key: str
    namespace: str
    value: bytes
    meta: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Store:
    def __init__(self, root: Path | str, policy: Policy, level: int = 6):
        self.root = Path(root)
        self.policy = policy
        self._level = level
        # zstandard contexts are not thread-safe: sharing one across the gateway's request threads
        # crashed it with a segmentation fault. One pair per thread.
        self._local = threading.local()

    @property
    def _cctx(self) -> zstandard.ZstdCompressor:
        if not hasattr(self._local, "cctx"):
            self._local.cctx = zstandard.ZstdCompressor(level=self._level)
        return self._local.cctx

    @property
    def _dctx(self) -> zstandard.ZstdDecompressor:
        if not hasattr(self._local, "dctx"):
            self._local.dctx = zstandard.ZstdDecompressor()
        return self._local.dctx

    # -- paths ---------------------------------------------------------------------------------
    def _ns_dir(self, ns: str) -> Path:
        return self.root / _check_ns(ns)

    def _blob_path(self, ns: str, digest: str) -> Path:
        return self._ns_dir(ns) / "objects" / digest[:2] / digest

    def _index_path(self, ns: str, key: str) -> Path:
        return self._ns_dir(ns) / "index" / key[:2] / f"{key}.json"

    def is_sealed(self, ns: str) -> bool:
        return (self._ns_dir(ns) / "SEALED").exists()

    # -- blobs ---------------------------------------------------------------------------------
    def _put_blob(self, ns: str, data: bytes) -> str:
        digest = sha256_bytes(data)
        path = self._blob_path(ns, digest)
        if not path.exists():
            _atomic_write(path, self._cctx.compress(data))
        return digest

    def _get_blob(self, ns: str, digest: str) -> bytes:
        path = self._blob_path(ns, digest)
        try:
            data = self._dctx.decompress(path.read_bytes())
        except FileNotFoundError as e:
            raise IntegrityError(f"{ns}: blob {digest} referenced but missing") from e
        if sha256_bytes(data) != digest:
            raise IntegrityError(f"{ns}: blob {digest} does not match its hash")
        return data

    # -- entries -------------------------------------------------------------------------------
    def _read(self, ns: str, key: str) -> Entry | None:
        path = self._index_path(ns, key)
        if not path.exists():
            return None
        rec = json.loads(path.read_text())
        return Entry(
            key=key,
            namespace=ns,
            value=self._get_blob(ns, rec["blob"]),
            meta=rec.get("meta", {}),
            files=rec.get("files", {}),
        )

    def get(self, key: str) -> Entry | None:
        """First entry for key in the policy's read order, or None (a cache miss)."""
        _check_key(key)
        for ns in self.policy.read:
            entry = self._read(ns, key)
            if entry is not None:
                return entry
        return None

    def get_file(self, entry: Entry, name: str) -> bytes:
        return self._get_blob(entry.namespace, entry.files[name])

    def put(
        self,
        key: str,
        value: bytes,
        meta: dict[str, Any] | None = None,
        files: dict[str, bytes] | None = None,
    ) -> Entry:
        """Record a call result in the policy's write namespace.

        First writer wins: if the key is already recorded there, the stored entry is returned
        unchanged. A second, different value for the same key is a determinism signal for the
        caller to log, never a silent overwrite.
        """
        _check_key(key)
        ns = self.policy.write
        if ns is None:
            raise WriteNotAllowed("this policy is replay-only")
        if self.is_sealed(ns):
            raise NamespaceSealed(f"namespace {ns!r} is sealed")
        existing = self._read(ns, key)
        if existing is not None:
            return existing
        file_digests = {name: self._put_blob(ns, data) for name, data in sorted((files or {}).items())}
        rec = {"key": key, "blob": self._put_blob(ns, value), "meta": meta or {}, "files": file_digests}
        _atomic_write(self._index_path(ns, key), json.dumps(rec, sort_keys=True).encode())
        return Entry(key=key, namespace=ns, value=value, meta=meta or {}, files=file_digests)

    def _check_readable(self, ns: str) -> None:
        if ns not in self.policy.read and ns != self.policy.write:
            raise CASError(f"namespace {ns!r} is outside this policy")

    def get_from(self, ns: str, key: str) -> Entry | None:
        """Read one specific namespace (must be in the policy's read set or its write namespace)."""
        _check_key(key)
        self._check_readable(ns)
        return self._read(ns, key)

    def scan(self, ns: str) -> Iterator[Entry]:
        """Every entry of one namespace, in key order, with its meta and files but an empty value
        (read the value with get_from, files with get_file). Used to find the recorded request
        nearest to one that missed."""
        self._check_readable(ns)
        for path in sorted((self._ns_dir(ns) / "index").glob("*/*.json")):
            rec = json.loads(path.read_text())
            yield Entry(key=rec["key"], namespace=ns, value=b"", meta=rec.get("meta", {}), files=rec.get("files", {}))

    def seal(self, ns: str) -> None:
        """Mark a namespace read-only and drop write permission on its files (best effort on NFS)."""
        if ns != self.policy.write:
            raise WriteNotAllowed(f"only the writer of {ns!r} may seal it")
        d = self._ns_dir(ns)
        d.mkdir(parents=True, exist_ok=True)
        (d / "SEALED").touch()
        for p in [d, *d.rglob("*")]:
            mode = p.stat().st_mode
            p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
