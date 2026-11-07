"""Submission bundles: a gzipped tar of the submission folder, unpacked safely.

Only regular files and folders are accepted: no links, devices, absolute paths or "..", within the
size and file-count limits. A single top-level folder (what `hack submit` sends) is stripped. The
static checks here are the free part of the validator that needs no sandbox; the full check
(hooks exercised twice on the public seed) runs in the worker before every eval.
"""

from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path, PurePosixPath

from harness.manifest import Manifest
from harness.manifest import load as load_manifest
from harness.validate import REQUIRED_FILES

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class BundleError(ValueError):
    """The bundle is refused; the message says why."""


def _members(tar: tarfile.TarFile, max_files: int, max_bytes: int) -> list[tuple[tarfile.TarInfo, PurePosixPath]]:
    members, total = [], 0
    for m in tar.getmembers():
        path = PurePosixPath(m.name)
        if path.is_absolute() or ".." in path.parts or m.name.startswith("/"):
            raise BundleError(f"unsafe path in the bundle: {m.name}")
        if not (m.isfile() or m.isdir()):
            raise BundleError(f"only files and folders are accepted: {m.name}")
        total += m.size
        if total > max_bytes:
            raise BundleError(f"the bundle unpacks to more than {max_bytes // 2**20} MB")
        members.append((m, path))
    if sum(m.isfile() for m, _ in members) > max_files:
        raise BundleError(f"the bundle holds more than {max_files} files")
    return members


def unpack(data: bytes, dest: Path, *, max_files: int = 200, max_bytes: int = 50 * 2**20) -> Path:
    """Unpacks a .tar.gz bundle into dest (which must not exist) and returns the submission folder."""
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
    except (tarfile.TarError, EOFError, OSError) as e:
        raise BundleError(f"not a readable tar archive: {e}") from None
    with tar:
        members = _members(tar, max_files, max_bytes)
        files = [p for m, p in members if m.isfile()]
        if not files:
            raise BundleError("the bundle is empty")
        tops = {p.parts[0] for _, p in members if p.parts and p.parts[0] != "."}
        strip = len(tops) == 1 and all(len(p.parts) > 1 for p in files)
        dest.mkdir(parents=True)
        for m, path in members:
            parts = [x for x in path.parts if x != "."][1 if strip else 0:]
            if not parts:
                continue
            target = dest.joinpath(*parts)
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            target.write_bytes(src.read() if src else b"")
    return dest


def static_check(folder: Path, team: str, tracks: dict) -> tuple[Manifest | None, list[str]]:
    """The manifest, or None and the problems: what the API refuses before queueing anything."""
    manifest, problems = load_manifest(folder)
    problems += [f"{name} is missing" for name in REQUIRED_FILES if not (folder / name).is_file()]
    if manifest is None:
        return None, problems
    if manifest.team != team:
        problems.append(f"manifest.yaml says team {manifest.team!r}, but this token is {team!r}'s")
    if not NAME.match(manifest.name):
        problems.append("manifest.yaml: name must be 1-64 letters, digits, '.', '_' or '-'")
    if manifest.target not in tracks:
        problems.append(f"target {manifest.target!r} is not open at this event (open: {sorted(tracks)})")
    return (None, problems) if problems else (manifest, [])


def pack(folder: Path) -> bytes:
    """A bundle of a submission folder, as `hack submit` sends it."""
    skip = {"__pycache__", ".git", ".ipynb_checkpoints"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(Path(folder).rglob("*")):
            rel = path.relative_to(folder)
            if skip & set(rel.parts) or path.is_symlink() or not path.is_file():
                continue
            tar.add(path, arcname=str(PurePosixPath(Path(folder).name) / rel.as_posix()), recursive=False)
    return buf.getvalue()
