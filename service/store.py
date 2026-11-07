"""The service's state in SQLite: submissions, evals, finalists, event flags and writeup scores.

Only the API side writes it. Workers return results through the queue (service/queue.py), so they
need no access to this file.

Eval kinds: preview and live (a team's own requests), background (the team's current best on the
hidden scoring seeds), finalist (designated finalists' missing scoring seeds, at the freeze) and
validation (the best attacks re-run with the cache off, at the freeze).
Eval statuses: queued, running, done, rejected (by the validator) and failed (the eval could not
run; it does not count against the team).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from harness.manifest import Manifest

KINDS = ("preview", "live", "background", "finalist", "validation")
TEAM_KINDS = ("preview", "live")  # what a team asks for, and sees before the reveal
OPEN = ("queued", "running")
_JSON = ("manifest", "check_report", "preview", "result", "team_view")

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY, team TEXT NOT NULL, name TEXT NOT NULL, attack_class TEXT NOT NULL,
    target TEXT NOT NULL, intended_direction TEXT NOT NULL, created REAL NOT NULL,
    bundle TEXT NOT NULL, manifest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evals (
    id TEXT PRIMARY KEY, submission_id TEXT NOT NULL REFERENCES submissions(id), team TEXT NOT NULL,
    seed TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
    started REAL, finished REAL, job_id TEXT, error TEXT, check_report TEXT, preview TEXT,
    result TEXT, team_view TEXT, trace TEXT);
CREATE INDEX IF NOT EXISTS evals_team ON evals(team, kind, status);
CREATE INDEX IF NOT EXISTS evals_submission ON evals(submission_id, seed);
CREATE TABLE IF NOT EXISTS finalists (
    team TEXT NOT NULL, submission_id TEXT NOT NULL REFERENCES submissions(id),
    PRIMARY KEY (team, submission_id));
CREATE TABLE IF NOT EXISTS flags (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS writeups (submission_id TEXT PRIMARY KEY, score REAL NOT NULL);
"""


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for key in _JSON:
        if key in out and out[key] is not None:
            out[key] = json.loads(out[key])
    return out


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.RLock()

    def _all(self, sql: str, args: Iterable = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [_row(r) for r in self._db.execute(sql, tuple(args)).fetchall()]

    def _one(self, sql: str, args: Iterable = ()) -> dict[str, Any] | None:
        with self._lock:
            return _row(self._db.execute(sql, tuple(args)).fetchone())

    def _exec(self, sql: str, args: Iterable = ()) -> None:
        with self._lock:
            self._db.execute(sql, tuple(args))

    # -- submissions -----------------------------------------------------------------------------
    def add_submission(self, manifest: Manifest, bundle: Path) -> dict[str, Any]:
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) FROM submissions WHERE team = ?", (manifest.team,)).fetchone()[0]
            sid = f"{manifest.team}-{n + 1:03d}"
            self._exec("INSERT INTO submissions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (sid, manifest.team, manifest.name, manifest.attack_class, manifest.target,
                        manifest.intended_direction, time.time(), str(bundle), json.dumps(asdict(manifest))))
        return self.submission(sid)

    def submission(self, sid: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM submissions WHERE id = ?", (sid,))

    def submissions(self, team: str | None = None) -> list[dict[str, Any]]:
        if team is None:
            return self._all("SELECT * FROM submissions ORDER BY created")
        return self._all("SELECT * FROM submissions WHERE team = ? ORDER BY created", (team,))

    # -- evals -----------------------------------------------------------------------------------
    def add_eval(self, submission: dict[str, Any], seed: str, kind: str) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        eid = f"{submission['id']}-{kind}-{seed}-{uuid.uuid4().hex[:6]}"
        self._exec("INSERT INTO evals (id, submission_id, team, seed, kind, status, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (eid, submission["id"], submission["team"], seed, kind, "queued", time.time()))
        return self.eval(eid)

    def update_eval(self, eid: str, **fields: Any) -> dict[str, Any]:
        cols = ", ".join(f"{k} = ?" for k in fields)
        args = [json.dumps(v) if k in _JSON and v is not None else v for k, v in fields.items()]
        self._exec(f"UPDATE evals SET {cols} WHERE id = ?", [*args, eid])
        return self.eval(eid)

    def eval(self, eid: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM evals WHERE id = ?", (eid,))

    def evals(self, *, team: str | None = None, submission_id: str | None = None, kinds: Iterable[str] | None = None,
              statuses: Iterable[str] | None = None, seed: str | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        for col, value in (("team", team), ("submission_id", submission_id), ("seed", seed)):
            if value is not None:
                where.append(f"{col} = ?")
                args.append(value)
        for col, values in (("kind", kinds), ("status", statuses)):
            if values is not None:
                values = list(values)
                where.append(f"{col} IN ({', '.join('?' * len(values))})")
                args += values
        sql = "SELECT * FROM evals" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created"
        return self._all(sql, args)

    # -- finalists, flags, writeups --------------------------------------------------------------
    def set_finalists(self, team: str, submission_ids: list[str]) -> None:
        with self._lock:
            self._db.execute("BEGIN")
            self._db.execute("DELETE FROM finalists WHERE team = ?", (team,))
            self._db.executemany("INSERT INTO finalists VALUES (?, ?)", [(team, s) for s in submission_ids])
            self._db.execute("COMMIT")

    def finalists(self, team: str | None = None) -> list[str]:
        rows = self._all("SELECT * FROM finalists" + (" WHERE team = ?" if team else ""), (team,) if team else ())
        return [r["submission_id"] for r in rows]

    def flag(self, name: str) -> bool:
        row = self._one("SELECT value FROM flags WHERE name = ?", (name,))
        return bool(row) and row["value"] == "1"

    def set_flag(self, name: str, value: bool) -> None:
        self._exec("INSERT INTO flags VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value = excluded.value",
                   (name, "1" if value else "0"))

    def set_writeup(self, submission_id: str, score: float) -> None:
        self._exec("INSERT INTO writeups VALUES (?, ?) ON CONFLICT(submission_id) DO UPDATE SET score = excluded.score",
                   (submission_id, float(score)))

    def writeups(self) -> dict[str, float]:
        return {r["submission_id"]: r["score"] for r in self._all("SELECT * FROM writeups")}
