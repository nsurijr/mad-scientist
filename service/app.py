"""The service's HTTP API and pages.

    SERVICE_HOME=state SERVICE_REDIS_URL=redis://localhost:6379/0 uvicorn --factory service.app:create_app

Teams (bearer token, service/auth.py):
    POST /api/submissions[?preview=true]   body: the bundle (.tar.gz); queues a live eval on the
                                           track's public seed, or a free preview
    POST /api/submissions/{id}/preview     a preview of a submission already uploaded
    GET  /api/status                       own submissions, evals, and finalists
    GET  /api/evals/{id}                   one eval: status and result (the team view)
    GET  /api/evals/{id}/trace             its trace viewer page
    PUT  /api/finalists                    {"submissions": [...]}: at most 3, before the freeze
Everyone:
    GET  /                                 the leaderboard page; GET /api/leaderboard its data
Organizers:
    POST /api/admin/freeze, POST /api/admin/reveal, GET /api/admin/ranking, GET /api/admin/evals,
    PUT /api/admin/writeups {"<submission id>": score in [0, 1]}
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import zstandard
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from service import leaderboard, policy, traces
from service.auth import Tokens, Who
from service.bundle import BundleError, static_check, unpack
from service.config import Settings
from service.queue import Queues
from service.store import OPEN, TEAM_KINDS, Store

STATIC = Path(__file__).parent / "static"
FAILED_FOR_TEAMS = "the eval could not run; it does not count against you, and the organizers can see why"


class Refused(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(detail)
        self.status, self.detail = status, detail


class Service:
    """Everything the API does, without HTTP: tests drive this directly or through the app."""

    def __init__(self, settings: Settings, store: Store | None = None, queues: Queues | None = None,
                 tokens: Tokens | None = None):
        self.settings = settings
        settings.home.mkdir(parents=True, exist_ok=True)
        self.store = store or Store(settings.home / "service.db")
        if queues is None:
            from redis import Redis
            queues = Queues(Redis.from_url(settings.redis_url), timeout_s=settings.job_timeout_s)
        self.queues = queues
        self.tokens = tokens or Tokens.load(settings.tokens_path)
        self._lock = threading.RLock()

    # -- submitting ------------------------------------------------------------------------------
    def submit(self, who: Who, data: bytes, preview: bool = False) -> dict[str, Any]:
        s = self.settings
        if self.store.flag("frozen"):
            raise Refused(409, "the event is frozen: no new submissions")
        if len(data) > s.max_bundle_bytes:
            raise Refused(413, f"the bundle is over {s.max_bundle_bytes // 2**20} MB")
        incoming = s.home / "incoming" / uuid.uuid4().hex
        try:
            try:
                folder = unpack(data, incoming, max_files=s.max_files, max_bytes=s.max_unpacked_bytes)
            except BundleError as e:
                raise Refused(400, [str(e)]) from None
            manifest, problems = static_check(folder, who.team, s.tracks())
            if manifest is None:
                raise Refused(400, problems)
            self._check_limits(who.team, "preview" if preview else "live")
            path = s.home / "bundles" / who.team / f"{uuid.uuid4().hex}.tar.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            with self._lock:
                sub = self.store.add_submission(manifest, path)
                ev = self._queue(sub, s.tracks()[manifest.target].live_seed, "preview" if preview else "live")
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
        self.collect()
        return {"submission": self._submission_view(sub), "eval": self.eval_view(who, ev["id"])}

    def preview(self, who: Who, submission_id: str) -> dict[str, Any]:
        sub = self.store.submission(submission_id)
        if sub is None or sub["team"] != who.team:
            raise Refused(404, f"no submission {submission_id}")
        if self.store.flag("frozen"):
            raise Refused(409, "the event is frozen")
        self._check_limits(who.team, "preview")
        with self._lock:
            ev = self._queue(sub, self.settings.tracks()[sub["target"]].live_seed, "preview")
        self.collect()
        return self.eval_view(who, ev["id"])

    def _check_limits(self, team: str, kind: str) -> None:
        s = self.settings
        if len(self.store.evals(team=team, kinds=TEAM_KINDS, statuses=OPEN)) >= s.in_flight:
            raise Refused(429, f"{s.in_flight} evals are already queued or running; wait for one to finish")
        if kind == "live" and len(self.store.evals(team=team, kinds=["live"], statuses=[*OPEN, "done"])) >= s.submission_cap:
            raise Refused(429, f"the cap of {s.submission_cap} live evals is used up")

    def _queue(self, sub: dict[str, Any], seed: str, kind: str) -> dict[str, Any]:
        ev = self.store.add_eval(sub, seed, kind)
        job_id = self.queues.enqueue(ev, Path(sub["bundle"]).read_bytes())
        return self.store.update_eval(ev["id"], job_id=job_id)

    # -- collecting results ----------------------------------------------------------------------
    def collect(self) -> int:
        """Records finished jobs, then queues the background runs that new bests call for. Repeats
        while that changes anything (with synchronous queues, a queued job finishes at once)."""
        total = 0
        with self._lock:
            for _ in range(100):
                changed = 0
                for ev in self.store.evals(statuses=OPEN):
                    state, payload = self.queues.state(ev["job_id"] or ev["id"])
                    if state == "running" and ev["status"] == "queued":
                        self.store.update_eval(ev["id"], status="running", started=time.time())
                    elif state == "done":
                        self._record(ev, payload)
                        changed += 1
                    elif state in ("failed", "lost"):
                        self.store.update_eval(ev["id"], status="failed", finished=time.time(),
                                               error=payload or "the queue lost the job")
                        changed += 1
                if changed:
                    for sid, seed, kind in policy.background_plan(self.store, self.settings):
                        self._queue(self.store.submission(sid), seed, kind)
                total += changed
                if not changed:
                    break
        return total

    def _record(self, ev: dict[str, Any], payload: dict[str, Any]) -> None:
        trace = None
        if payload.get("trace"):
            trace = self.settings.home / "traces" / f"{ev['id']}.json.zst"
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_bytes(payload["trace"])
        team_view = payload.get("team_view")
        if team_view is not None and trace is not None:
            team_view = {**team_view, "trace": f"/api/evals/{ev['id']}/trace"}
        self.store.update_eval(ev["id"], status=payload["status"], finished=time.time(),
                               started=ev["started"] or time.time(), check_report=payload.get("check_report"),
                               preview=payload.get("preview"), result=payload.get("result"), team_view=team_view,
                               trace=str(trace) if trace else None)

    # -- organizers ------------------------------------------------------------------------------
    def freeze(self) -> dict[str, Any]:
        with self._lock:
            self.store.set_flag("frozen", True)
            plan = policy.freeze_plan(self.store, self.settings)
            for sid, seed, kind in plan:
                self._queue(self.store.submission(sid), seed, kind)
        self.collect()
        return {"frozen": True, "queued": [f"{kind} {sid} {seed}" for sid, seed, kind in plan]}

    def reveal(self) -> dict[str, Any]:
        self.store.set_flag("revealed", True)
        return {"revealed": True}

    # -- views -----------------------------------------------------------------------------------
    @staticmethod
    def _submission_view(sub: dict[str, Any]) -> dict[str, Any]:
        return {k: sub[k] for k in ("id", "team", "name", "attack_class", "target", "intended_direction", "created")}

    def _visible(self, who: Who, ev: dict[str, Any] | None) -> bool:
        if ev is None:
            return False
        if who.organizer:
            return True
        return ev["team"] == who.team and (ev["kind"] in TEAM_KINDS or self.store.flag("revealed"))

    def eval_view(self, who: Who, eval_id: str) -> dict[str, Any]:
        ev = self.store.eval(eval_id)
        if not self._visible(who, ev):
            raise Refused(404, f"no eval {eval_id} visible to you")
        if who.organizer:
            return ev
        view = {k: ev[k] for k in ("id", "submission_id", "seed", "kind", "status", "created", "started", "finished")}
        if ev["status"] == "failed":
            view["error"] = FAILED_FOR_TEAMS
        if ev["check_report"]:
            view["check"] = {k: ev["check_report"].get(k) for k in ("verdict", "problems", "warnings", "fired", "replaced")}
        if ev["preview"]:
            view["preview"] = ev["preview"]
        if ev["team_view"]:
            view["result"] = ev["team_view"]
        return view

    def status(self, who: Who) -> dict[str, Any]:
        self.collect()
        subs = self.store.submissions(None if who.organizer else who.team)
        evals = [self.eval_view(who, e["id"]) for e in self.store.evals(team=None if who.organizer else who.team)
                 if self._visible(who, e)]
        hidden = [e for e in self.store.evals(team=who.team) if not self._visible(who, e)]
        progress: dict[str, dict[str, int]] = {}
        for e in hidden:
            if e["status"] in ("done", *OPEN):
                p = progress.setdefault(e["submission_id"], {"done": 0, "queued_or_running": 0})
                p["done" if e["status"] == "done" else "queued_or_running"] += 1
        return {"team": who.team, "frozen": self.store.flag("frozen"), "revealed": self.store.flag("revealed"),
                "submissions": [self._submission_view(s) for s in subs], "evals": evals,
                "scoring_seed_progress": progress, "finalists": self.store.finalists(None if who.organizer else who.team)}

    def set_finalists(self, who: Who, submission_ids: list[str]) -> dict[str, Any]:
        problems = policy.finalist_problems(self.store, self.settings, who.team, submission_ids)
        if problems:
            raise Refused(400, problems)
        self.store.set_finalists(who.team, list(dict.fromkeys(submission_ids)))
        return {"finalists": self.store.finalists(who.team)}

    def trace_html(self, who: Who, eval_id: str) -> str:
        ev = self.store.eval(eval_id)
        if not self._visible(who, ev) or not ev["trace"] or not Path(ev["trace"]).is_file():
            raise Refused(404, f"no trace for {eval_id} visible to you")
        digest = json.loads(zstandard.ZstdDecompressor().decompress(Path(ev["trace"]).read_bytes()))
        return traces.render(digest, f"{ev['submission_id']} · {ev['kind']} · {ev['seed']}")

    def leaderboard(self) -> dict[str, Any]:
        self.collect()
        return leaderboard.public(self.store, self.settings, self.tokens.teams)


def create_app(settings: Settings | None = None, service: Service | None = None) -> FastAPI:
    svc = service or Service(settings or Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = threading.Event()
        if svc.settings.poll_s > 0:
            def poll() -> None:
                while not stop.wait(svc.settings.poll_s):
                    try:
                        svc.collect()
                    except Exception as e:  # keep polling; the next pass retries
                        print(f"collect failed: {e}", flush=True)
            threading.Thread(target=poll, daemon=True).start()
        yield
        stop.set()

    app = FastAPI(title="Hackathon eval service", lifespan=lifespan)
    app.state.service = svc

    @app.exception_handler(Refused)
    async def refused(request: Request, exc: Refused) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    def who(authorization: str | None = Header(default=None)) -> Who:
        token = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None
        found = svc.tokens.lookup(token)
        if found is None:
            raise HTTPException(401, "a valid team token is needed (Authorization: Bearer <token>)")
        return found

    def organizer(w: Who = Depends(who)) -> Who:
        if not w.organizer:
            raise HTTPException(403, "organizers only")
        return w

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "leaderboard.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/leaderboard")
    def board() -> dict:
        return svc.leaderboard()

    @app.post("/api/submissions")
    async def submit(request: Request, preview: bool = False, w: Who = Depends(who)) -> dict:
        length = int(request.headers.get("content-length") or 0)
        if length > svc.settings.max_bundle_bytes:
            raise Refused(413, f"the bundle is over {svc.settings.max_bundle_bytes // 2**20} MB")
        return svc.submit(w, await request.body(), preview=preview)

    @app.post("/api/submissions/{submission_id}/preview")
    def preview_existing(submission_id: str, w: Who = Depends(who)) -> dict:
        return svc.preview(w, submission_id)

    @app.get("/api/status")
    def status(w: Who = Depends(who)) -> dict:
        return svc.status(w)

    @app.get("/api/evals/{eval_id}")
    def eval_view(eval_id: str, w: Who = Depends(who)) -> dict:
        svc.collect()
        return svc.eval_view(w, eval_id)

    @app.get("/api/evals/{eval_id}/trace", response_class=HTMLResponse)
    def trace(eval_id: str, w: Who = Depends(who)) -> str:
        return svc.trace_html(w, eval_id)

    @app.put("/api/finalists")
    def finalists(body: dict, w: Who = Depends(who)) -> dict:
        return svc.set_finalists(w, list(body.get("submissions") or []))

    @app.post("/api/admin/freeze")
    def freeze(w: Who = Depends(organizer)) -> dict:
        return svc.freeze()

    @app.post("/api/admin/reveal")
    def reveal(w: Who = Depends(organizer)) -> dict:
        return svc.reveal()

    @app.get("/api/admin/ranking")
    def ranking(w: Who = Depends(organizer)) -> dict:
        svc.collect()
        return leaderboard.ranking(svc.store, svc.settings)

    @app.get("/api/admin/evals")
    def all_evals(w: Who = Depends(organizer)) -> list:
        svc.collect()
        return svc.store.evals()

    @app.put("/api/admin/writeups")
    def writeups(body: dict, w: Who = Depends(organizer)) -> dict:
        for sid, score in body.items():
            if svc.store.submission(sid) is None or not 0 <= float(score) <= 1:
                raise Refused(400, f"{sid}: needs an existing submission and a score in [0, 1]")
            svc.store.set_writeup(sid, float(score))
        return svc.store.writeups()

    return app
