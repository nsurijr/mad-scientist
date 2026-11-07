"""The eval worker: runs queued jobs, live queue first.

    SERVICE_REDIS_URL=redis://... SERVICE_RUNNER=harness SERVICE_WORK=/scratch/evals python -m service.worker

A job (service/queue.py) carries the submission's bundle; run_job unpacks it, runs the validator
for a team's own requests (preview, live), then the preview or the eval, and returns the result.
It touches no store, so workers can run on another machine than the API.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import zstandard

from harness.scorer import team_view
from service import bundle, runners
from service.queue import BACKGROUND, LIVE

RUNNER: runners.Runner | None = None  # set by tests; otherwise from SERVICE_RUNNER


def runner() -> runners.Runner:
    global RUNNER
    if RUNNER is None:
        RUNNER = runners.from_env()
    return RUNNER


def pack_trace(digest: dict | None) -> bytes | None:
    return zstandard.ZstdCompressor().compress(json.dumps(digest).encode()) if digest is not None else None


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    base = Path(os.environ.get("SERVICE_WORK") or tempfile.gettempdir()) / "service-evals"
    work = base / job["eval_id"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    submission = bundle.unpack(job["bundle"], work / "submission")
    r = runner()
    out: dict[str, Any] = {"status": "done"}
    if job["kind"] in ("preview", "live"):
        report = r.check(submission)
        out["check_report"] = report
        if report.get("verdict") != "pass":
            return {**out, "status": "rejected"}
    if job["kind"] == "preview":
        preview, digest = r.preview(submission, job["eval_id"], job["team"], work)
        return {**out, "preview": preview, "trace": pack_trace(digest)}
    score, digest = r.evaluate(submission, job["seed"], job["eval_id"], job["team"], work,
                               cache=job["kind"] != "validation")
    return {**out, "result": score, "team_view": team_view(score), "trace": pack_trace(digest)}


def main(argv: list[str] | None = None) -> int:
    from redis import Redis
    from rq import SimpleWorker, Worker

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--simple", action="store_true", help="run jobs in this process (no fork), e.g. on macOS")
    ap.add_argument("--burst", action="store_true", help="stop when the queues are empty")
    args = ap.parse_args(argv)
    runner()  # fail now, not in the first job, if the runner is misconfigured
    connection = Redis.from_url(os.environ.get("SERVICE_REDIS_URL", "redis://localhost:6379/0"))
    cls = SimpleWorker if args.simple else Worker
    cls([LIVE, BACKGROUND], connection=connection).work(burst=args.burst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
