"""The two RQ queues: live (a team's preview and live evals, and finalists at the freeze)
and background (best attacks on the hidden seeds, and cache-off validation). A worker serves live
first.

A job is service.worker.run_job with everything it needs, the submission's bundle included, and
returns its result; the API collects results (service/app.py, Service.collect).
"""

from __future__ import annotations

from typing import Any

from rq import Queue
from rq.job import Job, JobStatus

LIVE, BACKGROUND = "live", "background"
QUEUE_OF = {"preview": LIVE, "live": LIVE, "finalist": LIVE, "background": BACKGROUND, "validation": BACKGROUND}
KEEP_S = 14 * 24 * 3600  # results and failures outlive the event


class Queues:
    def __init__(self, connection: Any, *, is_async: bool = True, timeout_s: int = 3 * 3600):
        self.connection = connection
        self.queues = {name: Queue(name, connection=connection, is_async=is_async, default_timeout=timeout_s)
                       for name in (LIVE, BACKGROUND)}

    def enqueue(self, ev: dict[str, Any], bundle: bytes) -> str:
        job = {"eval_id": ev["id"], "kind": ev["kind"], "seed": ev["seed"], "team": ev["team"],
               "submission_id": ev["submission_id"], "bundle": bundle}
        q = self.queues[QUEUE_OF[ev["kind"]]]
        return q.enqueue("service.worker.run_job", job, job_id=ev["id"], result_ttl=KEEP_S, failure_ttl=KEEP_S,
                         description=f"{ev['kind']} {ev['submission_id']} {ev['seed']}").id

    def state(self, job_id: str) -> tuple[str, Any]:
        """(state, payload): ("queued" | "running", None), ("done", the job's result),
        ("failed", the error) or ("lost", None) when Redis no longer knows the job."""
        try:
            job = Job.fetch(job_id, connection=self.connection)
        except Exception:  # NoSuchJobError, or the job expired
            return "lost", None
        status = job.get_status(refresh=True)
        if status == JobStatus.FINISHED:
            return "done", job.return_value()
        if status in (JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELED):
            result = job.latest_result()
            return "failed", (getattr(result, "exc_string", None) or str(status)).strip().splitlines()[-1]
        if status == JobStatus.STARTED:
            return "running", None
        return "queued", None
