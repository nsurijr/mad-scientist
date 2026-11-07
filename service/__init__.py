"""The event service: submissions, the eval queue, results, leaderboard, trace viewer.

    SERVICE_HOME=state uvicorn --factory service.app:create_app     # API and pages
    SERVICE_HOME=state SERVICE_RUNNER=harness python -m service.worker   # eval worker(s)
    hack submit <submission dir>                                     # teams (service/cli.py)

Parts:
- app.py: the HTTP API (FastAPI) and the two pages, leaderboard and trace viewer.
- store.py: submissions, evals, finalists and event flags in SQLite, owned by the API side.
- queue.py: two RQ queues, live (high) and background (low). A job carries the submission's
  bundle and returns its result, so workers never touch the store and may run on another machine
  (where they run is not decided yet).
- worker.py and runners.py: what a job does. FakeRunner for tests and local trials; HarnessRunner
  calls the harness (validator, stage runner, scorer) against a gateway in eval mode.
- policy.py: the event's rules: which seeds an attack runs on, a team's best result, the
  background runs it triggers, freeze and finalists.
- leaderboard.py and traces.py: what teams and organizers see.
- auth.py: one bearer token per team plus an organizer token (JupyterHub tokens later).

Teams see their live and preview results and traces. Runs on the hidden scoring seeds (s1–s5,
p1–p5) show only as progress (k/5) until the organizers reveal them after the freeze. There is
no monitor at the event, so a result is detected when it has a hard flag.
"""
