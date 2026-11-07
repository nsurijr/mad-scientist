"""The event's rules:

- A live eval runs on its track's public seed (s0, or p0 on the phantom track).
- A team's best is its live eval with the largest pull in the declared direction among those
  valid and undetected. Whenever it changes, the best submission is queued on every hidden scoring
  seed it has not yet run on (background).
- At the freeze, each designated finalist (at most 3 per team) is queued on its missing scoring
  seeds, and each team's best is re-run on the scoring seeds with the cache off (validation).

These functions only decide; Service (service/app.py) records and queues.
"""

from __future__ import annotations

from typing import Any

from service.config import Settings, Track
from service.store import OPEN, Store

SCORED = ("background", "finalist")  # kinds whose results count on the scoring seeds


def signed_pull(ev: dict[str, Any]) -> float | None:
    r = ev.get("result") or {}
    if r.get("pull") is None:
        return None
    return r["pull"] * (1 if r.get("intended_direction") == "+" else -1)


def counts(ev: dict[str, Any]) -> bool:
    """Valid and undetected."""
    r = ev.get("result") or {}
    return ev["status"] == "done" and bool(r.get("valid")) and not r.get("detected")


def best_live(store: Store, team: str, track: Track) -> dict[str, Any] | None:
    targets = {s["id"] for s in store.submissions(team) if s["target"] == track.target}
    good = [e for e in store.evals(team=team, kinds=["live"], statuses=["done"], seed=track.live_seed)
            if e["submission_id"] in targets and counts(e)]
    return max(good, key=lambda e: (signed_pull(e), -e["created"]), default=None)


def _has(store: Store, submission_id: str, seed: str, kinds: tuple[str, ...]) -> bool:
    return any(e["status"] in (*OPEN, "done")
               for e in store.evals(submission_id=submission_id, seed=seed, kinds=kinds))


def background_plan(store: Store, settings: Settings) -> list[tuple[str, str, str]]:
    """(submission_id, seed, kind) to queue so that every team's best runs on the scoring seeds."""
    plan = []
    for team in sorted({s["team"] for s in store.submissions()}):
        for track in settings.tracks().values():
            best = best_live(store, team, track)
            if best is None:
                continue
            plan += [(best["submission_id"], seed, "background") for seed in track.scoring_seeds
                     if not _has(store, best["submission_id"], seed, SCORED)]
    return plan


def freeze_plan(store: Store, settings: Settings) -> list[tuple[str, str, str]]:
    plan = []
    tracks = settings.tracks()
    for sid in store.finalists():
        sub = store.submission(sid)
        track = tracks.get(sub["target"]) if sub else None
        if track is None:
            continue
        plan += [(sid, seed, "finalist") for seed in track.scoring_seeds if not _has(store, sid, seed, SCORED)]
    for team in sorted({s["team"] for s in store.submissions()}):
        for track in tracks.values():
            best = best_live(store, team, track)
            if best is None:
                continue
            plan += [(best["submission_id"], seed, "validation") for seed in track.scoring_seeds
                     if not _has(store, best["submission_id"], seed, ("validation",))]
    return plan


def finalist_problems(store: Store, settings: Settings, team: str, submission_ids: list[str]) -> list[str]:
    problems = []
    if store.flag("frozen"):
        problems.append("the event is frozen: finalists can no longer change")
    if len(set(submission_ids)) > settings.max_finalists:
        problems.append(f"at most {settings.max_finalists} finalists")
    live = {e["submission_id"] for e in store.evals(team=team, kinds=["live"], statuses=["done"])}
    for sid in submission_ids:
        sub = store.submission(sid)
        if sub is None or sub["team"] != team:
            problems.append(f"{sid} is not one of your submissions")
        elif sid not in live:
            problems.append(f"{sid} has no finished live eval")
    return problems
