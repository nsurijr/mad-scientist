"""The public leaderboard and the organizers' ranking.

Public, per team and track: its best live result (suspicion, detected, pull), live evals used out
of the cap, live tokens spent, and how many hidden scoring seeds its best has finished (k/5).
Scores on the hidden seeds appear only after the organizers reveal them.

Ranking (organizers, and public after the reveal): per team, the best of its candidates by the
median pull over the scoring seeds counting only valid, undetected runs (harness.scorer.aggregate).
Candidates are its finalists, or, without finalists, every submission that ran on a scoring seed.
Where a cache-off validation run disagrees with the replayed one by more than the tolerance, the
validation run's result counts.
"""

from __future__ import annotations

import time
from typing import Any

from harness.scorer import aggregate, overall
from service import policy
from service.config import Settings, Track
from service.store import Store


def _seed_scores(store: Store, sid: str, track: Track, tolerance: float) -> tuple[list[dict], list[str]]:
    scores, disagreements = [], []
    for seed in track.scoring_seeds:
        done = [e for e in store.evals(submission_id=sid, seed=seed, statuses=["done"]) if e.get("result")]
        replay = next((e["result"] for e in reversed(done) if e["kind"] in policy.SCORED), None)
        live = next((e["result"] for e in reversed(done) if e["kind"] == "validation"), None)
        chosen = replay
        if live is not None and (replay is None or replay.get("pull") is None or live.get("pull") is None
                                 or abs(live["pull"] - replay["pull"]) > tolerance):
            chosen = live
            if replay is not None:
                disagreements.append(seed)
        if chosen is not None:
            scores.append(chosen)
    return scores, disagreements


def _candidates(store: Store, team: str, track: Track) -> list[str]:
    subs = {s["id"] for s in store.submissions(team) if s["target"] == track.target}
    finalists = [s for s in store.finalists(team) if s in subs]
    if finalists:
        return finalists
    ran = {e["submission_id"] for e in store.evals(team=team, kinds=[*policy.SCORED, "validation"], statuses=["done"])}
    return sorted(subs & ran)


def ranking(store: Store, settings: Settings) -> dict[str, Any]:
    writeups = store.writeups()
    out: dict[str, Any] = {}
    for name, track in settings.tracks().items():
        rows, best_per_class = [], {}
        for team in sorted({s["team"] for s in store.submissions()}):
            options = []
            for sid in _candidates(store, team, track):
                scores, disagree = _seed_scores(store, sid, track, settings.validation_tolerance)
                agg = aggregate(scores, track.scoring_seeds)
                sub = store.submission(sid)
                options.append({**agg, "submission_id": sid, "name": sub["name"], "attack_class": sub["attack_class"],
                                "validation_disagrees": disagree})
            if not options:
                continue
            best = max(options, key=lambda o: (o["median_effect"], o["undetected_fraction"]))
            for o in options:
                cls = o["attack_class"]
                if cls not in best_per_class or o["median_effect"] > best_per_class[cls]["median_effect"]:
                    best_per_class[cls] = {"team": team, **o}
            rows.append({"team": team, **best})
        g = overall([{**r, "submission": r["submission_id"]} for r in rows], writeups)
        for r in rows:
            r["writeup"] = writeups.get(r["submission_id"])
            r["overall"] = g.get(r["submission_id"]) if r["writeup"] is not None else None
        rows.sort(key=lambda r: (-r["median_effect"], -r["undetected_fraction"], r["team"]))
        out[name] = {"rows": rows, "best_per_attack_class": best_per_class}
    return out


def public(store: Store, settings: Settings, teams: list[str]) -> dict[str, Any]:
    revealed = store.flag("revealed")
    final = ranking(store, settings) if revealed else {}
    tracks = {}
    for name, track in settings.tracks().items():
        rows = []
        for team in sorted(set(teams) | {s["team"] for s in store.submissions()}):
            subs = {s["id"]: s for s in store.submissions(team)}
            own = store.evals(team=team, kinds=["preview", "live"])
            used = sum(e["kind"] == "live" and e["status"] in ("queued", "running", "done") for e in own)
            tokens = sum(((e.get("result") or {}).get("live_tokens") or 0) for e in own)
            best = policy.best_live(store, team, track)
            row: dict[str, Any] = {"team": team, "live_used": used, "cap": settings.submission_cap,
                                   "tokens_used": tokens, "best": None, "scoring_seeds_done": None}
            if best is not None:
                r = best["result"]
                done = {e["seed"] for e in store.evals(submission_id=best["submission_id"], kinds=policy.SCORED,
                                                       statuses=["done"])}
                row["best"] = {"submission_id": best["submission_id"], "name": subs[best["submission_id"]]["name"],
                               "suspicion": r["q"], "detected": r["detected"], "pull": r["pull"],
                               "effect": policy.signed_pull(best)}
                row["scoring_seeds_done"] = f"{len(done & set(track.scoring_seeds))}/{len(track.scoring_seeds)}"
            if revealed:
                row["final"] = next(({k: f[k] for k in ("submission_id", "median_effect", "undetected_fraction",
                                                        "scored")}
                                     for f in final.get(name, {}).get("rows", []) if f["team"] == team), None)
            rows.append(row)
        if revealed:
            rows.sort(key=lambda r: (-(r.get("final") or {}).get("median_effect", float("-inf")), r["team"]))
        else:
            rows.sort(key=lambda r: (r["best"] is None, -(r["best"] or {}).get("effect", 0.0), r["team"]))
        tracks[name] = {"live_seed": track.live_seed, "scoring_seeds": list(track.scoring_seeds), "rows": rows}
    return {"updated": time.time(), "frozen": store.flag("frozen"), "revealed": revealed, "tracks": tracks}
