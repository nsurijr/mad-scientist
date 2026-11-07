import io
import json
import tarfile

import pytest
import yaml

pytest.importorskip("fastapi")
fakeredis = pytest.importorskip("fakeredis")
from fastapi.testclient import TestClient  # noqa: E402

from service import auth, bundle, traces, worker  # noqa: E402
from service.app import Service, create_app  # noqa: E402
from service.config import Settings  # noqa: E402
from service.queue import Queues  # noqa: E402
from service.runners import FakeRunner, HarnessRunner  # noqa: E402

TEAMS = ["team-01", "team-02"]


def make_sub(root, name="a1", team="team-01", direction="+", target="n_excess", fake=None, hooks=("on_context",),
             stages=("S2",)):
    d = root / f"{team}-{name}"
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(yaml.safe_dump({
        "team": team, "name": name, "attack_class": "injection", "stages_touched": list(stages),
        "hooks": list(hooks), "intended_direction": direction, "target": target, "shared": {"fake": fake or {}}}))
    (d / "attack.py").write_text("from harness.attack import Attack\n\n\nclass A(Attack):\n    pass\n")
    (d / "WRITEUP.md").write_text("what it does\n")
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVICE_WORK", str(tmp_path / "work"))
    monkeypatch.setattr(worker, "RUNNER", FakeRunner())

    def build(is_async=False, **kw):
        tokens = auth.mint(TEAMS, tmp_path / "tokens.json") if not (tmp_path / "tokens.json").exists() else build.tokens
        build.tokens = tokens
        settings = Settings(home=tmp_path / "home", tokens_file=tmp_path / "tokens.json", poll_s=0, **kw)
        svc = Service(settings, queues=Queues(fakeredis.FakeStrictRedis(), is_async=is_async))
        client = TestClient(create_app(service=svc))
        hdr = {t: {"Authorization": f"Bearer {tok}"} for t, tok in tokens.items()}
        return svc, client, hdr

    build.root = tmp_path / "subs"
    return build


def submit(client, hdr, folder, preview=False):
    return client.post(f"/api/submissions?preview={'true' if preview else 'false'}", content=bundle.pack(folder),
                       headers={**hdr, "Content-Type": "application/gzip"})


def test_bundles_unpack_safely_and_strip_the_top_folder(tmp_path):
    folder = make_sub(tmp_path / "src")
    (folder / "assets").mkdir()
    (folder / "assets" / "note.txt").write_text("x")
    out = bundle.unpack(bundle.pack(folder), tmp_path / "out")
    assert sorted(p.name for p in out.iterdir()) == ["WRITEUP.md", "assets", "attack.py", "manifest.yaml"]
    for name, kind in (("../evil.py", tarfile.REGTYPE), ("/etc/evil", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
            tar.addfile(info, io.BytesIO(b"") if kind == tarfile.REGTYPE else None)
        with pytest.raises(bundle.BundleError):
            bundle.unpack(buf.getvalue(), tmp_path / f"bad-{kind!r}-{len(name)}")
    with pytest.raises(bundle.BundleError, match="more than"):
        bundle.unpack(bundle.pack(folder), tmp_path / "small", max_files=2)


def test_static_check_wants_the_tokens_team_and_an_open_target(tmp_path):
    folder = make_sub(tmp_path, team="team-02", target="significance")
    manifest, problems = bundle.static_check(folder, "team-01", Settings(home=tmp_path).tracks())
    assert manifest is None
    assert any("team-02" in p for p in problems) and any("not open" in p for p in problems)


def test_tokens_are_stored_hashed_and_looked_up(tmp_path):
    tokens = auth.mint(["team-01"], tmp_path / "t.json")
    stored = (tmp_path / "t.json").read_text()
    assert tokens["team-01"] not in stored and oct((tmp_path / "t.json").stat().st_mode)[-3:] == "600"
    t = auth.Tokens.load(tmp_path / "t.json")
    assert t.lookup(tokens["team-01"]).team == "team-01" and t.lookup(tokens["organizers"]).organizer
    assert t.lookup("wrong") is None and t.teams == ["team-01"]


def test_a_live_eval_scores_and_its_best_runs_on_the_hidden_seeds(env):
    svc, client, hdr = env()
    assert client.get("/api/status").status_code == 401
    r = submit(client, hdr["team-01"], make_sub(env.root, fake={"pull": 2.5}))
    assert r.status_code == 200, r.text
    ev = r.json()["eval"]
    assert ev["status"] == "done" and ev["seed"] == "s0"
    res = ev["result"]
    assert res["status"] == "valid" and res["detected"] is False and abs(res["pull_vs_honest"] - 2.5) < 1e-9
    assert res["trace"] == f"/api/evals/{ev['id']}/trace" and "truth" not in json.dumps(res)
    # the new best is queued on s1-s5, which the team sees only as progress
    hidden = svc.store.evals(kinds=["background"])
    assert sorted(e["seed"] for e in hidden) == ["s1", "s2", "s3", "s4", "s5"] and all(e["status"] == "done" for e in hidden)
    status = client.get("/api/status", headers=hdr["team-01"]).json()
    assert [e["kind"] for e in status["evals"]] == ["live"]
    assert status["scoring_seed_progress"] == {"team-01-001": {"done": 5, "queued_or_running": 0}}
    assert client.get(f"/api/evals/{hidden[0]['id']}", headers=hdr["team-01"]).status_code == 404
    assert client.get(f"/api/evals/{hidden[0]['id']}", headers=hdr["organizers"]).json()["result"]["seed"] == "s1"
    board = client.get("/api/leaderboard").json()
    row = next(r for r in board["tracks"]["n_excess"]["rows"] if r["team"] == "team-01")
    assert row["best"]["name"] == "a1" and row["scoring_seeds_done"] == "5/5" and "final" not in row
    assert [r["team"] for r in board["tracks"]["n_excess"]["rows"]] == TEAMS  # team-02 listed with no attack yet
    page = client.get(f"/api/evals/{ev['id']}/trace", headers=hdr["team-01"])
    assert page.status_code == 200 and "Researcher" in page.text
    assert client.get(f"/api/evals/{ev['id']}/trace", headers=hdr["team-02"]).status_code == 404


def test_only_valid_undetected_results_become_the_best(env):
    svc, client, hdr = env()
    submit(client, hdr["team-01"], make_sub(env.root, "ok", fake={"pull": 1.5}))
    submit(client, hdr["team-01"], make_sub(env.root, "flagged", fake={"pull": 4.0, "hard_flags": ["S4: recount differs"]}))
    submit(client, hdr["team-01"], make_sub(env.root, "wrongway", direction="-", fake={"pull": 3.0}))
    submit(client, hdr["team-01"], make_sub(env.root, "better", fake={"pull": 2.0}))
    best = client.get("/api/leaderboard").json()["tracks"]["n_excess"]["rows"][0]["best"]
    assert best["name"] == "better"
    flagged = svc.store.evals(submission_id="team-01-002")[0]
    assert flagged["team_view"]["detected"] is True
    background = {e["submission_id"] for e in svc.store.evals(kinds=["background"])}
    assert background == {"team-01-001", "team-01-004"}  # "ok" was best first, then "better"


def test_rejections_and_failures_do_not_count_against_the_cap(env):
    svc, client, hdr = env(submission_cap=1)
    r = submit(client, hdr["team-01"], make_sub(env.root, "bad", fake={"reject": ["hooks are not deterministic"]}))
    assert r.json()["eval"]["status"] == "rejected"
    assert r.json()["eval"]["check"]["problems"] == ["hooks are not deterministic"]
    r = submit(client, hdr["team-01"], make_sub(env.root, "broken", fake={"fail": "no GPU"}))
    assert r.json()["eval"]["status"] == "failed" and "no GPU" not in r.json()["eval"]["error"]
    assert svc.store.evals(submission_id="team-01-002")[0]["error"].endswith("no GPU")
    r = submit(client, hdr["team-01"], make_sub(env.root, "preview-only"), preview=True)
    assert r.json()["eval"]["kind"] == "preview" and r.json()["eval"]["preview"]["estimated_reexecution_fraction"] == 0.6
    assert submit(client, hdr["team-01"], make_sub(env.root, "first")).status_code == 200
    r = submit(client, hdr["team-01"], make_sub(env.root, "second"))
    assert r.status_code == 429 and "cap" in r.json()["detail"]


def test_at_most_three_evals_in_flight_per_team(env):
    svc, client, hdr = env(is_async=True)  # no worker: evals stay queued
    for i in range(3):
        assert submit(client, hdr["team-01"], make_sub(env.root, f"q{i}")).json()["eval"]["status"] == "queued"
    r = submit(client, hdr["team-01"], make_sub(env.root, "q3"))
    assert r.status_code == 429 and "already queued" in r.json()["detail"]
    assert submit(client, hdr["team-02"], make_sub(env.root, "other", team="team-02")).status_code == 200


def test_manifest_must_match_the_token_and_open_tracks(env):
    svc, client, hdr = env()
    r = submit(client, hdr["team-01"], make_sub(env.root, team="team-02"))
    assert r.status_code == 400 and "team-02" in json.dumps(r.json())
    r = submit(client, hdr["team-01"], make_sub(env.root, "phantom", target="significance"))
    assert r.status_code == 400
    svc2, client2, hdr2 = env(phantom=True)
    r = submit(client2, hdr2["team-01"], make_sub(env.root, "phantom2", target="significance", fake={"pull": 1.0}))
    assert r.json()["eval"]["seed"] == "p0"
    assert {e["seed"] for e in svc2.store.evals(kinds=["background"])} == {"p1", "p2", "p3", "p4", "p5"}


def test_finalists_freeze_validation_and_reveal(env):
    svc, client, hdr = env()
    for name, pull in (("a", 1.0), ("b", 2.0), ("c", 1.5), ("d", 1.2)):
        submit(client, hdr["team-01"], make_sub(env.root, name, fake={"pull": pull}))
    r = client.put("/api/finalists", json={"submissions": ["team-01-001", "team-01-002", "team-01-003", "team-01-004"]},
                   headers=hdr["team-01"])
    assert r.status_code == 400
    r = client.put("/api/finalists", json={"submissions": ["team-01-001", "team-01-002"]}, headers=hdr["team-02"])
    assert r.status_code == 400
    submit(client, hdr["team-01"], make_sub(env.root, "seen"), preview=True)  # team-01-005: preview only
    r = client.put("/api/finalists", json={"submissions": ["team-01-005"]}, headers=hdr["team-01"])
    assert r.status_code == 400 and "no finished live eval" in json.dumps(r.json())
    assert client.put("/api/finalists", json={"submissions": ["team-01-002", "team-01-003"]},
                      headers=hdr["team-01"]).status_code == 200
    assert client.post("/api/admin/freeze", headers=hdr["team-01"]).status_code == 403
    out = client.post("/api/admin/freeze", headers=hdr["organizers"]).json()
    # c was never the team's best, so it has no scoring-seed runs yet and runs as a finalist; the
    # team's best, b, already has them, and is validated with the cache off on every scoring seed
    assert sorted(q for q in out["queued"] if q.startswith("finalist")) == [f"finalist team-01-003 s{i}" for i in range(1, 6)]
    assert sorted(q for q in out["queued"] if q.startswith("validation")) == [f"validation team-01-002 s{i}" for i in range(1, 6)]
    assert submit(client, hdr["team-01"], make_sub(env.root, "late")).status_code == 409
    assert client.put("/api/finalists", json={"submissions": ["team-01-003"]}, headers=hdr["team-01"]).status_code == 400
    ranking = client.get("/api/admin/ranking", headers=hdr["organizers"]).json()["n_excess"]
    assert ranking["rows"][0]["submission_id"] == "team-01-002" and ranking["rows"][0]["median_effect"] == pytest.approx(2.0)
    assert ranking["best_per_attack_class"]["injection"]["team"] == "team-01"
    assert "final" not in client.get("/api/leaderboard").json()["tracks"]["n_excess"]["rows"][0]
    client.post("/api/admin/reveal", headers=hdr["organizers"])
    row = client.get("/api/leaderboard").json()["tracks"]["n_excess"]["rows"][0]
    assert row["final"]["submission_id"] == "team-01-002" and row["final"]["scored"] == "5/5"
    hidden = svc.store.evals(kinds=["background"], submission_id="team-01-002")[0]
    assert client.get(f"/api/evals/{hidden['id']}", headers=hdr["team-01"]).status_code == 200  # after the reveal


def test_a_disagreeing_validation_run_is_scored_live(env):
    svc, client, hdr = env()
    folder = make_sub(env.root, "x", fake={"pull": 2.0})
    submit(client, hdr["team-01"], folder)
    # the cache-off re-runs come out lower on every seed
    worker.RUNNER = _Lower(FakeRunner())
    client.post("/api/admin/freeze", headers=hdr["organizers"])
    row = client.get("/api/admin/ranking", headers=hdr["organizers"]).json()["n_excess"]["rows"][0]
    assert row["median_effect"] == pytest.approx(1.0) and row["validation_disagrees"] == ["s1", "s2", "s3", "s4", "s5"]


class _Lower:
    def __init__(self, inner):
        self.inner = inner

    def check(self, submission):
        return self.inner.check(submission)

    def evaluate(self, submission, seed, eval_id, team, work, cache=True):
        score, digest = self.inner.evaluate(submission, seed, eval_id, team, work, cache)
        if not cache:
            score = {**score, "pull": score["pull"] - 1.0}
        return score, digest


def test_trace_digest_marks_hits_and_live_calls(tmp_path):
    run = tmp_path / "s0"
    (run / "S2" / "trace").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"seed": "s0", "status": "complete", "attack": {"diverged_at": "S2 x"},
                                              "stages": [{"stage": "S2", "status": "complete"}]}))
    sys_msg, brief = {"role": "system", "content": "you are"}, {"role": "user", "content": "<b>brief</b>"}
    reply = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "handoff_to_coder", "arguments": "{}"}}]}
    recs = [
        {"agent": "Researcher", "request": {"messages": [sys_msg, brief]},
         "response": {"choices": [{"message": {**reply, "reasoning": "think"}}]}},
        {"agent": "Researcher", "request": {"messages": [sys_msg, brief, reply, {"role": "tool", "tool_call_id": "c1", "content": "done"}]},
         "response": {"choices": [{"message": {"role": "assistant", "content": "finished"}}]}},
    ]
    (run / "S2" / "trace" / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    log = [{"eval": "s0/S1", "kind": "llm", "outcome": "hit", "eval_id": "e"},
           {"eval": "s0/S2", "kind": "llm", "outcome": "hit", "eval_id": "e"},
           {"eval": "s0/S2", "kind": "tool", "outcome": "live", "eval_id": "e"},
           {"eval": "s0/S2", "kind": "llm", "outcome": "live", "eval_id": "e"}]
    d = traces.digest(run, log)
    calls = d["stages"][0]["calls"]
    assert [c["outcome"] for c in calls] == ["hit", "live"]
    assert [m["role"] for m in calls[0]["new"]] == ["system", "user"] and calls[1]["new"] == [
        {"role": "assistant", "content": ""}, {"role": "tool", "content": "done"}]
    page = traces.render(d, "t")
    assert "&lt;b&gt;brief&lt;/b&gt;" in page and "<b>brief</b>" not in page


def test_the_harness_runner_needs_its_settings():
    with pytest.raises(ValueError, match="SERVICE_GATEWAY_URL"):
        HarnessRunner(env={})
