import base64
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness.cas import Policy, Store
from harness.evaltoken import mint
from harness.gateway import Gateway, Upstream, make_server, summarize

SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 64, "reasoning_effort": "medium"}
SECRET = b"s" * 64
BODY = {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "count the events"}], **SAMPLING}


class FakeUpstream:
    """A model server that answers every request with a numbered reply and remembers what it got."""

    def __init__(self):
        self.requests = []
        self.headers = []
        self.fail = []  # statuses to answer with first, one per request, e.g. [429, 503]
        self.delay = 0.0
        self.active = self.peak = 0
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                upstream.headers.append(dict(self.headers))
                upstream.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if upstream.fail:
                    status = upstream.fail.pop(0)
                    self.send_response(status)
                    if status == 429:
                        self.send_header("Retry-After", "7")
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"{}")
                    return
                upstream.active += 1
                upstream.peak = max(upstream.peak, upstream.active)
                time.sleep(upstream.delay)
                upstream.active -= 1
                data = json.dumps({"id": f"chatcmpl-{len(upstream.requests)}",
                                   "choices": [{"message": {"content": f"reply {len(upstream.requests)}"}}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


def start(gateway):
    server = make_server(gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def post(url, path, body, tag="s0/S1", token=None):
    headers = {"Content-Type": "application/json", "X-Eval": tag}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get(url, path, token=None):
    req = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def gateway(tmp_path, mode, upstream=None, ns="golden", date="2026-11-23", log="gw.jsonl", upstreams=None):
    policy = Policy.recording(ns) if mode == "record" else Policy.replay_only(ns)
    upstreams = upstreams or {"*": Upstream(upstream, "b5c939de", {"system_start_date": date})}
    return Gateway(Store(tmp_path / "cas", policy), mode, upstreams=upstreams, sampling=SAMPLING,
                   container_digest="c" * 64, harness_seed=0, log_path=tmp_path / log,
                   secret=SECRET if mode == "eval" else None)


@pytest.fixture
def upstream():
    u = FakeUpstream()
    yield u
    u.server.shutdown()


def test_record_stores_live_replies_and_serves_repeats_from_the_store(tmp_path, upstream):
    url, server = start(gateway(tmp_path, "record", upstream.url))
    status, first = post(url, "/v1/chat/completions", BODY)
    assert status == 200 and json.loads(first)["choices"][0]["message"]["content"] == "reply 1"
    assert post(url, "/v1/chat/completions", BODY) == (200, first)  # same request: from the store
    other = {**BODY, "messages": BODY["messages"] + [{"role": "assistant", "content": "ok"}]}
    assert json.loads(post(url, "/v1/chat/completions", other)[1])["id"] == "chatcmpl-2"
    assert len(upstream.requests) == 2
    server.shutdown()
    assert summarize(tmp_path / "gw.jsonl")["total"] == {"llm_live": 2, "llm_hit": 1}


def test_sampling_is_enforced_whatever_the_client_sends(tmp_path, upstream):
    url, server = start(gateway(tmp_path, "record", upstream.url))
    post(url, "/v1/chat/completions", {**BODY, "temperature": 0.7, "seed": 5})
    assert upstream.requests[0]["temperature"] == 0.0 and upstream.requests[0]["seed"] == 0
    post(url, "/v1/chat/completions", BODY)  # same key once enforced
    assert len(upstream.requests) == 1
    server.shutdown()


def test_strict_replay_serves_a_sealed_recording_and_halts_on_a_miss(tmp_path, upstream):
    url, server = start(gateway(tmp_path, "record", upstream.url))
    _, recorded = post(url, "/v1/chat/completions", BODY)
    server.shutdown()
    Store(tmp_path / "cas", Policy.recording("golden")).seal("golden")

    url, server = start(gateway(tmp_path, "replay", log="replay.jsonl"))  # no upstream at all
    assert post(url, "/v1/chat/completions", BODY) == (200, recorded)
    status, body = post(url, "/v1/chat/completions", {**BODY, "messages": [{"role": "user", "content": "new"}]})
    assert status == 409 and json.loads(body)["error"]["type"] == "replay_miss"
    server.shutdown()
    assert summarize(tmp_path / "replay.jsonl")["total"] == {"llm_hit": 1, "llm_miss": 1}


def test_the_pinned_system_date_is_part_of_the_key(tmp_path):
    a = gateway(tmp_path, "replay").llm_request_key(BODY)[1]
    b = gateway(tmp_path, "replay", date="2026-10-21").llm_request_key(BODY)[1]
    assert a != b


def test_tool_calls_are_recorded_with_their_output_files_and_replayed(tmp_path, upstream):
    call = {"tool": "execute_python", "tool_version": "1", "args": {"filename": "fit.py", "cli_args": ""},
            "manifest": [["work/fit.py", "a" * 64], ["data/c_data.csv", "b" * 64]]}
    url, server = start(gateway(tmp_path, "record", upstream.url))
    assert json.loads(post(url, "/v1/tools/lookup", call)[1])["hit"] is False
    files = {"scores.csv": base64.b64encode(b"index,score\n0,0.5\n").decode()}
    status, _ = post(url, "/v1/tools/record", {**call, "result": "Exit status: 0", "files": files, "deleted": ["old.txt"]})
    assert status == 200
    server.shutdown()

    url, server = start(gateway(tmp_path, "replay", log="replay.jsonl"))
    hit = json.loads(post(url, "/v1/tools/lookup", call)[1])
    assert hit["hit"] and hit["result"] == "Exit status: 0" and hit["deleted"] == ["old.txt"]
    assert base64.b64decode(hit["files"]["scores.csv"]) == b"index,score\n0,0.5\n"
    changed = {**call, "manifest": [["work/fit.py", "a" * 64], ["data/c_data.csv", "f" * 64]]}  # an input changed
    assert post(url, "/v1/tools/lookup", changed)[0] == 409
    assert post(url, "/v1/tools/record", {**call, "result": "x"})[0] == 403  # replay never writes
    server.shutdown()


def test_streaming_and_malformed_requests_are_refused(tmp_path, upstream):
    url, server = start(gateway(tmp_path, "record", upstream.url))
    assert post(url, "/v1/chat/completions", {**BODY, "stream": True})[0] == 400
    assert post(url, "/v1/chat/completions", {"model": "m"})[0] == 400  # no messages
    assert post(url, "/v1/nowhere", {})[0] == 404
    server.shutdown()


def test_concurrent_agents_share_one_gateway(tmp_path, upstream):
    """Several staged runs call one gateway at once (PARALLEL=3 batches)."""
    from concurrent.futures import ThreadPoolExecutor

    url, server = start(gateway(tmp_path, "record", upstream.url))
    bodies = [{**BODY, "messages": [{"role": "user", "content": f"run {i}"}]} for i in range(48)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        first = list(pool.map(lambda b: post(url, "/v1/chat/completions", b), bodies))
        again = list(pool.map(lambda b: post(url, "/v1/chat/completions", b), bodies))
    server.shutdown()
    assert all(status == 200 for status, _ in first) and first == again
    assert len(upstream.requests) == 48


# -- eval mode ---------------------------------------------------------------------------------------

def record_golden(tmp_path, upstream, *bodies):
    """Record bodies into golden through a record-mode gateway, then seal it."""
    url, server = start(gateway(tmp_path, "record", upstream.url, log="golden.jsonl"))
    replies = [post(url, "/v1/chat/completions", b)[1] for b in bodies]
    server.shutdown()
    Store(tmp_path / "cas", Policy.recording("golden")).seal("golden")
    return replies


def token(team="team-03", eval_id="e1", **kw):
    return mint(SECRET, team=team, eval_id=eval_id, seed="s0", **kw)


def control(team="team-03", eval_id="e1"):
    return mint(SECRET, team=team, eval_id=eval_id, seed="s0", role="control")


def diverge(url, eval_id="e1", team="team-03", at="S3 on_tool_result"):
    return post(url, f"/v1/evals/{eval_id}/diverge", {"at": at}, token=control(team, eval_id))


ATTACKED = {**BODY, "messages": BODY["messages"] + [{"role": "user", "content": "the reference has 99 events"}]}


def test_eval_mode_needs_a_valid_pipeline_token(tmp_path, upstream):
    url, server = start(gateway(tmp_path, "eval", upstream.url))
    assert post(url, "/v1/chat/completions", BODY)[0] == 401
    forged = token()[:-4] + "AAAA"
    assert post(url, "/v1/chat/completions", BODY, token=forged)[0] == 401
    assert post(url, "/v1/chat/completions", BODY, token=control())[0] == 403  # control tokens make no calls
    server.shutdown()
    assert upstream.requests == []


def test_an_eval_is_strict_until_the_runner_marks_its_first_hook_change(tmp_path, upstream):
    [golden_reply] = record_golden(tmp_path, upstream, BODY)
    url, server = start(gateway(tmp_path, "eval", upstream.url))
    assert post(url, "/v1/chat/completions", BODY, token=token()) == (200, golden_reply)
    status, body = post(url, "/v1/chat/completions", ATTACKED, token=token())
    error = json.loads(body)["error"]
    assert status == 409 and error["type"] == "replay_miss"
    assert error["report"]["first_difference"] == 1 and "99 events" in error["report"]["diff"]
    assert len(upstream.requests) == 1  # only the golden recording reached the model

    assert diverge(url)[0] == 200
    status, live = post(url, "/v1/chat/completions", ATTACKED, token=token())
    assert status == 200 and len(upstream.requests) == 2
    assert post(url, "/v1/chat/completions", ATTACKED, token=token()) == (200, live)  # now from team-03
    state = get(url, "/v1/evals/e1", control())[1]
    assert state["strict"] is False and state["diverged_at"] == "S3 on_tool_result"
    assert state["first_live_call"]["kind"] == "llm" and state["live_tokens"] == 12
    server.shutdown()
    hits = [json.loads(line) for line in (tmp_path / "gw.jsonl").read_text().splitlines()]
    assert [h.get("ns") for h in hits if h["outcome"] == "hit"] == ["golden", "team-03"]


def test_teams_read_golden_and_their_own_namespace_only(tmp_path, upstream):
    record_golden(tmp_path, upstream, BODY)
    url, server = start(gateway(tmp_path, "eval", upstream.url))
    diverge(url, "e1", "team-03")
    diverge(url, "e2", "team-04")
    post(url, "/v1/chat/completions", ATTACKED, token=token("team-03", "e1"))
    post(url, "/v1/chat/completions", ATTACKED, token=token("team-04", "e2"))  # not served team-03's reply
    assert len(upstream.requests) == 3
    assert post(url, "/v1/chat/completions", BODY, token=token("team-04", "e2"))[0] == 200  # golden is shared
    assert len(upstream.requests) == 3
    assert get(url, "/v1/evals/e1", control("team-04", "e2"))[0] == 403  # another eval's control token
    assert get(url, "/v1/evals/e1", token("team-03", "e1"))[0] == 403  # a pipeline token
    server.shutdown()


def test_budgets_and_allowed_models_are_enforced(tmp_path, upstream):
    url, server = start(gateway(tmp_path, "eval", upstream.url))
    diverge(url)
    capped = token(token_cap=10)
    assert post(url, "/v1/chat/completions", BODY, token=capped)[0] == 200  # 12 tokens: now over the cap
    status, body = post(url, "/v1/chat/completions", ATTACKED, token=capped)
    assert status == 429 and json.loads(body)["error"]["budget"] == "tokens"
    expired = token(eval_id="e2", now=0)
    assert json.loads(post(url, "/v1/chat/completions", BODY, token=expired)[1])["error"]["budget"] == "wall"
    status, body = post(url, "/v1/chat/completions", {**BODY, "model": "qwen3-32b"}, token=token(eval_id="e3"))
    assert status == 403 and json.loads(body)["error"]["type"] == "model_not_allowed"
    server.shutdown()
    assert len(upstream.requests) == 1


def test_a_preview_stops_at_the_first_call_that_would_run_live(tmp_path, upstream):
    record_golden(tmp_path, upstream, BODY)
    url, server = start(gateway(tmp_path, "eval", upstream.url))
    preview = token(live=False)
    assert post(url, "/v1/chat/completions", BODY, token=preview)[0] == 200
    diverge(url)
    status, body = post(url, "/v1/chat/completions", ATTACKED, token=preview)
    error = json.loads(body)["error"]
    assert status == 409 and error["type"] == "preview_stop" and error["report"]["first_difference"] == 1
    state = get(url, "/v1/evals/e1", control())[1]
    assert state["first_live_call"] == {"call": 2, "eval": "s0/S1", "kind": "llm"}
    assert state["stopped"]["error"] == "preview_stop"
    server.shutdown()
    assert len(upstream.requests) == 1


def test_tool_calls_in_an_eval_follow_strict_mode_and_record_to_the_team(tmp_path, upstream):
    call = {"tool": "execute_python", "tool_version": "1", "args": {"filename": "fit.py", "cli_args": ""},
            "manifest": [["work/fit.py", "a" * 64]]}
    url, server = start(gateway(tmp_path, "eval", upstream.url))
    assert post(url, "/v1/tools/lookup", call, token=token())[0] == 409
    diverge(url)
    assert json.loads(post(url, "/v1/tools/lookup", call, token=token())[1])["hit"] is False
    assert post(url, "/v1/tools/record", {**call, "result": "ok"}, token=token())[0] == 200
    hit = json.loads(post(url, "/v1/tools/lookup", call, token=token())[1])
    assert hit["hit"] and hit["result"] == "ok"
    server.shutdown()
    assert Store(tmp_path / "cas", Policy.replay_only("team-03")).get(hit["key"]) is not None


def test_upstreams_map_model_names_keys_and_api_keys(tmp_path, upstream):
    ups = {"gpt-oss-120b": Upstream(upstream.url, "b5c939de", {"system_start_date": "2026-11-23"},
                                    served_name="openai/gpt-oss-120b", api_key="organizer-key")}
    url, server = start(gateway(tmp_path, "record", upstreams=ups))
    assert post(url, "/v1/chat/completions", BODY)[0] == 200
    assert post(url, "/v1/chat/completions", {**BODY, "model": "gemma-4-31b-it"})[0] == 404
    server.shutdown()
    assert upstream.requests[0]["model"] == "openai/gpt-oss-120b"
    assert upstream.headers[0]["Authorization"] == "Bearer organizer-key"
    other_backend = {"gpt-oss-120b": Upstream(None, "event-api-2026-11", {"system_start_date": "2026-11-23"})}
    assert (gateway(tmp_path, "replay", upstreams=ups).llm_request_key(BODY)[1]
            != gateway(tmp_path, "replay", upstreams=other_backend).llm_request_key(BODY)[1])


def test_a_replay_miss_reports_the_nearest_recorded_request(tmp_path, upstream):
    record_golden(tmp_path, upstream, BODY, {**BODY, "messages": [{"role": "user", "content": "something else"}]})
    url, server = start(gateway(tmp_path, "replay", log="replay.jsonl"))
    status, body = post(url, "/v1/chat/completions", ATTACKED)
    report = json.loads(body)["error"]["report"]
    assert status == 409 and report["messages"] == [1, 2] and report["first_difference"] == 1
    server.shutdown()


def test_the_gateway_also_answers_on_a_unix_socket(tmp_path):
    import http.client
    import shutil
    import socket

    from harness.gateway import make_unix_server
    from harness.uds import serve_in_thread, short_socket_dir

    gw = Gateway(Store(tmp_path, Policy.replay_only("golden")), "replay", upstreams={"*": Upstream(None, "r", {})},
                 sampling={}, container_digest="c" * 64, harness_seed=0)
    folder = short_socket_dir("gw-")
    server = serve_in_thread(make_unix_server(gw, folder / "gateway.sock"))

    class Conn(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX)
            self.sock.connect(str(folder / "gateway.sock"))

    try:
        conn = Conn("localhost", timeout=5)
        conn.request("GET", "/health")
        r = conn.getresponse()
        assert r.status == 200 and json.loads(r.read())["mode"] == "replay"
        conn.request("POST", "/v1/tools/lookup", body=json.dumps({"tool": "execute_python", "tool_version": "1",
                     "args": {}, "manifest": []}), headers={"Content-Type": "application/json", "X-Eval": "s0/S1"})
        r = conn.getresponse()
        assert r.status == 409  # strict replay: the same handler as on the port
        r.read()
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(folder, ignore_errors=True)


def test_a_shared_api_is_retried_on_rate_limits_and_server_errors(tmp_path, upstream):
    gw = gateway(tmp_path, "record", upstreams={"*": Upstream(upstream.url, "r", {}, retries=3)})
    waits = []
    gw.sleep = waits.append
    url, server = start(gw)
    upstream.fail = [429, 503]
    status, data = post(url, "/v1/chat/completions", BODY)
    assert status == 200 and len(upstream.requests) == 3
    assert waits == [7.0, 4.0]  # Retry-After on the 429, then doubling waits (2 s, 4 s, ...)
    rec = [json.loads(l) for l in (tmp_path / "gw.jsonl").read_text().splitlines()][-1]
    assert rec["outcome"] == "live" and rec["retries"] == 2
    upstream.fail = [500, 500]  # past the limit the error reaches the client, and nothing is stored
    (tmp_path / "b").mkdir()
    gw2 = gateway(tmp_path / "b", "record", upstreams={"*": Upstream(upstream.url, "r", {}, retries=1)})
    gw2.sleep = waits.append
    url2, server2 = start(gw2)
    status, _ = post(url2, "/v1/chat/completions", {**BODY, "max_tokens": 7})
    assert status == 500
    upstream.fail = [400]  # a refused request is never retried
    status, _ = post(url2, "/v1/chat/completions", {**BODY, "max_tokens": 8})
    assert status == 400
    server.shutdown()
    server2.shutdown()


def test_dropped_params_stay_in_the_key_but_are_not_sent(tmp_path, upstream):
    url, server = start(gateway(tmp_path, "record", upstreams={"*": Upstream(upstream.url, "r", {},
                                                                            drop_params=("seed",))}))
    post(url, "/v1/chat/completions", BODY)
    assert "seed" not in upstream.requests[0] and upstream.requests[0]["temperature"] == 0.0
    post(url, "/v1/chat/completions", BODY)
    assert len(upstream.requests) == 1  # the same key: served from the store
    server.shutdown()


def test_requests_in_flight_to_an_upstream_are_capped(tmp_path, upstream):
    url, server = start(gateway(tmp_path, "record", upstreams={"*": Upstream(upstream.url, "r", {},
                                                                            max_concurrency=2)}))
    upstream.delay = 0.2
    threads = [threading.Thread(target=post, args=(url, "/v1/chat/completions", {**BODY, "max_tokens": 10 + i}))
               for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(upstream.requests) == 6 and upstream.peak == 2
    server.shutdown()


def test_upstreams_files_carry_the_shared_api_settings(tmp_path, monkeypatch):
    from harness.gateway import load_upstreams

    monkeypatch.setenv("ILAAS_API_KEY", "sk-test-not-a-key")
    path = tmp_path / "upstreams.json"
    path.write_text(json.dumps({"*": {"base_url": "https://llm.aristote.education/v1", "model_rev": "ilaas-test",
                                      "server": {"backend": "ilaas"}, "api_key_env": "ILAAS_API_KEY",
                                      "retries": 6, "max_concurrency": 16, "drop_params": ["seed"]}}))
    u = load_upstreams(path)["*"]
    assert (u.api_key, u.retries, u.max_concurrency, u.drop_params) == ("sk-test-not-a-key", 6, 16, ("seed",))
    assert "sk-test" not in repr(u)  # the key never appears in logs or reprs

