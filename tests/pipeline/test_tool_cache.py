import json
import shutil
import threading

import pytest

from agents.agent import Agent
from harness.cas import Policy, Store
from harness.evaltoken import mint
from harness.gateway import Gateway, Upstream, make_server
from tools.execute_python import ExecutePython
from tools.tool import Tool
from utility import tool_cache
from utility.harness_halt import HarnessHalt
from utility.md_reporter import MDReporter

from test_chat_backend import FakeClient, completion

# Writes an output from a read-only input, and counts its own runs outside the working folder.
SCRIPT = """
import sys
open(sys.argv[2], "a").write("ran\\n")
open("out.txt", "w").write(open(sys.argv[1] + "/x.txt").read().upper())
print("done")
"""


SECRET = b"s" * 64


def serve(tmp_path, mode):
    policy = Policy.recording("golden") if mode == "record" else Policy.replay_only("golden")
    upstream = Upstream("http://unused" if mode == "record" else None, "r", {})
    gw = Gateway(Store(tmp_path / "cas", policy), mode, upstreams={"*": upstream}, sampling={},
                 container_digest="c" * 64, harness_seed=0, secret=SECRET if mode == "eval" else None)
    server = make_server(gw)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "x.txt").write_text("signal")
    monkeypatch.setenv("SAVE_EXECUTION_ENV", "True")
    monkeypatch.setenv(tool_cache.READ_ONLY_ROOTS_ENV, f"data={data}")
    return tmp_path


def work_dir(env, name):
    d = env / name
    d.mkdir()
    (d / "make_out.py").write_text(SCRIPT)
    return str(d) + "/"


def run(env, work, monkeypatch, server):
    monkeypatch.setenv("AOD_TOOL_CACHE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1/tools")
    tool = ExecutePython(work, MDReporter(work))
    return tool.execute_python("make_out.py", f"{env / 'data'} {env / 'runs.log'}")


def test_recorded_execution_is_replayed_without_running(env, monkeypatch):
    server = serve(env, "record")
    first = run(env, work_dir(env, "a"), monkeypatch, server)
    server.shutdown()
    assert "done" in first and (env / "a" / "out.txt").read_text() == "SIGNAL"

    server = serve(env, "replay")
    replayed = run(env, work_dir(env, "b"), monkeypatch, server)
    server.shutdown()
    assert replayed == first
    assert (env / "b" / "out.txt").read_text() == "SIGNAL"  # restored from the store
    assert (env / "runs.log").read_text() == "ran\n"  # the script ran once, during recording


def test_a_changed_input_misses_and_halts_strict_replay(env, monkeypatch):
    server = serve(env, "record")
    run(env, work_dir(env, "a"), monkeypatch, server)
    server.shutdown()
    (env / "data" / "x.txt").write_text("poisoned")
    server = serve(env, "replay")
    with pytest.raises(HarnessHalt, match="strict replay"):
        run(env, work_dir(env, "b"), monkeypatch, server)
    server.shutdown()


def test_in_an_eval_the_tool_cache_sends_the_eval_token(env, monkeypatch):
    server = serve(env, "eval")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(HarnessHalt, match="HTTP 401"):  # no token: the gateway refuses
        run(env, work_dir(env, "a"), monkeypatch, server)
    monkeypatch.setenv("OPENAI_API_KEY", mint(SECRET, team="team-03", eval_id="e1", seed="s0"))
    with pytest.raises(HarnessHalt, match="strict replay"):  # accepted, and strict before any hook change
        run(env, work_dir(env, "b"), monkeypatch, server)
    server.shutdown()


def test_aod_logs_do_not_change_the_key(env):
    work = work_dir(env, "a")
    before = tool_cache.manifest(tool_cache.snapshot(work))
    MDReporter(work).report_metrics("execution_time", 1.234, mode="overwrite")  # timings, every call
    (env / "a" / "conversation.md").write_text("# User Message\n")
    assert tool_cache.manifest(tool_cache.snapshot(work)) == before
    (env / "a" / "notes.txt").write_text("an agent's file counts")
    assert tool_cache.manifest(tool_cache.snapshot(work)) != before


def test_harness_halt_is_not_turned_into_a_tool_result(tmp_path):
    class Halting(Tool):
        def __init__(self, reporter):
            schema = Tool.build_function_schema("halt", "Halts", [Tool.build_parameter_schema("x", "x", "string")])
            super().__init__("halt", schema, self._halt, reporter)

        def _halt(self, x):
            raise HarnessHalt("strict replay: tool call not recorded")

    reporter = MDReporter(str(tmp_path) + "/")
    client = FakeClient([completion(tool_calls=[("halt", json.dumps({"x": "1"}))]), completion(content="never")])
    agent = Agent("m", "T", "sys", [Halting(reporter)], reporter, client=client)
    with pytest.raises(HarnessHalt):
        agent.call([{"role": "user", "content": "go"}])
    assert len(client.requests) == 1  # the agent never saw the halt


def test_executed_code_gets_neither_the_eval_token_nor_the_endpoints(env, monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AOD_HOOK_URL", "AOD_EVAL_TAG", "AOD_ROLE_OVERRIDES"):
        monkeypatch.setenv(key, f"harness-only-{key}")
    monkeypatch.delenv("AOD_TOOL_CACHE_URL", raising=False)  # run directly, without the gateway
    work = env / "w"
    work.mkdir()
    (work / "show_env.py").write_text("import os\n"
                                      "print('leaked:', sorted(k for k, v in os.environ.items() if 'harness-only' in v))\n")
    out = ExecutePython(str(work) + "/", MDReporter(str(work) + "/")).execute_python("show_env.py", "")
    assert "leaked: []" in out
