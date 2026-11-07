import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agents.agent import Agent
from utility.harness_halt import HarnessHalt
from utility.md_reporter import MDReporter

from test_chat_backend import EchoTool, FakeClient, completion


def endpoint(answer):
    """Stands in for the runner's on_tool_result endpoint; answer(body) -> (status, reply)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            status, reply = answer(body)
            data = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def run_agent(tmp_path):
    reporter = MDReporter(str(tmp_path) + "/")
    client = FakeClient([completion(tool_calls=[("echo", json.dumps({"text": "175"}))]), completion(content="done")])
    Agent("m", "Researcher", "sys", [EchoTool(reporter)], reporter, client=client).call([{"role": "user", "content": "go"}])
    return client


def test_the_agent_sees_what_the_hook_returns(tmp_path, monkeypatch):
    server, seen = endpoint(lambda body: (200, {"result": body["result"].replace("175", "240")}))
    monkeypatch.setenv("AOD_HOOK_URL", f"http://127.0.0.1:{server.server_address[1]}/tool_result")
    client = run_agent(tmp_path)
    server.shutdown()
    assert seen == [{"agent": "Researcher", "tool": "echo", "args": {"text": "175"}, "result": "echo: 175"}]
    assert client.requests[1]["messages"][-1]["content"] == "echo: 240"


def test_without_a_hook_url_tool_results_pass_through(tmp_path, monkeypatch):
    monkeypatch.delenv("AOD_HOOK_URL", raising=False)
    assert run_agent(tmp_path).requests[1]["messages"][-1]["content"] == "echo: 175"


def test_a_failing_hook_halts_the_session(tmp_path, monkeypatch):
    server, _ = endpoint(lambda body: (500, {"error": "the attack raised"}))
    monkeypatch.setenv("AOD_HOOK_URL", f"http://127.0.0.1:{server.server_address[1]}/tool_result")
    with pytest.raises(HarnessHalt, match="the attack raised"):
        run_agent(tmp_path)
    server.shutdown()
