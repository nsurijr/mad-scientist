import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("api_conformance", ROOT / "deploy" / "bouchet" / "api_conformance.py")
conformance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conformance)


class FakeApi:
    """An OpenAI-compatible server that wants one key and answers deterministically."""

    def __init__(self):
        api = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("x-ratelimit-remaining-requests", "99")
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self):
                if self.headers.get("Authorization") != "Bearer test-key":
                    self._send(401, {"error": "Unauthorized"})
                    return False
                return True

            def do_GET(self):  # noqa: N802
                if self._authorized():
                    self._send(200, {"data": [{"id": "gpt-oss-120b", "object": "model"}, {"id": "gemma-4-31b"}]})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.requests.append(body)
                if not self._authorized():
                    return
                n = min(body.get("max_tokens", 50), 50)
                msg = {"role": "assistant", "content": "OK", "reasoning_content": "thinking"}
                if body.get("tools"):
                    msg = {**msg, "content": None, "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "handoff_to_coder", "arguments": '{"a": 1}'}}]}
                self._send(200, {"model": "gpt-oss-120b", "system_fingerprint": "fp-test",
                                 "choices": [{"message": msg, "finish_reason": "length" if n < 50 else "stop"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": n}})

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


@pytest.fixture
def api():
    a = FakeApi()
    yield a
    a.server.shutdown()


def recorded(tmp_path):
    tools = [{"type": "function", "function": {"name": "handoff_to_coder", "strict": True, "parameters": {
        "type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}}}]
    paths = []
    for i, n in enumerate((2, 6)):
        messages = [{"role": "system", "content": "you are"}] + [{"role": "user", "content": f"m{j}"} for j in range(n - 1)]
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps({"model": "gpt-oss-120b", "messages": messages, "tools": tools, "seed": 0,
                                 "temperature": 0.0, "reasoning_effort": "medium", "max_tokens": 16384}))
        paths.append(str(p))
    return paths


def test_a_conforming_api_passes_and_the_report_records_what_it_found(tmp_path, api, capsys):
    (tmp_path / "key").write_text("test-key\n")
    out = tmp_path / "report.json"
    code = conformance.main(["--base-url", api.url, "--key-file", str(tmp_path / "key"), "--requests", *recorded(tmp_path),
                             "--repeats", "1", "--concurrent", "2", "--burst", "3", "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0 and "essential checks: PASS" in printed and "test-key" not in printed
    report = json.loads(out.read_text())
    assert report["models"]["pipeline_model"]["id"] == "gpt-oss-120b"
    assert [p["status"] for p in report["recorded"]] == [200, 200]
    assert all(p["repeat_identical"] and p["concurrent_identical"] for p in report["recorded"])
    assert report["recorded"][0]["system_fingerprint"] == "fp-test" and report["recorded"][0]["tools"] == 1
    assert report["params"]["seed"] == 200 and report["params"]["max_tokens_16_completion_tokens"] == 16
    assert report["burst"]["statuses"] == ["200"]
    assert report["headers"]["x-ratelimit-remaining-requests"] == ["99"]
    # each recorded request went out as recorded: 1 + 1 repeat + 2 concurrent
    assert sum(r.get("max_tokens") == 16384 for r in api.requests) == 2 * 4


def test_a_refused_key_fails_the_essential_checks(tmp_path, api, capsys):
    (tmp_path / "key").write_text("wrong")
    code = conformance.main(["--base-url", api.url, "--key-file", str(tmp_path / "key"), "--requests", *recorded(tmp_path),
                             "--repeats", "0", "--concurrent", "0"])
    printed = capsys.readouterr().out
    assert code == 1 and "FAIL /models: 401" in printed and "essential checks: FAIL" in printed
