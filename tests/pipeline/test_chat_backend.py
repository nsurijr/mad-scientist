import json
from types import SimpleNamespace

import pytest

from agents.agent import Agent
from agents.chat_backend import ChatResponse, to_chat_messages, to_chat_tools
from tools.tool import Tool
from utility import trace
from utility.md_reporter import MDReporter


def completion(content="", tool_calls=(), reasoning=None, cid="chatcmpl-random"):
    calls = [SimpleNamespace(id=f"chatcmpl-tool-{i}-random",
                             function=SimpleNamespace(name=n, arguments=a)) for i, (n, a) in enumerate(tool_calls)]
    msg = SimpleNamespace(content=content, tool_calls=calls or None, reasoning_content=reasoning)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                            prompt_tokens_details=None, completion_tokens_details=None)
    return SimpleNamespace(id=cid, choices=[SimpleNamespace(message=msg)], usage=usage,
                           model_dump=lambda mode="json": {"id": cid, "content": content})


class EchoTool(Tool):
    def __init__(self, reporter):
        schema = Tool.build_function_schema(
            "echo", "Echo text", [Tool.build_parameter_schema("text", "Text to echo", "string")])
        super().__init__("echo", schema, lambda text: f"echo: {text}", reporter)


class FakeClient:
    """Replays scripted completions and records every request it receives."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request)))
        return self.replies.pop(0)


def test_to_chat_tools_wraps_responses_schema(tmp_path):
    tools = to_chat_tools([EchoTool(MDReporter(str(tmp_path) + "/"))])
    fn = tools[0]["function"]
    assert tools[0]["type"] == "function" and fn["name"] == "echo" and fn["strict"] is True
    assert fn["parameters"]["required"] == ["text"]


def test_to_chat_messages_converts_tool_outputs_and_text_parts():
    out = to_chat_messages([
        {"role": "user", "content": "hi"},
        {"type": "function_call_output", "call_id": "call_1_0", "output": 42},
        {"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]},
    ])
    assert out == [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "call_1_0", "content": "42"},
        {"role": "user", "content": "a\nb"},
    ]


def test_images_are_refused():
    with pytest.raises(ValueError):
        to_chat_messages([{"role": "user", "content": [{"type": "input_image", "image_url": "data:"}]}])


def test_tool_call_ids_are_deterministic():
    r = ChatResponse(completion(tool_calls=[("echo", '{"text": "x"}')], reasoning="think"), "call_7")
    kinds = [o.type for o in r.output]
    assert kinds == ["reasoning", "function_call"]
    assert r.output[1].call_id == "call_7_0"
    assert r.history_message["tool_calls"][0]["id"] == "call_7_0"
    assert r.history_message["reasoning_content"] == "think"
    assert r.usage.input_tokens == 10 and r.usage.output_tokens == 5


def run_agent(tmp_path, vllm_ids):
    """One tool round trip; vLLM-style random ids differ per run."""
    trace.configure(str(tmp_path / f"trace-{vllm_ids}.jsonl"))
    client = FakeClient([
        completion(tool_calls=[("echo", '{"text": "hello"}')], cid=f"resp-{vllm_ids}-1"),
        completion(content="done", cid=f"resp-{vllm_ids}-2"),
    ])
    reporter = MDReporter(str(tmp_path) + "/", filename=f"conv-{vllm_ids}")
    agent = Agent("gpt-oss-120b", "Tester", "You are a test.", [EchoTool(reporter)], reporter, client=client)
    assert agent.call([{"role": "user", "content": "go"}]) == "done"
    return client.requests


def test_agent_sends_full_history_and_traces(tmp_path):
    requests = run_agent(tmp_path, "a")
    first, second = requests
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    assert second["messages"][3] == {"role": "tool", "tool_call_id": "call_1_0", "content": "echo: hello"}
    assert first["temperature"] == 0 and "seed" in first and "tools" in first
    lines = (tmp_path / "trace-a.jsonl").read_text().splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [1, 2]


def test_requests_are_identical_across_runs_despite_random_server_ids(tmp_path):
    assert run_agent(tmp_path, "a") == run_agent(tmp_path, "b")


class WriteTool(Tool):
    def __init__(self, reporter):
        schema = Tool.build_function_schema(
            "write", "Write a file", [Tool.build_parameter_schema("filename", "File name", "string")])
        super().__init__("write", schema, self._write, reporter)

    def _write(self, filename):
        with open(filename, "w"):
            return "written"


def test_failing_tool_calls_are_reported_to_the_agent(tmp_path):
    """An S4 crash in a staged run: write_python with an empty filename opened the work folder."""
    trace.configure(str(tmp_path / "trace.jsonl"))
    client = FakeClient([
        completion(tool_calls=[("write", json.dumps({"filename": str(tmp_path)})), ("write", "{not json")]),
        completion(content="recovered"),
    ])
    reporter = MDReporter(str(tmp_path) + "/")
    agent = Agent("m", "T", "sys", [WriteTool(reporter)], reporter, client=client)
    assert agent.call([{"role": "user", "content": "go"}]) == "recovered"
    results = [m["content"] for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert results[0].startswith("Tool call failed: IsADirectoryError")
    assert results[1].startswith("Tool call failed: JSONDecodeError")
    assert reporter.metrics["tool_call_errors"] == 2


def test_reset_history_forgets_conversation(tmp_path):
    reporter = MDReporter(str(tmp_path) + "/")
    agent = Agent("m", "T", "sys", [], reporter, client=FakeClient([completion(content="x")]))
    agent.call([{"role": "user", "content": "q"}])
    agent.reset_history()
    assert agent._history == [{"role": "system", "content": "sys"}]
