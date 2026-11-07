"""
Stateless Chat Completions backend for Agent (hackathon patch P1).

Upstream called the OpenAI Responses API and kept each conversation on the server through
previous_response_id. Here every request carries the agent's full message history, so each call
is self-contained: the harness can hash, cache and replay it, and it runs against a local vLLM
server. ChatResponse mimics the parts of a Responses `Response` the rest of the code reads
(.id, .output, .output_text, .usage).
"""
import os
from types import SimpleNamespace

from openai import OpenAI

_CLIENT = None


def default_client() -> OpenAI:
    """One shared client; OPENAI_BASE_URL points it at the harness gateway or a vLLM server."""
    global _CLIENT
    if _CLIENT is None:
        # P9: the harness gateway logs every call under its eval tag ("<seed>/<stage>").
        tag = os.environ.get("AOD_EVAL_TAG")
        _CLIENT = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=float(os.environ.get("AOD_REQUEST_TIMEOUT", "1800")),
            max_retries=0,
            default_headers={"X-Eval": tag} if tag else None,
        )
    return _CLIENT


def sampling() -> dict:
    """Sampling fixed for every call; the gateway enforces the same values."""
    params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": int(os.environ.get("HARNESS_LLM_SEED", "0")),
        "max_tokens": int(os.environ.get("AOD_MAX_TOKENS", "16384")),
    }
    effort = os.environ.get("AOD_REASONING_EFFORT", "medium")
    if effort:
        params["reasoning_effort"] = effort
    return params


def to_chat_tools(tools) -> list:
    """Responses-style tool schemas -> Chat Completions function tools."""
    out = []
    for tool in tools:
        s = tool.schema
        fn = {"name": s["name"], "description": s.get("description", ""), "parameters": s["parameters"]}
        if "strict" in s:
            fn["strict"] = s["strict"]
        out.append({"type": "function", "function": fn})
    return out


def _text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        kind = part.get("type")
        if kind in ("input_text", "text", "output_text"):
            parts.append(part["text"])
        elif kind == "input_image":
            raise ValueError("image inputs are disabled in this pipeline (patch P4: use view_plot_data)")
        else:
            parts.append(str(part))
    return "\n".join(parts)


def to_chat_messages(items) -> list:
    """Responses-style input items, as the agents and tools build them -> chat messages."""
    out = []
    for item in items:
        if item.get("type") == "function_call_output":
            out.append({"role": "tool", "tool_call_id": item["call_id"], "content": str(item["output"])})
        else:
            out.append({"role": item["role"], "content": _text(item["content"])})
    return out


class ChatResponse:
    """A Chat Completions result shaped like the Responses objects the agents read."""

    def __init__(self, completion, call_id_prefix: str):
        self.raw = completion
        self.id = completion.id
        msg = completion.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)

        self.output = []
        if reasoning:
            self.output.append(SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=reasoning)]))
        if msg.content:
            self.output.append(SimpleNamespace(type="message", content=[SimpleNamespace(text=msg.content)]))

        # vLLM draws tool-call ids at random and they are echoed into later prompts; replace them
        # with ids derived from the call sequence, or re-runs would never share a cache key.
        calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            call_id = f"{call_id_prefix}_{i}"
            self.output.append(SimpleNamespace(
                type="function_call", name=tc.function.name, arguments=tc.function.arguments, call_id=call_id))
            calls.append({"id": call_id, "type": "function",
                          "function": {"name": tc.function.name, "arguments": tc.function.arguments}})
        self.output_text = msg.content or ""

        # What goes back into this agent's history for its next call.
        self.history_message = {"role": "assistant", "content": msg.content or ""}
        if reasoning:
            self.history_message["reasoning_content"] = reasoning
        if calls:
            self.history_message["tool_calls"] = calls

        u = completion.usage
        self.usage = None
        if u is not None:
            prompt_details = getattr(u, "prompt_tokens_details", None)
            completion_details = getattr(u, "completion_tokens_details", None)
            self.usage = SimpleNamespace(
                input_tokens=u.prompt_tokens,
                output_tokens=u.completion_tokens,
                total_tokens=u.total_tokens,
                input_tokens_details=SimpleNamespace(
                    cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0),
                output_tokens_details=SimpleNamespace(
                    reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0),
            )
