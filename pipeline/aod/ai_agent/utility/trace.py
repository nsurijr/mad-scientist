"""
JSON-lines trace of every LLM request and response (hackathon patch P5).

Upstream logged Markdown only, without system prompts or tool schemas. Each line here holds the
complete request as sent (model, messages, tools, sampling) and the complete response, numbered in
call order. projects/default.py points it at <work_dir>/trace.jsonl.
"""
import json
import threading

_path = None
_seq = 0
_lock = threading.Lock()


def configure(path: str):
    global _path, _seq
    _path = path
    _seq = 0


def next_seq() -> int:
    global _seq
    with _lock:
        _seq += 1
        return _seq


def record(**fields):
    if _path is None:
        return
    response = fields.get("response")
    if hasattr(response, "model_dump"):
        fields["response"] = response.model_dump(mode="json")
    line = json.dumps(fields, ensure_ascii=False, default=str)
    with _lock, open(_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
