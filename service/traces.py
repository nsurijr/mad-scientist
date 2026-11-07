"""The trace viewer: one eval's stages, agents, messages and tool calls, with each LLM
call marked as a cache hit (replayed from the golden or the team's namespace) or live.

digest() runs where the eval ran and keeps only what a reader needs from each stage's trace.jsonl:
per LLM call, the agent, the messages new since that agent's previous call (tool results,
re-prompts), and the reply (reasoning, text, tool calls). Hits and live calls come from the
gateway's log, matched in order within each stage. render() turns a digest into one HTML page.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

STAGES = ("S1", "S2", "S3", "S4", "S5")
MAX_TEXT = 20_000  # characters kept per message


def _cut(text: Any) -> str:
    text = text if isinstance(text, str) else json.dumps(text)
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + f"\n... [{len(text) - MAX_TEXT:,} more characters]"


def digest(run_dir: Path, gateway_log: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").is_file() else {}
    outcomes: dict[str, list[str]] = {}
    for rec in gateway_log or []:
        if rec.get("kind") == "llm":
            outcomes.setdefault(str(rec.get("eval", "")).split("/")[-1], []).append(rec.get("outcome", "?"))
    stages = []
    for stage in STAGES:
        path = run_dir / stage / "trace" / "trace.jsonl"
        info = next((s for s in manifest.get("stages", []) if s.get("stage") == stage), {})
        calls, history = [], {}
        if path.is_file():
            for i, line in enumerate(l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()):
                rec = json.loads(line)
                agent, messages = rec.get("agent", "?"), (rec.get("request") or {}).get("messages", [])
                prev = history.get(agent, [])
                new = messages[len(prev):] if messages[:len(prev)] == prev else messages
                history[agent] = messages
                msg = (((rec.get("response") or {}).get("choices") or [{}])[0].get("message")) or {}
                stage_outcomes = outcomes.get(stage, [])
                calls.append({
                    "agent": agent, "outcome": stage_outcomes[i] if i < len(stage_outcomes) else None,
                    "new": [{"role": m.get("role"), "content": _cut(m.get("content") or "")} for m in new
                            if m.get("role") != "system" or not prev],
                    "reasoning": _cut(msg.get("reasoning") or msg.get("reasoning_content") or ""),
                    "content": _cut(msg.get("content") or ""),
                    "tool_calls": [{"name": (tc.get("function") or {}).get("name"),
                                    "arguments": _cut((tc.get("function") or {}).get("arguments") or "")}
                                   for tc in msg.get("tool_calls") or []],
                })
        if calls or info:
            stages.append({"stage": stage, "exit_code": info.get("exit_code"), "wall_s": info.get("wall_s"),
                           "replaced": bool(info.get("replaced")), "calls": calls})
    return {"seed": manifest.get("seed"), "status": manifest.get("status"),
            "diverged_at": (manifest.get("attack") or {}).get("diverged_at"), "stages": stages}


CSS = """
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6b6b70; --card: #f4f4f6; --line: #d8d8de;
        --hit: #2e7d32; --live: #b26a00; --unknown: #8a8a8f; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #16161a; --fg: #ececf1; --muted: #a0a0a8; --card: #212127; --line: #34343c;
          --hit: #66bb6a; --live: #ffb74d; --unknown: #9a9aa2; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0 auto;
       max-width: 1100px; padding: 16px; }
h1 { font-size: 1.3em; } h2 { font-size: 1.1em; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.meta { color: var(--muted); } details { background: var(--card); border-radius: 6px; margin: 6px 0; padding: 6px 10px; }
summary { cursor: pointer; } .badge { font-size: .8em; font-weight: 600; padding: 1px 6px; border-radius: 4px; color: var(--bg); }
.hit { background: var(--hit); } .live { background: var(--live); } .unknown { background: var(--unknown); }
pre { white-space: pre-wrap; word-break: break-word; font-size: 13px; margin: 4px 0 8px; }
.role { color: var(--muted); font-size: .85em; }
"""


def _pre(label: str, text: str) -> str:
    return f'<div class="role">{html.escape(label)}</div><pre>{html.escape(text)}</pre>' if text else ""


def render(d: dict[str, Any], title: str) -> str:
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width, initial-scale=1'><title>Eval trace</title><style>{CSS}</style></head><body>",
           f"<h1>{html.escape(title)}</h1>",
           f"<p class='meta'>seed {html.escape(str(d.get('seed')))} · status {html.escape(str(d.get('status')))} · "
           f"attack begins at {html.escape(str(d.get('diverged_at') or 'nowhere'))}</p>"]
    for s in d.get("stages", []):
        n_hit = sum(c["outcome"] == "hit" for c in s["calls"])
        n_live = sum(c["outcome"] == "live" for c in s["calls"])
        wall = f" · {s['wall_s'] / 60:.1f} min" if s.get("wall_s") else ""
        code = f" · exit code {s['exit_code']}" if s.get("exit_code") not in (None, 0) else ""
        out.append(f"<h2>{s['stage']}{' (replaced)' if s['replaced'] else ''}</h2>"
                   f"<p class='meta'>{len(s['calls'])} LLM calls: {n_hit} hits, {n_live} live{wall}{code}</p>")
        for i, c in enumerate(s["calls"], 1):
            kind = c["outcome"] if c["outcome"] in ("hit", "live") else "unknown"
            body = "".join(_pre(f"→ {m['role']}", m["content"]) for m in c["new"])
            body += _pre("reasoning", c["reasoning"]) + _pre("reply", c["content"])
            body += "".join(_pre(f"tool call: {t['name']}", t["arguments"]) for t in c["tool_calls"])
            tools = ", ".join(t["name"] or "?" for t in c["tool_calls"])
            out.append(f"<details><summary><span class='badge {kind}'>{kind}</span> {i}. "
                       f"{html.escape(c['agent'])}{' → ' + html.escape(tools) if tools else ''}</summary>{body}</details>")
    out.append("</body></html>")
    return "\n".join(out)
