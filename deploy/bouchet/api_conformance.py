"""Conformance check of an OpenAI-compatible API for the pipeline.

    PYTHONPATH=. python deploy/bouchet/api_conformance.py --base-url https://llm.aristote.education/v1 \
        --key-file secrets/ilaas.key --from-cas ../mad-scientist/cas --namespace stab3 --tag "*/S1" \
        --out runs/conformance-ilaas.json
    ... --base-url http://127.0.0.1:<port>/v1 ...    # our own vLLM, as the baseline to compare with

Run it before recording any golden traces on the API. It checks:
1. models: /models lists the pipeline's model, with whatever version fields the API gives.
2. recorded requests: the shortest, a middle and the longest recorded request whose eval tag
   matches --tag (a pattern such as "*/S1", from the store; or --requests files), sent exactly as recorded. Accepted (strict tool schemas, the agents'
   reasoning in their history)? Replies well formed (tool calls with JSON arguments)? Which model and
   version does the API report?
3. repeatability: each recorded request again, in sequence and concurrently. Identical replies?
4. parameters: seed, temperature, top_p, max_tokens and reasoning_effort, each accepted or refused,
   and whether max_tokens and reasoning_effort take effect.
5. the server's date: the date gpt-oss's system header gives, if any (it should be pinned).
6. limits: rate-limit headers, latencies, and optionally a long prompt (--long) and a burst (--burst).

The key is read from --key-file (or the variable named by --key-env) and never printed. A run uses
roughly the recorded requests' tokens times (2 + repeats + concurrent), plus a few thousand.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HEADER_PATTERN = re.compile(r"ratelimit|retry|request-id|model|version|fingerprint|server", re.I)
SMALL = [{"role": "user", "content": "Reply with the single word OK."}]


class Api:
    def __init__(self, base_url: str, key: str | None, timeout: float):
        self.base_url, self.key, self.timeout = base_url.rstrip("/"), key, timeout
        self.headers_seen: dict[str, set[str]] = {}

    def call(self, path: str, body: dict | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers)
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                status, raw, hdrs = r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            status, raw, hdrs = e.code, e.read(), e.headers
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return {"status": None, "error": f"{type(e).__name__}: {e}", "latency_s": round(time.time() - start, 2)}
        for k, v in (hdrs or {}).items():
            if HEADER_PATTERN.search(k):
                self.headers_seen.setdefault(k.lower(), set()).add(v)
        out: dict[str, Any] = {"status": status, "latency_s": round(time.time() - start, 2)}
        try:
            out["json"] = json.loads(raw)
        except ValueError:
            out["text"] = raw.decode(errors="replace")[:500]
        if status != 200:
            out["error"] = out.get("text") or json.dumps(out.get("json"))[:500]
        return out

    def chat(self, body: dict) -> dict[str, Any]:
        return self.call("/chat/completions", body)


def reply_of(result: dict) -> dict | None:
    """The comparable part of a reply: text, reasoning and tool calls, without ids or timings."""
    if result.get("status") != 200:
        return None
    msg = ((result["json"].get("choices") or [{}])[0].get("message")) or {}
    return {"content": msg.get("content"), "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
            "tool_calls": [((tc.get("function") or {}).get("name"), (tc.get("function") or {}).get("arguments"))
                           for tc in msg.get("tool_calls") or []]}


def well_formed(result: dict) -> list[str]:
    problems = []
    rep = reply_of(result)
    if rep is None:
        return [f"HTTP {result.get('status')}: {result.get('error', '')[:200]}"]
    if not rep["content"] and not rep["tool_calls"]:
        problems.append("neither text nor tool calls")
    for name, args in rep["tool_calls"]:
        try:
            json.loads(args or "")
        except ValueError:
            problems.append(f"tool call {name}: arguments are not JSON")
    if rep["reasoning"] is None:
        problems.append("no reasoning in this reply (normal for short tool calls; vLLM omits it too)")
    return problems


def recorded_requests(args) -> list[dict]:
    if args.requests:
        return [json.loads(Path(p).read_text()) for p in args.requests]
    from harness.cas import Policy, Store

    store = Store(args.from_cas, Policy.replay_only(args.namespace))
    found = []
    for entry in store.scan(args.namespace):
        tag = (entry.meta or {}).get("eval")  # entries recorded before tags existed have none
        if not tag or not fnmatch.fnmatch(tag, args.tag) or "request.json" not in (entry.files or {}):
            continue
        found.append(json.loads(store.get_file(entry, "request.json")))
    if not found:
        raise SystemExit(f"no recorded requests tagged {args.tag} in {args.namespace}")
    found.sort(key=lambda r: len(json.dumps(r)))
    picks = [found[0], found[len(found) // 2], found[-1]]
    return list({json.dumps(p, sort_keys=True): p for p in picks}.values())  # without repeats


def check(api: Api, args) -> dict[str, Any]:
    report: dict[str, Any] = {"base_url": api.base_url, "model": args.model, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}

    models = api.call("/models")
    listed = [m for m in (models.get("json") or {}).get("data", [])] if models.get("status") == 200 else []
    report["models"] = {"status": models.get("status"), "error": models.get("error"),
                        "ids": [m.get("id") for m in listed],
                        "pipeline_model": next((m for m in listed if m.get("id") == args.model), None)}

    probes = recorded_requests(args)
    report["recorded"] = []
    for i, req in enumerate(probes):
        req = {**req, "model": args.model}
        first = api.chat(req)
        again = [api.chat(req) for _ in range(args.repeats)]
        with concurrent.futures.ThreadPoolExecutor(args.concurrent or 1) as pool:
            together = list(pool.map(api.chat, [req] * args.concurrent))
        replies = [reply_of(r) for r in [first, *again, *together]]
        body = first.get("json") or {}
        report["recorded"].append({
            "probe": i, "messages": len(req.get("messages", [])), "tools": len(req.get("tools") or []),
            "status": first.get("status"), "problems": well_formed(first), "latency_s": first.get("latency_s"),
            "reported_model": body.get("model"), "system_fingerprint": body.get("system_fingerprint"),
            "usage": body.get("usage"),
            "repeat_identical": all(r == replies[0] for r in replies[1:1 + args.repeats]) if replies[0] else None,
            "concurrent_identical": all(r == replies[0] for r in replies[1 + args.repeats:]) if replies[0] else None,
            "distinct_replies": len({json.dumps(r, sort_keys=True) for r in replies if r is not None}),
            "errors": sorted({r.get("error", "")[:200] for r in [first, *again, *together] if r.get("status") != 200}),
        })

    params = {}
    base = {"model": args.model, "messages": SMALL, "max_tokens": 64}
    params["none"] = api.chat(base).get("status")
    for name, value in (("seed", 0), ("temperature", 0.0), ("top_p", 1.0), ("reasoning_effort", "low"),
                        ("reasoning_effort=high", "high")):
        key = name.split("=")[0]
        r = api.chat({**base, key: value})
        params[name] = r.get("status") if r.get("status") == 200 else f"{r.get('status')}: {r.get('error', '')[:200]}"
    long_answer = [{"role": "user", "content": "Write a 300-word story about a detector."}]
    capped = api.chat({"model": args.model, "messages": long_answer, "max_tokens": 16})
    usage = (capped.get("json") or {}).get("usage") or {}
    params["max_tokens_16_completion_tokens"] = usage.get("completion_tokens")
    params["max_tokens_16_finish_reason"] = ((capped.get("json") or {}).get("choices") or [{}])[0].get("finish_reason")
    effort = {}
    for level in ("low", "high"):
        r = api.chat({"model": args.model, "max_tokens": 4096, "reasoning_effort": level,
                      "messages": [{"role": "user", "content": "How many primes are there below 200? Answer with the number."}]})
        rep = reply_of(r) or {}
        effort[level] = {"completion_tokens": ((r.get("json") or {}).get("usage") or {}).get("completion_tokens"),
                         "reasoning_chars": len(rep.get("reasoning") or "")}
    params["reasoning_effort_effect"] = effort
    report["params"] = params

    date = api.chat({"model": args.model, "max_tokens": 2048, "temperature": 0.0, "messages": [
        {"role": "user", "content": "Your system message may state a current date. Reply with that date exactly as "
                                    "written there, or NONE if there is none."}]})
    report["server_date"] = (reply_of(date) or {}).get("content")

    if args.long:
        words = ("The detector recorded another collision event with two large-radius jets. " * (args.long // 12 + 1))
        r = api.chat({"model": args.model, "max_tokens": 8, "messages": [{"role": "user", "content": words + "\nReply OK."}]})
        report["long_prompt"] = {"target_tokens": args.long, "status": r.get("status"), "latency_s": r.get("latency_s"),
                                 "prompt_tokens": ((r.get("json") or {}).get("usage") or {}).get("prompt_tokens"),
                                 "error": r.get("error")}
    if args.burst:
        with concurrent.futures.ThreadPoolExecutor(args.burst) as pool:
            results = list(pool.map(api.chat, [base] * args.burst))
        lat = [r["latency_s"] for r in results if r.get("status") == 200]
        report["burst"] = {"requests": args.burst, "statuses": sorted({str(r.get("status")) for r in results}),
                           "rate_limited": sum(r.get("status") == 429 for r in results),
                           "median_latency_s": statistics.median(lat) if lat else None}
    report["headers"] = {k: sorted(v) for k, v in api.headers_seen.items()}
    return report


def summary(report: dict) -> tuple[list[str], bool]:
    lines, essential = [], True

    def line(level: str, text: str) -> None:
        lines.append(f"{level:<4} {text}")

    m = report["models"]
    if m["pipeline_model"] is not None:
        line("PASS", f"/models lists {report['model']} ({', '.join(i for i in m['ids'] if i)})")
    else:
        essential = False
        line("FAIL", f"/models: {m['status']} {m.get('error') or ''} (listed: {m['ids']})")
    for p in report["recorded"]:
        ok = p["status"] == 200 and not [x for x in p["problems"] if "reasoning" not in x]
        essential &= ok
        line("PASS" if ok else "FAIL", f"recorded request {p['probe']} ({p['messages']} messages, {p['tools']} tools): "
             f"HTTP {p['status']}, {p['latency_s']} s, model {p['reported_model']}, fingerprint {p['system_fingerprint']}"
             + (f"; {'; '.join(p['problems'])}" if p["problems"] else ""))
        same = p["repeat_identical"] and p["concurrent_identical"]
        line("PASS" if same else "WARN", f"  repeats identical: in sequence {p['repeat_identical']}, concurrently "
             f"{p['concurrent_identical']} ({p['distinct_replies']} distinct replies)")
    for name, value in report["params"].items():
        if name in ("max_tokens_16_completion_tokens", "max_tokens_16_finish_reason", "reasoning_effort_effect"):
            continue
        line("PASS" if value == 200 else "WARN", f"parameter {name}: {value}")
    p = report["params"]
    capped = p["max_tokens_16_completion_tokens"]
    line("PASS" if capped is not None and capped <= 16 else "WARN",
         f"max_tokens 16 gives {capped} completion tokens, finish_reason {p['max_tokens_16_finish_reason']}")
    eff = p["reasoning_effort_effect"]
    line("INFO", f"reasoning_effort low vs high: {eff['low']} vs {eff['high']}")
    line("INFO", f"server date in the system header: {report['server_date']!r} (should be the pinned 2026-11-23)")
    if "long_prompt" in report:
        lp = report["long_prompt"]
        line("PASS" if lp["status"] == 200 else "WARN", f"long prompt (~{lp['target_tokens']} tokens): HTTP {lp['status']}, "
             f"{lp['prompt_tokens']} prompt tokens, {lp['latency_s']} s")
    if "burst" in report:
        b = report["burst"]
        line("INFO", f"burst of {b['requests']}: statuses {b['statuses']}, {b['rate_limited']} rate-limited, "
             f"median {b['median_latency_s']} s")
    if report["headers"]:
        line("INFO", "headers: " + "; ".join(f"{k}={v}" for k, v in report["headers"].items()))
    return lines, essential


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--key-file", type=Path)
    ap.add_argument("--key-env", help="the environment variable holding the key, instead of --key-file")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--requests", nargs="+", help="recorded request.json files, instead of --from-cas")
    ap.add_argument("--from-cas", type=Path)
    ap.add_argument("--namespace", default="stab3")
    ap.add_argument("--tag", default="*/S1", help="pattern of the eval tags whose recorded requests to send")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrent", type=int, default=3)
    ap.add_argument("--long", type=int, default=0, help="also send a prompt of about this many tokens")
    ap.add_argument("--burst", type=int, default=0, help="also send this many small requests at once")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if not args.requests and not args.from_cas:
        ap.error("give --from-cas (with --namespace and --tag) or --requests")
    key = None
    if args.key_file:
        key = args.key_file.read_text().strip()
    elif args.key_env:
        key = os.environ.get(args.key_env)
    report = check(Api(args.base_url, key, args.timeout), args)
    lines, essential = summary(report)
    print("\n".join(lines))
    print(f"essential checks: {'PASS' if essential else 'FAIL'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if essential else 1


if __name__ == "__main__":
    sys.exit(main())
