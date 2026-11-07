"""Record fixed chat requests against a vLLM server, then compare recordings for bit-identical replies.

    python determinism_check.py run --port 8000 --model gpt-oss-120b --out a.json
    python determinism_check.py compare a.json b.json
    python determinism_check.py replay --port 8000 --requests probe_*.json --out r.json

`run` sends each short probe twice: alone, and again while other requests are in flight, so the
batch it lands in differs. With batch-invariant inference both replies must match exactly, and so
must a second `run` against a restarted server.

`replay` sends full recorded request bodies (real agent requests, thousands of tokens each): every
one alone, then all at once, then started a fraction of a second apart. Long concurrent prompts are
what the short probes never exercise: they overflow the scheduler's per-step token budget, so where
a prompt is split into prefill chunks depends on what else is running.

Standard library only, so it runs from any Python.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

SAMPLING = {"temperature": 0, "top_p": 1, "seed": 1234, "max_tokens": 600, "reasoning_effort": "low"}

EXECUTE_PYTHON = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute a python file in the working directory",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "The python file to execute"},
                "cli_args": {"type": "string", "description": "Command line arguments"},
            },
            "required": ["filename", "cli_args"],
            "additionalProperties": False,
        },
    },
}

PROBES = {
    "tool_call": {
        "messages": [
            {"role": "system", "content": "You are the data-preparation agent in a dijet anomaly search. "
             "Use tools to inspect data; never guess numbers."},
            {"role": "user", "content": "A script count_events.py prints the number of rows in /data/c_data.csv. "
             "Run it and report the count."},
        ],
        "tools": [EXECUTE_PYTHON],
    },
    "physics": {
        "messages": [
            {"role": "user", "content": "In two sentences: why does a resonance decaying to two boosted jets "
             "show up as a localized excess in the dijet invariant mass?"},
        ],
    },
    "estimator": {
        "messages": [
            {"role": "user", "content": "List five steps to estimate the number of signal events in a signal "
             "region using a background-only reference sample and an anomaly-score cut."},
        ],
    },
}


def post(port: int, model: str, probe: dict, timeout: float = 600) -> tuple[dict, float]:
    body = {"model": model, **SAMPLING, **probe}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def essence(resp: dict) -> dict:
    """The parts of a reply that must be identical; ids and timestamps are excluded."""
    choice = resp["choices"][0]
    msg = choice["message"]
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
             for c in (msg.get("tool_calls") or [])]
    return {
        "content": msg.get("content"),
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
        "tool_calls": calls,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
    }


def run(args: argparse.Namespace) -> None:
    results = []
    for name, probe in PROBES.items():
        resp, dt = post(args.port, args.model, probe)
        results.append({"probe": name, "mode": "alone", "latency_s": round(dt, 2), **essence(resp)})

    # Same probes again, each surrounded by other in-flight requests.
    noise = [{"messages": [{"role": "user", "content": f"Write a haiku about jet number {i}."}]}
             for i in range(args.noise)]
    with cf.ThreadPoolExecutor(max_workers=args.noise + 1) as pool:
        for name, probe in PROBES.items():
            futures = [pool.submit(post, args.port, args.model, n) for n in noise]
            resp, dt = post(args.port, args.model, probe)
            cf.wait(futures)
            results.append({"probe": name, "mode": "under_load", "latency_s": round(dt, 2), **essence(resp)})

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "sampling": SAMPLING, "results": results}, f, indent=2)
    for r in results:
        print(f"{r['probe']:10s} {r['mode']:10s} {r['latency_s']:7.2f}s  tokens={r['completion_tokens']}"
              f"  tool_calls={[c['name'] for c in r['tool_calls']]}")
    ok = check_pairs(results, "alone vs under_load (same server)")
    sys.exit(0 if ok else 1)


def check_pairs(results: list[dict], label: str, other: list[dict] | None = None) -> bool:
    keyed = {(r["probe"], r["mode"]): r for r in results}
    ok = True
    if other is None:
        pairs = [((p, "alone"), (p, "under_load")) for p in PROBES]
        b = keyed
    else:
        pairs = [(k, k) for k in keyed]
        b = {(r["probe"], r["mode"]): r for r in other}
    for ka, kb in pairs:
        fields = ("content", "reasoning", "tool_calls", "finish_reason")
        same = all(keyed[ka][f] == b[kb][f] for f in fields)
        ok &= same
        print(f"{'MATCH' if same else 'DIFF '}  {label}: {ka[0]} ({ka[1]} vs {kb[1]})")
    return ok


def compare(args: argparse.Namespace) -> None:
    a = json.load(open(args.a))["results"]
    b = json.load(open(args.b))["results"]
    ok = check_pairs(a, "server restart", other=b)
    print("overall:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


FIELDS = ("content", "reasoning", "tool_calls", "finish_reason")


def post_body(port: int, body: dict, timeout: float = 1800) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def replay(args: argparse.Namespace) -> None:
    bodies = [json.load(open(p)) for p in args.requests]
    for b in bodies:
        b["max_tokens"] = min(b.get("max_tokens") or args.max_tokens, args.max_tokens)
    names = [os.path.splitext(os.path.basename(p))[0] for p in args.requests]
    results = {}

    def send(i):
        resp, dt = post_body(args.port, bodies[i])
        return {**essence(resp), "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"), "latency_s": round(dt, 2)}

    for i, name in enumerate(names):
        results[(name, "alone")] = send(i)
    with cf.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        futures = {pool.submit(send, i): name for i, name in enumerate(names)}
        for f, name in futures.items():
            results[(name, "together")] = f.result()
        futures = {}
        for i, name in enumerate(names):
            futures[pool.submit(send, i)] = name
            time.sleep(args.stagger)
        for f, name in futures.items():
            results[(name, "staggered")] = f.result()

    ok = True
    for name in names:
        alone = results[(name, "alone")]
        for mode in ("together", "staggered"):
            same = all(results[(name, mode)][f] == alone[f] for f in FIELDS)
            ok &= same
            print(f"{'MATCH' if same else 'DIFF '}  {name} ({alone['prompt_tokens']} prompt tokens): alone vs {mode}")
    with open(args.out, "w") as f:
        json.dump({"%s|%s" % k: v for k, v in results.items()}, f, indent=2)
    print("overall:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--noise", type=int, default=8)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    p = sub.add_parser("replay")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--requests", nargs="+", required=True, help="JSON files, each a full request body")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=1024, help="cap on reply length, same in every mode")
    p.add_argument("--stagger", type=float, default=0.3, help="seconds between staggered starts")
    args = ap.parse_args()
    {"run": run, "compare": compare, "replay": replay}[args.cmd](args)


if __name__ == "__main__":
    main()
