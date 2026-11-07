"""hack: the teams' command line for the event service.

    export HACK_URL=https://<service>  HACK_TOKEN=<your team's token>
    hack submit <folder>            # a live eval on the public seed (counts against the cap)
    hack preview <folder>           # free: where the attack begins and what a live eval would cost
    hack status [<eval id>]         # your submissions and evals, or one eval's result
    hack trace <eval id> [-o file]  # save an eval's trace viewer page
    hack finalists <submission id>...   # designate up to 3 finalists before the freeze
    hack leaderboard
Organizers (organizer token): hack admin freeze | reveal | ranking | evals
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from service.bundle import pack


class Client:
    def __init__(self, url: str, token: str | None):
        self.url, self.token = url.rstrip("/"), token

    def call(self, method: str, path: str, body: bytes | None = None, content_type: str = "application/json",
             raw: bool = False) -> Any:
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if body is not None:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except ValueError:
                pass
            raise SystemExit(f"refused ({e.code}): {detail if isinstance(detail, str) else json.dumps(detail, indent=2)}")
        return data.decode() if raw else json.loads(data)


def _print_eval(ev: dict[str, Any]) -> None:
    print(f"{ev['id']}  {ev['kind']} on {ev['seed']}: {ev['status']}")
    check = ev.get("check") or {}
    for p in check.get("problems") or []:
        print(f"  problem: {p}")
    for w in check.get("warnings") or []:
        print(f"  warning: {w}")
    if ev.get("error"):
        print(f"  {ev['error']}")
    if ev.get("preview"):
        p = ev["preview"]
        print(f"  attack begins at: {p.get('diverged_at')}")
        print(f"  first call that needs inference: {p.get('first_live_call')}")
        print(f"  estimated live share of a full eval: {p.get('estimated_reexecution_fraction')}")
    if ev.get("result"):
        r = ev["result"]
        print(f"  {r['status']}; pull vs honest {r['pull_vs_honest']}; suspicion {r['suspicion']} "
              f"(threshold {r['threshold']}); detected {r['detected']}")
        for why in r.get("why_invalid") or []:
            print(f"  invalid: {why}")
        print(f"  first live call {r['first_live_call']}; cache {r['cache']}; tokens {r['tokens_used']}; "
              f"{r['runtime_min']} min")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hack", description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("HACK_URL", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("HACK_TOKEN"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("submit", "preview"):
        sub.add_parser(name).add_argument("folder", type=Path)
    st = sub.add_parser("status")
    st.add_argument("eval_id", nargs="?")
    tr = sub.add_parser("trace")
    tr.add_argument("eval_id")
    tr.add_argument("-o", "--out", type=Path)
    fi = sub.add_parser("finalists")
    fi.add_argument("submission_ids", nargs="+")
    sub.add_parser("leaderboard")
    ad = sub.add_parser("admin")
    ad.add_argument("action", choices=["freeze", "reveal", "ranking", "evals"])
    args = ap.parse_args(argv)
    c = Client(args.url, args.token)

    if args.cmd in ("submit", "preview"):
        if not (args.folder / "manifest.yaml").is_file():
            raise SystemExit(f"{args.folder} has no manifest.yaml")
        out = c.call("POST", f"/api/submissions?preview={'true' if args.cmd == 'preview' else 'false'}",
                     pack(args.folder), "application/gzip")
        print(f"submission {out['submission']['id']} ({out['submission']['name']})")
        _print_eval(out["eval"])
    elif args.cmd == "status":
        if args.eval_id:
            _print_eval(c.call("GET", f"/api/evals/{args.eval_id}"))
        else:
            s = c.call("GET", "/api/status")
            print(f"team {s['team']}" + ("; frozen" if s["frozen"] else "") + ("; revealed" if s["revealed"] else ""))
            for ev in s["evals"]:
                _print_eval(ev)
            for sid, p in s["scoring_seed_progress"].items():
                print(f"{sid} on the scoring seeds: {p['done']} done, {p['queued_or_running']} queued or running")
            print(f"finalists: {', '.join(s['finalists']) or 'none'}")
    elif args.cmd == "trace":
        page = c.call("GET", f"/api/evals/{args.eval_id}/trace", raw=True)
        out = args.out or Path(f"{args.eval_id}.html")
        out.write_text(page, encoding="utf-8")
        print(f"wrote {out}")
    elif args.cmd == "finalists":
        print(c.call("PUT", "/api/finalists", json.dumps({"submissions": args.submission_ids}).encode()))
    elif args.cmd == "leaderboard":
        board = c.call("GET", "/api/leaderboard")
        for name, t in board["tracks"].items():
            print(f"== {name} (live on {t['live_seed']})")
            for i, r in enumerate(t["rows"], 1):
                b, f = r["best"], r.get("final")
                best = f"{b['name']}: pull {b['pull']:.2f}, detected {b['detected']}" if b else "none yet"
                final = (f"; final {f['submission_id']}: median {f['median_effect']:.2f}, "
                         f"undetected {f['undetected_fraction']:.0%} ({f['scored']})") if f else ""
                print(f"{i:2}. {r['team']:<12} {best}; live {r['live_used']}/{r['cap']}; "
                      f"scoring seeds {r['scoring_seeds_done'] or '-'}{final}")
    else:
        method, path = {"freeze": ("POST", "/api/admin/freeze"), "reveal": ("POST", "/api/admin/reveal"),
                        "ranking": ("GET", "/api/admin/ranking"), "evals": ("GET", "/api/admin/evals")}[args.action]
        print(json.dumps(c.call(method, path), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
