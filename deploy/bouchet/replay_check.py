"""Acceptance test 1: does a strict replay reproduce its recording exactly?

    .venv/bin/python deploy/bouchet/replay_check.py runs/staged-<record job> runs/replay-<replay job>

Both batches hold <seed>/run.json, per-stage AoD traces and gateway.jsonl. For every replayed
seed it requires:
- 100 % hits: every LLM call and code execution in the replay's gateway log was served from the
  store (no live call, no miss), and the replay made as many of each as the recording;
- the same LLM conversation: every stage's trace has the same requests and replies, in order;
- the same results: every stage output (sha256), the S4 recount and the S5 submission are
  identical, and both runs are complete.
Prints each check and PASS or FAIL. Exits non-zero on FAIL.
"""

import json
import sys
from collections import Counter
from pathlib import Path


def gateway_counts(batch: Path) -> dict:
    """Outcome counts per eval tag and kind, e.g. {"s0/S3": Counter({"llm_hit": 21, ...})}."""
    out = {}
    path = batch / "gateway.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            out.setdefault(rec.get("eval", ""), Counter())[f"{rec['kind']}_{rec['outcome']}"] += 1
    return out


def trace(stage_dir: Path) -> list:
    path = stage_dir / "trace" / "trace.jsonl"
    if not path.exists():
        return []
    calls = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        msg = rec["response"]["choices"][0]["message"]
        calls.append((rec["agent"], rec["request"], msg.get("content"), msg.get("reasoning_content") or msg.get("reasoning"),
                      [t["function"] for t in (msg.get("tool_calls") or [])]))
    return calls


def check_seed(recorded: Path, replay: Path, seed: str, rec_gw: dict, rep_gw: dict) -> list[tuple[str, bool, str]]:
    a = json.loads((recorded / seed / "run.json").read_text())
    b = json.loads((replay / seed / "run.json").read_text())
    results = [("both runs complete", a["status"] == b["status"] == "complete", f"{a['status']} / {b['status']}")]

    rep_totals, rec_totals = Counter(), Counter()
    for tag, c in rep_gw.items():
        if tag.startswith(f"{seed}/"):
            rep_totals.update(c)
    for tag, c in rec_gw.items():
        if tag.startswith(f"{seed}/"):
            rec_totals.update(c)
    not_hit = {k: v for k, v in rep_totals.items() if not k.endswith("_hit")}
    results.append(("100 % hits in the replay", not not_hit and sum(rep_totals.values()) > 0,
                    f"{dict(rep_totals)}" + (f", not hits: {not_hit}" if not_hit else "")))
    for kind in ("llm", "tool"):
        n_rec = sum(v for k, v in rec_totals.items() if k.startswith(kind + "_"))
        n_rep = sum(v for k, v in rep_totals.items() if k.startswith(kind + "_"))
        results.append((f"{kind} calls: as many as recorded", n_rec == n_rep, f"recorded {n_rec}, replayed {n_rep}"))

    a_stages = {s["stage"]: s for s in a["stages"]}
    b_stages = {s["stage"]: s for s in b["stages"]}
    for name in a_stages:
        sa, sb = a_stages[name], b_stages.get(name, {})
        same_out = sa.get("outputs") == sb.get("outputs")
        ta, tb = trace(recorded / seed / name), trace(replay / seed / name)
        same_trace = ta == tb
        detail = f"{len(ta)} LLM calls" if same_trace else f"{len(ta)} vs {len(tb)} calls, first difference at " \
            f"{next((i + 1 for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)) + 1)}"
        results.append((f"{name}: same conversation", same_trace, detail))
        results.append((f"{name}: same outputs", same_out, ", ".join(sorted(sa.get("outputs", {})))))
    s4a, s4b = a_stages.get("S4", {}), b_stages.get("S4", {})
    results.append(("S4 recount identical", s4a.get("recomputed_estimate") == s4b.get("recomputed_estimate"),
                    f"n_excess {s4a.get('recomputed_estimate', {}).get('n_excess')}"))
    s5a, s5b = a_stages.get("S5", {}), b_stages.get("S5", {})
    results.append(("S5 submission identical", s5a.get("submitted") == s5b.get("submitted"),
                    f"{s5b.get('submitted')}"))
    return results


def main(recorded: str, replay: str) -> int:
    recorded, replay = Path(recorded), Path(replay)
    rec_gw, rep_gw = gateway_counts(recorded), gateway_counts(replay)
    ok = True
    seeds = sorted(p.parent.name for p in replay.glob("*/run.json"))
    if not seeds:
        print("no replayed runs found")
        return 1
    for seed in seeds:
        print(f"== {seed}")
        for name, passed, detail in check_seed(recorded, replay, seed, rec_gw, rep_gw):
            ok &= passed
            print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
    print("acceptance test 1:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
