"""Acceptance test 2 verdicts from an acc2 batch (deploy/bouchet/acceptance2.sbatch).

    python deploy/bouchet/acceptance2_check.py runs/acc2-<job> s4_context s4_tool_result

For each attack: no strict-mode miss anywhere, every call before the attack's first change served
from golden, and the eval stopped at its first live call, in the attack's stage (the next stage
for a scripted replacement, which makes no model calls).
"""

import json
import sys
from collections import Counter
from pathlib import Path

STAGES = ["S1", "S2", "S3", "S4", "S5"]


def check(batch: Path, attack: str, log: list[dict]) -> bool:
    manifest = json.loads((batch / attack / "s0" / "run.json").read_text())
    state = manifest.get("eval", {}).get("gateway", {})
    eval_id = manifest.get("eval", {}).get("eval_id")
    calls = [r for r in log if r.get("eval_id") == eval_id and r["kind"] != "control"]
    outcomes = Counter(f"{r['kind']}_{r['outcome']}" for r in calls)
    before = calls[:-1]  # everything before the preview stop
    stage = (manifest.get("attack") or {}).get("diverged_at", "") or ""
    first_live = state.get("first_live_call") or {}
    expect = stage.split(" ")[0]
    record = next((s for s in manifest["stages"] if s["stage"] == expect), {})
    if (record.get("replaced") or {}).get("kind") == "scripted":  # no model calls: the next stage needs the first
        expect = STAGES[STAGES.index(expect) + 1]
    checks = {
        "attack fired": bool(stage),
        "no strict-mode miss": outcomes["llm_miss"] + outcomes["tool_miss"] == 0,
        "every earlier call from golden": all(r["outcome"] == "hit" and r.get("ns") != "acc2" for r in before),
        "stopped at the first live call": (state.get("stopped") or {}).get("error") == "preview_stop",
        f"first live call in {expect}": first_live.get("eval") == f"s0/{expect}",
    }
    per_stage = Counter(r["eval"].split("/")[-1] for r in before)
    print(f"{attack}: diverged at {stage or '-'}; first live call {first_live}")
    print(f"  calls before it: {len(before)} ({dict(sorted(per_stage.items()))}); outcomes {dict(outcomes)}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(checks.values())


def main() -> int:
    batch, attacks = Path(sys.argv[1]), sys.argv[2:]
    log = [json.loads(line) for line in (batch / "gateway.jsonl").read_text().splitlines()]
    ok = [check(batch, a, log) for a in attacks]
    print(f"acceptance test 2: {'PASS' if all(ok) else 'FAIL'} ({sum(ok)}/{len(ok)} attacks)")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
