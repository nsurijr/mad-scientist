"""Validation and preview of a team's submission.

    python -m harness.validate check <submission> --golden-run runs/<batch>/s0 [--image pipeline.sif | --local]
    python -m harness.validate preview-report <preview run> --golden-run runs/<batch>/s0

check is free and runs before any eval:
1. Static: manifest.yaml is well formed (harness/manifest.py); attack.py and WRITEUP.md exist; the
   hooks attack.py implements are all declared.
2. Exercise, twice, each time in a fresh hook host: every hook gets what it would get on the public
   seed's golden run (golden_calls): at each stage replace_agent, on_input on the reference, on the
   seed's full-column files (offered empty) and on each hand-over file, on_context on the task
   files, and on_tool_result on every tool result the
   golden run's agents saw. The two runs must agree call for call (hooks must be deterministic),
   and no hook may change anything, or replace a stage, the manifest does not declare
   (runtime_problems, which the scorer applies again to every eval's run.json).
   Unlike a preview, this reaches all five stages, because it never runs the pipeline.

preview-report turns a preview eval (harness/runner.py --preview against a gateway in eval mode)
into what the team sees: where the attack begins, the first call that would need inference, the
diff against the nearest golden request, and the share of the golden run's tokens that a live
eval would re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from harness.hookhost import HookError, HookHost, apptainer_command
from harness.manifest import REQUIRED_FILES, STAGE_NAMES, Manifest
from harness.manifest import load as load_manifest
from harness.stages import DATA_FULL, REFERENCE, REFERENCE_FULL, ROOT, STAGE_BY_NAME, STAGES, Hooks

TASK_FILES = ("brief.txt", "reprompt.txt", "numeric_questions.json")


class GoldenRun:
    """A recorded honest run folder (harness/runner.py layout), as hooks would have seen it."""

    def __init__(self, run_dir: Path, seeds_dir: Path = ROOT / "data" / "seeds"):
        self.dir = Path(run_dir)
        self.manifest = json.loads((self.dir / "run.json").read_text())
        # The seed's full-column files, which the runner offers to on_input but never mounts unasked
        seed = self.manifest.get("seed")
        self.full = {n: p for n in (DATA_FULL, REFERENCE_FULL)
                     if seed and (p := Path(seeds_dir) / seed / n).is_file()}

    def readable(self, stage: str, name: str) -> bytes:
        """What ctx.read returns at a stage, by the runner's rule (StagedRun.readable)."""
        if name in ("c_data.csv", REFERENCE):
            return (self.dir / "data" / name).read_bytes()
        if name in self.full:
            return self.full[name].read_bytes()
        path = self.dir / stage / "in" / name
        if name in STAGE_BY_NAME[stage].inputs and path.is_file():
            return path.read_bytes()
        raise FileNotFoundError(f"{name} is not readable at {stage}")

    def docs(self, stage: str) -> dict[str, str]:
        return {n: (self.dir / stage / "task" / n).read_bytes().decode() for n in TASK_FILES
                if (self.dir / stage / "task" / n).is_file()}

    def tool_calls(self, stage: str) -> list[dict[str, Any]]:
        """Every tool call the stage's agents made, with the result they saw, in the order the
        results reached them. Taken from the requests, where AoD's history holds each call (with its
        rewritten id, patch P1) and its result; each agent's requests extend its previous one until
        a fresh instance (a new reviewer) starts over."""
        path = self.dir / stage / "trace" / "trace.jsonl"
        if not path.is_file():
            return []
        calls, history = [], {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            agent, messages = rec["agent"], rec["request"]["messages"]
            prev = history.get(agent, [])
            new = messages[len(prev):] if messages[:len(prev)] == prev else messages
            history[agent] = messages
            asked = {tc["id"]: tc["function"] for m in messages if m.get("role") == "assistant"
                     for tc in m.get("tool_calls") or []}
            for m in new:
                fn = asked.get(m.get("tool_call_id")) if m.get("role") == "tool" else None
                if fn is None:
                    continue
                try:
                    args = json.loads(fn["arguments"])
                except ValueError:
                    args = {}
                calls.append({"agent": agent, "tool": fn["name"], "args": args, "result": m["content"]})
        return calls

    def tokens(self) -> dict[str, int]:
        return {s["stage"]: (s.get("prompt_tokens") or 0) + (s.get("completion_tokens") or 0)
                for s in self.manifest["stages"]}


def exercise(host: HookHost, golden: GoldenRun) -> tuple[list[dict], list[str]]:
    """Every hook call the golden run would make, in the runner's order; returns the calls (with
    before and after digests) and the replaced stages."""
    host.reader = golden.readable
    hooks = Hooks(host)
    for stage in STAGE_NAMES:
        if hooks.replacement(stage) is not None:
            continue
        if hooks.has("on_input"):
            ref = golden.readable(stage, REFERENCE)
            hooks.apply("on_input", stage, REFERENCE, ref, {"name": REFERENCE, "data": ref})
            for name in golden.full:
                hooks.apply("on_input", stage, name, b"", {"name": name, "data": b""})
            for name in STAGE_BY_NAME[stage].inputs:
                data = golden.readable(stage, name)
                hooks.apply("on_input", stage, name, data, {"name": name, "data": data})
        docs = golden.docs(stage)
        hooks.apply("on_context", stage, "docs", docs, {"docs": docs})
        if hooks.has("on_tool_result"):
            for c in golden.tool_calls(stage):
                hooks.apply("on_tool_result", stage, c["tool"], c["result"],
                            {"tool": c["tool"], "args": c["args"], "result": c["result"]}, agent=c["agent"])
    return hooks.calls, sorted(hooks.replaced)


def runtime_problems(manifest: Manifest, calls: list[dict], replaced: list[str]) -> list[str]:
    """Hook changes or replacements at stages, or by hooks, the manifest does not declare."""
    problems = []
    for c in calls:
        if c["changed"] and (c["stage"] not in manifest.stages_touched or c["hook"] not in manifest.hooks):
            problems.append(f"{c['hook']} changed {c['what']} at {c['stage']}, which manifest.yaml does not declare")
    for stage in replaced:
        if stage not in manifest.stages_touched or "replace_agent" not in manifest.hooks:
            problems.append(f"replace_agent replaced {stage}, which manifest.yaml does not declare")
    return list(dict.fromkeys(problems))


def check(submission: Path, golden_dir: Path, make_host: Callable[[], HookHost],
          seeds_dir: Path | None = None) -> dict[str, Any]:
    """The check report; seeds_dir holds <seed>/ with the full-column files (default: data/seeds)."""
    report: dict[str, Any] = {"submission": str(submission), "problems": [], "warnings": [], "fired": [],
                              "replaced": [], "deterministic": None, "implemented": None}
    manifest, problems = load_manifest(submission)
    report["problems"] += problems
    report["problems"] += [f"{name} is missing" for name in REQUIRED_FILES if not (submission / name).is_file()]
    if report["problems"]:
        return {**report, "verdict": "reject"}

    runs = []
    for _ in range(2):
        try:
            host = make_host()
        except HookError as e:
            report["problems"].append(str(e))
            return {**report, "verdict": "reject"}
        try:
            report["implemented"] = host.hooks
            golden = GoldenRun(golden_dir) if seeds_dir is None else GoldenRun(golden_dir, seeds_dir)
            runs.append(exercise(host, golden))
        except HookError as e:
            report["problems"].append(str(e))
            return {**report, "verdict": "reject"}
        finally:
            host.close()

    undeclared = sorted(set(report["implemented"]) - set(manifest.hooks))
    if undeclared:
        report["problems"].append(f"attack.py implements {undeclared}, which manifest.yaml does not declare")
    (calls, replaced), (calls2, replaced2) = runs
    key = lambda c: (c["stage"], c["hook"], c["call_id"], c["after"])  # noqa: E731
    mismatch = next((a for a, b in zip(calls, calls2) if key(a) != key(b)), None)
    report["deterministic"] = mismatch is None and len(calls) == len(calls2) and replaced == replaced2
    if not report["deterministic"]:
        where = f" (first at {mismatch['stage']} {mismatch['hook']} {mismatch['call_id']})" if mismatch else ""
        report["problems"].append(f"the attack is not deterministic: two runs on the same inputs differ{where}")
    report["problems"] += runtime_problems(manifest, calls, replaced)
    report["fired"] = [f"{c['stage']} {c['hook']} {c['what']} ({c['call_id']})" for c in calls if c["changed"]]
    report["replaced"] = replaced
    handed_over = {name for stage in STAGES for name in stage.outputs}
    for stage, name in dict.fromkeys((c["stage"], c["what"]) for c in calls
                                     if c["changed"] and c["hook"] == "on_input" and c["what"] in handed_over):
        report["warnings"].append(f"{stage} on_input changes {name}, which an earlier stage wrote: the verifier "
                                  "detects any change to an earlier stage's output")
    if replaced and set(report["implemented"]) - {"replace_agent"}:
        report["warnings"].append(f"the other hooks are skipped at replaced stages {replaced}")
    for stage in manifest.stages_touched:
        if not any(f.startswith(stage + " ") for f in report["fired"]) and stage not in replaced:
            report["warnings"].append(f"nothing changed at {stage} on the public seed (it may on others)")
    report["calls"] = len(calls)
    return {**report, "verdict": "reject" if report["problems"] else "pass"}


def preview_report(preview_run: Path, golden_dir: Path) -> dict[str, Any]:
    """What a team sees for a preview eval."""
    m = json.loads((Path(preview_run) / "run.json").read_text())
    golden = GoldenRun(golden_dir)
    state = (m.get("eval") or {}).get("gateway") or {}
    first, stopped = state.get("first_live_call"), state.get("stopped") or {}
    tokens = golden.tokens()
    total = sum(tokens.values())
    reexecuted = 0
    if first is not None:
        live_stage = first["eval"].split("/")[-1]
        scripted = {s["stage"] for s in m["stages"] if (s.get("replaced") or {}).get("kind") == "scripted"}
        reused = {s["stage"]: (s.get("prompt_tokens") or 0) + (s.get("completion_tokens") or 0) for s in m["stages"]}
        later = STAGE_NAMES[STAGE_NAMES.index(live_stage) + 1:]
        reexecuted = max(0, tokens.get(live_stage, 0) - reused.get(live_stage, 0)) + sum(
            tokens.get(s, 0) for s in later if s not in scripted)
    return {
        "diverged_at": (m.get("attack") or {}).get("diverged_at"),
        "first_live_call": first,
        "stopped": stopped.get("error") or ("attack error" if (m.get("attack") or {}).get("error") else None),
        "attack_error": (m.get("attack") or {}).get("error"),
        "diff": stopped.get("report"),
        "cache": state.get("calls", {}),
        "golden_tokens": total,
        "estimated_live_tokens": reexecuted,
        "estimated_reexecution_fraction": round(reexecuted / total, 3) if total else None,
    }


def format_check(report: dict[str, Any]) -> str:
    lines = [f"{report['verdict'].upper()}: {report['submission']}"]
    lines += [f"  problem: {p}" for p in report["problems"]]
    lines += [f"  warning: {w}" for w in report["warnings"]]
    if report.get("calls") is not None:
        lines.append(f"  exercised {report['calls']} hook calls twice on the public seed; "
                     f"deterministic: {report['deterministic']}")
    lines += [f"  replaces {s}" for s in report["replaced"]]
    lines += [f"  changes {f}" for f in report["fired"][:20]]
    if len(report["fired"]) > 20:
        lines.append(f"  ... and {len(report['fired']) - 20} more changes")
    return "\n".join(lines)


def format_preview(report: dict[str, Any]) -> str:
    lines = [f"attack begins at: {report['diverged_at'] or 'nowhere (nothing changed: the run is the honest one)'}"]
    if report["attack_error"]:
        lines.append(f"the attack failed: {report['attack_error']}")
    if report["first_live_call"]:
        f = report["first_live_call"]
        lines.append(f"first call that needs inference: {f['eval']} call {f['call']} ({f['kind']})")
    lines.append(f"estimated live share of a full eval: {report['estimated_reexecution_fraction']} "
                 f"(~{report['estimated_live_tokens']:,} of {report['golden_tokens']:,} golden tokens)")
    lines.append(f"cache: {report['cache']}")
    diff = report.get("diff") or {}
    if diff.get("diff"):
        lines.append(f"request diff (message {diff['first_difference']} of {diff['messages'][1]}, against the "
                     f"nearest golden request):")
        lines.append(diff["diff"])
    return "\n".join(lines)


def local_host(submission: Path, shared: dict | None = None) -> HookHost:
    """A hook host as a plain subprocess: for team machines without apptainer, never for evals."""
    return HookHost([sys.executable, "-m", "harness.hookhost", "serve", str(submission)], seed="s0",
                    reader=lambda stage, name: b"", shared=shared, env={**os.environ, "PYTHONPATH": str(ROOT)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("submission", type=Path)
    c.add_argument("--golden-run", type=Path, required=True, help="the public seed's golden run folder")
    c.add_argument("--image", type=Path, help="the pipeline image: hooks run in the apptainer sandbox")
    c.add_argument("--local", action="store_true", help="run hooks as a plain subprocess (no sandbox)")
    c.add_argument("--json", type=Path, help="also write the report here")
    pr = sub.add_parser("preview-report")
    pr.add_argument("preview_run", type=Path)
    pr.add_argument("--golden-run", type=Path, required=True)
    pr.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "check":
        if not (args.image or args.local):
            ap.error("give --image (sandboxed) or --local")
        submission = args.submission.resolve()
        manifest, _ = load_manifest(submission)
        shared = manifest.shared if manifest else {}

        def make_host() -> HookHost:
            if args.local:
                return local_host(submission, shared)
            return HookHost(apptainer_command(args.image, submission, ROOT / "harness"), seed="s0",
                            reader=lambda stage, name: b"", shared=shared)

        report = check(submission, args.golden_run, make_host)
        print(format_check(report))
    else:
        report = preview_report(args.preview_run, args.golden_run)
        print(format_preview(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report.get("verdict", "pass") == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
