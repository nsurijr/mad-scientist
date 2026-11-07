"""What a worker runs for one eval: the validator, a preview, or an eval and its score.

A runner has three methods, each given an unpacked submission folder and a scratch folder:
- check(submission) -> the validator's report (harness/validate.py check);
- preview(submission, eval_id, team, work) -> (the preview report, a trace digest);
- evaluate(submission, seed, eval_id, team, work, cache) -> (the scorer's record, a trace digest).

HarnessRunner runs the harness as preview.sbatch and eval_run.sbatch do, against a gateway in eval
mode that the worker's job script starts (SERVICE_* variables below). FakeRunner makes up results
from the manifest, through the real scorer, for tests and local trials.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any, Protocol

from harness.manifest import load as load_manifest
from harness.scorer import HONEST, score_eval
from service import traces


class Runner(Protocol):
    def check(self, submission: Path) -> dict[str, Any]: ...

    def preview(self, submission: Path, eval_id: str, team: str, work: Path) -> tuple[dict, dict | None]: ...

    def evaluate(self, submission: Path, seed: str, eval_id: str, team: str, work: Path,
                 cache: bool = True) -> tuple[dict, dict | None]: ...


class FakeRunner:
    """Results made up from manifest.yaml's shared["fake"], scored by harness.scorer:

        shared:
          fake: {pull: 2.5, hard_flags: [], reject: [], fail: null, live_tokens: 400000,
                 seeds: {s3: {pull: -1.0}}}

    pull is the attack's true shift in σ_hon (default 2 in the declared direction); reject lists
    validator problems; fail raises, as an eval that could not run; seeds overrides per seed."""

    HONEST_N, SIGMA, HONEST_Z = 175.0, 50.0, 0.5
    CALIBRATION = {"tau": 0.5, "sigma_floor": {s: 0.2 for s in ("S1", "S2", "S3", "S4", "S5")}, "sigma_z_hon": 1.0,
                   "provisional": True}

    def _fake(self, submission: Path, seed: str | None = None) -> tuple[Any, dict]:
        manifest, problems = load_manifest(submission)
        if manifest is None:
            raise ValueError("; ".join(problems))
        fake = dict(manifest.shared.get("fake") or {})
        return manifest, {**fake, **((fake.get("seeds") or {}).get(seed) or {})}

    def check(self, submission: Path) -> dict[str, Any]:
        _, fake = self._fake(submission)
        problems = list(fake.get("reject") or [])
        return {"submission": str(submission), "verdict": "reject" if problems else "pass", "problems": problems,
                "warnings": [], "fired": [], "replaced": [], "deterministic": not problems}

    def preview(self, submission: Path, eval_id: str, team: str, work: Path) -> tuple[dict, dict | None]:
        manifest, _ = self._fake(submission)
        stage = manifest.stages_touched[0]
        return ({"diverged_at": f"{stage} {manifest.hooks[0]}", "first_live_call": {"eval": f"s0/{stage}", "call": 1,
                 "kind": "llm"}, "stopped": "preview", "attack_error": None, "diff": None, "cache": {"llm_hit": 3},
                 "golden_tokens": 500_000, "estimated_live_tokens": 300_000, "estimated_reexecution_fraction": 0.6},
                self._digest(seed="s0", stage=stage))

    def evaluate(self, submission: Path, seed: str, eval_id: str, team: str, work: Path,
                 cache: bool = True) -> tuple[dict, dict | None]:
        manifest, fake = self._fake(submission, seed)
        if fake.get("fail"):
            raise RuntimeError(str(fake["fail"]))
        stage = manifest.stages_touched[0]
        pull = float(fake.get("pull", 2.0 * manifest.sign))
        replaced = [stage] if manifest.hooks == ("replace_agent",) else []
        calls = [] if replaced else [{"stage": stage, "hook": manifest.hooks[0], "what": "x", "changed": True}]
        if manifest.target == "n_excess":
            values = {"n_excess": self.HONEST_N + pull * self.SIGMA, "n_excess_uncertainty": self.SIGMA, "p_value": 0.01}
        else:
            values = {"n_excess": 0.0, "n_excess_uncertainty": self.SIGMA,
                      "p_value": 1 - NormalDist().cdf(self.HONEST_Z + pull)}
        run = {"seed": seed, "status": "complete", "flags": list(fake.get("hard_flags") or []),
               "stages": [{"stage": "S5", "submitted": values, "wall_s": 600.0}],
               "attack": {"calls": calls, "replaced": replaced, "diverged_at": f"{stage} {manifest.hooks[0]}"},
               "eval": {"gateway": {"first_live_call": {"eval": f"{seed}/{stage}", "call": 1, "kind": "llm"},
                                    "calls": {"llm_hit": 10, "llm_live": 20},
                                    "live_tokens": int(fake.get("live_tokens", 400_000))}}}
        honest = {"n_excess": self.HONEST_N, "sigma": self.SIGMA, "z": self.HONEST_Z}
        return score_eval(run, manifest, honest, self.CALIBRATION), self._digest(seed, stage)

    @staticmethod
    def _digest(seed: str, stage: str) -> dict:
        call = {"agent": "Researcher", "outcome": "live", "new": [{"role": "user", "content": "the brief"}],
                "reasoning": "", "content": "", "tool_calls": [{"name": "handoff_to_coder", "arguments": "{}"}]}
        return {"seed": seed, "status": "complete", "diverged_at": stage,
                "stages": [{"stage": stage, "exit_code": 0, "wall_s": 600.0, "replaced": False, "calls": [call]}]}


class HarnessRunner:
    """The harness, as on Bouchet. Settings from the environment:

        SERVICE_REPO          the repository (default: this checkout)
        SERVICE_GATEWAY_URL   the gateway in eval mode, e.g. http://127.0.0.1:20001/v1
        SERVICE_GATEWAY_SOCKET  its Unix socket (gateway --socket): stage containers run with no network
        SERVICE_GATEWAY_LOG   its JSON-lines log, for the trace viewer's hits and live calls
        SERVICE_SECRET        the gateway's eval-token secret (secrets/gateway.key)
        SERVICE_IMAGE         the pipeline image (pipeline.sif)
        SERVICE_GOLDEN_RUN    the public seed's golden run folder (validator and preview)
        SERVICE_SEALED        sealed/ (honest baselines)
        SERVICE_CALIBRATION   calibration/*.json
        SERVICE_LIVE_URL      for validation evals: an endpoint with no cache (not yet tested)
    """

    def __init__(self, env: dict[str, str] | None = None, timeout_s: int = 3 * 3600):
        env = dict(os.environ if env is None else env)
        self.repo = Path(env.get("SERVICE_REPO", Path(__file__).resolve().parents[1]))
        self.gateway = env.get("SERVICE_GATEWAY_URL")
        self.gateway_socket = env.get("SERVICE_GATEWAY_SOCKET")
        self.gateway_log = Path(env["SERVICE_GATEWAY_LOG"]) if env.get("SERVICE_GATEWAY_LOG") else None
        self.secret = env.get("SERVICE_SECRET", "secrets/gateway.key")
        self.image = env.get("SERVICE_IMAGE")
        self.golden_run = env.get("SERVICE_GOLDEN_RUN")
        self.sealed = Path(env.get("SERVICE_SEALED", "sealed"))
        self.calibration = Path(env.get("SERVICE_CALIBRATION", "calibration/provisional.json"))
        self.live_url = env.get("SERVICE_LIVE_URL")
        self.timeout_s = timeout_s
        required = (("SERVICE_GATEWAY_URL", self.gateway), ("SERVICE_GATEWAY_SOCKET", self.gateway_socket),
                    ("SERVICE_IMAGE", self.image), ("SERVICE_GOLDEN_RUN", self.golden_run))
        missing = [k for k, v in required if not v]
        if missing:
            raise ValueError(f"HarnessRunner needs {missing}")

    def _run(self, args: list[str], log: Path) -> int:
        with open(log, "w") as f:
            return subprocess.run([sys.executable, "-m", *args], cwd=self.repo, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=self.timeout_s).returncode

    def _runner_args(self, submission: Path, seed: str, run_dir: Path) -> list[str]:
        return ["harness.runner", "--seed", seed, "--run-dir", str(run_dir), "--image", self.image,
                "--max-calls", "30", "--submission", str(submission)]

    def _eval_args(self, eval_id: str, team: str) -> list[str]:
        return ["--base-url", self.gateway, "--tool-cache-url", f"{self.gateway}/tools", "--eval-id", eval_id,
                "--team", team, "--token-secret", self.secret, "--gateway-socket", self.gateway_socket]

    def _log_records(self, eval_id: str) -> list[dict]:
        if self.gateway_log is None or not self.gateway_log.is_file():
            return []
        out = []
        for line in self.gateway_log.read_text(encoding="utf-8").splitlines():
            if f'"{eval_id}"' in line:
                rec = json.loads(line)
                if rec.get("eval_id") == eval_id:
                    out.append(rec)
        return out

    def check(self, submission: Path) -> dict[str, Any]:
        out = submission.parent / "check.json"
        self._run(["harness.validate", "check", str(submission), "--golden-run", self.golden_run, "--image",
                   self.image, "--json", str(out)], submission.parent / "check.log")
        if not out.is_file():
            raise RuntimeError("the validator wrote no report (see check.log)")
        return json.loads(out.read_text())

    def preview(self, submission: Path, eval_id: str, team: str, work: Path) -> tuple[dict, dict | None]:
        run_dir, out = work / "run", work / "preview.json"
        self._run([*self._runner_args(submission, "s0", run_dir), *self._eval_args(eval_id, team), "--preview"],
                  work / "runner.log")
        self._run(["harness.validate", "preview-report", str(run_dir), "--golden-run", self.golden_run,
                   "--json", str(out)], work / "preview.log")
        if not out.is_file():
            raise RuntimeError("the preview wrote no report (see runner.log)")
        return json.loads(out.read_text()), traces.digest(run_dir, self._log_records(eval_id))

    def evaluate(self, submission: Path, seed: str, eval_id: str, team: str, work: Path,
                 cache: bool = True) -> tuple[dict, dict | None]:
        run_dir = work / "run"
        if cache:
            args = [*self._runner_args(submission, seed, run_dir), *self._eval_args(eval_id, team)]
        elif self.live_url:
            args = [*self._runner_args(submission, seed, run_dir), "--base-url", self.live_url]
        else:
            raise RuntimeError("validation evals need SERVICE_LIVE_URL (an endpoint with no cache)")
        self._run(args, work / "runner.log")
        if not (run_dir / "run.json").is_file():
            raise RuntimeError("the runner wrote no run.json (see runner.log)")
        manifest, problems = load_manifest(submission)
        if manifest is None:
            raise ValueError("; ".join(problems))
        honest = json.loads((self.repo / self.sealed / seed / HONEST).read_text())
        calibration = json.loads((self.repo / self.calibration).read_text())
        score = score_eval(json.loads((run_dir / "run.json").read_text()), manifest, honest, calibration)
        return score, traces.digest(run_dir, self._log_records(eval_id))


def from_env() -> Runner:
    kind = os.environ.get("SERVICE_RUNNER", "fake")
    if kind == "fake":
        return FakeRunner()
    if kind == "harness":
        return HarnessRunner()
    raise ValueError(f"SERVICE_RUNNER must be fake or harness, not {kind!r}")
