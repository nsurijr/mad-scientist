import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sandbox(tmp_path, code, outputs=("estimate.json",)):
    """The folders a scripted replacement's sandbox mounts, as plain host folders."""
    for d in ("submission/agents", "task", "in", "data", "out"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "submission" / "agents" / "s4.py").write_text(code)
    (tmp_path / "in" / "scores_oof.csv").write_text("sample,index,fold,score\n")
    (tmp_path / "data" / "c_data.csv").write_text(",m_JJ\n0,3.5\n")
    (tmp_path / "data" / "c_bkg_data.csv").write_text(",m_JJ\n0,3.4\n")
    spec = {"stage": "S4", "seed": "s0", "harness_seed": 0, "shared": {"shift": 2}, "entrypoint": "agents/s4.py:run",
            "inputs": ["scores_oof.csv"], "outputs": list(outputs)}
    (tmp_path / "task" / "replacement.json").write_text(json.dumps(spec))
    return subprocess.run([sys.executable, "-m", "harness.replacement", "--submission", tmp_path / "submission",
                           "--task", tmp_path / "task" / "replacement.json", "--in", tmp_path / "in",
                           "--data", tmp_path / "data", "--out", tmp_path / "out"],
                          env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True)


def test_a_scripted_replacement_reads_its_inputs_and_writes_its_outputs(tmp_path):
    code = ("import json\n"
            "def run(inputs, ctx):\n"
            "    assert sorted(inputs) == ['c_bkg_data.csv', 'c_data.csv', 'scores_oof.csv']\n"
            "    header = inputs['scores_oof.csv'].decode().splitlines()[0]\n"
            "    return {'estimate.json': json.dumps({'header': header, 'shift': ctx.shared['shift'], 'stage': ctx.stage})}\n")
    proc = sandbox(tmp_path, code)
    assert proc.returncode == 0, proc.stderr
    assert json.loads((tmp_path / "out" / "estimate.json").read_text()) == {
        "header": "sample,index,fold,score", "shift": 2, "stage": "S4"}


def test_missing_or_extra_outputs_and_errors_fail_the_replacement(tmp_path):
    for i, code in enumerate(["def run(inputs, ctx):\n    return {}\n",
                              "def run(inputs, ctx):\n    return {'estimate.json': '{}', 'notes.txt': 'x'}\n",
                              "def run(inputs, ctx):\n    return {'estimate.json': inputs['truth.json']}\n"]):
        proc = sandbox(tmp_path / str(i), code)
        assert proc.returncode == 1 and "Traceback" in proc.stderr
        assert not (tmp_path / str(i) / "out" / "estimate.json").exists()
