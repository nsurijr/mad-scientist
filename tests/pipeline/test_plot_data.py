import json
import os
import subprocess
import sys

import pytest

from conftest import SITE
from tools.view_plot_data import ViewPlotData
from utility.md_reporter import MDReporter

pytest.importorskip("matplotlib")

SCRIPT = """
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
x = np.random.normal(3.5, 0.1, 1000)
print(repr(np.random.rand()))
plt.hist(x, bins=20, label="m_JJ")
plt.plot([3.3, 3.7], [0, 50], label="fit")
plt.xlabel("m_JJ [TeV]")
plt.ylabel("events")
plt.legend()
plt.savefig("mjj")
"""


def run(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(SITE), "HARNESS_SEED": "11", "MPLBACKEND": "Agg"}
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=tmp_path, env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_seeded_rng_repeats(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(), b.mkdir()
    assert run(a) == run(b)


def test_savefig_records_plot_data(tmp_path):
    run(tmp_path)
    data = json.loads((tmp_path / "mjj.png.json").read_text())
    ax = data["axes"][0]
    assert ax["xlabel"] == "m_JJ [TeV]" and ax["legend"] == ["m_JJ", "fit"]
    assert ax["bars"][0]["n"] == 20 and sum(ax["bars"][0]["height"]) == 1000
    assert ax["lines"][0]["y"] == [0.0, 50.0]


def test_view_plot_data_renders_text(tmp_path):
    run(tmp_path)
    tool = ViewPlotData(str(tmp_path) + "/", MDReporter(str(tmp_path) + "/"))
    text = tool.view_plot_data(["mjj.png", "missing.png"])
    assert "x: m_JJ [TeV] (linear" in text
    assert "bars 'm_JJ' (20 bars)" in text and "line 'fit' (2 points)" in text
    assert "No plot data recorded for: ['missing.png']" in text
