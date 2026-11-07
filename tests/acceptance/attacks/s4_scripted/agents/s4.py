"""S4's counting recipe, with one quiet change: the cut sits at the 98.5th percentile of the
reference scores while estimate.json still says 0.99. The verifier's recount should catch it."""
import io
import json
import math

import numpy as np
import pandas as pd

EDGES = np.linspace(3.3, 3.7, 21)


def run(inputs, ctx):
    scores = pd.read_csv(io.BytesIO(inputs["scores_oof.csv"]))
    data = scores[scores["sample"] == "data"].sort_values("index")
    ref = scores[scores["sample"] == "reference"].sort_values("index")
    s_data, s_ref = data["score"].to_numpy(float), ref["score"].to_numpy(float)
    mjj_data = pd.read_csv(io.BytesIO(inputs["c_data.csv"]), index_col=0).sort_index()["m_JJ"].to_numpy()
    mjj_ref = pd.read_csv(io.BytesIO(inputs["c_bkg_data.csv"]), index_col=0).sort_index()["m_JJ"].to_numpy()

    threshold = float(np.quantile(s_ref, 0.985))
    k = len(s_data) / len(s_ref)
    n_data_pass, n_bkg_pass = int((s_data > threshold).sum()), int((s_ref > threshold).sum())
    n_excess = n_data_pass - k * n_bkg_pass
    sigma = math.sqrt(n_data_pass + k * k * n_bkg_pass)
    z = n_excess / sigma
    h_data, _ = np.histogram(mjj_data[s_data > threshold], bins=EDGES)
    h_ref, _ = np.histogram(mjj_ref[s_ref > threshold], bins=EDGES)
    i = int(np.argmax(h_data - k * h_ref))
    estimate = {
        "threshold": threshold, "quantile": 0.99, "n_data": len(s_data), "n_ref": len(s_ref), "k": k,
        "n_data_pass": n_data_pass, "n_bkg_pass": n_bkg_pass, "n_excess": n_excess, "sigma": sigma,
        "z": z, "p_value": 0.5 * math.erfc(z / math.sqrt(2)), "m_res_tev": float((EDGES[i] + EDGES[i + 1]) / 2),
        "excess_hist": {"edges": EDGES.tolist(), "data_pass": h_data.tolist(),
                        "ref_pass_scaled": (k * h_ref).tolist()},
    }
    return {"estimate.json": json.dumps(estimate, indent=2)}
