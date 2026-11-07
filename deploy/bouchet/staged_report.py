"""Organizer-side report on staged runs (harness/runner.py), against the sealed truth labels.

    .venv/bin/python deploy/bouchet/staged_report.py runs/staged-<job> sealed

For each <batch>/<seed>/run.json it reports the run's status, per-stage cost, problems and flags,
the submitted n_excess and the recount, and the truth: S_pass, the injected signal events above
the honest threshold. n_excess estimates S_pass, so (n_excess - S_pass) / sigma should look like
a standard normal across seeds. Also the scores' signal-vs-background AUC (the paper's metric) and
maximum SIC, eps_S / sqrt(eps_B) over cuts keeping at least 25 background events,
both on the dataset's events. The paper's own maximum SIC (Agents of Discovery's
calc_sic_roc_auc, whose cuts keep about 5) is reported beside it, for comparison with the paper.
Writes <batch>/report.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from harness.stage_artifacts import max_sic, roc_auc  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline", "aod", "ai_agent"))
from utility.evaluation_functions import calc_sic_roc_auc  # noqa: E402


def truth_check(run_dir: str, seed: str, sealed: str, recount: dict) -> dict:
    truth = json.load(open(os.path.join(sealed, seed, "truth.json")))
    labels = pd.read_csv(os.path.join(sealed, seed, "c_labels.csv"), index_col=0).sort_index()["label"].to_numpy()
    scores = pd.read_csv(os.path.join(run_dir, "S4", "in", "scores_oof.csv"))
    s_data = scores[scores["sample"] == "data"].sort_values("index")["score"].to_numpy(dtype=float)
    is_sig = labels == 1
    s_pass = int((is_sig & (s_data > recount["threshold"])).sum())
    out = {"n_signal_injected": truth["n_signal_injected"], "s_pass": s_pass,
           "signal_auc": roc_auc(s_data[is_sig], s_data[~is_sig]) if is_sig.any() else None,
           "max_sic": max_sic(s_data[is_sig], s_data[~is_sig]),
           "max_sic_aod": float(calc_sic_roc_auc(labels, s_data)[4]) if is_sig.any() else None}
    if recount["sigma"] > 0:
        out["residual_sigma"] = (recount["n_excess"] - s_pass) / recount["sigma"]
    return out


def analyse(run_dir: str, sealed: str) -> dict:
    m = json.load(open(os.path.join(run_dir, "run.json")))
    stages = {s["stage"]: s for s in m["stages"]}
    row = {
        "name": os.path.basename(os.path.normpath(run_dir)), "seed": m["seed"], "null": m.get("null_variant"),
        "status": m["status"], "stopped_at": m["stopped_at"],
        "problems": m.get("problems", []), "flags": m.get("flags", []),
        "stages": {name: {"min": round(s["wall_s"] / 60, 1), "calls": s["llm_calls"],
                          "tokens": s["prompt_tokens"] + s["completion_tokens"],
                          "max_calls_reached": s["max_calls_reached"], "exit_code": s["exit_code"]}
                   for name, s in stages.items()},
    }
    row["tokens"] = sum(s["tokens"] for s in row["stages"].values())
    row["minutes"] = round(sum(s["min"] for s in row["stages"].values()), 1)
    if "S3" in stages:
        row["leakage"] = stages["S3"].get("leakage")
    if "S4" in stages and "recomputed_estimate" in stages["S4"]:
        row["recount"] = stages["S4"]["recomputed_estimate"]
        row["truth"] = truth_check(run_dir, m["seed"], sealed, row["recount"])
    if "S5" in stages:
        row["submitted"] = stages["S5"].get("submitted")
    return row


def show(row: dict) -> None:
    head = f"{row['name']}: {row['status']}" + (f" at {row['stopped_at']}" if row["stopped_at"] else "")
    print(f"{head}  {row['minutes']} min  {row['tokens']:,} tokens")
    for name, s in row["stages"].items():
        cap = "  (call cap reached)" if s["max_calls_reached"] else ""
        print(f"  {name}: {s['min']:>5} min  {s['tokens']:>9,} tok  exit {s['exit_code']}  {s['calls']}{cap}")
    if "recount" in row:
        r, t = row["recount"], row["truth"]
        sub = row.get("submitted") or {}
        print(f"  n_excess submitted {sub.get('n_excess')} +- {sub.get('n_excess_uncertainty')}  "
              f"recount {r['n_excess']:.1f} +- {r['sigma']:.1f}  z {r['z']:.2f}  m_res {r.get('m_res_tev')}")
        line = f"  truth: S_pass {t['s_pass']} of {t['n_signal_injected']} injected"
        if t.get("residual_sigma") is not None:
            line += f"  residual {t['residual_sigma']:+.2f} sigma"
        if t["signal_auc"] is not None:
            line += f"  signal AUC {t['signal_auc']:.3f}"
        if t.get("max_sic") is not None:
            line += f"  max SIC {t['max_sic']:.2f} (AoD's {t['max_sic_aod']:.2f})"
        print(line)
    if row.get("leakage"):
        lk = row["leakage"]
        print(f"  leakage stats: {lk['frac_data_above_ref_median']:.4f} above ref median, AUC {lk['auc_data_vs_ref']:.4f}")
    for p in row["problems"]:
        print(f"  PROBLEM {p}")
    for f in row["flags"]:
        print(f"  FLAG {f}")


def null_spread(rows: list[dict]) -> dict:
    """Per seed and perturbed stage: each null variant's n_excess shift from its base run, in units
    of the base run's sigma (sigma_hon), and the sd over base plus variants (sigma_null(k)
    must stay below 0.2 sigma_hon). A variant that did not complete counts as a failure. A base run
    with sigma 0 (scores tied at the threshold) takes sigma_hon as the
    median sigma of its completed variants."""
    base = {r["seed"]: r for r in rows if r["null"] is None and "recount" in r}
    out = {}
    for r in rows:
        if r["null"] is None or r["seed"] not in base:
            continue
        b = base[r["seed"]]["recount"]
        entry = out.setdefault(f"{r['seed']} {r['null']['stage']}",
                               {"sigma_hon": b["sigma"], "failed": [], "values": [b["n_excess"]], "sigmas": []})
        if "recount" in r and r["status"] == "complete":
            entry["values"].append(r["recount"]["n_excess"])
            entry["sigmas"].append(r["recount"]["sigma"])
        else:
            entry["failed"].append(r["name"])
    for entry in out.values():
        sigmas = [s for s in entry.pop("sigmas") if s > 0]
        if entry["sigma_hon"] == 0 and sigmas:
            entry["sigma_hon"], entry["sigma_hon_from"] = float(np.median(sigmas)), "variants"
        s, values = entry["sigma_hon"], entry["values"]
        entry["shifts_sigma"] = [(v - values[0]) / s for v in values[1:]] if s else []
        entry["sd_sigma"] = float(np.std(values, ddof=1) / s) if len(values) > 1 and s else None
    return out


def main(batch: str, sealed: str) -> None:
    rows = []
    for name in sorted(os.listdir(batch)):
        run_dir = os.path.join(batch, name)
        if os.path.isfile(os.path.join(run_dir, "run.json")):
            rows.append(analyse(run_dir, sealed))
            show(rows[-1])
    summary = {}
    # Calibration uses one honest run per seed; null variants of a seed are correlated with it.
    residuals = [r["truth"]["residual_sigma"] for r in rows
                 if r["null"] is None and r.get("truth", {}).get("residual_sigma") is not None]
    if len(residuals) > 1:
        summary["residuals"] = {"n": len(residuals), "mean": float(np.mean(residuals)), "sd": float(np.std(residuals, ddof=1))}
        print(f"\nresiduals over {len(residuals)} base runs: mean {np.mean(residuals):+.2f}, sd {np.std(residuals, ddof=1):.2f}")
    complete = [r for r in rows if r["status"] == "complete"]
    print(f"complete: {len(complete)} of {len(rows)} runs")
    spread = null_spread(rows)
    if spread:
        summary["null_spread"] = spread
        print("null variants (n_excess shift from the base run, in sigma_hon; gate: sd < 0.2):")
        for key, e in spread.items():
            shifts = ", ".join(f"{s:+.2f}" for s in e["shifts_sigma"])
            sd = f"{e['sd_sigma']:.2f}" if e["sd_sigma"] is not None else "-"
            sigma = f"  sigma_hon {e['sigma_hon']:.1f}" + (" (median of variants)" if e.get("sigma_hon_from") else "")
            print(f"  {key}: shifts [{shifts}]  sd {sd}{sigma}" + (f"  failed {e['failed']}" if e["failed"] else ""))
            print(f"    n_excess: {', '.join(f'{v:.1f}' for v in e['values'])} (base first)")
    leak = [r["leakage"] for r in rows if r.get("leakage")]
    if leak:
        frac = [x["frac_data_above_ref_median"] for x in leak]
        auc = [x["auc_data_vs_ref"] for x in leak]
        summary["leakage"] = {"n": len(leak), "frac_range": [min(frac), max(frac)], "auc_range": [min(auc), max(auc)]}
        print(f"leakage stats over {len(leak)} runs: fraction above reference median {min(frac):.4f}-{max(frac):.4f}, "
              f"AUC {min(auc):.4f}-{max(auc):.4f}")
    json.dump({"runs": rows, "summary": summary}, open(os.path.join(batch, "report.json"), "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
