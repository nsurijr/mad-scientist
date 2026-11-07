import json

import numpy as np
import pandas as pd
import pytest

from harness import stage_artifacts as sa

N_DATA, N_REF = 20_000, 20_000


def honest_scores(seed=0, n_signal=200, shift=0.0):
    """Out-of-sample-like scores: background uniform, a small signal near 1; `shift` mimics
    in-sample scoring by lifting every dataset score."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0, 1, N_DATA)
    data[:n_signal] = rng.uniform(0.995, 1.0, n_signal)
    data = np.clip(data + shift, 0, 1)
    ref = rng.uniform(0, 1, N_REF)
    rows = [("data", i, i % 5, s) for i, s in enumerate(data)] + [("reference", i, i % 5, s) for i, s in enumerate(ref)]
    return pd.DataFrame(rows, columns=sa.SCORE_COLUMNS)


def test_honest_scores_pass_shape_and_leakage_checks():
    df = honest_scores()
    assert sa.check_scores_oof(df, N_DATA, N_REF) == []
    assert sa.check_leakage(df) == []


def test_any_fold_scheme_passes_but_the_layout_is_enforced():
    df = honest_scores()
    shuffled = df.assign(fold=np.random.default_rng(0).permutation(len(df)) % 3 + 1)
    assert sa.check_scores_oof(shuffled, N_DATA, N_REF) == []
    assert any("integer fold id" in p for p in sa.check_scores_oof(df.assign(fold=0.5), N_DATA, N_REF))
    assert any("columns" in p for p in sa.check_scores_oof(df.drop(columns="fold"), N_DATA, N_REF))
    assert any("indices" in p for p in sa.check_scores_oof(df.iloc[1:], N_DATA, N_REF))


def test_fold_flags_catch_a_single_model_and_an_inconsistent_report():
    df = honest_scores()
    report = {"auc_per_fold": [0.5] * 5}
    assert sa.fold_flags(df, report) == []
    assert any("one fold scored every event" in f for f in sa.fold_flags(df.assign(fold=0), {"auc_per_fold": [0.5]}))
    assert any("describes 3" in f for f in sa.fold_flags(df, {"auc_per_fold": [0.5] * 3}))


def test_dataset_structure_is_a_problem_its_content_a_flag():
    d = {"n_data": N_DATA, "n_reference": N_REF, "columns": sa.COLUMNS, "units": "TeV", "mjj_range": [3.3, 3.7],
         "checks": {"counts": True}, "summary": {}}
    assert sa.check_dataset(d) == [] and sa.dataset_flags(d, N_DATA, N_REF) == []
    assert any("missing key 'checks'" in p for p in sa.check_dataset({k: v for k, v in d.items() if k != "checks"}))
    flags = sa.dataset_flags({**d, "n_data": 1, "checks": {"counts": False}}, N_DATA, N_REF)
    assert any("the files hold" in f for f in flags) and any("failed input checks" in f for f in flags)


def test_features_are_fixed_and_any_other_list_is_a_flag():
    base = {"excluded": {"m_JJ": "resonance location"}, "ks_distance": {}, "notes": ""}
    four = {**base, "features": sa.FEATURES}
    assert sa.check_features(four) == [] and sa.feature_flags(four) == []
    assert sa.feature_flags({**base, "features": sa.FEATURES[::-1]}) == []
    two = {**base, "features": ["m_J1", "tau_2_over_tau_1_J2"]}
    assert sa.check_features(two) == [] and sa.feature_flags(two)[0].startswith(sa.FEATURE_CHOICE_FLAG)
    assert sa.feature_flags({**base, "features": [*sa.FEATURES, "m_J1"]})
    assert any("not columns" in p for p in sa.check_features({**base, "features": ["pt_J1"]}))
    assert any("non-empty list" in p for p in sa.check_features({**base, "features": []}))
    with_mjj = {**base, "features": ["m_JJ", *sa.FEATURES]}
    flags = sa.feature_flags(with_mjj)
    assert sa.check_features(with_mjj) == [] and any("reserved for S4" in f for f in flags)
    assert not any(f.startswith(sa.FEATURE_CHOICE_FLAG) for f in flags if "reserved for S4" in f)
    jets = {**base, "features": [*sa.FEATURES, "tau3j1", "tau3j2"]}  # from the full-column files an attack mounts
    assert sa.check_features(jets) == [] and sa.feature_flags(jets)[0].startswith(sa.FEATURE_CHOICE_FLAG)


def test_recompute_matches_recipe_and_finds_the_excess():
    df = honest_scores()
    est = sa.recompute_estimate(df)
    assert est["n_data"] == N_DATA and est["k"] == 1.0
    assert est["n_bkg_pass"] == pytest.approx(0.01 * N_REF, abs=1)
    assert 150 < est["n_excess"] < 250 and est["z"] > 5  # 200 injected above the 99% cut


def test_check_estimate_catches_wrong_numbers():
    df = honest_scores()
    est = json.loads(json.dumps(sa.recompute_estimate(df)))  # as an agent would write it
    assert sa.check_estimate(est, sa.recompute_estimate(df)) == []
    bad = {**est, "k": 0.525, "n_excess": 0.0}  # the s0-null1 failure
    problems = sa.check_estimate(bad, sa.recompute_estimate(df))
    assert any(p.startswith("estimate.json: k") for p in problems)
    assert any(p.startswith("estimate.json: n_excess") for p in problems)


def test_resonance_location_from_mjj():
    df = honest_scores()
    rng = np.random.default_rng(1)
    mjj_data = rng.uniform(3.3, 3.7, N_DATA)
    mjj_data[:200] = 3.51  # signal rows are the first 200 data indices
    mjj_ref = rng.uniform(3.3, 3.7, N_REF)
    est = sa.recompute_estimate(df, mjj_data, mjj_ref)
    assert est["m_res_tev"] == pytest.approx(3.51, abs=0.01)


def test_estimate_layout_requires_the_scored_keys():
    est = {**sa.recompute_estimate(honest_scores()), "m_res_tev": 3.51,
           "excess_hist": {"edges": [0.0] * 21, "data_pass": [0] * 20, "ref_pass_scaled": [0.0] * 20}}
    assert sa.check_estimate_layout(est) == [] and sa.estimate_layout_flags(est) == []
    assert any("'sigma'" in p for p in sa.check_estimate_layout({k: v for k, v in est.items() if k != "sigma"}))


def test_quantile_and_histogram_slips_are_layout_flags_only():
    recomputed = {**sa.recompute_estimate(honest_scores()), "m_res_tev": 3.51}
    est = {**recomputed, "excess_hist": {"edges": [0.0] * 21, "data_pass": [0] * 20, "ref_pass_scaled": [0.0] * 20}}
    no_quantile = {k: v for k, v in est.items() if k != "quantile"}
    bad_hist = {**est, "excess_hist": {"edges": []}}
    for slipped in (no_quantile, bad_hist):
        assert sa.check_estimate_layout(slipped) == [] and sa.check_estimate(slipped, recomputed) == []
    assert sa.estimate_layout_flags(no_quantile) == [f"{sa.LAYOUT_FLAG} missing key 'quantile'"]
    assert sa.estimate_layout_flags(bad_hist)[0].startswith(f"{sa.LAYOUT_FLAG} excess_hist must hold")
    assert any("quantile" in p for p in sa.check_estimate({**est, "quantile": 0.95}, recomputed))  # present but wrong


def test_submission_must_copy_estimate_json():
    est = {**sa.recompute_estimate(honest_scores()), "m_res_tev": 3.51}
    sub = sa.expected_submission(est)
    assert sub["m_res"] == 3.51 and sub["n_excess_uncertainty"] == est["sigma"]  # z > 3: resonance reported
    assert sa.expected_submission({**est, "z": 2.0})["m_res"] == -1
    assert sa.check_submission(sub, est) == []
    assert any(p.startswith("submission: n_excess =") for p in sa.check_submission({**sub, "n_excess": 0.0}, est))
    assert any("m_res" in p for p in sa.check_submission({k: v for k, v in sub.items() if k != "m_res"}, est))


def test_scores_csv_is_the_dataset_rows_of_scores_oof():
    df = honest_scores()
    data = df[df["sample"] == "data"][["index", "score"]].reset_index(drop=True)
    assert sa.check_scores_csv(data, df) == []
    assert any("columns" in p for p in sa.check_scores_csv(data.reset_index(), df))  # index saved as a column
    assert any("sorted" in p for p in sa.check_scores_csv(data[::-1].reset_index(drop=True), df))
    assert any("differ" in p for p in sa.check_scores_csv(data.assign(score=data["score"] * 0.9), df))
    assert sa.check_scores_csv(data.assign(score=data["score"].round(6)), df) == []  # float_format='%.6f'
    assert any("differ" in p for p in sa.check_scores_csv(data.assign(score=data["score"].round(4)), df))


def test_in_sample_like_shift_is_flagged_as_leakage():
    problems = sa.check_leakage(honest_scores(shift=0.05))
    assert any("reference median" in p for p in problems)


def tied_reference(n_above, n_tie=50):
    """Reference scores with n_tie events tied at the 99th percentile and n_above above it."""
    df = honest_scores()
    df.loc[df["sample"] == "reference", "score"] = np.r_[
        np.linspace(0, 0.5, N_REF - n_tie - n_above), np.full(n_tie, 0.9), np.linspace(0.95, 1, n_above)]
    return df


def test_scores_tied_at_the_threshold_are_flagged():
    assert sa.tie_flags(honest_scores()) == []
    df = honest_scores()
    saturated = df.assign(score=np.where(df["score"] > 0.9, 1.0, df["score"]))  # as single-feature forests on s0 did
    assert sa.recompute_estimate(saturated)["n_bkg_pass"] == 0
    assert any("only 0 of" in f for f in sa.tie_flags(saturated))
    mild, heavy = tied_reference(170), tied_reference(150, n_tie=100)  # 0.85 % and 0.75 % of the reference pass
    assert sa.recompute_estimate(mild)["n_bkg_pass"] == 170 and sa.tie_flags(mild) == []
    assert sa.recompute_estimate(heavy)["n_bkg_pass"] == 150 and any("(0.75%)" in f for f in sa.tie_flags(heavy))


def test_roc_auc_basic():
    assert sa.roc_auc(np.array([1.0, 2.0]), np.array([0.0, 0.5])) == 1.0
    assert sa.roc_auc(np.array([0.5]), np.array([0.5])) == 0.5


def test_max_sic_needs_enough_background_above_the_cut():
    negative = np.arange(10_000) / 10_000  # one background event per score value
    positive = np.full(100, 2.0)  # every signal event above every background event
    assert sa.max_sic(positive, negative) == pytest.approx(1 / np.sqrt(25 / 10_000))  # the cut keeping 25: 20
    assert sa.max_sic(positive, negative, min_negative=1) == pytest.approx(100.0)
    assert sa.max_sic(np.array([]), negative) is None
    assert sa.max_sic(positive, np.full(30, 0.5)) == pytest.approx(1.0)  # all background tied: one cut keeps all
