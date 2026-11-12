"""Paired bootstrap of (raw tool-call count AUC minus detector AUC), ATBench held-out protocol (referee item 2).

Same 5-fold split over safe traces as heldout_check.run_atbench. Per resample: positives of one
category are resampled with replacement once; each fold's test negatives are resampled with
replacement; detector and count AUC are recomputed on that same resample, averaged over folds,
differenced. Count direction is the sign found by heldout_check.trivial on the full data.
Resampling uses multinomial weights over a precomputed pair-comparison matrix (exact weighted AUC).
Reported: seed 42 alone, and all seeds 42-46 (per resample, differences averaged over seeds).

Run: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/paired_bootstrap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_check as H  # noqa: E402
from heldout_check import (SEEDS, MODELS, AgentTraceFeatureExtractor, UBFSNormalizer, COUNT_COL,  # noqa: E402
                           load_atbench, fit_score, auc, trivial)

OUT = H.ROOT / "results" / "tables" / "paired_bootstrap.json"
B = 2000


def cmp_matrix(sp, sn):
    return (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])


def main():
    d = load_atbench()
    Xraw, _, _ = AgentTraceFeatureExtractor().extract_batch(d["trajectories"])
    X = UBFSNormalizer(method="zscore").fit_transform(Xraw)
    owasp = np.array(d["owasp_labels"])
    normal_idx = np.where(owasp == "")[0]
    cats = sorted(set(owasp) - {""})
    cat_pos = {c: np.where(owasp == c)[0] for c in cats}
    count = Xraw[:, COUNT_COL]
    sign = {c: (1 if trivial(np.r_[np.zeros(len(normal_idx)), np.ones(len(p))], count[np.r_[normal_idx, p]])["sign"] == "+" else -1)
            for c, p in cat_pos.items()}
    # M[(n, c)][seed] = list over folds of (M_detector, M_count)
    M = {}
    for n in MODELS:
        for seed in SEEDS:
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(normal_idx):
                s = fit_score(n, X[normal_idx[tr]], X, seed)
                for c, pos in cat_pos.items():
                    neg = normal_idx[te]
                    M.setdefault((n, c), {}).setdefault(seed, []).append(
                        (cmp_matrix(s[pos], s[neg]), cmp_matrix(sign[c] * count[pos], sign[c] * count[neg]), te))
        print("scored", n, flush=True)

    def wauc(Mx, w, v):
        return w @ Mx @ v / (w.sum() * v.sum())

    N = len(normal_idx)
    out = {}
    for n in MODELS:
        out[n] = {}
        for c in cats:
            npos = len(cat_pos[c])
            rng = np.random.RandomState(42)
            diffs_42, diffs_all = np.empty(B), np.empty(B)
            for b in range(B):
                w = rng.multinomial(npos, np.ones(npos) / npos).astype(float)
                # one weight per safe trace, shared by every seed and fold: the same traces are the negatives in all of them
                va = rng.multinomial(N, np.ones(N) / N).astype(float)
                per_seed = []
                for seed in SEEDS:
                    ds = []
                    for Md, Mc, te in M[(n, c)][seed]:
                        v = va[te]
                        if v.sum():
                            ds.append(wauc(Mc, w, v) - wauc(Md, w, v))
                    per_seed.append(np.mean(ds))
                diffs_42[b], diffs_all[b] = per_seed[0], np.mean(per_seed)
            pt = lambda seeds: float(np.mean([np.mean([wauc(Mc, np.ones(Mc.shape[0]), np.ones(Mc.shape[1])) -
                                                       wauc(Md, np.ones(Md.shape[0]), np.ones(Md.shape[1]))
                                                       for Md, Mc, _ in M[(n, c)][s]]) for s in seeds]))
            cell = {}
            for tag, dd, seeds in (("seed42", diffs_42, SEEDS[:1]), ("all_seeds", diffs_all, SEEDS)):
                lo, hi = np.percentile(dd, [2.5, 97.5])
                cell[tag] = {"point_count_minus_detector": pt(seeds), "ci95": [float(lo), float(hi)],
                             "excludes_0": bool(lo > 0 or hi < 0)}
            out[n][c] = cell
    json.dump({"resamples": B, "resample_seed": 42, "count_sign": sign,
               "note": "The count sign is fixed before resampling, from all data. Negative weights are per safe trace and shared across seeds.",
               "cells": out}, open(OUT, "w"), indent=2)
    for tag in ("seed42", "all_seeds"):
        print(f"\n### count AUC minus detector AUC, {tag}, {B} resamples\n")
        print("| category | detector | point | 95% CI | excludes 0 |")
        print("|---|---|---|---|---|")
        for c in cats:
            for n in MODELS:
                r = out[n][c][tag]
                print(f"| {c} | {n} | {r['point_count_minus_detector']:+.3f} | {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f} | {r['excludes_0']} |")


if __name__ == "__main__":
    main()
