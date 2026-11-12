"""Synthetic setting, folds grouped by base trace (referee item 1).

Each synthetic positive is a perturbed copy of one base trace. In heldout_check.py its base
is usually in the training normals. Here a positive is scored only by the fold model whose
test normals contain its base trace, so no model saw the positive's base.
Same folds, features and seeds as heldout_check.run_synthetic; `held_out` is recomputed and
checked against heldout_check.json.

Run: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/grouped_folds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_check as H  # noqa: E402
from heldout_check import (SEEDS, MODELS, AgentTraceFeatureExtractor, UBFSNormalizer, EXCESSIVE_AGENCY,  # noqa: E402
                           generate_anomalous_traces, load_trail_dataset, fit_score, auc)

OUT = H.ROOT / "results" / "tables" / "grouped_folds.json"


def main():
    base = load_trail_dataset()["traces"]
    cats = ["ASI01", "ASI02", "ASI06", EXCESSIVE_AGENCY, "ASI10"]
    res = {n: {c: {"held_out": [], "held_out_grouped": [], "n_pos_grouped": []} for c in cats} for n in MODELS}
    for seed in SEEDS:
        mixed, labels, owasp, bidx = generate_anomalous_traces(
            base, anomaly_ratio=0.5, categories=cats, seed=seed, return_base_idx=True)
        Xraw, _, _ = AgentTraceFeatureExtractor().extract_batch(mixed)
        X = UBFSNormalizer(method="zscore").fit_transform(Xraw)
        owasp = np.array(owasp)
        normal_idx = np.where(labels == 0)[0]
        cat_pos = {c: np.where(owasp == c)[0] for c in cats}
        for n in MODELS:
            fo = {c: [] for c in cats}; fg = {c: [] for c in cats}; npg = {c: [] for c in cats}
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(normal_idx):
                s = fit_score(n, X[normal_idx[tr]], X, seed)
                test_bases = set(bidx[normal_idx[te]].tolist())
                for c, pos in cat_pos.items():
                    ev = lambda p: (np.r_[normal_idx[te], p], np.r_[np.zeros(len(te)), np.ones(len(p))])
                    i, y = ev(pos)
                    fo[c].append(auc(y, s[i]))
                    pg = np.array([p for p in pos if bidx[p] in test_bases])
                    npg[c].append(len(pg))
                    if len(pg):
                        i, y = ev(pg)
                        fg[c].append(auc(y, s[i]))
            for c in cats:
                res[n][c]["held_out"].append(float(np.mean(fo[c])))
                res[n][c]["held_out_grouped"].append(float(np.mean(fg[c])))
                res[n][c]["n_pos_grouped"].append(float(np.mean(npg[c])))
            print("grouped", seed, n, {c: round(res[n][c]["held_out_grouped"][-1], 3) for c in cats}, flush=True)
    ref = json.load(open(H.ROOT / "results" / "tables" / "heldout_check.json"))["synthetic_exp3"]
    out = {}
    for n in MODELS:
        out[n] = {}
        for c in cats:
            r = {k: float(np.mean(v)) for k, v in res[n][c].items()}
            r["held_out_per_seed"], r["held_out_grouped_per_seed"] = res[n][c]["held_out"], res[n][c]["held_out_grouped"]
            r["caveat"] = "about 6 positives per fold and 30 per seed; compare the gap with the per-seed spread"
            r["held_out_matches_heldout_check_json"] = abs(r["held_out"] - ref[n][c]["held_out"]) < 1e-9
            out[n][c] = r
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n| category | detector | held_out | held_out_grouped | mean positives scored per fold | held_out == json |")
    print("|---|---|---|---|---|---|")
    for c in cats:
        for n in MODELS:
            r = out[n][c]
            print(f"| {c} | {n} | {r['held_out']:.3f} | {r['held_out_grouped']:.3f} | {r['n_pos_grouped']:.1f} | "
                  f"{r['held_out_matches_heldout_check_json']} |")


if __name__ == "__main__":
    main()
