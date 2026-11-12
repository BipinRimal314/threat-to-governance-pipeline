"""Detectability dial for the synthetic ASI02 profile (referee item 3).

asi02_strength s in [0, 1]: 0 = the original profile (numeric attributes rescaled by about +/-10%);
s > 0 also inserts round(s * 0.5 * n_spans) tool-call spans at random positions, with tool names
that occur in other traces but not in this one. Everything else is heldout_check.run_synthetic:
same traces (5 categories, ratio 0.5), features, 5-fold protocol over normals, seeds 42-46.
Only ASI02 is reported. Count baseline = heldout_check.trivial on the raw tool-call count.

Run: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/dial_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_check as H  # noqa: E402
from heldout_check import (SEEDS, MODELS, AgentTraceFeatureExtractor, UBFSNormalizer, COUNT_COL, EXCESSIVE_AGENCY,  # noqa: E402
                           generate_anomalous_traces, load_trail_dataset, kfold_setting, trivial)

OUT = H.ROOT / "results" / "tables" / "dial_check.json"
FIG = H.ROOT / "paper" / "short" / "figures" / "dial.pdf"
STRENGTHS = [0, 0.1, 0.25, 0.5, 0.75, 1.0]


def main():
    base = load_trail_dataset()["traces"]
    cats = ["ASI01", "ASI02", "ASI06", EXCESSIVE_AGENCY, "ASI10"]
    res = {}
    for s in STRENGTHS:
        per = {n: [] for n in MODELS}
        cnt = []
        for seed in SEEDS:
            mixed, labels, owasp = generate_anomalous_traces(base, anomaly_ratio=0.5, categories=cats, seed=seed,
                                                             asi02_strength=s)
            Xraw, _, _ = AgentTraceFeatureExtractor().extract_batch(mixed)
            X = UBFSNormalizer(method="zscore").fit_transform(Xraw)
            owasp = np.array(owasp)
            normal_idx = np.where(labels == 0)[0]
            pos = np.where(owasp == "ASI02")[0]
            cnt.append(trivial(np.r_[np.zeros(len(normal_idx)), np.ones(len(pos))], Xraw[np.r_[normal_idx, pos], COUNT_COL])["raw_auc"])
            for n in MODELS:
                _in_sample, held_out = kfold_setting(X, normal_idx, {"ASI02": pos}, seed, n)["ASI02"]
                per[n].append(held_out)
        a = float(np.mean(cnt))
        res[str(s)] = {**{n: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v} for n, v in per.items()},
                       "count": {"mean": a, "per_seed_raw": cnt}}  # sign fixed: more calls = anomalous
        print("dial", s, {k: round(v["mean"], 3) for k, v in res[str(s)].items()}, flush=True)
    json.dump({"seeds": SEEDS, "strengths": STRENGTHS, "asi02_auc_held_out": res}, open(OUT, "w"), indent=2)
    print("\n| s | IsolationForest | LSTMAutoencoder | DeepClustering | raw tool-call count |\n|---|---|---|---|---|")
    for s in STRENGTHS:
        r = res[str(s)]
        print(f"| {s} | " + " | ".join(f"{r[k]['mean']:.3f}" for k in [*MODELS, "count"]) + " |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    style = {"IsolationForest": ("k", "o", "-"), "LSTMAutoencoder": ("0.45", "s", "-"),
             "DeepClustering": ("k", "^", ":"), "count": ("0.45", "x", "--")}
    label = {"count": "Tool-call count"}
    for k, (col, mk, ls) in style.items():
        ax.plot(STRENGTHS, [res[str(s)][k]["mean"] for s in STRENGTHS], color=col, marker=mk, ls=ls, lw=1.2, ms=4,
                label=label.get(k, k))
    ax.axhline(0.5, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel("Profile strength (0 = original profile)")
    ax.set_ylabel("AUC-ROC, held-out normals")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout(pad=0.3)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)


if __name__ == "__main__":
    main()
