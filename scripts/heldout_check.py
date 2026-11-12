"""Held-out-negative check for Exp 1 (TRAIL), Exp 3 (synthetic) and Exp 12 (ATBench).

The published protocol fits each detector on ALL normal samples and then scores all
samples, so every evaluated negative was a training point. This script reruns that
protocol (`in_sample`) next to a protocol where the evaluated negatives are unseen
(`held_out`), plus a no-model baseline: AUC of the raw `primary_event_count` UBFS
column (tool_call_count for agent traces), taken before z-scoring.

Run: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/heldout_check.py
Note: Exp 3 fits the z-score normaliser on normals+anomalies together; kept as is.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_experiments import (  # noqa: E402
    SEEDS, AgentTraceFeatureExtractor, UBFSNormalizer, IsolationForestDetector,
    LSTMAutoencoderDetector, DeepClusteringDetector, compute_metrics,
    load_trail_dataset, get_trail_labels, generate_anomalous_traces,
    EXCESSIVE_AGENCY,
)
from src.data.atbench_loader import load_atbench  # noqa: E402
from src.features.ubfs_schema import ubfs_feature_names  # noqa: E402

# Copied from experiment_1/3/12 (they are local to those functions).
MODELS = {
    "IsolationForest": (IsolationForestDetector, {"n_estimators": 200, "contamination": "auto"}),
    "LSTMAutoencoder": (LSTMAutoencoderDetector, {"epochs": 30, "batch_size": 16, "device": "cpu", "verbose": False}),
    # device pinned: "auto" picks cuda or mps, and the scores then differ by up to 0.04 between machines;
    # cpu gives the same scores on macOS (torch 2.15) and Linux (torch 2.13).
    "DeepClustering": (DeepClusteringDetector, {"pretrain_epochs": 30, "batch_size": 16, "device": "cpu"}),
}
COUNT_COL = ubfs_feature_names().index("primary_event_count")
OUT = ROOT / "results" / "tables" / "heldout_check.json"


def fit_score(name, X_train, X_eval, seed):
    cls, kw = MODELS[name]
    m = cls(**{**kw, "seed": seed})
    if name == "LSTMAutoencoder":
        m.fit(X_train[:, None, :])
        return m.score(X_eval[:, None, :])
    m.fit(X_train)
    return m.score(X_eval)


def auc(y, s):
    return compute_metrics(y, s).auc_roc


def hanley_mcneil(a, n_p, n_n):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n_p - 1) * (q1 - a * a) + (n_n - 1) * (q2 - a * a)) / (n_p * n_n))
    return [float(max(0, a - 1.96 * se)), float(min(1, a + 1.96 * se))]


def trivial(y, count):
    a = float(roc_auc_score(y, count))
    return {"auc": max(a, 1 - a), "raw_auc": a, "sign": "+" if a >= .5 else "-"}  # "+": more calls = anomalous


def kfold_setting(X, normal_idx, cat_pos, seed, name):
    """One seed, one detector. Returns {cat: (in_sample_auc, held_out_auc)}."""
    ev = lambda idx_neg, idx_pos: (np.r_[idx_neg, idx_pos], np.r_[np.zeros(len(idx_neg)), np.ones(len(idx_pos))])
    s_all = fit_score(name, X[normal_idx], X, seed)
    out = {}
    for cat, pos in cat_pos.items():
        idx, y = ev(normal_idx, pos)
        out[cat] = [auc(y, s_all[idx]), []]
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(normal_idx):
        s = fit_score(name, X[normal_idx[tr]], X, seed)
        for cat, pos in cat_pos.items():
            idx, y = ev(normal_idx[te], pos)
            out[cat][1].append(auc(y, s[idx]))
    return {c: (v[0], float(np.mean(v[1]))) for c, v in out.items()}


def aggregate(per_seed, cats, n_pos, n_neg_total, triv):
    """per_seed: list over seeds of {cat: (in, held)} -> cell dict."""
    cells = {}
    for c in cats:
        i = float(np.mean([r[c][0] for r in per_seed]))
        h = float(np.mean([r[c][1] for r in per_seed]))
        n_fold, n_tot = n_neg_total // 5, n_neg_total
        n_p = n_pos[c]
        cells[c] = {"in_sample": i, "held_out": h, "n_pos": n_p, "n_neg_total": n_tot, "n_neg_per_fold": n_fold,
                    "ci95_hanley_mcneil_perfold_n": hanley_mcneil(h, n_p, n_fold),
                    "ci95_hanley_mcneil_total_n": hanley_mcneil(h, n_p, n_tot), "trivial": triv[c]}
    return cells


def run_synthetic():
    base = load_trail_dataset()["traces"]
    cats = ["ASI01", "ASI02", "ASI06", EXCESSIVE_AGENCY, "ASI10"]
    res = {n: [] for n in MODELS}
    triv_by_seed, npos, nneg = [], {}, None
    for seed in SEEDS:
        mixed, labels, owasp = generate_anomalous_traces(base, anomaly_ratio=0.5, categories=cats, seed=seed)
        Xraw, _, _ = AgentTraceFeatureExtractor().extract_batch(mixed)
        X = UBFSNormalizer(method="zscore").fit_transform(Xraw)
        owasp = np.array(owasp)
        normal_idx = np.where(labels == 0)[0]
        cat_pos = {c: np.where(owasp == c)[0] for c in cats}
        nneg, npos = len(normal_idx), {c: len(p) for c, p in cat_pos.items()}
        triv_by_seed.append({c: trivial(np.r_[np.zeros(nneg), np.ones(len(p))], Xraw[np.r_[normal_idx, p], COUNT_COL])
                             for c, p in cat_pos.items()})
        for n in MODELS:
            res[n].append(kfold_setting(X, normal_idx, cat_pos, seed, n))
            print("synthetic", seed, n, {c: round(v[1], 3) for c, v in res[n][-1].items()}, flush=True)
    triv = {}
    for c in cats:  # average the raw AUC over seeds first, so the sign belongs to the mean
        a = float(np.mean([t[c]["raw_auc"] for t in triv_by_seed]))
        triv[c] = {"auc": max(a, 1 - a), "raw_auc": a, "sign": "+" if a >= .5 else "-"}
    return {n: aggregate(res[n], cats, npos, nneg, triv) for n in MODELS}


def run_atbench():
    d = load_atbench()
    Xraw, _, _ = AgentTraceFeatureExtractor().extract_batch(d["trajectories"])
    X = UBFSNormalizer(method="zscore").fit_transform(Xraw)
    owasp, y = np.array(d["owasp_labels"]), d["labels"]
    normal_idx = np.where(owasp == "")[0]
    cats = sorted(set(owasp) - {""})
    cat_pos = {c: np.where(owasp == c)[0] for c in cats}
    npos = {c: len(p) for c, p in cat_pos.items()}
    triv = {c: trivial(np.r_[np.zeros(len(normal_idx)), np.ones(len(p))], Xraw[np.r_[normal_idx, p], COUNT_COL])
            for c, p in cat_pos.items()}
    out = {}
    for n in MODELS:
        per_seed = [kfold_setting(X, normal_idx, cat_pos, s, n) for s in SEEDS]
        out[n] = aggregate(per_seed, cats, npos, len(normal_idx), triv)
        print("atbench", n, {c: round(v["held_out"], 3) for c, v in out[n].items()}, flush=True)
    return out


def run_trail():
    trail = load_trail_dataset()
    y = get_trail_labels(trail["annotations"])
    Xraw, _, _ = AgentTraceFeatureExtractor().extract_batch(trail["traces"])
    X = UBFSNormalizer(method="zscore").fit_transform(Xraw)
    neg, pos = np.where(y == 0)[0], np.where(y == 1)[0]
    out = {}
    for n in MODELS:
        ins, held = [], []
        for seed in SEEDS:
            ins.append(auc(y, fit_score(n, X[neg], X, seed)))
            fr = []
            for k in range(len(neg)):
                s = fit_score(n, X[np.delete(neg, k)], X, seed)
                fr.append(np.mean((s[pos] > s[neg[k]]) + 0.5 * (s[pos] == s[neg[k]])))
            held.append(float(np.mean(fr)))
        h = float(np.mean(held))
        out[n] = {"in_sample": float(np.mean(ins)), "held_out": h, "n_pos": len(pos), "n_neg_total": len(neg),
                  "n_neg_per_fold": 1, "ci95_hanley_mcneil_total_n": hanley_mcneil(h, len(pos), len(neg)),
                  "trivial": trivial(y, Xraw[:, COUNT_COL])}
        print("trail", n, round(out[n]["in_sample"], 3), round(h, 3), flush=True)
    return out


def raw_atbench_length():
    """Length cue in the raw ATBench500 rows, with none of our parsing (paper Sec. IV)."""
    from datasets import load_dataset
    ds = load_dataset("AI45Research/ATBench", "ATBench500", split="test")
    y = np.array(ds["label"])
    n_msg = np.array([sum(len(conv) for conv in c) for c in ds["content"]])
    return {"auc_message_count": float(roc_auc_score(y, n_msg)),
            "auc_tools_offered": float(roc_auc_score(y, [len(t) for t in ds["tool_used"]])),
            "median_messages_safe": float(np.median(n_msg[y == 0])), "median_messages_unsafe": float(np.median(n_msg[y == 1])),
            "unsafe_with_10_or_more": int((n_msg[y == 1] >= 10).sum()), "safe_with_8_or_fewer": int((n_msg[y == 0] <= 8).sum()),
            "n_safe": int((y == 0).sum()), "n_unsafe": int((y == 1).sum())}


def published():
    t = ROOT / "results" / "tables"
    e3 = json.load(open(t / "experiment_3_owasp.json"))
    e12 = json.load(open(t / "experiment_12_real_owasp.json"))["phase_a_ubfs20"]
    e1 = json.load(open(t / "experiment_1_baselines.json"))["trail"]
    return ({n: {c: v["auc_roc"]["mean"] for c, v in e3[n]["per_category"].items()} for n in MODELS},
            {n: {c: v["auc_roc"]["mean"] for c, v in e12[n]["per_category"].items()} for n in MODELS},
            {n: e1[n]["auc_roc"]["mean"] for n in MODELS})


def table(title, res, pub):
    print(f"\n### {title}\n")
    print("| category | detector | in_sample | published | held_out | 95% CI (HM, per-fold n) | n_pos | n_neg (total/fold) | trivial (dir) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in next(iter(res.values())):
        for n in MODELS:
            r = res[n][c]
            ci = r["ci95_hanley_mcneil_perfold_n"]
            tv = r["trivial"]
            print(f"| {c} | {n} | {r['in_sample']:.3f} | {pub[n].get(c, float('nan')):.3f} | {r['held_out']:.3f} | "
                  f"{ci[0]:.2f}-{ci[1]:.2f} | {r['n_pos']} | {r['n_neg_total']}/{r['n_neg_per_fold']} | "
                  f"{tv['auc']:.3f} ({tv['sign']}) |")


if __name__ == "__main__":
    p3, p12, p1 = published()
    atb = run_atbench()
    load_trail_dataset()  # TRAIL is gated: fail here, before any result is written, if it is not cached
    syn, tr = run_synthetic(), run_trail()
    json.dump({"synthetic_exp3": syn, "atbench_exp12": atb, "trail_exp1": tr,
               "atbench_raw_length": raw_atbench_length(),
               "device": "cpu for LSTMAutoencoder and DeepClustering",
               "trivial_feature": "primary_event_count (tool_call_count), raw, before z-score",
               "published_check": {"exp3": p3, "exp12": p12, "exp1": p1}}, open(OUT, "w"), indent=2)
    table("Synthetic (Exp 3), 5-fold over normals, mean over 5 seeds", syn, p3)
    table("ATBench (Exp 12 phase A, UBFS-20), 5-fold over safe, mean over 5 seeds", atb, p12)
    print("\n### TRAIL in-domain (Exp 1), leave-one-normal-out\n")
    print("| detector | in_sample | published | held_out | HM CI (all negatives) | n_pos | n_neg | trivial (dir) |")
    print("|---|---|---|---|---|---|---|---|")
    for n in MODELS:
        r = tr[n]; ci = r["ci95_hanley_mcneil_total_n"]; tv = r["trivial"]
        print(f"| {n} | {r['in_sample']:.3f} | {p1[n]:.3f} | {r['held_out']:.3f} | {ci[0]:.2f}-{ci[1]:.2f} | "
              f"{r['n_pos']} | {r['n_neg_total']} | {tv['auc']:.3f} ({tv['sign']}) |")
    print("\nraw ATBench500 length cue:", json.load(open(OUT))["atbench_raw_length"])
    nan = float("nan")  # a category with no published value compares as different and is listed
    bad = [(s, n, c, round(r[n][c]["in_sample"], 3), round(p[n].get(c, nan), 3)) for s, r, p in
           (("syn", syn, p3), ("atb", atb, p12)) for n in MODELS for c in r[n]
           if not abs(r[n][c]["in_sample"] - p[n].get(c, nan)) <= .01]
    bad += [("trail", n, "-", round(tr[n]["in_sample"], 3), round(p1[n], 3)) for n in MODELS if abs(tr[n]["in_sample"] - p1[n]) > .01]
    print("\nin_sample vs published, cells off by >0.01:", bad or "none")
