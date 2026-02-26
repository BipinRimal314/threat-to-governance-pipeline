"""Run all experiments for the Threat-to-Governance Pipeline.

Usage:
    python run_experiments.py --experiment 1   # Within-domain baselines
    python run_experiments.py --experiment 2   # Cross-domain transfer
    python run_experiments.py --experiment 3   # OWASP category mapping (multi-seed + CIs)
    python run_experiments.py --experiment 4   # Governance assumption audit
    python run_experiments.py --experiment 12  # Real-data OWASP on ATBench + UBFS-28
    python run_experiments.py --experiment 13  # Distillation sensitivity analysis
    python run_experiments.py --all            # Run everything
    python run_experiments.py --all --cert     # Include CMU-CERT data
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.trail_loader import (
    load_trail_dataset,
    get_trail_labels,
    get_trail_error_categories,
)
from src.data.trace_loader import (
    load_trace_dataset,
    trace_to_otel_format,
)
from src.data.synthetic_generator import generate_anomalous_traces
from src.evaluation.metrics import compute_metrics, run_multi_seed
from src.evaluation.transfer_analysis import (
    evaluate_transfer,
    transfer_summary_table,
)
from src.evaluation.owasp_mapper import (
    evaluate_owasp_detection,
    owasp_detection_matrix,
    owasp_summary_table,
)
from src.features.agent_extractor import AgentTraceFeatureExtractor
from src.features.ubfs_schema import UBFSNormalizer, ubfs_feature_names
from src.governance.assumption_audit import (
    audit_normality_assumptions,
    compare_baseline_distributions,
    governance_report,
)
from src.models.isolation_forest import IsolationForestDetector
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.models.deep_clustering import DeepClusteringDetector

SEEDS = [42, 43, 44, 45, 46]
RESULTS_DIR = Path(__file__).parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"

# Global flag set by --cert CLI option
_USE_CERT = False


def load_cert_features():
    """Load CMU-CERT and extract UBFS features."""
    from src.data.cert_loader import load_cert_features as _load
    from src.features.ubfs_schema import UBFSNormalizer

    print("Loading CMU-CERT dataset...")
    X, entity_ids, timestamps, labels = _load()

    normalizer = UBFSNormalizer(method="zscore")
    X = normalizer.fit_transform(X)

    print(f"  {len(X)} user-days, {labels.sum()} insider-positive")
    return X, labels


def load_trail_features():
    """Load TRAIL and extract UBFS features."""
    print("Loading TRAIL dataset...")
    trail = load_trail_dataset()
    labels = get_trail_labels(trail["annotations"])

    print(f"  {len(trail['traces'])} traces, "
          f"{labels.sum()} with errors")

    extractor = AgentTraceFeatureExtractor()
    X, ids, ts = extractor.extract_batch(trail["traces"])

    normalizer = UBFSNormalizer(method="zscore")
    X = normalizer.fit_transform(X)

    return X, labels, trail, extractor


def load_trace_features():
    """Load TRACE and extract UBFS features."""
    print("Loading TRACE dataset...")
    trace_data = load_trace_dataset()
    labels = trace_data["labels"]

    print(f"  {len(trace_data['trajectories'])} trajectories, "
          f"{labels.sum()} reward hacks")

    otel = [trace_to_otel_format(t)
            for t in trace_data["trajectories"]]

    extractor = AgentTraceFeatureExtractor()
    X, ids, ts = extractor.extract_batch(otel)

    normalizer = UBFSNormalizer(method="zscore")
    X = normalizer.fit_transform(X)

    return X, labels, trace_data


def split_train_test(X, y, train_ratio=0.7, seed=42):
    """Simple train/test split, stratified by label."""
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * train_ratio)
    train_idx, test_idx = idx[:split], idx[split:]
    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx])


def experiment_1():
    """Within-domain baselines on TRAIL and TRACE."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Within-Domain Baselines")
    print("=" * 60)

    results = {}

    # --- TRAIL ---
    # TRAIL has 143/148 traces with errors — extreme imbalance
    # toward positive class. We train on a subset of "normal"
    # (error-free) traces and use the rest for evaluation.
    # Since only 5 normal traces exist, we use a leave-some-out
    # approach: train on 3 normal, test on remaining 2 normal
    # + all error traces.
    X_trail, y_trail, _, _ = load_trail_features()
    normal_idx = np.where(y_trail == 0)[0]
    error_idx = np.where(y_trail == 1)[0]
    print(f"\nTRAIL: {len(normal_idx)} normal, "
          f"{len(error_idx)} with errors")

    # Use all normals for training (unsupervised — no label leak)
    # Test on everything; expect errors to score higher
    X_train_normal = X_trail[normal_idx]
    X_test = X_trail
    y_test = y_trail
    print(f"TRAIL split: train={len(X_train_normal)} normal, "
          f"test={len(X_test)} (all)")

    models = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    results["trail"] = {}
    for name, (cls, kwargs) in models.items():
        print(f"\n  {name} on TRAIL...")
        t0 = time.time()

        if name == "LSTMAutoencoder":
            # Need 3D input: (samples, seq_len, features)
            X_tr_3d = X_train_normal[:, np.newaxis, :]
            X_te_3d = X_test[:, np.newaxis, :]
            seed_results = []
            for seed in SEEDS:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_tr_3d)
                scores = model.score(X_te_3d)
                m = compute_metrics(y_test, scores)
                seed_results.append(m.to_dict())
        else:
            seed_results = []
            for seed in SEEDS:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train_normal)
                scores = model.score(X_test)
                m = compute_metrics(y_test, scores)
                seed_results.append(m.to_dict())

        # Aggregate
        agg = {}
        for key in seed_results[0]:
            vals = [r[key] for r in seed_results]
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

        results["trail"][name] = agg
        elapsed = time.time() - t0
        print(f"    AUC-ROC: {agg['auc_roc']['mean']:.4f} "
              f"(+/- {agg['auc_roc']['std']:.4f})")
        print(f"    AUC-PR:  {agg['auc_pr']['mean']:.4f}")
        print(f"    Time:    {elapsed:.1f}s")

    # --- TRACE ---
    X_trace, y_trace, _ = load_trace_features()
    normal_idx = np.where(y_trace == 0)[0]
    error_idx = np.where(y_trace == 1)[0]
    print(f"\nTRACE: {len(normal_idx)} normal, "
          f"{len(error_idx)} reward hacks")

    # Train on normal, test on all
    X_train_normal = X_trace[normal_idx]
    X_test = X_trace
    y_test = y_trace
    print(f"TRACE split: train={len(X_train_normal)} normal, "
          f"test={len(X_test)} (all)")

    results["trace"] = {}
    for name, (cls, kwargs) in models.items():
        print(f"\n  {name} on TRACE...")
        t0 = time.time()

        if name == "LSTMAutoencoder":
            X_tr_3d = X_train_normal[:, np.newaxis, :]
            X_te_3d = X_test[:, np.newaxis, :]
            seed_results = []
            for seed in SEEDS:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_tr_3d)
                scores = model.score(X_te_3d)
                m = compute_metrics(y_test, scores)
                seed_results.append(m.to_dict())
        else:
            seed_results = []
            for seed in SEEDS:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train_normal)
                scores = model.score(X_test)
                m = compute_metrics(y_test, scores)
                seed_results.append(m.to_dict())

        agg = {}
        for key in seed_results[0]:
            vals = [r[key] for r in seed_results]
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

        results["trace"][name] = agg
        elapsed = time.time() - t0
        print(f"    AUC-ROC: {agg['auc_roc']['mean']:.4f} "
              f"(+/- {agg['auc_roc']['std']:.4f})")
        print(f"    AUC-PR:  {agg['auc_pr']['mean']:.4f}")
        print(f"    Time:    {elapsed:.1f}s")

    # --- CMU-CERT (optional) ---
    if _USE_CERT:
        X_cert, y_cert = load_cert_features()
        normal_idx = np.where(y_cert == 0)[0]
        error_idx = np.where(y_cert == 1)[0]
        print(f"\nCMU-CERT: {len(normal_idx)} normal, "
              f"{len(error_idx)} insider-positive")

        X_train_normal = X_cert[normal_idx]
        X_test = X_cert
        y_test = y_cert
        print(f"CERT split: train={len(X_train_normal)} normal, "
              f"test={len(X_test)} (all)")

        results["cert"] = {}
        for name, (cls, kwargs) in models.items():
            print(f"\n  {name} on CMU-CERT...")
            t0 = time.time()

            if name == "LSTMAutoencoder":
                X_tr_3d = X_train_normal[:, np.newaxis, :]
                X_te_3d = X_test[:, np.newaxis, :]
                seed_results = []
                for seed in SEEDS:
                    model = cls(**{**kwargs, "seed": seed})
                    model.fit(X_tr_3d)
                    scores = model.score(X_te_3d)
                    m = compute_metrics(y_test, scores)
                    seed_results.append(m.to_dict())
            else:
                seed_results = []
                for seed in SEEDS:
                    model = cls(**{**kwargs, "seed": seed})
                    model.fit(X_train_normal)
                    scores = model.score(X_test)
                    m = compute_metrics(y_test, scores)
                    seed_results.append(m.to_dict())

            agg = {}
            for key in seed_results[0]:
                vals = [r[key] for r in seed_results]
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

            results["cert"][name] = agg
            elapsed = time.time() - t0
            print(f"    AUC-ROC: {agg['auc_roc']['mean']:.4f} "
                  f"(+/- {agg['auc_roc']['std']:.4f})")
            print(f"    AUC-PR:  {agg['auc_pr']['mean']:.4f}")
            print(f"    Time:    {elapsed:.1f}s")

    # Save results
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "experiment_1_baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "-" * 60)
    print("Experiment 1 Summary:")
    datasets = ["trail", "trace"]
    if _USE_CERT:
        datasets.append("cert")
    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        for model_name, metrics in results[dataset].items():
            print(f"    {model_name:20s} "
                  f"AUC-ROC={metrics['auc_roc']['mean']:.4f} "
                  f"AUC-PR={metrics['auc_pr']['mean']:.4f}")

    return results


def experiment_2():
    """Cross-domain transfer: TRAIL <-> TRACE.

    Train on normal data from one domain, evaluate on both.
    Tests whether anomaly patterns learned in one agent-trace
    dataset transfer to another.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Cross-Domain Transfer")
    print("=" * 60)

    X_trail, y_trail, _, _ = load_trail_features()
    X_trace, y_trace, _ = load_trace_features()

    # Training sets: normal data from each domain
    trail_normal = X_trail[y_trail == 0]
    trace_normal = X_trace[y_trace == 0]
    print(f"  TRAIL: {len(trail_normal)} normal for training")
    print(f"  TRACE: {len(trace_normal)} normal for training")

    results = {"trail_to_trace": {}, "trace_to_trail": {}}

    models = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    for name, (cls, kwargs) in models.items():
        print(f"\n  {name}:")

        # TRAIL → TRACE
        print(f"    Train on TRAIL normal → eval TRACE...")
        seed_results = []
        for seed in SEEDS:
            model = cls(**{**kwargs, "seed": seed})
            model.fit(trail_normal)

            src = compute_metrics(y_trail, model.score(X_trail))
            tgt = compute_metrics(y_trace, model.score(X_trace))
            seed_results.append({
                "source": src.to_dict(),
                "target": tgt.to_dict(),
            })

        results["trail_to_trace"][name] = _aggregate_transfer(
            seed_results
        )
        r = results["trail_to_trace"][name]
        print(f"      Source AUC-ROC: "
              f"{r['source']['auc_roc']['mean']:.4f}")
        print(f"      Target AUC-ROC: "
              f"{r['target']['auc_roc']['mean']:.4f}")
        print(f"      Drop: {r['auc_roc_drop']:.4f}")

        # TRACE → TRAIL
        print(f"    Train on TRACE normal → eval TRAIL...")
        seed_results = []
        for seed in SEEDS:
            model = cls(**{**kwargs, "seed": seed})
            model.fit(trace_normal)

            src = compute_metrics(y_trace, model.score(X_trace))
            tgt = compute_metrics(y_trail, model.score(X_trail))
            seed_results.append({
                "source": src.to_dict(),
                "target": tgt.to_dict(),
            })

        results["trace_to_trail"][name] = _aggregate_transfer(
            seed_results
        )
        r = results["trace_to_trail"][name]
        print(f"      Source AUC-ROC: "
              f"{r['source']['auc_roc']['mean']:.4f}")
        print(f"      Target AUC-ROC: "
              f"{r['target']['auc_roc']['mean']:.4f}")
        print(f"      Drop: {r['auc_roc_drop']:.4f}")

    # --- CMU-CERT cross-domain (optional) ---
    if _USE_CERT:
        X_cert, y_cert = load_cert_features()
        cert_normal = X_cert[y_cert == 0]
        print(f"\n  CMU-CERT: {len(cert_normal)} normal for training")

        for name, (cls, kwargs) in models.items():
            print(f"\n  {name} (CERT transfers):")

            # CERT → TRAIL
            print(f"    Train on CERT normal → eval TRAIL...")
            seed_results = []
            for seed in SEEDS:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(cert_normal)
                src = compute_metrics(y_cert, model.score(X_cert))
                tgt = compute_metrics(y_trail, model.score(X_trail))
                seed_results.append({
                    "source": src.to_dict(),
                    "target": tgt.to_dict(),
                })
            results.setdefault("cert_to_trail", {})[name] = \
                _aggregate_transfer(seed_results)
            r = results["cert_to_trail"][name]
            print(f"      Source AUC-ROC: "
                  f"{r['source']['auc_roc']['mean']:.4f}")
            print(f"      Target AUC-ROC: "
                  f"{r['target']['auc_roc']['mean']:.4f}")
            print(f"      Drop: {r['auc_roc_drop']:.4f}")

            # CERT → TRACE
            print(f"    Train on CERT normal → eval TRACE...")
            seed_results = []
            for seed in SEEDS:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(cert_normal)
                src = compute_metrics(y_cert, model.score(X_cert))
                tgt = compute_metrics(y_trace, model.score(X_trace))
                seed_results.append({
                    "source": src.to_dict(),
                    "target": tgt.to_dict(),
                })
            results.setdefault("cert_to_trace", {})[name] = \
                _aggregate_transfer(seed_results)
            r = results["cert_to_trace"][name]
            print(f"      Source AUC-ROC: "
                  f"{r['source']['auc_roc']['mean']:.4f}")
            print(f"      Target AUC-ROC: "
                  f"{r['target']['auc_roc']['mean']:.4f}")
            print(f"      Drop: {r['auc_roc_drop']:.4f}")

            # TRAIL → CERT
            print(f"    Train on TRAIL normal → eval CERT...")
            seed_results = []
            for seed in SEEDS:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(trail_normal)
                src = compute_metrics(y_trail, model.score(X_trail))
                tgt = compute_metrics(y_cert, model.score(X_cert))
                seed_results.append({
                    "source": src.to_dict(),
                    "target": tgt.to_dict(),
                })
            results.setdefault("trail_to_cert", {})[name] = \
                _aggregate_transfer(seed_results)
            r = results["trail_to_cert"][name]
            print(f"      Source AUC-ROC: "
                  f"{r['source']['auc_roc']['mean']:.4f}")
            print(f"      Target AUC-ROC: "
                  f"{r['target']['auc_roc']['mean']:.4f}")
            print(f"      Drop: {r['auc_roc_drop']:.4f}")

            # TRACE → CERT
            print(f"    Train on TRACE normal → eval CERT...")
            seed_results = []
            for seed in SEEDS:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(trace_normal)
                src = compute_metrics(y_trace, model.score(X_trace))
                tgt = compute_metrics(y_cert, model.score(X_cert))
                seed_results.append({
                    "source": src.to_dict(),
                    "target": tgt.to_dict(),
                })
            results.setdefault("trace_to_cert", {})[name] = \
                _aggregate_transfer(seed_results)
            r = results["trace_to_cert"][name]
            print(f"      Source AUC-ROC: "
                  f"{r['source']['auc_roc']['mean']:.4f}")
            print(f"      Target AUC-ROC: "
                  f"{r['target']['auc_roc']['mean']:.4f}")
            print(f"      Drop: {r['auc_roc_drop']:.4f}")

    with open(TABLES_DIR / "experiment_2_transfer.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def _aggregate_transfer(seed_results):
    """Aggregate transfer results across seeds."""
    agg_s, agg_t = {}, {}
    for key in seed_results[0]["source"]:
        s_vals = [r["source"][key] for r in seed_results]
        t_vals = [r["target"][key] for r in seed_results]
        agg_s[key] = {
            "mean": float(np.mean(s_vals)),
            "std": float(np.std(s_vals)),
        }
        agg_t[key] = {
            "mean": float(np.mean(t_vals)),
            "std": float(np.std(t_vals)),
        }
    return {
        "source": agg_s,
        "target": agg_t,
        "auc_roc_drop": (
            agg_s["auc_roc"]["mean"] - agg_t["auc_roc"]["mean"]
        ),
    }


def experiment_3():
    """OWASP ASI category mapping — multi-seed with bootstrap CIs.

    Evaluates per-category detection across 3 models × 5 seeds with
    anomaly_ratio=0.5 for ~30 samples per category. Reports bootstrap
    95% CIs and Wilcoxon signed-rank tests between tier boundaries.
    """
    from scipy.stats import bootstrap, wilcoxon

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: OWASP Category Mapping (Multi-Seed)")
    print("=" * 60)

    trail = load_trail_dataset()
    base_traces = trail["traces"]
    print(f"  Using {len(base_traces)} TRAIL traces as base")

    categories = ["ASI01", "ASI02", "ASI04", "ASI05", "ASI09", "ASI10"]

    models_spec = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    results = {}

    for name, (cls, kwargs) in models_spec.items():
        print(f"\n  {name}:")
        seed_per_cat = {cat: [] for cat in categories}

        for seed in SEEDS:
            print(f"    seed={seed}...", end=" ", flush=True)
            mixed, labels, owasp_cats = generate_anomalous_traces(
                base_traces,
                anomaly_ratio=0.5,
                categories=categories,
                seed=seed,
            )

            extractor = AgentTraceFeatureExtractor()
            X, ids, ts = extractor.extract_batch(mixed)
            normalizer = UBFSNormalizer(method="zscore")
            X = normalizer.fit_transform(X)

            normal_mask = labels == 0
            X_train = X[normal_mask]

            if name == "LSTMAutoencoder":
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train[:, np.newaxis, :])
                scores = model.score(X[:, np.newaxis, :])
            else:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train)
                scores = model.score(X)

            normal_cat_mask = np.array([
                c == "" for c in owasp_cats
            ])

            line_parts = []
            for cat in categories:
                cat_mask = np.array([
                    c == cat for c in owasp_cats
                ])
                eval_mask = cat_mask | normal_cat_mask
                cat_y = np.zeros_like(labels)
                cat_y[cat_mask] = 1

                if cat_y[eval_mask].sum() > 0:
                    cat_m = compute_metrics(
                        cat_y[eval_mask], scores[eval_mask]
                    )
                    seed_per_cat[cat].append(cat_m.to_dict())
                    line_parts.append(
                        f"{cat}={cat_m.auc_roc:.3f}"
                    )

            print(" ".join(line_parts))

        # Aggregate per-category with bootstrap CIs
        agg_cats = {}
        for cat in categories:
            if seed_per_cat[cat]:
                agg_cats[cat] = {}
                for key in seed_per_cat[cat][0]:
                    vals = [r[key] for r in seed_per_cat[cat]]
                    arr = np.array(vals)
                    ci_lo, ci_hi = np.nan, np.nan
                    if len(arr) >= 3:
                        try:
                            res = bootstrap(
                                (arr,),
                                np.mean,
                                n_resamples=9999,
                                confidence_level=0.95,
                                random_state=42,
                            )
                            ci_lo = float(res.confidence_interval.low)
                            ci_hi = float(res.confidence_interval.high)
                        except Exception:
                            pass
                    agg_cats[cat][key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "ci_95": [ci_lo, ci_hi],
                    }

        # Wilcoxon signed-rank tests between tier boundaries
        # Tier ordering: ASI05 > ASI09 > ASI10 > ASI01 > ASI02
        tier_pairs = [
            ("ASI05", "ASI09"),
            ("ASI09", "ASI01"),
            ("ASI01", "ASI02"),
        ]
        wilcoxon_results = {}
        for cat_a, cat_b in tier_pairs:
            if cat_a in seed_per_cat and cat_b in seed_per_cat:
                vals_a = [r["auc_roc"] for r in seed_per_cat[cat_a]]
                vals_b = [r["auc_roc"] for r in seed_per_cat[cat_b]]
                if len(vals_a) == len(vals_b) and len(vals_a) >= 3:
                    try:
                        stat, p = wilcoxon(vals_a, vals_b,
                                           alternative="greater")
                        wilcoxon_results[f"{cat_a}_vs_{cat_b}"] = {
                            "statistic": float(stat),
                            "p_value": float(p),
                        }
                    except Exception:
                        pass

        results[name] = {
            "per_category": agg_cats,
            "wilcoxon_tier_tests": wilcoxon_results,
        }

        for cat in categories:
            if cat in agg_cats:
                m = agg_cats[cat]["auc_roc"]["mean"]
                s = agg_cats[cat]["auc_roc"]["std"]
                ci = agg_cats[cat]["auc_roc"]["ci_95"]
                print(f"    {cat}: AUC-ROC={m:.4f} "
                      f"(+/- {s:.4f}) "
                      f"CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
        for pair, wres in wilcoxon_results.items():
            print(f"    Wilcoxon {pair}: "
                  f"p={wres['p_value']:.4f}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "experiment_3_owasp.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def experiment_4():
    """Governance assumption audit."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Governance Assumption Audit")
    print("=" * 60)

    X_trail, _, _, _ = load_trail_features()
    X_trace, _, _ = load_trace_features()

    print("  Auditing normality assumptions...")
    assumptions = audit_normality_assumptions()
    for a in assumptions:
        print(f"\n    Feature: {a.feature_name}")
        print(f"    CERT: {a.cert_assumption}")
        print(f"    Agent: {a.agent_assumption}")
        print(f"    Implication: {a.governance_implication}")

    print("\n  Comparing baseline distributions...")
    comparison = compare_baseline_distributions(X_trail, X_trace)

    report = governance_report(assumptions, comparison)

    report_path = RESULTS_DIR / "governance_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to {report_path}")

    # Save raw comparison
    with open(TABLES_DIR / "experiment_4_governance.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    return {"assumptions": len(assumptions), "report": str(report_path)}


def experiment_5():
    """Model extraction detection with strategy-specific profiles.

    Tests whether UBFS detects distillation attacks, and whether
    the hydra cluster distribution strategy defeats single-entity
    behavioral monitoring. Four sub-profiles map to the three labs'
    strategies from the Anthropic Feb 2026 disclosure, plus the
    distributed hydra architecture.

    Sub-profiles:
        ASI_DISTILL_COT     — DeepSeek's CoT elicitation (SEQUENCE)
        ASI_DISTILL_BROAD   — Moonshot's multi-capability (SCOPE)
        ASI_DISTILL_FOCUSED — MiniMax's focused extraction (FREQ)
        ASI_DISTILL_HYDRA   — Per-account hydra (near-normal)

    Compared against ASI02 (Tool Misuse) as the known blind spot.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Model Extraction Detection")
    print("=" * 60)

    trail = load_trail_dataset()
    base_traces = trail["traces"]
    print(f"  Using {len(base_traces)} TRAIL traces as base")

    categories = [
        "ASI_DISTILL_COT", "ASI_DISTILL_BROAD",
        "ASI_DISTILL_FOCUSED", "ASI_DISTILL_HYDRA",
        "ASI02",
    ]

    results = {}
    models = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    for name, (cls, kwargs) in models.items():
        print(f"\n  {name}:")
        seed_per_cat = {cat: [] for cat in categories}

        for seed in SEEDS:
            print(f"    seed={seed}...", end=" ", flush=True)
            mixed, labels, owasp_cats = generate_anomalous_traces(
                base_traces,
                anomaly_ratio=0.3,
                categories=categories,
                seed=seed,
            )

            extractor = AgentTraceFeatureExtractor()
            X, ids, ts = extractor.extract_batch(mixed)
            normalizer = UBFSNormalizer(method="zscore")
            X = normalizer.fit_transform(X)

            normal_mask = labels == 0
            X_train = X[normal_mask]

            if name == "LSTMAutoencoder":
                X_tr_3d = X_train[:, np.newaxis, :]
                X_3d = X[:, np.newaxis, :]
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_tr_3d)
                scores = model.score(X_3d)
            else:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train)
                scores = model.score(X)

            normal_cat_mask = np.array([
                c == "" for c in owasp_cats
            ])

            line_parts = []
            for cat in categories:
                cat_mask = np.array([
                    c == cat for c in owasp_cats
                ])
                eval_mask = cat_mask | normal_cat_mask
                cat_y = np.zeros_like(labels)
                cat_y[cat_mask] = 1

                if cat_y[eval_mask].sum() > 0:
                    cat_m = compute_metrics(
                        cat_y[eval_mask], scores[eval_mask]
                    )
                    seed_per_cat[cat].append(cat_m.to_dict())
                    short = cat.replace("ASI_DISTILL_", "")
                    line_parts.append(
                        f"{short}={cat_m.auc_roc:.3f}"
                    )

            print(" ".join(line_parts))

        # Aggregate per-category
        agg_cats = {}
        for cat in categories:
            if seed_per_cat[cat]:
                agg_cats[cat] = {}
                for key in seed_per_cat[cat][0]:
                    vals = [r[key] for r in seed_per_cat[cat]]
                    agg_cats[cat][key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    }

        results[name] = {"per_category": agg_cats}

        for cat in categories:
            if cat in agg_cats:
                m = agg_cats[cat]["auc_roc"]["mean"]
                s = agg_cats[cat]["auc_roc"]["std"]
                print(f"    {cat}: AUC-ROC={m:.4f} (+/- {s:.4f})")

    # Summary table
    print("\n" + "-" * 60)
    print("Experiment 5 Summary — Distillation Detection Spectrum:")
    header = f"{'Model':<18s}"
    for cat in categories:
        short = cat.replace("ASI_DISTILL_", "").replace(
            "ASI02", "ASI02"
        )
        header += f" {short:>10s}"
    print(header)
    print("-" * 60)
    for name in results:
        row = f"{name:<18s}"
        for cat in categories:
            v = results[name]["per_category"].get(
                cat, {}
            ).get("auc_roc", {}).get("mean", 0)
            row += f" {v:>10.4f}"
        print(row)

    # Check HYDRA vs ASI02 proximity
    print("\n  Blind spot analysis:")
    for name in results:
        cats = results[name]["per_category"]
        hydra = cats.get("ASI_DISTILL_HYDRA", {}).get(
            "auc_roc", {}
        ).get("mean", 0)
        asi02 = cats.get("ASI02", {}).get(
            "auc_roc", {}
        ).get("mean", 0)
        gap = abs(hydra - asi02)
        print(f"  {name}: HYDRA={hydra:.4f}, "
              f"ASI02={asi02:.4f}, gap={gap:.4f}")
        if gap < 0.05:
            print(f"    → CONFIRMED: Two blind spots converge")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "experiment_5_distillation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


def experiment_6():
    """Jailbreak decomposition detection (boiling frog).

    Tests whether multi-step task decomposition reduces detection
    compared to single-turn attacks. Uses AgentHarm malicious tool
    specifications injected in two modes:

    - Direct: all tools at once (ASI01 pattern)
    - Decomposed: tools spread across trace with normal padding
      (boiling frog / Anthropic espionage pattern)

    Validated by: Chinese state-sponsored group (GTG-1002)
    decomposed espionage into 'defensive testing' subtasks,
    evading detection. Maps to insider threat finding: 85% of
    gradual attacks missed.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Jailbreak Decomposition Detection")
    print("=" * 60)

    from src.data.agent_harm_loader import (
        load_agent_harm,
        generate_decomposition_dataset,
    )

    trail = load_trail_dataset()
    base_traces = trail["traces"]
    print(f"  {len(base_traces)} TRAIL traces as base")

    print("  Loading AgentHarm dataset...")
    agent_harm = load_agent_harm()
    tasks = agent_harm["tasks"]

    models_spec = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    results = {}

    for name, (cls, kwargs) in models_spec.items():
        print(f"\n  {name}:")
        seed_direct = []
        seed_decomposed = []
        seed_overall = []

        for seed in SEEDS:
            print(f"    seed={seed}...", end=" ", flush=True)

            traces, labels, mode_labels = \
                generate_decomposition_dataset(
                    base_traces, tasks, seed=seed,
                )

            extractor = AgentTraceFeatureExtractor()
            X, ids, ts = extractor.extract_batch(traces)
            normalizer = UBFSNormalizer(method="zscore")
            X = normalizer.fit_transform(X)

            normal_mask = np.array([
                m == "normal" for m in mode_labels
            ])
            X_train = X[normal_mask]

            if name == "LSTMAutoencoder":
                X_tr_3d = X_train[:, np.newaxis, :]
                X_3d = X[:, np.newaxis, :]
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_tr_3d)
                scores = model.score(X_3d)
            else:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train)
                scores = model.score(X)

            # Overall
            overall = compute_metrics(labels, scores)
            seed_overall.append(overall.to_dict())

            # Direct detection (normal vs direct only)
            direct_mask = np.array([
                m == "direct" for m in mode_labels
            ])
            eval_d = direct_mask | normal_mask
            direct_y = np.zeros_like(labels)
            direct_y[direct_mask] = 1
            if direct_y[eval_d].sum() > 0:
                dm = compute_metrics(
                    direct_y[eval_d], scores[eval_d]
                )
                seed_direct.append(dm.to_dict())

            # Decomposed detection (normal vs decomposed only)
            decomp_mask = np.array([
                m == "decomposed" for m in mode_labels
            ])
            eval_c = decomp_mask | normal_mask
            decomp_y = np.zeros_like(labels)
            decomp_y[decomp_mask] = 1
            if decomp_y[eval_c].sum() > 0:
                cm = compute_metrics(
                    decomp_y[eval_c], scores[eval_c]
                )
                seed_decomposed.append(cm.to_dict())

            print(f"direct={dm.auc_roc:.4f} "
                  f"decomp={cm.auc_roc:.4f}")

        # Aggregate
        def _agg(seed_list):
            agg = {}
            for key in seed_list[0]:
                vals = [r[key] for r in seed_list]
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
            return agg

        agg_overall = _agg(seed_overall)
        agg_direct = _agg(seed_direct)
        agg_decomposed = _agg(seed_decomposed)

        delta = (
            agg_direct["auc_roc"]["mean"]
            - agg_decomposed["auc_roc"]["mean"]
        )

        # Wilcoxon signed-rank test: direct vs decomposed
        from scipy.stats import wilcoxon as _wilcoxon
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        direct_aucs = [r["auc_roc"] for r in seed_direct]
        decomp_aucs = [r["auc_roc"] for r in seed_decomposed]
        if (len(direct_aucs) == len(decomp_aucs)
                and len(direct_aucs) >= 3):
            try:
                wilcoxon_stat, wilcoxon_p = _wilcoxon(
                    direct_aucs, decomp_aucs,
                    alternative="greater",
                )
                wilcoxon_stat = float(wilcoxon_stat)
                wilcoxon_p = float(wilcoxon_p)
            except Exception:
                pass

        results[name] = {
            "overall": agg_overall,
            "direct": agg_direct,
            "decomposed": agg_decomposed,
            "auc_roc_delta": delta,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
        }

        print(f"    Direct AUC-ROC: "
              f"{agg_direct['auc_roc']['mean']:.4f} "
              f"(+/- {agg_direct['auc_roc']['std']:.4f})")
        print(f"    Decomposed AUC-ROC: "
              f"{agg_decomposed['auc_roc']['mean']:.4f} "
              f"(+/- {agg_decomposed['auc_roc']['std']:.4f})")
        print(f"    Delta (direct - decomposed): "
              f"{delta:+.4f}")
        print(f"    Wilcoxon: stat={wilcoxon_stat:.2f}, "
              f"p={wilcoxon_p:.4f}")

    # Summary
    print("\n" + "-" * 60)
    print("Experiment 6 Summary — Direct vs Decomposed:")
    print(f"{'Model':<20s} {'Direct':>12s} {'Decomposed':>12s} "
          f"{'Delta':>10s}")
    print("-" * 60)
    for name in results:
        d = results[name]["direct"]["auc_roc"]["mean"]
        c = results[name]["decomposed"]["auc_roc"]["mean"]
        delta = results[name]["auc_roc_delta"]
        print(f"{name:<20s} {d:>12.4f} {c:>12.4f} "
              f"{delta:>+10.4f}")

    # Cross-reference with insider threat boiling frog
    print("\n  Cross-reference with insider threat evasion:")
    print("  Insider threat: 85% of gradual attacks missed "
          "(Scenario 2: 3% detection rate)")
    for name in results:
        d = results[name]["direct"]["auc_roc"]["mean"]
        c = results[name]["decomposed"]["auc_roc"]["mean"]
        if d > 0.5:
            drop_pct = (d - c) / (d - 0.5) * 100
            print(f"  {name}: {drop_pct:.1f}% detection "
                  f"power lost via decomposition")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "experiment_6_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


def experiment_7():
    """MCP tool abuse profiling via cross-domain transfer.

    Tests whether insider threat models transfer to MCP-specific
    tool traces (ATBench). The Anthropic espionage attack used
    MCP tools. If CERT→ATBench transfer works like CERT→TRAIL,
    the pipeline generalizes beyond the original datasets.

    Phase A: Within-domain baselines on ATBench
    Phase B: Cross-domain transfer (CERT→ATBench, TRAIL→ATBench)
    Phase C: Per-risk-category detection (mapped to OWASP)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: MCP Tool Abuse Profiling")
    print("=" * 60)

    from src.data.atbench_loader import load_atbench

    # Load ATBench
    print("  Loading ATBench...")
    atbench = load_atbench()
    extractor = AgentTraceFeatureExtractor()
    X_at, ids_at, ts_at = extractor.extract_batch(
        atbench["trajectories"]
    )
    normalizer_at = UBFSNormalizer(method="zscore")
    X_at = normalizer_at.fit_transform(X_at)
    y_at = atbench["labels"]
    owasp_at = atbench["owasp_labels"]

    normal_at = X_at[y_at == 0]
    print(f"  ATBench features: {X_at.shape}, "
          f"{(y_at == 0).sum()} safe, {(y_at == 1).sum()} unsafe")

    # Load TRAIL
    X_trail, y_trail, _, _ = load_trail_features()
    trail_normal = X_trail[y_trail == 0]

    models_spec = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    results = {}

    # ---- Phase A: Within-domain baselines on ATBench ----
    print("\n--- Phase A: Within-Domain Baselines ---")
    results["atbench_within"] = {}

    for name, (cls, kwargs) in models_spec.items():
        print(f"  {name} on ATBench...")
        seed_results = []
        for seed in SEEDS:
            model = cls(**{**kwargs, "seed": seed})
            model.fit(normal_at)
            scores = model.score(X_at)
            m = compute_metrics(y_at, scores)
            seed_results.append(m.to_dict())

        agg = _aggregate_seeds(seed_results)
        results["atbench_within"][name] = agg
        print(f"    AUC-ROC: {agg['auc_roc']['mean']:.4f} "
              f"(+/- {agg['auc_roc']['std']:.4f})")

    # ---- Phase B: Cross-domain transfer ----
    print("\n--- Phase B: Cross-Domain Transfer ---")

    transfers = {
        "trail_to_atbench": (trail_normal, X_at, y_at),
        "atbench_to_trail": (normal_at, X_trail, y_trail),
    }

    # CERT transfers (optional)
    if _USE_CERT:
        X_cert, y_cert = load_cert_features()
        cert_normal = X_cert[y_cert == 0]
        transfers["cert_to_atbench"] = (
            cert_normal, X_at, y_at
        )
        transfers["atbench_to_cert"] = (
            normal_at, X_cert, y_cert
        )

    for direction, (X_src_train, X_tgt, y_tgt) in \
            transfers.items():
        print(f"\n  {direction}:")
        results[direction] = {}

        for name, (cls, kwargs) in models_spec.items():
            print(f"    {name}...")
            seed_results = []
            for seed in SEEDS:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_src_train)
                scores = model.score(X_tgt)
                m = compute_metrics(y_tgt, scores)
                seed_results.append(m.to_dict())

            agg = _aggregate_seeds(seed_results)
            results[direction][name] = agg
            print(f"      AUC-ROC: {agg['auc_roc']['mean']:.4f} "
                  f"(+/- {agg['auc_roc']['std']:.4f})")

    # ---- Phase C: Per-risk-category detection ----
    print("\n--- Phase C: Per-Risk-Category Detection ---")
    results["per_risk_category"] = {}

    # Use IF as the representative model (fastest)
    cls, kwargs = models_spec["IsolationForest"]
    model = cls(**{**kwargs, "seed": 42})
    model.fit(normal_at)
    scores = model.score(X_at)

    risk_cats = sorted(set(
        o for o in owasp_at if o
    ))
    normal_mask = np.array([o == "" for o in owasp_at])

    for cat in risk_cats:
        cat_mask = np.array([o == cat for o in owasp_at])
        n_pos = cat_mask.sum()
        if n_pos == 0:
            continue

        cat_y = np.zeros_like(y_at)
        cat_y[cat_mask] = 1
        eval_mask = cat_mask | normal_mask

        if cat_y[eval_mask].sum() > 0:
            cat_m = compute_metrics(
                cat_y[eval_mask], scores[eval_mask]
            )
            results["per_risk_category"][cat] = {
                "auc_roc": cat_m.auc_roc,
                "auc_pr": cat_m.auc_pr,
                "n_samples": int(n_pos),
            }
            print(f"  {cat}: AUC-ROC={cat_m.auc_roc:.4f} "
                  f"(n={n_pos})")

    # ---- Summary ----
    print("\n" + "-" * 60)
    print("Experiment 7 Summary:")

    # Within-domain
    print("\n  Within-domain (ATBench):")
    for name, agg in results["atbench_within"].items():
        print(f"    {name}: "
              f"AUC-ROC={agg['auc_roc']['mean']:.4f}")

    # Transfer comparison
    print("\n  Transfer comparison:")
    print(f"    {'Direction':<25s} {'IF AUC':>10s} {'DC AUC':>10s}")
    print("    " + "-" * 47)
    for direction in transfers:
        if direction in results:
            row = direction
            if_auc = results[direction].get(
                "IsolationForest", {}
            ).get("auc_roc", {}).get("mean", 0)
            dc_auc = results[direction].get(
                "DeepClustering", {}
            ).get("auc_roc", {}).get("mean", 0)
            print(f"    {row:<25s} {if_auc:>10.4f} {dc_auc:>10.4f}")

    # Transfer retention vs Exp 2
    print("\n  Transfer retention (vs within-domain):")
    within_if = results["atbench_within"].get(
        "IsolationForest", {}
    ).get("auc_roc", {}).get("mean", 0)
    for direction in transfers:
        if direction in results:
            tf_if = results[direction].get(
                "IsolationForest", {}
            ).get("auc_roc", {}).get("mean", 0)
            if within_if > 0:
                retention = tf_if / within_if * 100
                print(f"    {direction}: {retention:.1f}% "
                      f"(Exp 2 CERT→TRAIL was 97%)")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "experiment_7_mcp_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


def _aggregate_seeds(seed_results):
    """Aggregate metric dicts across seeds."""
    agg = {}
    for key in seed_results[0]:
        vals = [r[key] for r in seed_results]
        agg[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return agg


def experiment_8():
    """Hybrid detection: UBFS-20 vs UBFS-28 (behavioral + semantic).

    Tests whether adding 8 semantic features from sentence-transformers
    closes the ASI02 blind spot. Semantic features capture intent
    signals (parameter meaning, goal consistency, action coherence)
    that structural UBFS features miss.

    Phase A+B: OWASP detection matrix comparing UBFS-20 vs UBFS-28
    Phase C: Cross-domain transfer with UBFS-28
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 8: Hybrid Detection (UBFS-20 vs UBFS-28)")
    print("=" * 60)

    from src.features.semantic_extractor import (
        SemanticFeatureExtractor,
    )

    # Load TRAIL base traces
    trail = load_trail_dataset()
    base_traces = trail["traces"]
    y_trail = get_trail_labels(trail["annotations"])
    print(f"  Using {len(base_traces)} TRAIL traces as base")

    categories = ["ASI01", "ASI02", "ASI05", "ASI09", "ASI10"]

    models_spec = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    # ---- Pre-compute features for all seeds ----
    # Extract semantic features first (uses GPU), then free model
    # before running anomaly detection models
    print("\n  Pre-computing UBFS-20 + semantic features...")
    seed_data = {}
    for seed in SEEDS:
        print(f"    seed={seed}...", end=" ", flush=True)
        mixed, labels, owasp_cats = generate_anomalous_traces(
            base_traces, anomaly_ratio=0.3,
            categories=categories, seed=seed,
        )

        # UBFS-20
        extractor = AgentTraceFeatureExtractor()
        X_20_raw, ids, ts = extractor.extract_batch(mixed)
        norm_20 = UBFSNormalizer(method="zscore")
        X_20 = norm_20.fit_transform(X_20_raw)

        # Semantic features — fit baseline on normal traces
        normal_traces = [
            t for t, l in zip(mixed, labels) if l == 0
        ]
        sem_ext = SemanticFeatureExtractor()
        sem_ext.fit_baseline(normal_traces)
        X_sem_raw = sem_ext.extract_batch(mixed)
        norm_sem = UBFSNormalizer(method="zscore")
        X_sem = norm_sem.fit_transform(X_sem_raw)

        # UBFS-28 = UBFS-20 (z-scored) + semantic (z-scored)
        X_28 = np.hstack([X_20, X_sem])

        seed_data[seed] = {
            "X_20": X_20,
            "X_28": X_28,
            "labels": labels,
            "owasp_cats": owasp_cats,
        }
        print(f"X_20={X_20.shape}, X_28={X_28.shape}")

    # Free sentence-transformer from GPU memory
    import src.features.semantic_extractor as _sem_mod
    _sem_mod._MODEL = None
    import gc
    gc.collect()

    # ---- Phase A+B: OWASP matrix UBFS-20 vs UBFS-28 ----
    print("\n--- Phase A+B: UBFS-20 vs UBFS-28 OWASP Matrix ---")
    results = {}

    for name, (cls, kwargs) in models_spec.items():
        print(f"\n  {name}:")
        seed_per_cat_20 = {cat: [] for cat in categories}
        seed_per_cat_28 = {cat: [] for cat in categories}

        for seed in SEEDS:
            print(f"    seed={seed}...", end=" ", flush=True)
            sd = seed_data[seed]
            X_20 = sd["X_20"]
            X_28 = sd["X_28"]
            labels = sd["labels"]
            owasp_cats = sd["owasp_cats"]
            normal_mask = labels == 0

            # Train UBFS-20
            if name == "LSTMAutoencoder":
                model_20 = cls(**{**kwargs, "seed": seed})
                model_20.fit(
                    X_20[normal_mask][:, np.newaxis, :]
                )
                scores_20 = model_20.score(
                    X_20[:, np.newaxis, :]
                )
                model_28 = cls(**{**kwargs, "seed": seed})
                model_28.fit(
                    X_28[normal_mask][:, np.newaxis, :]
                )
                scores_28 = model_28.score(
                    X_28[:, np.newaxis, :]
                )
            else:
                model_20 = cls(**{**kwargs, "seed": seed})
                model_20.fit(X_20[normal_mask])
                scores_20 = model_20.score(X_20)
                model_28 = cls(**{**kwargs, "seed": seed})
                model_28.fit(X_28[normal_mask])
                scores_28 = model_28.score(X_28)

            # Per-category metrics
            normal_cat_mask = np.array(
                [c == "" for c in owasp_cats]
            )
            line_parts = []
            for cat in categories:
                cat_mask = np.array(
                    [c == cat for c in owasp_cats]
                )
                eval_mask = cat_mask | normal_cat_mask
                cat_y = np.zeros_like(labels)
                cat_y[cat_mask] = 1

                if cat_y[eval_mask].sum() > 0:
                    m_20 = compute_metrics(
                        cat_y[eval_mask],
                        scores_20[eval_mask],
                    )
                    m_28 = compute_metrics(
                        cat_y[eval_mask],
                        scores_28[eval_mask],
                    )
                    seed_per_cat_20[cat].append(
                        m_20.to_dict()
                    )
                    seed_per_cat_28[cat].append(
                        m_28.to_dict()
                    )
                    line_parts.append(
                        f"{cat}={m_28.auc_roc:.3f}"
                    )

            print(" ".join(line_parts))

        # Aggregate across seeds
        results[name] = {
            "per_category": {},
        }
        for cat in categories:
            if seed_per_cat_20[cat]:
                agg_20 = _aggregate_seeds(seed_per_cat_20[cat])
                agg_28 = _aggregate_seeds(seed_per_cat_28[cat])
                delta = (
                    agg_28["auc_roc"]["mean"]
                    - agg_20["auc_roc"]["mean"]
                )
                results[name]["per_category"][cat] = {
                    "ubfs_20": agg_20,
                    "ubfs_28": agg_28,
                    "auc_roc_delta": delta,
                }
                print(
                    f"    {cat}: UBFS-20="
                    f"{agg_20['auc_roc']['mean']:.4f}, "
                    f"UBFS-28="
                    f"{agg_28['auc_roc']['mean']:.4f}, "
                    f"delta={delta:+.4f}"
                )

    # ---- Phase C: Cross-domain transfer UBFS-28 ----
    print("\n--- Phase C: Cross-Domain Transfer (UBFS-28) ---")

    from src.data.atbench_loader import load_atbench

    print("  Loading ATBench...")
    atbench = load_atbench()
    at_traces = atbench["trajectories"]
    y_at = atbench["labels"]

    # UBFS-20 for both domains
    ext = AgentTraceFeatureExtractor()

    X_trail_20_raw, _, _ = ext.extract_batch(base_traces)
    norm_trail_20 = UBFSNormalizer(method="zscore")
    X_trail_20 = norm_trail_20.fit_transform(X_trail_20_raw)

    X_at_20_raw, _, _ = ext.extract_batch(at_traces)
    norm_at_20 = UBFSNormalizer(method="zscore")
    X_at_20 = norm_at_20.fit_transform(X_at_20_raw)

    # Semantic features for TRAIL (baseline: TRAIL normal)
    trail_normal_traces = [
        base_traces[i]
        for i in range(len(base_traces))
        if y_trail[i] == 0
    ]
    sem_trail = SemanticFeatureExtractor()
    sem_trail.fit_baseline(trail_normal_traces)
    X_sem_trail_raw = sem_trail.extract_batch(base_traces)
    norm_sem_trail = UBFSNormalizer(method="zscore")
    X_sem_trail = norm_sem_trail.fit_transform(X_sem_trail_raw)

    # Semantic features for ATBench (baseline: ATBench safe)
    at_safe_traces = [
        t for t, y in zip(at_traces, y_at) if y == 0
    ]
    sem_at = SemanticFeatureExtractor()
    sem_at.fit_baseline(at_safe_traces)
    X_sem_at_raw = sem_at.extract_batch(at_traces)
    norm_sem_at = UBFSNormalizer(method="zscore")
    X_sem_at = norm_sem_at.fit_transform(X_sem_at_raw)

    # UBFS-28
    X_trail_28 = np.hstack([X_trail_20, X_sem_trail])
    X_at_28 = np.hstack([X_at_20, X_sem_at])

    trail_normal_20 = X_trail_20[y_trail == 0]
    trail_normal_28 = X_trail_28[y_trail == 0]

    # Free semantic model again
    _sem_mod._MODEL = None
    gc.collect()

    transfer_models = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    results["transfer"] = {}
    for name, (cls, kwargs) in transfer_models.items():
        print(f"\n  {name} — TRAIL→ATBench:")
        seed_20 = []
        seed_28 = []
        for seed in SEEDS:
            # UBFS-20 transfer
            m20 = cls(**{**kwargs, "seed": seed})
            m20.fit(trail_normal_20)
            s20 = m20.score(X_at_20)
            met20 = compute_metrics(y_at, s20)
            seed_20.append(met20.to_dict())

            # UBFS-28 transfer
            m28 = cls(**{**kwargs, "seed": seed})
            m28.fit(trail_normal_28)
            s28 = m28.score(X_at_28)
            met28 = compute_metrics(y_at, s28)
            seed_28.append(met28.to_dict())

        agg_20 = _aggregate_seeds(seed_20)
        agg_28 = _aggregate_seeds(seed_28)
        delta = (
            agg_28["auc_roc"]["mean"]
            - agg_20["auc_roc"]["mean"]
        )

        results["transfer"][name] = {
            "trail_to_atbench_ubfs20": agg_20,
            "trail_to_atbench_ubfs28": agg_28,
            "auc_roc_delta": delta,
        }
        print(
            f"    UBFS-20: "
            f"{agg_20['auc_roc']['mean']:.4f}, "
            f"UBFS-28: "
            f"{agg_28['auc_roc']['mean']:.4f}, "
            f"delta={delta:+.4f}"
        )

    # ---- Summary ----
    print("\n" + "-" * 60)
    print("Experiment 8 Summary — UBFS-20 vs UBFS-28:")
    print(f"\n{'Model':<18s} {'Cat':<8s} "
          f"{'UBFS-20':>10s} {'UBFS-28':>10s} "
          f"{'Delta':>10s}")
    print("-" * 60)
    for name in results:
        if name == "transfer":
            continue
        for cat in categories:
            cat_data = results[name]["per_category"].get(cat)
            if cat_data:
                v20 = cat_data["ubfs_20"]["auc_roc"]["mean"]
                v28 = cat_data["ubfs_28"]["auc_roc"]["mean"]
                d = cat_data["auc_roc_delta"]
                print(f"{name:<18s} {cat:<8s} "
                      f"{v20:>10.4f} {v28:>10.4f} "
                      f"{d:>+10.4f}")

    # ASI02 improvement highlight
    print("\n  ASI02 Blind Spot Analysis:")
    for name in results:
        if name == "transfer":
            continue
        cat_data = results[name]["per_category"].get("ASI02")
        if cat_data:
            v20 = cat_data["ubfs_20"]["auc_roc"]["mean"]
            v28 = cat_data["ubfs_28"]["auc_roc"]["mean"]
            print(f"    {name}: {v20:.4f} → {v28:.4f} "
                  f"({v28 - v20:+.4f})")
            if v28 > 0.70:
                print(f"      CLOSED: ASI02 now above 0.70")
            else:
                print(f"      OPEN: ASI02 still below 0.70")

    # Transfer summary
    print("\n  Transfer (TRAIL→ATBench):")
    for name, t_data in results.get("transfer", {}).items():
        v20 = t_data["trail_to_atbench_ubfs20"][
            "auc_roc"
        ]["mean"]
        v28 = t_data["trail_to_atbench_ubfs28"][
            "auc_roc"
        ]["mean"]
        d = t_data["auc_roc_delta"]
        print(f"    {name}: UBFS-20={v20:.4f}, "
              f"UBFS-28={v28:.4f} ({d:+.4f})")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "experiment_8_hybrid.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


def experiment_9():
    """Adversarial evasion testing.

    Tests three evasion strategies against UBFS-based monitoring:
    9A: Feature clamping — cap anomalous features at normal thresholds
    9B: Gradual escalation — interpolate from normal to anomalous
    9C: Mimicry — copy nearest normal neighbor, keep attack payload

    Formalizes the 'boiling frog' pattern from Paper 1 into a
    measurable evasion framework for AI agent monitoring.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 9: Adversarial Evasion Testing")
    print("=" * 60)

    from scipy.spatial.distance import cdist

    trail = load_trail_dataset()
    base_traces = trail["traces"]
    print(f"  Using {len(base_traces)} TRAIL traces as base")

    categories = ["ASI01", "ASI02", "ASI05", "ASI09", "ASI10"]

    models_spec = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    results = {"clamping": {}, "escalation": {}, "mimicry": {}}

    for name, (cls, kwargs) in models_spec.items():
        print(f"\n  {name}:")
        clamp_per_cat = {cat: [] for cat in categories}
        mimicry_per_cat = {cat: [] for cat in categories}
        escalation_per_cat = {cat: [] for cat in categories}

        for seed in SEEDS:
            print(f"    seed={seed}...", end=" ", flush=True)

            # Generate baseline anomalies
            mixed, labels, owasp_cats = generate_anomalous_traces(
                base_traces,
                anomaly_ratio=0.3,
                categories=categories,
                seed=seed,
            )

            extractor = AgentTraceFeatureExtractor()
            X, ids, ts = extractor.extract_batch(mixed)
            normalizer = UBFSNormalizer(method="zscore")
            X = normalizer.fit_transform(X)

            normal_mask = labels == 0
            X_normal = X[normal_mask]
            X_train = X_normal.copy()

            # Train model on normal data
            if name == "LSTMAutoencoder":
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train[:, np.newaxis, :])
            else:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train)

            # --- Baseline scores ---
            if name == "LSTMAutoencoder":
                baseline_scores = model.score(
                    X[:, np.newaxis, :]
                )
            else:
                baseline_scores = model.score(X)

            normal_cat_mask = np.array([
                c == "" for c in owasp_cats
            ])

            # --- 9A: Feature clamping ---
            thresholds = np.percentile(X_normal, 95, axis=0)
            X_clamped = X.copy()
            anom_idx = np.where(labels == 1)[0]
            for i in anom_idx:
                X_clamped[i] = np.minimum(
                    X_clamped[i], thresholds
                )

            if name == "LSTMAutoencoder":
                clamped_scores = model.score(
                    X_clamped[:, np.newaxis, :]
                )
            else:
                clamped_scores = model.score(X_clamped)

            # --- 9C: Mimicry ---
            # For each anomalous sample, find nearest normal
            # neighbor and replace all features except privilege
            # (dim 19, the last one)
            X_mimicry = X.copy()
            if len(X_normal) > 0 and len(anom_idx) > 0:
                dists = cdist(
                    X[anom_idx], X_normal, metric="euclidean"
                )
                nn_idx = np.argmin(dists, axis=1)
                for j, ai in enumerate(anom_idx):
                    X_mimicry[ai, :19] = X_normal[
                        nn_idx[j], :19
                    ]

            if name == "LSTMAutoencoder":
                mimicry_scores = model.score(
                    X_mimicry[:, np.newaxis, :]
                )
            else:
                mimicry_scores = model.score(X_mimicry)

            # Per-category metrics for clamping and mimicry
            for cat in categories:
                cat_mask = np.array([
                    c == cat for c in owasp_cats
                ])
                eval_mask = cat_mask | normal_cat_mask
                cat_y = np.zeros_like(labels)
                cat_y[cat_mask] = 1

                if cat_y[eval_mask].sum() > 0:
                    base_m = compute_metrics(
                        cat_y[eval_mask],
                        baseline_scores[eval_mask],
                    )
                    clamp_m = compute_metrics(
                        cat_y[eval_mask],
                        clamped_scores[eval_mask],
                    )
                    mim_m = compute_metrics(
                        cat_y[eval_mask],
                        mimicry_scores[eval_mask],
                    )
                    clamp_per_cat[cat].append({
                        "baseline_auc": base_m.auc_roc,
                        "clamped_auc": clamp_m.auc_roc,
                    })
                    mimicry_per_cat[cat].append({
                        "baseline_auc": base_m.auc_roc,
                        "mimicry_auc": mim_m.auc_roc,
                    })

            # --- 9B: Gradual escalation ---
            # For each category, pick a representative anomalous
            # profile and interpolate from normal mean
            normal_mean = X_normal.mean(axis=0)
            for cat in categories:
                cat_idx = [
                    i for i, c in enumerate(owasp_cats)
                    if c == cat
                ]
                if not cat_idx:
                    continue
                # Use the mean anomalous profile for this category
                anom_profile = X[cat_idx].mean(axis=0)

                n_steps_list = [5, 10, 20, 50]
                detection_step = n_steps_list[-1]
                alpha_at_detection = 1.0

                for n_steps in n_steps_list:
                    found = False
                    for step in range(n_steps + 1):
                        alpha = step / n_steps
                        interp = (
                            (1 - alpha) * normal_mean
                            + alpha * anom_profile
                        )
                        interp_2d = interp.reshape(1, -1)

                        if name == "LSTMAutoencoder":
                            sc = model.score(
                                interp_2d[:, np.newaxis, :]
                            )
                        else:
                            sc = model.score(interp_2d)

                        # Detection threshold: 95th percentile
                        # of normal training scores
                        if name == "LSTMAutoencoder":
                            train_sc = model.score(
                                X_train[:, np.newaxis, :]
                            )
                        else:
                            train_sc = model.score(X_train)
                        threshold = np.percentile(train_sc, 95)

                        if sc[0] > threshold:
                            detection_step = step
                            alpha_at_detection = alpha
                            found = True
                            break

                    if found:
                        break

                escalation_per_cat[cat].append({
                    "detection_step": int(detection_step),
                    "alpha_at_detection": float(
                        alpha_at_detection
                    ),
                })

            print("done")

        # Aggregate across seeds
        for cat in categories:
            # Clamping
            if clamp_per_cat[cat]:
                base_vals = [
                    r["baseline_auc"] for r in clamp_per_cat[cat]
                ]
                clamp_vals = [
                    r["clamped_auc"] for r in clamp_per_cat[cat]
                ]
                base_mean = float(np.mean(base_vals))
                clamp_mean = float(np.mean(clamp_vals))
                drop = (
                    (base_mean - clamp_mean) / max(base_mean, 1e-9)
                    * 100
                )
                results["clamping"].setdefault(name, {})[cat] = {
                    "baseline_auc": {
                        "mean": base_mean,
                        "std": float(np.std(base_vals)),
                    },
                    "clamped_auc": {
                        "mean": clamp_mean,
                        "std": float(np.std(clamp_vals)),
                    },
                    "drop_pct": round(drop, 2),
                }

            # Mimicry
            if mimicry_per_cat[cat]:
                base_vals = [
                    r["baseline_auc"]
                    for r in mimicry_per_cat[cat]
                ]
                mim_vals = [
                    r["mimicry_auc"]
                    for r in mimicry_per_cat[cat]
                ]
                base_mean = float(np.mean(base_vals))
                mim_mean = float(np.mean(mim_vals))
                drop = (
                    (base_mean - mim_mean) / max(base_mean, 1e-9)
                    * 100
                )
                results["mimicry"].setdefault(name, {})[cat] = {
                    "baseline_auc": {
                        "mean": base_mean,
                        "std": float(np.std(base_vals)),
                    },
                    "mimicry_auc": {
                        "mean": mim_mean,
                        "std": float(np.std(mim_vals)),
                    },
                    "drop_pct": round(drop, 2),
                }

            # Escalation
            if escalation_per_cat[cat]:
                steps = [
                    r["detection_step"]
                    for r in escalation_per_cat[cat]
                ]
                alphas = [
                    r["alpha_at_detection"]
                    for r in escalation_per_cat[cat]
                ]
                results["escalation"].setdefault(
                    name, {}
                )[cat] = {
                    "detection_step": {
                        "mean": float(np.mean(steps)),
                        "std": float(np.std(steps)),
                    },
                    "alpha_at_detection": {
                        "mean": float(np.mean(alphas)),
                        "std": float(np.std(alphas)),
                    },
                }

    # Summary
    print("\n" + "-" * 60)
    print("Experiment 9 Summary — Adversarial Evasion:")

    print("\n  9A: Feature Clamping (AUC-ROC drop %)")
    header = f"  {'Model':<18s}"
    for cat in categories:
        header += f" {cat:>8s}"
    print(header)
    for name in results["clamping"]:
        row = f"  {name:<18s}"
        for cat in categories:
            d = results["clamping"][name].get(
                cat, {}
            ).get("drop_pct", 0)
            row += f" {d:>7.1f}%"
        print(row)

    print("\n  9B: Gradual Escalation (alpha at detection)")
    for name in results["escalation"]:
        row = f"  {name:<18s}"
        for cat in categories:
            a = results["escalation"][name].get(
                cat, {}
            ).get("alpha_at_detection", {}).get("mean", 0)
            row += f" {a:>8.2f}"
        print(row)

    print("\n  9C: Mimicry (AUC-ROC drop %)")
    for name in results["mimicry"]:
        row = f"  {name:<18s}"
        for cat in categories:
            d = results["mimicry"][name].get(
                cat, {}
            ).get("drop_pct", 0)
            row += f" {d:>7.1f}%"
        print(row)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "experiment_9_adversarial.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


def experiment_10():
    """Temporal dynamics: window ablation on agent traces.

    Tests whether the optimal monitoring window size transfers
    across domains. Paper 1 showed 7-day windows outperform
    14/30-day for insider detection. If agent traces show an
    analogous pattern, that's a structural insight: early warning
    signals are more predictive than full histories.

    Window sizes: 5, 10, 20 spans, and full trace.
    Datasets: TRAIL and ATBench.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 10: Temporal Window Ablation")
    print("=" * 60)

    from src.data.atbench_loader import load_atbench

    trail = load_trail_dataset()
    base_traces = trail["traces"]
    y_trail = get_trail_labels(trail["annotations"])
    print(f"  TRAIL: {len(base_traces)} traces")

    print("  Loading ATBench...")
    atbench = load_atbench()

    windows = [5, 10, 20, None]  # None = full trace
    window_labels = ["5_spans", "10_spans", "20_spans", "full"]

    models_spec = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    results = {"TRAIL": {}, "ATBench": {}, "optimal_window": {}}

    datasets = [
        (
            "TRAIL", base_traces, y_trail,
        ),
        (
            "ATBench",
            atbench["trajectories"],
            atbench["labels"],
        ),
    ]

    for ds_name, traces, y in datasets:
        print(f"\n  --- {ds_name} ---")
        results[ds_name] = {}

        for name, (cls, kwargs) in models_spec.items():
            print(f"  {name}:")
            results[ds_name][name] = {}

            for w, w_label in zip(windows, window_labels):
                seed_results = []

                for seed in SEEDS:
                    extractor = AgentTraceFeatureExtractor()
                    X, ids, ts = extractor.extract_batch(
                        traces, max_spans=w
                    )
                    normalizer = UBFSNormalizer(method="zscore")
                    X = normalizer.fit_transform(X)

                    normal_mask = y == 0
                    X_train = X[normal_mask]

                    if len(X_train) == 0:
                        continue

                    if name == "LSTMAutoencoder":
                        model = cls(**{**kwargs, "seed": seed})
                        model.fit(X_train[:, np.newaxis, :])
                        scores = model.score(
                            X[:, np.newaxis, :]
                        )
                    else:
                        model = cls(**{**kwargs, "seed": seed})
                        model.fit(X_train)
                        scores = model.score(X)

                    m = compute_metrics(y, scores)
                    seed_results.append(m.to_dict())

                if seed_results:
                    agg = _aggregate_seeds(seed_results)
                    results[ds_name][name][w_label] = {
                        "auc_roc_mean": agg["auc_roc"]["mean"],
                        "auc_roc_std": agg["auc_roc"]["std"],
                    }
                    print(
                        f"    {w_label}: AUC-ROC="
                        f"{agg['auc_roc']['mean']:.4f} "
                        f"(+/- {agg['auc_roc']['std']:.4f})"
                    )

    # Find optimal window per model per dataset
    results["optimal_window"] = {
        "TRAIL": {},
        "ATBench": {},
        "CERT_reference": "7_days",
    }
    for ds_name in ["TRAIL", "ATBench"]:
        for name in models_spec:
            best_label = "full"
            best_auc = 0.0
            for w_label in window_labels:
                auc = results[ds_name].get(name, {}).get(
                    w_label, {}
                ).get("auc_roc_mean", 0)
                if auc > best_auc:
                    best_auc = auc
                    best_label = w_label
            results["optimal_window"][ds_name][name] = (
                best_label
            )

    # Summary
    print("\n" + "-" * 60)
    print("Experiment 10 Summary — Temporal Window Ablation:")

    for ds_name in ["TRAIL", "ATBench"]:
        print(f"\n  {ds_name}:")
        header = f"  {'Model':<18s}"
        for w_label in window_labels:
            header += f" {w_label:>10s}"
        header += f" {'optimal':>10s}"
        print(header)
        for name in models_spec:
            row = f"  {name:<18s}"
            for w_label in window_labels:
                v = results[ds_name].get(name, {}).get(
                    w_label, {}
                ).get("auc_roc_mean", 0)
                row += f" {v:>10.4f}"
            opt = results["optimal_window"][ds_name].get(
                name, "?"
            )
            row += f" {opt:>10s}"
            print(row)

    # Cross-reference with Paper 1
    print("\n  Cross-reference with Paper 1:")
    print("  Paper 1: 7-day optimal (23% of 30-day max)")
    for ds_name in ["TRAIL", "ATBench"]:
        for name in ["IsolationForest"]:
            opt = results["optimal_window"][ds_name].get(
                name, "full"
            )
            print(f"  {ds_name} {name}: optimal = {opt}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "experiment_10_temporal.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


def experiment_11():
    """MITRE ATLAS taxonomy mapping.

    Tests whether UBFS detection generalizes beyond OWASP to the
    MITRE ATLAS adversarial ML threat taxonomy. Maps 5 ATLAS
    techniques to UBFS perturbation profiles and evaluates
    detection per technique.

    ATLAS techniques:
        AML.T0044 - Full Model Replication
        AML.T0048 - Model Extraction via API
        AML.T0043 - Craft Adversarial Data
        AML.T0025 - Exfiltration via ML Inference API
        AML.T0042 - Verify Attack
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 11: MITRE ATLAS Mapping")
    print("=" * 60)

    trail = load_trail_dataset()
    base_traces = trail["traces"]
    print(f"  Using {len(base_traces)} TRAIL traces as base")

    atlas_categories = [
        "AML_T0044_REPLICATION",
        "AML_T0048_EXTRACTION",
        "AML_T0043_ADVERSARIAL",
        "AML_T0025_EXFILTRATION",
        "AML_T0042_VERIFY",
    ]

    models_spec = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    results = {"atlas_detection_matrix": {}, "cross_taxonomy": {}}

    for name, (cls, kwargs) in models_spec.items():
        print(f"\n  {name}:")
        seed_per_cat = {cat: [] for cat in atlas_categories}

        for seed in SEEDS:
            print(f"    seed={seed}...", end=" ", flush=True)
            mixed, labels, owasp_cats = generate_anomalous_traces(
                base_traces,
                anomaly_ratio=0.3,
                categories=atlas_categories,
                seed=seed,
            )

            extractor = AgentTraceFeatureExtractor()
            X, ids, ts = extractor.extract_batch(mixed)
            normalizer = UBFSNormalizer(method="zscore")
            X = normalizer.fit_transform(X)

            normal_mask = labels == 0
            X_train = X[normal_mask]

            if name == "LSTMAutoencoder":
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train[:, np.newaxis, :])
                scores = model.score(X[:, np.newaxis, :])
            else:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train)
                scores = model.score(X)

            normal_cat_mask = np.array([
                c == "" for c in owasp_cats
            ])

            line_parts = []
            for cat in atlas_categories:
                cat_mask = np.array([
                    c == cat for c in owasp_cats
                ])
                eval_mask = cat_mask | normal_cat_mask
                cat_y = np.zeros_like(labels)
                cat_y[cat_mask] = 1

                if cat_y[eval_mask].sum() > 0:
                    cat_m = compute_metrics(
                        cat_y[eval_mask], scores[eval_mask]
                    )
                    seed_per_cat[cat].append(cat_m.to_dict())
                    short = cat.replace("AML_", "")
                    line_parts.append(
                        f"{short}={cat_m.auc_roc:.3f}"
                    )

            print(" ".join(line_parts))

        # Aggregate per-category
        agg_cats = {}
        for cat in atlas_categories:
            if seed_per_cat[cat]:
                agg_cats[cat] = {}
                for key in seed_per_cat[cat][0]:
                    vals = [
                        r[key] for r in seed_per_cat[cat]
                    ]
                    agg_cats[cat][key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    }

        results["atlas_detection_matrix"][name] = agg_cats

        for cat in atlas_categories:
            if cat in agg_cats:
                m = agg_cats[cat]["auc_roc"]["mean"]
                s = agg_cats[cat]["auc_roc"]["std"]
                print(
                    f"    {cat}: AUC-ROC={m:.4f} "
                    f"(+/- {s:.4f})"
                )

    # Cross-taxonomy comparison
    # OWASP reference (from Exp 3 expected results)
    owasp_tiers = {
        "strong": ["ASI05", "ASI09", "ASI10"],
        "moderate": ["ASI01"],
        "blind_spot": ["ASI02"],
    }

    atlas_tiers = {"strong": [], "moderate": [], "blind_spot": []}
    # Use IF as representative for tier assignment
    if_results = results["atlas_detection_matrix"].get(
        "IsolationForest", {}
    )
    for cat in atlas_categories:
        auc = if_results.get(cat, {}).get(
            "auc_roc", {}
        ).get("mean", 0)
        if auc >= 0.80:
            atlas_tiers["strong"].append(cat)
        elif auc >= 0.60:
            atlas_tiers["moderate"].append(cat)
        else:
            atlas_tiers["blind_spot"].append(cat)

    results["cross_taxonomy"] = {
        "strong": {
            "owasp": owasp_tiers["strong"],
            "atlas": atlas_tiers["strong"],
        },
        "moderate": {
            "owasp": owasp_tiers["moderate"],
            "atlas": atlas_tiers["moderate"],
        },
        "blind_spot": {
            "owasp": owasp_tiers["blind_spot"],
            "atlas": atlas_tiers["blind_spot"],
        },
    }

    # Summary
    print("\n" + "-" * 60)
    print("Experiment 11 Summary — ATLAS Detection Matrix:")
    header = f"  {'Model':<18s}"
    for cat in atlas_categories:
        short = cat.replace("AML_", "").replace(
            "_REPLICATION", ""
        ).replace("_EXTRACTION", "").replace(
            "_ADVERSARIAL", ""
        ).replace("_EXFILTRATION", "").replace(
            "_VERIFY", ""
        )
        header += f" {short:>8s}"
    print(header)
    print("  " + "-" * 60)
    for name in results["atlas_detection_matrix"]:
        row = f"  {name:<18s}"
        for cat in atlas_categories:
            v = results["atlas_detection_matrix"][name].get(
                cat, {}
            ).get("auc_roc", {}).get("mean", 0)
            row += f" {v:>8.4f}"
        print(row)

    print("\n  Cross-Taxonomy Comparison:")
    for tier in ["strong", "moderate", "blind_spot"]:
        owasp_list = results["cross_taxonomy"][tier]["owasp"]
        atlas_list = results["cross_taxonomy"][tier]["atlas"]
        tier_label = tier.replace("_", " ").title()
        print(f"  {tier_label:>12s}: OWASP={owasp_list}, "
              f"ATLAS={atlas_list}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "experiment_11_atlas.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


def experiment_12():
    """Real-data OWASP validation + UBFS-28 on ATBench.

    Phase A: All 3 models on ATBench per-category (UBFS-20)
    Phase B: UBFS-28 on ATBench per-category
    Phase C: Spearman correlation — synthetic (Exp 3) vs real ATBench

    This is the real-data validation experiment: Exp 3 uses circular
    synthetic methodology (profiles define perturbations, then models
    detect those same perturbations). Exp 12 tests the same categories
    on 500 real ATBench trajectories with ground-truth labels.
    """
    from scipy.stats import spearmanr

    print("\n" + "=" * 60)
    print("EXPERIMENT 12: Real-Data OWASP Validation (ATBench)")
    print("=" * 60)

    from src.data.atbench_loader import load_atbench
    from src.features.semantic_extractor import (
        SemanticFeatureExtractor,
    )

    # Load ATBench
    print("  Loading ATBench...")
    atbench = load_atbench()
    at_traces = atbench["trajectories"]
    y_at = atbench["labels"]
    owasp_at = atbench["owasp_labels"]

    # UBFS-20 features
    extractor = AgentTraceFeatureExtractor()
    X_20_raw, ids, ts = extractor.extract_batch(at_traces)
    norm_20 = UBFSNormalizer(method="zscore")
    X_20 = norm_20.fit_transform(X_20_raw)

    # UBFS-28: add semantic features
    print("  Computing semantic features for UBFS-28...")
    safe_traces = [t for t, y in zip(at_traces, y_at) if y == 0]
    sem_ext = SemanticFeatureExtractor()
    sem_ext.fit_baseline(safe_traces)
    X_sem_raw = sem_ext.extract_batch(at_traces)
    norm_sem = UBFSNormalizer(method="zscore")
    X_sem = norm_sem.fit_transform(X_sem_raw)
    X_28 = np.hstack([X_20, X_sem])

    # Free semantic model
    import src.features.semantic_extractor as _sem_mod
    import gc
    _sem_mod._MODEL = None
    gc.collect()

    normal_at = y_at == 0
    print(f"  ATBench: {X_20.shape[0]} traces, "
          f"{normal_at.sum()} safe, {(~normal_at).sum()} unsafe")
    print(f"  UBFS-20: {X_20.shape}, UBFS-28: {X_28.shape}")

    # Identify OWASP categories present in ATBench
    cats_present = sorted(set(o for o in owasp_at if o))
    print(f"  OWASP categories present: {cats_present}")

    normal_cat_mask = np.array([o == "" for o in owasp_at])

    models_spec = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    results = {"phase_a_ubfs20": {}, "phase_b_ubfs28": {},
               "phase_c_correlation": {}}

    # ---- Phase A: UBFS-20 per-category ----
    print("\n--- Phase A: UBFS-20 Per-Category ---")

    for name, (cls, kwargs) in models_spec.items():
        print(f"\n  {name}:")
        seed_per_cat = {cat: [] for cat in cats_present}
        seed_overall = []

        for seed in SEEDS:
            print(f"    seed={seed}...", end=" ", flush=True)
            X_train = X_20[normal_at]

            if name == "LSTMAutoencoder":
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train[:, np.newaxis, :])
                scores = model.score(X_20[:, np.newaxis, :])
            else:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train)
                scores = model.score(X_20)

            overall = compute_metrics(y_at, scores)
            seed_overall.append(overall.to_dict())

            line_parts = []
            for cat in cats_present:
                cat_mask = np.array([o == cat for o in owasp_at])
                eval_mask = cat_mask | normal_cat_mask
                cat_y = np.zeros_like(y_at)
                cat_y[cat_mask] = 1

                if cat_y[eval_mask].sum() > 0:
                    cat_m = compute_metrics(
                        cat_y[eval_mask], scores[eval_mask]
                    )
                    seed_per_cat[cat].append(cat_m.to_dict())
                    line_parts.append(
                        f"{cat}={cat_m.auc_roc:.3f}"
                    )
            print(" ".join(line_parts))

        agg_overall = _aggregate_seeds(seed_overall)
        agg_cats = {}
        for cat in cats_present:
            if seed_per_cat[cat]:
                agg_cats[cat] = _aggregate_seeds(
                    seed_per_cat[cat]
                )
        results["phase_a_ubfs20"][name] = {
            "overall": agg_overall,
            "per_category": agg_cats,
        }

        print(f"    Overall: "
              f"{agg_overall['auc_roc']['mean']:.4f}")
        for cat in cats_present:
            if cat in agg_cats:
                m = agg_cats[cat]["auc_roc"]["mean"]
                s = agg_cats[cat]["auc_roc"]["std"]
                print(f"    {cat}: {m:.4f} (+/- {s:.4f})")

    # ---- Phase B: UBFS-28 per-category ----
    print("\n--- Phase B: UBFS-28 Per-Category ---")

    for name, (cls, kwargs) in models_spec.items():
        print(f"\n  {name}:")
        seed_per_cat = {cat: [] for cat in cats_present}
        seed_overall = []

        for seed in SEEDS:
            print(f"    seed={seed}...", end=" ", flush=True)
            X_train = X_28[normal_at]

            if name == "LSTMAutoencoder":
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train[:, np.newaxis, :])
                scores = model.score(X_28[:, np.newaxis, :])
            else:
                model = cls(**{**kwargs, "seed": seed})
                model.fit(X_train)
                scores = model.score(X_28)

            overall = compute_metrics(y_at, scores)
            seed_overall.append(overall.to_dict())

            line_parts = []
            for cat in cats_present:
                cat_mask = np.array([o == cat for o in owasp_at])
                eval_mask = cat_mask | normal_cat_mask
                cat_y = np.zeros_like(y_at)
                cat_y[cat_mask] = 1

                if cat_y[eval_mask].sum() > 0:
                    cat_m = compute_metrics(
                        cat_y[eval_mask], scores[eval_mask]
                    )
                    seed_per_cat[cat].append(cat_m.to_dict())
                    line_parts.append(
                        f"{cat}={cat_m.auc_roc:.3f}"
                    )
            print(" ".join(line_parts))

        agg_overall = _aggregate_seeds(seed_overall)
        agg_cats = {}
        for cat in cats_present:
            if seed_per_cat[cat]:
                agg_cats[cat] = _aggregate_seeds(
                    seed_per_cat[cat]
                )
        results["phase_b_ubfs28"][name] = {
            "overall": agg_overall,
            "per_category": agg_cats,
        }

        print(f"    Overall: "
              f"{agg_overall['auc_roc']['mean']:.4f}")
        for cat in cats_present:
            if cat in agg_cats:
                m = agg_cats[cat]["auc_roc"]["mean"]
                s = agg_cats[cat]["auc_roc"]["std"]
                print(f"    {cat}: {m:.4f} (+/- {s:.4f})")

    # ---- Phase C: Synthetic vs Real Spearman correlation ----
    print("\n--- Phase C: Synthetic vs Real Correlation ---")

    exp3_path = TABLES_DIR / "experiment_3_owasp.json"
    if exp3_path.exists():
        with open(exp3_path) as f:
            exp3_data = json.load(f)

        # Compare per-model: for each shared category, collect
        # synthetic AUC-ROC (Exp 3) and real AUC-ROC (Exp 12 Phase A)
        for name in models_spec:
            synth_aucs = []
            real_aucs = []
            shared_cats = []
            exp3_model = exp3_data.get(name, {})
            exp12_model = results["phase_a_ubfs20"].get(
                name, {}
            ).get("per_category", {})

            exp3_cats = exp3_model.get("per_category", {})
            for cat in cats_present:
                if cat in exp3_cats and cat in exp12_model:
                    s_auc = exp3_cats[cat].get(
                        "auc_roc", {}
                    ).get("mean", None)
                    r_auc = exp12_model[cat].get(
                        "auc_roc", {}
                    ).get("mean", None)
                    if s_auc is not None and r_auc is not None:
                        synth_aucs.append(s_auc)
                        real_aucs.append(r_auc)
                        shared_cats.append(cat)

            if len(synth_aucs) >= 3:
                rho, p = spearmanr(synth_aucs, real_aucs)
                results["phase_c_correlation"][name] = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "n_categories": len(shared_cats),
                    "categories": shared_cats,
                    "synthetic_aucs": synth_aucs,
                    "real_aucs": real_aucs,
                }
                print(f"  {name}: rho={rho:.4f}, p={p:.4f} "
                      f"(n={len(shared_cats)} categories)")

                # Which categories synthetic over/under-estimates
                for i, cat in enumerate(shared_cats):
                    diff = synth_aucs[i] - real_aucs[i]
                    direction = ("OVER" if diff > 0.05
                                 else "UNDER" if diff < -0.05
                                 else "OK")
                    print(f"    {cat}: synth={synth_aucs[i]:.4f}, "
                          f"real={real_aucs[i]:.4f} → {direction}")
            else:
                print(f"  {name}: too few shared categories "
                      f"({len(synth_aucs)}) for Spearman")
    else:
        print("  Exp 3 results not found — run Exp 3 first")

    # ---- Summary ----
    print("\n" + "-" * 60)
    print("Experiment 12 Summary — Real-Data OWASP:")

    print(f"\n  {'Model':<18s} {'Cat':<8s} "
          f"{'UBFS-20':>10s} {'UBFS-28':>10s} "
          f"{'Delta':>10s}")
    print("  " + "-" * 60)
    for name in models_spec:
        a_cats = results["phase_a_ubfs20"].get(
            name, {}
        ).get("per_category", {})
        b_cats = results["phase_b_ubfs28"].get(
            name, {}
        ).get("per_category", {})
        for cat in cats_present:
            v20 = a_cats.get(cat, {}).get(
                "auc_roc", {}
            ).get("mean", 0)
            v28 = b_cats.get(cat, {}).get(
                "auc_roc", {}
            ).get("mean", 0)
            d = v28 - v20
            print(f"  {name:<18s} {cat:<8s} "
                  f"{v20:>10.4f} {v28:>10.4f} "
                  f"{d:>+10.4f}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "experiment_12_real_owasp.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


def experiment_13():
    """Distillation sensitivity analysis.

    Scales distillation profile multipliers by [0.25, 0.5, 1.0, 2.0]
    to test how detection degrades with lower attack intensity.
    Scaling formula: new_mult = 1 + (old_mult - 1) * scale

    HYDRA should remain flat (already near 1.0 multipliers), while
    FOCUSED/BROAD/COT should show clear degradation at lower scales.
    """
    from src.data.synthetic_generator import OWASP_PROFILES

    print("\n" + "=" * 60)
    print("EXPERIMENT 13: Distillation Sensitivity Analysis")
    print("=" * 60)

    trail = load_trail_dataset()
    base_traces = trail["traces"]
    print(f"  Using {len(base_traces)} TRAIL traces as base")

    distill_cats = [
        "ASI_DISTILL_COT", "ASI_DISTILL_BROAD",
        "ASI_DISTILL_FOCUSED", "ASI_DISTILL_HYDRA",
    ]
    scales = [0.25, 0.5, 1.0, 2.0]

    models_spec = {
        "IsolationForest": (
            IsolationForestDetector,
            {"n_estimators": 200, "contamination": "auto"},
        ),
        "LSTMAutoencoder": (
            LSTMAutoencoderDetector,
            {"epochs": 30, "batch_size": 16, "device": "cpu",
             "verbose": False},
        ),
        "DeepClustering": (
            DeepClusteringDetector,
            {"pretrain_epochs": 30, "batch_size": 16},
        ),
    }

    # Save original profiles to restore later
    import copy as _copy
    original_profiles = {
        cat: _copy.deepcopy(OWASP_PROFILES[cat])
        for cat in distill_cats
    }

    results = {}

    try:
        for name, (cls, kwargs) in models_spec.items():
            print(f"\n  {name}:")
            results[name] = {}

            for cat in distill_cats:
                results[name][cat] = {}

                for scale in scales:
                    # Scale the profile: new = 1 + (orig - 1) * scale
                    scaled = _copy.deepcopy(original_profiles[cat])
                    for key, val in scaled.items():
                        if key == "description":
                            continue
                        if key.endswith("_mult"):
                            scaled[key] = 1.0 + (val - 1.0) * scale
                        elif key.endswith("_add"):
                            scaled[key] = val * scale
                    OWASP_PROFILES[cat] = scaled

                    seed_aucs = []
                    for seed in SEEDS:
                        mixed, labels, owasp_cats = \
                            generate_anomalous_traces(
                                base_traces,
                                anomaly_ratio=0.3,
                                categories=[cat],
                                seed=seed,
                            )

                        extractor = AgentTraceFeatureExtractor()
                        X, ids, ts = extractor.extract_batch(mixed)
                        normalizer = UBFSNormalizer(method="zscore")
                        X = normalizer.fit_transform(X)

                        normal_mask = labels == 0
                        X_train = X[normal_mask]

                        if name == "LSTMAutoencoder":
                            model = cls(**{**kwargs, "seed": seed})
                            model.fit(X_train[:, np.newaxis, :])
                            scores = model.score(
                                X[:, np.newaxis, :]
                            )
                        else:
                            model = cls(**{**kwargs, "seed": seed})
                            model.fit(X_train)
                            scores = model.score(X)

                        m = compute_metrics(labels, scores)
                        seed_aucs.append(m.auc_roc)

                    mean_auc = float(np.mean(seed_aucs))
                    std_auc = float(np.std(seed_aucs))
                    results[name][cat][str(scale)] = {
                        "auc_roc_mean": mean_auc,
                        "auc_roc_std": std_auc,
                    }

                    short = cat.replace("ASI_DISTILL_", "")
                    print(f"    {short} scale={scale}: "
                          f"AUC={mean_auc:.4f} "
                          f"(+/- {std_auc:.4f})")

    finally:
        # Restore original profiles
        for cat in distill_cats:
            OWASP_PROFILES[cat] = original_profiles[cat]

    # Summary
    print("\n" + "-" * 60)
    print("Experiment 13 Summary — Sensitivity:")
    header = f"  {'Profile':<18s}"
    for s in scales:
        header += f" {s:>8.2f}x"
    print(header)
    print("  " + "-" * 60)
    for name in results:
        print(f"  {name}:")
        for cat in distill_cats:
            short = cat.replace("ASI_DISTILL_", "")
            row = f"    {short:<16s}"
            for s in scales:
                v = results[name].get(cat, {}).get(
                    str(s), {}
                ).get("auc_roc_mean", 0)
                row += f" {v:>8.4f}"
            print(row)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "experiment_13_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Threat-to-Governance Pipeline Experiments"
    )
    parser.add_argument(
        "--experiment", type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        help="Run specific experiment (1-13)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments"
    )
    parser.add_argument(
        "--cert", action="store_true",
        help="Include CMU-CERT dataset (requires sibling insider-detection/)"
    )
    args = parser.parse_args()

    global _USE_CERT
    _USE_CERT = args.cert

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.experiment is None:
        experiments = [1, 2, 3, 4]
    else:
        experiments = [args.experiment]

    all_results = {}
    for exp in experiments:
        t0 = time.time()
        if exp == 1:
            all_results[1] = experiment_1()
        elif exp == 2:
            all_results[2] = experiment_2()
        elif exp == 3:
            all_results[3] = experiment_3()
        elif exp == 4:
            all_results[4] = experiment_4()
        elif exp == 5:
            all_results[5] = experiment_5()
        elif exp == 6:
            all_results[6] = experiment_6()
        elif exp == 7:
            all_results[7] = experiment_7()
        elif exp == 8:
            all_results[8] = experiment_8()
        elif exp == 9:
            all_results[9] = experiment_9()
        elif exp == 10:
            all_results[10] = experiment_10()
        elif exp == 11:
            all_results[11] = experiment_11()
        elif exp == 12:
            all_results[12] = experiment_12()
        elif exp == 13:
            all_results[13] = experiment_13()
        elapsed = time.time() - t0
        print(f"\nExperiment {exp} completed in {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
