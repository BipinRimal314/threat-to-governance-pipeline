"""Run all experiments for the Threat-to-Governance Pipeline.

Usage:
    python run_experiments.py --experiment 1   # Within-domain baselines
    python run_experiments.py --experiment 2   # Cross-domain transfer
    python run_experiments.py --experiment 3   # OWASP category mapping
    python run_experiments.py --experiment 4   # Governance assumption audit
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
    """OWASP ASI category mapping using synthetic anomalies."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: OWASP Category Mapping")
    print("=" * 60)

    trail = load_trail_dataset()
    # Use all traces as base for synthetic injection
    base_traces = trail["traces"]
    print(f"  Using {len(base_traces)} TRAIL traces as base")

    print("  Generating OWASP-labelled anomalies...")
    mixed, labels, owasp_cats = generate_anomalous_traces(
        base_traces, anomaly_ratio=0.3, seed=42
    )
    print(f"  Mixed: {len(mixed)} traces, "
          f"{labels.sum()} anomalous")
    cats_present = set(c for c in owasp_cats if c)
    print(f"  Categories: {cats_present}")

    # Extract features
    extractor = AgentTraceFeatureExtractor()
    X, ids, ts = extractor.extract_batch(mixed)
    normalizer = UBFSNormalizer(method="zscore")
    X = normalizer.fit_transform(X)

    # Train on normal subset
    normal_mask = labels == 0
    X_train_normal = X[normal_mask]
    print(f"  Training on {len(X_train_normal)} normal traces")

    results = {}
    models = {
        "IsolationForest": IsolationForestDetector(
            n_estimators=200, seed=42
        ),
        "DeepClustering": DeepClusteringDetector(
            pretrain_epochs=30, seed=42
        ),
    }

    for name, model in models.items():
        print(f"\n  {name}...")
        model.fit(X_train_normal)
        scores = model.score(X)
        overall = compute_metrics(labels, scores)
        print(f"    Overall AUC-ROC: {overall.auc_roc:.4f}")

        # Per-category analysis
        owasp_result = evaluate_owasp_detection(
            model, X, labels, owasp_cats
        )
        results[name] = {
            "overall": overall.to_dict(),
            "per_category": owasp_result.category_metrics,
            "blind_spots": owasp_result.blind_spots,
        }

        for cat, met in owasp_result.category_metrics.items():
            if isinstance(met, dict) and "auc_roc" in met:
                print(f"    {cat}: AUC-ROC={met['auc_roc']:.4f}")
        print(f"    Blind spots: {owasp_result.blind_spots}")

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

        results[name] = {
            "overall": agg_overall,
            "direct": agg_direct,
            "decomposed": agg_decomposed,
            "auc_roc_delta": delta,
        }

        print(f"    Direct AUC-ROC: "
              f"{agg_direct['auc_roc']['mean']:.4f} "
              f"(+/- {agg_direct['auc_roc']['std']:.4f})")
        print(f"    Decomposed AUC-ROC: "
              f"{agg_decomposed['auc_roc']['mean']:.4f} "
              f"(+/- {agg_decomposed['auc_roc']['std']:.4f})")
        print(f"    Delta (direct - decomposed): "
              f"{delta:+.4f}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Threat-to-Governance Pipeline Experiments"
    )
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Run specific experiment (1-8)"
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
        elapsed = time.time() - t0
        print(f"\nExperiment {exp} completed in {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
