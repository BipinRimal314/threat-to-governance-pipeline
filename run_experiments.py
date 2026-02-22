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


def main():
    parser = argparse.ArgumentParser(
        description="Threat-to-Governance Pipeline Experiments"
    )
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2, 3, 4],
        help="Run specific experiment (1-4)"
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
        elapsed = time.time() - t0
        print(f"\nExperiment {exp} completed in {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
