"""Generate publication-ready figures for the pipeline.

Produces 6 figures in results/figures/:
1. Feature transfer heatmap
2. ROC curves (within-domain + cross-domain)
3. OWASP detection matrix
4. Reconstruction error distributions
5. t-SNE latent space visualization
6. Governance comparison table
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent))

from src.data.trail_loader import load_trail_dataset, get_trail_labels
from src.data.trace_loader import load_trace_dataset, trace_to_otel_format
from src.data.synthetic_generator import generate_anomalous_traces
from src.features.agent_extractor import AgentTraceFeatureExtractor
from src.features.ubfs_schema import (
    UBFSNormalizer,
    UBFSConfig,
    FeatureCategory,
    ubfs_feature_names,
)
from src.models.isolation_forest import IsolationForestDetector
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.models.deep_clustering import DeepClusteringDetector

FIGURES_DIR = Path(__file__).parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "trail": "#2196F3",
    "trace": "#FF9800",
    "normal": "#4CAF50",
    "anomalous": "#F44336",
    "if": "#2196F3",
    "lstm": "#FF9800",
    "dc": "#9C27B0",
}


def load_data():
    """Load and prepare all data for plotting."""
    print("Loading datasets...")
    trail = load_trail_dataset()
    trail_labels = get_trail_labels(trail["annotations"])

    trace_data = load_trace_dataset()
    trace_otel = [
        trace_to_otel_format(t)
        for t in trace_data["trajectories"]
    ]

    extractor_trail = AgentTraceFeatureExtractor()
    X_trail, _, _ = extractor_trail.extract_batch(trail["traces"])
    norm_trail = UBFSNormalizer(method="zscore")
    X_trail = norm_trail.fit_transform(X_trail)

    extractor_trace = AgentTraceFeatureExtractor()
    X_trace, _, _ = extractor_trace.extract_batch(trace_otel)
    norm_trace = UBFSNormalizer(method="zscore")
    X_trace = norm_trace.fit_transform(X_trace)

    return {
        "X_trail": X_trail,
        "y_trail": trail_labels,
        "trail": trail,
        "X_trace": X_trace,
        "y_trace": trace_data["labels"],
    }


def figure_1_feature_heatmap(data):
    """Feature correlation heatmap between TRAIL and TRACE."""
    print("  Figure 1: Feature transfer heatmap...")
    names = ubfs_feature_names()
    short_names = [n.replace("_", "\n") for n in names]

    # Compare feature distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Correlation within TRAIL
    corr_trail = np.corrcoef(data["X_trail"].T)
    sns.heatmap(
        corr_trail, ax=axes[0], cmap="RdBu_r", center=0,
        vmin=-1, vmax=1,
        xticklabels=short_names, yticklabels=short_names,
    )
    axes[0].set_title("TRAIL Feature Correlations")

    # Correlation within TRACE
    corr_trace = np.corrcoef(data["X_trace"].T)
    sns.heatmap(
        corr_trace, ax=axes[1], cmap="RdBu_r", center=0,
        vmin=-1, vmax=1,
        xticklabels=short_names, yticklabels=short_names,
    )
    axes[1].set_title("TRACE Feature Correlations")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_feature_heatmap.png")
    plt.close()


def figure_2_roc_curves(data):
    """ROC curves: 3 models x within + cross domain."""
    print("  Figure 2: ROC curves...")

    X_trail = data["X_trail"]
    y_trail = data["y_trail"]
    X_trace = data["X_trace"]
    y_trace = data["y_trace"]

    trail_normal = X_trail[y_trail == 0]
    trace_normal = X_trace[y_trace == 0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    models = [
        ("Isolation Forest", IsolationForestDetector,
         {"n_estimators": 200, "seed": 42}),
        ("LSTM Autoencoder", LSTMAutoencoderDetector,
         {"epochs": 30, "batch_size": 16, "device": "cpu",
          "seed": 42, "verbose": False}),
        ("Deep Clustering", DeepClusteringDetector,
         {"pretrain_epochs": 30, "batch_size": 16, "seed": 42}),
    ]

    for col, (name, cls, kwargs) in enumerate(models):
        # Within-domain (top row)
        ax = axes[0, col]
        for dataset, X, y, normal, color, label in [
            ("TRAIL", X_trail, y_trail, trail_normal,
             COLORS["trail"], "TRAIL"),
            ("TRACE", X_trace, y_trace, trace_normal,
             COLORS["trace"], "TRACE"),
        ]:
            model = cls(**kwargs)
            if name == "LSTM Autoencoder":
                model.fit(normal[:, np.newaxis, :])
                scores = model.score(X[:, np.newaxis, :])
            else:
                model.fit(normal)
                scores = model.score(X)
            if len(np.unique(y)) >= 2:
                fpr, tpr, _ = roc_curve(y, scores)
                ax.plot(fpr, tpr, color=color, label=label,
                        linewidth=1.5)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        if col == 0:
            ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name}\n(Within-Domain)")
        ax.legend(loc="lower right")

        # Cross-domain (bottom row)
        ax = axes[1, col]

        # Train TRAIL → eval TRACE
        model = cls(**kwargs)
        if name == "LSTM Autoencoder":
            model.fit(trail_normal[:, np.newaxis, :])
            scores = model.score(X_trace[:, np.newaxis, :])
        else:
            model.fit(trail_normal)
            scores = model.score(X_trace)
        if len(np.unique(y_trace)) >= 2:
            fpr, tpr, _ = roc_curve(y_trace, scores)
            ax.plot(fpr, tpr, color=COLORS["trail"],
                    label="TRAIL→TRACE", linewidth=1.5)

        # Train TRACE → eval TRAIL
        model = cls(**kwargs)
        if name == "LSTM Autoencoder":
            model.fit(trace_normal[:, np.newaxis, :])
            scores = model.score(X_trail[:, np.newaxis, :])
        else:
            model.fit(trace_normal)
            scores = model.score(X_trail)
        if len(np.unique(y_trail)) >= 2:
            fpr, tpr, _ = roc_curve(y_trail, scores)
            ax.plot(fpr, tpr, color=COLORS["trace"],
                    label="TRACE→TRAIL", linewidth=1.5)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        if col == 0:
            ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name}\n(Cross-Domain)")
        ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_roc_curves.png")
    plt.close()


def figure_3_owasp_matrix():
    """OWASP detection heatmap."""
    print("  Figure 3: OWASP detection matrix...")

    tables_dir = Path(__file__).parent / "results" / "tables"
    with open(tables_dir / "experiment_3_owasp.json") as f:
        results = json.load(f)

    categories = ["ASI01", "ASI02", "ASI05", "ASI09", "ASI10"]
    cat_labels = [
        "ASI01\nGoal Hijack",
        "ASI02\nTool Misuse",
        "ASI05\nMemory\nPoisoning",
        "ASI09\nExcessive\nAgency",
        "ASI10\nRogue\nAgents",
    ]
    model_names = list(results.keys())

    matrix = np.zeros((len(model_names), len(categories)))
    for i, model in enumerate(model_names):
        per_cat = results[model]["per_category"]
        for j, cat in enumerate(categories):
            if cat in per_cat and isinstance(per_cat[cat], dict):
                matrix[i, j] = per_cat[cat].get("auc_roc", 0)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=0, vmax=1,
        xticklabels=cat_labels,
        yticklabels=model_names,
        linewidths=0.5,
    )
    ax.set_ylabel("Model")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_owasp_matrix.png")
    plt.close()


def figure_4_recon_errors(data):
    """Reconstruction error distributions."""
    print("  Figure 4: Reconstruction error distributions...")

    X_trail = data["X_trail"]
    y_trail = data["y_trail"]
    X_trace = data["X_trace"]
    y_trace = data["y_trace"]

    trail_normal = X_trail[y_trail == 0]
    trace_normal = X_trace[y_trace == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # TRAIL: Deep Clustering reconstruction errors
    dc = DeepClusteringDetector(
        pretrain_epochs=30, batch_size=16, seed=42
    )
    dc.fit(trail_normal)
    trail_scores = dc.score(X_trail)

    ax = axes[0]
    ax.hist(
        trail_scores[y_trail == 0], bins=30, alpha=0.6,
        color=COLORS["normal"], label="Normal", density=True,
    )
    ax.hist(
        trail_scores[y_trail == 1], bins=30, alpha=0.6,
        color=COLORS["anomalous"], label="Anomalous", density=True,
    )
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("TRAIL (Deep Clustering)")
    ax.legend()

    # TRACE
    dc2 = DeepClusteringDetector(
        pretrain_epochs=30, batch_size=16, seed=42
    )
    dc2.fit(trace_normal)
    trace_scores = dc2.score(X_trace)

    ax = axes[1]
    ax.hist(
        trace_scores[y_trace == 0], bins=30, alpha=0.6,
        color=COLORS["normal"], label="Normal", density=True,
    )
    ax.hist(
        trace_scores[y_trace == 1], bins=30, alpha=0.6,
        color=COLORS["anomalous"], label="Anomalous", density=True,
    )
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("TRACE (Deep Clustering)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_score_distributions.png")
    plt.close()


def figure_5_tsne(data):
    """t-SNE of Deep Clustering latent space."""
    print("  Figure 5: t-SNE latent space...")

    X_trail = data["X_trail"]
    y_trail = data["y_trail"]
    X_trace = data["X_trace"]
    y_trace = data["y_trace"]

    # Combine and train DC on all data
    X_all = np.vstack([X_trail, X_trace])
    domain = np.array(
        ["TRAIL"] * len(X_trail) + ["TRACE"] * len(X_trace)
    )
    labels = np.concatenate([y_trail, y_trace])

    dc = DeepClusteringDetector(
        pretrain_epochs=50, batch_size=32, seed=42
    )
    dc.fit(X_all)
    latent = dc.encode(X_all)

    # t-SNE on latent
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=30
    )
    emb = tsne.fit_transform(latent)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color by domain
    ax = axes[0]
    for d, color in [("TRAIL", COLORS["trail"]),
                     ("TRACE", COLORS["trace"])]:
        mask = domain == d
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, alpha=0.5, s=15, label=d,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Coloured by Domain")
    ax.legend()

    # Color by anomaly status
    ax = axes[1]
    for lbl, color, label in [
        (0, COLORS["normal"], "Normal"),
        (1, COLORS["anomalous"], "Anomalous"),
    ]:
        mask = labels == lbl
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, alpha=0.5, s=15, label=label,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Coloured by Anomaly Status")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_tsne_latent.png")
    plt.close()


def figure_6_governance_table():
    """Governance comparison table as figure."""
    print("  Figure 6: Governance comparison table...")

    assumptions = [
        ("after_hours_ratio",
         "Off-hours = suspicious",
         "Off-schedule = anomalous",
         "Penalises non-standard patterns"),
        ("peer_distance",
         "Deviating from peers = suspicious",
         "Deviating from type baseline",
         "Assumes group homogeneity"),
        ("resource_breadth",
         "Many systems accessed = suspicious",
         "Many tools invoked = anomalous",
         "Privileges specialists"),
        ("data_volume_norm",
         "Large transfers = exfiltration risk",
         "High token usage = misuse",
         "Equates volume with risk"),
        ("action_entropy",
         "Unpredictable actions = suspicious",
         "High entropy = anomalous",
         "Penalises creativity"),
        ("privilege_deviation",
         "Above-level access = suspicious",
         "Above-scope tools = anomalous",
         "Assumes stable roles"),
    ]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    col_labels = [
        "UBFS Feature", "Insider Threat\nAssumption",
        "Agent Monitoring\nAssumption",
        "Governance\nImplication",
    ]
    cell_text = [list(row) for row in assumptions]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colColours=["#E3F2FD"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(4)))
    table.scale(1, 1.8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_governance_table.png")
    plt.close()


def main():
    print("Generating publication-ready figures...")
    data = load_data()

    figure_1_feature_heatmap(data)
    figure_2_roc_curves(data)
    figure_3_owasp_matrix()
    figure_4_recon_errors(data)
    figure_5_tsne(data)
    figure_6_governance_table()

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
