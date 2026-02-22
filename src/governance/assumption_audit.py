"""Governance assumption audit.

Analyses what "normal" means in each domain and interrogates
the assumptions embedded in anomaly detection systems. This
connects to the philosophical framing: both insider threat
and agent monitoring encode assumptions about trust.

Key question: Who decides what "normal" looks like, and what
does that decision exclude?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..features.ubfs_schema import (
    FEATURE_DEFINITIONS,
    FeatureCategory,
    UBFSConfig,
    ubfs_feature_names,
)


@dataclass
class NormalityAssumption:
    """An assumption embedded in the definition of 'normal'."""

    feature_name: str
    category: str
    cert_assumption: str
    agent_assumption: str
    governance_implication: str


# Curated list of governance-relevant assumptions
NORMALITY_ASSUMPTIONS = [
    NormalityAssumption(
        feature_name="after_hours_ratio",
        category="TEMPORAL",
        cert_assumption=(
            "Employees working outside 9-5 are suspicious."
        ),
        agent_assumption=(
            "Agents executing outside scheduled windows "
            "are anomalous."
        ),
        governance_implication=(
            "Penalises non-standard work patterns. In the "
            "human case, this disproportionately flags shift "
            "workers, caregivers, and employees in different "
            "time zones. In the agent case, it assumes agents "
            "should have 'work hours'."
        ),
    ),
    NormalityAssumption(
        feature_name="peer_distance",
        category="DEVIATION",
        cert_assumption=(
            "Employees who behave differently from their "
            "peer group are suspicious."
        ),
        agent_assumption=(
            "Agents that deviate from their type baseline "
            "are anomalous."
        ),
        governance_implication=(
            "Assumes homogeneity within groups. Penalises "
            "legitimate variation. In both cases, the "
            "definition of 'peer group' encodes "
            "organisational hierarchies."
        ),
    ),
    NormalityAssumption(
        feature_name="resource_breadth",
        category="SCOPE",
        cert_assumption=(
            "Accessing many different systems is suspicious."
        ),
        agent_assumption=(
            "Using many different tools is anomalous."
        ),
        governance_implication=(
            "Privileges the specialist over the generalist. "
            "Cross-functional employees and multi-tool agents "
            "are structurally more likely to be flagged."
        ),
    ),
    NormalityAssumption(
        feature_name="data_volume_norm",
        category="VOLUME",
        cert_assumption=(
            "Large data transfers indicate exfiltration risk."
        ),
        agent_assumption=(
            "High token usage indicates potential misuse."
        ),
        governance_implication=(
            "Equates volume with risk. Does not distinguish "
            "between legitimate large tasks and malicious "
            "data extraction."
        ),
    ),
    NormalityAssumption(
        feature_name="action_entropy",
        category="SEQUENCE",
        cert_assumption=(
            "Unpredictable action sequences are suspicious."
        ),
        agent_assumption=(
            "High entropy in tool-call patterns is anomalous."
        ),
        governance_implication=(
            "Penalises creative or exploratory behaviour. "
            "Rewards routine and predictability. This maps "
            "to broader questions about whether we value "
            "conformity over adaptability in both human and "
            "AI systems."
        ),
    ),
    NormalityAssumption(
        feature_name="privilege_deviation_index",
        category="PRIVILEGE",
        cert_assumption=(
            "Using access above your typical level is suspicious."
        ),
        agent_assumption=(
            "Invoking tools above granted scope is anomalous."
        ),
        governance_implication=(
            "Assumes stable role definitions. In dynamic "
            "organisations and agentic systems, legitimate "
            "scope expansion (promotions, new capabilities) "
            "looks identical to privilege escalation."
        ),
    ),
]


def audit_normality_assumptions() -> List[NormalityAssumption]:
    """Return the curated list of governance assumptions."""
    return NORMALITY_ASSUMPTIONS


def compare_baseline_distributions(
    cert_vectors: np.ndarray,
    agent_vectors: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compare what 'normal' looks like in each domain.

    Args:
        cert_vectors: UBFS vectors from CERT normal data.
        agent_vectors: UBFS vectors from agent normal data.

    Returns:
        Per-feature comparison of distributions.
    """
    names = ubfs_feature_names()
    comparison = {}

    for i, name in enumerate(names):
        cert_vals = cert_vectors[:, i]
        agent_vals = agent_vectors[:, i]

        comparison[name] = {
            "cert_mean": float(np.mean(cert_vals)),
            "cert_std": float(np.std(cert_vals)),
            "agent_mean": float(np.mean(agent_vals)),
            "agent_std": float(np.std(agent_vals)),
            "distribution_divergence": float(
                _kl_divergence_approx(cert_vals, agent_vals)
            ),
        }

    return comparison


def _kl_divergence_approx(
    p: np.ndarray, q: np.ndarray, n_bins: int = 50
) -> float:
    """Approximate KL divergence between two distributions."""
    # Create shared bins
    all_vals = np.concatenate([p, q])
    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalise
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def governance_report(
    assumptions: List[NormalityAssumption],
    distribution_comparison: Dict[str, Dict[str, float]],
) -> str:
    """Generate a governance analysis report as markdown."""
    lines = [
        "# Governance Assumption Audit",
        "",
        "## What 'Normal' Means: A Cross-Domain Comparison",
        "",
        "This report examines the assumptions embedded in the "
        "definition of 'normal behaviour' used by each anomaly "
        "detection model, and what these assumptions imply for "
        "governance.",
        "",
        "---",
        "",
    ]

    for assumption in assumptions:
        feat = assumption.feature_name
        lines.extend([
            f"### {feat} ({assumption.category})",
            "",
            f"**Insider Threat:** {assumption.cert_assumption}",
            "",
            f"**Agent Monitoring:** {assumption.agent_assumption}",
            "",
            f"**Governance Implication:** "
            f"{assumption.governance_implication}",
            "",
        ])

        if feat in distribution_comparison:
            comp = distribution_comparison[feat]
            lines.extend([
                f"**Distribution:** CERT mean={comp['cert_mean']:.3f} "
                f"(std={comp['cert_std']:.3f}), "
                f"Agent mean={comp['agent_mean']:.3f} "
                f"(std={comp['agent_std']:.3f}), "
                f"KL divergence={comp['distribution_divergence']:.3f}",
                "",
            ])

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
