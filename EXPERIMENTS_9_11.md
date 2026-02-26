# Experiments 9-11: Extension Plan

**Purpose:** Three new experiments that strengthen the paper for peer review submission. These address the two main questions reviewers will ask: (1) "What about adaptive adversaries?" and (2) "Does this generalize beyond OWASP?"

**Prerequisites:** Experiments 1-8 must be complete (results in `results/tables/`). All existing code and models must work.

**Run on:** Ubuntu PC (RTX 4060, Ryzen 5 5600X, 16GB RAM). CMU-CERT data must be at `../insider-detection/data/r4.2/`.

**Estimated total time:** ~8-10 hours (code implementation + experiment runs). Can run overnight.

---

## Experiment 9: Adversarial Evasion Testing

**The question:** If an attacker knows UBFS-based monitoring exists, how easily can they evade detection?

**Why this matters:** Every security venue (USENIX, CCS, IEEE S&P) reviewer will ask "what about adaptive adversaries?" This experiment provides the answer. It also formalizes the "boiling frog" pattern from Paper 1 into a measurable evasion framework.

**What to implement:**

### 9A: Feature Clamping Evasion

For each of the 20 UBFS features, compute the 95th percentile of the normal training distribution. Then for all anomalous samples, cap any feature value that exceeds this threshold back to the 95th percentile. This simulates an attacker who knows the monitoring thresholds and stays just below them.

```python
def clamp_evasion(X_anomalous, X_normal, percentile=95):
    """Cap anomalous features at the Nth percentile of normal."""
    thresholds = np.percentile(X_normal, percentile, axis=0)
    X_clamped = X_anomalous.copy()
    for i in range(X_clamped.shape[1]):
        X_clamped[:, i] = np.minimum(X_clamped[:, i], thresholds[i])
    return X_clamped
```

Evaluate: Run all 3 models on clamped anomalies. Report AUC-ROC before vs after clamping for each OWASP category.

### 9B: Gradual Escalation Evasion

Simulate the "boiling frog" at the feature level. Start from a normal profile and linearly interpolate toward the anomalous profile in N steps (N = 5, 10, 20, 50). At each step, evaluate detection.

```python
def gradual_escalation(X_normal_profile, X_anomalous_target, n_steps=20):
    """Interpolate from normal to anomalous in n_steps."""
    trajectory = []
    for step in range(n_steps + 1):
        alpha = step / n_steps
        X_step = (1 - alpha) * X_normal_profile + alpha * X_anomalous_target
        trajectory.append(X_step)
    return np.array(trajectory)
```

Report: For each OWASP category, at what step (what alpha) does the model first detect the anomaly? Plot detection threshold vs. escalation step.

### 9C: Mimicry Evasion

For each anomalous sample, find the nearest normal neighbor in UBFS space (Euclidean distance). Replace all features EXCEPT the privilege_deviation_index (dim 20) with the neighbor's values. This preserves the "attack payload" (privilege escalation) while making everything else look normal.

Report: AUC-ROC before vs after mimicry. Hypothesis: this should be devastatingly effective, especially for ASI02 which already relies on privilege features.

### Output

Save to `results/tables/experiment_9_adversarial.json`:
```json
{
  "clamping": {
    "model_name": {
      "category": {"baseline_auc": float, "clamped_auc": float, "drop_pct": float}
    }
  },
  "escalation": {
    "model_name": {
      "category": {"detection_step": int, "alpha_at_detection": float}
    }
  },
  "mimicry": {
    "model_name": {
      "category": {"baseline_auc": float, "mimicry_auc": float, "drop_pct": float}
    }
  }
}
```

**Expected results:**
- Feature clamping: 15-30% AUC-ROC drop across categories. ASI05 (Memory Poisoning, currently 0.969) should drop most because its detection relies on volume spikes easily capped.
- Gradual escalation: detection threshold around alpha=0.3-0.5 for most categories. ASI02 should never trigger (already near chance).
- Mimicry: most effective evasion. Should push most categories below 0.70. Confirms that behavioral monitoring's vulnerability is structural.

**Time estimate:** ~2 hours code, ~2 hours running (5 seeds x 3 models x 3 evasion strategies x 5 OWASP categories).

---

## Experiment 10: Temporal Dynamics in Agent Traces

**The question:** Does the optimal monitoring window size transfer across domains?

**Why this matters:** Paper 1 showed 7-day windows outperform 14 and 30-day windows for insider detection. If agent traces show an analogous pattern (shorter windows are better), that's a structural insight about behavioral monitoring: early warning signals are more predictive than full histories. This strengthens the "structural analogy" argument.

**What to implement:**

### 10A: Window Ablation on TRAIL

For TRAIL traces, vary the number of spans used for UBFS feature extraction:
- **5 spans** (first 5 spans only)
- **10 spans**
- **20 spans**
- **Full trace** (all spans, current default)

For each window, extract UBFS features, train models on normal, evaluate.

Implementation: Modify `AgentTraceFeatureExtractor.extract_single()` to accept a `max_spans` parameter. If the trace has more spans than `max_spans`, take only the first `max_spans`.

```python
def extract_single(self, trace, max_spans=None):
    spans = trace.get("spans", trace.get("child_spans", []))
    if max_spans is not None:
        spans = spans[:max_spans]
    # ... rest of extraction
```

### 10B: Window Ablation on ATBench

Same as 10A but on ATBench traces. Compare the optimal window across datasets.

### 10C: Cross-Reference with Paper 1

Compare the optimal window ratios:
- Paper 1: 7 days optimal out of 7/14/30 day options (23% of max)
- TRAIL: N spans optimal out of 5/10/20/full (compute ratio)
- ATBench: M spans optimal out of 5/10/20/full

If the optimal ratio is proportionally similar, the structural analogy goes deeper than feature-level.

### Output

Save to `results/tables/experiment_10_temporal.json`:
```json
{
  "TRAIL": {
    "model_name": {
      "5_spans": {"auc_roc_mean": float, "auc_roc_std": float},
      "10_spans": {"auc_roc_mean": float, "auc_roc_std": float},
      "20_spans": {"auc_roc_mean": float, "auc_roc_std": float},
      "full": {"auc_roc_mean": float, "auc_roc_std": float}
    }
  },
  "ATBench": { ... same structure ... },
  "optimal_window": {
    "TRAIL": {"model_name": "N_spans"},
    "ATBench": {"model_name": "N_spans"},
    "CERT_reference": "7_days"
  }
}
```

**Expected results:**
- Shorter windows (5-10 spans) should perform comparably or better than full traces for TRAIL (148 traces, most anomalies are early in execution).
- ATBench may differ (multi-turn traces where the dangerous action is later).
- If short windows win in both, that's a finding: behavioral monitoring should focus on early signals, not long histories.

**Time estimate:** ~1 hour code (minor modification to extractor), ~1.5 hours running.

---

## Experiment 11: MITRE ATLAS Mapping

**The question:** Does the UBFS detection framework map to MITRE ATLAS as well as it maps to OWASP?

**Why this matters:** OWASP ASI is one taxonomy. MITRE ATLAS is the other major adversarial ML threat taxonomy, used by the security community. Showing your framework detects across both makes it citable by a much wider audience and strengthens the paper for security venues (USENIX, IEEE S&P).

**What to implement:**

### 11A: ATLAS Technique Profiles

Map relevant MITRE ATLAS techniques to UBFS feature perturbation profiles, following the same pattern as the OWASP synthetic injection in Experiment 3.

Key ATLAS techniques to map:

| ATLAS ID | Technique | UBFS Mapping |
|----------|-----------|-------------|
| AML.T0043 | Craft Adversarial Data | VOLUME + SCOPE (data generation patterns) |
| AML.T0044 | Full Model Replication | FREQUENCY + VOLUME (systematic querying, high extraction) |
| AML.T0048 | Model Extraction via API | FREQUENCY + SEQUENCE (repeated similar queries) |
| AML.T0040 | ML Model Inference API Access | SCOPE + FREQUENCY (broad probing) |
| AML.T0042 | Verify Attack | SEQUENCE + DEVIATION (unusual test patterns) |
| AML.T0025 | Exfiltration via ML Inference API | VOLUME + PRIVILEGE (extracting model internals) |

Create synthetic profiles in `synthetic_generator.py` following the existing OWASP pattern:

```python
# ATLAS profiles — add to SYNTHETIC_PROFILES dict
"AML_T0044_REPLICATION": {
    "primary_event_count_mult": 10.0,    # Systematic querying
    "data_volume_mult": 15.0,            # High extraction
    "event_rate_add": 5.0,               # Elevated rate
    "resource_breadth_mult": 3.0,        # Multiple model endpoints
    "action_entropy_mult": 0.3,          # Low entropy (systematic)
    "repetition_add": 0.6,              # Repetitive patterns
    "description": "Full Model Replication (ATLAS AML.T0044)",
},
"AML_T0048_EXTRACTION": {
    "primary_event_count_mult": 8.0,
    "action_entropy_mult": 0.4,
    "repetition_add": 0.5,
    "data_volume_mult": 10.0,
    "resource_breadth_mult": 1.5,       # Narrow target
    "description": "Model Extraction via API (ATLAS AML.T0048)",
},
"AML_T0043_ADVERSARIAL": {
    "data_volume_mult": 8.0,
    "resource_breadth_mult": 5.0,
    "action_entropy_mult": 0.6,
    "transition_novelty_mult": 1.5,     # Novel patterns (adversarial search)
    "description": "Craft Adversarial Data (ATLAS AML.T0043)",
},
"AML_T0025_EXFILTRATION": {
    "data_volume_mult": 12.0,
    "privilege_deviation_add": 2.0,     # Needs elevated access
    "resource_breadth_mult": 2.0,
    "event_rate_add": 3.0,
    "description": "Exfiltration via Inference API (ATLAS AML.T0025)",
},
"AML_T0042_VERIFY": {
    "action_entropy_mult": 1.5,         # Unusual test patterns
    "transition_novelty_mult": 2.0,     # Novel sequences
    "peer_distance_add": 1.5,           # Deviates from normal
    "event_rate_add": 1.5,
    "description": "Verify Attack (ATLAS AML.T0042)",
},
```

### 11B: Detection Evaluation

Run the same pipeline as Experiment 3 but with ATLAS categories instead of OWASP:
1. Inject synthetic ATLAS anomalies into TRAIL traces
2. Train models on normal
3. Evaluate per ATLAS category
4. Report AUC-ROC per technique per model

### 11C: Cross-Taxonomy Comparison

Create a comparison table:

| Detection Tier | OWASP Categories | ATLAS Techniques |
|---------------|-----------------|-----------------|
| Strong (>0.80) | ASI05, ASI09, ASI10 | ? |
| Moderate (0.60-0.80) | ASI01 | ? |
| Blind spot (<0.60) | ASI02 | ? |

Hypothesis: AML.T0044 (Full Replication) and AML.T0048 (Extraction) will map to the same detection tier as the corresponding OWASP distillation profiles. AML.T0025 (Exfiltration) will be detectable due to VOLUME signature. The novel finding: does any ATLAS technique fall into a NEW blind spot not covered by the OWASP analysis?

### Output

Save to `results/tables/experiment_11_atlas.json`:
```json
{
  "atlas_detection_matrix": {
    "model_name": {
      "AML_T0044_REPLICATION": {"auc_roc_mean": float, "auc_roc_std": float},
      "AML_T0048_EXTRACTION": {...},
      "AML_T0043_ADVERSARIAL": {...},
      "AML_T0025_EXFILTRATION": {...},
      "AML_T0042_VERIFY": {...}
    }
  },
  "cross_taxonomy": {
    "strong": {"owasp": [...], "atlas": [...]},
    "moderate": {"owasp": [...], "atlas": [...]},
    "blind_spot": {"owasp": [...], "atlas": [...]}
  }
}
```

**Time estimate:** ~2 hours code (profiles + experiment function), ~2 hours running.

---

## Implementation Instructions for Claude on Ubuntu

### Step 1: Pull Latest Code
```bash
cd ~/threat-to-governance-pipeline  # or wherever the repo is
git pull origin main
```

### Step 2: Activate Environment
```bash
source .venv/bin/activate
```

### Step 3: Implement Experiments

Add the three experiment functions to `run_experiments.py` following the exact pattern of experiments 5-8. Each experiment:
- Uses the same model dictionary (IF, LSTM-AE, DC)
- Runs 5 seeds [42, 43, 44, 45, 46]
- Reports mean +/- std for AUC-ROC
- Saves results to `results/tables/experiment_{N}_{name}.json`

For Experiment 9: Work entirely with existing UBFS features. No new modules needed. Just manipulate the feature vectors post-extraction.

For Experiment 10: Modify `AgentTraceFeatureExtractor` to accept `max_spans` param. Or simpler: truncate traces before passing to extractor.

For Experiment 11: Add ATLAS profiles to `src/data/synthetic_generator.py` alongside existing OWASP profiles. The `generate_anomalous_traces()` function should already support arbitrary category names.

### Step 4: Update CLI
```python
# In main(), extend choices:
parser.add_argument(
    "--experiment", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    help="Run specific experiment (1-11)"
)

# Add to dispatch:
elif exp == 9:
    all_results[9] = experiment_9()
elif exp == 10:
    all_results[10] = experiment_10()
elif exp == 11:
    all_results[11] = experiment_11()
```

### Step 5: Run Overnight
```bash
# Run all three new experiments
python run_experiments.py --experiment 9 2>&1 | tee results/exp9.log
python run_experiments.py --experiment 10 2>&1 | tee results/exp10.log
python run_experiments.py --experiment 11 2>&1 | tee results/exp11.log

# Or chain them:
python run_experiments.py --experiment 9 2>&1 | tee results/exp9.log && \
python run_experiments.py --experiment 10 2>&1 | tee results/exp10.log && \
python run_experiments.py --experiment 11 2>&1 | tee results/exp11.log
```

### Step 6: Verify Results
```bash
ls -la results/tables/experiment_9_adversarial.json
ls -la results/tables/experiment_10_temporal.json
ls -la results/tables/experiment_11_atlas.json
```

### Step 7: Commit Results
```bash
git add results/tables/experiment_*.json run_experiments.py src/
git commit -m "feat(exp): add experiments 9-11 — adversarial evasion, temporal dynamics, ATLAS mapping"
git push origin main
```

---

## Notes

- All experiments use TRAIL as the base dataset for synthetic injection. CERT is optional (--cert flag) and only needed for cross-domain experiments.
- HuggingFace login required for TRAIL and ATBench: `huggingface-cli login`
- The RTX 4060 should handle all three experiments overnight. Exp 9 is the most compute-intensive (~4 hours). Exp 10 and 11 are lighter (~1.5-2 hours each).
- If VRAM runs low during LSTM training, reduce batch_size to 8 in the model kwargs.
