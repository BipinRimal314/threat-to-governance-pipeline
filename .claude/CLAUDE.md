# Threat-to-Governance Pipeline

Insider threat detection models applied to AI agent behavioral monitoring. Experiments 1-13 are complete; results in `results/tables/experiment_{1-13}_*.json`. Design specs for the later experiments: `EXPERIMENTS_5_8.md`, `EXPERIMENTS_9_11.md`.

## Current Status

- **Experiments 1-13:** complete. Results in `results/tables/`. Paper compiled at `paper/main.pdf`.
  - Exp 9: Adversarial evasion testing (clamping, gradual escalation, mimicry)
  - Exp 10: Temporal window ablation on agent traces
  - Exp 11: MITRE ATLAS taxonomy mapping

## Development Workflow

```bash
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Fresh run of all 13 on the Ubuntu box, into results/runs/<ts>/, compared to results/tables
./scripts/testrun.sh            # --check for preflight only

# Run all 13 in place (overwrites results/tables)
python run_experiments.py --all --cert

# Run specific experiment
python run_experiments.py --experiment 5

# Generate figures
python generate_figures.py
```

## Architecture

```
Data Layer (cert_loader, trail_loader, trace_loader, synthetic_generator)
    ↓
Feature Layer (cert_extractor, agent_extractor → UBFS 20-dim vectors)
    ↓
Model Layer (IsolationForest, LSTMAutoencoder, DeepClustering, Ensemble)
    ↓
Evaluation Layer (metrics, transfer_analysis, owasp_mapper)
    ↓
Governance Layer (assumption_audit)
```

## Target Machine

**Ubuntu PC (primary):** RTX 4060 (8GB VRAM), Ryzen 5 5600X, 16GB RAM, 1TB SSD.
**MacBook M4 Pro:** Development only. No CMU-CERT data.

PyTorch device config in `configs/model_configs.yaml` → `device: "auto"` (detects CUDA/MPS/CPU).

## Completed Experiments (1-4)

| Exp | What | Key Result |
|-----|------|------------|
| 1 | Within-domain baselines | DC 0.897 on TRAIL; TRACE ~0.50 (undetectable) |
| 2 | Cross-domain transfer | **CERT→TRAIL: 0.731→0.711 (within seed noise ±0.047; "transfer within noise", NOT "97% retention")** |
| 3 | OWASP detection matrix | ASI02 Tool Misuse 0.520 on **synthetic** data. **This is an artifact — see Common Mistakes.** |
| 4 | Governance assumptions | 6 embedded assumptions audited |

Results: `results/tables/experiment_{1-4}_*.json`
Figures: `results/figures/fig{1-6}_*.png`

## Common Mistakes — Read This First

- **The ASI02 blind spot is NOT real.** Synthetic Exp 3 puts ASI02 Tool Misuse at 0.520; real ATBench data (Exp 12) puts it at 0.871/0.823/0.910 — the *best*-detected of six categories. Spearman rho between the synthetic and real rankings is **-1.000**, an exact inversion, on all three models. Cause: the generator defines ASI02 to modify only privilege features while preserving all behavioural structure, so it is undetectable *by construction*. `paper/main.tex` was rewritten around this on 7 Aug 2026. **Do not restate the blind-spot claim.** Anything below in the EXPERIMENT EXTENSION PLAN predates this and is a historical record of what we believed, not current fact.
- **Synthetic per-category numbers do not predict real ones.** This applies to Exp 3, 8, 9 and 11 alike — all four inject perturbations we specify and then measure how visible they are. Treat them as hypothesis generation.
- **Exp 3/8/9 result JSONs were key-migrated, not re-run** (ASI05→ASI06, ASI09→EXCESSIVE_AGENCY, 7 Aug 2026). Values are unchanged. TRAIL is gated and this machine has no HF token; authenticate and re-run Exp 3 to verify.
- **Thesis AUC-ROC numbers**: IF 0.799, LSTM 0.770 (on full 24-dim features). Through UBFS (20-dim): IF 0.731, LSTM 0.723, DC 0.697. The lossy compression trades fidelity for cross-domain portability. **Never use 0.985.**
- **TRACE is structurally undetectable** (~0.50 AUC-ROC). Reward hacking produces traces identical to normal. This is expected, not a bug.
- **TRAIL has extreme class imbalance**: 143/148 traces have errors. Only 5 normal traces. Train on normal subset, test on all.
- **TRAIL trace field is JSON string** with nested `child_spans`. Loader flattens recursively.
- **TRACE label "0" = benign**, anything else (e.g. "1.1.1") = reward hack. Some turns have `content: None`.
- **CMU-CERT needs sibling directory**: `../insider-detection/data/r4.2/` with raw CSVs.
- **5 random seeds**: [42, 43, 44, 45, 46]. All results report mean ± std.

## UBFS (20 dimensions, 7 categories)

| Category | Dims | Key Features |
|----------|------|-------------|
| TEMPORAL | 4 | activity_hour_mean, session_duration_norm, after_hours_ratio, weekend_activity_flag |
| FREQUENCY | 4 | primary_event_count, secondary_event_count, peripheral_event_count, event_rate_zscore |
| VOLUME | 3 | data_volume_norm, output_count_norm, volume_variability |
| SCOPE | 3 | resource_breadth, target_breadth, breadth_ratio |
| SEQUENCE | 3 | action_entropy, transition_novelty, repetition_score |
| DEVIATION | 2 | peer_distance, self_deviation |
| PRIVILEGE | 1 | privilege_deviation_index |

Schema definition: `src/features/ubfs_schema.py`

## Existing Datasets

| Dataset | Source | Size | Loader |
|---------|--------|------|--------|
| CMU-CERT | Local `../insider-detection/data/r4.2/` | ~1000 users, 500+ days | `cert_loader.py` |
| TRAIL | `PatronusAI/TRAIL` (gated) | 148 traces, 841 errors | `trail_loader.py` |
| TRACE | `PatronusAI/trace-dataset` (gated) | 517 trajectories | `trace_loader.py` |
| AgentHarm | `ai-safety-institute/AgentHarm` | 440 tasks | Not yet loaded via UBFS |

HuggingFace login required: `huggingface-cli login`

## Publication Target

With Exp 5-8 + Anthropic validation narrative:
- IEEE S&P Workshop on AI Security (strong fit, thesis already IEEE format)
- NeurIPS 2026 Workshop on Red Teaming (deadline ~August)
- USENIX Security (stretch, but cross-domain transfer + real-world validation is compelling)

The narrative (revised 7 Aug 2026): "We built cross-domain behavioural monitoring, identified a structural blind spot from synthetic profiling, and then tested it against real data — where it inverted. The blind spot that survives is architectural (per-entity monitors cannot see coordination), and the one that did not was an artifact of our own generator. Synthetic perturbation profiling cannot locate blind spots."

## Code Style

- Google Python Style Guide
- 80 char line length
- Type annotations on public APIs
- Commit format: `<type>(<scope>): <description>`
- All experiments: 5 seeds [42, 43, 44, 45, 46], report mean ± std
