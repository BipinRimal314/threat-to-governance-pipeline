# Threat-to-Governance Pipeline — CLAUDE.md

## Project Status: All Experiments Complete, Ready for CMU-CERT Integration

**Last updated:** 2026-02-21
**Current state:** Source code tested (50/50), data loaders implemented, 4 experiments run on agent datasets (TRAIL + TRACE), results generated. Ubuntu PC set up with Python 3.11 venv. Next: run full pipeline with HuggingFace datasets + CMU-CERT integration.

## What This Is

Research project repurposing MSc insider threat detection models (Isolation Forest 0.807, LSTM Autoencoder 0.774, Deep Clustering — on CMU-CERT) for AI agent behavioural monitoring. Maps detections to OWASP Top 10 for Agentic Applications. Includes governance analysis of normality assumptions.

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

## Target Environment

**Primary execution target: Ubuntu PC**
- GPU: NVIDIA RTX 4060 (8 GB VRAM) — use CUDA backend for PyTorch
- CPU: AMD Ryzen 5 5600X (6-core)
- RAM: 16 GB
- Storage: 1 TB SSD
- OS: Ubuntu (fresh boot)
- CMU-CERT dataset will be stored here (too large for Mac)

**Development machine (Mac):**
- MacBook M4 Pro — used for code development and initial experiments
- PyTorch MPS backend available but not needed for final runs
- No CMU-CERT data (storage constraint)

## Setup on Ubuntu PC

```bash
# 1. Install Python 3.11
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev

# 2. Clone and create virtualenv
git clone <repo-url> && cd threat-to-governance-pipeline
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA (RTX 4060)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install project dependencies
pip install -e ".[dev,notebooks]"

# 5. Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 6. Login to HuggingFace (TRAIL and TRACE are gated)
huggingface-cli login

# 7. Run tests
pytest tests/ -v  # expect 33/33

# 8. Run all experiments
python run_experiments.py --all

# 9. Generate figures
python generate_figures.py
```

## Completed Work

### Source Code (all tested, 50/50 passing)
- `src/features/ubfs_schema.py` — 20-dim UBFS with 7 categories, normalizer
- `src/features/cert_extractor.py` — CMU-CERT → UBFS vectors
- `src/features/agent_extractor.py` — OTel traces → UBFS vectors (handles TRAIL nested spans, ISO durations)
- `src/models/isolation_forest.py` — sklearn IsolationForest wrapper
- `src/models/lstm_autoencoder.py` — PyTorch LSTM-AE (encoder/decoder/latent)
- `src/models/deep_clustering.py` — PyTorch AE + KMeans
- `src/models/ensemble.py` — weighted/majority/cascade voting
- `src/data/trail_loader.py` — TRAIL from HuggingFace (handles 2 splits, nested child_spans, JSON labels)
- `src/data/trace_loader.py` — TRACE from HuggingFace (conversation JSON, taxonomy label codes)
- `src/data/cert_loader.py` — CMU-CERT loading via MSc pipeline
- `src/data/synthetic_generator.py` — OWASP ASI01-10 anomaly injection
- `src/evaluation/metrics.py` — AUC-ROC, AUC-PR, recall@FPR
- `src/evaluation/transfer_analysis.py` — cross-domain transfer framework
- `src/evaluation/owasp_mapper.py` — detection → OWASP category mapping
- `src/governance/assumption_audit.py` — normality assumption audit

### Experiment Results (with CMU-CERT)
**Experiment 1 — Within-Domain Baselines:**
| Domain | IF | LSTM | DC |
|--------|-------|-------|-------|
| TRAIL | 0.577 | 0.685 | **0.897** |
| TRACE | 0.501 | 0.521 | 0.496 |
| CMU-CERT (UBFS) | **0.731** | 0.723 | 0.697 |

- **Experiment 2 — Transfer (agent-only):** TRACE→TRAIL improves (+0.11 IF, +0.14 DC). TRAIL→TRACE degrades.
- **Experiment 2 --cert — CERT cross-domain:** CERT→TRAIL IF=0.711 (97% retention), TRACE→CERT DC=0.719 (+0.22 improvement)
- **Experiment 3 — OWASP:** ASI02=0.57-0.59 (blind spot), ASI05=0.94-0.97 (best)
- **Experiment 4 — Governance:** Report generated

### Artifacts
- `results/tables/experiment_{1-4}_*.json` — all experiment results
- `results/figures/fig{1-6}_*.png` — 6 publication-ready figures
- `results/governance_report.md` — governance assumption analysis
- `notebooks/01-05_*.ipynb` — 5 Jupyter notebooks
- `README.md` — paper-style research argument with results

## Still TODO

- [x] Set up Ubuntu environment with Python 3.11 venv
- [x] Implement data loaders (trail_loader, trace_loader, synthetic_generator, cert_loader)
- [x] Add data loader tests (15 tests, 50/50 total)
- [x] Login to HuggingFace and download TRAIL + TRACE gated datasets
- [x] Run experiments 1-4 end-to-end on Ubuntu PC (agent-only)
- [x] Integrate CMU-CERT via cert_loader (330,344 user-days, 20-dim UBFS, 1,364 insider-positive)
- [x] Run Experiment 1 with CMU-CERT baseline (IF=0.731, LSTM=0.723, DC=0.697)
- [x] Run Experiment 2 with CMU-CERT ↔ Agent transfer (CERT→TRAIL: 0.711, TRACE→CERT: 0.719)
- [x] Generate all 6 figures (fig1-6 in results/figures/)
- [ ] Verify notebooks run end-to-end in Jupyter

## Key Technical Details

### UBFS (20 dimensions, 7 categories)
| Category | Dims | Features |
|----------|------|----------|
| TEMPORAL | 4 | activity_hour_mean, session_duration_norm, after_hours_ratio, weekend_activity_flag |
| FREQUENCY | 4 | primary_event_count, secondary_event_count, peripheral_event_count, event_rate_zscore |
| VOLUME | 3 | data_volume_norm, output_count_norm, volume_variability |
| SCOPE | 3 | resource_breadth, target_breadth, breadth_ratio |
| SEQUENCE | 3 | action_entropy, transition_novelty, repetition_score |
| DEVIATION | 2 | peer_distance, self_deviation |
| PRIVILEGE | 1 | privilege_deviation_index |

### Models (all PyTorch except IF)
- **IsolationForest:** sklearn, 200 trees, auto contamination, threshold at 95th percentile
- **LSTM Autoencoder:** PyTorch, encoder [64,32] → latent 16 → decoder [32,64], MSE loss, 100 epochs, early stopping 15
- **Deep Clustering:** PyTorch, encoder [128,64] → latent 32 → 5 KMeans clusters, dual score (recon + distance)
- **Ensemble:** weighted voting (IF 0.3, LSTM 0.4, DC 0.3), threshold 0.7

### Key Findings (So Far)
- Tool Misuse (ASI02) is structurally undetectable — parameter changes without pattern changes
- Reward hacking (TRACE) is undetectable by structural anomaly detection (~0.50 AUC-ROC)
- Cross-domain transfer works when source domain has richer normal baseline
- Deep Clustering outperforms IF and LSTM-AE on TRAIL agent traces (0.897 vs 0.685 vs 0.577)

### Datasets
| Dataset | Source | Size | Role |
|---------|--------|------|------|
| CMU-CERT | Local (PC only) | ~1000 users, 500+ days | Insider threat baseline |
| TRAIL | `PatronusAI/TRAIL` (gated) | 148 traces, 841 errors | Agent anomaly detection |
| TRACE | `PatronusAI/trace-dataset` (gated) | 517 trajectories | Reward hack detection |
| AgentHarm | `ai-safety-institute/AgentHarm` | 440 tasks (3 configs) | Malicious agent behaviour |

### Data Loader Notes (Important)
- **TRAIL:** `trace` field is JSON string with nested `child_spans` — loader flattens recursively. `labels` is JSON with errors array. Two splits: `gaia` (117) + `swe_bench` (31).
- **TRACE:** `conversation` field is JSON string of `[{role, content}, ...]`. Label "0" = benign, anything else (e.g. "1.1.1") = reward hack. Some turns have `content: None`.
- **CMU-CERT:** Uses `cert_loader.py` which wraps MSc thesis data pipeline. Needs raw CSVs in `data/all_data/r1/`.

## Commands

```bash
# Run tests
source .venv/bin/activate
pytest tests/ -v  # 50 tests

# Run all experiments
python run_experiments.py --all

# Run single experiment
python run_experiments.py --experiment 1

# Generate figures
python generate_figures.py

# Download HuggingFace datasets (after login)
python -c "from datasets import load_dataset; d = load_dataset('PatronusAI/TRAIL'); print(d)"
```

## Code Style
- Google Python Style Guide
- 80 char line length
- Type annotations on public APIs
- Commit format: `<type>(<scope>): <description>`
