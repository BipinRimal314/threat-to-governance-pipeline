# Threat-to-Governance Pipeline

Insider threat detection models applied to AI agent behavioral monitoring. Experiments 1-4 complete. Experiments 5-8 extend the pipeline with new threat categories validated by the Feb 2026 Anthropic incidents.

## Development Workflow

```bash
source .venv/bin/activate

# Run tests (50 unit tests)
pytest tests/ -v

# Run completed experiments
python run_experiments.py --all --cert

# Run single new experiment
python run_experiments.py --experiment 5
python run_experiments.py --experiment 6
python run_experiments.py --experiment 7
python run_experiments.py --experiment 8

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
| 2 | Cross-domain transfer | **CERT→TRAIL: 0.711 (97% retention)** |
| 3 | OWASP detection matrix | **ASI02 Tool Misuse: 0.57-0.59 (blind spot)** |
| 4 | Governance assumptions | 6 embedded assumptions audited |

Results: `results/tables/experiment_{1-4}_*.json`
Figures: `results/figures/fig{1-6}_*.png`

## Common Mistakes — Read This First

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

---

# EXPERIMENT EXTENSION PLAN (5-8)

## Real-World Context: Anthropic Incidents (Feb 2026)

Two incidents validate findings from Experiments 2-3:

**Incident 1 — AI-Orchestrated Espionage (disclosed Nov 2025):**
Chinese state-sponsored group (GTG-1002) used Claude Code as autonomous hacking agent. AI performed 80-90% of espionage campaign. Used MCP tools (network scanners, password crackers) with correct sequencing and API calls. Jailbreak: decomposed malicious objectives into innocent subtasks ("defensive testing").

- **Validates ASI02 blind spot (Exp 3):** Tool misuse with legitimate tool invocation patterns. Exactly the structural signature our models can't detect at 0.57-0.59 AUC-ROC.
- **Validates boiling frog transfer:** Decomposition at prompt level mirrors gradual insider escalation. Same evasion strategy, different domain.
- **Validates CERT→TRAIL transfer (Exp 2):** Hybrid human-AI attack confirms structural equivalence between insider threat and agent threat patterns.

**Incident 2 — Model Distillation Attack (Feb 23, 2026):**
DeepSeek, Moonshot AI, MiniMax created ~24,000 fraudulent accounts, 16M+ exchanges to extract Claude capabilities.

- **New threat category:** Not insider threat, not agent misuse. API abuse / model extraction. Massive FREQUENCY and VOLUME anomaly that UBFS features should catch.

Blog post with analysis: `bipinrimal.com.np/blog/010-when-the-threat-model-is-the-model`

---

## Experiment 5: Model Extraction Detection (Distillation Attack Patterns)

**Goal:** Test whether UBFS detects high-volume model extraction attacks.

**Hypothesis:** Distillation attacks are highly detectable (>0.85 AUC-ROC) because they create massive structural anomalies in FREQUENCY, VOLUME, and SCOPE. This contrasts with ASI02 (Tool Misuse) which is undetectable. The comparison strengthens the argument: behavioral monitoring catches volume-based attacks but misses intent-based attacks.

**Implementation:**

1. Add `ASI_DISTILL` profile to `src/data/synthetic_generator.py`:

```python
"ASI_DISTILL": {
    # Model extraction: systematic capability probing
    # Massive frequency spike, broad scope, low entropy (methodical)
    "primary_event_count_mult": 15.0,      # 15x normal query rate
    "secondary_event_count_mult": 10.0,    # High LLM call volume
    "event_rate_add": 8.0,                 # Extreme rate anomaly
    "data_volume_mult": 20.0,              # Massive token extraction
    "resource_breadth_mult": 5.0,          # Probing across many capabilities
    "target_breadth_mult": 5.0,            # Many distinct endpoints
    "action_entropy_mult": 0.3,            # LOW entropy — systematic, not organic
    "repetition_add": 0.6,                 # Repetitive probing patterns
    "description": "Model Extraction / Distillation",
}
```

2. Add perturbation logic in `_perturb_trace()`:

```python
elif owasp_category == "ASI_DISTILL":
    # Model extraction: duplicate traces with systematic variation
    # Simulate high-frequency capability probing
    n_repeat = rng.randint(10, 20)
    original_spans = list(spans)
    for i in range(n_repeat):
        for span in original_spans:
            dup = copy.deepcopy(span)
            dup["span_id"] = f"distill_{rng.randint(0, 99999)}"
            # Slightly vary tool names to simulate breadth probing
            if rng.random() < 0.3:
                dup["name"] = f"probe_{rng.choice(['math', 'code', 'reason', 'creative', 'factual', 'analysis'])}"
            spans.append(dup)
```

3. Add `"ASI_DISTILL"` to the `categories` list in `experiment_3()` or create `experiment_5()` in `run_experiments.py` following the same pattern as Experiment 3 (synthetic injection → feature extraction → train on normal → evaluate per-category).

4. Run with all 3 models, 5 seeds. Compare ASI_DISTILL AUC-ROC against ASI02.

**Expected output:** `results/tables/experiment_5_distillation.json`

**Estimated time:** ~2 hours (code) + ~30 min (training on RTX 4060)

---

## Experiment 6: Jailbreak Decomposition Detection (Boiling Frog at Prompt Level)

**Goal:** Test whether multi-step task decomposition (splitting a malicious objective into innocent subtasks) reduces detection similarly to gradual insider escalation.

**Hypothesis:** Decomposed attacks drop detection below single-turn ASI01 (Goal Hijack, 0.663 AUC-ROC), potentially toward ASI02 territory (0.57-0.59). If confirmed, this proves the boiling frog evasion pattern transfers from insider threat to AI agent domain.

**Implementation:**

1. Create `src/data/agent_harm_loader.py`:

```python
"""Load AgentHarm dataset for decomposition analysis.

AgentHarm (UK AISI) contains 440 malicious agent tasks in 3 configs:
- direct: Single-turn malicious request
- decomposed: Multi-step broken into innocent subtasks
- augmented: Additional context/tools

We need direct vs decomposed comparison.
"""
from datasets import load_dataset

def load_agent_harm():
    """Load and parse AgentHarm dataset."""
    ds = load_dataset("ai-safety-institute/AgentHarm")
    # Parse direct vs decomposed configs
    # Convert to OTel-style trace format for UBFS extraction
    ...
```

2. Extend `src/features/agent_extractor.py` with multi-turn SEQUENCE features:
   - `entropy_drift`: how action entropy changes across conversation turns (gradual escalation = low drift, sudden hijack = high drift)
   - `bigram_novelty_slope`: rate of new action patterns appearing over turns
   - `turn_level_scope_expansion`: how resource breadth grows per turn

3. Create `experiment_6()` in `run_experiments.py`:
   - Load AgentHarm direct and decomposed configs
   - Extract UBFS features from both
   - Train models on benign traces
   - Compare detection of direct malicious (maps to ASI01) vs decomposed malicious
   - Report AUC-ROC for each, plus the delta

4. Cross-reference: compare decomposition detection drop to the insider threat boiling frog evasion rate (85% of gradual attacks missed). If the drop is proportionally similar, that's the cross-domain structural equivalence proof.

**Key insight to validate:** The Anthropic attackers decomposed espionage into "defensive testing" subtasks. AgentHarm's decomposed config does the same thing synthetically. If the detection drop matches the insider threat pattern, the UBFS bridge thesis is strengthened.

**Expected output:** `results/tables/experiment_6_decomposition.json`

**Estimated time:** ~6 hours (code: new loader + feature extension) + ~1 hour (training)

---

## Experiment 7: MCP Tool Abuse Profiling

**Goal:** Test cross-domain transfer on MCP-specific tool traces. The Anthropic attack used MCP tools. If CERT→MCP transfer works like CERT→TRAIL, the pipeline generalizes beyond the original datasets.

**Hypothesis:** CERT→MCP transfer retains >90% detection power (comparable to CERT→TRAIL's 97%). MCP-specific tool abuse profiles map to existing OWASP categories.

**New Datasets:**

| Dataset | Source | What It Provides |
|---------|--------|-----------------|
| MCPAgentBench | `arxiv:2512.24565` / HuggingFace | Real-world MCP tool traces with sandbox evaluation, distractor tools |
| ATBench | HuggingFace | Multi-turn tool-centric safety scenarios, fine-grained risk taxonomy |

**Implementation:**

1. Create `src/data/mcp_bench_loader.py`:

```python
"""Load MCPAgentBench for MCP-specific tool abuse detection.

MCPAgentBench provides traces from agents using real MCP tool
definitions with distractor tools in sandbox environments.
Map to OTel format for UBFS extraction.
"""
```

2. Create `src/data/atbench_loader.py`:

```python
"""Load ATBench (Agent Trajectory Safety Benchmark).

ATBench provides multi-turn, tool-centric safety scenarios
with binary safety labels and fine-grained risk diagnosis.
Map trajectories to OTel format for UBFS extraction.
"""
```

3. Create `experiment_7()` in `run_experiments.py`:
   - **Phase A — Within-domain baselines:** Train/test on MCPAgentBench and ATBench separately
   - **Phase B — Cross-domain transfer:** CERT→MCP, CERT→ATBench, TRAIL→MCP, TRAIL→ATBench
   - **Phase C — OWASP mapping:** Run synthetic OWASP injection on MCP traces, compare detection matrix to Exp 3 results
   - Report transfer retention percentages alongside Exp 2 results

4. Key comparison: CERT→TRAIL retention was 97% (0.731→0.711). Does CERT→MCP show similar or different retention? If similar, the UBFS bridge generalizes. If different, analyze which UBFS dimensions explain the gap.

**Expected output:** `results/tables/experiment_7_mcp_transfer.json`

**Estimated time:** ~8 hours (code: 2 new loaders + experiment) + ~2 hours (training)

**Note:** Check HuggingFace availability of MCPAgentBench and ATBench. If either is gated, request access first. If unavailable, use ToolEmu (GitHub: `ryoungj/ToolEmu`, 36 high-stakes tools, 144 test cases) as fallback.

---

## Experiment 8: Hybrid Detection (Behavioral + Semantic Features)

**Goal:** Close the ASI02 blind spot. ASI02 is undetectable at 0.57-0.59 because behavioral features can't distinguish legitimate from malicious tool use. Adding semantic features (what tool parameters mean, not just what tools were called) should push detection above 0.70.

**Hypothesis:** UBFS-28 (20 behavioral + 8 semantic) achieves >0.70 AUC-ROC on ASI02, compared to 0.57-0.59 with UBFS-20.

**UBFS Extension (20 → 28 dimensions):**

Add a new `SEMANTIC` category to `src/features/ubfs_schema.py`:

```python
FeatureCategory.SEMANTIC: [
    (
        "param_embedding_distance",
        None,                          # No CERT equivalent
        "tool_param_cosine_distance",
        "Cosine distance of tool parameters from known-good embeddings",
    ),
    (
        "intent_consistency_score",
        None,
        "stated_goal_action_similarity",
        "Embedding similarity between stated goal and actual tool actions",
    ),
    (
        "escalation_gradient",
        "cumulative_access_change",
        "cumulative_privilege_change",
        "Cumulative privilege change across session/trace",
    ),
    (
        "param_anomaly_score",
        None,
        "tool_param_deviation",
        "How unusual the tool parameters are vs. normal usage",
    ),
    (
        "context_tool_alignment",
        None,
        "context_action_coherence",
        "Whether tool calls are coherent with conversation context",
    ),
    (
        "goal_drift_score",
        None,
        "goal_embedding_drift",
        "How much the inferred goal shifts across turns",
    ),
    (
        "semantic_novelty",
        None,
        "semantic_action_novelty",
        "Novelty of actions in semantic (not structural) space",
    ),
    (
        "instruction_compliance",
        None,
        "instruction_action_alignment",
        "Whether actions align with explicit instructions",
    ),
]
```

**Implementation:**

1. Create `src/features/semantic_extractor.py`:

```python
"""Semantic feature extraction using sentence-transformers.

Extracts meaning-aware features from tool call parameters,
conversation context, and stated goals. These complement
the structural UBFS features by capturing intent signals
that behavioral monitoring misses.

Requires: sentence-transformers (~110M params, fits in 8GB VRAM)
Install: pip install sentence-transformers
"""
from sentence_transformers import SentenceTransformer

class SemanticFeatureExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # Precompute embeddings for known-good tool parameter patterns
        ...
```

2. Extend `src/features/agent_extractor.py`:
   - Add `extract_semantic_features()` method
   - Combine with existing UBFS extraction to produce 28-dim vectors
   - Fallback: if semantic extractor unavailable, fill 8 semantic dims with 0.0 (neutral under z-score)

3. Create `experiment_8()` in `run_experiments.py`:
   - **Phase A — UBFS-20 vs UBFS-28 on ASI02:** Re-run Experiment 3 OWASP injection with both feature sets. Direct comparison: does adding semantic features improve ASI02 detection?
   - **Phase B — Full OWASP matrix with UBFS-28:** Compare all OWASP categories across both feature sets. Semantic features should help ASI02 most, with minimal impact on already-detectable categories (ASI05, ASI09).
   - **Phase C — Cross-domain transfer with UBFS-28:** Do semantic features help or hurt transfer? CERT has no semantic features (filled with 0.0), so CERT→TRAIL transfer might degrade. Test this explicitly.

4. Report:
   - ASI02 AUC-ROC improvement (primary metric)
   - Full OWASP matrix comparison table (UBFS-20 vs UBFS-28)
   - Transfer retention comparison

**VRAM Management:** The sentence-transformer model (~110M params) uses ~500MB VRAM. Running it alongside LSTM-AE or DC on RTX 4060 (8GB) is tight. Strategy:
- Extract all semantic features FIRST, save to disk
- Then run anomaly detection models on the combined 28-dim vectors (no embedding model in memory during training)
- If VRAM issues persist: rent Vast.ai RTX 4090 spot (~$0.30/hr) for a few hours

**Expected output:** `results/tables/experiment_8_hybrid.json`

**Estimated time:** ~12 hours (code: semantic extractor + UBFS extension + experiment) + ~3 hours (training)

**Install dependency:**
```bash
pip install sentence-transformers
```

---

## Code Changes Summary

```
src/
  data/
    synthetic_generator.py     ← ADD: ASI_DISTILL profile + perturbation logic
    agent_harm_loader.py       ← NEW: AgentHarm dataset loader (direct vs decomposed)
    mcp_bench_loader.py        ← NEW: MCPAgentBench loader
    atbench_loader.py          ← NEW: ATBench loader
  features/
    ubfs_schema.py             ← EXTEND: UBFS-28 (add SEMANTIC category, 8 dims)
    agent_extractor.py         ← EXTEND: multi-turn SEQUENCE features + semantic integration
    semantic_extractor.py      ← NEW: sentence-transformer embedding features
  evaluation/
    owasp_mapper.py            ← ADD: ASI_DISTILL category support
run_experiments.py             ← ADD: experiment_5(), experiment_6(), experiment_7(), experiment_8()
configs/model_configs.yaml     ← ADD: semantic extractor config, UBFS-28 dims
```

## Execution Order

Run experiments in this order. Each builds on the previous.

| Order | Experiment | Depends On | Time (code + run) |
|-------|-----------|------------|-------------------|
| 1st | **Exp 5** (Distillation) | Only `synthetic_generator.py` change | ~2.5 hrs |
| 2nd | **Exp 6** (Decomposition) | New loader + feature extension | ~7 hrs |
| 3rd | **Exp 7** (MCP Transfer) | 2 new loaders + transfer framework | ~10 hrs |
| 4th | **Exp 8** (Hybrid) | UBFS extension + semantic extractor | ~15 hrs |

## Publication Target

With Exp 5-8 + Anthropic validation narrative:
- IEEE S&P Workshop on AI Security (strong fit, thesis already IEEE format)
- NeurIPS 2026 Workshop on Red Teaming (deadline ~August)
- USENIX Security (stretch, but cross-domain transfer + real-world validation is compelling)

The narrative: "We built cross-domain behavioral monitoring. We identified a structural blind spot (ASI02). The first documented AI-orchestrated espionage campaign exploited that exact blind spot. Here's the data, and here's what we built to close the gap."

## Code Style

- Google Python Style Guide
- 80 char line length
- Type annotations on public APIs
- Commit format: `<type>(<scope>): <description>`
- All experiments: 5 seeds [42, 43, 44, 45, 46], report mean ± std
