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
DeepSeek, Moonshot AI, MiniMax created ~24,000 fraudulent accounts, 16M+ exchanges to extract Claude capabilities via "hydra cluster" architectures (20,000+ accounts, no single point of failure).

Each lab ran a different extraction strategy:
- **DeepSeek** (150K+ exchanges): Chain-of-thought elicitation — prompted Claude to articulate step-by-step reasoning for completed responses. Generates reasoning training data. Also tested censorship-alternative queries.
- **Moonshot AI** (3.4M+ exchanges): Multi-capability targeting — agentic reasoning, tool use, coding, data analysis, computer-use agents, computer vision.
- **MiniMax** (13M+ exchanges): Focused extraction — coding and tool orchestration functions.

Detection: Anthropic used behavioral fingerprinting, chain-of-thought elicitation classifiers, IP/metadata correlation, and coordinated activity tracking across account networks.

- **New threat category:** Not insider threat, not agent misuse. API abuse / model extraction. Three distinct behavioral signatures (SEQUENCE, SCOPE, FREQUENCY). Plus a coordination problem: hydra clusters distribute traffic so per-account volume is normal. Anomaly is only visible in aggregate.

Blog post with analysis: `bipinrimal.com.np/blog/010-when-the-threat-model-is-the-model`

---

## Experiment 5: Model Extraction Detection (Distillation Attack Patterns)

**Goal:** Test whether UBFS detects distillation attacks, and specifically whether the "hydra cluster" distribution strategy defeats single-entity behavioral monitoring.

**Key insight from Anthropic disclosure (Feb 23, 2026):** The distillation attack was NOT one pattern. Three labs ran three different extraction strategies, and the hydra cluster architecture (20,000+ accounts) distributed traffic so per-account behavior might look normal. This creates a more nuanced detection problem than "high volume = easy to catch."

**Hypotheses:**
1. **Single-entity high-volume distillation** is highly detectable (>0.85 AUC-ROC) via FREQUENCY/VOLUME anomalies
2. **Strategy-specific signatures** have different detection profiles: CoT elicitation (SEQUENCE), multi-capability probing (SCOPE), focused extraction (FREQUENCY)
3. **Hydra-distributed distillation** per-account is near-undetectable (~0.55-0.65 AUC-ROC) because per-account volume falls within normal bounds
4. If hypothesis 3 confirms: behavioral monitoring has TWO structural blind spots, not one. Intent-based (ASI02) and coordination-based (HYDRA).

**Implementation:**

1. Add FOUR distillation sub-profiles to `src/data/synthetic_generator.py`:

```python
# --- DeepSeek pattern: Chain-of-thought elicitation ---
"ASI_DISTILL_COT": {
    # Systematic reasoning extraction via step-by-step elicitation
    # Distinctive SEQUENCE signature: low entropy, high repetition, specific prompt patterns
    "primary_event_count_mult": 3.0,       # Moderate volume (150K over time)
    "action_entropy_mult": 0.2,            # VERY low — methodical CoT prompting
    "repetition_add": 0.8,                 # Highly repetitive (same elicitation pattern)
    "transition_novelty_mult": 0.3,        # Low novelty — same prompt structure repeated
    "resource_breadth_mult": 2.0,          # Moderate breadth (reasoning across topics)
    "data_volume_mult": 5.0,              # High output extraction (long CoT responses)
    "description": "Chain-of-Thought Elicitation (DeepSeek pattern)",
},

# --- Moonshot pattern: Multi-capability targeting ---
"ASI_DISTILL_BROAD": {
    # Broad capability probing across reasoning, tool use, coding, CV
    # Distinctive SCOPE signature: extreme breadth, moderate depth per capability
    "primary_event_count_mult": 5.0,       # 3.4M exchanges
    "resource_breadth_mult": 8.0,          # EXTREME breadth — many capability domains
    "target_breadth_mult": 6.0,            # Many distinct endpoints/tools
    "action_entropy_mult": 0.7,            # Moderate entropy — varied but structured
    "data_volume_mult": 8.0,              # High extraction volume
    "event_rate_add": 3.0,                # Elevated but not extreme rate
    "description": "Multi-Capability Probing (Moonshot pattern)",
},

# --- MiniMax pattern: Focused extraction ---
"ASI_DISTILL_FOCUSED": {
    # Narrow, deep extraction on coding + tool orchestration
    # Distinctive FREQUENCY signature: massive volume on narrow target
    "primary_event_count_mult": 15.0,      # 13M exchanges — highest volume
    "secondary_event_count_mult": 10.0,    # Heavy tool call volume
    "event_rate_add": 8.0,                 # Extreme rate
    "data_volume_mult": 20.0,             # Massive extraction
    "resource_breadth_mult": 1.5,          # NARROW scope — coding/tools only
    "target_breadth_mult": 2.0,            # Few distinct targets, hit hard
    "action_entropy_mult": 0.4,            # Low entropy — repetitive deep probing
    "repetition_add": 0.5,                # High repetition
    "description": "Focused Extraction (MiniMax pattern)",
},

# --- Hydra cluster: Distributed across N accounts ---
"ASI_DISTILL_HYDRA": {
    # Per-account behavior after distributing across ~20K accounts
    # Each account's volume is NORMAL. Anomaly is in coordination, not individual behavior.
    "primary_event_count_mult": 1.05,      # ~5% above normal (barely detectable)
    "secondary_event_count_mult": 1.1,     # Slight elevation
    "event_rate_add": 0.2,                 # Within normal variance
    "data_volume_mult": 1.1,              # Normal-ish
    "resource_breadth_mult": 1.3,          # Slightly broader than typical user
    "action_entropy_mult": 0.8,            # Slightly more systematic
    "repetition_add": 0.1,                # Barely elevated
    "description": "Hydra Cluster Per-Account (distributed distillation)",
},
```

2. Add perturbation logic in `_perturb_trace()`:

```python
elif owasp_category.startswith("ASI_DISTILL"):
    if owasp_category == "ASI_DISTILL_COT":
        # CoT elicitation: repeat similar prompt-response patterns
        # Simulate systematic "explain your reasoning" queries
        n_repeat = rng.randint(5, 10)
        original_spans = list(spans)
        for i in range(n_repeat):
            for span in original_spans:
                dup = copy.deepcopy(span)
                dup["span_id"] = f"cot_{rng.randint(0, 99999)}"
                # Add CoT-style tool names
                dup["name"] = f"elicit_{rng.choice(['reasoning', 'step_by_step', 'chain_of_thought', 'explain', 'elaborate'])}"
                spans.append(dup)

    elif owasp_category == "ASI_DISTILL_BROAD":
        # Multi-capability: probe across many different tool types
        n_repeat = rng.randint(3, 8)
        capability_domains = ['math', 'code', 'reason', 'creative', 'factual',
                              'analysis', 'vision', 'tool_use', 'planning', 'search']
        original_spans = list(spans)
        for i in range(n_repeat):
            for span in original_spans:
                dup = copy.deepcopy(span)
                dup["span_id"] = f"broad_{rng.randint(0, 99999)}"
                dup["name"] = f"probe_{rng.choice(capability_domains)}"
                spans.append(dup)

    elif owasp_category == "ASI_DISTILL_FOCUSED":
        # Focused: massive repetition on narrow target
        n_repeat = rng.randint(15, 25)
        original_spans = list(spans)
        for i in range(n_repeat):
            for span in original_spans:
                dup = copy.deepcopy(span)
                dup["span_id"] = f"focused_{rng.randint(0, 99999)}"
                # Narrow: only coding/tool targets
                if rng.random() < 0.3:
                    dup["name"] = f"extract_{rng.choice(['code_gen', 'code_review', 'tool_call', 'tool_chain'])}"
                spans.append(dup)

    elif owasp_category == "ASI_DISTILL_HYDRA":
        # Hydra: minimal perturbation — each account looks nearly normal
        # Only very slight parameter shifts applied via profile multipliers
        # No span duplication — the attack is invisible at individual level
        pass
```

3. Create `experiment_5()` in `run_experiments.py`:
   - Run all 4 distillation sub-profiles through the Exp 3 pipeline (synthetic injection → feature extraction → train on normal → evaluate)
   - Report AUC-ROC for each sub-profile separately
   - Compare: COT vs BROAD vs FOCUSED vs HYDRA vs ASI02 (tool misuse)
   - Key result table: a spectrum from "easily detectable" (FOCUSED) to "structurally undetectable" (HYDRA, ASI02)

4. Run with all 3 models, 5 seeds.

**Expected output:** `results/tables/experiment_5_distillation.json` with per-sub-profile AUC-ROC

**Expected results pattern:**
| Sub-profile | Expected AUC-ROC | Why |
|---|---|---|
| ASI_DISTILL_FOCUSED (MiniMax) | >0.90 | Extreme FREQUENCY spike |
| ASI_DISTILL_BROAD (Moonshot) | ~0.80-0.85 | Clear SCOPE anomaly |
| ASI_DISTILL_COT (DeepSeek) | ~0.70-0.80 | SEQUENCE anomaly (testable) |
| ASI_DISTILL_HYDRA | ~0.55-0.65 | Per-account normal — blind spot |
| ASI02 Tool Misuse (baseline) | 0.57-0.59 | Known blind spot |

If HYDRA lands near ASI02: two distinct blind spots in the behavioral monitoring paradigm. Different failure modes (intent vs coordination), same ceiling.

**Estimated time:** ~4 hours (code: 4 profiles + perturbation logic + experiment) + ~1 hour (training on RTX 4060)

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
| 1st | **Exp 5** (Distillation) | 4 sub-profiles + perturbation logic | ~5 hrs |
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
