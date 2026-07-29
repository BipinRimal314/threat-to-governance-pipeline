---
title: Development Plan — Threat-to-Governance Pipeline
branch: develop/hydra-coordination
focus: Multi-account / multi-agent coordination detection (HYDRA)
status: draft plan — implement next
updated: 2026-07-09
related:
  - README.md
  - EXPERIMENTS_9_11.md
  - paper/main.tex
---

# DEVELOPMENT.md — How We Develop This Further

**Project:** [threat-to-governance-pipeline](https://github.com/BipinRimal314/threat-to-governance-pipeline)  
**Branch:** `develop/hydra-coordination`  
**Role in the portfolio:** Flagship **AI safety research** artifact (Lane 1 — fellowships, grants, AF/LW, safety orgs)

This document is the operating plan for the next development cycle. It is deliberately narrow: we do **not** rebuild the whole thesis. We extend the one finding that survived the synthetic→real sanity check.

---

## 1. North star

> Single-entity behavioral monitoring fails on **coordinated multi-account / multi-agent (HYDRA-style) attacks**. Build the smallest credible system that detects coordination patterns the current pipeline cannot see, measure it honestly, and publish the result.

### What we already proved (do not re-litigate)

| Finding | Status | Implication |
|---|---|---|
| CERT → TRAIL transfer (~97% detection power retained) | Solid | Keep as baseline story |
| ASI02 Tool Misuse ~0.57 on *synthetic* data | **Mostly artifact** | Do not lead with “paradigm blind spot” for ASI02 |
| ASI02 on real ATBench trajectories ~0.86–0.94 | Corrected | Document in README/paper; Experiments 12–13 |
| HYDRA / coordination ~0.538 AUC-ROC | **Still real** | This is the development target |
| Boiling-frog / decomposition evasion | Real, smaller on AgentHarm | Secondary; optional Phase 3 |

### What “done” looks like for this cycle

1. A **coordination detector** (or multi-entity scorer) that beats chance on a defined HYDRA benchmark.
2. A **reproducible experiment** in `run_experiments.py` (or a new `run_hydra_experiments.py`) with fixed seeds and tables under `results/`.
3. An updated **README section** + short **write-up** suitable for LessWrong / Alignment Forum / fellowship work sample appendix.
4. Tests that lock the schema and the evaluation protocol so results don’t silently drift.

---

## 2. Design thesis

### Why single-entity monitoring fails

Current stack:

```
trace / user-day
  → feature extractor (UBFS 20-dim or semantic extension)
  → anomaly model (IF / LSTM / Deep Clustering)
  → per-entity score
```

A HYDRA attack **splits** malicious behavior across many entities so each vector looks normal. No amount of improving single-vector AUC fixes that. We need a second layer:

```
set of entity traces over a time window
  → aggregate / graph / correlation features
  → coordination score
  → optional: which accounts participate
```

### What we are *not* building this cycle

- Full production SOC product
- Real-time streaming multi-tenant SaaS
- Mechanistic interpretability of the agent model
- Another Isolation Forest bake-off on CMU-CERT alone

---

## 3. Architecture (proposed)

```
src/
  features/
    ubfs_schema.py              # existing — keep stable
    agent_extractor.py          # existing
    cert_extractor.py           # existing
    semantic_extractor.py       # existing
    coordination_schema.py      # NEW — multi-entity feature space
    coordination_extractor.py   # NEW — windowed aggregate features
  models/
    isolation_forest.py         # existing
    coordination_detector.py    # NEW — detector over coordination vectors
  data/
    synthetic_generator.py      # existing — extend carefully
    hydra_generator.py          # NEW — multi-account attack simulator
    multi_agent_loader.py       # NEW — load multi-agent / multi-session traces
  evaluation/
    owasp_mapper.py             # existing
    hydra_eval.py               # NEW — metrics for coordination detection
```

### Coordination feature candidates (v0.1)

Start small (8–15 dims), not a kitchen sink. Examples:

| Feature family | Examples | Why |
|---|---|---|
| **Rate coincidence** | burst correlation across accounts in same window | HYDRA clusters fire together |
| **Behavioral similarity** | pairwise cosine of UBFS vectors within window | cloned or templated agents |
| **Goal / prompt overlap** | embedding similarity of goals or first messages | capability distillation clusters |
| **Tool-set Jaccard** | shared tool vocabulary across accounts | same attack playbook |
| **Temporal graph** | density of near-simultaneous starts | campaign structure |
| **Volume partitioning** | entropy of volume across accounts (too flat = split) | classic HYDRA signature |
| **Cross-account sequence** | shared tool bigrams across entities | coordinated playbooks |

**Hard rule from Experiments 12–13:** do not generate “coordination attacks” by perturbing a single dimension of one entity and calling it multi-account. The generator must **actually split** behavior across N entities so that *per-entity* scores stay near-normal while *aggregate* scores should rise.

---

## 4. Phased plan

### Phase 0 — Housekeeping (2–3 days)

Goal: make the research repo safe to extend without rotting the old results.

- [ ] Pin README “current best statement” to the **post–Exp 12–13** conclusion (coordination is the real gap; ASI02 was mostly synthetic artifact).
- [ ] Add a short `CHANGELOG.md` or README “Corrections” section so external readers don’t cite the outdated ASI02 claim.
- [ ] Verify `pytest` still passes; fix any bitrot on loaders.
- [ ] Ensure experiment tables under `results/tables/` are documented (which experiment number produced which file).
- [ ] Confirm Docker path still runs the main experiment suite (even if slow).

**Exit criteria:** Clean main narrative + green tests + one-line “how to reproduce Exp 1–3.”

### Phase 1 — HYDRA benchmark (1–2 weeks)

Goal: a **definition of success** that is not circular.

1. **Threat model (write it down first)**
   - N accounts (start N ∈ {10, 50, 200, 1000})
   - Each account’s individual behavior is drawn from “normal” TRAIL-like or synthetic normal distributions
   - Attack total work is partitioned so no single account exceeds per-entity anomaly thresholds
   - Variants: simultaneous burst, staggered, capability-distillation style (many accounts, similar goals)

2. **Generator: `src/data/hydra_generator.py`**
   - Inputs: N, window_size, attack_budget, seed
   - Outputs: multi-entity dataset + labels (coordinated campaign vs independent benign multi-user traffic)
   - Benign control: N independent normal users/agents with natural coincidences

3. **Baseline that should fail**
   - Score each entity with existing Isolation Forest (UBFS)
   - Aggregate with max/mean/topk — document that this still fails (near 0.54–0.60 territory is fine; we need the *baseline*)

4. **Metrics**
   - AUC-ROC / PR-AUC at **campaign** level (detect that a window contains a coordinated attack)
   - Optional: attribution F1 (which accounts are in the attack set)
   - Report per-entity false positive rate so we don’t “win” by flagging everyone

**Exit criteria:** Benchmark + failing single-entity baseline committed with seeds and plots.

### Phase 2 — Coordination detector v0.1 (2–3 weeks)

Goal: beat the baseline on the HYDRA benchmark with a simple, explainable model.

1. Implement `coordination_schema.py` + `coordination_extractor.py`
2. Implement `coordination_detector.py` (start with Isolation Forest or logistic regression on coordination features — prefer **interpretable** over deep)
3. Ablations: which feature families move AUC?
4. Stress tests:
   - Benign correlated traffic (same timezone, same org tools)
   - Different N
   - Staggered vs simultaneous

**Success bar (honest, not heroic):**

| Metric | Target |
|---|---|
| Campaign-level AUC-ROC on synthetic HYDRA | **≥ 0.75** (stretch ≥ 0.85) |
| Per-entity FPR on pure benign | **≤ baseline** (don’t regress) |
| Ablation: remove volume-partition features | AUC should drop (proves we’re not magic) |

If we cannot beat 0.75 with simple features, document *why* (e.g., benign correlation is indistinguishable) — that negative result is still publishable.

**Exit criteria:** Experiment script + tables + figure + tests for schema/extractor.

### Phase 3 — Real / semi-real data (2–4 weeks)

Goal: avoid another synthetic-only story (lesson of Exp 12–13).

Options, pick what is available:

| Data source | Use |
|---|---|
| Multi-session Claude Code / multi-agent traces you already have | Benign multi-entity baselines |
| ATBench / AgentHarm extended to multi-agent if possible | Malicious patterns |
| Public multi-agent benchmarks (TRAIL multi-agent, LangGraph traces) | Transfer test |
| Carefully labeled public “bot farm” style logs if any open data exists | External validity |

Protocol:

1. Train or calibrate on synthetic HYDRA
2. Test on real multi-agent *benign* (FPR check)
3. Test on any real multi-agent *malicious* available (even small N)
4. Report Spearman-style rank correlation between synthetic and real difficulty **if** multiple attack types exist

**Exit criteria:** One “real-ish” evaluation section that either validates or constrains the synthetic win.

### Phase 4 — Publish & package (1 week, parallelizable)

- [ ] LessWrong / AF post using the correction framing (draft already exists in personal wiki)
- [ ] Update Zenodo preprint or add a short technical report PDF under `paper/`
- [ ] 10–15 min recorded walkthrough (slides already exist in `research-slides`)
- [ ] Pin this repo; lead fellowship applications with HYDRA v0.1 results
- [ ] Optional: tiny CLI `python -m src.hydra_demo` that scores a folder of multi-account traces

---

## 5. Experiment numbering (proposed)

| ID | Name | Purpose |
|---|---|---|
| Exp 12–13 | Already done | Synthetic vs real ASI02 correction |
| **Exp 14** | HYDRA single-entity baseline | Show max/mean aggregation fails |
| **Exp 15** | Coordination features + detector | Main positive result |
| **Exp 16** | Ablations | Which features matter |
| **Exp 17** | Real multi-agent transfer / FPR | External validity |

Record each in `results/tables/exp14_*.csv` etc. and document in README.

---

## 6. Engineering standards

- **Reproducibility:** fixed seeds, pinned dependency versions in `pyproject.toml`, Docker path remains the gold path for full runs.
- **Honesty:** if a metric is campaign-level, never report it as if it were per-trace OWASP ASI02.
- **Tests:** unit tests for feature shape, generator invariants (e.g., “no single account exceeds budget B”), and eval wiring — not only end-to-end AUC.
- **No silent renorming of old claims:** README leads with corrected narrative.

### Generator invariants (must test)

```
1. For every attack campaign, max per-entity anomaly score under baseline model ≤ threshold T
   (or mean score within ε of benign mean)
2. Sum of attack work across accounts ≈ total attack budget
3. Benign multi-user controls have natural correlation but no shared attack goal label
4. Seeds fully determine dataset
```

---

## 7. Relationship to AI Trace Auditor

These two projects are **siblings**, not competitors:

| | This repo | `ai-trace-auditor` |
|---|---|---|
| Question | Can we *detect* coordinated agent misuse? | Can we *document* regulatory evidence from code/traces? |
| Audience | Safety researchers, fellowships | Compliance/product engineers, GRC buyers |
| Output | Papers, AUC tables, threat model | CLI reports, Annex IV, CI gates |

Optional later bridge (Phase 5+, not now): export coordination scores as **trace-level governance signals** that Trace Auditor can flag as “monitoring gap: multi-agent correlation not implemented.” That is a product story, not a research blocker.

---

## 8. Time budget (recommended)

| Allocation | Focus |
|---|---|
| **70%** | This repo — HYDRA Phases 0–2 |
| **30%** | `ai-trace-auditor` — hiring/demo edge (see that repo’s DEVELOPMENT.md) |

If a fellowship deadline hits, freeze at Phase 2 with clean tables and publish. Do not leave mid-refactor.

---

## 9. First concrete PR sequence

1. **docs:** README correction block + this `DEVELOPMENT.md` (this branch)
2. **feat:** `hydra_generator.py` + unit tests for invariants
3. **exp:** Exp 14 baseline script + results tables
4. **feat:** coordination schema + extractor + tests
5. **feat:** coordination detector + Exp 15–16
6. **docs:** results write-up + figure for AF post appendix

---

## 10. Open questions (resolve during Phase 1)

1. What window length (hours/days/tokens) matches “campaign” for agent traces?
2. Is the unit of analysis **account**, **session**, or **agent-id** in multi-agent frameworks?
3. How do we label “benign correlation” (same company, same CI bot fleet) so we don’t ship a detector that is just “many similar workers”?
4. Can we get even a tiny real multi-agent attack dataset, or is public evaluation stuck on synthetic + benign real?

---

## Bottom line

**Develop the coordination layer.** Everything else in this repo is already strong enough for a portfolio. The edge comes from turning the unsolved HYDRA finding into a measured v0.1 detector — with the same epistemic honesty (Exp 12–13 style) that made the original work credible.
