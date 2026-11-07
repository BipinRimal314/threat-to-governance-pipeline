# Agent-trace anomaly detection and behavioural monitoring, 2025-2026 landscape

Overlap scale vs our claim (agent-monitor evaluation validity: behaviour-only 20-dim schema, IForest/LSTM-AE/Deep Clustering ported from CERT to agent traces, ASI/ATLAS mapping): none / adjacent / close / same.
Verification note: abstract pages were fetched for most items; full texts were mostly NOT read. Numbers are from abstracts unless flagged "secondary".

## 1. Agent-trace anomaly/monitoring benchmarks and detectors, 2025-2026

### Takeaway
The 2025-26 field splits into (a) LLM-judged or fine-tuned-LLM trace auditors (TRAIL, TrajAD, OpenClawBench, ATBench, TraceDance, 2610.07256), (b) one structural/behavioural rule-induction detector (TraceAegis), and (c) one head-to-head feature study (2605.11053) showing metadata-only features plateau at ~0.64 AUROC and that random splits inflate AUROC by up to 26 points. Item (c) is the closest to our paper in both substance and in the validity angle, and is the main threat to a behaviour-only claim.

### Cited Findings
- TRAIL (Patronus AI): 148 human-annotated traces (1,987 OTel spans, 575 with at least one error) from GAIA and SWE-Bench; best model Gemini-2.5-Pro scores 11%. Task is LLM-driven error localisation and categorisation, not statistical anomaly detection. Overlap: adjacent. — [arXiv 2505.08638](https://arxiv.org/abs/2505.08638) (counts via [search summary of v3](https://arxiv.org/html/2505.08638v3); the 11% is in the abstract)
- TRACE (Patronus AI, ICML 2026 proceedings): "Testing Reward Anomalies in Code Environments"; 54-category reward-exploit taxonomy, 517 human-verified trajectories (reported as 249 benign / 268 hacked, secondary summary), framed as contrastive anomaly detection; models do better in contrastive than isolated classification. Dataset on HF as PatronusAI/trace-dataset. Overlap: adjacent (our "TRACE" dataset is presumably this one; confirm). — [arXiv 2601.20103](https://arxiv.org/abs/2601.20103); [PMLR v306](https://proceedings.mlr.press/v306/deshpande26a.html)
- ATBench: 1,000 trajectories (503 safe / 497 unsafe), avg 9.01 turns, 3.95k tokens, 1,954 tool invocations from 2,084-tool pools; indexed by risk source, failure mode, real-world harm; delayed-trigger protocol; GPT-5.4 gets only 76.7% F1 on binary safety classification (v2 figure via search summary). Abstract does not say whether trajectories are synthetic; construction uses rule + LLM filtering and a human audit, so likely generated. Extensions ATBench-Claw / ATBench-CodeX. Overlap: adjacent. — [arXiv 2604.02022](https://arxiv.org/abs/2604.02022); extensions [arXiv 2604.14858](https://arxiv.org/abs/2604.14858)
- OpenClawBench: 31,264 silver-labelled OpenClaw/BFCL trajectories from 6 source models; 31,135 pass the oracle, 2,904 (about 9.3%) of those are process-anomalous under FullTax labels (the "Outcome-Process Gap"); LoRA-tuned Gemma 3 12B reaches binary F1 0.729 on a cleaner-label held-out split. Silver labels: a validity caveat in itself. Overlap: adjacent. — [arXiv 2605.29253](https://arxiv.org/abs/2605.29253)
- TraceAegis: hierarchical, provenance-based detector built from tool-invocation traces (input, output, order) at gateway level; TraceAegis-Bench has healthcare and procurement scenarios, 1,300 benign + 300 abnormal each (3,200 traces). Abstract reports no F1; a third party (2605.11053) characterises it as F1 > 0.94 but needing labelled per-scenario templates (unverified). Closest structural/behavioural (non-LLM-judge) prior art. Overlap: close (method), none (validity). — [arXiv 2510.11203](https://arxiv.org/abs/2510.11203)
- TrajAD: step-level verifier fine-tuned on TrajBench (perturb-and-complete synthetic anomalies); zero-shot LLMs struggle; outputs verdict + first erroneous step. Abstract has no numbers; secondary source claims +11.38 Macro-F1 over Qwen3-8B (unverified). Synthetic perturbation labels are a validity concern. Overlap: adjacent. — [arXiv 2602.06443](https://arxiv.org/abs/2602.06443)
- TraceDance (ByteDance/UIC authors incl. Philip S. Yu): builds behaviour benchmarks from 252,557 real deployment sessions into 107 benchmarks / 4,125 instances; 84% of sampled instances human-confirmed; nine frontier LLMs average 26.7% pass rate; LLM-judged next-turn grading. A benchmark builder, not a detector. Overlap: adjacent. — [arXiv 2609.33295](https://arxiv.org/abs/2609.33295)
- 2610.07256 "Efficient Auditing of Adversarial AI Agent Behavior from Agent Traces" (Zhang, Yang, Xu): two-stage gate (single-event + sequence rules, then LLM audit agent). On OpenAgentSafety: LLM audits per run 8.15 to 2.33; tokens 47.8k to 14.6k; detection 72.8% vs 81.5% for audit-everything; multi-agent case studies flag all malicious traces with >80% token saving. Cheap rule gate plus LLM = the hybrid pattern that undercuts pure statistical detectors. Overlap: adjacent. — [arXiv 2610.07256](https://arxiv.org/abs/2610.07256)
- 2605.11053 (Zavrak; first titled MCPShield) "Content-Aware Attack Detection in LLM Agent Tool-Call Traffic": session graphs with SBERT node features; GAT/GCN/GraphSAGE, MLP, XGBoost/RF/LR/SVM on RAS-Eval, ATBench, combined. Metadata-only plateaus ~0.64 AUROC for every architecture; content embeddings >0.89; tree ensembles on pooled SBERT 0.975 (GNN 0.917, MLP 0.896 on RAS-Eval); random-split evaluation inflates AUROC up to 26 points vs task-disjoint splits; self-supervised pretraining gave no label-efficiency gain. Uses ATBench, which we also use. Overlap: SAME on the evaluation-protocol point, close on the feature-schema point. — [arXiv 2605.11053](https://arxiv.org/abs/2605.11053)
- SentinelAgent (2025): graph-based anomaly detection in multi-agent systems at node/edge/path level; only two case studies (email assistant, Magentic-One). Overlap: adjacent. — [arXiv 2505.24201](https://arxiv.org/abs/2505.24201)
- AgentHarm (ICLR 2025, background): 110 malicious tasks (440 with augmentations), 11 harm categories, 2-8 tools each; a refusal/compliance benchmark, not a trace-anomaly set, so using it as a monitor-detection set needs justification. Overlap: adjacent. — [arXiv 2410.09024](https://arxiv.org/abs/2410.09024)

### Inferences
- A reviewer who knows 2605.11053 will ask why we claim anything from behaviour-only features without a content-aware comparison on the same data. Its 0.64 ceiling is a direct competitor number to ours.
- Most of the field (TRAIL, TrajAD, OpenClawBench, ATBench, TraceDance) labels with LLMs or synthetic perturbation. A behaviour-only detector on those labels may be learning generator artefacts. This supports a validity framing.
- No paper in this set runs Isolation Forest, LSTM-AE or Deep Clustering on agent traces. Our method choice is unusual rather than standard; the 2026 norm is LLM judge or fine-tuned small LLM.

### Gaps
- Full texts not read for any item; per-model tables (TRAIL, ATBench, OpenClawBench) and TraceAegis F1 unverified.
- ATBench synthetic status unconfirmed.
- Whether any of these released CERT-style behavioural baselines: not found.

## 2. LLM-as-monitor vs classical behavioural anomaly detection

### Takeaway
LLM monitors dominate the 2025-26 literature and lab practice (CoT, action, hybrid). Classical behavioural detection appears mainly as rule/graph/structure detectors (TraceAegis, SentinelAgent) or as a cheap pre-filter in front of an LLM (2610.07256). Evidence on LLM monitors' own validity is thin and getting critical.

### Cited Findings
- SHADE-Arena: benign main task plus hidden malicious side task; best sabotage success 27% (Claude 3.7 Sonnet) and 15% (Gemini 2.5 Pro); best monitor Gemini 2.5 Pro AUC 0.87; hidden scratchpad helps agents evade, visible reasoning helps monitors. — [arXiv 2506.15740](https://arxiv.org/abs/2506.15740); [Anthropic post](https://www.anthropic.com/research/shade-arena-sabotage-monitoring)
- OpenAI, Monitoring Reasoning Models: a weaker LLM reading CoT catches reward hacking better than action-only; training against the CoT monitor yields obfuscated reward hacking; argues for a "monitorability tax". — [arXiv 2503.11926](https://arxiv.org/abs/2503.11926)
- CoT Red-Handed (NeurIPS 2025): CoT monitoring improves detection by up to 27 points where action-only fails on subtle sabotage, but misleading rationales hurt on obvious sabotage; a hybrid scoring reasoning and output separately beats both. — [arXiv 2505.23575](https://arxiv.org/abs/2505.23575)
- Ctrl-Z (Redwood): BashBench, 257 tasks; resampling + earlier-step analysis cut attack success from 58% to 7% at about 5% usefulness cost. Control-evaluation framing with a trusted weak monitor. — [arXiv 2504.10374](https://arxiv.org/abs/2504.10374)
- Chain of Thought Monitorability position paper (multi-lab, July 2025): CoT monitoring promising but fragile. — [arXiv 2507.11473](https://arxiv.org/abs/2507.11473)
- Monitoring Monitorability (OpenAI-led, ICML 2026): monitorability metric and three evaluation archetypes; CoT monitoring beats action-only; frontier models generally but not perfectly monitorable. — [PMLR v306](https://proceedings.mlr.press/v306/guan26h.html) (no arXiv id found)
- CoT Monitoring Can Fail Even When CoT Is Necessary (Oct 2026): failures at all three lowest of four necessity levels; in a medical task, patient sex and a clinical attribute influenced at least 19% and 24% of correct answers yet were rarely verbalised. — [arXiv 2610.18105](https://arxiv.org/abs/2610.18105)
- AutoMonitor-Bench (ACL 2026 Findings): 3,010 paired samples; 12 proprietary + 10 open LLM monitors; consistent miss-rate vs false-alarm trade-off; fine-tuned Qwen3-4B on 153,581 samples. — [arXiv 2601.05752](https://arxiv.org/abs/2601.05752)

### Inferences
- A behaviour-only detector is a different threat model, not a rival: it needs no CoT access and no inference-time LLM, which is its real selling point. Reviewers will still want the LLM-monitor number beside it on the same split.
- The CoT literature concentrates on intent-revealing text; our schema deliberately excludes content, so it cannot detect anything that is only visible in content (2605.11053 quantifies this loss).

### Gaps
- No head-to-head of LLM monitor vs IForest/LSTM-AE on identical agent traces found.
- No 2026 Anthropic/METR/Redwood paper on classical statistical monitoring found; coverage of lab blogs was shallow.

## 3. Insider-threat bridge and OWASP agentic Top 10 evaluations

### Takeaway
The "agents as insiders" idea is mostly vendor marketing (Exabeam) plus surveys. Our SSRN preprint appears to be the only published attempt at CERT-to-agent-trace transfer; nothing peer-reviewed competes. OWASP ASI empirical work exists as Inspect benchmarks and drafts, not arXiv papers.

### Cited Findings
- Exabeam white paper and blog argue CERT insider-threat principles extend to agents and non-human identities, and sell "Agent Behavior Analytics". Vendor claims, no detection data. Overlap: adjacent (conceptual). — [Exabeam white paper](https://www.exabeam.com/resources/white-papers/modernizing-the-cert-insider-threat-framework-for-the-agentic-enterprise/)
- SSRN, "From Rogue Employees to Rogue Agents: Repurposing Insider Threat Detection for AI Agent Governance" (Bipin Rimal): shared feature schema, Isolation Forest + two detectors; surfaced in search with a summary saying the IForest trained on insider data retains most detection power on agent traces. This is the authors' own prior paper; page returned HTTP 403 so unread by me. Overlap: same (prior work, must be cited/disclosed). — [SSRN 6355658](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6355658)
- LLM zero-shot insider/APT detection on CERT r5.2 (F1 +11.40 over GABM on CERT; +31.50 on PicoDomain); no agent traces. Shows the CERT side is now LLM-driven too. Overlap: adjacent. — [arXiv 2608.16508](https://arxiv.org/abs/2608.16508)
- Hybrid insider-threat framework with multi-agent simulation and SIEM correlation (different: agents as simulated users). — [arXiv 2601.04243](https://arxiv.org/abs/2601.04243)
- Systematic survey of agent security recommends an action-log schema with per-deployment behavioural baseline and drift detection (recommendation, not evaluation). — [arXiv 2604.23338](https://arxiv.org/abs/2604.23338)
- OWASP Top 10 for Agentic Applications published 9 Dec 2025 by the OWASP GenAI Security Project. — see [AgentThreatBench write-up](https://ukgovernmentbeis.github.io/inspect_evals/evals/agent_threat_bench/index.html)
- AgentThreatBench: OWASP-agentic tasks in Inspect AI, merged into UK AISI inspect_evals; two of three tasks test ASI01 goal hijack. No arXiv paper found. — [inspect_evals page](https://ukgovernmentbeis.github.io/inspect_evals/evals/agent_threat_bench/index.html); [blog](https://dev.to/vaishnavi_gudur/agentthreatbench-the-first-owasp-agentic-top-10-security-benchmark-6pp)
- AASTF (50 scenarios across ASI01-10) and CSA RiskRubric Agentic Benchmark Suite (draft, 2026-03-27): drafts/white papers, not confirmed on arXiv. — [AASTF draft](https://anonymousaak.github.io/aastf/research/arXiv-paper-draft/); [CSA](https://labs.cloudsecurityalliance.org/agentic/agentic-riskrubric-benchmark-suite-v1/)

### Inferences
- Mapping features to ASI01-10 is novel enough to keep, but the evidence base is grey literature; word the mapping as a taxonomy exercise, not a validated coverage claim.

### Gaps
- No arXiv empirical paper evaluating detectors per ASI category found.
- No CERT-Insider-Threat-Center-authored agent paper found.
- MITRE ATLAS-specific agent-trace detection work not searched; open.

## 4. What an evaluation-validity paper must cite

### Takeaway
The validity literature is new (2026) and directly relevant. Four items are close to our claim; cite them before a reviewer does.

### Cited Findings
- 2605.11053 (random vs task-disjoint splits, up to 26 AUROC points inflation; metadata ceiling 0.64). Overlap: SAME. — [arXiv 2605.11053](https://arxiv.org/abs/2605.11053)
- "Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks" (R-Judge, InjecAgent, AgentHarm, AgentDojo; up to 22 models; always-positive F1 = 2π/(1+π) = 0.690 on R-Judge beats five of 21 models; capability vs misalignment-safety ρ = −0.44, n=21). We use AgentHarm, so directly applicable; also the always-positive baseline should be reported in our tables. Overlap: close. — [arXiv 2607.28685](https://arxiv.org/abs/2607.28685)
- "Are Agent-Safety Benchmarks Ready for Monitor-Portfolio Procurement?" (NeurIPS 2026 workshop "Who Verifies the Agents?"): InjecAgent has 63,240 rows but only 17 source templates, no natural benign controls; 404 independent benign templates needed vs 5 available. Template-poor data cannot support confirmatory monitor comparison. Overlap: close. Workshop venue is a likely home or peer for our paper. No arXiv id found. — [NeurIPS page](https://neurips.cc/virtual/2026/172899)
- "Taxonomy and Consistency Analysis of Safety Benchmarks for AI Agents": 40 benchmarks, Kendall's W = 0.10 (p = 0.94), no ranking agreement. Overlap: adjacent. — [arXiv 2605.16282](https://arxiv.org/abs/2605.16282)
- "Agent-Safety Evaluations as Load-Bearing Evidence": reconstructability metric; replay preconditions unmet in every scored trace. Overlap: adjacent. — [arXiv 2607.12469](https://arxiv.org/abs/2607.12469)
- HackDetect: score inflation 0.45-1.00 in paired comparisons when agents exploit evaluation artefacts (secondary search snippet; not fetched). Overlap: adjacent, unverified. — [arXiv 2607.22368](https://papers.cool/arxiv/2607.22368)
- AutoMonitor-Bench (monitor reliability, miss/false-alarm trade-off). Overlap: adjacent. — [arXiv 2601.05752](https://arxiv.org/abs/2601.05752)

### Rating table (overlap with our claim)
| Item | Overlap |
|---|---|
| 2605.11053 content-aware tool-call detection | same |
| 2607.28685 validity audit | close |
| NeurIPS-26 monitor-portfolio readiness | close |
| TraceAegis 2510.11203 | close (method) |
| Rimal SSRN 6355658 | same (own prior work) |
| 2605.16282 benchmark taxonomy | adjacent |
| 2607.12469 reconstructability | adjacent |
| ATBench 2604.02022, TRACE 2601.20103, TRAIL 2505.08638, OpenClawBench 2605.29253, TrajAD 2602.06443, TraceDance 2609.33295, 2610.07256, SentinelAgent, AgentHarm, AutoMonitor-Bench | adjacent |
| SHADE-Arena, CoT monitoring papers, Ctrl-Z, 2610.18105 | adjacent (different monitor type) |
| Exabeam, 2608.16508, AgentThreatBench | none to adjacent |

### Inferences
- Minimum cite set: 2605.11053, 2607.28685, the monitor-portfolio paper, TraceAegis, SHADE-Arena, CoT Red-Handed or 2503.11926, ATBench, TRAIL, TRACE, AgentHarm, OpenClawBench (silver-label point), and our own SSRN paper.
- Outdated in our approach: no content-aware or LLM-monitor baseline; no always-positive/trivial baseline; random splits if used (task-disjoint required); possibly using synthetic or LLM-labelled sets as ground truth; the closest metadata-only number in the literature (0.64) is a ceiling we should compare against.

### Gaps
- OpenReview paper for the monitor-portfolio work has no arXiv id found; fetch OpenReview w0x22PRthi if citation needs it.
- I did not verify HackDetect, TraceAegis F1, TrajAD +11.38, or Monitoring Monitorability beyond snippets.
- The user-supplied IDs (2605.29253, 2510.11203, 2609.33295, 2610.07256, 2605.11053) all resolved and matched their descriptions.
