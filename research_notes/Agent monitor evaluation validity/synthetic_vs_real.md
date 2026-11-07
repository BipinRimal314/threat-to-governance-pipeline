# Synthetic vs real evaluation of detectors and agent monitors: prior work

Overlap scale: none / adjacent / close / same, judged against our claim: "synthetic perturbation profiling of agent monitors gives confident but wrong per-category answers; ASI02 'tool misuse' ~0.52 AUC synthetic vs 0.82-0.95 on 500 real ATBench trajectories; ranking inverted."

## Q1. Classic security / anomaly-detection literature on synthetic-data validity

### Takeaway
The general lesson (synthetic or simulated attack/benign data produces over-optimistic or misleading detector results) is 25 years old and must be cited as foundation. No classic paper shows a per-category ranking inversion; they show inflated absolute scores and artefacts. All are overlap "adjacent".

### Cited Findings
- Sommer & Paxson, "Outside the Closed World: On Using ML for Network Intrusion Detection", IEEE S&P 2010. Argues ML finds similarity better than novelty, misclassification cost is high, evaluation is hard; gives guidelines. Overlap: adjacent (motivation, not a synthetic-vs-real measurement). DOI 10.1109/SP.2010.25 — [IEEE](https://ieeexplore.ieee.org/document/5504793); [ICSI PDF](https://www1.icsi.berkeley.edu/pubs/networking/outsidethe10.pdf)
- McHugh, "Testing IDS: a critique of the 1998 and 1999 DARPA IDS evaluations as performed by Lincoln Laboratory", ACM TISSEC 3(4):262-294, 2000. Criticises unvalidated test data, attack taxonomy from attacker's view, false-alarm treatment, low traffic rates, ROC construction. Overlap: adjacent (closest classic in spirit: synthetic benchmark validity). [PDF copy](https://www.cs.ucsb.edu/~kemm/courses/CS595/TestingIDSs/mchugh_ll_critique.pdf.gz); RAID 2000 version [DOI 10.1007/3-540-39945-3_10](https://dx.doi.org/10.1007/3-540-39945-3_10). (ACM DOI not verified in this session.)
- Mahoney & Chan, "An Analysis of the 1999 DARPA/Lincoln Laboratory Evaluation Data for Network Anomaly Detection", RAID 2003, LNCS 2820:220-237. Simulation artefacts in the synthetic background traffic "would lead to overoptimistic evaluation"; on mixed real+simulated traffic detectors catch fewer attacks. Overlap: adjacent, nearest classic to "synthetic is optimistic". [DOI 10.1007/978-3-540-45248-5_13](https://doi.org/10.1007/978-3-540-45248-5_13); [tech report](https://cs.fit.edu/media/TechnicalReports/cs-2003-02.pdf)
- Arp et al., "Dos and Don'ts of Machine Learning in Computer Security", USENIX Security 2022. Reviews 30 top-venue papers for pitfalls (sampling bias, unrealistic settings etc.). Overlap: adjacent; cite for "pitfalls widespread". [arXiv 2010.09470](https://arxiv.org/abs/2010.09470)
- Engelen, Rimmer, Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study", IEEE S&P Workshops 2021. Found errors in traffic generation, labelling (~10% of samples), artefacts (~20% of data); persists in CSE-CIC-IDS2018. Overlap: adjacent. [KU Leuven PDF](https://lirias.kuleuven.be/retrieve/713070)
- CERT insider-threat data: a 2025-26 systematic review of 131 studies reports performance inflation from dataset-version bias and inconsistent evaluation [Jurisic & Tomicic](https://www.fvv.um.si/rV/arhiv/2026/2026-02-Jurisic-Tomicic-E.html); Chimera paper's experts rated CERT realism 1.78/5 vs 4.20 for its own synthetic logs (authors have a competing dataset, so weak evidence) [arXiv 2508.07745](https://arxiv.org/pdf/2508.07745); Facade says predictable benign activity underestimates false positives [arXiv 2412.06700](https://arxiv.org/pdf/2412.06700). Overlap: adjacent. Use sparingly.
- Log anomaly detection: Le & Zhang, "Log-based Anomaly Detection with Deep Learning: How Far Are We?", ICSE 2022: models report F>0.9 on HDFS but none works consistently across settings [arXiv 2202.04301](https://arxiv.org/abs/2202.04301). Landauer, Skopik, Wurzenberger, "A Critical Review of Common Log Data Sets...", FSE 2024: most anomalies are not sequence patterns; simple methods score high [arXiv 2309.02854](https://arxiv.org/abs/2309.02854). Overlap: adjacent.
- Xiao et al. 2026 (Google Research page, no arXiv found): random splits give near-ceiling scores on HDFS/BGL; under grouped/chronological splits "scores drop and model rankings shift". This is a ranking change across evaluation protocol (models, not categories). Overlap: adjacent-to-close on the "ranking shifts" point. [page](https://research.google/pubs/protocol-sensitive-evaluation-of-log-anomaly-detection-component-costs-and-target-access-sensitivity-on-hdfs-and-bgl-2/)
- Wu & Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress", IEEE TKDE (arXiv v5 2022): four flaws in popular benchmarks; published comparisons may be unreliable. Overlap: adjacent. [arXiv 2009.13807](https://arxiv.org/abs/2009.13807)

### Inferences
- Cite Sommer-Paxson, McHugh, Mahoney-Chan, Arp as the lineage; our contribution is the new domain plus a per-category, ranking-level inversion measured against real data, which these do not report.
- Mahoney-Chan is the only classic with a synthetic-vs-real measurement of the same detectors; it shows absolute inflation, not inversion.

### Gaps
- Tavallaee et al. (KDD Cup 99 analysis) not searched; CERT critique is thin and partly from rival-dataset authors.
- McHugh ACM DOI and Engelen DOI not verified; use the links above or look up before submission.

## Q2. AI control / monitor evaluation

### Takeaway
Control literature worries that synthetic or elicited attacks under-test monitors, and one paper shows prompt-elicited vs training-time reward-hacking monitors fail to transfer. Nobody found compares per-category monitor performance on synthetic perturbations vs real trajectories.

### Cited Findings
- Kutasov et al., SHADE-Arena (17 task pairs; best monitor AUC 0.87 in v1; success needs hidden scratchpad). Model-organism style attacks by prompted agents. Overlap: adjacent (setting we implicitly contrast with). [arXiv 2506.15740](https://arxiv.org/abs/2506.15740)
- Loughridge et al., "Optimizing AI Agent Attacks With Synthetic Data" (Nov 2025): tunes attacks in a probabilistic simulation and shows transfer to SHADE-Arena; safety score 0.87 -> 0.41. Counter-example: calibrated synthetic data DID transfer. Must be cited to be fair. Overlap: adjacent (opposite finding). [arXiv 2511.02823](https://arxiv.org/abs/2511.02823)
- Najt et al., SLEIGHT-Bench (May 2026): 40 hand-crafted synthetic transcripts, 11 attack categories; Opus 4.6 monitor catches 32% at 1% FPR, 20/40 never caught. Authors state hand-crafted transcripts trade environmental realism for coverage. Overlap: adjacent; it is itself a synthetic-attack benchmark we are critiquing in kind. [arXiv 2605.16626](https://arxiv.org/abs/2605.16626)
- MonitoringBench (Matthews et al., 2026): argues current red-teaming under-elicits attacks and overstates monitor performance; single setting BashArena; reported catch rate falling 94.9% -> 60.3% for best refined attacks (figures from search snippet of author listing, not checked in the PDF). Overlap: adjacent. [arXiv 2605.09684](https://arxiv.org/pdf/2605.09684)
- Li et al. 2026, "Do Prompt-Elicited Trajectories Reflect Training-Time Reward Hacking?" (Trace-and-Amplify): monitors trained on prompt-elicited hacking data often fail on hacking that emerges during RL; conclusion "prompted hacking data may not fully reflect real training-time behavior" and can mislead. Table values (accuracy %, via fetch summary, verify against PDF): zero-shot gpt-4.1 gets 97.5 on prompt-elicited hacking but 3.7 on training-time hacking; prompt-trained LogReg 98.0 vs 8.79. Paper reports no ranking reversal and no AUC. Overlap: CLOSE on the thesis "synthetic/elicited eval is confidently wrong vs naturally arising", but domain (code reward hacking), unit (accuracy, trained probes) and claim (generalisation gap, not inverted category ranking) differ. Closest prior work found. [arXiv 2604.23488](https://arxiv.org/abs/2604.23488)
  - Caution: a first search summary described this paper as "near-random on in-the-wild" with "synthetic" wording; the abstract page uses "prompt-elicited" vs "training-time". Use the paper's own terms.
- AutoMonitor-Bench (ACL 2026 Findings): 3,010 samples, 22 LLM monitors, miss-rate/false-alarm tension; tests whether training on easy known misbehaviour helps with unseen implicit misbehaviour. Overlap: adjacent. [arXiv 2601.05752](https://arxiv.org/abs/2601.05752)
- Lindner et al., "Practical challenges of control monitoring in frontier AI deployments" (Dec 2025): deployment complications; abstract says nothing on synthetic vs real evaluation. Overlap: none/adjacent. [arXiv 2512.22154](https://arxiv.org/abs/2512.22154)
- "Async Control: stress-testing asynchronous..." (Dec 2025) surfaced in search, not read. [arXiv 2512.13526](https://arxiv.org/abs/2512.13526). Anthropic "strengthening red teams" post surfaced, not read: [post](https://alignment.anthropic.com/2025/strengthening-red-teams/).

### Inferences
- The control field already says "attack elicitation is the weak point"; our paper adds a measured case where the synthetic profile mis-ranks categories against real labelled data.
- Li et al. 2026 is the paper a reviewer will raise first. Position: they show a generalisation gap in one domain; we show a category-ranking inversion in agent-trajectory monitors.

### Gaps
- No Redwood "AI Control" (Greenblatt et al. 2023) or untrusted-monitoring paper was fetched; cite from memory only after verifying arXiv ids (2312.06942 believed, unverified).
- No Anthropic/OpenAI report comparing monitors on naturally arising misbehaviour vs model-organism attacks found.

## Q3. Agent security benchmarks with injected attacks, and transfer tests

### Takeaway
All the major agent-security benchmarks inject synthetic attacks; few test transfer to real ones. AgentDyn and BrowseSafe++ argue static or synthetic injection benchmarks mislead, but neither shows a category ranking inversion.

### Cited Findings
- AgentDojo (97 tasks, 629 security cases) [arXiv 2406.13352](https://arxiv.org/abs/2406.13352); InjecAgent (Zhan et al., Mar 2024; ~1,054 cases per secondary source) [arXiv 2403.02691](https://arxiv.org/abs/2403.02691); ASB (ICLR 2025) [arXiv 2410.02644](https://arxiv.org/abs/2410.02644); AgentHarm (ICLR 2025) [arXiv 2410.09024](https://arxiv.org/abs/2410.09024); RAS-Eval (80 cases, 3,802 attack tasks, 11 CWE categories) [arXiv 2506.15253](https://arxiv.org/abs/2506.15253). All synthetic or injected attacks. Overlap: none (they are the object of critique, not critics).
- ATBench (Li et al., Apr 2026; v-latest 1,000 trajectories, 503 safe / 497 unsafe; taxonomy of risk source, failure mode, real-world harm). Abstract does not say trajectories are real, and does not mention OWASP; it describes quality control with LLM filtering plus human audit. Our use of "500 real" corresponds to the earlier 500-case release; verify that "real" is defensible, since the trajectories appear generated (the fetch said the page does not confirm). Overlap: none (our evaluation data). [arXiv 2604.02022](https://arxiv.org/abs/2604.02022)
- AgentDyn (2026): nine defenses that score well on existing static benchmarks fail on a dynamic open-ended one; real injections are sparse. Overlap: adjacent-to-close on "benchmarks overstate defenses". [arXiv 2602.03117](https://arxiv.org/abs/2602.03117)
- BrowseSafe++ (Perplexity): detectors trained on separately collected attacked/clean pages learn topic/layout shortcuts; uses real-site injections and matched pairs. Overlap: adjacent (shortcut risk). [post](https://perplexity.ai/hub/blog/browsesafe-plus-plus)

### Inferences
- A synthetic ASI02 blind spot could be a shortcut artefact (perturbation style) not a monitor property; BrowseSafe++ supports that mechanism, worth one sentence.

### Gaps
- ATBench "real" status; ATBench-Claw/CodeX (arXiv 2604.14858) not read.
- No paper found that injects attacks into agent traces and then validates against real in-the-wild attacks across categories.

## Q4. Has anyone shown a synthetic-vs-real ranking inversion for detector categories?

### Takeaway
Not found. Nearest: Li et al. 2026 (elicited vs training-time monitor failure), Xiao et al. 2026 (model rankings shift with protocol), Mahoney-Chan 2003 (synthetic optimistic). A per-category inversion (worst on synthetic = best on real) appears to be novel, with the caveat that the search was about 22 calls and not exhaustive.

### Cited Findings
- Ranking shifts across protocols for models: [Xiao et al. 2026](https://research.google/pubs/protocol-sensitive-evaluation-of-log-anomaly-detection-component-costs-and-target-access-sensitivity-on-hdfs-and-bgl-2/); synthetic-to-real gap for monitors: [arXiv 2604.23488](https://arxiv.org/abs/2604.23488); synthetic optimism for detectors: [Mahoney-Chan](https://doi.org/10.1007/978-3-540-45248-5_13).

### Inferences
- Frame the novelty narrowly: "per-category ranking inverted, measured against labelled real trajectories", and state the novelty claim as "we found no prior report", not "none exists".

### Gaps
- Not searched: Google Scholar forward citations of Li et al.; Alignment Forum / LessWrong posts; METR posts; OWASP ASI taxonomy source for our own ASI02 labels.
