# Base-rate and few-negative artifacts in detector evaluation (TRAIL claim)

Overlap scale with our claim (transfer IF AUROC 0.69-0.71 > in-domain 0.58 on TRAIL, ~97% positive, ~5 negatives, so score reflects base rate and a few negatives): none / adjacent / close / same.

## 1. AUC-ROC vs AUC-PR under imbalance; variance of AUC with very few examples of one class

### Takeaway
The classic literature (Davis & Goadrich, Saito & Rehmsmeier) says ROC can look good under imbalance, but McDermott et al. 2024 argue that imbalance alone is not the key factor. Hanley-McNeil gives the standard-error tools. No source found treats the case of only ~5 negatives with a majority-positive class. Note: TRAIL is the reverse of the usual rare-positive case (positives are 97%), so PR-vs-ROC advice must be flipped or applied to the negative class.

### Cited Findings
- Saito & Rehmsmeier 2015: ROC plots can be deceptive on imbalanced data; PR plots give a more accurate picture of future performance. Overlap: adjacent (it covers rare positives, not rare negatives, and does not address small-n variance). — [PLoS ONE 10(3):e0118432, doi:10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432); [PMC4349800](https://pmc.ncbi.nlm.nih.gov/articles/PMC4349800)
- Davis & Goadrich 2006: a curve dominates in ROC space iff it dominates in PR space; optimizing AUROC does not guarantee optimizing AUPRC; linear interpolation in PR space is incorrect. Overlap: adjacent. — [ICML 2006, doi:10.1145/1143844.1143874](https://doi.org/10.1145/1143844.1143874); [MLAnthology](https://mlanthology.org/icml/2006/davis2006icml-relationship)
- McDermott et al. 2024 (NeurIPS): the two metrics differ in how they weight false positives; the deciding factor is the use case, not class imbalance. AUPRC can favor high-prevalence subgroups. Useful as a counterweight: do not cite Saito as "AUROC is invalid under imbalance". Overlap: adjacent. — [arXiv:2401.06091](https://arxiv.org/abs/2401.06091); [NeurIPS record](https://mlanthology.org/neurips/2024/mcdermott2024neurips-closer)
- Hanley & McNeil 1982: AUC equals the probability a random positive outranks a random negative (Wilcoxon/Mann-Whitney); gives closed-form standard error and sample-size relations. This is the direct basis for "5 negatives means a very wide interval". Overlap: adjacent (method, not a finding about our setup). — [Radiology 143(1):29-36, doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747); [PubMed](https://pubmed.ncbi.nlm.nih.gov/7063747/)
- LeDell, Petersen, van der Laan 2015: influence-curve confidence intervals for cross-validated AUC (package cvAUC). Overlap: adjacent (CI method). — [EJS 9:1583-1607, doi:10.1214/15-EJS1035](https://doi.org/10.1214/15-EJS1035)

### Inferences
- With n_neg ~ 5 and n_pos ~ 143 there are only ~715 positive-negative pairs, and they come from 5 negative traces. AUC is then effectively an average over 5 negatives' ranks. A Hanley-McNeil or bootstrap CI will probably span most of 0.3-0.9, so 0.58 vs 0.70 is likely not a significant difference. We should compute this and report it (not yet computed here).
- "AUC measures P(pos > neg)" implies the majority-class prior drops out of AUROC in theory. So a "base rate" explanation must be argued as: few negatives make rank-AUC noisy, and any nuisance feature correlated with those few negatives (trace length, source GAIA vs SWE-Bench) can drive it. Phrase the claim carefully to avoid contradicting McDermott's point.

### Gaps
- No paper found on AUROC specifically with ~5 negatives and ~97% positives. No paper found giving exact-permutation or small-n guidance for extreme minority negatives (not searched beyond one query).
- Did not open Hanley-McNeil or LeDell full text; only abstracts and search summaries.

## 2. Base-rate fallacy in intrusion detection and successors

### Takeaway
Axelsson is the canonical citation. It shows that at low attack prevalence, false-alarm rate dominates usefulness. It is the opposite regime to TRAIL (very high prevalence), so it supports "report prevalence-aware metrics" but not the specific artifact.

### Cited Findings
- Axelsson 1999 (ACM CCS, pp. 1-7) and the extended 2000 version, "The Base-Rate Fallacy and the Difficulty of Intrusion Detection", ACM TISSEC 3(3):186-205. Argues false-alarm rate is the limiting factor for IDS performance. Overlap: adjacent. — [CERIAS record](https://www.cerias.purdue.edu/apps/reports_and_papers/view/2815); [full text, TISSEC 2000](https://www.profsandhu.com/cs6393_s20/p186-axelsson.pdf)

### Inferences
- Axelsson is about the posterior P(intrusion | alarm) under rare intrusions. Our case is the mirror image (rare benign). Cite it as the origin of "prevalence changes what a detector score means", and state that our regime is inverted.

### Gaps
- Did not search for successors (e.g. Gates & Taylor 2006 "challenging the anomaly detection paradigm", Sommer & Paxson 2010 "outside the closed world"). These are from my background knowledge and have NOT been verified in this session; verify before citing.

## 3. Benchmark evaluation flaws in anomaly/agent detection

### Takeaway
Two sources are close in spirit: arXiv 2605.11053 (evaluation protocol inflates AUROC) and Wu & Keogh (benchmark flaws create illusion of progress). Neither studies few-negatives AUROC or uses transfer-beats-in-domain as a diagnostic.

### Cited Findings
- arXiv 2605.11053 (Zavrak, May 2026): on MCP tool-call attack detection, "naive random-split evaluation inflates AUROC by up to 26 percentage points" vs task-disjoint splits, attributed to task memorization on RAS-Eval (metadata-only features). Metadata-only AUROC plateaus ~0.64 vs content embeddings above 0.89. ATBench used label-stratified splits; the sources I read do not show a random-split inflation result for ATBench, so "RAS-Eval" is the safe attribution. Overlap: close (same family: a protocol artifact inflates AUROC for agent-trace detectors), but the mechanism is leakage, not base rate or few negatives. — [arXiv:2605.11053](https://arxiv.org/abs/2605.11053)
- Wu & Keogh: popular time-series anomaly benchmarks (Yahoo, Numenta, NASA) suffer triviality, unrealistic anomaly density, mislabeled ground truth, run-to-failure bias; apparent progress may be illusory. Introduces the UCR Anomaly Archive. Overlap: adjacent (dataset-level validity argument we can parallel; "unrealistic anomaly density" is nearest to our prevalence point). — [arXiv:2009.13807](https://arxiv.org/abs/2009.13807)
- Ascarate et al. 2026, "Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection": class-split evaluation can be ill-posed; scores can collapse to chance or invert, and preferred score direction depends on the unknown anomaly class. Proposes a training-free "neighborhood class leakage" diagnostic. Closest found to a "diagnose a broken evaluation before trusting AUROC" paper. Isolation Forest is not named in the abstract I read (a search snippet suggested it is among the methods; unverified). Overlap: close (diagnostic for invalid AUROC), though the cause differs (score direction, not few negatives). — [arXiv:2606.02601](https://arxiv.org/abs/2606.02601)
- Sadhu et al. 2026, "What Do Compliance Detectors Read?": detectors' verdicts do not depend on the stated rule ("rule blindness"); a bag-of-words model matches the proposed method's pooled generalisation. A search snippet claimed all 380 ordered cross-domain pairs had AUROC at or above 0.79; I could not find that in the abstract, so treat it as unverified. Overlap: adjacent (trivial baseline as a validity check). — [arXiv:2608.16852](https://arxiv.org/abs/2608.16852)

### Inferences
- Zavrak's metadata-only result is a template: show that a content-free or trivial feature (trace length, span count, source dataset) reproduces the Isolation Forest AUROC on TRAIL. That would support the "reflects artifacts, not detection skill" claim.
- Cite Wu & Keogh for the general principle and Zavrak for the agent-detection precedent, and say plainly that our artifact (few negatives) differs from theirs.

### Gaps
- Did not find a paper on leakage/memorization confounds in agent-trace anomaly detection beyond Zavrak.
- Wu & Keogh claim (via Keogh slides) that ~95% of papers have flaws is the author's own claim; not independently checked.

## 4. Has anyone flagged TRAIL, or used "transfer beats in-domain" as a metric diagnostic?

### Takeaway
I found no one who has flagged TRAIL as unsuitable for AUC-based evaluation, and no source using "transfer beats in-domain" as a broken-metric diagnostic. This looks like a gap our claim can fill, but absence of search hits is not proof of absence.

### Cited Findings
- TRAIL: 148 traces (118 GAIA, 30 SWE-Bench), 841 errors, 1,987 OpenTelemetry spans, 575 spans with at least one error; designed for error localization and classification scored by joint accuracy, not for trace-level anomaly detection. Overlap with our claim: none (it is the dataset under test). — [arXiv:2505.08638](https://arxiv.org/abs/2505.08638); [dataset card](https://huggingface.co/datasets/PatronusAI/TRAIL/blob/main/README.md)
- The dataset card states limits: errors partly induced (SWE-Bench CodeAct traces had instructional constraints added to introduce errors), output-generation errors are ~42% of all errors, text-only, performance falls with trace length. The card does not mention AUROC or negatives. — [dataset card](https://huggingface.co/datasets/PatronusAI/TRAIL/blob/main/README.md)
- Related inversion examples: a LessWrong post on deception probes reports cross-dataset mean AUROC 0.468 with 11 of 30 values below 0.5. This is a blog post, not peer reviewed, and it is about probes, not Isolation Forest. Overlap: adjacent. — [LessWrong](https://www.lesswrong.com/posts/MFdGxip7TdQS8eNc2/cross-dataset-transfer-evaluation-of-deception-probes-in)

### Inferences
- The "~97% error-positive" figure is consistent with TRAIL being an error-annotation benchmark in which nearly every trace has an error (574/... spans in the card are span-level; the trace-level figure and the ~5 negatives come from our own data, not from a source I read). State that these counts are ours.
- Because TRAIL traces were curated and partly induced to contain errors, "negative" traces are an accident of curation, not a sampled benign class. Cite the dataset card for this.

### Gaps
- No TRAIL citer was found that applies AUROC; I did not run a Semantic Scholar citation crawl.
- "Transfer beats in-domain as diagnostic" has no prior source found. Closest conceptual relatives: Ascarate (leakage diagnostic), Zavrak (protocol comparison). Overlap with our diagnostic: none/adjacent.

## Summary table

| Item | Link | Overlap |
|---|---|---|
| Zavrak, random-split inflates AUROC (RAS-Eval) | https://arxiv.org/abs/2605.11053 | close |
| Ascarate et al., score-direction instability | https://arxiv.org/abs/2606.02601 | close |
| McDermott et al., AUROC vs AUPRC | https://arxiv.org/abs/2401.06091 | adjacent |
| Saito & Rehmsmeier | https://doi.org/10.1371/journal.pone.0118432 | adjacent |
| Davis & Goadrich | https://doi.org/10.1145/1143844.1143874 | adjacent |
| Hanley & McNeil | https://doi.org/10.1148/radiology.143.1.7063747 | adjacent |
| LeDell et al. cvAUC | https://doi.org/10.1214/15-EJS1035 | adjacent |
| Axelsson 2000 | https://www.profsandhu.com/cs6393_s20/p186-axelsson.pdf | adjacent |
| Wu & Keogh | https://arxiv.org/abs/2009.13807 | adjacent |
| Sadhu et al. | https://arxiv.org/abs/2608.16852 | adjacent |
| TRAIL paper | https://arxiv.org/abs/2505.08638 | none (subject dataset) |
| Anyone flagging TRAIL / transfer-beats-in-domain diagnostic | not found | same: none found |
