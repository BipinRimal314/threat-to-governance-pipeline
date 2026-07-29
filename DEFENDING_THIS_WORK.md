# Defending This Work: A Briefing

*A claim-by-claim guide to what the threat-to-governance-pipeline paper actually shows, where it is strong, where it is fragile, and how to answer the hardest questions honestly.*

This document is written for you — the author — to walk into a lab meeting, a
reviewer response, or an interview and speak about this work with precision.
The organizing principle is simple: **know exactly what each number supports,
and never claim more than the evidence carries.** A claim you can defend at its
true strength is worth more than a claim you have to walk back under one good
question.

---

## Part 1 — The one-paragraph honest summary

> We built a shared feature representation (UBFS) that maps both insider-threat
> logs and AI-agent execution traces into the same 20-dimensional space, and we
> ran three off-the-shelf anomaly detectors across both domains. The headline
> finding is that detectors trained on one domain are **not degraded** when
> applied to the other — and on a balanced agent benchmark, cross-domain
> transfer measurably *beats* within-domain training. The equally important
> finding is that the **blind spots transfer too**: attacks that are
> structurally indistinguishable from legitimate behavior (Tool Misuse,
> distributed distillation, decomposed tasks) evade detection in both domains
> for the same reason. The contribution is the bridge and the symmetry of the
> failure modes, not a new state-of-the-art detector.

If you can say that paragraph and defend every clause in it, you are in good
shape. The rest of this document is how to defend each clause.

---

## Part 2 — The claims, ranked by how well the evidence supports them

### TIER A — Strong claims you can defend aggressively

**A1. "The blind spots transfer across domains."**
This is the paper's most defensible and most interesting claim. Tool Misuse
(ASI02) is near-chance on synthetic data; decomposition costs IF/DC 5–6%
detection power (p=0.031, Wilcoxon); HYDRA distributed distillation sits at
0.54 AUC-ROC. The mechanism — "structurally legitimate behavior evades
structural anomaly detection" — is the same argument the insider-threat
literature makes about quiet-exfiltration insiders. This is a real, coherent,
and useful result. *Lead with this, not with the transfer numbers.*

**A2. "MCP transfer beats within-domain (104.8%)."**
TRAIL→ATBench IF = 0.878 vs within-domain 0.838. This gain (0.041) is ~18× the
seed-to-seed standard deviation (±0.002) and is measured on a **balanced**
benchmark (ATBench AUC-PR 0.77–0.83) where AUC-ROC is a trustworthy metric.
This one is statistically robust. You can defend it hard.

**A3. "Real ASI02 detection is far better than synthetic suggested."**
Real ATBench ASI02: 0.81–0.94 AUC-ROC across all models/feature sets, versus
~0.52 synthetic. You *found and corrected your own artifact* — this is a
strength, not a weakness. It demonstrates methodological honesty and it is the
kind of result reviewers respect. Present it as: "our synthetic generator
produced a circular blind spot; real data revealed it, and we report both."

### TIER B — Claims that are true but need the right framing

**B1. "CERT→TRAIL retains 97% of detection power."**  ⚠️ *Reframed in the revised paper.*
The honest version: an IF trained on CERT is **not degraded** on TRAIL
(0.731→0.711). But the 0.019 drop is *smaller than the target's own seed
standard deviation* (±0.047), so it is within noise — you cannot claim a
measured "97% retention" of a real effect. Two things make this fragile:
- TRAIL is ~97% error-positive, so its AUC-ROC is estimated against only ~5
  negatives and is inherently unstable.
- A detector **trained on TRAIL itself** scores 0.577 — *lower* than the 0.711
  from CERT transfer. Genuine skill transfer cannot beat in-domain training;
  this inversion proves the TRAIL AUC-ROC is dominated by evaluation noise.

*How to say it:* "CERT-trained detectors are not degraded on TRAIL, but I don't
lean on the 97% number — TRAIL's near-total positive prevalence makes its
AUC-ROC an unstable estimand, so I treat that result as 'no degradation within
noise' and put the weight on the balanced-benchmark transfer instead."

**B2. "TRACE→CERT improves Deep Clustering by +0.223."**
This one is large enough to trust (an order of magnitude above within-domain
variance). The mechanism you claim — more "normal" training data from any
domain sharpens the boundary — is plausible. Defensible, but frame it as "more
data helps even across domains," not as evidence of deep semantic transfer.

### TIER C — Claims to state carefully or soften

**C1. "UBFS costs only 9% vs the full pipeline (0.731 vs 0.807/0.799)."**  ⚠️ *Fixed to 0.799 / 8.5%.*
Two problems, both now handled in the revised paper:
- The number was **0.807**, but the companion repo's canonical value is
  **0.799 ± 0.017** (0.807 is a stale value that survived in the companion
  *manuscript* after the code was corrected). They agree within CIs; cite 0.799.
- More importantly: the companion paper's **entire thesis is that AUC-ROC is
  misleading here.** IF "wins" on AUC-ROC but the LSTM catches 3.8× more attacks
  at the 5% FPR where teams operate. So "9% worse on AUC-ROC" is a
  *ranking-quality* cost on the very metric that paper argues against — not an
  operational one. Say so.

**C2. "Distillation sensitivity: HYDRA stays near-chance regardless of scale."**
Fine as stated, but remember HYDRA is a **synthetic** profile injected by the
same generator that produced the ASI02 artifact. The honest caveat: "this is a
synthetic construction of a distributed attack; we have not validated it on real
distributed-distillation traces." Don't let anyone push you into claiming HYDRA
is an empirical finding about real attacks.

---

## Part 3 — The eight hardest questions and how to answer them

**Q1. "Your transfer 'beats' within-domain training. Isn't that impossible, and
doesn't it mean your metric is broken?"**
→ *Correct on TRAIL, and I say so in the paper.* On TRAIL specifically, yes —
transfer (0.711) beating in-domain (0.577) is a tell that the TRAIL AUC-ROC is
noise-dominated because TRAIL is ~97% positive. On **ATBench**, which is
balanced, transfer also beats within-domain, but there the gain is 18× the seed
noise and the metric is trustworthy, so that one is real. The lesson is exactly
the one the paper now makes: separate the noisy-metric settings from the
trustworthy ones.

**Q2. "TRAIL is 97% positive. Isn't your AUC-PR of 0.98 just the base rate?"**
→ Yes. AUC-PR ≈ prevalence when prevalence is near 1.0, so 0.98 on TRAIL is not
evidence of skill. That's why the revised paper adds a prevalence row to every
baseline and leads with recall@fixed-FPR. I don't cite TRAIL AUC-PR as
detection evidence.

**Q3. "Your ASI02 blind spot is 0.52 synthetic but 0.81–0.94 real. Doesn't that
mean the blind-spot finding is wrong?"**
→ The *synthetic* blind spot is an artifact — our generator perturbs only
privilege features by ±10% with an explicit "minimal/undetectable" construction,
so of course a structural detector can't see it; that's circular. The real-data
result (0.81–0.94) is the trustworthy one and we report it prominently. What
survives is the *conceptual* point: a truly structure-preserving attack is
undetectable by structural methods. We just can't claim ASI02-in-practice is
that attack — real ASI02 is detectable.

**Q4. "Is the 0.807 or 0.799 the real number? Your paper and the companion
paper disagree."**
→ 0.799 ± 0.017 is canonical — it's what the released reproducibility harness
produces and what the companion README/CLAUDE notes standardize on. 0.807 is a
stale manuscript value. They overlap within CIs (Δ=0.008). I cite 0.799 and
footnote the difference.

**Q5. "You use AUC-ROC as your primary metric across datasets with wildly
different base rates. Is that valid?"**
→ AUC-ROC is base-rate *independent* in principle, which is why we chose it for
cross-dataset comparison. The problem is not bias, it's *variance*: when one
class is tiny (TRAIL: ~5 negatives), the estimate is unstable. So AUC-ROC is the
right choice for comparability but must be read alongside its seed variance and
alongside recall@FPR, which the revised tables now do.

**Q6. "How much of this is synthetic?"**
→ Be exact and unembarrassed: OWASP profiling (Exp 3), distillation (Exp 5, 13),
and decomposition perturbations use synthetic injection. Real data: CERT (Exp
1–2), TRAIL/TRACE (Exp 1–2), ATBench (Exp 7, 8, 12). The synthetic experiments
generate *hypotheses about failure modes*; the real experiments (especially Exp
12) *test* them. Exp 12 is the one that catches the synthetic artifact — that's
the point of having it.

**Q7. "What's the actual contribution? These are off-the-shelf detectors."**
→ Correct, and that's deliberate. The contribution is (a) the UBFS bridge that
makes two domains comparable at all, and (b) the empirical finding that the
*failure modes* are symmetric across domains — structurally-legitimate attacks
evade both. We are not claiming a better detector; we're claiming that the
governance implication ("behavioral monitoring alone will miss the
sophisticated threats") holds identically in insider-threat and agent-safety
settings. That's a policy argument backed by measurement.

**Q8. "If TRAIL's metric is unreliable, why is it in the paper at all?"**
→ Because the *negative* result — that TRAIL's within-domain AUC-ROC is 0.577
and unstable — is itself informative: it shows the limits of structural
detection on a near-degenerate label distribution, and it's the control that
reveals the transfer-beats-in-domain inversion. Removing it would hide the very
evidence that keeps us honest about the noise.

---

## Part 4 — How to talk about the fixes you made

If someone asks "did you change anything after review?", the honest and
strong answer is:

> "Yes. On audit I found three framing issues and corrected them: I reframed the
> '97% retention' claim as 'transfer within noise' because the drop is smaller
> than the seed variance; I added positive-prevalence to every table so the
> TRAIL base-rate caveat is visible; I moved the real-data ASI02 result ahead of
> the synthetic one and labeled the synthetic 0.52 as a methodological artifact;
> and I reconciled a 0.807-vs-0.799 citation to the reproducible value. The code
> and the numbers were all correct and reproduce exactly — these were
> presentation and interpretation fixes, not data fixes."

That answer signals exactly the competence you're building toward: you can find
the weak points in your own work before a reviewer does, and you fix them by
tightening the claim to match the evidence rather than by overclaiming.

---

## Part 5 — The three sentences to memorize

1. **"The models port across domains — and so do their blind spots."** (The
   thesis. Defensible.)
2. **"I separate the transfer results by whether the metric is trustworthy:
   balanced benchmarks yes, near-degenerate label distributions no."** (Your
   statistical maturity in one sentence.)
3. **"The synthetic experiments generate hypotheses about failure modes; the
   real-data experiments test them — and one of them caught my own artifact."**
   (Why the methodology is honest.)
