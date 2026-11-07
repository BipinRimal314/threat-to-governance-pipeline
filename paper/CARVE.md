# Carve: thesis B (chosen 2026-11-07)

Status: DRAFT abstract. Bipin approves it before `main.tex` changes.

## Claim, in one sentence

Two standard ways of evaluating behavioural monitors for AI agents, synthetic perturbation profiling and AUC-ROC on
error-heavy trace benchmarks, give confident answers that real data contradicts.

## Draft abstract

Behavioural anomaly detection is a natural candidate for monitoring AI agents, and evaluating it usually rests on
two shortcuts: injecting synthetic attack profiles into agent traces, and scoring detectors with AUC-ROC on public
trace benchmarks. We test both shortcuts against real data and find that each produces a confident, wrong answer.
First, synthetic OWASP profiling ranks Tool Misuse (ASI02) as a blind spot (about 0.52 AUC-ROC). On 500 real ATBench
trajectories, ASI02 is the best-detected of six categories (0.82 to 0.95), and the ranking of the three categories
both experiments share is exactly inverted for all three models (Spearman rho = -1.000). The cause is in the
generator: the ASI02 profile perturbs only privilege features, so it is undetectable by construction, and adding
eight semantic features does not move it. Second, on TRAIL, which is about 97% error-positive, an Isolation Forest
transferred from insider-threat logs scores higher (0.71) than one trained on TRAIL itself (0.58). Genuine skill
transfer cannot do that, so the score measures the benchmark's base rate, not detection. A third case shows why
neither failure can be seen from inside the experiment: a synthetic prediction of the same kind, that distributed
distillation (HYDRA) stays near chance, was later corroborated by a real-world disclosure. Nothing distinguished the
two predictions beforehand. We argue that synthetic profiles are hypothesis generators, not evidence, and we report
AUC-PR and recall at fixed false-positive rates for every benchmark with extreme prevalence.

## What stays in the body

| Section | Why |
|---|---|
| UBFS, datasets, models (Methodology) | Needed to read any result |
| Exp 1 baselines, with the prevalence row | Sets up the TRAIL artifact |
| Exp 2 transfer, the 0.58 vs 0.71 inversion | Failure 2 |
| Exp 3 synthetic OWASP matrix | The prediction that fails |
| Exp 12 real-data OWASP validation | Failure 1 |
| Exp 8, synthetic ASI02 does not move | Shows the profile, not the detector, sets the score |
| Exp 7 per-risk ATBench table (ASI02 0.871) | Independent confirmation of Failure 1 |
| Exp 5 HYDRA + the disclosure, vs Exp 12 | The "nothing distinguished them" argument |
| Limitations | Must include the n = 3 point below |

## What moves to the appendix (labelled as hypotheses where synthetic)

Exp 4 governance audit, Exp 6 decomposition, rest of Exp 7 transfer, Exp 9 adversarial evasion (synthetic),
Exp 10 temporal windows, Exp 11 ATLAS mapping (synthetic), Exp 13 distillation sensitivity (synthetic),
latent-space structure.

## Weak points a reviewer will find

1. **rho = -1.000 is over three categories.** With three items, a perfect inversion happens by chance 1 time in 6.
   The strong evidence is the size of the gap (0.52 synthetic against 0.82 to 0.95 real) and the mechanism, not
   rho. Say so in the abstract or the results, before a reviewer does.
2. **TRAIL has about 5 negatives.** That is the point of Failure 2, but the exact numbers are unstable. The fresh
   run (results/runs/20261107-164136) must reproduce 0.577 and 0.711 within noise before submission.
4. **The CERT numbers cannot be rerun (found 2026-11-07).** `cert_loader.py` imports
   `insider-detection/src/data/{loader,preprocessing,features}.py`. Those files were never committed: that repo's
   `.gitignore` line `data/` also ignores `src/data/`. They are on no machine and in no git history, so every CERT
   number (Exp 1 CERT column, Exp 2 CERT transfers, the 0.711) is unreproducible. Failure 2 does not need CERT:
   TRACE -> TRAIL IF scores 0.614 against TRAIL in-domain 0.577 (corrected 2026-11-07; an earlier note said 0.690, an arithmetic slip: +0.113 is against TRACE in-domain).
   Either lead with that, or rebuild the CERT features on `insider-budget/features.py` (the preregistered pipeline)
   and report new numbers.
3. **One real dataset.** ATBench is the only real test of the OWASP ranking. Name it as a limitation.

## Fresh rerun and literature sweep (2026-11-07)

Runs: `results/runs/20261107-164136` (all 13, Arch PC) and `20261107-165123` (1, 2, 7 without CERT).
Report: `reports/Agent monitor evaluation validity.md`; notes in `research_notes/`.

- 10 of 13 experiments match the published tables. Exp 1 and 2 match on every non-CERT number.
- **rho = -1.000 is gone.** The corrected taxonomy gives 4 shared categories (ASI06 joins), rho = -0.40,
  exact p = 0.75. With n = 4 no rho can be significant (smallest p = 0.083). Drop rank correlation; lead with the
  per-category gaps: ASI02 synthetic 0.51-0.52 vs ATBench 0.82-0.91; ASI06 synthetic 0.92-0.98 vs ATBench 0.80-0.91.
- **ATBench is not "real".** Its paper says trajectories are constructed, then LLM- and rule-filtered and
  human-audited (arXiv 2604.02022). Say "an independently constructed benchmark", never "real data".
- **TRAIL cannot rank detectors.** With 5 negatives, Hanley-McNeil 95% CI is about +/-0.24 (0.577 -> 0.33-0.82); 0.614 and 0.774 (ATBench -> TRAIL) both sit inside it.
  Do not say "base rate inflates AUROC"; say 5 negatives make the statistic too noisy to compare anything.
- Exp 7 per-category n changed with the taxonomy fix (ASI02 58 -> 29); ASI02 AUC 0.871, unchanged.
- Nobody found showing either claim. Closest: arXiv 2604.23488 (prompted vs training-time reward hacking) and
  2605.11053 (random splits inflate AUROC up to 26 points; metadata-only plateaus ~0.64 on ATBench).
- Reviewer gaps: no content-aware or LLM-monitor baseline, no trivial baselines (always-positive, trace length),
  no CIs on AUCs, no task-disjoint splits. Disclose SSRN 6355658.

Revised claim: how the attack data is constructed decides which attacks look detectable (two independent
constructions of the same OWASP categories disagree by up to 0.4 AUC per category), and a widely used trace
benchmark has too few negatives to rank detectors at all.

- **ASI02 mechanism, corrected:** `src/data/synthetic_generator.py` (ASI02 branch) changes no spans, tool names or order; it rescales numeric span attributes by +/-10%. It does not touch privilege features, despite main.tex and CLAUDE.md. The short paper describes the code.
- Short paper: `paper/short/` (main.tex, refs.bib, NUMBERS.md). Check every bib entry before submission.
