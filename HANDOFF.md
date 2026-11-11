# Handoff

## 2026-11-11: workshop paper rewritten on held-out numbers

Branch `paper/heldout-rewrite` (local, not pushed). The paper is `paper/short/main.tex`, 4 pages. Target: an ICLR 2027 workshop (suggested paper deadline 1 Feb 2027; the workshop list is due about 29 Nov 2026).

What changed:

- `scripts/heldout_check.py` re-runs Exp 1, 3 and 12 with test negatives held out of training, adds a tool-call-count baseline and Hanley-McNeil intervals, and writes `results/tables/heldout_check.json`. It makes every number in Tables I to III. `paper/short/NUMBERS.md` maps each number to its key.
- Three findings now carry the paper: the two constructions disagree per category (ASI02 0.50 against 0.82-0.89); on ATBench a raw count scores 0.89-0.94 and no detector improves on it; on TRAIL, deep clustering falls from 0.893 to 0.598 when its negatives are held out.
- TRAIL has 4 error-free traces, not 5.
- `refs.bib`: 10 of 21 entries were wrong and are corrected; 13 were checked against source pages.

Reviewed once by two fresh reviewers (structure; correctness and numbers). All findings were fixed, and the script was re-run afterwards (deterministic). No second review round was run.

Next steps:

1. Bipin reads `paper/short/main.pdf` and approves or changes the claim.
2. Find why deep clustering scores 0.02-0.04 lower than `results/tables/` on ATBench (torch 2.15 on CPU here). Run `heldout_check.py` on the Arch PC and compare.
3. Re-run the synthetic setting with folds grouped by base trace (the paper lists this as a limitation).
4. Add a paired bootstrap of count minus best detector, so "no detector improves on a count" has an interval.
5. Experiment E1 in `../PAPER_REVIEW_2026-11-11.md`: a strength parameter for the ASI02 profile, and a third construction (AgentHarm).
6. Check the remaining 8 older bibliography entries against their sources.
7. Pick the workshop when the ICLR list is out; convert the template; push after Bipin says yes.

Run: `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/heldout_check.py` (about 10 minutes; TRAIL and ATBench must be in the HuggingFace cache).

## 2026-11-07: the paper is the main work again

Decided on the Mac, 2026-11-07. Two lanes until a job or fellowship lands:

1. **This paper (main work).** No gatekeeper can block it until arXiv. Finish it first.
2. **Applications (background).** About 2 a day, technical-writer roles. Tracked outside this repo.

Everything else is parked (Clip Farm, videos, apps). KIAI belongs to the KIAI account.

### The carve

`paper/main.tex` has 13 experiments and no single claim. Do not add experiments. Carve one paper with
one thesis. There are two candidates, and Bipin picks one first:

> A (July note): Behavioural anomaly detection catches sloppy agents, not patient or imitative ones.
> This is a structural limit, not a tuning problem.

> B (CLAUDE.md, 7 Aug rewrite of main.tex): Synthetic perturbation profiling cannot locate blind
> spots. The synthetic ASI02 blind spot inverted on real data (rho -1.000). The blind spot that
> survives is architectural: per-entity monitors cannot see coordination.

Read `paper/main.tex` as it is now before choosing; it already follows B.

The evidence is already in the repo: mimicry evasion (5–25% AUC drop), clamping (1–8%), TRACE reward
hacking at chance, HYDRA distributed distillation at 0.54 whatever the scale, and ASI05 as the exception
with an irreducible structural signature. Transfer, ATLAS, temporal and UBFS-28 go to an appendix or a
second paper. The ASI02 blind spot is retracted (see README and CLAUDE.md); never bring it back.

The fresh-run harness (`TESTRUN.md`, `scripts/testrun.sh --all`, experiments 1–13) had not been run as
of 2026-09-28. Run it before submitting, so the numbers in the paper match a clean run.

### Working rules

- Bipin directs, Claude builds. Bipin owns every claim and writes his predictions before each run.
- He under-claims. Check the artifact before you agree with his self-assessment.
- Lead with the one-sentence version, then the detail.

### Next steps

1. On the Arch PC: `git status`. Commit any paper work done there that is not on GitHub yet.
2. Write the one-paragraph abstract for the carved claim. Bipin approves it before anything else.
3. Cut `main.tex` to the experiments that support that claim. Move the rest to the appendix.
4. Run `scripts/testrun.sh --all` and compare the results with `scripts/compare_results.py`.
5. Find an arXiv endorser for cs.LG or cs.CR if this is his first submission.
