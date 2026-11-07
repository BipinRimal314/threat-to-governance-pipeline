# Handoff

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
