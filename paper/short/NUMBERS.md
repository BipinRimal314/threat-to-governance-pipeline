# Numbers in paper/short/main.tex and their sources

Paths are relative to the repo root. `T/` = `results/tables/`.

The fresh-run directories `results/runs/20261107-164136/` and `results/runs/20261107-165123/`
are not on this machine (they were produced on the Arch PC and are not committed). Per
`paper/CARVE.md`, Exp 1 and 2 match the published tables on every non-CERT number, and
Exp 7 ASI02 AUC 0.871 is unchanged. So the values below come from `results/tables/` and CARVE.
Re-check against the run directories when they are copied over.

## Table I and abstract: synthetic vs ATBench, four shared categories (UBFS-20, AUC-ROC means)

Synthetic: `T/experiment_3_owasp.json` -> `<Model>.per_category.<Cat>.auc_roc.mean`.
ATBench: `T/experiment_12_real_owasp.json` -> `phase_a_ubfs20.<Model>.per_category.<Cat>.auc_roc.mean`.
Delta = ATBench - synthetic, computed from the unrounded JSON values, then rounded to 2 dp.

| Cat | Model | Syn | ATB | Delta |
|---|---|---|---|---|
| ASI01 | IF | .642 | .843 | +.201 -> +.20 |
| ASI01 | LSTM | .637 | .785 | +.148 -> +.15 |
| ASI01 | DC | .621 | .894 | +.273 -> +.27 |
| ASI02 | IF | .520 | .871 | +.351 -> +.35 |
| ASI02 | LSTM | .510 | .823 | +.312 -> +.31 |
| ASI02 | DC | .524 | .910 | +.386 -> +.39 |
| ASI06 | IF | .920 | .869 | -.052 -> -.05 |
| ASI06 | LSTM | .975 | .798 | -.177 -> -.18 |
| ASI06 | DC | .981 | .906 | -.075 -> -.08 |
| ASI10 | IF | .729 | .764 | +.036 -> +.04 |
| ASI10 | LSTM | .696 | .714 | +.019 -> +.02 |
| ASI10 | DC | .730 | .850 | +.120 -> +.12 |

- "up to 0.39" (abstract, intro, Sec. III): max |Delta| = 0.386 (DC, ASI02). Derived from the rows above. CARVE says "up to 0.4".
- "0.51-0.52" synthetic ASI02, "0.82-0.91" ATBench ASI02, "0.92-0.98" synthetic ASI06, "0.80-0.91" ATBench ASI06: ranges of the rows above; same ranges are stated in CARVE.md (last section).
- "at most 0.037" synthetic ASI02 seed std: `T/experiment_3_owasp.json` `DeepClustering.per_category.ASI02.auc_roc.std` = 0.0367 (IF 0.025, LSTM 0.019).

## Sec. III, UBFS-28 on synthetic ASI02

`T/experiment_8_hybrid.json` -> `<Model>.per_category.ASI02.ubfs_20|ubfs_28.auc_roc.mean`.
- DC 0.537 -> 0.559. IF 0.514 -> 0.494.

## Sec. III, Spearman

- rho = -0.40, n = 4 shared categories, smallest attainable exact p = 0.083: `paper/CARVE.md`, last section. (Reported once, with the caveat. The p = 0.75 is not quoted in the paper.)

## Table II and Sec. IV, TRAIL

- IF in-domain 0.577 +/- .021, DC in-domain 0.897 +/- .021: `T/experiment_1_baselines.json` `trail.<Model>.auc_roc`.
- IF in-domain AUC-PR 0.982: `T/experiment_1_baselines.json` `trail.IsolationForest.auc_pr.mean` = 0.98200.
- TRACE->TRAIL IF 0.614 +/- .007, DC 0.631 +/- .030: `T/experiment_2_transfer.json` `trace_to_trail.<Model>.target.auc_roc`.
- ATBench->TRAIL IF 0.774 +/- .009, DC 0.734 +/- .050: `T/experiment_7_mcp_transfer.json` `atbench_to_trail.<Model>.auc_roc`.
- "0.007-0.050" seed std range: min and max of the six std values above.
- Hanley-McNeil 95% CI 0.33-0.82 around 0.577, "about +/-0.24": `paper/CARVE.md` last section. (Re-derived with the formula in Appendix B, n_pos = 143, n_neg = 5: [0.33, 0.82].)
- "about five negatives", "about 97% error-positive": `paper/CARVE.md`.

## Sec. V, distillation (synthetic)

`T/experiment_5_distillation.json` -> `<Model>.per_category.<Profile>.auc_roc.mean`.
- HYDRA: IF 0.538, LSTM 0.538, DC 0.540.
- FOCUSED: IF 0.995, LSTM 0.999, DC 0.936.

## Sec. VI-VII, literature numbers

- Random splits inflate AUROC "up to 26 points"; metadata-only "near 0.64": `paper/CARVE.md` last section (citing arXiv 2605.11053).
- Content-aware "about 0.89 or higher": `reports/Agent monitor evaluation validity.md` ("content-aware features reach above 0.89"), arXiv 2605.11053. Not in the allowed result files; the task brief required it.

## Sec. VII, limitations and reproduction

- ASI02 has 29 positive trajectories in ATBench: `paper/CARVE.md` ("ASI02 58 -> 29").
- "10 of" 13 experiments reproduced: `paper/CARVE.md` last section.
- Four shared categories: `paper/CARVE.md`.

## Methodology counts (not results)

- 500 ATBench trajectories: `paper/CARVE.md` (draft abstract).
- 20-dimensional UBFS, seven groups; 28 with eight semantic features: `src/features/ubfs_schema.py`; UBFS-28 keys in `T/experiment_8_hybrid.json`.
- Five seeds: `configs/model_configs.yaml` (`n_seeds: 5`).
- 200 trees for Isolation Forest: `configs/model_configs.yaml` (`n_estimators: 200`).
- Three detectors, six ATBench categories (Appendix A): keys of `T/experiment_12_real_owasp.json`.
- 95%, 1.96: the Hanley-McNeil interval definition.

## Appendix A, ATBench full table

`T/experiment_12_real_owasp.json` -> `phase_a_ubfs20` (rows "20") and `phase_b_ubfs28` (rows "28"),
`<Model>.per_category.<Cat>.auc_roc.mean`, rounded to 3 dp. IF UBFS-28 ASI02 0.954 is quoted in Sec. III.

## Numbers deliberately not used

- Spearman rho = -1.000 (n = 3) in `T/experiment_12_real_owasp.json` `phase_c_correlation`: superseded by CARVE.
- Every CMU-CERT number (including CERT->TRAIL 0.711): loader code lost.
- 0.690 for TRACE->TRAIL IF and its CI 0.49-0.89 (CARVE.md, literature report): **not in any result file.**
  `T/experiment_2_transfer.json` gives TRACE->TRAIL IF target = 0.614. The +0.113 in the JSON is
  `auc_roc_drop` relative to the TRACE in-domain source (0.501), not to TRAIL in-domain (0.577);
  0.577 + 0.113 = 0.690 is an arithmetic slip. The paper uses 0.614.
