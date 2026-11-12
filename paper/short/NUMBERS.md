# Numbers in paper/short/main.tex and their sources

Rewritten 2026-11-12. Paths are relative to the repo root. `T/` = `results/tables/`.

Run everything (about 40 minutes on CPU; TRAIL and ATBench must be in the HuggingFace cache):

    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    for s in heldout_check grouped_folds paired_bootstrap dial_check; do .venv/bin/python scripts/$s.py; done

The LSTM and deep-clustering detectors are pinned to CPU in `scripts/heldout_check.py`. All four scripts import its `MODELS`.

## Tables I, II, III and the abstract: `scripts/heldout_check.py` -> `T/heldout_check.json` (`H`)

- Table I. Syn.: `H.synthetic_exp3.<Model>.<Cat>.held_out`. ATB.: `H.atbench_exp12.<Model>.<Cat>.held_out`. Count: `...trivial.auc` (raw `primary_event_count`; the raw AUC is averaged over seeds; no cell needed a sign reversal).
- Synthetic ASI02 .503, .503, .505 ("0.50-0.51"). ATBench ASI02 .870, .819, .902 ("0.82-0.90"). Gap .37, .32, .40.
- Synthetic ASI06 .894, .959, .965 ("0.89-0.96"; the largest is 0.9649). ATBench ASI06 .866, .794, .899 ("0.79-0.90").
- Intervals in Sec. III: `ci95_hanley_mcneil_perfold_n`. Synthetic ASI02 0.35-0.65 (DC 0.36-0.65). ATBench ASI02 lower bounds 0.78, 0.71, 0.82.
- "at most 0.014" (ATBench) and "at most 0.039" (synthetic): max of `abs(in_sample - held_out)`.
- Count margin over the best detector on ATBench: 0.035 to 0.062 ("0.04-0.06").
- Table II. `H.trail_exp1.<Model>.{in_sample, held_out, ci95_hanley_mcneil_total_n}`; count `trivial.auc` = 0.569. 148 traces, 144 with errors, 4 without.
- Raw ATBench rows (Sec. IV): `H.atbench_raw_length`. Message count 0.924; medians 8 and 10; 226 of 250; 213 of 250; tools offered 0.509.

## Sec. III, dial: `scripts/dial_check.py` -> `T/dial_check.json`, `paper/short/figures/dial.pdf`

- s = 0: 0.503, 0.503, 0.505. s = 1: IF 0.848, LSTM 0.821, DC 0.848 ("0.82-0.85"). s = 0.1: DC 0.484.
- Count at each strength: 0.531, 0.726, 0.758, 0.819, 0.852, 0.887. It is above every detector at every strength.

## Sec. IV, bootstrap: `scripts/paired_bootstrap.py` -> `T/paired_bootstrap.json`

- 2,000 resamples. `all_seeds`: the interval excludes zero in 18 of 18 cells (smallest lower bound 0.003). `seed42`: 17 of 18; DC ASI01 includes zero.
- Negative weights are drawn per safe trace and shared across seeds. The count sign is fixed before resampling, from all data.

## Sec. VII, grouped folds: `scripts/grouped_folds.py` -> `T/grouped_folds.json`

- Synthetic ASI02 grouped: 0.509, 0.508, 0.508 ("0.51"). Largest change in any cell: 0.039. About 6 positives per fold.

## Sec. VII, device: one-off check, 2026-11-12 (not in a script in this repo)

DC, in-sample ATBench, mean of five seeds, ASI01 / ASI02 / ASI10:
CPU on macOS (torch 2.15) 0.896 / 0.916 / 0.838; CPU on Linux (torch 2.13) the same; CUDA (RTX 4060, torch 2.13) 0.894 / 0.910 / 0.850, which equals `T/experiment_12_real_owasp.json`; Apple MPS 0.869 / 0.888 / 0.808.

## Not re-run, quoted from older tables

- TRACE -> TRAIL IF 0.614 (`T/experiment_2_transfer.json`); ATBench -> TRAIL IF 0.774 (`T/experiment_7_mcp_transfer.json`). Transfer tests use no TRAIL trace in training.
- IF in-sample AUC-PR on TRAIL 0.982 (`T/experiment_1_baselines.json`).

## Bibliography

19 entries, all cited. Each was checked against its source on 2026-11-11 or 2026-11-12 (arXiv or publisher page, or the DOI record), except the author's own SSRN page, which returned HTTP 403 to an automated fetch.

## Removed from the paper on 2026-11-12

The distillation section (a corroboration in kind, not a measurement), the semantic-feature numbers and the seed-deviation paragraph: none was re-run with held-out negatives.
