# Numbers in paper/short/main.tex and their sources

Rewritten 2026-11-11. Paths are relative to the repo root.

**One script makes Tables I, II and III:** `scripts/heldout_check.py` writes `results/tables/heldout_check.json` (`H` below).
Run: `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/heldout_check.py` (about 10 minutes on CPU; torch 2.15).
Older numbers come from `results/tables/experiment_*.json` (`T/`), which use the in-sample protocol.

## Table I and abstract (held-out, UBFS-20)

- Syn.: `H.synthetic_exp3.<Model>.<Cat>.held_out`. ATB.: `H.atbench_exp12.<Model>.<Cat>.held_out`.
- Count: `H.<setting>.<Model>.<Cat>.trivial.auc` (raw `primary_event_count`; the raw AUC is averaged over seeds first; no cell needed a sign reversal). Synthetic ASI02 count is 0.531.
- "0.50-0.51" synthetic ASI02 (.503, .503, .506). "0.82-0.89" ATBench ASI02 (.870, .819, .894). Gap 0.32-0.39.
- "0.89-0.96" synthetic ASI06 (.894, .959, .964). "0.79-0.89" ATBench ASI06 (.866, .794, .886).
- "up to 0.39": DC ASI02, .894 - .506 = .388.
- Synthetic ASI02 interval 0.35-0.65 and ATBench ASI02 lower bounds 0.78, 0.71, 0.81: `ci95_hanley_mcneil_perfold_n` (the negatives of one test fold: 29 synthetic, 50 ATBench). Table II uses all four TRAIL negatives.
- Positives per category: `n_pos`. Synthetic 29-30. ATBench 61, 29, 29, 73, 29, 29.
- "at most 0.009" (ATBench) and "at most 0.044" (synthetic): max of `abs(in_sample - held_out)` over all cells.
- Count margin over the best detector, ATBench: 0.046 to 0.079.

## Sec. IV, length in the raw ATBench rows

Computed on `load_dataset("AI45Research/ATBench", "ATBench500", split="test")`, with none of our parsing:
message count AUC-ROC 0.924; median 8 messages (safe), 10 (unsafe); 226 of 250 unsafe have 10 or more; 213 of 250 safe have 8 or fewer; tools offered 0.509.
Source: `H.atbench_raw_length` (`raw_atbench_length()` in the script).

## Table II and Sec. V, TRAIL

- 148 traces, 144 with errors, 4 without (3 GAIA, 1 SWE-bench): `get_trail_labels`. Older notes said 143 and 5.
- In-sample, held-out, interval, count: `H.trail_exp1.<Model>.{in_sample, held_out, ci95_hanley_mcneil_total_n, trivial.auc}`.
- TRACE -> TRAIL IF 0.614: `T/experiment_2_transfer.json` `trace_to_trail`. ATBench -> TRAIL IF 0.774: `T/experiment_7_mcp_transfer.json` `atbench_to_trail`. Not re-run.
- Seed std 0.007-0.050: the six std values of the old Table II (`T/experiment_1`, `_2`, `_7`).
- IF in-sample AUC-PR 0.982: `T/experiment_1_baselines.json`.

## Not re-run with held-out negatives (said so in the paper)

- Semantic features on synthetic ASI02 (DC 0.537 -> 0.559, IF 0.514 -> 0.494): `T/experiment_8_hybrid.json`.
- Distillation (0.538-0.540 distributed; 0.936-0.999 focused): `T/experiment_5_distillation.json`.

## Reproduction

- In-sample IF and LSTM match `T/` to three decimals in every cell.
- In-sample DC on ATBench is 0.019-0.040 below `T/` in all six cells; synthetic DC ASI01 0.639 against 0.621. Deterministic here. Cause not found.

## Literature numbers

Zavrak (arXiv 2605.11053) abstract: "up to 26 percentage points", "around 0.64", "above 0.89". Checked on the arXiv page, 2026-11-11.
All 21 bibliography entries: see `refs.bib` header. 13 were checked against source pages on 2026-11-11.

## Deliberately not used

Spearman rho (-1.000 over three, -0.40 over four); every CMU-CERT number; the old "143 errors, 5 negatives" count.
