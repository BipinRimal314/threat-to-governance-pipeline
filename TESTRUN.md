# Fresh test run on the Ubuntu box

A full rerun of experiments 1–13 into its own folder, compared against the
published `results/tables/`. The published tables are never written.

## Once per machine

1. Python 3.10+ and venv: `sudo apt install python3.11 python3.11-venv`
2. NVIDIA driver, so `nvidia-smi` shows the RTX 4060. Without it the run
   still works on CPU, but the LSTM and clustering models take much longer.
3. HuggingFace: accept the terms on
   <https://huggingface.co/datasets/PatronusAI/TRAIL> and
   <https://huggingface.co/datasets/PatronusAI/trace-dataset>, then, after the
   first `--check` has created `.venv`, log in with
   `.venv/bin/hf auth login`.
4. Optional, CMU-CERT: raw CSVs at `../insider-detection/data/r4.2/`.
   Without them the run skips the CERT arms, and Experiment 2 will not match
   the published numbers.

## Run

```bash
git pull
tmux new -s ttg                 # hours long; survives a dropped SSH session
./scripts/testrun.sh --check    # setup + access checks only, a few minutes
./scripts/testrun.sh            # all 13 experiments
./scripts/testrun.sh 3 12       # or a subset, run in the order given
```

`--check` stops with the exact fix if you are not logged in or a dataset's
terms are not accepted. Run it first, so a missing login fails now and not
three hours in.

## What comes out

`results/runs/<timestamp>/` (ignored by git):

| File | What it holds |
|---|---|
| `summary.tsv` | ok / FAILED and seconds, per experiment |
| `compare.txt` | every number that moved more than 0.01 from `results/tables/`, worst first |
| `tables/` | the fresh result JSONs |
| `logs/` | `pytest.log` and one `expN.log` per experiment |
| `meta.txt` | commit, host, GPU, Python, whether CERT was used |

A failed experiment is logged and the run moves on. Differences at the third
decimal are normal across machines; what matters is whether a headline number
moved.

**Read Experiment 3 first.** Its result labels were renamed and the numbers
have never been rerun since; this run is the first real check of them. To
compare two directories by hand:
`.venv/bin/python scripts/compare_results.py results/tables results/runs/<ts>/tables`.

If a fresh run should replace the published numbers, copy its `tables/` over
`results/tables/` and commit that deliberately.
