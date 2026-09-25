#!/usr/bin/env bash
# Fresh end-to-end run of experiments 1-13 on the Ubuntu box, written to its
# own directory and compared against the published results/tables.
#
#   tmux new -s ttg                  # the full run takes hours; survive SSH drops
#   ./scripts/testrun.sh             # all 13 experiments
#   ./scripts/testrun.sh 3 12        # just these
#   ./scripts/testrun.sh --check     # preflight only, run nothing
#
# Output: results/runs/<timestamp>/ with tables/, logs/, meta.txt,
# summary.tsv and compare.txt. The published results/ is never written.
set -uo pipefail
cd "$(dirname "$0")/.."

CHECK_ONLY=0; EXPS=()
for a in "$@"; do
  case "$a" in
    --check) CHECK_ONLY=1 ;;
    *) EXPS+=("$a") ;;
  esac
done
[ ${#EXPS[@]} -eq 0 ] && EXPS=($(seq 1 13))

die() { echo "ERROR: $*" >&2; exit 1; }

# --- Python: 3.10+ required by pyproject ------------------------------------
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' && { PY=$c; break; }
done
[ -n "$PY" ] || die "need Python 3.10+ (sudo apt install python3.11 python3.11-venv)"

if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv with $PY"
  "$PY" -m venv .venv || die "venv failed (sudo apt install python3-venv)"
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[dev]" || die "pip install failed"
PYV=.venv/bin/python

# --- GPU ---------------------------------------------------------------------
$PYV - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__}, CUDA {torch.version.cuda})")
else:
    print("WARNING: torch sees no CUDA device; LSTM-AE and DC will run on CPU and take much longer.")
EOF

# --- Gated HuggingFace datasets: fail now, not three hours in ---------------
$PYV - <<'EOF' || die "fix HuggingFace access above, then re-run"
import sys, tempfile
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
api = HfApi()
try:
    print("HF user:", api.whoami()["name"])
except Exception:
    sys.exit("Not logged in to HuggingFace. Run: .venv/bin/hf auth login")
bad = []
DATA = (".parquet", ".json", ".jsonl", ".csv", ".arrow")
for repo in ("PatronusAI/TRAIL", "PatronusAI/trace-dataset",
             "ai-safety-institute/AgentHarm", "AI45Research/ATBench"):
    # Listing files succeeds even without access to a gated repo; only an
    # actual download proves the terms were accepted.
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
        probe = next((f for f in files if f.endswith(DATA)), files[0])
        hf_hub_download(repo, probe, repo_type="dataset", local_dir=tempfile.mkdtemp())
        print("  ok   ", repo)
    except (GatedRepoError, RepositoryNotFoundError) as e:
        print("  DENIED", repo, "->", type(e).__name__)
        bad.append(repo)
if bad:
    sys.exit("Accept the terms on each denied dataset's page: " +
             ", ".join(f"https://huggingface.co/datasets/{r}" for r in bad))
EOF

# --- CMU-CERT (optional; experiments run without it, minus the CERT arms) ----
CERT_FLAG=""
if [ -d ../insider-detection/data/r4.2 ]; then
  CERT_FLAG="--cert"; echo "CMU-CERT: found, running with --cert"
else
  echo "WARNING: ../insider-detection/data/r4.2 not found; running WITHOUT CMU-CERT."
  echo "         Cross-domain transfer (Exp 2) will not match the published CERT numbers."
fi

[ "$CHECK_ONLY" = 1 ] && { echo "Preflight passed."; exit 0; }

# --- Run ---------------------------------------------------------------------
RUN_DIR="results/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR/logs"
export TTG_RESULTS_DIR="$PWD/$RUN_DIR"
{
  echo "started:  $(date -Iseconds)"
  echo "host:     $(hostname)"
  echo "commit:   $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
  echo "python:   $($PYV --version 2>&1)"
  echo "gpu:      $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo none)"
  echo "cert:     ${CERT_FLAG:-no}"
  echo "exps:     ${EXPS[*]}"
} > "$RUN_DIR/meta.txt"
cat "$RUN_DIR/meta.txt"

echo "Unit tests..."
$PYV -m pytest -q > "$RUN_DIR/logs/pytest.log" 2>&1 \
  && echo "  tests passed" || echo "  TESTS FAILED (see logs/pytest.log); continuing"

printf "exp\tstatus\tseconds\n" > "$RUN_DIR/summary.tsv"
for n in "${EXPS[@]}"; do
  echo "Experiment $n..."
  t0=$(date +%s)
  # Experiments run one per process and in order: Exp 12 reads Exp 3's table.
  if $PYV run_experiments.py --experiment "$n" $CERT_FLAG > "$RUN_DIR/logs/exp$n.log" 2>&1; then
    s=ok; else s=FAILED; fi
  dt=$(( $(date +%s) - t0 ))
  printf "%s\t%s\t%s\n" "$n" "$s" "$dt" >> "$RUN_DIR/summary.tsv"
  echo "  $s in ${dt}s"
done
echo "finished: $(date -Iseconds)" >> "$RUN_DIR/meta.txt"

$PYV scripts/compare_results.py results/tables "$RUN_DIR/tables" > "$RUN_DIR/compare.txt"
echo
column -t "$RUN_DIR/summary.tsv"
echo
head -40 "$RUN_DIR/compare.txt"
echo
echo "Everything is in $RUN_DIR"
