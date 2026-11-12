"""Does the deep-clustering detector's score depend on the device? (paper/short, Limitations)

In-sample ATBench protocol (Exp 12), mean of five seeds, on every device this machine has.
Writes results/tables/device_check_<platform>.json.
Run: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/device_check.py
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_experiments import SEEDS, AgentTraceFeatureExtractor, UBFSNormalizer, DeepClusteringDetector, compute_metrics  # noqa: E402
from src.data.atbench_loader import load_atbench  # noqa: E402

CATS = ["ASI01", "ASI02", "ASI10"]

d = load_atbench()
Xraw, _, _ = AgentTraceFeatureExtractor().extract_batch(d["trajectories"])
X = UBFSNormalizer(method="zscore").fit_transform(Xraw)
owasp = np.array(d["owasp_labels"])
normal = np.where(owasp == "")[0]
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])
out = {"torch": torch.__version__, "platform": platform.platform(), "protocol": "in-sample, Exp 12, UBFS-20, mean of five seeds", "auc_roc": {}}
for dev in devices:
    scores = {s: None for s in SEEDS}
    for s in SEEDS:
        m = DeepClusteringDetector(pretrain_epochs=30, batch_size=16, seed=s, device=dev)
        m.fit(X[normal])
        scores[s] = m.score(X)
    out["auc_roc"][dev] = {}
    for cat in CATS:
        pos = np.where(owasp == cat)[0]
        idx, y = np.r_[normal, pos], np.r_[np.zeros(len(normal)), np.ones(len(pos))]
        out["auc_roc"][dev][cat] = round(float(np.mean([compute_metrics(y, scores[s][idx]).auc_roc for s in SEEDS])), 3)
    print(dev, out["auc_roc"][dev], flush=True)
name = "device_check_" + ("macos" if platform.system() == "Darwin" else platform.system().lower()) + ".json"
json.dump(out, open(ROOT / "results" / "tables" / name, "w"), indent=2)
