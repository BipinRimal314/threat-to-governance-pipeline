"""Results API for the Threat-to-Governance Pipeline.

Serves experiment results as a REST API. Designed to run as a
container service alongside the experiment runner.

Endpoints:
    GET /health              — Service health check
    GET /experiments         — List all completed experiments
    GET /experiments/{id}    — Full results for one experiment
    GET /summary             — Cross-experiment summary table
    GET /owasp-matrix        — OWASP detection matrix (Exp 3)
    GET /transfer-matrix     — Cross-domain transfer results (Exp 2)
    GET /models              — Model comparison across all experiments
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Threat-to-Governance Pipeline API",
    description=(
        "Cross-domain threat transfer: insider threat detection models "
        "applied to AI agent governance. 11 experiments, 3 models, "
        "5 random seeds."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

RESULTS_DIR = Path("/app/results/tables")

EXPERIMENT_NAMES = {
    1: "Within-Domain Baselines",
    2: "Cross-Domain Transfer",
    3: "OWASP Detection Matrix",
    4: "Governance Assumption Audit",
    5: "Distillation Attack Spectrum",
    6: "Decomposition Evasion",
    7: "MCP Transfer Generalisation",
    8: "Hybrid Detection (UBFS-28)",
    9: "Adversarial Evasion Testing",
    10: "Temporal Window Ablation",
    11: "MITRE ATLAS Mapping",
}


def _load_result(experiment_id: int) -> Optional[dict]:
    """Load a single experiment's results from disk."""
    pattern = f"experiment_{experiment_id}_*.json"
    files = list(RESULTS_DIR.glob(pattern))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


@app.get("/health")
def health():
    """Service health check."""
    completed = [
        eid for eid in EXPERIMENT_NAMES
        if list(RESULTS_DIR.glob(f"experiment_{eid}_*.json"))
    ]
    return {
        "status": "healthy",
        "experiments_completed": len(completed),
        "experiments_total": len(EXPERIMENT_NAMES),
    }


@app.get("/experiments")
def list_experiments():
    """List all experiments with completion status."""
    experiments = []
    for eid, name in EXPERIMENT_NAMES.items():
        files = list(RESULTS_DIR.glob(f"experiment_{eid}_*.json"))
        experiments.append({
            "id": eid,
            "name": name,
            "completed": len(files) > 0,
            "file": files[0].name if files else None,
        })
    return {"experiments": experiments}


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: int):
    """Full results for a specific experiment."""
    if experiment_id not in EXPERIMENT_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {experiment_id} not found. "
                   f"Valid: {list(EXPERIMENT_NAMES.keys())}",
        )
    result = _load_result(experiment_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {experiment_id} has not been run yet.",
        )
    return {
        "id": experiment_id,
        "name": EXPERIMENT_NAMES[experiment_id],
        "results": result,
    }


@app.get("/summary")
def experiment_summary():
    """Cross-experiment summary with key metrics."""
    summary = {}
    for eid, name in EXPERIMENT_NAMES.items():
        result = _load_result(eid)
        if result is None:
            summary[eid] = {"name": name, "status": "not_run"}
            continue
        summary[eid] = {
            "name": name,
            "status": "completed",
            "key_metrics": _extract_key_metrics(eid, result),
        }
    return {"summary": summary}


@app.get("/owasp-matrix")
def owasp_matrix():
    """OWASP ASI detection matrix from Experiment 3."""
    result = _load_result(3)
    if result is None:
        raise HTTPException(404, "Experiment 3 not yet completed.")
    return {"owasp_detection_matrix": result}


@app.get("/transfer-matrix")
def transfer_matrix():
    """Cross-domain transfer results from Experiment 2."""
    result = _load_result(2)
    if result is None:
        raise HTTPException(404, "Experiment 2 not yet completed.")
    return {"transfer_results": result}


@app.get("/models")
def model_comparison():
    """Compare model performance across all completed experiments."""
    models = {}
    for eid in EXPERIMENT_NAMES:
        result = _load_result(eid)
        if result is None:
            continue
        # Extract per-model results where available
        for key, val in result.items():
            if isinstance(val, dict) and "auc_roc_mean" in str(val):
                if key not in models:
                    models[key] = {}
                models[key][f"experiment_{eid}"] = val
    return {"model_comparison": models}


def _extract_key_metrics(eid: int, result: dict) -> dict:
    """Pull the most important metric from each experiment."""
    if eid == 2:
        # Transfer retention percentage
        for key, val in result.items():
            if "retention" in str(val).lower():
                return {"transfer": val}
    if eid == 3:
        # ASI02 blind spot
        for key, val in result.items():
            if "ASI02" in str(key):
                return {"ASI02_auc": val}
    # Default: return first few keys
    keys = list(result.keys())[:3]
    return {k: result[k] for k in keys if not isinstance(result[k], dict)}
