"""Load ATBench (Agent Trajectory Safety Benchmark).

ATBench (Shanghai AI Lab, Jan 2026) provides 500 safety-labelled
agent execution trajectories with structured tool calls. Each
trajectory is a multi-turn conversation where agent actions are
JSON objects with tool name and arguments.

Dataset: AI45Research/ATBench on HuggingFace (Apache 2.0)
Paper: arxiv:2601.18491 (AgentDoG framework)

Structure:
    500 trajectories: 250 safe, 250 unsafe
    ~8.97 turns per trajectory, 1575 unique tools
    3-dimensional risk taxonomy for unsafe cases:
        risk_source (8 categories)
        failure_mode (14 categories)
        real_world_harm (10 categories)

OWASP mapping (risk_source → ASI code):
    malicious_user_instruction_or_jailbreak → ASI01
    direct_prompt_injection → ASI01
    indirect_prompt_injection → ASI06
    tool_description_injection → ASI02
    malicious_tool_execution → ASI02
    corrupted_tool_feedback → ASI05
    unreliable_or_misinformation → ASI04
    inherent_agent_failures → ASI10
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

# Maps ATBench risk_source to OWASP ASI categories
RISK_TO_OWASP = {
    "malicious_user_instruction_or_jailbreak": "ASI01",
    "direct_prompt_injection": "ASI01",
    "indirect_prompt_injection": "ASI06",
    "tool_description_injection": "ASI02",
    "malicious_tool_execution": "ASI02",
    "corrupted_tool_feedback": "ASI05",
    "unreliable_or_misinformation": "ASI04",
    "inherent_agent_failures": "ASI10",
}


def load_atbench() -> Dict:
    """Load ATBench dataset from HuggingFace.

    Returns:
        Dict with:
            'trajectories': list of trajectory dicts
            'labels': numpy array (0=safe, 1=unsafe)
            'risk_sources': list of risk_source strings
            'owasp_labels': list of mapped OWASP ASI codes
    """
    from datasets import load_dataset

    ds = load_dataset("AI45Research/ATBench", split="test")

    trajectories = []
    labels = []
    risk_sources = []
    owasp_labels = []

    for row in ds:
        traj = _parse_trajectory(row)
        trajectories.append(traj)
        labels.append(row["label"])

        risk = row.get("risk_source", "")
        risk_sources.append(risk)
        # Only assign OWASP labels to unsafe trajectories.
        # Safe trajectories also have risk_source (describes
        # the scenario), but they should be labelled normal.
        if row["label"] == 1:
            owasp = RISK_TO_OWASP.get(risk, "")
        else:
            owasp = ""
        owasp_labels.append(owasp)

    labels = np.array(labels, dtype=np.int32)

    n_safe = (labels == 0).sum()
    n_unsafe = (labels == 1).sum()
    print(f"  ATBench: {len(trajectories)} trajectories "
          f"({n_safe} safe, {n_unsafe} unsafe)")

    return {
        "trajectories": trajectories,
        "labels": labels,
        "risk_sources": risk_sources,
        "owasp_labels": owasp_labels,
    }


def _parse_trajectory(row: dict) -> dict:
    """Parse ATBench row into OTel-style trace.

    Converts the multi-turn content field into a list of
    spans. Agent action turns become tool-call spans;
    environment responses are stored as span attributes.

    Args:
        row: ATBench dataset row.

    Returns:
        OTel-style trace dict with spans list.
    """
    spans = []
    content = row.get("content", [[]])
    turns = content[0] if content else []

    span_idx = 0
    pending_env_response = None

    for turn in turns:
        role = turn.get("role", "")

        if role == "agent":
            action_str = turn.get("action", "")
            thought = turn.get("thought", "")

            # Parse tool call from action JSON
            tool_name = "unknown_action"
            tool_args = {}

            if action_str:
                # Handle "Complete{...}" sentinel
                if action_str.startswith("Complete"):
                    tool_name = "action_complete"
                    try:
                        json_part = action_str[len("Complete"):]
                        tool_args = json.loads(json_part)
                    except (json.JSONDecodeError, ValueError):
                        pass
                else:
                    try:
                        parsed = json.loads(action_str)
                        tool_name = parsed.get(
                            "name", "unknown_action"
                        )
                        tool_args = parsed.get("arguments", {})
                    except (json.JSONDecodeError, ValueError):
                        tool_name = "action_parse_error"

            # Prefix for _is_tool_call() detection
            if not any(
                kw in tool_name.lower()
                for kw in ["tool", "function", "action", "execute"]
            ):
                span_name = f"tool_call_{tool_name}"
            else:
                span_name = tool_name

            span = {
                "name": span_name,
                "span_id": f"atbench_{span_idx}",
                "parent_span_id": "",
                "start_time": span_idx * 1.0,
                "end_time": span_idx * 1.0 + 0.5,
                "attributes": {},
                "status": "OK",
                "events": [],
            }

            # Add tool arguments as attributes
            if isinstance(tool_args, dict):
                for k, v in tool_args.items():
                    span["attributes"][f"tool.param.{k}"] = str(v)

            # Add thought as attribute
            if thought:
                span["attributes"]["agent.thought"] = (
                    thought[:500]
                )

            spans.append(span)
            span_idx += 1

        elif role == "environment":
            # Attach env response to the last span
            env_content = turn.get("content", "")
            if spans and env_content:
                try:
                    env_data = json.loads(env_content)
                    status = env_data.get("status", "unknown")
                    if status != "success":
                        spans[-1]["status"] = "ERROR"
                except (json.JSONDecodeError, ValueError):
                    pass

    return {
        "trace_id": row.get("conv_id", "unknown"),
        "spans": spans,
    }


def atbench_to_otel(trajectories: List[dict]) -> List[dict]:
    """Convert parsed ATBench trajectories to OTel format.

    This is a pass-through since _parse_trajectory already
    produces OTel-style dicts. Provided for API consistency
    with trace_loader.trace_to_otel_format().
    """
    return trajectories


def load_atbench_features():
    """Load ATBench and extract UBFS features.

    Convenience function that handles the full pipeline:
    load → parse → extract features → normalize.

    Returns:
        Tuple of (X, labels, owasp_labels, data_dict)
    """
    from src.features.agent_extractor import (
        AgentTraceFeatureExtractor,
    )
    from src.features.ubfs_schema import UBFSNormalizer

    data = load_atbench()

    extractor = AgentTraceFeatureExtractor()
    X, ids, ts = extractor.extract_batch(data["trajectories"])

    normalizer = UBFSNormalizer(method="zscore")
    X = normalizer.fit_transform(X)

    return X, data["labels"], data["owasp_labels"], data
