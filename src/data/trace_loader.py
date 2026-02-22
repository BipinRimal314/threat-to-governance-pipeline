"""Load TRACE benchmark dataset from HuggingFace.

TRACE (PatronusAI/trace-dataset) contains agentic trajectories
labelled for reward hacking. Each row has a JSON-encoded
conversation and a taxonomy label code.

Dataset schema:
    - conversation: JSON string → [{role, content}, ...]
    - label: string → "0" (benign) or taxonomy code like "1.1.1"
    - Some turns have content: None

References:
    https://huggingface.co/datasets/PatronusAI/trace-dataset
"""

import json
from typing import Dict, List

import numpy as np


def _parse_conversation(row: dict) -> dict:
    """Parse a single TRACE dataset row into a trajectory dict."""
    conv_str = row.get("conversation", "[]")
    if isinstance(conv_str, str):
        conversation = json.loads(conv_str)
    else:
        conversation = conv_str

    # Clean None content values
    cleaned = []
    for turn in conversation:
        cleaned.append({
            "role": turn.get("role", "unknown"),
            "content": turn.get("content") or "",
        })

    label_str = str(row.get("label", "0"))
    is_reward_hack = label_str != "0"

    return {
        "conversation": cleaned,
        "label": label_str,
        "is_anomalous": is_reward_hack,
    }


def load_trace_dataset(
    cache_dir: str = None,
) -> Dict[str, object]:
    """Load TRACE dataset from HuggingFace.

    Requires HuggingFace authentication (gated dataset).

    Args:
        cache_dir: Optional cache directory for HuggingFace.

    Returns:
        Dictionary with:
            trajectories: List of trajectory dicts
            labels: numpy array (0=benign, 1=reward hack)
            label_codes: List of taxonomy code strings
    """
    from datasets import load_dataset

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    ds = load_dataset("PatronusAI/trace-dataset", **kwargs)

    # TRACE has a single split (usually "train" or "test")
    split_name = list(ds.keys())[0]
    split = ds[split_name]

    trajectories = []
    labels = []
    label_codes = []

    for row in split:
        parsed = _parse_conversation(row)
        trajectories.append(parsed)
        labels.append(1 if parsed["is_anomalous"] else 0)
        label_codes.append(parsed["label"])

    labels = np.array(labels, dtype=np.int32)
    n_normal = (labels == 0).sum()
    n_anomalous = (labels == 1).sum()
    print(f"  Loaded {len(trajectories)} trajectories "
          f"({n_normal} normal, {n_anomalous} reward hacks)")

    return {
        "trajectories": trajectories,
        "labels": labels,
        "label_codes": label_codes,
    }


def trace_to_otel_format(trajectory: dict) -> dict:
    """Convert a TRACE trajectory to OTel-compatible trace format.

    Maps conversation turns to spans so the AgentTraceFeatureExtractor
    can process them. Each turn becomes a span with:
        - name: "{role}_turn" (e.g., "user_turn", "assistant_turn")
        - duration: proportional to content length
        - attributes: content length, role metadata

    Args:
        trajectory: Single trajectory dict from load_trace_dataset.

    Returns:
        OTel-compatible trace dict with spans list.
    """
    spans = []
    base_time = 1700000000.0  # Fixed reference timestamp

    conversation = trajectory.get("conversation", [])
    current_time = base_time

    for i, turn in enumerate(conversation):
        role = turn.get("role", "unknown")
        content = turn.get("content", "") or ""
        content_len = len(content)

        # Duration proportional to content length
        # (rough proxy for processing time)
        duration = max(0.1, content_len / 1000.0)

        span = {
            "span_id": f"span_{i:04d}",
            "parent_span_id": "" if i == 0 else f"span_{i-1:04d}",
            "name": f"{role}_turn",
            "start_time": current_time,
            "end_time": current_time + duration,
            "span_attributes": {
                "role": role,
                "content_length": content_len,
                "turn_index": i,
            },
            "status_code": "OK",
            "events": [],
        }

        # Mark tool calls if content suggests tool use
        if role == "assistant" and any(
            kw in content.lower()
            for kw in ["function_call", "tool_use", "action"]
        ):
            span["name"] = "tool_execution"
            span["span_attributes"]["tool.name"] = "inferred"

        # Mark LLM calls
        if role == "assistant":
            span["span_attributes"]["gen_ai.usage.output_tokens"] = (
                content_len // 4  # Rough token estimate
            )
        if role == "user":
            span["span_attributes"]["gen_ai.usage.input_tokens"] = (
                content_len // 4
            )

        spans.append(span)
        current_time += duration + 0.01  # Small gap between turns

    return {
        "trace_id": f"trace_{id(trajectory)}",
        "spans": spans,
    }
