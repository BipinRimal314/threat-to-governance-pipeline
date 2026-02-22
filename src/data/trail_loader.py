"""Load TRAIL benchmark dataset from HuggingFace.

TRAIL (PatronusAI/TRAIL) contains agent execution traces from
GAIA and SWE-bench benchmarks. Each row has a JSON-encoded trace
with nested child_spans and a JSON-encoded labels field.

Dataset schema:
    - trace: JSON string → {"spans": [...], "child_spans": {...}}
    - labels: JSON string → {"errors": [...]}
    - Two splits: "gaia" (117 rows) and "swe_bench" (31 rows)

References:
    https://huggingface.co/datasets/PatronusAI/TRAIL
"""

import json
from typing import Dict, List, Tuple

import numpy as np


def _flatten_spans(span: dict, flat: list) -> None:
    """Recursively flatten nested child_spans into a flat list."""
    flat.append(span)
    for child in span.get("child_spans", []):
        _flatten_spans(child, flat)


def _parse_trace(row: dict) -> dict:
    """Parse a single TRAIL dataset row into a trace dict.

    Handles the JSON-encoded trace field and flattens nested
    child_spans into a flat span list compatible with
    AgentTraceFeatureExtractor.
    """
    trace_str = row.get("trace", "{}")
    if isinstance(trace_str, str):
        trace_data = json.loads(trace_str)
    else:
        trace_data = trace_str

    # Flatten spans (TRAIL nests child_spans inside parent spans)
    raw_spans = trace_data.get("spans", [])
    flat_spans = []
    for span in raw_spans:
        _flatten_spans(span, flat_spans)

    # Also check top-level child_spans dict
    child_spans_dict = trace_data.get("child_spans", {})
    for span_id, children in child_spans_dict.items():
        if isinstance(children, list):
            for child in children:
                _flatten_spans(child, flat_spans)

    # Build OTel-compatible trace
    otel_spans = []
    for span in flat_spans:
        otel_span = {
            "span_id": span.get("span_id", ""),
            "parent_span_id": span.get("parent_span_id", ""),
            "name": span.get("span_name", span.get("name", "unknown")),
            "start_time": span.get("start_time", span.get("timestamp", "")),
            "duration": span.get("duration", ""),
            "span_attributes": span.get("span_attributes", {}),
            "status_code": span.get("status_code", "OK"),
            "events": span.get("events", []),
        }
        otel_spans.append(otel_span)

    trace_id = row.get("trace_id", trace_data.get("trace_id", "unknown"))
    return {
        "trace_id": str(trace_id),
        "spans": otel_spans,
    }


def _parse_labels(row: dict) -> dict:
    """Parse labels field from a TRAIL dataset row.

    Handles trailing commas (common in human-edited JSON) by
    stripping them before parsing.
    """
    labels_str = row.get("labels", "{}")
    if isinstance(labels_str, str):
        try:
            labels_data = json.loads(labels_str)
        except json.JSONDecodeError:
            import re
            # Remove trailing commas before ] or }
            cleaned = re.sub(r",\s*([}\]])", r"\1", labels_str)
            labels_data = json.loads(cleaned)
    else:
        labels_data = labels_str
    return labels_data


def load_trail_dataset(
    cache_dir: str = None,
) -> Dict[str, list]:
    """Load TRAIL dataset from HuggingFace.

    Downloads both splits (gaia + swe_bench) and combines them.
    Requires HuggingFace authentication (gated dataset).

    Args:
        cache_dir: Optional cache directory for HuggingFace.

    Returns:
        Dictionary with:
            traces: List of OTel-compatible trace dicts
            annotations: List of label dicts (one per trace)
            split_ids: List of split names ("gaia" or "swe_bench")
    """
    from datasets import load_dataset

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    ds = load_dataset("PatronusAI/TRAIL", **kwargs)

    traces = []
    annotations = []
    split_ids = []

    for split_name in ds:
        split = ds[split_name]
        for row in split:
            trace = _parse_trace(row)
            labels = _parse_labels(row)
            traces.append(trace)
            annotations.append(labels)
            split_ids.append(split_name)

    print(f"  Loaded {len(traces)} traces "
          f"({sum(1 for s in split_ids if s == 'gaia')} gaia, "
          f"{sum(1 for s in split_ids if s != 'gaia')} swe_bench)")

    return {
        "traces": traces,
        "annotations": annotations,
        "split_ids": split_ids,
    }


def get_trail_labels(annotations: List[dict]) -> np.ndarray:
    """Extract binary labels from TRAIL annotations.

    A trace is labelled positive (1) if it contains any errors.

    Args:
        annotations: List of annotation dicts from load_trail_dataset.

    Returns:
        Binary label array (n_traces,). 1 = has errors.
    """
    labels = np.zeros(len(annotations), dtype=np.int32)
    for i, ann in enumerate(annotations):
        errors = ann.get("errors", [])
        if errors:
            labels[i] = 1
    return labels


def get_trail_error_categories(
    annotations: List[dict],
) -> List[str]:
    """Extract primary error category for each trace.

    Returns the first error category for traces with errors,
    empty string for traces without errors.

    Args:
        annotations: List of annotation dicts.

    Returns:
        List of error category strings.
    """
    categories = []
    for ann in annotations:
        errors = ann.get("errors", [])
        if errors:
            first = errors[0]
            if isinstance(first, dict):
                cat = first.get("category", first.get("type", "unknown"))
            elif isinstance(first, str):
                cat = first
            else:
                cat = "unknown"
            categories.append(str(cat))
        else:
            categories.append("")
    return categories
