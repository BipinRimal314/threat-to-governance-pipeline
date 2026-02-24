"""Load AgentHarm dataset for decomposition analysis.

AgentHarm (UK AISI, ICLR 2025) contains 440 harmful agent tasks
with tool specifications. We use the target_functions field to
create synthetic traces in two injection modes:

- direct: All malicious tool calls injected at one point
  (maps to ASI01-style sudden hijacking)
- decomposed: Tool calls spread across trace, each padded
  by normal-looking actions (maps to boiling frog escalation)

This tests whether multi-step decomposition reduces behavioural
detection, mirroring the insider threat evasion pattern where
85% of gradual attacks evade detection.

Dataset structure:
    Configs: harmful, harmless_benign, chat
    Splits: test_public (176 harmful), validation (32 harmful)
    Key fields: prompt, target_functions, category,
                detailed_prompt, hint_included
"""

import copy
from typing import Dict, List, Tuple

import numpy as np


def load_agent_harm() -> Dict:
    """Load AgentHarm dataset from HuggingFace.

    Returns:
        Dict with 'tasks' (list of task dicts) and
        'malicious_tools' (sorted unique tool names).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "ai-safety-institute/AgentHarm", "harmful",
        split="test_public",
    )

    all_tools = set()
    tasks = []
    for row in ds:
        target_fns = row.get("target_functions", [])
        if isinstance(target_fns, list) and target_fns:
            all_tools.update(target_fns)
            tasks.append({
                "id": row["id"],
                "category": row.get("category", "unknown"),
                "target_functions": target_fns,
                "detailed_prompt": row.get(
                    "detailed_prompt", False
                ),
            })

    print(f"  AgentHarm: {len(tasks)} tasks, "
          f"{len(all_tools)} unique tools")
    return {
        "tasks": tasks,
        "malicious_tools": sorted(all_tools),
    }


def inject_direct(
    trace: dict,
    tools: List[str],
    rng: np.random.RandomState,
) -> dict:
    """Inject all malicious tools at one point (ASI01 pattern).

    All target function spans are inserted at the trace midpoint,
    creating a sudden burst of novel tool calls. This maps to
    single-turn goal hijacking where malicious intent is
    concentrated in one action cluster.

    Args:
        trace: Base normal trace to perturb.
        tools: List of tool function names to inject.
        rng: Random state.

    Returns:
        Perturbed trace with clustered tool injection.
    """
    perturbed = copy.deepcopy(trace)
    spans = perturbed.get("spans", [])
    if not spans or not tools:
        return perturbed

    insert_pos = max(1, len(spans) // 2)
    for j, fn in enumerate(tools):
        template = spans[min(insert_pos, len(spans) - 1)]
        new_span = copy.deepcopy(template)
        # Prefix with "function_call_" so _is_tool_call() matches
        new_span["name"] = f"function_call_{fn}"
        new_span["span_id"] = f"direct_{j}"
        spans.insert(insert_pos + j, new_span)

    perturbed["spans"] = spans
    return perturbed


def inject_decomposed(
    trace: dict,
    tools: List[str],
    rng: np.random.RandomState,
    disguise_rate: float = 0.7,
) -> dict:
    """Spread malicious tools with partial disguise (boiling frog).

    Injects the SAME number of spans as inject_direct, but:
    - Tools are spread across the trace (not clustered)
    - With probability disguise_rate, each tool's span copies
      an existing span name instead of using its real name

    This models the Anthropic espionage pattern where attackers
    decomposed malicious objectives into 'defensive testing'
    subtasks. Each disguised action looks like normal activity;
    only the undisguised fraction creates a structural anomaly.

    Args:
        trace: Base normal trace to perturb.
        tools: List of tool function names to inject.
        rng: Random state.
        disguise_rate: Fraction of tools disguised as normal
            (default 0.7 = 70% of actions look innocent).

    Returns:
        Perturbed trace with distributed, partially disguised
        tool injection.
    """
    perturbed = copy.deepcopy(trace)
    spans = perturbed.get("spans", [])
    if not spans or not tools:
        return perturbed

    n_tools = len(tools)
    step = max(1, len(spans) // (n_tools + 1))

    offset = 0
    for i, fn in enumerate(tools):
        pos = min(step * (i + 1) + offset, len(spans))
        template = spans[min(pos, len(spans) - 1)]
        new_span = copy.deepcopy(template)
        new_span["span_id"] = f"decomp_{i}"

        if rng.random() >= disguise_rate:
            # Not disguised: use original malicious tool name
            new_span["name"] = f"function_call_{fn}"
        # else: keep template's name (disguised as normal)

        spans.insert(pos, new_span)
        offset += 1

    perturbed["spans"] = spans
    return perturbed


def generate_decomposition_dataset(
    base_traces: List[dict],
    tasks: List[dict],
    seed: int = 42,
) -> Tuple[List[dict], np.ndarray, List[str]]:
    """Generate matched direct vs decomposed attack traces.

    For each AgentHarm task, creates both a direct and decomposed
    version using the same base trace and tools. Normal traces
    are included for training.

    Args:
        base_traces: Normal TRAIL traces.
        tasks: AgentHarm tasks with target_functions.
        seed: Random seed.

    Returns:
        Tuple of (traces, labels, mode_labels) where:
            traces: normal + direct + decomposed traces
            labels: 0=normal, 1=anomalous
            mode_labels: "normal", "direct", or "decomposed"
    """
    rng = np.random.RandomState(seed)

    direct_traces = []
    decomposed_traces = []

    for i, task in enumerate(tasks):
        base = base_traces[rng.randint(0, len(base_traces))]
        tools = task["target_functions"]

        direct_traces.append(inject_direct(base, tools, rng))
        decomposed_traces.append(
            inject_decomposed(base, tools, rng)
        )

    n_normal = len(base_traces)
    n_direct = len(direct_traces)
    n_decomposed = len(decomposed_traces)

    all_traces = (
        list(base_traces) + direct_traces + decomposed_traces
    )
    labels = np.concatenate([
        np.zeros(n_normal, dtype=np.int32),
        np.ones(n_direct, dtype=np.int32),
        np.ones(n_decomposed, dtype=np.int32),
    ])
    mode_labels = (
        ["normal"] * n_normal
        + ["direct"] * n_direct
        + ["decomposed"] * n_decomposed
    )

    # Shuffle
    perm = rng.permutation(len(all_traces))
    all_traces = [all_traces[i] for i in perm]
    labels = labels[perm]
    mode_labels = [mode_labels[i] for i in perm]

    print(f"  Generated {len(all_traces)} traces: "
          f"{n_normal} normal, {n_direct} direct, "
          f"{n_decomposed} decomposed")

    return all_traces, labels, mode_labels
