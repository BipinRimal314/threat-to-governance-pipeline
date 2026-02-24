"""Semantic feature extraction using sentence-transformers.

Extracts meaning-aware features from tool call names and
parameters. These 8 features complement the 20 structural
UBFS features by capturing intent signals that behavioral
monitoring misses.

The key insight: ASI02 (Tool Misuse) is structurally
undetectable because the same tools are called with similar
frequency. Semantic features can detect subtle parameter
deviations and goal-action misalignment that structural
features cannot.

Requires: sentence-transformers (~110M params, ~500MB VRAM)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

_MODEL = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts.

    Args:
        texts: List of text strings.

    Returns:
        Embeddings array of shape (n_texts, embed_dim).
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    return model.encode(texts, show_progress_bar=False)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


class SemanticFeatureExtractor:
    """Extracts 8 semantic features per trace.

    Features:
        0: param_embedding_distance — avg cosine distance
           of tool parameters from mean parameter embedding
        1: intent_consistency_score — similarity between
           first turn context and aggregate action embedding
        2: escalation_gradient — rate of embedding drift
           from trace start to end (privilege escalation proxy)
        3: param_anomaly_score — max parameter deviation
           from centroid (outlier detection)
        4: context_tool_alignment — avg similarity between
           consecutive action embeddings (coherence)
        5: goal_drift_score — total embedding displacement
           across turns (goal shift detection)
        6: semantic_novelty — fraction of action embeddings
           far from centroid (novelty in semantic space)
        7: instruction_compliance — uniformity of action
           embeddings (high = repetitive, low = diverse)
    """

    SEMANTIC_DIM = 8

    def __init__(self):
        self._baseline_centroid: Optional[np.ndarray] = None
        self._baseline_param_centroid: Optional[np.ndarray] = None

    def fit_baseline(
        self, traces: List[dict]
    ) -> "SemanticFeatureExtractor":
        """Compute baseline centroids from normal traces.

        Args:
            traces: List of parsed trace dicts (with spans).

        Returns:
            Self for chaining.
        """
        all_names = []
        all_params = []
        for trace in traces:
            spans = trace.get("spans", [])
            for span in spans:
                all_names.append(span.get("name", "unknown"))
                attrs = span.get("attributes",
                    span.get("span_attributes", {}))
                param_str = " ".join(
                    f"{k}={v}" for k, v in attrs.items()
                    if not k.startswith("agent.thought")
                )
                if param_str.strip():
                    all_params.append(param_str[:200])

        if all_names:
            name_embs = _embed_texts(all_names)
            self._baseline_centroid = np.mean(
                name_embs, axis=0
            )
        else:
            self._baseline_centroid = np.zeros(
                384, dtype=np.float32
            )

        if all_params:
            param_embs = _embed_texts(all_params)
            self._baseline_param_centroid = np.mean(
                param_embs, axis=0
            )
        else:
            self._baseline_param_centroid = np.zeros(
                384, dtype=np.float32
            )

        return self

    def extract_trace(self, trace: dict) -> np.ndarray:
        """Extract 8 semantic features from a single trace.

        Args:
            trace: Parsed trace dict with spans.

        Returns:
            Feature array of shape (8,).
        """
        features = np.zeros(self.SEMANTIC_DIM, dtype=np.float32)
        spans = trace.get("spans", [])

        if not spans:
            return features

        # Collect text for embedding
        names = [s.get("name", "unknown") for s in spans]
        param_strs = []
        for s in spans:
            attrs = s.get("attributes",
                s.get("span_attributes", {}))
            p = " ".join(
                f"{k}={v}" for k, v in attrs.items()
                if not k.startswith("agent.thought")
            )
            param_strs.append(p[:200] if p.strip() else "")

        # Embed span names
        name_embs = _embed_texts(names)
        if len(name_embs) == 0:
            return features

        centroid = np.mean(name_embs, axis=0)

        # 0: param_embedding_distance
        non_empty_params = [p for p in param_strs if p]
        if non_empty_params and self._baseline_param_centroid is not None:
            param_embs = _embed_texts(non_empty_params)
            dists = [
                _cosine_distance(e, self._baseline_param_centroid)
                for e in param_embs
            ]
            features[0] = float(np.mean(dists))

        # 1: intent_consistency_score
        # Similarity between first span and overall action centroid
        features[1] = 1.0 - _cosine_distance(
            name_embs[0], centroid
        )

        # 2: escalation_gradient
        # Rate of drift from start to end in embedding space
        if len(name_embs) >= 2:
            half = len(name_embs) // 2
            first_half = np.mean(name_embs[:half], axis=0)
            second_half = np.mean(name_embs[half:], axis=0)
            features[2] = _cosine_distance(
                first_half, second_half
            )

        # 3: param_anomaly_score
        # Max deviation from baseline centroid
        if self._baseline_centroid is not None:
            dists = [
                _cosine_distance(e, self._baseline_centroid)
                for e in name_embs
            ]
            features[3] = float(np.max(dists))

        # 4: context_tool_alignment
        # Avg cosine similarity between consecutive spans
        if len(name_embs) >= 2:
            consec_sims = []
            for i in range(len(name_embs) - 1):
                sim = 1.0 - _cosine_distance(
                    name_embs[i], name_embs[i + 1]
                )
                consec_sims.append(sim)
            features[4] = float(np.mean(consec_sims))

        # 5: goal_drift_score
        # Total cumulative displacement across trace
        if len(name_embs) >= 2:
            total_drift = 0.0
            for i in range(len(name_embs) - 1):
                total_drift += _cosine_distance(
                    name_embs[i], name_embs[i + 1]
                )
            features[5] = total_drift / len(name_embs)

        # 6: semantic_novelty
        # Fraction of actions far from centroid (>0.5 distance)
        if self._baseline_centroid is not None:
            dists = [
                _cosine_distance(e, self._baseline_centroid)
                for e in name_embs
            ]
            features[6] = float(
                np.mean([d > 0.5 for d in dists])
            )

        # 7: instruction_compliance (uniformity)
        # Low std = high uniformity = repetitive actions
        std_norm = float(
            np.mean(np.std(name_embs, axis=0))
        )
        features[7] = std_norm

        return features

    def extract_batch(
        self, traces: List[dict]
    ) -> np.ndarray:
        """Extract semantic features for a batch of traces.

        Args:
            traces: List of parsed trace dicts.

        Returns:
            Feature matrix of shape (n_traces, 8).
        """
        X = np.zeros(
            (len(traces), self.SEMANTIC_DIM),
            dtype=np.float32,
        )
        for i, trace in enumerate(traces):
            X[i] = self.extract_trace(trace)
        return X


def extract_semantic_features(
    traces: List[dict],
    baseline_traces: Optional[List[dict]] = None,
) -> np.ndarray:
    """Convenience function for semantic feature extraction.

    Args:
        traces: All traces to extract features from.
        baseline_traces: Normal traces for baseline computation.
            If None, uses all traces as baseline.

    Returns:
        Feature matrix of shape (n_traces, 8).
    """
    extractor = SemanticFeatureExtractor()
    extractor.fit_baseline(baseline_traces or traces)
    return extractor.extract_batch(traces)
