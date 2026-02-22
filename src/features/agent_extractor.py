"""Extract UBFS features from OpenTelemetry agent traces.

Maps agent behaviour dimensions (tool calls, span sequences,
token usage, error rates) to the Unified Behavioural Feature
Schema. This is the counterpart to cert_extractor.py.

Supported input formats:
    - OpenTelemetry JSON (OpenInference standard, as in TRAIL)
    - TRAIL benchmark dataset rows (HuggingFace format)
    - Raw span dictionaries
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ubfs_schema import (
    FeatureCategory,
    UBFSConfig,
    UBFSVector,
)


def _safe_int(value) -> int:
    """Safely convert to int, returning 0 on failure."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return 0
    return 0


def _parse_iso_duration(dur: str) -> float:
    """Parse ISO 8601 duration like 'PT1M24.635189S' to seconds."""
    import re
    total = 0.0
    m = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", dur
    )
    if m:
        if m.group(1):
            total += int(m.group(1)) * 3600
        if m.group(2):
            total += int(m.group(2)) * 60
        if m.group(3):
            total += float(m.group(3))
    return total


class AgentTraceFeatureExtractor:
    """Extracts UBFS vectors from OpenTelemetry agent traces.

    Each trace is a sequence of spans representing tool calls,
    LLM invocations, and other agent actions. Features are
    extracted at the trace level (one UBFS vector per trace).
    """

    def __init__(self):
        self.config = UBFSConfig()
        self._tool_vocab: Dict[str, int] = {}
        self._baseline_bigrams: Optional[Counter] = None

    def parse_otel_trace(self, trace_data: dict) -> dict:
        """Parse OpenTelemetry trace into structured span list.

        Handles both raw OTel JSON and TRAIL benchmark format.

        Args:
            trace_data: Raw trace dictionary.

        Returns:
            Parsed trace with sorted span list.
        """
        spans = []

        # Handle TRAIL format (spans as list of dicts)
        raw_spans = trace_data.get("spans", [])
        if not raw_spans:
            # Try OpenTelemetry resourceSpans format
            for rs in trace_data.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    raw_spans.extend(ss.get("spans", []))

        for span in raw_spans:
            parsed = {
                "span_id": span.get(
                    "spanId",
                    span.get("span_id", ""),
                ),
                "parent_span_id": span.get(
                    "parentSpanId",
                    span.get("parent_span_id", ""),
                ),
                "name": span.get(
                    "name",
                    span.get("span_name", "unknown"),
                ),
                "start_time": self._parse_time(
                    span.get(
                        "startTimeUnixNano",
                        span.get("start_time",
                                 span.get("timestamp", 0)),
                    )
                ),
                "end_time": self._parse_end_time(span),
                "attributes": self._parse_attributes(
                    span.get(
                        "attributes",
                        span.get("span_attributes", {}),
                    )
                ),
                "status": self._parse_status(
                    span.get(
                        "status",
                        span.get("status_code", "OK"),
                    )
                ),
                "events": span.get("events", []),
            }
            spans.append(parsed)

        spans.sort(key=lambda s: s["start_time"])
        return {
            "trace_id": trace_data.get("trace_id", "unknown"),
            "spans": spans,
        }

    def extract_trace_features(
        self, parsed_trace: dict
    ) -> np.ndarray:
        """Extract UBFS vector from a single parsed trace.

        Args:
            parsed_trace: Output of parse_otel_trace().

        Returns:
            UBFS vector as numpy array (total_dim,).
        """
        spans = parsed_trace["spans"]
        features = np.zeros(
            self.config.total_dim, dtype=np.float32
        )
        slices = self.config.category_slices

        if not spans:
            return features

        # Durations
        durations = [
            max(s["end_time"] - s["start_time"], 0)
            for s in spans
        ]
        total_duration = sum(durations)

        # Classify spans
        tool_spans = [
            s for s in spans
            if self._is_tool_call(s["name"])
        ]
        llm_spans = [
            s for s in spans
            if self._is_llm_call(s["name"])
        ]
        error_spans = [
            s for s in spans if s["status"] != "OK"
        ]

        # TEMPORAL
        s = slices[FeatureCategory.TEMPORAL]
        if spans[0]["start_time"] > 0:
            from datetime import datetime, timezone
            start_dt = datetime.fromtimestamp(
                spans[0]["start_time"], tz=timezone.utc
            )
            features[s.start + 0] = start_dt.hour
        features[s.start + 1] = total_duration
        features[s.start + 2] = 0.0  # Off-schedule: needs config
        features[s.start + 3] = 0.0  # Weekend flag

        # FREQUENCY
        s = slices[FeatureCategory.FREQUENCY]
        features[s.start + 0] = len(tool_spans)
        features[s.start + 1] = len(llm_spans)
        features[s.start + 2] = len(error_spans)
        # Span rate z-score: filled at batch level
        features[s.start + 3] = 0.0

        # VOLUME
        s = slices[FeatureCategory.VOLUME]
        token_counts = []
        for sp in spans:
            attrs = sp["attributes"]
            tokens = (
                _safe_int(attrs.get(
                    "gen_ai.usage.input_tokens", 0))
                + _safe_int(attrs.get(
                    "gen_ai.usage.output_tokens", 0))
                + _safe_int(attrs.get(
                    "llm.token_count.prompt", 0))
                + _safe_int(attrs.get(
                    "llm.token_count.completion", 0))
            )
            token_counts.append(tokens)
        features[s.start + 0] = sum(token_counts)
        # Output artifacts: number of spans with output
        features[s.start + 1] = sum(
            1 for sp in spans
            if sp["attributes"].get("output.value")
        )
        features[s.start + 2] = (
            float(np.std(token_counts)) if token_counts else 0.0
        )

        # SCOPE
        s = slices[FeatureCategory.SCOPE]
        unique_tools = set(sp["name"] for sp in tool_spans)
        features[s.start + 0] = len(unique_tools)
        unique_endpoints = set()
        for sp in spans:
            url = sp["attributes"].get("http.url", "")
            if url:
                unique_endpoints.add(url.split("?")[0])
        features[s.start + 1] = len(unique_endpoints)
        features[s.start + 2] = (
            len(unique_tools) / max(len(tool_spans), 1)
        )

        # SEQUENCE
        s = slices[FeatureCategory.SEQUENCE]
        action_seq = [sp["name"] for sp in spans]
        features[s.start + 0] = self._sequence_entropy(action_seq)
        features[s.start + 1] = self._bigram_novelty(action_seq)
        features[s.start + 2] = self._repetition_rate(action_seq)

        # DEVIATION (batch-level, placeholder)
        s = slices[FeatureCategory.DEVIATION]
        features[s.start + 0] = 0.0
        features[s.start + 1] = 0.0

        # PRIVILEGE
        s = slices[FeatureCategory.PRIVILEGE]
        features[s.start + 0] = 0.0  # Requires permission model

        return features

    def extract_batch(
        self, traces: List[dict]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract UBFS vectors for a batch of traces.

        Args:
            traces: List of raw trace dictionaries.

        Returns:
            Tuple of:
                X: UBFS matrix (n_traces, ubfs_dim)
                entity_ids: List of trace/agent IDs
                timestamps: List of trace start times
        """
        n = len(traces)
        X = np.zeros((n, self.config.total_dim), dtype=np.float32)
        entity_ids = []
        timestamps = []

        parsed_traces = []
        for trace in traces:
            parsed = self.parse_otel_trace(trace)
            parsed_traces.append(parsed)

        # Build tool vocabulary from all traces
        self._build_vocab(parsed_traces)

        for i, parsed in enumerate(parsed_traces):
            X[i] = self.extract_trace_features(parsed)
            entity_ids.append(parsed["trace_id"])
            if parsed["spans"]:
                timestamps.append(
                    str(parsed["spans"][0]["start_time"])
                )
            else:
                timestamps.append("")

        # Batch-level features
        X = self._compute_batch_features(X)

        return X, entity_ids, timestamps

    def _compute_batch_features(
        self, X: np.ndarray
    ) -> np.ndarray:
        """Fill batch-level features (z-scores, deviations)."""
        slices = self.config.category_slices

        # Event rate z-score
        freq_s = slices[FeatureCategory.FREQUENCY]
        total_events = X[:, freq_s.start] + X[:, freq_s.start + 1]
        mean_rate = np.mean(total_events)
        std_rate = np.std(total_events)
        if std_rate > 0:
            X[:, freq_s.start + 3] = (
                (total_events - mean_rate) / std_rate
            )

        # Peer distance (global mean)
        dev_s = slices[FeatureCategory.DEVIATION]
        global_mean = np.mean(X[:, :dev_s.start], axis=0)
        for i in range(len(X)):
            X[i, dev_s.start + 0] = float(
                np.linalg.norm(X[i, :dev_s.start] - global_mean)
            )

        return X

    def _build_vocab(self, traces: List[dict]) -> None:
        """Build tool call vocabulary from training traces."""
        all_names = []
        for trace in traces:
            for span in trace["spans"]:
                all_names.append(span["name"])
        self._tool_vocab = {
            name: i
            for i, name in enumerate(sorted(set(all_names)))
        }
        # Build baseline bigram distribution
        all_bigrams = Counter()
        for trace in traces:
            names = [s["name"] for s in trace["spans"]]
            for a, b in zip(names, names[1:]):
                all_bigrams[(a, b)] += 1
        self._baseline_bigrams = all_bigrams

    def _sequence_entropy(self, sequence: List[str]) -> float:
        """Shannon entropy of action sequence."""
        if not sequence:
            return 0.0
        counts = Counter(sequence)
        total = len(sequence)
        return -sum(
            (c / total) * np.log2(c / total)
            for c in counts.values()
        )

    def _bigram_novelty(self, sequence: List[str]) -> float:
        """Fraction of bigrams not in baseline distribution."""
        if len(sequence) < 2 or self._baseline_bigrams is None:
            return 0.0
        bigrams = list(zip(sequence, sequence[1:]))
        novel = sum(
            1 for bg in bigrams
            if bg not in self._baseline_bigrams
        )
        return novel / len(bigrams)

    def _repetition_rate(self, sequence: List[str]) -> float:
        """Fraction of consecutive duplicate actions."""
        if len(sequence) < 2:
            return 0.0
        repeats = sum(
            1 for a, b in zip(sequence, sequence[1:]) if a == b
        )
        return repeats / (len(sequence) - 1)

    @staticmethod
    def _is_tool_call(name: str) -> bool:
        lower = name.lower()
        return any(
            kw in lower
            for kw in ["tool", "function", "action", "execute"]
        )

    @staticmethod
    def _is_llm_call(name: str) -> bool:
        lower = name.lower()
        return any(
            kw in lower
            for kw in ["llm", "chat", "completion", "generate"]
        )

    @staticmethod
    def _parse_time(value) -> float:
        """Convert time value to seconds (float)."""
        if isinstance(value, (int, float)):
            if value > 1e15:  # nanoseconds
                return value / 1e9
            if value > 1e12:  # milliseconds
                return value / 1e3
            return float(value)
        if isinstance(value, str) and value:
            from datetime import datetime, timezone
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.replace(
                        tzinfo=timezone.utc
                    ).timestamp()
                except ValueError:
                    continue
        return 0.0

    @staticmethod
    def _parse_end_time(span: dict) -> float:
        """Derive end time from span, handling duration."""
        end = span.get(
            "endTimeUnixNano", span.get("end_time", None)
        )
        if end is not None:
            return AgentTraceFeatureExtractor._parse_time(end)
        # TRAIL uses ISO duration like "PT1M24.635189S"
        start = AgentTraceFeatureExtractor._parse_time(
            span.get(
                "startTimeUnixNano",
                span.get("start_time",
                         span.get("timestamp", 0)),
            )
        )
        dur = span.get("duration", "")
        if isinstance(dur, str) and dur.startswith("PT"):
            secs = _parse_iso_duration(dur)
            return start + secs
        if isinstance(dur, (int, float)):
            return start + dur
        return start

    @staticmethod
    def _parse_attributes(attrs) -> dict:
        """Normalise OTel attribute formats."""
        if isinstance(attrs, list):
            # OTel JSON format: list of {key, value} dicts
            result = {}
            for item in attrs:
                key = item.get("key", "")
                val = item.get("value", {})
                if isinstance(val, dict):
                    val = (
                        val.get("intValue")
                        or val.get("stringValue")
                        or val.get("doubleValue")
                        or val.get("boolValue")
                        or ""
                    )
                result[key] = val
            return result
        return attrs if isinstance(attrs, dict) else {}

    @staticmethod
    def _parse_status(status) -> str:
        """Extract status code string."""
        if isinstance(status, dict):
            code = status.get("code", status.get("statusCode", 0))
            if code in (0, "STATUS_CODE_UNSET", "OK", "UNSET"):
                return "OK"
            return "ERROR"
        return "OK"

    def to_ubfs_vectors(
        self,
        X: np.ndarray,
        entity_ids: List[str],
        timestamps: List[str],
    ) -> List[UBFSVector]:
        """Wrap raw arrays into UBFSVector objects."""
        return [
            UBFSVector(
                values=X[i],
                entity_id=entity_ids[i],
                domain="agent",
                timestamp=timestamps[i],
            )
            for i in range(len(X))
        ]
