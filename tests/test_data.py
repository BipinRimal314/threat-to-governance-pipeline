"""Tests for data loaders (using synthetic data, no HF needed)."""

import numpy as np
import pytest

from src.data.trail_loader import (
    _flatten_spans,
    _parse_trace,
    _parse_labels,
    get_trail_labels,
    get_trail_error_categories,
)
from src.data.trace_loader import (
    _parse_conversation,
    trace_to_otel_format,
)
from src.data.synthetic_generator import (
    generate_anomalous_traces,
    OWASP_PROFILES,
)


class TestTrailLoader:
    """Tests for TRAIL data parsing (no HuggingFace needed)."""

    def test_flatten_spans_simple(self):
        span = {"name": "root", "child_spans": [
            {"name": "child1", "child_spans": []},
            {"name": "child2", "child_spans": [
                {"name": "grandchild", "child_spans": []},
            ]},
        ]}
        flat = []
        _flatten_spans(span, flat)
        assert len(flat) == 4
        assert flat[0]["name"] == "root"
        assert flat[-1]["name"] == "grandchild"

    def test_parse_trace_basic(self):
        import json
        row = {
            "trace": json.dumps({
                "spans": [
                    {"span_name": "tool_call_1", "start_time": "2024-01-01T00:00:00Z"},
                    {"span_name": "tool_call_2", "start_time": "2024-01-01T00:01:00Z"},
                ],
            }),
            "trace_id": "test_trace",
        }
        parsed = _parse_trace(row)
        assert parsed["trace_id"] == "test_trace"
        assert len(parsed["spans"]) == 2

    def test_parse_labels(self):
        import json
        row = {"labels": json.dumps({"errors": ["timeout", "wrong_answer"]})}
        labels = _parse_labels(row)
        assert len(labels["errors"]) == 2

    def test_get_trail_labels(self):
        annotations = [
            {"errors": ["e1"]},
            {"errors": []},
            {"errors": ["e1", "e2"]},
            {},
        ]
        labels = get_trail_labels(annotations)
        assert labels.shape == (4,)
        np.testing.assert_array_equal(labels, [1, 0, 1, 0])

    def test_get_trail_error_categories(self):
        annotations = [
            {"errors": [{"category": "timeout"}]},
            {"errors": []},
            {"errors": ["wrong_answer"]},
        ]
        cats = get_trail_error_categories(annotations)
        assert cats[0] == "timeout"
        assert cats[1] == ""
        assert cats[2] == "wrong_answer"


class TestTraceLoader:
    """Tests for TRACE data parsing."""

    def test_parse_conversation_benign(self):
        import json
        row = {
            "conversation": json.dumps([
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]),
            "label": "0",
        }
        parsed = _parse_conversation(row)
        assert len(parsed["conversation"]) == 2
        assert not parsed["is_anomalous"]

    def test_parse_conversation_reward_hack(self):
        import json
        row = {
            "conversation": json.dumps([
                {"role": "user", "content": "Do X"},
                {"role": "assistant", "content": None},
            ]),
            "label": "1.1.1",
        }
        parsed = _parse_conversation(row)
        assert parsed["is_anomalous"]
        assert parsed["conversation"][1]["content"] == ""

    def test_trace_to_otel_format(self):
        trajectory = {
            "conversation": [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hi there, how can I help?"},
                {"role": "user", "content": "Do something"},
            ],
        }
        otel = trace_to_otel_format(trajectory)
        assert "trace_id" in otel
        assert len(otel["spans"]) == 3
        assert otel["spans"][0]["name"] == "user_turn"
        assert otel["spans"][1]["name"] == "assistant_turn"

    def test_otel_format_has_timestamps(self):
        trajectory = {
            "conversation": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"},
            ],
        }
        otel = trace_to_otel_format(trajectory)
        for span in otel["spans"]:
            assert span["start_time"] > 0
            assert span["end_time"] > span["start_time"]


class TestSyntheticGenerator:
    """Tests for OWASP anomaly injection."""

    @pytest.fixture
    def base_traces(self):
        """Create minimal normal traces for testing."""
        return [
            {
                "trace_id": f"normal_{i}",
                "spans": [
                    {
                        "name": "llm_call",
                        "span_id": f"s{i}_0",
                        "parent_span_id": "",
                        "start_time": 1700000000.0 + i,
                        "end_time": 1700000001.0 + i,
                        "span_attributes": {"gen_ai.usage.output_tokens": 100},
                        "status_code": "OK",
                        "events": [],
                    },
                    {
                        "name": "tool_call_search",
                        "span_id": f"s{i}_1",
                        "parent_span_id": f"s{i}_0",
                        "start_time": 1700000001.0 + i,
                        "end_time": 1700000002.0 + i,
                        "span_attributes": {},
                        "status_code": "OK",
                        "events": [],
                    },
                ],
            }
            for i in range(20)
        ]

    def test_generate_returns_correct_types(self, base_traces):
        mixed, labels, cats = generate_anomalous_traces(
            base_traces, anomaly_ratio=0.3, seed=42
        )
        assert isinstance(mixed, list)
        assert isinstance(labels, np.ndarray)
        assert isinstance(cats, list)
        assert len(mixed) == len(labels) == len(cats)

    def test_anomaly_ratio(self, base_traces):
        mixed, labels, cats = generate_anomalous_traces(
            base_traces, anomaly_ratio=0.3, seed=42
        )
        actual_ratio = labels.sum() / len(labels)
        assert 0.15 < actual_ratio < 0.45  # Approximate

    def test_owasp_categories_present(self, base_traces):
        mixed, labels, cats = generate_anomalous_traces(
            base_traces, anomaly_ratio=0.3, seed=42
        )
        anomalous_cats = [c for c in cats if c]
        for cat in OWASP_PROFILES:
            assert cat in anomalous_cats

    def test_specific_category(self, base_traces):
        mixed, labels, cats = generate_anomalous_traces(
            base_traces, anomaly_ratio=0.3,
            categories=["ASI01"], seed=42
        )
        anomalous_cats = [c for c in cats if c]
        assert all(c == "ASI01" for c in anomalous_cats)

    def test_asi02_minimal_perturbation(self, base_traces):
        """ASI02 (Tool Misuse) should produce minimal structural change."""
        mixed, labels, cats = generate_anomalous_traces(
            base_traces, anomaly_ratio=0.5,
            categories=["ASI02"], seed=42
        )
        for trace, label, cat in zip(mixed, labels, cats):
            if cat == "ASI02":
                # Span count should be same as originals
                assert len(trace["spans"]) == 2

    def test_reproducibility(self, base_traces):
        m1, l1, c1 = generate_anomalous_traces(
            base_traces, seed=42
        )
        m2, l2, c2 = generate_anomalous_traces(
            base_traces, seed=42
        )
        np.testing.assert_array_equal(l1, l2)
        assert c1 == c2
