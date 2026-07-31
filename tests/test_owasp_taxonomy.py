"""Pin the OWASP ASI taxonomy to the published standard.

Nothing else in this repository verifies a claim about an external standard.
That is how six of ten category titles came to be wrong while the file header
correctly cited the December 2025 release: the names were inherited from OWASP's
pre-release "Agentic AI — Threats and Mitigations" material, nothing compared
them to anything, and the wrong labels reached a public README and a preprint.

Same failure class as the "8 of 20 features" claim in the sibling Rust project:
a confident, plausible statement about an outside fact that no test touched.
"""

import json
import re
from pathlib import Path

import pytest

from src.data.atbench_loader import RISK_TO_OWASP
from src.data.synthetic_generator import EXCESSIVE_AGENCY, OWASP_PROFILES
from src.evaluation.owasp_mapper import OWASP_CATEGORIES

REPO = Path(__file__).resolve().parents[1]

# OWASP Top 10 for Agentic Applications, release of 9 December 2025.
# https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/
#
# Written out here independently of the source constant, so that the code is
# checked against the standard rather than against itself.
PUBLISHED = {
    "ASI01": "Agent Goal Hijack",
    "ASI02": "Tool Misuse",
    "ASI03": "Identity & Privilege Abuse",
    "ASI04": "Agentic Supply Chain Vulnerabilities",
    "ASI05": "Unexpected Code Execution",
    "ASI06": "Memory & Context Poisoning",
    "ASI07": "Insecure Inter-Agent Communication",
    "ASI08": "Cascading Failures",
    "ASI09": "Human-Agent Trust Exploitation",
    "ASI10": "Rogue Agents",
}

# Names from the superseded taxonomy. Each is plausible, familiar, and wrong;
# "Excessive Agency" is from the OWASP *LLM* Top 10 and has no ASI equivalent
# at all.
SUPERSEDED = [
    "Excessive Agency",
    "Cascading Hallucinations",
    "Inadequate Sandboxing",
    "Unsafe Code Generation",
    "Supply Chain Compromise",
    "Sensitive Information Disclosure",
    "Model Denial of Service",
    "Multi-Agent Manipulation",
]


def test_categories_match_the_published_list():
    assert OWASP_CATEGORIES == PUBLISHED, (
        "OWASP_CATEGORIES has drifted from the published December 2025 list. "
        "Check identifiers as well as titles — the common error is inheriting "
        "names from the pre-release taxonomy, which puts the right words "
        "against the wrong number."
    )


@pytest.mark.parametrize("identifier", sorted(PUBLISHED))
def test_each_identifier_carries_its_published_title(identifier):
    assert OWASP_CATEGORIES.get(identifier) == PUBLISHED[identifier]


def test_no_synthetic_profile_claims_an_asi_number_it_is_not():
    """A profile may sit outside the taxonomy; it may not misrepresent it.

    The generator also carries distillation sub-profiles and MITRE ATLAS
    techniques, which are deliberately not ASI categories and whose keys do not
    look like one. The invariant is narrower than "everything is an ASI code":
    anything shaped like ASInn must actually be that category.
    """
    for key in OWASP_PROFILES:
        if re.fullmatch(r"ASI\d{2}", key):
            assert key in PUBLISHED, (
                f"synthetic profile {key!r} looks like an ASI identifier "
                "but is not one in the published Top 10"
            )


def test_excessive_agency_does_not_claim_an_asi_identifier():
    assert not re.fullmatch(r"ASI\d{2}", EXCESSIVE_AGENCY), (
        "Excessive Agency is OWASP LLM Top 10, not Agentic Top 10. Giving it an "
        "ASI identifier asserts membership of a taxonomy it is not in."
    )


def test_atbench_risk_sources_map_to_real_identifiers():
    for risk, identifier in RISK_TO_OWASP.items():
        assert identifier in PUBLISHED, (
            f"ATBench risk source {risk!r} maps to {identifier!r}, "
            "which is not in the Top 10"
        )


def test_tool_description_injection_is_supply_chain_not_tool_misuse():
    """The correction that changes a reported number rather than a label.

    Poisoned tool *descriptions* are the canonical MCP tool-poisoning attack,
    which ASI04 covers explicitly. Mapping them to ASI02 put 29 supply-chain
    samples into the tool-misuse bucket alongside 29 genuine ones — an exact
    50/50 split of the result the paper reports as "real ASI02".
    """
    assert RISK_TO_OWASP["tool_description_injection"] == "ASI04"
    assert RISK_TO_OWASP["malicious_tool_execution"] == "ASI02"


def _sources_using_superseded_names():
    """Files that state category names, and should state the published ones."""
    here = Path(__file__).resolve()
    for rel in ("README.md", "src", "tests", "run_experiments.py",
                "generate_figures.py"):
        path = REPO / rel
        if path.is_file():
            yield path
        elif path.is_dir():
            # This file defines the list of superseded names, so scanning it
            # would flag its own definition.
            yield from (p for p in path.rglob("*.py") if p.resolve() != here)


def test_superseded_names_are_not_used_as_labels():
    """A superseded name is allowed only where the text says it is superseded."""
    offenders = []
    for path in _sources_using_superseded_names():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in SUPERSEDED:
            for m in re.finditer(re.escape(name), text):
                # Both directions: a table row may legitimately carry a
                # footnote marker whose explanation follows it.
                window = text[max(0, m.start() - 500):m.start() + 800]
                excused = any(
                    marker in window
                    for marker in ("superseded", "NOT in this taxonomy",
                                   "Threats and Mitigations", "LLM Top 10",
                                   "not an ASI", "SUPERSEDED")
                )
                if not excused:
                    offenders.append(f"{path.relative_to(REPO)}: {name!r}")
    assert not offenders, (
        "superseded OWASP category names used without saying so:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )
