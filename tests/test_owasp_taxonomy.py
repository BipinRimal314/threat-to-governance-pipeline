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
import os
import re
from pathlib import Path

import pytest

from src.data.atbench_loader import ATBENCH_CONFIG, RISK_TO_OWASP
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


# ATBench per-category sample counts, measured 7 Aug 2026 against the
# ATBench500 config after the ASI taxonomy correction (c41bd17).
#
# These are the counts Experiment 12's per-category table is computed over.
# They are pinned because the identical failure the module docstring describes
# is available here in a second form: the published dataset now carries two
# configs, an unnamed load raises rather than defaulting, and the 1000-row
# `ATBench` config would run perfectly cleanly while reporting a different
# population against the same category names.
ATBENCH500_BUCKETS = {
    "ASI01": 61,
    "ASI02": 29,
    "ASI04": 29,
    "ASI06": 73,
    "ASI08": 29,
    "ASI10": 29,
}
ATBENCH500_SAFE = 250


def test_atbench_config_is_pinned():
    """The split every reported number comes from, named explicitly."""
    assert ATBENCH_CONFIG == "ATBench500"


def test_atbench_buckets_are_reachable_from_the_mapping():
    """Every pinned bucket is one the risk-source mapping can produce."""
    assert set(ATBENCH500_BUCKETS) == set(RISK_TO_OWASP.values()), (
        "pinned ATBench buckets and the risk-source mapping disagree; one of "
        "them moved without the other"
    )


def test_atbench_bucket_totals_are_balanced():
    """250 unsafe against 250 safe — the balance Exp 12 scores against."""
    assert sum(ATBENCH500_BUCKETS.values()) == ATBENCH500_SAFE


@pytest.mark.skipif(
    not os.environ.get("TTG_NETWORK_TESTS"),
    reason="downloads ATBench; set TTG_NETWORK_TESTS=1 to run",
)
def test_atbench_download_matches_pinned_buckets():
    """The pinned counts against the live dataset, when asked for."""
    from collections import Counter

    from src.data.atbench_loader import load_atbench

    data = load_atbench()
    counts = Counter(o for o in data["owasp_labels"] if o)
    assert dict(counts) == ATBENCH500_BUCKETS
    assert int((data["labels"] == 0).sum()) == ATBENCH500_SAFE


# ---------------------------------------------------------------------------
# The paper.
#
# The scan above deliberately does not include paper/main.tex: the paper
# reports Excessive Agency as a result under its own name, with a footnote
# saying it is an LLM Top 10 category, and a per-occurrence proximity check
# would flag every one of those legitimate mentions.
#
# What the paper needs instead is the binding check — every ASInn in the text
# must be followed by *its own* title and no other. That is the failure this
# file exists to catch: the numbers were right, the words were right, and six
# of them were against each other's identifiers. main.tex carried those
# pairings for a month after the code was fixed, in an abstract, four tables
# and a figure caption, because nothing here looked at it.

# Abbreviations main.tex uses inside table rows, expanded before matching.
# Without these, "Ctx." and "Excess." are not prefixes of the words they stand
# for and would have to be excused by a looser matcher — and a looser matcher
# accepts "ASI05 Mem. Poisoning", which is the superseded pairing itself.
PAPER_ABBREVIATIONS = {
    "mem": "memory",
    "ctx": "context",
    "excess": "excessive",
}

# ASInn followed by a run of title-shaped words, optionally parenthesised as in
# a figure caption. Stops at the first lowercase word, digit or maths, so
# ordinary prose ("ASI04 and ASI08 are covered") yields no run.
#
# A title word is capitalised and then lowercase. Requiring that second
# lowercase letter is what keeps table headers ("ASI01 & ASI02 & ASI04") and
# model acronyms ("ASI08 (IF: +13.3%)") out: every published title word has
# the shape, and no identifier or model abbreviation does.
_PAPER_TITLE_RUN = re.compile(
    r"ASI(\d\d)\s+\(?((?:(?:[A-Z][a-z][A-Za-z-]*\.?|&)\s*){1,5})"
)


def _plain_tex(tex):
    """Strip the LaTeX that sits between an identifier and its title."""
    tex = tex.replace("\\&", "&").replace("\\ ", " ").replace("~", " ")
    tex = re.sub(r"\\(?:textbf|textit|emph)\{([^}]*)\}", r"\1", tex)
    tex = re.sub(r"\$\^\{?\\dagger\}?\$", " ", tex)
    return tex


def _title_words(identifier):
    return {w.lower() for w in PUBLISHED[identifier].split() if w != "&"}


def _binds(token, identifier):
    """Does this word belong to this category's published title?

    Prefix-compatible in both directions, so "Hijacking" matches "Hijack" and
    the table's "Agentic Supply Chain" matches the full "...Vulnerabilities".
    Abbreviations are expanded first, never prefix-matched.
    """
    token = PAPER_ABBREVIATIONS.get(token, token)
    return any(
        word.startswith(token) or token.startswith(word)
        for word in _title_words(identifier)
    )


def test_paper_binds_each_identifier_to_its_published_title():
    tex = _plain_tex((REPO / "paper" / "main.tex").read_text(encoding="utf-8"))
    offenders = []
    for match in _PAPER_TITLE_RUN.finditer(tex):
        identifier = "ASI" + match.group(1)
        tokens = [
            t.rstrip(".").lower()
            for t in match.group(2).split()
            if t not in ("&",)
        ]
        stray = [t for t in tokens if not _binds(t, identifier)]
        if stray:
            line = tex[:match.start()].count("\n") + 1
            offenders.append(
                f"main.tex:{line}: {identifier} followed by "
                f"{' '.join(tokens)!r} — published title is "
                f"{PUBLISHED[identifier]!r}"
            )
    assert not offenders, (
        "the paper pairs an identifier with a title that is not its own:\n  "
        + "\n  ".join(offenders)
    )


def test_paper_marks_excessive_agency_as_not_an_asi_category():
    """It is reported as a result, so it has to be labelled as an outsider."""
    tex = (REPO / "paper" / "main.tex").read_text(encoding="utf-8")
    if "Excessive Agency" not in tex and "Excess." not in tex:
        pytest.skip("the paper no longer reports Excessive Agency")
    assert "LLM" in tex and "Top 10" in tex, (
        "the paper reports Excessive Agency without anywhere saying it is an "
        "OWASP LLM Top 10 category rather than an ASI one"
    )
    assert not re.search(r"ASI\d\d\s+\(?Excess", _plain_tex(tex)), (
        "Excessive Agency is presented under an ASI identifier; it has none"
    )
