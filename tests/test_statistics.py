"""Pin the reported statistics against their own degenerate cases.

Experiment 12's Phase C compares synthetic detection (Exp 3) against real
ATBench detection over the categories the two experiments share. After the
ASI taxonomy correction there are three such categories, and the rank
correlation came out at exactly -1.0 for all three models.

``scipy.stats.spearmanr`` reported ``p = 0.0`` for that, which is a division
by zero surfacing as a p-value: its t-approximation is
``t = rho * sqrt((n - 2) / (1 - rho**2))``, and ``1 - rho**2`` is exactly
zero at a perfect correlation. Reported as-is it would have put "rho = -1.0,
p < 0.001" into the paper — an overwhelming significance claim drawn from
three points, at the one value where the approximation is guaranteed to
break.

Same failure class as the taxonomy bug that ``test_owasp_taxonomy.py``
exists for: a confident, plausible, checkable number that nothing compared
against anything.
"""

from itertools import permutations

import pytest
from scipy.stats import spearmanr

from run_experiments import _exact_spearman_p

# The three categories Exp 3 and Exp 12 share, Isolation Forest, measured
# 7 Aug 2026. Synthetic ranks ASI02 lowest; real data ranks it highest.
SHARED_SYNTHETIC = [0.6423, 0.5200, 0.7287]
SHARED_REAL = [0.8433, 0.8711, 0.7645]


def test_perfect_correlation_does_not_report_zero_p():
    """The defect itself: p=0 from a perfect rank correlation."""
    _, asymptotic = spearmanr(SHARED_SYNTHETIC, SHARED_REAL)
    assert asymptotic == 0.0, (
        "scipy no longer underflows to 0 at |rho|=1; the guard below may "
        "no longer be needed, but check before removing it"
    )

    exact, _ = _exact_spearman_p(SHARED_SYNTHETIC, SHARED_REAL)
    assert exact > 0.05, (
        "a perfect rank correlation over three points is not significant"
    )


def test_three_points_cannot_reach_significance():
    """The floor is the point: n=3 admits 6 orderings, 2 of them perfect."""
    _, floor = _exact_spearman_p(SHARED_SYNTHETIC, SHARED_REAL)
    assert floor == pytest.approx(2 / 6)
    assert floor > 0.05, (
        "no arrangement of three points can produce a significant Spearman "
        "p, so a significance claim at n=3 is unsupportable regardless of "
        "which way the data fell"
    )


def test_exact_p_equals_the_permutation_definition():
    """The helper against a directly written enumeration."""
    observed = abs(spearmanr(SHARED_SYNTHETIC, SHARED_REAL).statistic)
    orderings = list(permutations(SHARED_REAL))
    expected = sum(
        1 for perm in orderings
        if abs(spearmanr(SHARED_SYNTHETIC, list(perm)).statistic)
        >= observed - 1e-12
    ) / len(orderings)

    exact, _ = _exact_spearman_p(SHARED_SYNTHETIC, SHARED_REAL)
    assert exact == pytest.approx(expected)


def test_exact_and_asymptotic_agree_where_the_answer_is_not_marginal():
    """Away from the boundary the two tests reach the same decision."""
    clear_signal_x = [1, 2, 3, 4, 5, 6, 7, 8]
    clear_signal_y = [1, 2, 3, 5, 4, 6, 7, 8]
    exact, _ = _exact_spearman_p(clear_signal_x, clear_signal_y)
    assert exact < 0.05
    assert spearmanr(clear_signal_x, clear_signal_y).pvalue < 0.05

    no_signal_x = [1, 2, 3, 4, 5, 6, 7, 8]
    no_signal_y = [5, 2, 8, 1, 7, 3, 6, 4]
    exact, _ = _exact_spearman_p(no_signal_x, no_signal_y)
    assert exact > 0.05
    assert spearmanr(no_signal_x, no_signal_y).pvalue > 0.05


def test_the_approximation_is_anti_conservative_near_the_boundary():
    """Documented because it decides a marginal result the wrong way.

    At n=6 and rho=0.829 the t-approximation reports p=0.042 and the exact
    permutation test reports p=0.058. The approximation calls this
    significant; the exact test does not, and the exact test is right —
    there is no arrangement of six points whose true tail probability is
    what the approximation claims.

    This is the same defect as the n=3 case, in its non-degenerate form:
    the approximation is optimistic at small n, and only becomes an
    outright division by zero at |rho| = 1.
    """
    x = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    y = [0.15, 0.05, 0.35, 0.30, 0.65, 0.55]
    exact, floor = _exact_spearman_p(x, y)
    asymptotic = spearmanr(x, y).pvalue

    assert floor == pytest.approx(2 / 720)
    assert asymptotic < 0.05 < exact, (
        "the boundary disagreement this test documents has moved; recheck "
        "which of the two the paper reports"
    )


def test_large_n_falls_back_rather_than_enumerating():
    """Above the cap it must not attempt n! work."""
    x = list(range(12))
    y = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
    exact, floor = _exact_spearman_p(x, y)
    assert exact == pytest.approx(spearmanr(x, y).pvalue)
    assert floor == 0.0
