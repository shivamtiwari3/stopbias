"""The vectorised signed-rank test must agree with scipy, or the sample sizes it produces are
fiction. Includes the tie-heavy and zero-heavy cases that Silero's 32 ms quantisation creates.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from stopbias.power import ALPHA, _wilcoxon_reject, required_n, simulate_power


def _scipy_p(d: np.ndarray) -> float:
    nz = d[d != 0]
    if nz.size < 6:
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox", method="approx").pvalue)


@pytest.mark.parametrize("kind", ["continuous", "quantised_32ms", "zero_heavy", "shifted"])
def test_agrees_with_scipy_row_by_row(kind):
    rng = np.random.default_rng(3)
    n_rows, n = 300, 40
    if kind == "continuous":
        d = rng.normal(5, 40, size=(n_rows, n))
    elif kind == "quantised_32ms":
        d = np.round(rng.normal(16, 48, size=(n_rows, n)) / 32.0) * 32.0
    elif kind == "zero_heavy":
        d = np.round(rng.normal(0, 12, size=(n_rows, n)) / 32.0) * 32.0
    else:
        d = np.round(rng.normal(64, 40, size=(n_rows, n)) / 16.0) * 16.0

    ours = _wilcoxon_reject(d)
    theirs = np.array([_scipy_p(r) < ALPHA for r in d])
    disagree = int(np.sum(ours != theirs))
    assert disagree == 0, f"{kind}: disagreed with scipy on {disagree}/{n_rows} rows"


def test_tie_correction_actually_matters():
    """If the tie term were dropped the variance would be too large, T-mn/se too small, and the
    test too conservative on quantised data. Assert the correction is not a no-op."""
    rng = np.random.default_rng(5)
    d = np.round(rng.normal(16, 32, size=(400, 40)) / 32.0) * 32.0
    ad = np.abs(d)
    from stopbias.power import _tie_term
    assert float(np.mean(_tie_term(ad))) > 0.0


def test_zero_only_rows_never_reject():
    d = np.zeros((10, 30))
    assert not _wilcoxon_reject(d).any()


def test_tiny_samples_never_reject():
    rng = np.random.default_rng(7)
    d = rng.normal(500, 1.0, size=(50, 5))  # huge effect, n below the threshold
    assert not _wilcoxon_reject(d).any()


def test_power_rises_with_n_and_with_effect():
    rng = np.random.default_rng(11)
    errors = rng.normal(0, 40, size=200)
    small_n = simulate_power(errors, 50.0, 10, rng, n_sim=800)
    big_n = simulate_power(errors, 50.0, 120, rng, n_sim=800)
    assert big_n > small_n
    weak = simulate_power(errors, 10.0, 40, rng, n_sim=800)
    strong = simulate_power(errors, 100.0, 40, rng, n_sim=800)
    assert strong > weak


def test_null_effect_holds_the_nominal_false_positive_rate():
    """With no effect, rejection should sit near alpha. A badly wrong variance shows up here."""
    rng = np.random.default_rng(13)
    errors = rng.normal(0, 40, size=400)
    p = simulate_power(errors, 0.0, 60, rng, n_sim=4000)
    assert p < 0.08, f"false positive rate {p:.3f} exceeds nominal {ALPHA}"


def test_required_n_is_monotone_in_effect_size():
    rng = np.random.default_rng(17)
    errors = rng.normal(0, 40, size=300)
    n_small, _ = required_n(errors, 25.0, rng)
    n_large, _ = required_n(errors, 100.0, rng)
    assert n_small is not None and n_large is not None
    assert n_large <= n_small
