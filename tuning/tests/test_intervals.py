# SPDX-License-Identifier: Apache-2.0
"""The interval math against known values: the beta quantiles are the binomial bounds."""
import numpy as np
import pytest
from scipy.stats import binom

from eldpctune.stats import (clopper_pearson_lower, clopper_pearson_upper, derive_seed)


def test_upper_zero_failures_has_closed_form():
    # P(X <= 0 | n, U) = 0.01  =>  U = 1 - 0.01**(1/n)
    for n in (1, 10, 100, 46052):
        expected = 1.0 - 0.01 ** (1.0 / n)
        assert clopper_pearson_upper(0, n) == pytest.approx(expected, rel=1e-12)


def test_lower_against_binomial_survival():
    for failures, shots in [(1, 100), (5, 1000), (17, 500), (0, 50), (90, 100)]:
        lower = clopper_pearson_lower(failures, shots)
        if failures == 0:
            assert lower == 0.0
            continue
        # P(X >= k | n, lower) = 0.01
        assert binom.sf(failures - 1, shots, lower) == pytest.approx(0.01, abs=1e-10)


def test_upper_against_binomial_cdf():
    for failures, shots in [(0, 100), (1, 100), (50, 100), (99, 100), (3, 131072)]:
        upper = clopper_pearson_upper(failures, shots)
        if failures == shots:
            assert upper == 1.0
            continue
        # P(X <= k | n, upper) = 0.01
        assert binom.cdf(failures, shots, upper) == pytest.approx(0.01, abs=1e-9)


def test_known_values():
    # Values computed independently from the defining binomial equations.
    assert clopper_pearson_upper(0, 100) == pytest.approx(0.0450074139785640)
    assert clopper_pearson_lower(1, 100) == pytest.approx(0.000100498308241668)
    assert clopper_pearson_upper(50, 100) == pytest.approx(0.6192825330930475)


def test_monotonicity():
    assert clopper_pearson_upper(3, 1000) < clopper_pearson_upper(3, 100)
    assert clopper_pearson_upper(3, 1000) < clopper_pearson_upper(4, 1000)
    assert clopper_pearson_lower(3, 1000) > clopper_pearson_lower(2, 1000)
    assert clopper_pearson_lower(3, 1000) < clopper_pearson_lower(3, 100)


def test_edge_and_invalid():
    assert clopper_pearson_upper(100, 100) == 1.0
    assert clopper_pearson_lower(0, 100) == 0.0
    for bad in [(-1, 10), (11, 10)]:
        with pytest.raises(ValueError):
            clopper_pearson_upper(*bad)
        with pytest.raises(ValueError):
            clopper_pearson_lower(*bad)
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 0)


def test_interval_contains_estimate():
    for failures, shots in [(0, 100), (1, 100), (50, 100), (99, 100)]:
        lo = clopper_pearson_lower(failures, shots)
        hi = clopper_pearson_upper(failures, shots)
        assert lo <= failures / shots <= hi


def test_seeds_are_stage_and_value_specific():
    base = 7
    seeds = {
        derive_seed(base, "coarse", "bb72:p=0.001"),
        derive_seed(base, "refine", "bb72:p=0.001"),
        derive_seed(base, "coarse", "bb72:p=0.002"),
        derive_seed(base, "coarse", "bb144xyz:p=0.001"),
        derive_seed(base + 1, "coarse", "bb72:p=0.001"),
    }
    assert len(seeds) == 5
    assert all(0 <= s < 2 ** 63 for s in seeds)
    assert derive_seed(base, "coarse", "x") == derive_seed(base, "coarse", "x")


def test_quantiles():
    from eldpctune.stats import quantiles

    samples = np.arange(1, 101, dtype=np.float64)
    q = quantiles(samples)
    assert q["calls"] == 100
    assert q["min_us"] == 1.0
    assert q["max_us"] == 100.0
    assert q["p50_us"] == pytest.approx(np.percentile(samples, 50))
    assert q["p99_us"] == pytest.approx(np.percentile(samples, 99))
