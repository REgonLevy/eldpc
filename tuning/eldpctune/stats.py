# SPDX-License-Identifier: Apache-2.0
"""Statistics used by the knob tuner.

Everything here is deterministic: the same inputs give the same outputs.  The interval functions
implement the one-sided 99% Clopper-Pearson bounds of a binomial proportion.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.stats import beta as _beta

#: Confidence of the one-sided intervals used for every accuracy decision.
CONFIDENCE = 0.99


def clopper_pearson_upper(failures: int, shots: int, confidence: float = CONFIDENCE) -> float:
    """One-sided upper Clopper-Pearson bound for the failure probability.

    The returned p satisfies P(X <= failures | n=shots, p) = 1 - confidence; it is the smallest
    value of the true failure probability that the data still supports at the given confidence.
    """
    failures = int(failures)
    shots = int(shots)
    if shots <= 0:
        raise ValueError("shots must be >= 1")
    if failures < 0 or failures > shots:
        raise ValueError("failures must be in [0, shots]")
    if failures >= shots:
        return 1.0
    return float(_beta.ppf(confidence, failures + 1, shots - failures))


def clopper_pearson_lower(failures: int, shots: int, confidence: float = CONFIDENCE) -> float:
    """One-sided lower Clopper-Pearson bound for the failure probability.

    The returned p satisfies P(X >= failures | n=shots, p) = 1 - confidence.
    """
    failures = int(failures)
    shots = int(shots)
    if shots <= 0:
        raise ValueError("shots must be >= 1")
    if failures < 0 or failures > shots:
        raise ValueError("failures must be in [0, shots]")
    if failures <= 0:
        return 0.0
    return float(_beta.ppf(1.0 - confidence, failures, shots - failures + 1))


def clamp_probability(p: float, floor: float = 0.0) -> float:
    """Clamp a probability into [floor, 1.0] (used only for display and log scores)."""
    return float(min(1.0, max(floor, p)))


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stage seed from the run seed and a tuple of stage identifiers.

    Each stage uses its own derived seed, so shot sets are never shared across stages even when the
    identifiers repeat between runs.  The result fits in 63 bits so it is safe for every sampler.
    """
    h = hashlib.sha256()
    h.update(b"eldpc-tune-seed-v1")
    h.update(str(int(base_seed)).encode("ascii"))
    for part in parts:
        h.update(b"\x00")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little") & ((1 << 63) - 1)


def quantiles(samples_us: np.ndarray) -> dict:
    """Latency summary of one measurement: count, min, mean, p50, p99, max (microseconds).

    Quantiles are numpy's default linear interpolation.  The method string is recorded next to the
    numbers so a reader can reproduce them.
    """
    a = np.asarray(samples_us, dtype=np.float64)
    if a.size == 0:
        raise ValueError("no latency samples")
    return {
        "calls": int(a.size),
        "min_us": float(a.min()),
        "mean_us": float(a.mean()),
        "p50_us": float(np.percentile(a, 50)),
        "p99_us": float(np.percentile(a, 99)),
        "max_us": float(a.max()),
        "quantile_method": "numpy.percentile linear",
    }


#: A p99 is taken to sit on the cliff between the converged calls and the slow class when the
#: slow-class share is within a factor of two of 1 % (0.5 % to 2 %).
SLOW_FRACTION_LOW = 0.005
SLOW_FRACTION_HIGH = 0.02


def slow_class(samples_us, statuses) -> dict:
    """Split one latency measurement into its converged and slow (non-converged) calls.

    ``statuses`` are the status words of eldpc.h (bit 0 is ST_BP_CONVERGED).  ``slow_fraction`` is
    the share of timed calls whose status word lacks that flag.  ``p99_unreliable`` is true exactly
    when that share lies in [0.005, 0.02]: at a slow share of about 1 % the 99th percentile is one
    of the slow calls and moves with how many happened to fall in the sample, so it is not a stable
    number; ``slow_median_us`` is the median of the slow calls, the latency a p99 budget should be
    read against in that regime.
    """
    a = np.asarray(samples_us, dtype=np.float64)
    st = np.asarray(statuses, dtype=np.uint64)
    if a.size != st.size:
        raise ValueError("samples and statuses must have the same length")
    out = {"converged_calls": 0, "slow_calls": 0, "slow_fraction": None,
           "p99_unreliable": False, "slow_median_us": None}
    if a.size == 0:
        return out
    converged = (st & np.uint64(0x0001)) != 0
    slow = ~converged
    fraction = float(np.count_nonzero(slow)) / float(a.size)
    out["converged_calls"] = int(np.count_nonzero(converged))
    out["slow_calls"] = int(np.count_nonzero(slow))
    out["slow_fraction"] = fraction
    out["p99_unreliable"] = bool(SLOW_FRACTION_LOW <= fraction <= SLOW_FRACTION_HIGH)
    if out["slow_calls"]:
        out["slow_median_us"] = float(np.percentile(a[slow], 50))
    return out


def latency_summary(samples_us, statuses) -> dict:
    """The quantiles of a measurement plus its slow-class fields."""
    out = quantiles(samples_us)
    out.update(slow_class(samples_us, statuses))
    return out


def ratio(num: float, den: float) -> float:
    if den == 0:
        return float("inf") if num > 0 else 1.0
    return float(num / den)


def log10_or_floor(value: float, floor: float = 1e-300) -> float:
    return math.log10(max(float(value), floor))
