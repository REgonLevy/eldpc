# SPDX-License-Identifier: Apache-2.0
"""Latency statistics and the Wilson interval for the LER check.

Quantiles are nearest-rank: for n sorted samples, the q quantile is sample[max(1, ceil(q*n)) - 1]
(no interpolation).  A latency reported at p99 is therefore an observed
sample, not an interpolated value, and is robust to the heavy tail of GPU launch latency.

All latency numbers are in nanoseconds internally and microseconds in the results.
"""
from __future__ import annotations

import math

import numpy as np

QUANTILES = (("p50", 0.50), ("p99", 0.99), ("p99_9", 0.999))


def nearest_rank(sorted_or_unsorted, q: float) -> float:
    a = np.sort(np.asarray(sorted_or_unsorted, dtype=np.float64))
    n = a.size
    if n == 0:
        return float("nan")
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    k = int(math.ceil(q * n - 1e-9))
    k = min(max(k, 1), n)
    return float(a[k - 1])


def summarize_ns(durations_ns) -> dict:
    """p50/p99/p99.9/mean (microseconds) plus counts and the sample mean/extremes."""
    a = np.asarray(durations_ns, dtype=np.float64)
    out = {"shots": int(a.size)}
    if a.size == 0:
        out.update({k: None for k, _ in QUANTILES})
        out.update({"mean_us": None, "min_us": None, "max_us": None})
        return out
    for name, q in QUANTILES:
        out[name + "_us"] = round(nearest_rank(a, q) / 1000.0, 4)
    out["mean_us"] = round(float(a.mean()) / 1000.0, 4)
    out["min_us"] = round(float(a.min()) / 1000.0, 4)
    out["max_us"] = round(float(a.max()) / 1000.0, 4)
    return out


#: A p99 sits on the cliff between the converged calls and the slow class when the slow-class
#: share is within a factor of two of 1 % (0.5 % to 2 %).
SLOW_FRACTION_LOW = 0.005
SLOW_FRACTION_HIGH = 0.02


def slow_class(durations_ns, statuses) -> dict:
    """The slow (non-converged) calls of one measurement.

    ``statuses`` are the status words of eldpc.h: bit 0 is ST_BP_CONVERGED.  ``slow_fraction`` is
    the share of timed calls whose status word lacks the flag.  ``p99_unreliable`` is true exactly
    when that share is in [0.005, 0.02]: at about 1 % slow calls the 99th percentile is one of
    them and moves with how many happened to fall in the sample.  ``slow_p50_us`` is the median
    (nearest rank) of the slow calls.
    """
    a = np.asarray(durations_ns, dtype=np.float64)
    st = np.asarray(statuses, dtype=np.uint64)
    if a.size != st.size:
        raise ValueError("durations and statuses must have the same length")
    out = {"converged_calls": 0, "slow_calls": 0, "slow_fraction": None,
           "p99_unreliable": False, "slow_p50_us": None}
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
        out["slow_p50_us"] = round(nearest_rank(a[slow], 0.50) / 1000.0, 4)
    return out


def summarize_with_slow(durations_ns, statuses) -> dict:
    """``summarize_ns`` plus the slow-class fields of the same calls."""
    out = summarize_ns(durations_ns)
    out.update(slow_class(durations_ns, statuses))
    return out


def delta(a: dict, b: dict) -> dict:
    """a - b for each latency statistic, in microseconds (None when a side is missing)."""
    out = {}
    for key in ["p50_us", "p99_us", "p99_9_us", "mean_us"]:
        va, vb = a.get(key), b.get(key)
        out[key] = None if va is None or vb is None else round(va - vb, 4)
    return out


def wilson_interval(errors: int, shots: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (z = 1.96, 95%)."""
    if shots <= 0:
        return (float("nan"), float("nan"))
    p = errors / shots
    denom = 1.0 + z * z / shots
    centre = (p + z * z / (2 * shots)) / denom
    half = z * math.sqrt(p * (1.0 - p) / shots + z * z / (4.0 * shots * shots)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pooled_stats(per_process: list[dict], key: str = "dur_ns") -> dict:
    """Pool the per-shot samples of several processes and summarize them."""
    parts = [np.asarray(p[key], dtype=np.float64) for p in per_process if len(p.get(key, ())) > 0]
    if not parts:
        return summarize_ns([])
    return summarize_ns(np.concatenate(parts))


def per_process_deltas(per_process_a: list[dict], per_process_b: list[dict]) -> dict:
    """For each process i, stat(a_i) - stat(b_i), for every latency statistic."""
    n = min(len(per_process_a), len(per_process_b))
    rows = []
    for i in range(n):
        sa = summarize_ns(per_process_a[i]["dur_ns"])
        sb = summarize_ns(per_process_b[i]["dur_ns"])
        rows.append(delta(sa, sb))
    return rows


def band(rows: list[dict]) -> dict:
    """Range of a set of per-process differences, per statistic."""
    out = {}
    for key in ["p50_us", "p99_us", "p99_9_us", "mean_us"]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = [round(min(vals), 4), round(max(vals), 4)] if vals else None
    return out


def band_width(band_entry) -> float | None:
    if not band_entry:
        return None
    return round(band_entry[1] - band_entry[0], 4)
