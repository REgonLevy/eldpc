# SPDX-License-Identifier: Apache-2.0
"""Latency measurement: single-shot calls through the public decoder API.

Every timed call goes through ``Decoder.decode_full`` on one sampled syndrome.  Two configurations
are compared in an ABBA schedule of runs (ref, candidate, candidate, ref) so that slow drift of the
machine affects both sides equally while each run sees the application's call cadence between its
own calls.  An A/A experiment -- two handles of the same configuration through the same schedule --
gives the band of ratios the machine produces when nothing differs.

At a non-zero call cadence the loop waits between calls (sleep plus a short busy-wait).  With
keep-warm enabled it calls ``Decoder.keep_warm()`` every 500 us while waiting and stops 50 us
before the next decode, as the library's keep-warm contract requires.

Every timed call's status word is kept: the pooled summary splits the calls into the converged
calls (the ``ST_BP_CONVERGED`` flag) and the slow class (BP did not converge), and records
``slow_fraction``, ``p99_unreliable`` and the median latency of the slow calls.
"""
from __future__ import annotations

import time

import numpy as np

from . import stats

KEEP_WARM_PERIOD_US = 500.0
KEEP_WARM_STOP_US = 50.0
#: Keep-warm is evaluated only above this call cadence; below it the call stream itself holds the
#: clock and the keep-warm tick (period 500 us, stop 50 us before the call) does not fit.
KEEP_WARM_MIN_CADENCE_US = 500.0
#: Wall-clock budget of one ABBA experiment (both sides combined), used to scale the call count at
#: sparse call cadences.
EXPERIMENT_SECONDS = 2.0


def _sleep_until(target_ns: int) -> None:
    while True:
        now = time.perf_counter_ns()
        remaining = target_ns - now
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 1_000_000) / 1e9)
        # short remainder: busy-wait for precision


def _status_word(decoder, result) -> int:
    """The status word of one ``decode_full`` call.

    ``Decoder.decode_full`` returns (correction, observables, status) and also stores the word in
    ``last_status``; a decoder double of a test may return something else and expose no status, in
    which case the call counts as not converged (the conservative side for the slow-class split).
    """
    if isinstance(result, tuple) and len(result) == 3:
        return int(result[2])
    return int(getattr(decoder, "last_status", 0) or 0)


class LatencyMeter:
    """Times single-shot decodes on a fixed pool of syndromes at a fixed call cadence."""

    def __init__(self, syndromes: np.ndarray, cadence_us: float):
        s = np.ascontiguousarray(syndromes, dtype=np.uint8)
        if s.ndim != 2 or s.shape[0] == 0:
            raise ValueError("syndromes must be a non-empty (shots, detectors) array")
        self._shots = s
        self._cadence_ns = int(round(float(cadence_us) * 1000.0))
        if self._cadence_ns < 0:
            raise ValueError("cadence must be >= 0")
        self.cadence_us = float(cadence_us)

    @property
    def syndromes(self) -> np.ndarray:
        return self._shots

    def wait(self, target_ns: int, decoder, keep_warm: bool) -> None:
        if self._cadence_ns <= 0:
            return
        if not keep_warm:
            _sleep_until(target_ns)
            return
        stop_ns = target_ns - int(KEEP_WARM_STOP_US * 1000.0)
        while time.perf_counter_ns() < stop_ns:
            decoder.keep_warm()
            _sleep_until(min(time.perf_counter_ns() + int(KEEP_WARM_PERIOD_US * 1000.0), stop_ns))
        _sleep_until(target_ns)

    def run(self, decoder, calls: int, keep_warm: bool = False, warmup: int = 20,
            start_index: int = 0) -> dict:
        """One timed run of ``calls`` single-shot decodes at the meter's cadence.

        ``start_index`` selects the first syndrome; paired runs use the same offset on both sides
        so they decode exactly the same shots in the same order.
        """
        calls = int(calls)
        if calls < 1:
            raise ValueError("calls must be >= 1")
        n_shots = self._shots.shape[0]
        for i in range(min(warmup, calls)):
            decoder.decode_full(self._shots[(start_index + i) % n_shots])
        samples = np.empty(calls, dtype=np.float64)
        statuses = np.empty(calls, dtype=np.uint64)
        target_ns = time.perf_counter_ns()
        for j in range(calls):
            self.wait(target_ns, decoder, keep_warm)
            syndrome = self._shots[(start_index + j) % n_shots]
            t0 = time.perf_counter_ns()
            result = decoder.decode_full(syndrome)
            t1 = time.perf_counter_ns()
            samples[j] = (t1 - t0) / 1000.0
            statuses[j] = _status_word(decoder, result)
            target_ns = max(target_ns + self._cadence_ns, t1)
        out = stats.latency_summary(samples, statuses)
        out["keep_warm"] = bool(keep_warm)
        out["cadence_us"] = self.cadence_us
        out["_samples"] = samples
        out["_statuses"] = statuses
        return out


def scale_calls(cadence_us: float, max_calls: int) -> int:
    """Call count per side: the full count at dense cadences, fewer at sparse ones."""
    budget_us = EXPERIMENT_SECONDS * 1e6 / 2.0  # both sides share the budget
    if cadence_us <= 0:
        return int(max_calls)
    per_side = int(budget_us / max(1.0, cadence_us))
    return max(50, min(int(max_calls), per_side))


def measure_pair(meter: LatencyMeter, ref_decoder, cand_decoder, calls: int,
                 keep_warm: bool = False, warmup: int = 20) -> dict:
    """ABBA comparison: ref, candidate, candidate, ref; each side gets ``calls`` timed calls.

    The ratio uses the pooled quantiles of the two runs of each side; the two runs of a side also
    give that side's run-to-run drift.
    """
    calls = int(calls)
    if calls < 2:
        raise ValueError("a paired measurement needs at least 2 calls per side")
    half = calls // 2
    halves = [half, calls - half]
    n_shots = meter._shots.shape[0]
    for i in range(min(warmup, calls)):
        ref_decoder.decode_full(meter._shots[i % n_shots])
        cand_decoder.decode_full(meter._shots[i % n_shots])
    ref_runs, cand_runs = [], []
    # Both sides decode the same syndromes per run: run 0 starts at shot 0, run 1 at ``half``.
    schedule = [("ref", 0, 0), ("cand", 0, 0), ("cand", 1, half), ("ref", 1, half)]
    for side, half_index, offset in schedule:
        decoder = ref_decoder if side == "ref" else cand_decoder
        part = meter.run(decoder, halves[half_index], keep_warm=keep_warm, warmup=0,
                         start_index=offset)
        (ref_runs if side == "ref" else cand_runs).append(part)
    ref = _pool(ref_runs)
    cand = _pool(cand_runs)
    for r in ref_runs + cand_runs:
        r.pop("_samples", None)
        r.pop("_statuses", None)
    return {
        "calls_per_side": calls,
        "calls_per_run": halves,
        "keep_warm": bool(keep_warm),
        "cadence_us": meter.cadence_us,
        "schedule": "ref,cand,cand,ref",
        "ref": ref,
        "cand": cand,
        "ratio_p50": stats.ratio(cand["p50_us"], ref["p50_us"]),
        "ratio_p99": stats.ratio(cand["p99_us"], ref["p99_us"]),
        "ref_runs": ref_runs,
        "cand_runs": cand_runs,
        "ref_drift_p50": stats.ratio(ref_runs[1]["p50_us"], ref_runs[0]["p50_us"]),
        "ref_drift_p99": stats.ratio(ref_runs[1]["p99_us"], ref_runs[0]["p99_us"]),
        "cand_drift_p50": stats.ratio(cand_runs[1]["p50_us"], cand_runs[0]["p50_us"]),
        "cand_drift_p99": stats.ratio(cand_runs[1]["p99_us"], cand_runs[0]["p99_us"]),
    }


def _pool(runs: list[dict]) -> dict:
    """Pooled summary of runs: exact quantiles of the union of their samples and status words."""
    samples = np.concatenate([np.asarray(r["_samples"], dtype=np.float64) for r in runs])
    statuses = np.concatenate([np.asarray(r["_statuses"], dtype=np.uint64) for r in runs])
    out = stats.latency_summary(samples, statuses)
    out["pooling"] = "exact: union of the run samples"
    return out


def measure_keep_warm(meter: LatencyMeter, decoder, calls: int) -> dict:
    """ABBA comparison of the same decoder with keep-warm off and on: off, on, on, off.

    The decoder is the same handle in every run; only the keep-warm calls during the idle gaps
    differ.  Returns the pooled summary of each mode and their ratio.
    """
    calls = int(calls)
    if calls < 2:
        raise ValueError("a keep-warm comparison needs at least 2 calls per mode")
    half = calls // 2
    halves = [half, calls - half]
    n_shots = meter.syndromes.shape[0]
    for i in range(min(20, calls)):
        decoder.decode_full(meter.syndromes[i % n_shots])
    runs = {"off": [], "on": []}
    schedule = [("off", 0, 0), ("on", 0, 0), ("on", 1, half), ("off", 1, half)]
    for mode, half_index, offset in schedule:
        part = meter.run(decoder, halves[half_index], keep_warm=(mode == "on"), warmup=0,
                         start_index=offset)
        runs[mode].append(part)
    off = _pool(runs["off"])
    on = _pool(runs["on"])
    for run in runs["off"] + runs["on"]:
        run.pop("_samples", None)
        run.pop("_statuses", None)
    return {
        "calls_per_mode": calls,
        "calls_per_run": halves,
        "schedule": "off,on,on,off",
        "cadence_us": meter.cadence_us,
        "off": off,
        "on": on,
        "ratio_p50": stats.ratio(on["p50_us"], off["p50_us"]),
        "ratio_p99": stats.ratio(on["p99_us"], off["p99_us"]),
        "off_runs": runs["off"],
        "on_runs": runs["on"],
    }


def aa_band(meter: LatencyMeter, ref_decoder, second_decoder, calls: int,
            keep_warm: bool = False, cycles: int = 2, meters=None) -> dict:
    """A/A experiment: the same configuration on two handles through the pair schedule.

    The band is the largest deviation from 1 of the paired ratio of pooled quantiles over
    ``cycles`` repetitions -- the same statistic the candidate-versus-best comparison uses, so
    differences below the band are indistinguishable from measuring the same configuration twice.
    One cycle is returned (and the others summarized) to keep the results file small.

    ``meters`` optionally gives one freshly sampled syndrome pool per cycle (the tuner's latency
    stage does); without it every cycle reuses ``meter``.
    """
    cycles = max(1, int(cycles))
    pairs = [measure_pair((meters[i] if meters is not None else meter),
                          ref_decoder, second_decoder, calls, keep_warm)
             for i in range(cycles)]
    first = pairs[0]
    first["aa_cycles"] = cycles
    first["aa_ratios"] = [{"ratio_p50": p["ratio_p50"], "ratio_p99": p["ratio_p99"]}
                          for p in pairs]
    first["band_p50"] = float(max(abs(p["ratio_p50"] - 1.0) for p in pairs))
    first["band_p99"] = float(max(abs(p["ratio_p99"] - 1.0) for p in pairs))
    first["band_basis"] = ("max deviation of the paired pooled-quantile ratio over the A/A "
                           "cycles; the per-run drift values are diagnostics only")
    return first
