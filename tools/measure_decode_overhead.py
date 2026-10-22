#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the Python per-call overhead of Decoder.decode over the C entry point.

Both sides run in one process, back to back on one warm decoder and the same syndrome, with the
process pinned to one CPU (run it under ``taskset -c N``):

- ``Decoder.decode``: the Python API of the package.
- ``eldpc_decode``: the same C call through ctypes with preallocated buffers, i.e. without the
  Python-side argument checks, buffer allocation and result copying.

The shot is chosen to converge in BP (no OSD), so the C call's duration is stable. Prints one
JSON object with the medians (and other percentiles) in microseconds.

    taskset -c 2 python tools/measure_decode_overhead.py --dem model.dem --calls 20000
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import eldpc
from eldpc._abi import _C_U8P, _C_U32P


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def find_converged_shot(dec, H, priors, seed=12345, tries=20000):
    """A syndrome whose decode converges in BP (no post-processing), or fail.

    Syndromes are sampled from the problem's own priors: each mechanism fires independently and
    the syndrome is H times the error pattern over GF(2).
    """
    import scipy.sparse as sp

    Hc = sp.csc_matrix(H, dtype=np.int32) if sp.issparse(H) else sp.csc_matrix(np.asarray(H), dtype=np.int32)
    rng = np.random.default_rng(seed)
    weights = np.asarray(priors, dtype=np.float64)
    for _ in range(tries):
        error = (rng.random(Hc.shape[1]) < weights).astype(np.int32)
        syndrome = np.asarray(Hc @ error, dtype=np.int32) & 1
        _, _, status = dec.decode_full(syndrome.astype(np.uint8))
        if eldpc.status.converged(status) and not eldpc.status.postprocessed(status):
            return syndrome.astype(np.uint8), int(status)
    raise SystemExit("no converging syndrome found: cannot measure a shot without post-processing")


def timed_samples(fn, calls):
    samples = np.empty(calls, dtype=np.int64)
    start = time.perf_counter_ns
    for i in range(calls):
        t0 = start()
        fn()
        samples[i] = start() - t0
    return samples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dem", required=True, help="stim detector error model file")
    ap.add_argument("--config", default="a03125c3", help="preset name or ABI config string")
    ap.add_argument("--calls", type=int, default=20000)
    ap.add_argument("--out", default="", help="write the JSON result here too")
    args = ap.parse_args(argv)
    if args.calls < 1000:
        ap.error("--calls must be at least 1000")

    text = Path(args.dem).read_text()
    dec = eldpc.Decoder.from_dem(text, config=args.config)
    try:
        H, _, priors = eldpc.dem_to_arrays(text)
        syndrome, status = find_converged_shot(dec, H, priors)
        n, k = dec.num_mechanisms, dec.num_observables
        # Warm up both paths (GPU context, caches, clocks).
        for _ in range(50):
            dec.decode(syndrome)

        # The C side: eldpc_decode straight through ctypes with preallocated buffers.
        front = eldpc._abi.load()
        handle = dec._h._h
        s = np.ascontiguousarray(syndrome, dtype=np.uint8)
        corr = np.zeros(max(n, 1), dtype=np.uint8)
        obs = np.zeros(max(k, 1), dtype=np.uint8)
        st = np.zeros(1, dtype=np.uint32)
        sp = s.ctypes.data_as(_C_U8P)
        cp = corr.ctypes.data_as(_C_U8P)
        op = obs.ctypes.data_as(_C_U8P)
        stp = st.ctypes.data_as(_C_U32P)
        decode_c = front.lib.eldpc_decode

        def c_call():
            decode_c(handle, sp, cp, op, stp)

        py = timed_samples(lambda: dec.decode(syndrome), args.calls)
        c = timed_samples(c_call, args.calls)
        py_us = py / 1000.0
        c_us = c / 1000.0
        result = {
            "probe": "decoder.decode over the C entry point",
            "dem": str(args.dem),
            "config": args.config,
            "config_string": dec.config_string,
            "num_detectors": int(dec.num_detectors),
            "num_mechanisms": int(dec.num_mechanisms),
            "num_observables": int(dec.num_observables),
            "shot_status": status,
            "calls": int(args.calls),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu": cpu_model(),
            "gpu": [d["name"] for d in eldpc.devices()],
            "isa": eldpc.isa(),
            "python": sys.version.split()[0],
            "median_py_us": float(statistics.median(py_us)),
            "median_c_us": float(statistics.median(c_us)),
            "median_overhead_us": float(statistics.median(py_us) - statistics.median(c_us)),
            "p10_py_us": float(np.percentile(py_us, 10)),
            "p90_py_us": float(np.percentile(py_us, 90)),
            "p10_c_us": float(np.percentile(c_us, 10)),
            "p90_c_us": float(np.percentile(c_us, 90)),
            "mean_py_us": float(np.mean(py_us)),
            "mean_c_us": float(np.mean(c_us)),
        }
        line = json.dumps(result)
        print(line)
        if args.out:
            Path(args.out).write_text(line + "\n")
        return 0
    finally:
        dec.close()


if __name__ == "__main__":
    sys.exit(main())
