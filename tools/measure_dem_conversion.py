#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the detector-error-model -> (H, O, priors) conversion on a DEM file.

    python tools/measure_dem_conversion.py --dem model.dem [--repeats 3]

Prints one JSON object with the median and minimum wall time of ``eldpc.dem_to_arrays`` over the
model, its dimensions, and the size of the model. CPU only: no decoder is created.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import eldpc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dem", required=True, help="stim detector error model file")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.repeats < 1:
        ap.error("--repeats must be at least 1")

    import stim

    text = Path(args.dem).read_text()
    dem = stim.DetectorErrorModel(text)
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        H, O, priors = eldpc.dem_to_arrays(dem)
        times.append(time.perf_counter() - t0)
    result = {
        "dem": str(args.dem),
        "dem_text_bytes": len(text),
        "num_detectors": int(H.shape[0]),
        "num_mechanisms": int(H.shape[1]),
        "num_observables": int(O.shape[0]),
        "nnz_h": int(H.nnz),
        "nnz_o": int(O.nnz),
        "repeats": int(args.repeats),
        "conversion_ms_median": float(statistics.median(times) * 1000.0),
        "conversion_ms_min": float(min(times) * 1000.0),
        "conversion_ms_max": float(max(times) * 1000.0),
    }
    line = json.dumps(result)
    print(line)
    if args.out:
        Path(args.out).write_text(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
