#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the create-time cost of the per-handle core-variant choice.

A decoder handle gets its core variant at ``eldpc_create()`` time: ``ELDPC_DEVICE_CLASS`` when it
is set, otherwise the device-class table, whose input (the target device's compute capability and
SM count) is read through the CUDA driver API and cached per process. This tool separates the two
costs of that design:

  (a) the one-time device query on the first create without the override -- ``dlopen`` of
      ``libcuda.so.1``, ``cuInit``, ``cuDeviceGet`` and three ``cuDeviceGetAttribute`` calls --
      measured as the paired difference between fresh processes that create their first decoder
      with ``ELDPC_DEVICE_CLASS=default`` (no query) and without it (the query);
  (b) the marginal cost of later creates in one process, where the device properties are cached,
      measured by repeated create/destroy cycles with and without the override.

Both quantities include everything else a create does (loading the front, the decoder core's own
CUDA initialisation, its allocation); only the difference between the two modes is attributed to
the variant choice. ``ELDPC_DEVICE_CLASS=ada64`` is measured as well, because the first ada64
handle also loads the ada64 core library. Finally, the same driver-query call sequence is timed in
isolation in fresh processes, twice each: the first call pays the driver's initialisation and the
second shows what a cached driver costs. That is the query's own price; in the package it is paid
before the core's CUDA initialisation, which then finds the driver already up.

    bash scripts/gpu_run.sh --timeout 600 python tools/measure_create_cost.py \
        --pairs 15 --repeats 40 --out tools/results/create-cost.json

Prints one JSON object; ``--out`` writes it to a file too. Run it under the machine's GPU lock: the
child processes create decoders and use the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

# The child process: import the package, create one decoder (or a series), time each create.
CHILD = r'''
import json
import sys
import time

import numpy as np

import eldpc

mode = sys.argv[1]
repeats = int(sys.argv[2])
H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)


def create():
    t0 = time.perf_counter_ns()
    dec = eldpc.Decoder(H, priors=[0.1, 0.1, 0.1], config="ms30")
    elapsed = time.perf_counter_ns() - t0
    variant = dec.core_variant
    dec.close()
    return elapsed, variant


if mode == "first":
    elapsed, variant = create()
    print(json.dumps({"first_ns": elapsed, "variant": variant, "isa": eldpc.isa()}))
elif mode == "repeat":
    runs = [create() for _ in range(repeats)]
    print(json.dumps({"times_ns": [r[0] for r in runs], "variant": runs[0][1],
                      "isa": eldpc.isa()}))
else:
    raise SystemExit("unknown child mode " + mode)
'''

# The driver-query child: the exact call sequence the front runs for the device class, through
# ctypes, timed twice in one fresh process (the first pays the driver's initialisation, the second
# is the cached-driver cost). This isolates the query from the decoder core's own CUDA startup.
QUERY_CHILD = r'''
import ctypes
import json
import time


def query():
    t0 = time.perf_counter_ns()
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit.restype = ctypes.c_int
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuDeviceGet.restype = ctypes.c_int
    lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.cuDeviceGetAttribute.restype = ctypes.c_int
    lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    rc = lib.cuInit(0)
    dev = ctypes.c_int(0)
    rc_dev = lib.cuDeviceGet(ctypes.byref(dev), 0)
    values = []
    for attribute in (75, 76, 16):  # compute capability major/minor, multiprocessor count
        value = ctypes.c_int(0)
        lib.cuDeviceGetAttribute(ctypes.byref(value), attribute, dev)
        values.append(int(value.value))
    return time.perf_counter_ns() - t0, rc, rc_dev, values


first, rc, rc_dev, values = query()
second, _, _, _ = query()
print(json.dumps({"fresh_ns": first, "second_ns": second, "rc": rc, "rc_dev": rc_dev,
                  "attrs": values}))
'''


def child_env(side: str) -> dict:
    env = dict(os.environ)
    env.pop("ELDPC_DEVICE_CLASS", None)
    if side == "default":
        env["ELDPC_DEVICE_CLASS"] = "default"
    elif side == "ada64":
        env["ELDPC_DEVICE_CLASS"] = "ada64"
    return env


def run_child(mode: str, repeats: int, side: str) -> dict:
    cmd = [sys.executable, "-c", CHILD, mode, str(repeats)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=child_env(side))
    if proc.returncode != 0:
        raise SystemExit(f"child ({mode}, {side}) failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_query_child() -> dict:
    proc = subprocess.run([sys.executable, "-c", QUERY_CHILD], capture_output=True, text=True,
                          timeout=300, env=child_env("table"))
    if proc.returncode != 0:
        raise SystemExit(f"driver-query child failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def describe(values: list[int]) -> dict:
    return {
        "median_ns": int(statistics.median(values)),
        "min_ns": int(min(values)),
        "max_ns": int(max(values)),
        "mean_ns": int(statistics.fmean(values)),
        "count": len(values),
    }


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=15,
                    help="fresh-process pairs for the first-create comparison (default 15)")
    ap.add_argument("--repeats", type=int, default=40,
                    help="create/destroy cycles per process for the cached-query measurement")
    ap.add_argument("--query-runs", type=int, default=9,
                    help="fresh processes that time the driver query in isolation (default 9)")
    ap.add_argument("--out", default="", help="write the JSON result here too")
    args = ap.parse_args(argv)
    if args.pairs < 2:
        ap.error("--pairs must be at least 2")
    if args.repeats < 4:
        ap.error("--repeats must be at least 4")
    if args.query_runs < 2:
        ap.error("--query-runs must be at least 2")

    import eldpc  # noqa: E402 - imported here so --help works without the package

    devices = eldpc.devices()
    first = {"table": [], "default": [], "ada64": []}
    variants = {}
    isas = set()
    for index in range(args.pairs):
        # Alternate the order so a drift over the run does not favour one side.
        order = ("table", "default", "ada64") if index % 2 == 0 else ("ada64", "default", "table")
        for side in order:
            out = run_child("first", 1, side)
            first[side].append(int(out["first_ns"]))
            variants[side] = out["variant"]
            isas.add(out["isa"])

    repeated = {}
    for side in ("table", "default"):
        repeated[side] = run_child("repeat", args.repeats, side)

    query_runs = [run_query_child() for _ in range(args.query_runs)]

    paired = [t - d for t, d in zip(first["table"], first["default"])]
    rest = {side: repeated[side]["times_ns"][1:] for side in ("table", "default")}
    result = {
        "probe": "create-time cost of the per-handle core-variant choice (eldpc.Decoder construction)",
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "commit": git_commit(Path(__file__).resolve().parents[2]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "environment": {
            "gpu": devices[0] if devices else None,
            "cpu": cpu_model(),
            "isa": sorted(isas),
            "version": getattr(eldpc, "__version__", "?"),
            "variants": variants,
        },
        "first_create": {
            "pairs": args.pairs,
            "table_ns": first["table"],
            "default_ns": first["default"],
            "ada64_ns": first["ada64"],
            "table": describe(first["table"]),
            "default": describe(first["default"]),
            "ada64": describe(first["ada64"]),
            "paired_difference_table_minus_default_ns": paired,
            "paired_difference_table_minus_default": describe(paired),
        },
        "repeated_create": {
            "repeats": args.repeats,
            "table_first_ns": repeated["table"]["times_ns"][0],
            "default_first_ns": repeated["default"]["times_ns"][0],
            "table_rest_ns": rest["table"],
            "default_rest_ns": rest["default"],
            "table_rest": describe(rest["table"]),
            "default_rest": describe(rest["default"]),
            "rest_median_difference_table_minus_default_ns":
                int(statistics.median(rest["table"])) - int(statistics.median(rest["default"])),
        },
        "driver_query": {
            "runs": args.query_runs,
            "fresh_ns": [q["fresh_ns"] for q in query_runs],
            "second_ns": [q["second_ns"] for q in query_runs],
            "fresh": describe([q["fresh_ns"] for q in query_runs]),
            "second": describe([q["second_ns"] for q in query_runs]),
            "attrs": query_runs[0]["attrs"],
            "rc": query_runs[0]["rc"],
            "rc_dev": query_runs[0]["rc_dev"],
        },
    }
    line = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(line + "\n", encoding="utf-8")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
