#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the CPU core's thread placement on the packaged wheel.

    python tools/cpu_placement.py [--corpus DIR] [--shots 1000] [--rounds 4]
        [--out tools/results/cpu/placement.json] [--md tools/results/cpu/placement.md]

Every arm decodes the same latency shots through the installed package's C API (`eldpc_decode`,
`device=cpu`) with the calling thread pinned and the started threads pinned through `cpus=`. On the
recorded machine (AMD Ryzen 9 7950X3D: CCD0 = CPUs 0-7,16-23 with 96 MB L3, CCD1 = CPUs 8-15,24-31
with 32 MB L3):

- `threads8_ccd0`: the reference placement, 8 threads on one CCD (caller on CPU 0, started threads
  1-7), plus an A/A duplicate of it;
- `threads16_both_ccds`: 16 threads spanning both CCDs (started threads 1-15);
- `threads16_ccd0_smt`: 16 threads on the one CCD, one per logical CPU (started 1-7 and 16-23),
  i.e. using the SMT second contexts of that CCD;
- `threads8_ccd0_smt`: 8 threads on the second SMT contexts of that CCD (started 17-23);
- `threads8_ccd1`: the same 8-thread shape on the other CCD (caller 8, started 9-15).

It writes one JSON file and the markdown table generated from it; both say that this is a screen
on a shared development machine, not a benchmark.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from cpu_screen import (Decoder, Front, Problem, config_string, git_commit, load_problem,
                        machine_record, stats)

BASE_CONFIG = ("decoder=bposd;bp_method=ms;ms_scaling_factor=0.3125;max_iter=3;schedule=parallel;"
               "osd_method=osd0;osd_order=0")

ARMS = [
    # name, calling CPU, threads, cpus (exactly `threads` entries; the started threads pin to the
    # first threads-1)
    ("threads8_ccd0", 0, 8, "1,2,3,4,5,6,7,1"),
    ("threads8_ccd0_aa", 0, 8, "1,2,3,4,5,6,7,1"),
    ("threads16_both_ccds", 0, 16, "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1"),
    ("threads16_ccd0_smt", 0, 16, "1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,1"),
    ("threads8_ccd0_smt", 0, 8, "17,18,19,20,21,22,23,17"),
    ("threads8_ccd1", 8, 8, "9,10,11,12,13,14,15,9"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="")
    ap.add_argument("--problem", default="bb144_unif_p1e-3_r12_xyz")
    ap.add_argument("--config", default=BASE_CONFIG)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="tools/results/cpu/placement.json")
    ap.add_argument("--md", default="tools/results/cpu/placement.md")
    a = ap.parse_args(argv)

    import eldpc

    package = Path(eldpc.__file__).resolve().parent
    libs = package / "_libs"
    source_root = Path(__file__).resolve().parents[1]
    corpus = Path(a.corpus) if a.corpus else Path("build/eldpc-corpus")
    if not (corpus / "problems" / a.problem).is_dir():
        raise SystemExit(f"problem {a.problem} not found under {corpus}")
    m, n, k, (hc, hi), (oc, oi), priors, dets = load_problem(corpus, a.problem)
    shots = min(a.shots, dets.shape[0])
    dets = dets[:shots]
    u32p = ctypes.POINTER(ctypes.c_uint32)
    f64p = ctypes.POINTER(ctypes.c_double)
    problem = Problem(1, m, n, k, hc.ctypes.data_as(u32p), hi.ctypes.data_as(u32p),
                      oc.ctypes.data_as(u32p), oi.ctypes.data_as(u32p),
                      priors.ctypes.data_as(f64p), None)
    front = Front(libs / "libeldpc.so")
    isa = (front.lib.eldpc_isa() or b"").decode()

    original_affinity = os.sched_getaffinity(0)
    samples = {name: [] for name, _, _, _ in ARMS}
    durations = {name: [] for name, _, _, _ in ARMS}
    variants = {}
    outputs = {}
    corr = np.empty(max(n, 1), dtype=np.uint8)
    obs = np.empty(max(k, 1), dtype=np.uint8)
    status = np.empty(1, dtype=np.uint32)
    started = time.time()
    try:
        for round_index in range(a.rounds):
            order = ARMS if round_index % 2 == 0 else list(reversed(ARMS))
            for name, call_cpu, threads, cpus in order:
                os.sched_setaffinity(0, {call_cpu})
                session = time.time()
                dec = Decoder(front, config_string(a.config, "cpu", threads, cpus), problem)
                try:
                    if dec.variant != "cpu":
                        raise SystemExit(f"{name}: core variant {dec.variant!r}, expected cpu")
                    variants[name] = dec.variant
                    for i in range(min(a.warmup, shots)):
                        dec.decode(dets[i], corr, obs, status)
                    digest = hashlib.sha256()
                    times = samples[name]
                    for i in range(shots):
                        t0 = time.perf_counter_ns()
                        rc = dec.decode(dets[i], corr, obs, status)
                        t1 = time.perf_counter_ns()
                        if rc != 0:
                            raise SystemExit(f"{name}: eldpc_decode returned {rc}")
                        times.append(t1 - t0)
                        if round_index == 0 and i < min(50, shots):
                            digest.update(corr[:n].tobytes() + obs[:k].tobytes() +
                                          status.astype("<u4").tobytes())
                    if round_index == 0:
                        outputs[name] = digest.hexdigest()
                finally:
                    dec.close()
                durations[name].append(round(time.time() - session, 3))
        if len(set(outputs.values())) != 1:
            raise SystemExit("the arms do not return identical outputs: " + json.dumps(outputs))
        elapsed = time.time() - started
    finally:
        os.sched_setaffinity(0, original_affinity)

    lat = {name: stats(values) for name, values in samples.items()}
    ref = "threads8_ccd0"
    comparisons = {"aa_p50": lat["threads8_ccd0_aa"]["p50"] / lat[ref]["p50"],
                   "aa_p99": lat["threads8_ccd0_aa"]["p99"] / lat[ref]["p99"]}
    for name, _, _, _ in ARMS:
        if name in (ref, "threads8_ccd0_aa"):
            continue
        comparisons[f"{name}_over_threads8_ccd0_p50"] = lat[name]["p50"] / lat[ref]["p50"]
        comparisons[f"{name}_over_threads8_ccd0_p99"] = lat[name]["p99"] / lat[ref]["p99"]
    result = {
        "schema": "eldpc.cpu_placement.v1",
        "command": "python tools/cpu_placement.py " + " ".join(sys.argv[1:]),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(source_root),
        "machine": machine_record(),
        "package": {"version": eldpc.__version__, "isa": isa,
                    "libeldpc_sha256": hashlib.sha256((libs / "libeldpc.so").read_bytes()).hexdigest(),
                    "cpu_library": f"libeldpc_cpu_{isa}.so",
                    "cpu_library_sha256": hashlib.sha256((libs / f"libeldpc_cpu_{isa}.so").read_bytes()).hexdigest()},
        "problem": {"id": a.problem, "num_detectors": m, "num_mechanisms": n,
                    "num_observables": k, "shots": shots, "rounds": a.rounds,
                    "warmup": min(a.warmup, shots)},
        "config": a.config,
        "arms": {name: {"call_cpu": call_cpu, "threads": threads, "cpus": cpus,
                        "variant": variants.get(name), "session_seconds": durations[name],
                        "latency_ns": lat[name]} for name, call_cpu, threads, cpus in ARMS},
        "comparisons": comparisons,
        "outputs_sha256": outputs,
        "elapsed_s": round(elapsed, 2),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if a.md:
        md = Path(a.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(result))
    print(json.dumps({k: result[k] for k in ("problem", "comparisons")}, sort_keys=True))
    print(f"wrote {out}" + (f" and {a.md}" if a.md else ""))
    return 0


def render_markdown(r: dict) -> str:
    lines = [
        "# CPU core thread placement (installed package)",
        "",
        "Generated by `tools/cpu_placement.py`; every number is from the JSON written by the same",
        "run. This is a **screen** on a shared development machine, not a benchmark.",
        "",
        "| item | value |",
        "|---|---|",
        f"| command | `{r['command']}` |",
        f"| generated | {r['generated_utc']} |",
        f"| commit | `{r['git_commit']}` |",
        f"| CPU | {r['machine']['cpu']} |",
        f"| kernel | {r['machine']['kernel']} |",
        f"| load average at the start | {r['machine']['loadavg']} |",
        f"| problem | {r['problem']['id']} ({r['problem']['num_detectors']} detectors, "
        f"{r['problem']['num_mechanisms']} mechanisms) |",
        f"| shots / rounds per arm | {r['problem']['shots']} / {r['problem']['rounds']} |",
        "",
        "| arm | calling CPU | threads | cpus | p50 (us) | p99 (us) | p99.9 (us) |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for name, arm in r["arms"].items():
        s = arm["latency_ns"]
        lines.append(f"| `{name}` | {arm['call_cpu']} | {arm['threads']} | `{arm['cpus']}` | "
                     f"{s['p50'] / 1000:.1f} | {s['p99'] / 1000:.1f} | {s['p999'] / 1000:.1f} |")
    lines += ["", "| comparison | ratio |", "|---|---:|"]
    for key, value in r["comparisons"].items():
        lines.append(f"| {key} | {value:.3f} |")
    lines += [
        "",
        "`threads8_ccd0` is the reference placement; `threads8_ccd0_aa` is its A/A control.",
        "",
        "All arms returned identical corrections, observables and status words on the compared",
        f"shots (one SHA-256: `{list(r['outputs_sha256'].values())[0]}`).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
