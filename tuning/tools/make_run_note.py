#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a RUN.md next to a committed tuner or benchmark results.json.

The note repeats, from the results file itself, the command and the environment (GPU, CPU, driver,
toolkit); it adds no number that is not in the results file, and it names this script as its
author.  A short label says whether the timings are timing-grade or screens.

    python tuning/tools/make_run_note.py --results tuning/results/<name>/results.json \
        --commit <commit of the code that ran> --label "screen: development machine"

Neither results file records the commit of the code that produced it, and the commit checked out
when the note is written can postdate the run, so the note names a commit only when ``--commit``
gives it (the person who made the run knows which code ran); otherwise it says "not recorded".
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _tuner_note(result: dict) -> tuple[str, list[tuple[str, str]]]:
    command = " ".join(result.get("tool", {}).get("command") or ["unknown"])
    env = result.get("environment") or {}
    machine = env.get("machine") or {}
    gpu = machine.get("gpu") or {}
    devices = gpu.get("devices") or []
    device = devices[0] if devices else {}
    commands = {entry.get("command", ""): entry for entry in env.get("commands") or []}
    rows = [
        ("GPU", str(device.get("name") or "none reported")),
        ("GPU compute capability / SMs / L2",
         f"{device.get('compute_capability')} / {device.get('sm_count')} / {device.get('l2_bytes')} B"),
        ("GPU driver", str(device.get("nvidia_driver_version") or device.get("driver_version"))),
        ("CPU", f"{machine.get('cpu_model')} ({machine.get('cpu_count')} threads)"),
        ("Package", f"eldpc {env.get('eldpc', {}).get('version')}, ISA {env.get('eldpc', {}).get('isa')}"),
        ("Python / numpy / scipy / stim / qLDPC",
         " / ".join(str((env.get("python") or {}).get(k)) for k in
                    ("version", "numpy", "scipy", "stim", "qldpc"))),
    ]
    toolkit = env.get("toolkit")
    if toolkit:   # tuner 2.2+: the loaded library's toolkit (build_info), else nvcc --version
        rows.append(("CUDA toolkit", f"{toolkit.get('version')} ({toolkit.get('nvcc')}, source: "
                                     f"{toolkit.get('source')})"))
    for name in ("nvidia-smi", "nvcc", "CUDACXX"):
        for key, entry in commands.items():
            if name in key:
                rows.append((key, (entry.get("stdout") or entry.get("stderr") or "").replace("\n", "; ")))
                break
    extra = [
        ("Started (UTC)", str(result.get("tool", {}).get("started_utc"))),
        ("Wall time (s)", f"{result.get('tool', {}).get('duration_seconds'):.1f}"),
        ("Quick", str(bool(result.get("tool", {}).get("quick")))),
        ("Schema", str(result.get("schema"))),
    ]
    rows.extend(extra)
    return command, rows


def _bench_note(result: dict) -> tuple[str, list[tuple[str, str]]]:
    command = str(result.get("command", "unknown"))
    env = result.get("environment") or {}
    gpu = env.get("gpu") or {}
    cpu = env.get("cpu") or {}
    toolkit = env.get("toolkit") or {}
    rows = [
        ("GPU", str(gpu.get("name"))),
        ("GPU compute capability / SMs / L2 / memory",
         f"{gpu.get('compute_capability')} / {gpu.get('sm_count')} / {gpu.get('l2_bytes')} B / "
         f"{gpu.get('memory_mib')} MiB"),
        ("GPU driver", str(env.get("driver"))),
        ("CUDA runtime / toolkit", f"{env.get('cuda_runtime')} / {toolkit.get('version')}"),
        ("CPU", f"{cpu.get('model')} ({cpu.get('logical_cores')} logical cores)"),
        ("Pinned CPU", str(cpu.get("pinned_cpu"))),
        ("Package", f"eldpc {env.get('version')}, ISA {env.get('isa')}"),
        ("Generated (UTC)", str(result.get("generated_utc"))),
        ("Quick", str(bool(result.get("quick")))),
        ("Schema", str(result.get("schema"))),
    ]
    return command, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="path to results.json")
    parser.add_argument("--label", default="",
                        help="one line on the grade of the timings (for example: screens on a "
                             "shared development machine)")
    parser.add_argument("--commit", default=None,
                        help="commit of the code that produced the run (default: not recorded)")
    args = parser.parse_args(argv)
    with open(args.results, "r", encoding="utf-8") as f:
        result = json.load(f)
    schema = str(result.get("schema", ""))
    if schema.startswith("eldpc.tune."):
        command, rows = _tuner_note(result)
    elif schema.startswith("eldpc.bench."):
        command, rows = _bench_note(result)
    else:
        print(f"{args.results}: unknown schema {schema!r}", file=sys.stderr)
        return 2
    commit = (f"`{args.commit}` (the code that produced the run)" if args.commit else
              "not recorded")
    lines = ["# Run record", "",
             f"- command: `{command}`",
             f"- commit: {commit}",
             f"- results: `{os.path.basename(args.results)}`",
             f"- written by: `tuning/tools/make_run_note.py` from `{os.path.basename(args.results)}`"]
    if args.label:
        lines.append(f"- timings: {args.label}")
    lines += ["", "| | |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in rows]
    lines.append("")
    out = os.path.join(os.path.dirname(os.path.abspath(args.results)), "RUN.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
