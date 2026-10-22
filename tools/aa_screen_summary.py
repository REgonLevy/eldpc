#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compact, path-free summary of a paired before/after eldpc_decode screen.

    python tools/aa_screen_summary.py \
        --results DIR/results.json \
        --before-commit <git sha of the before build> --after-commit <git sha of the after build> \
        --before-lib PATH --after-lib PATH --problem PATH --out-dir DIR \
        --out tools/results/aa-screen/summary.json

The input is the ``results.json`` that ``bench.py c-latency`` writes for two builds of the package
(``--lib`` and ``--lib-b``). It contains an A/B comparison with its own A/A control: the same
estimator run as A against A, whose spread is this machine's noise band; a difference inside that
band means the two builds cannot be told apart on the measured problem.

The output is the same measurement in a compact, committable form: the environment (GPU, CPU,
driver, CUDA runtime, ISA), the SHA-256 of both libraries, the commits they were built from, the
per-mode pooled statistics and the A/A band, with every machine path replaced by a placeholder.
A full copy of the input with the same replacements is written next to it
(``raw-results.sanitized.json``), so the summary can be checked against the raw numbers.

The screen itself is run under the machine's GPU lock; the command is recorded in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(value, replacements: list[tuple[str, str]]):
    """Recursively replace machine paths in every string of a JSON value."""
    if isinstance(value, dict):
        return {key: sanitize(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            if old:
                value = value.replace(old, new)
        return value
    return value


def find_absolute_paths(value, path="") -> list[str]:
    """Every string value that still looks like an absolute machine path."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found += find_absolute_paths(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += find_absolute_paths(item, f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith("/") and not value.startswith("/dev/"):
        found.append(f"{path}={value}")
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="bench.py c-latency results.json")
    ap.add_argument("--before-commit", required=True, help="git commit the before build came from")
    ap.add_argument("--after-commit", required=True, help="git commit the after build came from")
    ap.add_argument("--before-lib", required=True, help="libeldpc.so of the before build")
    ap.add_argument("--after-lib", required=True, help="libeldpc.so of the after build")
    ap.add_argument("--problem", required=True, help="the DEM file the screen decoded")
    ap.add_argument("--out-dir", required=True, help="the bench --out directory")
    ap.add_argument("--out", required=True, help="write the summary JSON here")
    args = ap.parse_args(argv)

    results_path = Path(args.results)
    raw = json.loads(results_path.read_text(encoding="utf-8"))
    latency = raw["c_latency"]
    env = raw["environment"]

    before_lib = Path(args.before_lib).resolve()
    after_lib = Path(args.after_lib).resolve()
    replacements = [
        (str(before_lib), "<before>/libeldpc.so"),
        (str(after_lib), "<after>/libeldpc.so"),
        (str(Path(args.problem).resolve()), "<problem>"),
        (str(Path(args.out_dir).resolve()), "<out>"),
        (str(before_lib.parent), "<before>"),
        (str(after_lib.parent), "<after>"),
        (str(results_path.resolve().parent), "<out>"),
    ]

    modes = {}
    for name, mode in latency["modes"].items():
        modes[name] = {
            "mode": mode.get("mode"),
            "gap_us": mode.get("gap_us"),
            "sessions_per_pair": mode.get("sessions_per_pair"),
            "shots_per_variant_per_session": mode.get("shots_per_variant_per_session"),
            "variants": {
                side: {
                    "label": info.get("label"),
                    "config_string": info.get("config_string"),
                    "pooled": info.get("pooled"),
                    "status": info.get("status"),
                }
                for side, info in mode["variants"].items()
            },
            "comparison": {
                key: mode["comparison"].get(key)
                for key in ("delta", "delta_blocked", "aa_band", "aa_band_sessions", "aa_band_note",
                            "within_band", "within_band_blocked", "checksums", "checksums_equal",
                            "sides_equal", "sem", "per_session_delta")
            },
            "warnings": mode.get("warnings"),
        }

    summary = {
        "probe": "paired A/A and A/B of eldpc_decode (C API single-shot latency), before vs after",
        "command": sanitize(raw.get("command"), replacements),
        "source_results": "raw-results.sanitized.json",
        "source_results_sha256": sha256_file(results_path),
        "generated_utc": raw.get("generated_utc"),
        "quick": raw.get("quick"),
        "settings": raw.get("settings"),
        "elapsed_s": raw.get("elapsed_s"),
        "commits": {"before": args.before_commit, "after": args.after_commit},
        "libraries": {
            "A_before": {"role": "before", "sha256": sha256_file(before_lib)},
            "B_after": {"role": "after", "sha256": sha256_file(after_lib)},
        },
        "environment": {
            "gpu": env.get("gpu"),
            "cpu": env.get("cpu"),
            "driver": env.get("driver"),
            "cuda_runtime": env.get("cuda_runtime"),
            "toolkit": env.get("toolkit"),
            "isa": env.get("isa"),
            "version": env.get("version"),
            "build_info": env.get("build_info"),
            "clocks": env.get("clocks"),
            "loadavg": env.get("loadavg"),
            "loadavg_after": env.get("loadavg_after"),
            "gpu_processes": env.get("gpu_processes"),
            "gpu_processes_after": env.get("gpu_processes_after"),
        },
        "problem": {
            "name": raw.get("problem", {}).get("name"),
            "detectors": raw.get("problem", {}).get("m"),
            "mechanisms": raw.get("problem", {}).get("n"),
            "observables": raw.get("problem", {}).get("k"),
            "dem_sha256": raw.get("problem", {}).get("dem_sha256"),
            "latency_shots": raw.get("problem", {}).get("shots", {}).get("latency", {}).get("shots"),
        },
        "modes": modes,
        "warnings": raw.get("warnings"),
        "failures": raw.get("failures"),
    }

    summary = sanitize(summary, replacements)
    raw_sanitized = sanitize(raw, replacements)
    leftover = find_absolute_paths(summary) + find_absolute_paths(raw_sanitized)
    if leftover:
        print("aa_screen_summary: machine paths left after sanitizing:\n  " + "\n  ".join(leftover[:20]),
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out.parent / "raw-results.sanitized.json").write_text(
        json.dumps(raw_sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["modes"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
