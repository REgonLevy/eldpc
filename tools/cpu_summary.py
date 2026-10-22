#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the CPU-core numbers quoted in the package documentation.

    python tools/cpu_summary.py [--results tools/results/cpu]

Reads the measurement files of the CPU core (``tools/results/cpu/``, recorded on the machine they
name) and prints, in one line per fact, the numbers the documents cite: the same-session CPU / GPU
comparison, the thread placement ratios, and the exactness counts. Every number in the documents is
one of these printed values; this script is the generator, so no number is typed by hand.

The files are results of a *screen* on a shared development machine (other processes run on it);
they are not a benchmark of a quiet machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args()
    results = Path(args.results)

    def load(name: str) -> dict:
        return json.loads((results / name).read_text())

    same = load("gpu_same_machine.json")
    cpu = same["cpu_latency_ns"]
    gpu = same["gpu_latency_ns"]
    print(f"machine.cpu {same['machine']['cpu']}")
    print(f"machine.gpu {same['nvidia_smi'].splitlines()[1].split(',')[0]}")
    print(f"same_session.problem {same['problem']} config {same['config']}")
    print(f"same_session.cpu_threads {same['cpu_config'].split('threads=')[1].split(';')[0]}")
    print(f"same_session.cpu_p50_us {cpu['p50'] / 1000.0:.1f}")
    print(f"same_session.cpu_p99_us {cpu['p99'] / 1000.0:.1f}")
    print(f"same_session.gpu_p50_us {gpu['p50'] / 1000.0:.1f}")
    print(f"same_session.gpu_p99_us {gpu['p99'] / 1000.0:.1f}")
    print(f"same_session.cpu_over_gpu_p50 {same['cpu_over_gpu']['p50']:.2f}")
    print(f"same_session.cpu_over_gpu_p99 {same['cpu_over_gpu']['p99']:.2f}")

    placement = load("final_ab.json")
    for row in placement["rows"]:
        if "placement A/B" in row["note"] or "hybrid A/B" in row["note"]:
            print(f"placement.threads_{row['threads']}_over_8_p50 {row['p50']['ratio']:.2f}")
            print(f"placement.threads_{row['threads']}_over_8_p99 {row['p99']['ratio']:.2f}")

    exact_rows = []
    for name in ("exactness.json", "exactness_extra.json", "exactness_deep.json",
                 "exactness_iter.json"):
        exact_rows.extend(load(name)["rows"])
    print(f"exactness.rows {len(exact_rows)}")
    print(f"exactness.shots {sum(int(r.get('shots', 0)) for r in exact_rows)}")
    print(f"exactness.all_exact {all(bool(r.get('exact')) for r in exact_rows)}")
    mismatched = sum(sum(v.values()) if isinstance(v, dict) else int(v)
                     for r in exact_rows for v in [r.get("mismatches", 0)])
    print(f"exactness.mismatched_field_values {mismatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
