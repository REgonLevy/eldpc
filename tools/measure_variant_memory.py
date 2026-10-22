#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the host-memory cost of loading the second core variant in one process.

The front loads the default core of the host ISA when the library is loaded (eldpc_isa() and
eldpc_build_info() report it). The ada64 core of the same ISA is loaded on the first handle that
needs it; a process that holds handles of both variants therefore maps two cores. This tool
measures what that costs in host address space and resident memory, in fresh processes:

  1. after ``eldpc.load()``                     -- the front and the default core,
  2. after a default handle                     -- the first CUDA context and the handle,
  3. after a second default handle              -- control for the handle allocation alone,
  4. after an ada64 handle                      -- the ada64 core load and its handle,
  5. after closing every handle                 -- what is released.

The interesting quantity is (4) - (3) minus the control growth in (3) - (2): the memory the second
core adds on first use. ``/proc/self/maps`` is checked at every step for the core libraries that
are mapped. RSS is host-resident memory; device memory is not part of it.

    bash scripts/gpu_run.sh --timeout 600 python tools/measure_variant_memory.py \
        --runs 5 --out tools/results/variant-memory.json

Prints one JSON object; ``--out`` writes it to a file too. Run it under the machine's GPU lock:
the child processes create decoders and use the GPU.
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

CHILD = r'''
import json
import os
from pathlib import Path

import numpy as np

import eldpc

H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)


def snapshot():
    status = Path("/proc/self/status").read_text()
    memory = {}
    for line in status.splitlines():
        for key in ("VmSize", "VmRSS", "VmHWM"):
            if line.startswith(key + ":"):
                memory[key] = int(line.split(":", 1)[1].split()[0]) * 1024
    cores = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split()
        if parts and "libeldpc_core_" in parts[-1]:
            cores.add(parts[-1].rsplit("/", 1)[-1])
    return {"memory": memory, "cores": sorted(cores)}


steps = {}
steps["base"] = snapshot()
eldpc.load()
steps["after_load"] = snapshot()
os.environ["ELDPC_DEVICE_CLASS"] = "default"
first = eldpc.Decoder(H, priors=[0.1, 0.1, 0.1], config="ms30")
steps["after_default"] = snapshot()
second = eldpc.Decoder(H, priors=[0.1, 0.1, 0.1], config="ms30")
steps["after_second_default"] = snapshot()
os.environ["ELDPC_DEVICE_CLASS"] = "ada64"
ada = eldpc.Decoder(H, priors=[0.1, 0.1, 0.1], config="ms30")
steps["after_ada64"] = snapshot()
variants = {"default": first.core_variant, "second_default": second.core_variant,
            "ada64": ada.core_variant}
first.close()
second.close()
ada.close()
steps["after_close"] = snapshot()
print(json.dumps({"steps": steps, "variants": variants}))
'''


def child_env() -> dict:
    env = dict(os.environ)
    env.pop("ELDPC_DEVICE_CLASS", None)
    return env


def run_child() -> dict:
    proc = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True,
                          timeout=600, env=child_env())
    if proc.returncode != 0:
        raise SystemExit(f"child failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def describe(values: list[int]) -> dict:
    return {
        "median": int(statistics.median(values)),
        "min": int(min(values)),
        "max": int(max(values)),
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


STEPS = ("base", "after_load", "after_default", "after_second_default", "after_ada64", "after_close")
DELTAS = (
    ("load_front_and_default_core", "base", "after_load"),
    ("first_default_handle_cuda_context", "after_load", "after_default"),
    ("second_default_handle_control", "after_default", "after_second_default"),
    ("ada64_handle_and_second_core", "after_second_default", "after_ada64"),
    ("close_every_handle", "after_ada64", "after_close"),
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=5, help="fresh processes to measure (default 5)")
    ap.add_argument("--out", default="", help="write the JSON result here too")
    args = ap.parse_args(argv)
    if args.runs < 2:
        ap.error("--runs must be at least 2")

    import eldpc  # noqa: E402

    devices = eldpc.devices()
    runs = [run_child() for _ in range(args.runs)]

    per_step = {}
    for step in STEPS:
        per_step[step] = {
            key: [run["steps"][step]["memory"][key] for run in runs]
            for key in ("VmSize", "VmRSS", "VmHWM")
        }
        per_step[step]["cores"] = runs[0]["steps"][step]["cores"]

    deltas = {}
    for name, before, after in DELTAS:
        deltas[name] = {
            key: describe([run["steps"][after]["memory"][key] - run["steps"][before]["memory"][key]
                           for run in runs])
            for key in ("VmSize", "VmRSS", "VmHWM")
        }

    control = deltas["second_default_handle_control"]["VmRSS"]["median"]
    second = deltas["ada64_handle_and_second_core"]["VmRSS"]["median"]
    result = {
        "probe": "host-memory cost of loading the second core variant on first use",
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "commit": git_commit(Path(__file__).resolve().parents[2]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "environment": {
            "gpu": devices[0] if devices else None,
            "cpu": cpu_model(),
            "version": getattr(eldpc, "__version__", "?"),
        },
        "runs": args.runs,
        "steps": per_step,
        "deltas": deltas,
        "ada64_core_cost_over_control": {
            "VmRSS_median_minus_control": second - control,
            "note": ("median VmRSS growth of the ada64 handle (which loads the ada64 core) minus the "
                     "median VmRSS growth of a second default handle (same allocation work, no new core)"),
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
