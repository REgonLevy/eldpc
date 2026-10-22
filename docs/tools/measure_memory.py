#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure what one decoder handle adds to host RSS and to the process's GPU memory.

ARCH.md derives the memory of a handle from the allocation sites in the core.  This tool measures
it: it creates decoders for one detector error model and records the process's RSS (from
/proc/self/status) and the GPU memory nvidia-smi attributes to the process before and after.

The CUDA context is created by a first, tiny decoder and measured separately, so the per-handle
numbers do not include it.  A second handle is created while the first is alive, which shows the
marginal cost of another handle on the same process.  Every measurement waits briefly for the
driver's accounting to settle and is repeated, keeping the median; the four numbers (RSS and GPU,
first and second handle) are printed and written to the JSON and Markdown files.

The tool needs the installed package, a CUDA GPU and nvidia-smi (for the per-process GPU memory).
Run it from the package root.  It writes no absolute path of the machine and no GPU UUID into its
output: the DEM argument must be a relative path.

    python docs/tools/measure_memory.py --dem docs/results/tuning-bb144/model.dem \
        --out docs/results/memory/decoder-memory.json --markdown docs/results/memory/decoder-memory.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# One generic check for an absolute path of the machine that produced a record: a POSIX path of at
# least two components (a "/" at the start of the string or after whitespace, a quote, "=", "(",
# ",", ":" or similar, not part of "://", then a name and one more "/") or a Windows drive path.
# Relative paths, URLs and ratios such as "us/shot" or "p50/p99" do not match.
_ABS_PATH = re.compile(
    r"""(?:^|[\s"'`=(\[<,:;])/(?!/)[\w.+@~-]+(?:/[\w.+@~-]*)+"""  # POSIX, >= 2 components
    r"""|\b[A-Za-z]:\\\S"""                                         # Windows drive path
)


def looks_like_abs_path(s: str) -> bool:
    return _ABS_PATH.search(s) is not None


def rss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def gpu_mib(pid: int) -> int | None:
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    total = 0
    found = False
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                total += int(float(parts[1]))
                found = True
            except ValueError:
                pass
    return total if found else 0


def sample(pid: int, repeats: int, settle_s: float) -> dict:
    rss, gpu = [], []
    for _ in range(repeats):
        time.sleep(settle_s)
        r = rss_kib()
        g = gpu_mib(pid)
        if r is not None:
            rss.append(r)
        if g is not None:
            gpu.append(g)
    return {"rss_kib": int(statistics.median(rss)) if rss else None,
            "gpu_mib": int(statistics.median(gpu)) if gpu else None}


def delta(after: dict, before: dict, key: str) -> int | None:
    a, b = after.get(key), before.get(key)
    return None if a is None or b is None else a - b


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def smi_driver_version() -> str | None:
    """The NVIDIA driver version from nvidia-smi (used when NVML gives none through eldpc)."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        return p.stdout.splitlines()[0].strip() if p.returncode == 0 and p.stdout else None
    except (OSError, subprocess.SubprocessError):
        return None


def commit_of(package_dir: Path) -> str:
    """HEAD of the git repository that contains the package (the package may be its root)."""
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(package_dir),
                           capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def package_tree_clean(package_dir: Path) -> bool:
    """True when the package's sources match the recorded commit (pathspecs relative to it)."""
    watched = ["src", "python", "include", "CMakeLists.txt", "pyproject.toml", "docs/tools"]
    try:
        p = subprocess.run(["git", "status", "--porcelain", "--", *watched], cwd=str(package_dir),
                           capture_output=True, text=True, timeout=15)
        return p.returncode == 0 and p.stdout.strip() == ""
    except (OSError, subprocess.SubprocessError):
        return False


def version_of(package_dir: Path) -> str | None:
    try:
        m = re.search(r'version\s*=\s*"([^"]+)"',
                      (package_dir / "pyproject.toml").read_text(encoding="utf-8"))
        return m.group(1) if m else None
    except OSError:
        return None


def check_paths(obj, where: str) -> None:
    if isinstance(obj, str):
        if looks_like_abs_path(obj):
            raise SystemExit(f"measure_memory: {where} names a machine path: {obj!r}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_paths(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_paths(v, f"{where}[{i}]")


def markdown(data: dict) -> str:
    env = data["environment"]
    p = data["problem"]
    lines = [
        "# Decoder memory (measured)",
        "",
        "Generated by `docs/tools/measure_memory.py`; do not edit.  Every number is read from the",
        "JSON record written by the same run.",
        "",
        f"**Measured on:** {env['gpu']['name']} (compute capability {env['gpu']['compute_capability']}, "
        f"{env['gpu']['sm_count']} SMs); {env['cpu']}; NVIDIA driver {env['driver']} "
        f"(CUDA driver API {env.get('cuda_driver_version')}).  ",
        f"**Command:** `{data['command']}`  ",
        f"**Commit:** `{data['commit']}`",
        "",
        f"Problem: m = {p['m']}, n = {p['n']}, k = {p['k']}, nnz(H) = {p['nnz_h']}, "
        f"nnz(O) = {p['nnz_o']}; DEM sha256 `{p['dem_sha256']}`.",
        "",
        "| measurement | RSS (MiB) | GPU memory (MiB) |",
        "|---|---:|---:|",
    ]

    def mib_rss(v):
        return "n/a" if v is None else f"{v / 1024:.1f}"

    def mib_gpu(v):
        return "n/a" if v is None else f"{v}"

    lines.append(f"| CUDA context (tiny decoder, then closed) | "
                 f"{mib_rss(data['context']['rss_kib_delta'])} | {mib_gpu(data['context']['gpu_mib_delta'])} |")
    lines.append(f"| first handle for the problem | "
                 f"{mib_rss(data['first_handle']['rss_kib_delta'])} | {mib_gpu(data['first_handle']['gpu_mib_delta'])} |")
    lines.append(f"| second handle, first alive | "
                 f"{mib_rss(data['second_handle']['rss_kib_delta'])} | {mib_gpu(data['second_handle']['gpu_mib_delta'])} |")
    lines += [
        "",
        "The context row is the one-time cost of bringing up CUDA (the device context and the state of",
        "the statically linked CUDA runtime); it is paid once per process.  The handle rows are what each additional decoder adds.  RSS is read",
        "from the kernel's per-process status (VmRSS) in KiB and shown in MiB; GPU memory comes from nvidia-smi's",
        "per-process accounting in MiB.  Both are sampled several times after the allocation and the",
        "median is reported.  RSS includes the create-time temporaries and the allocator's retained",
        "free space, so it is larger than the sum of the live host tables listed in ARCH.md; the GPU",
        "number is the driver's per-process accounting, so the deltas assume nothing else in the",
        "process allocates between the samples.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    here = Path(__file__).resolve()
    package_dir = here.parent.parent.parent   # <package>/docs/tools/this -> <package>

    p = argparse.ArgumentParser(description="measure the memory one decoder handle adds")
    p.add_argument("--dem", required=True, help="stim detector error model file (relative path)")
    p.add_argument("--out", required=True, help="JSON output file")
    p.add_argument("--markdown", default=None, help="optional Markdown output file")
    p.add_argument("--repeats", type=int, default=3, help="samples per measurement (default 3)")
    p.add_argument("--settle-ms", type=int, default=400, help="wait between samples, ms (default 400)")
    p.add_argument("--commit", default=None, help="commit to record (default: HEAD)")
    args = p.parse_args(argv)

    if looks_like_abs_path(args.dem):
        raise SystemExit("measure_memory: --dem must be a relative path without a machine prefix")
    for value in (args.dem, args.out, args.markdown):
        if value is not None and Path(value).is_absolute():
            raise SystemExit("measure_memory: --dem, --out and --markdown must be relative paths "
                             "(they are recorded in the output)")

    import eldpc
    import stim

    text = Path(args.dem).read_text(encoding="utf-8")
    dem = stim.DetectorErrorModel(text)
    H, O, priors = eldpc.dem_to_arrays(dem)
    problem = {"m": int(H.shape[0]), "n": int(H.shape[1]), "k": int(O.shape[0]),
               "nnz_h": int(H.nnz), "nnz_o": int(O.nnz),
               "dem_sha256": hashlib.sha256(text.encode()).hexdigest()}

    pid = os.getpid()
    settle = args.settle_ms / 1000.0
    base = sample(pid, args.repeats, settle)

    # Establish the CUDA context with a tiny decoder, then close it: its cost is per process.
    tiny = stim.DetectorErrorModel("error(0.01) D0")
    t = eldpc.Decoder.from_dem(tiny)
    t.close()
    after_context = sample(pid, args.repeats, settle)

    first = eldpc.Decoder(H, O, priors, config="a03125c3")
    after_first = sample(pid, args.repeats, settle)

    second = eldpc.Decoder(H, O, priors, config="a03125c3")
    after_second = sample(pid, args.repeats, settle)

    devices = eldpc.devices()
    dev = devices[0] if devices else {}
    data = {
        "schema": "eldpc.results.memory.v1",
        "kind": "memory",
        "commit": args.commit or commit_of(package_dir),
        "package_tree_clean": package_tree_clean(package_dir),
        "package_version": version_of(package_dir),
        "environment": {
            "gpu": {"name": dev.get("name"), "compute_capability": dev.get("compute_capability"),
                    "sm_count": dev.get("sm_count"), "l2_bytes": dev.get("l2_bytes")},
            "cpu": cpu_model(),
            "driver": dev.get("nvidia_driver_version") or smi_driver_version(),
            "cuda_driver_version": dev.get("cuda_driver_version"),
            "python": sys.version.split()[0],
            "eldpc": {"version": getattr(eldpc, "__version__", None), "isa": eldpc.isa(),
                      "build_info": eldpc.build_info()},
        },
        "problem": problem,
        "base": base,
        "context": {"rss_kib_delta": delta(after_context, base, "rss_kib"),
                    "gpu_mib_delta": delta(after_context, base, "gpu_mib")},
        "first_handle": {"rss_kib_delta": delta(after_first, after_context, "rss_kib"),
                         "gpu_mib_delta": delta(after_first, after_context, "gpu_mib")},
        "second_handle": {"rss_kib_delta": delta(after_second, after_first, "rss_kib"),
                          "gpu_mib_delta": delta(after_second, after_first, "gpu_mib")},
        "settle_ms": args.settle_ms,
        "repeats": args.repeats,
    }
    data["command"] = "python docs/tools/measure_memory.py " + " ".join(
        sys.argv[1:] if argv is None else argv)
    first.close()
    second.close()

    check_paths(data, "output")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        md = Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(markdown(data), encoding="utf-8")
    print(json.dumps({k: data[k] for k in ("context", "first_handle", "second_handle")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
