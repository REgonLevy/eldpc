#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A short latency screen of the installed package's CPU core against its GPU core.

    python tools/cpu_screen.py [--corpus DIR] [--problem bb144_unif_p1e-3_r12_xyz] [--config a03125c3]
        [--shots 2000] [--rounds 3] [--out tools/results/cpu/screen.json]
        [--md tools/results/cpu/screen.md] [--no-gpu]

The screen calls the C API (`eldpc_create` / `eldpc_decode`) of the **installed** package through
ctypes, one shot per call, on the latency shots of a decoder-bench corpus problem:

- `cpu_packaged_t1`, `cpu_packaged_t8`: `device=cpu` at `threads=1` and `threads=8`; the calling
  thread is pinned to CPU 0 and the 8-thread handle pins its started threads to CPUs 1-7
  (`cpus=1,2,3,4,5,6,7,1`), i.e. one CCD (one L3) on the recorded machine;
- `cpu_packaged_t8_aa`: a second, independent handle at the same settings -- the A/A control;
- `cpu_rebuilt_t1`, `cpu_rebuilt_t8`: the same sources compiled by this script with the same
  flags into a temporary library (the "measured build" the package ships), to show that packaging
  did not change the speed;
- `gpu_core`: `device=gpu` through the same front, the GPU core in the same session.

Every arm decodes the same shots; their corrections, observables and status words are compared
(they must be equal). Each arm runs as its own session (create, warm up, time, close) because a
CPU handle with `threads > 1` keeps `threads - 1` threads busy-waiting for its whole life; the
rounds repeat every session, so a drift of the machine shows in every arm. Each latency is one
`eldpc_decode` call in nanoseconds; the tables report p50/p90/p99/p99.9.

This is a *screen* on a shared development machine, not a benchmark: the JSON records the machine,
the load average, the driver and toolkit versions, the commit and the exact command, and the
markdown it writes says so. Run it while no other process uses the GPU.

The corpus is a directory in the decoder-bench layout (``problems/<name>/H.csc``, ``O.csc``,
``priors.dba`` and ``shots/<name>/latency/dets.dba``; the DBCSC / DBA binary formats), read here
directly.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

OK = 0
_U8P = ctypes.POINTER(ctypes.c_uint8)
_U32P = ctypes.POINTER(ctypes.c_uint32)
_F64P = ctypes.POINTER(ctypes.c_double)


class Problem(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("num_detectors", ctypes.c_uint32),
        ("num_mechanisms", ctypes.c_uint32),
        ("num_observables", ctypes.c_uint32),
        ("h_colptr", _U32P),
        ("h_rowind", _U32P),
        ("o_colptr", _U32P),
        ("o_rowind", _U32P),
        ("priors", _F64P),
        ("dem_text", ctypes.c_char_p),
    ]


# ---------------------------------------------------------------------------------------------
# the corpus file formats (DBCSC / DBA), read directly


def read_csc(path: Path):
    b = path.read_bytes()
    if b[:8] != b"DBCSC001":
        raise SystemExit(f"{path}: not a DBCSC matrix")
    nrows = int.from_bytes(b[8:16], "little")
    ncols = int.from_bytes(b[16:24], "little")
    nnz = int.from_bytes(b[24:32], "little")
    colptr = np.frombuffer(b, dtype="<u4", count=ncols + 1, offset=32).astype(np.uint32)
    rowind = np.frombuffer(b, dtype="<u4", count=nnz, offset=32 + 4 * (ncols + 1)).astype(np.uint32)
    return int(nrows), int(ncols), np.ascontiguousarray(colptr), np.ascontiguousarray(rowind)


def read_dba(path: Path):
    b = path.read_bytes()
    if b[:8] != b"DBARRAY1":
        raise SystemExit(f"{path}: not a DBA array")
    code = int.from_bytes(b[8:12], "little")
    nrows = int.from_bytes(b[16:24], "little")
    ncols = int.from_bytes(b[24:32], "little")
    payload = b[32:]
    if code == 4:  # bits: nrows rows of ceil(ncols/8) bytes, LSB first
        rb = (ncols + 7) // 8
        packed = np.frombuffer(payload, dtype=np.uint8).reshape(nrows, rb)
        return np.unpackbits(packed, axis=1, count=ncols, bitorder="little").astype(np.uint8)
    if code == 3:
        return np.frombuffer(payload, dtype="<f8").reshape(nrows, ncols)
    if code == 1:
        return np.frombuffer(payload, dtype=np.uint8).reshape(nrows, ncols)
    raise SystemExit(f"{path}: unsupported DBA dtype {code}")


def load_problem(corpus: Path, problem: str):
    pdir = corpus / "problems" / problem
    sdir = corpus / "shots" / problem / "latency"
    m, n, hc, hi = read_csc(pdir / "H.csc")
    k, n_o, oc, oi = read_csc(pdir / "O.csc")
    assert n_o == n
    priors = np.ascontiguousarray(read_dba(pdir / "priors.dba").reshape(-1), dtype=np.float64)
    dets = np.ascontiguousarray(read_dba(sdir / "dets.dba"), dtype=np.uint8)
    return m, n, k, (hc, hi), (oc, oi), priors, dets


# ---------------------------------------------------------------------------------------------
# the C API of the installed package


class Front:
    """libeldpc.so: the eldpc_* C API (the installed package's front)."""

    kind = "eldpc"

    def __init__(self, path: Path):
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        L = self.lib
        L.eldpc_create.restype = ctypes.c_void_p
        L.eldpc_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(Problem), ctypes.c_char_p,
                                   ctypes.c_size_t]
        L.eldpc_decode.restype = ctypes.c_int
        L.eldpc_decode.argtypes = [ctypes.c_void_p, _U8P, _U8P, _U8P, _U32P]
        L.eldpc_destroy.restype = None
        L.eldpc_destroy.argtypes = [ctypes.c_void_p]
        L.eldpc_core_variant.restype = ctypes.c_char_p
        L.eldpc_core_variant.argtypes = [ctypes.c_void_p]
        L.eldpc_isa.restype = ctypes.c_char_p
        L.eldpc_isa.argtypes = []
        L.eldpc_keep_warm.restype = ctypes.c_int
        L.eldpc_keep_warm.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    def create(self, config: str, problem: Problem):
        err = ctypes.create_string_buffer(1024)
        handle = self.lib.eldpc_create(config.encode(), ctypes.byref(problem), err, len(err))
        if not handle:
            raise SystemExit(f"eldpc_create({config!r}) failed: {err.value.decode()}")
        return handle

    def variant(self, handle):
        return (self.lib.eldpc_core_variant(handle) or b"").decode()

    def decode(self, handle, syndrome, corr, obs, status) -> int:
        return self.lib.eldpc_decode(handle, syndrome.ctypes.data_as(_U8P),
                                     corr.ctypes.data_as(_U8P), obs.ctypes.data_as(_U8P),
                                     status.ctypes.data_as(_U32P))

    def destroy(self, handle):
        self.lib.eldpc_destroy(handle)


class CoreFront:
    """A decoder core library (the decoder-bench ABI: dbench_*), for the rebuilt CPU build."""

    kind = "core"

    def __init__(self, path: Path):
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        L = self.lib
        L.dbench_create.restype = ctypes.c_void_p
        L.dbench_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(Problem), ctypes.c_char_p,
                                    ctypes.c_size_t]
        L.dbench_decode.restype = ctypes.c_int
        L.dbench_decode.argtypes = [ctypes.c_void_p, _U8P, _U8P, _U8P, _U32P]
        L.dbench_destroy.restype = None
        L.dbench_destroy.argtypes = [ctypes.c_void_p]

    def create(self, config: str, problem: Problem):
        err = ctypes.create_string_buffer(1024)
        handle = self.lib.dbench_create(config.encode(), ctypes.byref(problem), err, len(err))
        if not handle:
            raise SystemExit(f"dbench_create({config!r}) failed: {err.value.decode()}")
        return handle

    def variant(self, handle):
        return "cpu"

    def decode(self, handle, syndrome, corr, obs, status) -> int:
        return self.lib.dbench_decode(handle, syndrome.ctypes.data_as(_U8P),
                                      corr.ctypes.data_as(_U8P), obs.ctypes.data_as(_U8P),
                                      status.ctypes.data_as(_U32P))

    def destroy(self, handle):
        self.lib.dbench_destroy(handle)


def config_string(base: str, device: str, threads=None, cpus=None) -> str:
    parts = [base]
    if device == "cpu":
        parts.append("device=cpu")
    if threads is not None:
        parts.append(f"threads={threads}")
    if cpus is not None:
        parts.append(f"cpus={cpus}")
    return ";".join(parts)


class Decoder:
    def __init__(self, front, config: str, problem: Problem):
        self.front = front
        self._h = front.create(config, problem)
        self.variant = front.variant(self._h)

    def decode(self, syndrome: np.ndarray, corr: np.ndarray, obs: np.ndarray, status: np.ndarray) -> int:
        return self.front.decode(self._h, syndrome, corr, obs, status)

    def close(self):
        if getattr(self, "_h", None):
            self.front.destroy(self._h)
            self._h = None


# ---------------------------------------------------------------------------------------------
# the rebuilt library (the same sources and flags as the packaged CPU core)


def build_cpu_library(package: Path, isa: str, dest: Path) -> Path:
    source = package / "src" / "cpu" / "fastbpcpu6g.cpp"
    flags = ["-O2", "-DNDEBUG", "-ffp-contract=off", "-fno-fast-math", "-O3"]
    if isa == "avx512":
        flags += ["-mavx2", "-mavx512f", "-mavx512vl", "-mavx512dq", "-mavx512bw"]
    else:
        flags += ["-mavx2"]
    info = dest / "build-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "dbench_candidate_build_info.h").write_text(
        '#define DBENCH_CANDIDATE_NAME "fastbpcpu6g"\n'
        '#define DBENCH_CANDIDATE_BUILD_INFO "compiler=screen; flags=' + " ".join(flags) +
        '; build_type=Screen; isa=' + isa + '; lib=fastbpcpu6g; threads=1..64"\n')
    lib = dest / f"libeldpc_cpu_{isa}_rebuilt.so"
    command = ["g++", "-std=c++17", "-fPIC", "-shared", *flags, str(source),
               "-I", str(package / "src" / "port"),
               "-I", str(package / "src" / "third_party" / "ldpc" / "src_cpp"),
               "-I", str(package / "src" / "third_party" / "ldpc" / "include" / "robin_map"),
               "-I", str(info), "-pthread", "-o", str(lib)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("cannot rebuild the CPU core:\n" + proc.stderr[-2000:])
    return lib


# ---------------------------------------------------------------------------------------------
# the screen


def machine_record() -> dict:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        model = next(line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                     if line.startswith("model name"))
    except (OSError, StopIteration):
        model = platform.processor()
    try:
        loadavg = Path("/proc/loadavg").read_text().strip()
    except OSError:
        loadavg = "?"
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "?"
    return {
        "cpu": model,
        "cpu_count": os.cpu_count(),
        "kernel": platform.platform(),
        "loadavg": loadavg,
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv"]),
        # The toolkit the package was built with: CUDACXX names it; PATH's nvcc is a fallback.
        "nvcc": run([os.environ.get("CUDACXX") or "nvcc", "--version"]),
        "python": platform.python_version(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }


def git_commit(package: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def stats(values: list[int]) -> dict:
    a = np.asarray(values, dtype=np.int64)
    return {
        "n": int(a.size),
        "min": int(a.min()),
        "mean": float(a.mean()),
        "p50": int(np.percentile(a, 50)),
        "p90": int(np.percentile(a, 90)),
        "p99": int(np.percentile(a, 99)),
        "p999": int(np.percentile(a, 99.9)),
        "max": int(a.max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="")
    ap.add_argument("--problem", default="bb144_unif_p1e-3_r12_xyz")
    ap.add_argument("--config", default="decoder=bposd;bp_method=ms;ms_scaling_factor=0.3125;"
                                         "max_iter=3;schedule=parallel;osd_method=osd0;osd_order=0")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--call-cpu", type=int, default=0)
    ap.add_argument("--cpu-list", default="1,2,3,4,5,6,7,1")
    ap.add_argument("--out", default="tools/results/cpu/screen.json")
    ap.add_argument("--md", default="tools/results/cpu/screen.md")
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args(argv)

    import eldpc  # the installed package this screen measures

    package = Path(eldpc.__file__).resolve().parent
    libs = package / "_libs"
    # The rebuilt arm needs the CPU core sources, which the wheel does not ship: they are in the
    # package source tree this script belongs to.
    source_root = Path(__file__).resolve().parents[1]
    corpus = Path(a.corpus) if a.corpus else Path("build/eldpc-corpus")
    if not (corpus / "problems" / a.problem).is_dir():
        raise SystemExit(f"problem {a.problem} not found under {corpus}")
    m, n, k, (hc, hi), (oc, oi), priors, dets = load_problem(corpus, a.problem)
    shots = min(a.shots, dets.shape[0])
    dets = dets[:shots]

    problem = Problem(1, m, n, k, hc.ctypes.data_as(_U32P), hi.ctypes.data_as(_U32P),
                      oc.ctypes.data_as(_U32P), oi.ctypes.data_as(_U32P),
                      priors.ctypes.data_as(_F64P), None)
    front = Front(libs / "libeldpc.so")
    isa = (front.lib.eldpc_isa() or b"").decode()

    # The calling thread sits on the call CPU for every arm; the 8-thread CPU arms pin their
    # started threads to the rest of the CCD.
    original_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {a.call_cpu})

    tmp = Path(tempfile.mkdtemp(prefix="eldpc-cpu-screen-"))
    sources = source_root / "src" / "cpu" / "fastbpcpu6g.cpp"
    rebuilt = build_cpu_library(source_root, isa, tmp) if sources.is_file() else None

    arm_defs = [
        # name, front ("eldpc" = the installed front; "core" = the rebuilt CPU core), config
        ("cpu_packaged_t1", "eldpc", config_string(a.config, "cpu", 1, a.call_cpu)),
        ("cpu_packaged_t8", "eldpc", config_string(a.config, "cpu", 8, a.cpu_list)),
        ("cpu_packaged_t8_aa", "eldpc", config_string(a.config, "cpu", 8, a.cpu_list)),
    ]
    if rebuilt is not None:
        arm_defs += [
            # A core library takes the core's own keys only: no device key.
            ("cpu_rebuilt_t1", "core", config_string(a.config, "", 1, a.call_cpu)),
            ("cpu_rebuilt_t8", "core", config_string(a.config, "", 8, a.cpu_list)),
        ]
    if not a.no_gpu:
        arm_defs.append(("gpu_core", "eldpc", a.config))

    core_front = CoreFront(rebuilt) if rebuilt is not None else None
    samples = {name: [] for name, _, _ in arm_defs}
    outputs = {}
    durations = {}
    variants = {}
    corr = np.empty(max(n, 1), dtype=np.uint8)
    obs = np.empty(max(k, 1), dtype=np.uint8)
    status = np.empty(1, dtype=np.uint32)
    started = time.time()
    try:
        # One handle at a time: a CPU handle with threads > 1 keeps threads-1 threads busy-waiting
        # for its whole life, and two such teams would compete for the same cores. Each round runs
        # every arm as its own session (create, warm up, time, close), like one process per arm;
        # the rounds are the repetition that shows the machine's drift.
        for round_index in range(a.rounds):
            # Alternate the order of the arms every round, so a drift of the machine falls on
            # every arm equally (the A/A arm is the control).
            order = arm_defs if round_index % 2 == 0 else list(reversed(arm_defs))
            for name, front_kind, cfg in order:
                session = time.time()
                dec = Decoder(front if front_kind == "eldpc" else core_front, cfg, problem)
                try:
                    expected = {"cpu"} if name.startswith("cpu_") else {"default", "ada64"}
                    if dec.variant not in expected:
                        raise SystemExit(f"{name}: core variant {dec.variant!r}, expected {expected}")
                    variants[name] = dec.variant
                    for i in range(min(a.warmup, shots)):
                        dec.decode(dets[i], corr, obs, status)
                    digest = hashlib.sha256()
                    times = samples[name]
                    for i in range(shots):
                        s = dets[i]
                        t0 = time.perf_counter_ns()
                        rc = dec.decode(s, corr, obs, status)
                        t1 = time.perf_counter_ns()
                        if rc != OK:
                            raise SystemExit(f"{name}: eldpc_decode returned {rc}")
                        times.append(t1 - t0)
                        if round_index == 0 and i < min(50, shots):
                            digest.update(corr[:n].tobytes() + obs[:k].tobytes() +
                                          status.astype("<u4").tobytes())
                    if round_index == 0:
                        outputs[name] = digest.hexdigest()
                finally:
                    dec.close()
                durations.setdefault(name, []).append(round(time.time() - session, 3))
        if len(set(outputs.values())) != 1:
            raise SystemExit("the arms do not return identical outputs: " + json.dumps(outputs))
        elapsed = time.time() - started
    finally:
        os.sched_setaffinity(0, original_affinity)
        shutil.rmtree(tmp, ignore_errors=True)

    lat = {name: stats(values) for name, values in samples.items()}
    def ratio(a_name, b_name, key="p50"):
        return lat[a_name][key] / lat[b_name][key]
    comparisons = {
        "cpu_t8_over_gpu_p50": ratio("cpu_packaged_t8", "gpu_core") if "gpu_core" in lat else None,
        "cpu_t8_over_gpu_p99": ratio("cpu_packaged_t8", "gpu_core", "p99") if "gpu_core" in lat else None,
        "cpu_t1_over_gpu_p50": ratio("cpu_packaged_t1", "gpu_core") if "gpu_core" in lat else None,
        "aa_t8_p50": ratio("cpu_packaged_t8_aa", "cpu_packaged_t8"),
        "aa_t8_p99": ratio("cpu_packaged_t8_aa", "cpu_packaged_t8", "p99"),
        "packaged_over_rebuilt_t8_p50":
            ratio("cpu_packaged_t8", "cpu_rebuilt_t8") if "cpu_rebuilt_t8" in lat else None,
        "packaged_over_rebuilt_t8_p99":
            ratio("cpu_packaged_t8", "cpu_rebuilt_t8", "p99") if "cpu_rebuilt_t8" in lat else None,
        "packaged_over_rebuilt_t1_p50":
            ratio("cpu_packaged_t1", "cpu_rebuilt_t1") if "cpu_rebuilt_t1" in lat else None,
        "t8_over_t1_p50": ratio("cpu_packaged_t8", "cpu_packaged_t1"),
    }
    result = {
        "schema": "eldpc.cpu_screen.v1",
        "command": "python tools/cpu_screen.py " + " ".join(sys.argv[1:]),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(source_root),
        "machine": machine_record(),
        "package": {"version": eldpc.__version__, "isa": isa, "build_info": eldpc.build_info(),
                    "libeldpc_sha256": hashlib.sha256((libs / "libeldpc.so").read_bytes()).hexdigest(),
                    "cpu_library": f"libeldpc_cpu_{isa}.so",
                    "cpu_library_sha256": hashlib.sha256((libs / f"libeldpc_cpu_{isa}.so").read_bytes()).hexdigest()},
        "problem": {"id": a.problem, "num_detectors": m, "num_mechanisms": n, "num_observables": k,
                    "shots": shots, "rounds": a.rounds, "warmup": min(a.warmup, shots)},
        "config": a.config,
        "call_cpu": a.call_cpu,
        "cpu_list_8": a.cpu_list,
        "arms": {name: {"config": cfg, "variant": variants.get(name),
                        "library": ("installed libeldpc.so" if front_kind == "eldpc"
                                    else "rebuilt from src/cpu with the same flags"),
                        "session_seconds": durations.get(name, []),
                        "latency_ns": lat[name]} for name, front_kind, cfg in arm_defs},
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
    print(json.dumps({k: result[k] for k in ("problem", "config", "comparisons")}, sort_keys=True))
    print(f"wrote {out}" + (f" and {a.md}" if a.md else ""))
    return 0


def render_markdown(r: dict) -> str:
    lines = [
        "# CPU core screen (installed package against the GPU core)",
        "",
        "Generated by `tools/cpu_screen.py`; every number below is from the JSON written by the same",
        "run. This is a **screen** on a shared development machine, not a benchmark of a quiet machine.",
        "",
        "| item | value |",
        "|---|---|",
        f"| command | `{r['command']}` |",
        f"| generated | {r['generated_utc']} |",
        f"| commit | `{r['git_commit']}` |",
        f"| CPU | {r['machine']['cpu']} (affinity {r['machine']['affinity']}) |",
        f"| kernel | {r['machine']['kernel']} |",
        f"| load average at the start | {r['machine']['loadavg']} |",
        f"| GPU / driver / compute capability | {r['machine']['nvidia_smi'].splitlines()[-1] if r['machine']['nvidia_smi'] else '?'} |",
        f"| nvcc | {(r['machine']['nvcc'].splitlines()[-1] if r['machine']['nvcc'] else '?')} |",
        f"| problem | {r['problem']['id']} ({r['problem']['num_detectors']} detectors, "
        f"{r['problem']['num_mechanisms']} mechanisms, {r['problem']['num_observables']} observables) |",
        f"| shots / rounds per arm | {r['problem']['shots']} / {r['problem']['rounds']} |",
        f"| config | `{r['config']}` |",
        "",
        "| arm | core | shots | p50 (us) | p90 (us) | p99 (us) | p99.9 (us) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, arm in r["arms"].items():
        s = arm["latency_ns"]
        lines.append(f"| `{name}` | {arm['variant']} | {s['n']} | {s['p50'] / 1000:.1f} | "
                     f"{s['p90'] / 1000:.1f} | {s['p99'] / 1000:.1f} | {s['p999'] / 1000:.1f} |")
    lines += ["", "| comparison | ratio |", "|---|---:|", ]
    for key, value in r["comparisons"].items():
        if value is not None:
            lines.append(f"| {key} | {value:.3f} |")
    lines += [
        "",
        "The A/A rows (`cpu_packaged_t8_aa` against `cpu_packaged_t8`) are the noise control of the",
        "same session: their ratio bounds how much of the other ratios is machine noise.",
        "",
        "All arms returned identical corrections, observables and status words on the compared shots",
        f"(one SHA-256 over the three output arrays: `{list(r['outputs_sha256'].values())[0]}`).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
