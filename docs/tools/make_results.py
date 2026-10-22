#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate docs/RESULTS.md from the results files in docs/results/.

Every number in RESULTS.md is read from a committed JSON file under docs/results/ and every table
names the run it came from: the GPU, CPU, drivers, CUDA toolkit and clocks, the command, and the
git commit.  Nothing is typed by hand, so the document cannot drift from the measurements.

    python docs/tools/make_results.py            # write docs/RESULTS.md
    python docs/tools/make_results.py --check    # exit 1 when the file differs from this output

The results files are written by docs/tools/collect_results.py (and the memory record by
docs/tools/measure_memory.py) from the output of the package's own commands; see the
"Reproducing" section of the generated document for the full command sequence.

The script uses the Python standard library only, so --check works in a fresh checkout without
the package installed and without a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent          # <package>/docs
RESULTS_DIR = DOCS_DIR / "results"
OUTPUT = DOCS_DIR / "RESULTS.md"

# The raw outputs live outside the package tree; the commands below are run from the package root.
RAW = "../eldpc-results"

REPRODUCE_COMMANDS = [
    "cmake -S . -B build-ref -DCMAKE_BUILD_TYPE=Release    # with CUDACXX naming the package's toolkit",
    "cmake --build build-ref -j --target eldpc_core_avx2",
    f"ELDPC_REFERENCE_LIB=build-ref/lib/libeldpc_core_avx2.so python verify/run_verify.py --quick "
    f"--out {RAW}/verify",
    f"python bench/bench.py all --quick --problem bb144 --out {RAW}/bench",
    "python bench/bench.py ler --problem surface --config a03125c3 --config ms30 "
    f"--ler-shots 500000 --out {RAW}/ler",
    "mkdir -p docs/results/tuning-bb144",
    f"cp {RAW}/bench/data/model.dem docs/results/tuning-bb144/model.dem",
    "python tuning/eldpc_tune.py --dem docs/results/tuning-bb144/model.dem --ler-target 0.005 "
    "--p99-budget-us 500 --auto --quick --out docs/results/tuning-bb144",
    "python docs/tools/measure_memory.py --dem docs/results/tuning-bb144/model.dem "
    "--out docs/results/memory/decoder-memory.json --markdown docs/results/memory/decoder-memory.md",
    f"python docs/tools/collect_results.py --verify {RAW}/verify/summary.json "
    f"--bench {RAW}/bench/results.json --ler {RAW}/ler/results.json "
    "--tune docs/results/tuning-bb144/results.json --out docs/results",
    "python docs/tools/make_results.py",
]

TARGET_COMMANDS = [
    REPRODUCE_COMMANDS[0],
    REPRODUCE_COMMANDS[1],
    f"ELDPC_REFERENCE_LIB=build-ref/lib/libeldpc_core_avx2.so python verify/run_verify.py --quick "
    f"--out {RAW}/target-verify",
    f"python bench/bench.py all --problem bb144 --out {RAW}/target-bench",
    "python bench/bench.py ler --problem surface --config a03125c3 --config ms30 "
    f"--ler-shots 500000 --out {RAW}/target-ler",
    f"python docs/tools/collect_results.py --verify {RAW}/target-verify/summary.json "
    f"--bench {RAW}/target-bench/results.json --ler {RAW}/target-ler/results.json "
    "--out docs/results/target",
    "python docs/tools/make_results.py",
]

# The deployment class of the target section is the RTX 6000 Ada board with a Sapphire Rapids Xeon W
# host.  Records from other parts render as a proxy for it, with what relates each part to the class
# (keyed by the GPU and CPU model names the records carry).
TARGET_BOARD = "NVIDIA RTX 6000 Ada Generation"
TARGET_PROXY_NOTES = {
    "NVIDIA L40S": "the same AD102 chip configuration as the RTX 6000 Ada, compute capability 8.9 "
                   "and 142 SMs, on a different board whose clocks and power limit differ",
    "Intel(R) Xeon(R) w9-3495X": "a Sapphire Rapids-W workstation CPU, the same family as the "
                                 "Xeon W5-3435X",
}


# ---------------------------------------------------------------------------
# formatting


def us(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def num(value) -> str:
    return "n/a" if value is None else f"{value}"


def sci(value) -> str:
    return "n/a" if value is None else f"{value:.3e}"


def mhz(value) -> str:
    return "n/a" if value is None else f"{value:g} MHz"


def bytes_(value) -> str:
    if value is None:
        return "n/a"
    if value >= 1 << 20:
        return f"{value / (1 << 20):.1f} MiB"
    return f"{value} B"


def code_block(commands) -> list[str]:
    out = ["```bash"]
    for command in commands:
        out.append(command)
    out.append("```")
    return out


def clocks_text(clocks) -> str:
    if not clocks:
        return "n/a"
    if "graphics_mhz" in clocks:
        text = f"graphics {mhz(clocks.get('graphics_mhz'))}, memory {mhz(clocks.get('memory_mhz'))}"
    else:
        before, after = clocks.get("before") or {}, clocks.get("after") or {}
        parts = []
        for side, rec in (("before", before), ("after", after)):
            if rec:
                parts.append(f"{side} graphics {mhz(rec.get('graphics_mhz'))}, "
                             f"memory {mhz(rec.get('memory_mhz'))}")
        text = "; ".join(parts) if parts else "n/a"
    source = clocks.get("source")
    if source:
        text += f" ({source})"
    return text


def driver_text(env: dict) -> str:
    """The NVIDIA driver version and the CUDA version it supports, as far as the record has them."""
    driver = env.get("driver")
    cuda = env.get("cuda_driver_version")
    if driver and cuda:
        return f"NVIDIA driver {driver} (CUDA driver API {cuda})"
    if driver:
        return f"NVIDIA driver {driver}"
    if cuda:
        return f"CUDA driver API {cuda}"
    return "driver n/a"


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def gpu_cpu(env: dict) -> tuple[str, str]:
    """The GPU and CPU model names of one record (the tuner records the CPU as a string)."""
    gpu = (env.get("gpu") or {}).get("name") or "unknown GPU"
    cpu = env.get("cpu")
    cpu = (cpu.get("model") if isinstance(cpu, dict) else cpu) or "unknown CPU"
    return gpu, cpu


def machines_text(records: list[dict]) -> str:
    """The hardware classes of a set of records, by GPU and CPU model (no host names)."""
    seen = []
    for data in records:
        env = data.get("environment") or {}
        gpu, cpu = gpu_cpu(env)
        cc = (env.get("gpu") or {}).get("compute_capability")
        item = f"{gpu}{f' (compute capability {cc})' if cc else ''} with {cpu}"
        if item not in seen:
            seen.append(item)
    return "; ".join(seen) if seen else "an unrecorded machine"


def run_line(data: dict) -> str:
    """The one-line provenance a table carries: machine, command, commit."""
    env = data.get("environment") or {}
    gpu = env.get("gpu") or {}
    cpu = env.get("cpu") or {}
    gpu_bits = [gpu.get("name") or "unknown GPU"]
    if gpu.get("compute_capability"):
        gpu_bits.append(f"compute capability {gpu['compute_capability']}")
    if gpu.get("sm_count"):
        gpu_bits.append(f"{gpu['sm_count']} SMs")
    if gpu.get("l2_bytes"):
        gpu_bits.append(f"{bytes_(gpu['l2_bytes'])} L2")
    cpu_bits = [cpu.get("model") or "unknown CPU"]
    if cpu.get("logical_cores"):
        cpu_bits.append(f"{cpu['logical_cores']} logical cores")
    if cpu.get("pinned_cpu") is not None:
        cpu_bits.append(f"pinned to CPU {cpu['pinned_cpu']}")
    return (
        f"**Measured on:** {'; '.join([', '.join(gpu_bits), ', '.join(cpu_bits)])}; "
        f"{driver_text(env)}; CUDA toolkit {env.get('toolkit') or 'n/a'}; "
        f"clocks {clocks_text(env.get('clocks'))}.  \n"
        f"**Command:** `{data.get('command') or 'n/a'}`  \n"
        f"**Commit:** `{data.get('commit') or 'n/a'}`"
        + ("" if data.get("package_tree_clean") else
           " (the package's sources or measurement code had uncommitted changes when the run "
           "was collected)")
    )


def env_table(env: dict) -> str:
    gpu = env.get("gpu") or {}
    cpu = env.get("cpu") or {}
    freq = env.get("cpu_frequency") or {}
    if cpu.get("physical_cores") is not None:
        cores = f"{num(cpu.get('physical_cores'))} physical / {num(cpu.get('logical_cores'))} logical"
    else:
        cores = f"{num(cpu.get('logical_cores'))} logical"
    rows = [
        ("GPU", gpu.get("name") or "n/a"),
        ("GPU SMs / L2", f"{num(gpu.get('sm_count'))} / {bytes_(gpu.get('l2_bytes'))}"),
        ("GPU compute capability", gpu.get("compute_capability") or "n/a"),
        ("CPU", cpu.get("model") or "n/a"),
        ("CPU cores", cores),
        ("Pinned CPU", num(cpu.get("pinned_cpu"))),
        ("NVIDIA driver", env.get("driver") or "n/a"),
        ("CUDA driver API", env.get("cuda_driver_version") or "n/a"),
        ("CUDA runtime (in eldpc)", env.get("cuda_runtime") or "n/a"),
        ("CUDA toolkit (nvcc on the machine)", env.get("toolkit") or "n/a"),
        ("GPU clocks", clocks_text(env.get("clocks"))),
        ("Python", env.get("python") or "n/a"),
        ("Platform", env.get("platform") or "n/a"),
    ]
    if freq.get("governor") or freq.get("cur_freq_khz"):
        rows.insert(6, ("CPU governor / frequency",
                        f"{freq.get('governor') or 'n/a'} / {num(freq.get('cur_freq_khz'))} kHz / "
                        f"boost {freq.get('boost') or 'unknown'}"))
    eldpc = env.get("eldpc") or {}
    if eldpc.get("version"):
        rows.append(("eldpc", f"{eldpc.get('version')} (ISA {eldpc.get('isa') or 'n/a'})"))
    rows = [r for r in rows if r[1] not in (None, "n/a", "None", "None / n/a")
            and not all(part.strip() in ("n/a", "None", "") for part in str(r[1]).split("/"))]
    return table(["field", "value"], rows)


def require(data: dict, kind: str, path: Path) -> dict:
    if data.get("kind") != kind:
        raise SystemExit(f"make_results: {path} has kind {data.get('kind')!r}, expected {kind!r}")
    if not data.get("commit"):
        raise SystemExit(f"make_results: {path} names no commit")
    return data


def has_reference_lib(rows) -> bool:
    return any(row.get("status_mismatches_vs_reference_lib") is not None for row in rows or [])


# ---------------------------------------------------------------------------
# sections


def header(records: list[dict], target_number: int | None = None) -> list[str]:
    other = ([f"Section {target_number} was measured on a different machine, named there with the",
              "warnings of its runs.", ""] if target_number else [])
    return [
        "# eldpc results",
        "",
        "Generated by `python docs/tools/make_results.py` from the results files in `docs/results/`.",
        "Do not edit this file by hand: every number below is read from those files, and",
        "`python docs/tools/make_results.py --check` fails when this file differs from what the",
        "script generates.",
        "",
        "**All timings in this document are screens from a shared development machine, not",
        "benchmark results.**  The runs were taken on a machine of this class: "
        f"{machines_text(records)}.  The machine was shared with other work, so latency and",
        "throughput are indicative only, and the benchmark's warnings for each run are reproduced",
        "with it.  Measure latency on your own machine with `bench/bench.py` (see `docs/BENCH.md`).",
        "",
        "The package's timing-grade numbers are machine A's, at stock GPU clocks and with a validated",
        "overclock (`README.md`, \"Results\"); the overclock profile, how it was validated and its",
        "caveat are in [SUPPORT.md, \"The validated GPU overclock\"](SUPPORT.md#the-validated-gpu-overclock).",
        "",
        *other,
        "Decoding outputs are exact and do not depend on the machine: the corrections, observables",
        "and status words of a shot are fixed by the problem and the configuration.  Latency,",
        "throughput and the width of an LER interval are properties of the machine and of the",
        "number of shots.",
        "",
    ]


def lib_status_total(data: dict):
    """The status-word mismatches against the separately built library over all rows: the
    summary's total when it records one, else the sum of the rows' column; None without a library."""
    total = (data.get("totals") or {}).get("status_mismatches_vs_reference_lib")
    if total is not None:
        return total
    rows = data.get("rows") or []
    if not has_reference_lib(rows):
        return None
    return sum(int(r.get("status_mismatches_vs_reference_lib") or 0) for r in rows)


def verify_section(data: dict) -> list[str]:
    totals = data.get("totals") or {}
    env = data.get("environment") or {}
    rows_in = data.get("rows") or []
    lines = ["## 1. Exactness: bit identity against the reference decoder", ""]
    lines += [
        "`verify/run_verify.py` decodes the same public-seeded shots with `eldpc` and with the packaged",
        "reference (ldpc 2.4.1's BP+OSD-0 with the deterministic OSD tie order), through the public",
        "Python API only.  A row passes when every correction and every observable prediction is",
        "bit-identical, single-shot and in batch, and when the batch equals the single-shot loop.",
        "",
        run_line(data),
        "",
        f"**Result: {(data.get('status') or 'unknown').upper()}** -- {num(totals.get('rows'))} rows, "
        f"{num(totals.get('problems'))} problems, {num(totals.get('configs'))} presets, "
        f"{num(totals.get('shots'))} decoded shots, {num(totals.get('mismatches'))} mismatches"
        + (f", including {num(lib_status_total(data))} status-word mismatches against the separately "
           f"built library." if lib_status_total(data) is not None else "."),
        "",
    ]
    with_lib = has_reference_lib(rows_in)
    headers = ["problem", "config", "m", "n", "k", "shots", "mismatches", "corr", "obs",
               "batch corr", "batch obs", "single vs batch", "status vs reference"]
    if with_lib:
        headers.append("status vs reference lib")
    rows = []
    for row in rows_in:
        cells = [row.get("problem"), row.get("config"), num(row.get("num_detectors")),
                 num(row.get("num_mechanisms")), num(row.get("num_observables")),
                 num(row.get("shots")), num(row.get("mismatches")),
                 num(row.get("correction_mismatches")), num(row.get("observables_mismatches")),
                 num(row.get("batch_correction_mismatches")),
                 num(row.get("batch_observables_mismatches")),
                 num(row.get("single_batch_mismatches")),
                 num(row.get("status_mismatches_vs_reference"))]
        if with_lib:
            cells.append(num(row.get("status_mismatches_vs_reference_lib")))
        rows.append(cells)
    lines.append(table(headers, rows))
    lines.append("")
    lib = data.get("reference_lib")
    if with_lib:
        lines += [
            "The column \"status vs reference lib\" compares the status words with a separately built",
            f"library of the same decoder sources (`{(lib or {}).get('name') or 'unnamed'}`, given",
            "with `--reference-lib` or `ELDPC_REFERENCE_LIB`), independent of the installed package;",
            "the other columns compare with the packaged reference.",
            "",
        ]
    else:
        lines += [
            "This run had no separately built library of the decoder sources (`--reference-lib` or",
            "`ELDPC_REFERENCE_LIB`), so the status words were compared with the packaged reference only.",
            "",
        ]
    lines += [
        "Environment of this run:",
        "",
        env_table(env),
        "",
        f"Reference build: `{data.get('reference_build_info') or 'n/a'}`.",
        "",
    ]
    return lines


def bench_section(data: dict) -> list[str]:
    env = data.get("environment") or {}
    problem = data.get("problem") or {}
    settings = data.get("settings") or {}
    lines = ["## 2. Benchmark screen (shared development machine)", ""]
    lines += [
        "Every timing in this section is a **screen from a shared development machine, not a",
        "benchmark result**: it was taken with the benchmark's quick settings on a machine shared",
        "with other work, and the benchmark's own warnings for the run are reproduced below.",
        "Decoding outputs are machine-independent (section 1); only latency and throughput are",
        "properties of the machine and its load.",
        "",
        run_line(data),
        "",
        "Problem: "
        f"`{problem.get('name') or 'n/a'}` ({problem.get('source') or 'n/a'}), "
        f"m = {num(problem.get('m'))}, n = {num(problem.get('n'))}, k = {num(problem.get('k'))}, "
        f"nnz(H) = {num(problem.get('nnz_h'))}, nnz(O) = {num(problem.get('nnz_o'))}; "
        f"DEM sha256 `{problem.get('dem_sha256') or 'n/a'}`.",
        "",
    ]
    warnings = data.get("warnings") or []
    if warnings:
        lines.append("Warnings recorded by the benchmark for this run:")
        lines.append("")
        lines += [f"- {w}" for w in warnings]
        lines.append("")
    else:
        lines += ["The benchmark recorded no warnings for this run.", ""]

    # -- C single-shot latency
    c = data.get("c_latency") or {}
    modes = c.get("modes") or {}
    lines += [
        "### C API single-shot latency",
        "",
        "One `eldpc_decode` call per shot, `CLOCK_MONOTONIC` around the call; creating the decoder and",
        "the warm-up shots are outside.  Percentiles are nearest-rank over the pooled per-shot samples.",
        "",
        "| mode | gap (us) | variant | calls | p50 (us) | p99 (us) | p99.9 (us) | mean (us) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("back_to_back", "cadence", "cadence_keep_warm", "cadence_keep_warm_pair"):
        mode = modes.get(name)
        if not mode:
            continue
        for side in ("A", "B"):
            v = (mode.get("variants") or {}).get(side)
            if not v:
                continue
            pooled = v.get("pooled") or {}
            label = v.get("label") or side
            if mode.get("description"):
                label = f"{label} ({'keep-warm' if side == 'A' else 'no keep-warm'})"
            lines.append(
                f"| {name} | {num(mode.get('gap_us'))} | {label} | {num(pooled.get('shots'))} | "
                f"{us(pooled.get('p50_us'))} | {us(pooled.get('p99_us'))} | "
                f"{us(pooled.get('p99_9_us'))} | {us(pooled.get('mean_us'))} |")
    lines.append("")
    control = (modes.get("back_to_back") or {}).get("a_a_control") or {}
    band = control.get("band") or {}
    if band:
        p99 = band.get("p99_us") or [None, None]
        lines += [
            "The A/A control (two handles of the same decoder, same shots) gives the machine's own",
            "run-to-run band, i.e. the smallest difference this setup resolves: "
            f"p50 [{us((band.get('p50_us') or [None, None])[0])}, "
            f"{us((band.get('p50_us') or [None, None])[1])}] us, "
            f"p99 [{us(p99[0])}, {us(p99[1])}] us.",
            "",
        ]
        if p99[0] is not None and p99[1] is not None and p99[1] - p99[0] > 100:
            lines += [
                "The p99 band is wide because the tail of this run was dominated by interference",
                "(see the warnings above); for this run the p50 band is the resolution figure that",
                "means something, and the p99 and p99.9 columns are not differences the machine can",
                "resolve.",
                "",
            ]
    kw = modes.get("cadence_keep_warm_pair") or {}
    kwc = kw.get("comparison") or {}
    if kwc and kwc.get("delta"):
        delta = kwc["delta"]
        lines += [
            "Keep-warm against no keep-warm at the same cadence (paired, same sessions): "
            f"p50 difference {us(delta.get('p50_us'))} us, p99 difference {us(delta.get('p99_us'))} us "
            f"({num(kwc.get('aa_band_sessions'))} control session deltas in the band).",
            "",
        ]

    # -- Python latency
    py = data.get("py_latency") or {}
    pymode = (py.get("modes") or {}).get("back_to_back") or {}
    pyv = ((pymode.get("variants") or {}).get("A") or {}).get("pooled") or {}
    if pyv:
        lines += [
            "### Python per-shot latency",
            "",
            "`Decoder.decode(syndrome)` from Python with the syndrome already a `uint8` numpy array;",
            "includes the Python and ctypes crossing and the allocation of the returned correction.",
            "",
            "| variant | calls | p50 (us) | p99 (us) | p99.9 (us) | mean (us) |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {((pymode.get('variants') or {}).get('A') or {}).get('label') or 'A'} | "
            f"{num(pyv.get('shots'))} | {us(pyv.get('p50_us'))} | {us(pyv.get('p99_us'))} | "
            f"{us(pyv.get('p99_9_us'))} | {us(pyv.get('mean_us'))} |",
            "",
        ]

    # -- throughput
    th = data.get("throughput") or {}
    variants = th.get("variants") or {}
    if variants:
        lines += [
            "### Batch throughput",
            "",
            f"`Decoder.decode_batch` over batches of {num(th.get('batch_shots'))} shots, as shots per",
            "second.  The C-driver row is the same workload through `eldpc_decode_batch` with reused",
            "output buffers (the decoder's own batch throughput); the Python row includes the Python",
            "layer.",
            "",
            "| path | shots | seconds | shots/s |",
            "|---|---:|---:|---:|",
        ]
        for side in ("A", "B"):
            v = variants.get(side)
            if not v:
                continue
            lines.append(f"| Python, {v.get('label') or side} | {num(v.get('shots'))} | "
                         f"{num(v.get('seconds'))} | {num(v.get('shots_per_s'))} |")
        cdriver = th.get("c_driver") or {}
        cvars = cdriver.get("variants") or {}
        for side in ("A", "B"):
            v = cvars.get(side)
            if not v:
                continue
            lines.append(f"| C driver, {v.get('label') or side} | {num(v.get('shots'))} | "
                         f"{num(v.get('seconds'))} | {num(v.get('shots_per_s'))} |")
        lines.append("")

    # -- LER of the bench run
    ler = data.get("ler") or {}
    lvars = ler.get("variants") or {}
    if lvars:
        lines += [
            "### Logical error rate of the benchmark problem",
            "",
            f"{num(ler.get('shots'))} shots were sampled from the problem's model and decoded with",
            "`decode_batch`; a shot is wrong when any predicted observable differs from the true one.",
            "The interval is the 95% Wilson score interval.",
            "",
            "| variant | shots | errors | LER | 95% CI |",
            "|---|---:|---:|---:|---|",
        ]
        for side in ("A", "B"):
            v = lvars.get(side)
            if not v:
                continue
            ci = v.get("ci95") or [None, None]
            lines.append(f"| {v.get('label') or side} | {num(v.get('shots'))} | {num(v.get('errors'))} | "
                         f"{sci(v.get('ler'))} | [{sci(ci[0])}, {sci(ci[1])}] |")
        lines.append("")

    speed = {"c_latency": settings.get("c_shots"), "ler_shots": settings.get("ler_shots"),
             "batch_shots": settings.get("throughput_batch"), "processes": settings.get("processes")}
    lines += [
        "Measurement settings of the run: "
        + ", ".join(f"{k} = {v}" for k, v in speed.items() if v is not None) + ".",
        "",
        "Environment of this run:",
        "",
        env_table(env),
        "",
    ]
    return lines


def ler_section(data: dict) -> list[str]:
    env = data.get("environment") or {}
    problem = data.get("problem") or {}
    ler = data.get("ler") or {}
    lines = ["## 3. LER check", ""]
    lines += [
        "A separate `bench.py ler` run with more shots than the quick benchmark: shots are sampled",
        "from the problem's model and decoded through the public API; the LER is the fraction of shots",
        "whose predicted observables differ from the true ones.  Two presets decode the same shots,",
        "so their difference is paired.  The interval is the 95% Wilson score interval.",
        "",
        run_line(data),
        "",
        "Problem: "
        f"`{problem.get('name') or 'n/a'}` ({problem.get('source') or 'n/a'}), "
        f"m = {num(problem.get('m'))}, n = {num(problem.get('n'))}, k = {num(problem.get('k'))}; "
        f"DEM sha256 `{problem.get('dem_sha256') or 'n/a'}`; shot seed "
        f"{num(((problem.get('shots') or {}).get('ler') or {}).get('seed'))}.",
        "",
        "| preset | shots | errors | LER | 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for side in ("A", "B"):
        v = (ler.get("variants") or {}).get(side)
        if not v:
            continue
        ci = v.get("ci95") or [None, None]
        lines.append(f"| {v.get('label') or side} | {num(v.get('shots'))} | {num(v.get('errors'))} | "
                     f"{sci(v.get('ler'))} | [{sci(ci[0])}, {sci(ci[1])}] |")
    lines.append("")
    comp = ler.get("comparison") or {}
    if comp:
        lines.append(f"Difference (A - B): {sci(comp.get('delta_ler'))}; 95% intervals disjoint: "
                     f"{comp.get('ci95_disjoint')}.")
        lines.append("")
    warnings = data.get("warnings") or []
    if warnings:
        lines.append("Warnings recorded by the benchmark for this run:")
        lines.append("")
        lines += [f"- {w}" for w in warnings]
        lines.append("")
    lines += ["Environment of this run:", "", env_table(env), ""]
    return lines


def choice_lines(data: dict) -> list[str]:
    """The chosen configuration of one tuning record: the v1 screen line, or the v2 confirmation."""
    choice = data.get("choice") or {}
    tool = data.get("tool") or {}
    version = f"eldpc-tune {tool.get('version') or 'n/a'}"
    if choice.get("ler_source") != "accuracy_confirmation":  # an eldpc.tune.v1 record
        return [
            f"Chosen configuration: `{choice.get('config_string') or 'n/a'}` -- LER upper bound "
            f"{sci(choice.get('ler_upper'))}, p50 {us(choice.get('p50_us'))} us, p99 "
            f"{us(choice.get('p99_us'))} us ({version}, "
            f"{num(choice.get('eligible'))} eligible of {num(choice.get('measured'))} measured points)."]
    ac = data.get("accuracy_confirmation") or {}
    if not choice.get("config"):
        return [f"No configuration was accepted ({version}): {choice.get('reason') or 'n/a'}."]
    screen = choice.get("screen") or {}
    latency = data.get("latency_confirmation") or {}
    lines = [
        f"Chosen configuration: `{choice.get('config_string') or 'n/a'}` -- LER "
        f"{sci(choice.get('ler'))} with one-sided upper bound {sci(choice.get('ler_upper'))} from "
        f"the accuracy confirmation ({num(ac.get('planned_shots'))} fresh shots per case, no early "
        f"stop; the screen's upper bound {sci(screen.get('upper'))} is not the reported number); "
        f"p50 {us(choice.get('p50_us'))} us, p99 {us(choice.get('p99_us'))} us from the latency "
        f"confirmation ({num(latency.get('distinct_syndromes'))} distinct syndromes, slow fraction "
        f"{sci(choice.get('slow_fraction'))}{', p99 unreliable' if choice.get('p99_unreliable') else ''}) "
        f"({version}, {num(choice.get('eligible'))} eligible of {num(choice.get('measured'))} "
        f"measured points)."]
    variants = data.get("core_variants") or {}
    if variants.get("evaluated"):
        faster = variants.get("faster")
        verdict = faster if faster == "no difference" else f"{faster} is faster"
        env = variants.get("recommended_env")
        if env:
            recommendation = f"recommended: `{env}`"
        else:  # the device-class table's own choice stands: nothing to set
            recommendation = (f"recommended: keep the device-class table's "
                              f"`{variants.get('table_variant') or 'n/a'}` (no variable to set)")
        lines += ["", f"Core variants: {verdict} (ada64 / default p50 "
                  f"x{variants.get('ratio_p50'):.3f}, p99 x{variants.get('ratio_p99'):.3f}); "
                  f"{recommendation}."]
    return lines


def tuner_text(data: dict) -> str:
    """The tuner version a tuning record was written by: another version can choose another point
    from the same kind of run (tuning/README.md has the rules of each)."""
    version = (data.get("tool") or {}).get("version")
    if not version:
        return "version not recorded"
    return (f"eldpc-tune {version} (the choice below is that version's; another tuner version can "
            f"choose another point from the same run)")


def tuning_section(records: list[dict]) -> list[str]:
    lines = ["## 4. Tuning run (shared development machine)", ""]
    lines += [
        "A run of the package's tuner (`tuning/eldpc_tune.py`) with `--quick`, recorded to show the",
        "documented workflow end to end: it samples shots from a detector error model, measures the",
        "LER and latency of the preset grid, refines around the frontier and writes a configuration",
        "that `eldpc.Decoder(config=...)` loads.  `--quick` is a smoke test, not a tuning run, and the",
        "latencies are screens from a shared development machine: the chosen configuration is not a",
        "recommendation for another machine or code.",
        "",
    ]
    for data in records:
        env = data.get("environment") or {}
        gpu = env.get("gpu") or {}
        problem = data.get("problem") or {}
        directory = data.get("_dir")
        if (data.get("choice") or {}).get("config"):
            files = (f"The full record (the command, the environment, the Pareto table and the chosen "
                     f"config) is in [`results/{directory}/`](results/{directory}/); the configuration "
                     f"file is [`results/{directory}/config.json`](results/{directory}/config.json).")
        else:
            files = (f"The full record (the command, the environment and the Pareto table) is in "
                     f"[`results/{directory}/`](results/{directory}/); no configuration was accepted, "
                     f"so the run wrote no `config.json`.")
        lines += [
            f"**Measured on:** {gpu.get('name') or 'n/a'} (compute capability "
            f"{gpu.get('compute_capability') or 'n/a'}, {num(gpu.get('sm_count'))} SMs); "
            f"{env.get('cpu') or 'n/a'} ({num(env.get('logical_cores'))} logical cores); "
            f"{driver_text(env)}; CUDA toolkit {env.get('toolkit') or 'n/a'}; "
            "clocks not recorded by the tuner.  ",
            f"**Command:** `{data.get('command') or 'n/a'}`  ",
            f"**Commit:** `{data.get('commit') or 'n/a'}`  ",
            f"**Tuner:** {tuner_text(data)}",
            "",
            f"Problem: `{problem.get('label') or 'n/a'}` ({problem.get('source') or 'n/a'}), "
            f"m = {num(problem.get('num_detectors'))}, n = {num(problem.get('num_mechanisms'))}, "
            f"k = {num(problem.get('num_observables'))}; DEM file sha256 "
            f"`{problem.get('dem_file_sha256') or 'n/a'}`; parsed-DEM sha256 "
            f"`{problem.get('dem_sha256') or 'n/a'}` (the tuner records the hash of the parsed model "
            "text, which differs from the file's bytes for a file that ends in a newline).",
            "",
            *choice_lines(data),
            "",
            files,
            "",
        ]
    return lines


def memory_section(number: int, data: dict) -> list[str]:
    env = data.get("environment") or {}
    gpu = env.get("gpu") or {}
    p = data.get("problem") or {}

    def mib_rss(v):
        return "n/a" if v is None else f"{v / 1024:.1f}"

    def mib_gpu(v):
        return "n/a" if v is None else f"{v}"

    rows = [
        ("CUDA context (tiny decoder, then closed)", data.get("context") or {}),
        ("first handle for the problem", data.get("first_handle") or {}),
        ("second handle, first alive", data.get("second_handle") or {}),
    ]
    lines = [f"## {number}. Decoder memory", ""]
    lines += [
        "What one decoder handle adds to the process, measured by `docs/tools/measure_memory.py`:",
        "host RSS and the driver's per-process GPU memory accounting, before and after creating the",
        "handles (median of repeated samples).  `docs/ARCH.md` section 8 derives the live tables",
        "from the allocation sites; RSS also counts the create-time temporaries and the allocator's",
        "retained space.",
        "",
        f"**Measured on:** {gpu.get('name') or 'n/a'} (compute capability "
        f"{gpu.get('compute_capability') or 'n/a'}, {num(gpu.get('sm_count'))} SMs); "
        f"{env.get('cpu') or 'n/a'}; {driver_text(env)}.  ",
        f"**Command:** `{data.get('command') or 'n/a'}`  ",
        f"**Commit:** `{data.get('commit') or 'n/a'}`",
        "",
        f"Problem: m = {num(p.get('m'))}, n = {num(p.get('n'))}, k = {num(p.get('k'))}, "
        f"nnz(H) = {num(p.get('nnz_h'))}, nnz(O) = {num(p.get('nnz_o'))}; DEM sha256 "
        f"`{p.get('dem_sha256') or 'n/a'}`.",
        "",
        "| measurement | RSS (MiB) | GPU memory (MiB) |",
        "|---|---:|---:|",
    ]
    for label, rec in rows:
        lines.append(f"| {label} | {mib_rss(rec.get('rss_kib_delta'))} | "
                     f"{mib_gpu(rec.get('gpu_mib_delta'))} |")
    lines += ["", "The record is [`results/memory/decoder-memory.json`](results/memory/decoder-memory.json).", ""]
    return lines


def _pooled(bench: dict, kind: str, mode: str = "back_to_back", side: str = "A") -> dict:
    section = bench.get(kind) or {}
    modes = section.get("modes") or {}
    return ((((modes.get(mode) or {}).get("variants") or {}).get(side) or {}).get("pooled") or {})


def target_section(number: int, main_records: list[dict], target: dict | None = None) -> list[str]:
    lines = [f"## {number}. The RTX 6000 Ada class with a Sapphire Rapids Xeon", ""]
    if not target:
        return lines + [
            "**To be measured.**  The build carries SASS for `sm_89` (the RTX 6000 Ada class) and an",
            "AVX-512 host core whose instruction set Sapphire Rapids Xeons support, but the records",
            f"above ran on {machines_text(main_records)}.  This section is a placeholder printed",
            "while `docs/results/target/` holds no records.  To produce them, run from the package",
            "root on such a machine:",
            "",
            *code_block(TARGET_COMMANDS),
            "",
            "`--out docs/results/target` keeps that machine's three records apart from the ones",
            "above; the regenerated document then renders them in this section, each with its own",
            "environment block and commit.  The C latency measurement needs an idle machine with",
            "exclusive use of the GPU and a pinned CPU (see `docs/BENCH.md`); the verification and",
            "LER numbers do not depend on the timing conditions.",
            "",
        ]
    envs = [(target[k].get("environment") or {}) for k in ("verify", "bench", "ler") if target.get(k)]
    parts = []
    for env in envs:
        for name in gpu_cpu(env):
            if name not in parts:
                parts.append(name)
    described = " with ".join(f"{name} ({TARGET_PROXY_NOTES[name]})" if name in TARGET_PROXY_NOTES
                              else name for name in parts)
    if parts[:1] == [TARGET_BOARD]:
        lines += [f"Measured on {described}."]
    else:
        lines[0] = f"## {number}. The RTX 6000 Ada class with a Sapphire Rapids Xeon (proxy machine)"
        lines += [
            f"Measured on a proxy for that class, not on the deployment pair itself (an RTX 6000 Ada with a "
            f"Sapphire Rapids host): {described}.",
            "Device-side absolute times stand in for the board only as far as its clocks and power",
            "limit match; the host side is the measured CPU's.",
        ]
    lines += [
        "The full records are committed under `results/target/` and each subsection names the",
        "command, commit and environment of its run.",
        "",
    ]
    return lines + _target_subsections(number, target)


def _target_subsections(number: int, target: dict) -> list[str]:
    lines = []
    tv = target.get("verify")
    if tv:
        totals = tv.get("totals") or {}
        lib_total = lib_status_total(tv)
        lines += [
            f"### {number}.1 Verification",
            "",
            run_line(tv),
            "",
            f"**Result: {(tv.get('status') or 'unknown').upper()}** -- {num(totals.get('rows'))} rows, "
            f"{num(totals.get('problems'))} problems, {num(totals.get('configs'))} presets, "
            f"{num(totals.get('shots'))} shots, {num(totals.get('mismatches'))} mismatches"
            + (f", including {num(lib_total)} status-word mismatches against the separately built "
               f"library." if lib_total is not None else "."),
            "",
        ]
        lib_info = ((tv.get("reference_lib") or {}).get("build_info") or "")
        toolkit = re.search(r"cuda=[^;]*; cuda_arch=[^;]*", lib_info)
        if lib_total is not None and toolkit:
            lines += [f"Separately built library (`docs/VERIFY.md`, \"Status words\"): `{toolkit.group(0)}`.",
                      ""]
    tb = target.get("bench")
    if tb:
        problem = tb.get("problem") or {}
        c = _pooled(tb, "c_latency")
        py = _pooled(tb, "py_latency")
        th = ((tb.get("throughput") or {}).get("variants") or {}).get("A") or {}
        lines += [
            f"### {number}.2 Benchmark",
            "",
            run_line(tb),
            "",
            f"Problem `{problem.get('name') or 'n/a'}` (m = {num(problem.get('m'))}, "
            f"n = {num(problem.get('n'))}, k = {num(problem.get('k'))}).",
            "",
            "| measurement | value |",
            "|---|---:|",
            f"| C single-shot p50 (us) | {us(c.get('p50_us'))} |",
            f"| C single-shot p99 (us) | {us(c.get('p99_us'))} |",
            f"| Python single-shot p50 (us) | {us(py.get('p50_us'))} |",
            f"| batch throughput (shots/s) | {num(th.get('shots_per_s'))} |",
            f"| batch size (shots) | {num((tb.get('throughput') or {}).get('batch_shots'))} |",
            "",
        ]
        warnings = tb.get("warnings") or []
        if warnings:
            lines.append("Warnings recorded by the benchmark:")
            lines.append("")
            lines += [f"- {w}" for w in warnings]
            lines.append("")
    tl = target.get("ler")
    if tl:
        ler = tl.get("ler") or {}
        problem = tl.get("problem") or {}
        lines += [
            f"### {number}.3 LER check",
            "",
            run_line(tl),
            "",
            f"Problem `{problem.get('name') or 'n/a'}` (m = {num(problem.get('m'))}, "
            f"n = {num(problem.get('n'))}, k = {num(problem.get('k'))}).",
            "",
            "| preset | shots | errors | LER | 95% CI |",
            "|---|---:|---:|---:|---|",
        ]
        for side in ("A", "B"):
            v = (ler.get("variants") or {}).get(side)
            if not v:
                continue
            ci = v.get("ci95") or [None, None]
            lines.append(f"| {v.get('label') or side} | {num(v.get('shots'))} | {num(v.get('errors'))} | "
                         f"{sci(v.get('ler'))} | [{sci(ci[0])}, {sci(ci[1])}] |")
        lines.append("")
        env = tl.get("environment") or {}
        lines += ["Environment of the runs on this class:", "", env_table(env), ""]
        return lines
    if tv:
        lines += ["Environment of the runs on this class:", "",
                  env_table(tv.get("environment") or {}), ""]
    return lines


def reproducing_section(number: int) -> list[str]:
    return [
        f"## {number}. Reproducing these results",
        "",
        "The results files in `docs/results/` are produced by the package's own commands, run from",
        "the package root against the installed package, and collected by a script.  The raw",
        f"outputs go to a directory outside the package tree (`{RAW}` here):",
        "",
        *code_block(REPRODUCE_COMMANDS),
        "",
        "The first two commands build the separately built library of the same decoder sources",
        "(`docs/VERIFY.md`, \"Status words\"; `CUDACXX` names the package's toolkit, as for the",
        "install), and the third",
        "gives it to the verification suite through the environment, so that the status words are",
        "compared with it as well; `collect_results.py` refuses a recorded command that names an",
        "absolute path, so pass a relative path through the variable rather than `--reference-lib`.",
        "",
        "The commands above replace the committed records.  To check a number without replacing them,",
        "give the measurement a relative output path outside `docs/results/` (for example",
        f"`--out {RAW}/memory.json --markdown {RAW}/memory.md` for `measure_memory.py`, and",
        f"`--out {RAW}/tune` for the tuner) and compare the two files; the tools refuse absolute",
        "paths, because the output records the command.",
        "",
        "`collect_results.py` keeps the fields this document prints, removes machine-specific paths",
        "and GPU UUIDs, and records the commit of the package tree the measurement ran on.",
        "`make_results.py` renders this file from those JSON files and nothing else; `--check` exits",
        "non-zero when this file differs from what it generates.",
        "",
        "Decoding is exact and machine-independent, so the verification result transfers between",
        "machines.  Latency, throughput and the LER confidence intervals are measurements of the",
        "machine and the sampling; treat the timings above as screens, not as benchmark results.",
        "",
    ]


# ---------------------------------------------------------------------------


def generate(results_dir: Path) -> str:
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"make_results: no results files in {results_dir}")
    data = {}
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"make_results: cannot read {path}: {exc}")
        kind = obj.get("kind")
        if kind in data:
            raise SystemExit(f"make_results: two results files of kind {kind!r}")
        data[kind] = require(obj, kind, path)
    for kind in ("verify", "bench", "ler"):
        if kind not in data:
            raise SystemExit(f"make_results: no results file of kind {kind!r} in {results_dir}")

    tuning_records = []
    for path in sorted(results_dir.glob("tuning-*/provenance.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("kind") != "tuning":
            raise SystemExit(f"make_results: {path} is not a tuning record")
        require(record, "tuning", path)
        record["_dir"] = path.parent.name
        tuning_records.append(record)
    memory = None
    memory_path = results_dir / "memory" / "decoder-memory.json"
    if memory_path.is_file():
        memory = require(json.loads(memory_path.read_text(encoding="utf-8")), "memory", memory_path)

    main_records = [data["verify"], data["bench"], data["ler"]] + tuning_records
    if memory:
        main_records.append(memory)
    target = {}
    # The target machine's records, by kind (the benchmark's file is bench-quick.json for a
    # --quick run and bench.json for a full-length one; collect_results.bench_record_name).
    for path in sorted((results_dir / "target").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        kind = record.get("kind")
        if kind not in ("verify", "bench", "ler"):
            continue
        if kind in target:
            raise SystemExit(f"make_results: two target records of kind {kind!r}")
        target[kind] = require(record, kind, path)
    target_number = 4 + bool(tuning_records) + bool(memory)
    lines = header(main_records, target_number if target else None)
    lines += verify_section(data["verify"])
    lines += bench_section(data["bench"])
    lines += ler_section(data["ler"])
    number = 4
    if tuning_records:
        lines += tuning_section(tuning_records)
        number += 1
    if memory:
        lines += memory_section(number, memory)
        number += 1
    lines += target_section(number, main_records, target)
    lines += reproducing_section(number + 1)
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="generate docs/RESULTS.md from docs/results/")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when docs/RESULTS.md differs from the generated text")
    parser.add_argument("--results", default=str(RESULTS_DIR), help="results directory")
    parser.add_argument("--out", default=str(OUTPUT), help="output file")
    args = parser.parse_args(argv)
    results_dir = Path(args.results)
    out = Path(args.out)
    text = generate(results_dir)
    if args.check:
        try:
            current = out.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"make_results: cannot read {out}: {exc}", file=sys.stderr)
            return 1
        if current == text:
            print(f"make_results: {out} is up to date")
            return 0
        import difflib

        diff = list(difflib.unified_diff(current.splitlines(), text.splitlines(),
                                         fromfile=str(out), tofile="generated", lineterm=""))
        print("\n".join(diff[:80]), file=sys.stderr)
        print(f"make_results: {out} is out of date ({len(diff)} diff lines)", file=sys.stderr)
        return 1
    out.write_text(text, encoding="utf-8")
    print(f"make_results: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
