#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""eldpc benchmark: measure the decoder on this machine and write DIR/results.json and DIR/results.md.

    python bench.py all --out DIR            # every measurement (default: longer)
    python bench.py c-latency --out DIR      # (a) C API single-shot call
    python bench.py py-latency --out DIR     # (b) Python per-shot call
    python bench.py throughput --out DIR     # (c) batch throughput
    python bench.py ler --out DIR            # (d) logical error rate over sampled shots
    python bench.py all --quick --out DIR    # 2-5 minutes on the default problem, by machine
    python bench.py summary --out DIR        # the headline numbers of an existing results.json
    python bench.py compare --a DIR_A --b DIR_B   # diff two runs, refusing different problems/configs

The three latency numbers are different quantities and are reported separately.  --problem selects a
built-in circuit-level problem (bb144, bb144z, bb72, bb288, hgp98, surface) or a stim DEM file; --config
selects one or two presets (or ABI config strings); --cpu pins the process to a CPU (--cpu auto picks
one) and timing refuses to run unpinned unless --allow-unpinned.  See docs/BENCH.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from benchlib import cdriver, environment, problems  # noqa: E402
from benchlib.measure import Bench, MeasurementError  # noqa: E402
from benchlib.report import build_markdown, core_variants_text, write_json  # noqa: E402

MEASUREMENTS = {
    "c-latency": ("c_latency", "measure_c_latency"),
    "py-latency": ("py_latency", "measure_py_latency"),
    "throughput": ("throughput", "measure_throughput"),
    "ler": ("ler", "measure_ler"),
}
TIMING_COMMANDS = {"c-latency", "py-latency", "throughput", "all"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench.py", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("c-latency", "py-latency", "throughput", "ler", "all"):
        sp = sub.add_parser(name, help={
            "c-latency": "time the C API single-shot call (eldpc_decode)",
            "py-latency": "time the Python per-shot call (Decoder.decode)",
            "throughput": "measure decode_batch throughput in shots per second",
            "ler": "measure the logical error rate over sampled shots",
            "all": "run all four measurements",
        }[name])
        sp.add_argument("--quick", action="store_true", help="fewer shots and sessions (all: 2-5 minutes on the default problem, by machine)")
        sp.add_argument("--problem", default=problems.DEFAULT_PROBLEM,
                        help="built-in problem name or a stim DEM file (default: bb144 when qLDPC is "
                             "installed, else surface)")
        sp.add_argument("--config", action="append", default=None,
                        help="preset name, ABI config string or config .json; give twice to compare two "
                             "configs (default: a03125c3)")
        sp.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu",
                        help="the engine to measure: gpu (the GPU core, default), cpu (the CPU core, "
                             "several times the GPU core's time per shot; use a small --problem with "
                             "--quick) or auto (the GPU when it is usable, else the CPU core); added "
                             "to every config string as device=...")
        sp.add_argument("--lib", default=None, help="the eldpc C library to measure (default: find it)")
        sp.add_argument("--lib-b", default=None, help="a second C library, to compare two builds")
        sp.add_argument("--cpu", default="auto",
                        help="CPU to pin to, or 'auto' (pick one) or 'none' (default: auto)")
        sp.add_argument("--allow-unpinned", action="store_true",
                        help="allow timing without CPU pinning")
        sp.add_argument("--out", default="eldpc-bench-results", help="output directory (default: %(default)s)")
        sp.add_argument("--gap-us", type=int, default=1000,
                        help="idle gap before each call in the cadence modes, microseconds (default: 1000)")
        sp.add_argument("--processes", type=int, default=None,
                        help="sessions per pair (default: quick 4, full 6)")
        sp.add_argument("--shots", type=int, default=None,
                        help="latency calls per variant per session (default: quick 300, full 1500); "
                             "each session draws its own distinct syndromes, so the distinct count per "
                             "handle per pair is shots x processes (use at least 8000 for a p99 budget)")
        sp.add_argument("--ler-shots", type=int, default=None, help="shots for the LER check")
        sp.add_argument("--batch-shots", type=int, default=None, help="shots per decode_batch call")
        sp.add_argument("--order-design", choices=("alternating", "all4"), default="alternating",
                        help="session order assignment: 'alternating' swaps the create and call order "
                             "together between sessions (default); 'all4' runs every combination of create "
                             "order and call order once per four sessions")
        sp.add_argument("--modes", default=None,
                        help="comma-separated c-latency modes to run: back_to_back, cadence, "
                             "cadence_keep_warm, cadence_keep_warm_pair (default: all)")
        sp.add_argument("--no-c-batch", action="store_true",
                        help="skip the C-driver half of the throughput measurement (eldpc_decode_batch "
                             "called from C, next to the Python decode_batch number)")
    sp = sub.add_parser("summary", help="print the key numbers of an existing results.json")
    sp.add_argument("--out", default="eldpc-bench-results", help="directory with results.json")
    sp = sub.add_parser("compare", help="compare two results.json files, refusing different problems, "
                                        "configs or batch sizes")
    sp.add_argument("--a", required=True, help="results directory of side A")
    sp.add_argument("--b", required=True, help="results directory of side B")
    sp.add_argument("--out", default=None, help="also write the comparison text to this file")
    return p


def gpu_error_hint(error, device: str) -> str:
    """For a GPU create error (no usable GPU, an unsupported one, a CPU-only build): how to run
    the benchmark on the CPU core instead; "" for any other error."""
    if device == "gpu" and "to run on the CPU core" in str(error):
        return " (bench.py measures the CPU core with --device cpu, or --device auto)"
    return ""


def parse_configs(values) -> list[str]:
    """The configs of the --config options: each option is one config, and a value without "="
    may also list two preset names or files separated by a comma ("a03125c3,ms30").  A value with
    "=" (an ABI string, or a preset with extra keys) is never split: its cpus= list holds commas."""
    names = []
    for v in values or ["a03125c3"]:
        if "=" in v:
            names.append(v.strip())
        else:
            names.extend([x.strip() for x in v.split(",") if x.strip()])
    if not names:
        raise SystemExit("bench.py: --config needs at least one name")
    if len(names) > 2:
        raise SystemExit("bench.py: at most two configs can be compared")
    return names


def choose_cpu(args, parser, timing: bool):
    """Pin this process (and its children) to one CPU.

    Returns (cpu or None, why --cpu auto chose it, or None when the CPU was not chosen automatically).
    """
    if args.cpu == "none":
        if timing and not args.allow_unpinned:
            parser.error("refusing to time unpinned: pass --cpu N (or --cpu auto), or --allow-unpinned")
        return None, None
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = list(range(os.cpu_count() or 1))
    if args.cpu == "auto":
        cpu, auto = environment.auto_cpu(allowed, environment.gpu_local_cpus(), environment.smt_siblings(0))
    else:
        try:
            cpu = int(args.cpu)
        except ValueError:
            parser.error(f"--cpu must be an integer, 'auto' or 'none': {args.cpu!r}")
        auto = None
        if cpu not in allowed:
            parser.error(f"CPU {cpu} is not available to this process (affinity: {allowed[:16]}"
                         f"{'...' if len(allowed) > 16 else ''})")
    try:
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError) as e:
        if timing and not args.allow_unpinned:
            parser.error(f"cannot pin to CPU {cpu} ({e}); pass --allow-unpinned to time unpinned")
        return None, auto
    return cpu, auto


def summarize(out_dir: str) -> int:
    """Print the headline numbers of an existing results.json (no decoder or GPU needed)."""
    import json

    path = Path(out_dir) / "results.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)
    except (OSError, ValueError) as e:
        print(f"bench.py: cannot read {path}: {e}", file=sys.stderr)
        return 2
    env = r.get("environment") or {}
    gpu, cpu = env.get("gpu") or {}, env.get("cpu") or {}
    tk = env.get("toolkit") or {}
    prob = r.get("problem") or {}
    print(f"results: {path}")
    print(f"problem: {prob.get('name')} ({prob.get('source')}), m={prob.get('m')} n={prob.get('n')} "
          f"k={prob.get('k')}")
    print(f"gpu: {gpu.get('name')} ({gpu.get('sm_count')} SMs, {gpu.get('l2_bytes')} B L2, "
          f"{gpu.get('memory_bytes')} B, cc {gpu.get('compute_capability')})")
    print(f"cpu: {cpu.get('model')} ({cpu.get('logical_cores')} logical), pinned {cpu.get('pinned_cpu')}, "
          f"governor {cpu.get('frequency', {}).get('governor')}")
    print(f"driver: {env.get('driver')}  cuda runtime: {env.get('cuda_runtime')}  "
          f"toolkit: {tk.get('version')}  isa: {env.get('isa')}  version: {env.get('version')}")
    print(f"core variants of the timed handles: {core_variants_text(env.get('core_variants'))}")
    clocks = env.get("clocks") or {}
    b, a = clocks.get("before") or {}, clocks.get("after") or {}
    print(f"gpu clocks before/after: graphics {b.get('graphics_mhz')}/{a.get('graphics_mhz')} MHz, "
          f"memory {b.get('memory_mhz')}/{a.get('memory_mhz')} MHz")
    warns = r.get("warnings") or []
    print(f"warnings: {len(warns)}" + ("" if not warns else " -> " + "; ".join(warns)))
    for key, title in (("c_latency", "C API single-shot"), ("py_latency", "Python per-shot")):
        data = r.get(key)
        if not data:
            continue
        print(f"{title}:")
        for mode_name, mode in (data.get("modes") or {}).items():
            for side in ("A", "B"):
                v = (mode.get("variants") or {}).get(side)
                if not v:
                    continue
                p = v.get("pooled") or {}
                extra = f" [{v.get('treatment')}]" if v.get("treatment") else ""
                print(f"  {mode_name} {side}{extra}: p50 {p.get('p50_us')} us, p99 {p.get('p99_us')} us, "
                      f"p99.9 {p.get('p99_9_us')} us, mean {p.get('mean_us')} us, shots {p.get('shots')}")
            comp = mode.get("comparison")
            if comp:
                print(f"  {mode_name} delta ({comp.get('a')} - {comp.get('b')}): "
                      f"p50 {comp.get('delta', {}).get('p50_us')} us, "
                      f"p99 {comp.get('delta', {}).get('p99_us')} us, "
                      f"p99.9 {comp.get('delta', {}).get('p99_9_us')} us, "
                      f"mean {comp.get('delta', {}).get('mean_us')} us; "
                      f"A/A band (delta p50 range) {comp.get('aa_band', {}).get('p50_us')}")
                if comp.get("delta_blocked"):
                    print(f"  {mode_name} order-blocked delta: "
                          f"p50 {comp['delta_blocked'].get('p50_us')} us, "
                          f"p99 {comp['delta_blocked'].get('p99_us')} us, "
                          f"p99.9 {comp['delta_blocked'].get('p99_9_us')} us, "
                          f"mean {comp['delta_blocked'].get('mean_us')} us; "
                          f"blocks {[(b['create_order'], b['call_order'], b['sessions']) for b in comp.get('order_blocks', [])]}")
            ctl = mode.get("a_a_control") or {}
            if "sides_equal" in ctl:
                print(f"  {mode_name} A/A handles identical outputs for the same shots: "
                      f"{ctl['sides_equal']} (each session decodes its own syndromes)")
            if ctl.get("sem"):
                print(f"  {mode_name} A/A per-session delta SEM: p50 {ctl['sem'].get('p50_us')} us, "
                      f"mean {ctl['sem'].get('mean_us')} us"
                      + (f"; order-blocked delta p50 {ctl['delta_blocked'].get('p50_us')} us"
                         if ctl.get("delta_blocked") else ""))
    tp = r.get("throughput")
    if tp:
        v = (tp.get("variants") or {}).get("A") or {}
        print(f"throughput: {v.get('shots_per_s')} shots/s at {tp.get('batch_shots')} shots per batch")
        comp = tp.get("comparison")
        if comp:
            print(f"throughput ratio B/A: {comp.get('ratio_b_over_a')} (A/A band {comp.get('aa_band_ratio')} over "
                  f"{comp.get('aa_band_sessions')} control sessions)")
        cd = tp.get("c_driver")
        if cd and cd.get("error"):
            print(f"throughput (C driver decode_batch): not measured: {cd['error']}")
        elif cd:
            cv = (cd.get("variants") or {}).get("A") or {}
            print(f"throughput (C driver decode_batch): {cv.get('shots_per_s')} shots/s at "
                  f"{cd.get('batch_shots')} shots per batch")
    ler = r.get("ler")
    if ler:
        for side, v in (ler.get("variants") or {}).items():
            print(f"LER {side}: {v.get('errors')}/{v.get('shots')} = {v.get('ler')} "
                  f"(95% CI {v.get('ci95')})")
    for f in r.get("failures") or []:
        print(f"failure: {f.get('measurement')}: {f.get('error')}")
    return 0


def compare_results(a_dir: str, b_dir: str, out: str | None = None) -> int:
    """Compare two results directories; refuse unless the problem, config and batch size match."""
    from benchlib import compare

    try:
        a = compare.load_results(a_dir)
        b = compare.load_results(b_dir)
        checks = compare.check_comparable(a, b)
    except compare.CompareError as e:
        print(f"bench.py: not comparable: {e}", file=sys.stderr)
        return 2
    text = compare.compare_text(a, b, checks)
    print(text)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "summary":
        return summarize(args.out)
    if args.command == "compare":
        return compare_results(args.a, args.b, args.out)
    args.config = parse_configs(args.config)
    if args.gap_us < 0:
        parser.error("--gap-us must be >= 0")
    for name in ("processes", "shots", "ler_shots", "batch_shots"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    try:
        import eldpc
    except Exception as e:
        print(f"bench.py: cannot import the eldpc package: {type(e).__name__}: {e}\n"
              "Install eldpc, or set PYTHONPATH to the package tree.", file=sys.stderr)
        return 2
    bench = Bench(args, eldpc, BENCH_DIR)
    bench.cpu, auto = choose_cpu(args, parser, timing=args.command in TIMING_COMMANDS)

    bench.S["processes"] = args.processes or bench.S["processes"]
    bench.S["c_shots"] = args.shots or bench.S["c_shots"]
    bench.S["ler_shots"] = args.ler_shots or bench.S["ler_shots"]
    bench.S["throughput_batch"] = args.batch_shots or bench.S["throughput_batch"]
    bench.order_design = args.order_design
    if args.modes:
        valid_modes = {"back_to_back", "cadence", "cadence_keep_warm", "cadence_keep_warm_pair"}
        enabled = [m.strip() for m in args.modes.split(",") if m.strip()]
        unknown = sorted(set(enabled) - valid_modes)
        if unknown:
            parser.error(f"unknown --modes {unknown}; choose from {sorted(valid_modes)}")
        bench.enabled_modes = set(enabled)

    results = {
        "schema": "eldpc.bench.v1",
        "command": "bench.py " + " ".join(sys.argv[1:]),
        "quick": bool(args.quick),
        "settings": dict(bench.S),
    }
    failures = []
    started = time.monotonic()
    try:
        bench.prepare_problem()
        bench.prepare_variants()
        env = environment.snapshot(bench.cpu, eldpc)
        env["cpu"]["auto_selected"] = bool(auto)
        if auto:
            env["cpu"]["auto_reason"] = auto
        bench.env = env
        if auto:
            bench.warn(f"CPU pinned automatically to CPU {bench.cpu} ({auto}); pass --cpu N to fix "
                       "the choice")
        names = ["c-latency", "py-latency", "throughput", "ler"] if args.command == "all" \
            else [args.command]
        for name in names:
            key, method = MEASUREMENTS[name]
            t0 = time.monotonic()
            try:
                data = getattr(bench, method)()
                data["wall_s"] = round(time.monotonic() - t0, 3)
                results[key] = data
                print(f"bench.py: {name}: {data['wall_s']:.1f} s", file=sys.stderr)
                environment.note_gpu_processes(bench.env, name)
            except (MeasurementError, cdriver.DriverError, problems.ProblemError, OSError) as e:
                error = f"{type(e).__name__}: {e}{gpu_error_hint(e, args.device)}"
                results[key] = {"available": False, "error": error}
                failures.append({"measurement": name, "error": error})
                bench.warn(f"{name} could not be measured: {error}")
                print(f"bench.py: {name} failed: {e}{gpu_error_hint(e, args.device)}", file=sys.stderr)
                if args.command != "all":
                    raise
    except Exception as e:
        if not failures:
            print(f"bench.py: {type(e).__name__}: {e}", file=sys.stderr)
            if os.environ.get("ELDPC_BENCH_DEBUG"):
                traceback.print_exc()
            return 2
    results["problem"] = bench.problem_info
    if bench.env is not None:
        # The core variant (Decoder.core_variant / eldpc_core_variant) of every handle timed.
        bench.env["core_variants"] = bench.core_variant_record()
        if any(side == "cpu" for sides in bench.env["core_variants"].values()
               for side in (sides or {}).values()):
            bench.warn("handles ran on the CPU core (--device cpu, or auto without a usable GPU): "
                       "these are CPU-core numbers, not GPU-core numbers")
    results["environment"] = bench.env
    results["warnings"] = bench.warnings
    results["failures"] = failures
    results["elapsed_s"] = round(time.monotonic() - started, 3)
    if bench.env is not None:
        for w in environment.finish(bench.env):
            if w not in bench.warnings:
                bench.warnings.append(w)
        results["warnings"] = bench.warnings

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "results.json", results)
    (out / "results.md").write_text(build_markdown(results), encoding="utf-8")
    print(f"bench.py: wrote {out / 'results.json'} and {out / 'results.md'}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
