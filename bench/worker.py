#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One measurement process of the Python side of the eldpc benchmark.

bench.py starts this script as a fresh process for every session: a Python per-shot latency session
(Decoder.decode, alternating two decoders), a batch-throughput session (Decoder.decode_batch), or a
logical-error-rate session.  A job with "sequential": true (a CPU-core handle) times the two decoders
one at a time instead of alternating them (docs/BENCH.md, "CPU core").  Running each session in its own process keeps sessions independent for the
paired A/A design and keeps one decoder's state out of another measurement.

    python worker.py JOB.json

The job is JSON; the result is written to job["out"] as JSON.  The process prints nothing on success.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

NS = 1_000_000_000


def hash_update(h, data) -> None:
    """Feed an array's bytes to a hashlib object without a Python-level byte loop."""
    a = np.ascontiguousarray(data).reshape(-1)
    h.update(memoryview(a))


def hash_update_packed(h, data) -> None:
    """Feed the packed bits of a 0/1 array (8x fewer bytes; still every bit)."""
    hash_update(h, np.packbits(np.ascontiguousarray(data, dtype=np.uint8)))


def load_problem(path):
    from benchlib.formats import read_problem

    return read_problem(path)


def load_shots(path):
    from benchlib.formats import read_shots

    return read_shots(path)


def core_variants(decoders) -> list[str]:
    """``Decoder.core_variant`` of each handle ("default", "ada64" or "cpu"; "unknown" if absent)."""
    return [str(getattr(d, "core_variant", None) or "unknown") for d in decoders]


def make_decoder(eldpc, problem_file, config_string):
    H, O, priors, _dem = load_problem(problem_file)
    return eldpc.Decoder(H, O, priors, config=config_string)


def pin(cpu):
    if cpu is None:
        return None
    try:
        os.sched_setaffinity(0, {int(cpu)})
        return int(cpu)
    except (AttributeError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def busy_wait_until(t_end_ns: int):
    while time.perf_counter_ns() < t_end_ns:
        pass


def keep_warm_gap(decoder, gap_us: int, final_us: int = 25, period_us: int = 500):
    """Busy-wait a gap, calling decoder.keep_warm() every period_us and once shortly before the call."""
    gap_ns = gap_us * 1000
    gs = time.perf_counter_ns()
    ge = gs + gap_ns
    fin = ge - final_us * 1000
    next_call = gs
    final_done = False
    calls = 0
    errors = 0
    t = time.perf_counter_ns()
    while t < ge:
        if not final_done and t >= fin:
            try:
                decoder.keep_warm()
            except Exception:
                errors += 1
            calls += 1
            final_done = True
        elif t >= next_call:
            try:
                decoder.keep_warm()
            except Exception:
                errors += 1
            calls += 1
            next_call = t + period_us * 1000
        t = time.perf_counter_ns()
    if not final_done:
        try:
            decoder.keep_warm()
        except Exception:
            errors += 1
        calls += 1
    return calls, errors


def run_py_latency_sequential(eldpc, job):
    """The CPU-core session design: one live decoder at a time, in call order.

    Each decoder is created, warmed up, timed for all its calls and closed before the next one is
    created, so the busy-waiting worker threads of two CPU-core handles never share the CPUs.  Decoder
    v decodes shots 0, 1, 2, ... as in the interleaved design, where call pair i decodes shot i.
    """
    dets, _obs = load_shots(job["shots_file"])
    shots = dets.shape[0]
    variants = job["variants"]
    order = job["order"]
    n = int(job["shots_per_variant"])
    warmup = int(job.get("warmup", 0))
    gap_us = int(job.get("gap_us", 0))
    keep_warm = bool(job.get("keep_warm", False))
    per = [{"dur_ns": [], "status": [], "checksum": hashlib.sha256()} for _ in range(2)]
    keep = [{"calls": 0, "errors": 0} for _ in range(2)]
    variants_run = ["unknown", "unknown"]
    syn = [dets[i % shots] for i in range(n)]
    for v in order:
        dec = make_decoder(eldpc, job["problem_file"], variants[v]["config_string"])
        try:
            variants_run[v] = core_variants([dec])[0]
            t0 = time.perf_counter_ns()
            for w in range(warmup):
                if w >= 10 and time.perf_counter_ns() - t0 > 1_500_000_000:
                    break
                dec.decode(dets[w % shots])
            for s in syn:
                if gap_us and keep_warm:
                    calls, errs = keep_warm_gap(dec, gap_us)
                    keep[v]["calls"] += calls
                    keep[v]["errors"] += errs
                elif gap_us:
                    t_end = time.perf_counter_ns() + gap_us * 1000
                    busy_wait_until(t_end)
                t0 = time.perf_counter_ns()
                corr = dec.decode(s)
                t1 = time.perf_counter_ns()
                st = int(dec.last_status)
                per[v]["dur_ns"].append(t1 - t0)
                per[v]["status"].append(st)
                hash_update(per[v]["checksum"], corr)
                hash_update(per[v]["checksum"], np.frombuffer(st.to_bytes(4, "little"), dtype=np.uint8))
        finally:
            dec.close()
    return {"variants": [
        {"dur_ns": per[v]["dur_ns"], "status": per[v]["status"],
         "checksum": per[v]["checksum"].hexdigest(), "keep_warm": keep[v]} for v in range(2)],
        "config_strings": [variants[i]["config_string"] for i in range(2)],
        "core_variants": variants_run, "handles": "sequential"}


def run_py_latency(eldpc, job):
    if job.get("sequential"):
        return run_py_latency_sequential(eldpc, job)
    dets, _obs = load_shots(job["shots_file"])
    shots = dets.shape[0]
    variants = job["variants"]
    order = job["order"]
    create = job.get("create_order", order)
    decoders = [None, None]
    for v in create:  # create order is separate from the call order, as in the C driver
        decoders[v] = make_decoder(eldpc, job["problem_file"], variants[v]["config_string"])
    n = int(job["shots_per_variant"])
    warmup = int(job.get("warmup", 0))
    gap_us = int(job.get("gap_us", 0))
    keep_warm = bool(job.get("keep_warm", False))
    per = [{"dur_ns": [], "status": [], "checksum": hashlib.sha256()} for _ in range(2)]
    keep = [{"calls": 0, "errors": 0} for _ in range(2)]

    # Warm up each decoder, in create order.
    for v in order:
        t0 = time.perf_counter_ns()
        for w in range(warmup):
            if w >= 10 and time.perf_counter_ns() - t0 > 1_500_000_000:
                break
            decoders[v].decode(dets[w % shots])
    syn = [dets[i % shots] for i in range(n)]
    for i in range(2 * n):
        v = order[i & 1]
        s = syn[i >> 1]
        if gap_us and keep_warm:
            calls, errs = keep_warm_gap(decoders[v], gap_us)
            keep[v]["calls"] += calls
            keep[v]["errors"] += errs
        elif gap_us:
            t_end = time.perf_counter_ns() + gap_us * 1000
            busy_wait_until(t_end)
        t0 = time.perf_counter_ns()
        corr = decoders[v].decode(s)
        t1 = time.perf_counter_ns()
        st = int(decoders[v].last_status)
        per[v]["dur_ns"].append(t1 - t0)
        per[v]["status"].append(st)
        hash_update(per[v]["checksum"], corr)
        hash_update(per[v]["checksum"], np.frombuffer(st.to_bytes(4, "little"), dtype=np.uint8))
    variants_run = core_variants(decoders)
    for v in range(2):
        decoders[v].close()
    return {"variants": [
        {"dur_ns": per[v]["dur_ns"], "status": per[v]["status"],
         "checksum": per[v]["checksum"].hexdigest(), "keep_warm": keep[v]} for v in range(2)],
        "config_strings": [variants[i]["config_string"] for i in range(2)],
        "core_variants": variants_run}


def run_throughput_sequential(eldpc, job):
    """decode_batch throughput with one live decoder at a time, in call order (the CPU core; see
    run_py_latency_sequential)."""
    dets, _obs = load_shots(job["shots_file"])
    shots = dets.shape[0]
    variants = job["variants"]
    order = job["order"]
    batch = int(job["batch_shots"])
    reps = int(job["repetitions"])
    warmup = int(job.get("warmup", 0))
    batches = []
    checks = [hashlib.sha256(), hashlib.sha256()]
    variants_run = ["unknown", "unknown"]
    for v in order:
        dec = make_decoder(eldpc, job["problem_file"], variants[v]["config_string"])
        try:
            variants_run[v] = core_variants([dec])[0]
            t0 = time.perf_counter_ns()
            for w in range(max(1, warmup // max(batch, 1))):
                dec.decode_batch(dets[(w * batch) % shots:(w * batch) % shots + batch])
                if time.perf_counter_ns() - t0 > 1_000_000_000:
                    break
            for r in range(reps):
                start = (r * batch) % shots
                idx = (start + np.arange(batch)) % shots
                synd = dets[idx]
                t0 = time.perf_counter_ns()
                corr, obs, st = dec.decode_batch(synd)
                t1 = time.perf_counter_ns()
                batches.append({"variant": v, "shots": int(batch), "dur_ns": int(t1 - t0)})
                hash_update_packed(checks[v], corr)
                hash_update(checks[v], obs)
                hash_update(checks[v], st)
        finally:
            dec.close()
    return {"batches": batches,
            "checksum": [checks[0].hexdigest(), checks[1].hexdigest()],
            "config_strings": [variants[i]["config_string"] for i in range(2)],
            "core_variants": variants_run, "handles": "sequential"}


def run_throughput(eldpc, job):
    if job.get("sequential"):
        return run_throughput_sequential(eldpc, job)
    dets, _obs = load_shots(job["shots_file"])
    shots = dets.shape[0]
    variants = job["variants"]
    order = job["order"]
    batch = int(job["batch_shots"])
    reps = int(job["repetitions"])
    decoders = [None, None]
    for v in job.get("create_order", order):
        decoders[v] = make_decoder(eldpc, job["problem_file"], variants[v]["config_string"])
    warmup = int(job.get("warmup", 0))
    for v in job.get("create_order", order):
        t0 = time.perf_counter_ns()
        for w in range(max(1, warmup // max(batch, 1))):
            decoders[v].decode_batch(dets[(w * batch) % shots:(w * batch) % shots + batch])
            if time.perf_counter_ns() - t0 > 1_000_000_000:
                break
    batches = []
    checks = [hashlib.sha256(), hashlib.sha256()]
    for r in range(reps):
        for k in range(2):
            v = order[k]
            start = (r * batch) % shots
            idx = (start + np.arange(batch)) % shots
            synd = dets[idx]
            t0 = time.perf_counter_ns()
            corr, obs, st = decoders[v].decode_batch(synd)
            t1 = time.perf_counter_ns()
            batches.append({"variant": v, "shots": int(batch), "dur_ns": int(t1 - t0)})
            hash_update_packed(checks[v], corr)
            hash_update(checks[v], obs)
            hash_update(checks[v], st)
    variants_run = core_variants(decoders)
    for v in range(2):
        decoders[v].close()
    return {"batches": batches,
            "checksum": [checks[0].hexdigest(), checks[1].hexdigest()],
            "config_strings": [variants[i]["config_string"] for i in range(2)],
            "core_variants": variants_run}


def run_ler(eldpc, job):
    dets, obs_true = load_shots(job["shots_file"])
    shots = dets.shape[0]
    variant = job["variants"][0]
    dec = make_decoder(eldpc, job["problem_file"], variant["config_string"])
    batch = int(job.get("batch_shots", 2048))
    errors = 0
    done = 0
    checksum = hashlib.sha256()
    t_dec0 = time.perf_counter_ns()
    while done < shots:
        chunk = dets[done:done + batch]
        corr, obs, st = dec.decode_batch(chunk)
        if obs.shape[1]:
            wrong = (obs != obs_true[done:done + len(chunk)]).any(axis=1)
            errors += int(wrong.sum())
        hash_update(checksum, obs)
        hash_update(checksum, st)
        done += len(chunk)
    t_dec1 = time.perf_counter_ns()
    variants_run = core_variants([dec])
    dec.close()
    return {"shots": int(shots), "errors": int(errors), "checksum": checksum.hexdigest(),
            "decode_wall_s": (t_dec1 - t_dec0) / NS, "config_strings": [variant["config_string"]],
            "core_variants": variants_run}


def main(argv):
    if len(argv) != 2:
        print("usage: worker.py JOB.json", file=sys.stderr)
        return 2
    with open(argv[1], "r", encoding="utf-8") as f:
        job = json.load(f)
    out = {"kind": job["kind"], "pid": os.getpid()}
    try:
        import eldpc
    except Exception as e:
        out["error"] = f"cannot import eldpc: {type(e).__name__}: {e}"
        with open(job["out"], "w", encoding="utf-8") as f:
            json.dump(out, f)
        return 1
    out["pinned"] = pin(job.get("cpu"))
    out["isa"] = getattr(eldpc, "isa", lambda: None)()
    out["version"] = getattr(eldpc, "__version__", None)
    t0 = time.monotonic()
    try:
        if job["kind"] == "py_latency":
            out.update(run_py_latency(eldpc, job))
        elif job["kind"] == "throughput":
            out.update(run_throughput(eldpc, job))
        elif job["kind"] == "ler":
            out.update(run_ler(eldpc, job))
        else:
            raise ValueError(f"unknown job kind {job['kind']!r}")
        out["wall_s"] = round(time.monotonic() - t0, 3)
    except Exception as e:
        import traceback

        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        with open(job["out"], "w", encoding="utf-8") as f:
            json.dump(out, f)
        return 1
    with open(job["out"], "w", encoding="utf-8") as f:
        json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
