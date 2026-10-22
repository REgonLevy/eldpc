# SPDX-License-Identifier: Apache-2.0
"""Write DIR/results.json and DIR/results.md.

results.json is the machine-readable record; results.md is the same content as a report a person can
read.  Every number in the Markdown is read from the JSON object, so the two cannot disagree.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone


def write_json(path, results: dict) -> None:
    results.setdefault("generated_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")


def _fmt(v, digits=3, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}{suffix}"
    return f"{v}{suffix}"


def _bytes(n):
    if n is None:
        return "n/a"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    return f"{n} B"


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def core_variants_text(record) -> str:
    """``c_latency A=ada64, B=ada64; ler A=ada64`` from the environment's ``core_variants``."""
    if not record:
        return "n/a"
    parts = []
    for measurement, sides in record.items():
        names = []
        for side, variant in sides.items():
            value = "/".join(variant) if isinstance(variant, list) else str(variant)
            names.append(f"{side}={value}")
        parts.append(f"{measurement} {', '.join(names)}")
    return "; ".join(parts)


def _toolkit_text(tk: dict) -> str:
    """``13.3 (V13.3.73, from the library's build_info)``; ``n/a`` when no toolkit is known."""
    if not tk.get("version"):
        return "n/a"
    where = {"build_info": "from the library's build_info",
             "nvcc": "nvcc on PATH or CUDACXX"}.get(tk.get("source"))
    detail = ", ".join(x for x in (tk.get("nvcc"), where) if x)
    return tk["version"] + (f" ({detail})" if detail else "")


def _env_section(env: dict) -> str:
    gpu = env.get("gpu", {})
    cpu = env.get("cpu", {})
    freq = cpu.get("frequency", {})
    clocks = env.get("clocks", {})
    b, a = clocks.get("before") or {}, clocks.get("after") or {}
    tk = env.get("toolkit") or {}
    rows = [
        ("GPU", gpu.get("name") or "n/a"),
        ("GPU SMs / L2 / memory", f"{_fmt(gpu.get('sm_count'))} / {_bytes(gpu.get('l2_bytes'))} / "
                                  f"{_fmt(gpu.get('memory_mib'), 0, ' MiB')}"),
        ("GPU compute capability", gpu.get("compute_capability") or "n/a"),
        ("CPU", f"{cpu.get('model') or 'n/a'} ({_fmt(cpu.get('logical_cores'))} logical cores)"),
        ("Pinned CPU", _fmt(cpu.get("pinned_cpu"))),
        ("CPU governor / frequency", f"{freq.get('governor', 'n/a')} / "
                                     f"{_fmt(freq.get('cur_freq_khz'), 0)} kHz"),
        ("CPU energy preference / SMT sibling",
         f"{freq.get('energy_performance_preference', 'n/a')} / "
         f"{freq.get('thread_siblings_list', 'n/a')}"),
        ("CPU scaling driver / boost", f"{freq.get('scaling_driver', 'n/a')} / "
                                       f"{freq.get('boost', 'n/a')}"),
        ("Driver", env.get("driver") or "n/a"),
        ("CUDA runtime", env.get("cuda_runtime") or "n/a"),
        ("CUDA toolkit", _toolkit_text(tk)),
        ("GPU clocks before / after", f"graphics {_fmt(b.get('graphics_mhz'), 0)} MHz, memory "
                                      f"{_fmt(b.get('memory_mhz'), 0)} MHz / graphics "
                                      f"{_fmt(a.get('graphics_mhz'), 0)} MHz, memory "
                                      f"{_fmt(a.get('memory_mhz'), 0)} MHz"),
        ("Host load average before / after",
         f"{env.get('loadavg')} / {env.get('loadavg_after')}"),
        ("Virtual machine", "yes" if cpu.get("hypervisor") else "no"),
        ("Package version / ISA", f"{env.get('version')} / {env.get('isa')}"),
        ("Core variant of the timed handles", core_variants_text(env.get("core_variants"))),
    ]
    return _table(["", ""], rows) + "\nBuild info: `" + str(env.get("build_info") or "n/a") + "`\n"


def _latency_section(title: str, data: dict) -> str:
    if not data or data.get("error"):
        return f"## {title}\n\nNot measured: {data.get('error') if data else 'not requested'}.\n\n"
    lines = [f"## {title}\n",
             f"What is timed: {data.get('what_is_timed', 'n/a')}.\n"]
    if data.get("what_is_not_timed"):
        lines.append(f"Not timed: {data['what_is_not_timed']}.\n")
    for mode_name, mode in data.get("modes", {}).items():
        label = {"back_to_back": "back to back", "cadence": f"fixed cadence, {mode.get('gap_us')} us gap",
                 "cadence_keep_warm": f"fixed cadence, {mode.get('gap_us')} us gap with keep-warm",
                 "cadence_keep_warm_pair": f"fixed cadence, {mode.get('gap_us')} us gap: keep-warm (A) "
                                           "against no keep-warm (B), same sessions"}
        lines.append(f"\n### {label.get(mode_name, mode_name)}\n")
        rows = []
        for side in ("A", "B"):
            v = mode.get("variants", {}).get(side)
            if not v:
                continue
            p = v.get("pooled", {})
            name = v.get("label")
            if v.get("treatment") == "keep_warm":
                name += " (keep-warm)"
            elif v.get("treatment") == "idle":
                name += " (idle)"
            rows.append([side, name, _fmt(p.get("shots")), _fmt(p.get("p50_us")),
                         _fmt(p.get("p99_us")), _fmt(p.get("p99_9_us")), _fmt(p.get("mean_us"))])
        lines.append(_table(["side", "config", "shots", "p50 us", "p99 us", "p99.9 us", "mean us"], rows))
        for side in ("A", "B"):
            v = mode.get("variants", {}).get(side)
            if not v:
                continue
            p = v.get("pooled") or {}
            if p.get("p99_unreliable"):
                lines.append(f"\n{side}: slow_fraction {_fmt(p.get('slow_fraction'), 4)} is within "
                             f"2x of 1 %: the p99 is unreliable (it sits on the cliff between the "
                             f"converged calls and the slow class); the median of the slow calls is "
                             f"{_fmt(p.get('slow_p50_us'))} us.\n")
        ctl = mode.get("a_a_control")
        if ctl:
            d = ctl.get("delta", {})
            band = ctl.get("band", {})
            lines.append(
                f"\nA/A control ({len(ctl.get('per_session_delta') or [])} sessions, per-shot interleaved): "
                f"delta p50 {_fmt(d.get('p50_us'))} us, p99 {_fmt(d.get('p99_us'))} us, "
                f"p99.9 {_fmt(d.get('p99_9_us'))} us, mean {_fmt(d.get('mean_us'))} us; "
                f"noise band p50 {band.get('p50_us')} us, p99 {band.get('p99_us')} us, "
                f"p99.9 {band.get('p99_9_us')} us, mean {band.get('mean_us')} us.\n")
        for ctl_name, ctl in (mode.get("a_a_controls") or {}).items():
            d = ctl.get("delta", {})
            band = ctl.get("band", {})
            lines.append(
                f"\nA/A control, {ctl_name.replace('_', ' ')} "
                f"({len(ctl.get('per_session_delta') or [])} sessions): delta p50 {_fmt(d.get('p50_us'))} us, "
                f"p99 {_fmt(d.get('p99_us'))} us, p99.9 {_fmt(d.get('p99_9_us'))} us, mean "
                f"{_fmt(d.get('mean_us'))} us; band p50 {band.get('p50_us')} us, "
                f"p99 {band.get('p99_us')} us, p99.9 {band.get('p99_9_us')} us, mean "
                f"{band.get('mean_us')} us.\n")
        comp = mode.get("comparison")
        if comp:
            d = comp.get("delta", {})
            band = comp.get("aa_band") or {}
            wb = comp.get("within_band", {})
            lines.append(
                f"\nA vs B: delta p50 {_fmt(d.get('p50_us'))} us, p99 {_fmt(d.get('p99_us'))} us, "
                f"p99.9 {_fmt(d.get('p99_9_us'))} us, mean {_fmt(d.get('mean_us'))} us. "
                f"A/A band (same estimator, A vs A, {comp.get('aa_band_sessions', 'n/a')} control "
                f"sessions): p50 {band.get('p50_us')}, p99 {band.get('p99_us')}, "
                f"p99.9 {band.get('p99_9_us')}, mean {band.get('mean_us')} us; inside the band: "
                + ", ".join(f"{k.replace('_us', '')}={'yes' if v else 'no'}" for k, v in wb.items()) + ".\n")
        kw = mode.get("keep_warm_stats")
        if kw:
            lines.append(
                f"\nKeep-warm: {kw.get('calls')} calls ({kw.get('errors')} errors), periodic mean "
                f"{_fmt(kw.get('periodic_mean_us'))} us/max {_fmt(kw.get('periodic_max_us'))} us, "
                f"final-before-call mean {_fmt(kw.get('final_mean_us'))} us/max "
                f"{_fmt(kw.get('final_max_us'))} us.\n")
    return "\n".join(lines) + "\n"


def _throughput_section(data: dict) -> str:
    if not data or data.get("error"):
        return f"## Batch throughput\n\nNot measured: {data.get('error') if data else 'not requested'}.\n\n"
    lines = ["## Batch throughput\n",
             f"What is measured: {data.get('what_is_measured', 'n/a')} (decode_batch, "
             f"{data.get('batch_shots')} shots per batch).\n"]
    rows = []
    for side in ("A", "B"):
        v = data.get("variants", {}).get(side)
        if v:
            rows.append([side, v.get("label"), _fmt(v.get("shots")), _fmt(v.get("seconds"), 3),
                         _fmt(v.get("shots_per_s"), 1)])
    lines.append(_table(["side", "config", "shots", "seconds", "shots/s"], rows))
    ctl = data.get("a_a_control", {})
    if ctl.get("ratio_b_over_a") is not None:
        lines.append(f"\nA/A control ({_fmt(ctl.get('sessions'))} sessions) ratio (B/A): "
                     f"{_fmt(ctl.get('ratio_b_over_a'), 4)}; per-session shots/s A "
                     f"{ctl.get('per_session_shots_per_s', {}).get('A')}, B "
                     f"{ctl.get('per_session_shots_per_s', {}).get('B')}.\n")
    comp = data.get("comparison")
    if comp:
        inside = comp.get("within_aa_band")
        lines.append(f"\nA vs B throughput ratio (B/A): {_fmt(comp.get('ratio_b_over_a'), 4)}. A/A band "
                     f"(range of the B/A ratio over the {_fmt(comp.get('aa_band_sessions'))} A/A control "
                     f"sessions): {comp.get('aa_band_ratio')}; inside the band: "
                     f"{'n/a' if inside is None else ('yes' if inside else 'no')}. Range of the "
                     f"per-session A/B ratio over {_fmt(comp.get('ab_sessions'))} sessions: "
                     f"{comp.get('ab_session_ratio_range')}.\n")
    return "\n".join(lines) + "\n"


def _ler_section(data: dict) -> str:
    if not data or data.get("error"):
        return f"## Logical error rate\n\nNot measured: {data.get('error') if data else 'not requested'}.\n\n"
    lines = ["## Logical error rate\n", f"What is measured: {data.get('what_is_measured', 'n/a')}.\n"]
    rows = []
    for side, v in data.get("variants", {}).items():
        rows.append([side, v.get("label"), _fmt(v.get("shots")), _fmt(v.get("errors")),
                     _fmt(v.get("ler"), 6), f"[{_fmt(v.get('ci95', [None, None])[0], 6)}, "
                                             f"{_fmt(v.get('ci95', [None, None])[1], 6)}]"])
    lines.append(_table(["side", "config", "shots", "errors", "LER", "95% Wilson CI"], rows))
    comp = data.get("comparison")
    if comp:
        lines.append(f"\nA - B LER: {_fmt(comp.get('delta_ler'), 6)}; the 95% intervals are disjoint: "
                     f"{'yes' if comp.get('ci95_disjoint') else 'no'}.\n")
    return "\n".join(lines) + "\n"


def build_markdown(results: dict) -> str:
    lines = ["# eldpc benchmark results\n",
             f"Generated {results.get('generated_utc', 'n/a')} by `{results.get('command', 'bench.py')}`"
             f" ({'quick' if results.get('quick') else 'full'} mode).\n",
             "The four measurement kinds are independent: (a) the C API single-shot call, (b) the Python "
             "per-shot call, (c) batch throughput and (d) a logical error rate check. They are not "
             "convertible into each other; see docs/BENCH.md for what each one does and does not mean.\n"]
    prob = results.get("problem") or {}
    lines += ["## Problem\n",
              _table(["", ""], [
                  ("Name", prob.get("name")),
                  ("Source", prob.get("source")),
                  ("Detectors / mechanisms / observables", f"{prob.get('m')} / {prob.get('n')} / "
                                                            f"{prob.get('k')}"),
                  ("Recipe", json.dumps(prob.get("recipe"), sort_keys=True) if prob.get("recipe") else "n/a"),
                  ("Latency shots / LER shots",
                   f"{(prob.get('shots') or {}).get('latency', {}).get('shots')} / "
                   f"{(prob.get('shots') or {}).get('ler', {}).get('shots')}"),
                  ("DEM sha256", prob.get("dem_sha256") or "n/a"),
              ]) + "\n",
              "## Environment\n", _env_section(results.get("environment") or {}) + "\n"]
    warns = results.get("warnings") or []
    lines.append("## Warnings\n")
    lines.append("None.\n" if not warns else "".join(f"- {w}\n" for w in warns))
    lines.append("\nThe warnings list conditions that make a timing unrepresentative (unpinned run, a busy "
                 "GPU or host, a CPU governor that moves clocks, a virtual machine). A results file "
                 "without warnings was produced under the conditions this benchmark checks for.\n")
    for key, title, fn in (("c_latency", "C API single-shot latency", _latency_section),
                           ("py_latency", "Python per-shot latency", _latency_section)):
        if key in results:
            lines.append(fn(title, results[key]))
    if "throughput" in results:
        lines.append(_throughput_section(results["throughput"]))
    if "ler" in results:
        lines.append(_ler_section(results["ler"]))
    failures = results.get("failures") or []
    if failures:
        lines.append("## Measurement failures\n")
        lines += [f"- **{f.get('measurement')}**: {f.get('error')}\n" for f in failures]
    return "\n".join(lines).rstrip() + "\n"
