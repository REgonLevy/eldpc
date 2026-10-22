# SPDX-License-Identifier: Apache-2.0
"""Compare two benchmark results files, refusing to diff measurements of different things.

Two results are comparable only when they measured the same problem (same DEM), the same decoder
configuration, the same batch size and the same measurement kinds.  The comparison prints the
environment of both sides, the pooled statistics of each measurement side by side, the difference and
each side's A/A band, so a cross-machine comparison cannot silently mix a 1024-shot batch with a
4096-shot one or two different codes.

    python bench.py compare --a DIR_A --b DIR_B

Exit 0 when the two are comparable and the comparison was printed; 2 when a check fails (the reason is
printed) or a file cannot be read.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "eldpc.bench.v1"
LATENCY_KINDS = ("c_latency", "py_latency")


class CompareError(RuntimeError):
    pass


def load_results(directory) -> dict:
    path = Path(directory) / "results.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise CompareError(f"cannot read {path}: {e}") from e


def _problem_identity(r: dict) -> dict:
    p = r.get("problem") or {}
    return {
        "name": p.get("name"), "m": p.get("m"), "n": p.get("n"), "k": p.get("k"),
        "dem_sha256": p.get("dem_sha256"),
        "problem_file_sha256": (p.get("problem_file") or {}).get("sha256"),
    }


def _config_of(r: dict, kind: str, side: str = "A"):
    """The config string a results file records for one measurement's side A."""
    m = r.get(kind) or {}
    cfg = (m.get("configs") or {}).get(side) or {}
    if cfg.get("config_string"):
        return cfg["config_string"]
    v = (m.get("variants") or {}).get(side) or {}
    return v.get("config_string")


def check_comparable(a: dict, b: dict) -> list[dict]:
    """Return the list of checks; raises CompareError on the first hard mismatch."""
    checks = []

    def ok(name, detail, passed):
        checks.append({"check": name, "ok": bool(passed), "detail": detail})
        if not passed:
            raise CompareError(f"{name}: {detail}")

    ok("results schema", f"{a.get('schema')} vs {b.get('schema')}",
       a.get("schema") == SCHEMA and b.get("schema") == SCHEMA)

    pa, pb = _problem_identity(a), _problem_identity(b)
    ok("problem dimensions", f"m/n/k {pa['m']}/{pa['n']}/{pa['k']} vs {pb['m']}/{pb['n']}/{pb['k']}",
       (pa["m"], pa["n"], pa["k"]) == (pb["m"], pb["n"], pb["k"]))
    if pa["dem_sha256"] and pb["dem_sha256"]:
        ok("problem DEM sha256", f"{pa['dem_sha256']} vs {pb['dem_sha256']}",
           pa["dem_sha256"] == pb["dem_sha256"])
    elif pa["problem_file_sha256"] and pb["problem_file_sha256"]:
        ok("problem file sha256", f"{pa['problem_file_sha256']} vs {pb['problem_file_sha256']}",
           pa["problem_file_sha256"] == pb["problem_file_sha256"])
    else:
        ok("problem name", f"{pa['name']} vs {pb['name']}", pa["name"] == pb["name"])

    for kind in LATENCY_KINDS + ("throughput", "ler"):
        if kind in a and kind in b:
            ca, cb = _config_of(a, kind), _config_of(b, kind)
            if ca is not None or cb is not None:
                ok(f"{kind} config (side A)", f"{ca} vs {cb}",
                   ca is not None and cb is not None and ca == cb)

    if "throughput" in a and "throughput" in b:
        ba = a["throughput"].get("batch_shots")
        bb = b["throughput"].get("batch_shots")
        ok("throughput batch size", f"{ba} vs {bb}", ba is not None and ba == bb)
        # The C-driver block uses the same batch size; check it too when both have it.
        if "batch_shots" in (a["throughput"].get("c_driver") or {}) \
                and "batch_shots" in (b["throughput"].get("c_driver") or {}):
            ca = a["throughput"]["c_driver"].get("batch_shots")
            cb = b["throughput"]["c_driver"].get("batch_shots")
            ok("C-driver throughput batch size", f"{ca} vs {cb}", ca is not None and ca == cb)

    ok("common measurements", "at least one of c_latency/py_latency/throughput/ler",
       any(k in a and k in b for k in LATENCY_KINDS + ("throughput", "ler")))
    return checks


def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _env_line(env: dict) -> str:
    gpu = env.get("gpu") or {}
    cpu = env.get("cpu") or {}
    tk = env.get("toolkit") or {}
    return (f"{gpu.get('name')} ({gpu.get('sm_count')} SMs, cc {gpu.get('compute_capability')}) | "
            f"{cpu.get('model')} ({cpu.get('logical_cores')} logical, pinned {cpu.get('pinned_cpu')}) | "
            f"driver {env.get('driver')} | runtime {env.get('cuda_runtime')} | toolkit {tk.get('version')} | "
            f"isa {env.get('isa')} | version {env.get('version')}")


def _latency_block(kind: str, a: dict, b: dict) -> list[str]:
    lines = []
    modes_a = (a.get(kind) or {}).get("modes") or {}
    modes_b = (b.get(kind) or {}).get("modes") or {}
    for mode in sorted(set(modes_a) & set(modes_b)):
        ma, mb = modes_a[mode], modes_b[mode]
        va = (ma.get("variants") or {}).get("A") or {}
        vb = (mb.get("variants") or {}).get("A") or {}
        pa, pb = va.get("pooled") or {}, vb.get("pooled") or {}
        if not pa or not pb:
            continue
        delta = {k: (round(pa[k] - pb[k], 3) if pa.get(k) is not None and pb.get(k) is not None else None)
                 for k in ("p50_us", "p99_us", "p99_9_us", "mean_us")}
        ca = (ma.get("a_a_control") or {}).get("band") or {}
        cb = (mb.get("a_a_control") or {}).get("band") or {}
        lines.append(f"\n### {kind} / {mode}\n")
        lines.append(f"| | p50 us | p99 us | p99.9 us | mean us | shots |")
        lines.append(f"|---|---|---|---|---|---|")
        lines.append(f"| A {va.get('label')} | {_fmt(pa.get('p50_us'))} | {_fmt(pa.get('p99_us'))} | "
                     f"{_fmt(pa.get('p99_9_us'))} | {_fmt(pa.get('mean_us'))} | {pa.get('shots')} |")
        lines.append(f"| B {vb.get('label')} | {_fmt(pb.get('p50_us'))} | {_fmt(pb.get('p99_us'))} | "
                     f"{_fmt(pb.get('p99_9_us'))} | {_fmt(pb.get('mean_us'))} | {pb.get('shots')} |")
        lines.append(f"| A - B | {_fmt(delta['p50_us'])} | {_fmt(delta['p99_us'])} | "
                     f"{_fmt(delta['p99_9_us'])} | {_fmt(delta['mean_us'])} | |")
        lines.append("")
        lines.append(f"A/A band A: p50 {ca.get('p50_us')} us, p99 {ca.get('p99_us')} us; "
                     f"A/A band B: p50 {cb.get('p50_us')} us, p99 {cb.get('p99_us')} us.")
    return lines


def compare_text(a: dict, b: dict, checks: list[dict] | None = None) -> str:
    lines = ["# eldpc benchmark comparison\n",
             f"A: {a.get('command')}  ({a.get('generated_utc')})",
             f"B: {b.get('command')}  ({b.get('generated_utc')})\n"]
    if checks:
        lines.append("## Comparability checks\n")
        lines += [f"- {c['check']}: ok ({c['detail']})" for c in checks]
        lines.append("")
    lines += ["## Environment\n",
             f"- A: {_env_line(a.get('environment') or {})}",
             f"- B: {_env_line(b.get('environment') or {})}"]
    for kind, title in (("c_latency", "C API single-shot latency"),
                        ("py_latency", "Python per-shot latency")):
        if kind in a and kind in b:
            block = _latency_block(kind, a, b)
            if block:
                lines.append(f"\n## {title}")
                lines += block
    if "throughput" in a and "throughput" in b:
        lines.append("\n## Batch throughput\n")
        for label, key, title in (("Python decode_batch", None, "Python"), ("C driver", "c_driver", "C")):
            ta = a["throughput"] if key is None else (a["throughput"].get(key) or {})
            tb = b["throughput"] if key is None else (b["throughput"].get(key) or {})
            va = (ta.get("variants") or {}).get("A") or {}
            vb = (tb.get("variants") or {}).get("A") or {}
            if va and vb:
                ra, rb = va.get("shots_per_s"), vb.get("shots_per_s")
                ratio = round(ra / rb, 4) if ra and rb else None
                lines.append(f"- {label}: A {_fmt(ra, 1)} shots/s vs B {_fmt(rb, 1)} shots/s "
                             f"(A/B {ratio}); batch {ta.get('batch_shots')} shots")
    if "ler" in a and "ler" in b:
        lines.append("\n## Logical error rate\n")
        va = (a["ler"].get("variants") or {}).get("A") or {}
        vb = (b["ler"].get("variants") or {}).get("A") or {}
        if va and vb:
            lines.append(f"- A: {va.get('errors')}/{va.get('shots')} = {va.get('ler')} (95% CI {va.get('ci95')})")
            lines.append(f"- B: {vb.get('errors')}/{vb.get('shots')} = {vb.get('ler')} (95% CI {vb.get('ci95')})")
    wa = a.get("warnings") or []
    wb = b.get("warnings") or []
    lines.append("\n## Warnings\n")
    lines.append(f"- A: {len(wa)}" + ("" if not wa else " -> " + "; ".join(wa)))
    lines.append(f"- B: {len(wb)}" + ("" if not wb else " -> " + "; ".join(wb)))
    return "\n".join(lines) + "\n"
