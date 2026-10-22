# SPDX-License-Identifier: Apache-2.0
"""The verification run: generate corpora, decode with the reference and eldpc, compare, report.

One row per (problem, config). Every shot of the corpus is decoded by the reference library (CPU)
and by eldpc, once shot by shot and once through ``decode_batch``. The pass criterion is bit
identity of corrections and observables between eldpc and the reference in both paths. When the
caller supplies a separately built library of the same decoder sources (``--reference-lib`` or
``ELDPC_REFERENCE_LIB``), the status words of eldpc are compared with that library's as well. The upstream wheel column compares the reference with the installed
``ldpc`` 2.4.1 wheel on the same shots; it is informational and never decides pass or fail.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import SUMMARY_SCHEMA
from . import upstream as upstream_mod
from .abi import ST_BP_CONVERGED, AbiDecoder, AbiLibrary, AbiError
from .config import default_cache_root, load_config
from .corpus import CorpusError, describe_generator, ensure_corpus, generator_versions
from .reference import build_reference


class VerifyError(RuntimeError):
    """The verification run cannot start or continue (environment, configuration, I/O)."""


def config_string(params: dict) -> str:
    """The canonical ABI config string of a preset dict (keys sorted, as the reference construction
    sorts them)."""
    def fmt(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise VerifyError("non-finite config value")
            return repr(value)
        return str(value)

    return ";".join(f"{key}={fmt(params[key])}" for key in sorted(params))


# --------------------------------------------------------------------------
# environment fingerprint
# --------------------------------------------------------------------------
def _run_capture(cmd, timeout=15):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_info():
    out = _run_capture(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                        "--format=csv,noheader"])
    if not out:
        return None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return {"raw": first}
    return {"name": parts[0], "driver_version": parts[1], "compute_capability": parts[2]}


def _cpu_info():
    model = None
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"model": model or platform.processor() or "unknown",
            "logical_cpus": os.cpu_count()}


def _nvcc_info():
    """Version of the CUDA compiler: $CUDACXX when set (the toolkit the package is built with),
    else nvcc from PATH."""
    exe = os.environ.get("CUDACXX") or "nvcc"
    out = _run_capture([exe, "--version"])
    return out.splitlines()[-1].strip() if out else None


def _package_versions():
    versions = {}
    for name in ("numpy", "scipy", "stim", "qldpc", "ldpc"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - a missing optional package is reported as None
            versions[name] = None
    return versions


def _eldpc_info(module) -> dict:
    info = {"version": getattr(module, "__version__", None)}
    for name, key in (("build_info", "build_info"), ("isa", "isa"), ("devices", "devices")):
        try:
            value = getattr(module, name)()
        except Exception as exc:  # noqa: BLE001 - best effort fingerprint
            value = f"unavailable: {exc}"
        info[key] = value
    return info


def _cuda_field(build_info) -> str | None:
    """The ``cuda=`` field of a build-info string ("NVIDIA 13.3.73"), or None."""
    for part in str(build_info or "").split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key == "cuda":
            return value.strip()
    return None


def toolkit_line(reference_build_info, package_build_info) -> str:
    """The log line that names the toolkits of the separately built library and of the package.

    docs/VERIFY.md asks for the separately built library to be built with the package's toolkit
    (the same CUDACXX); this line shows which ones were used, and says so when they differ.
    """
    ref, pkg = _cuda_field(reference_build_info), _cuda_field(package_build_info)
    line = (f"separately built library: cuda={ref or 'not reported'}; the package's GPU cores: "
            f"cuda={pkg or 'not reported'}")
    if ref and pkg and ref != pkg:
        line += (" (different toolkits: build the separately built library with the package's "
                 "toolkit, docs/VERIFY.md)")
    return line


def _safe_build_info(lib_path):
    try:
        return AbiLibrary(lib_path).build_info
    except AbiError as exc:
        return f"unavailable: {exc}"


# --------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------
def _per_shot_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean per shot: any element differs (False for empty rows)."""
    if a.shape != b.shape:
        raise VerifyError(f"comparison shape mismatch: {a.shape} != {b.shape}")
    if a.ndim == 1:
        return a != b
    if a.shape[1] == 0:
        return np.zeros(a.shape[0], dtype=bool)
    return np.any(a != b, axis=tuple(range(1, a.ndim)))


def _first_shots(mask: np.ndarray, limit: int = 20) -> list:
    return [int(i) for i in np.flatnonzero(mask)[:limit]]


def _decode_reference(path, cs, corpus):
    """Decode every shot once -> (corrections, observables, status, seconds)."""
    shots, n, k = corpus.shots, corpus.num_mechanisms, corpus.num_observables
    corrections = np.empty((shots, n), dtype=np.uint8)
    observables = np.empty((shots, k), dtype=np.uint8)
    status = np.empty(shots, dtype=np.uint32)
    decoder = AbiDecoder(path, cs, corpus.H, corpus.O, corpus.priors, corpus.dem_text)
    try:
        t0 = time.perf_counter()
        for i in range(shots):
            obs, corr, st = decoder.decode(corpus.dets[i])
            corrections[i] = corr
            observables[i] = obs
            status[i] = st
        seconds = time.perf_counter() - t0
    finally:
        decoder.close()
    return corrections, observables, status, seconds


def _decode_eldpc(decoder, corpus):
    """Decode every shot with the public eldpc API -> (corr, obs, status, seconds)."""
    shots, n, k = corpus.shots, corpus.num_mechanisms, corpus.num_observables
    corrections = np.empty((shots, n), dtype=np.uint8)
    observables = np.empty((shots, k), dtype=np.uint8)
    status = np.empty(shots, dtype=np.uint32)
    t0 = time.perf_counter()
    for i in range(shots):
        corr, obs, st = decoder.decode_full(corpus.dets[i])
        corrections[i] = np.asarray(corr, dtype=np.uint8).reshape(-1)
        observables[i] = np.asarray(obs, dtype=np.uint8).reshape(-1)
        status[i] = int(st)
    return corrections, observables, status, time.perf_counter() - t0


def _upstream_column(corpus, params, reference_status, reference_corrections,
                     reference_observables, reference_path, log) -> dict:
    """Informational comparison with the upstream ldpc wheel on the same shots."""
    available, description = upstream_mod.availability()
    if not available:
        return {"available": False, "reason": description}
    try:
        upstream = upstream_mod.UpstreamDecoder(corpus.H, corpus.O, corpus.priors, params)
    except Exception as exc:  # noqa: BLE001 - the column is informational only
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    differing = []
    correction_mismatches = 0
    observables_mismatches = 0
    masks = None
    auxiliary = None
    cs = config_string(params)
    del log  # kept in the signature for symmetry with the other helpers
    try:
        t0 = time.perf_counter()
        for i in range(corpus.shots):
            corr, obs = upstream.decode(corpus.dets[i])
            corr_differs = not np.array_equal(corr, reference_corrections[i])
            obs_differs = not np.array_equal(obs, reference_observables[i])
            if not (corr_differs or obs_differs):
                continue
            correction_mismatches += int(corr_differs)
            observables_mismatches += int(obs_differs)
            if masks is None:
                masks = upstream_mod.column_masks(corpus.H)
            status = int(reference_status[i])
            if status & ST_BP_CONVERGED:
                tie_free, detail = True, None
            else:
                if auxiliary is None:
                    auxiliary = AbiDecoder(reference_path, cs, corpus.H, corpus.O,
                                           corpus.priors, corpus.dem_text)
                _, _, re_status = auxiliary.decode(corpus.dets[i])
                soft = auxiliary.soft_output()
                tie_free, detail = upstream_mod.shot_cause(
                    int(re_status), soft, corpus.dets[i], params, masks)
            differing.append({
                "shot": int(i),
                "correction_differs": bool(corr_differs),
                "observables_differ": bool(obs_differs),
                "cause": "tie_order" if not tie_free else "unexplained",
                "detail": detail,
            })
        seconds = time.perf_counter() - t0
    finally:
        if auxiliary is not None:
            auxiliary.close()
    causes = {"tie_order": 0, "unexplained": 0}
    for entry in differing:
        causes[entry["cause"]] = causes.get(entry["cause"], 0) + 1
    return {"available": True, "description": description,
            "compared_shots": corpus.shots,
            "differing_shots": len(differing),
            "correction_mismatches": int(correction_mismatches),
            "observables_mismatches": int(observables_mismatches),
            "causes": causes, "differing": differing, "seconds": round(seconds, 4)}


# --------------------------------------------------------------------------
# one row
# --------------------------------------------------------------------------
def _gpu_error_hint(error, device: str) -> str:
    """For a GPU create error: how to verify the CPU core instead ("" for any other error)."""
    if device == "gpu" and "to run on the CPU core" in str(error):
        return ("; run_verify.py compares the CPU core's outputs with --device cpu (or --device "
                "auto)")
    return ""


def run_row(module, presets, corpus, config_name, reference_path, reference_library, use_upstream,
            log, device="gpu") -> dict:
    params = presets[config_name]
    cs = config_string(params)
    shots = corpus.shots

    corr_ref, obs_ref, st_ref, t_ref = _decode_reference(reference_path, cs, corpus)

    # The GPU core is the package's default device; other devices are named explicitly.
    kwargs = {} if device == "gpu" else {"device": device}
    try:
        decoder = module.Decoder(corpus.H, corpus.O, corpus.priors, config=config_name, **kwargs)
    except Exception as exc:  # noqa: BLE001 - report the package's message
        raise VerifyError(f"eldpc.Decoder({corpus.id}, {config_name}) failed: "
                          f"{exc}{_gpu_error_hint(exc, device)}") from None
    used_device = getattr(decoder, "device", device)
    try:
        corr_single, obs_single, st_single, t_single = _decode_eldpc(decoder, corpus)
        t0 = time.perf_counter()
        corr_batch, obs_batch, st_batch = decoder.decode_batch(corpus.dets)
        t_batch = time.perf_counter() - t0
    finally:
        close = getattr(decoder, "close", None)
        if close is not None:
            close()

    corr_batch = np.asarray(corr_batch, dtype=np.uint8).reshape(shots, corpus.num_mechanisms)
    obs_batch = np.asarray(obs_batch, dtype=np.uint8).reshape(shots, corpus.num_observables)
    st_batch = np.asarray(st_batch, dtype=np.uint32).reshape(shots)

    corr_mm = _per_shot_diff(corr_single, corr_ref)
    obs_mm = _per_shot_diff(obs_single, obs_ref)
    batch_corr_mm = _per_shot_diff(corr_batch, corr_ref)
    batch_obs_mm = _per_shot_diff(obs_batch, obs_ref)
    single_batch_mm = (_per_shot_diff(corr_single, corr_batch)
                       | _per_shot_diff(obs_single, obs_batch)
                       | (st_single != st_batch))
    status_ref_mm = st_single != st_ref

    mismatches = int(corr_mm.sum() + obs_mm.sum() + batch_corr_mm.sum()
                     + batch_obs_mm.sum() + single_batch_mm.sum())

    row = {
        "problem": corpus.id,
        "config": config_name,
        "config_string": cs,
        "device": used_device,
        "shots": int(shots),
        "num_detectors": corpus.num_detectors,
        "num_mechanisms": corpus.num_mechanisms,
        "num_observables": corpus.num_observables,
        "mismatches": mismatches,
        "correction_mismatches": int(corr_mm.sum()),
        "observables_mismatches": int(obs_mm.sum()),
        "batch_correction_mismatches": int(batch_corr_mm.sum()),
        "batch_observables_mismatches": int(batch_obs_mm.sum()),
        "single_batch_mismatches": int(single_batch_mm.sum()),
        "status_mismatches_vs_reference": int(status_ref_mm.sum()),
        "first_mismatched_shots": _first_shots(corr_mm | obs_mm | batch_corr_mm | batch_obs_mm
                                               | single_batch_mm),
        "reference_seconds": round(t_ref, 4),
        "eldpc_single_seconds": round(t_single, 4),
        "eldpc_batch_seconds": round(t_batch, 4),
        "upstream": {"available": False, "reason": "disabled"},
    }

    if reference_library is not None:
        ref_lib = AbiDecoder(reference_library, cs, corpus.H, corpus.O, corpus.priors,
                             corpus.dem_text)
        try:
            t0 = time.perf_counter()
            h_obs, h_corr, h_status = ref_lib.decode_batch(corpus.dets)
            t_reflib = time.perf_counter() - t0
        finally:
            ref_lib.close()
        h_corr = np.asarray(h_corr, dtype=np.uint8).reshape(shots, corpus.num_mechanisms)
        h_obs = np.asarray(h_obs, dtype=np.uint8).reshape(shots, corpus.num_observables)
        h_status = np.asarray(h_status, dtype=np.uint32).reshape(shots)
        status_reflib_single = st_single != h_status
        status_reflib_batch = st_batch != h_status
        reflib_status_mm = int(status_reflib_single.sum() + status_reflib_batch.sum())
        row.update({
            "status_mismatches_vs_reference_lib": reflib_status_mm,
            "reference_lib_correction_mismatches": int(_per_shot_diff(h_corr, corr_ref).sum()),
            "reference_lib_observables_mismatches": int(_per_shot_diff(h_obs, obs_ref).sum()),
            "reference_lib_seconds": round(t_reflib, 4),
            "first_status_mismatched_shots": _first_shots(status_reflib_single
                                                          | status_reflib_batch),
        })
        row["mismatches"] += reflib_status_mm
    else:
        row["status_mismatches_vs_reference_lib"] = None

    if use_upstream:
        row["upstream"] = _upstream_column(corpus, params, st_ref, corr_ref, obs_ref,
                                           reference_path, log)

    if row["mismatches"]:
        log(f"  {corpus.id} / {config_name}: {row['mismatches']} mismatches "
            f"(corr {row['correction_mismatches']}, obs {row['observables_mismatches']}, "
            f"batch corr {row['batch_correction_mismatches']}, "
            f"batch obs {row['batch_observables_mismatches']}, "
            f"single-vs-batch {row['single_batch_mismatches']})")
    return row


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def _select(available, requested, what):
    """Selected entries of ``available`` in their configured order."""
    if requested is None:
        return list(available)
    unknown = [x for x in requested if x not in available]
    if unknown:
        raise VerifyError(f"unknown {what} {unknown} (configured: {list(available)})")
    wanted = set(requested)
    return [x for x in available if x in wanted]


def _reference_lib_path(args):
    """The optional separately built library: ``--reference-lib``, else $ELDPC_REFERENCE_LIB.

    An explicit empty ``--reference-lib`` disables the comparison even when the variable is set.
    These are the only two sources; nothing is guessed from the file system.
    """
    if args.reference_lib is not None:
        return args.reference_lib or None
    return os.environ.get("ELDPC_REFERENCE_LIB") or None


def _refuse_own_library(eldpc, args, reference_lib_path) -> None:
    """The status comparison is only meaningful against a library other than eldpc's own.

    Refused: the front given with --lib, the installed package's libeldpc.so, and every library
    installed next to it (the package's own cores)."""
    own = []
    if args.lib:
        own.append(Path(args.lib))
    try:
        front = Path(eldpc._abi.library_path())
        own.append(front)
        own.extend(front.parent.glob("*.so*"))
    except Exception:  # noqa: BLE001 - a package without this helper simply skips the guard
        pass
    for candidate in own:
        try:
            if Path(candidate).resolve() == Path(reference_lib_path).resolve():
                raise VerifyError(
                    f"{reference_lib_path} is one of eldpc's own installed libraries; the status "
                    "comparison needs a separately built library of the same decoder sources "
                    "(docs/VERIFY.md, \"Status words\", says how to build one)")
        except OSError:
            pass


def run(args, log=print) -> int:
    started = time.time()
    t_start = time.perf_counter()
    mode = "full" if args.full else "quick"
    cfg_path = Path(args.config) if args.config else Path(__file__).resolve().parent.parent / "config.json"
    cfg = load_config(cfg_path)
    problem_by_id = {p["id"]: p for p in cfg["problems"]}
    ordered_ids = [p["id"] for p in cfg["problems"]]
    problem_ids = _select(ordered_ids, args.problems, "problems")
    if not problem_ids:
        raise VerifyError("no problems selected")

    for module_name in ("numpy", "scipy", "stim", "qldpc"):
        try:
            __import__(module_name)
        except ImportError as exc:
            if module_name == "qldpc":
                raise VerifyError(
                    f"the verification suite needs qLDPC to build its corpora ({exc}). Install "
                    "the tested revision from the package root with: pip install \".[qldpc]\" "
                    "(docs/VERIFY.md, \"Running it\")") from None
            raise VerifyError(f"the verification suite needs {module_name} ({exc}); it is a "
                              "dependency of the package: pip install .") from None
    log(f"corpus generator: {describe_generator(generator_versions())} (the corpus cache is "
        "keyed on these versions)")

    cache_root = Path(args.cache) if args.cache else default_cache_root()
    out_dir = Path(args.out)

    try:
        import eldpc
    except Exception as exc:  # noqa: BLE001
        raise VerifyError(
            "cannot import the eldpc package; install it (pip install .) or put it on "
            f"PYTHONPATH before running the suite: {exc}") from None
    missing = [name for name in ("Decoder", "presets", "load") if not hasattr(eldpc, name)]
    if missing:
        raise VerifyError(
            f"the eldpc module at {getattr(eldpc, '__file__', None) or 'sys.path'} is not the "
            f"package (missing {missing}); a directory named eldpc on sys.path can shadow the "
            "installed package. Install the package (pip install .) or put its python/ directory "
            "on PYTHONPATH.")

    try:
        presets = eldpc.presets()
    except Exception as exc:  # noqa: BLE001 - report the package's message
        raise VerifyError(f"eldpc.presets() failed: {exc}") from None
    if args.configs:
        config_names = list(args.configs)
    else:
        config_names = list(cfg["configs"][mode])
    unknown = [name for name in config_names if name not in presets]
    if unknown:
        raise VerifyError(f"unknown preset(s) {unknown}; eldpc.presets() has "
                          f"{sorted(presets)}")
    if not config_names:
        raise VerifyError("no configs selected")
    if args.lib:
        try:
            eldpc.load(str(args.lib))
        except Exception as exc:  # noqa: BLE001
            raise VerifyError(f"eldpc.load({args.lib!r}) failed: {exc}") from None

    t_build = time.perf_counter()
    try:
        ref_info = build_reference(cache_root, jobs=args.jobs, log=log)
    except OSError as exc:
        raise VerifyError(f"cannot use the cache directory {cache_root}: {exc}") from None
    reference_build_seconds = time.perf_counter() - t_build
    reference_library = None
    reference_lib_path = _reference_lib_path(args)
    if reference_lib_path:
        _refuse_own_library(eldpc, args, reference_lib_path)
        try:
            reference_library = AbiLibrary(reference_lib_path)
        except AbiError as exc:
            raise VerifyError(str(exc)) from None
        log(f"separately built library: {reference_library.path} ({reference_library.name}); "
            "status words are compared with it")
        log(toolkit_line(reference_library.build_info, _eldpc_info(eldpc).get("build_info")))
    else:
        log("separately built library: none given, so status words are not compared with one "
            "(docs/VERIFY.md, \"Status words\")")

    use_upstream = bool(cfg["upstream"]) and not args.no_upstream
    if use_upstream:
        ok, description = upstream_mod.availability()
        log(f"upstream: {description}" if ok else f"upstream: unavailable ({description})")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VerifyError(f"cannot create the output directory {out_dir}: {exc}") from None
    rows = []
    corpus_seconds = 0.0
    for pid in problem_ids:
        problem = problem_by_id[pid]
        shots = cfg["shots"][mode][pid]
        parent = parent_shots = None
        if problem.get("shape", {}).get("kind") == "window":
            parent = problem_by_id[problem["shape"]["parent"]]
            parent_shots = cfg["shots"][mode][parent["id"]]
        t0 = time.perf_counter()
        try:
            corpus = ensure_corpus(cache_root, problem, cfg["base_seed"], shots,
                                   parent=parent, parent_shots=parent_shots, log=log)
        except (CorpusError, ValueError, OSError) as exc:
            raise VerifyError(f"corpus for {pid}: {exc}") from None
        corpus_seconds += time.perf_counter() - t0
        for config_name in config_names:
            log(f"verify: {pid} / {config_name} ({shots} shots)")
            try:
                row = run_row(eldpc, presets, corpus, config_name, ref_info.path,
                              reference_library, use_upstream, log,
                              device=getattr(args, "device", "gpu"))
            except VerifyError:
                raise
            except AbiError as exc:
                raise VerifyError(f"{pid} / {config_name}: {exc}") from None
            except Exception as exc:  # noqa: BLE001 - report package errors by name
                raise VerifyError(
                    f"{pid} / {config_name}: {type(exc).__name__}: {exc}") from None
            rows.append(row)

    elapsed = time.perf_counter() - t_start
    mismatched_rows = [r for r in rows if r["mismatches"]]
    status = "pass" if not mismatched_rows else "fail"
    total_shots = int(sum(r["shots"] for r in rows))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": status,
        "mode": mode,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "rows": rows,
        "totals": {
            "rows": len(rows),
            "problems": len({r["problem"] for r in rows}),
            "configs": len({r["config"] for r in rows}),
            "shots": total_shots,
            "mismatches": int(sum(r["mismatches"] for r in rows)),
            "status_mismatches_vs_reference": int(sum(r["status_mismatches_vs_reference"]
                                                      for r in rows)),
            "upstream_differing_shots": int(sum(r["upstream"].get("differing_shots", 0)
                                                for r in rows)),
            "upstream_unexplained_shots": int(sum(r["upstream"].get("causes", {})
                                                  .get("unexplained", 0) for r in rows)),
            # None when no separately built library was given (status words not compared with one).
            "status_mismatches_vs_reference_lib": (
                int(sum(r["status_mismatches_vs_reference_lib"] for r in rows))
                if reference_library is not None else None),
        },
        "timings": {
            "total_seconds": round(elapsed, 3),
            "reference_build_seconds": round(reference_build_seconds, 3),
            "corpus_seconds": round(corpus_seconds, 3),
            "reference_decode_seconds": round(sum(r["reference_seconds"] for r in rows), 3),
            "eldpc_single_seconds": round(sum(r["eldpc_single_seconds"] for r in rows), 3),
            "eldpc_batch_seconds": round(sum(r["eldpc_batch_seconds"] for r in rows), 3),
        },
        "selection": {
            "mode": mode,
            "config_path": str(cfg_path),
            "base_seed": cfg["base_seed"],
            "problems": problem_ids,
            "configs": config_names,
            "shots": {pid: cfg["shots"][mode][pid] for pid in problem_ids},
        },
        "environment": _environment(cfg_path, cfg, eldpc, ref_info, reference_library,
                                    args, cache_root),
    }
    if reference_library is not None:
        summary["timings"]["reference_lib_seconds"] = round(
            sum(r.get("reference_lib_seconds", 0.0) for r in rows), 3)
    if use_upstream:
        summary["timings"]["upstream_seconds"] = round(
            sum(r["upstream"].get("seconds", 0.0) for r in rows), 3)

    summary_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, sort_keys=True, allow_nan=False)
            f.write("\n")
        md_path.write_text(render_summary(summary), encoding="utf-8")
    except OSError as exc:
        raise VerifyError(f"cannot write the summary to {out_dir}: {exc}") from None
    log(f"summary: {summary_path}")
    log(f"summary: {md_path}")
    if reference_library is not None:
        compared = ("corrections and observables vs the reference, status words vs the separately "
                    "built library")
    else:
        compared = ("corrections and observables vs the reference; status words not compared "
                    "(no separately built library, see docs/VERIFY.md)")
    devices = sorted({row.get("device") for row in rows if row.get("device")})
    if devices and devices != ["gpu"]:
        # The default (GPU) line stays as it was; another engine is named in it.
        compared = f"eldpc on the {' and '.join(d.upper() for d in devices)} core: " + compared
    if status == "pass":
        log(f"PASS: {len(rows)} rows, {total_shots} shots, 0 mismatches; {compared} "
            f"({elapsed:.1f} s)")
        return 0
    log(f"FAIL: {len(mismatched_rows)} of {len(rows)} rows have mismatches "
        f"({summary['totals']['mismatches']} comparisons differ); {compared}")
    return 1


def _environment(cfg_path, cfg, eldpc, ref_info, reference_library, args, cache_root) -> dict:
    env = {
        "gpu": _gpu_info(),
        "cpu": _cpu_info(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "nvcc": _nvcc_info(),
        "packages": _package_versions(),
        "eldpc": _eldpc_info(eldpc),
        "reference": ref_info.as_dict(),
        "reference_library_build_info": _safe_build_info(ref_info.path),
        "reference_lib": None,
        "cache_root": str(cache_root),
        "arguments": {
            "mode": "full" if args.full else "quick",
            "configs": args.configs,
            "problems": args.problems,
            "lib": str(args.lib) if args.lib else None,
            "device": getattr(args, "device", "gpu"),
            "reference_lib": str(_reference_lib_path(args)) if _reference_lib_path(args) else None,
        },
        "config_sha256": None,
        "base_seed": cfg["base_seed"],
        "command": " ".join([sys.executable] + sys.argv),
    }
    if reference_library is not None:
        env["reference_lib"] = {"path": reference_library.path, "name": reference_library.name,
                                "build_info": reference_library.build_info}
    try:
        h = hashlib.sha256()
        with open(cfg_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        env["config_sha256"] = h.hexdigest()
    except OSError:
        pass
    return env


# --------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------
def _fmt_seconds(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "-"


def render_summary(summary: dict) -> str:
    env = summary["environment"]
    sel = summary["selection"]
    totals = summary["totals"]
    lines = [f"# eldpc verification: {summary['mode']}", ""]
    gpu = env.get("gpu") or {}
    cpu = env.get("cpu") or {}
    eldpc = env.get("eldpc") or {}
    lines += [
        f"- GPU: {gpu.get('name', 'unknown')} (compute capability "
        f"{gpu.get('compute_capability', 'unknown')}, driver "
        f"{gpu.get('driver_version', 'unknown')})",
        f"- CPU: {cpu.get('model', 'unknown')} ({cpu.get('logical_cpus', '?')} logical CPUs)",
        f"- eldpc: version {eldpc.get('version', 'unknown')}, ISA {eldpc.get('isa', 'unknown')}",
        f"- eldpc build: {eldpc.get('build_info', 'unknown')}",
        f"- Reference: {env.get('reference_library_build_info', 'unknown')}",
        f"- Config: {sel['config_path']} (base seed {sel['base_seed']})",
        f"- Shots per problem: " + ", ".join(f"{k} {v}" for k, v in sel["shots"].items()),
        "",
        f"## Result: {summary['status'].upper()}",
        "",
        f"{totals['rows']} rows, {totals['shots']} shots, "
        f"{totals['mismatches']} mismatches between eldpc and the reference "
        f"(corrections and observables, single-shot and batch).",
        "",
    ]
    if totals.get("status_mismatches_vs_reference_lib") is not None:
        lines += [f"Status words against the separately built library: "
                  f"{totals['status_mismatches_vs_reference_lib']} mismatches (single-shot and batch).", ""]
    lines += [
        "| problem | config | shots | mismatches | corr | obs | batch corr | batch obs | "
        "single vs batch | ref s | eldpc single s | eldpc batch s | upstream differing |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        up = row.get("upstream") or {}
        if up.get("available"):
            upstream_cell = f"{up['differing_shots']}/{up['compared_shots']}"
            if up["differing_shots"]:
                upstream_cell += f" (tie order {up['causes'].get('tie_order', 0)}, " \
                                 f"unexplained {up['causes'].get('unexplained', 0)})"
        else:
            upstream_cell = "n/a"
        lines.append(
            f"| {row['problem']} | {row['config']} | {row['shots']} | {row['mismatches']} | "
            f"{row['correction_mismatches']} | {row['observables_mismatches']} | "
            f"{row['batch_correction_mismatches']} | {row['batch_observables_mismatches']} | "
            f"{row['single_batch_mismatches']} | {_fmt_seconds(row['reference_seconds'])} | "
            f"{_fmt_seconds(row['eldpc_single_seconds'])} | "
            f"{_fmt_seconds(row['eldpc_batch_seconds'])} | {upstream_cell} |")
    lines += ["", "## Status words", ""]
    if env.get("reference_lib"):
        lines.append(f"Status words were compared with the separately built library "
                     f"{env['reference_lib']['path']} ({env['reference_lib']['name']}).")
        lines.append("")
        lines.append("| problem | config | status mismatches vs separate library |")
        lines.append("|---|---|---:|")
        for row in summary["rows"]:
            lines.append(f"| {row['problem']} | {row['config']} | "
                         f"{row['status_mismatches_vs_reference_lib']} |")
    else:
        lines.append("No separately built library was given (--reference-lib or "
                     "ELDPC_REFERENCE_LIB), so status words were not compared with one; "
                     "docs/VERIFY.md (\"Status words\") says how to build one. Status "
                     "words against the reference are in summary.json "
                     f"(total differing: {totals['status_mismatches_vs_reference']}).")
    lines += ["", "## Upstream wheel (informational)", ""]
    differing = []
    for row in summary["rows"]:
        up = row.get("upstream") or {}
        if not up.get("available"):
            continue
        for entry in up.get("differing", []):
            differing.append((row["problem"], row["config"], entry))
    if not any((r.get("upstream") or {}).get("available") for r in summary["rows"]):
        reasons = sorted({str((r.get("upstream") or {}).get("reason")) for r in summary["rows"]})
        if reasons == ["disabled"]:
            lines.append("The upstream wheel comparison was skipped (--no-upstream or the "
                         "configuration disables it).")
        else:
            lines.append("The upstream ldpc wheel was not available, so this column was "
                         "skipped: " + "; ".join(reasons))
    elif not differing:
        lines.append("The upstream ldpc 2.4.1 wheel agreed with the reference on every shot of "
                     "this run (corrections and observables).")
    else:
        lines.append("Shots where the upstream wheel differs from the reference:")
        lines.append("")
        lines.append("| problem | config | shot | correction | observables | cause | detail |")
        lines.append("|---|---|---:|---|---|---|---|")
        for problem, config, entry in differing:
            detail = entry.get("detail")
            detail_text = "" if detail is None else json.dumps(detail, sort_keys=True)
            lines.append(f"| {problem} | {config} | {entry['shot']} | "
                         f"{'differs' if entry['correction_differs'] else 'same'} | "
                         f"{'differs' if entry['observables_differ'] else 'same'} | "
                         f"{entry['cause']} | {detail_text} |")
    t = summary["timings"]
    lines += ["", "## Timing", "",
              f"- Whole run: {t['total_seconds']} s "
              f"(reference build {_fmt_seconds(t.get('reference_build_seconds'))} s, "
              f"corpus generation {t['corpus_seconds']} s)",
              f"- Reference decode: {t['reference_decode_seconds']} s",
              f"- eldpc single-shot decode: {t['eldpc_single_seconds']} s",
              f"- eldpc batch decode: {t['eldpc_batch_seconds']} s"]
    if "upstream_seconds" in t:
        lines.append(f"- Upstream wheel decode: {t['upstream_seconds']} s")
    if "reference_lib_seconds" in t:
        lines.append(f"- Separately built library decode: {t['reference_lib_seconds']} s")
    lines.append("")
    return "\n".join(lines)
