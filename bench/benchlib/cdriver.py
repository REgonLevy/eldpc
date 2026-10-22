# SPDX-License-Identifier: Apache-2.0
"""Build and run the C driver (bench/c/bench_driver.c) and find the eldpc C library and header.

The driver is compiled on the user's machine with the system C compiler the first time the benchmark
runs.  It is an ordinary consumer of the public C API: it includes the installed eldpc.h and loads the
libeldpc.so to measure with dlopen, because one process must hold two libraries at once to compare two
builds (symbols of the same name cannot be linked twice).  bench.py finds the library explicitly
(--lib / ELDPC_LIB), through the Python package (the path eldpc.load() reports), or next to the installed
package (<site-packages>/eldpc/_libs/libeldpc.so).  The header comes from ELDPC_INCLUDE, from pkg-config
(with <site-packages>/eldpc/lib/pkgconfig added to PKG_CONFIG_PATH), or from <site-packages>/eldpc/include.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import formats


class DriverError(RuntimeError):
    pass


def find_compiler() -> str:
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand and shutil.which(cand):
            return cand
    raise DriverError("no C compiler found (set CC or install cc/gcc/clang)")


def _probe(path: str) -> bool:
    """True if the library exports the eldpc C API entry points the driver needs."""
    try:
        lib = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_LOCAL)
    except OSError:
        return False
    return all(hasattr(lib, name) for name in ("eldpc_create", "eldpc_decode", "eldpc_destroy"))


def _python_hint(eldpc_module) -> str | None:
    """The library path the Python package loaded, if it reports one.

    eldpc.load() returns the loaded library object, which carries its path.  That is the library the
    Python measurements use, and the C driver should load the same one.
    """
    try:
        ret = eldpc_module.load()
    except Exception:
        return None
    for obj in (ret, getattr(ret, "lib", None)):
        if obj is None:
            continue
        for attr in ("path", "_name"):
            p = getattr(obj, attr, None)
            if isinstance(p, str) and p and os.path.exists(p):
                return os.path.abspath(p)
    return None


def _package_dir(eldpc_module) -> Path | None:
    try:
        return Path(eldpc_module.__file__).resolve().parent
    except Exception:
        return None


def _package_candidates(eldpc_module) -> list[str]:
    pkg = _package_dir(eldpc_module)
    if pkg is None:
        return []
    return [str(pkg / "_libs" / "libeldpc.so")]


def find_library(eldpc_module, explicit: str | None = None) -> dict:
    """Return {'path', 'source'} for the C library to measure."""
    tried = []
    candidates = []
    if explicit:
        candidates.append((explicit, "command line"))
    envlib = os.environ.get("ELDPC_LIB")
    if envlib:
        candidates.append((envlib, "ELDPC_LIB"))
    hint = _python_hint(eldpc_module)
    if hint:
        candidates.append((hint, "loaded by the Python package"))
    candidates += [(p, "installed with the package") for p in _package_candidates(eldpc_module)]
    found = ctypes.util.find_library("eldpc")
    if found:
        candidates.append((found, "loader search path"))
    for path, source in candidates:
        if not path or not os.path.exists(path):
            tried.append(f"{path} ({source}): does not exist")
            continue
        if not _probe(path):
            tried.append(f"{path} ({source}): does not export the eldpc C API")
            continue
        return {"path": os.path.abspath(path), "source": source}
    detail = "; ".join(tried) if tried else "no candidate library was found"
    raise DriverError(
        "cannot find the eldpc C library (libeldpc.so). Pass --lib PATH (or set ELDPC_LIB). Tried: "
        + detail)


def _header_dirs_from_pkg_config(pkg: Path | None) -> list[str]:
    exe = shutil.which("pkg-config") or shutil.which("pkgconf")
    if not exe:
        return []
    env = dict(os.environ)
    if pkg is not None:
        extra = str(pkg / "lib" / "pkgconfig")
        env["PKG_CONFIG_PATH"] = extra + (os.pathsep + env["PKG_CONFIG_PATH"] if env.get("PKG_CONFIG_PATH")
                                          else "")
    try:
        p = subprocess.run([exe, "--cflags-only-I", "eldpc"], capture_output=True, text=True, env=env,
                           timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if p.returncode != 0:
        return []
    return [tok[2:] for tok in p.stdout.split() if tok.startswith("-I") and len(tok) > 2]


def find_include_dirs(eldpc_module) -> list[str]:
    """Directories to pass as -I so that the C driver includes the installed public header eldpc.h.

    Order: ELDPC_INCLUDE; pkg-config (`pkg-config --cflags eldpc`, with <site-packages>/eldpc/lib/pkgconfig
    appended to PKG_CONFIG_PATH); <site-packages>/eldpc/include.  Raises DriverError when none holds
    eldpc.h.
    """
    pkg = _package_dir(eldpc_module)
    candidates = []
    if os.environ.get("ELDPC_INCLUDE"):
        candidates.append(os.environ["ELDPC_INCLUDE"])
    candidates += _header_dirs_from_pkg_config(pkg)
    if pkg is not None:
        candidates.append(str(pkg / "include"))
    for d in candidates:
        if (Path(d) / "eldpc.h").is_file():
            return [str(Path(d).resolve())]
    raise DriverError(
        "cannot find eldpc.h, the header of the eldpc C API. It is installed at <site-packages>/eldpc/"
        "include/eldpc.h; set ELDPC_INCLUDE to its directory, or PKG_CONFIG_PATH to "
        "<site-packages>/eldpc/lib/pkgconfig. Tried: " + (", ".join(candidates) or "nothing"))


def _needs_rebuild(out: Path, sources: list[Path], stamp: str) -> bool:
    if not out.exists():
        return True
    stamp_file = out.with_suffix(out.suffix + ".stamp")
    if not stamp_file.exists() or stamp_file.read_text() != stamp:
        return True
    mtime = out.stat().st_mtime
    return any(s.stat().st_mtime > mtime for s in sources)


def _compile(compiler: str, src: Path, out: Path, includes: list[str], shared: bool, extra: list[str],
             stamp: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [compiler, "-O2", "-std=c11", "-Wall", "-Wextra"]
    if shared:
        cmd += ["-fPIC", "-shared"]
    for d in includes:
        cmd += ["-I", d]
    cmd += [str(src), "-o", str(out)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise DriverError(f"compiling {src.name} failed:\n{' '.join(cmd)}\n{p.stderr.strip()}")
    out.with_suffix(out.suffix + ".stamp").write_text(stamp)
    return out


def build_driver(bench_dir: Path, workdir: Path, include_dirs: list[str], compiler: str | None = None) -> Path:
    compiler = compiler or find_compiler()
    src = bench_dir / "c" / "bench_driver.c"
    header = Path(include_dirs[0]) / "eldpc.h"
    out = workdir / "bench_driver"
    stamp = "|".join([compiler] + include_dirs + [str(src), str(header)])
    if _needs_rebuild(out, [src, header], stamp):
        _compile(compiler, src, out, include_dirs, shared=False, extra=["-ldl"], stamp=stamp)
    return out


def driver_command(driver: Path, problem_file, shots_file, out_file, library_a: str, library_b: str,
                   config_a: str, config_b: str, order: str, mode: str, gap_us: int, keep_warm_a: bool,
                   keep_warm_b: bool, shots_per_variant: int, warmup: int, cpu: int | None,
                   warmup_ms: int = 1500, create_order: str | None = None,
                   batch_shots: int | None = None, batch_reps: int | None = None,
                   sequential: bool = False) -> list[str]:
    """`mode` is the driver mode, "b2b", "cadence" (a cadence may have a zero gap) or "batch".  `order`
    is the call order; `create_order` (default: the same) is the order the two handles are created in.
    `batch_shots`/`batch_reps` are passed only in batch mode.  `sequential` (the CPU core) adds
    ``--handles sequential``: one live handle at a time, created, warmed, timed and destroyed in call
    order; without it the command is the interleaved one, unchanged."""
    cmd = [str(driver),
           "--problem", str(problem_file), "--shots", str(shots_file), "--out", str(out_file),
           "--lib-a", str(library_a), "--lib-b", str(library_b),
           "--config-a", config_a, "--config-b", config_b,
           "--order", order, "--create-order", create_order or order,
           "--mode", mode, "--gap-us", str(int(gap_us)),
           "--keep-warm-a", "1" if keep_warm_a else "0",
           "--keep-warm-b", "1" if keep_warm_b else "0",
           "--shots-per-variant", str(int(shots_per_variant)), "--warmup", str(int(warmup)),
           "--warmup-ms", str(int(warmup_ms)), "--cpu", str(-1 if cpu is None else int(cpu))]
    if batch_shots is not None:
        cmd += ["--batch-shots", str(int(batch_shots))]
    if batch_reps is not None:
        cmd += ["--batch-reps", str(int(batch_reps))]
    if sequential:
        cmd += ["--handles", "sequential"]
    return cmd


def core_variants_from_stderr(text: str) -> list[str]:
    """[variant of handle 0, variant of handle 1] from the driver's ``core_variant`` lines.

    A handle the driver did not report (an older driver) is ``"unknown"``.
    """
    found = dict(re.findall(r"^bench_driver: core_variant ([01]) (\S+)\s*$", text or "", re.M))
    return [found.get("0", "unknown"), found.get("1", "unknown")]


def run_driver(cmd: list[str], timeout: float = 1800.0) -> dict:
    out_file = cmd[cmd.index("--out") + 1]
    t0 = time.monotonic()
    try:
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise DriverError(f"the C driver did not finish within {timeout:.0f} s") from e
    wall = time.monotonic() - t0
    if p.returncode == 2 or not os.path.exists(out_file):
        raise DriverError(f"the C driver failed (exit {p.returncode}): {p.stderr.strip()}\n"
                          f"command: {' '.join(str(c) for c in cmd)}")
    result = formats.read_driver_output(out_file)
    result["stderr"] = p.stderr.strip()
    result["core_variants"] = core_variants_from_stderr(p.stderr)
    result["wall_s"] = round(wall, 3)
    result["exit"] = p.returncode
    return result
