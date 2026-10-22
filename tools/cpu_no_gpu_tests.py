#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the package tests with no CUDA device visible, for each CPU-core ISA, and record the counts.

    CUDA_VISIBLE_DEVICES= ELDPC_REFERENCE_LIB=... ELDPC_REFERENCE_LDPC_LIB=... \
        python tools/cpu_no_gpu_tests.py [--isa avx512,avx2] [--out tools/results/cpu/no-gpu-tests.json]

For every ISA it runs `pytest -m cpu_core` and `pytest -m "not cpu_core"` with
``CUDA_VISIBLE_DEVICES`` empty and ``ELDPC_ISA`` set, reads the JUnit XML, and writes one JSON
record: the command, the commit, the ISA, the libraries' SHA-256 and the test/failure/error/skip
counts of both selections. The reference-lib environment variables of the package tests are passed
through when set (their presence is recorded; their machine paths are not).

This is the recorded evidence for "the CPU core and its tests run with no GPU visible"; docs/BUILD.md
("Tests without a GPU") gives the same two selections.

    python tools/cpu_no_gpu_tests.py --no-driver [--isa avx512,avx2] [--out tools/results/cpu/no-driver-run.json]

``--no-driver`` runs without the NVIDIA driver instead: it compiles a small ``LD_PRELOAD`` shim (C
source below) that makes every ``dlopen`` of ``libcuda.so*`` and ``libnvidia-ml.so*`` fail, so the
process sees no driver at all (``CUDA_VISIBLE_DEVICES`` is left unset), and for every ISA runs
``pytest -m cpu_core`` and a check: ``eldpc.devices()`` is empty, a ``device="cpu"`` decode returns
the README's output, a ``device="gpu"`` create fails with a message that names ``device="cpu"``,
``eldpc.check()``'s last line, and no driver library is mapped into the process (docs/SUPPORT.md,
"The CPU core").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1] / "tests"
RESULTS = Path(__file__).resolve().parent / "results" / "cpu"


# The LD_PRELOAD shim of --no-driver: dlopen of the NVIDIA driver libraries fails, everything else
# goes to the real dlopen.
SHIM_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>

void *dlopen(const char *file, int mode) {
    static void *(*real_dlopen)(const char *, int) = NULL;
    if (file != NULL && (strstr(file, "libcuda.so") != NULL || strstr(file, "libnvidia-ml.so") != NULL)) {
        return NULL;
    }
    if (real_dlopen == NULL) real_dlopen = (void *(*)(const char *, int))dlsym(RTLD_NEXT, "dlopen");
    return real_dlopen(file, mode);
}
"""

# What --no-driver checks in a fresh process with the shim preloaded (the README's H/O problem).
NO_DRIVER_CHECK = r"""
import io, json
import numpy as np
import eldpc
H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
O = np.array([[1, 0, 1]], dtype=np.uint8)
out = {"devices": eldpc.devices(), "isa": eldpc.isa()}
with eldpc.Decoder(H, O, [0.01] * 3, config="ms30", device="cpu") as dec:
    c, o, s = dec.decode_full([1, 0])
out["cpu_decode"] = [c.tolist(), o.tolist(), int(s)]
try:
    eldpc.Decoder(H, O, [0.01] * 3, config="ms30").close()
    out["gpu_create_error"] = None
except eldpc.EldpcError as exc:
    out["gpu_create_error"] = str(exc)
buf = io.StringIO()
try:
    out["check_returned"] = eldpc.check(file=buf)
except eldpc.EldpcError as exc:
    out["check_returned"] = f"raised: {exc}"
out["check_last_line"] = buf.getvalue().splitlines()[-1]
maps = open("/proc/self/maps").read().splitlines()
out["driver_libraries_mapped"] = sorted({line.split()[-1] for line in maps
                                         if "libcuda.so" in line or "libnvidia-ml.so" in line})
print("NO_DRIVER_CHECK=" + json.dumps(out))
"""


def build_shim(workdir: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise SystemExit("--no-driver needs a C compiler (cc) to build the LD_PRELOAD shim")
    src = workdir / "no_driver_shim.c"
    src.write_text(SHIM_C)
    lib = workdir / "no_driver_shim.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(lib), str(src), "-ldl"], check=True)
    return lib


def no_driver_check(env: dict, workdir: Path) -> dict:
    proc = subprocess.run([sys.executable, "-c", NO_DRIVER_CHECK], cwd=workdir, env=env,
                          capture_output=True, text=True, timeout=600)
    for line in proc.stdout.splitlines():
        if line.startswith("NO_DRIVER_CHECK="):
            data = json.loads(line[len("NO_DRIVER_CHECK="):])
            break
    else:
        return {"returncode": proc.returncode, "stderr_tail": proc.stderr[-2000:], "ok": False}
    gpu_error = data.get("gpu_create_error") or ""
    data["returncode"] = proc.returncode
    data["ok"] = (proc.returncode == 0 and data["devices"] == []
                  and data["cpu_decode"] == [[1, 0, 0], [1], 131073]
                  and 'device="cpu"' in gpu_error.replace("device=cpu", 'device="cpu"')
                  and data["driver_libraries_mapped"] == []
                  and data["check_last_line"].startswith("CPU only:"))
    return data


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def counts(xml: Path) -> dict:
    root = ET.parse(xml).getroot()
    s = root if root.tag == "testsuite" else root.find("testsuite")
    return {k: int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}


def run_selection(pytest: list[str], selection: str, env: dict, workdir: Path, name: str) -> dict:
    xml = workdir / f"{name}.xml"
    cmd = pytest + ["-q", "-m", selection, "--junitxml", str(xml), str(TESTS)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)
    elapsed = round(time.time() - started, 1)
    result = {"selection": selection, "returncode": proc.returncode, "elapsed_s": elapsed}
    if xml.is_file():
        result.update(counts(xml))
    else:
        result["tests"] = 0
        result["stdout_tail"] = proc.stdout[-2000:]
        result["stderr_tail"] = proc.stderr[-2000:]
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--isa", default="avx512,avx2")
    ap.add_argument("--no-driver", action="store_true",
                    help="hide the NVIDIA driver with an LD_PRELOAD shim instead of CUDA_VISIBLE_DEVICES=")
    ap.add_argument("--out", default=None,
                    help="the record (default: tools/results/cpu/no-gpu-tests.json, or no-driver-run.json "
                         "with --no-driver)")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(RESULTS / ("no-driver-run.json" if a.no_driver else "no-gpu-tests.json"))

    import eldpc  # the installed package under test

    package = Path(eldpc.__file__).resolve().parent
    libs = package / "_libs"
    # The commit to record is the source tree's (the wheel installed elsewhere has none).
    root = Path(__file__).resolve().parents[1]
    workdir = Path(tempfile.mkdtemp(prefix="eldpc-cpu-tests-"))
    env = os.environ.copy()
    if a.no_driver:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["LD_PRELOAD"] = str(build_shim(workdir))
    else:
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "":
            raise SystemExit('run with CUDA_VISIBLE_DEVICES set to the empty string: CUDA_VISIBLE_DEVICES= ...')
        env["CUDA_VISIBLE_DEVICES"] = ""
    test_sets = []
    for isa in a.isa.split(","):
        isa = isa.strip()
        if not isa:
            continue
        env_isa = dict(env)
        env_isa["ELDPC_ISA"] = isa
        lib = libs / f"libeldpc_cpu_{isa}.so"
        record = {"isa": isa, "cpu_library": lib.name,
                  "cpu_library_sha256": hashlib.sha256(lib.read_bytes()).hexdigest() if lib.is_file() else None,
                  "selections": []}
        selections = [("cpu_core", f"cpu_core_{isa}")]
        if not a.no_driver:
            selections.append(("not cpu_core", f"rest_{isa}"))
        for selection, name in selections:
            record["selections"].append(run_selection([sys.executable, "-m", "pytest"], selection,
                                                       env_isa, workdir, name))
        if a.no_driver:
            record["no_driver_check"] = no_driver_check(env_isa, workdir)
        test_sets.append(record)

    prefix = "" if a.no_driver else "CUDA_VISIBLE_DEVICES= "
    result = {
        "schema": "eldpc.cpu_no_gpu_tests.v1",
        "command": prefix + "python tools/cpu_no_gpu_tests.py " + " ".join(sys.argv[1:]),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(root),
        "package_version": eldpc.__version__,
        "cuda_visible_devices": None if a.no_driver else "",
        "no_driver_shim": ("LD_PRELOAD shim: dlopen of libcuda.so* and libnvidia-ml.so* fails "
                           "(SHIM_C in this script)") if a.no_driver else None,
        "reference_env": sorted(k for k in env if k.startswith("ELDPC_") and k != "ELDPC_ISA"),
        "python": sys.version.split()[0],
        "sets": test_sets,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for record in test_sets:
        for sel in record["selections"]:
            print(record["isa"], sel["selection"], "rc", sel["returncode"], "tests", sel.get("tests"),
                  "failures", sel.get("failures"), "errors", sel.get("errors"), "skipped", sel.get("skipped"))
        if "no_driver_check" in record:
            print(record["isa"], "no-driver check ok", record["no_driver_check"].get("ok"))
    print(f"wrote {out}")
    ok = all(sel["returncode"] == 0 for record in test_sets for sel in record["selections"])
    ok = ok and all(record.get("no_driver_check", {"ok": True}).get("ok") for record in test_sets)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
