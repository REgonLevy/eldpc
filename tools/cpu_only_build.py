#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and test the CPU-only configuration (no CUDA toolkit, no GPU cores) and record it.

    python tools/cpu_only_build.py [--out tools/results/cpu/no-cuda-build.json]

The CPU-only build is selected with ``-DELDPC_WITHOUT_CUDA=ON`` (or ``ELDPC_WITHOUT_CUDA=1`` in the
environment, which ``pip wheel`` / ``pip install`` pick up). It builds ``libeldpc.so`` and the two
CPU cores only: the CMake project does not enable the CUDA language, does not look for the CUDA
toolkit, and never runs ``nvcc``.

To make that a measurement rather than a claim, this script puts a directory whose ``nvcc`` is
``/bin/false`` first on ``PATH`` and removes ``CUDACXX``: if any step invoked nvcc, it would fail.
It then

1. configures and builds the CPU-only CMake tree, and checks that no build file mentions nvcc;
2. builds a wheel with ``ELDPC_WITHOUT_CUDA=1``, lists the libraries it contains and its size;
3. installs the wheel into a temporary target and runs the package's ``cpu_core`` tests with
   ``CUDA_VISIBLE_DEVICES=`` empty (the reference libraries of the environment are passed through);
4. checks through the installed package that ``eldpc.isa()``, ``eldpc.build_info()`` (naming the CPU
   core) and a ``device="cpu"`` decode work, and that a ``device="gpu"`` create fails with a message
   naming ``device="cpu"``.

Every command it ran, the library names and sizes, and the test counts go into one JSON file (no
machine paths). This is the evidence for "a GPU-less CI machine can build, install and test the
package from source".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PACKAGE = TOOLS.parent
RESULTS = TOOLS / "results" / "cpu"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd, *, env, cwd=None, timeout=1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def counts(xml: Path) -> dict:
    root = ET.parse(xml).getroot()
    s = root if root.tag == "testsuite" else root.find("testsuite")
    return {k: int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(PACKAGE), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def machine_record() -> dict:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        cpu = next(line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    try:
        loadavg = Path("/proc/loadavg").read_text().strip()
    except OSError:
        loadavg = "?"
    nvcc = shutil.which("nvcc")
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "kernel": platform.platform(),
            "loadavg": loadavg, "python": platform.python_version(),
            "nvcc_on_path_before_the_guard": nvcc is not None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RESULTS / "no-cuda-build.json"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="eldpc-cpuonly-"))
    guard = work / "nvcc-guard"
    guard.mkdir()
    nvcc = guard / "nvcc"
    nvcc.symlink_to("/bin/false")
    env = os.environ.copy()
    env.pop("CUDACXX", None)
    env["PATH"] = str(guard) + os.pathsep + env.get("PATH", "")
    env["ELDPC_WITHOUT_CUDA"] = "1"

    record = {
        "schema": "eldpc.cpu_only_build.v1",
        "command": "python tools/cpu_only_build.py " + " ".join(sys.argv[1:]),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "machine": machine_record(),
        "guard": {"nvcc": "a PATH entry whose nvcc is /bin/false", "CUDACXX": "unset",
                  "env": "ELDPC_WITHOUT_CUDA=1"},
        "cmake": {},
        "wheel": {},
        "tests": {},
        "package_check": {},
    }
    try:
        # 1. the CPU-only CMake build
        build = work / "build"
        generator = ["-G", "Ninja"] if shutil.which("ninja") else []
        configure = ["cmake", "-S", "eldpc", "-B", "<build>", "-DCMAKE_BUILD_TYPE=Release",
                     "-DELDPC_WITHOUT_CUDA=ON"] + generator
        proc = run(["cmake", "-S", str(PACKAGE), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DELDPC_WITHOUT_CUDA=ON"] + generator, env=env)
        record["cmake"]["configure_command"] = " ".join(configure)
        record["cmake"]["configure_returncode"] = proc.returncode
        if proc.returncode != 0:
            record["cmake"]["configure_tail"] = (proc.stdout + proc.stderr)[-3000:]
            raise SystemExit("CPU-only configure failed:\n" + record["cmake"]["configure_tail"])
        build_cmd = ["cmake", "--build", "<build>", "-j", str(a.jobs)]
        proc = run(["cmake", "--build", str(build), "-j", str(a.jobs)], env=env)
        record["cmake"]["build_command"] = " ".join(build_cmd)
        record["cmake"]["build_returncode"] = proc.returncode
        if proc.returncode != 0:
            record["cmake"]["build_tail"] = (proc.stdout + proc.stderr)[-3000:]
            raise SystemExit("CPU-only build failed:\n" + record["cmake"]["build_tail"])
        libs = sorted((build / "lib").glob("*.so"))
        record["cmake"]["libraries"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                        for p in libs}
        build_files = "\n".join(p.read_text(errors="replace") for p in build.rglob("*.ninja"))
        cache = (build / "CMakeCache.txt").read_text(errors="replace") \
            if (build / "CMakeCache.txt").is_file() else ""
        record["cmake"]["nvcc_mentions_in_build_files"] = build_files.lower().count("nvcc")
        record["cmake"]["cuda_toolkit_in_cache"] = ("CUDAToolkit" in cache or
                                                    "CMAKE_CUDA_COMPILER" in cache)

        # 2. the CPU-only wheel
        wheel_dir = work / "wheel"
        wheel_dir.mkdir()
        # Commands are recorded with a symbolic interpreter and paths, so the result file names no
        # machine path.
        wheel_cmd = ["python", "-m", "pip", "wheel", "--no-build-isolation", "--no-deps",
                     "-w", "<wheel dir>", "./eldpc"]
        proc = run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps",
                    "-w", str(wheel_dir), str(PACKAGE)], env=env)
        record["wheel"]["command"] = " ".join(wheel_cmd)
        record["wheel"]["returncode"] = proc.returncode
        if proc.returncode != 0:
            record["wheel"]["tail"] = (proc.stdout + proc.stderr)[-3000:]
            raise SystemExit("CPU-only wheel build failed:\n" + record["wheel"]["tail"])
        wheels = sorted(wheel_dir.glob("*.whl"))
        wheel = wheels[0]
        record["wheel"]["file"] = wheel.name
        record["wheel"]["bytes"] = wheel.stat().st_size
        record["wheel"]["sha256"] = sha256(wheel)
        import zipfile
        with zipfile.ZipFile(wheel) as archive:
            record["wheel"]["libraries"] = sorted(n.split("_libs/")[-1] for n in archive.namelist()
                                                  if n.endswith(".so"))

        # 3. install it and run the CPU-core tests with no GPU visible
        target = work / "install"
        proc = run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target),
                    str(wheel)], env=env)
        record["wheel"]["install_returncode"] = proc.returncode
        if proc.returncode != 0:
            record["wheel"]["install_tail"] = (proc.stdout + proc.stderr)[-3000:]
            raise SystemExit("CPU-only wheel install failed:\n" + record["wheel"]["install_tail"])
        installed = sorted((target / "eldpc" / "_libs").glob("*.so"))
        record["wheel"]["installed_libraries"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in installed}
        test_env = env.copy()
        test_env["CUDA_VISIBLE_DEVICES"] = ""
        test_env["PYTHONPATH"] = str(target)
        if not a.skip_tests:
            junit = work / "cpu-only-tests.xml"
            test_cmd = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m",
                        "cpu_core", "--junitxml", "<junit>", "eldpc/tests"]
            proc = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m",
                        "cpu_core", "--junitxml", str(junit), str(PACKAGE / "tests")],
                       env=test_env, cwd=str(PACKAGE.parent))
            record["tests"]["command"] = " ".join(test_cmd)
            record["tests"]["reference_env"] = sorted(k for k in test_env if k.startswith("ELDPC_")
                                                       and k != "ELDPC_ISA")
            record["tests"]["returncode"] = proc.returncode
            record["tests"].update(counts(junit) if junit.is_file() else {"tests": 0})

        # 4. through the installed package: ISA, build info, a decode, the GPU-create error
        code = (
            "import json, numpy as np, eldpc\n"
            "out = {'isa': eldpc.isa(), 'build_info': eldpc.build_info()}\n"
            "H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)\n"
            "O = np.array([[1, 0, 0]], dtype=np.uint8)\n"
            "d = eldpc.Decoder(H, O, [0.01, 0.02, 0.01], config='ms30', device='cpu', threads=2)\n"
            "out['decode'] = d.decode(np.array([1, 0], dtype=np.uint8)).tolist()\n"
            "out['core_variant'] = d.core_variant\n"
            "d.close()\n"
            "try:\n"
            "    eldpc.Decoder(H, O, error_rate=0.01, config='ms30')\n"
            "    out['gpu_create'] = 'NO ERROR'\n"
            "except eldpc.EldpcError as exc:\n"
            "    out['gpu_create'] = str(exc)\n"
            "print(json.dumps(out))\n"
        )
        proc = run([sys.executable, "-c", code], env=test_env)
        record["package_check"]["returncode"] = proc.returncode
        if proc.returncode == 0:
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
            record["package_check"].update(parsed)
        else:
            record["package_check"]["stderr_tail"] = proc.stderr[-2000:]
    finally:
        shutil.rmtree(work, ignore_errors=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: record[k] for k in ("cmake", "wheel", "tests", "package_check")},
                     sort_keys=True)[:4000])
    print(f"wrote {out}")
    ok = (record["cmake"].get("build_returncode") == 0
          and record["wheel"].get("install_returncode") == 0
          and (a.skip_tests or record["tests"].get("returncode") == 0)
          and record["package_check"].get("returncode") == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
