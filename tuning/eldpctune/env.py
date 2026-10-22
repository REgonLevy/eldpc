# SPDX-License-Identifier: Apache-2.0
"""Environment capture: every results file records the machine and the toolchain it ran on.

The GPU name, driver and compute capability are recorded from ``nvidia-smi``.  The CUDA toolkit is
the loaded library's: the ``cuda=`` field of ``eldpc.build_info()`` names the toolkit that compiled
its GPU cores (whose runtime is linked into them), recorded as ``toolkit`` with ``source:
"build_info"``; only when build_info names none (a CPU-only build, or no library loaded) is the
CUDA compiler asked (``nvcc --version``, and ``$CUDACXX --version`` when that is set), with
``source: "nvcc"``.  The CUDA devices visible to the process are recorded from ``eldpc.devices()``
together with ``CUDA_VISIBLE_DEVICES``.  Paths and host names are deliberately not recorded.
"""
from __future__ import annotations

import datetime
import os
import platform
import re
import subprocess

GPU_QUERY = ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv"]
NVCC_QUERY = ["nvcc", "--version"]


def run_command(argv: list[str], timeout: float = 30.0, display: str | None = None) -> dict:
    """Run one query command; record the command line, exit code and output."""
    record = {"command": display or " ".join(argv), "exit_code": None, "stdout": "",
              "stderr": ""}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        record["exit_code"] = int(proc.returncode)
        record["stdout"] = proc.stdout.strip()
        record["stderr"] = proc.stderr.strip()
    except FileNotFoundError:
        record["exit_code"] = None
        record["stderr"] = "not found"
    except subprocess.TimeoutExpired:
        record["exit_code"] = None
        record["stderr"] = "timed out"
    return record


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 - optional diagnostics
        return None


_BUILD_INFO_CUDA = re.compile(r"(?:^|[ ;])cuda=\S+ (\d+)\.(\d+)((?:\.\d+)*)")
_NVCC_RELEASE = re.compile(r"release\s+([\d.]+),\s*(V[\d.]+)")


def toolkit_record(build_info, commands: list[dict]) -> dict | None:
    """The CUDA toolkit of the measured library: ``{version, nvcc, first_line, source}``.

    From the ``cuda=<compiler id> <version>`` field of ``build_info`` (``source: "build_info"``),
    else from the first recorded compiler query that names a release (``source: "nvcc"``, with
    the command in ``command``); None when neither names a toolkit.  The same rule as the
    benchmark's environment record.
    """
    m = _BUILD_INFO_CUDA.search(build_info if isinstance(build_info, str) else "")
    if m:
        full = f"{m.group(1)}.{m.group(2)}{m.group(3)}"
        return {"version": f"{m.group(1)}.{m.group(2)}", "nvcc": f"V{full}",
                "first_line": None, "source": "build_info"}
    for command in commands:
        if "nvcc" not in command["command"] and "CUDACXX" not in command["command"]:
            continue
        text = (command.get("stdout") or "").strip()
        if command.get("exit_code") != 0 or not text:
            continue
        m = _NVCC_RELEASE.search(text)
        lines = text.splitlines()
        return {"version": m.group(1) if m else None, "nvcc": m.group(2) if m else None,
                "first_line": lines[0] if lines else None, "source": "nvcc",
                "command": command["command"]}
    return None


def capture_environment() -> dict:
    """Machine, GPU, toolkit and library versions of this run."""
    import eldpc

    commands = [run_command(GPU_QUERY)]
    try:
        build_info = eldpc.build_info()
    except Exception as exc:  # noqa: BLE001
        build_info = f"unavailable: {exc}"
    if toolkit_record(build_info, []) is None:
        # The library names no toolkit (CPU-only build or not loaded): ask the compiler.
        cudacxx = os.environ.get("CUDACXX")
        if cudacxx:
            # Record the variable rather than the machine-dependent absolute path of the compiler.
            commands.append(run_command([cudacxx, "--version"], display="$CUDACXX --version"))
        commands.append(run_command(NVCC_QUERY))
    try:
        isa = eldpc.isa()
    except Exception as exc:  # noqa: BLE001
        isa = f"unavailable: {exc}"
    try:
        devices = eldpc.devices()
    except Exception as exc:  # noqa: BLE001
        devices = [{"error": str(exc)}]
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commands": commands,
        "toolkit": toolkit_record(build_info, commands),
        "eldpc": {
            "version": str(getattr(eldpc, "__version__", "unknown")),
            "build_info": build_info,
            "isa": isa,
            "devices": devices,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": _module_version("numpy"),
            "scipy": _module_version("scipy"),
            "stim": _module_version("stim"),
            "qldpc": _module_version("qldpc"),
        },
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "gpu": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "devices": devices},
        },
    }
