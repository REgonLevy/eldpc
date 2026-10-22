# SPDX-License-Identifier: Apache-2.0
"""The installation check, :func:`eldpc.check`, and the warning at ``import eldpc`` about a
CPU-only build on a machine with a GPU.

``pip install .`` builds the CPU-only package when CMake finds no CUDA compiler (no ``nvcc`` on
``PATH``, ``CUDACXX`` unset; docs/BUILD.md, "CPU-only build"), and pip shows CMake's warning about
that only with ``pip install -v``. Two things make it visible where a user looks:

- At import, :func:`warn_at_import` warns once per process when the installed build has no GPU
  cores *because CMake found no CUDA compiler* and the CUDA driver reports a device. It reads the
  build record that CMake installs next to the libraries (``_libs/build.json``) and loads no eldpc
  library, so the ISA choice of :func:`eldpc.load` is still open after the import. On a build
  with GPU cores, on a CPU-only build chosen with ``ELDPC_WITHOUT_CUDA=1`` and without a build
  record (a source-tree run) it makes no CUDA call. Otherwise it queries device 0 the way
  ``device="auto"`` does (:func:`eldpc._auto.probe_device0`: ``cuInit``, ``cuDeviceGetCount`` and
  attribute queries; no CUDA context is created). It never raises.
- :func:`check` prints the build (the GPU cores with their CUDA images and toolkit, the CPU cores,
  the host ISA), the devices the CUDA driver reports, a test decode on each usable core, and a last
  line that says ``GPU ready: ...`` or ``CPU only: <why>``; or ``NOT READY: <why>`` when no core
  can decode, after which it raises :class:`eldpc.EldpcError` (so ``python -c "import eldpc;
  eldpc.check()"`` exits non-zero).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import warnings
from pathlib import Path

from . import _abi, _auto, _devices

#: The README section that says how to install the CUDA toolkit and build the GPU cores.
README_TOOLKIT = "README.md#cuda-13-toolkit"

#: Why a build has no GPU cores, when its build record says CMake found no CUDA compiler.
NO_NVCC_REASON = "this build has no GPU cores because nvcc was not found at build time"

#: Why a build has no GPU cores, when it was built CPU-only on purpose.
EXPLICIT_CPU_ONLY_REASON = ("this build has no GPU cores: it was built CPU-only on purpose "
                            "(ELDPC_WITHOUT_CUDA=1)")

#: What to do about a build without GPU cores.
GPU_CORES_HINT = (f"{README_TOOLKIT}: install the CUDA 13 toolkit, put its nvcc on PATH or set "
                  f"CUDACXX, then pip install . again")

#: The README section that lists the GPUs the build supports by name.
README_GPUS = "README.md#supported-gpus"

#: The file name of the build record in the package's ``_libs`` directory.
BUILD_RECORD = "build.json"

# The test problem of the README's "Quick start: Python from H, O and priors" and its documented
# output at ms30: correction [1 0 0], observable [1], status 131073 (converged, 2 iterations).
_TEST_H = ((1, 1, 0), (0, 1, 1))
_TEST_O = ((1, 0, 1),)
_TEST_PRIORS = (0.01, 0.01, 0.01)
_TEST_SYNDROME = (1, 0)
_TEST_EXPECTED = ([1, 0, 0], [1], 131073)

_WARNED = False
_WARN_LOCK = threading.Lock()


def build_record_path() -> Path:
    """The build record of the installed libraries: ``<package>/_libs/build.json``."""
    return _abi.library_path().parent / BUILD_RECORD


def read_build_record(path=None):
    """The build record as a dict, or None when it is missing or unreadable.

    CMake writes it at configure time: ``{"schema": "eldpc.build.v1", "gpu_cores": true|false,
    "cuda_choice": "auto"|"explicit"}``; ``"auto"`` means that CMake looked for a CUDA compiler
    (``ELDPC_WITHOUT_CUDA`` unset), ``"explicit"`` that ``ELDPC_WITHOUT_CUDA`` decided.
    """
    p = Path(path) if path is not None else build_record_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def known_build_info():
    """What this process knows of the build's GPU cores, as a build-info string, or None.

    The loaded library's ``build_info()`` when the library is loaded; otherwise the build record
    (``lib=``, ``cuda=`` and ``cuda_arch=`` rebuilt from it), so that :func:`eldpc.devices` can
    say whether the build supports a device without loading a library; None without either.
    """
    front = _abi._LIBRARY
    if front is not None and front.load_error() is None:
        return front.build_info()
    record = read_build_record()
    if not isinstance(record, dict) or not isinstance(record.get("gpu_cores"), bool):
        return None
    if not record["gpu_cores"]:
        return "lib=fastbpcpu6g"
    return (f"lib=fastbpgpu6g; cuda={record.get('cuda') or ''}; "
            f"cuda_arch={record.get('cuda_arch') or ''}")


def support_rule():
    """A function ``(index, name, (major, minor), driver_version) -> (supported, reason)`` that
    applies this build's device rule (:func:`eldpc._auto.device_support`) to one device, for
    :func:`eldpc.devices`. ``supported`` is None when the build is unknown (no build record and no
    loaded library) or its GPU images are not reported."""
    info = known_build_info()
    if info is None:
        unknown = ("unknown: this process has no build record of the eldpc libraries and has not "
                   "loaded one")
        return lambda index, name, cc, driver_version: (None, unknown)
    if "lib=fastbpgpu6g" not in info:
        front = _abi._LIBRARY
        reason = _cpu_only_reason(front.path if front is not None else str(_abi.library_path()))
        return lambda index, name, cc, driver_version: (False, reason)
    images = _auto.parse_cuda_arch(_auto.build_info_field(info, "cuda_arch"))
    cuda = _auto.build_info_field(info, "cuda").strip()
    cuda_version = cuda.rsplit(" ", 1)[-1] if cuda else None

    def rule(index, name, cc, driver_version):
        if not images:
            return None, "unknown: the build reports no CUDA images for its GPU cores"
        probe = _auto.Probe(gpu_cores=True, images=images, cuda_version=cuda_version,
                            device_name=name, compute_capability=cc,
                            driver_version=driver_version)
        return _auto.device_support(probe, index)
    return rule


def cpu_only_without_nvcc(record) -> bool:
    """Whether ``record`` is that of a CPU-only build CMake chose because it found no CUDA compiler."""
    return (isinstance(record, dict) and record.get("gpu_cores") is False
            and record.get("cuda_choice") == "auto")


def import_warning_text(record, device0):
    """The text of the import warning, or None when there is nothing to warn about.

    ``record`` is the build record (or None); ``device0`` is a callable that returns the tuple of
    :func:`eldpc._auto.probe_device0`. It is called only for a CPU-only build without nvcc.
    """
    if not cpu_only_without_nvcc(record):
        return None
    error, name, _cc, _version = device0()
    if error is not None:
        return None
    gpu = f" ({name})" if name else ""
    return (f'eldpc: {NO_NVCC_REASON}, but the CUDA driver reports a GPU{gpu}, so device="gpu" '
            f'fails (device="cpu" and device="auto" run on the CPU core). To build the GPU cores, '
            f'see {README_TOOLKIT}: put nvcc on PATH or set CUDACXX, then pip install . again. '
            f'python -c "import eldpc; eldpc.check()" prints what was built. This warning is '
            f'shown once per process; ELDPC_WITHOUT_CUDA=1 builds the CPU-only package on purpose, '
            f'without it.')


def warn_at_import(stacklevel: int = 3) -> None:
    """The warning at ``import eldpc`` (see the module docstring): at most once per process.

    It never raises: a missing or broken driver, an unreadable record, and warnings turned into
    errors (``python -W error``) all leave the import working.
    """
    global _WARNED
    try:
        with _WARN_LOCK:
            if _WARNED:
                return
            _WARNED = True
        text = import_warning_text(read_build_record(), _auto.probe_device0)
        if text:
            warnings.warn(text, RuntimeWarning, stacklevel=stacklevel)
    except Exception:  # noqa: BLE001 - the import must never fail because of this check
        pass


def _cpu_only_reason(front_path: str) -> str:
    """Why the loaded library has no GPU cores, from the build record when it describes it."""
    try:
        installed = Path(front_path).resolve() == _abi.library_path()
    except OSError:
        installed = False
    record = read_build_record() if installed else None
    if cpu_only_without_nvcc(record):
        return NO_NVCC_REASON
    if isinstance(record, dict) and record.get("gpu_cores") is False:
        return EXPLICIT_CPU_ONLY_REASON
    return _auto.CPU_ONLY_REASON


def _test_decode(device: str):
    """``None`` when a decode of the README's H/O/priors problem on ``device`` returns the
    documented output, otherwise a one-line description of what went wrong."""
    import numpy as np

    from . import Decoder

    try:
        with Decoder(np.array(_TEST_H, dtype=np.uint8), np.array(_TEST_O, dtype=np.uint8),
                     list(_TEST_PRIORS), config="ms30", device=device) as dec:
            correction, observables, status = dec.decode_full(list(_TEST_SYNDROME))
    except Exception as exc:  # noqa: BLE001 - report any failure, do not raise
        return f"{type(exc).__name__}: {exc}"
    got = (correction.tolist(), observables.tolist(), int(status))
    if got != _TEST_EXPECTED:
        return f"unexpected output {got}, expected {_TEST_EXPECTED}"
    return None


def _device_line(index: int, d: dict) -> str:
    driver = d.get("nvidia_driver_version") or "release unknown"
    supported = {True: "yes", False: "NO", None: "unknown"}.get(d.get("supported"), "unknown")
    return (f"  {index}: {d.get('name')}, compute capability {d.get('compute_capability')}, "
            f"{d.get('sm_count')} SMs, NVIDIA driver {driver} (CUDA {d.get('cuda_driver_version')}), "
            f"core variant {d.get('core_variant')}; supported by this build: {supported}")


def _say_environment(say) -> None:
    shown = [f"{k}={os.environ[k]!r}" for k in ("CUDA_VISIBLE_DEVICES", "ELDPC_ISA", "ELDPC_DEVICE_CLASS")
             if k in os.environ]
    if shown:
        say(f"environment: {', '.join(shown)}")


def _say_devices(say):
    """Print the devices lines; return (devices, why there is none or None)."""
    try:
        devices = _devices.devices()
    except Exception:  # noqa: BLE001 - devices() does not raise; be safe anyway
        devices = []
    if devices:
        say(f"devices:     {len(devices)} CUDA device{'s' if len(devices) != 1 else ''} "
            f"(eldpc uses device 0)")
        for i, d in enumerate(devices):
            say(_device_line(i, d))
        return devices, None
    device_error = _auto.probe_device0()[0] or "the CUDA driver reports no device"
    say(f"devices:     none: {device_error}")
    return devices, device_error


def _report_load_error(say, front, load_error: str, gpu_libs, cpu_libs) -> None:
    """The report when the front loaded no core: the error as the reason on the ISA, GPU-cores and
    test-decode lines; what was built from the build record (the libraries say nothing now)."""
    say(f"host ISA:    none: {load_error}")
    record = read_build_record()
    if isinstance(record, dict) and record.get("gpu_cores") is True:
        say(f"GPU cores:   built (build record: cuda={record.get('cuda') or '?'}, "
            f"cuda_arch={record.get('cuda_arch') or '?'}), but not usable: no decoder core could be "
            f"loaded (host ISA above)")
        say(f"             {', '.join(gpu_libs) or 'no core library found next to the front'}")
    elif isinstance(record, dict) and record.get("gpu_cores") is False:
        say(f"GPU cores:   no: {_cpu_only_reason(front.path)}")
    else:
        say("GPU cores:   unknown: no decoder core could be loaded (host ISA above) and there is no "
            "build record next to the libraries")
    say(f"CPU cores:   {', '.join(cpu_libs) if cpu_libs else 'none found next to the front'}")
    _say_environment(say)
    _say_devices(say)
    say("test decode: not run: no decoder core could be loaded")
    say(f"NOT READY: no decoder core could be loaded: {load_error}")


def check(version: str, file=None) -> bool:
    """Print what this installation built and sees, and whether the GPU is ready (see eldpc.check)."""
    out = sys.stdout if file is None else file

    def say(text: str = "") -> None:
        print(text, file=out)

    say(f"eldpc {version} ({Path(__file__).resolve().parent})")
    try:
        front = _abi.load()
    except Exception as exc:  # noqa: BLE001 - say what failed, then raise below
        say(f"library:     cannot be loaded: {exc}")
        say(f"NOT READY: the eldpc library cannot be loaded ({exc})")
        raise _abi.EldpcError(f"NOT READY: the eldpc library cannot be loaded ({exc})") from None
    lib_dir = Path(front.path).resolve().parent
    gpu_libs = sorted(p.name for p in lib_dir.glob("libeldpc_core_*.so"))
    cpu_libs = sorted(p.name for p in lib_dir.glob("libeldpc_cpu_*.so"))
    say(f"library:     {front.path}")
    load_error = front.load_error()
    if load_error is not None:
        _report_load_error(say, front, load_error, gpu_libs, cpu_libs)
        raise _abi.EldpcError(f"NOT READY: no decoder core could be loaded: {load_error}")
    info = front.build_info()
    has_gpu_cores = "lib=fastbpgpu6g" in info
    isa = front.isa()
    say(f"host ISA:    {isa} (the {isa} build of each core is used; ELDPC_ISA=avx2 or avx512 "
        f"forces one)")
    if has_gpu_cores:
        cuda = _auto.build_info_field(info, "cuda").strip()
        toolkit = f"the CUDA {cuda.rsplit(' ', 1)[-1]} toolkit" if cuda else "an unnamed CUDA toolkit"
        images = _auto.parse_cuda_arch(_auto.build_info_field(info, "cuda_arch"))
        say(f"GPU cores:   yes, built with {toolkit} (build_info: cuda={cuda or '?'})")
        say(f"             {_auto.images_text(images)}")
        say(f"             {', '.join(gpu_libs) or 'no core library found next to the front'}")
        cpu_only_reason = None
    else:
        cpu_only_reason = _cpu_only_reason(front.path)
        say(f"GPU cores:   no: {cpu_only_reason}")
    say(f"CPU cores:   {', '.join(cpu_libs) if cpu_libs else 'none found next to the front'}")
    _say_environment(say)
    devices, device_error = _say_devices(say)

    if has_gpu_cores:
        device, reason = _auto.choose(_auto.probe(info))
    else:
        device, reason = "cpu", cpu_only_reason
    gpu_error = _test_decode("gpu") if device == "gpu" else None
    cpu_error = _test_decode("cpu")
    parts = []
    if device == "gpu":
        parts.append("GPU core ok" if gpu_error is None else f"GPU core FAILED: {gpu_error}")
    parts.append("CPU core ok" if cpu_error is None else f"CPU core FAILED: {cpu_error}")
    say(f"test decode: {'; '.join(parts)}")

    if device == "gpu" and gpu_error is None:
        name = devices[0].get("name") if devices else "device 0"
        say(f'GPU ready: device="gpu" (the default) decodes on device 0 ({name}); device="cpu" '
            f'runs the CPU core.')
        return True
    if cpu_error is not None:
        line = ("NOT READY: the CPU core failed its test decode"
                + ("" if device != "gpu" else " and so did the GPU core"))
        say(line)
        raise _abi.EldpcError(f"{line}: {cpu_error}")
    hint = ""
    if device == "gpu":
        why = f"the GPU core failed its test decode ({gpu_error})"
    elif not has_gpu_cores:
        why = reason if devices else f"{reason}, and no CUDA device is visible: {device_error}"
        hint = f" To build the GPU cores, see {GPU_CORES_HINT}."
    else:
        why = reason
        if "which the GPU cores of this build do not support" in reason:
            hint = f" The GPUs this build supports: {README_GPUS}."
    say(f'CPU only: {why}. device="gpu" fails; device="cpu" and device="auto" run on the CPU '
        f'core.{hint}')
    return False
