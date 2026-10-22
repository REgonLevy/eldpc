# SPDX-License-Identifier: Apache-2.0
"""``device="auto"``: the GPU core when device 0 is usable, else the CPU core with a warning.

The choice is made in Python, before the native create, so that the warning can say why and so that
``threads`` applies only when the CPU core is chosen. :func:`probe` collects what the choice needs
(the build's GPU cores and their CUDA images from ``eldpc.build_info()``, and device 0 through the
CUDA driver API); :func:`choose` applies the rules to a :class:`Probe`, so the rules can be tested
with any probe result and no GPU. The C front's ``device=auto`` makes the same choice
(``src/port/eldpc_front.cpp``, ``gpu_usable``); the compute-capability rules and the image text
mirror ``src/port/gpu_arch.h``.

The rules, in order: a CPU-only build (no GPU cores) -> CPU; no NVIDIA driver, a failed ``cuInit``
or no device 0 -> CPU; a driver that does not support the CUDA major version the GPU cores were
compiled with -> CPU; a compute capability the cores carry no SASS or PTX image for -> CPU;
otherwise the GPU.
"""

from __future__ import annotations

import ctypes
import re
import threading
import warnings
from dataclasses import dataclass

from . import _devices

#: The reason of a CPU-only build (the package was built without a CUDA toolkit).
CPU_ONLY_REASON = "this build has no GPU cores (built without a CUDA toolkit)"

_CUDA_ERROR_NO_DEVICE = 100
_ATTR_COMPUTE_CAPABILITY_MAJOR = 75
_ATTR_COMPUTE_CAPABILITY_MINOR = 76
_ATTR_MULTIPROCESSOR_COUNT = 16

_IMAGE_RE = re.compile(r"^(sm|compute)_(\d{2,})(a|f)?$")


@dataclass(frozen=True)
class CudaImage:
    """One CUDA image of the GPU cores: ``sm_XY`` (SASS) or ``compute_XY`` (PTX)."""

    name: str
    major: int
    minor: int
    ptx: bool
    arch_specific: bool = False


@dataclass(frozen=True)
class Probe:
    """What ``device="auto"`` decides from.

    ``gpu_cores``: whether the build has GPU cores. ``images``: the cores' CUDA images (empty when
    the build does not report them; the compute-capability check is then skipped).
    ``cuda_version``: the CUDA toolkit version the cores were compiled with (``"13.3.73"``), or
    None. ``device_error``: why device 0 could not be queried (None when it was).
    ``device_name``, ``compute_capability`` (major, minor) and ``driver_version`` (the
    ``cuDriverGetVersion`` integer, 0 when unknown) describe device 0.
    """

    gpu_cores: bool
    images: tuple = ()
    cuda_version: str | None = None
    device_error: str | None = None
    device_name: str = ""
    compute_capability: tuple | None = None
    driver_version: int = 0


def build_info_field(info: str, key: str) -> str:
    """The value of ``key=`` in a build-info string, or ``""`` (mirror of gpu_arch.h)."""
    match = re.search(r"(?:^|[ ;])" + re.escape(key) + r"=([^;]*)", info)
    return match.group(1) if match else ""


def parse_cuda_arch(text: str) -> tuple:
    """``"sm_89,sm_120,compute_120"`` -> the images; entries of another form are skipped."""
    out = []
    for item in text.split(","):
        item = item.strip()
        match = _IMAGE_RE.match(item)
        if not match:
            continue
        number = int(match.group(2))
        out.append(CudaImage(item, number // 10, number % 10, match.group(1) == "compute",
                             match.group(3) == "a"))
    return tuple(out)


def device_supported(images, major: int, minor: int) -> bool:
    """Whether a device of compute capability major.minor can run one of the images.

    SASS ``sm_XY`` runs on X.Z for Z >= Y (``sm_XYa`` only on X.Y); PTX ``compute_XY`` on X.Y or
    newer.
    """
    for im in images:
        if im.ptx:
            if (major, minor) >= (im.major, im.minor):
                return True
        elif im.arch_specific:
            if (major, minor) == (im.major, im.minor):
                return True
        elif major == im.major and minor >= im.minor:
            return True
    return False


def images_text(images) -> str:
    """``"SASS for sm_89, sm_120 and PTX for compute_120 (compute capability 8.9; 12.0 or newer)"``."""
    sass = ", ".join(im.name for im in images if not im.ptx)
    ptx = ", ".join(im.name for im in images if im.ptx)
    ptx_images = [im for im in images if im.ptx]
    covers = []
    for im in images:
        if im.ptx or any((im.major, im.minor) >= (p.major, p.minor) for p in ptx_images):
            continue
        covers.append(f"{im.major}.{im.minor}")
    covers += [f"{im.major}.{im.minor} or newer" for im in ptx_images]
    parts = []
    if sass:
        parts.append("SASS for " + sass)
    if ptx:
        parts.append("PTX for " + ptx)
    if not parts:
        return "no CUDA images"
    return " and ".join(parts) + " (compute capability " + "; ".join(covers) + ")"


def unsupported_reason(device_name: str, major: int, minor: int, images, index: int = 0) -> str:
    who = f"device {index}" + (f" ({device_name})" if device_name else "")
    return (f"{who} has compute capability {major}.{minor}, which the GPU cores of this build do "
            f"not support: they carry {images_text(images)}")


def _cuda_major(cuda_version: str | None) -> int:
    try:
        return int(str(cuda_version).split(".")[0])
    except (TypeError, ValueError):
        return 0


def device_support(probe: Probe, index: int = 0) -> tuple:
    """``(True, None)`` when the probed device can run this build's GPU cores, else
    ``(False, reason)``: the rules of :func:`choose` after the device query (no GPU cores, a
    driver too old for the cores' CUDA version, a compute capability without an image).
    ``index`` names the device in the reason."""
    if not probe.gpu_cores:
        return False, CPU_ONLY_REASON
    cuda_major = _cuda_major(probe.cuda_version)
    if cuda_major > 0 and probe.driver_version > 0 and probe.driver_version // 1000 < cuda_major:
        v = probe.driver_version
        return False, (f"the NVIDIA driver supports CUDA {v // 1000}.{(v % 1000) // 10}, but the GPU "
                       f"cores of this build were compiled with CUDA {probe.cuda_version} and need a "
                       f"driver that supports CUDA {cuda_major}.x")
    if probe.images and probe.compute_capability is not None:
        major, minor = probe.compute_capability
        if not device_supported(probe.images, major, minor):
            return False, unsupported_reason(probe.device_name, major, minor, probe.images, index)
    return True, None


def choose(probe: Probe) -> tuple:
    """``("gpu", None)`` or ``("cpu", reason)`` for this probe result."""
    if not probe.gpu_cores:
        return "cpu", CPU_ONLY_REASON
    if probe.device_error is not None:
        return "cpu", "no usable CUDA device: " + probe.device_error
    ok, reason = device_support(probe)
    return ("gpu", None) if ok else ("cpu", reason)


def _error_text(lib, rc: int) -> str:
    """``"CUDA_ERROR_NO_DEVICE (100)"`` from cuGetErrorName when the driver has it."""
    fn = getattr(lib, "cuGetErrorName", None)
    if fn is not None:
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        name = ctypes.c_char_p()
        if fn(rc, ctypes.byref(name)) == 0 and name.value:
            return f"{name.value.decode('utf-8', 'replace')} ({rc})"
    return f"CUresult {rc}"


def probe_device0(driver=None):
    """Device 0 through the CUDA driver API: ``(None, name, (major, minor), driver_version)``, or
    ``(reason, "", None, 0)`` when there is no driver, ``cuInit`` fails or device 0 cannot be read.

    ``driver`` overrides the ``(library, error)`` pair of ``eldpc._devices._driver()`` (tests).
    """
    lib, error = driver if driver is not None else _devices._driver()
    if lib is None:
        return ("the NVIDIA driver library libcuda.so.1 cannot be loaded (no NVIDIA driver is "
                "installed, or it is not visible to this process)"), "", None, 0
    rc = lib.cuInit(0)
    if rc != 0:
        reason = f"cuInit failed with {_error_text(lib, rc)}"
        if rc == _CUDA_ERROR_NO_DEVICE:
            reason += ": the driver sees no CUDA device (check CUDA_VISIBLE_DEVICES)"
        return reason, "", None, 0
    count = ctypes.c_int(0)
    rc = lib.cuDeviceGetCount(ctypes.byref(count))
    if rc != 0 or count.value <= 0:
        detail = _error_text(lib, rc) if rc != 0 else "no device"
        return f"the CUDA driver could not query device 0 ({detail})", "", None, 0
    values = []
    for attr in (_ATTR_COMPUTE_CAPABILITY_MAJOR, _ATTR_COMPUTE_CAPABILITY_MINOR,
                 _ATTR_MULTIPROCESSOR_COUNT):
        value = ctypes.c_int(0)
        rc = lib.cuDeviceGetAttribute(ctypes.byref(value), attr, 0)
        if rc != 0:
            return f"the CUDA driver could not query device 0 ({_error_text(lib, rc)})", "", None, 0
        values.append(int(value.value))
    if values[2] <= 0:
        return "the CUDA driver could not query device 0 (no multiprocessors)", "", None, 0
    version = ctypes.c_int(0)
    driver_version = 0
    if lib.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value > 0:
        driver_version = int(version.value)
    return None, _devices._device_name(lib, 0), (values[0], values[1]), driver_version


def probe(build_info: str, driver=None) -> Probe:
    """The probe for this process: the build's GPU cores from ``build_info`` (the loaded
    library's ``eldpc.build_info()``), then device 0. A CPU-only build makes no CUDA call."""
    if "lib=fastbpgpu6g" not in build_info:
        return Probe(gpu_cores=False)
    images = parse_cuda_arch(build_info_field(build_info, "cuda_arch"))
    cuda = build_info_field(build_info, "cuda").strip()
    cuda_version = cuda.rsplit(" ", 1)[-1] if cuda else None
    device_error, name, cc, driver_version = probe_device0(driver)
    return Probe(gpu_cores=True, images=images, cuda_version=cuda_version, device_error=device_error,
                 device_name=name, compute_capability=cc, driver_version=driver_version)


_WARNED = False
_WARN_LOCK = threading.Lock()


def warn_cpu_fallback(reason: str, stacklevel: int = 5) -> None:
    """Warn once per process that ``device="auto"`` chose the CPU core, and why."""
    global _WARNED
    with _WARN_LOCK:
        if _WARNED:
            return
        _WARNED = True
    hint = ""
    if reason == CPU_ONLY_REASON:
        hint = ("; for the GPU cores install the CUDA 13 toolkit "
                "(https://developer.nvidia.com/cuda-downloads), put its nvcc on PATH or set CUDACXX, "
                "and reinstall eldpc")
    warnings.warn(f'eldpc: device="auto" runs on the CPU core: {reason}{hint}. This warning is '
                  f'shown once per process; device="cpu" selects the CPU core without it.',
                  RuntimeWarning, stacklevel=stacklevel)


def resolve(front) -> tuple:
    """``("gpu", None)`` or ``("cpu", reason)`` for this process, with the warning on "cpu"."""
    device, reason = choose(probe(front.build_info()))
    if device == "cpu":
        warn_cpu_fallback(reason)
    return device, reason
