# SPDX-License-Identifier: Apache-2.0
"""CUDA device enumeration through the CUDA driver API (libcuda.so.1).

The package links the CUDA runtime statically into its decoder cores, and nothing in it loads
libcudart at run time. Device enumeration therefore goes through the driver interface every
NVIDIA installation provides: ``cuInit``, ``cuDeviceGetCount``, ``cuDeviceGetName``,
``cuDeviceGetAttribute`` and ``cuDriverGetVersion`` from libcuda.so.1. This needs no decoder core
to be loaded and works on a host whose CPU cannot run either core.

Two driver versions are reported, and they are different quantities:

- ``cuda_driver_version`` -- the CUDA version the installed driver supports, from the CUDA
  driver API (``cuDriverGetVersion``), as ``"major.minor"``.
- ``nvidia_driver_version`` -- the version of the installed NVIDIA display driver, from NVML
  (``nvmlSystemGetDriverVersion`` of libnvidia-ml.so.1), as the driver prints it, e.g.
  ``"580.159.03"``. NVML is part of the driver installation, not of the package; when it is
  missing the value is ``None`` and ``nvidia_driver_version_reason`` says why.
"""

from __future__ import annotations

import ctypes
import os

from ._variant import effective_core_variant

# CUdevice_attribute values of the fields reported here (cuda.h; stable across CUDA versions).
_ATTR_MULTIPROCESSOR_COUNT = 16
_ATTR_L2_CACHE_SIZE = 38
_ATTR_COMPUTE_CAPABILITY_MAJOR = 75
_ATTR_COMPUTE_CAPABILITY_MINOR = 76

_CUDA_SUCCESS = 0
_NVML_SUCCESS = 0

# The driver ships libcuda.so.1; the unversioned name exists only where a development package
# creates the linker symlink, so it is the fallback, not the first choice.
_DRIVER_SONAMES = ("libcuda.so.1", "libcuda.so")
_NVML_SONAMES = ("libnvidia-ml.so.1", "libnvidia-ml.so")

# nvmlSystemGetDriverVersion writes a version string such as "580.159.03" (NVML documents at
# least 80 bytes); a small buffer is enough.
_NVML_VERSION_BUFFER = 80

# Successful NVML reads, per tuple of library names tried. One nvmlInit/query/nvmlShutdown cycle
# costs about 14 ms on a 580 driver (the shutdown after the first query tears down the device
# state NVML initialised lazily), which is too much to pay on every informational devices() call.
# The value cannot change while the process runs, so it is cached; failures are not cached, so a
# host where NVML appears later still reports the version.
_NVML_VERSION_CACHE: dict[tuple, str] = {}


def _open(names):
    """``(ctypes.CDLL, None)`` for the first loadable library of ``names``, or ``(None, reason)``."""
    errors = []
    for name in names:
        try:
            return ctypes.CDLL(name), None
        except OSError as exc:
            errors.append(str(exc))
    if not errors:
        return None, "no library name to try"
    return None, "; ".join(errors)


def _driver():
    """The CUDA driver library with the entry points bound, or ``(None, reason)``."""
    lib, error = _open(_DRIVER_SONAMES)
    if lib is None:
        return None, error
    try:
        lib.cuInit.restype = ctypes.c_int
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDeviceGetCount.restype = ctypes.c_int
        lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuDeviceGetName.restype = ctypes.c_int
        lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.cuDeviceGetAttribute.restype = ctypes.c_int
        lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        lib.cuDriverGetVersion.restype = ctypes.c_int
        lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        # Optional: a driver too old for cuDeviceGetUuid_v2 still enumerates devices.
        uuid_fn = getattr(lib, "cuDeviceGetUuid_v2", None)
        if uuid_fn is None:
            uuid_fn = getattr(lib, "cuDeviceGetUuid", None)
        if uuid_fn is not None:
            uuid_fn.restype = ctypes.c_int
            uuid_fn.argtypes = [ctypes.c_char_p, ctypes.c_int]
    except AttributeError as exc:
        return None, f"the CUDA driver library has no {exc} entry point"
    return lib, None


def _nvml_error_string(lib, code):
    try:
        fn = lib.nvmlErrorString
        fn.restype = ctypes.c_char_p
        fn.argtypes = [ctypes.c_int]
        text = fn(code)
        if text:
            return text.decode("utf-8", "replace")
    except AttributeError:
        pass
    return f"NVML error {code}"


def _nvml_driver_version(sonames=None):
    """``(version, reason)`` of the NVIDIA display driver version from NVML.

    ``version`` is the string nvmlSystemGetDriverVersion returns (for example "580.159.03"), or
    ``None``; ``reason`` is None when a version was read and a one-sentence explanation
    otherwise (no NVML in the process, nvmlInit failed, or the query failed). ``sonames``
    overrides the library names tried; the test suite passes an empty sequence to exercise the
    missing-NVML path without uninstalling anything. A successful read is cached per name tuple
    (the driver version cannot change while the process runs); failures are retried.
    """
    names = _NVML_SONAMES if sonames is None else tuple(sonames)
    cached = _NVML_VERSION_CACHE.get(names)
    if cached is not None:
        return cached, None
    lib, error = _open(names)
    if lib is None:
        return None, ("the NVIDIA Management Library (libnvidia-ml.so.1) is not available "
                      f"({error})")
    # nvml.h maps nvmlInit to nvmlInit_v2 in every current NVML; try both names.
    init = getattr(lib, "nvmlInit_v2", None)
    if init is None:
        init = getattr(lib, "nvmlInit", None)
    if init is None:
        return None, "the NVIDIA Management Library has no nvmlInit entry point"
    init.restype = ctypes.c_int
    init.argtypes = []
    rc = init()
    if rc != _NVML_SUCCESS:
        return None, f"nvmlInit failed: {_nvml_error_string(lib, rc)}"
    try:
        try:
            query = lib.nvmlSystemGetDriverVersion
        except AttributeError:
            return None, "the NVIDIA Management Library has no nvmlSystemGetDriverVersion entry point"
        query.restype = ctypes.c_int
        query.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        buf = ctypes.create_string_buffer(_NVML_VERSION_BUFFER)
        rc = query(buf, len(buf))
        if rc != _NVML_SUCCESS:
            return None, f"nvmlSystemGetDriverVersion failed: {_nvml_error_string(lib, rc)}"
        version = buf.value.decode("utf-8", "replace")
        if not version:
            return None, "nvmlSystemGetDriverVersion returned an empty version string"
        _NVML_VERSION_CACHE[names] = version
        return version, None
    finally:
        try:
            lib.nvmlShutdown.restype = ctypes.c_int
            lib.nvmlShutdown.argtypes = []
            lib.nvmlShutdown()
        except AttributeError:
            pass


def _attribute(lib, attr: int, device: int) -> int:
    value = ctypes.c_int(0)
    rc = lib.cuDeviceGetAttribute(ctypes.byref(value), attr, device)
    return int(value.value) if rc == _CUDA_SUCCESS else -1


def _device_name(lib, device: int) -> str:
    # cuDeviceGetName writes at most 256 bytes including the terminator.
    buf = ctypes.create_string_buffer(256)
    rc = lib.cuDeviceGetName(buf, len(buf), device)
    return buf.value.decode("utf-8", "replace") if rc == _CUDA_SUCCESS else ""


def _device_uuid(lib, device: int):
    """The device UUID as 8-4-4-4-12 lowercase hex, or None when the driver has no UUID call."""
    fn = getattr(lib, "cuDeviceGetUuid_v2", None)
    if fn is None:
        fn = getattr(lib, "cuDeviceGetUuid", None)
    if fn is None:
        return None
    buf = ctypes.create_string_buffer(16)  # CUuuid: 16 bytes
    if fn(buf, device) != _CUDA_SUCCESS:
        return None
    h = buf.raw.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _cuda_version(value: int) -> str:
    """The "major.minor" form of a CUDA version integer (13030 -> "13.3")."""
    return f"{value // 1000}.{(value % 1000) // 10}"


def devices() -> list[dict]:
    """The CUDA devices visible to this process, queried through the CUDA driver API.

    Each entry has:

    - ``name`` (str): the device name, e.g. ``"NVIDIA GeForce RTX 5080"``;
    - ``uuid`` (str or None): the device UUID as 8-4-4-4-12 lowercase hex, matching the
      ``GPU-<uuid>`` form of nvidia-smi; None when the driver has no UUID entry point;
    - ``compute_capability`` ("major.minor"), ``sm_count`` (int), ``l2_bytes`` (int);
    - ``cuda_driver_version`` (str or None): the CUDA version the installed driver supports,
      from ``cuDriverGetVersion``;
    - ``driver_version``: an alias of ``cuda_driver_version``, kept for compatibility;
    - ``nvidia_driver_version`` (str or None): the installed NVIDIA display driver version from
      NVML, e.g. ``"580.159.03"``;
    - ``nvidia_driver_version_reason`` (str or None): why ``nvidia_driver_version`` is None
      (NVML missing, or a failed NVML call); None when a version was read;
    - ``core_variant`` ("default" or "ada64"): the decoder core variant a handle on this device
      would get -- the device-class table's choice for its compute capability and SM count, or
      the ``ELDPC_DEVICE_CLASS`` value when that variable is set to a valid variant;
    - ``supported`` (True, False or None): whether this build's GPU cores can run on the device
      -- the rule of ``device="auto"``: GPU cores in the build, a driver for their CUDA version, a
      compute capability they carry a SASS or PTX image for -- read from the loaded library's
      build info, or from the build record next to the libraries when none is loaded (no library
      is loaded for it); None when neither is available;
    - ``supported_reason`` (str or None): why ``supported`` is not True; None when it is.

    An empty list means that no CUDA driver or no device is available.
    """
    lib, _error = _driver()
    if lib is None:
        return []
    if lib.cuInit(0) != _CUDA_SUCCESS:
        return []
    count = ctypes.c_int(0)
    if lib.cuDeviceGetCount(ctypes.byref(count)) != _CUDA_SUCCESS or count.value <= 0:
        return []
    driver = ctypes.c_int(0)
    cuda_driver_version = None
    if lib.cuDriverGetVersion(ctypes.byref(driver)) == _CUDA_SUCCESS and driver.value > 0:
        cuda_driver_version = _cuda_version(int(driver.value))
    nvidia_driver_version, nvml_reason = _nvml_driver_version()
    override = os.environ.get("ELDPC_DEVICE_CLASS")
    from . import _check  # here, not at module level: _check imports this module

    rule = _check.support_rule()
    out = []
    for index in range(count.value):
        major = _attribute(lib, _ATTR_COMPUTE_CAPABILITY_MAJOR, index)
        minor = _attribute(lib, _ATTR_COMPUTE_CAPABILITY_MINOR, index)
        sm_count = _attribute(lib, _ATTR_MULTIPROCESSOR_COUNT, index)
        name = _device_name(lib, index)
        supported, supported_reason = rule(index, name, (major, minor), int(driver.value))
        out.append({
            "name": name,
            "uuid": _device_uuid(lib, index),
            "compute_capability": f"{major}.{minor}",
            "sm_count": sm_count,
            "l2_bytes": _attribute(lib, _ATTR_L2_CACHE_SIZE, index),
            "cuda_driver_version": cuda_driver_version,
            "driver_version": cuda_driver_version,
            "nvidia_driver_version": nvidia_driver_version,
            "nvidia_driver_version_reason": nvml_reason,
            "core_variant": effective_core_variant(major, minor, sm_count, override),
            "supported": supported,
            "supported_reason": supported_reason,
        })
    return out
