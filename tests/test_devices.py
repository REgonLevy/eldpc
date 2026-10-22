# SPDX-License-Identifier: Apache-2.0
"""eldpc.devices(): the CUDA driver API path, the two driver versions, and no dynamic libcudart."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import eldpc
from eldpc import _devices

from eldpc_testlib import requires_gpu


def test_nvml_missing_reports_none_with_a_reason():
    """The missing-NVML path, exercised without touching the installed NVML.

    ``_nvml_driver_version`` takes the library names to try; an empty sequence is the same code
    path as a host that has no libnvidia-ml.so.1.
    """
    version, reason = _devices._nvml_driver_version(())
    assert version is None
    assert isinstance(reason, str) and "nvidia-ml" in reason


def test_nvml_unloadable_library_reports_none_with_a_reason():
    version, reason = _devices._nvml_driver_version(("libnvidia-ml-eldpc-no-such-library.so.1",))
    assert version is None
    assert isinstance(reason, str) and "nvidia-ml" in reason


def test_nvml_version_cache_hit(monkeypatch):
    """A cached read is returned without opening the library again."""
    monkeypatch.setitem(_devices._NVML_VERSION_CACHE, ("eldpc-test",), "1.2.3")
    assert _devices._nvml_driver_version(("eldpc-test",)) == ("1.2.3", None)


def test_nvml_failure_is_not_cached():
    """A failed read is retried next time instead of being cached as a permanent failure."""
    names = ("libnvidia-ml-eldpc-no-such-library.so.1",)
    assert _devices._nvml_driver_version(names)[0] is None
    assert names not in _devices._NVML_VERSION_CACHE


def test_no_driver_library_gives_no_devices(monkeypatch):
    """Without a loadable libcuda.so.1, devices() is an empty list (as with no GPU)."""
    monkeypatch.setattr(_devices, "_DRIVER_SONAMES", ())
    assert eldpc.devices() == []


def test_devices_respects_empty_cuda_visible_devices():
    """CUDA_VISIBLE_DEVICES='' hides every device from the driver API, as from the runtime."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("ELDPC_ISA", None)
    proc = subprocess.run([sys.executable, "-c",
                           "import eldpc; print('DEVICES=' + repr(eldpc.devices()))"],
                          capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "DEVICES=[]" in proc.stdout, proc.stdout


@requires_gpu
def test_devices_reports_both_driver_versions():
    devices = eldpc.devices()
    assert devices
    for dev in devices:
        for key in ("name", "uuid", "compute_capability", "sm_count", "l2_bytes",
                    "cuda_driver_version", "nvidia_driver_version", "nvidia_driver_version_reason",
                    "driver_version", "core_variant", "supported", "supported_reason"):
            assert key in dev, key
        # requires_gpu means device 0 can run this build's GPU cores: devices() must agree.
        assert devices[0]["supported"] is True and devices[0]["supported_reason"] is None
        # driver_version is the documented alias of the CUDA driver API version.
        assert dev["driver_version"] == dev["cuda_driver_version"]
        assert isinstance(dev["cuda_driver_version"], str)
        assert re.fullmatch(r"\d+\.\d+", dev["cuda_driver_version"]), dev["cuda_driver_version"]
        if dev["nvidia_driver_version"] is None:
            assert isinstance(dev["nvidia_driver_version_reason"], str)
            assert dev["nvidia_driver_version_reason"]
        else:
            assert dev["nvidia_driver_version_reason"] is None
            assert re.fullmatch(r"\d+(\.\d+)+", dev["nvidia_driver_version"]), \
                dev["nvidia_driver_version"]


@requires_gpu
def test_devices_uuid():
    """The UUID is 8-4-4-4-12 lowercase hex and, when nvidia-smi is available, its GPU-<uuid>."""
    devices = eldpc.devices()
    assert devices
    smi = shutil.which("nvidia-smi")
    smi_uuids = []
    if smi is not None:
        proc = subprocess.run([smi, "--query-gpu=uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            smi_uuids = [line.strip().lower().removeprefix("gpu-")
                         for line in proc.stdout.splitlines() if line.strip()]
    for dev in devices:
        uuid = dev["uuid"]
        if uuid is None:
            continue
        assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", uuid), uuid
        if smi_uuids:
            assert uuid in smi_uuids, (uuid, smi_uuids)


@requires_gpu
def test_devices_matches_nvidia_smi():
    """Name, compute capability, driver version and UUID agree with nvidia-smi's view.

    Rows are matched by UUID, so the test works when CUDA_VISIBLE_DEVICES hides some devices
    from this process; devices without a UUID are not compared.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        pytest.skip("nvidia-smi is not available")
    proc = subprocess.run([smi, "--query-gpu=name,driver_version,compute_cap,uuid",
                           "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        pytest.skip("nvidia-smi could not query the devices")
    rows = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4 and parts[3].lower().startswith("gpu-"):
            rows[parts[3].lower()[len("gpu-"):]] = parts
    if not rows:
        pytest.skip("nvidia-smi reported no device UUIDs")
    devices = eldpc.devices()
    assert devices
    compared = 0
    for dev in devices:
        row = rows.get(dev["uuid"])
        if row is None:
            continue
        name, driver, cc, _uuid = row
        assert dev["name"] == name, (dev["name"], name)
        assert dev["compute_capability"] == cc, (dev["compute_capability"], cc)
        if dev["nvidia_driver_version"] is not None:
            assert dev["nvidia_driver_version"] == driver, (dev["nvidia_driver_version"], driver)
        compared += 1
    assert compared, "no device matched an nvidia-smi UUID"


@requires_gpu
def test_devices_missing_nvml_gives_none_versions(monkeypatch):
    """With NVML unavailable, every device still enumerates and says why the version is None."""
    monkeypatch.setattr(_devices, "_NVML_SONAMES", ())
    devices = eldpc.devices()
    assert devices
    for dev in devices:
        assert dev["nvidia_driver_version"] is None
        assert isinstance(dev["nvidia_driver_version_reason"], str)
        assert "nvidia-ml" in dev["nvidia_driver_version_reason"]
        # The CUDA driver API version comes from libcuda, not from NVML, and stays available.
        assert dev["cuda_driver_version"] is not None


def _package_libs() -> Path:
    return Path(eldpc.__file__).resolve().parent / "_libs"


def test_installed_libraries_do_not_need_libcudart():
    """readelf -d shows no libcudart dependency, and ldd's closure contains none either."""
    libs = sorted(_package_libs().glob("*.so"))
    if not libs:
        pytest.skip("no installed native libraries next to the package (source-tree run)")
    readelf = shutil.which("readelf")
    if readelf is None:
        pytest.skip("readelf is not available")
    for lib in libs:
        out = subprocess.run([readelf, "-d", str(lib)], capture_output=True, text=True, check=True).stdout
        assert "libcudart" not in out, f"{lib.name} needs libcudart"
    ldd = shutil.which("ldd")
    if ldd is None:
        return
    for lib in libs:
        proc = subprocess.run([ldd, str(lib)], capture_output=True, text=True)
        assert "libcudart" not in proc.stdout, f"{lib.name} pulls libcudart transitively"


def _fresh_process_maps(code: str) -> dict:
    """Run `code` in a fresh interpreter and split its LIBRARIES=... output lines by prefix."""
    maps = Path("/proc/self/maps")
    if not maps.exists():
        pytest.skip("no /proc/self/maps on this system")
    env = dict(os.environ)
    env.pop("ELDPC_ISA", None)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                          env=env)
    assert proc.returncode == 0, proc.stderr
    out = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition("=")
        if key.startswith("LIBRARIES_"):
            out[key[len("LIBRARIES_"):]] = value
    return out


def test_devices_does_not_load_libcudart():
    """A fresh process that only queries devices() never maps libcudart."""
    code = ("import eldpc\n"
            "eldpc.devices()\n"
            "text = open('/proc/self/maps', encoding='utf-8', errors='replace').read()\n"
            "bad = sorted({line.split()[-1] for line in text.splitlines() if 'libcudart' in line})\n"
            "print('LIBRARIES_CUDART=' + ','.join(bad))\n")
    assert _fresh_process_maps(code) == {"CUDART": ""}


def test_devices_does_not_load_the_native_library():
    """devices() reports core_variant from the Python table, so it must not load libeldpc.so.

    Loading the native library would fix the host ISA at the first devices() call and would make
    enumeration depend on a CPU that can run a core, which docs/BUILD.md promises it does not.
    """
    code = ("import eldpc, eldpc._abi\n"
            "eldpc.devices()\n"
            "print('LIBRARIES_NATIVE_LOADED=' + str(eldpc._abi._LIBRARY is not None))\n")
    assert _fresh_process_maps(code) == {"NATIVE_LOADED": "False"}


@requires_gpu
def test_decode_ignores_a_broken_libcudart_on_the_loader_path(tmp_path):
    """A package that never opens libcudart cannot pick up a broken one from the loader path."""
    (tmp_path / "libcudart.so.13").write_bytes(b"this is not a shared object\n")
    code = ("import numpy as np, eldpc\n"
            "H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)\n"
            "dec = eldpc.Decoder(H, priors=[0.1, 0.1, 0.1])\n"
            "correction = dec.decode([1, 0])\n"
            "dec.close()\n"
            "print('OK', correction.tolist())\n")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(tmp_path) + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.pop("ELDPC_ISA", None)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                          env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("OK "), proc.stdout


@requires_gpu
def test_decode_does_not_load_libcudart():
    """A fresh process that decodes maps the driver (libcuda.so.1) but not libcudart."""
    code = ("import numpy as np, eldpc\n"
            "H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)\n"
            "dec = eldpc.Decoder(H, priors=[0.1, 0.1, 0.1])\n"
            "correction = dec.decode([1, 0])\n"
            "dec.close()\n"
            "assert correction.shape == (3,), correction\n"
            "text = open('/proc/self/maps', encoding='utf-8', errors='replace').read()\n"
            "names = [line.split()[-1] for line in text.splitlines()]\n"
            "print('LIBRARIES_CUDART=' + ','.join(sorted({n for n in names if 'libcudart' in n})))\n"
            # The loader maps the SONAME's target file (libcuda.so.<driver version>), so match the
            # library's base name, not the .so.1 name.
            "print('LIBRARIES_CUDA=' + ','.join(sorted({n for n in names if 'libcuda.so' in n})))\n")
    maps = _fresh_process_maps(code)
    assert maps["CUDART"] == ""
    assert maps["CUDA"], "the static runtime did not map the driver library (libcuda.so.1)"
