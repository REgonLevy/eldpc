# SPDX-License-Identifier: Apache-2.0
"""The installation check: the build record, the warning at ``import eldpc`` and ``eldpc.check()``.

The warning's rules are tested with simulated build records and device probes (no GPU needed); the
end-to-end tests run on whatever the machine has: a build with GPU cores and a visible device must
say "GPU ready" and not warn at import; with no device visible the check must say "CPU only" and
why; a CPU-only build that found no nvcc must warn exactly once at import when a GPU is visible.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

import pytest

import eldpc
from eldpc import _auto, _check
from eldpc_testlib import CUDA_DEVICES, cpu_core, requires_gpu

AUTO_CPU_ONLY = {"schema": "eldpc.build.v1", "gpu_cores": False, "cuda_choice": "auto"}
DEVICE0 = (None, "NVIDIA Test GPU", (12, 0), 13020)
NO_DEVICE = ("cuInit failed with CUDA_ERROR_NO_DEVICE (100): the driver sees no CUDA device "
             "(check CUDA_VISIBLE_DEVICES)", "", None, 0)
WARNING_START = "eldpc: this build has no GPU cores because nvcc was not found at build time"


def probe_returning(result, calls):
    def probe():
        calls.append(result)
        return result
    return probe


def run_python(code: str, **env_extra):
    env = os.environ.copy()
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          timeout=300)


@cpu_core
def test_import_warning_rules():
    calls = []
    text = _check.import_warning_text(AUTO_CPU_ONLY, probe_returning(DEVICE0, calls))
    assert text.startswith(WARNING_START) and len(calls) == 1
    assert "(NVIDIA Test GPU)" in text and 'device="gpu" fails' in text
    assert _check.README_TOOLKIT in text and "README.md#cuda-13-toolkit" == _check.README_TOOLKIT
    assert "nvcc on PATH or set CUDACXX" in text and 'eldpc.check()' in text
    assert "ELDPC_WITHOUT_CUDA=1" in text
    # no visible device: no warning
    assert _check.import_warning_text(AUTO_CPU_ONLY, probe_returning(NO_DEVICE, [])) is None
    # a build with GPU cores, a CPU-only build chosen on purpose, no or a broken record: no warning
    # and no driver call at all
    for record in ({"gpu_cores": True, "cuda_choice": "auto"},
                   {"gpu_cores": False, "cuda_choice": "explicit"}, None, {}, [], "x"):
        calls = []
        assert _check.import_warning_text(record, probe_returning(DEVICE0, calls)) is None
        assert calls == []


@cpu_core
def test_import_warning_is_shown_once_and_never_raises(monkeypatch):
    monkeypatch.setattr(_check, "_WARNED", False)
    monkeypatch.setattr(_check, "read_build_record", lambda path=None: dict(AUTO_CPU_ONLY))
    monkeypatch.setattr(_auto, "probe_device0", lambda driver=None: DEVICE0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _check.warn_at_import()
        _check.warn_at_import()
    ours = [w for w in rec if str(w.message).startswith(WARNING_START)]
    assert len(ours) == 1 and ours[0].category is RuntimeWarning
    # warnings turned into errors, or a probe that raises: the import must still work
    monkeypatch.setattr(_check, "_WARNED", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check.warn_at_import()

    def broken(driver=None):
        raise OSError("a broken driver")
    monkeypatch.setattr(_check, "_WARNED", False)
    monkeypatch.setattr(_auto, "probe_device0", broken)
    _check.warn_at_import()


@cpu_core
def test_build_record_reader(tmp_path):
    p = tmp_path / "build.json"
    assert _check.read_build_record(p) is None
    p.write_text("{not json", encoding="utf-8")
    assert _check.read_build_record(p) is None
    p.write_text("[1, 2]", encoding="utf-8")
    assert _check.read_build_record(p) is None
    p.write_text(json.dumps(AUTO_CPU_ONLY), encoding="utf-8")
    assert _check.read_build_record(p) == AUTO_CPU_ONLY
    assert _check.cpu_only_without_nvcc(_check.read_build_record(p))


@cpu_core
def test_installed_build_record_matches_the_library():
    record = _check.read_build_record()
    if record is None:
        pytest.skip("no build record next to the package's libraries (a source-tree run)")
    assert record["schema"] == "eldpc.build.v1"
    assert record["cuda_choice"] in ("auto", "explicit")
    assert record["gpu_cores"] is ("lib=fastbpgpu6g" in eldpc.build_info())


@requires_gpu
def test_gpu_build_says_gpu_ready_and_does_not_warn_at_import():
    proc = run_python("import eldpc\nprint('RETURNED', eldpc.check())\n")
    assert proc.returncode == 0, proc.stderr
    assert WARNING_START not in proc.stderr and "RuntimeWarning" not in proc.stderr, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[-1] == "RETURNED True", proc.stdout
    assert lines[-2].startswith('GPU ready: device="gpu" (the default) decodes on device 0'), proc.stdout
    text = proc.stdout
    assert "GPU cores:   yes, built with the CUDA 13." in text and "(build_info: cuda=NVIDIA 13." in text
    assert "SASS for sm_89, sm_120 and PTX for compute_120" in text
    assert "libeldpc_core_avx2.so" in text and "libeldpc_cpu_avx512.so" in text
    assert f"host ISA:    {eldpc.isa()}" in text
    assert f"  0: {CUDA_DEVICES[0]['name']}, compute capability" in text
    assert "test decode: GPU core ok; CPU core ok" in text


@cpu_core
def test_no_visible_device_says_cpu_only_and_why():
    proc = run_python("import eldpc\nprint('RETURNED', eldpc.check())\n", CUDA_VISIBLE_DEVICES="")
    assert proc.returncode == 0, proc.stderr
    assert WARNING_START not in proc.stderr, proc.stderr     # no device: nothing to warn about
    lines = proc.stdout.splitlines()
    assert lines[-1] == "RETURNED False", proc.stdout
    last = lines[-2]
    assert last.startswith("CPU only: "), proc.stdout
    assert 'device="gpu" fails; device="cpu" and device="auto" run on the CPU core.' in last
    if "lib=fastbpgpu6g" in eldpc.build_info():
        # With the driver installed the devices are hidden (cuInit fails); with no driver at all
        # (tools/cpu_no_gpu_tests.py --no-driver) libcuda.so.1 cannot be loaded.
        assert ("no usable CUDA device: cuInit failed" in last
                or "no usable CUDA device: the NVIDIA driver library libcuda.so.1 cannot be loaded"
                in last), proc.stdout
    else:
        assert "this build has no GPU cores" in last and "no CUDA device is visible: " in last
        assert last.endswith(f"To build the GPU cores, see {_check.GPU_CORES_HINT}."), proc.stdout
    assert "environment: CUDA_VISIBLE_DEVICES=''" in proc.stdout
    assert "devices:     none: " in proc.stdout
    assert "test decode: CPU core ok" in proc.stdout


def _cpu_only_build_without_nvcc() -> bool:
    return _check.cpu_only_without_nvcc(_check.read_build_record())


@cpu_core
@pytest.mark.skipif(not _cpu_only_build_without_nvcc(),
                    reason="the installed package is not a CPU-only build that found no nvcc")
def test_cpu_only_build_without_nvcc_warns_once_at_import_when_a_gpu_is_visible():
    error, name, _, _ = _auto.probe_device0()
    if error is not None:
        pytest.skip(f"no CUDA device is visible: {error}")
    proc = run_python("import eldpc\nimport importlib\nimportlib.reload(eldpc)\n"
                      "print('RETURNED', eldpc.check())\n")
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.count(WARNING_START) == 1, proc.stderr
    assert "RuntimeWarning" in proc.stderr and name in proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[-1] == "RETURNED False"
    assert lines[-2] == (f'CPU only: {_check.NO_NVCC_REASON}. device="gpu" fails; device="cpu" and '
                         f'device="auto" run on the CPU core. To build the GPU cores, see '
                         f'{_check.GPU_CORES_HINT}.'), proc.stdout
    assert f"GPU cores:   no: {_check.NO_NVCC_REASON}" in proc.stdout


@cpu_core
def test_check_on_an_unsupported_gpu_says_cpu_only_and_names_the_compute_capability(monkeypatch):
    """A simulated RTX 3090 (compute capability 8.6) on a build with GPU cores: the device line
    says the build does not support it, and the last line says why and where the supported GPUs
    are listed; no GPU decode is attempted."""
    import io

    from eldpc import _devices
    if "lib=fastbpgpu6g" not in eldpc.build_info():
        pytest.skip("a CPU-only build has no GPU cores to check a device against")
    from test_device_auto import FakeDriverWithL2

    fake = FakeDriverWithL2(cc=(8, 6), sms=82, name=b"NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(_devices, "_driver", lambda: (fake, None))
    out = io.StringIO()
    assert eldpc.check(file=out) is False
    lines = out.getvalue().splitlines()
    assert any(line.startswith("  0: NVIDIA GeForce RTX 3090, compute capability 8.6")
               and line.endswith("supported by this build: NO") for line in lines), lines
    assert "test decode: CPU core ok" in lines
    assert lines[-1].startswith("CPU only: device 0 (NVIDIA GeForce RTX 3090) has compute "
                                "capability 8.6, which the GPU cores of this build do not support")
    assert lines[-1].endswith(f"The GPUs this build supports: {_check.README_GPUS}."), lines[-1]
