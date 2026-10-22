# SPDX-License-Identifier: Apache-2.0
"""device="auto": the selection rules, the warning, config files, and the C front's device=auto.

The rules are tested with simulated probe results (no GPU needed): a CPU-only build, no driver, no
device, a driver too old for the GPU cores' CUDA version, and a compute capability the cores carry
no image for (the last cannot be produced on the test machines, whose GPUs are all supported, so
the message is checked here, for the Python mirror and for the C front's header, gpu_arch.h). The
end-to-end tests run on whatever the machine has: with a usable GPU, "auto" must choose it; with
CUDA_VISIBLE_DEVICES empty or in a CPU-only build, the CPU core with a warning.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

import eldpc
from eldpc import _auto
from eldpc._abi import EldpcError, Handle
from eldpc._config import config_string as abi_config_string
from eldpc._matrix import build_problem, canonical_csc
from eldpc_testlib import CUDA_DEVICES, DEM_TEXT, cpu_core, requires_gpu, tiny_problem

TESTS = Path(__file__).resolve().parent
PACKAGE = TESTS.parent
GPU_ARCH_H = PACKAGE / "src" / "port" / "gpu_arch.h"
BUILD_IMAGES = _auto.parse_cuda_arch("sm_89,sm_120,compute_120")
GPU_BUILD_INFO = ("eldpc 0.1.0; front=GNU 13.3.0; host_isa=avx512; core=compiler=GNU 13.3.0; "
                  "flags=-O2 -DNDEBUG -ffp-contract=off -fno-fast-math -O3 -mavx2; build_type=Release; "
                  "isa=avx512; lib=fastbpgpu6g; cuda=NVIDIA 13.3.73; cuda_arch=sm_89,sm_120,compute_120; "
                  "cuda_flags=-O2 -DNDEBUG -Xfatbin=--ident=compute_120 -fmad=false; gpu_sm_budget=0")
CPU_BUILD_INFO = ("eldpc 0.1.0; front=GNU 13.3.0; host_isa=avx512; core=compiler=GNU 13.3.0; "
                  "flags=-O2; build_type=Release; isa=avx512; lib=fastbpcpu6g; threads=1..64")


def gpu_probe(cc=(12, 0), driver_version=13000, **kw):
    args = dict(gpu_cores=True, images=BUILD_IMAGES, cuda_version="13.3.73", device_error=None,
                device_name="NVIDIA Test GPU", compute_capability=cc, driver_version=driver_version)
    args.update(kw)
    return _auto.Probe(**args)


# ---------------------------------------------------------------------------------------------
# the rules, with simulated probes


@cpu_core
def test_cpu_only_build_chooses_the_cpu_core():
    assert _auto.choose(_auto.Probe(gpu_cores=False)) == ("cpu", _auto.CPU_ONLY_REASON)
    assert "this build has no GPU cores (built without a CUDA toolkit)" == _auto.CPU_ONLY_REASON


@cpu_core
def test_no_device_chooses_the_cpu_core_and_says_why():
    device, reason = _auto.choose(gpu_probe(device_error="cuInit failed with CUDA_ERROR_NO_DEVICE (100)"))
    assert device == "cpu"
    assert reason.startswith("no usable CUDA device: cuInit failed") and "CUDA_ERROR_NO_DEVICE" in reason


@cpu_core
def test_old_driver_chooses_the_cpu_core():
    device, reason = _auto.choose(gpu_probe(driver_version=12040))
    assert device == "cpu"
    assert "supports CUDA 12.4" in reason and "CUDA 13.3.73" in reason and "CUDA 13.x" in reason
    # Any 13.x driver is enough (CUDA minor-version compatibility).
    assert _auto.choose(gpu_probe(driver_version=13000)) == ("gpu", None)
    # An unknown driver version skips the check.
    assert _auto.choose(gpu_probe(driver_version=0)) == ("gpu", None)


@pytest.mark.parametrize("cc,expected", [
    ((8, 9), "gpu"), ((12, 0), "gpu"), ((12, 1), "gpu"), ((13, 0), "gpu"),
    ((8, 6), "cpu"), ((8, 0), "cpu"), ((8, 7), "cpu"), ((9, 0), "cpu"), ((10, 0), "cpu"),
    ((11, 0), "cpu"), ((7, 5), "cpu"),
])
@cpu_core
def test_compute_capability_rule(cc, expected):
    device, reason = _auto.choose(gpu_probe(cc=cc))
    assert device == expected
    if expected == "cpu":
        assert f"compute capability {cc[0]}.{cc[1]}" in reason
        assert "sm_89, sm_120" in reason and "compute_120" in reason
        assert "NVIDIA Test GPU" in reason


@cpu_core
def test_unreported_images_skip_the_compute_capability_check():
    assert _auto.choose(gpu_probe(cc=(8, 6), images=())) == ("gpu", None)


@cpu_core
def test_image_list_parsing_and_text():
    assert [(im.name, im.major, im.minor, im.ptx) for im in BUILD_IMAGES] == [
        ("sm_89", 8, 9, False), ("sm_120", 12, 0, False), ("compute_120", 12, 0, True)]
    assert _auto.images_text(BUILD_IMAGES) == (
        "SASS for sm_89, sm_120 and PTX for compute_120 (compute capability 8.9; 12.0 or newer)")
    assert _auto.parse_cuda_arch("sm_90a, bogus, compute_x, sm_8") == (
        _auto.CudaImage("sm_90a", 9, 0, False, True),)
    assert _auto.device_supported(_auto.parse_cuda_arch("sm_90a"), 9, 0)
    assert not _auto.device_supported(_auto.parse_cuda_arch("sm_90a"), 9, 1)
    assert _auto.build_info_field(GPU_BUILD_INFO, "cuda") == "NVIDIA 13.3.73"
    assert _auto.build_info_field(GPU_BUILD_INFO, "cuda_arch") == "sm_89,sm_120,compute_120"
    assert _auto.build_info_field(CPU_BUILD_INFO, "cuda_arch") == ""


# ---------------------------------------------------------------------------------------------
# the probe, with simulated driver libraries


class FakeDriver:
    """A stand-in for the bound libcuda.so.1 of eldpc._devices._driver()."""

    def __init__(self, init_rc=0, count=1, cc=(8, 6), sms=82, version=13000, name=b"Fake GPU"):
        self.init_rc, self.count, self.cc, self.sms = init_rc, count, cc, sms
        self.version, self.name = version, name
        self.calls = []

    def cuInit(self, flags):
        self.calls.append("cuInit")
        return self.init_rc

    def cuDeviceGetCount(self, out):
        out._obj.value = self.count
        return 0

    def cuDeviceGetAttribute(self, out, attr, device):
        out._obj.value = {75: self.cc[0], 76: self.cc[1], 16: self.sms}[attr]
        return 0

    def cuDriverGetVersion(self, out):
        out._obj.value = self.version
        return 0

    def cuDeviceGetName(self, buf, size, device):
        ctypes.memmove(buf, self.name + b"\0", len(self.name) + 1)
        return 0


@cpu_core
def test_probe_without_a_driver():
    reason, name, cc, version = _auto.probe_device0((None, "libcuda.so.1: cannot open shared object file"))
    assert "libcuda.so.1 cannot be loaded" in reason and cc is None
    device, why = _auto.choose(_auto.probe(GPU_BUILD_INFO, driver=(None, "missing")))
    assert device == "cpu" and "no usable CUDA device" in why and "libcuda.so.1" in why


@cpu_core
def test_probe_with_no_visible_device():
    reason, _, cc, _ = _auto.probe_device0((FakeDriver(init_rc=100), None))
    assert reason.startswith("cuInit failed with") and "100" in reason
    assert "sees no CUDA device" in reason and "CUDA_VISIBLE_DEVICES" in reason and cc is None
    reason, _, cc, _ = _auto.probe_device0((FakeDriver(count=0), None))
    assert "could not query device 0" in reason and cc is None


@cpu_core
def test_probe_of_an_unsupported_device():
    probe = _auto.probe(GPU_BUILD_INFO, driver=(FakeDriver(cc=(8, 6), name=b"NVIDIA GeForce RTX 3090"), None))
    assert probe.gpu_cores and probe.compute_capability == (8, 6) and probe.cuda_version == "13.3.73"
    assert probe.device_name == "NVIDIA GeForce RTX 3090" and probe.driver_version == 13000
    device, reason = _auto.choose(probe)
    assert device == "cpu"
    assert reason == ("device 0 (NVIDIA GeForce RTX 3090) has compute capability 8.6, which the GPU "
                      "cores of this build do not support: they carry SASS for sm_89, sm_120 and PTX "
                      "for compute_120 (compute capability 8.9; 12.0 or newer)")
    ok = _auto.probe(GPU_BUILD_INFO, driver=(FakeDriver(cc=(12, 0)), None))
    assert _auto.choose(ok) == ("gpu", None)


# CUDA's rules for the build's images (SASS sm_89, sm_120; PTX compute_120): SASS runs on devices
# of the same major compute capability with an equal or higher minor one; PTX is JIT-compiled for
# its own or any higher compute capability, never a lower one. So Turing, Ampere, Hopper,
# data-center Blackwell and Thor are out, and every 12.x device is in.
SUPPORT_BY_CC = {(7, 5): False, (8, 0): False, (8, 6): False, (8, 7): False, (8, 9): True,
                 (9, 0): False, (10, 0): False, (10, 3): False, (11, 0): False, (12, 0): True,
                 (12, 1): True}


@cpu_core
def test_device_support_rule_by_compute_capability():
    for cc, ok in SUPPORT_BY_CC.items():
        supported, reason = _auto.device_support(gpu_probe(cc=cc), index=1)
        assert supported is ok, cc
        if ok:
            assert reason is None
        else:
            assert reason == _auto.unsupported_reason("NVIDIA Test GPU", cc[0], cc[1], BUILD_IMAGES, 1)
            assert reason.startswith(f"device 1 (NVIDIA Test GPU) has compute capability {cc[0]}.{cc[1]}")
    supported, reason = _auto.device_support(gpu_probe(cc=(12, 0), driver_version=12080))
    assert supported is False and "the NVIDIA driver supports CUDA 12.8" in reason
    assert _auto.device_support(_auto.Probe(gpu_cores=False)) == (False, _auto.CPU_ONLY_REASON)


class FakeDriverWithL2(FakeDriver):
    def cuDeviceGetAttribute(self, out, attr, device):
        if attr == 38:  # the L2 cache size, which devices() reports
            out._obj.value = 6 * 1024 * 1024
            return 0
        return super().cuDeviceGetAttribute(out, attr, device)


@cpu_core
def test_devices_reports_whether_the_build_supports_each_device(monkeypatch):
    """eldpc.devices()'s supported / supported_reason, with a simulated RTX 3090 (8.6) and RTX 5080
    (12.0) and simulated builds: the signal a user sees before the first create."""
    from eldpc import _check, _devices
    fake = FakeDriverWithL2(cc=(8, 6), sms=82, name=b"NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(_devices, "_driver", lambda: (fake, None))
    monkeypatch.setattr(_check, "known_build_info", lambda: GPU_BUILD_INFO)
    [dev] = eldpc.devices()
    assert dev["compute_capability"] == "8.6" and dev["supported"] is False
    assert dev["supported_reason"] == _auto.unsupported_reason("NVIDIA GeForce RTX 3090", 8, 6,
                                                               BUILD_IMAGES)
    fake.cc, fake.name = (12, 0), b"NVIDIA GeForce RTX 5080"
    [dev] = eldpc.devices()
    assert dev["supported"] is True and dev["supported_reason"] is None
    monkeypatch.setattr(_check, "known_build_info", lambda: CPU_BUILD_INFO)
    [dev] = eldpc.devices()
    assert dev["supported"] is False and "no GPU cores" in dev["supported_reason"]
    monkeypatch.setattr(_check, "known_build_info", lambda: None)
    [dev] = eldpc.devices()
    assert dev["supported"] is None and dev["supported_reason"].startswith("unknown")


@cpu_core
def test_the_build_record_describes_the_gpu_images(monkeypatch):
    """Without a loaded library, devices() reads the cores' images from the build record."""
    from eldpc import _abi, _check
    monkeypatch.setattr(_abi, "_LIBRARY", None)
    monkeypatch.setattr(_check, "read_build_record", lambda path=None: {
        "schema": "eldpc.build.v1", "gpu_cores": True, "cuda_choice": "auto",
        "cuda": "NVIDIA 13.3.73", "cuda_arch": "sm_89,sm_120,compute_120"})
    info = _check.known_build_info()
    assert _auto.build_info_field(info, "cuda_arch") == "sm_89,sm_120,compute_120"
    rule = _check.support_rule()
    assert rule(0, "NVIDIA GeForce RTX 3090", (8, 6), 13000)[0] is False
    assert rule(0, "NVIDIA GeForce RTX 4090", (8, 9), 13000) == (True, None)
    monkeypatch.setattr(_check, "read_build_record", lambda path=None: {
        "schema": "eldpc.build.v1", "gpu_cores": False, "cuda_choice": "auto"})
    assert _check.known_build_info() == "lib=fastbpcpu6g"
    monkeypatch.setattr(_check, "read_build_record", lambda path=None: None)
    assert _check.known_build_info() is None


@cpu_core
def test_cpu_only_build_makes_no_driver_call():
    driver = FakeDriver()
    probe = _auto.probe(CPU_BUILD_INFO, driver=(driver, None))
    assert probe == _auto.Probe(gpu_cores=False) and driver.calls == []


# ---------------------------------------------------------------------------------------------
# Decoder(device="auto") with a simulated probe: the warning, the reported device, threads


@pytest.fixture
def fresh_warning(monkeypatch):
    monkeypatch.setattr(_auto, "_WARNED", False)


def _eldpc_warnings(record):
    return [w for w in record if "device=\"auto\"" in str(w.message)]


@cpu_core
def test_auto_falls_back_to_the_cpu_core_with_one_warning(monkeypatch, fresh_warning):
    monkeypatch.setattr(_auto, "probe", lambda info, driver=None: gpu_probe(cc=(8, 6)))
    H, O, priors = tiny_problem()
    with pytest.warns(RuntimeWarning, match=r'device="auto" runs on the CPU core: device 0 .*8\.6') as rec:
        dec = eldpc.Decoder(H, O, priors, device="auto", threads=2)
    try:
        assert dec.device == "cpu" and dec.core_variant == "cpu" and dec.threads == 2
        assert "device=cpu" in dec.config_string and "threads=2" in dec.config_string
        assert len(_eldpc_warnings(rec)) == 1
    finally:
        dec.close()
    # Once per process: a second fallback is silent.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dec = eldpc.Decoder.from_dem(DEM_TEXT, device="auto")
        dec.close()
    assert not _eldpc_warnings(rec)
    assert dec.device == "cpu"


@cpu_core
def test_cpu_only_warning_says_how_to_get_the_gpu_cores(monkeypatch, fresh_warning):
    monkeypatch.setattr(_auto, "probe", lambda info, driver=None: _auto.Probe(gpu_cores=False))
    H, O, priors = tiny_problem()
    with pytest.warns(RuntimeWarning) as rec:
        eldpc.Decoder(H, O, priors, device="auto").close()
    text = str(_eldpc_warnings(rec)[0].message)
    assert "this build has no GPU cores (built without a CUDA toolkit)" in text
    assert "https://developer.nvidia.com/cuda-downloads" in text and "CUDACXX" in text


@requires_gpu
def test_auto_chooses_the_gpu_and_drops_threads(monkeypatch, fresh_warning):
    H, O, priors = tiny_problem()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dec = eldpc.Decoder(H, O, priors, device="auto", threads=4)
    try:
        assert not _eldpc_warnings(rec)
        assert dec.device == "gpu" and dec.core_variant in ("default", "ada64")
        assert dec.threads is None and "threads" not in dec.config_string
        assert "device" not in dec.config_string  # the string a GPU-only package would pass
    finally:
        dec.close()


@cpu_core
def test_auto_validation():
    H, O, priors = tiny_problem()
    for kwargs in ({"device": "AUTO"}, {"device": "auto", "threads": 0}, {"device": "auto", "threads": 65},
                   {"device": "auto", "threads": 2.5}):
        with pytest.raises(ValueError):
            eldpc.Decoder(H, O, priors, **kwargs)
    with pytest.raises(ValueError, match='"gpu", "cpu" or "auto"'):
        eldpc.Decoder(H, O, priors, device="tpu")
    with pytest.raises(ValueError, match="conflicts"):
        eldpc.Decoder(H, O, priors, config={"device": "auto"}, device="cpu")
    with pytest.raises(ValueError, match="conflicts"):
        eldpc.Decoder(H, O, priors, config={"device": "cpu"}, device="auto")
    with pytest.raises(ValueError, match="threads"):
        eldpc.Decoder(H, O, priors, config={"threads": 2})  # device="gpu" still rejects threads


@cpu_core
def test_auto_from_a_config_dict_and_a_json_file(monkeypatch, fresh_warning, tmp_path):
    monkeypatch.setattr(_auto, "probe", lambda info, driver=None: _auto.Probe(gpu_cores=False))
    H, O, priors = tiny_problem()
    config = dict(eldpc.presets()["a03125c3"], device="auto", threads=2)
    path = tmp_path / "auto.json"
    path.write_text(json.dumps(config))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cfg in (config, str(path)):
            dec = eldpc.Decoder(H, O, priors, config=cfg)
            try:
                assert dec.device == "cpu" and dec.core_variant == "cpu" and dec.threads == 2
                assert "device" not in dec.config and "threads" not in dec.config
            finally:
                dec.close()


# ---------------------------------------------------------------------------------------------
# the real machine: Python's choice, the C front's device=auto, and the device="gpu" error


def _front_handle(config: str):
    H, O, priors = tiny_problem()
    problem, _keep = build_problem(canonical_csc(H), canonical_csc(O), np.asarray(priors, float), None)
    return Handle(eldpc.load(), config, problem)


@cpu_core
def test_python_and_c_front_choose_the_same_device():
    expected, _reason = _auto.choose(_auto.probe(eldpc.build_info()))
    base = abi_config_string(eldpc.presets()["a03125c3"])
    h = _front_handle(base + ";device=auto;threads=2")
    try:
        assert (h.core_variant == "cpu") == (expected == "cpu"), (h.core_variant, expected)
    finally:
        h.close()
    with pytest.raises(EldpcError, match='device must be "gpu", "cpu" or "auto"'):
        _front_handle(base + ";device=tpu")


@cpu_core
def test_with_no_device_visible_auto_runs_on_the_cpu_core_and_gpu_fails_clearly():
    """With CUDA_VISIBLE_DEVICES empty: auto -> CPU core with one warning that names the reason
    (no device, or a CPU-only build); device="gpu" -> an error that says why and names
    device="cpu"; the C front's device=auto -> "cpu"."""
    code = textwrap.dedent("""
        import warnings, numpy as np, eldpc
        from eldpc._abi import Handle
        from eldpc._matrix import build_problem, canonical_csc
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        O = np.array([[1, 0, 0]], dtype=np.uint8)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            d = eldpc.Decoder(H, O, [0.01, 0.02, 0.01], config="ms30", device="auto")
        print("CHOSE", d.device, d.core_variant)
        print("DECODE", d.decode(np.array([1, 0], dtype=np.uint8)).tolist())
        d.close()
        for w in rec:
            print("WARNING", w.category.__name__, w.message)
        try:
            eldpc.Decoder(H, O, error_rate=0.01, config="ms30", device="gpu")
        except eldpc.EldpcError as exc:
            print("GPUERROR", exc)
        else:
            print("NO ERROR")
        problem, keep = build_problem(canonical_csc(H), canonical_csc(O), np.array([0.01, 0.02, 0.01]), None)
        h = Handle(eldpc.load(), "decoder=bposd;bp_method=ms;schedule=parallel;osd_method=osd0;"
                   "osd_order=0;ms_scaling_factor=0.625;max_iter=30;device=auto", problem)
        print("CAUTO", h.core_variant)
        h.close()
        print("BUILD", "lib=fastbpgpu6g" in eldpc.build_info())
    """)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "CHOSE cpu cpu" in out and "CAUTO cpu" in out
    warnings_seen = [line for line in out.splitlines() if line.startswith("WARNING")]
    assert len(warnings_seen) == 1 and "RuntimeWarning" in warnings_seen[0]
    gpu_error = next(line for line in out.splitlines() if line.startswith(("GPUERROR", "NO ERROR")))
    assert gpu_error.startswith("GPUERROR") and 'device="cpu"' in gpu_error
    if "BUILD True" in out:
        assert "no usable CUDA device" in warnings_seen[0]
        assert "no usable CUDA device" in gpu_error and 'device="auto"' in gpu_error
    else:
        assert _auto.CPU_ONLY_REASON in warnings_seen[0]
        assert "this build has no GPU cores (built without a CUDA toolkit)" in gpu_error
        assert "https://developer.nvidia.com/cuda-downloads" in gpu_error


# ---------------------------------------------------------------------------------------------
# the C front's unsupported-device message (gpu_arch.h), compiled on its own


ARCH_PROGRAM = r"""
#include <cstdio>
#include "gpu_arch.h"
int main() {
    const char *lists[] = {"sm_89,sm_120,compute_120", "sm_90a", "sm_80,sm_86", "compute_75"};
    const int ccs[][2] = {{7, 5}, {8, 0}, {8, 6}, {8, 9}, {9, 0}, {10, 0}, {12, 0}, {12, 1}, {13, 0}};
    for (const char *list : lists) {
        const auto images = eldpc::parse_cuda_arch(list);
        std::printf("TEXT|%s|%s\n", list, eldpc::images_text(images).c_str());
        for (const auto &cc : ccs) {
            std::printf("SUP|%s|%d.%d|%d\n", list, cc[0], cc[1],
                        eldpc::device_supported(images, cc[0], cc[1]) ? 1 : 0);
        }
    }
    const std::string info = "eldpc 0.1.0; core=compiler=GNU; lib=fastbpgpu6g; cuda=NVIDIA 13.3.73; "
                             "cuda_arch=sm_89,sm_120,compute_120; cuda_flags=-Xfatbin=--ident=compute_120";
    std::printf("FIELD|%s|%s\n", eldpc::build_info_field(info, "cuda").c_str(),
                eldpc::build_info_field(info, "cuda_arch").c_str());
    std::printf("MSG|%s\n", eldpc::unsupported_device_message(
        "NVIDIA GeForce RTX 3090", 8, 6, eldpc::parse_cuda_arch("sm_89,sm_120,compute_120")).c_str());
    return 0;
}
"""


def test_c_front_unsupported_device_message(tmp_path):
    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    if not GPU_ARCH_H.is_file():
        pytest.skip("the source tree's src/port/gpu_arch.h is not available")
    src = tmp_path / "arch.cpp"
    src.write_text(ARCH_PROGRAM)
    exe = tmp_path / "arch"
    build = subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{GPU_ARCH_H.parent}",
                            str(src), "-o", str(exe)], capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    for line in run.stdout.splitlines():
        kind, *rest = line.split("|")
        if kind == "TEXT":
            assert rest[1] == _auto.images_text(_auto.parse_cuda_arch(rest[0])), line
        elif kind == "SUP":
            major, minor = (int(x) for x in rest[1].split("."))
            assert int(rest[2]) == int(_auto.device_supported(_auto.parse_cuda_arch(rest[0]), major, minor)), line
        elif kind == "FIELD":
            assert rest == ["NVIDIA 13.3.73", "sm_89,sm_120,compute_120"]
        elif kind == "MSG":
            message = rest[0]
            assert message == (
                "eldpc: device 0 (NVIDIA GeForce RTX 3090) has compute capability 8.6, which the GPU "
                "cores of this build do not support: they carry SASS for sm_89, sm_120 and PTX for "
                "compute_120 (compute capability 8.9; 12.0 or newer); to run on the CPU core, create "
                "the decoder with device=cpu, or with device=auto to use the GPU only when it is usable "
                "(Python: device=\"cpu\" or device=\"auto\"; C: device=cpu or device=auto in the config "
                "string)")
            assert message[len("eldpc: "):].startswith(
                _auto.unsupported_reason("NVIDIA GeForce RTX 3090", 8, 6, BUILD_IMAGES))
    assert "MSG|" in run.stdout and run.stdout.count("SUP|") == 36
