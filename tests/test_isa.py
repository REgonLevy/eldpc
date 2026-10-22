# SPDX-License-Identifier: Apache-2.0
"""The C front's host-ISA choice, and forced-ISA runs in fresh processes."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import eldpc
from eldpc_testlib import requires_gpu, tiny_problem

DECODE_SCRIPT = textwrap.dedent("""
    import hashlib
    import numpy as np
    import eldpc
    H = np.array([[1, 1, 0, 0, 1], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=np.uint8)
    O = np.array([[0, 1, 0, 1, 1]], dtype=np.uint8)
    priors = np.array([0.05, 0.10, 0.02, 0.20, 0.01])
    syndromes = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=np.uint8)
    dec = eldpc.Decoder(H, O, priors, config="a03125c3")
    corr, obs, st = dec.decode_batch(syndromes)
    print(eldpc.isa())
    print(hashlib.sha256(corr.tobytes() + obs.tobytes() + st.astype("<u4").tobytes()).hexdigest())
    dec.close()
""")

ERROR_SCRIPT = textwrap.dedent("""
    import eldpc
    for name in ("isa", "build_info"):
        try:
            value = getattr(eldpc, name)()
        except eldpc.EldpcError as exc:
            print(name, "EldpcError:", exc)
        else:
            print(name, "NO ERROR", repr(value))
    try:
        eldpc.Decoder.from_dem("error(0.1) D0")
    except eldpc.EldpcError as exc:
        print("EldpcError:", exc)
    else:
        print("NO ERROR")
""")

LOAD_TWICE_SCRIPT = textwrap.dedent("""
    import eldpc
    eldpc.load()
    try:
        eldpc.load(isa="avx2")
    except eldpc.EldpcError as exc:
        print("EldpcError:", exc)
    else:
        print("NO ERROR")
""")


def device_decode_hash() -> str:
    H, O, priors = tiny_problem()
    syndromes = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=np.uint8)
    dec = eldpc.Decoder(H, O, priors, config="a03125c3")
    try:
        corr, obs, st = dec.decode_batch(syndromes)
    finally:
        dec.close()
    return hashlib.sha256(corr.tobytes() + obs.tobytes() + st.astype("<u4").tobytes()).hexdigest()


def run_python(code: str, extra_env: dict | None = None, timeout: float = 600) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("ELDPC_ISA", None)
    package_parent = str(Path(eldpc.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout)


def cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def cpu_has_avx512() -> bool:
    return {"avx2", "avx512f", "avx512vl", "avx512dq", "avx512bw"} <= cpu_flags()


def test_auto_isa_follows_the_cpu():
    proc = run_python("import eldpc; print(eldpc.isa())")
    if proc.returncode != 0 and "EldpcError" in proc.stderr:
        pytest.skip("no decoder core could be loaded (the CPU or the installed libraries are not usable here)")
    assert proc.returncode == 0, proc.stderr
    expected = "avx512" if cpu_has_avx512() else "avx2"
    assert proc.stdout.strip() == expected


@requires_gpu
def test_forced_avx2_gives_the_same_results():
    proc = run_python(DECODE_SCRIPT, {"ELDPC_ISA": "avx2"})
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.split()
    assert lines[0] == "avx2"
    assert lines[1] == device_decode_hash()


@requires_gpu
@pytest.mark.skipif(not cpu_has_avx512(), reason="the CPU does not support the AVX-512 core's instruction set")
def test_forced_avx512_gives_the_same_results():
    proc = run_python(DECODE_SCRIPT, {"ELDPC_ISA": "avx512"})
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.split()
    assert lines[0] == "avx512"
    assert lines[1] == device_decode_hash()


def test_unknown_eldpc_isa_is_a_clear_error():
    """isa(), build_info() and a create all raise the front's load error, which names the
    variable, the value and what this CPU and OS support."""
    proc = run_python(ERROR_SCRIPT, {"ELDPC_ISA": "sse4"})
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0].startswith('isa EldpcError: eldpc: ELDPC_ISA must be avx2 or avx512 (got "sse4")')
    assert lines[1].startswith('build_info EldpcError: eldpc: ELDPC_ISA must be avx2 or avx512')
    assert "this CPU and OS support" in lines[0]
    # What to do ends the message: on an AVX-512 CPU it names the automatic choice in a closed
    # clause, not as a trailing word after a comma.
    if cpu_has_avx512():
        assert lines[0].endswith("; unset ELDPC_ISA to let eldpc choose (the automatic choice is avx512)")
    else:
        assert lines[0].endswith("this CPU and OS support the avx2 core only; unset ELDPC_ISA (or set it "
                                 "to avx2) to use it")
    assert not lines[0].endswith(", avx512")
    assert lines[2].startswith("EldpcError:") and "ELDPC_ISA" in lines[2]
    assert "NO ERROR" not in proc.stdout


@pytest.mark.skipif(cpu_has_avx512(), reason="this CPU supports AVX-512, so forcing it cannot fail")
def test_forced_avx512_on_a_cpu_without_it_is_a_clear_error():
    proc = run_python(ERROR_SCRIPT, {"ELDPC_ISA": "avx512"})
    lines = proc.stdout.splitlines()
    assert lines[0].startswith("isa EldpcError: eldpc: ELDPC_ISA=avx512 was requested")
    assert "this CPU and OS support the avx2 core only" in lines[0]
    assert lines[1].startswith("build_info EldpcError:")
    assert "EldpcError:" in lines[2] and "avx512" in lines[2]
    assert "NO ERROR" not in proc.stdout


def test_check_with_an_unusable_eldpc_isa_names_it_and_exits_nonzero():
    """eldpc.check() under a bad ELDPC_ISA: the ISA error is the reason on the ISA line and in the
    last line (NOT READY), the GPU-cores line does not claim a build without a CUDA toolkit, and
    the one-line command exits non-zero."""
    proc = run_python("import eldpc; eldpc.check()", {"ELDPC_ISA": "bogus"})
    assert proc.returncode != 0, proc.stdout
    lines = proc.stdout.splitlines()
    isa_line = next(line for line in lines if line.startswith("host ISA:"))
    assert isa_line.startswith('host ISA:    none: eldpc: ELDPC_ISA must be avx2 or avx512 (got "bogus")')
    gpu_line = next(line for line in lines if line.startswith("GPU cores:"))
    assert "built without a CUDA toolkit" not in gpu_line
    # From the build record next to the libraries (none in a source-tree run): what was built.
    assert gpu_line.startswith(("GPU cores:   built (build record:", "GPU cores:   no: this build has no GPU cores",
                                "GPU cores:   unknown: no decoder core could be loaded")), gpu_line
    assert "test decode: not run" in proc.stdout
    assert lines[-1].startswith("NOT READY: no decoder core could be loaded: eldpc: ELDPC_ISA")
    assert "EldpcError: NOT READY" in proc.stderr


def test_forced_isa_reports_build_info():
    proc = run_python("import eldpc; print(eldpc.isa()); print(eldpc.build_info())",
                      {"ELDPC_ISA": "avx2"})
    if proc.returncode != 0 and "EldpcError" in proc.stderr:
        pytest.skip("no decoder core could be loaded (the CPU or the installed libraries are not usable here)")
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0] == "avx2"
    assert "host_isa=avx2" in lines[1] and "core=" in lines[1]
    assert "lib=fastbpgpu6g" in lines[1] or "lib=fastbpcpu6g" in lines[1]


@requires_gpu
def test_isa_cannot_change_after_load():
    proc = run_python(LOAD_TWICE_SCRIPT)
    assert proc.returncode == 0, proc.stderr
    assert "EldpcError: the eldpc library is already loaded" in proc.stdout


@requires_gpu
def test_explicit_library_path_finds_its_cores(tmp_path):
    import shutil

    libs = Path(eldpc.__file__).resolve().parent / "_libs"
    for name in ("libeldpc.so", "libeldpc_core_avx2.so", "libeldpc_core_avx512.so",
                 "libeldpc_core_avx2_ada64.so", "libeldpc_core_avx512_ada64.so"):
        shutil.copy2(libs / name, tmp_path / name)
    code = (f"import eldpc\n"
            f"eldpc.load(path={str(tmp_path / 'libeldpc.so')!r})\n"
            f"print(eldpc.isa())\n")
    proc = run_python(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() in ("avx2", "avx512")


@requires_gpu
def test_loading_a_non_eldpc_library_is_a_clear_error(tmp_path):
    import shutil

    core = Path(eldpc.__file__).resolve().parent / "_libs" / "libeldpc_core_avx2.so"
    shutil.copy2(core, tmp_path / "not_eldpc.so")
    proc = run_python("import eldpc\n"
                      f"try:\n"
                      f"    eldpc.load(path={str(tmp_path / 'not_eldpc.so')!r})\n"
                      f"except eldpc.EldpcError as exc:\n"
                      f"    print('EldpcError:', exc)\n")
    assert proc.returncode == 0, proc.stderr
    assert "not an eldpc library" in proc.stdout


def test_missing_library_path_is_a_clear_error():
    proc = run_python("import eldpc\n"
                      "try:\n"
                      "    eldpc.load(path='/nonexistent/libeldpc.so')\n"
                      "except eldpc.EldpcError as exc:\n"
                      "    print('EldpcError:', exc)\n")
    assert proc.returncode == 0, proc.stderr
    assert "EldpcError: cannot load the eldpc library" in proc.stdout
