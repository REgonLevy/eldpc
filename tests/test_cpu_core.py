# SPDX-License-Identifier: Apache-2.0
"""The optional CPU core: device="cpu", threads, validation, and bit identity.

These tests need no CUDA device (the ``cpu_core`` marker) and always run on the CPU core, with or
without a GPU visible; the comparison with the GPU core is in test_cpu_gpu_agreement.py. The
comparison with a separately built reference (``ELDPC_REFERENCE_LDPC_LIB``) is not marked: it
skips when that library is not given.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import eldpc
from eldpc._config import config_string as abi_config_string
from eldpc_testlib import (DEM_TEXT, cpu_core, demo_problem, random_problem, sampled_syndromes,
                           tiny_problem)

TESTS = Path(__file__).resolve().parent
PACKAGE = TESTS.parent
REFERENCE_LDPC = Path(os.environ.get("ELDPC_REFERENCE_LDPC_LIB", "/nonexistent"))

# The packaged reference decoder of the verification suite (ldpc 2.4.1 with the determinism
# patch), built on first use into a session cache.
_verify_path = str(PACKAGE / "verify")
if _verify_path not in sys.path:
    sys.path.insert(0, _verify_path)
from eldpc_verify.abi import AbiDecoder  # noqa: E402
from eldpc_verify.reference import build_reference  # noqa: E402


@pytest.fixture(scope="session")
def packaged_reference(tmp_path_factory):
    return build_reference(tmp_path_factory.mktemp("cpu-core-reference"), jobs=2)


def cpu_has_avx512() -> bool:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    for line in text.splitlines():
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            return {"avx2", "avx512f", "avx512vl", "avx512dq", "avx512bw"} <= flags
    return False


def run_python(code: str, extra_env: dict | None = None, timeout: float = 300) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("ELDPC_ISA", None)
    env["PYTHONPATH"] = str(PACKAGE) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          timeout=timeout)


# ---------------------------------------------------------------------------------------------
# device and threads: argument validation (ValueError, before any native call)


@pytest.mark.parametrize("kwargs", [
    {"device": "tpu"},
    {"device": None},
    {"device": 3},
    {"device": "CPU"},
    {"device": "gpu", "threads": 2},
    {"device": "gpu", "threads": 1},
    {"device": "cpu", "threads": 0},
    {"device": "cpu", "threads": 65},
    {"device": "cpu", "threads": -1},
    {"device": "cpu", "threads": 2.5},
    {"device": "cpu", "threads": True},
    {"device": "cpu", "threads": "4"},
])
@cpu_core
def test_invalid_device_or_threads_raise_value_error(kwargs):
    H, O, priors = tiny_problem()
    with pytest.raises(ValueError):
        eldpc.Decoder(H, O, priors, **kwargs)
    with pytest.raises(ValueError):
        eldpc.Decoder.from_dem(DEM_TEXT, **kwargs)


@pytest.mark.parametrize("threads", [None, 1, 4])
@cpu_core
def test_valid_cpu_threads(threads):
    H, O, priors = tiny_problem()
    dec = eldpc.Decoder(H, O, priors, device="cpu", threads=threads)
    try:
        assert dec.device == "cpu" and dec.core_variant == "cpu"
        assert dec.threads == threads
        syndrome = sampled_syndromes(H, priors, 1, seed=3)[0]
        assert dec.decode(syndrome).shape == (H.shape[1],)
    finally:
        dec.close()


@cpu_core
def test_device_argument_errors_precede_any_native_call(monkeypatch):
    """A bad argument raises ValueError without loading or calling the native library."""
    H, O, priors = tiny_problem()

    def boom(*args, **kwargs):
        raise AssertionError("the native library was loaded before the argument was validated")

    monkeypatch.setattr(eldpc._abi, "load", boom)
    with pytest.raises(ValueError):
        eldpc.Decoder(H, O, priors, device="tpu")
    with pytest.raises(ValueError):
        eldpc.Decoder(H, O, priors, device="gpu", threads=2)
    with pytest.raises(ValueError):
        eldpc.Decoder.from_dem(DEM_TEXT, device="gpu", threads=4)
    with pytest.raises(ValueError):
        eldpc.Decoder(H, O, priors, config={"threads": 2})


@cpu_core
def test_config_may_not_conflict_with_the_device_arguments():
    H, O, priors = tiny_problem()
    with pytest.raises(ValueError, match="conflicts"):
        eldpc.Decoder(H, O, priors, config={"device": "gpu"}, device="cpu")
    with pytest.raises(ValueError, match="conflicts"):
        eldpc.Decoder(H, O, priors, config={"device": "cpu", "threads": 4}, device="cpu",
                      threads=2)
    # A config that only carries device=cpu is accepted and selects the CPU core.
    dec = eldpc.Decoder(H, O, priors, config={"device": "cpu", **eldpc.presets()["a03125c3"]})
    try:
        assert dec.device == "cpu" and dec.core_variant == "cpu"
        assert "device=cpu" in dec.config_string
    finally:
        dec.close()


@cpu_core
def test_an_abi_config_string_may_carry_device_and_threads():
    """The C front's string ("device=cpu;threads=2;...") works in Python too: threads arrives as
    the text "2" and is taken as the integer 2 (the benchmark's Python measurements pass the C
    driver's config string unchanged)."""
    H, O, priors = tiny_problem()
    text = abi_config_string(dict(eldpc.presets()["ms30"], device="cpu", threads=2))
    with eldpc.Decoder(H, O, priors, config=text) as dec, \
            eldpc.Decoder(H, O, priors, config="ms30", device="cpu", threads=2) as ref:
        assert dec.device == "cpu" and dec.core_variant == "cpu" and dec.threads == 2
        assert dec.config_string == ref.config_string == text
        for syndrome in ([1, 0, 0], [0, 1, 1], [1, 1, 0]):
            got, want = dec.decode_full(syndrome), ref.decode_full(syndrome)
            assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
            assert int(got[2]) == int(want[2])
    with pytest.raises(ValueError, match="threads"):
        eldpc.Decoder(H, O, priors, config=text.replace("threads=2", "threads=two"))
    with pytest.raises(ValueError, match="threads"):
        eldpc.Decoder(H, O, priors, config=text.replace("threads=2", "threads=0"))


# ---------------------------------------------------------------------------------------------
# the CPU handle's attributes and config string


@cpu_core
def test_cpu_decoder_attributes_and_config_string():
    H, O, priors = tiny_problem()
    dec = eldpc.Decoder(H, O, priors, config="ms30", device="cpu", threads=4)
    try:
        assert dec.device == "cpu"
        assert dec.core_variant == "cpu"
        assert dec.threads == 4
        assert "device=cpu" in dec.config_string
        assert "threads=4" in dec.config_string
        # The resolved config is the preset; device and threads are constructor arguments.
        assert dec.config == eldpc.presets()["ms30"]
        # The config string is what a C user would pass to eldpc_create for the same handle.
        assert dec.config_string == abi_config_string(
            dict(eldpc.presets()["ms30"], device="cpu", threads=4))
    finally:
        dec.close()

    default = eldpc.Decoder(H, O, priors, config="ms30", device="cpu")
    try:
        assert default.threads is None
        assert default.config_string == abi_config_string(dict(eldpc.presets()["ms30"], device="cpu"))
    finally:
        default.close()


# ---------------------------------------------------------------------------------------------
# thread counts and CPU lists: identical outputs


@cpu_core
def test_threads_1_2_and_4_give_the_same_outputs():
    H, O, priors = random_problem(12, 30, 3, seed=21, column_weight=3)
    syndromes = sampled_syndromes(H, priors, 40, seed=22)
    results = []
    for threads in (1, 2, 4):
        dec = eldpc.Decoder(H, O, priors, config="a03125c3", device="cpu", threads=threads)
        try:
            batch = dec.decode_batch(syndromes)
            # single-shot decodes must equal the batch, row by row
            for i in range(4):
                corr, obs, status = dec.decode_full(syndromes[i])
                assert np.array_equal(corr, batch[0][i])
                assert np.array_equal(obs, batch[1][i])
                assert int(status) == int(batch[2][i])
            results.append(batch)
        finally:
            dec.close()
    for other in results[1:]:
        assert np.array_equal(results[0][0], other[0])
        assert np.array_equal(results[0][1], other[1])
        assert np.array_equal(results[0][2], other[2])


@cpu_core
def test_an_explicit_cpus_list_gives_the_same_outputs():
    """threads=2 pinned to the two lowest CPUs of this process equals threads=2 without cpus."""
    allowed = sorted(os.sched_getaffinity(0))
    assert len(allowed) >= 2, "the test needs two CPUs of the process's affinity mask"
    cpus = f"{allowed[0]},{allowed[1]}"
    H, O, priors = random_problem(10, 24, 2, seed=23, column_weight=3)
    syndromes = sampled_syndromes(H, priors, 30, seed=24)
    pinned_config = dict(eldpc.presets()["a03125c3"], cpus=cpus)
    pinned = eldpc.Decoder(H, O, priors, config=pinned_config, device="cpu", threads=2)
    plain = eldpc.Decoder(H, O, priors, config="a03125c3", device="cpu", threads=2)
    try:
        assert pinned.config_string.endswith(f"threads=2") and f"cpus={cpus}" in pinned.config_string
        a, b = pinned.decode_batch(syndromes), plain.decode_batch(syndromes)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
    finally:
        pinned.close()
        plain.close()


@cpu_core
def test_cpu_single_equals_batch_and_arrays_equal_dem():
    import stim

    dem = stim.DetectorErrorModel(DEM_TEXT)
    H, O, priors = eldpc.dem_to_arrays(dem)
    syndromes = sampled_syndromes(H, priors, 25, seed=25)
    from_dem = eldpc.Decoder.from_dem(dem, config="a03125c3", device="cpu", threads=2)
    from_arrays = eldpc.Decoder(H, O, priors, config="a03125c3", device="cpu", threads=4)
    try:
        batch = from_dem.decode_batch(syndromes)
        for i in range(syndromes.shape[0]):
            single = from_dem.decode_full(syndromes[i])
            assert np.array_equal(single[0], batch[0][i])
            assert np.array_equal(single[1], batch[1][i])
            assert int(single[2]) == int(batch[2][i])
        other = from_arrays.decode_batch(syndromes)
        for x, y in zip(batch, other):
            assert np.array_equal(x, y)
    finally:
        from_dem.close()
        from_arrays.close()


@cpu_core
def test_keep_warm_is_a_no_op_on_the_cpu():
    H, O, priors = demo_problem()
    dec = eldpc.Decoder(H, O, priors, config="a03125c3", device="cpu", threads=1)
    syndromes = sampled_syndromes(H, priors, 20, seed=26)
    try:
        before = dec.decode_batch(syndromes)
        assert dec.keep_warm() is None
        after = dec.decode_batch(syndromes)
        for x, y in zip(before, after):
            assert np.array_equal(x, y)
    finally:
        dec.close()


@cpu_core
def test_cpu_soft_output_and_set_priors():
    H, O, priors = demo_problem()
    other = np.linspace(0.005, 0.25, H.shape[1])
    syndromes = sampled_syndromes(H, priors, 20, seed=27)
    dec = eldpc.Decoder(H, O, priors, config="ms30", device="cpu", threads=2)
    reference = eldpc.Decoder(H, O, other, config="ms30", device="cpu", threads=1)
    try:
        original = dec.decode_batch(syndromes)
        changed = reference.decode_batch(syndromes)
        corrected = dec.decode_full(syndromes[0])[0]
        assert np.array_equal(original[0][0], corrected)
        llr = dec.soft_output()
        assert llr.shape == (H.shape[1],) and llr.dtype == np.float64
        assert np.all(np.isfinite(llr))
        dec.set_priors(other)
        after = dec.decode_batch(syndromes)
        for x, y in zip(changed, after):
            assert np.array_equal(x, y)
    finally:
        dec.close()
        reference.close()


# ---------------------------------------------------------------------------------------------
# the host ISA choice: the CPU core of the chosen ISA is the one mapped

ISA_SCRIPT = textwrap.dedent("""
    import re
    import numpy as np
    import eldpc
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    O = np.array([[1, 0, 0]], dtype=np.uint8)
    dec = eldpc.Decoder(H, O, [0.01, 0.02, 0.01], config="ms30", device="cpu", threads=1)
    corr, obs, st = dec.decode_full(np.array([1, 0], dtype=np.uint8))
    print(eldpc.isa())
    maps = sorted(set(re.findall(r"libeldpc_cpu_[a-z0-9_]+\\.so", open("/proc/self/maps").read())))
    print(" ".join(maps))
    print(int(corr[0]), int(corr[1]), int(corr[2]), int(st) > 0)
    dec.close()
""")


@cpu_core
def test_forced_avx2_maps_the_avx2_cpu_core():
    proc = run_python(ISA_SCRIPT, {"ELDPC_ISA": "avx2", "CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0] == "avx2"
    assert lines[1] == "libeldpc_cpu_avx2.so"


@cpu_core
def test_forced_avx512_maps_the_avx512_cpu_core():
    proc = run_python(ISA_SCRIPT, {"ELDPC_ISA": "avx512", "CUDA_VISIBLE_DEVICES": ""})
    if cpu_has_avx512():
        assert proc.returncode == 0, proc.stderr
        lines = proc.stdout.splitlines()
        assert lines[0] == "avx512"
        assert lines[1] == "libeldpc_cpu_avx512.so"
    else:
        # Forcing an ISA the CPU does not support is a load error, not a fallback.
        assert proc.returncode != 0
        assert "avx512" in proc.stderr.lower()


@cpu_core
def test_cpu_decoder_works_with_no_gpu_visible():
    """With CUDA_VISIBLE_DEVICES empty, a CPU handle decodes and a GPU create names device="cpu"."""
    code = textwrap.dedent("""
        import numpy as np, eldpc
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        O = np.array([[1, 0, 0]], dtype=np.uint8)
        d = eldpc.Decoder(H, O, [0.01, 0.02, 0.01], config="ms30", device="cpu", threads=1)
        print(d.decode(np.array([1, 0], dtype=np.uint8)).tolist())
        d.close()
        try:
            eldpc.Decoder(H, O, error_rate=0.01, config="ms30")
        except eldpc.EldpcError as exc:
            print("EldpcError:", exc)
        else:
            print("NO ERROR")
    """)
    proc = run_python(code, {"CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode == 0, proc.stderr
    assert "NO ERROR" not in proc.stdout
    assert 'device="cpu"' in proc.stdout or "device='cpu'" in proc.stdout


# ---------------------------------------------------------------------------------------------
# bit identity against the reference BP+OSD-0


def _shots(H, priors, shots=30, seed=41):
    return sampled_syndromes(H, priors, shots, seed=seed)


def _soft_bits(values):
    """The IEEE-754 bit pattern of a soft-output vector, for a bit-for-bit comparison."""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64)).view(np.uint64)


def _compare_with_reference(reference, dec, syndromes, soft_shots=10):
    """Compare corrections, observables, status words and soft output shot by shot, and batches."""
    ref_soft = []
    for i in range(syndromes.shape[0]):
        r_obs, r_corr, r_status = reference.decode(syndromes[i])
        corr, obs, status = dec.decode_full(syndromes[i])
        assert np.array_equal(corr, r_corr), f"correction differs at shot {i}"
        assert np.array_equal(obs, r_obs), f"observables differ at shot {i}"
        assert int(status) == int(r_status), f"status differs at shot {i}: {status} vs {r_status}"
        if i < soft_shots:
            ref_soft.append(_soft_bits(reference.soft_output()))
            assert np.array_equal(_soft_bits(dec.soft_output()), ref_soft[i]), \
                f"soft output differs at shot {i}"
    b_corr, b_obs, b_status = dec.decode_batch(syndromes)
    r_obs, r_corr, r_status = reference.decode_batch(syndromes)
    assert np.array_equal(b_corr, r_corr)
    assert np.array_equal(b_obs, r_obs)
    assert np.array_equal(b_status, r_status)


@pytest.mark.parametrize("config_name", ["ms30", "a03125c3", "a03125c3.bp"])
@cpu_core
def test_cpu_bit_identity_with_the_packaged_reference(config_name, packaged_reference):
    cfg = abi_config_string(eldpc.presets()[config_name])
    problems = [demo_problem(), random_problem(10, 24, 2, seed=61, column_weight=3)]
    for H, O, priors in problems:
        syndromes = _shots(H, priors)
        reference = AbiDecoder(packaged_reference.path, cfg, H, O, priors)
        dec = eldpc.Decoder(H, O, priors, config=config_name, device="cpu", threads=2)
        try:
            _compare_with_reference(reference, dec, syndromes)
        finally:
            reference.close()
            dec.close()


@cpu_core
def test_cpu_bit_identity_on_a_dem(packaged_reference):
    import stim

    dem = stim.DetectorErrorModel(DEM_TEXT)
    H, O, priors = eldpc.dem_to_arrays(dem)
    syndromes = _shots(H, priors, shots=30, seed=42)
    cfg = abi_config_string(eldpc.presets()["a03125c3"])
    reference = AbiDecoder(packaged_reference.path, cfg, H, O, priors, dem_text=str(dem))
    dec = eldpc.Decoder.from_dem(dem, config="a03125c3", device="cpu", threads=4)
    try:
        _compare_with_reference(reference, dec, syndromes)
    finally:
        reference.close()
        dec.close()


@pytest.mark.skipif(not REFERENCE_LDPC.is_file(),
                    reason="ELDPC_REFERENCE_LDPC_LIB is not set (the optional comparison with the "
                           "reference construction's own build of the reference decoder)")
@pytest.mark.parametrize("config_name", ["a03125c3", "ms30"])
def test_cpu_bit_identity_with_the_reference_construction_build(config_name):
    """The CPU core against the reference construction's own build of the reference decoder
    (``ELDPC_REFERENCE_LDPC_LIB``, as in tests/verify/test_reference.py)."""
    from refabi import RefDecoder, RefLibrary

    H, O, priors = demo_problem()
    syndromes = _shots(H, priors, shots=25, seed=43)
    library = RefLibrary(REFERENCE_LDPC)
    reference = RefDecoder(library, abi_config_string(eldpc.presets()[config_name]), H, O, priors)
    dec = eldpc.Decoder(H, O, priors, config=config_name, device="cpu", threads=2)
    try:
        r_corr, r_obs, r_status = reference.decode_batch(syndromes)
        b_corr, b_obs, b_status = dec.decode_batch(syndromes)
        assert np.array_equal(b_corr, r_corr)
        assert np.array_equal(b_obs, r_obs)
        assert np.array_equal(b_status, r_status)
        for i in range(syndromes.shape[0]):
            corr, obs, status = dec.decode_full(syndromes[i])
            assert np.array_equal(corr, r_corr[i])
            assert np.array_equal(obs, r_obs[i])
            assert int(status) == int(r_status[i])
    finally:
        reference.close()
        dec.close()
