# SPDX-License-Identifier: Apache-2.0
"""The per-handle core variant: the device-class table, the override, and both variants together.

The table itself is the C one in ``src/port/core_variant.h``; the exported lookup
``eldpc_core_variant_for_device`` is compared against the Python mirror that ``eldpc.devices()``
uses, over a grid of simulated device properties, so the two copies cannot drift apart. The
handles' variants are then checked against the real device of this machine, the override, and the
lazy loading of the ada64 core.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import eldpc
from eldpc import _variant
from eldpc_testlib import requires_gpu, sampled_syndromes, tiny_problem

# The table cases of the task: the Ada 142-SM class (8.9 / 142), the row boundary (8.9 / 128),
# a smaller Ada part (8.9 / 76), and both Blackwell parts (12.0 / 84 and 12.0 / 170).
TABLE_CASES = (
    ((8, 9, 142), "ada64"),
    ((8, 9, 128), "ada64"),
    ((8, 9, 76), "default"),
    ((12, 0, 84), "default"),
    ((12, 0, 170), "default"),
)

# A grid of device properties around the table's rows, for the C/Python agreement test.
GRID_CC = ((7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (13, 0))
GRID_SMS = (1, 2, 32, 63, 76, 84, 96, 100, 127, 128, 129, 140, 142, 144, 168, 170, 200, 256)


def front():
    """The loaded front library (``eldpc.load()``); its constructor loads the default core."""
    return eldpc.load()


def c_table(cc_major: int, cc_minor: int, sm_count: int) -> str:
    return front().core_variant_for_device(cc_major, cc_minor, sm_count)


def expected_variant_for_this_gpu(devices=None) -> str:
    """The table's choice for device 0, which is the device a handle decodes on."""
    dev = (devices or eldpc.devices())[0]
    major, minor = (int(part) for part in dev["compute_capability"].split("."))
    return _variant.core_variant_for_device(major, minor, int(dev["sm_count"]))


def decode_some(dec, shots=50, seed=11):
    H, O, priors = tiny_problem()
    syndromes = np.asarray(sampled_syndromes(H, priors, shots, seed=seed), dtype=np.uint8)
    return dec.decode_batch(syndromes)


# -- the table ---------------------------------------------------------------


@pytest.mark.parametrize("props,expected", TABLE_CASES)
def test_table_cases(props, expected):
    assert c_table(*props) == expected


def test_table_python_mirror_matches_the_c_table():
    for major, minor in GRID_CC:
        for sms in GRID_SMS:
            assert c_table(major, minor, sms) == _variant.core_variant_for_device(major, minor, sms), \
                (major, minor, sms)


def test_table_invalid_properties_are_default():
    for props in ((-1, 9, 142), (8, -1, 142), (8, 9, 0), (8, 9, -5), (-1, -1, -1)):
        assert c_table(*props) == "default", props
        assert _variant.core_variant_for_device(*props) == "default", props


def test_table_lookup_never_returns_null():
    text = front().lib.eldpc_core_variant_for_device(-1, -1, -1)
    assert text is not None and text.decode() == "default"


def test_table_lookup_extreme_values():
    """C int extremes match no row and give default (no overflow, no crash)."""
    int_max, int_min = 2 ** 31 - 1, -(2 ** 31)
    assert c_table(int_max, int_max, int_max) == "default"
    assert c_table(int_min, int_min, int_min) == "default"
    assert c_table(int_min, 9, int_max) == "default"


# -- the devices() report ----------------------------------------------------


@requires_gpu
def test_devices_report_the_table_variant():
    devices = eldpc.devices()
    assert devices
    for dev in devices:
        assert dev["core_variant"] in ("default", "ada64")
        major, minor = (int(part) for part in dev["compute_capability"].split("."))
        assert dev["core_variant"] == _variant.core_variant_for_device(major, minor, dev["sm_count"])


@requires_gpu
def test_devices_report_a_valid_override(monkeypatch):
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "ada64")
    assert all(dev["core_variant"] == "ada64" for dev in eldpc.devices())
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "default")
    assert all(dev["core_variant"] == "default" for dev in eldpc.devices())


@requires_gpu
def test_devices_ignore_an_invalid_override(monkeypatch):
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "bogus")
    devices = eldpc.devices()
    assert all(dev["core_variant"] in ("default", "ada64") for dev in devices)
    assert devices[0]["core_variant"] == expected_variant_for_this_gpu(devices)


# -- the handle's variant ----------------------------------------------------


@requires_gpu
def test_decoder_variant_follows_the_table(monkeypatch):
    monkeypatch.delenv("ELDPC_DEVICE_CLASS", raising=False)
    H, O, priors = tiny_problem()
    dec = eldpc.Decoder(H, O, priors)
    try:
        assert dec.core_variant == expected_variant_for_this_gpu()
    finally:
        dec.close()


@pytest.mark.parametrize("forced", ["default", "ada64"])
@requires_gpu
def test_override_selects_the_handle_variant(monkeypatch, forced):
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", forced)
    H, O, priors = tiny_problem()
    dec = eldpc.Decoder(H, O, priors)
    try:
        assert dec.core_variant == forced
        corr, obs, st = decode_some(dec)
        assert corr.shape == (50, H.shape[1]) and obs.shape == (50, O.shape[0]) and st.shape == (50,)
    finally:
        dec.close()


@requires_gpu
def test_empty_override_means_the_table(monkeypatch):
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "")
    H, O, priors = tiny_problem()
    dec = eldpc.Decoder(H, O, priors)
    try:
        assert dec.core_variant == expected_variant_for_this_gpu()
    finally:
        dec.close()


@requires_gpu
def test_invalid_override_is_a_clear_create_error(monkeypatch):
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "sse4")
    H, O, priors = tiny_problem()
    with pytest.raises(eldpc.EldpcError) as excinfo:
        eldpc.Decoder(H, O, priors)
    message = str(excinfo.value)
    assert "ELDPC_DEVICE_CLASS" in message
    assert "sse4" in message
    assert "default" in message and "ada64" in message


@pytest.mark.parametrize("first", ["ada64", "default"])
@requires_gpu
def test_both_variants_coexist_and_agree(monkeypatch, first):
    """Two handles of the two variants in one process: both alive, same bits, no interference."""
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", first)
    H, O, priors = tiny_problem()
    a = eldpc.Decoder(H, O, priors)
    monkeypatch.setenv("ELDPC_DEVICE_CLASS",
                       "default" if first == "ada64" else "ada64")
    b = eldpc.Decoder(H, O, priors)
    try:
        assert a.core_variant == first
        assert b.core_variant != first
        ca, oa, sa = decode_some(a)
        cb, ob, sb = decode_some(b)
        assert np.array_equal(ca, cb), "the two variants returned different corrections"
        assert np.array_equal(oa, ob), "the two variants returned different observables"
        assert np.array_equal(sa, sb), "the two variants returned different status words"
    finally:
        a.close()
        b.close()


@requires_gpu
def test_variant_is_fixed_per_handle(monkeypatch):
    """Changing ELDPC_DEVICE_CLASS later does not change an existing handle."""
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "ada64")
    H, O, priors = tiny_problem()
    dec = eldpc.Decoder(H, O, priors)
    try:
        monkeypatch.setenv("ELDPC_DEVICE_CLASS", "default")
        second = eldpc.Decoder(H, O, priors)
        try:
            assert second.core_variant == "default"
            assert dec.core_variant == "ada64"
            ca, _, _ = decode_some(dec)
            cb, _, _ = decode_some(second)
            assert np.array_equal(ca, cb)
        finally:
            second.close()
    finally:
        dec.close()


def test_c_core_variant_of_a_null_handle_is_empty():
    assert front().lib.eldpc_core_variant(None) in (None, b"")


@requires_gpu
def test_decode_does_not_read_the_device_class(monkeypatch):
    """ELDPC_DEVICE_CLASS is read at create only: changing it afterwards cannot affect a handle."""
    H, O, priors = tiny_problem()
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "default")
    dec = eldpc.Decoder(H, O, priors)
    try:
        first = dec.decode_batch(np.asarray(sampled_syndromes(H, priors, 20, seed=3), dtype=np.uint8))
        monkeypatch.setenv("ELDPC_DEVICE_CLASS", "not-a-variant")
        second = dec.decode_batch(np.asarray(sampled_syndromes(H, priors, 20, seed=3), dtype=np.uint8))
        for a, b in zip(first, second):
            assert np.array_equal(a, b)
        assert dec.core_variant == "default"
    finally:
        dec.close()


@requires_gpu
def test_build_info_keeps_reporting_the_default_core(monkeypatch):
    """eldpc_build_info() reports the default core of the ISA, not an ada64 handle's core."""
    H, O, priors = tiny_problem()
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "ada64")
    dec = eldpc.Decoder(H, O, priors)
    try:
        assert dec.core_variant == "ada64"
        info = eldpc.build_info()
        assert "gpu_sm_budget=0" in info, info
        assert "isa=" in info and "lib=fastbpgpu6g" in info
        assert eldpc.isa() in ("avx2", "avx512")
    finally:
        dec.close()


@requires_gpu
def test_core_variant_property_is_read_only(monkeypatch):
    H, O, priors = tiny_problem()
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "default")
    dec = eldpc.Decoder(H, O, priors)
    try:
        with pytest.raises(AttributeError):
            dec.core_variant = "ada64"
    finally:
        dec.close()


@requires_gpu
def test_missing_ada64_core_is_a_clear_create_error(tmp_path):
    """A front whose directory has no ada64 core says which core it could not load."""
    import shutil

    libs = Path(eldpc.__file__).resolve().parent / "_libs"
    for name in ("libeldpc.so", "libeldpc_core_avx2.so", "libeldpc_core_avx512.so"):
        shutil.copy2(libs / name, tmp_path / name)
    package_parent = str(Path(eldpc.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = package_parent + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["ELDPC_DEVICE_CLASS"] = "ada64"
    code = (
        "import numpy as np, eldpc\n"
        f"eldpc.load(path={str(tmp_path / 'libeldpc.so')!r})\n"
        "H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)\n"
        "try:\n"
        "    eldpc.Decoder(H, priors=[0.1, 0.1, 0.1])\n"
        "except eldpc.EldpcError as exc:\n"
        "    print('EldpcError:', exc)\n"
        "else:\n"
        "    print('NO ERROR')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                          env=env)
    assert proc.returncode == 0, proc.stderr
    assert "EldpcError:" in proc.stdout and "NO ERROR" not in proc.stdout
    assert "ada64" in proc.stdout and "libeldpc_core_" in proc.stdout


@requires_gpu
def test_both_variants_decode_concurrently(monkeypatch):
    """Handles of the two variants, each decoded from its own thread, give the same bits."""
    import threading

    H, O, priors = tiny_problem()
    syndromes = np.asarray(sampled_syndromes(H, priors, 64, seed=5), dtype=np.uint8)
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "ada64")
    a = eldpc.Decoder(H, O, priors)
    monkeypatch.setenv("ELDPC_DEVICE_CLASS", "default")
    b = eldpc.Decoder(H, O, priors)
    results, errors = {}, []

    def work(name, dec):
        try:
            results[name] = dec.decode_batch(syndromes)
        except Exception as exc:  # noqa: BLE001 - reported through the assertion below
            errors.append(exc)

    try:
        threads = [threading.Thread(target=work, args=("a", a)),
                   threading.Thread(target=work, args=("b", b))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors
        assert a.core_variant == "ada64" and b.core_variant == "default"
        for key in ("corr", "obs", "status"):
            assert np.array_equal(results["a"][("corr", "obs", "status").index(key)],
                                  results["b"][("corr", "obs", "status").index(key)]), key
    finally:
        a.close()
        b.close()


# -- lazy loading ------------------------------------------------------------


@requires_gpu
def test_ada64_core_loads_only_with_an_ada64_handle():
    """A fresh process maps the default core after load(); an ada64 handle maps the ada64 core."""
    package_parent = str(Path(eldpc.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = package_parent + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("ELDPC_DEVICE_CLASS", None)
    code = (
        "import numpy as np, eldpc\n"
        "def cores():\n"
        "    text = open('/proc/self/maps', encoding='utf-8', errors='replace').read()\n"
        "    return sorted({line.split()[-1].split('/')[-1] for line in text.splitlines()\n"
        "                   if 'libeldpc_core_' in line})\n"
        "eldpc.load()\n"
        "print('LOADED=' + ','.join(cores()))\n"
        "H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)\n"
        "import os\n"
        "os.environ['ELDPC_DEVICE_CLASS'] = 'ada64'\n"
        "assert eldpc.load().core_variant_for_device(8, 9, 142) == 'ada64'\n"
        "dec = eldpc.Decoder(H, priors=[0.1, 0.1, 0.1])\n"
        "print('VARIANT=' + dec.core_variant)\n"
        "print('ADA=' + ','.join(cores()))\n"
        "dec.close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                          env=env)
    assert proc.returncode == 0, proc.stderr
    out = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    assert out["VARIANT"] == "ada64"
    loaded = out["LOADED"]
    assert "libeldpc_core_avx512.so" in loaded or "libeldpc_core_avx2.so" in loaded, loaded
    # The default core does not need an ada64 sibling, and before the ada64 handle none is mapped.
    assert all("_ada64" not in name for name in loaded.split(",")), loaded
    assert "libeldpc_core_avx512_ada64.so" in out["ADA"] or "libeldpc_core_avx2_ada64.so" in out["ADA"], \
        out["ADA"]
