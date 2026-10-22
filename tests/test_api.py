# SPDX-License-Identifier: Apache-2.0
"""The Python API of the package: shapes, dtypes, errors, presets, configs, lifetime, knobs.

The decoders run on the GPU core when a CUDA device is visible and on the CPU core otherwise
(``DEVICE_KWARGS``), so the same tests check either core.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import scipy.sparse as sp

import eldpc
from eldpc_testlib import (DEM_TEXT, DEVICE_KWARGS, cpu_core, demo_problem, requires_gpu,
                           sampled_syndromes, tiny_problem)


@pytest.fixture()
def demo():
    H, O, priors = demo_problem()
    dec = eldpc.Decoder(H, O, priors, config="a03125c3", **DEVICE_KWARGS)
    yield dec, H, O, priors
    dec.close()


@cpu_core
def test_shapes_and_dtypes(demo):
    dec, H, O, priors = demo
    syndromes = sampled_syndromes(H, priors, 20, seed=1)
    corr, obs, st = dec.decode_full(syndromes[0])
    n, k = H.shape[1], O.shape[0]
    assert corr.shape == (n,) and corr.dtype == np.uint8
    assert obs.shape == (k,) and obs.dtype == np.uint8
    assert isinstance(st, int) and st >= 0
    assert dec.last_status == st
    assert np.array_equal(dec.decode(syndromes[0]), corr)
    assert np.array_equal(dec.decode_observables(syndromes[0]), obs)
    assert np.array_equal(obs, (O.astype(np.int64) @ corr.astype(np.int64)) % 2)
    b_corr, b_obs, b_st = dec.decode_batch(syndromes)
    assert b_corr.shape == (20, n) and b_corr.dtype == np.uint8
    assert b_obs.shape == (20, k) and b_obs.dtype == np.uint8
    assert b_st.shape == (20,) and b_st.dtype == np.uint32
    for i in range(20):
        c, o, s = dec.decode_full(syndromes[i])
        assert np.array_equal(b_corr[i], c)
        assert np.array_equal(b_obs[i], o)
        assert int(b_st[i]) == s
    assert np.array_equal(b_obs, (np.asarray(b_corr).astype(np.int64) @ O.astype(np.int64).T) % 2)


@cpu_core
def test_zero_syndrome_short_circuit(demo):
    dec, H, O, priors = demo
    corr, obs, st = dec.decode_full(np.zeros(H.shape[0], dtype=np.uint8))
    assert not corr.any() and not obs.any()
    assert eldpc.status.converged(st)
    assert eldpc.status.iterations(st) == 0


@cpu_core
def test_no_observables():
    H, _, priors = tiny_problem()
    dec = eldpc.Decoder(H, priors=priors, **DEVICE_KWARGS)
    try:
        corr, obs, st = dec.decode_full(np.zeros(H.shape[0], dtype=np.uint8))
        assert obs.shape == (0,) and corr.shape == (H.shape[1],)
        b_corr, b_obs, b_st = dec.decode_batch(np.zeros((3, H.shape[0]), dtype=np.uint8))
        assert b_obs.shape == (3, 0) and b_corr.shape == (3, H.shape[1]) and b_st.shape == (3,)
    finally:
        dec.close()


@cpu_core
def test_config_forms_agree(tmp_path):
    H, O, priors = tiny_problem()
    base = eldpc.presets()["a03125c3"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(base))
    canonical = ("bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;"
                 "osd_method=osd0;osd_order=0;schedule=parallel")
    forms = ["a03125c3", base, canonical, str(path)]
    decs = [eldpc.Decoder(H, O, priors, config=f, **DEVICE_KWARGS) for f in forms]
    try:
        assert len({d.config_string for d in decs}) == 1
        assert decs[0].config == base
        syndrome = sampled_syndromes(H, priors, 1, seed=4)[0]
        results = [d.decode_full(syndrome) for d in decs]
        for corr, obs, st in results[1:]:
            assert np.array_equal(corr, results[0][0])
            assert np.array_equal(obs, results[0][1])
            assert st == results[0][2]
    finally:
        for d in decs:
            d.close()
    # An ABI config string naming the same keys is equivalent too.
    dec = eldpc.Decoder(H, O, priors, config=decs[0].config_string, **DEVICE_KWARGS)
    dec.close()


@cpu_core
def test_all_presets_construct_and_decode():
    H, _, priors = tiny_problem()
    syndrome = sampled_syndromes(H, priors, 1, seed=2)[0]
    for name in sorted(eldpc.presets()):
        dec = eldpc.Decoder(H, priors=priors, config=name, **DEVICE_KWARGS)
        try:
            corr = dec.decode(syndrome)
            assert corr.shape == (H.shape[1],) and corr.dtype == np.uint8
            assert dec.config_string
        finally:
            dec.close()


@cpu_core
def test_config_errors():
    H, O, priors = tiny_problem()
    with pytest.raises(eldpc.EldpcError):
        eldpc.Decoder(H, O, priors, config="bogus_key=1", **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError):
        eldpc.Decoder(H, O, priors, config="a03125c3;ms_scaling_factor=0", **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError):
        eldpc.Decoder(H, O, priors, config="no-such-preset", **DEVICE_KWARGS)


@cpu_core
def test_input_errors():
    H, O, priors = tiny_problem()
    dec = eldpc.Decoder(H, O, priors, **DEVICE_KWARGS)
    try:
        with pytest.raises(eldpc.EldpcError, match="syndrome must have"):
            dec.decode(np.zeros(H.shape[0] + 1, dtype=np.uint8))
        with pytest.raises(eldpc.EldpcError, match="integer or bool"):
            dec.decode(np.zeros(H.shape[0], dtype=np.float64))
        bad = np.zeros(H.shape[0], dtype=np.uint8)
        bad[0] = 2
        with pytest.raises(eldpc.EldpcError, match="0 or 1"):
            dec.decode(bad)
        with pytest.raises(eldpc.EldpcError, match="shape"):
            dec.decode_batch(np.zeros((2, H.shape[0] + 1), dtype=np.uint8))
        with pytest.raises(eldpc.EldpcError, match="shape"):
            dec.decode_batch(np.zeros(H.shape[0], dtype=np.uint8))
        with pytest.raises(eldpc.EldpcError, match="priors"):
            dec.set_priors(np.full(H.shape[1], 1.5))
        with pytest.raises(eldpc.EldpcError, match="priors"):
            dec.set_priors(np.zeros(H.shape[1] + 1))
    finally:
        dec.close()


@cpu_core
def test_constructor_errors():
    H, O, priors = tiny_problem()
    n, m = H.shape[1], H.shape[0]
    with pytest.raises(eldpc.EldpcError, match="error_rate"):
        eldpc.Decoder(H, O, **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError, match="not both"):
        eldpc.Decoder(H, O, priors, error_rate=0.1, **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError, match="strictly between"):
        eldpc.Decoder(H, O, np.zeros(n), **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError, match="strictly between"):
        eldpc.Decoder(H, O, np.full(n, 1.0), **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError, match="strictly between"):
        eldpc.Decoder(H, O, np.full(n, np.nan), **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError, match="entries"):
        eldpc.Decoder(H, O, priors[:-1], **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError, match="as many columns"):
        eldpc.Decoder(H, np.zeros((1, n + 1), dtype=np.uint8), priors, **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError, match="2-D"):
        eldpc.Decoder(np.zeros(4, dtype=np.uint8), priors=priors, **DEVICE_KWARGS)
    with pytest.raises(eldpc.EldpcError, match="error_rate"):
        eldpc.Decoder(H, O, error_rate="high", **DEVICE_KWARGS)
    assert m > 0


@cpu_core
def test_error_rate_equals_uniform_priors():
    H, O, priors = tiny_problem()
    dec = eldpc.Decoder(H, O, error_rate=0.1, **DEVICE_KWARGS)
    ref = eldpc.Decoder(H, O, np.full(H.shape[1], 0.1), **DEVICE_KWARGS)
    try:
        syndrome = sampled_syndromes(H, priors, 1, seed=3)[0]
        a, b = dec.decode_full(syndrome), ref.decode_full(syndrome)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    finally:
        dec.close()
        ref.close()


@cpu_core
def test_context_manager_and_close(demo):
    dec, H, O, _ = demo
    dec.close()
    dec.close()  # idempotent
    with pytest.raises(eldpc.EldpcError, match="closed"):
        dec.decode(np.zeros(H.shape[0], dtype=np.uint8))
    with pytest.raises(eldpc.EldpcError, match="closed"):
        dec.soft_output()
    with eldpc.Decoder(H, O, error_rate=0.05, **DEVICE_KWARGS) as d2:
        assert d2.decode(np.zeros(H.shape[0], dtype=np.uint8)).shape == (H.shape[1],)
    with pytest.raises(eldpc.EldpcError, match="closed"):
        d2.decode(np.zeros(H.shape[0], dtype=np.uint8))
    assert "eldpc.Decoder" in repr(dec)


@cpu_core
def test_keep_warm_does_not_change_results(demo):
    dec, H, O, priors = demo
    syndromes = sampled_syndromes(H, priors, 25, seed=5)
    before = dec.decode_batch(syndromes)
    assert dec.keep_warm() is None
    after = dec.decode_batch(syndromes)
    for a, b in zip(before, after):
        assert np.array_equal(a, b)


@cpu_core
def test_set_priors_round_trip():
    H, O, priors = demo_problem()
    other = np.linspace(0.005, 0.25, H.shape[1])
    dec = eldpc.Decoder(H, O, priors, **DEVICE_KWARGS)
    ref_other = eldpc.Decoder(H, O, other, **DEVICE_KWARGS)
    syndromes = sampled_syndromes(H, priors, 30, seed=6)
    try:
        original = dec.decode_batch(syndromes)
        changed = ref_other.decode_batch(syndromes)
        dec.set_priors(other)
        after = dec.decode_batch(syndromes)
        for a, b in zip(changed, after):
            assert np.array_equal(a, b)
        dec.set_priors(priors)
        back = dec.decode_batch(syndromes)
        for a, b in zip(original, back):
            assert np.array_equal(a, b)
    finally:
        dec.close()
        ref_other.close()


@cpu_core
def test_soft_output(demo):
    dec, H, O, priors = demo
    syndrome = sampled_syndromes(H, priors, 1, seed=8)[0]
    dec.decode_full(syndrome)
    llr = dec.soft_output()
    assert llr.shape == (H.shape[1],) and llr.dtype == np.float64
    assert np.all(np.isfinite(llr))
    zero = np.zeros(H.shape[0], dtype=np.uint8)
    dec.decode_full(zero)
    prior_llr = np.array([math.log((1.0 - p) / p) for p in priors])
    np.testing.assert_allclose(dec.soft_output(), prior_llr, rtol=0, atol=1e-12)


@cpu_core
def test_from_dem_agrees_with_matrices():
    import stim

    model = stim.DetectorErrorModel(DEM_TEXT)
    H, O, priors = eldpc.dem_to_arrays(model)
    from_dem = eldpc.Decoder.from_dem(model, config="a03125c3", **DEVICE_KWARGS)
    from_text = eldpc.Decoder.from_dem(DEM_TEXT, config="a03125c3", **DEVICE_KWARGS)
    from_arrays = eldpc.Decoder(H, O, priors, config="a03125c3", **DEVICE_KWARGS)
    try:
        assert (from_dem.num_detectors, from_dem.num_mechanisms, from_dem.num_observables) == \
               (H.shape[0], H.shape[1], O.shape[0])
        syndromes = sampled_syndromes(H, priors, 10, seed=9)
        for i in range(10):
            base = from_arrays.decode_full(syndromes[i])
            for dec in (from_dem, from_text):
                r = dec.decode_full(syndromes[i])
                assert np.array_equal(r[0], base[0])
                assert np.array_equal(r[1], base[1])
                assert r[2] == base[2]
    finally:
        from_dem.close()
        from_text.close()
        from_arrays.close()


@cpu_core
def test_from_dem_errors():
    with pytest.raises(eldpc.EldpcError, match="detector error model"):
        eldpc.Decoder.from_dem("this is not a dem")


@cpu_core
def test_build_info_and_isa():
    info = eldpc.build_info()
    assert isinstance(info, str) and info
    for field in ("compiler=", "isa="):
        assert field in info
    # The process's core is the GPU core in a normal build and the CPU core in a CPU-only build.
    assert "lib=fastbpgpu6g" in info or "lib=fastbpcpu6g" in info
    if "lib=fastbpgpu6g" in info:
        assert "cuda=" in info
    current = eldpc.isa()
    assert current in ("avx2", "avx512")
    assert f"host_isa={current}" in info
    assert eldpc.load().version() == eldpc.__version__


@requires_gpu
def test_devices():
    devs = eldpc.devices()
    assert devs, "no CUDA device reported"
    for d in devs:
        assert isinstance(d["name"], str) and d["name"]
        assert isinstance(d["compute_capability"], str) and "." in d["compute_capability"]
        assert int(d["sm_count"]) > 0
        assert int(d["l2_bytes"]) > 0
        assert isinstance(d["driver_version"], str) and "." in d["driver_version"]
        assert d["core_variant"] in ("default", "ada64")


@cpu_core
def test_load_is_idempotent():
    assert eldpc.load() is eldpc.load()
    with pytest.raises(eldpc.EldpcError, match="already loaded"):
        eldpc.load(isa="avx2")
    with pytest.raises(eldpc.EldpcError, match="isa"):
        eldpc.load(isa="sse4")


@cpu_core
def test_outputs_are_fresh_arrays(demo):
    dec, H, O, priors = demo
    syndrome = sampled_syndromes(H, priors, 1, seed=16)[0]
    c1, o1, _ = dec.decode_full(syndrome)
    c2, o2, _ = dec.decode_full(syndrome)
    assert c1 is not c2 and o1 is not o2
    assert not np.shares_memory(c1, c2) and not np.shares_memory(o1, o2)
    assert not np.shares_memory(c1, syndrome)
    batch1 = dec.decode_batch(np.stack([syndrome]))
    batch2 = dec.decode_batch(np.stack([syndrome]))
    assert not np.shares_memory(batch1[0], batch2[0])
    assert not np.shares_memory(batch1[1], batch2[1])
    # Mutating a returned correction does not affect later decodes.
    c1[:] = 0
    c3, _, _ = dec.decode_full(syndrome)
    assert np.array_equal(c3, c2)


@cpu_core
def test_empty_batch(demo):
    dec, H, O, _ = demo
    corr, obs, st = dec.decode_batch(np.zeros((0, H.shape[0]), dtype=np.uint8))
    assert corr.shape == (0, H.shape[1]) and obs.shape == (0, O.shape[0]) and st.shape == (0,)


@cpu_core
def test_batch_does_not_change_last_status(demo):
    dec, H, O, priors = demo
    syndrome = sampled_syndromes(H, priors, 1, seed=13)[0]
    _, _, st = dec.decode_full(syndrome)
    dec.decode_batch(np.stack([syndrome, syndrome]))
    assert dec.last_status == st


@cpu_core
def test_two_handles_give_identical_results(demo):
    dec, H, O, priors = demo
    other = eldpc.Decoder(H, O, priors, config="a03125c3", **DEVICE_KWARGS)
    try:
        syndromes = sampled_syndromes(H, priors, 20, seed=14)
        a = dec.decode_batch(syndromes)
        b = other.decode_batch(syndromes)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
    finally:
        other.close()


@cpu_core
def test_concurrent_decoders():
    from concurrent.futures import ThreadPoolExecutor

    H, O, priors = demo_problem()
    syndromes = sampled_syndromes(H, priors, 40, seed=15)
    single = eldpc.Decoder(H, O, priors, **DEVICE_KWARGS)
    try:
        expected = single.decode_batch(syndromes)
    finally:
        single.close()

    def run(chunk):
        dec = eldpc.Decoder(H, O, priors, **DEVICE_KWARGS)
        try:
            return dec.decode_batch(chunk)
        finally:
            dec.close()

    chunks = [syndromes[i::4] for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, chunks))
    for index, (corr, obs, st) in enumerate(results):
        rows = np.arange(index, len(syndromes), 4)
        assert np.array_equal(corr, expected[0][rows])
        assert np.array_equal(obs, expected[1][rows])
        assert np.array_equal(st, expected[2][rows])


@cpu_core
def test_extreme_priors_with_finite_llrs():
    H, O, _ = tiny_problem()
    for probability in (1e-300, 0.9999999999999999):
        dec = eldpc.Decoder(H, O, np.full(H.shape[1], probability), **DEVICE_KWARGS)
        try:
            assert dec.decode(np.zeros(H.shape[0], dtype=np.uint8)).shape == (H.shape[1],)
        finally:
            dec.close()


@cpu_core
def test_sparse_and_dense_inputs_agree():
    H, O, priors = tiny_problem()
    dense = eldpc.Decoder(H, O, priors, **DEVICE_KWARGS)
    sparse = eldpc.Decoder(sp.csr_matrix(H), sp.csc_matrix(O), priors, **DEVICE_KWARGS)
    booled = eldpc.Decoder(H.astype(bool), O.astype(bool), priors, **DEVICE_KWARGS)
    try:
        syndrome = sampled_syndromes(H, priors, 1, seed=11)[0]
        a = dense.decode_full(syndrome)
        for dec in (sparse, booled):
            b = dec.decode_full(syndrome)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    finally:
        dense.close()
        sparse.close()
        booled.close()
