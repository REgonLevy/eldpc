# SPDX-License-Identifier: Apache-2.0
"""Bit identity of the package against a separately built reference library of the same sources.

The reference is the decoder ABI v1 library named by ``ELDPC_REFERENCE_LIB``; the tests
compare correction, observables and status word shot by shot, and the batch call against the
reference's batch call.
"""

from __future__ import annotations

import numpy as np
import pytest

import eldpc
from eldpc_testlib import DEM_TEXT, demo_problem, random_problem, requires_gpu, \
    requires_reference_lib, sampled_syndromes, tiny_problem
from refabi import RefDecoder

pytestmark = [requires_gpu, requires_reference_lib]

CONFIGS = ("a03125c3", "ms30")


def preset_string(name: str) -> str:
    return eldpc._config.config_string(eldpc.presets()[name])


def compare(reference_lib, H, O, priors, config, shots=200, seed=0):
    syndromes = np.asarray(sampled_syndromes(H, priors, shots, seed=seed), dtype=np.uint8)
    ref = RefDecoder(reference_lib, preset_string(config), H, O, priors)
    dec = eldpc.Decoder(H, O, priors, config=config)
    try:
        r_corr, r_obs, r_st = ref.decode_batch(syndromes)
        b_corr, b_obs, b_st = dec.decode_batch(syndromes)
        assert b_corr.shape == r_corr.shape and b_obs.shape == r_obs.shape and b_st.shape == r_st.shape
        assert np.array_equal(b_corr, r_corr), "batch corrections differ"
        assert np.array_equal(b_obs, r_obs), "batch observables differ"
        assert np.array_equal(b_st, r_st), "batch status words differ"
        # Observables are O * correction (mod 2) on every shot.
        expected_obs = (b_corr.astype(np.int64) @ np.asarray(O.toarray() if hasattr(O, "toarray") else O,
                                                             dtype=np.int64).T) % 2
        assert np.array_equal(b_obs, expected_obs)
        for i in range(shots):
            c, o, st = dec.decode_full(syndromes[i])
            assert np.array_equal(c, r_corr[i]), f"shot {i} correction differs"
            assert np.array_equal(o, r_obs[i]), f"shot {i} observables differ"
            assert st == int(r_st[i]), f"shot {i} status differs ({st} vs {r_st[i]})"
    finally:
        dec.close()
        ref.close()


@pytest.mark.parametrize("config", CONFIGS)
def test_bit_identity_tiny(reference_lib, config):
    H, O, priors = tiny_problem()
    compare(reference_lib, H, O, priors, config)


@pytest.mark.parametrize("config", CONFIGS)
def test_bit_identity_demo(reference_lib, config):
    H, O, priors = demo_problem()
    compare(reference_lib, H, O, priors, config, shots=150, seed=1)


@pytest.mark.parametrize("config", CONFIGS)
def test_bit_identity_random(reference_lib, config):
    H, O, priors = random_problem(12, 24, 3, seed=21, column_weight=3, prior_lo=0.002, prior_hi=0.25)
    compare(reference_lib, H, O, priors, config, shots=100, seed=2)


@pytest.mark.parametrize("config", CONFIGS)
def test_bit_identity_dem(reference_lib, config):
    model = __import__("stim").DetectorErrorModel(DEM_TEXT)
    H, O, priors = eldpc.dem_to_arrays(model)
    dec = eldpc.Decoder.from_dem(model, config=config)
    ref = RefDecoder(reference_lib, preset_string(config), H, O, priors)
    try:
        syndromes = np.asarray(sampled_syndromes(H, priors, 120, seed=3), dtype=np.uint8)
        r_corr, r_obs, r_st = ref.decode_batch(syndromes)
        b_corr, b_obs, b_st = dec.decode_batch(syndromes)
        assert np.array_equal(b_corr, r_corr)
        assert np.array_equal(b_obs, r_obs)
        assert np.array_equal(b_st, r_st)
    finally:
        dec.close()
        ref.close()


@pytest.mark.parametrize("config", CONFIGS)
def test_bit_identity_zero_syndrome(reference_lib, config):
    H, O, priors = demo_problem()
    ref = RefDecoder(reference_lib, preset_string(config), H, O, priors)
    dec = eldpc.Decoder(H, O, priors, config=config)
    try:
        zero = np.zeros(H.shape[0], dtype=np.uint8)
        r_corr, r_obs, r_st = ref.decode_batch(zero[None, :])
        c, o, st = dec.decode_full(zero)
        assert np.array_equal(c, r_corr[0]) and np.array_equal(o, r_obs[0]) and st == int(r_st[0])
    finally:
        dec.close()
        ref.close()
