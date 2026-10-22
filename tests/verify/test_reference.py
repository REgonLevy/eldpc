# SPDX-License-Identifier: Apache-2.0
"""The packaged reference decoder builds from the package tree alone and decodes correctly.

The test builds the CMake target in a temporary cache (the same path the runner uses), then checks
the ABI behaviour on a tiny problem: the zero-syndrome short circuit, syndromes in the column space
of H, observables equal to O*correction, batch equal to a loop of single decodes, deterministic
repeats, and a sane build description.

One further test compares the packaged reference with the reference construction's own build of the
same reference decoder on the same shots. It needs that library and is skipped, with the reason
shown by ``pytest -rs``, unless ``ELDPC_REFERENCE_LDPC_LIB`` points to it; a run without the
variable therefore reports one skipped test more than a run with it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]      # the package root (tests/verify/x.py -> root)
if str(ROOT / "verify") not in sys.path:
    sys.path.insert(0, str(ROOT / "verify"))

from eldpc_verify.abi import (ST_BP_CONVERGED, ST_ITER_SHIFT, AbiDecoder, AbiError,  # noqa: E402
                              AbiLibrary)
from eldpc_verify.dem import canonical_csc  # noqa: E402
from eldpc_verify.reference import build_reference  # noqa: E402

CONFIG = ("bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;"
          "osd_method=osd0;osd_order=0;schedule=parallel")
H = canonical_csc(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
O = canonical_csc(np.array([[1, 0, 1]], dtype=np.uint8))
PRIORS = np.array([0.1, 0.1, 0.1], dtype=np.float64)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("reference-cache")


@pytest.fixture(scope="module")
def reference(cache_dir):
    return build_reference(cache_dir, jobs=2)


def _decoder(reference, config=CONFIG):
    return AbiDecoder(reference.path, config, H, O, PRIORS)


def test_build_reuses_its_cache(reference, cache_dir):
    again = build_reference(cache_dir, jobs=2)
    assert again.cached
    assert again.path == reference.path
    assert again.source_sha256 == reference.source_sha256


def test_build_info_names_ldpc_and_the_patch(reference):
    info = reference.build_info or ""
    assert "ldpc=2.4.1" in info
    assert "patch_sha256=" in info
    digest = info.split("patch_sha256=", 1)[1].split(";")[0].strip()
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    assert "compiler=" in info and "flags=" in info


def test_zero_syndrome_short_circuit(reference):
    decoder = _decoder(reference)
    try:
        obs, corr, status = decoder.decode(np.zeros(2, dtype=np.uint8))
    finally:
        decoder.close()
    assert np.array_equal(corr, np.zeros(3, dtype=np.uint8))
    assert np.array_equal(obs, np.zeros(1, dtype=np.uint8))
    assert status & ST_BP_CONVERGED
    assert status >> ST_ITER_SHIFT == 0


def test_syndromes_in_column_space(reference):
    rng = np.random.default_rng(7)
    decoder = _decoder(reference)
    try:
        for _ in range(40):
            truth = rng.integers(0, 2, size=3).astype(np.uint8)
            syndrome = np.asarray((H @ truth) % 2, dtype=np.uint8).reshape(-1)
            obs, corr, status = decoder.decode(syndrome)
            assert corr.shape == (3,)
            assert np.array_equal((H @ corr) % 2, syndrome)
            assert np.array_equal(obs, (O @ corr) % 2)
            assert (status >> ST_ITER_SHIFT) <= 3
            soft = decoder.soft_output()
            assert soft.shape == (3,) and np.all(np.isfinite(soft))
    finally:
        decoder.close()


def test_batch_equals_loop_and_repeats_are_deterministic(reference):
    rng = np.random.default_rng(11)
    truths = rng.integers(0, 2, size=(12, 3)).astype(np.uint8)
    dets = np.asarray((H @ truths.T) % 2, dtype=np.uint8).T
    first = _decoder(reference)
    second = _decoder(reference)
    try:
        obs_batch, corr_batch, status_batch = first.decode_batch(dets)
        for i in range(dets.shape[0]):
            obs, corr, status = second.decode(dets[i])
            assert np.array_equal(corr, corr_batch[i])
            assert np.array_equal(obs, obs_batch[i])
            assert int(status) == int(status_batch[i])
    finally:
        first.close()
        second.close()


def test_invalid_config_is_rejected_with_a_message(reference):
    with pytest.raises(AbiError, match="dbench_create failed"):
        AbiDecoder(reference.path, "decoder=nonesuch", H, O, PRIORS)


def test_matches_development_reference_library(reference):
    value = os.environ.get("ELDPC_REFERENCE_LDPC_LIB")
    if not value:
        pytest.skip("ELDPC_REFERENCE_LDPC_LIB is not set (the optional comparison with the "
                    "reference construction's own build of the reference decoder)")
    reference_lib = Path(value)
    if not reference_lib.is_file():
        pytest.skip(f"ELDPC_REFERENCE_LDPC_LIB does not name a file: {value}")
    if AbiLibrary(reference_lib).build_info is None:
        pytest.skip("the reference construction's reference build has no build description")
    rng = np.random.default_rng(19)
    truths = rng.integers(0, 2, size=(25, 3)).astype(np.uint8)
    dets = np.asarray((H @ truths.T) % 2, dtype=np.uint8).T
    ours = _decoder(reference)
    theirs = AbiDecoder(reference_lib, CONFIG, H, O, PRIORS)
    try:
        for i in range(dets.shape[0]):
            o1, c1, s1 = ours.decode(dets[i])
            o2, c2, s2 = theirs.decode(dets[i])
            assert np.array_equal(c1, c2)
            assert np.array_equal(o1, o2)
            assert int(s1) == int(s2)
    finally:
        ours.close()
        theirs.close()
