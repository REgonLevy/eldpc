# SPDX-License-Identifier: Apache-2.0
"""Contract tests of the EldpcDecoder: error inference, batches, observables, and statuses."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from eldpc_testlib import DEVICE_KWARGS, cpu_core, require_qldpc_layer

require_qldpc_layer()  # a missing or too-old qLDPC skips this module instead of failing collection

import galois
from qldpc import codes
from qldpc.decoders import (
    BatchErrorDecoder,
    DetectorErrorModelArrays,
    ErrorDecoder,
    ObservableDecoder,
)
from qldpc.decoders.protocols import BatchObservableDecoder

import eldpc as eldpc_package

from eldpc.qldpc import EldpcDecoder, eldpc

# These tests need no CUDA device: they run on the GPU core when one is visible, else on the CPU
# core (DEVICE_KWARGS).
pytestmark = cpu_core


@pytest.fixture(scope="module")
def gross_h() -> np.ndarray:
    """The Z-sector parity-check matrix of the [[144, 12, 12]] bivariate bicycle code."""
    return np.asarray(codes.GrossCode().matrix_z, dtype=np.uint8)


@pytest.fixture(scope="module")
def gross_decoder(gross_h: np.ndarray) -> EldpcDecoder:
    """An eldpc decoder for the gross code's Z sector, with its matrix-column priors."""
    return eldpc(config="a03125c3", error_channel=1e-3, **DEVICE_KWARGS).build(gross_h)


def _sample(H: np.ndarray, shots: int, probability: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample errors and their syndromes for a parity-check matrix."""
    rng = np.random.default_rng(seed)
    errors = (rng.random((shots, H.shape[1])) < probability).astype(np.uint8)
    return errors, (errors @ H.T) % 2


def test_decoder_protocols(gross_decoder: EldpcDecoder, gross_h: np.ndarray) -> None:
    """The decoder satisfies qLDPC's error and observable decoding protocols."""
    decoder = gross_decoder
    assert isinstance(decoder, ErrorDecoder)
    assert isinstance(decoder, BatchErrorDecoder)
    assert isinstance(decoder, ObservableDecoder)
    assert isinstance(decoder, BatchObservableDecoder)
    assert decoder.has_erasure_bit is False
    assert decoder.num_detectors == gross_h.shape[0]
    assert decoder.num_mechanisms == gross_h.shape[1]
    assert decoder.num_observables == 0
    assert decoder.config["decoder"] == "bposd"
    assert "decoder=bposd" in decoder.config_string
    assert repr(decoder).startswith("EldpcDecoder(")


def test_error_repairs_non_trivial_syndromes(
    gross_decoder: EldpcDecoder, gross_h: np.ndarray
) -> None:
    """Every decoded error reproduces its (non-trivial) syndrome: H @ e == s."""
    _, syndromes = _sample(gross_h, 120, 0.01, seed=7)
    assert syndromes.any(), "the test needs non-trivial syndromes"
    corrections = gross_decoder.decode_errors_batch(syndromes)
    assert corrections.shape == (len(syndromes), gross_h.shape[1])
    mismatched = np.any((corrections.astype(np.int64) @ gross_h.T) % 2 != syndromes, axis=1)
    assert not mismatched.any(), f"{int(mismatched.sum())} decodes do not reproduce the syndrome"


def test_single_and_batch_agree(gross_decoder: EldpcDecoder, gross_h: np.ndarray) -> None:
    """Batch methods take one syndrome per row and agree with the single-shot methods."""
    _, syndromes = _sample(gross_h, 16, 0.02, seed=8)
    batch_corrections = gross_decoder.decode_errors_batch(syndromes)
    batch_observables = gross_decoder.decode_observables_batch(syndromes)
    assert batch_observables.shape == (len(syndromes), 0)
    for syndrome, correction, observables in zip(
        syndromes, batch_corrections, batch_observables
    ):
        assert np.array_equal(gross_decoder.decode_errors(syndrome), correction)
        assert np.array_equal(gross_decoder.decode(syndrome), correction)
        assert np.array_equal(gross_decoder.decode_observables(syndrome), observables)
        assert np.array_equal(gross_decoder.decode_batch(syndrome[None, :])[0], correction)


def test_empty_batch_shapes(gross_decoder: EldpcDecoder, gross_h: np.ndarray) -> None:
    """An empty batch yields empty two-dimensional predictions."""
    syndromes = np.zeros((0, gross_h.shape[0]), dtype=np.uint8)
    corrections = gross_decoder.decode_errors_batch(syndromes)
    observables = gross_decoder.decode_observables_batch(syndromes)
    assert corrections.shape == (0, gross_h.shape[1])
    assert observables.shape == (0, 0)


@pytest.mark.parametrize(
    ("syndromes", "message"),
    [
        (np.zeros(72, dtype=np.uint8), "must have shape \\(shots, 72\\)"),
        (np.zeros((2, 71), dtype=np.uint8), "must have shape \\(shots, 72\\)"),
        (np.zeros((2, 3, 72), dtype=np.uint8), "must have shape \\(shots, 72\\)"),
        (np.full((2, 72), 2, dtype=np.uint8), "entries must be 0 or 1"),
        (np.zeros((2, 72), dtype=float), "integer or boolean array"),
    ],
)
def test_batch_validation(
    gross_decoder: EldpcDecoder, syndromes: np.ndarray, message: str
) -> None:
    """Malformed batches are rejected with a clear error."""
    with pytest.raises(ValueError, match=message):
        gross_decoder.decode_errors_batch(syndromes)
    with pytest.raises(ValueError, match=message):
        gross_decoder.decode_observables_batch(syndromes)


@pytest.mark.parametrize(
    ("syndrome", "message"),
    [
        (np.zeros(71, dtype=np.uint8), "must have shape \\(72,\\)"),
        (np.zeros((1, 72), dtype=np.uint8), "must have shape \\(72,\\)"),
        (np.full(72, 2, dtype=np.uint8), "entries must be 0 or 1"),
        (np.zeros(72, dtype=float), "integer or boolean array"),
    ],
)
def test_single_validation(
    gross_decoder: EldpcDecoder, syndrome: np.ndarray, message: str
) -> None:
    """Malformed syndromes are rejected with a clear error."""
    with pytest.raises(ValueError, match=message):
        gross_decoder.decode_errors(syndrome)
    with pytest.raises(ValueError, match=message):
        gross_decoder.decode_observables(syndrome)


def test_status_word_and_detailed_result(
    gross_decoder: EldpcDecoder, gross_h: np.ndarray
) -> None:
    """The status word and the detailed result report the BP/OSD outcome."""
    _, syndromes = _sample(gross_h, 10, 0.01, seed=9)
    for syndrome in syndromes:
        correction, observables, status = gross_decoder._decoder.decode_full(syndrome)
        assert np.array_equal(correction, gross_decoder.decode_errors(syndrome))
        assert observables.shape == (0,)
        assert eldpc_package.status.iterations(status) <= 3
        assert eldpc_package.status.converged(status) or eldpc_package.status.postprocessed(
            status
        )
        detailed = gross_decoder.decode_errors_detailed(syndrome)
        assert np.array_equal(detailed.error, correction)
        assert detailed.diagnostics["eldpc.status"] == status
        assert detailed.diagnostics["eldpc.converged"] == eldpc_package.status.converged(status)
        assert detailed.diagnostics["eldpc.iterations"] == eldpc_package.status.iterations(status)


def test_observables_match_the_native_builder(
    repetition_dem_and_shots: tuple[stim.DetectorErrorModel, np.ndarray, np.ndarray],
) -> None:
    """Observable predictions equal eldpc.Decoder's own on the same detector error model and shots."""
    dem, detections, observables = repetition_dem_and_shots
    decoder = eldpc(config="a03125c3", **DEVICE_KWARGS).build_observable_decoder(dem)
    native = eldpc_package.Decoder.from_dem(dem, **DEVICE_KWARGS)
    batch = decoder.decode_observables_batch(detections)
    assert batch.shape == (len(detections), dem.num_observables)
    assert batch.dtype == np.uint8
    for shot, prediction in zip(detections, batch):
        assert np.array_equal(decoder.decode_observables(shot), prediction)
        assert np.array_equal(native.decode_observables(shot), prediction)
    assert np.array_equal(native.decode_batch(detections)[1], batch)
    # the predictions are the observables of the inferred errors
    flips = DetectorErrorModelArrays(dem, simplify=False).observable_flip_matrix
    corrections = decoder.decode_errors_batch(detections)
    assert np.array_equal(np.asarray(corrections.astype(np.int64) @ flips.T) % 2, batch)
    # the decoder is useful: it predicts most sampled shots correctly
    logical_error_rate = np.any(batch != observables, axis=1).mean()
    assert logical_error_rate <= 0.05, logical_error_rate


def test_dem_error_decoder(
    repetition_dem_and_shots: tuple[stim.DetectorErrorModel, np.ndarray, np.ndarray],
) -> None:
    """A decoder built from a detector error model infers errors consistent with its syndromes."""
    dem, detections, _ = repetition_dem_and_shots
    decoder = eldpc(**DEVICE_KWARGS).build(dem)
    flip_matrix = DetectorErrorModelArrays(dem, simplify=False).detector_flip_matrix
    assert decoder.num_detectors == dem.num_detectors
    assert decoder.num_mechanisms == flip_matrix.shape[1]
    corrections = decoder.decode_errors_batch(detections)
    assert np.array_equal(np.asarray(corrections.astype(np.int64) @ flip_matrix.T) % 2, detections)
    expected_probs = []
    for instruction in dem.flattened():
        if instruction.type == "error":
            expected_probs.append(instruction.args_copy()[0])
    assert np.allclose(decoder.error_channel, expected_probs)


def test_galois_binary_matrix_accepted(gross_h: np.ndarray) -> None:
    """A galois.GF(2) parity-check matrix is accepted like any other binary matrix."""
    matrix = galois.GF(2)(gross_h)
    decoder = eldpc(config="a03125c3", error_channel=1e-3, **DEVICE_KWARGS).build(matrix)
    assert decoder.num_detectors == gross_h.shape[0]
    assert decoder.num_mechanisms == gross_h.shape[1]


@pytest.mark.parametrize(("osd_method", "osd_order"), [("osd_e", 1), ("osd_cs", 2)])
def test_higher_order_osd_methods(osd_method: str, osd_order: int) -> None:
    """The exhaustive and combination-sweep OSD methods decode consistent corrections."""
    H = np.asarray(codes.SurfaceCode(3).matrix, dtype=np.uint8)[:8]
    rng = np.random.default_rng(77)
    errors = (rng.random((5, H.shape[1])) < 0.05).astype(np.uint8)
    syndromes = (errors @ H.T) % 2
    decoder = eldpc(
        config="a03125c3",
        error_channel=1e-3,
        osd_method=osd_method,
        osd_order=osd_order,
        **DEVICE_KWARGS,
    ).build(H)
    assert decoder.config["osd_method"] == osd_method
    assert decoder.config["osd_order"] == osd_order
    corrections = decoder.decode_errors_batch(syndromes)
    assert np.array_equal(
        np.asarray(corrections.astype(np.int64) @ H.T) % 2, syndromes
    )


def test_osd_order_rejected_for_osd0() -> None:
    """osd_order must be 0 for osd0; the backend rejects the combination."""
    H = np.eye(4, dtype=np.uint8)
    with pytest.raises(ValueError, match="osd_order must be 0 for osd_method=osd0"):
        eldpc(config="a03125c3", osd_order=1, **DEVICE_KWARGS).build(H)


def test_plain_bp_decoder() -> None:
    """A plain-BP configuration builds and decodes, and its config carries no OSD keys."""
    H = np.eye(6, dtype=np.uint8)
    decoder = eldpc(config="a03125c3.bp", error_channel=0.01, **DEVICE_KWARGS).build(H)
    assert decoder.config["decoder"] == "bp"
    assert "osd_method" not in decoder.config
    correction = decoder.decode_errors(np.array([1, 0, 0, 0, 1, 0], dtype=np.uint8))
    assert correction.shape == (6,)
    assert correction[0] == 1 and correction[4] == 1


def test_dem_columns_are_never_merged() -> None:
    """One decoder column per error instruction: identical instructions stay separate."""
    dem = stim.DetectorErrorModel(
        "error(0.1) D0\nerror(0.2) D0\nerror(0.3) D1\ndetector D0\ndetector D1"
    )
    decoder = eldpc(**DEVICE_KWARGS).build(dem)
    assert decoder.num_detectors == 2
    assert decoder.num_mechanisms == 3
    assert np.allclose(decoder.error_channel, [0.1, 0.2, 0.3])
    correction = decoder.decode_errors(np.array([1, 0], dtype=np.uint8))
    assert correction.shape == (3,)


def test_closed_decoder_rejects_use(gross_h: np.ndarray) -> None:
    """A closed decoder refuses further decodes with a clear error."""
    decoder = eldpc(config="a03125c3", error_channel=1e-3, **DEVICE_KWARGS).build(gross_h)
    decoder.close()
    with pytest.raises(ValueError, match="has been closed"):
        decoder.decode_errors(np.zeros(gross_h.shape[0], dtype=np.uint8))
    with pytest.raises(ValueError, match="has been closed"):
        decoder.decode_errors_batch(np.zeros((1, gross_h.shape[0]), dtype=np.uint8))
    decoder.close()  # closing twice is harmless


def test_dem_with_observable_only_error() -> None:
    """An error that flips only an observable is a column with no detector and decodes safely."""
    dem = stim.DetectorErrorModel(
        "error(0.1) D0\nerror(0.2) L0\ndetector D0\nlogical_observable L0"
    )
    decoder = eldpc(**DEVICE_KWARGS).build_observable_decoder(dem)
    assert decoder.num_detectors == 1
    assert decoder.num_mechanisms == 2
    assert decoder.num_observables == 1
    flip_matrix = DetectorErrorModelArrays(dem, simplify=False).detector_flip_matrix
    obs_matrix = DetectorErrorModelArrays(dem, simplify=False).observable_flip_matrix
    for syndrome in (np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8)):
        correction = np.asarray(decoder.decode_errors(syndrome), dtype=np.uint8)
        assert correction.shape == (2,)
        assert np.array_equal(np.asarray(correction.astype(np.int64) @ flip_matrix.T) % 2, syndrome)
        prediction = np.asarray(decoder.decode_observables(syndrome))
        assert np.array_equal(prediction, np.asarray(correction.astype(np.int64) @ obs_matrix.T) % 2)
