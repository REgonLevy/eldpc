# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of the eldpc decoder with qLDPC's workflows: Sinter, windows, code capacity."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import stim

from eldpc_testlib import DEVICE_KWARGS, cpu_core, require_qldpc_layer

require_qldpc_layer()  # a missing or too-old qLDPC skips this module instead of failing collection

from qldpc import codes, decoders
from qldpc.decoders import DetectorErrorModelArrays

import eldpc as eldpc_package

from eldpc.qldpc import eldpc

# These tests need no CUDA device: they run on the GPU core when one is visible, else on the CPU
# core (DEVICE_KWARGS).
pytestmark = cpu_core


def _ldpc_bp_osd(max_iter: int, ms_scaling_factor: float, **options: object) -> decoders.DecoderSpec:
    """qLDPC's own ldpc BP+OSD specification at the settings of an eldpc preset (options such as
    error_channel are passed through, so that both decoders see the same priors)."""
    return decoders.bp_osd(
        max_iter=max_iter,
        bp_method="ms",
        ms_scaling_factor=ms_scaling_factor,
        schedule="parallel",
        osd_method="OSD_0",
        osd_order=0,
        **options,
    )


def _surface_code(distance: int, rounds: int, p: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
    )


def test_sinter_collect() -> None:
    """Sinter collects shots with the eldpc decoder in two worker processes."""
    import sinter

    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    decoder = decoders.SinterDecoder(decoder=eldpc(config="a03125c3", **DEVICE_KWARGS))
    stats = sinter.collect(
        tasks=[sinter.Task(circuit=circuit, decoder="eldpc")],
        custom_decoders={"eldpc": decoder},
        num_workers=2,
        max_shots=300,
        max_errors=50,
        print_progress=False,
    )
    assert len(stats) == 1
    result = stats[0]
    assert 200 <= result.shots <= 400, result.shots
    assert result.discards == 0
    assert result.errors / result.shots <= 0.05


def test_sinter_predictions_match_the_native_decoder(
    repetition_dem_and_shots: tuple[stim.DetectorErrorModel, np.ndarray, np.ndarray],
) -> None:
    """Sinter's compiled decoder predicts what eldpc.Decoder does on the same simplified model."""
    dem, detections, _ = repetition_dem_and_shots
    sinter_decoder = decoders.SinterDecoder(decoder=eldpc(config="a03125c3", **DEVICE_KWARGS))
    compiled = sinter_decoder.compile_decoder_for_dem(dem)
    simplified_dem = DetectorErrorModelArrays(dem, simplify=True, decompose_errors=False).to_dem()
    native = eldpc_package.Decoder.from_dem(simplified_dem, **DEVICE_KWARGS)
    predictions = compiled.decode_shots(detections)
    assert predictions.shape == (len(detections), dem.num_observables)
    for shot, prediction in zip(detections, predictions):
        assert np.array_equal(native.decode_observables(shot), prediction)


def test_sinter_predictions_match_ldpc() -> None:
    """The Sinter predictions equal qLDPC's ldpc BP+OSD integration at the same settings."""
    circuit = _surface_code(distance=3, rounds=3, p=0.01)
    dem = circuit.detector_error_model()
    detections, _ = circuit.compile_detector_sampler(seed=77).sample(
        200, separate_observables=True
    )
    ours = decoders.SinterDecoder(decoder=eldpc(config="a03125c3", **DEVICE_KWARGS))
    reference = decoders.SinterDecoder(decoder=_ldpc_bp_osd(3, 0.3125))
    ours_predictions = ours.compile_decoder_for_dem(dem).decode_shots(detections)
    reference_predictions = reference.compile_decoder_for_dem(dem).decode_shots(detections)
    assert np.array_equal(ours_predictions, reference_predictions)


def test_sliding_window_decoder() -> None:
    """qLDPC's SlidingWindowDecoder works with an eldpc error decoder."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=5,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    detections, observables = circuit.compile_detector_sampler(seed=31).sample(
        100, separate_observables=True
    )
    window = decoders.SlidingWindowDecoder(
        window_size=2, stride=1, decoder=eldpc(config="a03125c3", **DEVICE_KWARGS)
    )
    assert isinstance(window, decoders.SinterDecoder)
    compiled = window.compile_decoder_for_dem(dem)
    predictions = compiled.decode_shots(detections)
    assert predictions.shape == (len(detections), dem.num_observables)
    logical_error_rate = np.any(predictions != observables, axis=1).mean()
    assert logical_error_rate <= 0.1, logical_error_rate


def test_sliding_window_predictions_match_ldpc() -> None:
    """The window decoder's predictions equal qLDPC's ldpc integration window for window."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=5,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    detections, _ = circuit.compile_detector_sampler(seed=32).sample(
        100, separate_observables=True
    )
    ours = decoders.SlidingWindowDecoder(
        window_size=2, stride=1, decoder=eldpc(config="a03125c3", **DEVICE_KWARGS)
    )
    reference = decoders.SlidingWindowDecoder(
        window_size=2, stride=1, decoder=_ldpc_bp_osd(3, 0.3125)
    )
    ours_predictions = ours.compile_decoder_for_dem(dem).decode_shots(detections)
    reference_predictions = reference.compile_decoder_for_dem(dem).decode_shots(detections)
    assert np.array_equal(ours_predictions, reference_predictions)


def test_code_capacity_classical_matches_ldpc() -> None:
    """The classical code-capacity estimator matches qLDPC's ldpc integration sample for sample."""
    code = codes.RepetitionCode(5)
    np.random.seed(2024)
    ours = code.get_logical_error_rate_func(200, decoder=eldpc(config="a03125c3", **DEVICE_KWARGS))
    np.random.seed(2024)
    reference = code.get_logical_error_rate_func(
        200, decoder=_ldpc_bp_osd(3, 0.3125)
    )
    assert np.array_equal(ours.num_failures, reference.num_failures)
    assert np.array_equal(ours.num_discards, reference.num_discards)
    logical_error_rate, stderr = ours(0.05)
    assert 0.0 <= logical_error_rate <= 1.0 and stderr >= 0.0


def test_code_capacity_quantum() -> None:
    """A quantum code-capacity estimate runs with the native observable decoder."""
    code = codes.SurfaceCode(3)
    func = code.get_logical_error_rate_func(100, decoder=eldpc(config="a03125c3", **DEVICE_KWARGS))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logical_error_rate, stderr = func(0.01)
    fallbacks = [w for w in caught if issubclass(w.category, decoders.ObservableDecodingFallbackWarning)]
    assert not fallbacks
    assert 0.0 <= logical_error_rate <= 1.0
    assert stderr >= 0.0


@pytest.mark.parametrize(
    ("config", "max_iter", "ms_scaling_factor"),
    [("a03125c3", 3, 0.3125), ("ms30", 30, 0.625)],
)
def test_error_corrections_match_ldpc(
    config: str, max_iter: int, ms_scaling_factor: float
) -> None:
    """Corrections equal qLDPC's ldpc BP+OSD integration at the same settings on the same matrix."""
    H = np.asarray(codes.GrossCode().matrix_z, dtype=np.uint8)
    priors = np.full(H.shape[1], 2e-3)  # not the qLDPC placeholder rate, so the priors must really be passed
    rng = np.random.default_rng(404)
    errors = (rng.random((100, H.shape[1])) < 0.01).astype(np.uint8)
    syndromes = (errors @ H.T) % 2
    syndromes = syndromes[np.flatnonzero(syndromes.any(axis=1))]
    assert len(syndromes) > 20, "the test needs non-trivial syndromes"

    ours = eldpc(config=config, error_channel=priors, **DEVICE_KWARGS).build(H)
    reference = _ldpc_bp_osd(max_iter, ms_scaling_factor, error_channel=priors).build(H)
    for syndrome in syndromes:
        assert np.array_equal(
            np.asarray(ours.decode_errors(syndrome), dtype=np.uint8),
            np.asarray(reference.decode_errors(syndrome), dtype=np.uint8),
        )


def test_sinter_without_simplification(
    repetition_dem_and_shots: tuple[stim.DetectorErrorModel, np.ndarray, np.ndarray],
) -> None:
    """With simplify=False, the compiled decoder predicts what eldpc does for the unsimplified DEM."""
    dem, detections, _ = repetition_dem_and_shots
    sinter_decoder = decoders.SinterDecoder(decoder=eldpc(config="a03125c3", **DEVICE_KWARGS), simplify=False)
    compiled = sinter_decoder.compile_decoder_for_dem(dem)
    native = eldpc_package.Decoder.from_dem(compiled.dem_arrays.to_dem(), **DEVICE_KWARGS)
    predictions = compiled.decode_shots(detections)
    for shot, prediction in zip(detections, predictions):
        assert np.array_equal(native.decode_observables(shot), prediction)


def test_sinter_with_decomposed_errors() -> None:
    """With decompose_errors=True, the compiled decoder predicts what eldpc does for the split DEM."""
    circuit = _surface_code(distance=3, rounds=3, p=0.01)
    dem = circuit.detector_error_model(decompose_errors=True)
    detections, _ = circuit.compile_detector_sampler(seed=55).sample(
        100, separate_observables=True
    )
    sinter_decoder = decoders.SinterDecoder(
        decoder=eldpc(config="a03125c3", **DEVICE_KWARGS), decompose_errors=True
    )
    compiled = sinter_decoder.compile_decoder_for_dem(dem)
    native = eldpc_package.Decoder.from_dem(compiled.dem_arrays.to_dem(), **DEVICE_KWARGS)
    predictions = compiled.decode_shots(detections)
    for shot, prediction in zip(detections, predictions):
        assert np.array_equal(native.decode_observables(shot), prediction)


def test_code_capacity_quantum_matches_ldpc() -> None:
    """The quantum code-capacity estimator matches qLDPC's ldpc integration sample for sample."""
    for code in (codes.SurfaceCode(3), codes.SteaneCode()):
        np.random.seed(303)
        ours = code.get_logical_error_rate_func(200, decoder=eldpc(config="a03125c3", **DEVICE_KWARGS))
        np.random.seed(303)
        reference = code.get_logical_error_rate_func(200, decoder=_ldpc_bp_osd(3, 0.3125))
        assert np.array_equal(ours.num_failures, reference.num_failures)
        assert np.array_equal(ours.num_discards, reference.num_discards)


@pytest.mark.parametrize(
    ("osd_method", "osd_order", "reference_method"),
    [("osd_e", 1, "OSD_E"), ("osd_cs", 2, "OSD_CS")],
)
def test_higher_order_osd_matches_ldpc(
    osd_method: str, osd_order: int, reference_method: str
) -> None:
    """The exhaustive and combination-sweep OSD methods match qLDPC's ldpc integration."""
    H = np.asarray(codes.GrossCode().matrix_z, dtype=np.uint8)
    priors = np.full(H.shape[1], 1e-3)
    rng = np.random.default_rng(606)
    errors = (rng.random((100, H.shape[1])) < 0.01).astype(np.uint8)
    syndromes = (errors @ H.T) % 2
    syndromes = syndromes[np.flatnonzero(syndromes.any(axis=1))]
    assert len(syndromes) > 20, "the test needs non-trivial syndromes"
    ours = eldpc(
        config="a03125c3",
        error_channel=priors,
        osd_method=osd_method,
        osd_order=osd_order,
        **DEVICE_KWARGS,
    ).build(H)
    reference = decoders.bp_osd(
        error_channel=priors,
        max_iter=3,
        bp_method="ms",
        ms_scaling_factor=0.3125,
        schedule="parallel",
        osd_method=reference_method,
        osd_order=osd_order,
    ).build(H)
    for syndrome in syndromes:
        assert np.array_equal(
            np.asarray(ours.decode_errors(syndrome), dtype=np.uint8),
            np.asarray(reference.decode_errors(syndrome), dtype=np.uint8),
        )
