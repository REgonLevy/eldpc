# SPDX-License-Identifier: Apache-2.0
"""Tests of the eldpc decoder specification: options, pickling, and input validation."""

from __future__ import annotations

import inspect
import pickle

import numpy as np
import pytest
import stim

from eldpc_testlib import DEVICE_KWARGS, cpu_core, require_qldpc_layer

require_qldpc_layer()  # a missing or too-old qLDPC skips this module instead of failing collection

import galois
from qldpc import decoders

from eldpc.qldpc import EldpcDecoder, eldpc

# These tests need no CUDA device: they run on the GPU core when one is visible, else on the CPU
# core (DEVICE_KWARGS).
pytestmark = cpu_core

# a small binary parity-check matrix; validation failures happen before any decoder is created
H = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=np.uint8)
DEM = stim.DetectorErrorModel("""
    error(0.1) D0 D1 L0
    error(0.2) D1 D2
    error(0.05) D2
    detector D0
    detector D1
    detector D2
    logical_observable L0
""")


def test_spec_is_decoder_spec() -> None:
    """eldpc(...) returns a typed qLDPC decoder specification."""
    spec = eldpc(**DEVICE_KWARGS)
    assert isinstance(spec, decoders.DecoderSpec)
    assert spec.infers_errors
    assert spec.predicts_observables_natively
    assert spec.options["config"] == "a03125c3"
    assert spec.options["error_channel"] is None


def test_spec_signature_and_documentation() -> None:
    """The helper has a typed option signature and written documentation."""
    signature = inspect.signature(eldpc)
    for option in (
        "config",
        "error_channel",
        "error_rate",
        "decoder",
        "bp_method",
        "max_iter",
        "ms_scaling_factor",
        "schedule",
        "osd_method",
        "osd_order",
        "add_erasure_bit",
        "device",
        "threads",
    ):
        assert option in signature.parameters, option
    assert signature.return_annotation is not inspect.Signature.empty
    assert eldpc.__doc__ is not None
    assert "Args:" in eldpc.__doc__ and "Returns:" in eldpc.__doc__
    assert "error_channel" in eldpc.__doc__ and "decode_observables" in eldpc.__doc__


def test_spec_options_and_repr() -> None:
    """Named options are stored in the specification and shown by its repr."""
    spec = eldpc(config="ms30", max_iter=5, ms_scaling_factor=0.5, **DEVICE_KWARGS)
    assert spec.options["config"] == "ms30"
    assert spec.options["max_iter"] == 5
    assert spec.options["ms_scaling_factor"] == 0.5
    assert "max_iter=5" in repr(spec)
    assert "ms_scaling_factor=0.5" in repr(spec)


def test_spec_pickles() -> None:
    """The specification pickles, so Sinter can send it to worker processes."""
    spec = eldpc(config="a03125c3", max_iter=7, error_channel=0.01, **DEVICE_KWARGS)
    restored = pickle.loads(pickle.dumps(spec))
    assert isinstance(restored, decoders.DecoderSpec)
    assert restored.options == spec.options
    assert restored.infers_errors and restored.predicts_observables_natively


def test_misspelled_option_rejected() -> None:
    """A misspelled option is rejected when the specification is created."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_itr'"):
        eldpc(max_itr=3, **DEVICE_KWARGS)


def test_unknown_preset_rejected() -> None:
    """An unknown preset name is rejected with the available names."""
    with pytest.raises(ValueError, match="unknown eldpc preset 'nope'"):
        eldpc(config="nope", **DEVICE_KWARGS).build(H)
    with pytest.raises(TypeError, match="config must be the name of an eldpc preset"):
        eldpc(config={"max_iter": 3}, **DEVICE_KWARGS).build(H)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter must be at least 1"),
        ({"max_iter": 2.5}, "max_iter must be an integer"),
        ({"ms_scaling_factor": 0.0}, "ms_scaling_factor must be a finite number greater than 0"),
        ({"ms_scaling_factor": -1}, "ms_scaling_factor must be a finite number greater than 0"),
        ({"decoder": "bplsd"}, "decoder must be 'bposd' or 'bp'"),
        ({"bp_method": "ps"}, "bp_method must be 'ms'"),
        ({"schedule": "serial"}, "schedule must be 'parallel'"),
        ({"osd_method": "OSD_0"}, "osd_method must be 'osd0', 'osd_e' or 'osd_cs'"),
        ({"osd_order": -1}, "osd_order must be at least 0"),
    ],
)
def test_invalid_options_rejected(kwargs: dict[str, object], message: str) -> None:
    """Invalid named options are rejected when the decoder is built."""
    with pytest.raises(ValueError, match=message):
        eldpc(**kwargs, **DEVICE_KWARGS).build(H)  # type: ignore[arg-type]


def test_erasure_not_supported() -> None:
    """The decoder never signals erasure, so add_erasure_bit is rejected."""
    with pytest.raises(ValueError, match="cannot signal erasure"):
        eldpc(add_erasure_bit=True, **DEVICE_KWARGS).build(H)
    assert not eldpc(add_erasure_bit=False, **DEVICE_KWARGS).build(H).has_erasure_bit


def test_non_binary_field_rejected() -> None:
    """A matrix over a non-binary field is rejected with a clear error."""
    matrix = galois.GF(3)([[1, 2, 0], [0, 1, 2]])
    with pytest.raises(ValueError, match="binary decoder.*GF\\(3"):
        eldpc(**DEVICE_KWARGS).build(matrix)


@pytest.mark.parametrize("shape", [(0, 5), (3, 0), (0, 0)])
def test_empty_matrix_rejected(shape: tuple[int, int]) -> None:
    """An empty parity-check matrix is rejected."""
    with pytest.raises(ValueError, match="non-empty parity-check matrix"):
        eldpc(**DEVICE_KWARGS).build(np.zeros(shape, dtype=np.uint8))


def test_wrong_shape_rejected() -> None:
    """A matrix that is not two-dimensional is rejected."""
    with pytest.raises(ValueError, match="must be two-dimensional"):
        eldpc(**DEVICE_KWARGS).build(np.array([1, 0, 1], dtype=np.uint8))


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[2, 0], [0, 1]], dtype=np.uint8),
        np.array([[0.5, 0.0], [0.0, 1.0]]),
        np.array([[2, 1], [0, 1]], dtype=np.int64),
    ],
)
def test_non_binary_matrix_entries_rejected(matrix: np.ndarray) -> None:
    """Matrices with entries other than 0 and 1 are rejected."""
    with pytest.raises(ValueError, match="0 and 1"):
        eldpc(**DEVICE_KWARGS).build(matrix)


def test_object_matrix_rejected() -> None:
    """A matrix of a non-numeric data type is rejected."""
    with pytest.raises(ValueError, match="integer or boolean entries"):
        eldpc(**DEVICE_KWARGS).build(np.array([[True, None], [False, True]], dtype=object))


def test_string_input_rejected() -> None:
    """A string is not a parity-check matrix or a detector error model."""
    with pytest.raises(TypeError, match="not a string"):
        eldpc(**DEVICE_KWARGS).build("0 1 1")


def test_sparse_matrix_accepted() -> None:
    """A scipy sparse parity-check matrix is accepted, and validated like a dense one."""
    import scipy.sparse

    sparse = scipy.sparse.csc_matrix(H)
    decoder = eldpc(error_channel=0.01, **DEVICE_KWARGS).build(sparse)
    assert decoder.num_detectors == H.shape[0] and decoder.num_mechanisms == H.shape[1]
    with pytest.raises(ValueError, match="0 and 1"):
        eldpc(**DEVICE_KWARGS).build(scipy.sparse.csc_matrix(np.array([[2, 0], [0, 1]])))


@pytest.mark.parametrize(
    ("error_channel", "message"),
    [
        (0.0, "strictly between 0 and 1"),
        (1.0, "strictly between 0 and 1"),
        (-0.1, "strictly between 0 and 1"),
        (float("nan"), "strictly between 0 and 1"),
        (float("inf"), "strictly between 0 and 1"),
        (np.array([0.1, 0.2]), "of shape \\(5,\\) are required"),
        (np.array([0.1, 0.2, 0.3, 0.4, 0.0]), "strictly between 0 and 1"),
    ],
)
def test_invalid_error_channel_rejected(error_channel: object, message: str) -> None:
    """Error probabilities outside (0, 1) and of the wrong shape are rejected."""
    with pytest.raises(ValueError, match=message):
        eldpc(error_channel=error_channel, **DEVICE_KWARGS).build(H)  # type: ignore[arg-type]


def test_error_channel_default_and_scalar_broadcast() -> None:
    """A scalar error_channel is broadcast to every column; the default is 1e-3."""
    assert np.allclose(eldpc(error_channel=0.02, **DEVICE_KWARGS).build(H).error_channel, 0.02)
    assert np.allclose(eldpc(**DEVICE_KWARGS).build(H).error_channel, 1e-3)
    per_column = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    assert np.allclose(eldpc(error_channel=per_column, **DEVICE_KWARGS).build(H).error_channel, per_column)


def test_deprecated_error_rate() -> None:
    """error_rate is a deprecated scalar alias for error_channel."""
    with pytest.warns(DeprecationWarning, match="error_rate=0.02.*error_channel=0.02"):
        decoder = eldpc(error_rate=0.02, **DEVICE_KWARGS).build(H)
    assert np.allclose(decoder.error_channel, 0.02)
    with pytest.raises(ValueError, match="cannot both be specified"):
        eldpc(error_channel=0.01, error_rate=0.02, **DEVICE_KWARGS).build(H)


def test_dem_probabilities_come_from_the_dem() -> None:
    """A detector error model supplies its own probabilities and rejects an override."""
    decoder = eldpc(**DEVICE_KWARGS).build(DEM)
    assert isinstance(decoder, EldpcDecoder)
    assert decoder.num_detectors == 3
    assert decoder.num_mechanisms == 3
    assert decoder.num_observables == 1
    assert np.allclose(decoder.error_channel, [0.1, 0.2, 0.05])
    with pytest.raises(ValueError, match="supplies its own error probabilities"):
        eldpc(error_channel=0.01, **DEVICE_KWARGS).build(DEM)
    with pytest.raises(ValueError, match="supplies its own error probabilities"):
        eldpc(error_rate=0.01, **DEVICE_KWARGS).build(DEM)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("error(1.0) D0\n detector D0", "strictly between 0 and 1"),
        ("error(0.0) D0\n detector D0", "strictly between 0 and 1"),
        ("detector D0", "at least one error instruction"),
        ("error(0.1) L0\n logical_observable L0", "at least one detector"),
    ],
)
def test_unsupported_dem_rejected(text: str, message: str) -> None:
    """Detector error models with unusable probabilities or structure are rejected."""
    with pytest.raises(ValueError, match=message):
        eldpc(**DEVICE_KWARGS).build(stim.DetectorErrorModel(text))


def test_config_switch_keeps_keys_consistent() -> None:
    """Switching the decoder kind by option drops or adds the OSD keys of the preset."""
    bposd = eldpc(config="a03125c3", decoder="bposd", **DEVICE_KWARGS).build(H)
    assert bposd.config["decoder"] == "bposd" and bposd.config["osd_method"] == "osd0"
    bp = eldpc(config="a03125c3", decoder="bp", **DEVICE_KWARGS).build(H)
    assert bp.config["decoder"] == "bp"
    assert "osd_method" not in bp.config and "osd_order" not in bp.config
    from_bp_preset = eldpc(config="a03125c3.bp", decoder="bposd", **DEVICE_KWARGS).build(H)
    assert from_bp_preset.config["osd_method"] == "osd0"
    assert from_bp_preset.config["osd_order"] == 0


def test_cpu_device_and_threads() -> None:
    """device="cpu" and threads reach the package decoder, with the outputs of the default device;
    threads with device="gpu", or an unknown device, is a ValueError."""
    spec = eldpc(config="a03125c3", error_channel=0.05, device="cpu", threads=2)
    assert spec.options["device"] == "cpu" and spec.options["threads"] == 2
    syndromes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    default = eldpc(config="a03125c3", error_channel=0.05, **DEVICE_KWARGS)
    with spec.build(H) as cpu, default.build(H) as other:
        assert cpu.device == "cpu" and cpu.threads == 2
        assert np.array_equal(cpu.decode_batch(syndromes), other.decode_batch(syndromes))
    with pytest.raises(ValueError, match="threads"):
        eldpc(device="gpu", threads=2).build(H)
    with pytest.raises(ValueError, match="device"):
        eldpc(device="tpu").build(H)


def test_presets_are_available() -> None:
    """Every preset name of the package can be used as config."""
    import eldpc as eldpc_package

    for name in eldpc_package.presets():
        spec = eldpc(config=name, **DEVICE_KWARGS)
        assert spec.options["config"] == name
