# SPDX-License-Identifier: Apache-2.0
"""qLDPC integration: the exact GPU BP+OSD-0 decoder of the ``eldpc`` package as a qLDPC decoder.

This module lets qLDPC use ``eldpc`` wherever qLDPC accepts a decoder specification::

    from eldpc.qldpc import eldpc
    from qldpc import codes, decoders

    spec = eldpc(config="a03125c3")
    estimator = codes.ClassicalCode(matrix).get_logical_error_rate_func(
        1000, decoder=spec
    )
    sinter_decoder = decoders.SinterDecoder(decoder=spec)
    window_decoder = decoders.SlidingWindowDecoder(
        window_size=5, stride=2, decoder=spec
    )

``eldpc(...)`` returns a :class:`qldpc.decoders.DecoderSpec` (built with qLDPC's own
``decoder_spec`` helper), so it works anywhere qLDPC accepts ``decoder=``.  Building the
specification for a parity-check matrix returns an :class:`EldpcDecoder` that infers errors; building
it for a detector error model (DEM) returns one that also predicts observable flips, using the DEM's
own probability model.  The decoder is binary: matrices over fields other than GF(2) and detector
error models from non-binary codes are rejected with a clear error.

The module needs qLDPC 0.4.2.dev20261004 (qLDPC git main, commit 4c4f801) or newer: the released
qLDPC 0.4.0 and 0.4.1 lack the decoder-specification API it builds on.  Importing it with a missing
or older qLDPC raises an ImportError that says so and names the install command
(``pip install ".[qldpc]"`` from the eldpc source tree).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

#: The qLDPC revision this module is tested with, and how to install it.
QLDPC_REQUIREMENT = "qLDPC 0.4.2.dev20261004 (qLDPC git main, commit 4c4f801) or newer"
QLDPC_INSTALL = ('pip install ".[qldpc]" from the eldpc source tree, or pip install '
                 '"qldpc @ git+https://github.com/qLDPCOrg/qLDPC@'
                 '4c4f80109215fa4a28b4d94e2c285e4683487bc1"')

try:
    import qldpc as _qldpc

    if getattr(_qldpc, "__file__", None) is None:
        # A directory named qldpc without the package (a namespace package), e.g. a source tree's
        # tests/qldpc on sys.path: not an installed qLDPC.
        raise ModuleNotFoundError(
            f"the name qldpc resolves to a directory without the package "
            f"({', '.join(getattr(_qldpc, '__path__', []))})", name="qldpc")
except ImportError as _exc:
    raise ModuleNotFoundError(
        f"eldpc.qldpc needs {QLDPC_REQUIREMENT}, and qLDPC cannot be imported ({_exc}). "
        f"Install it with: {QLDPC_INSTALL}", name="qldpc") from _exc
try:
    from qldpc.decoders.common import PLACEHOLDER_ERROR_RATE
    from qldpc.decoders.construction.specs import PcmOrDem, decoder_spec
    from qldpc.decoders.protocols import (
        BatchErrorDecoder,
        BatchObservableDecoder,
        ErrorDecodeResult,
    )
except ImportError as _exc:
    raise ImportError(
        f"eldpc.qldpc needs {QLDPC_REQUIREMENT}; the installed qLDPC "
        f"{getattr(_qldpc, '__version__', '(version unknown)')} lacks its decoder-specification "
        f"API ({_exc}). The released qLDPC 0.4.0 and 0.4.1 are too old. Install the supported "
        f"revision with: {QLDPC_INSTALL}") from _exc

import galois  # a qLDPC dependency

import eldpc as _eldpc

__all__ = ["EldpcDecoder", "eldpc"]


####################################################################################################
# the decoder


class EldpcDecoder(BatchErrorDecoder, BatchObservableDecoder):
    """A qLDPC error and observable decoder backed by ``eldpc.Decoder``.

    Construct instances through the :func:`eldpc` decoder specification, which builds one from a
    parity-check matrix (error decoding only) or a detector error model (error decoding and native
    observable prediction).  ``decode_errors`` returns one entry per matrix column over GF(2), and
    ``decode_observables`` returns one predicted flip per observable of the model.

    The decoder does not signal erasure: ``has_erasure_bit`` is False, and no prediction ends with
    an erasure flag.
    """

    has_erasure_bit = False

    def __init__(self, decoder: _eldpc.Decoder, error_channel: npt.NDArray[np.floating]) -> None:
        """Wrap a created ``eldpc.Decoder``.  Use the ``eldpc`` specification instead."""
        self._decoder = decoder
        self.num_detectors = int(decoder.num_detectors)
        self.num_mechanisms = int(decoder.num_mechanisms)
        self.num_observables = int(decoder.num_observables)
        self.config = dict(decoder.config)
        self.config_string = str(decoder.config_string)
        self.device = str(decoder.device)
        self.threads = decoder.threads
        self.error_channel = np.array(error_channel, dtype=float)

    def __repr__(self) -> str:
        return (
            f"EldpcDecoder(num_detectors={self.num_detectors},"
            f" num_mechanisms={self.num_mechanisms},"
            f" num_observables={self.num_observables}, config={self.config_string!r})"
        )

    def close(self) -> None:
        """Release the decoder's resources."""
        decoder = getattr(self, "_decoder", None)
        if decoder is not None:
            decoder.close()
            self._decoder = None

    def _backend(self) -> _eldpc.Decoder:
        """Return the wrapped eldpc decoder, or raise a clear error if this decoder is closed."""
        decoder = getattr(self, "_decoder", None)
        if decoder is None:
            raise ValueError("this EldpcDecoder has been closed")
        return decoder

    def __enter__(self) -> EldpcDecoder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001  (a destroyed interpreter may fail here)
            pass

    # ---- error decoding

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return the inferred error, one entry per matrix column."""
        checked = self._check_syndrome(syndrome)
        return self._backend().decode(checked)

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return the inferred error (alias for ``decode_errors``)."""
        return self.decode_errors(syndrome)

    def decode_errors_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes, one per row, and return one inferred error per row."""
        checked = self._check_syndromes(syndromes)
        backend = self._backend()
        if checked.shape[0] == 0:
            return np.zeros((0, self.num_mechanisms), dtype=np.uint8)
        corrections, _, _ = backend.decode_batch(checked)
        return corrections

    def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes (alias for ``decode_errors_batch``)."""
        return self.decode_errors_batch(syndromes)

    # ---- observable prediction

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a syndrome and return one predicted observable flip per observable."""
        checked = self._check_syndrome(syndrome)
        return self._backend().decode_observables(checked)

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes, one per row, and return one prediction per row."""
        checked = self._check_syndromes(syndromes)
        backend = self._backend()
        if checked.shape[0] == 0:
            return np.zeros((0, self.num_observables), dtype=np.uint8)
        _, observables, _ = backend.decode_batch(checked)
        return observables

    # ---- optional detailed results

    def decode_errors_detailed(self, syndrome: npt.NDArray[np.int_]) -> ErrorDecodeResult:
        """Decode an error syndrome and return the error with BP/OSD status diagnostics."""
        checked = self._check_syndrome(syndrome)
        corrections, _, status = self._backend().decode_full(checked)
        return ErrorDecodeResult(corrections, diagnostics=_status_diagnostics(status))

    # ---- input validation

    def _check_syndrome(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.uint8]:
        """Validate one syndrome and return it as a contiguous uint8 array."""
        array = _as_integer_array(syndrome, "syndrome")
        if array.ndim != 1 or array.shape[0] != self.num_detectors:
            raise ValueError(
                f"a syndrome must have shape ({self.num_detectors},) for this decoder, but got"
                f" shape {array.shape}"
            )
        return _check_binary_entries(array, "syndrome")

    def _check_syndromes(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.uint8]:
        """Validate a batch of syndromes, one per row, and return contiguous uint8 data."""
        array = _as_integer_array(syndromes, "syndromes")
        if array.ndim != 2 or array.shape[1] != self.num_detectors:
            raise ValueError(
                f"a batch of syndromes must have shape (shots, {self.num_detectors}), but got"
                f" shape {array.shape}"
            )
        return _check_binary_entries(array, "syndromes")


def _as_integer_array(array: object, name: str) -> npt.NDArray[np.int_]:
    """Return an integer or boolean array, rejecting other data types with a clear error."""
    converted = np.asarray(array)
    if converted.dtype.kind not in "biu":
        raise ValueError(
            f"{name} must be an integer or boolean array of 0s and 1s, but got dtype"
            f" {converted.dtype}"
        )
    return converted


def _check_binary_entries(array: npt.NDArray[np.int_], name: str) -> npt.NDArray[np.uint8]:
    """Check that every entry is 0 or 1, and return a contiguous uint8 array."""
    if array.size:
        # one or two whole-array reductions, without the large temporaries of a mask comparison
        if array.dtype.kind == "u":
            invalid = bool(array.max() > 1)
        else:
            invalid = bool(array.min() < 0 or array.max() > 1)
        if invalid:
            raise ValueError(f"{name} entries must be 0 or 1")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _status_diagnostics(status: int) -> dict[str, object]:
    """Return the eldpc status word of one decode as named diagnostics."""
    return {
        "eldpc.status": int(status),
        "eldpc.converged": _eldpc.status.converged(status),
        "eldpc.postprocessed": _eldpc.status.postprocessed(status),
        "eldpc.iterations": _eldpc.status.iterations(status),
    }


####################################################################################################
# construction


def _build_eldpc(
    pcm_or_dem: PcmOrDem,
    *,
    config: str = "a03125c3",
    error_channel: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    error_rate: float | None = None,
    decoder: Literal["bposd", "bp"] | None = None,
    bp_method: Literal["ms"] | None = None,
    max_iter: int | None = None,
    ms_scaling_factor: float | None = None,
    schedule: Literal["parallel"] | None = None,
    osd_method: Literal["osd0", "osd_e", "osd_cs"] | None = None,
    osd_order: int | None = None,
    add_erasure_bit: bool = False,
    device: str = "gpu",
    threads: int | None = None,
) -> EldpcDecoder:
    """Build an eldpc decoder for a parity-check matrix or detector error model.

    Args:
        pcm_or_dem: A binary parity-check matrix (dense or sparse, over GF(2)) or a
            ``stim.DetectorErrorModel``.  A matrix gives error decoding only; a detector error
            model gives error decoding and native observable prediction, with the probabilities of
            the model.
        config: Name of an ``eldpc`` preset (for example ``"a03125c3"``, the default, or
            ``"a03125c3.bp"`` for plain BP); see ``eldpc.presets()``.  The preset supplies a
            complete configuration, which the named options below override one key at a time.
        error_channel: Error probability for every matrix column, or one probability per column.
            Defaults to ``1e-3`` when the input is a matrix.  A detector error model supplies its
            own probabilities, so it does not accept this option (or ``error_rate``).
        error_rate: Deprecated alias for a scalar ``error_channel``.
        decoder: ``"bposd"`` (belief propagation followed by an ordered-statistics decoder) or
            ``"bp"`` (belief propagation only).
        bp_method: Belief-propagation method; only ``"ms"`` (minimum sum) is supported.
        max_iter: Maximum number of belief-propagation iterations, a positive integer.
        ms_scaling_factor: Minimum-sum scaling factor, a positive number.
        schedule: Belief-propagation update schedule; only ``"parallel"`` is supported.
        osd_method: Ordered-statistics method for ``decoder="bposd"``: ``"osd0"`` (order 0),
            ``"osd_e"`` (exhaustive) or ``"osd_cs"`` (combination sweep).
        osd_order: Ordered-statistics order; must be 0 for ``"osd0"`` and at most 15 for
            ``"osd_e"``.
        add_erasure_bit: Not supported; the decoder never signals erasure, so True is rejected.
        device: ``"gpu"`` (the default) runs the fastbpgpu6g GPU core, ``"cpu"`` the optional CPU
            core on the host (for machines without a usable GPU), ``"auto"`` the GPU when it is
            usable and the CPU core otherwise (with a warning that says why). Results are the same
            on both.
        threads: The CPU core's threads per shot (1..64, default 1, the calling thread included);
            invalid with ``device="gpu"``; with ``"auto"`` used only if the CPU core is chosen.
            Pinning (the ``cpus`` key) is not exposed here; build an
            ``eldpc.Decoder`` with a config dict for that.

    Returns:
        An :class:`EldpcDecoder`.  For a matrix it infers errors (``decode_errors``); for a
        detector error model it also predicts observable flips (``decode_observables``).

    Raises:
        ValueError: For a non-binary field, a malformed or empty matrix or detector error model,
            error probabilities outside (0, 1), or an unusable option value.
        TypeError: For an input that is neither a matrix nor a detector error model.
    """
    if add_erasure_bit:
        raise ValueError(
            "The eldpc decoder cannot signal erasure, so it does not accept the add_erasure_bit"
            " argument"
        )
    resolved = _resolve_config(
        config,
        decoder=decoder,
        bp_method=bp_method,
        max_iter=max_iter,
        ms_scaling_factor=ms_scaling_factor,
        schedule=schedule,
        osd_method=osd_method,
        osd_order=osd_order,
    )

    device_kwargs = {"device": device, "threads": threads}
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        _reject_dem_error_probabilities(error_channel, error_rate)
        probabilities = _check_dem(pcm_or_dem)
        try:
            backend = _eldpc.Decoder.from_dem(pcm_or_dem, config=resolved, **device_kwargs)
        except _eldpc.EldpcError as error:
            raise ValueError(f"eldpc could not build a decoder for this detector error model: {error}") from error
    else:
        matrix = _check_matrix(pcm_or_dem)
        probabilities = _check_error_channel(matrix.shape[1], error_channel, error_rate)
        try:
            backend = _eldpc.Decoder(matrix, priors=probabilities, config=resolved, **device_kwargs)
        except _eldpc.EldpcError as error:
            raise ValueError(f"eldpc could not build a decoder for this matrix: {error}") from error
    return EldpcDecoder(backend, probabilities)


####################################################################################################
# option handling


def _resolve_config(
    config: str,
    *,
    decoder: object,
    bp_method: object,
    max_iter: object,
    ms_scaling_factor: object,
    schedule: object,
    osd_method: object,
    osd_order: object,
) -> dict[str, object]:
    """Resolve a preset name and the named option overrides into one eldpc configuration."""
    if not isinstance(config, str):
        raise TypeError(f"config must be the name of an eldpc preset, not {type(config).__name__}")
    presets = _eldpc.presets()
    if config not in presets:
        raise ValueError(
            f"unknown eldpc preset {config!r}; the presets are {', '.join(sorted(presets))}"
        )
    resolved: dict[str, object] = dict(presets[config])
    explicit = {
        "decoder": decoder,
        "bp_method": bp_method,
        "max_iter": max_iter,
        "ms_scaling_factor": ms_scaling_factor,
        "schedule": schedule,
        "osd_method": osd_method,
        "osd_order": osd_order,
    }
    for key, value in explicit.items():
        if value is not None:
            resolved[key] = _check_option(key, value)

    # Keep the configuration well formed when a named option switches the decoder kind: a preset
    # for one kind carries keys that the other kind rejects.
    if resolved.get("decoder") == "bp":
        for key in ("osd_method", "osd_order"):
            if explicit[key] is None:
                resolved.pop(key, None)
    elif resolved.get("decoder") == "bposd":
        if "osd_method" not in resolved:
            resolved["osd_method"] = "osd0"
        if "osd_order" not in resolved:
            resolved["osd_order"] = 0
    return resolved


def _check_option(key: str, value: object) -> object:
    """Validate one named configuration option and return it."""
    if key == "decoder":
        if value not in ("bposd", "bp"):
            raise ValueError(f"decoder must be 'bposd' or 'bp', not {value!r}")
        return value
    if key == "bp_method":
        if value != "ms":
            raise ValueError(f"bp_method must be 'ms' (minimum sum), not {value!r}")
        return value
    if key == "schedule":
        if value != "parallel":
            raise ValueError(f"schedule must be 'parallel', not {value!r}")
        return value
    if key == "osd_method":
        if value not in ("osd0", "osd_e", "osd_cs"):
            raise ValueError(f"osd_method must be 'osd0', 'osd_e' or 'osd_cs', not {value!r}")
        return value
    if key == "max_iter":
        return _check_int(key, value, minimum=1)
    if key == "osd_order":
        return _check_int(key, value, minimum=0)
    if key != "ms_scaling_factor":
        raise AssertionError(f"unhandled eldpc option {key!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ms_scaling_factor must be a number greater than 0, not {value!r}")
    number = float(value)
    if not np.isfinite(number) or not number > 0:
        raise ValueError(f"ms_scaling_factor must be a finite number greater than 0, not {value!r}")
    return value


def _check_int(key: str, value: object, *, minimum: int) -> int:
    """Validate an integer option."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{key} must be an integer, not {value!r}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{key} must be at least {minimum}, not {value!r}")
    return number


####################################################################################################
# problem input handling


def _check_matrix(matrix: object) -> scipy.sparse.csc_matrix:
    """Validate a parity-check matrix and return it as a canonical binary CSC matrix."""
    if isinstance(matrix, galois.FieldArray):
        if type(matrix).order != 2:
            raise ValueError(
                "eldpc is a binary decoder, so it cannot decode a matrix over"
                f" {type(matrix).name}; pass a decoder specification for non-binary codes"
            )
        matrix = matrix.view(np.ndarray)
    if isinstance(matrix, (str, bytes)):
        raise TypeError(
            "pcm_or_dem must be a parity-check matrix or a stim.DetectorErrorModel, not a string"
        )
    if scipy.sparse.issparse(matrix):
        sparse = scipy.sparse.csc_matrix(matrix)
        sparse.sum_duplicates()
        sparse.eliminate_zeros()
        _check_binary_entries_in(
            sparse.data, "parity-check matrix", "column entries that sum to an even count"
        )
        output = scipy.sparse.csc_matrix(
            (np.ones(sparse.nnz, dtype=np.uint8), sparse.indices.copy(), sparse.indptr.copy()),
            shape=sparse.shape,
        )
    else:
        array = np.asarray(matrix)
        if array.dtype.kind == "f":
            if array.size and np.any((array != 0) & (array != 1)):
                raise ValueError("the parity-check matrix must have only 0 and 1 entries")
            array = array.astype(np.uint8)
        elif array.dtype.kind == "c" or array.dtype.kind == "O":
            raise ValueError(
                f"the parity-check matrix must have integer or boolean entries, not dtype"
                f" {array.dtype}"
            )
        if array.ndim != 2:
            raise ValueError(
                f"the parity-check matrix must be two-dimensional, but got shape {array.shape}"
            )
        _check_binary_entries_in(array.reshape(-1), "parity-check matrix", "")
        array = np.ascontiguousarray(array, dtype=np.uint8)
        output = scipy.sparse.csc_matrix(array)
    if output.shape[0] == 0 or output.shape[1] == 0:
        raise ValueError(
            f"eldpc needs a non-empty parity-check matrix, but got shape {output.shape}"
        )
    return output


def _check_binary_entries_in(values: npt.NDArray[np.generic], what: str, detail: str) -> None:
    """Check that an array holds only 0s and 1s."""
    if values.size and np.any((values != 0) & (values != 1)):
        suffix = f" ({detail})" if detail else ""
        raise ValueError(f"{what} must have only 0 and 1 entries{suffix}")


def _check_error_channel(
    num_mechanisms: int,
    error_channel: float | npt.NDArray[np.floating] | Sequence[float] | None,
    error_rate: float | None,
) -> npt.NDArray[np.floating]:
    """Normalize matrix-column error probabilities and check them against eldpc's range."""
    if error_rate is not None:
        if error_channel is not None:
            raise ValueError("error_rate and error_channel cannot both be specified")
        warnings.warn(
            f"error_rate={error_rate!r} is deprecated; use error_channel={error_rate!r} instead",
            DeprecationWarning,
            stacklevel=4,
        )
        error_channel = error_rate
    if error_channel is None:
        error_channel = PLACEHOLDER_ERROR_RATE
    if np.isscalar(error_channel):
        try:
            scalar = float(error_channel)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(
                f"error_channel must be a probability or one probability per matrix column, but"
                f" got {error_channel!r}"
            ) from None
        probabilities = np.full(num_mechanisms, scalar, dtype=float)
    else:
        try:
            probabilities = np.asarray(error_channel, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"error_channel must be a probability or one probability per matrix column, but"
                f" got {error_channel!r}"
            ) from None
        if probabilities.shape != (num_mechanisms,):
            raise ValueError(
                f"error probabilities of shape ({num_mechanisms},) are required in error_channel,"
                f" but got shape {probabilities.shape}"
            )
    if probabilities.size and (
        np.any(~np.isfinite(probabilities)) or np.any((probabilities <= 0) | (probabilities >= 1))
    ):
        raise ValueError(
            "eldpc requires every error probability to be strictly between 0 and 1;"
            f" error_channel has entries outside (0, 1)"
        )
    return probabilities


def _reject_dem_error_probabilities(error_channel: object, error_rate: object) -> None:
    """Reject explicit error probabilities for a detector error model."""
    options = {"error_channel": error_channel, "error_rate": error_rate}
    if specified := [f"{name}={value!r}" for name, value in options.items() if value is not None]:
        raise ValueError(
            "A detector error model supplies its own error probabilities, so"
            f" {' and '.join(specified)} cannot be specified with one"
        )


def _check_dem(dem: stim.DetectorErrorModel) -> npt.NDArray[np.floating]:
    """Validate a detector error model and return its error probabilities."""
    probabilities: list[float] = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        if not (np.isfinite(probability) and 0.0 < probability < 1.0):
            raise ValueError(
                "eldpc requires every error probability to be strictly between 0 and 1, but this"
                f" detector error model has an error instruction with probability {probability!r}"
            )
        probabilities.append(probability)
    if dem.num_detectors == 0:
        raise ValueError(
            "eldpc needs a detector error model with at least one detector, but this model has"
            " none"
        )
    if not probabilities:
        raise ValueError(
            "eldpc needs a detector error model with at least one error instruction, but this"
            " model has none"
        )
    return np.array(probabilities, dtype=float)


####################################################################################################
# the public decoder specification


eldpc = decoder_spec(
    "eldpc",
    _build_eldpc,
    _build_eldpc,
    doc="""Configure the exact GPU belief-propagation + ordered-statistics decoder of eldpc.

``eldpc`` decodes binary syndromes with minimum-sum belief propagation (parallel schedule) on a
CUDA GPU, followed by an exact bit-packed OSD-0 (or a higher-order OSD method) on the host.  For a
detector error model it predicts observable flips natively, with the probabilities of the model;
for a parity-check matrix it infers errors, using ``error_channel`` as the prior of every column.

Args:
    config: Name of a preset from ``eldpc.presets()``, such as ``"a03125c3"`` (the default: minimum
        sum with scaling factor 0.3125, up to 3 BP iterations, then OSD-0) or ``"ms30"``.  The
        suffix ``".bp"`` selects plain belief propagation without OSD.  A preset sets the whole
        configuration; the named options below override its keys.
    error_channel: One error probability for every matrix column, or one per column.  Defaults to
        1e-3 for a matrix.  A detector error model supplies its own probabilities, so no
        probability may be given with one.
    error_rate: Deprecated alias for a scalar ``error_channel``.
    decoder: ``"bposd"`` or ``"bp"``.
    bp_method: Belief-propagation method; only ``"ms"`` (minimum sum) is supported.
    max_iter: Maximum number of belief-propagation iterations, at least 1.
    ms_scaling_factor: Minimum-sum scaling factor, greater than 0.
    schedule: Belief-propagation schedule; only ``"parallel"`` is supported.
    osd_method: ``"osd0"``, ``"osd_e"`` (exhaustive) or ``"osd_cs"`` (combination sweep), for
        ``decoder="bposd"``.
    osd_order: Ordered-statistics order; 0 for ``"osd0"``, at most 15 for ``"osd_e"``.
    add_erasure_bit: Not supported; True is rejected, because eldpc never signals erasure.
    device: ``"gpu"`` (the default) for the fastbpgpu6g GPU core, ``"cpu"`` for the optional CPU
        core on the host (machines without a usable GPU; several times the GPU core's latency),
        ``"auto"`` for the GPU when it is usable and the CPU core otherwise (with a warning that
        says why). Every output is the same on both.
    threads: The CPU core's threads per shot (1..64, default 1, the calling thread included);
        invalid with ``device="gpu"``; with ``"auto"`` used only if the CPU core is chosen.
        Pinning (the ``cpus`` key) is not exposed here; build an
        ``eldpc.Decoder`` with a config dict for that.

Returns:
    A decoder specification.  ``build(pcm_or_dem)`` returns an ``EldpcDecoder`` that infers
    errors; for a detector error model, ``build_observable_decoder(dem)`` returns one that also
    predicts observable flips, natively from the model.  ``decode_errors`` returns one entry per
    matrix column over GF(2), ``decode_observables`` one flip per observable, and the batch methods
    (``decode_errors_batch``, ``decode_observables_batch``) take one syndrome per row.
""",
)
