# Using eldpc with qLDPC

`eldpc.qldpc` exposes the exact GPU BP+OSD decoder of the `eldpc` package as a qLDPC decoder
specification.  The decoder runs minimum-sum belief propagation with a parallel schedule on a CUDA
GPU, followed by an exact bit-packed ordered-statistics decoder on the host.

```python
from eldpc.qldpc import eldpc
```

`eldpc(...)` returns a `qldpc.decoders.DecoderSpec`, so it can be passed anywhere qLDPC accepts
`decoder=`: code-capacity estimators, `decoders.SinterDecoder`, and the sliding-window decoders.

## Requirements

`eldpc.qldpc` needs qLDPC 0.4.2.dev20261004 (qLDPC git main, commit `4c4f801`) or a newer revision
with the same decoder-specification API; it is tested with that commit. The released qLDPC 0.4.0
and 0.4.1 on PyPI are too old: neither has `ErrorDecodeResult`, and 0.4.0 also lacks
`decoder_spec`, both of which `eldpc.qldpc` imports; neither has `codes.GrossCode`, which the
examples below use (checked on their PyPI wheels, for example `pip download qldpc==0.4.1 --no-deps`
and a search of the unpacked package). From the eldpc package root, the `qldpc` extra installs
the tested revision (from git, so it needs `git`), `ldpc` 2.4.1 and, through qLDPC, `sinter`:

```bash
pip install ".[qldpc]"
```

With a missing or older qLDPC, `import eldpc.qldpc` raises an `ImportError` that names the
requirement and this command, and `python -m pytest tests` skips `tests/qldpc` with the same reason.

## Options

The options of `eldpc(...)` select one `eldpc` configuration and the error probabilities of a
problem that is given as a parity-check matrix.

| option | meaning |
|---|---|
| `config` | Name of an `eldpc` preset, from `eldpc.presets()`.  The default is `"a03125c3"`: minimum-sum scaling factor 0.3125, up to 3 BP iterations, then OSD-0.  A `".bp"` suffix selects plain BP without OSD. |
| `error_channel` | One error probability for every matrix column, or one probability per column.  Defaults to `1e-3` for a matrix.  A detector error model supplies its own probabilities, so no probability may be given with one. |
| `error_rate` | Deprecated alias for a scalar `error_channel`. |
| `decoder` | `"bposd"` (BP then OSD) or `"bp"` (BP only). |
| `bp_method` | Only `"ms"` (minimum sum) is supported. |
| `max_iter` | Maximum number of BP iterations, at least 1. |
| `ms_scaling_factor` | Minimum-sum scaling factor, greater than 0. |
| `schedule` | Only `"parallel"` is supported. |
| `osd_method` | `"osd0"`, `"osd_e"` (exhaustive) or `"osd_cs"` (combination sweep), for `decoder="bposd"`. |
| `osd_order` | Ordered-statistics order; 0 for `"osd0"`, at most 15 for `"osd_e"`. |
| `add_erasure_bit` | Not supported; a value of True is rejected with a clear error. |
| `device` | `"gpu"` (the default) runs the fastbpgpu6g GPU core, `"cpu"` the optional CPU core on the host (machines without a usable GPU).  The decoder specification and every workflow built from it use the device the specification was created with; the outputs are the same on both. |
| `threads` | The CPU core's threads per shot (1..64, default 1, the calling thread included); invalid with `device="gpu"` (a `ValueError`).  The specification takes preset names only, so explicit CPU pinning (`cpus`) is not exposed here; build an `eldpc.Decoder` directly for that. |

A preset supplies a complete configuration.  The named options override its keys, so
`eldpc(config="ms30", max_iter=10)` is the `ms30` preset with the iteration cap lowered.  Switching
the decoder kind keeps the configuration well formed: `eldpc(config="a03125c3", decoder="bp")`
drops the OSD keys of the preset, and `eldpc(config="a03125c3.bp", decoder="bposd")` adds OSD-0.

## Code capacity

Pass the specification as `decoder=` to a code-capacity estimator.  For a binary code the
estimator builds a detector error model from the code and the specification decodes it with its
native observable prediction, so no error-to-observable conversion is needed.

```python
from qldpc import codes

from eldpc.qldpc import eldpc

code = codes.GrossCode()          # the [[144, 12, 12]] bivariate bicycle code
estimator = code.get_logical_error_rate_func(
    2000,
    decoder=eldpc(config="a03125c3"),
)
# the estimator samples errors when it is constructed, so its physical-rate argument is free
logical_error_rate, stderr = estimator(0.05)
print(logical_error_rate, stderr)    # about 0.023 (typically within +-0.008) and 0.004; changes from run to run
```

The estimator supplies fixed placeholder probabilities to the decoder, since an estimate reuses one
set of decoding outcomes at every physical error rate. Its sampling is unseeded, so the numbers
vary from run to run: at 0.05 it returns a logical error rate of about 0.023 with a standard error
of about 0.004, and runs spread by about as much: typically within +-0.008 of 0.023 (29 runs on
three machines gave 0.018 to 0.031; samples, not bounds; the README, "Quick start: qLDPC"). At a
rate where none of the 2,000 sampled errors that matter fails, such as 1e-3 for this distance-12
code, the value is a tiny number (below about 1e-10) and the standard
error, a Jeffreys-posterior uncertainty, is about 3.5e-4: the estimate then only says that the
logical error rate is below a few times 1e-4 (`help(qldpc.codes.ErrorRateFunc)` explains both
numbers).

## Sinter

Wrap the specification in qLDPC's `SinterDecoder` (or a subclass) and register it under the name
the tasks use:

```python
import sinter
import stim
from qldpc import decoders

from eldpc.qldpc import eldpc

if __name__ == "__main__":       # sinter starts worker processes; keep the run under this guard
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=5,
        rounds=5,
        after_clifford_depolarization=1e-3,
    )
    decoder = decoders.SinterDecoder(decoder=eldpc(config="a03125c3"))
    stats = sinter.collect(
        tasks=[sinter.Task(circuit=circuit, decoder="eldpc")],
        custom_decoders={"eldpc": decoder},
        num_workers=2,
        max_shots=10000,
        max_errors=100,
    )
    print(stats[0].shots, stats[0].errors)
```

Sinter compiles the decoder for each detector error model that it samples.  The specification
pickles, so Sinter's worker processes build their own decoders.  The compiled decoder predicts
observable flips natively, from the model's detector-flip and observable-flip matrices and its
error probabilities.

## Windows

The sliding-window and sequential-window decoders need an error decoder, which this specification
provides:

```python
import sinter
import stim
from qldpc import decoders

from eldpc.qldpc import eldpc

if __name__ == "__main__":       # sinter starts worker processes; keep the run under this guard
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=5,
        rounds=9,
        after_clifford_depolarization=1e-3,
    )
    window_decoder = decoders.SlidingWindowDecoder(
        window_size=5,
        stride=2,
        decoder=eldpc(config="a03125c3"),
    )
    stats = sinter.collect(
        tasks=[sinter.Task(circuit=circuit, decoder="window")],
        custom_decoders={"window": window_decoder},
        num_workers=2,
        max_shots=2000,
        max_errors=100,
    )
    print(stats[0].shots, stats[0].errors)     # the shots decoded and the logical errors among them
```

A `SlidingWindowDecoder` is itself a Sinter decoder, so it is registered the same way as in the
previous section.

Each window builds its own `eldpc` decoder from that window's detector error model and commits the
inferred errors of its commit region.

## Direct use

`DecoderSpec.build` builds the decoder for a parity-check matrix or a detector error model:

```python
import numpy as np
import stim

from eldpc.qldpc import eldpc

matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)   # m x n over GF(2)
decoder = eldpc(error_channel=0.01).build(matrix)
correction = decoder.decode_errors(np.array([1, 0], dtype=np.uint8))
assert decoder.decode_errors_batch(np.array([[1, 0], [0, 1]]).astype(np.uint8)).shape == (2, 3)

circuit = stim.Circuit.generated(
    "repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.01)
dem = circuit.detector_error_model()                  # any stim.DetectorErrorModel
dem_decoder = eldpc().build(dem)
flips = dem_decoder.decode_observables(np.zeros(dem.num_detectors, dtype=np.uint8))
```

An `EldpcDecoder` built from a matrix infers errors and has no observables; one built from a
detector error model also predicts observable flips.  Its public attributes are `num_detectors`,
`num_mechanisms` (matrix columns), `num_observables`, `config`, `config_string`, and
`error_channel` (the probabilities it decodes with).  `decode_errors_detailed` returns the inferred
error with the status word of the decode as named diagnostics (`eldpc.status`,
`eldpc.converged`, `eldpc.postprocessed`, `eldpc.iterations`).

## Output contract

- `decode_errors(syndrome)` returns one entry per matrix column over GF(2); `decode(syndrome)` is
  its alias.  A syndrome has shape `(num_detectors,)`; a batch has shape
  `(shots, num_detectors)` and both batch methods take one syndrome per row.
- `decode_observables(syndrome)` returns one predicted flip per observable of the detector error
  model; `decode_observables_batch(syndromes)` does the same for a batch.
- Predictions are `numpy.uint8` arrays.  An empty batch gives an empty two-dimensional array.
- The decoder never signals erasure: `has_erasure_bit` is False and no prediction ends with an
  erasure flag.

## Supported inputs

Only binary problems are supported.  A parity-check matrix may be a dense or sparse integer or
boolean array over GF(2), or a `galois.GF(2)` field array; matrices over other fields are rejected.
A detector error model is a `stim.DetectorErrorModel`.  The decoder rejects, with a clear error:

- a matrix over a field other than GF(2), a matrix that is not two-dimensional, or a matrix with
  zero rows or zero columns;
- a matrix with entries other than 0 and 1;
- error probabilities outside the open interval (0, 1), or a probability vector of the wrong
  length;
- a detector error model with no detectors, with no error instructions, or with an error
  probability outside (0, 1);
- an explicit `error_channel` or `error_rate` together with a detector error model;
- a malformed syndrome (wrong shape, non-integer data, or entries other than 0 and 1);
- an unsupported option value (`bp_method` other than `"ms"`, `schedule` other than `"parallel"`,
  an unknown preset, a negative `max_iter` or `osd_order`, and so on).

`decoder="bplsd"`, product-sum BP, and serial schedules are not supported by `eldpc` and are
rejected by the decoder backend.

## Relation to qLDPC's ldpc integration

At the same settings - minimum-sum BP, the same scaling factor and iteration cap, a parallel
schedule, and OSD-0 - the corrections of this decoder equal those of qLDPC's ldpc BP+OSD
integration on the same parity-check matrix and the same probabilities.  That comparison covers
OSD-0; the exhaustive and combination-sweep methods are passed through to the backend, and one
setting of each is also compared with the ldpc integration in the test suite.  Observable predictions
equal that integration's on the same detector error models when Sinter simplifies the model, which
is its default.  With `simplify=False` and `decompose_errors=True`, qLDPC's ldpc integration merges
identical error mechanisms internally while this decoder decodes the model as given (one variable
per error instruction); the two can then differ on a shot, and an `eldpc` decoder built for the
merged model matches the ldpc integration.  The test suite checks the equality on a bivariate
bicycle code and circuit-level detector error models, and compares the two decoders through Sinter,
through the sliding-window decoder, and through the code-capacity estimator.
