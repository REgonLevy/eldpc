# SPDX-License-Identifier: Apache-2.0
"""Detector error model -> (H, O, priors).

Rule (the one the decoder bench's problem files use):

- Walk ``dem.flattened()`` (REPEAT blocks unrolled, shift_detectors applied).
- Every ``error(p) ...`` instruction becomes one column ``j``, in order of appearance. Columns
  are never merged, dropped or reordered, even if identical.
- ``priors[j] = p``. ``H[i, j]`` is the parity of the number of times detector ``D_i`` appears
  in the instruction's targets; ``O[l, j]`` likewise for observable ``L_l``. Separators (``^``)
  are ignored: the column is the XOR of all components.
- ``m = dem.num_detectors`` and ``k = dem.num_observables``.

The conversion walks the canonical text of the flattened model rather than the Python object
tree of the instruction targets: one line per instruction, and a target is one ``D<index>`` or
``L<index>`` token. That is the same walk, about 2.5x faster on models with tens of thousands of
mechanisms (the object API creates a Python object per target).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_MASK32 = 0xFFFFFFFF


def _parse_flattened_text(text: str):
    """One (probability, target keys) entry per ``error`` line of a flattened model's text.

    A key packs the column (error index) and the detector/observable index into one int64:
    ``column << 32 | index``. Detector keys and observable keys are kept apart. Lines that are
    not ``error`` instructions (``detector``/``logical_observable`` declarations) are skipped.
    """
    probs: list[float] = []
    dkeys: list[int] = []
    lkeys: list[int] = []
    dap, lap, pap = dkeys.append, lkeys.append, probs.append
    column = 0
    for line in text.splitlines():
        if not line.startswith("error("):
            continue
        close = line.index(")")
        pap(float(line[6:close]))
        base = column << 32
        for token in line[close + 1:].split():
            kind = token[0]
            if kind == "D":
                dap(base | int(token[1:]))
            elif kind == "L":
                lap(base | int(token[1:]))
            # '^' separators carry no target; XOR semantics are the parity of the target list.
        column += 1
    return probs, dkeys, lkeys


def _odd_keys(keys: list[int]) -> np.ndarray:
    """The sorted keys that occur an odd number of times (XOR of the target multiset)."""
    if not keys:
        return np.zeros(0, dtype=np.int64)
    key = np.asarray(keys, dtype=np.int64)
    key.sort()
    change = np.empty(key.size, dtype=bool)
    change[0] = True
    np.not_equal(key[1:], key[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    ends = np.empty(starts.size, dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = key.size
    return key[starts[((ends - starts) & 1) == 1]]


def _csc(keys: np.ndarray, num_columns: int, num_rows: int) -> sp.csc_matrix:
    """CSC matrix (num_rows x num_columns) from sorted ``column << 32 | row`` keys with odd parity."""
    if keys.size == 0:
        return sp.csc_matrix((num_rows, num_columns), dtype=np.uint8)
    cols = (keys >> 32).astype(np.int32)
    rows = (keys & _MASK32).astype(np.int32)
    indptr = np.zeros(num_columns + 1, dtype=np.int64)
    np.cumsum(np.bincount(cols, minlength=num_columns), out=indptr[1:])
    return sp.csc_matrix((np.ones(keys.size, dtype=np.uint8), rows, indptr.astype(np.int32)),
                         shape=(num_rows, num_columns), dtype=np.uint8)


def dem_to_arrays(dem) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    """Convert a ``stim.DetectorErrorModel`` (or DEM text) to ``(H, O, priors)``.

    ``H`` is ``m x n``, ``O`` is ``k x n``, both canonical CSC with uint8 data; ``priors`` is a
    float64 array of ``n`` error probabilities.
    """
    import stim

    if not isinstance(dem, stim.DetectorErrorModel):
        try:
            dem = stim.DetectorErrorModel(str(dem))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"not a detector error model: {exc}") from None
    probs, dkeys, lkeys = _parse_flattened_text(str(dem.flattened()))
    n = len(probs)
    return (_csc(_odd_keys(dkeys), n, dem.num_detectors),
            _csc(_odd_keys(lkeys), n, dem.num_observables),
            np.asarray(probs, dtype=np.float64))
