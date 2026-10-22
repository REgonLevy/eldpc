# SPDX-License-Identifier: Apache-2.0
"""Detector error model -> (H, O, priors), the rule of the reference construction.

H, O and priors are derived from the DEM exactly as the reference construction's ``dem_to_arrays``
derives them:

- Walk ``dem.flattened()`` (REPEAT blocks unrolled, shift_detectors applied).
- Every ``error(p) ...`` instruction becomes one column j, in order of appearance. Columns are
  never merged, dropped or reordered, even if identical.
- ``priors[j] = p``. ``H[i, j]`` is the parity of the number of times detector ``D_i`` appears in
  the instruction's targets; ``O[l, j]`` likewise for observable ``L_l``. Separators (``^``) are
  ignored: the column is the XOR of all components.
- ``m = dem.num_detectors``, ``k = dem.num_observables``.

Keeping this rule identical to the reference construction's is what makes the suite's problems the
same decoding problems the reference decoder was developed against.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def arrays_to_dem_text(H, O, priors) -> str:
    """Inverse direction (the reference construction's arrays_to_dem_text), for window problems.

    One line per column: 'error(p) D.. D.. L.. L..' with indices ascending, then
    'detector D{m-1}' (if m > 0) and 'logical_observable L{k-1}' (if k > 0), so the DEM has the
    same detector and observable counts even when trailing rows are empty."""
    H = canonical_csc(H)
    O = canonical_csc(O)
    priors = np.asarray(priors, dtype=np.float64)
    m, n = H.shape
    k = O.shape[0]
    lines = []
    for j in range(n):
        d = H.indices[H.indptr[j]:H.indptr[j + 1]]
        o = O.indices[O.indptr[j]:O.indptr[j + 1]]
        targets = " ".join([f"D{int(x)}" for x in d] + [f"L{int(x)}" for x in o])
        lines.append(f"error({float(priors[j])!r})" + (" " + targets if targets else ""))
    if m > 0:
        lines.append(f"detector D{m - 1}")
    if k > 0:
        lines.append(f"logical_observable L{k - 1}")
    return "\n".join(lines) + "\n"


def extract_window(H, O, priors, det_round, start_round: int, w: int):
    """A window of a parent problem (the reference construction's rule).

    rows = ascending detector indices with start_round <= det_round[i] < start_round + w;
    cols = ascending mechanism indices whose column of H has a nonzero in some row of `rows`.
    Returns (H_w, O_w, priors_w, rows, cols) with canonical matrices; ValueError if no rows.
    """
    H = canonical_csc(H)
    O = canonical_csc(O)
    priors = np.asarray(priors, dtype=np.float64)
    det_round = np.asarray(det_round, dtype=np.int64).reshape(-1)
    m = H.shape[0]
    if det_round.shape[0] != m:
        raise ValueError(f"det_round has {det_round.shape[0]} entries, expected {m}")
    start_round = int(start_round)
    w = int(w)
    rows = np.nonzero((det_round >= start_round) & (det_round < start_round + w))[0].astype(np.int64)
    if rows.size == 0:
        raise ValueError(f"window [{start_round}, {start_round + w}) selects no detectors")
    sub = sp.csc_matrix(H[rows, :])
    sub.sum_duplicates()
    sub.eliminate_zeros()
    sub.sort_indices()
    cols = np.unique(sub.nonzero()[1]).astype(np.int64)
    return (canonical_csc(H[rows][:, cols]), canonical_csc(O[:, cols]), priors[cols].copy(),
            rows, cols)


def canonical_csc(matrix) -> sp.csc_matrix:
    """GF(2)-canonical CSC: entries mod 2, explicit zeros dropped, indices sorted, dtype uint8."""
    c = sp.csc_matrix(matrix, dtype=np.int64)
    c.sum_duplicates()
    c.data %= 2
    c.eliminate_zeros()
    c.sort_indices()
    c.data = np.ones_like(c.data, dtype=np.uint8)
    return sp.csc_matrix((c.data, c.indices.astype(np.int32), c.indptr.astype(np.int32)),
                         shape=c.shape, dtype=np.uint8)


def dem_to_arrays(dem) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    """(H (m x n), O (k x n), priors (n,)) of a stim DetectorErrorModel or DEM text."""
    import stim

    if not isinstance(dem, stim.DetectorErrorModel):
        dem = stim.DetectorErrorModel(str(dem))
    hr, hc, orow, ocol, probs = [], [], [], [], []
    j = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        probs.append(float(inst.args_copy()[0]))
        dets: dict[int, int] = {}
        obs: dict[int, int] = {}
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets[t.val] = dets.get(t.val, 0) ^ 1
            elif t.is_logical_observable_id():
                obs[t.val] = obs.get(t.val, 0) ^ 1
        for d, b in dets.items():
            if b:
                hr.append(d)
                hc.append(j)
        for o, b in obs.items():
            if b:
                orow.append(o)
                ocol.append(j)
        j += 1
    m, k, n = dem.num_detectors, dem.num_observables, j
    H = sp.csc_matrix((np.ones(len(hr), dtype=np.uint8),
                       (np.array(hr, dtype=np.int64), np.array(hc, dtype=np.int64))), shape=(m, n))
    O = sp.csc_matrix((np.ones(len(orow), dtype=np.uint8),
                       (np.array(orow, dtype=np.int64), np.array(ocol, dtype=np.int64))), shape=(k, n))
    return canonical_csc(H), canonical_csc(O), np.array(probs, dtype=np.float64)
