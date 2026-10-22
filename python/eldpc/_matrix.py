# SPDX-License-Identifier: Apache-2.0
"""GF(2) matrices in the form the C API expects (compressed sparse columns, row indices ascending)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._abi import ELDPC_API_VERSION, _C_F64P, _C_U32P, _Problem, _f64, _ptr, _u32, EldpcError


def canonical_csc(matrix) -> sp.csc_matrix:
    """GF(2)-canonical CSC: entries mod 2, explicit zeros dropped, row indices sorted, dtype uint8.

    ``matrix`` is a scipy sparse matrix or a 2-D dense array-like; non-zero entries are taken
    mod 2 (values are implicitly 1 in the CSC form the C API takes).
    """
    try:
        if sp.issparse(matrix):
            c = sp.csc_matrix(matrix, dtype=np.int64)
        else:
            a = np.asarray(matrix)
            if a.ndim != 2:
                raise EldpcError(f"a matrix must be 2-D, got shape {a.shape}")
            c = sp.csc_matrix(a.astype(np.int64))
    except EldpcError:
        raise
    except (TypeError, ValueError) as exc:
        raise EldpcError(f"cannot interpret the matrix: {exc}") from None
    c.sum_duplicates()
    c.data %= 2
    c.eliminate_zeros()
    c.sort_indices()
    c.data = np.ones(c.data.shape, dtype=np.uint8)
    return sp.csc_matrix((c.data, c.indices.astype(np.int32), c.indptr.astype(np.int32)),
                         shape=c.shape, dtype=np.uint8)


def build_problem(H: sp.csc_matrix, O: sp.csc_matrix, priors: np.ndarray, dem_text: str | None):
    """Build the ctypes ``eldpc_problem`` for canonical ``H``/``O`` and ``priors``.

    Returns ``(problem, keepalive)``; the arrays in ``keepalive`` must outlive ``eldpc_create``.
    """
    m, n = H.shape
    k = O.shape[0]
    hp = _u32(H.indptr)
    hi = _u32(H.indices)
    op = _u32(O.indptr)
    oi = _u32(O.indices)
    pr = _f64(priors)
    problem = _Problem(ELDPC_API_VERSION, m, n, k,
                       _ptr(hp, _C_U32P), _ptr(hi, _C_U32P), _ptr(op, _C_U32P), _ptr(oi, _C_U32P),
                       _ptr(pr, _C_F64P),
                       dem_text.encode("utf-8") if dem_text is not None else None)
    return problem, (hp, hi, op, oi, pr)
