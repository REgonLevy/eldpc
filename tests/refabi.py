# SPDX-License-Identifier: Apache-2.0
"""A minimal ctypes binding to a decoder ABI v1 library (dbench_* symbols), for the tests only.

The bit-identity tests compare the package against a separately built reference library of the same
decoder sources (the path in ``ELDPC_REFERENCE_LIB``, built by the reference construction). This
module speaks that library's ``dbench_*`` symbols directly; the package itself never imports it.
"""

from __future__ import annotations

import ctypes

import numpy as np
import scipy.sparse as sp

_C_U8P = ctypes.POINTER(ctypes.c_uint8)
_C_U32P = ctypes.POINTER(ctypes.c_uint32)
_C_F64P = ctypes.POINTER(ctypes.c_double)


class DbenchProblem(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("num_detectors", ctypes.c_uint32),
        ("num_mechanisms", ctypes.c_uint32),
        ("num_observables", ctypes.c_uint32),
        ("h_colptr", _C_U32P),
        ("h_rowind", _C_U32P),
        ("o_colptr", _C_U32P),
        ("o_rowind", _C_U32P),
        ("priors", _C_F64P),
        ("dem_text", ctypes.c_char_p),
    ]


def _ptr(a, typ):
    return a.ctypes.data_as(typ)


class RefLibrary:
    SYMBOLS = ("dbench_abi_version", "dbench_decoder_name", "dbench_capabilities", "dbench_create",
               "dbench_set_priors", "dbench_decode", "dbench_decode_batch", "dbench_get_soft_output",
               "dbench_destroy")

    def __init__(self, path):
        self.path = str(path)
        self.lib = ctypes.CDLL(self.path)
        for name in self.SYMBOLS:
            if not hasattr(self.lib, name):
                raise RuntimeError(f"{path}: missing symbol {name}")
        L = self.lib
        L.dbench_abi_version.restype = ctypes.c_uint32
        L.dbench_decoder_name.restype = ctypes.c_char_p
        L.dbench_capabilities.restype = ctypes.c_uint32
        L.dbench_create.restype = ctypes.c_void_p
        L.dbench_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(DbenchProblem), ctypes.c_char_p,
                                    ctypes.c_size_t]
        L.dbench_set_priors.restype = ctypes.c_int
        L.dbench_set_priors.argtypes = [ctypes.c_void_p, _C_F64P, ctypes.c_uint32]
        L.dbench_decode.restype = ctypes.c_int
        L.dbench_decode.argtypes = [ctypes.c_void_p, _C_U8P, _C_U8P, _C_U8P, _C_U32P]
        L.dbench_decode_batch.restype = ctypes.c_int
        L.dbench_decode_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, _C_U8P, _C_U8P, _C_U8P, _C_U32P]
        L.dbench_get_soft_output.restype = ctypes.c_int
        L.dbench_get_soft_output.argtypes = [ctypes.c_void_p, _C_F64P, ctypes.c_uint32]
        L.dbench_destroy.restype = None
        L.dbench_destroy.argtypes = [ctypes.c_void_p]
        if int(L.dbench_abi_version()) != 1:
            raise RuntimeError(f"{path}: not ABI version 1")


def _canonical_csc(m) -> sp.csc_matrix:
    c = sp.csc_matrix(m, dtype=np.int64)
    c.sum_duplicates()
    c.data %= 2
    c.eliminate_zeros()
    c.sort_indices()
    return sp.csc_matrix((np.ones(c.data.shape, dtype=np.uint8), c.indices.astype(np.int32),
                          c.indptr.astype(np.int32)), shape=c.shape, dtype=np.uint8)


class RefDecoder:
    def __init__(self, library: RefLibrary, config: str, H, O, priors):
        self.lib = library
        H = _canonical_csc(H)
        O = _canonical_csc(O)
        self.m, self.n = H.shape
        self.k = O.shape[0]
        hp = np.ascontiguousarray(H.indptr, dtype=np.uint32)
        hi = np.ascontiguousarray(H.indices, dtype=np.uint32)
        op = np.ascontiguousarray(O.indptr, dtype=np.uint32)
        oi = np.ascontiguousarray(O.indices, dtype=np.uint32)
        pr = np.ascontiguousarray(priors, dtype=np.float64)
        problem = DbenchProblem(1, self.m, self.n, self.k, _ptr(hp, _C_U32P), _ptr(hi, _C_U32P),
                                _ptr(op, _C_U32P), _ptr(oi, _C_U32P), _ptr(pr, _C_F64P), None)
        err = ctypes.create_string_buffer(1024)
        h = library.lib.dbench_create(config.encode(), ctypes.byref(problem), err, len(err))
        if not h:
            raise RuntimeError(f"dbench_create failed: {err.value.decode()}")
        self._h = ctypes.c_void_p(h)

    def decode_batch(self, syndromes):
        d = np.ascontiguousarray(syndromes, dtype=np.uint8)
        shots = d.shape[0]
        corr = np.zeros((max(shots, 1), max(self.n, 1)), dtype=np.uint8)
        obs = np.zeros((max(shots, 1), max(self.k, 1)), dtype=np.uint8)
        st = np.zeros(max(shots, 1), dtype=np.uint32)
        if shots:
            rc = self._lib_call_batch(shots, d, corr, obs, st)
            if rc != 0:
                raise RuntimeError(f"dbench_decode_batch returned {rc}")
        return corr[:shots, :self.n].copy(), obs[:shots, :self.k].copy(), st[:shots].copy()

    def _lib_call_batch(self, shots, d, corr, obs, st):
        return self.lib.lib.dbench_decode_batch(self._h, shots, _ptr(d, _C_U8P), _ptr(corr, _C_U8P),
                                                _ptr(obs, _C_U8P), _ptr(st, _C_U32P))

    def close(self):
        if getattr(self, "_h", None):
            self.lib.lib.dbench_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
