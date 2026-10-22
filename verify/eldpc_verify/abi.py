# SPDX-License-Identifier: Apache-2.0
"""ctypes binding for decoder libraries that implement the decoder ABI v1.

The verification suite uses it for two libraries: the reference decoder it builds from
``verify/reference`` and, optionally (``--reference-lib`` or ``ELDPC_REFERENCE_LIB``), a separately
built library of the same decoder sources. Both export the same C symbols (``dbench_*``,
``verify/reference/include/decoder_abi.h``); this module is self-contained and needs no code from
the reference construction.

Syndromes and outputs are unpacked uint8 arrays; batches are shot-major. Library handles are not
thread-safe and are used from one thread only.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

ABI_VERSION = 1
OK, EINVAL, EINTERNAL, ENOTSUP = 0, -1, -2, -3

ST_BP_CONVERGED = 0x0001
ST_POSTPROC_INVOKED = 0x0002
ST_HIGHER_ORDER_WIN = 0x0004
ST_ABSTAIN = 0x0008
ST_SYNDROME_UNMATCHED = 0x0010
ST_FLAGS_MASK = 0x0000FFFF
ST_ITER_SHIFT = 16
ST_ITER_MAX = 0xFFFF

CAP_CORRECTION = 0x01
CAP_SET_PRIORS = 0x02
CAP_SOFT_OUTPUT = 0x04
CAP_NATIVE_BATCH = 0x08

_u8p = ctypes.POINTER(ctypes.c_uint8)
_u32p = ctypes.POINTER(ctypes.c_uint32)
_f64p = ctypes.POINTER(ctypes.c_double)


class AbiError(RuntimeError):
    """A library is missing, cannot be loaded, or returned a failing status code."""


class DbenchProblem(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("num_detectors", ctypes.c_uint32),
        ("num_mechanisms", ctypes.c_uint32),
        ("num_observables", ctypes.c_uint32),
        ("h_colptr", _u32p),
        ("h_rowind", _u32p),
        ("o_colptr", _u32p),
        ("o_rowind", _u32p),
        ("priors", _f64p),
        ("dem_text", ctypes.c_char_p),
    ]


def _ptr(array: np.ndarray, typ):
    return array.ctypes.data_as(typ)


class AbiLibrary:
    """A loaded decoder library; checks every required symbol and the ABI version."""

    SYMBOLS = ("dbench_abi_version", "dbench_decoder_name", "dbench_capabilities", "dbench_create",
               "dbench_set_priors", "dbench_decode", "dbench_decode_batch", "dbench_get_soft_output",
               "dbench_destroy")

    def __init__(self, path):
        self.path = os.path.abspath(os.fspath(path))
        try:
            self.lib = ctypes.CDLL(self.path)
        except OSError as exc:
            raise AbiError(f"cannot load {self.path}: {exc}") from exc
        for symbol in self.SYMBOLS:
            if not hasattr(self.lib, symbol):
                raise AbiError(f"{self.path}: missing symbol {symbol}")
        L = self.lib
        L.dbench_abi_version.restype = ctypes.c_uint32
        L.dbench_abi_version.argtypes = []
        L.dbench_decoder_name.restype = ctypes.c_char_p
        L.dbench_decoder_name.argtypes = []
        L.dbench_capabilities.restype = ctypes.c_uint32
        L.dbench_capabilities.argtypes = []
        L.dbench_create.restype = ctypes.c_void_p
        L.dbench_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(DbenchProblem),
                                    ctypes.c_char_p, ctypes.c_size_t]
        L.dbench_set_priors.restype = ctypes.c_int
        L.dbench_set_priors.argtypes = [ctypes.c_void_p, _f64p, ctypes.c_uint32]
        L.dbench_decode.restype = ctypes.c_int
        L.dbench_decode.argtypes = [ctypes.c_void_p, _u8p, _u8p, _u8p, _u32p]
        L.dbench_decode_batch.restype = ctypes.c_int
        L.dbench_decode_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, _u8p, _u8p, _u8p, _u32p]
        L.dbench_get_soft_output.restype = ctypes.c_int
        L.dbench_get_soft_output.argtypes = [ctypes.c_void_p, _f64p, ctypes.c_uint32]
        L.dbench_destroy.restype = None
        L.dbench_destroy.argtypes = [ctypes.c_void_p]
        version = int(L.dbench_abi_version())
        if version != ABI_VERSION:
            raise AbiError(f"{self.path}: ABI version {version} != {ABI_VERSION}")
        self.name = (L.dbench_decoder_name() or b"").decode("utf-8", "replace")
        self.capabilities = int(L.dbench_capabilities())

    @property
    def build_info(self) -> str | None:
        """The optional dbench_decoder_build_info() string, or None when the library has none."""
        try:
            f = self.lib.dbench_decoder_build_info
        except AttributeError:
            return None
        f.restype = ctypes.c_char_p
        f.argtypes = []
        value = f()
        return value.decode("utf-8", "replace") if value else ""


def library_build_info(path) -> str | None:
    return AbiLibrary(path).build_info


class AbiDecoder:
    """One decoder handle of a library, over one problem (H, O, priors)."""

    def __init__(self, library, config: str, H, O, priors,
                 dem_text: str | None = None):
        self.library = library if isinstance(library, AbiLibrary) else AbiLibrary(library)
        if not isinstance(config, str):
            raise AbiError("config must be an ABI config string")
        self.config = config
        if H.shape[1] != O.shape[1]:
            raise AbiError("H and O have different numbers of columns")
        self.m, self.n = int(H.shape[0]), int(H.shape[1])
        self.k = int(O.shape[0])
        hp = np.ascontiguousarray(H.indptr, dtype=np.uint32)
        hi = np.ascontiguousarray(H.indices, dtype=np.uint32)
        op = np.ascontiguousarray(O.indptr, dtype=np.uint32)
        oi = np.ascontiguousarray(O.indices, dtype=np.uint32)
        pr = np.ascontiguousarray(priors, dtype=np.float64)
        if pr.shape != (self.n,):
            raise AbiError("priors length != num_mechanisms")
        dem = dem_text.encode() if dem_text is not None else None
        problem = DbenchProblem(ABI_VERSION, self.m, self.n, self.k,
                                _ptr(hp, _u32p), _ptr(hi, _u32p), _ptr(op, _u32p),
                                _ptr(oi, _u32p), _ptr(pr, _f64p), dem)
        err = ctypes.create_string_buffer(1024)
        handle = self.library.lib.dbench_create(config.encode(), ctypes.byref(problem), err, 1024)
        if not handle:
            raise AbiError(f"{self.library.path}: dbench_create failed: "
                           f"{err.value.decode('utf-8', 'replace')}")
        self._h = ctypes.c_void_p(handle)

    @property
    def has_correction(self) -> bool:
        return bool(self.library.capabilities & CAP_CORRECTION)

    def decode(self, syndrome):
        """Decode one shot -> (obs (k,), correction (n,) or None, status int)."""
        s = np.ascontiguousarray(syndrome, dtype=np.uint8).reshape(-1)
        if s.shape[0] != self.m:
            raise AbiError("syndrome length != num_detectors")
        obs = np.zeros(max(self.k, 1), dtype=np.uint8)
        corr = np.zeros(max(self.n, 1), dtype=np.uint8) if self.has_correction else None
        status = np.zeros(1, dtype=np.uint32)
        rc = self.library.lib.dbench_decode(
            self._h, _ptr(s, _u8p),
            _ptr(corr, _u8p) if corr is not None else None, _ptr(obs, _u8p), _ptr(status, _u32p))
        if rc != 0:
            raise AbiError(f"dbench_decode returned {rc}")
        return (obs[:self.k].copy(), corr[:self.n].copy() if corr is not None else None,
                int(status[0]))

    def decode_batch(self, dets):
        """Decode a (shots, m) 0/1 array -> (obs (shots,k), corrections (shots,n) or None, status)."""
        d = np.ascontiguousarray(dets, dtype=np.uint8)
        if d.ndim != 2 or d.shape[1] != self.m:
            raise AbiError("dets must be (shots, num_detectors)")
        shots = int(d.shape[0])
        obs = np.zeros((shots, max(self.k, 1)), dtype=np.uint8)
        corr = (np.zeros((shots, max(self.n, 1)), dtype=np.uint8)
                if self.has_correction else None)
        status = np.zeros(max(shots, 1), dtype=np.uint32)
        if shots:
            obs_buf = obs if self.k else np.zeros((shots, 1), dtype=np.uint8)
            corr_buf = corr if (corr is not None and self.n) else None
            rc = self.library.lib.dbench_decode_batch(
                self._h, shots, _ptr(d, _u8p),
                _ptr(corr_buf, _u8p) if corr_buf is not None else None,
                _ptr(obs_buf, _u8p), _ptr(status, _u32p))
            if rc != 0:
                raise AbiError(f"dbench_decode_batch returned {rc}")
        obs = obs[:, :self.k] if self.k else np.zeros((shots, 0), dtype=np.uint8)
        if corr is not None:
            corr = corr[:, :self.n]
        return np.ascontiguousarray(obs), corr, status[:shots].copy()

    def set_priors_llr(self, llr) -> None:
        a = np.ascontiguousarray(llr, dtype=np.float64)
        rc = self.library.lib.dbench_set_priors(self._h, _ptr(a, _f64p), a.shape[0])
        if rc != 0:
            raise AbiError(f"dbench_set_priors returned {rc}")

    def soft_output(self):
        if not (self.library.capabilities & CAP_SOFT_OUTPUT):
            raise AbiError("decoder has no soft output")
        out = np.zeros(self.n, dtype=np.float64)
        rc = self.library.lib.dbench_get_soft_output(self._h, _ptr(out, _f64p), self.n)
        if rc != 0:
            raise AbiError(f"dbench_get_soft_output returned {rc}")
        return out

    def close(self) -> None:
        handle = getattr(self, "_h", None)
        if handle:
            self.library.lib.dbench_destroy(handle)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
