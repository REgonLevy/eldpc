# SPDX-License-Identifier: Apache-2.0
"""ctypes binding to libeldpc.so, the package's C API front (``include/eldpc.h``).

The front selects the host ISA and loads the default core of that ISA at load time, and a core
variant (default or ada64) per handle at create time; this module only talks to the ``eldpc_*``
entry points, never to a ``dbench_*`` symbol. A module-level singleton holds the loaded library,
so the ISA choice (and the ``ELDPC_ISA`` environment override) is made once per process, when the
first :class:`~eldpc.Decoder` (or one of the informational calls) needs it.
"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

import numpy as np

ELDPC_API_VERSION = 1
ELDPC_OK = 0
ELDPC_EINVAL = -1
ELDPC_EINTERNAL = -2
ELDPC_ENOTSUP = -3

_C_U8P = ctypes.POINTER(ctypes.c_uint8)
_C_U32P = ctypes.POINTER(ctypes.c_uint32)
_C_F64P = ctypes.POINTER(ctypes.c_double)


class EldpcError(RuntimeError):
    """An error reported by the eldpc native library or by this package."""


class _Problem(ctypes.Structure):
    """ctypes mirror of ``eldpc_problem`` (field for field)."""

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


def _ptr(array: np.ndarray, typ):
    """The array's data as a ctypes pointer (used by the problem builder)."""
    return array.ctypes.data_as(typ)


def _u32(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uint32)


def _f64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


class Front:
    """A loaded libeldpc.so with its entry points bound."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(path)
        try:
            self.lib = ctypes.CDLL(self.path)
        except OSError as exc:
            raise EldpcError(f"cannot load the eldpc library {self.path}: {exc}") from None
        lib = self.lib
        try:
            lib.eldpc_api_version.restype = ctypes.c_uint32
            lib.eldpc_api_version.argtypes = []
            for name in ("eldpc_version", "eldpc_build_info", "eldpc_isa"):
                fn = getattr(lib, name)
                fn.restype = ctypes.c_char_p
                fn.argtypes = []
            lib.eldpc_core_variant.restype = ctypes.c_char_p
            lib.eldpc_core_variant.argtypes = [ctypes.c_void_p]
            lib.eldpc_core_variant_for_device.restype = ctypes.c_char_p
            lib.eldpc_core_variant_for_device.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.eldpc_create.restype = ctypes.c_void_p
            lib.eldpc_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Problem), ctypes.c_char_p,
                                         ctypes.c_size_t]
            lib.eldpc_set_priors.restype = ctypes.c_int
            lib.eldpc_set_priors.argtypes = [ctypes.c_void_p, _C_F64P, ctypes.c_uint32]
            lib.eldpc_decode.restype = ctypes.c_int
            lib.eldpc_decode.argtypes = [ctypes.c_void_p, _C_U8P, _C_U8P, _C_U8P, _C_U32P]
            lib.eldpc_decode_batch.restype = ctypes.c_int
            lib.eldpc_decode_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, _C_U8P, _C_U8P, _C_U8P,
                                               _C_U32P]
            lib.eldpc_get_soft_output.restype = ctypes.c_int
            lib.eldpc_get_soft_output.argtypes = [ctypes.c_void_p, _C_F64P, ctypes.c_uint32]
            lib.eldpc_keep_warm.restype = ctypes.c_int
            lib.eldpc_keep_warm.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            lib.eldpc_destroy.restype = None
            lib.eldpc_destroy.argtypes = [ctypes.c_void_p]
        except AttributeError as exc:
            raise EldpcError(f"{self.path}: not an eldpc library ({exc})") from None
        version = int(lib.eldpc_api_version())
        if version != ELDPC_API_VERSION:
            raise EldpcError(f"{self.path}: eldpc API version {version}, expected {ELDPC_API_VERSION}")

    # -- informational -----------------------------------------------------

    def version(self) -> str:
        return (self.lib.eldpc_version() or b"").decode("utf-8", "replace")

    def build_info(self) -> str:
        return (self.lib.eldpc_build_info() or b"").decode("utf-8", "replace")

    def isa(self) -> str:
        return (self.lib.eldpc_isa() or b"").decode("utf-8", "replace")

    def load_error(self) -> str | None:
        """Why the front loaded no decoder core (an ``ELDPC_ISA`` value this CPU cannot use, a
        missing core library, ...), or None when it loaded one.

        The C API reports that case as an empty ``eldpc_isa()`` and the error text in place of
        ``eldpc_build_info()``; :func:`eldpc.isa` and :func:`eldpc.build_info` raise it instead.
        """
        if self.isa():
            return None
        return self.build_info() or "eldpc: no core loaded"

    def core_variant_for_device(self, cc_major: int, cc_minor: int, sm_count: int) -> str:
        """The device-class table's variant for these device properties ("default"/"ada64")."""
        text = self.lib.eldpc_core_variant_for_device(int(cc_major), int(cc_minor), int(sm_count))
        return (text or b"").decode("utf-8", "replace")


class Handle:
    """One ``eldpc_decoder`` handle with its buffers.

    A handle must not be used by two threads at once; distinct handles may run concurrently.
    """

    def __init__(self, front: Front, config: str, problem: _Problem):
        self.front = front
        self._h = None
        err = ctypes.create_string_buffer(1024)
        h = front.lib.eldpc_create(config.encode(), ctypes.byref(problem), err, len(err))
        if not h:
            raise EldpcError(err.value.decode("utf-8", "replace") or "eldpc_create failed")
        self._h = ctypes.c_void_p(h)
        self.num_detectors = int(problem.num_detectors)
        self.num_mechanisms = int(problem.num_mechanisms)
        self.num_observables = int(problem.num_observables)
        # The variant the front chose for this handle ("default" or "ada64").
        self.core_variant = (front.lib.eldpc_core_variant(self._h) or b"").decode("utf-8", "replace")
        # The entry points are stable for the library's lifetime; keeping them here saves an
        # attribute chain on every call.
        lib = front.lib
        self._decode = lib.eldpc_decode
        self._decode_batch = lib.eldpc_decode_batch
        self._get_soft_output = lib.eldpc_get_soft_output
        self._keep_warm = lib.eldpc_keep_warm
        self._set_priors = lib.eldpc_set_priors
        self._destroy = lib.eldpc_destroy

    def set_priors_llr(self, llr: np.ndarray) -> None:
        a = _f64(llr)
        rc = self._set_priors(self._h, a.ctypes.data_as(_C_F64P), ctypes.c_uint32(a.shape[0]))
        if rc != ELDPC_OK:
            raise EldpcError(f"eldpc_set_priors returned {rc}")

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """One shot -> (correction uint8[n], observables uint8[k], status int).

        ``syndrome`` must be a C-contiguous uint8 array of the handle's m 0/1 bytes; the caller
        (:class:`eldpc.Decoder`) validates it. The C call writes every output byte, so the output
        arrays are allocated uninitialised and returned as they are (a fresh buffer per call).
        """
        n, k = self.num_mechanisms, self.num_observables
        corr = np.empty(n if n else 1, dtype=np.uint8)
        obs = np.empty(k if k else 1, dtype=np.uint8)
        st = np.empty(1, dtype=np.uint32)
        rc = self._decode(self._h, syndrome.ctypes.data_as(_C_U8P), corr.ctypes.data_as(_C_U8P),
                          obs.ctypes.data_as(_C_U8P), st.ctypes.data_as(_C_U32P))
        if rc != ELDPC_OK:
            raise EldpcError(f"eldpc_decode returned {rc}")
        return corr[:n], obs[:k], int(st[0])

    def decode_batch(self, syndromes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(shots, m) shots -> (corrections (shots, n), observables (shots, k), status (shots,)).

        ``syndromes`` must be a C-contiguous uint8 (shots, m) array; the caller validates it.
        """
        d = syndromes
        shots = int(d.shape[0])
        n, k = self.num_mechanisms, self.num_observables
        corr = np.empty((max(shots, 1), max(n, 1)), dtype=np.uint8)
        obs = np.empty((max(shots, 1), max(k, 1)), dtype=np.uint8)
        st = np.empty(max(shots, 1), dtype=np.uint32)
        if shots:
            rc = self._decode_batch(self._h, ctypes.c_uint32(shots), d.ctypes.data_as(_C_U8P),
                                    corr.ctypes.data_as(_C_U8P), obs.ctypes.data_as(_C_U8P),
                                    st.ctypes.data_as(_C_U32P))
            if rc != ELDPC_OK:
                raise EldpcError(f"eldpc_decode_batch returned {rc}")
        return corr[:shots, :n], obs[:shots, :k], st[:shots]

    def soft_output(self) -> np.ndarray:
        n = self.num_mechanisms
        out = np.empty(max(n, 1), dtype=np.float64)
        rc = self._get_soft_output(self._h, out.ctypes.data_as(_C_F64P), ctypes.c_uint32(n))
        if rc != ELDPC_OK:
            raise EldpcError(f"eldpc_get_soft_output returned {rc}")
        return out[:n]

    def keep_warm(self, flags: int) -> None:
        rc = self._keep_warm(self._h, ctypes.c_uint32(flags))
        if rc != ELDPC_OK:
            raise EldpcError(f"eldpc_keep_warm returned {rc}")

    def close(self) -> None:
        if self._h is not None:
            self._destroy(self._h)
            self._h = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - a destructor must not raise
            pass


_LIBRARY: Front | None = None
_LOAD_LOCK = threading.Lock()


def library_path() -> Path:
    """The libeldpc.so installed with this package."""
    return Path(__file__).resolve().parent / "_libs" / "libeldpc.so"


def load(path: str | os.PathLike[str] | None = None, isa: str | None = None) -> Front:
    """Load the eldpc library, once per process.

    ``path`` selects an explicit libeldpc.so (default: the one installed with the package).
    ``isa`` forces ``"avx2"`` or ``"avx512"`` by setting ``ELDPC_ISA`` before the library is
    loaded; it must be given before the first load, because the front reads the variable once.
    """
    global _LIBRARY
    with _LOAD_LOCK:
        if isa is not None:
            if isa not in ("avx2", "avx512"):
                raise EldpcError(f'isa must be "avx2" or "avx512", got {isa!r}')
            if _LIBRARY is not None:
                raise EldpcError("the eldpc library is already loaded; its ISA cannot change in this process")
            os.environ["ELDPC_ISA"] = isa
        if _LIBRARY is None:
            _LIBRARY = Front(Path(path) if path is not None else library_path())
        elif path is not None and Path(path).resolve() != Path(_LIBRARY.path).resolve():
            raise EldpcError(f"the eldpc library is already loaded from {_LIBRARY.path}")
        return _LIBRARY
