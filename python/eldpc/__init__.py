# SPDX-License-Identifier: Apache-2.0
"""eldpc: exact GPU BP+OSD decoding for quantum LDPC codes.

A :class:`Decoder` holds one qLDPC detector error model (H, O, priors) and a configuration, and
decodes syndromes one shot at a time or in batches. The native library behind it, libeldpc.so,
loads the AVX-512 build of the fastbpgpu6g decoder when the CPU and OS support it, else the AVX2
build (``ELDPC_ISA=avx2|avx512`` forces one; see :func:`isa`). Each handle then runs on one of two
devices: a GPU handle runs one of two core variants, ``"default"``, or ``"ada64"`` for the 142-SM
Ada parts (see :attr:`Decoder.core_variant`; ``ELDPC_DEVICE_CLASS=default|ada64`` forces the
choice); a CPU handle (``device="cpu"``) runs the optional CPU core on the host;
``device="auto"`` uses the GPU when it is usable and the CPU core otherwise. Results are exact
and reproducible: every shot returns the same correction, observables and status word as the
decoder's reference implementation at the same settings, whichever device and variant run it.

Example::

    import eldpc
    import stim

    dem = stim.DetectorErrorModel(...)
    with eldpc.Decoder.from_dem(dem, config="a03125c3") as dec:
        correction = dec.decode(syndrome)
        observables, status = dec.decode_observables(syndrome), dec.last_status
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.sparse as sp

from . import _abi, _auto, _check, _dem, _devices, _placement
from ._abi import EldpcError
from ._config import config_string as _config_string
from ._config import presets, resolve_config as _resolve_config
from ._dem import dem_to_arrays
from ._matrix import build_problem, canonical_csc
from ._status import (
    ST_ABSTAIN,
    ST_BP_CONVERGED,
    ST_FLAGS_MASK,
    ST_HIGHER_ORDER_WIN,
    ST_ITER_MAX,
    ST_ITER_SHIFT,
    ST_POSTPROC_INVOKED,
    ST_SYNDROME_UNMATCHED,
    status,
)

__version__ = "0.1.0"

#: ``eldpc_keep_warm()`` flag: enqueue one empty kernel on the handle's GPU stream.
KEEP_WARM_GPU = 1

__all__ = [
    "Decoder",
    "EldpcError",
    "__version__",
    "build_info",
    "check",
    "dem_to_arrays",
    "devices",
    "isa",
    "load",
    "presets",
    "status",
    "KEEP_WARM_GPU",
    "ST_BP_CONVERGED",
    "ST_POSTPROC_INVOKED",
    "ST_HIGHER_ORDER_WIN",
    "ST_ABSTAIN",
    "ST_SYNDROME_UNMATCHED",
    "ST_FLAGS_MASK",
    "ST_ITER_SHIFT",
    "ST_ITER_MAX",
]


def load(path=None, isa=None):
    """Load the native library explicitly (optional; otherwise the first :class:`Decoder` loads it).

    ``path`` selects a libeldpc.so other than the installed one. ``isa`` forces ``"avx2"`` or
    ``"avx512"`` by setting ``ELDPC_ISA`` before the library is loaded; because the C front reads
    that variable once, ``isa`` works only before the first load in the process.
    """
    return _abi.load(path=path, isa=isa)


def build_info() -> str:
    """Compiler, CUDA toolkit, GPU architectures, host ISA and core identity of the loaded library.

    ``core=`` names the process's core: the default GPU core, or the CPU core in a CPU-only build
    (the ``ELDPC_WITHOUT_CUDA`` build, which has no GPU cores). Raises :class:`EldpcError` with
    the library's message when it could load no core (for example ``ELDPC_ISA`` names an ISA this
    CPU and OS do not support, or a value other than ``avx2`` and ``avx512``).
    """
    front = load()
    error = front.load_error()
    if error is not None:
        raise EldpcError(error)
    return front.build_info()


def isa() -> str:
    """The decoder core this process loaded: ``"avx512"`` or ``"avx2"``.

    Raises :class:`EldpcError` with the library's message when it could load no core (see
    :func:`build_info`); the message names ``ELDPC_ISA`` and what this CPU and OS support.
    """
    front = load()
    error = front.load_error()
    if error is not None:
        raise EldpcError(error)
    return front.isa()


def check(file=None) -> bool:
    """Print what this installation built and sees, and whether the GPU is ready.

    The report names the GPU cores (yes or no, their SASS and PTX targets and the CUDA toolkit of
    :func:`build_info`), the CPU cores, the host ISA chosen, the CUDA devices of :func:`devices`,
    the result of a test decode on each usable core, and ends with ``GPU ready: ...`` or
    ``CPU only: <why>``. Run it after installing::

        python -c "import eldpc; eldpc.check()"

    ``file`` is a text stream (default ``sys.stdout``). Returns True when the GPU is ready: the
    build has GPU cores, device 0 can run them and the test decode on it returned the documented
    output; False when only the CPU core is usable (``CPU only: ...``). When no core can decode --
    the library or its decoder core cannot be loaded (for example an ``ELDPC_ISA`` value this CPU
    cannot use), or the CPU core fails its test decode -- the last line is ``NOT READY: <why>``
    and it raises :class:`EldpcError` with that line after printing the report, so the command
    above exits non-zero. It creates a decoder on each usable core, so it initialises the CUDA
    device.
    """
    return _check.check(__version__, file)


def devices() -> list[dict]:
    """The CUDA devices visible to this process, queried through the CUDA driver API.

    Each entry has ``name``, ``uuid`` (8-4-4-4-12 lowercase hex), ``compute_capability``
    (``"major.minor"``), ``sm_count``, ``l2_bytes``, ``cuda_driver_version`` (the CUDA version the installed driver supports, from
    ``cuDriverGetVersion``), ``driver_version`` (an alias of ``cuda_driver_version``),
    ``nvidia_driver_version`` (the installed NVIDIA display driver version from NVML, or None
    with the reason in ``nvidia_driver_version_reason`` when NVML is unavailable) and
    ``core_variant`` (the decoder core variant a handle on that device would get, ``"default"``
    or ``"ada64"``), ``supported`` (whether this build's GPU cores can run on the device: the
    rule of ``device="auto"``, from the loaded library or the build record next to it; None when
    neither is available) and ``supported_reason`` (why not, or None). An empty list means that
    no CUDA driver or no device is available.
    """
    return _devices.devices()


def _resolve_priors(priors, error_rate, n: int) -> np.ndarray:
    if priors is not None and error_rate is not None:
        raise EldpcError("give priors or error_rate, not both")
    if priors is None:
        if error_rate is None:
            raise EldpcError("give priors (n probabilities) or error_rate")
        try:
            priors = np.full(n, float(error_rate))
        except (TypeError, ValueError):
            raise EldpcError(f"error_rate must be a number, got {error_rate!r}") from None
    try:
        p = np.asarray(priors, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise EldpcError("priors must be an array of probabilities") from None
    if p.shape != (n,):
        raise EldpcError(f"priors must have {n} entries (one per H column), got {p.size}")
    if not np.all(np.isfinite(p)) or not np.all((p > 0.0) & (p < 1.0)):
        raise EldpcError("priors must be finite probabilities strictly between 0 and 1")
    return p


#: The values the ``device`` argument of :class:`Decoder` accepts.
DEVICES = ("gpu", "cpu", "auto")

#: The CPU core's thread range (threads per shot, the calling thread included).
MIN_THREADS = 1
MAX_THREADS = 64


def _check_device(device) -> str:
    """Validate the device argument; ValueError before any core is called."""
    if not isinstance(device, str) or device not in DEVICES:
        raise ValueError(f'device must be "gpu", "cpu" or "auto", got {device!r}')
    return device


def _check_threads(device: str, threads):
    """Validate the threads argument for this device; ValueError before any core is called.

    ``None`` means the core's default (1 thread per shot) and is returned unchanged. Threads are a
    CPU-core setting: a GPU handle decodes one shot on its stream and rejects the argument. With
    ``device="auto"`` they are validated here and used only if the CPU core is chosen.
    """
    if threads is None:
        return None
    if device == "gpu":
        raise ValueError('threads is only valid with device="cpu": a GPU handle decodes one shot at '
                         'a time on its stream')
    if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)):
        raise ValueError(f"threads must be an integer between {MIN_THREADS} and {MAX_THREADS}, "
                         f"got {threads!r}")
    value = int(threads)
    if not MIN_THREADS <= value <= MAX_THREADS:
        raise ValueError(f"threads must be between {MIN_THREADS} and {MAX_THREADS}, got {value}")
    return value


def _resolve_device_and_threads(config: dict, device, threads):
    """Take device/threads from a config mapping when present, and validate the pair.

    The C front's ``device`` key is not a decoder-core key, so a config dict may carry it (a C user
    passes it in the config string). The keyword arguments are the Python API; a config that names
    the same setting must agree with them. Returns (config without those keys, device, threads).
    """
    config = dict(config)
    configured_device = config.pop("device", None)
    configured_threads = config.pop("threads", None)
    if configured_device is not None:
        configured_device = _check_device(configured_device)
        if device != "gpu" and device != configured_device:
            raise ValueError(f'device={device!r} conflicts with device={configured_device!r} in the config')
        device = configured_device
    if configured_threads is not None:
        if isinstance(configured_threads, str) and configured_threads.strip().isdigit():
            # An ABI config string's value ("device=cpu;threads=4;..."), as the C front takes it.
            configured_threads = int(configured_threads.strip())
        if device == "gpu":
            raise ValueError('a config with threads= is only valid with device="cpu" (or "auto")')
        configured_threads = _check_threads("cpu", configured_threads)
        if threads is not None and int(threads) != configured_threads:
            raise ValueError(f"threads={threads!r} conflicts with threads={configured_threads!r} in the config")
        threads = configured_threads
    return config, _check_device(device), _check_threads(device, threads)


def _resolve_auto(device: str, threads):
    """Resolve ``device="auto"`` to ``"gpu"`` or ``"cpu"`` (see :mod:`eldpc._auto`).

    Other devices are returned unchanged. When the GPU is chosen, ``threads`` (a CPU-core
    setting) is dropped; when the CPU core is chosen, a warning says why, once per process.
    """
    if device != "auto":
        return device, threads
    chosen, _reason = _auto.resolve(load())
    return chosen, (threads if chosen == "cpu" else None)


def _check_bits(a: np.ndarray, what: str) -> np.ndarray:
    """Validate a 0/1 integer or bool array and return it as a C-contiguous uint8 array."""
    kind = a.dtype.kind
    if kind == "b":
        pass
    elif kind == "u":
        if a.size and int(a.max()) > 1:
            raise EldpcError(f"{what} values must be 0 or 1")
    elif kind == "i":
        if a.size and (int(a.min()) < 0 or int(a.max()) > 1):
            raise EldpcError(f"{what} values must be 0 or 1")
    else:
        raise EldpcError(f"{what} must be an integer or bool array, got dtype {a.dtype}")
    return np.ascontiguousarray(a, dtype=np.uint8)


class Decoder:
    """A BP+OSD-0 decoder for one problem: H (m x n), O (k x n) and n priors.

    ``config`` is a preset name (see :func:`presets`), a dict of ABI config keys, an ABI config
    string (``"decoder=bposd;..."``), a preset name followed by extra ABI keys that add to or
    override the preset's (``"a03125c3;threads=8"``), or a path to a JSON file holding a config dict.

    ``device`` selects the engine: ``"gpu"`` (the default) runs the fastbpgpu6g core on the GPU,
    ``"cpu"`` runs the optional CPU core on the host, for machines without a usable GPU. A CPU
    handle makes no CUDA call, so it works with no NVIDIA driver and with ``CUDA_VISIBLE_DEVICES``
    set empty. ``"auto"`` runs on the GPU when device 0 is usable (an NVIDIA driver and a device,
    a driver for the GPU cores' CUDA major version, a compute capability the cores support, and a
    build with GPU cores) and on the CPU core otherwise, with a ``RuntimeWarning`` once per process
    that says why; :attr:`device` and :attr:`core_variant` report the choice. A config dict or JSON
    file may carry ``"device"`` (``"gpu"``, ``"cpu"`` or ``"auto"``) instead of the argument.
    ``threads`` (1..64, default 1) is the CPU core's threads per shot, the calling thread included;
    it is invalid for a GPU handle, and with ``"auto"`` it applies only if the CPU core is chosen.
    The CPU core starts ``threads - 1`` threads at
    construction and they busy-wait between decodes until :meth:`close`, so give each CPU decoder
    its own cores, and keep them and the calling thread on one CCD (one L3) or one socket.

    A decoder owns one native handle. Use it from one thread at a time; several decoders may be
    used from several threads concurrently. ``close()``, ``with`` and ``__del__`` free the handle.

    A GPU handle runs one core variant, :attr:`core_variant`: ``"default"``, or ``"ada64"`` on the
    142-SM Ada device class (compute capability 8.9 with at least 128 SMs), where the create path
    plans for 64 SMs. Set ``ELDPC_DEVICE_CLASS=default`` or ``ELDPC_DEVICE_CLASS=ada64`` to force
    a variant; it is read at every construction, so handles of both variants can coexist in one
    process. A CPU handle's variant is ``"cpu"``. All of them return bit-identical results.
    """

    def __init__(self, H, O=None, priors=None, *, config="a03125c3", error_rate=None,
                 device="gpu", threads=None):
        # The device arguments are validated first: an argument error is a ValueError and must be
        # raised before any native call (the library may not even be loadable on this machine).
        device = _check_device(device)
        cfg, device, threads = _resolve_device_and_threads(_resolve_config(config), device, threads)
        device, threads = _resolve_auto(device, threads)
        Hc = canonical_csc(H)
        n = Hc.shape[1]
        if O is None:
            Oc = canonical_csc(np.zeros((0, n), dtype=np.uint8))
        else:
            Oc = canonical_csc(O)
            if Oc.shape[1] != n:
                raise EldpcError(f"O must have as many columns as H ({n}), got {Oc.shape[1]}")
        pr = _resolve_priors(priors, error_rate, n)
        self._setup(Hc, Oc, pr, None, cfg, device, threads)

    @classmethod
    def from_dem(cls, dem, *, config="a03125c3", device="gpu", threads=None):
        """Build a decoder from a ``stim.DetectorErrorModel`` or detector error model text.

        H, O and the priors are converted by the rule of :func:`dem_to_arrays`: one column per
        ``error`` instruction of the flattened model, in order, never merged. ``device`` and
        ``threads`` are as for :class:`Decoder`.
        """
        import stim

        device = _check_device(device)
        cfg, device, threads = _resolve_device_and_threads(_resolve_config(config), device, threads)
        device, threads = _resolve_auto(device, threads)
        if isinstance(dem, stim.DetectorErrorModel):
            model = dem
            text = str(dem)
        else:
            text = str(dem)
            try:
                model = stim.DetectorErrorModel(text)
            except (ValueError, IndexError) as exc:
                raise EldpcError(f"not a detector error model: {exc}") from None
        H, O, pr = dem_to_arrays(model)
        self = cls.__new__(cls)
        self._setup(H, O, pr, text, cfg, device, threads)
        return self

    def _setup(self, H, O, priors, dem_text, config, device, threads):
        self.config = dict(config)
        self._device = device
        self._threads = threads
        # The string handed to eldpc_create: the core's keys, then the front's device key and the
        # CPU core's threads key. The GPU default is not written out (device=gpu is the front's
        # default), so a GPU handle's string is the one a GPU-only package would pass.
        create_config = dict(self.config)
        if device == "cpu":
            create_config["device"] = "cpu"
            if threads is not None:
                create_config["threads"] = threads
        self.config_string = _config_string(create_config)
        self.num_detectors, self.num_mechanisms = int(H.shape[0]), int(H.shape[1])
        self.num_observables = int(O.shape[0])
        self.last_status = 0
        self._h = None
        problem, _keepalive = build_problem(H, O, priors, dem_text)
        self._h = _abi.Handle(load(), self.config_string, problem)
        self._core_variant = self._h.core_variant
        if device == "cpu":
            cpus = self.config.get("cpus")
            text = _placement.placement_warning(None if cpus is None else str(cpus), threads)
            if text is not None:
                warnings.warn(text, RuntimeWarning, stacklevel=3)

    @property
    def device(self) -> str:
        """The engine this handle runs on: ``"gpu"`` or ``"cpu"`` (also after ``device="auto"``)."""
        return self._device

    @property
    def threads(self):
        """The CPU core's threads per shot (``None`` on a GPU handle, and when the CPU core's default is used)."""
        return self._threads

    @property
    def core_variant(self) -> str:
        """The decoder core variant this handle runs: ``"default"``, ``"ada64"`` or ``"cpu"``.

        The GPU variant is chosen for each handle when it is created: ``ELDPC_DEVICE_CLASS`` when
        that variable is set to a valid value, otherwise the device-class table's choice for the
        target device (compute capability and SM count). See :func:`devices`. A CPU handle is
        always ``"cpu"``.
        """
        return self._core_variant

    # -- decoding ----------------------------------------------------------

    def _handle(self) -> _abi.Handle:
        h = getattr(self, "_h", None)
        if h is None:
            raise EldpcError("the decoder is closed")
        return h

    def _syndrome(self, syndrome) -> np.ndarray:
        a = np.asarray(syndrome)
        if a.size != self.num_detectors:
            raise EldpcError(f"syndrome must have {self.num_detectors} entries, got {a.size}")
        return _check_bits(a.reshape(-1), "syndrome")

    def decode(self, syndrome) -> np.ndarray:
        """Decode one shot; returns the correction, a uint8 array of ``num_mechanisms`` bits."""
        return self.decode_full(syndrome)[0]

    def decode_observables(self, syndrome) -> np.ndarray:
        """Decode one shot; returns the observable flips, a uint8 array of ``num_observables`` bits."""
        return self.decode_full(syndrome)[1]

    def decode_full(self, syndrome) -> tuple[np.ndarray, np.ndarray, int]:
        """Decode one shot; returns ``(correction, observables, status)``.

        ``status`` is the status word of ``eldpc.h`` (also stored in :attr:`last_status`).
        """
        h = self._handle()
        corr, obs, st = h.decode(self._syndrome(syndrome))
        self.last_status = st
        return corr, obs, st

    def decode_batch(self, syndromes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode a ``(shots, num_detectors)`` array of syndromes.

        Returns ``(corrections (shots, n), observables (shots, k), status (shots,))``; the result
        of every row equals :meth:`decode_full` on that row.
        """
        h = self._handle()
        a = np.asarray(syndromes)
        if a.ndim != 2 or a.shape[1] != self.num_detectors:
            raise EldpcError(f"syndromes must have shape (shots, {self.num_detectors}), got {a.shape}")
        return h.decode_batch(_check_bits(a, "syndromes"))

    def soft_output(self) -> np.ndarray:
        """Posterior LLRs (``num_mechanisms`` doubles) of the most recent single-shot decode."""
        return self._handle().soft_output()

    # -- knobs -------------------------------------------------------------

    def set_priors(self, priors) -> None:
        """Replace the error probabilities of subsequent decodes; ``priors`` has n entries in (0, 1)."""
        h = self._handle()
        p = _resolve_priors(priors, None, self.num_mechanisms)
        # The core converts priors with llr = log((1-p)/p) in C++; math.log is the same libm
        # function, so a round trip through set_priors is exact.
        llr = np.fromiter((math.log((1.0 - x) / x) for x in p.tolist()), dtype=np.float64, count=p.size)
        if not np.all(np.isfinite(llr)):
            raise EldpcError("priors are too close to 0 or 1 for a finite log-likelihood ratio")
        h.set_priors_llr(llr)

    def keep_warm(self) -> None:
        """Keep an idle decoder's GPU clocks up: enqueue one empty kernel on its stream.

        A no-op on a CPU handle: the CPU core has no GPU stream and no idle device work.
        """
        h = self._handle()
        if self._device == "cpu":
            return None
        h.keep_warm(KEEP_WARM_GPU)

    # -- lifetime ----------------------------------------------------------

    def close(self) -> None:
        """Free the decoder handle. Idempotent; the decoder cannot be used afterwards."""
        h = getattr(self, "_h", None)
        if h is not None:
            self._h = None
            h.close()

    def __enter__(self) -> "Decoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - a destructor must not raise
            pass

    def __repr__(self) -> str:
        return (f"<eldpc.Decoder m={self.num_detectors} n={self.num_mechanisms} "
                f"k={self.num_observables} config={self.config_string!r}>")


# A build without GPU cores because no nvcc was found at build time, on a machine whose CUDA driver
# reports a GPU: say so once, at import (eldpc._check; never raises, loads no eldpc library).
_check.warn_at_import()
