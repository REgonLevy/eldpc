# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the eldpc test suite (problems, markers, reference paths).

Imported by name (``from eldpc_testlib import ...``); conftest.py holds only the fixtures. A
module name of its own keeps it apart from tests/qldpc/conftest.py when the whole suite runs.

GPU tests skip with a clear reason when no CUDA device is visible. The bit-identity tests also
need the separately built reference library of the same decoder sources; they skip when it is not
there.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

TESTS = Path(__file__).resolve().parent
PACKAGE = TESTS.parent          # the package root
# ELDPC_REFERENCE_LIB: a separately built library of the same decoder sources with the decoder
# ABI v1 entry points (dbench_*); the bit-identity tests skip when it is not set.
REFERENCE_LIB = Path(os.environ.get("ELDPC_REFERENCE_LIB", "/nonexistent"))
CORE_TARBALL = Path(os.environ.get("ELDPC_CORE_TARBALL", "/nonexistent"))
CORE_SNAPSHOT = Path(os.environ.get("ELDPC_CORE_SNAPSHOT", "/nonexistent"))


def _usable_gpu():
    """``(devices, None)`` when device 0 can run this build's GPU cores, else ``([], why)``.

    The rule is the one of ``device="auto"`` (eldpc._auto.choose): a build with GPU cores, an NVIDIA
    driver and a visible device, a driver new enough for the cores' CUDA version, and a compute
    capability the cores carry an image for. A visible GPU the build cannot run on (for example
    an RTX 3090, compute capability 8.6) therefore counts as no GPU: the GPU tests skip and say
    why, and the device-independent tests run on the CPU core (DEVICE_KWARGS).
    """
    try:
        import eldpc
        from eldpc import _auto

        device, reason = _auto.choose(_auto.probe(eldpc.build_info()))
        if device != "gpu":
            return [], reason
        devices = eldpc.devices()
        return (devices, None) if devices else ([], "eldpc.devices() lists no CUDA device")
    except Exception as exc:  # noqa: BLE001 - no library, no runtime, no device: all mean "skip"
        return [], f"the eldpc library or the CUDA driver could not be queried ({type(exc).__name__}: {exc})"


#: The CUDA devices when device 0 is usable by this build's GPU cores, else empty; NO_GPU_REASON
#: says why it is empty.
CUDA_DEVICES, NO_GPU_REASON = _usable_gpu()
# ELDPC_REQUIRE_GPU=1: a machine that should have a working GPU installation fails here instead of
# skipping every GPU test (a missing or broken driver, or an unsupported GPU, otherwise looks like
# "no GPU").
if os.environ.get("ELDPC_REQUIRE_GPU") == "1" and not CUDA_DEVICES:
    raise pytest.UsageError(f"ELDPC_REQUIRE_GPU=1 but there is no GPU this build can run on: "
                            f"{NO_GPU_REASON}")
requires_gpu = pytest.mark.skipif(
    not CUDA_DEVICES,
    reason=f"needs a GPU this build's GPU cores can run on: {NO_GPU_REASON}")
requires_reference_lib = pytest.mark.skipif(
    not REFERENCE_LIB.is_file(),
    reason="ELDPC_REFERENCE_LIB is not set to a built reference library")


def _qldpc_installed() -> bool:
    """Whether qLDPC is installed. While this suite runs, tests/ is on sys.path, so without an
    installed qLDPC the name qldpc resolves to the tests/qldpc directory as a namespace package
    (no origin); that is not qLDPC."""
    spec = importlib.util.find_spec("qldpc")
    return spec is not None and spec.origin not in (None, "namespace")


# The verification suite's corpora (tests/verify) build their circuits with qLDPC; any installed
# qLDPC builds them, so these tests need only its presence (eldpc.qldpc needs a newer one, below).
QLDPC_INSTALLED = _qldpc_installed()
requires_qldpc = pytest.mark.skipif(
    not QLDPC_INSTALLED,
    reason='qLDPC is not installed (it builds the verification corpora); install it with '
           'pip install ".[qldpc]" from the package root (docs/BUILD.md, "Tests")')


def require_qldpc_layer() -> None:
    """Skip the calling test module when qLDPC is missing or older than eldpc.qldpc needs.

    Call it at module level, before any qLDPC import. eldpc.qldpc raises an ImportError naming the
    supported qLDPC and the install command in both cases; that message becomes the skip reason.
    Any other import error is a real failure and propagates.
    """
    try:
        import eldpc.qldpc  # noqa: F401
    except ImportError as exc:
        if "eldpc.qldpc needs" not in str(exc):
            raise
        reason = str(exc)
        # When the name qldpc matches only a directory (tests/qldpc, with the tests run from the
        # package root), the cause is simply that qLDPC is not installed: do not show the path.
        reason = re.sub(r"qLDPC cannot be imported \(the name qldpc resolves to a directory "
                        r"without the package \(.*?\)\)", "qLDPC is not installed", reason)
        pytest.skip(reason, allow_module_level=True)


def tiny_problem():
    """A 3 x 5 problem with one observable: H and O as dense lists, per-column priors."""
    H = np.array([
        [1, 1, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
    ], dtype=np.uint8)
    O = np.array([[0, 1, 0, 1, 1]], dtype=np.uint8)
    priors = np.array([0.05, 0.10, 0.02, 0.20, 0.01])
    return H, O, priors


def random_problem(m, n, k, seed, column_weight=2, prior_lo=0.005, prior_hi=0.2):
    """A random sparse GF(2) problem; every column has `column_weight` H entries and 0-1 O entries."""
    rng = np.random.default_rng(seed)
    rows = []
    cols = []
    for j in range(n):
        for r in rng.choice(m, size=min(column_weight, m), replace=False):
            rows.append(int(r))
            cols.append(j)
    H = sp.csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(m, n), dtype=np.uint8)
    orows, ocols = [], []
    for j in range(n):
        for r in np.flatnonzero(rng.random(k) < 0.5):
            orows.append(int(r))
            ocols.append(j)
    O = sp.csc_matrix((np.ones(len(orows), dtype=np.uint8), (orows, ocols)), shape=(k, n), dtype=np.uint8)
    priors = rng.uniform(prior_lo, prior_hi, size=n)
    return H, O, priors


def demo_problem():
    """A 4 x 8 problem with two observables, varied column weights and priors."""
    return random_problem(4, 8, 2, seed=7, column_weight=2, prior_lo=0.01, prior_hi=0.3)


#: The device of the device-independent tests (the API and qLDPC tests): the GPU core when device
#: 0 can run it (CUDA_DEVICES), else the CPU core. The same tests thus check the GPU core on a
#: supported GPU and the CPU core on a machine without one (no device, ``CUDA_VISIBLE_DEVICES`` set
#: empty, a CPU-only build, or a GPU the build has no image for); the tests in test_cpu_*.py name
#: ``device="cpu"`` themselves and check the CPU core on every machine.
DEVICE_KWARGS = {} if CUDA_DEVICES else {"device": "cpu"}

#: The registered pytest marker of a test that needs no CUDA device: with none visible it runs on
#: the CPU core. A marked test does not skip for want of a GPU or of a separately built reference
#: library, so ``CUDA_VISIBLE_DEVICES= pytest -m cpu_core`` runs every one of them (the qLDPC tests
#: among them need the ``qldpc`` extra installed).
cpu_core = pytest.mark.cpu_core


def sampled_syndromes(H, priors, shots, seed):
    """Sample (shots, m) syndromes by flipping each column independently with its prior."""
    H = sp.csc_matrix(H, dtype=np.int64)
    m, n = H.shape
    rng = np.random.default_rng(seed)
    errors = (rng.random((shots, n)) < np.asarray(priors, dtype=np.float64)[None, :]).astype(np.uint8)
    dense = H.toarray().astype(np.uint8) % 2
    return (errors @ dense.T % 2).astype(np.uint8)


DEM_TEXT = """\
error(0.05) D0 L0
error(0.05) D0 D1
error(0.04) D1 D2
error(0.04) D2 L0
error(0.03) D0 D1 D2 L0
"""
