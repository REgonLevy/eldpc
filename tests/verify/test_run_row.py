# SPDX-License-Identifier: Apache-2.0
"""The row comparison: a decoder that is exactly the reference passes, a broken one fails.

``run_row`` is exercised directly with a test double for the ``eldpc`` module whose Decoder wraps the
packaged reference library. This checks the comparison bookkeeping (single shot and batch against
the reference, single vs batch) without needing a GPU, and that a deliberately wrong correction is
counted instead of being silently accepted.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]      # the package root (tests/verify/x.py -> root)
if str(ROOT / "verify") not in sys.path:
    sys.path.insert(0, str(ROOT / "verify"))

from eldpc_verify import runner  # noqa: E402
from eldpc_verify.abi import AbiDecoder  # noqa: E402
from eldpc_verify.config import load_config  # noqa: E402
from eldpc_verify.corpus import ensure_corpus  # noqa: E402
from eldpc_verify.reference import build_reference  # noqa: E402
from eldpc_testlib import requires_qldpc  # noqa: E402

pytestmark = requires_qldpc  # the corpus fixture is built with qLDPC

A03125C3 = {"decoder": "bposd", "bp_method": "ms", "schedule": "parallel",
            "max_iter": 3, "ms_scaling_factor": 0.3125, "osd_method": "osd0", "osd_order": 0}
PRESETS = {"a03125c3": A03125C3}
SHOTS = 8


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return build_reference(tmp_path_factory.mktemp("reference-cache"), jobs=2)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    cfg = load_config()
    problem = next(p for p in cfg["problems"] if p["id"] == "bb72_unif_p1e-3_r6_z")
    return ensure_corpus(tmp_path_factory.mktemp("corpus-cache"), problem, cfg["base_seed"], SHOTS)


def _module(reference, corrupt=False):
    class FakeDecoder:
        def __init__(self, H, O, priors, *, config="a03125c3", device="gpu"):
            self.device = device
            self._d = AbiDecoder(reference.path, runner.config_string(PRESETS[config]),
                                 H, O, priors)
            self.last_status = 0
            self._shot = 0

        def decode_full(self, syndrome):
            obs, corr, status = self._d.decode(syndrome)
            if corrupt and self._shot == 0:
                corr = corr.copy()
                corr[0] ^= 1
            self._shot += 1
            self.last_status = status
            return corr, obs, status

        def decode_batch(self, syndromes):
            # the public API returns (corrections, observables, status)
            obs, corr, status = self._d.decode_batch(syndromes)
            return corr, obs, status

        def close(self):
            self._d.close()

    return types.SimpleNamespace(Decoder=FakeDecoder)


def test_exact_decoder_has_no_mismatches(reference, corpus):
    row = runner.run_row(_module(reference), PRESETS, corpus, "a03125c3", reference.path,
                         None, False, lambda *a: None)
    assert row["problem"] == "bb72_unif_p1e-3_r6_z"
    assert row["config"] == "a03125c3"
    assert row["shots"] == SHOTS
    assert row["mismatches"] == 0
    assert row["correction_mismatches"] == 0
    assert row["observables_mismatches"] == 0
    assert row["batch_correction_mismatches"] == 0
    assert row["batch_observables_mismatches"] == 0
    assert row["single_batch_mismatches"] == 0
    assert row["status_mismatches_vs_reference"] == 0
    assert row["status_mismatches_vs_reference_lib"] is None


def test_a_corrupted_correction_is_counted(reference, corpus):
    row = runner.run_row(_module(reference, corrupt=True), PRESETS, corpus, "a03125c3",
                         reference.path, None, False, lambda *a: None)
    assert row["mismatches"] > 0
    assert row["correction_mismatches"] >= 1
    assert 0 in row["first_mismatched_shots"]
    # The single vs batch comparison catches it too: only the single path was corrupted.
    assert row["single_batch_mismatches"] >= 1


def test_the_device_is_passed_on_and_recorded(reference, corpus):
    """--device cpu (or auto) reaches eldpc.Decoder and every row records the engine it ran on;
    the default passes no device argument (the package's default, the GPU core)."""
    row = runner.run_row(_module(reference), PRESETS, corpus, "a03125c3", reference.path,
                         None, False, lambda *a: None)
    assert row["device"] == "gpu" and row["mismatches"] == 0
    row = runner.run_row(_module(reference), PRESETS, corpus, "a03125c3", reference.path,
                         None, False, lambda *a: None, device="cpu")
    assert row["device"] == "cpu" and row["mismatches"] == 0


def test_a_gpu_create_error_points_to_the_cpu_core(reference, corpus):
    class Unsupported:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("eldpc: device 0 (Some GPU) has compute capability 8.6, which the GPU "
                               "cores of this build do not support: ...; to run on the CPU core, "
                               "create the decoder with device=cpu")
    module = types.SimpleNamespace(Decoder=Unsupported)
    with pytest.raises(runner.VerifyError, match="--device cpu"):
        runner.run_row(module, PRESETS, corpus, "a03125c3", reference.path, None, False,
                       lambda *a: None)
    with pytest.raises(runner.VerifyError) as info:
        runner.run_row(module, PRESETS, corpus, "a03125c3", reference.path, None, False,
                       lambda *a: None, device="cpu")
    assert "--device cpu" not in str(info.value)
