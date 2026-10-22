# SPDX-License-Identifier: Apache-2.0
"""GPU and CPU handles in one process: bit-identical corrections, observables and status words.

These tests need a CUDA device (they are not marked ``cpu_core`` and skip, with a reason, when no
device is visible).
"""

from __future__ import annotations

import numpy as np
import pytest

import eldpc
from eldpc_testlib import DEM_TEXT, demo_problem, random_problem, requires_gpu, sampled_syndromes

pytestmark = requires_gpu


def _same(a, b, what):
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y)), f"{what} differ"


def _check_pair(gpu, cpu, syndromes):
    _same(gpu.decode_batch(syndromes), cpu.decode_batch(syndromes), "batch outputs")
    for i in range(syndromes.shape[0]):
        g = gpu.decode_full(syndromes[i])
        c = cpu.decode_full(syndromes[i])
        assert np.array_equal(g[0], c[0]), f"correction differs at shot {i}"
        assert np.array_equal(g[1], c[1]), f"observables differ at shot {i}"
        assert int(g[2]) == int(c[2]), f"status differs at shot {i}: {g[2]} vs {c[2]}"


@pytest.mark.parametrize("config_name", ["a03125c3", "ms30"])
def test_gpu_and_cpu_handles_agree_in_one_process(config_name):
    H, O, priors = random_problem(12, 30, 3, seed=51, column_weight=3)
    syndromes = sampled_syndromes(H, priors, 40, seed=52)
    gpu = eldpc.Decoder(H, O, priors, config=config_name)
    cpu = eldpc.Decoder(H, O, priors, config=config_name, device="cpu", threads=2)
    try:
        assert gpu.device == "gpu" and gpu.core_variant in ("default", "ada64")
        assert cpu.device == "cpu" and cpu.core_variant == "cpu"
        # Both handles are alive in this process; decode on each in turn.
        _check_pair(gpu, cpu, syndromes)
        _check_pair(cpu, gpu, syndromes)
    finally:
        gpu.close()
        cpu.close()


def test_gpu_and_cpu_handles_agree_on_a_dem():
    import stim

    dem = stim.DetectorErrorModel(DEM_TEXT)
    H, O, priors = eldpc.dem_to_arrays(dem)
    syndromes = sampled_syndromes(H, priors, 30, seed=53)
    gpu = eldpc.Decoder.from_dem(dem, config="a03125c3")
    cpu = eldpc.Decoder.from_dem(dem, config="a03125c3", device="cpu", threads=4)
    try:
        _check_pair(gpu, cpu, syndromes)
    finally:
        gpu.close()
        cpu.close()


def test_gpu_and_cpu_agree_at_every_preset():
    H, O, priors = demo_problem()
    syndromes = sampled_syndromes(H, priors, 10, seed=54)
    for name in sorted(eldpc.presets()):
        gpu = eldpc.Decoder(H, O, priors, config=name)
        cpu = eldpc.Decoder(H, O, priors, config=name, device="cpu", threads=1)
        try:
            assert gpu.device == "gpu" and cpu.device == "cpu"
            _check_pair(gpu, cpu, syndromes)
        finally:
            gpu.close()
            cpu.close()
