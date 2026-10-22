# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the tests of the eldpc qLDPC layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import stim

# The shared helpers (eldpc_testlib) live in the parent tests directory. pytest adds it to the
# import path when it loads that directory's conftest; make it explicit so these tests also work
# when this directory alone is passed to pytest.
_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))


@pytest.fixture(scope="session")
def repetition_dem_and_shots() -> tuple[stim.DetectorErrorModel, np.ndarray, np.ndarray]:
    """A detector error model of a small repetition-code memory experiment and sampled shots."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    detections, observables = circuit.compile_detector_sampler(seed=1234).sample(
        200, separate_observables=True
    )
    return dem, detections, observables


@pytest.fixture(scope="session")
def repetition_shots() -> tuple[np.ndarray, np.ndarray]:
    """Sampled detection events and observables of a repetition-code memory experiment."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    return circuit.compile_detector_sampler(seed=1234).sample(200, separate_observables=True)
