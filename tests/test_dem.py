# SPDX-License-Identifier: Apache-2.0
"""The DEM -> (H, O, priors) rule; no GPU needed."""

from __future__ import annotations

import numpy as np
import pytest

from eldpc_testlib import DEM_TEXT, cpu_core
from eldpc._dem import dem_to_arrays

pytestmark = cpu_core


def dense_arrays(dem):
    H, O, priors = dem_to_arrays(dem)
    return H.toarray(), O.toarray(), np.asarray(priors)


def test_dem_text_rule():
    H, O, priors = dense_arrays(DEM_TEXT)
    expected_H = np.array([
        [1, 1, 0, 0, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 1],
    ], dtype=np.uint8)
    expected_O = np.array([[1, 0, 0, 1, 1]], dtype=np.uint8)
    assert H.shape == (3, 5) and O.shape == (1, 5)
    np.testing.assert_array_equal(H, expected_H)
    np.testing.assert_array_equal(O, expected_O)
    np.testing.assert_array_equal(priors, np.array([0.05, 0.05, 0.04, 0.04, 0.03]))


def test_repeat_is_unrolled_with_shift_detectors():
    dem = """
        repeat 2 {
            error(0.1) D0
            shift_detectors 1
        }
        error(0.1) D0
    """
    H, O, priors = dense_arrays(dem)
    # The two copies of the block address D0 and D1; the trailing instruction is D2.
    assert H.shape == (3, 3)
    np.testing.assert_array_equal(H, np.eye(3, dtype=np.uint8))
    assert O.shape == (0, 3)
    np.testing.assert_array_equal(priors, [0.1, 0.1, 0.1])


def test_identical_columns_are_never_merged():
    H, O, priors = dense_arrays("error(0.1) D0\nerror(0.2) D0 L0")
    assert H.shape == (1, 2)
    np.testing.assert_array_equal(H, np.ones((1, 2), dtype=np.uint8))
    np.testing.assert_array_equal(O, np.array([[0, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(priors, [0.1, 0.2])


def test_separator_is_xor_of_components():
    H, O, _ = dense_arrays("error(0.1) D0 ^ D0 D1 L0")
    # D0 cancels between the two components; D1 and L0 remain.
    np.testing.assert_array_equal(H, np.array([[0], [1]], dtype=np.uint8))
    np.testing.assert_array_equal(O, np.array([[1]], dtype=np.uint8))


def test_repeated_target_in_one_component_cancels():
    H, _, _ = dense_arrays("error(0.1) D0 D0 D1")
    np.testing.assert_array_equal(H, np.array([[0], [1]], dtype=np.uint8))


def test_empty_dem():
    H, O, priors = dense_arrays("")
    assert H.shape == (0, 0) and O.shape == (0, 0) and priors.shape == (0,)


def test_accepts_a_stim_model():
    import stim

    model = stim.DetectorErrorModel(DEM_TEXT)
    H1, O1, p1 = dem_to_arrays(model)
    H2, O2, p2 = dem_to_arrays(DEM_TEXT)
    np.testing.assert_array_equal(H1.toarray(), H2.toarray())
    np.testing.assert_array_equal(O1.toarray(), O2.toarray())
    np.testing.assert_array_equal(p1, p2)


def test_bad_text_raises():
    with pytest.raises(ValueError):
        dem_to_arrays("not a detector error model")


def _reference_dem_to_arrays():
    """The frozen reference conversion of the reference construction, when the checkout has it."""
    import sys
    from pathlib import Path

    import os

    repo_python = Path(os.environ.get("ELDPC_REFERENCE_PYTHON", "/nonexistent"))
    if not (repo_python / "dbench" / "demarrays.py").is_file():
        pytest.skip("ELDPC_REFERENCE_PYTHON does not hold the reference DEM conversion")
    sys.path.insert(0, str(repo_python))
    from dbench.demarrays import dem_to_arrays as reference
    return reference


def _check_against_reference(text):
    import stim

    reference = _reference_dem_to_arrays()
    model = stim.DetectorErrorModel(text)
    H1, O1, p1 = dem_to_arrays(model)
    H2, O2, p2 = reference(model)
    np.testing.assert_array_equal(H1.toarray(), H2.toarray())
    np.testing.assert_array_equal(O1.toarray(), O2.toarray())
    np.testing.assert_array_equal(p1, p2)
    assert (H1.shape, O1.shape) == (H2.shape, O2.shape)


def test_conversion_matches_the_reference_rule_on_dem_text():
    _check_against_reference(DEM_TEXT)


def test_conversion_matches_the_reference_rule_on_repeats_and_shifts():
    _check_against_reference("""
        repeat 3 {
            error(0.02) D0 L0
            error(0.03) D0 D1 ^ D2
            shift_detectors 2
            error(0.01) D0
        }
        error(0.04) D0 D5 L1
    """)


def test_conversion_matches_the_reference_rule_on_a_circuit_model():
    import stim

    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                                     after_clifford_depolarization=0.001,
                                     before_measure_flip_probability=0.001,
                                     after_reset_flip_probability=0.001)
    _check_against_reference(str(circuit.detector_error_model(decompose_errors=False)))
