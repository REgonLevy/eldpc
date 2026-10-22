# SPDX-License-Identifier: Apache-2.0
"""config.json is loadable by ``eldpc.Decoder(config=...)`` and gives identical outputs."""
import json
import os

import numpy as np
import pytest

import eldpc

from eldpctune import grid, problems

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="module")
def small_problem():
    prob = problems.problem_from_circuit_file(os.path.join(DATA, "surface_d3_r2.stim"))
    try:
        decoder = eldpc.Decoder(prob.H, prob.O, prob.priors,
                                config=grid.Point("bposd", 0.5, 3).config())
        decoder.close()
    except Exception as exc:  # noqa: BLE001 - any failure means no usable decoder here
        pytest.skip(f"no GPU decoder available: {exc}")
    return prob


def _outputs(decoder, dets):
    return [decoder.decode_full(shot) for shot in dets]


@pytest.mark.parametrize("point", [grid.Point("bposd", 0.5, 3), grid.Point("bp", 0.3125, 3)])
def test_config_json_roundtrip(tmp_path, small_problem, point):
    config = point.config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    dets = problems.unpack_bits(small_problem.shot_source(101).sample(8)[0], small_problem.m)

    first = eldpc.Decoder(small_problem.H, small_problem.O, small_problem.priors, config=config)
    try:
        expected = _outputs(first, dets)
    finally:
        first.close()
    second = eldpc.Decoder(small_problem.H, small_problem.O, small_problem.priors,
                           config=str(path))
    loaded_config = dict(second.config)
    try:
        actual = _outputs(second, dets)
    finally:
        second.close()
    assert loaded_config == config
    for expected_shot, actual_shot in zip(expected, actual):
        assert np.array_equal(expected_shot[0], actual_shot[0])
        assert np.array_equal(expected_shot[1], actual_shot[1])
        assert int(expected_shot[2]) == int(actual_shot[2])


def test_arrays_path_matches_from_dem(small_problem):
    """The tuner's H/O/priors port gives the same decoder as the public from_dem constructor."""
    import stim
    point = grid.Point("bposd", 0.5, 3)
    dem = stim.DetectorErrorModel(small_problem.dem_text)
    dets = problems.unpack_bits(small_problem.shot_source(202).sample(20)[0], small_problem.m)
    from_dem = eldpc.Decoder.from_dem(dem, config=point.config())
    from_arrays = eldpc.Decoder(small_problem.H, small_problem.O, small_problem.priors,
                                config=point.config())
    try:
        expected = _outputs(from_dem, dets)
        actual = _outputs(from_arrays, dets)
    finally:
        from_dem.close()
        from_arrays.close()
    for expected_shot, actual_shot in zip(expected, actual):
        assert np.array_equal(expected_shot[0], actual_shot[0])
        assert np.array_equal(expected_shot[1], actual_shot[1])
        assert int(expected_shot[2]) == int(actual_shot[2])


def test_written_config_is_a_plain_abi_dict(small_problem):
    point = grid.Point("bposd", 0.3125, 5)
    config = point.config()
    assert set(config) <= {"decoder", "bp_method", "max_iter", "ms_scaling_factor",
                           "schedule", "osd_method", "osd_order"}
    assert isinstance(config["ms_scaling_factor"], float)
    assert isinstance(config["max_iter"], int)
    decoder = eldpc.Decoder(small_problem.H, small_problem.O, small_problem.priors, config=config)
    decoder.close()
