# SPDX-License-Identifier: Apache-2.0
"""Problem recipes, DEM conversion, matrix files and the code-capacity logical basis."""
import importlib.util
import os

import numpy as np
import pytest
import scipy.sparse as sp
import stim

from eldpctune import problems

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#: The built-in recipes build their circuits with qLDPC, which is optional.
needs_qldpc = pytest.mark.skipif(importlib.util.find_spec("qldpc") is None,
                                 reason="the built-in recipes need the qldpc package")


@needs_qldpc
def test_builtin_problem_sizes():
    # Detector and observable counts of the circuits these recipes define.
    expected = {
        "bb72": (252, 12),
        "bb144": (936, 12),
        "bb144xyz": (1872, 24),
        "hgp98": (245, 18),
        "sc5": (120, 1),
    }
    for name, (m, k) in expected.items():
        prob = problems.build_problem(name, 0.001, None)
        assert (prob.m, prob.k) == (m, k), name
        assert prob.H.shape == (prob.m, prob.n)
        assert prob.O.shape == (prob.k, prob.n)
        assert prob.priors.shape == (prob.n,)
        assert np.all((prob.priors > 0) & (prob.priors < 1))
        assert prob.n > 0


@needs_qldpc
def test_rounds_override_changes_the_detector_count():
    five = problems.build_problem("sc5", 0.001, 5)
    seven = problems.build_problem("sc5", 0.001, 8)
    assert seven.m > five.m


def test_dem_to_arrays_rule():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D2 L0
        error(0.2) D1 ^ D0 L1
        error(0.3) D2
        detector D2
        logical_observable L1
    """)
    H, O, priors = problems.dem_to_arrays(dem)
    assert (H.shape, O.shape) == ((3, 3), (2, 3))
    assert np.allclose(priors, [0.1, 0.2, 0.3])
    Hd = H.toarray()
    assert list(Hd[0]) == [1, 1, 0]   # column 1 has D1 ^ D0: the separator XORs to D0
    assert list(Hd[1]) == [0, 1, 0]   # column 1 is D1 ^ D0: both detectors flip
    assert list(Hd[2]) == [1, 0, 1]
    Od = O.toarray()
    assert list(Od[0]) == [1, 0, 0]
    assert list(Od[1]) == [0, 1, 0]


@needs_qldpc
def test_dem_arrays_match_the_shot_source():
    prob = problems.build_problem("sc5", 0.01, 2)
    source = prob.shot_source(11)
    dets_p, obs_p = source.sample(50)
    dets = problems.unpack_bits(dets_p, prob.m)
    obs = problems.unpack_bits(obs_p, prob.k)
    assert dets.shape == (50, prob.m)
    assert obs.shape == (50, prob.k)
    assert set(np.unique(dets)) <= {0, 1}
    # the same seed gives the same shots; another seed does not
    again = problems.unpack_bits(prob.shot_source(11).sample(50)[0], prob.m)
    other = problems.unpack_bits(prob.shot_source(12).sample(50)[0], prob.m)
    assert np.array_equal(dets, again)
    assert not np.array_equal(dets, other)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    a = (rng.random((37, 13)) < 0.5).astype(np.uint8)
    packed = problems.pack_bits(a)
    assert packed.shape == (37, 2)
    assert np.array_equal(problems.unpack_bits(packed, 13), a)


def test_text_matrix_file(tmp_path):
    path = tmp_path / "h.txt"
    path.write_text("# a comment\n3 5\n1 0 1 0 1\n0 1 1 0 0\n1 1 0 0 1\n")
    H = problems.read_matrix_file(str(path))
    assert H.shape == (3, 5)
    assert np.array_equal(H.toarray(), [[1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 0, 1]])
    bad = tmp_path / "bad.txt"
    bad.write_text("3 5\n1 0 1\n")
    with pytest.raises(ValueError):
        problems.read_matrix_file(str(bad))
    bad2 = tmp_path / "bad2.txt"
    bad2.write_text("1 3\n1 2 0\n")
    with pytest.raises(ValueError):
        problems.read_matrix_file(str(bad2))


def test_logicals_from_a_self_orthogonal_h():
    # A single even-weight row: ker(H)/row(H) has dimension n - 2 = 2.
    H = sp.csc_matrix(np.asarray([[1, 1, 0, 0]], dtype=np.uint8))
    O = problems.logical_observables_from_h(H)
    assert O.shape == (2, 4)
    assert (H.astype(np.int32) @ O.T.astype(np.int32)).nnz == 0
    # H and O are independent: rank([H; O]) = 3 over GF(2)
    stacked = sp.vstack([H, O]).toarray().astype(np.uint8)
    packed = problems._packed_rows(stacked)
    basis = {}
    independent = sum(1 for row in packed if problems._basis_insert(basis, row))
    assert independent == 3


def test_logicals_need_self_orthogonality():
    H = sp.csc_matrix(np.asarray([[1, 0, 1, 1]], dtype=np.uint8))
    with pytest.raises(ValueError, match="self-orthogonal"):
        problems.logical_observables_from_h(H)


def test_code_capacity_problem_from_h(tmp_path):
    path = tmp_path / "h.txt"
    path.write_text("1 4\n1 1 0 0\n")
    prob = problems.problem_from_h_file(str(path), 0.1)
    assert (prob.m, prob.n, prob.k) == (1, 4, 2)
    assert np.allclose(prob.priors, 0.1)
    dets_p, obs_p = prob.shot_source(3).sample(200)
    dets = problems.unpack_bits(dets_p, 1)
    obs = problems.unpack_bits(obs_p, 2)
    assert dets.shape == (200, 1) and obs.shape == (200, 2)
    assert set(np.unique(dets)) <= {0, 1}
    assert set(np.unique(obs)) <= {0, 1}


def test_observable_file_for_h(tmp_path):
    h = tmp_path / "h.txt"
    h.write_text("1 3\n1 1 1\n")  # not self-orthogonal, but O is supplied
    o = tmp_path / "o.txt"
    o.write_text("1 3\n1 0 0\n")
    prob = problems.problem_from_h_file(str(h), 0.2, str(o))
    assert (prob.m, prob.n, prob.k) == (1, 3, 1)
    with pytest.raises(ValueError, match="self-orthogonal"):
        problems.problem_from_h_file(str(h), 0.2)


def test_committed_input_files():
    """The committed demonstration inputs are read by the three file paths."""
    circuit = problems.problem_from_circuit_file(os.path.join(DATA, "surface_d3_r2.stim"))
    assert (circuit.m, circuit.k) == (16, 1)
    assert circuit.circuit is not None and circuit.n > 0
    dem = problems.problem_from_dem_file(os.path.join(DATA, "surface_d3_r2.dem"))
    assert (dem.m, dem.k) == (16, 1)
    assert dem.circuit is None and dem.n > 0
    h = problems.problem_from_h_file(os.path.join(DATA, "h_selforthogonal.txt"), 0.05)
    assert (h.m, h.n, h.k) == (1, 4, 2)
    assert h.meta["observables_derived"] is True


@needs_qldpc
def test_surface_code_uniform_noise_override():
    default = problems.build_problem("sc5", 0.001, None)
    uniform = problems.build_problem("sc5", 0.001, None, noise_model="uniform")
    assert uniform.meta["noise_model"] == "uniform"
    assert default.meta["noise_model"] == "si1000"
    assert uniform.key != default.key and uniform.key.endswith(":noise=uniform")
    assert (uniform.m, uniform.k) == (default.m, default.k)
    # the two circuits differ: uniform depolarizing is not SI1000
    assert uniform.dem_sha256() != default.dem_sha256()
    # an explicit si1000 is the default recipe and keeps its key
    same = problems.build_problem("sc5", 0.001, None, noise_model="si1000")
    assert same.key == default.key


@needs_qldpc
def test_uniform_only_families_reject_other_models():
    with pytest.raises(ValueError, match="uniform"):
        problems.build_problem("bb72", 0.001, None, noise_model="si1000")
    with pytest.raises(ValueError, match="si1000 and uniform"):
        problems.build_problem("sc5", 0.001, None, noise_model="custom")


def test_circuit_dem_is_undecomposed_by_default(tmp_path):
    """--circuit uses decompose_errors=False; --decompose changes both the DEM and its mechanisms."""
    from eldpctune.runner import TuneConfig, build_cases

    path = os.path.join(DATA, "surface_d3_r2.stim")
    plain = build_cases(TuneConfig(circuit=path, out=str(tmp_path), quick=True))[0]["problem"]
    decomposed = build_cases(TuneConfig(circuit=path, out=str(tmp_path), quick=True,
                                        decompose=True))[0]["problem"]
    assert plain.summary()["dem_options"] == {"decompose_errors": False,
                                              "approximate_disjoint_errors": True}
    assert decomposed.summary()["dem_options"]["decompose_errors"] is True
    assert plain.dem_sha256() != decomposed.dem_sha256()
    assert plain.n != decomposed.n


def test_a_dem_file_records_no_derived_options():
    prob = problems.problem_from_dem_file(os.path.join(DATA, "surface_d3_r2.dem"))
    assert prob.summary()["dem_options"] is None


def test_dem_options_dict_is_what_is_passed_to_stim():
    assert problems.dem_options() == {"decompose_errors": False,
                                      "approximate_disjoint_errors": True}
    assert problems.dem_options(True) == {"decompose_errors": True,
                                          "approximate_disjoint_errors": True,
                                          "ignore_decomposition_failures": True}


def test_decompose_is_refused_where_there_is_no_derived_dem(tmp_path):
    from eldpctune.runner import TuneConfig

    with pytest.raises(ValueError, match="--decompose applies"):
        TuneConfig(dem="x.dem", decompose=True, out=str(tmp_path)).validate()
    with pytest.raises(ValueError, match="--decompose applies"):
        TuneConfig(h="x", ps=[0.1], decompose=True, out=str(tmp_path)).validate()
    TuneConfig(circuit="x.stim", decompose=True, out=str(tmp_path)).validate()
    TuneConfig(problem="bb72", decompose=True, out=str(tmp_path)).validate()


def test_cli_lists_the_confirmation_pruning_and_dem_options():
    from eldpctune.cli import build_parser

    parser = build_parser()
    help_text = " ".join(parser.format_help().split())
    for option in ("--confirm-shots", "--confirm-candidates", "--no-prune", "--decompose",
                   "--no-decompose"):
        assert option in help_text
    base = ["--circuit", "x", "--out", "/tmp/x"]
    assert parser.parse_args(base).decompose is False
    assert parser.parse_args(base + ["--decompose"]).decompose is True
    assert parser.parse_args(base + ["--decompose", "--no-decompose"]).decompose is False
    assert parser.parse_args(base).no_prune is False
    assert parser.parse_args(base + ["--no-prune"]).no_prune is True
    assert parser.parse_args(base).confirm_candidates is None     # no cap by default
