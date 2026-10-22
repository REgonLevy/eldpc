# SPDX-License-Identifier: Apache-2.0
"""The suite's corpus rules are the reference construction's, on every suite problem.

The comparison builds each problem's circuit twice -- once with the suite's own port and once with
the reference construction's corpus code (the ``dbench`` Python package) -- and checks that the
circuits, the DEMs, the matrices, the priors and a seeded shot set are identical. Loss problems
include the heralded erasure insertion: the suite shifts the measurement records after the final
measurement (``insert_final_herald``), and when the reference construction's circuit differs it is
compared after ``correct_herald_records`` applies that shift to it (an older reference construction
copies the targets unchanged); window problems are compared with the reference construction's window
extraction on the parent's stored entry.

The reference construction package is optional and never imported by the suite itself. Set
``ELDPC_REFERENCE_PYTHON`` to the reference construction's ``python/`` directory (the one that
contains ``dbench/``) to run these tests; without it the whole module is skipped, and ``pytest -rs``
shows the reason. The pass/skip counts of a run therefore depend on that variable.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]      # the package root (tests/verify/x.py -> root)
if str(ROOT / "verify") not in sys.path:
    sys.path.insert(0, str(ROOT / "verify"))

_REFERENCE_PYTHON = os.environ.get("ELDPC_REFERENCE_PYTHON")
if not _REFERENCE_PYTHON:
    pytest.skip("ELDPC_REFERENCE_PYTHON is not set (the optional comparison with the build "
                "reference construction's corpus code)", allow_module_level=True)
if not (Path(_REFERENCE_PYTHON) / "dbench" / "demarrays.py").is_file():
    pytest.skip(f"ELDPC_REFERENCE_PYTHON does not contain the dbench package: {_REFERENCE_PYTHON}",
                allow_module_level=True)
sys.path.append(_REFERENCE_PYTHON)

dbench_corpus = pytest.importorskip("dbench.corpus.circuits")
dbench_dem = pytest.importorskip("dbench.corpus.dem")
dbench_sampling = pytest.importorskip("dbench.corpus.sampling")
dbench_spec = pytest.importorskip("dbench.corpus.spec")
dbench_demarrays = pytest.importorskip("dbench.demarrays")
dbench_common = pytest.importorskip("dbench.common")

import stim  # noqa: E402

from eldpc_verify import circuits as verify_circuits  # noqa: E402
from eldpc_verify import dem as verify_dem  # noqa: E402
from eldpc_verify.config import load_config  # noqa: E402
from eldpc_verify.corpus import ensure_corpus  # noqa: E402

CONFIG = load_config()
PROBLEMS = CONFIG["problems"]
BY_ID = {p["id"]: p for p in PROBLEMS}
SHOTS = 8


def _is_window(problem) -> bool:
    return problem.get("shape", {}).get("kind") == "window"


def _dbench_problem(problem):
    """The problem object the reference construction expects (window problems: id + shape only)."""
    if _is_window(problem):
        shape = problem["shape"]
        return {"id": problem["id"],
                "shape": {"kind": "window", "parent": shape["parent"],
                          "start_round": shape["start_round"], "w": shape["w"],
                          "commit": shape["commit"]}}
    return problem


def _reference_specs():
    cfg, specs = dbench_spec.load_problem_specs({
        "schema": dbench_spec.CONFIG_SCHEMA,
        "dev_base_seed": CONFIG["base_seed"],
        "problems": [_dbench_problem(p) for p in PROBLEMS],
    })
    return specs


def _reference_circuit(spec, problem):
    """(circuit, heralds) from the reference construction; for a loss problem whose reference
    circuit lacks the measurement-record shift, the shift is applied (see the module docstring)."""
    circuit, heralds = dbench_corpus.build_circuit(spec)
    if problem.get("loss") is not None:
        ours, _ = verify_circuits.build_circuit_and_heralds(problem)
        if str(circuit) != str(ours):
            circuit = verify_circuits.correct_herald_records(circuit)
    return circuit, heralds


@pytest.mark.parametrize("problem", PROBLEMS, ids=[p["id"] for p in PROBLEMS])
def test_circuit_matches_reference(problem):
    if _is_window(problem):
        pytest.skip("window problems have no circuit of their own")
    spec = next(s for s in _reference_specs() if s.id == problem["id"])
    ours, our_heralds = verify_circuits.build_circuit_and_heralds(problem)
    theirs, their_heralds = _reference_circuit(spec, problem)
    assert str(ours) == str(theirs)
    assert our_heralds == their_heralds
    if problem.get("loss") is not None:
        assert our_heralds, "a loss problem must have herald detectors"


@pytest.mark.parametrize("problem", PROBLEMS, ids=[p["id"] for p in PROBLEMS])
def test_dem_and_arrays_match_reference(problem):
    spec = next(s for s in _reference_specs() if s.id == problem["id"])
    if _is_window(problem):
        shape = problem["shape"]
        parent = BY_ID[shape["parent"]]
        with tempfile.TemporaryDirectory(prefix="eldpc-window-test-") as tmp:
            parent_corpus = ensure_corpus(tmp, parent, CONFIG["base_seed"], SHOTS)
            parent_spec = next(s for s in _reference_specs() if s.id == shape["parent"])
            parent_circuit, parent_heralds = dbench_corpus.build_circuit(parent_spec)
            parent_round, _kind = dbench_corpus.detector_metadata(
                parent_spec, parent_circuit, parent_heralds,
                code=dbench_corpus.build_code(parent_spec))
            assert np.array_equal(parent_corpus.det_round, parent_round)
            win = dbench_dem.extract_window(
                parent_corpus.H, parent_corpus.O, parent_corpus.priors,
                parent_corpus.det_round, shape["start_round"], shape["w"])
            window = ensure_corpus(tmp, problem, CONFIG["base_seed"], SHOTS,
                                   parent=parent, parent_shots=SHOTS)
        assert (window.H != win.H).nnz == 0
        assert (window.O != win.O).nnz == 0
        assert np.array_equal(window.priors, win.priors)
        assert window.manifest["window"]["rows"] == [int(x) for x in win.rows]
        assert window.manifest["window"]["cols"] == [int(x) for x in win.cols]
        assert np.array_equal(window.dets, parent_corpus.dets[:SHOTS][:, win.rows])
        assert np.array_equal(window.obs, parent_corpus.obs[:SHOTS])
        assert window.dem_text == verify_dem.arrays_to_dem_text(win.H, win.O, win.priors)
        return

    our_circuit, our_heralds = verify_circuits.build_circuit_and_heralds(problem)
    ref_circuit, ref_heralds = _reference_circuit(spec, problem)
    assert our_heralds == ref_heralds
    if problem.get("loss") is not None:
        our_round = verify_circuits.detector_rounds(problem, our_circuit, our_heralds)
        ref_round, _kind = dbench_corpus.detector_metadata(spec, ref_circuit, ref_heralds)
        assert np.array_equal(our_round, ref_round)
    our_dem_text = str(verify_circuits.circuit_dem(our_circuit, problem)) + "\n"
    ref_dem = dbench_dem.circuit_dem(ref_circuit, spec)
    ref_dem_text = str(ref_dem) + "\n"
    assert our_dem_text == ref_dem_text

    H, O, priors = verify_dem.dem_to_arrays(stim.DetectorErrorModel(our_dem_text))
    H_ref, O_ref, priors_ref = dbench_demarrays.dem_to_arrays(
        stim.DetectorErrorModel(ref_dem_text))
    assert H.shape == H_ref.shape and O.shape == O_ref.shape
    assert (H != H_ref).nnz == 0
    assert (O != O_ref).nnz == 0
    assert np.array_equal(priors, priors_ref)


@pytest.mark.parametrize("problem", PROBLEMS, ids=[p["id"] for p in PROBLEMS])
def test_seeded_shots_match_reference(problem):
    if _is_window(problem):
        pytest.skip("window shots are the parent's first shots (checked in the window test)")
    spec = next(s for s in _reference_specs() if s.id == problem["id"])
    circuit = verify_circuits.build_circuit(problem)
    seed = dbench_common.derive_seed(CONFIG["base_seed"], "dev", problem["id"], "latency")
    assert verify_circuits.derive_seed(CONFIG["base_seed"], "dev", problem["id"], "latency") == seed
    our_dets, our_obs = verify_circuits.sample_shots(circuit, SHOTS, seed)
    ref_circuit, _heralds = _reference_circuit(spec, problem)
    ref_dets, ref_obs = dbench_sampling.sample_shots(ref_circuit, SHOTS, seed)
    assert np.array_equal(our_dets, ref_dets)
    assert np.array_equal(our_obs, ref_obs)
