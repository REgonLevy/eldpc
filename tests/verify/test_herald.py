# SPDX-License-Identifier: Apache-2.0
"""The heralded-erasure insertion of the loss problems: record targets and the p_erase = 0 check.

``HERALDED_ERASE`` writes one measurement record per qubit, so inserting it before the final
measurement moves every earlier record one step further back for each herald. The insertion
rewrites the later record targets accordingly; these tests check the rewrite on a small circuit
and check, on every loss problem of the suite, that with p_erase = 0 the inserted circuit's
detector error model describes exactly the clean circuit's error distribution (mechanisms of equal
symptom combined) and that its herald detectors never fire.
"""
from __future__ import annotations

import collections
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]      # the package root (tests/verify/x.py -> root)
if str(ROOT / "verify") not in sys.path:
    sys.path.insert(0, str(ROOT / "verify"))

stim = pytest.importorskip("stim")

from eldpc_verify import circuits as verify_circuits  # noqa: E402
from eldpc_verify.config import load_config  # noqa: E402
from eldpc_testlib import requires_qldpc  # noqa: E402

LOSS_PROBLEMS = [p for p in load_config()["problems"] if p.get("loss") is not None]


def test_record_targets_after_the_final_measurement_are_shifted():
    clean = stim.Circuit("""
        R 0 1 2
        M 0 1 2
        DETECTOR rec[-1]
        M 0 1
        DETECTOR rec[-1] rec[-4]
        DETECTOR rec[-2] rec[-5]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
    """)
    inserted, heralds = verify_circuits.insert_final_herald(clean, 0.25)
    expected = stim.Circuit("""
        R 0 1 2
        M 0 1 2
        DETECTOR rec[-1]
        HERALDED_ERASE(0.25) 0 1
        DETECTOR(-1, 0) rec[-2]
        DETECTOR(-1, 1) rec[-1]
        M 0 1
        DETECTOR rec[-1] rec[-6]
        DETECTOR rec[-2] rec[-7]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
    """)
    assert str(inserted) == str(expected)
    assert heralds == [1, 2]


def test_correct_herald_records_shifts_an_unshifted_insertion():
    unshifted = stim.Circuit("""
        R 0 1 2
        M 0 1 2
        HERALDED_ERASE(0.25) 0 1
        DETECTOR(-1, 0) rec[-2]
        DETECTOR(-1, 1) rec[-1]
        M 0 1
        DETECTOR rec[-1] rec[-4]
    """)
    corrected = verify_circuits.correct_herald_records(unshifted)
    assert str(corrected).splitlines()[-1] == "DETECTOR rec[-1] rec[-6]"


def _unshifted_insertion(circuit, p_erase):
    """The insertion without the record shift (the defect the suite's rule avoids)."""
    f = circuit.flattened()
    instructions = list(f)
    index = max(i for i, inst in enumerate(instructions)
                if inst.name in verify_circuits.MEASURE_GATES)
    qubits = [t.qubit_value for t in instructions[index].targets_copy()]
    n = len(qubits)
    new = f[0:index]
    new.append("HERALDED_ERASE", qubits, float(p_erase))
    for j in range(n):
        new.append("DETECTOR", [stim.target_rec(-(n - j))], [-1.0, float(j)])
    new += f[index:]
    return stim.Circuit(str(new))


def _mechanisms(circuit, heralds=()):
    """{(detectors, observables): probability} over the DEM's error mechanisms with p > 0, with
    mechanisms of the same symptom combined (p = p1 (1 - p2) + p2 (1 - p1)): stim may list such
    mechanisms separately or merged depending on the instruction stream, and both describe the
    same distribution. Herald detectors must not appear; later detector ids are renumbered as if
    the heralds were absent."""
    heralds = sorted(heralds)
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    combined = collections.defaultdict(float)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        if p == 0:
            continue
        dets, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                assert t.val not in heralds, f"herald detector D{t.val} fires at p_erase = 0"
                dets.append(t.val - sum(1 for h in heralds if h < t.val))
            elif t.is_logical_observable_id():
                obs.append(t.val)
        key = (tuple(sorted(dets)), tuple(sorted(obs)))
        q = combined[key]
        combined[key] = q * (1 - p) + p * (1 - q)
    return {key: round(q, 12) for key, q in combined.items()}


@requires_qldpc  # the loss problems are qLDPC circuits
@pytest.mark.parametrize("problem", LOSS_PROBLEMS, ids=[p["id"] for p in LOSS_PROBLEMS])
def test_zero_erasure_insertion_keeps_the_clean_dem(problem):
    zero = copy.deepcopy(problem)
    zero["loss"]["p_erase"] = 0.0
    inserted, heralds = verify_circuits.build_circuit_and_heralds(zero)
    assert heralds, "a loss problem must have herald detectors"
    clean = stim.Circuit(str(verify_circuits.build_base_circuit(problem)))
    assert inserted.num_detectors == clean.num_detectors + len(heralds)
    assert _mechanisms(inserted, heralds) == _mechanisms(clean)


@requires_qldpc  # the loss problems are qLDPC circuits
@pytest.mark.parametrize("problem", LOSS_PROBLEMS, ids=[p["id"] for p in LOSS_PROBLEMS])
def test_unshifted_insertion_changes_the_dem(problem):
    """Regression guard: without the shift the final detectors read herald records, so the
    p_erase = 0 DEM differs from the clean one (if this ever passes for a problem, the check
    above no longer discriminates on it)."""
    clean = stim.Circuit(str(verify_circuits.build_base_circuit(problem)))
    unshifted = _unshifted_insertion(clean, 0.0)
    d = sum(1 for inst in clean.flattened() if inst.name == "DETECTOR")
    final = max(i for i, inst in enumerate(clean.flattened())
                if inst.name in verify_circuits.MEASURE_GATES)
    before = sum(1 for inst in list(clean.flattened())[:final] if inst.name == "DETECTOR")
    n = len(list(clean.flattened())[final].targets_copy())
    heralds = list(range(before, before + n))
    assert unshifted.num_detectors == d + n
    try:
        assert _mechanisms(unshifted, heralds) != _mechanisms(clean)
    except ValueError:
        pass    # stim may also reject the unshifted circuit outright; either way it differs
