# SPDX-License-Identifier: Apache-2.0
"""Circuit constructions and seeded sampling for the verification corpora.

The recipes are those of the reference construction, reproduced here so that the
verification suite does not depend on it:

    uniform -> qldpc.circuits.DepolarizingNoiseModel(p)
    si1000  -> qldpc.circuits.SI1000NoiseModel(p)
    custom  -> qldpc.circuits.NoiseModel(clifford_1q_error=p1, clifford_2q_error=p2,
                   readout_error=p_meas, reset_error=p_reset, idle_error=p_idle or False)

    bb  -> qldpc.codes.BBCode({x: l, y: m}, A, B), A/B sums of x**a * y**b
    hgp -> qldpc.codes.HGPCode(ClassicalCode(H1), ClassicalCode(H2)),
           where a circulant row0 gives entry (i, j) = row0[(j - i) mod n]
    css -> qldpc.codes.CSSCode(hx, hz)
    surface -> stim.Circuit.generated("surface_code:rotated_memory_<basis>", ...), then the
               noise model's noisy_circuit(); bb/hgp/css use qldpc's memory experiment with
               basis Pauli.Z, Pauli.X or None ("z", "x", "xyz")
    finally circuit = stim.Circuit(str(circuit)) -- canonicalise before deriving anything, so the
            DEM and every sample come from exactly the circuit that circuit.stim describes.

Sampling is ``circuit.compile_detector_sampler(seed=seed).sample(shots=N,
separate_observables=True, bit_packed=True)`` in one call, and the seed of a shot set is
``derive_seed(base_seed, split, problem_id, "latency")`` -- the reference construction's derivation.
With the same base seed, problem and shot count, the suite therefore samples the same shot set the
reference construction builds.
"""
from __future__ import annotations

import hashlib

import numpy as np

_LOSS_KEY = "loss"
MEASURE_GATES = ("M", "MX", "MY", "MZ")


def noise_model(noise: dict):
    """The qLDPC noise model of a validated noise dict."""
    from qldpc import circuits as qcircuits

    model = noise["model"]
    if model == "uniform":
        return qcircuits.DepolarizingNoiseModel(float(noise["p"]))
    if model == "si1000":
        return qcircuits.SI1000NoiseModel(float(noise["p"]))
    if model == "custom":
        p_idle = float(noise["p_idle"])
        return qcircuits.NoiseModel(clifford_1q_error=float(noise["p1"]),
                                    clifford_2q_error=float(noise["p2"]),
                                    readout_error=float(noise["p_meas"]),
                                    reset_error=float(noise["p_reset"]),
                                    idle_error=p_idle if p_idle > 0 else False)
    raise ValueError(f"unknown noise model {model!r}")


def _hgp_matrix(M):
    """The rows of an hgp code's matrix (a list of rows, or the two documented dict forms)."""
    if isinstance(M, dict):
        if set(M) == {"rows"}:
            return np.asarray(M["rows"], dtype=np.uint8)
        if set(M) == {"circulant"}:
            row0 = list(M["circulant"])
            n = len(row0)
            return np.asarray([[row0[(j - i) % n] for j in range(n)] for i in range(n)],
                              dtype=np.uint8)
        raise ValueError(f"unknown hgp matrix form {sorted(M)}")
    return np.asarray(M, dtype=np.uint8)


def build_code(problem: dict):
    """The qLDPC code object of a bb/hgp/css problem."""
    from qldpc import codes as qcodes

    family = problem["family"]
    code = problem["code"]
    if family == "bb":
        from sympy.abc import x, y

        A = 0
        for a, b in code["A"]:
            A = A + x ** int(a) * y ** int(b)
        B = 0
        for a, b in code["B"]:
            B = B + x ** int(a) * y ** int(b)
        return qcodes.BBCode({x: int(code["l"]), y: int(code["m"])}, A, B)
    if family == "hgp":
        H1 = _hgp_matrix(code["h1"])
        H2 = _hgp_matrix(code["h2"])
        return qcodes.HGPCode(qcodes.ClassicalCode(H1), qcodes.ClassicalCode(H2))
    if family == "css":
        hx = np.asarray(code["hx"], dtype=np.uint8)
        hz = np.asarray(code["hz"], dtype=np.uint8)
        return qcodes.CSSCode(hx, hz)
    raise ValueError(f"build_code is defined for bb/hgp/css, not {family!r}")


def _qldpc_basis(basis: str):
    from qldpc.objects import Pauli

    return {"z": Pauli.Z, "x": Pauli.X, "xyz": None}[basis]


def build_base_circuit(problem: dict):
    """The noisy stim circuit of a problem, before loss insertion and text canonicalisation."""
    import stim

    if problem.get("shape", {}).get("kind") == "window":
        raise ValueError(f"problem {problem.get('id')!r} is a window and has no circuit")
    family = problem["family"]
    if family == "surface":
        circuit = stim.Circuit.generated(f"surface_code:rotated_memory_{problem['basis']}",
                                         distance=int(problem["code"]["d"]),
                                         rounds=int(problem["rounds"]))
        return noise_model(problem["noise"]).noisy_circuit(circuit)
    if family in ("bb", "hgp", "css"):
        from qldpc import circuits as qcircuits

        code = build_code(problem)
        return qcircuits.get_memory_experiment(code, basis=_qldpc_basis(problem["basis"]),
                                               num_rounds=int(problem["rounds"]),
                                               noise_model=noise_model(problem["noise"]))
    raise ValueError(f"unsupported family {family!r}")


def insert_final_herald(circuit, p_erase: float):
    """Insert heralded erasure before the final data readout (the reference construction's loss
    rule, with the measurement-record shift).

    Returns (new_circuit, herald_detector_ids):

    1. f = circuit.flattened(); let I be the index of the LAST instruction whose name is a
       measurement gate; q_1..q_t are its targets' qubit values, in order.
    2. Insert immediately before f[I]:
         HERALDED_ERASE(p_erase) q_1 ... q_t
         DETECTOR(-1, j) rec[-(t - j)]        for j = 0 .. t-1
       All other instructions are copied in order. HERALDED_ERASE writes t measurement records,
       so in every instruction after f[I] a record target rec[-k] with k > t (one written before
       the insertion point) becomes rec[-(k + t)]; targets with k <= t (the final measurement's
       own records) are unchanged. Without the shift the final detectors would read the herald
       records instead of the earlier measurements they were written for (the reference
       construction's insertion copies them unchanged; ``correct_herald_records`` applies the shift
       to such a circuit). With p_erase = 0 the result has the clean circuit's error distribution
       plus t herald detectors that never fire (tests/verify/test_herald.py).
    3. herald_detector_ids = [D, ..., D+t-1], D = number of DETECTOR instructions before I.
    """
    import stim

    f = circuit.flattened()
    instructions = list(f)
    index = None
    for i, inst in enumerate(instructions):
        if inst.name in MEASURE_GATES:
            index = i
    if index is None:
        raise ValueError("circuit has no measurement instruction")
    inst = instructions[index]
    qubits = []
    for target in inst.targets_copy():
        if not target.is_qubit_target:
            raise ValueError("final measurement has a non-qubit target")
        qubits.append(target.qubit_value)
    n = len(qubits)
    d = sum(1 for ins in instructions[:index] if ins.name == "DETECTOR")

    new = f[0:index]
    if n:
        new.append("HERALDED_ERASE", qubits, float(p_erase))
        for j in range(n):
            new.append("DETECTOR", [stim.target_rec(-(n - j))], [-1.0, float(j)])
        new.append(inst)
        new += _shift_records(instructions[index + 1:], n)
    else:
        new += f[index:]
    return new, [d + j for j in range(n)]


def _shift_records(instructions, n: int):
    """The instructions as a circuit, with every record target rec[-k], k > n, moved to
    rec[-(k + n)] (n records were inserted before the instruction that wrote rec[-n] .. rec[-1])."""
    import stim

    out = stim.Circuit()
    for inst in instructions:
        targets = inst.targets_copy()
        if any(t.is_measurement_record_target and -t.value > n for t in targets):
            targets = [stim.target_rec(t.value - n)
                       if t.is_measurement_record_target and -t.value > n else t
                       for t in targets]
            out.append(inst.name, targets, inst.gate_args_copy())
        else:
            out.append(inst)
    return out


def correct_herald_records(circuit):
    """Apply the measurement-record shift of ``insert_final_herald`` to a circuit whose herald
    block was inserted before the final measurement WITHOUT it (the reference construction's
    insertion): every record target rec[-k], k > t, after the final measurement becomes rec[-(k +
    t)], t = the final measurement's target count. Only for such a circuit: applied to a corrected
    one it
    shifts a second time."""
    import stim

    f = circuit.flattened()
    instructions = list(f)
    index = None
    for i, inst in enumerate(instructions):
        if inst.name in MEASURE_GATES:
            index = i
    if index is None:
        raise ValueError("circuit has no measurement instruction")
    n = len(instructions[index].targets_copy())
    new = f[0:index + 1]
    new += _shift_records(instructions[index + 1:], n)
    return stim.Circuit(str(new))


def build_circuit_and_heralds(problem: dict):
    """(circuit, herald detector ids): the base circuit, loss insertion, then canonicalisation
    through the stim text -- the reference construction's build_circuit."""
    import stim

    circuit = build_base_circuit(problem)
    heralds: list[int] = []
    loss = problem.get(_LOSS_KEY)
    if loss is not None:
        circuit, heralds = insert_final_herald(circuit, float(loss["p_erase"]))
    circuit = stim.Circuit(str(circuit))
    return circuit, [int(h) for h in heralds]


def build_circuit(problem: dict):
    """The problem's circuit."""
    return build_circuit_and_heralds(problem)[0]


def detector_rounds(problem: dict, circuit, heralds) -> np.ndarray:
    """det_round uint32[m]: the measurement round of every detector (the reference construction's
    rule).

    surface: int(coords[-1]); bb/hgp/css: int(coords[0]). Herald detectors (no round of their
    own) take the maximum round of the non-herald detectors."""
    m = int(circuit.num_detectors)
    coords = circuit.get_detector_coordinates()
    herald_set = {int(h) for h in heralds}
    det_round = np.zeros(m, dtype=np.uint32)
    for i in range(m):
        if i in herald_set:
            continue
        if i not in coords or not coords[i]:
            raise ValueError(f"detector {i} has no coordinates")
        c = coords[i]
        det_round[i] = int(c[-1]) if problem["family"] in ("surface", "tcnot") else int(c[0])
    non_herald = [int(det_round[i]) for i in range(m) if i not in herald_set]
    herald_round = max(non_herald) if non_herald else 0
    for h in herald_set:
        if not (0 <= h < m):
            raise ValueError(f"herald detector id {h} out of range")
        det_round[h] = herald_round
    return det_round


def dem_options(problem: dict) -> dict:
    """Keyword arguments of stim's detector_error_model for this problem."""
    decompose = bool(problem.get("dem", {}).get("decompose_errors", False))
    return {"decompose_errors": decompose, "approximate_disjoint_errors": True,
            "ignore_decomposition_failures": decompose}


def circuit_dem(circuit, problem: dict):
    return circuit.detector_error_model(**dem_options(problem))


def sample_shots(circuit, num_shots: int, seed: int):
    """(dets, obs) as bit-packed uint8 arrays, LSB-first rows, as stim returns them."""
    num_shots = int(num_shots)
    if num_shots <= 0:
        raise ValueError("num_shots must be >= 1")
    sampler = circuit.compile_detector_sampler(seed=int(seed))
    dets, obs = sampler.sample(shots=num_shots, separate_observables=True, bit_packed=True)
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8) if obs is not None else np.zeros((num_shots, 0), np.uint8)
    return dets, obs


def derive_seed(base_seed: int, *labels) -> int:
    """Shot-set seed: little-endian uint64 from the first 8 bytes of
    sha256(f"{base_seed}|{label1}|{label2}|..."), masked to 63 bits (the reference construction's
    derive_seed)."""
    msg = "|".join([str(int(base_seed))] + [str(x) for x in labels]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "little") & ((1 << 63) - 1)
