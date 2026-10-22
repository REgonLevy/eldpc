# SPDX-License-Identifier: Apache-2.0
"""Problems for the eldpc benchmark.

A problem is (H, O, priors) plus a way to sample syndromes and the true observable flips.  Two sources:

- Built-in problems: circuit-level memory experiments with uniform depolarizing noise at p = 1e-3.  The
  rotated surface code ("surface") is built with stim alone.  The bivariate-bicycle and hypergraph-product
  codes (bb144, bb144z, bb72, bb288, hgp98) are built with the circuit builder of the optional qLDPC
  package (pip install ".[qldpc]" from the package root); without it they raise a clear error and the
  default problem falls back to the surface code.  The matrices are derived from the circuit's detector error model with the rule of
  eldpc.dem_to_arrays (one column per error instruction of the flattened DEM, in order, with no columns combined; H
  and O record the parity of the detectors and observables it flips; priors[j] is its probability).
- A stim detector error model file (--problem PATH.dem), the way to benchmark any other code.  The same
  rule derives (H, O, priors), and shots are sampled from the DEM.

Shots for the latency measurement and shots for the LER check are sampled from the circuit (exact model)
or the DEM, with seeds derived from the problem name and the purpose, so a run is reproducible.

The only imports are numpy, scipy and stim, plus qLDPC for the bivariate-bicycle and hypergraph-product
built-ins.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import scipy.sparse as sp


class ProblemError(RuntimeError):
    pass


# Code definitions: BB(l, m, A, B) is the bivariate bicycle code with A = sum x^a y^b, B = sum x^a y^b;
# HGP(h1, h2) is the hypergraph product of two classical circulant codes; the surface code is stim's
# rotated_memory_z.
BUILTINS = {
    "bb144": {
        "id": "bb144_unif_p1e-3_r12_xyz", "family": "bb", "basis": "xyz", "rounds": 12, "p": 1e-3,
        "l": 12, "m": 6, "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]],
        "description": "[[144,12,12]] bivariate bicycle code, 12 rounds, X and Z decoded jointly",
        "dem_decompose_errors": False,
    },
    "bb144z": {
        "id": "bb144_unif_p1e-3_r12_z", "family": "bb", "basis": "z", "rounds": 12, "p": 1e-3,
        "l": 12, "m": 6, "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]],
        "description": "[[144,12,12]] bivariate bicycle code, 12 rounds, Z basis",
        "dem_decompose_errors": False,
    },
    "bb72": {
        "id": "bb72_unif_p1e-3_r6_z", "family": "bb", "basis": "z", "rounds": 6, "p": 1e-3,
        "l": 6, "m": 6, "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]],
        "description": "[[72,12,6]] bivariate bicycle code, 6 rounds, Z basis",
        "dem_decompose_errors": False,
    },
    "bb288": {
        "id": "bb288_unif_p1e-3_r18_z", "family": "bb", "basis": "z", "rounds": 18, "p": 1e-3,
        "l": 12, "m": 12, "A": [[3, 0], [0, 2], [0, 7]], "B": [[0, 3], [1, 0], [2, 0]],
        "description": "[[288,12,18]] bivariate bicycle code, 18 rounds, Z basis",
        "dem_decompose_errors": False,
    },
    "hgp98": {
        "id": "hgp98_unif_p1e-3_r4_z", "family": "hgp", "basis": "z", "rounds": 4, "p": 1e-3,
        "h1": {"circulant": [1, 1, 0, 1, 0, 0, 0]}, "h2": {"circulant": [1, 1, 0, 1, 0, 0, 0]},
        "description": "hypergraph product of two [7,4,3] circulant codes, 4 rounds, Z basis",
        "dem_decompose_errors": False,
    },
    "surface": {
        "id": "sc9_unif_p1e-3_z", "family": "surface", "basis": "z", "rounds": 9, "p": 1e-3, "d": 9,
        "description": "rotated surface code d=9, 9 rounds, Z basis", "dem_decompose_errors": True,
    },
}
PREFERRED_DEFAULT = "bb144"
FALLBACK_DEFAULT = "surface"   # needs only stim
DEFAULT_PROBLEM = "default"    # resolved by resolve_default() when the run starts


def resolve_default() -> tuple[str, str | None]:
    """(problem name, note): bb144 when qLDPC is installed, else the stim-only surface code and a note."""
    try:
        import qldpc  # noqa: F401
        from qldpc import circuits  # noqa: F401
    except Exception:
        return FALLBACK_DEFAULT, (f"qLDPC is not installed, so the default problem is {FALLBACK_DEFAULT} "
                                  f"(stim only) instead of {PREFERRED_DEFAULT}; pip install "
                                  "\".[qldpc]\" from the package root, or pass --problem, to choose")
    return PREFERRED_DEFAULT, None


def dem_to_arrays(dem):
    """(H, O, priors) of a stim DetectorErrorModel by the rule of eldpc.dem_to_arrays.

    Walk dem.flattened() (REPEAT blocks unrolled, shift_detectors applied).  Every error(p) instruction
    becomes one column j, in order, with no columns combined or reordered; H[i, j] is the parity of the number of
    times detector i appears among its targets and O[l, j] the same for observable l (separators XOR);
    priors[j] = p.  This is the same rule the decoder's from_dem() uses.
    """
    import stim

    if not isinstance(dem, stim.DetectorErrorModel):
        dem = stim.DetectorErrorModel(str(dem))
    hr, hc, orow, ocol, probs = [], [], [], [], []
    j = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        args = inst.args_copy()
        probs.append(float(args[0]))
        dets, obs = {}, {}
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets[t.val] = dets.get(t.val, 0) ^ 1
            elif t.is_logical_observable_id():
                obs[t.val] = obs.get(t.val, 0) ^ 1
        for d, b in dets.items():
            if b:
                hr.append(d)
                hc.append(j)
        for o, b in obs.items():
            if b:
                orow.append(o)
                ocol.append(j)
        j += 1
    m, k, n = dem.num_detectors, dem.num_observables, j
    H = sp.csc_matrix((np.ones(len(hr), dtype=np.uint8), (np.array(hr, np.int64), np.array(hc, np.int64))),
                      shape=(m, n))
    O = sp.csc_matrix((np.ones(len(orow), dtype=np.uint8), (np.array(orow, np.int64),
                                                             np.array(ocol, np.int64))), shape=(k, n))
    H.sum_duplicates()
    O.sum_duplicates()
    H.sort_indices()
    O.sort_indices()
    return H, O, np.array(probs, dtype=np.float64)


def _code(spec):
    from qldpc import codes as qcodes

    if spec["family"] == "bb":
        from sympy.abc import x, y

        A = 0
        for a, b in spec["A"]:
            A = A + x ** int(a) * y ** int(b)
        B = 0
        for a, b in spec["B"]:
            B = B + x ** int(a) * y ** int(b)
        return qcodes.BBCode({x: int(spec["l"]), y: int(spec["m"])}, A, B)
    if spec["family"] == "hgp":
        return qcodes.HGPCode(qcodes.ClassicalCode(_hgp_matrix(spec["h1"])),
                              qcodes.ClassicalCode(_hgp_matrix(spec["h2"])))
    raise ProblemError(f"no code object for family {spec['family']!r}")


def _hgp_matrix(M):
    if isinstance(M, dict) and "circulant" in M:
        row0 = list(M["circulant"])
        n = len(row0)
        return np.asarray([[row0[(j - i) % n] for j in range(n)] for i in range(n)], dtype=np.uint8)
    return np.asarray(M, dtype=np.uint8)


def _circuit(spec):
    """The circuit-level memory experiment of a built-in problem, canonicalised as text, then its detector
    error model (decomposed for the surface code)."""
    import stim

    try:
        from qldpc import circuits as qcircuits
    except ImportError as e:  # pragma: no cover - depends on the installation
        raise ProblemError(
            "the built-in BB/HGP problems need qLDPC to build their circuits; install it with "
            "pip install \".[qldpc]\" from the package root, or pass a stim detector error model "
            "file (--problem PATH.dem)") from e

    noise = qcircuits.DepolarizingNoiseModel(float(spec["p"]))
    if spec["family"] == "surface":
        circ = stim.Circuit.generated(f"surface_code:rotated_memory_{spec['basis']}",
                                      distance=int(spec["d"]), rounds=int(spec["rounds"]))
        circ = noise.noisy_circuit(circ)
    elif spec["family"] in ("bb", "hgp"):
        from qldpc.objects import Pauli

        basis = {"z": Pauli.Z, "x": Pauli.X, "xyz": None}[spec["basis"]]
        circ = qcircuits.get_memory_experiment(_code(spec), basis=basis, num_rounds=int(spec["rounds"]),
                                               noise_model=noise)
    else:
        raise ProblemError(f"unknown built-in family {spec['family']!r}")
    circ = stim.Circuit(str(circ))
    decompose = bool(spec.get("dem_decompose_errors", False))
    dem = circ.detector_error_model(decompose_errors=decompose, approximate_disjoint_errors=True,
                                    ignore_decomposition_failures=decompose)
    return circ, dem


def _seed(name: str, purpose: str) -> int:
    h = hashlib.sha256(f"eldpc-bench|{name}|{purpose}".encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def session_seed(name: str, purpose: str) -> int:
    """A reproducible seed for one sampled set of a problem (public wrapper of the shot seed)."""
    return _seed(name, purpose)


class Problem:
    """A decoded problem: matrices, DEM text, provenance, and a deterministic shot sampler."""

    def __init__(self, name, source, H, O, priors, dem_text, recipe, circuit=None, dem=None):
        self.name = name
        self.source = source
        self.H = sp.csc_matrix(H, dtype=np.uint8)
        self.O = sp.csc_matrix(O, dtype=np.uint8)
        self.priors = np.asarray(priors, dtype=np.float64)
        self.dem_text = dem_text
        self.recipe = recipe
        self._circuit = circuit
        self._dem = dem
        if self._circuit is None and self._dem is None:
            self._dem = _parse_dem(dem_text)
        self.m, self.n = self.H.shape
        self.k = self.O.shape[0]
        if self.O.shape[1] != self.n:
            raise ProblemError("O must have as many columns as H")
        if self.priors.shape != (self.n,):
            raise ProblemError("priors length does not match the number of mechanisms")

    def sample(self, shots: int, purpose: str):
        """Sample `shots` syndromes and true observable flips; deterministic in (problem, purpose, shots)."""
        seed = _seed(self.name, f"{purpose}:{int(shots)}")
        if self._circuit is not None:
            sampler = self._circuit.compile_detector_sampler(seed=seed)
            dets, obs = sampler.sample(shots=int(shots), separate_observables=True)
        else:
            sampler = self._dem.compile_sampler(seed=seed)
            r = sampler.sample(shots=int(shots))
            if isinstance(r, tuple):  # stim returns (detectors, observables[, errors])
                dets, obs = np.asarray(r[0], dtype=np.uint8), np.asarray(r[1], dtype=np.uint8)
            else:                     # older stim: one array, detectors first, observables after
                both = np.asarray(r, dtype=np.uint8)
                md = int(self._dem.num_detectors)
                dets, obs = both[:, :md], both[:, md:]
        dets = np.asarray(dets, dtype=np.uint8)
        obs = np.zeros((dets.shape[0], self.k), dtype=np.uint8) if obs is None else np.asarray(obs, dtype=np.uint8)
        if dets.shape != (shots, self.m):
            raise ProblemError(f"sampler returned detector data of shape {dets.shape}, expected {(shots, self.m)}")
        if obs.shape != (shots, self.k):
            raise ProblemError(f"sampler returned observable data of shape {obs.shape}, expected {(shots, self.k)}")
        return dets, obs, seed

    def info(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "m": int(self.m),
            "n": int(self.n),
            "k": int(self.k),
            "nnz_h": int(self.H.nnz),
            "nnz_o": int(self.O.nnz),
            "recipe": self.recipe,
        }


def _parse_dem(text: str, where: str = "the detector error model"):
    try:
        import stim
    except ImportError as e:  # pragma: no cover - depends on the installation
        raise ProblemError(f"{where}: stim is needed to read a detector error model") from e
    try:
        return stim.DetectorErrorModel(text)
    except Exception as e:
        raise ProblemError(f"{where}: not a valid stim detector error model: {e}") from e


def open_problem(name: str) -> Problem:
    """A built-in problem by name, or a stim DEM file (any name ending in .dem or naming a file)."""
    if name == DEFAULT_PROBLEM and not os.path.isfile(name):
        name, _note = resolve_default()
    if name in BUILTINS:
        spec = BUILTINS[name]
        circ, dem = _circuit(spec)
        H, O, priors = dem_to_arrays(dem)
        recipe = {"kind": "built-in circuit", "family": spec["family"], "description": spec["description"],
                  "basis": spec["basis"], "rounds": int(spec["rounds"]), "p": float(spec["p"]),
                  "noise": "uniform depolarizing",
                  "dem_decompose_errors": bool(spec.get("dem_decompose_errors", False))}
        return Problem(spec["id"], "built-in", H, O, priors, str(dem) + "\n", recipe, circuit=circ, dem=dem)
    if os.path.isfile(name):
        try:
            text = open(name, "r", encoding="utf-8").read()
        except (OSError, UnicodeError) as e:
            raise ProblemError(f"cannot read {name}: {e}") from e
        dem = _parse_dem(text, name)
        H, O, priors = dem_to_arrays(dem)
        if priors.size and (not np.all(np.isfinite(priors)) or not np.all((priors > 0) & (priors < 1))):
            bad = int(np.flatnonzero(~((priors > 0) & (priors < 1)))[0])
            raise ProblemError(f"{name}: error instruction {bad} has probability {priors[bad]!r} outside "
                               "(0, 1)")
        recipe = {"kind": "detector error model file", "path": name}
        return Problem(os.path.basename(name), "dem file", H, O, priors, text, recipe, dem=dem)
    known = ", ".join(sorted(BUILTINS))
    raise ProblemError(f"unknown problem {name!r}: use one of {known}, or a stim DEM file path")
