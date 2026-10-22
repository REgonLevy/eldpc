# SPDX-License-Identifier: Apache-2.0
"""Problems and shot sources for the knob tuner.

A problem is one code at one physical error rate: a detector error model, the (H, O, priors) arrays
the decoder consumes, and a seeded source of fresh shots (detection events + observable flips).

The built-in problem recipes are the standard memory-experiment circuits of these codes:

    bb72       bivariate-bicycle code l=6,  m=6,  rounds 6,  Z memory
    bb144      bivariate-bicycle code l=12, m=6,  rounds 12, Z memory
    bb144xyz   the same code, X and Z decoded jointly ("xyz"), rounds 12
    bb288      bivariate-bicycle code l=12, m=12, rounds 18, Z memory
    hgp98      hypergraph-product code from two [7, ...] circulants, rounds 4, Z memory
    sc5/sc9/sc13/sc17  rotated surface code, distance = rounds = 5/9/13/17, SI1000 noise

The bivariate-bicycle and hypergraph-product circuits are built with qLDPC's memory experiment, the
surface code with stim's generated circuit and qLDPC's noise models.  Uniform depolarizing noise is
used for bb/hgp, SI1000 for the surface code.  The built-in recipes therefore need the optional
``qldpc`` package (and its ``sympy`` dependency); ``--dem``, ``--circuit`` and ``--h`` need only
numpy, scipy and stim.

Decoding goes through the public ``eldpc`` API only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# problem definitions (circuit recipes)
# ---------------------------------------------------------------------------

#: bivariate-bicycle polynomials, as (x exponent, y exponent) terms.
_BB72 = {"l": 6, "m": 6,
         "A": [(3, 0), (0, 1), (0, 2)], "B": [(0, 3), (1, 0), (2, 0)]}
_BB144 = {"l": 12, "m": 6,
          "A": [(3, 0), (0, 1), (0, 2)], "B": [(0, 3), (1, 0), (2, 0)]}
_BB288 = {"l": 12, "m": 12,
          "A": [(3, 0), (0, 2), (0, 7)], "B": [(0, 3), (1, 0), (2, 0)]}
_HGP98 = {"h1": [1, 1, 0, 1, 0, 0, 0], "h2": [1, 1, 0, 1, 0, 0, 0]}

PROBLEMS = {
    "bb72": {"family": "bb", "code": _BB72, "basis": "z", "rounds": 6, "noise": "uniform",
             "description": "bivariate-bicycle [[72,12,6]] code, Z memory"},
    "bb144": {"family": "bb", "code": _BB144, "basis": "z", "rounds": 12, "noise": "uniform",
              "description": "bivariate-bicycle [[144,12,12]] code, Z memory"},
    "bb144xyz": {"family": "bb", "code": _BB144, "basis": "xyz", "rounds": 12, "noise": "uniform",
                 "description": "bivariate-bicycle [[144,12,12]] code, X and Z decoded jointly"},
    "bb288": {"family": "bb", "code": _BB288, "basis": "z", "rounds": 18, "noise": "uniform",
              "description": "bivariate-bicycle [[288,12,18]] code, Z memory"},
    "hgp98": {"family": "hgp", "code": _HGP98, "basis": "z", "rounds": 4, "noise": "uniform",
              "description": "hypergraph-product [[98,...]] code, Z memory"},
    "sc5": {"family": "surface", "code": {"d": 5}, "basis": "z", "rounds": 5, "noise": "si1000",
            "description": "rotated surface code d=5, SI1000 noise"},
    "sc9": {"family": "surface", "code": {"d": 9}, "basis": "z", "rounds": 9, "noise": "si1000",
            "description": "rotated surface code d=9, SI1000 noise"},
    "sc13": {"family": "surface", "code": {"d": 13}, "basis": "z", "rounds": 13, "noise": "si1000",
             "description": "rotated surface code d=13, SI1000 noise"},
    "sc17": {"family": "surface", "code": {"d": 17}, "basis": "z", "rounds": 17, "noise": "si1000",
             "description": "rotated surface code d=17, SI1000 noise"},
}

#: DEM options of the built-in recipes and of --circuit files.  The undecomposed model is the
#: default: BP+OSD decodes the DEM's error mechanisms directly, and for a surface code the
#: decomposed DEM has a different mechanism list, i.e. it is a different model.  --decompose adds
#: the decomposition options of stim.
DEM_OPTIONS = {"decompose_errors": False, "approximate_disjoint_errors": True}


def dem_options(decompose: bool = False) -> dict:
    """The ``detector_error_model()`` keyword options used for a built-in recipe or a circuit."""
    options = dict(DEM_OPTIONS)
    options["decompose_errors"] = bool(decompose)
    if decompose:
        options["ignore_decomposition_failures"] = True
    return options


def _bb_polynomial(terms):
    from sympy.abc import x, y

    out = 0
    for a, b in terms:
        out = out + x ** int(a) * y ** int(b)
    return out


def _qldpc_basis(basis: str):
    from qldpc.objects import Pauli

    return {"z": Pauli.Z, "x": Pauli.X, "xyz": None}[basis]


def _circulant_matrix(row):
    row0 = [int(v) for v in row]
    n = len(row0)
    return np.asarray([[row0[(j - i) % n] for j in range(n)] for i in range(n)], dtype=np.uint8)


def build_circuit(name: str, p: float, rounds: int | None = None,
                  noise_model: str | None = None):
    """Build the noisy stim circuit of a built-in problem.

    ``noise_model`` overrides the recipe's default: surface codes default to SI1000 and also
    support uniform depolarizing noise; bivariate-bicycle and hypergraph-product codes always use
    uniform depolarizing noise.
    """
    import stim

    if name not in PROBLEMS:
        raise ValueError(f"unknown problem {name!r} (choose from {', '.join(sorted(PROBLEMS))})")
    try:  # every recipe takes its noise model, and most take their code, from qLDPC
        from qldpc import circuits as _circuits  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            f"the built-in recipe {name!r} needs qLDPC, which cannot be imported ({exc}); install "
            "it with pip install \".[qldpc]\" from the repository root, or give the problem as "
            "--dem, --circuit or --h") from None
    spec = PROBLEMS[name]
    model_name = spec["noise"] if noise_model is None else str(noise_model)
    rounds = int(spec["rounds"] if rounds is None else rounds)
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    family = spec["family"]
    if family == "surface":
        from qldpc.circuits import DepolarizingNoiseModel, SI1000NoiseModel

        if model_name == "si1000":
            model = SI1000NoiseModel(float(p))
        elif model_name == "uniform":
            model = DepolarizingNoiseModel(float(p))
        else:
            raise ValueError(f"{name} supports the si1000 and uniform noise models, "
                             f"not {model_name!r}")
        d = int(spec["code"]["d"])
        noiseless = stim.Circuit.generated(f"surface_code:rotated_memory_{spec['basis']}",
                                           distance=d, rounds=rounds)
        circuit = model.noisy_circuit(noiseless)
    else:
        if model_name != "uniform":
            raise ValueError(f"{name} uses uniform depolarizing noise; "
                             f"the noise model cannot be overridden")
        from qldpc import circuits as qcircuits, codes as qcodes

        if family == "bb":
            from sympy.abc import x, y

            code = qcodes.BBCode({x: int(spec["code"]["l"]), y: int(spec["code"]["m"])},
                                 _bb_polynomial(spec["code"]["A"]),
                                 _bb_polynomial(spec["code"]["B"]))
        elif family == "hgp":
            code = qcodes.HGPCode(
                qcodes.ClassicalCode(_circulant_matrix(spec["code"]["h1"])),
                qcodes.ClassicalCode(_circulant_matrix(spec["code"]["h2"])))
        else:  # pragma: no cover - the table above has no other family
            raise ValueError(f"unsupported family {family!r}")
        circuit = qcircuits.get_memory_experiment(
            code, basis=_qldpc_basis(spec["basis"]), num_rounds=rounds,
            noise_model=qcircuits.DepolarizingNoiseModel(float(p)))
    # Text canonicalisation: the DEM and every sample are derived from exactly this circuit.
    return stim.Circuit(str(circuit))


def circuit_dem(circuit, decompose: bool = False):
    """DEM of a circuit with the options of the built-in recipes (undecomposed by default)."""
    return circuit.detector_error_model(**dem_options(decompose))


# ---------------------------------------------------------------------------
# DEM -> (H, O, priors)
# ---------------------------------------------------------------------------

def dem_to_arrays(dem):
    """Convert a detector error model to ``(H, O, priors)``.

    Walk ``dem.flattened()``; every ``error`` instruction becomes one column, in order.  Columns are
    never combined, dropped or reordered.  ``priors[j]`` is the instruction's probability, ``H[i,j]``
    the parity of detector D_i in the instruction's targets, ``O[l,j]`` the parity of observable
    L_l.  Separators (``^``) are ignored: a column is the XOR of its components.
    """
    import stim

    if not isinstance(dem, stim.DetectorErrorModel):
        dem = stim.DetectorErrorModel(str(dem))
    h_rows, h_cols, o_rows, o_cols, probs = [], [], [], [], []
    j = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        probs.append(float(inst.args_copy()[0]))
        dets: dict[int, int] = {}
        obs: dict[int, int] = {}
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets[t.val] = dets.get(t.val, 0) ^ 1
            elif t.is_logical_observable_id():
                obs[t.val] = obs.get(t.val, 0) ^ 1
        for d, b in dets.items():
            if b:
                h_rows.append(d)
                h_cols.append(j)
        for o, b in obs.items():
            if b:
                o_rows.append(o)
                o_cols.append(j)
        j += 1
    m, k, n = int(dem.num_detectors), int(dem.num_observables), j
    H = sp.csc_matrix((np.ones(len(h_rows), dtype=np.uint8),
                       (np.asarray(h_rows, dtype=np.int64), np.asarray(h_cols, dtype=np.int64))),
                      shape=(m, n))
    O = sp.csc_matrix((np.ones(len(o_rows), dtype=np.uint8),
                       (np.asarray(o_rows, dtype=np.int64), np.asarray(o_cols, dtype=np.int64))),
                      shape=(k, n))
    priors = np.asarray(probs, dtype=np.float64)
    return _canonical_csc(H), _canonical_csc(O), priors


def _canonical_csc(matrix) -> sp.csc_matrix:
    m = sp.csc_matrix(matrix, dtype=np.uint8)
    m.sum_duplicates()
    m.eliminate_zeros()
    m.sort_indices()
    return m


# ---------------------------------------------------------------------------
# bit packing helpers (LSB-first, the stim convention)
# ---------------------------------------------------------------------------

def pack_bits(unpacked: np.ndarray) -> np.ndarray:
    a = np.asarray(unpacked, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError("pack_bits expects a 2-D array")
    return np.packbits(a, axis=1, bitorder="little")


def unpack_bits(packed: np.ndarray, ncols: int) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError("unpack_bits expects a 2-D array")
    return np.unpackbits(p, axis=1, count=int(ncols), bitorder="little").astype(np.uint8)


# ---------------------------------------------------------------------------
# shot sources
# ---------------------------------------------------------------------------

class ShotSource:
    """A seeded stream of fresh shots.  ``sample(n)`` returns (dets, obs) bit-packed rows."""

    def sample(self, shots: int):  # pragma: no cover - interface
        raise NotImplementedError


class CircuitShots(ShotSource):
    def __init__(self, circuit, seed: int):
        self._sampler = circuit.compile_detector_sampler(seed=int(seed))

    def sample(self, shots: int):
        out = self._sampler.sample(shots=int(shots), separate_observables=True, bit_packed=True)
        dets, obs = out[0], out[1]
        return np.asarray(dets, dtype=np.uint8), (None if obs is None else np.asarray(obs, dtype=np.uint8))


class DemShots(ShotSource):
    def __init__(self, dem, seed: int):
        self._sampler = dem.compile_sampler(seed=int(seed))

    def sample(self, shots: int):
        dets, obs, _errors = self._sampler.sample(shots=int(shots), bit_packed=True)
        return np.asarray(dets, dtype=np.uint8), (None if obs is None else np.asarray(obs, dtype=np.uint8))


class BernoulliShots(ShotSource):
    """Code-capacity errors: every mechanism flips independently with probability ``p``."""

    def __init__(self, H, O, p: float, seed: int):
        self._H = sp.csc_matrix(H, dtype=np.uint32)
        self._O = sp.csc_matrix(O, dtype=np.uint32)
        self._p = float(p)
        self._rng = np.random.default_rng(int(seed))

    def sample(self, shots: int):
        n = self._H.shape[1]
        errors = (self._rng.random((int(shots), n)) < self._p).astype(np.uint32)
        dets = np.asarray(self._H @ errors.T, dtype=np.uint32) & 1
        obs = np.asarray(self._O @ errors.T, dtype=np.uint32) & 1
        return pack_bits(dets.T), pack_bits(obs.T)


# ---------------------------------------------------------------------------
# the Problem object
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    label: str
    key: str
    m: int
    n: int
    k: int
    H: sp.csc_matrix
    O: sp.csc_matrix
    priors: np.ndarray
    source_factory: object
    dem_text: str | None
    circuit: object = None
    meta: dict = field(default_factory=dict)

    def shot_source(self, seed: int) -> ShotSource:
        return self.source_factory(int(seed))

    def dem_sha256(self) -> str | None:
        if self.dem_text is None:
            return None
        return hashlib.sha256(self.dem_text.encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        return {
            "label": self.label,
            "key": self.key,
            "num_detectors": self.m,
            "num_mechanisms": self.n,
            # The keyword options passed to stim's detector_error_model(), when this tool built
            # the model; None for a DEM read from a file or for an H/O problem (no DEM).
            "dem_options": self.meta.get("dem_options"),
            "num_observables": self.k,
            "dem_sha256": self.dem_sha256(),
            "circuit_present": self.circuit is not None,
            **self.meta,
        }


def _problem_from_dem(label: str, key: str, dem, circuit=None, meta=None) -> Problem:
    H, O, priors = dem_to_arrays(dem)
    m, n = H.shape
    k = O.shape[0]
    dem_text = str(dem)

    def factory(seed):
        if circuit is not None:
            return CircuitShots(circuit, seed)
        return DemShots(dem, seed)

    return Problem(label=label, key=key, m=m, n=n, k=k, H=H, O=O, priors=priors,
                   source_factory=factory, dem_text=dem_text, circuit=circuit,
                   meta=dict(meta or {}))


def build_problem(name: str, p: float, rounds: int | None = None,
                  noise_model: str | None = None, decompose: bool = False) -> Problem:
    """One built-in problem at physical error rate ``p`` (undecomposed DEM by default)."""
    if name not in PROBLEMS:
        raise ValueError(f"unknown problem {name!r} (choose from {', '.join(sorted(PROBLEMS))})")
    spec = PROBLEMS[name]
    circuit = build_circuit(name, p, rounds, noise_model)
    dem = circuit_dem(circuit, decompose)
    rounds_used = int(spec["rounds"] if rounds is None else rounds)
    model = spec["noise"] if noise_model is None else str(noise_model)
    # The default recipe keeps its historical seed key; an override gets its own.  The DEM options
    # do not change the shots (they are sampled from the same circuit), so they are not in the key.
    suffix = "" if model == spec["noise"] else f":noise={model}"
    meta = {"family": spec["family"], "basis": spec["basis"], "rounds": rounds_used,
            "p": float(p), "noise_model": model, "description": spec["description"],
            "dem_options": dem_options(decompose)}
    return _problem_from_dem(name, f"{name}:p={float(p):g}:r={rounds_used}{suffix}",
                             dem, circuit, meta)


def problem_from_dem_file(path: str) -> Problem:
    """A problem read from a stim detector error model file; shots are sampled from the DEM."""
    import stim

    text = _read_text(path)
    dem = stim.DetectorErrorModel(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return _problem_from_dem(f"dem:{digest}", f"dem:{digest}", dem,
                             meta={"source": "dem", "path": str(path), "dem_options": None})


def problem_from_circuit_file(path: str, decompose: bool = False) -> Problem:
    """A problem read from a stim circuit file; shots are sampled from the circuit."""
    import stim

    text = _read_text(path)
    circuit = stim.Circuit(str(stim.Circuit(text)))
    dem = circuit_dem(circuit, decompose)
    digest = hashlib.sha256(str(circuit).encode("utf-8")).hexdigest()[:16]
    return _problem_from_dem(f"circuit:{digest}", f"circuit:{digest}", dem, circuit,
                             meta={"source": "circuit", "path": str(path),
                                   "dem_options": dem_options(decompose)})


def h_digest(H) -> str:
    """A stable identity of a parity-check matrix (shape and CSC contents)."""
    H = _canonical_csc(H)
    h = hashlib.sha256()
    h.update(f"{H.shape[0]}x{H.shape[1]}:".encode("ascii"))
    h.update(np.ascontiguousarray(H.indptr, dtype="<u4").tobytes())
    h.update(np.ascontiguousarray(H.indices, dtype="<u4").tobytes())
    return h.hexdigest()[:16]


def problem_from_h_arrays(H, O, p: float, *, meta: dict | None = None) -> Problem:
    """A code-capacity problem from a parity-check matrix and its observables.

    One error mechanism per column; mechanism j flips with probability ``p``, flips the detectors of
    column j of H and the observables of column j of O.
    """
    H = _canonical_csc(H)
    O = _canonical_csc(O)
    m, n = H.shape
    if O.shape[1] != n:
        raise ValueError(f"observable matrix has {O.shape[1]} columns, H has {n}")
    if O.shape[0] < 1:
        raise ValueError("the problem has no logical observables, so a logical error rate "
                         "cannot be measured")
    priors = np.full(n, float(p), dtype=np.float64)
    digest = h_digest(H)

    def factory(seed):
        return BernoulliShots(H, O, p, seed)

    return Problem(label=f"h:{digest}", key=f"h:{digest}:p={float(p):g}", m=m, n=n,
                   k=O.shape[0], H=H, O=O, priors=priors, source_factory=factory,
                   dem_text=None, circuit=None,
                   meta=dict({"source": "h", "p": float(p), "dem_options": None},
                             **(meta or {})))


def problem_from_h_file(h_path: str, p: float, o_path: str | None = None) -> Problem:
    """A code-capacity problem read from files.

    If ``o_path`` is not given, the observable matrix is derived as a basis of ker(H)/row(H); that
    requires H H^T = 0 over GF(2) (the rows are the commuting stabilizer generators of a stabilizer
    code), and the tool says so if it does not hold.  H H^T = 0 is necessary but not sufficient for
    a meaningful result: H must be the *full* stabilizer matrix.  A CSS half (for example Hx with
    Hx Hx^T = 0) is not: the derived rows then include the stabilizers of the other type, and a
    decoded stabilizer component would be counted as a logical error.  For a CSS half pass the
    observable matrix explicitly.
    """
    H = read_matrix_file(h_path)
    if o_path is not None:
        O = read_matrix_file(o_path)
    else:
        O = logical_observables_from_h(H)
    meta = {"path": str(h_path), "observables_derived": o_path is None}
    if o_path is not None:
        meta["observable_path"] = str(o_path)
    return problem_from_h_arrays(H, O, p, meta=meta)


# ---------------------------------------------------------------------------
# matrix files and GF(2) code-capacity logicals
# ---------------------------------------------------------------------------

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_matrix_file(path: str) -> sp.csc_matrix:
    """Read a binary matrix: the DBCSC001 binary format or text ``m n`` + m rows of 0/1.

    DBCSC001 is: the 8 bytes ``DBCSC001``, then nrows, ncols and nnz as little-endian uint64, then
    ncols + 1 uint32 column pointers and nnz uint32 row indices (a 0/1 CSC matrix).

    The text format is strict: the first non-comment line is ``m n``, followed by m lines with n
    whitespace-separated 0/1 values each.  Lines starting with ``#`` and blank lines are ignored.
    """
    with open(path, "rb") as f:
        head = f.read(8)
    if head == b"DBCSC001":
        return _read_dbcsc(path)
    rows = []
    m = n = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            if m is None:
                if len(tokens) != 2:
                    raise ValueError(f"{path}: first line must be 'm n', got {line!r}")
                m, n = int(tokens[0]), int(tokens[1])
                if m < 0 or n < 0:
                    raise ValueError(f"{path}: negative matrix size")
                continue
            if len(tokens) != n:
                raise ValueError(f"{path}: row {len(rows)} has {len(tokens)} entries, expected {n}")
            vals = [int(t) for t in tokens]
            if any(v not in (0, 1) for v in vals):
                raise ValueError(f"{path}: entries must be 0 or 1")
            rows.append(vals)
    if m is None:
        raise ValueError(f"{path}: empty matrix file")
    if len(rows) != m:
        raise ValueError(f"{path}: header says {m} rows, found {len(rows)}")
    dense = np.asarray(rows, dtype=np.uint8) if m else np.zeros((0, n), dtype=np.uint8)
    return sp.csc_matrix(dense)


def _read_dbcsc(path: str) -> sp.csc_matrix:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 32:
        raise ValueError(f"{path}: truncated DBCSC header")
    if data[:8] != b"DBCSC001":
        raise ValueError(f"{path}: not a DBCSC file")
    nrows = int.from_bytes(data[8:16], "little")
    ncols = int.from_bytes(data[16:24], "little")
    nnz = int.from_bytes(data[24:32], "little")
    need = 32 + 4 * (ncols + 1) + 4 * nnz
    if len(data) != need:
        raise ValueError(f"{path}: DBCSC size mismatch (expected {need} bytes, have {len(data)})")
    colptr = np.frombuffer(data, dtype="<u4", count=ncols + 1, offset=32).astype(np.int64)
    rowind = np.frombuffer(data, dtype="<u4", count=nnz, offset=32 + 4 * (ncols + 1)).astype(np.int64)
    if colptr[0] != 0 or colptr[-1] != nnz or np.any(np.diff(colptr) < 0):
        raise ValueError(f"{path}: invalid DBCSC column pointers")
    m = sp.csc_matrix((np.ones(nnz, dtype=np.uint8), rowind, colptr), shape=(nrows, ncols))
    return _canonical_csc(m)


def _packed_rows(dense: np.ndarray) -> np.ndarray:
    """(r, n) 0/1 uint8 -> (r, ceil(n/64)) uint64 rows (LSB-first)."""
    dense = np.asarray(dense, dtype=np.uint8)
    r, n = dense.shape
    words = (n + 63) // 64
    bytes_per_row = 8 * words
    packed = np.packbits(dense, axis=1, bitorder="little")
    if packed.shape[1] < bytes_per_row:
        packed = np.pad(packed, ((0, 0), (0, bytes_per_row - packed.shape[1])))
    return np.ascontiguousarray(packed).view("<u8").reshape(r, words)


def _unpack_rows(packed: np.ndarray, n: int) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype="<u8")
    raw = packed.view(np.uint8).reshape(packed.shape[0], -1)
    return np.unpackbits(raw, axis=1, count=int(n), bitorder="little").astype(np.uint8)


def _lowest_set_bit(row: np.ndarray):
    for w, word in enumerate(row):
        v = int(word)
        if v:
            return w * 64 + (v & -v).bit_length() - 1
    return None


def _basis_insert(basis: dict, vector: np.ndarray) -> bool:
    """Insert a packed GF(2) vector into a row-echelon basis; True if it was independent."""
    v = vector.copy()
    while True:
        pivot = _lowest_set_bit(v)
        if pivot is None:
            return False
        if pivot in basis:
            v ^= basis[pivot]
        else:
            basis[pivot] = v
            return True


def _rref_packed(rows: np.ndarray):
    """Reduced row echelon form of packed rows; returns (rows, pivot_columns)."""
    rows = rows.copy()
    nrows = rows.shape[0]
    nwords = rows.shape[1]
    ncols = nwords * 64
    rank = 0
    pivots = []
    all_rows = np.arange(nrows)
    for c in range(ncols):
        word, bit = divmod(c, 64)
        col = (rows[:, word] >> np.uint64(bit)) & np.uint64(1)
        candidates = all_rows[rank:][col[rank:] != 0]
        if candidates.size == 0:
            continue
        pivot = int(candidates[0])
        if pivot != rank:
            rows[[rank, pivot]] = rows[[pivot, rank]]
        col = (rows[:, word] >> np.uint64(bit)) & np.uint64(1)
        mask = col.astype(bool)
        mask[rank] = False
        rows[mask] ^= rows[rank]
        pivots.append(c)
        rank += 1
        if rank == nrows:
            break
    return rows[:rank], pivots


def gf2_nullspace_packed(H_dense: np.ndarray) -> np.ndarray:
    """Basis of {x : H x = 0} as packed uint64 rows, from a dense 0/1 matrix."""
    dense = np.asarray(H_dense, dtype=np.uint8)
    m, n = dense.shape
    words = (n + 63) // 64
    packed = _packed_rows(dense)
    rref, pivots = _rref_packed(packed) if m else (np.zeros((0, words), dtype=np.uint64), [])
    pivot_set = set(pivots)
    basis = []
    for free in range(n):
        if free in pivot_set:
            continue
        vec = np.zeros(words, dtype=np.uint64)
        vec[free // 64] |= np.uint64(1) << np.uint64(free % 64)
        for i, pivot in enumerate(pivots):
            if (int(rref[i, free // 64]) >> (free % 64)) & 1:
                vec[pivot // 64] |= np.uint64(1) << np.uint64(pivot % 64)
        basis.append(vec)
    if not basis:
        return np.zeros((0, words), dtype=np.uint64)
    return np.asarray(basis, dtype=np.uint64)


def logical_observables_from_h(H) -> sp.csc_matrix:
    """A basis of ker(H)/row(H), for H H^T = 0 over GF(2); raises ValueError otherwise.

    For a stabilizer code whose rows are the commuting generators, ker(H) is the group of Pauli
    operators that commute with every generator and row(H) is the stabilizer group itself, so the
    quotient is the logical Pauli group.  The returned rows are an arbitrary (not canonical) basis.

    This is not a CSS logical basis: for one CSS half (say Hx with Hx Hx^T = 0) the quotient
    ker(Hx)/row(Hx) contains the Z-stabilizers of the code, which are not logical operators.  Such
    a problem needs an explicit observable matrix (``--o``).
    """
    H = _canonical_csc(H)
    m, n = H.shape
    if m == 0:
        raise ValueError("H has no rows; pass --o with the observable matrix")
    # Self-orthogonality: every pair of rows must commute (GF(2) dot product 0).
    gram = (H.astype(np.int32) @ H.T.astype(np.int32)).tocoo()
    if gram.nnz and np.any(gram.data % 2):
        raise ValueError("H is not self-orthogonal over GF(2) (H H^T != 0), so the logical "
                         "observables cannot be derived from H alone; pass --o FILE with the "
                         "observable matrix")
    dense = np.asarray(H.toarray(), dtype=np.uint8)
    null = gf2_nullspace_packed(dense)
    if null.shape[0] == 0:
        raise ValueError("H has a trivial nullspace; there are no observables to measure")
    row_basis: dict[int, np.ndarray] = {}
    for row in _packed_rows(dense):
        _basis_insert(row_basis, row)
    logicals = []
    for vec in null:
        v = vec.copy()
        # reduce against the rowspace, then keep it if it is independent
        while True:
            pivot = _lowest_set_bit(v)
            if pivot is None:
                break
            if pivot in row_basis:
                v ^= row_basis[pivot]
            else:
                break
        if _lowest_set_bit(v) is not None:
            _basis_insert(row_basis, v)
            logicals.append(v)
    if not logicals:
        raise ValueError("the code has no logical observables (ker(H) == row(H))")
    rows = _unpack_rows(np.asarray(logicals, dtype=np.uint64), n)
    return _canonical_csc(sp.csc_matrix(rows))
