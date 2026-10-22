# SPDX-License-Identifier: Apache-2.0
"""The verification suite's configuration file (``verify/config.json``).

Schema ``eldpc.verify.config.v1``::

    {"schema": "eldpc.verify.config.v1",
     "base_seed": <int >= 0>,
     "problems": [<problem>, ...],
     "shots": {"quick": {<id>: <int >= 1>, ...}, "full": {...}},
     "configs": {"quick": [<preset name>, ...], "full": [...]},
     "upstream": <bool, optional, default true>}

A problem uses the fields of the reference construction: ``id``, ``family`` (surface, bb, hgp, css),
``code`` (family-specific), ``basis`` (z/x for surface, z/x/xyz for the others), ``rounds``,
``noise`` (uniform, si1000 or custom), ``dem`` (``{"decompose_errors": bool}``, optional), ``loss``
(``{"p_erase": p, "where": "final"}``, optional: heralded erasure before the final readout) and
``shape`` (optional; ``{"kind": "window", "parent": <id>, "start_round": r, "w": w,
"commit": c}`` for a windowed problem, whose family/code/basis/rounds/noise/dem must repeat the
parent's values). Every shot count is per problem and per mode; every mode needs at least one
problem, one preset and a shot count for every listed problem. Validation errors are raised as
``ConfigError`` with the offending location in the message.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from . import CONFIG_SCHEMA

FAMILIES = ("surface", "bb", "hgp", "css")
MODES = ("quick", "full")
_ALLOWED_PROBLEM_KEYS = frozenset({"id", "family", "code", "basis", "rounds", "noise", "dem",
                                   "loss", "shape"})
_REQUIRED_PROBLEM_KEYS = ("id", "family", "code", "basis", "rounds", "noise")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_WINDOW_KEYS = ("kind", "parent", "start_round", "w", "commit")


class ConfigError(ValueError):
    """Invalid suite configuration."""


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _err(where: str, msg: str):
    raise ConfigError(f"{where}: {msg}")


def _check_keys(obj: dict, allowed, required, where: str) -> None:
    extra = sorted(set(obj) - set(allowed))
    if extra:
        _err(where, f"unknown keys {extra}")
    missing = sorted(set(required) - set(obj))
    if missing:
        _err(where, f"missing keys {missing}")


def _matrix(value, where: str) -> list:
    """A non-empty list of equal-length rows of 0/1 ints."""
    if not isinstance(value, list) or not value:
        _err(where, "must be a non-empty list of rows")
    width = None
    out = []
    for i, row in enumerate(value):
        if not isinstance(row, list) or not row:
            _err(where, f"row {i} must be a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            _err(where, f"row {i} has {len(row)} entries, expected {width}")
        for v in row:
            if not _is_int(v) or v not in (0, 1):
                _err(where, f"row {i} has non-0/1 entry {v!r}")
        out.append([int(v) for v in row])
    return out


def _resolve_matrix(value, where: str) -> list:
    """hgp matrix forms: rows, {"rows": rows} or {"circulant": row0}."""
    if isinstance(value, dict):
        if set(value) == {"rows"}:
            return _matrix(value["rows"], where)
        if set(value) == {"circulant"}:
            row0 = value["circulant"]
            if not isinstance(row0, list) or not row0:
                _err(where, '{"circulant": ...} needs a non-empty row')
            for v in row0:
                if not _is_int(v) or v not in (0, 1):
                    _err(where, f"circulant has non-0/1 entry {v!r}")
            n = len(row0)
            return [[int(row0[(j - i) % n]) for j in range(n)] for i in range(n)]
        _err(where, f"unknown matrix form {sorted(value)}")
    return _matrix(value, where)


def _validate_code(family: str, code, where: str) -> dict:
    if not isinstance(code, dict):
        _err(where, f"must be an object, got {type(code).__name__}")
    if family == "surface":
        _check_keys(code, ("d",), ("d",), where)
        d = code["d"]
        if not _is_int(d) or d < 3 or d % 2 == 0:
            _err(where, f'"d" must be an odd int >= 3, got {d!r}')
        return {"d": int(d)}
    if family == "bb":
        _check_keys(code, ("l", "m", "A", "B"), ("l", "m", "A", "B"), where)
        out = {}
        for key in ("l", "m"):
            v = code[key]
            if not _is_int(v) or v < 1:
                _err(where, f'"{key}" must be an int >= 1, got {v!r}')
            out[key] = int(v)
        for key in ("A", "B"):
            monos = code[key]
            if not isinstance(monos, list) or not monos:
                _err(where, f'"{key}" must be a non-empty list of [a, b] monomials')
            parsed = []
            for i, mono in enumerate(monos):
                if not isinstance(mono, (list, tuple)) or len(mono) != 2:
                    _err(where, f'"{key}"[{i}] must be a [a, b] pair of ints')
                for x in mono:
                    if not _is_int(x) or x < 0:
                        _err(where, f'"{key}"[{i}] has invalid exponent {x!r}')
                parsed.append([int(mono[0]), int(mono[1])])
            out[key] = parsed
        return out
    if family == "hgp":
        _check_keys(code, ("h1", "h2"), ("h1", "h2"), where)
        return {"h1": _resolve_matrix(code["h1"], f"{where}.h1"),
                "h2": _resolve_matrix(code["h2"], f"{where}.h2")}
    if family == "css":
        _check_keys(code, ("hx", "hz"), ("hx", "hz"), where)
        hx = _matrix(code["hx"], f"{where}.hx")
        hz = _matrix(code["hz"], f"{where}.hz")
        if len(hx[0]) != len(hz[0]):
            _err(where, f"hx has {len(hx[0])} columns but hz has {len(hz[0])}")
        import numpy as np

        if np.any((np.asarray(hx, dtype=np.int64) @ np.asarray(hz, dtype=np.int64).T) % 2):
            _err(where, "some hx row does not commute with some hz row (odd overlap)")
        return {"hx": hx, "hz": hz}
    _err(where, f"unknown family {family!r}")


def _validate_noise(noise, where: str) -> dict:
    if not isinstance(noise, dict):
        _err(where, f"must be an object, got {type(noise).__name__}")
    model = noise.get("model")
    if model in ("uniform", "si1000"):
        _check_keys(noise, ("model", "p"), ("model", "p"), where)
        p = noise["p"]
        if not _is_num(p) or not (0.0 < float(p) < 0.5):
            _err(where, f'"p" must be a float in (0, 0.5), got {p!r}')
        return {"model": model, "p": float(p)}
    if model == "custom":
        keys = ("p1", "p2", "p_meas", "p_reset", "p_idle")
        _check_keys(noise, ("model",) + keys, ("model",) + keys, where)
        out = {"model": "custom"}
        for key in keys:
            v = noise[key]
            if not _is_num(v) or not (0.0 <= float(v) < 0.5):
                _err(where, f'"{key}" must be a float in [0, 0.5), got {v!r}')
            out[key] = float(v)
        return out
    _err(where, f'unknown noise model {model!r} (expected "uniform", "si1000" or "custom")')


def _validate_loss(loss, where: str):
    """Optional heralded-erasure loss before the final readout (the reference construction's loss
    rule)."""
    if loss is None:
        return None
    if not isinstance(loss, dict):
        _err(where, f"must be an object or null, got {type(loss).__name__}")
    _check_keys(loss, ("p_erase", "where"), ("p_erase", "where"), where)
    p = loss["p_erase"]
    if not _is_num(p) or not (0.0 < float(p) <= 0.5):
        _err(where, f'"p_erase" must be a float in (0, 0.5], got {p!r}')
    if loss["where"] != "final":
        _err(where, f'"where" must be "final", got {loss["where"]!r}')
    return {"p_erase": float(p), "where": "final"}


def _validate_shape(shape, where: str) -> dict:
    """Optional problem shape: full (default) or a window of a parent problem."""
    if shape is None:
        return {"kind": "full"}
    if not isinstance(shape, dict):
        _err(where, f"must be an object, got {type(shape).__name__}")
    kind = shape.get("kind")
    if kind == "full":
        _check_keys(shape, ("kind",), ("kind",), where)
        return {"kind": "full"}
    if kind != "window":
        _err(where, f'"kind" must be "full" or "window", got {kind!r}')
    _check_keys(shape, _WINDOW_KEYS, _WINDOW_KEYS, where)
    parent = shape["parent"]
    if not isinstance(parent, str) or not _ID_RE.match(parent):
        _err(where, f'"parent" must be a problem id string, got {parent!r}')
    start = shape["start_round"]
    if not _is_int(start) or start < 0:
        _err(where, f'"start_round" must be an int >= 0, got {start!r}')
    w = shape["w"]
    if not _is_int(w) or w < 1:
        _err(where, f'"w" must be an int >= 1, got {w!r}')
    commit = shape["commit"]
    if not _is_int(commit) or not (1 <= commit <= w):
        _err(where, f'"commit" must be an int with 1 <= commit <= w, got {commit!r}')
    return {"kind": "window", "parent": parent, "start_round": int(start), "w": int(w),
            "commit": int(commit)}


def _validate_problem(obj, where: str) -> dict:
    if not isinstance(obj, dict):
        _err(where, f"must be an object, got {type(obj).__name__}")
    _check_keys(obj, _ALLOWED_PROBLEM_KEYS, _REQUIRED_PROBLEM_KEYS, where)
    pid = obj["id"]
    if not isinstance(pid, str) or not _ID_RE.match(pid):
        _err(where, f"invalid id {pid!r}")
    family = obj["family"]
    if family not in FAMILIES:
        _err(where, f"unknown family {family!r} (expected one of {list(FAMILIES)})")
    code = _validate_code(family, obj["code"], f"{where}.code")
    basis = obj["basis"]
    allowed = ("z", "x") if family == "surface" else ("z", "x", "xyz")
    if basis not in allowed:
        _err(where, f"basis {basis!r} not allowed for family {family!r}")
    rounds = obj["rounds"]
    if not _is_int(rounds) or rounds < 1:
        _err(where, f'"rounds" must be an int >= 1, got {rounds!r}')
    noise = _validate_noise(obj["noise"], f"{where}.noise")
    dem = obj.get("dem", {"decompose_errors": False})
    if not isinstance(dem, dict):
        _err(where + ".dem", f"must be an object, got {type(dem).__name__}")
    _check_keys(dem, ("decompose_errors",), ("decompose_errors",), where + ".dem")
    if not isinstance(dem["decompose_errors"], bool):
        _err(where + ".dem", '"decompose_errors" must be a bool')
    loss = _validate_loss(obj.get("loss"), where + ".loss")
    shape = _validate_shape(obj.get("shape"), where + ".shape")
    if shape["kind"] == "window" and loss is not None:
        _err(where, "a window problem cannot have loss")
    return {"id": pid, "family": family, "code": code, "basis": basis, "rounds": int(rounds),
            "noise": noise, "dem": {"decompose_errors": dem["decompose_errors"]},
            "loss": loss, "shape": shape}


def _validate_shots(shots, problem_ids, where: str) -> dict:
    if not isinstance(shots, dict):
        _err(where, f"must be an object, got {type(shots).__name__}")
    _check_keys(shots, MODES, MODES, where)
    out = {}
    for mode in MODES:
        table = shots[mode]
        if not isinstance(table, dict):
            _err(f"{where}.{mode}", f"must be an object, got {type(table).__name__}")
        extra = sorted(set(table) - set(problem_ids))
        if extra:
            _err(f"{where}.{mode}", f"unknown problem ids {extra}")
        missing = sorted(set(problem_ids) - set(table))
        if missing:
            _err(f"{where}.{mode}", f"missing shot counts for {missing}")
        out[mode] = {}
        for pid, n in table.items():
            if not _is_int(n) or n < 1:
                _err(f"{where}.{mode}.{pid}", f"must be an int >= 1, got {n!r}")
            out[mode][pid] = int(n)
    return out


def _validate_configs(configs, where: str) -> dict:
    if not isinstance(configs, dict):
        _err(where, f"must be an object, got {type(configs).__name__}")
    _check_keys(configs, MODES, MODES, where)
    out = {}
    for mode in MODES:
        names = configs[mode]
        if not isinstance(names, list) or not names:
            _err(f"{where}.{mode}", "must be a non-empty list of preset names")
        for name in names:
            if not isinstance(name, str) or not name:
                _err(f"{where}.{mode}", f"invalid preset name {name!r}")
        if len(set(names)) != len(names):
            _err(f"{where}.{mode}", f"duplicate preset names in {names}")
        out[mode] = list(names)
    return out


def validate_config(cfg) -> dict:
    """Validate a parsed configuration object and return it normalised."""
    if not isinstance(cfg, dict):
        raise ConfigError(f"configuration must be an object, got {type(cfg).__name__}")
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f'config schema must be "{CONFIG_SCHEMA}", got {cfg.get("schema")!r}')
    allowed = {"schema", "comment", "base_seed", "problems", "shots", "configs", "upstream"}
    extra = sorted(set(cfg) - allowed)
    if extra:
        raise ConfigError(f"unknown keys {extra}")
    base_seed = cfg.get("base_seed")
    if not _is_int(base_seed) or base_seed < 0:
        raise ConfigError(f'"base_seed" must be an int >= 0, got {base_seed!r}')
    problems = cfg.get("problems")
    if not isinstance(problems, list) or not problems:
        raise ConfigError('"problems" must be a non-empty list')
    parsed = []
    seen = {}
    for i, obj in enumerate(problems):
        p = _validate_problem(obj, f"problems[{i}]")
        if p["id"] in seen:
            raise ConfigError(f"problems[{i}]: duplicate problem id {p['id']!r}")
        if p["shape"]["kind"] == "window":
            parent_id = p["shape"]["parent"]
            if parent_id not in seen:
                raise ConfigError(f"problems[{i}]: window parent {parent_id!r} is not an earlier "
                                  f"problem")
            parent = seen[parent_id]
            if parent["shape"]["kind"] == "window":
                raise ConfigError(f"problems[{i}]: window parent {parent_id!r} is itself a window")
            if parent["loss"] is not None:
                raise ConfigError(f"problems[{i}]: window parent {parent_id!r} has loss")
            for key in ("family", "code", "basis", "rounds", "noise", "dem"):
                if p[key] != parent[key]:
                    raise ConfigError(f"problems[{i}].{key} must repeat the parent's value")
        seen[p["id"]] = p
        parsed.append(p)
    ids = [p["id"] for p in parsed]
    shots = _validate_shots(cfg.get("shots"), ids, "shots")
    configs = _validate_configs(cfg.get("configs"), "configs")
    upstream = cfg.get("upstream", True)
    if not isinstance(upstream, bool):
        raise ConfigError(f'"upstream" must be a bool, got {upstream!r}')
    out = {"schema": CONFIG_SCHEMA, "base_seed": int(base_seed), "problems": parsed,
           "shots": shots, "configs": configs, "upstream": upstream}
    if "comment" in cfg:
        if not isinstance(cfg["comment"], str):
            raise ConfigError('"comment" must be a string')
        out["comment"] = cfg["comment"]
    return out


def load_config(path=None) -> dict:
    """Load and validate the configuration; ``path=None`` uses the suite's own config.json."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.json"
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"configuration {path} is not valid JSON: {exc}") from exc
    try:
        return validate_config(cfg)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def problem_sha256(problem: dict) -> str:
    """SHA-256 of a problem object's canonical JSON (the corpus cache key component)."""
    blob = json.dumps(problem, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()


def config_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_cache_root() -> Path:
    """$ELDPC_VERIFY_CACHE, else $XDG_CACHE_HOME/eldpc/verify, else ~/.cache/eldpc/verify."""
    explicit = os.environ.get("ELDPC_VERIFY_CACHE")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "eldpc" / "verify"
