# SPDX-License-Identifier: Apache-2.0
"""Preset names in config strings: ``"a03125c3"`` and ``"a03125c3;threads=8"``.

A config string may start with a preset name, followed by extra ``key=value`` segments that add to
the preset's keys or override them. The C front (src/port/eldpc_front.cpp, ``expand_preset``) and
the Python package (``eldpc._config.resolve_config``) expand it the same way; the C front's table
must hold exactly the package's presets.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import eldpc
from eldpc import _abi
from eldpc._config import config_string, presets, resolve_config
from eldpc._matrix import build_problem, canonical_csc
from eldpc_testlib import CUDA_DEVICES, cpu_core, random_problem, sampled_syndromes

FRONT = Path(__file__).resolve().parents[1] / "src" / "port" / "eldpc_front.cpp"
CPU = ";device=cpu"


def _front_table() -> dict[str, str]:
    source = FRONT.read_text(encoding="utf-8")
    block = source[source.index("const Preset kPresets[] = {"):]
    block = block[:block.index("\n};")]
    return dict(re.findall(r'\{"([A-Za-z0-9_.]+)",\s*"([^"]+)"\}', block))


def test_the_c_front_table_holds_the_package_presets():
    table = _front_table()
    assert table == {name: config_string(cfg) for name, cfg in presets().items()}


class _Raw:
    """Handles created straight through the C API with a raw config string (no Python resolution)."""

    def __init__(self):
        H, O, priors = random_problem(24, 60, 2, seed=11)
        self.problem, self._keepalive = build_problem(canonical_csc(H), canonical_csc(O), priors, None)
        self.syndromes = sampled_syndromes(H, priors, 120, seed=5)

    def outputs(self, config: str):
        h = _abi.Handle(eldpc.load(), config, self.problem)
        try:
            out = [h.decode(s) for s in self.syndromes]
        finally:
            h.close()
        return [(c.tobytes(), o.tobytes(), st) for c, o, st in out]

    def error(self, config: str) -> str:
        with pytest.raises(eldpc.EldpcError) as info:
            _abi.Handle(eldpc.load(), config, self.problem).close()
        return str(info.value)


@pytest.fixture(scope="module")
def raw():
    return _Raw()


@cpu_core
def test_every_preset_name_creates_the_presets_handle(raw):
    for name, cfg in presets().items():
        assert raw.outputs(name + CPU) == raw.outputs(config_string(cfg) + CPU), name


@cpu_core
def test_extra_keys_add_to_the_preset(raw):
    explicit = config_string(presets()["a03125c3"])
    got = raw.outputs("a03125c3;threads=2" + CPU)
    assert got == raw.outputs(explicit + ";threads=2" + CPU)
    assert got == raw.outputs(" a03125c3 ; device=cpu ; threads=2 ;")
    assert raw.outputs("a03125c3;" + CPU) == raw.outputs(explicit + CPU)


@cpu_core
def test_extra_keys_override_the_preset(raw):
    explicit = config_string(dict(presets()["a03125c3"], max_iter=7))
    # the preset's max_iter=3 is replaced, not repeated (a repeat would be a duplicate key)
    assert raw.outputs("a03125c3;max_iter=7" + CPU) == raw.outputs(explicit + CPU)
    assert "invalid max_iter 'x'" in raw.error("a03125c3;max_iter=x" + CPU)


@cpu_core
def test_preset_prefix_errors(raw):
    assert "unknown preset \"nosuch\"" in raw.error("nosuch;threads=2" + CPU)
    assert "duplicate config key 'threads'" in raw.error("a03125c3;threads=2;threads=3" + CPU)
    assert "malformed config segment 'ms30'" in raw.error("a03125c3;ms30" + CPU)


def test_a_bare_preset_on_the_default_device(raw):
    suffix = "" if CUDA_DEVICES else CPU
    assert raw.outputs("a03125c3" + suffix) == raw.outputs(config_string(presets()["a03125c3"]) + suffix)


@cpu_core
def test_python_resolves_a_preset_with_extra_keys(raw):
    base = presets()["a03125c3"]
    assert resolve_config("a03125c3;threads=8") == {**base, "threads": "8"}
    assert resolve_config("ms30; max_iter = 50")["max_iter"] == "50"
    assert resolve_config("a03125c3;") == base  # as in the C front: empty segments are ignored
    with pytest.raises(eldpc.EldpcError, match="unknown preset 'nosuch'"):
        resolve_config("nosuch;threads=2")
    with pytest.raises(eldpc.EldpcError, match="duplicate"):
        resolve_config("a03125c3;threads=2;threads=3")
    H, O, priors = random_problem(24, 60, 2, seed=11)
    syndromes = sampled_syndromes(H, priors, 60, seed=9)
    with eldpc.Decoder(H, O, priors, config="a03125c3;threads=2", device="cpu") as a, \
            eldpc.Decoder(H, O, priors, config="a03125c3", device="cpu", threads=2) as b:
        assert a.config_string == b.config_string
        assert a.config_string == config_string({**base, "device": "cpu", "threads": 2})
        for s in syndromes:
            ca, oa, sa = a.decode_full(s)
            cb, ob, sb = b.decode_full(s)
            assert np.array_equal(ca, cb) and np.array_equal(oa, ob) and sa == sb
    with pytest.raises(ValueError, match="threads"):  # threads= is a CPU-core key
        eldpc.Decoder(H, O, priors, config="a03125c3;threads=2", device="gpu")
