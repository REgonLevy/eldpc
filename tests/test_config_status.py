# SPDX-License-Identifier: Apache-2.0
"""Configurations and status helpers; no GPU needed."""

from __future__ import annotations

import json
import re

import pytest

import eldpc
from eldpc_testlib import cpu_core
from eldpc._config import config_string, presets, resolve_config

pytestmark = cpu_core


def test_presets_names_and_values():
    p = presets()
    expected = {
        "ms30": (0.625, 30), "ms100": (0.625, 100), "ms30a075": (0.75, 30), "a05ms31": (0.5, 31),
        "a05c15": (0.5, 15), "a05c11": (0.5, 11), "a0375c7": (0.375, 7), "a04375c3": (0.4375, 3),
        "a03125c3": (0.3125, 3),
    }
    assert len(p) == 2 * len(expected)
    for name, (alpha, iters) in expected.items():
        cfg = p[name]
        assert cfg == {"decoder": "bposd", "bp_method": "ms", "schedule": "parallel",
                       "osd_method": "osd0", "osd_order": 0,
                       "ms_scaling_factor": alpha, "max_iter": iters}
        bp = p[name + ".bp"]
        assert bp == {"decoder": "bp", "bp_method": "ms", "schedule": "parallel",
                      "ms_scaling_factor": alpha, "max_iter": iters}


def test_config_string_is_canonical():
    cfg = resolve_config("a03125c3")
    assert config_string(cfg) == ("bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;"
                                  "osd_method=osd0;osd_order=0;schedule=parallel")


def test_resolve_preset_dict_string_and_file(tmp_path):
    assert resolve_config("ms30")["max_iter"] == 30
    assert resolve_config("ms30.bp")["decoder"] == "bp"
    d = {"decoder": "bp", "bp_method": "ms", "max_iter": 5, "schedule": "parallel"}
    assert resolve_config(d) == d
    assert resolve_config(d) is not d
    parsed = resolve_config("decoder=bposd;bp_method=ms;max_iter=3;schedule=parallel;"
                            "osd_method=osd0;osd_order=0;ms_scaling_factor=0.3125")
    assert parsed["max_iter"] == "3" and parsed["ms_scaling_factor"] == "0.3125"
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"decoder": "bp", "max_iter": 2}))
    assert resolve_config(str(path)) == {"decoder": "bp", "max_iter": 2}
    assert resolve_config(path) == {"decoder": "bp", "max_iter": 2}


def test_config_errors(tmp_path):
    with pytest.raises(eldpc.EldpcError, match="unknown preset"):
        resolve_config("no-such-preset")
    with pytest.raises(eldpc.EldpcError, match="not found"):
        resolve_config(str(tmp_path / "missing.json"))
    with pytest.raises(eldpc.EldpcError, match="malformed"):
        resolve_config("decoder=bposd;a=b=c")
    with pytest.raises(eldpc.EldpcError, match="duplicate"):
        resolve_config("max_iter=1;max_iter=2")
    with pytest.raises(eldpc.EldpcError, match="invalid config key"):
        resolve_config("Max=1")
    with pytest.raises(eldpc.EldpcError, match="config values"):
        config_string({"max_iter": [3]})
    with pytest.raises(eldpc.EldpcError, match="config must be"):
        resolve_config(42)


def test_config_string_escapes_nothing_but_validates_keys():
    assert config_string({"a": True, "b": False, "c": 7, "d": 0.5, "e": "x"}) == "a=true;b=false;c=7;d=0.5;e=x"
    with pytest.raises(eldpc.EldpcError, match="invalid config key"):
        config_string({"bad-key": 1})


def test_status_helpers():
    assert eldpc.ST_BP_CONVERGED == 0x1
    assert eldpc.ST_POSTPROC_INVOKED == 0x2
    assert eldpc.ST_HIGHER_ORDER_WIN == 0x4
    assert eldpc.ST_ABSTAIN == 0x8
    assert eldpc.ST_SYNDROME_UNMATCHED == 0x10
    assert eldpc.ST_ITER_SHIFT == 16
    st = 0x0003 | (7 << 16)
    assert eldpc.status.converged(st) is True
    assert eldpc.status.postprocessed(st) is True
    assert eldpc.status.iterations(st) == 7
    assert eldpc.status.converged(0x0002) is False
    assert eldpc.status.postprocessed(0x0001) is False
    assert eldpc.status.iterations(0xFFFF_0001) == 0xFFFF
    assert eldpc.ST_FLAGS_MASK == 0xFFFF


def test_version_matches_the_pyproject():
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("the test suite is running outside the source tree")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    assert m and m.group(1) == eldpc.__version__
