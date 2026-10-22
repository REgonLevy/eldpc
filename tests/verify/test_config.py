# SPDX-License-Identifier: Apache-2.0
"""Validation of the suite configuration and of the ABI config strings derived from presets."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]      # the package root (tests/verify/x.py -> root)
if str(ROOT / "verify") not in sys.path:
    sys.path.insert(0, str(ROOT / "verify"))

from eldpc_verify.config import (ConfigError, load_config, problem_sha256,  # noqa: E402
                                 validate_config)
from eldpc_verify.runner import config_string  # noqa: E402

RAW = json.loads((ROOT / "verify" / "config.json").read_text())


def _cfg():
    return copy.deepcopy(RAW)


def test_shipped_config_is_valid():
    cfg = load_config()
    assert cfg["schema"] == "eldpc.verify.config.v1"
    assert cfg["base_seed"] == 20261006
    assert len(cfg["problems"]) == 9
    assert set(cfg["shots"]) == {"quick", "full"}
    for mode in ("quick", "full"):
        assert set(cfg["shots"][mode]) == {p["id"] for p in cfg["problems"]}
        assert cfg["configs"][mode]
    assert cfg["configs"]["quick"][:2] == ["a03125c3", "ms30"]
    assert {"a05c11", "a0375c7", "a05ms31"} <= set(cfg["configs"]["full"])
    loss = [p for p in cfg["problems"] if p["loss"] is not None]
    windows = [p for p in cfg["problems"] if p["shape"]["kind"] == "window"]
    assert [p["id"] for p in loss] == ["bb72_unif_p1e-3_r6_z_loss1e-2"]
    assert [p["id"] for p in windows] == ["bb144_unif_p1e-3_r12_z_win3c1"]
    assert windows[0]["shape"]["parent"] in {p["id"] for p in cfg["problems"]}


def test_problem_hash_is_stable_and_content_addressed():
    problem = RAW["problems"][0]
    assert problem_sha256(problem) == problem_sha256(copy.deepcopy(problem))
    other = copy.deepcopy(problem)
    other["rounds"] += 1
    assert problem_sha256(problem) != problem_sha256(other)


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c.update(schema="other"), "schema"),
    (lambda c: c.update(base_seed=-1), "base_seed"),
    (lambda c: c["problems"][0].update(unknown_key=1), "unknown keys"),
    (lambda c: c["problems"][0].update(family="nope"), "unknown family"),
    (lambda c: c["problems"][0].update(basis="w"), "basis"),
    (lambda c: c["problems"][0].update(rounds=0), "rounds"),
    (lambda c: next(p for p in c["problems"] if p["family"] == "surface")["code"].update(d=4),
     "odd int"),
    (lambda c: c["problems"][0]["noise"].update(p=0.9), "0.5"),
    (lambda c: c["problems"][1].update(id=c["problems"][0]["id"]), "duplicate problem id"),
    (lambda c: c["shots"]["quick"].pop("bb72_unif_p1e-3_r6_z"), "missing shot counts"),
    (lambda c: c["shots"]["quick"].update(no_such_problem=10), "unknown problem ids"),
    (lambda c: c["configs"]["quick"].append("a03125c3"), "duplicate preset names"),
    (lambda c: c["configs"].pop("full"), "missing keys"),
    (lambda c: c["problems"][1]["loss"].update(p_erase=0.9), "0.5"),
    (lambda c: c["problems"][1]["loss"].update(where="start"), "final"),
    (lambda c: c["problems"][3]["shape"].update(parent="no_such_problem"),
     "not an earlier problem"),
    (lambda c: c["problems"][3].update(rounds=11), "must repeat the parent"),
    (lambda c: c["problems"][3]["shape"].update(commit=4), "1 <= commit <= w"),
    (lambda c: c["problems"][3]["shape"].update(parent="bb72_unif_p1e-3_r6_z_loss1e-2"),
     "has loss"),
])
def test_invalid_configs_are_rejected(mutate, message):
    cfg = _cfg()
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        validate_config(cfg)


def test_extra_shot_count_for_unknown_problem_names_the_location():
    cfg = _cfg()
    cfg["shots"]["full"]["nope"] = 5
    with pytest.raises(ConfigError, match="shots.full"):
        validate_config(cfg)


def test_config_string_is_canonical():
    params = {"schedule": "parallel", "decoder": "bposd", "max_iter": 3,
              "ms_scaling_factor": 0.3125, "bp_method": "ms", "osd_method": "osd0",
              "osd_order": 0}
    text = config_string(params)
    assert text == ("bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;"
                    "osd_method=osd0;osd_order=0;schedule=parallel")
    assert config_string({"b": True, "a": False, "c": 2}) == "a=false;b=true;c=2"
