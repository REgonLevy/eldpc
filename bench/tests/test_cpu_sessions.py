# SPDX-License-Identifier: Apache-2.0
"""CPU-core sessions: one live handle at a time, the threads/CPU check, and preset-prefixed configs."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import worker  # noqa: E402
from benchlib import cdriver, formats, measure  # noqa: E402

PRESET = {"decoder": "bposd", "max_iter": 3, "ms_scaling_factor": 0.3125, "schedule": "parallel",
          "bp_method": "ms", "osd_method": "osd0", "osd_order": 0}
PRESET_STRING = ("bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;osd_method=osd0;"
                 "osd_order=0;schedule=parallel")


class _FakeEldpc:
    def presets(self):
        return {"a03125c3": dict(PRESET)}


def _bench(tmp_path):
    args = types.SimpleNamespace(out=str(tmp_path), quick=True)
    return measure.Bench(args, None, BENCH)


def _command(sequential):
    return cdriver.driver_command("drv", "p", "s", "o", "la", "lb", "ca", "cb", "ab", "b2b", 0, False,
                                  False, 10, 5, None, sequential=sequential)


def test_driver_command_is_unchanged_unless_sequential():
    assert "--handles" not in _command(False)
    assert _command(True)[-2:] == ["--handles", "sequential"]
    assert _command(True)[:-2] == _command(False)


def test_driver_source_has_the_sequential_schedule():
    source = (BENCH / "c" / "bench_driver.c").read_text(encoding="utf-8")
    assert '"--handles"' in source and "handles=sequential" in source
    assert 'die("--handles must be both or sequential", handles);' in source


def test_device_of_a_config_string():
    assert measure.resolve_device(PRESET_STRING, None) == "gpu"
    assert measure.resolve_device(PRESET_STRING + ";device=cpu;threads=8", None) == "cpu"
    assert measure.resolve_device("device=gpu", None) == "gpu"
    # device=auto that cannot be resolved here counts as the GPU (and is checked after the session)
    assert measure.resolve_device("device=auto", None) == "gpu"


def test_handle_schedule_record(tmp_path):
    bench = _bench(tmp_path)
    assert bench.handle_schedule()["design"] == "interleaved"
    bench.sequential, bench.cpu_core_sides = True, ["A"]
    rec = bench.handle_schedule()
    assert rec["design"] == "sequential" and rec["cpu_core_sides"] == ["A"]
    assert "one live handle at a time" in rec["what"]


def test_two_live_cpu_handles_are_warned(tmp_path):
    bench = _bench(tmp_path)
    bench.check_handle_schedule("c_latency", ["default", "default"])
    assert not bench.warnings
    bench.check_handle_schedule("c_latency", ["cpu", "cpu"])
    assert any("two CPU-core handles were live" in w for w in bench.warnings)
    quiet = _bench(tmp_path)
    quiet.sequential = True
    quiet.check_handle_schedule("c_latency", ["cpu", "cpu"])
    assert not quiet.warnings


@pytest.mark.parametrize("config,affinity,warned", [
    ("device=cpu;threads=8", {0}, True),                 # --cpu pins the run to one CPU
    ("device=cpu;threads=8", set(range(8)), False),      # taskset -c 0-7
    ("device=cpu;threads=8", set(range(4)), True),
    ("device=cpu", {0}, False),                          # threads default to 1
    ("device=cpu;threads=4;cpus=0-3", {0}, False),       # an explicit CPU list
    ("device=cpu;threads=4;cpus=0,0,1,1", set(range(8)), True),
])
def test_more_threads_than_cpus_is_warned(tmp_path, monkeypatch, config, affinity, warned):
    monkeypatch.setattr(measure.os, "sched_getaffinity", lambda pid: set(affinity), raising=False)
    bench = _bench(tmp_path)
    bench._check_cpu_threads(measure.Variant("A", "x", config, {"path": "lib"}))
    assert any("busy-waiting workers share cores" in w for w in bench.warnings) == warned


def test_cpu_list_size():
    assert measure.cpu_list_size("0-7") == 8
    assert measure.cpu_list_size("0,0,1") == 2
    assert measure.cpu_list_size("3-1") is None
    assert measure.cpu_list_size("x") is None


def _files(tmp_path):
    H = sp.csc_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    O = sp.csc_matrix(np.array([[1, 0, 0]], dtype=np.uint8))
    problem_file = tmp_path / "problem.bin"
    shots_file = tmp_path / "shots.bin"
    formats.write_problem(problem_file, H, O, np.array([0.1, 0.2, 0.3]), None)
    dets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    formats.write_shots(shots_file, dets, np.zeros((4, 1), dtype=np.uint8))
    return problem_file, shots_file


class _CountingEldpc:
    """A fake package whose decoders record how many are live at once and which config decodes."""

    def __init__(self):
        self.live = 0
        self.max_live = 0
        self.calls = []

    def Decoder(self, H_, O_, priors_, config=None):  # noqa: N802 - the package API
        outer = self

        class FakeDecoder:
            last_status = 1
            core_variant = "cpu"

            def __init__(self):
                outer.live += 1
                outer.max_live = max(outer.max_live, outer.live)
                self.open = True

            def decode(self, syndrome):
                outer.calls.append(config)
                return np.zeros(3, dtype=np.uint8)

            def decode_batch(self, syndromes):
                outer.calls.append(config)
                n = len(syndromes)
                return (np.zeros((n, 3), dtype=np.uint8), np.zeros((n, 1), dtype=np.uint8),
                        np.ones(n, dtype=np.uint32))

            def close(self):
                if self.open:
                    outer.live -= 1
                    self.open = False

        return FakeDecoder()


@pytest.mark.parametrize("sequential,max_live", [(False, 2), (True, 1)])
def test_worker_sequential_sessions_hold_one_decoder(tmp_path, sequential, max_live):
    problem_file, shots_file = _files(tmp_path)
    fake = _CountingEldpc()
    job = {"kind": "py_latency", "problem_file": str(problem_file), "shots_file": str(shots_file),
           "variants": [{"config_string": "A"}, {"config_string": "B"}],
           "order": [1, 0], "create_order": [1, 0], "shots_per_variant": 3,
           "warmup": 0, "gap_us": 0, "keep_warm": False, "sequential": sequential}
    out = worker.run_py_latency(fake, job)
    assert fake.max_live == max_live and fake.live == 0
    assert [len(v["dur_ns"]) for v in out["variants"]] == [3, 3]
    assert out["core_variants"] == ["cpu", "cpu"]
    if sequential:  # B's block first (order [1, 0]), then A's
        assert fake.calls == ["B"] * 3 + ["A"] * 3
    else:
        assert fake.calls == ["B", "A"] * 3
    tp = {"kind": "throughput", "problem_file": str(problem_file), "shots_file": str(shots_file),
          "variants": [{"config_string": "A"}, {"config_string": "B"}], "order": [0, 1],
          "create_order": [0, 1], "batch_shots": 2, "repetitions": 2, "warmup": 0,
          "sequential": sequential}
    fake = _CountingEldpc()
    res = worker.run_throughput(fake, tp)
    assert fake.max_live == max_live and fake.live == 0
    assert sorted(b["variant"] for b in res["batches"]) == [0, 0, 1, 1]
    assert res["checksum"][0] == res["checksum"][1]


def test_preset_name_with_extra_keys():
    label, s = measure.resolve_config("a03125c3;threads=8", _FakeEldpc())
    assert label == "a03125c3;threads=8"
    assert s == PRESET_STRING + ";threads=8"
    # an extra key overrides the preset's value; the result stays in key order
    assert measure.resolve_config("a03125c3; max_iter = 7", _FakeEldpc())[1] == \
        PRESET_STRING.replace("max_iter=3", "max_iter=7")
    with measure_error("unknown preset"):
        measure.resolve_config("nosuch;threads=8", _FakeEldpc())
    with measure_error("duplicate"):
        measure.resolve_config("a03125c3;threads=8;threads=4", _FakeEldpc())
    assert measure.resolve_config("a03125c3;", _FakeEldpc())[1] == PRESET_STRING
    # an ABI string without a preset is unchanged (sorted)
    assert measure.resolve_config("b=1;a=2", _FakeEldpc())[1] == "a=2;b=1"


def measure_error(match):
    return pytest.raises(measure.MeasurementError, match=match)
