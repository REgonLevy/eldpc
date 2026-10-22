# SPDX-License-Identifier: Apache-2.0
"""The core variant of every timed handle: driver output, worker records, environment, report."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import worker  # noqa: E402
from benchlib import cdriver, formats, measure, report  # noqa: E402


def test_driver_core_variant_lines_are_parsed():
    text = ("bench_driver: core_variant 0 ada64\n"
            "bench_driver: core_variant 1 default\n"
            "bench_driver: 300 shots per variant, mode=b2b gap=0us keep_warm=0/0, rc!=0: 0, "
            "checksum 0123456789abcdef\n")
    assert cdriver.core_variants_from_stderr(text) == ["ada64", "default"]
    # an older driver that prints no such line
    assert cdriver.core_variants_from_stderr("bench_driver: 300 shots per variant") == [
        "unknown", "unknown"]
    assert cdriver.core_variants_from_stderr("") == ["unknown", "unknown"]


def test_driver_source_reports_the_variant_of_both_handles():
    source = (BENCH / "c" / "bench_driver.c").read_text(encoding="utf-8")
    assert 'dlsym(so[v], "eldpc_core_variant")' in source
    assert "bench_driver: core_variant %d %s" in source


def _bench(tmp_path):
    args = types.SimpleNamespace(out=str(tmp_path), quick=True)
    return measure.Bench(args, None, BENCH)


def test_environment_record_names_the_variant_of_every_side(tmp_path):
    bench = _bench(tmp_path)
    for _ in range(3):
        bench.note_core_variants("c_latency", ("A", "A"), ["ada64", "ada64"])
        bench.note_core_variants("c_latency", ("A", "B"), ["ada64", "ada64"])
    bench.note_core_variants("py_latency", ("A", "B"), ["default", "default"])
    bench.note_core_variants("ler", ("A",), ["cpu"])
    bench.note_core_variants("throughput", ("A", "A"), None)  # a worker that reported nothing
    record = bench.core_variant_record()
    assert record == {"c_latency": {"A": "ada64", "B": "ada64"},
                      "py_latency": {"A": "default", "B": "default"},
                      "ler": {"A": "cpu"},
                      "throughput": {"A": "unknown"}}
    assert bench.warnings == []


def test_mixed_variants_on_one_side_are_listed_and_warned(tmp_path):
    bench = _bench(tmp_path)
    bench.note_core_variants("c_latency", ("A", "A"), ["ada64", "default"])
    record = bench.core_variant_record()
    assert record["c_latency"]["A"] == ["ada64", "default"]
    assert any("different core variants" in w for w in bench.warnings)


def test_worker_reports_the_core_variant_of_its_handles(tmp_path):
    H = sp.csc_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    O = sp.csc_matrix(np.array([[1, 0, 0]], dtype=np.uint8))
    problem_file = tmp_path / "problem.bin"
    shots_file = tmp_path / "shots.bin"
    formats.write_problem(problem_file, H, O, np.array([0.1, 0.2, 0.3]), None)
    dets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    formats.write_shots(shots_file, dets, np.zeros((4, 1), dtype=np.uint8))

    class FakeDecoder:
        last_status = 1

        def __init__(self, variant):
            self.core_variant = variant

        def decode(self, syndrome):
            return np.zeros(3, dtype=np.uint8)

        def decode_batch(self, syndromes):
            n = len(syndromes)
            return (np.zeros((n, 3), dtype=np.uint8), np.zeros((n, 1), dtype=np.uint8),
                    np.ones(n, dtype=np.uint32))

        def close(self):
            pass

    variants = iter(["ada64", "default", "ada64"])

    class FakeEldpc:
        def Decoder(self, H_, O_, priors_, config=None):  # noqa: N802 - the package API
            return FakeDecoder(next(variants))

    job = {"kind": "py_latency", "problem_file": str(problem_file),
           "shots_file": str(shots_file),
           "variants": [{"config_string": "decoder=bposd"}, {"config_string": "decoder=bposd"}],
           "order": [0, 1], "create_order": [0, 1], "shots_per_variant": 4,
           "warmup": 0, "gap_us": 0, "keep_warm": False}
    assert worker.run_py_latency(FakeEldpc(), job)["core_variants"] == ["ada64", "default"]
    ler_job = {"kind": "ler", "problem_file": str(problem_file), "shots_file": str(shots_file),
               "variants": [{"config_string": "decoder=bposd"}], "batch_shots": 2}
    assert worker.run_ler(FakeEldpc(), ler_job)["core_variants"] == ["ada64"]
    # a decoder object without the attribute is recorded as unknown, not dropped
    assert worker.core_variants([object()]) == ["unknown"]


def test_markdown_environment_table_names_the_core_variants():
    env = {"gpu": {}, "cpu": {}, "clocks": {}, "toolkit": {},
           "core_variants": {"c_latency": {"A": "ada64", "B": "ada64"},
                             "ler": {"A": "ada64"}}}
    results = {"generated_utc": "t", "command": "bench.py all --quick", "quick": True,
               "problem": {"name": "p", "m": 1, "n": 2, "k": 1, "recipe": {}, "shots": {}},
               "environment": env, "warnings": []}
    text = report.build_markdown(results)
    assert "| Core variant of the timed handles | c_latency A=ada64, B=ada64; ler A=ada64 |" in text
    assert report.core_variants_text(None) == "n/a"
    assert report.core_variants_text({"py_latency": {"A": ["ada64", "default"]}}) == (
        "py_latency A=ada64/default")
