# SPDX-License-Identifier: Apache-2.0
"""Per-session latency syndromes and the slow-class records of the benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import worker  # noqa: E402
from benchlib import formats, measure, problems, stats  # noqa: E402


# -- the per-session syndrome draw ----------------------------------------
def test_session_purposes_are_distinct_for_every_session():
    purposes = {measure.session_purpose(mode, pair, session)
                for mode in ("back_to_back", "cadence", "cadence_keep_warm_pair")
                for pair in ("a_a", "a_b", "keep_warm_ab")
                for session in range(6)}
    assert len(purposes) == 3 * 3 * 6
    # the same session of the same pair always derives the same seed
    p = measure.session_purpose("cadence", "a_b", 3)
    assert problems.session_seed("problem", p) == problems.session_seed("problem", p)
    assert problems.session_seed("problem", p) != problems.session_seed(
        "problem", measure.session_purpose("cadence", "a_b", 4))


def test_syndrome_record_counts_each_session_and_pair():
    records = []
    for pair, sessions in (("a_a", 4), ("a_b", 4)):
        for i in range(sessions):
            records.append({"mode": "back_to_back", "pair": pair, "session": i,
                            "shots": 300, "seed": 100 * sessions + i,
                            "sha256": f"{pair}-{i}"})
    record = measure.syndrome_record(records)
    assert record["distinct"] == 4 * 300          # per handle, minimum over pairs
    assert len(record["per_session"]) == 8
    assert sorted({entry["pair"] for entry in record["per_session"]}) == ["a_a", "a_b"]
    assert all(entry["shots"] == 300 for entry in record["per_session"])
    # the minimum is over pairs: a pair with three sessions gives three sessions' worth
    record = measure.syndrome_record(records[:4] + [
        {"mode": "back_to_back", "pair": "keep_warm_ab", "session": i, "shots": 300,
         "seed": i, "sha256": f"kw-{i}"} for i in range(3)])
    assert record["distinct"] == 3 * 300


def test_worker_times_the_same_syndromes_on_both_handles(tmp_path):
    """One shots file per session; both variants of the session decode exactly those shots."""
    H = sp.csc_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    O = sp.csc_matrix(np.array([[1, 0, 0]], dtype=np.uint8))
    priors = np.array([0.1, 0.2, 0.3])
    dets = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 1]], dtype=np.uint8)
    obs = np.zeros((6, 1), dtype=np.uint8)
    problem_file = tmp_path / "problem.bin"
    shots_file = tmp_path / "shots.bin"
    formats.write_problem(problem_file, H, O, priors, None)
    formats.write_shots(shots_file, dets, obs)

    class RecordingDecoder:
        def __init__(self):
            self.seen = []
            self.last_status = 1

        def decode(self, syndrome):
            self.seen.append(bytes(np.ascontiguousarray(syndrome, dtype=np.uint8).tobytes()))
            return np.zeros(2, dtype=np.uint8)

        def close(self):
            pass

    class FakeEldpc:
        def __init__(self):
            self.instances = []

        def Decoder(self, H_, O_, priors_, config=None):  # noqa: N802 - the package API
            decoder = RecordingDecoder()
            self.instances.append(decoder)
            return decoder

    fake = FakeEldpc()
    job = {"kind": "py_latency", "problem_file": str(problem_file),
           "shots_file": str(shots_file),
           "variants": [{"config_string": "decoder=bposd"},
                        {"config_string": "decoder=bposd"}],
           "order": [0, 1], "create_order": [0, 1], "shots_per_variant": 6,
           "warmup": 2, "gap_us": 0, "keep_warm": False}
    result = worker.run_py_latency(fake, job)
    assert len(fake.instances) == 2
    # warm-up plus the six timed calls, in the same order on both handles
    assert fake.instances[0].seen == fake.instances[1].seen
    assert len(fake.instances[0].seen) == 8
    assert [len(v["dur_ns"]) for v in result["variants"]] == [6, 6]


# -- slow-class records ----------------------------------------------------
def _sessions(n_sessions=2, per=200, slow_positions=(0,)):
    """Two-handle sessions; side v0 has the given slow shot indices in every session."""
    sessions = []
    for _ in range(n_sessions):
        status = np.ones(per, dtype=np.uint64)      # converged
        for i in slow_positions:
            status[i] = 0                            # slow (no ST_BP_CONVERGED)
        dur = np.full(per, 1000, dtype=np.int64)     # 1 us
        dur[list(slow_positions)] = 5000             # slow calls are slower
        sessions.append({
            "v0": {"dur_ns": dur, "status": status, "checksum": "aa"},
            "v1": {"dur_ns": np.full(per, 1000, dtype=np.int64),
                   "status": np.ones(per, dtype=np.uint64), "checksum": "aa"},
            "create_order": "ab", "call_order": "ab", "meta": {}, "stderr": "", "exit": 0})
    return sessions


def test_pooled_slow_fraction_is_the_status_count_of_the_pooled_calls():
    pair = measure._pair_stats(_sessions(n_sessions=2, per=200, slow_positions=(0,)))
    shots = pair["a_stats"]["shots"]
    converged = pair["a_stats"]["converged_calls"]
    assert shots == 400 and converged == 398
    assert pair["a_stats"]["slow_fraction"] == pytest.approx((shots - converged) / shots)
    assert pair["a_stats"]["slow_calls"] == 2
    # 2 of 400 = 0.5 % is just inside the band; the median of the slow calls is reported
    assert pair["a_stats"]["p99_unreliable"] is True
    assert pair["a_stats"]["slow_p50_us"] == pytest.approx(5.0)
    assert pair["b_stats"]["slow_fraction"] == 0.0
    assert pair["b_stats"]["p99_unreliable"] is False


def test_slow_fraction_band_edges():
    durations = list(range(1000))
    for slow, expected in ((4, False), (5, True), (20, True), (21, False), (0, False)):
        statuses = [0] * slow + [1] * (1000 - slow)
        out = stats.slow_class(durations, statuses)
        assert out["slow_fraction"] == pytest.approx(slow / 1000)
        assert out["p99_unreliable"] is expected
        assert out["slow_calls"] == slow


def test_slow_class_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        stats.slow_class([1, 2], [1])


def test_library_ref_records_no_absolute_path(tmp_path):
    """A results file must not embed the installation path of the measured library."""
    lib = tmp_path / "libeldpc.so"
    lib.write_bytes(b"ELF-ish bytes")
    ref = measure.library_ref({"path": str(lib), "source": "test"})
    assert ref["name"] == "libeldpc.so"
    assert ref["source"] == "test"
    assert "path" not in ref
    assert ref["sha256"] == measure.sha256_file(lib)
    assert str(tmp_path) not in json.dumps(ref)


def test_full_default_controls_carry_at_least_8000_distinct_syndromes():
    # full defaults: 1500 calls x 6 sessions per pair, the half-size A/A controls raised to 6
    assert measure.control_sessions(6, 1500, False, True) == 6
    assert 1500 * 6 >= measure.FULL_DISTINCT_SYNDROMES
    # a quick run keeps its small control count
    assert measure.control_sessions(4, 300, True, True) == 2
    # an explicit --shots or --processes keeps the cost the user asked for
    assert measure.control_sessions(6, 1500, False, False) == 3
    # other default combinations are raised only as far as 8000 needs
    assert measure.control_sessions(6, 2000, False, True) == 4
