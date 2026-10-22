# SPDX-License-Identifier: Apache-2.0
"""The sequential stopping rule, exercised with scripted failures on a fake decoder."""
import numpy as np

from eldpctune.accuracy import StageSettings, evaluate_point
from eldpctune.stats import clopper_pearson_lower, clopper_pearson_upper


class FakeStore:
    def __init__(self, m=1, k=1, total=10_000):
        self.m = m
        self.k = k
        self.total = total
        self.shots = 0

    def ensure(self, shots):
        self.shots = max(self.shots, min(int(shots), self.total))

    def get(self, start, stop):
        n = stop - start
        return np.zeros((n, self.m), dtype=np.uint8), np.zeros((n, self.k), dtype=np.uint8)


class FakeDecoder:
    num_detectors = 1
    num_observables = 1

    def __init__(self, failing_shots):
        self.failing = set(int(i) for i in failing_shots)
        self.position = 0

    def decode_batch(self, dets):
        n = len(dets)
        predicted = np.zeros((n, 1), dtype=np.uint8)
        for i in range(n):
            if self.position + i in self.failing:
                predicted[i, 0] = 1
        self.position += n
        return None, predicted, np.zeros(n, dtype=np.uint32)


def run(failing, target, max_shots, chunk):
    settings = StageSettings(max_shots=max_shots, chunk_shots=chunk, target=target)
    decoder = FakeDecoder(failing)
    record = evaluate_point(decoder, FakeStore(), settings, "test", "case", None)
    return record


def test_zero_failures_pass_at_the_first_chunk_that_decides():
    record = run(failing=[], target=0.05, max_shots=4000, chunk=500)
    assert record["status"] == "pass"
    assert record["shots"] == 500
    assert record["failures"] == 0
    assert record["cp99_upper"] < 0.05
    # exactly one chunk was decoded
    assert len(record["chunks"]) == 1
    assert record["cp99_upper"] == clopper_pearson_upper(0, 500)


def test_all_failures_reject_at_the_first_chunk_that_decides():
    record = run(failing=range(5000), target=0.05, max_shots=4000, chunk=500)
    assert record["status"] == "fail"
    assert record["shots"] == 500
    assert record["cp99_lower"] > 0.05


def test_undecided_at_the_shot_limit_is_inconclusive():
    failing = range(0, 1000, 22)  # 46 failures at 1000 shots, straddling 5 %
    settings = StageSettings(max_shots=1000, chunk_shots=100, target=0.05)
    record = evaluate_point(FakeDecoder(failing), FakeStore(), settings, "test", "case", None)
    assert record["status"] == "inconclusive"
    assert record["shots"] == 1000
    assert record["failures"] == 46
    assert record["cp99_lower"] < 0.05 < record["cp99_upper"]
    # every intermediate chunk was undecided
    for chunk in record["chunks"][:-1]:
        assert chunk["cp99_lower"] <= 0.05 <= chunk["cp99_upper"]


def test_decision_is_taken_at_the_first_crossing_chunk():
    # 3 % failures: the upper bound is above 5 % at 500 shots and below it at 1000.
    failing = range(0, 10_000, 33)  # ~3 %
    record = run(failing=failing, target=0.05, max_shots=4000, chunk=500)
    assert record["status"] == "pass"
    assert record["shots"] < 4000
    assert record["cp99_upper"] < 0.05
    previous = record["chunks"][-2]
    assert previous["cp99_upper"] >= 0.05


def test_no_target_measures_to_the_limit():
    record = run(failing=range(0, 1000, 22), target=None, max_shots=1000, chunk=100)
    assert record["status"] == "measured"
    assert record["shots"] == 1000
    assert record["failures"] == 46
    # a measured record reports both one-sided bounds
    assert record["cp99_lower"] == clopper_pearson_lower(46, 1000)
    assert record["cp99_upper"] == clopper_pearson_upper(46, 1000)


def test_failure_rate_at_the_target_is_not_accepted_cheaply():
    # With exactly 5 % failures the upper bound never falls below 5 %; the point must not pass.
    record = run(failing=range(0, 4000, 20), target=0.05, max_shots=4000, chunk=500)
    assert record["status"] in ("fail", "inconclusive")
    assert not (record["cp99_upper"] < 0.05)
