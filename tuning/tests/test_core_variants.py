# SPDX-License-Identifier: Apache-2.0
"""The core-variant comparison of the chosen point, and the per-candidate p99 warnings.

The comparison is exercised with decoder doubles whose speed depends on the ELDPC_DEVICE_CLASS
value at their create, so the stage's forcing, restoring, pairing and verdict are covered on any
machine; the run-schema tests run it end to end on the GPU at hand.
"""
import os
import time

import numpy as np
import pytest

from eldpctune import runner
from eldpctune.grid import Point

ADA_CLASS = {"name": "Ada 142-SM class", "compute_capability": "8.9", "sm_count": 142}
ONE_VARIANT = {"name": "one-variant GPU", "compute_capability": "12.0", "sm_count": 84}


class _Source:
    def sample(self, n):
        rng = np.random.default_rng(int(n))
        return rng.integers(0, 256, size=(int(n), 1), dtype=np.uint8), None


class _Problem:
    m = 8

    def shot_source(self, seed):  # noqa: ARG002 - any seed gives a valid pool
        return _Source()


class _TimedDecoder:
    """Busy-waits per call: the "default" double is ten times faster than the "ada64" one."""

    def __init__(self):
        self.core_variant = os.environ.get(runner.VARIANT_VARIABLE) or "unset"
        self.wait_ns = 20_000 if self.core_variant == "default" else 200_000

    def decode_full(self, syndrome):  # noqa: ARG002
        end = time.perf_counter_ns() + self.wait_ns
        while time.perf_counter_ns() < end:
            pass
        return None, None, 1  # status word with the converged flag

    def close(self):
        pass


def _cases():
    return [{"label": "case", "p": 0.001, "seed_key": "k", "problem": _Problem()}]


def _cfg():
    return runner.TuneConfig(dem="x.dem", ler_target=0.01, p99_budget_us=1e6, quick=True,
                             auto=True, out="unused")


def test_two_variant_device_times_both_and_names_the_faster(monkeypatch):
    created = []

    def make(point, case):  # noqa: ARG001
        decoder = _TimedDecoder()
        created.append(decoder.core_variant)
        return decoder

    monkeypatch.setattr(runner, "_make_decoder", make)
    monkeypatch.delenv(runner.VARIANT_VARIABLE, raising=False)
    record = runner.run_core_variant_comparison(_cfg(), _cases(), Point("bposd", 0.5, 3), 200, 1,
                                                log=lambda *_: None, device=ADA_CLASS)
    assert record["evaluated"] is True and record["table_variant"] == "ada64"
    # one handle of each variant for the pair, a second table-variant handle for the A/A
    assert sorted(created) == ["ada64", "ada64", "default"]
    assert runner.VARIANT_VARIABLE not in os.environ  # restored: it was unset before
    assert record["ratio_p50"] > 1.0 + record["band_p50"]
    assert record["faster"] == "default"
    assert record["recommended_env"] == "ELDPC_DEVICE_CLASS=default"
    assert "config.json" in record["config_json"]
    pair = record["per_case"][0]
    assert pair["distinct_syndromes"] == 200 and isinstance(pair["syndrome_seed"], int)
    assert len(record["aa"][0]["syndrome_seeds"]) == 1
    assert pair["syndrome_seed"] not in record["aa"][0]["syndrome_seeds"]


def test_a_preset_variable_is_restored(monkeypatch):
    monkeypatch.setattr(runner, "_make_decoder", lambda point, case: _TimedDecoder())
    monkeypatch.setenv(runner.VARIANT_VARIABLE, "ada64")
    record = runner.run_core_variant_comparison(_cfg(), _cases(), Point("bposd", 0.5, 3), 100, 1,
                                                log=lambda *_: None, device=ADA_CLASS)
    assert record["environment_value"] == "ada64"
    assert os.environ[runner.VARIANT_VARIABLE] == "ada64"


def test_a_handle_of_the_wrong_variant_is_an_error(monkeypatch):
    class Stuck(_TimedDecoder):
        def __init__(self):
            super().__init__()
            self.core_variant = "default"

    monkeypatch.setattr(runner, "_make_decoder", lambda point, case: Stuck())
    with pytest.raises(RuntimeError, match="gave a handle of variant"):
        runner.run_core_variant_comparison(_cfg(), _cases(), Point("bposd", 0.5, 3), 50, 1,
                                           log=lambda *_: None, device=ADA_CLASS)


def test_one_variant_device_skips_the_stage(monkeypatch):
    def refuse(point, case):  # noqa: ARG001
        raise AssertionError("no handle may be created on a one-variant device")

    monkeypatch.setattr(runner, "_make_decoder", refuse)
    record = runner.run_core_variant_comparison(_cfg(), _cases(), Point("bposd", 0.5, 3), 100, 1,
                                                log=lambda *_: None, device=ONE_VARIANT)
    assert record["evaluated"] is False and record["table_variant"] == "default"
    assert "one core variant" in record["reason"]
    assert runner.table_core_variant(None) == "default"


def test_p99_warnings_cover_every_reported_candidate():
    unreliable = {"p99_unreliable": True, "slow_fraction": 0.01, "slow_median_us": 812.0}
    reliable = {"p99_unreliable": False, "slow_fraction": 0.3, "slow_median_us": 900.0}
    accuracy = {"chosen": "bposd:a=0.5:c=3", "candidates": [
        {"label": "bposd:a=0.5:c=3", "latency": reliable},
        {"label": "bposd:a=0.5:c=5", "latency": unreliable},
        {"label": "bposd:a=0.4375:c=3", "latency": dict(unreliable, slow_median_us=None)}]}
    rows = [{"label": "bposd:a=0.5:c=3", "p99_unreliable": True, "slow_fraction": 0.02,
             "slow_median_us": 700.0},
            {"label": "bposd:a=0.5:c=7", "p99_unreliable": False}]
    warnings = runner.p99_warnings(accuracy, rows)
    assert len(warnings) == 3
    assert warnings[0].startswith("bposd:a=0.5:c=5: the p99 of its latency confirmation")
    assert "812.0 us" in warnings[0]
    assert "not recorded" in warnings[1]
    assert warnings[2].startswith("bposd:a=0.5:c=3 (chosen): the p99 of its latency-stage")
    assert runner.p99_warnings({"candidates": []}, []) == []
