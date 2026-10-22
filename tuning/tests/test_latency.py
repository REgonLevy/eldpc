# SPDX-License-Identifier: Apache-2.0
"""Latency measurement mechanics: paired syndrome sequences and call-count scaling (no GPU)."""
import numpy as np

from eldpctune.latency import LatencyMeter, aa_band, measure_pair, scale_calls


class RecordingDecoder:
    """Records the first detector of every syndrome it is asked to decode."""

    def __init__(self):
        self.seen = []

    def decode_full(self, syndrome):
        self.seen.append(int(syndrome[0]))
        return None

    def keep_warm(self):
        pass


def _meter(shots=8):
    syndromes = np.arange(shots, dtype=np.uint8).reshape(shots, 1)
    return LatencyMeter(syndromes, cadence_us=0.0)


def test_pair_uses_the_same_syndromes_on_both_sides():
    meter = _meter(8)
    ref = RecordingDecoder()
    cand = RecordingDecoder()
    pair = measure_pair(meter, ref, cand, calls=4)
    # warm-up is the first `calls` shots on both sides; the timed runs cover shots 0..calls-1
    assert ref.seen[-4:] == cand.seen[-4:] == [0, 1, 2, 3]
    assert pair["calls_per_side"] == 4
    assert pair["calls_per_run"] == [2, 2]


def test_run_offsets_select_different_shots():
    meter = _meter(8)
    decoder = RecordingDecoder()
    meter.run(decoder, calls=3, start_index=5)
    assert decoder.seen[-3:] == [5, 6, 7]


def test_aa_band_records_its_cycles():
    meter = _meter(8)
    band = aa_band(meter, RecordingDecoder(), RecordingDecoder(), calls=4, cycles=2)
    assert band["aa_cycles"] == 2
    assert len(band["aa_ratios"]) == 2
    assert band["band_p50"] >= 0.0 and band["band_p99"] >= 0.0
    assert "band_basis" in band
    single = aa_band(meter, RecordingDecoder(), RecordingDecoder(), calls=4, cycles=1)
    assert single["aa_cycles"] == 1


def test_scale_calls_dense_and_sparse():
    assert scale_calls(0.0, 2000) == 2000
    assert scale_calls(0.0, 400) == 400
    # 2 s budget shared by both sides: 1,000,000 us / cadence
    assert scale_calls(1000.0, 2000) == 1000
    assert scale_calls(10000.0, 2000) == 100
    assert scale_calls(100000.0, 2000) == 50   # floor
    assert scale_calls(10.0, 2000) == 2000     # capped by max_calls


def test_meter_rejects_bad_input():
    import pytest

    with pytest.raises(ValueError):
        LatencyMeter(np.zeros((0, 3), dtype=np.uint8), 0.0)
    with pytest.raises(ValueError):
        LatencyMeter(np.zeros((2, 3), dtype=np.uint8), -1.0)
    with pytest.raises(ValueError):
        measure_pair(_meter(8), RecordingDecoder(), RecordingDecoder(), calls=1)
