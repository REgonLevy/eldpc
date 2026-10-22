# SPDX-License-Identifier: Apache-2.0
"""Unit tests of the benchmark's own logic (no GPU, no decoder, no qLDPC circuits).

    python -m pytest -q bench/tests

The C driver, the Python worker and the built-in circuit problems are exercised end to end by
`bench.py all --quick`; these tests cover the pieces whose failures would be silent: the binary formats,
the statistics, the DEM rule, the config resolution and the report.
"""
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

from benchlib import cdriver, environment, formats, measure, problems, report, stats  # noqa: E402


# -- statistics ------------------------------------------------------------
def test_nearest_rank_is_an_observed_sample():
    a = list(range(1, 101))
    assert stats.nearest_rank(a, 0.50) == 50
    assert stats.nearest_rank(a, 0.99) == 99
    assert stats.nearest_rank(a, 0.999) == 100
    assert stats.nearest_rank([7], 0.999) == 7
    assert stats.nearest_rank(a, 1.0) == 100


def test_summarize_ns_reports_microseconds():
    s = stats.summarize_ns([1000, 2000, 3000, 4000])
    assert s["shots"] == 4
    assert s["p50_us"] == 2.0          # nearest rank: the 2nd of 4 samples, not an interpolation
    assert s["p99_us"] == 4.0
    assert s["mean_us"] == 2.5


def test_wilson_interval_known_values():
    lo, hi = stats.wilson_interval(50, 100)
    assert abs(lo - 0.4038) < 0.001 and abs(hi - 0.5962) < 0.001
    lo, hi = stats.wilson_interval(0, 100)
    assert abs(lo) < 1e-12 and 0.035 < hi < 0.038
    lo, hi = stats.wilson_interval(3, 100)
    assert abs(lo - 0.01025) < 0.001 and abs(hi - 0.08452) < 0.001


def test_band_is_the_range_of_per_session_deltas():
    rows = [{"p50_us": 1.0, "p99_us": 2.0, "p99_9_us": 3.0, "mean_us": 1.5},
            {"p50_us": -1.0, "p99_us": 4.0, "p99_9_us": 2.0, "mean_us": 2.5}]
    b = stats.band(rows)
    assert b["p50_us"] == [-1.0, 1.0]
    assert b["p99_us"] == [2.0, 4.0]
    assert stats.band_width(b["p50_us"]) == 2.0


# -- binary formats --------------------------------------------------------
def test_problem_file_round_trip(tmp_path):
    H = sp.csc_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8))
    O = sp.csc_matrix(np.array([[0, 1, 1]], dtype=np.uint8))
    pri = np.array([0.01, 0.02, 0.03])
    p = tmp_path / "p.bin"
    formats.write_problem(p, H, O, pri, "error(0.01) D0\ndetector D0\n")
    H2, O2, pri2, dem = formats.read_problem(p)
    assert (H != H2).nnz == 0 and (O != O2).nnz == 0
    assert np.array_equal(pri, pri2)
    assert dem.startswith("error(0.01)")


def test_shots_file_round_trip(tmp_path):
    dets = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.uint8)
    obs = np.array([[1], [0], [1]], dtype=np.uint8)
    p = tmp_path / "s.bin"
    formats.write_shots(p, dets, obs)
    d2, o2 = formats.read_shots(p)
    assert np.array_equal(dets, d2) and np.array_equal(obs, o2)


def test_problem_file_rejects_bad_priors(tmp_path):
    H = sp.csc_matrix(np.array([[1]], dtype=np.uint8))
    O = sp.csc_matrix((0, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        formats.write_problem(tmp_path / "x.bin", H, O, np.array([1.0]))


def test_driver_output_round_trip(tmp_path):
    """Build a driver output file by hand and read it back with the documented layout."""
    import struct

    def blob(s):
        b = s.encode()
        return struct.pack("<I", len(b)) + b

    cfg, lib = "decoder=bposd", "libeldpc.so"
    out_bytes = bytearray(formats.OUTPUT_MAGIC)
    out_bytes += struct.pack("<12I", 1, 2, 3, 4, 1, 2, 1, 100, 1, 0, 5, 0xFFFFFFFF)
    out_bytes += blob(lib) + blob(lib) + blob(cfg) + blob(cfg)
    out_bytes += struct.pack("<Q", 12345)
    for t0, t1, rc, st in [(100, 200, 0, 0), (150, 260, 0, 1 | (1 << 32))] * 2:
        out_bytes += struct.pack("<4Q", t0, t1, rc, st)
    out_bytes += struct.pack("<10Q", 4, 0, 10, 3, 10, 3, 11, 12, 13, formats.OUTPUT_MARKER)
    p = tmp_path / "o.bin"
    p.write_bytes(bytes(out_bytes))
    out = formats.read_driver_output(p)
    assert out["meta"]["records_per_variant"] == 2
    assert out["shots_per_record"] is None
    assert out["meta"]["m"] == 3 and out["meta"]["k"] == 1
    assert out["meta"]["configs"] == [cfg, cfg]
    assert out["meta"]["libraries"] == [lib, lib]
    assert list(out["variant"]) == [0, 1, 0, 1]
    assert list(out["dur_ns"]) == [100, 110, 100, 110]
    assert out["checksums"] == {"all": 11, "variant0": 12, "variant1": 13}
    assert out["keep"]["n"] == 4


def test_driver_batch_output_records_shots_per_batch(tmp_path):
    import struct

    def blob(s):
        b = s.encode()
        return struct.pack("<I", len(b)) + b

    cfg, lib = "decoder=bposd", "libeldpc.so"
    out_bytes = bytearray(formats.OUTPUT_MAGIC)
    out_bytes += struct.pack("<12I", 1, 2, 3, 4, 1, 2, 2, 0, 0, 0, 5, 0xFFFFFFFF)  # mode 2 = batch
    out_bytes += blob(lib) + blob(lib) + blob(cfg) + blob(cfg)
    out_bytes += struct.pack("<Q", 12345)
    for t0, t1, rc, low, v in [(100, 200, 0, 64, 0), (150, 260, 0, 64, 1),
                               (120, 230, 0, 64, 0), (170, 290, 0, 64, 1)]:
        out_bytes += struct.pack("<4Q", t0, t1, rc, low | (v << 32))
    out_bytes += struct.pack("<10Q", 0, 0, 0, 0, 0, 0, 21, 22, 23, formats.OUTPUT_MARKER)
    p = tmp_path / "b.bin"
    p.write_bytes(bytes(out_bytes))
    out = formats.read_driver_output(p)
    assert out["meta"]["mode"] == 2 and out["meta"]["records_per_variant"] == 2
    assert list(out["shots_per_record"]) == [64, 64, 64, 64]
    assert out["status"] is None
    assert out["checksums"] == {"all": 21, "variant0": 22, "variant1": 23}


# -- DEM rule --------------------------------------------------------------
def test_dem_rule_xors_duplicates_and_separators():
    text = ("error(0.1) D0 D0 D1 L0 L0 L1\n"
            "error(0.2) D1\n"
            "error(0.3) D0 ^ D1 L0\n"
            "detector D1\n"
            "logical_observable L1\n")
    H, O, pri = problems.dem_to_arrays(text)
    assert H.shape == (2, 3) and O.shape == (2, 3)
    assert pri.tolist() == [0.1, 0.2, 0.3]
    dens = lambda M: [[int(v) for v in M.getrow(i).toarray().ravel()] for i in range(M.shape[0])]
    assert dens(H) == [[0, 0, 1], [1, 1, 1]]      # D0 appears twice in column 0 -> cancels
    assert dens(O) == [[0, 0, 1], [1, 0, 0]]


def test_open_problem_from_dem_file(tmp_path):
    p = tmp_path / "m.dem"
    p.write_text("error(0.01) D0\nerror(0.02) D0 L0\ndetector D0\nlogical_observable L0\n")
    prob = problems.open_problem(str(p))
    assert (prob.m, prob.n, prob.k) == (1, 2, 1)
    dets, obs, seed = prob.sample(64, "latency")
    assert dets.shape == (64, 1) and obs.shape == (64, 1)
    assert prob.source == "dem file"
    assert isinstance(seed, int)


def test_corrupt_dem_file_has_a_clear_error(tmp_path):
    p = tmp_path / "bad.dem"
    p.write_text("this is not a detector error model\n")
    with pytest.raises(problems.ProblemError) as e:
        problems.open_problem(str(p))
    assert str(p) in str(e.value) and "not a valid stim detector error model" in str(e.value)


def test_unknown_problem_lists_builtins():
    with pytest.raises(problems.ProblemError) as e:
        problems.open_problem("no-such-problem")
    assert "bb144" in str(e.value)


def test_builtin_definitions_are_complete():
    for name, spec in problems.BUILTINS.items():
        assert spec["family"] in ("bb", "hgp", "surface"), name
        assert spec["basis"] in ("z", "x", "xyz"), name
        assert 0 < spec["p"] < 1
        assert isinstance(spec["dem_decompose_errors"], bool)
        if spec["family"] == "bb":
            assert len(spec["A"]) == len(spec["B"]) == 3
        if spec["family"] == "surface":
            assert spec["d"] % 2 == 1


# -- configs and report ----------------------------------------------------
class _FakeEldpc:
    def presets(self):
        return {"a03125c3": {"decoder": "bposd", "max_iter": 3, "ms_scaling_factor": 0.3125,
                             "schedule": "parallel", "bp_method": "ms", "osd_method": "osd0",
                             "osd_order": 0}}


def test_config_string_is_canonical():
    label, s = measure.resolve_config("a03125c3", _FakeEldpc())
    assert label == "a03125c3"
    assert s == ("bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;osd_method=osd0;"
                 "osd_order=0;schedule=parallel")
    assert measure.resolve_config("a=1;b=0.5", _FakeEldpc())[1] == "a=1;b=0.5"


def test_config_string_from_dict_formats_bools_and_floats():
    assert measure.config_string_from_dict({"b": True, "a": 0.5, "n": 3}) == "a=0.5;b=true;n=3"


def test_resolve_config_from_json_file(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"decoder": "bp", "max_iter": 5, "bp_method": "ms"}')
    label, s = measure.resolve_config(str(p), _FakeEldpc())
    assert label == str(p)
    assert s == "bp_method=ms;decoder=bp;max_iter=5"


def test_dem_rule_unrolls_repeat_blocks():
    H, O, pri = problems.dem_to_arrays("repeat 3 {\n  error(0.01) D0 L0\n}\ndetector D0\nlogical_observable L0\n")
    assert H.shape == (1, 3) and O.shape == (1, 3)
    assert np.all(H.toarray() == 1) and np.all(O.toarray() == 1)
    assert pri.tolist() == [0.01, 0.01, 0.01]


def test_summary_reports_a_missing_results_file(tmp_path, capsys):
    import bench
    assert bench.summarize(str(tmp_path)) == 2
    assert "cannot read" in capsys.readouterr().err


def test_unknown_config_lists_presets():
    with pytest.raises(measure.MeasurementError) as e:
        measure.resolve_config("nope", _FakeEldpc())
    assert "a03125c3" in str(e.value)


def _fake_sessions(n_sessions=2, per=4, v0_ns=1000, v1_ns=1000, v1_checksum="aa"):
    return [{"v0": {"dur_ns": np.full(per, v0_ns), "status": np.zeros(per, np.uint64),
                    "checksum": "aa"},
             "v1": {"dur_ns": np.full(per, v1_ns), "status": np.zeros(per, np.uint64),
                    "checksum": v1_checksum},
             "keep": {"n": 0, "errors": 0, "periodic_sum_ns": 0, "periodic_max_ns": 0,
                      "final_sum_ns": 0, "final_max_ns": 0},
             "meta": {}, "stderr": "", "exit": 0} for _ in range(n_sessions)]


def _variant(side="A"):
    return measure.Variant(side, "a03125c3", "decoder=bposd", {"path": "libeldpc.so"})


def test_assemble_latency_mode_single_variant():
    mode = {"name": "back_to_back", "gap_us": 0, "kw": (0, 0)}
    out = measure._assemble_latency_mode(mode, {"a_a": _fake_sessions()}, [_variant()],
                                         {"c_shots": 4, "processes": 2})
    assert out["a_a_control"]["delta"]["p50_us"] == 0.0
    assert out["variants"]["A"]["pooled"]["shots"] == 16      # both handles pooled
    assert out["comparison"] is None
    assert out["keep_warm_a"] == 0 and out["warnings"] == []


def test_blocked_delta_weights_order_blocks_equally():
    keys = ("p50_us", "p99_us", "p99_9_us", "mean_us")
    # Two sessions created A first (delta +2 us), one created B first (delta -2 us): the plain mean is
    # +0.67 us, the block-balanced mean is 0.
    rows = [{k: v for k in keys} for v in (2.0, 2.0, -2.0)]
    for r, v in zip(rows, (2.0, 2.0, -2.0)):
        for k in keys:
            r[k] = v
    sessions = [{"create_order": "ab", "call_order": "ab"}, {"create_order": "ab", "call_order": "ab"},
                {"create_order": "ba", "call_order": "ba"}]
    blocked, blocks = measure._blocked_delta(rows, sessions)
    assert blocked["p50_us"] == 0.0
    assert sorted(b["sessions"] for b in blocks) == [1, 2]
    # Equal block counts: the blocked mean equals the plain mean.
    rows = [{"p50_us": 1.0 + i, "p99_us": 0.0, "p99_9_us": 0.0, "mean_us": 0.0} for i in range(4)]
    sessions = [{"create_order": o, "call_order": o} for o in ("ab", "ba", "ab", "ba")]
    blocked, _ = measure._blocked_delta(rows, sessions)
    assert blocked["p50_us"] == 2.5
    # Sessions without recorded orders (older results) give no blocked estimate.
    assert measure._blocked_delta(rows, [{}, {}, {}, {}])[0] is None


def test_an_outlier_shot_raises_a_warning():
    mode = {"name": "back_to_back", "gap_us": 0, "kw": (0, 0)}
    sessions = _fake_sessions()
    sessions[0]["v0"]["dur_ns"][0] = 2_000_000        # 2000x the 1000 ns p50, above the 1 ms floor
    out = measure._assemble_latency_mode(mode, {"a_a": sessions}, [_variant()],
                                         {"c_shots": 4, "processes": 2})
    assert any("one shot took" in w for w in out["warnings"])
    # A sub-millisecond outlier on a microsecond decode is launch latency, not interference.
    sessions = _fake_sessions()
    sessions[0]["v0"]["dur_ns"][0] = 200_000          # 200x the p50 but only 0.2 ms
    out = measure._assemble_latency_mode(mode, {"a_a": sessions}, [_variant()],
                                         {"c_shots": 4, "processes": 2})
    assert not any("one shot took" in w for w in out["warnings"])


def test_assemble_latency_mode_flags_different_side_outputs():
    mode = {"name": "back_to_back", "gap_us": 0, "kw": (0, 0)}
    out = measure._assemble_latency_mode(mode, {"a_a": _fake_sessions(v1_checksum="bb")}, [_variant()],
                                         {"c_shots": 4, "processes": 2})
    assert out["a_a_control"]["sides_equal"] is False
    assert any("different outputs" in w for w in out["warnings"])


def test_assemble_latency_mode_comparison_has_a_a_band():
    mode = {"name": "cadence", "gap_us": 1000, "kw": (0, 0)}
    per_pair = {"a_a": _fake_sessions(), "a_b": _fake_sessions(v1_ns=3000)}
    out = measure._assemble_latency_mode(mode, per_pair, [_variant("A"), _variant("B")],
                                         {"c_shots": 4, "processes": 2})
    comp = out["comparison"]
    assert comp["delta"]["p50_us"] == -2.0
    assert comp["aa_band"]["p50_us"] == [0.0, 0.0]
    assert comp["within_band"]["p50_us"] is False
    assert out["variants"]["B"]["pooled"]["p50_us"] == 3.0


def test_assemble_keep_warm_pair_mode():
    mode = {"name": "cadence_keep_warm_pair", "gap_us": 1000, "kw": (1, 0)}
    per_pair = {"keep_warm_on": _fake_sessions(), "keep_warm_off": _fake_sessions(),
                "keep_warm_ab": _fake_sessions(v0_ns=2000, v1_ns=1000)}
    out = measure._assemble_kw_pair_mode(mode, per_pair, _variant(), {"c_shots": 4, "processes": 2})
    assert out["comparison"]["delta"]["p50_us"] == 1.0        # warm minus idle
    assert out["variants"]["A"]["treatment"] == "keep_warm"
    assert out["variants"]["B"]["treatment"] == "idle"
    assert out["a_a_controls"]["keep_warm_on"]["delta"]["p50_us"] == 0.0


def _fake_results(dem="dem1", cfg="decoder=bposd", batch=1024, m=10):
    return {
        "schema": "eldpc.bench.v1", "command": "bench.py all", "generated_utc": "t",
        "problem": {"name": "p", "m": m, "n": 10, "k": 1, "dem_sha256": dem},
        "environment": {"gpu": {}, "cpu": {}, "toolkit": {}},
        "warnings": [],
        "c_latency": {"modes": {}, "configs": {"A": {"config_string": cfg}}},
        "throughput": {"batch_shots": batch,
                       "variants": {"A": {"config_string": cfg}}, "a_a_control": {}},
    }


def test_compare_accepts_matching_results():
    from benchlib import compare
    checks = compare.check_comparable(_fake_results(), _fake_results())
    assert all(c["ok"] for c in checks)
    text = compare.compare_text(_fake_results(), _fake_results(), checks)
    assert "Comparability checks" in text and "Environment" in text


@pytest.mark.parametrize("mutation", [
    {"dem": "other-dem"}, {"cfg": "decoder=bp"}, {"batch": 4096}, {"m": 11}])
def test_compare_refuses_different_things(mutation):
    from benchlib import compare
    a = _fake_results()
    b = _fake_results(**mutation)
    with pytest.raises(compare.CompareError):
        compare.check_comparable(a, b)


def test_report_has_every_section():
    results = {
        "generated_utc": "2026-01-01T00:00:00Z", "command": "bench.py all --quick", "quick": True,
        "problem": {"name": "p", "source": "dem file", "m": 1, "n": 2, "k": 1, "recipe": {}, "shots": {}},
        "environment": {"gpu": {}, "cpu": {}, "clocks": {}, "toolkit": {}, "driver": "d",
                        "cuda_runtime": "r", "isa": "avx2", "version": "0"},
        "warnings": ["w"],
        "c_latency": {"modes": {"back_to_back": {
            "variants": {"A": {"label": "a", "pooled": {"shots": 2, "p50_us": 1.0}}},
            "a_a_control": {"delta": {}, "band": {}}}}},
        "py_latency": {"modes": {"back_to_back": {"variants": {}, "a_a_control": {"delta": {}, "band": {}}}}},
        "throughput": {"variants": {}, "a_a_control": {}},
        "ler": {"variants": {"A": {"label": "a", "shots": 4, "errors": 1, "ler": 0.25,
                                   "ci95": [0.05, 0.7]}}},
    }
    text = report.build_markdown(results)
    for heading in ("# eldpc benchmark results", "## Problem", "## Environment", "## Warnings",
                    "## C API single-shot latency", "## Python per-shot latency",
                    "## Batch throughput", "## Logical error rate"):
        assert heading in text
    assert "- w" in text


# -- A/A bands and labels --------------------------------------------------
def _tp_sessions(durations_ns, shots=100):
    """Throughput sessions from (A duration, B duration) pairs, one batch per side per session."""
    return [{"batches": [{"variant": 0, "shots": shots, "dur_ns": a}, {"variant": 1, "shots": shots, "dur_ns": b}],
             "checksums": ["x", "x"]} for a, b in durations_ns]


def test_throughput_aa_band_comes_from_the_control_sessions():
    S = {"throughput_batch": 100, "throughput_reps": 1, "processes": 2}
    control = _tp_sessions([(1_000_000, 1_010_000), (1_000_000, 990_000), (1_000_000, 1_000_000)])
    ab = _tp_sessions([(1_000_000, 2_800_000), (1_000_000, 2_790_000)])
    out = measure._assemble_throughput({"a_a": control, "a_b": ab}, [_variant("A"), _variant("B")], S)
    comp = out["comparison"]
    # The A/B ratio is about 0.358.  A band built from the A/B sessions (the old behaviour) would sit
    # around 0.358; the band of the A/A control sessions is around 1.
    assert 0.35 < comp["ratio_b_over_a"] < 0.37
    assert comp["aa_band_ratio"] == [round(1_000_000 / 1_010_000, 4), round(1_000_000 / 990_000, 4)]
    assert comp["aa_band_ratio"][0] > 0.98 and comp["aa_band_ratio"][1] < 1.02
    assert comp["within_aa_band"] is False
    assert comp["aa_band_sessions"] == 3 and comp["ab_sessions"] == 2
    assert comp["ab_session_ratio_range"][1] < 0.4
    assert out["a_a_control"]["sessions"] == 3


def test_throughput_inside_the_band_when_both_pairs_are_the_same():
    S = {"throughput_batch": 100, "throughput_reps": 1, "processes": 2}
    sessions = _tp_sessions([(1_000_000, 1_010_000), (1_000_000, 990_000)])
    out = measure._assemble_throughput({"a_a": sessions, "a_b": sessions}, [_variant("A"), _variant("B")], S)
    assert out["comparison"]["within_aa_band"] is True


def test_report_prints_the_real_control_session_counts():
    mode = {"name": "cadence_keep_warm_pair", "gap_us": 1000, "kw": (1, 0)}
    per_pair = {"keep_warm_on": _fake_sessions(2), "keep_warm_off": _fake_sessions(2),
                "keep_warm_ab": _fake_sessions(4, v0_ns=2000, v1_ns=1000)}
    out = measure._assemble_kw_pair_mode(mode, per_pair, _variant(), {"c_shots": 4, "processes": 4})
    text = report._latency_section("t", {"what_is_timed": "x", "modes": {"cadence_keep_warm_pair": out}})
    assert text.count("(2 sessions)") == 2          # the warm/warm and idle/idle controls
    assert "(4 sessions)" not in text
    assert "4 control sessions" in text             # the band pools both controls
    tp = measure._assemble_throughput(
        {"a_a": _tp_sessions([(1, 1)] * 3), "a_b": _tp_sessions([(1, 1)] * 5)},
        [_variant("A"), _variant("B")], {"throughput_batch": 100, "throughput_reps": 1, "processes": 5})
    text = report._throughput_section(tp)
    assert "3 A/A control sessions" in text and "5 sessions" in text


def test_abi_config_string_is_put_in_key_order():
    assert measure.resolve_config("b=1; a=2", _FakeEldpc())[1] == "a=2;b=1"


# -- environment, problems and the C driver build --------------------------
def test_boost_is_always_recorded(monkeypatch):
    files = {}
    monkeypatch.setattr(environment, "_read", lambda path, default=None: files.get(path, default))
    assert environment._cpu_boost("cpu0") == ("unknown", None)
    files["/sys/devices/system/cpu/intel_pstate/no_turbo"] = "1"
    assert environment._cpu_boost("cpu0") == ("0", "/sys/devices/system/cpu/intel_pstate/no_turbo")
    files["/sys/devices/system/cpu/cpufreq/boost"] = "1"
    assert environment._cpu_boost("cpu0")[0] == "1"
    files["cpu0/cpufreq/boost"] = "0"
    assert environment._cpu_boost("cpu0") == ("0", "cpu0/cpufreq/boost")
    assert environment._cpu_freq(None)["boost"] in ("0", "1", "unknown")


def test_epp_warning_is_kept():
    env = {"gpu": {"name": "g"}, "cpu": {"pinned_cpu": 0, "logical_cores": 4,
                                         "frequency": {"energy_performance_preference": "balance_power"}},
           "clocks": {"before": None}, "pid": 1}
    assert any("energy-performance preference" in w for w in environment.finish(env))


def test_default_problem_falls_back_to_stim_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "qldpc", None)
    name, note = problems.resolve_default()
    assert name == problems.FALLBACK_DEFAULT and "qLDPC" in note


def test_include_dir_discovery(tmp_path, monkeypatch):
    pkg = tmp_path / "site" / "eldpc"
    (pkg / "include").mkdir(parents=True)
    (pkg / "include" / "eldpc.h").write_text("/* header */\n")

    class FakePackage:
        __file__ = str(pkg / "__init__.py")

    monkeypatch.delenv("ELDPC_INCLUDE", raising=False)
    monkeypatch.setenv("PATH", "")                  # no pkg-config: fall back to the package directory
    assert cdriver.find_include_dirs(FakePackage) == [str((pkg / "include").resolve())]
    other = tmp_path / "other"
    other.mkdir()
    (other / "eldpc.h").write_text("/* header */\n")
    monkeypatch.setenv("ELDPC_INCLUDE", str(other))
    assert cdriver.find_include_dirs(FakePackage) == [str(other.resolve())]
    monkeypatch.delenv("ELDPC_INCLUDE")
    (pkg / "include" / "eldpc.h").unlink()
    with pytest.raises(cdriver.DriverError) as e:
        cdriver.find_include_dirs(FakePackage)
    assert "eldpc.h" in str(e.value)


# -- the CUDA toolkit record -----------------------------------------------
GPU_BUILD_INFO = ("eldpc 0.1.0; front=GNU 13.3.0; host_isa=avx512; core=compiler=GNU 13.3.0; "
                  "isa=avx512; lib=fastbpgpu6g; cuda=NVIDIA 13.3.73; cuda_arch=sm_89,sm_120,compute_120; "
                  "cuda_flags=-O3 -fmad=false")
CPU_BUILD_INFO = "eldpc 0.1.0; front=GNU 13.3.0; host_isa=avx512; core=compiler=GNU 13.3.0; lib=fastbpcpu6g"


def test_toolkit_comes_from_build_info_without_nvcc(monkeypatch):
    """The build toolkit is the library's (build_info's cuda=), even with no nvcc on PATH."""
    def no_run(*a, **k):
        raise AssertionError("nvcc must not be run when build_info names the toolkit")
    monkeypatch.setattr(environment, "_run", no_run)
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("CUDACXX", raising=False)
    tk = environment._toolkit(GPU_BUILD_INFO)
    assert tk == {"version": "13.3", "nvcc": "V13.3.73", "first_line": None, "source": "build_info"}
    assert report._toolkit_text(tk) == "13.3 (V13.3.73, from the library's build_info)"
    assert environment._toolkit(GPU_BUILD_INFO.replace("13.3.73", "13.4.92"))["nvcc"] == "V13.4.92"


def test_toolkit_falls_back_to_nvcc_when_build_info_has_none(monkeypatch, tmp_path):
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    monkeypatch.setenv("CUDACXX", str(nvcc))
    text = ("nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Cuda compilation tools, release 13.2, V13.2.86\nBuild cuda_13.2.r13.2/compiler.0\n")
    monkeypatch.setattr(environment, "_run", lambda cmd, timeout=15: (0, text, ""))
    tk = environment._toolkit(CPU_BUILD_INFO)
    assert tk["version"] == "13.2" and tk["nvcc"] == "V13.2.86" and tk["source"] == "nvcc"
    assert report._toolkit_text(tk) == "13.2 (V13.2.86, nvcc on PATH or CUDACXX)"
    monkeypatch.setenv("CUDACXX", str(tmp_path / "missing"))
    monkeypatch.setenv("PATH", "")
    assert environment._toolkit(None) is None
    assert report._toolkit_text({}) == "n/a"


# -- --device ----------------------------------------------------------------
def test_device_option_adds_the_front_key_to_every_config_string():
    s = "bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125"
    assert measure.with_device(s, "gpu") == s                      # the default: unchanged strings
    assert measure.with_device(s, "cpu") == ("bp_method=ms;decoder=bposd;device=cpu;max_iter=3;"
                                             "ms_scaling_factor=0.3125")
    assert measure.with_device(s, "auto").count("device=auto") == 1
    named = s + ";device=cpu;threads=4"
    assert measure.with_device(named, "cpu") == named and measure.with_device(named, "gpu") == named
    with pytest.raises(measure.MeasurementError, match="conflicts"):
        measure.with_device(named, "auto")


def test_a_gpu_create_error_points_to_the_device_option():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_main", BENCH / "bench.py")
    bench_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench_main)
    error = ("EldpcError: eldpc: device 0 (Some GPU) has compute capability 8.6, which the GPU cores "
             "of this build do not support: ...; to run on the CPU core, create the decoder with "
             "device=cpu")
    assert "--device cpu" in bench_main.gpu_error_hint(error, "gpu")
    assert bench_main.gpu_error_hint(error, "cpu") == ""
    assert bench_main.gpu_error_hint("some other error", "gpu") == ""
    args = bench_main.build_parser().parse_args(["all", "--quick"])
    assert args.device == "gpu"
    assert bench_main.build_parser().parse_args(["ler", "--device", "cpu"]).device == "cpu"


def test_cgroup_cpu_quota_v2_and_v1(tmp_path):
    (tmp_path / "cpu.max").write_text("5375999 100000\n")
    assert abs(environment.cpu_quota(str(tmp_path)) - 53.75999) < 1e-9
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert environment.cpu_quota(str(tmp_path)) is None
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("400000\n")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert environment.cpu_quota(str(v1)) == 4.0
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    assert environment.cpu_quota(str(v1)) is None
    assert environment.cpu_quota(str(tmp_path / "missing")) is None


def test_host_load_warning_threshold(tmp_path, monkeypatch):
    """A loaded host warns (load 52 on 112 logical CPUs, in a 54-CPU cgroup quota, did not with the
    old rule of half the logical CPUs); a quiet many-core host does not, whatever the quota of the
    container or the pinned mask of the run."""
    loadavg = tmp_path / "loadavg"
    reads = {"/proc/loadavg": loadavg}
    real_read = environment._read
    monkeypatch.setattr(environment, "_read",
                        lambda path, default=None: (reads[path].read_text() if path in reads
                                                    else real_read(path, default)))
    monkeypatch.setattr(environment, "_smi_gpus", lambda: [])
    monkeypatch.setattr(environment, "_smi_compute_apps", lambda: [])

    def run(before, after, logical, quota):
        env = {"gpu": {"name": "g"}, "cpu": {"pinned_cpu": 1, "logical_cores": logical,
                                             "affinity_cpus": logical, "cpu_quota": quota,
                                             "frequency": {}},
               "clocks": {"before": None}, "pid": 1, "loadavg": [before, 0.0, 0.0]}
        loadavg.write_text(f"{after} 1.00 1.00 1/100 12345\n")
        return [w for w in environment.finish(env) if "host load" in w], env

    warned, env = run(52.1, 52.2, 112, 53.76)
    assert len(warned) == 1 and "112 logical CPUs" in warned[0] and "6.60" in warned[0]
    # The load is read before and after the run only, and the warning says so.
    assert "(sampled before and after the run)" in warned[0]
    # A quiet host: neither the pinned mask (1 CPU) nor a 31-CPU quota shrinks the threshold.
    env = {"gpu": {"name": "g"}, "cpu": {"pinned_cpu": 1, "logical_cores": 128, "affinity_cpus": 1,
                                         "cpu_quota": 30.72, "frequency": {}},
           "clocks": {"before": None}, "pid": 1, "loadavg": [4.08, 0.0, 0.0]}
    loadavg.write_text("3.9 1.00 1.00 1/100 12345\n")
    assert [w for w in environment.finish(env) if "host load" in w] == []
    assert run(4.9, 4.5, 128, None)[0] == []                 # 128 CPUs: threshold 7.4
    assert len(run(9.7, 3.0, 128, None)[0]) == 1             # high before the run counts too
    assert run(1.5, 1.6, 16, None)[0] == []                  # 16 CPUs: threshold 1.8
    rose = run(0.2, 1.5, 128, None)[0][0]
    assert "rose during the run" in rose and "(sampled before and after the run)" in rose
    assert environment.load_threshold(112) == 1.0 + 0.05 * 112


def test_auto_cpu_avoids_cpu0_and_prefers_the_gpus_numa_node():
    node0 = set(range(0, 32)) | set(range(64, 96))
    allowed = list(range(128))
    cpu, why = environment.auto_cpu(allowed, node0, {0, 64})
    assert cpu == 1 and "NUMA node" in why
    node1 = set(range(32, 64)) | set(range(96, 128))
    assert environment.auto_cpu(allowed, node1, {0, 64})[0] == 32
    assert environment.auto_cpu(list(range(16)), None, {0, 8})[0] == 1
    assert environment.auto_cpu([0, 8, 9], {0, 8}, {0, 8})[0] == 9
    assert "none of the GPU's NUMA node" in environment.auto_cpu([0, 8, 9], {0, 8}, {0, 8})[1]
    assert environment.auto_cpu([0], None, {0})[0] == 0
    assert environment.auto_cpu([0, 64], None, {0, 64})[0] == 0


def test_cpu_list_parser():
    assert environment.parse_cpu_list("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    assert environment.parse_cpu_list("") is None
    assert environment.parse_cpu_list("3-1") is None
    assert environment.parse_cpu_list("x") is None


def test_a_config_with_a_cpus_list_is_one_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_main_cfg", BENCH / "bench.py")
    bench_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench_main)
    assert bench_main.parse_configs(["a03125c3;threads=8;cpus=8-11,72-75"]) == \
        ["a03125c3;threads=8;cpus=8-11,72-75"]
    assert bench_main.parse_configs(["a03125c3,ms30"]) == ["a03125c3", "ms30"]
    assert bench_main.parse_configs(["a03125c3", "ms30;max_iter=5"]) == ["a03125c3", "ms30;max_iter=5"]
    assert bench_main.parse_configs(None) == ["a03125c3"]
