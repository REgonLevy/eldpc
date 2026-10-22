# SPDX-License-Identifier: Apache-2.0
"""Tests of the results tools (collect_results.py, make_results.py and measure_memory.py).

Run from the package root:

    python -m pytest -q docs/tools/test_results_tools.py

They need only the Python standard library and pytest: no package, no GPU, no network (the last
test compiles the README's C example when the package is installed and skips otherwise).  The
inputs are small synthetic records in the shape the package's verify, bench and tune commands
write, so the tests do not depend on any measured results file.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collect = _load("collect_results")
make = _load("make_results")
memory_probe = _load("measure_memory")

ABS = "/data/scratch/"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
UUID = "01234567-89ab-cdef-0123-456789abcdef"
BUILD_INFO = "eldpc 0.1.0; core=compiler=GNU 13.3.0; cuda=NVIDIA 13.3.73; cuda_arch=sm_89,sm_120"


def _device(**extra):
    dev = {"name": "GPU X", "uuid": UUID, "compute_capability": "9.9", "sm_count": 42,
           "l2_bytes": 1 << 20, "cuda_driver_version": "13.0", "driver_version": "13.0",
           "nvidia_driver_version": "580.1.2", "nvidia_driver_version_reason": None}
    dev.update(extra)
    return dev


def _env_verify(with_reference_lib: bool = True):
    env = {
        "gpu": {"name": "GPU X", "driver_version": "580.1.2", "compute_capability": "9.9"},
        "cpu": {"model": "CPU X", "logical_cpus": 8},
        "platform": "Linux-test",
        "python": "3.12.0",
        "nvcc": "Cuda compilation tools, release 13.3, V13.3.73",
        "eldpc": {"version": "0.1.0", "isa": "avx2", "build_info": BUILD_INFO,
                  "devices": [_device()]},
        "reference_library_build_info": "reference-x",
        "reference_lib": None,
        "command": "python verify/run_verify.py --quick --out out-verify",
    }
    if with_reference_lib:
        env["reference_lib"] = {"path": "/opt/not-real/libdbench_x.so", "name": "fastbpgpu6g",
                                "build_info": "reflib-x"}
    return env


def _verify_raw(path: Path, with_reference_lib: bool = True):
    row = {"problem": "p", "config": "c", "shots": 10, "num_detectors": 2,
           "num_mechanisms": 3, "num_observables": 1, "mismatches": 0,
           "correction_mismatches": 0, "observables_mismatches": 0,
           "batch_correction_mismatches": 0, "batch_observables_mismatches": 0,
           "single_batch_mismatches": 0, "status_mismatches_vs_reference": 0}
    if with_reference_lib:
        row.update({"status_mismatches_vs_reference_lib": 0,
                    "reference_lib_correction_mismatches": 0,
                    "reference_lib_observables_mismatches": 0})
    else:
        row["status_mismatches_vs_reference_lib"] = None
    raw = {
        "status": "pass", "mode": "quick", "started_utc": "2026-01-01T00:00:00Z",
        "rows": [row],
        "totals": {"rows": 1, "problems": 1, "configs": 1, "shots": 10, "mismatches": 0},
        "timings": {"total_seconds": 1.0},
        "selection": {"base_seed": 1, "problems": ["p"], "configs": ["c"], "shots": {"p": 10}},
        "environment": _env_verify(with_reference_lib),
    }
    path.write_text(json.dumps(raw))


def _bench_raw(path: Path, command: str = "bench.py all --quick --out out-bench"):
    raw = {
        "command": command, "quick": True, "generated_utc": "2026-01-01T00:00:00Z",
        "elapsed_s": 1.0, "settings": {"processes": 2, "c_shots": 10, "ler_shots": 10,
                                       "throughput_batch": 4},
        "problem": {"name": "p", "source": "built-in", "m": 2, "n": 3, "k": 1,
                    "nnz_h": 4, "nnz_o": 1, "recipe": {"kind": "built-in"}, "dem_sha256": "ab",
                    "shots": {"latency": {"shots": 10, "seed": 1, "sha256": "cd"},
                              "ler": {"shots": 10, "seed": 2, "sha256": "ef"}}},
        "environment": {
            "gpu": {"name": "GPU X", "sm_count": 42, "l2_bytes": 1 << 20,
                    "memory_bytes": 1 << 30, "memory_mib": 1024, "compute_capability": "9.9"},
            "cpu": {"model": "CPU X", "physical_cores": 4, "logical_cores": 8, "pinned_cpu": 3,
                    "frequency": {"governor": "performance", "cur_freq_khz": 1000, "boost": "1",
                                  "boost_source": "/sys/not/a/real/file"}},
            "driver": "580.1.2", "cuda_runtime": "12.0.0",
            "cuda_runtime_library": "/usr/local/not-real/libcudart.so.12",
            "toolkit": {"version": "13.3", "nvcc": "V13.3.73"},
            "clocks": {"before": {"graphics_mhz": 1, "memory_mhz": 2},
                       "after": {"graphics_mhz": 1, "memory_mhz": 2}},
            "isa": "avx2", "version": "0.1.0", "build_info": BUILD_INFO,
            "devices_api": [_device()],
            "python_packages": {"stim": "1.0"},
        },
        "warnings": [], "failures": [],
        "c_latency": {"modes": {"back_to_back": {
            "gap_us": 0,
            "variants": {"A": {"label": "c", "config_string": "s",
                               "library": {"path": "/usr/local/not-real/libeldpc.so",
                                           "source": "installed"},
                               "pooled": {"shots": 10, "p50_us": 1.0, "p99_us": 2.0,
                                          "p99_9_us": 3.0, "mean_us": 1.5},
                               "status": {}, "checksum": "0"}},
            "a_a_control": {"band": {"p50_us": [0.0, 0.1], "p99_us": [0.0, 0.2]}}}}},
        "py_latency": {"modes": {"back_to_back": {"variants": {
            "A": {"label": "c", "pooled": {"shots": 10, "p50_us": 2.0, "p99_us": 3.0,
                                           "p99_9_us": 4.0, "mean_us": 2.5}}}}}},
        "throughput": {"batch_shots": 4,
                       "variants": {"A": {"label": "c", "shots": 8, "seconds": 1.0,
                                          "shots_per_s": 8.0}},
                       "c_driver": {"variants": {"A": {"label": "c", "shots": 8, "seconds": 1.0,
                                                       "shots_per_s": 9.0}}}},
        "ler": {"shots": 10, "batch_shots": 4, "variants": {
            "A": {"label": "c", "shots": 10, "errors": 1, "ler": 0.1, "ci95": [0.01, 0.3]}}},
    }
    path.write_text(json.dumps(raw))


def _collect(tmp_path: Path, monkeypatch, *, verify_command=None, with_reference_lib=True,
             extra=()):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    _verify_raw(raw / "verify.json", with_reference_lib)
    if verify_command is not None:
        v = json.loads((raw / "verify.json").read_text())
        v["environment"]["command"] = verify_command
        (raw / "verify.json").write_text(json.dumps(v))
    _bench_raw(raw / "bench.json")
    _bench_raw(raw / "ler.json", command="bench.py ler --problem p --ler-shots 10 --out out-ler")
    out = tmp_path / "out"
    monkeypatch.chdir(tmp_path)
    rc = collect.main(["--verify", str(raw / "verify.json"), "--bench", str(raw / "bench.json"),
                       "--ler", str(raw / "ler.json"), "--out", str(out), "--commit", COMMIT,
                       *extra])
    assert rc == 0
    return out


def test_collect_writes_three_records_and_scrubs_paths_and_uuids(tmp_path, monkeypatch):
    out = _collect(tmp_path, monkeypatch)
    for name, kind in (("verify-quick.json", "verify"), ("bench-quick.json", "bench"),
                       ("ler-check.json", "ler")):
        data = json.loads((out / name).read_text())
        assert data["kind"] == kind
        assert data["commit"] == COMMIT
        assert data["command"].startswith("ELDPC_REFERENCE_LIB=" if kind == "verify" else "python ")
        text = json.dumps(data)
        assert not collect.looks_like_abs_path(text)
        assert "/usr/local/not-real" not in text
        assert "/opt/not-real" not in text
        assert "boost_source" not in text
        assert UUID not in text and '"uuid"' not in text
    bench = json.loads((out / "bench-quick.json").read_text())
    assert bench["c_latency"]["modes"]["back_to_back"]["variants"]["A"]["library"] == {
        "source": "installed"}


def test_collect_with_one_input_replaces_only_that_record(tmp_path, monkeypatch):
    """--verify alone (one machine's verification run re-recorded) writes verify-quick.json with
    the variable in its command and leaves the other records of --out as they are."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _verify_raw(raw / "verify.json")
    out = tmp_path / "out"
    out.mkdir()
    kept = {"bench.json": '{"kind": "bench"}\n', "ler-check.json": '{"kind": "ler"}\n'}
    for name, text in kept.items():
        (out / name).write_text(text)
    monkeypatch.chdir(tmp_path)
    assert collect.main(["--verify", str(raw / "verify.json"), "--out", str(out), "--commit", COMMIT]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["bench.json", "ler-check.json", "verify-quick.json"]
    for name, text in kept.items():
        assert (out / name).read_text() == text
    verify = json.loads((out / "verify-quick.json").read_text())
    assert verify["commit"] == COMMIT and verify["command"].startswith("ELDPC_REFERENCE_LIB=")
    with pytest.raises(SystemExit):
        collect.main(["--out", str(out), "--commit", COMMIT])


def test_collect_records_both_driver_versions_and_the_static_runtime(tmp_path, monkeypatch):
    out = _collect(tmp_path, monkeypatch)
    verify = json.loads((out / "verify-quick.json").read_text())
    bench = json.loads((out / "bench-quick.json").read_text())
    for env in (verify["environment"], bench["environment"]):
        assert env["driver"] == "580.1.2"
        assert env["cuda_driver_version"] == "13.0"
        # The cores link the runtime statically: the record names the build toolkit's runtime,
        # not a shared libcudart the benchmark may have found on the machine.
        assert env["cuda_runtime"].startswith("NVIDIA 13.3.73, linked statically")
    assert "12.0.0" not in json.dumps(bench)
    assert verify["environment"]["eldpc"]["devices"][0]["nvidia_driver_version"] == "580.1.2"


def test_collect_keeps_the_reference_lib_columns(tmp_path, monkeypatch):
    out = _collect(tmp_path, monkeypatch)
    verify = json.loads((out / "verify-quick.json").read_text())
    row = verify["rows"][0]
    assert row["status_mismatches_vs_reference_lib"] == 0
    assert row["reference_lib_correction_mismatches"] == 0
    assert row["reference_lib_observables_mismatches"] == 0
    assert verify["reference_lib"] == {"name": "fastbpgpu6g", "build_info": "reflib-x"}


def test_collect_refuses_a_machine_path_in_a_kept_field(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    _verify_raw(raw / "verify.json")
    v = json.loads((raw / "verify.json").read_text())
    v["environment"]["command"] = "python verify/run_verify.py --out " + ABS + "x/y"
    (raw / "verify.json").write_text(json.dumps(v))
    _bench_raw(raw / "bench.json")
    _bench_raw(raw / "ler.json")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        collect.main(["--verify", str(raw / "verify.json"), "--bench", str(raw / "bench.json"),
                      "--ler", str(raw / "ler.json"), "--out", str(tmp_path / "out"),
                      "--commit", COMMIT])


@pytest.mark.parametrize("arguments", ["--out /srv/results", "--reference-lib=/srv/lib.so"])
def test_collect_refuses_an_absolute_path_in_a_command(tmp_path, monkeypatch, arguments):
    with pytest.raises(SystemExit, match="absolute path"):
        _collect(tmp_path, monkeypatch,
                 verify_command=f"python verify/run_verify.py --quick {arguments}")


def test_make_results_roundtrip_and_check(tmp_path, monkeypatch):
    out = _collect(tmp_path, monkeypatch)
    doc = tmp_path / "RESULTS.md"
    assert make.main(["--results", str(out), "--out", str(doc)]) == 0
    text = doc.read_text()
    assert "## 1. Exactness" in text
    assert "The RTX 6000 Ada class with a Sapphire Rapids Xeon" in text
    assert "To be measured" in text
    assert "screens from a shared development machine, not" in text
    assert "benchmark results" in text
    assert "GPU X (compute capability 9.9) with CPU X" in text
    assert "NVIDIA driver 580.1.2 (CUDA driver API 13.0)" in text
    assert "status vs reference lib" in text
    assert COMMIT in text
    assert "Measured on:" in text
    assert not collect.looks_like_abs_path(text)
    assert make.main(["--results", str(out), "--out", str(doc), "--check"]) == 0
    doc.write_text(text + "\n- tampered\n")
    assert make.main(["--results", str(out), "--out", str(doc), "--check"]) == 1


def test_make_results_without_a_reference_lib(tmp_path, monkeypatch):
    out = _collect(tmp_path, monkeypatch, with_reference_lib=False)
    doc = tmp_path / "RESULTS.md"
    assert make.main(["--results", str(out), "--out", str(doc)]) == 0
    text = doc.read_text()
    assert "status vs reference lib" not in text
    assert "compared with the packaged reference only" in text


def test_make_results_refuses_a_missing_kind(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "verify-quick.json").write_text(json.dumps({"kind": "verify", "commit": COMMIT}))
    rc = subprocess.run([sys.executable, str(TOOLS / "make_results.py"), "--results", str(out),
                         "--out", str(tmp_path / "RESULTS.md"), "--check"],
                        capture_output=True, text=True)
    assert rc.returncode != 0
    assert "no results file of kind" in rc.stderr


def test_make_results_refuses_a_record_without_a_commit(tmp_path, monkeypatch):
    out = _collect(tmp_path, monkeypatch)
    data = json.loads((out / "bench-quick.json").read_text())
    data["commit"] = ""
    (out / "bench-quick.json").write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        make.generate(out)


def _tune_raw(path: Path, v2: bool = False):
    raw = {
        "tool": {"name": "eldpc-tune", "version": "1.1", "seed": 7, "quick": True, "auto": True,
                 "command": ["eldpc-tune", "--dem", "docs/results/tuning-test/model.dem",
                             "--ler-target", "0.005", "--auto", "--quick",
                             "--out", "docs/results/tuning-test"],
                 "started_utc": "2026-01-01T00:00:00Z", "duration_seconds": 1.0},
        "elapsed_seconds": 1.0,
        "problem_cases": [{"label": "dem:abc", "source": "dem", "num_detectors": 2,
                           "num_mechanisms": 3, "num_observables": 1, "dem_sha256": "ab",
                           "path": "docs/results/tuning-test/model.dem"}],
        "choice": {"config": {"decoder": "bposd", "bp_method": "ms", "schedule": "parallel",
                              "ms_scaling_factor": 0.4375, "max_iter": 1, "osd_method": "osd0",
                              "osd_order": 0},
                   "ler_upper": 0.004, "p50_us": 29.7, "p99_us": 80.2, "eligible": 1,
                   "measured": 1, "fastest": "bposd:a=0.4375:c=1",
                   "tie_set": [{"label": "bposd:a=0.4375:c=1", "ler_upper": 0.004,
                                "p50_us": 29.7, "p99_us": 80.2, "max_iter": 1,
                                "ms_scaling_factor": 0.4375, "chosen": True}]},
        "environment": {"machine": {"cpu_model": "CPU X", "cpu_count": 8,
                                    "gpu": {"cuda_visible_devices": None,
                                            "devices": [_device()]}},
                        "eldpc": {"version": "0.1.0", "build_info": BUILD_INFO, "isa": "avx2",
                                  "devices": [_device()]},
                        "commands": [{"command": "$CUDACXX --version",
                                      "stdout": "Cuda compilation tools, release 13.3, V13.3.73"}]},
    }
    if v2:  # the eldpc.tune.v2 records: the reported LER and latency come from the confirmations
        raw["schema"] = "eldpc.tune.v2"
        raw["tool"]["version"] = "2.0"
        raw["choice"].update({
            "ler": 0.001, "ler_upper": 0.0031, "ler_source": "accuracy_confirmation",
            "screen": {"shots": 512, "failures": 0, "ler": 0.0, "upper": 0.009, "source": "screen"},
            "slow_fraction": 0.012, "p99_unreliable": True, "reason": "first accepted"})
        raw["accuracy_confirmation"] = {
            "planned_shots": 1944, "shots_rule": "doubling then bisection", "confidence": 0.99,
            "target": 0.005, "chosen": "bposd:a=0.4375:c=1",
            "candidates": [{"label": "bposd:a=0.4375:c=1", "shots": 1944, "failures": 2,
                            "ler": 0.001, "upper": 0.0031, "passed": True,
                            "latency_within": True, "accepted": True, "per_p": []}]}
        raw["confirmation"] = {"distinct_syndromes": 8000, "core_variant": "ada64"}
        raw["core_variants"] = {"evaluated": True, "table_variant": "ada64", "faster": "default",
                                "ratio_p50": 1.25, "ratio_p99": 1.3, "band_p50": 0.02,
                                "band_p99": 0.05,
                                "recommended_env": "ELDPC_DEVICE_CLASS=default"}
        raw["warnings"] = ["bposd:a=0.4375:c=1 (chosen): the p99 is unreliable"]
    path.write_text(json.dumps(raw, indent=1) + "\n")


def _tune_dir(tmp_path: Path, v2: bool = False) -> Path:
    tune = tmp_path / "docs" / "results" / "tuning-test"
    (tune / "raw").mkdir(parents=True)
    _tune_raw(tune / "results.json", v2=v2)
    (tune / "model.dem").write_text("error(0.01) D0\n")
    (tune / "raw" / "environment.json").write_text(
        json.dumps({"machine": {"gpu": {"devices": [_device()]}}}, indent=1) + "\n")
    (tune / "raw" / "checkpoint.json").write_text("{}\n")
    (tune / "pareto.md").write_text("# Tuning result\n")
    return tune


def test_tune_provenance_section_and_clean_directory(tmp_path, monkeypatch):
    tune = _tune_dir(tmp_path)
    out = _collect(tmp_path, monkeypatch,
                   extra=("--tune", str(tune / "results.json")))
    prov = json.loads((tune / "provenance.json").read_text())
    assert prov["kind"] == "tuning"
    assert prov["commit"] == COMMIT
    assert prov["command"].startswith("python tuning/eldpc_tune.py --dem ")
    assert prov["choice"]["config_string"].startswith("bp_method=ms;decoder=bposd;max_iter=1;")
    assert prov["environment"]["toolkit"] == "CUDA 13.3 (V13.3.73)"
    assert prov["environment"]["driver"] == "580.1.2"
    assert prov["environment"]["cuda_driver_version"] == "13.0"
    assert prov["problem"]["dem_file_sha256"]
    # The tuner's own files are cleaned in place: no UUID, no resume checkpoint.
    assert not (tune / "raw" / "checkpoint.json").exists()
    for path in (tune / "results.json", tune / "raw" / "environment.json", tune / "provenance.json"):
        text = path.read_text()
        assert UUID not in text and '"uuid"' not in text
        assert not collect.looks_like_abs_path(text)
    results = json.loads((tune / "results.json").read_text())
    assert results["choice"]["p50_us"] == 29.7
    # make_results finds tuning-*/provenance.json under the results directory.
    shutil.copytree(tune, out / "tuning-test")
    doc = tmp_path / "RESULTS.md"
    assert make.main(["--results", str(out), "--out", str(doc)]) == 0
    rendered = doc.read_text()
    assert "## 4. Tuning run" in rendered
    assert "## 5. The RTX 6000 Ada class" in rendered
    assert "bp_method=ms;decoder=bposd;max_iter=1;" in rendered
    assert "(results/tuning-test/config.json)" in rendered
    assert make.main(["--results", str(out), "--out", str(doc), "--check"]) == 0


def test_tune_v2_records_the_confirmation_numbers(tmp_path, monkeypatch):
    tune = _tune_dir(tmp_path, v2=True)
    (tune / "raw" / "confirmation-checkpoint.json").write_text("{}\n")
    out = _collect(tmp_path, monkeypatch,
                   extra=("--tune", str(tune / "results.json")))
    prov = json.loads((tune / "provenance.json").read_text())
    assert prov["tune_schema"] == "eldpc.tune.v2"
    assert prov["choice"]["ler_source"] == "accuracy_confirmation"
    assert prov["choice"]["ler"] == 0.001 and prov["choice"]["screen"]["upper"] == 0.009
    assert prov["accuracy_confirmation"]["planned_shots"] == 1944
    assert prov["accuracy_confirmation"]["candidates"][0]["accepted"] is True
    assert prov["latency_confirmation"]["distinct_syndromes"] == 8000
    assert prov["core_variants"]["recommended_env"] == "ELDPC_DEVICE_CLASS=default"
    assert prov["warnings"]
    assert not (tune / "raw" / "confirmation-checkpoint.json").exists()
    shutil.copytree(tune, out / "tuning-test")
    doc = tmp_path / "RESULTS.md"
    assert make.main(["--results", str(out), "--out", str(doc)]) == 0
    rendered = doc.read_text()
    assert "from the accuracy confirmation (1944 fresh shots per case" in rendered
    assert "the screen's upper bound 9.000e-03 is not the reported number" in rendered
    assert "8000 distinct syndromes" in rendered and "p99 unreliable" in rendered
    assert "Core variants: default is faster (ada64 / default p50 x1.250" in rendered
    assert "recommended: `ELDPC_DEVICE_CLASS=default`" in rendered
    assert make.main(["--results", str(out), "--out", str(doc), "--check"]) == 0


def test_tune_v1_record_renders_the_screen_line_unchanged(tmp_path, monkeypatch):
    """A provenance record from the v1 tuner keeps its RESULTS.md text (the committed results)."""
    record = {"choice": {"config_string": "a=1", "ler_upper": 0.004, "p50_us": 29.7,
                         "p99_us": 80.2, "eligible": 1, "measured": 1},
              "tool": {"version": "1.1"}}
    assert make.choice_lines(record) == [
        "Chosen configuration: `a=1` -- LER upper bound 4.000e-03, p50 29.70 us, p99 80.20 us "
        "(eldpc-tune 1.1, 1 eligible of 1 measured points)."]


def test_core_variant_line_names_the_table_choice_when_nothing_is_to_set():
    """When the device-class table's variant wins, no variable is recommended: the line says so
    instead of rendering the empty recommendation as a variable name."""
    record = {"choice": {"config_string": "a=1", "ler_source": "accuracy_confirmation",
                         "config": {"max_iter": 1}},
              "tool": {"version": "2.0"},
              "core_variants": {"evaluated": True, "table_variant": "ada64", "faster": "ada64",
                                "ratio_p50": 0.65, "ratio_p99": 0.82, "recommended_env": None}}
    line = make.choice_lines(record)[-1]
    assert line == ("Core variants: ada64 is faster (ada64 / default p50 x0.650, p99 x0.820); "
                    "recommended: keep the device-class table's `ada64` (no variable to set).")
    assert "no variable`" not in line


def _memory_record():
    return {
        "schema": "eldpc.results.memory.v1", "kind": "memory",
        "command": "python docs/tools/measure_memory.py --dem model.dem --out memory.json",
        "commit": COMMIT, "package_version": "0.1.0", "package_tree_clean": True,
        "environment": {"gpu": {"name": "GPU X", "compute_capability": "9.9", "sm_count": 42},
                        "cpu": "CPU X", "driver": "580.1.2", "cuda_driver_version": "13.0",
                        "eldpc": {"version": "0.1.0"}},
        "problem": {"m": 2, "n": 3, "k": 1, "nnz_h": 4, "nnz_o": 1, "dem_sha256": "ab"},
        "base": {"rss_kib": 1000, "gpu_mib": 0},
        "context": {"rss_kib_delta": 131136, "gpu_mib_delta": 264},
        "first_handle": {"rss_kib_delta": 82108, "gpu_mib_delta": 26},
        "second_handle": {"rss_kib_delta": 71524, "gpu_mib_delta": 26},
    }


def test_memory_markdown_units_and_provenance():
    data = _memory_record()
    text = memory_probe.markdown(data)
    # RSS is KiB in the record and MiB in the table; GPU memory is already MiB.
    assert "| 128.1 | 264 |" in text
    assert "| 80.2 | 26 |" in text
    assert COMMIT in text
    assert "docs/tools/measure_memory.py" in text
    assert "NVIDIA driver 580.1.2 (CUDA driver API 13.0)" in text
    assert not memory_probe.looks_like_abs_path(text)
    assert UUID not in text and '"uuid"' not in text


def test_memory_check_paths_rejects_a_machine_path():
    data = _memory_record()
    data["command"] = "python docs/tools/measure_memory.py --out " + ABS + "x/y"
    with pytest.raises(SystemExit):
        memory_probe.check_paths(data, "record")


def test_memory_refuses_absolute_paths(tmp_path):
    with pytest.raises(SystemExit):
        memory_probe.main(["--dem", str(tmp_path / "model.dem"), "--out", "m.json"])
    with pytest.raises(SystemExit):
        memory_probe.main(["--dem", "model.dem", "--out", str(tmp_path / "m.json")])


def test_memory_record_renders_in_results(tmp_path, monkeypatch):
    out = _collect(tmp_path, monkeypatch)
    (out / "memory").mkdir()
    (out / "memory" / "decoder-memory.json").write_text(json.dumps(_memory_record()))
    doc = tmp_path / "RESULTS.md"
    assert make.main(["--results", str(out), "--out", str(doc)]) == 0
    text = doc.read_text()
    assert "## 4. Decoder memory" in text
    assert "| first handle for the problem | 80.2 | 26 |" in text
    assert "## 5. The RTX 6000 Ada class" in text
    assert make.main(["--results", str(out), "--out", str(doc), "--check"]) == 0


def test_target_records_render_in_their_section(tmp_path, monkeypatch):
    out = _collect(tmp_path, monkeypatch)
    target = out / "target"
    target.mkdir()
    for name in ("verify-quick.json", "bench-quick.json", "ler-check.json"):
        (target / name).write_text((out / name).read_text())
    doc = tmp_path / "RESULTS.md"
    assert make.main(["--results", str(out), "--out", str(doc)]) == 0
    text = doc.read_text()
    assert "## 4. The RTX 6000 Ada class with a Sapphire Rapids Xeon" in text
    assert re.search(r"^### 4\.1 Verification$", text, re.M)
    assert re.search(r"^### 4\.2 Benchmark$", text, re.M)
    assert re.search(r"^### 4\.3 LER check$", text, re.M)
    assert "To be measured" not in text
    assert "Measured on:" in text
    # records from another GPU than the RTX 6000 Ada board render as a proxy for the class
    assert "## 4. The RTX 6000 Ada class with a Sapphire Rapids Xeon (proxy machine)" in text
    assert ("Measured on a proxy for that class, not on the deployment pair itself (an RTX 6000 Ada with a "
            "Sapphire Rapids host): GPU X with CPU X.") in text
    assert "Section 4 was measured on a different machine" in text
    # The target verification compared the status words with a separately built library too: the
    # result line says so, and names the library's toolkit and architectures when it records them.
    section = text.split("### 4.1 Verification")[1].split("### 4.2")[0]
    assert "0 mismatches, including 0 status-word mismatches against the separately built library." in section
    assert make.main(["--results", str(out), "--out", str(doc), "--check"]) == 0
    record = json.loads((target / "verify-quick.json").read_text())
    record["reference_lib"]["build_info"] = "lib=x; cuda=NVIDIA 13.3.73; cuda_arch=sm_89,sm_120,compute_120; y=1"
    (target / "verify-quick.json").write_text(json.dumps(record))
    assert make.main(["--results", str(out), "--out", str(doc)]) == 0
    assert ("Separately built library (`docs/VERIFY.md`, \"Status words\"): `cuda=NVIDIA 13.3.73; "
            "cuda_arch=sm_89,sm_120,compute_120`.") in doc.read_text()


def test_readme_c_example_compiles(tmp_path):
    """The C quick start must still compile against an installed package.

    Compiling is the check that does not need a GPU; running the program needs one and is
    covered by tests/test_c_api.py in the package.
    """
    eldpc = pytest.importorskip("eldpc")
    package = getattr(eldpc, "__file__", None)
    if not package:
        pytest.skip("eldpc is not installed (a namespace package has no package directory)")
    pkg = Path(package).resolve().parent
    header = pkg / "include" / "eldpc.h"
    lib = pkg / "_libs" / "libeldpc.so"
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    if not header.is_file() or not lib.is_file():
        pytest.skip("the installed package does not ship the C header and library")
    out = tmp_path / "example"
    proc = subprocess.run(
        [cc, "-std=c11", "-I", str(header.parent), str(TOOLS / "example.c"),
         "-L", str(lib.parent), "-leldpc", f"-Wl,-rpath,{lib.parent}", "-o", str(out)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()


def test_readme_c_example_matches_the_file():
    """README.md's C quick start and docs/tools/example.c are the same program."""
    readme = (TOOLS.parent.parent / "README.md").read_text(encoding="utf-8")
    m = re.search(r"## Quick start: C\n.*?```c\n(.*?)```", readme, re.S)
    assert m, "README.md has no C quick start block"
    example = (TOOLS / "example.c").read_text(encoding="utf-8")
    assert m.group(1).strip() in example


ABS_PATH_POSITIVES = [
    ABS + "x/y", "/home/u/a", "/opt/foo/bar", "--out /tmp/x/y", "out=/mnt/data/run1",
    "'/var/lib/x'", "(/usr/lib/x.so)", "path: /root/a/b", "C:\\Users\\a",
]
ABS_PATH_NEGATIVES = [
    "https://github.com/a/b", "us/shot", "p50/p99", "shots/s", "docs/results/a.json",
    "../eldpc-results/verify", "1/2000", "and/or", "a / b", "ftp://host/x/y", "",
]


@pytest.mark.parametrize("text", ABS_PATH_POSITIVES)
@pytest.mark.parametrize("module", [collect, memory_probe])
def test_abs_path_check_flags_absolute_paths(module, text):
    assert module.looks_like_abs_path(text)


@pytest.mark.parametrize("text", ABS_PATH_NEGATIVES)
@pytest.mark.parametrize("module", [collect, memory_probe])
def test_abs_path_check_ignores_urls_ratios_and_relative_paths(module, text):
    assert not module.looks_like_abs_path(text)


def test_collect_drops_uuid_keys_from_everything_it_writes(tmp_path):
    nested = {"environment": {"devices_api": [{"name": "GPU X", "uuid": UUID}], "uuid": UUID}}
    cleaned = collect.drop_identifying_keys(nested)
    assert UUID not in json.dumps(cleaned) and "uuid" not in json.dumps(cleaned)
    out = tmp_path / "record.json"
    collect.write_record({"command": "python x.py", **nested}, out)
    assert UUID not in out.read_text() and '"uuid"' not in out.read_text()


def test_verify_record_command_names_the_separately_built_library(tmp_path, monkeypatch):
    """A run with a separately built library records ELDPC_REFERENCE_LIB in its command (its path
    only when it lies inside the package root), and RESULTS.md prints the status-word total."""
    out = _collect(tmp_path, monkeypatch)
    data = json.loads((out / "verify-quick.json").read_text())
    assert data["command"].startswith("ELDPC_REFERENCE_LIB=<separately-built-library> "
                                      "python verify/run_verify.py --quick")
    assert collect.reference_lib_display(str(Path.cwd() / "build-ref" / "lib" / "x.so")) == \
        str(Path("build-ref") / "lib" / "x.so")
    doc = tmp_path / "RESULTS.md"
    assert make.main(["--results", str(out), "--out", str(doc)]) == 0
    assert "including 0 status-word mismatches against the separately built library" in doc.read_text()
    (tmp_path / "plain").mkdir()
    out2 = _collect(tmp_path / "plain", monkeypatch, with_reference_lib=False)
    assert json.loads((out2 / "verify-quick.json").read_text())["command"].startswith("python ")


def test_a_full_length_bench_record_is_bench_json_and_renders_as_target(tmp_path, monkeypatch):
    assert collect.bench_record_name({"quick": True}) == "bench-quick.json"
    assert collect.bench_record_name({"quick": False}) == "bench.json"
    out = _collect(tmp_path, monkeypatch)
    target = out / "target"
    target.mkdir()
    for src, dst in (("verify-quick.json", "verify-quick.json"), ("bench-quick.json", "bench.json"),
                     ("ler-check.json", "ler-check.json")):
        (target / dst).write_text((out / src).read_text())
    doc = tmp_path / "RESULTS.md"
    assert make.main(["--results", str(out), "--out", str(doc)]) == 0
    assert re.search(r"^### 4\.2 Benchmark$", doc.read_text(), re.M)


def test_tuning_section_names_the_tuner_version():
    assert make.tuner_text({"tool": {"version": "2.2"}}).startswith("eldpc-tune 2.2 (")
    assert make.tuner_text({}) == "version not recorded"
    lines = make.tuning_section([{"tool": {"version": "2.0"}, "choice": {}, "_dir": "tuning-x"}])
    assert any(line.startswith("**Tuner:** eldpc-tune 2.0") for line in lines)


def test_tune_provenance_takes_the_toolkit_of_a_tuner_2_2_environment(tmp_path):
    """Tuner 2.2 records environment.toolkit from build_info and runs no nvcc: the provenance keeps
    the toolkit (RESULTS.md section 4 printed "CUDA toolkit n/a" without this)."""
    _tune_raw(tmp_path / "results.json", v2=True)
    raw = json.loads((tmp_path / "results.json").read_text())
    raw["tool"]["version"] = "2.2"
    raw["environment"]["commands"] = [
        {"command": "nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv",
         "stdout": "name, driver_version, compute_cap\nGPU X, 580.1.2, 9.9"}]
    raw["environment"]["toolkit"] = {"version": "13.3", "nvcc": "V13.3.73", "first_line": None,
                                     "source": "build_info"}
    data = collect.collect_tune(raw, command="python tuning/eldpc_tune.py --dem m.dem", commit=COMMIT,
                                tree_clean=True, package_version="0.1.0")
    assert data["environment"]["toolkit"] == "CUDA 13.3 (V13.3.73)"
    lines = make.tuning_section([dict(data, _dir="tuning-test")])
    assert any("CUDA toolkit CUDA 13.3 (V13.3.73)" in line for line in lines)
    assert not any("CUDA toolkit n/a" in line for line in lines)
