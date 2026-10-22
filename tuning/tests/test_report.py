# SPDX-License-Identifier: Apache-2.0
"""The output files: contents, the config file, and the removal of stale outputs."""
import json

from eldpctune import report


def _environment():
    return {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "commands": [
            {"command": "nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv",
             "exit_code": 0, "stdout": "name, driver_version, compute_cap\nGPU, 1.0, 12.0", "stderr": ""},
            {"command": "nvcc --version", "exit_code": 0, "stdout": "release 13.3", "stderr": ""},
        ],
        "eldpc": {"version": "1", "build_info": "", "isa": "avx2", "devices": []},
        "python": {"version": "3.12"},
        "machine": {"system": "Linux", "release": "", "machine": "x86_64",
                    "cpu_model": "CPU", "cpu_count": 4,
                    "gpu": {"cuda_visible_devices": None, "devices": []}},
    }


def _pareto_row():
    return {
        "label": "bposd:a=0.5:c=1", "decoder": "bposd", "alpha": 0.5,
        "ms_scaling_factor": 0.5, "max_iter": 1, "status": "pass",
        "per_case": [{"case": "p=0.001", "p": 0.001, "shots": 1000, "failures": 0,
                      "ler": 0.0, "cp99_lower": 0.0, "cp99_upper": 0.0045}],
        "ler_upper_max": 0.0045, "decode_us_per_shot": 10.0,
        "p50_us": 20.0, "p99_us": 30.0,
        "budget": {"within": True, "checks": {"p99": {"budget_us": 100.0, "measured_us": 30.0,
                                                      "within": True}}},
        "pareto": True,
    }


def _results(with_choice=True):
    config = {"decoder": "bposd", "bp_method": "ms", "schedule": "parallel",
              "osd_method": "osd0", "osd_order": 0, "ms_scaling_factor": 0.5, "max_iter": 1}
    if with_choice:
        choice = {"target_present": True, "eligible": 1, "measured": 1,
                  "ranking": "lowest p50", "config": config,
                  "point": {"decoder": "bposd", "ms_scaling_factor": 0.5, "max_iter": 1,
                            "config": config},
                  "reason": "fastest measured point whose LER upper bound is below the target",
                  "ler_upper": 0.0045, "p50_us": 20.0, "p99_us": 30.0,
                  "budget": {"within": True, "checks": {}},
                  "fastest": "bposd:a=0.5:c=1",
                  "tie_rule": "within a tie: lower LER upper bound, then fewer BP iterations",
                  "tie_set": [{"label": "bposd:a=0.5:c=1", "ler_upper": 0.0045, "p50_us": 20.0,
                               "p99_us": 30.0, "max_iter": 1, "ms_scaling_factor": 0.5,
                               "chosen": True}]}
    else:
        choice = {"target_present": True, "eligible": 0, "measured": 1,
                  "config": None, "reason": "no point that meets the accuracy target is within "
                                            "the latency budget",
                  "nearest": [{"point": {"decoder": "bposd", "ms_scaling_factor": 0.5,
                                         "max_iter": 1, "config": config},
                               "label": "bposd:a=0.5:c=1", "status": "pass",
                               "ler_upper": 0.0045, "p50_us": 20.0, "p99_us": 30.0,
                               "budget": {"within": False, "checks": {}},
                               "budget_distance": 1.5}]}
    return {
        "schema": "eldpc.tune.v1",
        "tool": {"name": "eldpc-tune", "version": "1", "seed": 1, "quick": True, "auto": True,
                 "command": ["eldpc-tune", "--problem", "x"], "started_utc": "", "duration_seconds": 0.0},
        "environment": _environment(),
        "inputs": {"problem": "x", "dem": None, "circuit": None, "h": None, "o": None,
                   "p": [0.001], "rounds": None, "p50_budget_us": None, "p99_budget_us": 100.0,
                   "ler_target": 1e-4, "weight": None, "cadence_us": 0.0, "confidence": 0.99,
                   "max_shots": 1000, "chunk_shots": 100, "target_reachable_with_max_shots": True},
        "problem_cases": [{"label": "p=0.001"}],
        "stages": [],
        "latency": {},
        "confirmation": None,
        "keep_warm": None,
        "pareto": [_pareto_row()],
        "choice": choice,
        "elapsed_seconds": 0.0,
    }


def test_write_outputs_with_a_choice(tmp_path):
    out = tmp_path / "tune"
    written = report.write_outputs(_results(True), str(out))
    assert set(written["files"]) >= {"results.json", "config.json", "pareto.md"}
    config = json.loads((out / "config.json").read_text())
    assert config["decoder"] == "bposd" and config["max_iter"] == 1
    text = (out / "pareto.md").read_text()
    assert "CHOSEN" in text
    assert "bposd:a=0.5:c=1" in text
    results = json.loads((out / "results.json").read_text())
    assert results["choice"]["config"] == config
    assert (out / "raw" / "environment.json").is_file()
    assert (out / "raw" / "latency.json").is_file()


def test_report_prints_the_tie_set_the_devices_and_the_keep_warm_note(tmp_path):
    results = _results(True)
    results["keep_warm"] = {
        "evaluated": True, "recommended": False,
        "off": {"p50_us": 20.0, "p99_us": 30.0}, "on": {"p50_us": 20.0, "p99_us": 31.0},
        "p99_note": "the decision rests on p50: the p99 A/A band is wide; this is not a p99 guard"}
    text = report.render_pareto_md(results)
    assert "tie set (1; fastest `bposd:a=0.5:c=1`)" in text
    assert "<- chosen" in text
    assert "CUDA_VISIBLE_DEVICES: (unset)" in text
    assert "not a p99 guard" in text


def test_write_outputs_without_a_choice_removes_a_stale_config(tmp_path):
    out = tmp_path / "tune"
    report.write_outputs(_results(True), str(out))
    stale = out / "raw" / "stage-coarse.json"
    stale.write_text("{}")
    written = report.write_outputs(_results(False), str(out))
    assert "config.json" not in written["files"]
    assert not (out / "config.json").exists()
    assert (out / "config.nearest.json").is_file()
    assert json.loads((out / "config.nearest.json").read_text())["decoder"] == "bposd"
    assert not stale.exists()
    text = (out / "pareto.md").read_text()
    assert "no point that meets the accuracy target is within the latency budget" in text
    assert "1.50x the budget" in text


def test_write_outputs_removes_a_stale_confirmation_checkpoint(tmp_path):
    out = tmp_path / "tune"
    raw = out / "raw"
    raw.mkdir(parents=True)
    stale = raw / "confirmation-checkpoint.json"
    stale.write_text("{}")
    report.write_outputs(_results(True), str(out))
    assert not stale.exists()


def test_results_json_is_serializable_with_numpy_values(tmp_path):
    import numpy as np

    results = _results(True)
    results["numpy_value"] = np.float64(1.5)
    results["numpy_array"] = np.asarray([1, 2, 3])
    report.write_json(str(tmp_path / "r.json"), results)
    loaded = json.loads((tmp_path / "r.json").read_text())
    assert loaded["numpy_value"] == 1.5
    assert loaded["numpy_array"] == [1, 2, 3]
