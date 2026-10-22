# SPDX-License-Identifier: Apache-2.0
"""Same seed, same shots, same accuracy decisions."""
import os

import pytest

import eldpc

from eldpctune import accuracy, problems
from eldpctune.grid import Point
from eldpctune.runner import TuneConfig, run_accuracy_stage
from eldpctune.stats import derive_seed

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_TIMING_KEYS = {"wall_seconds", "create_seconds", "decode_seconds", "sample_seconds", "sample_seconds_this_run",
                "decode_us_per_shot", "elapsed_seconds"}


def _strip_timing(value):
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k not in _TIMING_KEYS}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


@pytest.fixture(scope="module")
def small_problem():
    prob = problems.problem_from_circuit_file(os.path.join(DATA, "surface_d3_r2.stim"))
    try:
        decoder = eldpc.Decoder(prob.H, prob.O, prob.priors, config=Point("bposd", 0.5, 3).config())
        decoder.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GPU decoder available: {exc}")
    return prob


def _run(prob, seed, stage_name="coarse", target=None, max_shots=600, chunk=200):
    cfg = TuneConfig(seed=seed, quick=True, out="/unused")
    cases = [{"label": "p=0.02", "p": 0.02, "problem": prob, "seed_key": prob.key}]
    settings = accuracy.StageSettings(max_shots=max_shots, chunk_shots=chunk, target=target)
    points = [Point("bposd", 0.5, 3), Point("bposd", 0.3125, 5)]
    return run_accuracy_stage(stage_name, points, cases, settings, cfg, log=lambda *_: None)


def test_same_seed_same_accuracy_record(small_problem):
    first = _run(small_problem, seed=1234)
    second = _run(small_problem, seed=1234)
    assert _strip_timing(first) == _strip_timing(second)
    # the records contain actual shots and failures, not an empty run
    assert all(point["shots"] > 0 for point in first["points"])


def test_stages_derive_different_seeds(small_problem):
    coarse = _run(small_problem, seed=99, stage_name="coarse")
    refine = _run(small_problem, seed=99, stage_name="refine")
    assert coarse["cases"][0]["seed"] != refine["cases"][0]["seed"]
    assert coarse["cases"][0]["seed"] == derive_seed(99, "coarse", small_problem.key)
    assert refine["cases"][0]["seed"] == derive_seed(99, "refine", small_problem.key)


def test_different_seed_changes_the_shots(small_problem):
    a = _run(small_problem, seed=1)
    b = _run(small_problem, seed=2)
    assert a["cases"][0]["seed"] != b["cases"][0]["seed"]
    # Different shot sets are very unlikely to give identical failure counts and chunk bounds
    # for both points; compare the failure counts of all chunks.
    fa = [c["failures"] for p in a["points"] for c in p["per_p"][0]["chunks"]]
    fb = [c["failures"] for p in b["points"] for c in p["per_p"][0]["chunks"]]
    assert fa != fb


def test_target_decision_is_reproducible(small_problem):
    first = _run(small_problem, seed=7, target=0.5)
    second = _run(small_problem, seed=7, target=0.5)
    assert _strip_timing(first) == _strip_timing(second)
    assert all(p["status"] in ("pass", "fail", "inconclusive") for p in first["points"])


def _point_map(stage):
    return {point["label"]: {"status": point["status"],
                             "per_p": _strip_timing(point["per_p"])}
            for point in stage["points"]}


def test_parallel_evaluation_matches_sequential(small_problem):
    """Workers only change wall time: the shot records and decisions are identical."""
    cfg = TuneConfig(seed=5, quick=True, out="/unused")
    cases = [{"label": "p=0.02", "p": 0.02, "problem": small_problem,
              "seed_key": small_problem.key}]
    settings = accuracy.StageSettings(max_shots=600, chunk_shots=200, target=None)
    points = [Point("bposd", 0.5, 3), Point("bposd", 0.3125, 5)]
    serial = run_accuracy_stage("coarse", points, cases, settings, cfg,
                                log=lambda *_: None, workers=1)
    parallel = run_accuracy_stage("coarse", points, cases, settings, cfg,
                                  log=lambda *_: None, workers=2)
    assert _point_map(serial) == _point_map(parallel)
    assert all(point.get("group_size") == 2 for point in parallel["points"])
    assert all("group_size" not in point for point in serial["points"])


def test_resume_reuses_completed_points(tmp_path, small_problem):
    """A checkpoint with one recorded point resumes it without changing the final records."""
    from eldpctune.runner import Checkpoint, run_fingerprint

    cfg = TuneConfig(seed=3, quick=True, out=str(tmp_path))
    cases = [{"label": "p=0.02", "p": 0.02, "problem": small_problem,
              "seed_key": small_problem.key}]
    settings = accuracy.StageSettings(max_shots=600, chunk_shots=200, target=None)
    points = [Point("bposd", 0.5, 3), Point("bposd", 0.3125, 5)]
    full = run_accuracy_stage("coarse", points, cases, settings, cfg, log=lambda *_: None)

    path = tmp_path / "raw" / "checkpoint.json"
    path.parent.mkdir(parents=True)
    fingerprint = run_fingerprint(cfg, cases, settings)
    partial = Checkpoint(str(path), fingerprint)
    partial.save_stage(dict(full, points=[full["points"][0]]), complete=False)
    resumed_checkpoint = Checkpoint.load(str(path), fingerprint)

    logs = []
    resumed = run_accuracy_stage("coarse", points, cases, settings, cfg, log=logs.append,
                                 checkpoint=resumed_checkpoint)
    assert any("resuming: 1 of 2 points" in line for line in logs)
    assert _point_map(resumed) == _point_map(full)
    # the stage record is consistent with the points it holds, resumed ones included
    case, full_case = resumed["cases"][0], full["cases"][0]
    assert case["shots_sampled"] >= full_case["shots_sampled"] > 0
    assert resumed["resumed_points"] == 1
    saved = resumed_checkpoint.saved_stage("coarse")
    assert saved is not None and saved["complete"] is True
    assert len(saved["points"]) == 2


def test_no_prune_evaluates_every_point(small_problem):
    """With pruning disabled no record is skipped, whatever the converged shares are."""
    cfg = TuneConfig(seed=13, quick=True, out="/unused")
    cases = [{"label": "p=0.02", "p": 0.02, "problem": small_problem,
              "seed_key": small_problem.key}]
    settings = accuracy.StageSettings(max_shots=400, chunk_shots=200, target=0.5)
    points = [Point("bposd", 0.5, 1), Point("bposd", 0.5, 3), Point("bposd", 0.5, 15)]
    stage = run_accuracy_stage("coarse", points, cases, settings, cfg, log=lambda *_: None,
                               pruning={"enabled": False, "min_converged": 0.99,
                                        "confidence": 0.99})
    assert [p["status"] for p in stage["points"]] != ["skipped"] * 3
    assert all(p["status"] != "skipped" for p in stage["points"])
    assert all(p["converged_share"] is not None for p in stage["points"])


def test_parallel_stage_keeps_dominance_pruning(small_problem):
    """Groups never mix two caps of one (decoder, alpha) row, so the skipped set matches the sequential run."""
    cfg = TuneConfig(seed=11, quick=True, out="/unused")
    cases = [{"label": "p=0.02", "p": 0.02, "problem": small_problem,
              "seed_key": small_problem.key}]
    settings = accuracy.StageSettings(max_shots=400, chunk_shots=200, target=0.5)
    points = [Point("bposd", 0.5, 1), Point("bposd", 0.5, 3), Point("bposd", 0.3125, 1)]

    def statuses(workers):
        stage = run_accuracy_stage("coarse", points, cases, settings, cfg,
                                   log=lambda *_: None, workers=workers)
        return {point["label"]: point["status"] for point in stage["points"]}

    serial = statuses(1)
    parallel = statuses(2)
    assert serial == parallel
    if serial["bposd:a=0.5:c=1"] == "pass":
        assert parallel["bposd:a=0.5:c=3"] == "skipped"
