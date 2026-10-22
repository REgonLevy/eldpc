# SPDX-License-Identifier: Apache-2.0
"""Configuration validation, case construction and stage settings (no decoder needed)."""
import importlib.util
import os

import pytest

from eldpctune import accuracy, grid, problems, runner
from eldpctune.grid import Point
from eldpctune.runner import (TuneConfig, auto_max_shots, build_cases, run_accuracy_stage,
                              run_fingerprint, stage_settings, zero_failure_floor)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEM = os.path.join(DATA, "surface_d3_r2.dem")

#: The built-in recipes build their circuits with qLDPC, which is optional.
needs_qldpc = pytest.mark.skipif(importlib.util.find_spec("qldpc") is None,
                                 reason="the built-in recipes need the qldpc package")


def _dem_cases():
    prob = problems.problem_from_dem_file(DEM)
    return [{"label": "dem", "p": None, "problem": prob, "seed_key": prob.key}]


def _h_file(tmp_path):
    path = tmp_path / "h.txt"
    path.write_text("1 4\n1 1 0 0\n")
    return path


def test_validation_requires_exactly_one_input(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        TuneConfig(out=str(tmp_path)).validate()
    with pytest.raises(ValueError, match="exactly one"):
        TuneConfig(problem="bb72", dem="x", out=str(tmp_path)).validate()


def test_validation_of_option_ranges(tmp_path):
    out = str(tmp_path)
    with pytest.raises(ValueError, match="--p"):
        TuneConfig(h="x", out=out).validate()
    with pytest.raises(ValueError, match="--p applies"):
        TuneConfig(dem="x", ps=[0.1], out=out).validate()
    with pytest.raises(ValueError, match="distinct"):
        TuneConfig(h="x", ps=[0.1, 0.1], out=out).validate()
    with pytest.raises(ValueError, match="ler-target"):
        TuneConfig(problem="bb72", ler_target=0.0, out=out).validate()
    with pytest.raises(ValueError, match="p99-budget"):
        TuneConfig(problem="bb72", p99_budget_us=0.0, out=out).validate()
    with pytest.raises(ValueError, match="cadence"):
        TuneConfig(problem="bb72", cadence_us=-1.0, out=out).validate()
    with pytest.raises(ValueError, match="weight"):
        TuneConfig(problem="bb72", weight=-1.0, out=out).validate()
    with pytest.raises(ValueError, match="rounds"):
        TuneConfig(problem="bb72", rounds=0, out=out).validate()
    with pytest.raises(ValueError, match="max-shots"):
        TuneConfig(problem="bb72", max_shots=0, out=out).validate()
    with pytest.raises(ValueError, match="workers"):
        TuneConfig(problem="bb72", workers=0, out=out).validate()
    with pytest.raises(ValueError, match="rounds applies"):
        TuneConfig(dem="x", rounds=3, out=out).validate()
    with pytest.raises(ValueError, match="--o supplies"):
        TuneConfig(problem="bb72", o="x", out=out).validate()
    with pytest.raises(ValueError, match="out"):
        TuneConfig(problem="bb72").validate()
    TuneConfig(problem="bb72", out=out).validate()


def test_stage_settings_defaults_and_quick(tmp_path):
    normal = stage_settings(TuneConfig(problem="bb72", out=str(tmp_path)))
    assert normal.max_shots == 131072 and normal.chunk_shots == 2048
    quick = stage_settings(TuneConfig(problem="bb72", quick=True, out=str(tmp_path)))
    assert quick.max_shots == 2048 and quick.chunk_shots == 512
    small = stage_settings(TuneConfig(problem="bb72", max_shots=100, out=str(tmp_path)))
    assert small.max_shots == 100 and small.chunk_shots == 100
    targeted = stage_settings(TuneConfig(problem="bb72", ler_target=1e-4,
                                         out=str(tmp_path)))
    assert targeted.target == 1e-4


def test_h_cases_one_per_p(tmp_path):
    h = _h_file(tmp_path)
    cfg = TuneConfig(h=str(h), ps=[0.05, 0.1], out=str(tmp_path)).validate()
    cases = build_cases(cfg)
    assert [c["label"] for c in cases] == ["p=0.05", "p=0.1"]
    assert [c["p"] for c in cases] == [0.05, 0.1]
    assert all(c["problem"].m == 1 and c["problem"].n == 4 for c in cases)
    assert all(c["problem"].k == 2 for c in cases)
    assert cases[0]["seed_key"] != cases[1]["seed_key"]
    assert cases[0]["problem"].priors[0] == pytest.approx(0.05)


@needs_qldpc
def test_builtin_problem_case_label(tmp_path):
    cfg = TuneConfig(problem="bb72", out=str(tmp_path))
    cases = build_cases(cfg)
    assert len(cases) == 1
    assert cases[0]["label"] == "p=0.001"   # the documented default physical error rate
    assert cases[0]["problem"].m == 252


def test_checkpoint_roundtrip_and_fingerprint_guard(tmp_path):
    from eldpctune.report import write_json
    from eldpctune.runner import Checkpoint

    path = tmp_path / "checkpoint.json"
    checkpoint = Checkpoint(str(path), "fingerprint-a")
    stage = {"name": "coarse", "settings": {"max_shots": 100}, "cases": [],
             "points": [{"point": {"decoder": "bposd", "ms_scaling_factor": 0.5,
                                   "max_iter": 1}, "status": "pass"}]}
    checkpoint.save_stage(stage, complete=False)
    loaded = Checkpoint.load(str(path), "fingerprint-a")
    saved = loaded.saved_stage("coarse")
    assert saved is not None and saved["complete"] is False
    assert saved["points"][0]["status"] == "pass"
    loaded.save_stage(stage, complete=True)
    assert Checkpoint.load(str(path), "fingerprint-a").saved_stage("coarse")["complete"] is True
    with pytest.raises(ValueError, match="different problem"):
        Checkpoint.load(str(path), "fingerprint-b")
    bad = tmp_path / "bad.json"
    write_json(str(bad), {"schema": "something-else"})
    with pytest.raises(ValueError, match="not an eldpc-tune checkpoint"):
        Checkpoint.load(str(bad), "fingerprint-a")


def test_confirmation_checkpoint_round_trip_and_guards(tmp_path):
    from eldpctune.report import write_json
    from eldpctune.runner import ConfirmationCheckpoint

    path = tmp_path / "confirmation-checkpoint.json"
    record = {"planned_shots": 100, "confidence": 0.99,
              "cases": [{"label": "c", "seed": 5}],
              "candidates": {"p": {"c": {"shots": 48, "failures": 1, "converged": 40}}},
              "complete": False}
    checkpoint = ConfirmationCheckpoint(str(path), "fp", 100)
    assert checkpoint.progress() is None
    checkpoint.save(record)
    loaded = ConfirmationCheckpoint.load(str(path), "fp", 100)
    assert loaded.progress() == {"planned_shots": 100, "cases": record["cases"],
                                 "candidates": record["candidates"]}
    with pytest.raises(ValueError, match="different problem"):
        ConfirmationCheckpoint.load(str(path), "other-fingerprint", 100)
    with pytest.raises(ValueError, match="shots per candidate"):
        ConfirmationCheckpoint.load(str(path), "fp", 200)
    bad = tmp_path / "bad.json"
    write_json(str(bad), {"schema": "something-else"})
    with pytest.raises(ValueError, match="not an eldpc-tune confirmation checkpoint"):
        ConfirmationCheckpoint.load(str(bad), "fp", 100)


def test_target_derived_shot_limit(tmp_path):
    out = str(tmp_path)
    base = stage_settings(TuneConfig(problem="bb72", out=out))
    assert base.max_shots == 131072
    # a target below the default's zero-failure floor grows the limit to about 4.7/target
    tight = stage_settings(TuneConfig(problem="bb72", ler_target=1e-5, out=out))
    assert tight.max_shots == 470000
    assert zero_failure_floor(131072) > 1e-5 > zero_failure_floor(470000)
    # a loose target keeps the default
    assert stage_settings(TuneConfig(problem="bb72", ler_target=1e-4, out=out)).max_shots == 131072
    # an explicit limit and quick mode are never overridden
    assert stage_settings(TuneConfig(problem="bb72", ler_target=1e-5, max_shots=200000,
                                     out=out)).max_shots == 200000
    assert stage_settings(TuneConfig(problem="bb72", ler_target=1e-5, quick=True,
                                     out=out)).max_shots == 2048
    assert auto_max_shots(TuneConfig(problem="bb72", ler_target=1e-5, out=out)) == 470000
    assert auto_max_shots(TuneConfig(problem="bb72", out=out)) is None


def test_noise_model_validation(tmp_path):
    out = str(tmp_path)
    with pytest.raises(ValueError, match="noise-model applies"):
        TuneConfig(dem="x", noise_model="uniform", out=out).validate()
    with pytest.raises(ValueError, match="uniform"):
        TuneConfig(problem="bb72", noise_model="si1000", out=out).validate()
    TuneConfig(problem="sc5", noise_model="uniform", out=out).validate()


def test_noise_model_with_unknown_problem_is_a_clear_error(tmp_path):
    cfg = TuneConfig(problem="nope", noise_model="uniform", out=str(tmp_path))
    cfg.validate()  # the family check cannot run without a known problem
    with pytest.raises(ValueError, match="unknown problem"):
        build_cases(cfg)


def test_resume_rejects_a_checkpoint_for_different_inputs(tmp_path):
    """run() refuses to resume a checkpoint whose fingerprint does not match the inputs."""
    import json as _json

    from eldpctune.runner import Checkpoint, run
    from eldpctune.report import write_json

    raw = tmp_path / "raw"
    raw.mkdir()
    write_json(str(raw / "checkpoint.json"),
               {"schema": Checkpoint.SCHEMA, "fingerprint": "not-this-run", "stages": {}})
    cfg = TuneConfig(dem=DEM, quick=True, resume=True, out=str(tmp_path))
    with pytest.raises(ValueError, match="different problem"):
        run(cfg, log=lambda *_: None)
    # without --resume the stale checkpoint is not consulted (the run would proceed and only
    # stop at the first decoder creation, which is not exercised here)
    data = _json.loads((raw / "checkpoint.json").read_text())
    assert data["fingerprint"] == "not-this-run"


def test_fingerprint_covers_the_grid_and_the_tuner_version(tmp_path, monkeypatch):
    cases = _dem_cases()
    for quick, attribute in ((True, "QUICK_ALPHAS"), (False, "COARSE_CAPS")):
        cfg = TuneConfig(dem=DEM, quick=quick, out=str(tmp_path))
        settings = stage_settings(cfg)
        base = run_fingerprint(cfg, cases, settings)
        assert base == run_fingerprint(cfg, cases, settings)
        with monkeypatch.context() as patch:   # another coarse grid
            patch.setattr(grid, attribute, getattr(grid, attribute) + (99,))
            assert run_fingerprint(cfg, cases, settings) != base
        with monkeypatch.context() as patch:   # another refinement rule
            patch.setattr(grid, "REFINE_ALPHA_STEPS", (-0.125, 0.125))
            assert run_fingerprint(cfg, cases, settings) != base
        with monkeypatch.context() as patch:   # another tuner version
            patch.setattr(runner, "__version__", "next")
            assert run_fingerprint(cfg, cases, settings) != base
        assert run_fingerprint(cfg, cases, settings) == base
    quick_cfg = TuneConfig(dem=DEM, quick=True, out=str(tmp_path))
    full_cfg = TuneConfig(dem=DEM, quick=False, out=str(tmp_path))
    assert grid.grid_digest(True) != grid.grid_digest(False)
    assert (run_fingerprint(quick_cfg, cases, stage_settings(quick_cfg))
            != run_fingerprint(full_cfg, cases, stage_settings(full_cfg)))


def _record(point, shots=4096):
    return {"point": point.as_dict(), "label": point.label(), "status": "pass",
            "per_p": [{"case": "dem", "p": None, "shots": shots, "failures": 0}]}


def _stage_dict(points, records, digest=None, sample_seconds=1.5, shots=4096):
    return {"name": "coarse", "settings": {"max_shots": shots},
            "cases": [{"label": "dem", "sample_seconds": sample_seconds,
                       "shots_sampled": shots}],
            "points": records,
            "points_digest": grid.points_digest(points) if digest is None else digest}


def test_resume_refuses_a_stage_made_with_another_point_list(tmp_path):
    from eldpctune.runner import Checkpoint

    cfg = TuneConfig(dem=DEM, quick=True, out=str(tmp_path))
    settings = accuracy.StageSettings(max_shots=4096, chunk_shots=512, target=None)
    points = [Point("bposd", 0.5, 3), Point("bposd", 0.5, 5)]
    other = [Point("bposd", 0.5, 3), Point("bposd", 0.625, 5)]
    path = tmp_path / "checkpoint.json"
    checkpoint = Checkpoint(str(path), "fp")
    checkpoint.save_stage(_stage_dict(other, [_record(other[0])]), complete=False)
    with pytest.raises(ValueError, match="different point list"):
        run_accuracy_stage("coarse", points, _dem_cases(), settings, cfg, log=lambda *_: None,
                           checkpoint=Checkpoint.load(str(path), "fp"))
    # the same list is accepted
    checkpoint.save_stage(_stage_dict(points, [_record(points[0]), _record(points[1])]),
                          complete=True)
    stage = run_accuracy_stage("coarse", points, _dem_cases(), settings, cfg,
                               log=lambda *_: None, checkpoint=Checkpoint.load(str(path), "fp"))
    assert [p["label"] for p in stage["points"]] == [p.label() for p in points]


def test_a_fully_resumed_stage_reports_the_resumed_counts(tmp_path):
    from eldpctune.runner import Checkpoint

    cfg = TuneConfig(dem=DEM, quick=True, out=str(tmp_path))
    settings = accuracy.StageSettings(max_shots=4096, chunk_shots=512, target=None)
    points = [Point("bposd", 0.5, 3), Point("bposd", 0.5, 5)]
    path = tmp_path / "checkpoint.json"
    Checkpoint(str(path), "fp").save_stage(
        _stage_dict(points, [_record(p) for p in points]), complete=True)
    stage = run_accuracy_stage("coarse", points, _dem_cases(), settings, cfg,
                               log=lambda *_: None, checkpoint=Checkpoint.load(str(path), "fp"))
    case = stage["cases"][0]
    assert stage["resumed_points"] == 2
    # the stream the recorded points decoded, not zero; this run sampled nothing
    assert case["shots_sampled"] == 4096 and case["shots_sampled_this_run"] == 0
    assert case["sample_seconds"] == 1.5 and case["sample_seconds_this_run"] == 0.0
    # and the checkpoint written by the resumed run keeps the same counts
    saved = Checkpoint.load(str(path), "fp").saved_stage("coarse")
    assert saved["cases"][0]["shots_sampled"] == 4096 and saved["complete"] is True


def test_cli_help_warns_about_derived_observables_for_a_css_half():
    from eldpctune.cli import build_parser

    help_text = " ".join(build_parser().format_help().split())
    assert "NOT a CSS logical basis" in help_text
    assert "--o FILE" in help_text and "CSS half" in help_text
