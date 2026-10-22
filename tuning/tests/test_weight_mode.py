# SPDX-License-Identifier: Apache-2.0
"""Runs without --ler-target (the --weight recipe): one regression test per defect of tuner 2.0.

A re-validation of tuner 2.0 on an RTX 6000 Ada with a two-socket EPYC 7542 host found four
defects of ``--weight 0``:
(1) config.json written for a choice whose own latency confirmation was over the budget; (2) BP
alone (LER 0.9935) chosen on bb288 because there was no LER floor; (3) the reported LER was the
2048-shot confirmation number; (4) budget membership decided on 2000-call latency runs whose p99
scattered 0.6-1.9x.  Every test here rebuilds the failing situation from synthetic records, fake
stages or fake decoders; no GPU is needed except in the last two tests (a quick run end to end).
"""
import json
import os

import numpy as np
import pytest

from eldpctune import cli, grid, latency, report, runner, stats
from eldpctune.grid import STOCK_POINT, Point
from eldpctune.runner import TuneConfig

SHOTS = 131072
BUDGET = 110.0


def _entry(failures, shots=SHOTS, stage="coarse", case="dem", confidence=stats.CONFIDENCE):
    return {"stage": stage, "case": case, "p": None, "status": "measured", "shots": shots,
            "failures": failures, "ler": failures / shots,
            "cp99_lower": stats.clopper_pearson_lower(failures, shots, confidence),
            "cp99_upper": stats.clopper_pearson_upper(failures, shots, confidence),
            "converged_share": 0.5}


def _record(point, failures, shots=SHOTS, stage="coarse", decode_us=50.0):
    entry = _entry(failures, shots, stage)
    return {"point": point.as_dict(), "label": point.label(), "status": "measured",
            "per_p": [entry], "shots": shots, "failures": failures,
            "cp99_upper_max": entry["cp99_upper"], "cp99_lower_min": entry["cp99_lower"],
            "decode_us_per_shot": decode_us, "converged_share": 0.5}


def _summary(p50, p99, calls=2000):
    return {"p50_us": float(p50), "p99_us": float(p99), "mean_us": float(p50), "calls": calls,
            "slow_fraction": 0.0, "p99_unreliable": False, "slow_calls": 0,
            "slow_median_us": None}


def _environment():
    return {"timestamp_utc": "2026-01-01T00:00:00+00:00",
            "commands": [{"command": "nvidia-smi --query-gpu=name --format=csv", "exit_code": 0,
                          "stdout": "GPU", "stderr": ""}],
            "eldpc": {"version": "1", "build_info": "", "isa": "avx2", "devices": []},
            "python": {"version": "3.12"},
            "machine": {"system": "Linux", "release": "", "machine": "x86_64",
                        "cpu_model": "CPU", "cpu_count": 4,
                        "gpu": {"cuda_visible_devices": None, "devices": []}}}


# -- the LER floor (defect 2) ------------------------------------------------------------------
def test_stock_point_is_the_ms30_preset():
    import eldpc

    assert STOCK_POINT.config() == eldpc.presets()[grid.STOCK_LABEL]
    assert STOCK_POINT not in grid.coarse_points() and STOCK_POINT not in grid.coarse_points(True)


def _floor_stage(name, ms30_failures, records):
    return {"name": name, "points": list(records.values()),
            "references": [_record(STOCK_POINT, ms30_failures, stage=name)]}


def test_defect2_bp_alone_is_below_the_stock_floor_and_never_chosen():
    """bb288: only BP-alone points screened inside the budget, and BP alone was chosen."""
    bp = [Point("bp", 0.5, 15), Point("bp", 0.3125, 3)]
    osd = [Point("bposd", 0.5, 11), Point("bposd", 0.5, 15)]
    results = {bp[0]: _record(bp[0], 130218), bp[1]: _record(bp[1], 130300),
               osd[0]: _record(osd[0], 80), osd[1]: _record(osd[1], 75)}
    cfg = TuneConfig(dem="x.dem", out="/unused", weight=0.0, p99_budget_us=250.0)
    floor = runner.ler_floor_record(cfg, [_floor_stage("coarse", 82, results)])
    runner.apply_ler_floor(results, floor, log=lambda *_: None)
    assert floor["source"].startswith("stock ms30")
    assert floor["choosable"] == 2 and set(floor["excluded"]) == {p.label() for p in bp}
    assert all(results[p]["ler_floor"]["passed"] is False for p in bp)
    assert runner._survivors(results, target_present=False) == osd
    # BP alone within the budget, BP+OSD over it (as on the Zen 2 host): no choice, not BP alone
    stage = {"_measured_points": {p: {"cand": _summary(122.0, 208.0), "verdict": "faster"}
                                  for p in bp}, "aa": {"band_p50": 0.0, "band_p99": 0.0}}
    stage["_measured_points"].update(
        {p: {"cand": _summary(115.0, 300.0), "verdict": "slower"} for p in osd})
    choice = runner.choose(cfg, results, stage, [0.5, 0.625])
    assert choice["config"] is None
    assert choice["reason"] == "no point that passes the LER floor is within the latency budget"
    assert {near["label"] for near in choice["nearest"]} == {p.label() for p in osd}


def test_the_floor_is_paired_with_the_reference_of_the_same_stage():
    coarse = Point("bposd", 0.5, 7)
    refine = Point("bposd", 0.46875, 7)
    results = {coarse: _record(coarse, 90), refine: _record(refine, 95, stage="refine")}
    stages = [_floor_stage("coarse", 80, {coarse: results[coarse]}),
              _floor_stage("refine", 100, {refine: results[refine]})]
    floor = runner.ler_floor_record(TuneConfig(dem="x", out="/u", weight=0.0), stages)
    runner.apply_ler_floor(results, floor, log=lambda *_: None)
    assert results[coarse]["ler_floor"]["passed"] is False   # 90 > 80 on the coarse shots
    assert results[refine]["ler_floor"]["passed"] is True    # 95 <= 100 on the refine shots
    assert results[refine]["ler_floor"]["checks"][0]["floor_failures"] == 100
    # equal failures on the same shots pass ("no worse than"); no reference on a stage fails
    tie = Point("bposd", 0.5, 31)
    assert runner.floor_verdict(_record(tie, 80), floor)["passed"] is True
    orphan = _record(tie, 1, stage="other")
    assert runner.floor_verdict(orphan, floor)["passed"] is False


def test_max_ler_replaces_the_floor_and_is_recorded():
    bp = Point("bp", 0.5, 15)
    osd = Point("bposd", 0.5, 11)
    results = {bp: _record(bp, 130218), osd: _record(osd, 80)}
    cfg = TuneConfig(dem="x.dem", out="/unused", weight=0.0, max_ler=1.0)
    assert runner.floor_reference_points(cfg) == []
    floor = runner.ler_floor_record(cfg, [])
    runner.apply_ler_floor(results, floor, log=lambda *_: None)
    assert floor["source"] == "--max-ler" and floor["max_ler"] == 1.0
    assert floor["choosable"] == 2
    strict = runner.ler_floor_record(TuneConfig(dem="x", out="/u", max_ler=1e-3), [])
    assert runner.floor_verdict(results[bp], strict)["passed"] is False
    assert runner.floor_verdict(results[osd], strict)["passed"] is True
    # the target path has no floor, and --max-ler is refused with --ler-target
    assert runner.ler_floor_record(TuneConfig(dem="x", out="/u", ler_target=1e-3), []) is None
    assert runner.floor_reference_points(TuneConfig(dem="x", out="/u", ler_target=1e-3)) == []
    with pytest.raises(ValueError, match="--max-ler"):
        TuneConfig(dem="x", out="/u", ler_target=1e-3, max_ler=1e-3).validate()
    with pytest.raises(ValueError, match="--max-ler"):
        TuneConfig(dem="x", out="/u", max_ler=0.0).validate()


class ParityDecoder:
    """Predicts the parity of the syndrome: about half the shots fail, and which ones (and how
    many) depends on the shots decoded."""

    num_observables = 1

    def __init__(self, counter):
        self.counter = counter

    def decode_batch(self, dets):
        self.counter.append(len(dets))
        parity = (np.asarray(dets, dtype=np.int64).sum(axis=1) % 2).astype(np.uint8)
        return None, parity[:, None], np.zeros(len(dets), dtype=np.uint32)

    def close(self):
        pass


def test_the_reference_decodes_the_stage_shots_and_is_not_a_candidate(tmp_path, monkeypatch):
    from eldpctune import accuracy, problems

    dem = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "surface_d3_r2.dem")
    prob = problems.problem_from_dem_file(dem)
    cases = [{"label": "dem", "p": None, "problem": prob, "seed_key": prob.key}]
    decoded = []
    monkeypatch.setattr(runner, "_make_decoder", lambda point, case: ParityDecoder(decoded))
    cfg = TuneConfig(dem=dem, out=str(tmp_path), weight=0.0)
    settings = accuracy.StageSettings(max_shots=1024, chunk_shots=256, target=None)
    points = [Point("bposd", 0.5, 3), Point("bposd", 0.5, 5)]
    path = str(tmp_path / "checkpoint.json")
    checkpoint = runner.Checkpoint(path, "fp")
    stage = runner.run_accuracy_stage("coarse", points, cases, settings, cfg, lambda *_: None,
                                      checkpoint=checkpoint, reference_points=[STOCK_POINT])
    assert [record["label"] for record in stage["points"]] == [p.label() for p in points]
    reference = stage["references"][0]
    assert reference["label"] == STOCK_POINT.label() and reference["role"] == "ler_floor_reference"
    # the same decoder on the same shots gives the same failures: the reference is paired
    assert reference["failures"] == stage["points"][0]["failures"] > 0
    assert reference["per_p"][0]["stage"] == "coarse"
    # a resumed stage reuses the saved reference and decodes nothing
    before = len(decoded)
    resumed = runner.run_accuracy_stage("coarse", points, cases, settings, cfg, lambda *_: None,
                                        checkpoint=runner.Checkpoint.load(path, "fp"),
                                        reference_points=[STOCK_POINT])
    assert len(decoded) == before
    assert resumed["references"][0]["failures"] == reference["failures"]


# -- budget membership (defect 4) ----------------------------------------------------------------
def _row(point, p50, p99, cfg):
    summary = _summary(p50, p99)
    return {"point": point, "record": _record(point, 50), "summary": summary,
            "budget": runner._budget_status(summary, cfg), "p50": p50, "p99": p99,
            "ler_upper": 1e-3, "band_p50": 0.0, "band_p99": 0.0}


def _retimed(cfg, p50, p99, calls=8000):
    cand = _summary(p50, p99, calls)
    return {"cand": cand, "ref": _summary(55.0, 140.0, calls), "calls_per_side": calls,
            "distinct_syndromes": calls, "budget": runner._budget_status(cand, cfg)}


def test_defect4_a_lucky_screen_p99_inside_the_margin_is_decided_by_the_re_timing():
    """bb90: (0.375, 3) screened inside 110 us on a lucky 2000-call draw (true p99 ~140)."""
    cfg = TuneConfig(dem="x", out="/u", weight=0.0, p99_budget_us=BUDGET)
    lucky = Point("bposd", 0.375, 3)       # screen p99 80 (x0.73), re-timed 139.9: out
    unlucky = Point("bposd", 0.5, 3)       # screen p99 120 (x1.09), re-timed 104: in
    clear_in = Point("bposd", 0.25, 1)     # screen p99 50 (x0.45): the screen decides
    clear_out = Point("bposd", 0.5, 31)    # screen p99 257 (x2.34): the screen decides
    rows = [_row(lucky, 57.0, 80.0, cfg), _row(unlucky, 57.0, 120.0, cfg),
            _row(clear_in, 30.0, 50.0, cfg), _row(clear_out, 90.0, 257.0, cfg)]
    plan = runner.plan_budget_retime(rows)
    assert {row["point"] for row, _r in plan["retime"]} == {lucky, unlucky}
    assert {row["point"] for row, _r in plan["decided"]} == {clear_in, clear_out}
    assert plan["capped"] == []
    # nearest to the budget first: |log 1.09| < |log 0.73|
    assert [row["point"] for row, _r in plan["retime"]] == [unlucky, lucky]
    retimed = {lucky: _retimed(cfg, 57.0, 139.9), unlucky: _retimed(cfg, 57.0, 104.0)}
    membership, record = runner.decide_budget_membership(cfg, rows, plan, retimed)
    assert membership[lucky]["budget"]["within"] is False
    assert membership[lucky]["basis"] == "re-timed"
    assert membership[lucky]["screen_budget"]["within"] is True      # the screen said "in"
    assert membership[unlucky]["budget"]["within"] is True
    assert membership[clear_in]["budget"]["within"] is True
    assert membership[clear_out]["budget"]["within"] is False
    assert record["margin"] == [0.5, 2.0] and len(record["retimed"]) == 2
    # the choice uses the decision: the lucky point is no longer eligible
    results = {row["point"]: row["record"] for row in rows}
    stage = {"_measured_points": {row["point"]: {"cand": row["summary"], "verdict": "faster"}
                                  for row in rows},
             "aa": {"band_p50": 0.0, "band_p99": 0.0}, "_membership": membership}
    eligible = [r["point"] for r in runner.latency_rows(cfg, results, stage)
                if r["budget"]["within"]]
    assert set(eligible) == {unlucky, clear_in}


def test_re_timing_is_capped_nearest_to_the_budget_first():
    cfg = TuneConfig(dem="x", out="/u", weight=0.0, p99_budget_us=100.0)
    points = [Point("bposd", 0.5, c) for c in (3, 5, 7)]
    rows = [_row(points[0], 50.0, 99.0, cfg), _row(points[1], 50.0, 180.0, cfg),
            _row(points[2], 50.0, 60.0, cfg)]
    plan = runner.plan_budget_retime(rows, max_points=2)
    assert [row["point"] for row, _r in plan["retime"]] == [points[0], points[2]]
    assert [row["point"] for row, _r in plan["capped"]] == [points[1]]
    membership, record = runner.decide_budget_membership(
        cfg, rows, plan, {p: _retimed(cfg, 50.0, 90.0) for p in (points[0], points[2])})
    assert record["capped"] == [{"label": points[1].label(), "screen_ratio": 1.8,
                                 "within": False}]
    assert "beyond the re-timing cap" in membership[points[1]]["basis"]


class SleepFreeDecoder:
    """A decoder double for the latency loop: counts its calls, returns a converged status."""

    def __init__(self, label, calls):
        self.label = label
        self.calls = calls

    def decode_full(self, syndrome):
        self.calls[self.label] = self.calls.get(self.label, 0) + 1
        return None, None, 1

    def close(self):
        pass


class RandomSource:
    def __init__(self, m, seed):
        self.m = m
        self.rng = np.random.default_rng(int(seed))

    def sample(self, shots):
        bits = (self.rng.random((int(shots), self.m)) < 0.5).astype(np.uint8)
        return np.packbits(bits, axis=1, bitorder="little"), None


class RandomProblem:
    m, k = 8, 1

    def shot_source(self, seed):
        return RandomSource(self.m, seed)


def test_the_re_timing_uses_the_confirmation_design(monkeypatch):
    """>= 8000 calls per side, fresh syndromes per point, paired with the A/A reference."""
    calls = {}
    monkeypatch.setattr(runner, "_make_decoder",
                        lambda point, case: SleepFreeDecoder(point.label(), calls))
    cases = [{"label": "dem", "p": None, "problem": RandomProblem(), "seed_key": "k"}]
    cfg = TuneConfig(dem="x", out="/u", weight=0.0, p99_budget_us=BUDGET)
    reference = Point("bposd", 0.3125, 3)
    points = [Point("bposd", 0.5, 3), Point("bposd", 0.5, 5)]
    out = runner.run_budget_retime(cfg, cases, points, reference, log=lambda *_: None)
    assert set(out) == set(points)
    for point in points:
        entry = out[point]
        assert entry["calls_per_side"] >= runner.CONFIRMATION_MIN_SYNDROMES == 8000
        assert entry["distinct_syndromes"] >= 8000
        assert entry["cand"]["calls"] >= 8000 and entry["ref"]["calls"] >= 8000
        assert entry["ref_label"] == reference.label()
        assert entry["per_case"][0]["schedule"] == "ref,cand,cand,ref"
        assert entry["budget"]["within"] is True       # the double is instant
        assert calls[point.label()] >= 8000
    # the reference handle is timed in every pairing
    assert calls[reference.label()] >= 2 * 8000
    seeds = {out[p]["syndrome_seeds"]["dem"] for p in points}
    other = {stats.derive_seed(cfg.seed, stage, "k") for stage in
             ("latency-confirm", "coarse", "refine", "confirm")}
    assert len(seeds) == 2 and not seeds & other


# -- the whole weight-mode path on fake stages (defects 1, 2, 3, 4) ------------------------------
class FakeCaseProblem:
    m, n, k = 5, 10, 1
    key = "fake"

    def dem_sha256(self):
        return "0" * 64

    def summary(self):
        return {"m": self.m, "n": self.n, "k": self.k}


class Scenario:
    """Screen failures, latency-stage summaries, re-timing and latency confirmation per point."""

    def __init__(self, points, ms30_failures, screen_latency, retime, confirm_latency,
                 confirm_failures=None):
        self.points = points            # {point: screen failures}
        self.ms30_failures = ms30_failures
        self.screen_latency = screen_latency
        self.retime = retime
        self.confirm_latency = confirm_latency
        self.confirm_failures = confirm_failures or {}
        self.latency_measured = []
        self.retimed = []
        self.confirmed = []


def _install(monkeypatch, scenario):
    case = {"label": "dem", "p": None, "problem": FakeCaseProblem(), "seed_key": "fake"}
    monkeypatch.setattr(runner.env, "capture_environment", _environment)
    monkeypatch.setattr(runner, "build_cases", lambda cfg: [case])
    monkeypatch.setattr(runner.grid, "coarse_points", lambda quick=False: list(scenario.points))
    monkeypatch.setattr(runner.grid, "refine_points", lambda quick, summary, already: [])

    def accuracy_stage(name, points, cases, settings, cfg, log, prior_results=None, workers=1,
                       checkpoint=None, pruning=None, reference_points=None):
        stage = {"name": name, "settings": {"max_shots": settings.max_shots}, "cases": [],
                 "points": [_record(p, scenario.points[p], stage=name) for p in points]}
        if settings.target is not None:   # a target run: every scripted point passes
            for record in stage["points"]:
                record["status"] = "pass"
        if reference_points:
            stage["references"] = [_record(p, scenario.ms30_failures, stage=name)
                                   for p in reference_points]
        return stage

    def latency_stage(cfg, cases, results, log):
        survivors = runner._survivors(results, cfg.ler_target is not None)
        scenario.latency_measured = list(survivors)
        if not survivors:
            return {"experiments": [], "aa": None, "survivors": [], "note": "none"}
        measured = {p: {"cand": _summary(*scenario.screen_latency[p]), "verdict": "faster"}
                    for p in survivors}
        return {"experiments": [], "aa": {"band_p50": 0.0, "band_p99": 0.0},
                "_measured_points": measured, "_reference_point": survivors[0],
                "calls_top": 8000, "aa_cycles": 2}

    def retime(cfg, cases, points, reference, log):
        scenario.retimed = list(points)
        return {p: dict(_retimed(cfg, *scenario.retime[p]), label=p.label()) for p in points}

    def accuracy_confirmation(make_decoder, candidates, cases, case_seeds, n_shots, confidence,
                              log=print, progress=None, save_progress=None):
        out = []
        for rank, (point, screen) in enumerate(candidates, 1):
            failures = scenario.confirm_failures.get(point, 1)
            upper = stats.clopper_pearson_upper(failures, n_shots, confidence)
            lower = stats.clopper_pearson_lower(failures, n_shots, confidence)
            out.append({"label": point.label(), "point": point.as_dict(), "rank": rank,
                        "screen": screen,
                        "per_p": [{"label": "dem", "p": None, "shots": n_shots,
                                   "failures": failures, "ler": failures / n_shots,
                                   "lower": lower, "upper": upper}],
                        "shots": n_shots, "failures": failures, "ler": failures / n_shots,
                        "upper": upper, "resumed_case_shots": 0})
        return out

    def latency_confirmation(cfg, cases, point, log):
        scenario.confirmed.append(point)
        summary = _summary(*scenario.confirm_latency[point], calls=8000)
        return {"summary": summary, "distinct_syndromes": 8000, "syndrome_seeds": {"dem": 1},
                "core_variant": "default", "budget": runner._budget_status(summary, cfg)}

    monkeypatch.setattr(runner, "run_accuracy_stage", accuracy_stage)
    monkeypatch.setattr(runner, "run_latency_stage", latency_stage)
    monkeypatch.setattr(runner, "run_budget_retime", retime)
    monkeypatch.setattr(runner.confirmation, "run_accuracy_confirmation", accuracy_confirmation)
    monkeypatch.setattr(runner, "run_latency_confirmation", latency_confirmation)
    monkeypatch.setattr(runner, "run_core_variant_comparison",
                        lambda *args, **kwargs: {"evaluated": False, "reason": "test double"})


def _cli(tmp_path, *extra):
    out = tmp_path / "out"
    code = cli.main(["--dem", "fake.dem", "--weight", "0", "--p99-budget-us", str(BUDGET),
                     "--auto", "--out", str(out), *extra])
    results = json.loads((out / "results.json").read_text())
    return code, out, results


A = Point("bposd", 0.375, 3)     # the screen's lowest-LER point inside the budget
B = Point("bposd", 0.5, 5)
C = Point("bposd", 0.5, 15)      # lowest LER, but over the budget
BP = Point("bp", 0.5, 15)        # fast and useless


def test_defect1_an_over_budget_confirmation_is_never_emitted(tmp_path, monkeypatch, capsys):
    """bb90: the only in-budget candidate's own latency confirmation read 139.9 us > 110."""
    scenario = Scenario(points={A: 113, C: 66, BP: 130000}, ms30_failures=120,
                        screen_latency={A: (57.0, 100.0), C: (88.0, 157.0)},
                        retime={A: (57.0, 105.0), C: (88.0, 150.0)},
                        confirm_latency={A: (57.0, 139.9)})
    _install(monkeypatch, scenario)
    code, out, results = _cli(tmp_path)
    capsys.readouterr()
    choice = results["choice"]
    assert choice["config"] is None and choice["point"] is None
    assert choice["reason"].startswith("no confirmed candidate is within the latency budget")
    assert code == runner.NO_QUALIFIER_EXIT != 0
    assert not (out / "config.json").exists()
    assert (out / "config.nearest.json").exists()
    # the nearest list keeps the confirmation's reading of A and is in results.json and pareto.md
    nearest = {near["label"]: near for near in choice["nearest"]}
    assert nearest[A.label()]["p99_us"] == 139.9
    assert nearest[A.label()]["basis"] == "latency confirmation"
    candidates = results["accuracy_confirmation"]["candidates"]
    assert [c["accepted"] for c in candidates] == [False]
    assert results["accuracy_confirmation"]["chosen"] is None
    pareto = (out / "pareto.md").read_text()
    assert "no configuration qualifies" in pareto and A.label() in pareto


def test_defect1_the_next_confirmed_candidate_is_chosen(tmp_path, monkeypatch, capsys):
    scenario = Scenario(points={A: 100, B: 110, C: 66}, ms30_failures=120,
                        screen_latency={A: (57.0, 100.0), B: (60.0, 101.0), C: (88.0, 157.0)},
                        retime={A: (57.0, 105.0), B: (60.0, 104.0), C: (88.0, 150.0)},
                        confirm_latency={A: (57.0, 139.9), B: (60.0, 107.0)})
    _install(monkeypatch, scenario)
    code, out, results = _cli(tmp_path)
    capsys.readouterr()
    choice = results["choice"]
    assert code == 0
    assert choice["point"]["label"] == B.label()
    assert json.loads((out / "config.json").read_text()) == B.config()
    assert choice["budget"]["within"] is True and choice["p99_us"] == 107.0
    assert [item["label"] for item in choice["passed_over"]] == [A.label()]
    assert any("ranked above the choice" in w for w in results["warnings"])
    assert scenario.confirmed == [A, B]


def test_defect2_and_3_the_whole_path_with_a_floor_and_the_screen_claim(tmp_path, monkeypatch,
                                                                        capsys):
    """The floor is recorded and applied end to end; the LER claim is the screen's estimate."""
    scenario = Scenario(points={A: 66, C: 60, BP: 130218}, ms30_failures=120,
                        screen_latency={A: (67.0, 98.0), C: (88.0, 157.0)},
                        retime={A: (67.0, 100.0), C: (88.0, 150.0)},
                        confirm_latency={A: (67.7, 104.0)},
                        confirm_failures={A: 5})    # 5/2048 = 2.44e-3: 4.8x the screen
    _install(monkeypatch, scenario)
    code, out, results = _cli(tmp_path)
    capsys.readouterr()
    assert code == 0
    # (2) BP alone is below the floor: not latency-measured, not choosable, recorded
    assert BP not in scenario.latency_measured
    floor = results["ler_floor"]
    assert floor["source"] == "stock ms30 on the same screen shots"
    assert floor["references"][0]["per_case"][0]["failures"] == 120
    assert floor["excluded"] == [BP.label()]
    bp_row = next(row for row in results["pareto"] if row["label"] == BP.label())
    assert bp_row["ler_floor_passed"] is False and bp_row["p50_us"] is None
    # (3) the claim is the 131,072-shot screen estimate with its interval, not 5/2048
    choice = results["choice"]
    assert choice["point"]["label"] == A.label()
    assert choice["ler_source"] == "screen"
    assert choice["ler"] == pytest.approx(66 / SHOTS)
    assert choice["ler_shots_per_case"] == SHOTS
    assert choice["ler_interval"] == [pytest.approx(stats.clopper_pearson_lower(66, SHOTS)),
                                      pytest.approx(stats.clopper_pearson_upper(66, SHOTS))]
    check = choice["confirmation_check"]
    assert check["failures"] == 5 and check["ler"] == pytest.approx(5 / 2048)
    assert check["consistent"] is False
    assert any("not consistent with the screen" in w for w in results["warnings"])
    # (4) every point inside the margin was re-timed before the decision
    assert set(scenario.retimed) == {A, C}
    assert results["budget_retime"]["margin"] == [0.5, 2.0]
    pareto = (out / "pareto.md").read_text()
    assert "## LER floor" in pareto and "## Budget membership" in pareto
    assert "LER (screen, 131072 shots per case, worst case)" in pareto
    assert "a consistency check, not the reported number" in pareto


def test_defect2_only_bp_alone_in_budget_gives_no_qualifier(tmp_path, monkeypatch, capsys):
    """bb288: with the floor, BP alone inside the budget no longer becomes the choice."""
    scenario = Scenario(points={BP: 130218, C: 80}, ms30_failures=82,
                        screen_latency={C: (132.0, 298.0)}, retime={C: (132.0, 303.0)},
                        confirm_latency={})
    _install(monkeypatch, scenario)
    code, out, results = _cli(tmp_path, "--p99-budget-us", "250")
    capsys.readouterr()
    assert code == runner.NO_QUALIFIER_EXIT
    assert results["choice"]["config"] is None
    assert results["choice"]["reason"] == ("no point that passes the LER floor is within the "
                                           "latency budget")
    assert not (out / "config.json").exists()
    assert scenario.latency_measured == [C]
    # --max-ler 1 restores the old, floor-free behaviour explicitly (and records it)
    scenario = Scenario(points={BP: 130218, C: 80}, ms30_failures=82,
                        screen_latency={BP: (122.0, 208.0), C: (132.0, 298.0)},
                        retime={BP: (122.0, 208.5), C: (132.0, 303.0)},
                        confirm_latency={BP: (122.3, 208.5)})
    _install(monkeypatch, scenario)
    code, out, results = _cli(tmp_path, "--p99-budget-us", "250", "--max-ler", "1")
    capsys.readouterr()
    assert code == 0 and results["choice"]["point"]["label"] == BP.label()
    assert results["ler_floor"]["source"] == "--max-ler"
    assert results["inputs"]["max_ler"] == 1.0


def test_both_paths_exit_3_and_write_the_same_files_when_nothing_qualifies(tmp_path, monkeypatch,
                                                                         capsys):
    """No qualifier: exit 3, results.json + pareto.md + config.nearest.json, no config.json."""
    written = {}
    for mode, extra in (("weight", ["--weight", "0"]), ("target", ["--ler-target", "0.01"])):
        scenario = Scenario(points={A: 66, C: 60}, ms30_failures=120,
                            screen_latency={A: (67.0, 98.0), C: (88.0, 105.0)},
                            retime={A: (67.0, 100.0), C: (88.0, 104.0)},
                            confirm_latency={A: (67.0, 140.0), C: (88.0, 150.0)},
                            confirm_failures={A: 0, C: 0})
        _install(monkeypatch, scenario)
        out = tmp_path / mode
        code = cli.main(["--dem", "fake.dem", *extra, "--p99-budget-us", str(BUDGET), "--auto",
                         "--out", str(out), "--confirm-shots", "4096"])
        capsys.readouterr()
        results = json.loads((out / "results.json").read_text())
        assert code == runner.NO_QUALIFIER_EXIT == 3, mode
        assert results["choice"]["config"] is None
        assert [n["label"] for n in results["choice"]["nearest"]] == [A.label(), C.label()]
        assert results["choice"]["nearest"][0]["basis"] == "latency confirmation"
        assert json.loads((out / "config.nearest.json").read_text()) == A.config()
        written[mode] = sorted(p.name for p in out.iterdir() if p.is_file())
    assert written["weight"] == written["target"] == ["config.nearest.json", "pareto.md",
                                                       "results.json"]


def test_a_target_candidate_that_failed_accuracy_sorts_after_the_others():
    cfg = TuneConfig(dem="x", out="/u", ler_target=1e-3, p99_budget_us=BUDGET)
    rows = [_row(A, 57.0, 100.0, cfg), _row(C, 88.0, 105.0, cfg)]
    confirmed = [{"label": A.label(), "passed": False, "upper": 2e-3,
                  "latency": _summary(57.0, 100.0), "latency_budget": rows[0]["budget"]},
                 {"label": C.label(), "passed": True, "upper": 5e-4,
                  "latency": _summary(88.0, 130.0),
                  "latency_budget": runner._budget_status(_summary(88.0, 130.0), cfg)}]
    nearest = runner.nearest_after_confirmation(cfg, rows, confirmed)
    assert [n["label"] for n in nearest] == [C.label(), A.label()]
    assert nearest[1]["accuracy_passed"] is False and nearest[1]["confirmation_ler_upper"] == 2e-3


def test_the_cli_documents_the_exit_status():
    help_text = " ".join(cli.build_parser().format_help().split())
    assert "Exit status: 0 = a choice was written" in help_text
    assert "3 = no configuration qualifies" in help_text


def test_the_target_path_keeps_its_choice_rule(tmp_path, monkeypatch, capsys):
    """--ler-target: no floor, the first accepted candidate; since tuner 2.2 the points near the
    budget are re-timed first, as without a target (tests/test_confirmation_loop.py)."""
    scenario = Scenario(points={A: 66, C: 60}, ms30_failures=120,
                        screen_latency={A: (67.0, 98.0), C: (88.0, 105.0)},
                        retime={A: (67.0, 101.0), C: (88.0, 104.0)},
                        confirm_latency={A: (67.0, 120.0), C: (88.0, 104.0)},
                        confirm_failures={A: 0, C: 0})
    _install(monkeypatch, scenario)
    out = tmp_path / "out"
    code = cli.main(["--dem", "fake.dem", "--ler-target", "0.01", "--p99-budget-us",
                     str(BUDGET), "--auto", "--out", str(out), "--confirm-shots", "4096"])
    capsys.readouterr()
    results = json.loads((out / "results.json").read_text())
    assert code == 0
    assert results["ler_floor"] is None
    assert set(scenario.retimed) == {A, C} and len(results["budget_retime"]["retimed"]) == 2
    assert results["choice"]["ler_source"] == "accuracy_confirmation"
    assert results["choice"]["point"]["label"] == C.label()
    assert scenario.confirmed == [A, C]


# -- a quick run end to end on the GPU ------------------------------------------------------------
STIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "surface_d3_r2.stim")


def _gpu_or_skip():
    import eldpc
    from eldpctune import problems

    prob = problems.problem_from_circuit_file(STIM)
    try:
        eldpc.Decoder(prob.H, prob.O, prob.priors, config=Point("bposd", 0.5, 3).config()).close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GPU decoder available: {exc}")


def test_quick_weight_run_records_the_floor_and_the_screen_claim(tmp_path):
    _gpu_or_skip()
    cfg = TuneConfig(circuit=STIM, weight=0.0, p99_budget_us=1_000_000.0, quick=True, auto=True,
                     out=str(tmp_path))
    cfg.command = ["eldpc-tune", "--test"]
    results = runner.run(cfg, log=lambda *_: None)
    floor = results["ler_floor"]
    assert floor["references"] and floor["references"][0]["stage"] == "coarse"
    assert floor["choosable"] >= 1
    assert all(ref["label"] == "ms30" for ref in floor["references"])
    coarse = results["stages"][0]
    assert coarse["references"][0]["shots"] == coarse["points"][0]["shots"]
    choice = results["choice"]
    assert choice["config"] is not None and choice["ler_source"] == "screen"
    chosen = next(r for r in results["pareto"] if r["label"] == choice["point"]["label"])
    assert chosen["ler_floor_passed"] is True
    assert choice["ler"] == max(case["ler"] for case in chosen["per_case"])
    assert choice["budget"]["within"] is True
    assert results["budget_retime"]["retimed"] == []     # a budget of 1 s: nothing near it


def test_quick_weight_run_with_an_impossible_budget_has_no_choice(tmp_path):
    _gpu_or_skip()
    cfg = TuneConfig(circuit=STIM, weight=0.0, p99_budget_us=1.0, quick=True, auto=True,
                     out=str(tmp_path))
    cfg.command = ["eldpc-tune", "--test"]
    results = runner.run(cfg, log=lambda *_: None)
    assert results["choice"]["config"] is None
    written = report.write_outputs(results, str(tmp_path))
    assert "config.json" not in written["files"]
    assert not os.path.exists(os.path.join(str(tmp_path), "config.json"))
