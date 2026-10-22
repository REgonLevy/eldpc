# SPDX-License-Identifier: Apache-2.0
"""Tuner 2.2 follow-ups of the confirmed choice, in both paths, one regression test per failure.

Tuner 2.1: (1) the --ler-target path decided budget membership on the 2000-call latency-stage
p99, so its first candidates could be points whose lucky stage p99 sat just inside the budget and
whose 8000-call confirmations read over it; (2) both paths stopped after a fixed three
confirmations, so a run could say "no configuration qualifies" while in-budget screen passes were
never confirmed; (3) a nearest-list line printed the latency-stage p50 / p99 next to the re-timed
budget ratio; (4) the tie set marked the screen's pick "chosen" when the confirmation had passed
it over.  Every test rebuilds the situation from synthetic
records and fake stages (the scaffolding of test_weight_mode.py); no GPU is needed.
"""
import json

import pytest

from eldpctune import cli, report, runner
from eldpctune.grid import Point

from test_weight_mode import BUDGET, Scenario, _install

TARGET = ["--ler-target", "0.01", "--confirm-shots", "4096"]
WEIGHT = ["--weight", "0"]


def _run(tmp_path, mode_args, *extra, name="out"):
    out = tmp_path / name
    code = cli.main(["--dem", "fake.dem", *mode_args, "--p99-budget-us", str(BUDGET), "--auto",
                     "--out", str(out), *extra])
    results = json.loads((out / "results.json").read_text())
    return code, out, results


def _row(results, point):
    return next(row for row in results["pareto"] if row["label"] == point.label())


# -- Budget membership: the --ler-target path re-times near-budget points before the in/out decision -----------
LUCKY1 = Point("bposd", 0.375, 3)    # stage p99 inside the budget, re-timed far over it
LUCKY2 = Point("bposd", 0.625, 5)    # stage p99 inside, re-timed over
LUCKY3 = Point("bposd", 0.375, 5)    # stage p99 inside, re-timed over
GOOD = Point("bposd", 0.5, 11)       # stage p99 just over, re-timed inside
FAR = Point("bposd", 0.5, 31)        # beyond 2x the budget: the latency stage decides


def test_the_target_path_re_times_near_budget_points_before_the_decision(tmp_path, monkeypatch,
                                                                         capsys):
    """Here tuner 2.1 would rank the three lucky points first on their stage p99 (0.85x, 0.92x and
    0.95x the budget), confirm them (8000-call readings 1.35x-1.19x) and stop at its three with no
    choice, never reaching GOOD (stage 1.02x, re-timed 0.88x).  Now the re-timing decides before
    any candidate is ranked."""
    scenario = Scenario(points={LUCKY1: 0, LUCKY2: 0, LUCKY3: 0, GOOD: 0, FAR: 0},
                        ms30_failures=0,
                        screen_latency={LUCKY1: (52.0, 93.0), LUCKY2: (55.0, 101.0),
                                        LUCKY3: (57.0, 104.0), GOOD: (71.0, 112.0),
                                        FAR: (130.0, 240.0)},
                        retime={LUCKY1: (52.0, 150.0), LUCKY2: (55.0, 126.0),
                                LUCKY3: (57.0, 131.0), GOOD: (71.0, 97.0)},
                        confirm_latency={LUCKY1: (52.0, 149.0), LUCKY2: (55.0, 127.0),
                                         LUCKY3: (57.0, 131.0), GOOD: (71.0, 98.0)},
                        confirm_failures={LUCKY1: 0, LUCKY2: 0, LUCKY3: 0, GOOD: 0})
    _install(monkeypatch, scenario)
    code, out, results = _run(tmp_path, TARGET)
    capsys.readouterr()
    assert code == 0
    retime = results["budget_retime"]
    assert retime["margin"] == [0.5, 2.0] and retime["max_points"] == runner.RETIME_MAX_POINTS
    lucky_and_good = {LUCKY1, LUCKY2, LUCKY3, GOOD}
    assert {item["label"] for item in retime["retimed"]} == {p.label() for p in lucky_and_good}
    assert set(scenario.retimed) == lucky_and_good and FAR not in scenario.retimed
    for point, within in ((LUCKY1, False), (LUCKY2, False), (LUCKY3, False), (GOOD, True)):
        row = _row(results, point)
        assert row["budget_basis"] == "re-timed" and row["budget"]["within"] is within
    assert _row(results, FAR)["budget_basis"] == "latency stage (above 2x the budget)"
    # the lucky points are never candidates: GOOD is the only one confirmed, and it is chosen
    assert scenario.confirmed == [GOOD]
    assert results["accuracy_confirmation"]["in_budget_order"] == [GOOD.label()]
    assert results["choice"]["point"]["label"] == GOOD.label()
    assert results["choice"]["ler_source"] == "accuracy_confirmation"
    assert "## Budget membership" in (out / "pareto.md").read_text()


@pytest.mark.parametrize("mode", [TARGET, WEIGHT], ids=["target", "weight"])
def test_the_re_timing_cap_and_its_warning_are_the_same_in_both_paths(tmp_path, monkeypatch,
                                                                      capsys, mode):
    plan = runner.plan_budget_retime
    monkeypatch.setattr(runner, "plan_budget_retime", lambda rows: plan(rows, max_points=1))
    a, b = Point("bposd", 0.5, 3), Point("bposd", 0.5, 5)
    scenario = Scenario(points={a: 50, b: 60}, ms30_failures=120,
                        screen_latency={a: (50.0, 100.0), b: (55.0, 80.0)},
                        retime={a: (50.0, 104.0), b: (55.0, 85.0)},
                        confirm_latency={a: (50.0, 105.0), b: (55.0, 86.0)},
                        confirm_failures={a: 0, b: 0})
    _install(monkeypatch, scenario)
    code, _out, results = _run(tmp_path, mode)
    capsys.readouterr()
    assert code == 0
    assert scenario.retimed == [a]                     # nearest to the budget first
    assert results["budget_retime"]["capped"] == [{"label": b.label(), "screen_ratio":
                                                   pytest.approx(80.0 / BUDGET), "within": True}]
    assert any("inside the re-timing margin were not re-timed" in w for w in results["warnings"])


# -- Confirmed choice: confirm in rank order until one is accepted or the in-budget list is exhausted ---------
P = [Point("bposd", 0.375, 5), Point("bposd", 0.375, 7), Point("bposd", 0.5, 5),
     Point("bposd", 0.5, 7), Point("bposd", 0.5, 11)]
OVER = Point("bposd", 0.25, 3)       # fastest and lowest LER, but over the budget


def _five_candidates(mode, last_latency=(58.0, 100.0)):
    """Five in-budget screen passes (ranked P[0] first in both paths) and one over the budget;
    the first four fail their confirmations: with --ler-target accuracy, latency, both,
    latency; without it (no accuracy bar) all four on latency."""
    failures = {P[0]: 60, P[1]: 62, P[2]: 64, P[3]: 66, P[4]: 68, OVER: 50}
    screen = {p: (50.0 + 2 * i, 80.0) for i, p in enumerate(P)}
    screen[OVER] = (40.0, 300.0)
    return Scenario(points=failures, ms30_failures=120, screen_latency=screen,
                    retime={p: (50.0 + 2 * i, 85.0) for i, p in enumerate(P)},
                    confirm_latency={P[0]: (50.0, 90.0 if mode is TARGET else 128.0),
                                     P[1]: (52.0, 130.0),
                                     P[2]: (54.0, 140.0), P[3]: (56.0, 125.0),
                                     P[4]: last_latency},
                    # with --ler-target 0.01 at 4096 shots: 80 failures fail the bound
                    confirm_failures={P[0]: 80, P[1]: 3, P[2]: 80, P[3]: 3, P[4]: 3})


@pytest.mark.parametrize("mode", [TARGET, WEIGHT], ids=["target", "weight"])
def test_a_fifth_candidate_is_confirmed_and_chosen(tmp_path, monkeypatch, capsys, mode):
    scenario = _five_candidates(mode)
    _install(monkeypatch, scenario)
    code, out, results = _run(tmp_path, mode)
    console = capsys.readouterr().out
    assert code == 0
    ac = results["accuracy_confirmation"]
    assert scenario.confirmed == P and OVER not in scenario.confirmed
    assert ac["max_candidates"] is None
    assert ac["in_budget_candidates"] == 5 and ac["confirmed"] == 5
    assert ac["stopped"] == "accepted" and ac["unconfirmed"] == []
    assert ac["in_budget_order"] == [p.label() for p in P]
    assert [c["rank"] for c in ac["candidates"]] == [1, 2, 3, 4, 5]
    assert [c["accepted"] for c in ac["candidates"]] == [False] * 4 + [True]
    reasons = [c["failure_reasons"] for c in ac["candidates"]]
    assert reasons[4] == []
    assert all(r and all(x.startswith(("accuracy:", "latency:")) for x in r) for r in reasons[:4])
    assert any(x.startswith("latency: p99 130.0 us is over the 110 us budget") for x in reasons[1])
    if mode is TARGET:
        assert len(reasons[0]) == 1 and reasons[0][0].startswith("accuracy: upper bound")
        assert [x.split(":")[0] for x in reasons[2]] == ["accuracy", "latency"]
    else:   # no accuracy bar: only the latency confirmation can fail
        assert all([x.split(":")[0] for x in r] == ["latency"] for r in reasons[:4])
    assert results["choice"]["point"]["label"] == P[4].label()
    assert json.loads((out / "config.json").read_text()) == P[4].config()
    text = "confirmed 5 of 5 in-budget candidate(s), one at a time in the choice's order"
    assert text in (out / "pareto.md").read_text() and text in console
    assert f"{P[1].label()} (rank 2) not accepted: latency: p99 130.0 us" in console


@pytest.mark.parametrize("mode", [TARGET, WEIGHT], ids=["target", "weight"])
def test_a_confirmation_cap_that_leaves_candidates_unconfirmed_warns(tmp_path, monkeypatch,
                                                                     capsys, mode):
    scenario = _five_candidates(mode)
    _install(monkeypatch, scenario)
    code, out, results = _run(tmp_path, mode, "--confirm-candidates", "3")
    capsys.readouterr()
    assert code == runner.NO_QUALIFIER_EXIT
    ac = results["accuracy_confirmation"]
    assert scenario.confirmed == P[:3]
    assert ac["max_candidates"] == 3 and ac["confirmed"] == 3 and ac["stopped"] == "cap"
    assert ac["unconfirmed"] == [P[3].label(), P[4].label()]
    assert results["inputs"]["confirm_candidates"] == 3
    assert "left 2 in-budget candidate(s) unconfirmed" in results["choice"]["reason"]
    assert any(w.startswith("--confirm-candidates 3 stopped the confirmation after 3 "
                            "candidate(s) with 2 in-budget candidate(s) never confirmed")
               for w in results["warnings"])
    assert not (out / "config.json").exists() and (out / "config.nearest.json").exists()


@pytest.mark.parametrize("mode", [TARGET, WEIGHT], ids=["target", "weight"])
def test_an_exhausted_in_budget_list_gives_no_choice(tmp_path, monkeypatch, capsys, mode):
    scenario = _five_candidates(mode, last_latency=(58.0, 120.0))
    _install(monkeypatch, scenario)
    code, out, results = _run(tmp_path, mode)
    capsys.readouterr()
    assert code == runner.NO_QUALIFIER_EXIT
    ac = results["accuracy_confirmation"]
    assert scenario.confirmed == P
    assert ac["confirmed"] == 5 and ac["stopped"] == "exhausted" and ac["unconfirmed"] == []
    assert all(c["failure_reasons"] for c in results["choice"]["confirmation_candidates"])
    assert not (out / "config.json").exists() and (out / "config.nearest.json").exists()


# -- Nearest list: a line prints the p50 / p99 of the measurement behind its ratio -----------
NEAR = Point("bposd", 0.625, 7)
SLOW = Point("bposd", 0.5, 31)


def test_nearest_lines_without_a_confirmed_candidate_print_the_re_timing(tmp_path, monkeypatch,
                                                                         capsys):
    """Tuner 2.1 printed NEAR as "p99 97.0 us, 1.21x the budget": the stage p99 next to the
    re-timed ratio (re-timed p99 121.0).  Nothing is in budget, so nothing is confirmed."""
    scenario = Scenario(points={NEAR: 60, SLOW: 50}, ms30_failures=120,
                        screen_latency={NEAR: (60.0, 97.0), SLOW: (150.0, 260.0)},
                        retime={NEAR: (63.0, 121.0)}, confirm_latency={})
    _install(monkeypatch, scenario)
    code, out, results = _run(tmp_path, WEIGHT, "--p99-budget-us", "100")
    console = capsys.readouterr().out
    assert code == runner.NO_QUALIFIER_EXIT and scenario.confirmed == []
    near, slow = results["choice"]["nearest"]
    assert near["label"] == NEAR.label() and slow["label"] == SLOW.label()
    assert (near["p50_us"], near["p99_us"]) == (63.0, 121.0)
    assert near["budget_distance"] == pytest.approx(1.21)
    assert near["measurement"] == "re-timed at 8000 calls per side" and near["calls"] == 8000
    assert (slow["p99_us"], slow["measurement"]) == (260.0, "latency stage, 2000 calls per side")
    pareto = (out / "pareto.md").read_text()
    line = (f"nearest `{NEAR.label()}`: LER upper {report.fmt_prob(near['ler_upper'])}, p50 "
            f"63.0 us, p99 121.0 us, 1.21x the budget (re-timed at 8000 calls per side)")
    assert line in pareto and line.replace("`", "") in console
    assert ("p99 260.0 us, 2.60x the budget (latency stage, 2000 calls per side; above 2x the "
            "budget)") in pareto
    assert "p99 97.0 us, 1.21x" not in pareto and "p99 97.0 us, 1.21x" not in console


@pytest.mark.parametrize("mode", [TARGET, WEIGHT], ids=["target", "weight"])
def test_nearest_lines_after_the_confirmation_print_one_measurement_each(tmp_path, monkeypatch,
                                                                         capsys, mode):
    """A confirmed candidate by its latency confirmation, an unconfirmed point by its re-timing."""
    cand, out_point = Point("bposd", 0.5, 5), Point("bposd", 0.5, 7)
    scenario = Scenario(points={cand: 60, out_point: 55}, ms30_failures=120,
                        screen_latency={cand: (55.0, 90.0), out_point: (58.0, 99.0)},
                        retime={cand: (55.0, 95.0), out_point: (58.5, 118.0)},
                        confirm_latency={cand: (56.0, 130.0)}, confirm_failures={cand: 0})
    _install(monkeypatch, scenario)
    code, out, results = _run(tmp_path, mode)
    capsys.readouterr()
    assert code == runner.NO_QUALIFIER_EXIT and scenario.confirmed == [cand]
    first, second = results["choice"]["nearest"]
    assert first["label"] == out_point.label()            # 118 / 110 before 130 / 110
    assert (first["p50_us"], first["p99_us"]) == (58.5, 118.0)
    assert first["budget_distance"] == pytest.approx(118.0 / BUDGET)
    assert first["measurement"] == "re-timed at 8000 calls per side"
    assert (second["p99_us"], second["measurement"]) == (130.0, "latency confirmation, 8000 calls")
    pareto = (out / "pareto.md").read_text()
    assert "p99 118.0 us, 1.07x the budget (re-timed at 8000 calls per side)" in pareto
    assert "p99 130.0 us, 1.18x the budget (latency confirmation, 8000 calls)" in pareto
    assert "p99 99.0 us, 1.07x" not in pareto


# -- Tie set: the marks follow the accepted choice ---------------------------------------------------
def test_the_tie_set_marks_the_accepted_choice_and_the_screen_pick(tmp_path, monkeypatch, capsys):
    pick, accepted = Point("bposd", 0.375, 3), Point("bposd", 0.5, 5)
    scenario = Scenario(points={pick: 100, accepted: 110}, ms30_failures=120,
                        screen_latency={pick: (57.0, 100.0), accepted: (60.0, 101.0)},
                        retime={pick: (57.0, 105.0), accepted: (60.0, 104.0)},
                        confirm_latency={pick: (57.0, 139.0), accepted: (60.0, 107.0)})
    _install(monkeypatch, scenario)
    code, out, results = _run(tmp_path, WEIGHT)
    console = capsys.readouterr().out
    assert code == 0
    choice = results["choice"]
    assert choice["point"]["label"] == accepted.label()
    assert choice["screen_choice_label"] == pick.label()
    assert [(t["label"], t["chosen"], t["screen_pick"]) for t in choice["tie_set"]] == \
        [(pick.label(), False, True)]
    pareto = (out / "pareto.md").read_text()
    tie_line = next(line for line in pareto.splitlines()
                    if line.startswith(f"  - `{pick.label()}`: p50"))
    assert tie_line.endswith("(screen pick)") and "<- chosen" not in tie_line
    assert "<- chosen" not in pareto and "<- chosen" not in console
    table = {line.split(" | ")[0].lstrip("| "): line for line in pareto.splitlines()
             if line.startswith("| bposd")}
    assert table[accepted.label()].endswith("; CHOSEN |")
    assert table[pick.label()].endswith("; screen pick (not accepted) |")


def test_the_report_marks_a_tie_set_with_both():
    tie = {"p50_us": 50.0, "p99_us": 90.0, "ler_upper": 1e-3, "max_iter": 3,
           "ms_scaling_factor": 0.5}
    assert report.tie_mark(dict(tie, chosen=True, screen_pick=False)) == "  <- chosen"
    assert report.tie_mark(dict(tie, chosen=False, screen_pick=True)) == "  (screen pick)"
    assert report.tie_mark(dict(tie, chosen=True, screen_pick=True)) == "  <- chosen"
    assert report.tie_mark(dict(tie, chosen=False)) == ""
