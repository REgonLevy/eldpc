# SPDX-License-Identifier: Apache-2.0
"""One quick tuner run on the committed small circuit: the eldpc.tune.v2 records, end to end.

The run needs a GPU decoder; it is skipped when none can be created (and fails instead with
ELDPC_REQUIRE_GPU=1, see conftest.py).  It keeps the schema, the confirmation arithmetic, the
choice rule, the per-candidate latency records and the core-variant stage covered end to end.
"""
import json
import os

import pytest

import eldpc
from eldpctune import problems, runner
from eldpctune.grid import Point
from eldpctune.stats import clopper_pearson_upper

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STIM = os.path.join(DATA, "surface_d3_r2.stim")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    prob = problems.problem_from_circuit_file(STIM)
    try:
        decoder = eldpc.Decoder(prob.H, prob.O, prob.priors,
                                config=Point("bposd", 0.5, 3).config())
        decoder.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GPU decoder available: {exc}")
    cfg = runner.TuneConfig(circuit=STIM, ler_target=0.5, p99_budget_us=1_000_000.0,
                            quick=True, auto=True, out=str(tmp_path_factory.mktemp("tune")))
    cfg.command = ["eldpc-tune", "--test"]
    return runner.run(cfg, log=lambda *_: None)


def test_schema_and_accuracy_confirmation(results):
    assert results["schema"] == "eldpc.tune.v2"
    ac = results["accuracy_confirmation"]
    assert ac["planned_shots"] > 0 and ac["shots_rule"]
    assert ac["cases"] and all("seed" in case for case in ac["cases"])
    assert ac["candidates"] and ac["max_candidates"] is None     # no cap by default
    assert ac["confirmed"] == len(ac["candidates"]) <= ac["in_budget_candidates"]
    assert ac["stopped"] in ("accepted", "exhausted")
    assert ac["candidate_order"] == ac["in_budget_order"][:ac["confirmed"]]
    n = int(ac["planned_shots"])
    for candidate in ac["candidates"]:
        assert candidate["per_p"] and all(int(e["shots"]) == n for e in candidate["per_p"])
        for entry in candidate["per_p"]:
            expected = clopper_pearson_upper(int(entry["failures"]), int(entry["shots"]),
                                             ac["confidence"])
            assert entry["upper"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert candidate["upper"] == max(e["upper"] for e in candidate["per_p"])
        assert candidate["ler"] == max(e["ler"] for e in candidate["per_p"])
        passed = all(e["upper"] < ac["target"] for e in candidate["per_p"])
        assert candidate["passed"] is passed
        assert candidate["accepted"] is (passed and candidate["latency_within"] is True)
        assert candidate["screen"]["shots"] > 0


def test_the_choice_and_its_ler_come_from_the_confirmation(results):
    ac = results["accuracy_confirmation"]
    choice = results["choice"]
    if ac["chosen"] is None:
        assert choice.get("config") is None
        return
    first = next(c for c in ac["candidates"] if c["label"] == ac["chosen"])
    assert choice["ler_source"] == "accuracy_confirmation"
    assert choice["ler"] == first["ler"] and choice["ler_upper"] == first["upper"]
    assert choice["screen"]["source"] == "screen"
    assert choice["point"]["decoder"] == first["point"]["decoder"]
    assert choice["point"]["max_iter"] == first["point"]["max_iter"]
    assert results["confirmation"]["label"] == first["label"]
    assert results["confirmation"]["budget"]["within"] is True
    assert 0.0 <= choice["slow_fraction"] <= 1.0
    assert choice["p99_unreliable"] == (0.005 <= choice["slow_fraction"] <= 0.02)


def test_pruning_converged_shares_and_slow_flags(results):
    pruning = results["pruning"]
    assert pruning["min_converged"] == 0.99 and pruning["quantile"] == "p99"
    assert pruning["confidence"] == 0.99
    for stage in results["stages"]:
        for point in stage["points"]:
            if point["status"] != "skipped":
                assert 0.0 <= point["converged_share"] <= 1.0
                for entry in point["per_p"]:
                    assert 0.0 <= entry["converged_share"] <= 1.0
            else:
                assert "dominator_converged_share" in point
    rows = [row for row in results["pareto"] if row.get("p99_us") is not None]
    assert rows
    for row in rows:
        assert row["p99_unreliable"] == (0.005 <= row["slow_fraction"] <= 0.02)
        if row["status"] == "skipped":
            assert row["dominator_converged_share"] >= pruning["min_converged"]


def test_latency_measurements_have_their_own_syndromes(results):
    latency = results["latency"]
    assert latency["aa"]["distinct_syndromes"] > 0
    labels = {case["label"] for case in latency["cases"]}
    seeds = []
    for entry in latency["experiments"]:
        assert entry["distinct_syndromes"] > 0
        assert set(entry["syndrome_seeds"]) == labels
        seeds += [entry["syndrome_seeds"][label] for label in labels]
    for case in latency["aa"]["cases"]:
        seeds += list(case["syndrome_seeds"])
    # no two latency measurements share a syndrome stream
    assert len(seeds) == len(set(seeds))
    confirmation = results["confirmation"]
    assert confirmation["distinct_syndromes"] > 0
    assert len(set(confirmation["syndrome_seeds"].values())) == len(
        confirmation["syndrome_seeds"])


def test_every_confirmed_candidate_records_its_latency_syndromes(results):
    """Chosen or not, every candidate's latency confirmation keeps its syndrome fields."""
    candidates = results["accuracy_confirmation"]["candidates"]
    assert candidates
    labels = {case["label"] for case in results["latency"]["cases"]}
    for candidate in candidates:
        latency = candidate["latency"]
        assert latency["distinct_syndromes"] > 0
        assert set(latency["syndrome_seeds"]) == labels
        assert latency["core_variant"] in ("default", "ada64")
        assert "slow_fraction" in latency and "p99_unreliable" in latency
    chosen = results["accuracy_confirmation"]["chosen"]
    if chosen is not None:
        record = next(c for c in candidates if c["label"] == chosen)
        assert record["latency"]["syndrome_seeds"] == results["confirmation"]["syndrome_seeds"]
        assert record["latency"]["distinct_syndromes"] == results["confirmation"][
            "distinct_syndromes"]


def test_p99_warnings_are_recorded_for_every_flagged_candidate(results):
    flagged = [c["label"] for c in results["accuracy_confirmation"]["candidates"]
               if c["latency"]["p99_unreliable"]]
    flagged += [row["label"] for row in results["pareto"] if row.get("p99_unreliable")]
    assert isinstance(results["warnings"], list)
    assert len(results["warnings"]) == len(flagged)
    for label, warning in zip(flagged, results["warnings"]):
        assert warning.startswith(label)


def test_core_variant_stage_follows_the_device_class(results):
    """Both variants are timed on a two-variant device; elsewhere the stage records a skip."""
    record = results["core_variants"]
    assert record is not None and record["label"] == results["accuracy_confirmation"]["chosen"]
    table = runner.table_core_variant((eldpc.devices() or [None])[0])
    assert record["table_variant"] == table
    if table == "default":
        assert record["evaluated"] is False and record["reason"]
    else:
        assert record["evaluated"] is True
        assert record["faster"] in ("default", "ada64", "no difference")
        assert record["default"]["calls"] > 0 and record["ada64"]["calls"] > 0
        assert (record["recommended_env"] is None) == (record["faster"] in (
            "no difference", record["table_variant"]))
    assert "ELDPC_DEVICE_CLASS" not in json.dumps(results["choice"]["config"])


def test_no_accepted_candidate_gives_no_choice(tmp_path):
    """A target the confirmation cannot pass at 10 shots leaves the run without a choice."""
    prob = problems.problem_from_circuit_file(STIM)
    try:
        decoder = eldpc.Decoder(prob.H, prob.O, prob.priors,
                                config=Point("bposd", 0.5, 3).config())
        decoder.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GPU decoder available: {exc}")
    cfg = runner.TuneConfig(circuit=STIM, ler_target=0.3, confirm_shots=10, max_shots=128,
                            p99_budget_us=1_000_000.0, quick=True, auto=True,
                            out=str(tmp_path))
    cfg.command = ["eldpc-tune", "--test"]
    results = runner.run(cfg, log=lambda *_: None)
    ac = results["accuracy_confirmation"]
    assert ac["planned_shots"] == 10
    assert ac["candidates"], "the screen has survivors to confirm"
    assert all(c["passed"] is False for c in ac["candidates"])
    assert all(c["accepted"] is False for c in ac["candidates"])
    assert ac["chosen"] is None
    assert results["choice"].get("config") is None
    assert results["confirmation"] is None
    assert results["choice"]["confirmation_candidates"]
    assert results["choice"]["reason"]
    # the nearest list (and so config.nearest.json) is written as without a target
    assert results["choice"]["nearest"]
    assert all(near.get("accuracy_passed") is False for near in results["choice"]["nearest"]
               if "accuracy_passed" in near)


def test_resume_continues_the_confirmation_from_its_checkpoint(tmp_path):
    """A second run with --resume reuses the saved confirmation progress and counts.

    Ten confirmation shots cannot prove a 0.3 target, so every in-budget candidate is rejected and
    both runs confirm the whole in-budget list (the confirmation stops at the first accepted
    candidate, and the re-measured latency may reorder the list): the runs share candidates."""
    prob = problems.problem_from_circuit_file(STIM)
    try:
        decoder = eldpc.Decoder(prob.H, prob.O, prob.priors,
                                config=Point("bposd", 0.5, 3).config())
        decoder.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GPU decoder available: {exc}")

    def run(resume: bool):
        cfg = runner.TuneConfig(circuit=STIM, ler_target=0.3, confirm_shots=10, max_shots=128,
                                p99_budget_us=1_000_000.0, quick=True, auto=True,
                                out=str(tmp_path), resume=resume)
        cfg.command = ["eldpc-tune", "--test"]
        logs = []
        return runner.run(cfg, log=logs.append), logs

    first, logs = run(False)
    checkpoint_path = tmp_path / "raw" / "confirmation-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["complete"] is True
    assert first["accuracy_confirmation"]["stopped"] == "exhausted"
    # one confirmation call per candidate: the saved state accumulates every candidate
    assert set(checkpoint["candidates"]) == {candidate["label"] for candidate in
                                             first["accuracy_confirmation"]["candidates"]}
    assert all(candidate["resumed_case_shots"] == 0
               for candidate in first["accuracy_confirmation"]["candidates"])
    second, logs = run(True)
    assert any("confirmation progress loaded" in line for line in logs)
    assert any("resumed with" in line for line in logs)
    assert second["accuracy_confirmation"]["resumed_case_shots"] > 0
    assert any(candidate["resumed_case_shots"] > 0
               for candidate in second["accuracy_confirmation"]["candidates"])
    first_failures = {candidate["label"]: [entry["failures"] for entry in candidate["per_p"]]
                      for candidate in first["accuracy_confirmation"]["candidates"]}
    second_failures = {candidate["label"]: [entry["failures"] for entry in candidate["per_p"]]
                       for candidate in second["accuracy_confirmation"]["candidates"]}
    common = set(first_failures) & set(second_failures)
    assert common and all(first_failures[label] == second_failures[label] for label in common)


def test_dem_options_are_recorded_and_undecomposed_by_default(results):
    case = results["problem_cases"][0]
    assert case["dem_options"]["decompose_errors"] is False
    assert case["dem_options"]["approximate_disjoint_errors"] is True
    assert case["num_mechanisms"] > 0 and case["dem_sha256"]
