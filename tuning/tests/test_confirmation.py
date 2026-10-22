# SPDX-License-Identifier: Apache-2.0
"""The fixed-size accuracy confirmation: shot-count rule, seeds, shots and verdicts."""
import numpy as np
import pytest

from eldpctune import confirmation, stats
from eldpctune.confirmation import (PASS_PROBABILITY, planned_shots, pass_probability,
                                    run_accuracy_confirmation)
from eldpctune.grid import Point
from eldpctune.runner import TuneConfig, pruning_record
from eldpctune.stats import derive_seed


# -- the shot-count rule ---------------------------------------------------
def test_planned_shots_is_the_crossing_the_documented_search_returns():
    """N meets the margin and N - 1 does not: the doubling-then-bisection result, as recorded."""
    target, confidence = 0.05, 0.99
    n, rule = planned_shots(target, confidence)
    assert n > 0 and "target/1.5" in rule and "not monotone" in rule
    assert pass_probability(n, target, confidence, target / 1.5) >= PASS_PROBABILITY
    if n > 1:
        assert pass_probability(n - 1, target, confidence, target / 1.5) < PASS_PROBABILITY
    # a tighter target needs more shots
    assert planned_shots(1e-3, confidence)[0] > n


def test_planned_shots_is_not_claimed_to_be_the_smallest_n():
    """The pass probability is not monotone in N, so a smaller N can also meet the margin.

    The rule text and the docs describe the search, not a minimum; this pins the documented
    example (target 0.05, confidence 0.99: the search returns 1944 while 1899 also passes).
    """
    target, confidence = 0.05, 0.99
    n, rule = planned_shots(target, confidence)
    assert n == 1944
    assert "smallest" not in rule
    assert pass_probability(1899, target, confidence, target / 1.5) >= PASS_PROBABILITY
    smaller = [m for m in range(1800, n)
               if pass_probability(m, target, confidence, target / 1.5) >= PASS_PROBABILITY]
    assert smaller and min(smaller) == 1899


def test_planned_shots_override_and_quick_cap():
    n, rule = planned_shots(0.05, 0.99, override=777)
    assert n == 777 and "--confirm-shots" in rule
    with pytest.raises(ValueError):
        planned_shots(0.05, 0.99, override=0)
    quick_n, quick_rule = planned_shots(0.05, 0.99, quick=True)
    full_n = planned_shots(0.05, 0.99)[0]
    assert quick_n == min(full_n, confirmation.QUICK_MAX_SHOTS)
    if full_n > confirmation.QUICK_MAX_SHOTS:
        assert "--quick caps N" in quick_rule
    assert planned_shots(None, 0.99, quick=True)[0] == confirmation.NO_TARGET_SHOTS[True]


def test_confirmation_seeds_are_fresh():
    """The confirmation seed stream is distinct from every screen, latency and keep-warm stream."""
    seed, key = 20261006, "problem:key"
    other = {derive_seed(seed, stage, key)
             for stage in ("coarse", "refine", "latency-shots", "latency-confirm",
                           "keep-warm-shots")}
    confirm_seed = derive_seed(seed, "confirm", key)
    assert confirm_seed not in other
    assert confirm_seed == derive_seed(seed, "confirm", key)


# -- the confirmation stage on a scripted problem --------------------------
class FakeSource:
    """Random packed shots from a seed; the observable vector is all zero."""

    def __init__(self, m: int, k: int, seed: int):
        self.m, self.k = int(m), int(k)
        self._rng = np.random.default_rng(int(seed))

    def sample(self, shots: int):
        bits = (self._rng.random((int(shots), self.m)) < 0.5).astype(np.uint8)
        dets = np.packbits(bits, axis=1, bitorder="little")
        obs = np.zeros((int(shots), (self.k + 7) // 8), dtype=np.uint8)
        return dets, obs


class FakeProblem:
    def __init__(self, m=8, k=1):
        self.m, self.k = int(m), int(k)

    def shot_source(self, seed):
        return FakeSource(self.m, self.k, seed)


class CountingDecoder:
    """Predicts a wrong observable on the first ``failing`` shots of its own stream."""

    num_observables = 1

    def __init__(self, failing: int):
        self.failing = int(failing)
        self.position = 0

    def decode_batch(self, dets):
        n = len(dets)
        predicted = np.zeros((n, 1), dtype=np.uint8)
        predicted[:max(0, min(n, self.failing - self.position)), 0] = 1
        self.position += n
        return None, predicted, np.zeros(n, dtype=np.uint32)

    def close(self):
        pass


class DetectorDecoder:
    """Predicts the first detector of every shot: a seed-dependent failure count."""

    num_observables = 1

    def decode_batch(self, dets):
        n = len(dets)
        return None, np.asarray(dets)[:, :1].copy(), np.zeros(n, dtype=np.uint32)

    def close(self):
        pass


def _case():
    prob = FakeProblem()
    return {"label": "case", "p": 0.02, "problem": prob, "seed_key": prob_key()}


def prob_key():
    return "fake:key"


def _run(failing_by_cap, n=500, seed=123, confidence=0.99):
    cases = [_case()]
    points = [Point("bposd", 0.5, cap) for cap in failing_by_cap]
    seeds = {case["label"]: derive_seed(seed, "confirm", case["seed_key"]) for case in cases}
    screens = {"bposd:a=0.5:c=%d" % cap: {"shots": 100, "failures": 1, "ler": 0.01,
                                          "upper": 0.05, "source": "screen"}
               for cap in failing_by_cap}

    def make(point, case):
        return CountingDecoder(failing_by_cap[point.max_iter])

    return run_accuracy_confirmation(
        make, [(p, screens[p.label()]) for p in points], cases, seeds, n, confidence,
        log=lambda *_: None)


def test_every_candidate_gets_the_fixed_shot_count_and_none_stops_early():
    records = _run({1: 0, 2: 50})
    assert all(entry["shots"] == 500 for record in records for entry in record["per_p"])
    # the confirmation has no early stop: a point that would pass at 500 shots still decodes all
    assert [entry["shots"] for entry in records[0]["per_p"]] == [500]


def test_verdict_bounds_are_the_hand_computed_clopper_pearson_bounds():
    records = _run({1: 0, 2: 50})
    upper0 = stats.clopper_pearson_upper(0, 500, 0.99)
    upper50 = stats.clopper_pearson_upper(50, 500, 0.99)
    assert records[0]["per_p"][0]["failures"] == 0
    assert records[1]["per_p"][0]["failures"] == 50
    assert records[0]["upper"] == pytest.approx(upper0, rel=1e-12)
    assert records[1]["upper"] == pytest.approx(upper50, rel=1e-12)
    assert records[0]["per_p"][0]["lower"] == pytest.approx(
        stats.clopper_pearson_lower(0, 500, 0.99), rel=1e-12)
    # worst case over cases is what the candidate reports
    assert records[1]["ler"] == pytest.approx(50 / 500)


def test_round_measurements_are_deterministic():
    first = _run({1: 0, 2: 5})
    second = _run({1: 0, 2: 5})
    assert [(r["failures"], r["upper"]) for r in first] == \
           [(r["failures"], r["upper"]) for r in second]
    # the candidate decodes the same shot stream in the two runs
    assert [r["per_p"][0]["failures"] for r in first] == \
           [r["per_p"][0]["failures"] for r in second]


def test_a_different_confirmation_seed_changes_the_shots():
    cases = [_case()]
    points = [Point("bposd", 0.5, 1)]

    def run(seed):
        seeds = {case["label"]: derive_seed(seed, "confirm", case["seed_key"]) for case in cases}
        return run_accuracy_confirmation(DetectorDecoder_factory, [(p, {}) for p in points],
                                         cases, seeds, 400, 0.99, log=lambda *_: None)

    a, b = run(1), run(2)
    assert a[0]["per_p"][0]["failures"] != b[0]["per_p"][0]["failures"]


def DetectorDecoder_factory(point, case):  # noqa: N802 - used as a callable above
    return DetectorDecoder()


def test_candidate_order_puts_the_screen_choice_first_then_the_eligible_points():
    """The order: the chosen point, then the other budget-eligible points by p50; a point over
    the budget is never a candidate, and the list is not cut (the confirmation loop applies an
    optional --confirm-candidates cap)."""
    from eldpctune.runner import candidate_order

    slow = Point("bposd", 0.5, 3)       # the screen choice, but the slowest eligible point
    fast = Point("bposd", 0.375, 3)
    middle = Point("bposd", 0.5, 5)
    over = Point("bposd", 0.5, 7)       # over the p99 budget
    results = {p: {"status": "pass", "cp99_upper_max": 1e-5, "shots": 1000,
                   "failures": 0, "per_p": []} for p in (slow, fast, middle, over)}

    def summary(p50, p99):
        return {"p50_us": p50, "p99_us": p99, "mean_us": p50, "calls": 1000,
                "slow_fraction": 0.0, "p99_unreliable": False, "slow_calls": 0,
                "slow_median_us": None}

    stage = {"_measured_points": {
        slow: {"cand": summary(50, 80), "verdict": "initial best"},
        fast: {"cand": summary(30, 40), "verdict": "faster"},
        middle: {"cand": summary(40, 60), "verdict": "within_noise"},
        over: {"cand": summary(10, 200), "verdict": "faster"}},
        "aa": {"band_p50": 0.0, "band_p99": 0.0}}
    choice = {"point": slow.as_dict()}
    cfg = TuneConfig(out="/unused", ler_target=1e-4, p99_budget_us=100.0,
                     confirm_candidates=3)
    assert [row["point"] for row in candidate_order(cfg, results, stage, choice)] == \
        [slow, fast, middle]
    few = TuneConfig(out="/unused", ler_target=1e-4, p99_budget_us=100.0,
                     confirm_candidates=1)
    assert [row["point"] for row in candidate_order(few, results, stage, choice)] == \
        [slow, fast, middle]
    # no accuracy bar: the chosen point, then the other in-budget points in the ranking (p50,
    # or the weight score); the first whose latency confirmation passes is chosen
    no_target = TuneConfig(out="/unused", p99_budget_us=100.0, confirm_candidates=3)
    assert [row["point"] for row in candidate_order(no_target, results, stage, choice)] == \
        [slow, fast, middle]
    one = TuneConfig(out="/unused", p99_budget_us=100.0, confirm_candidates=1)
    assert [row["point"] for row in candidate_order(one, results, stage, choice)] == \
        [slow, fast, middle]


def test_a_saved_partial_state_resumes_to_the_uninterrupted_counts():
    """Resuming after one decoded chunk gives exactly the counts of a full run."""
    cases = [_case()]
    seed = 99
    seeds = {case["label"]: derive_seed(seed, "confirm", case["seed_key"]) for case in cases}
    point = Point("bposd", 0.5, 1)
    full = run_accuracy_confirmation(DetectorDecoder_factory, [(point, {})], cases, seeds,
                                     8192, 0.99, log=lambda *_: None)
    first_chunk = run_accuracy_confirmation(DetectorDecoder_factory, [(point, {})], cases,
                                            seeds, 4096, 0.99, log=lambda *_: None)
    progress = {
        "planned_shots": 8192,
        "cases": [{"label": "case", "seed": seeds["case"]}],
        "candidates": {point.label(): {"case": {
            "shots": 4096,
            "failures": first_chunk[0]["per_p"][0]["failures"],
            "converged": first_chunk[0]["per_p"][0]["converged_shots"]}}},
    }
    saved = []
    resumed = run_accuracy_confirmation(
        DetectorDecoder_factory, [(point, {})], cases, seeds, 8192, 0.99,
        log=lambda *_: None, progress=progress, save_progress=saved.append)
    assert resumed[0]["per_p"][0]["failures"] == full[0]["per_p"][0]["failures"]
    assert resumed[0]["per_p"][0]["converged_shots"] == full[0]["per_p"][0]["converged_shots"]
    assert resumed[0]["upper"] == pytest.approx(full[0]["upper"], rel=1e-15)
    assert resumed[0]["resumed_case_shots"] == 4096
    assert saved and saved[-1]["complete"] is True
    assert saved[-1]["candidates"][point.label()]["case"]["shots"] == 8192


def test_a_complete_saved_state_is_not_decoded_again():
    cases = [_case()]
    seeds = {case["label"]: derive_seed(7, "confirm", case["seed_key"]) for case in cases}
    point = Point("bposd", 0.5, 1)
    full = run_accuracy_confirmation(DetectorDecoder_factory, [(point, {})], cases, seeds,
                                     1000, 0.99, log=lambda *_: None)

    class ExplodingDecoder:
        num_observables = 1

        def decode_batch(self, dets):
            raise AssertionError("a fully resumed candidate must not decode")

        def close(self):
            pass

    progress = {
        "planned_shots": 1000,
        "cases": [{"label": "case", "seed": seeds["case"]}],
        "candidates": {point.label(): {"case": {
            "shots": 1000,
            "failures": full[0]["per_p"][0]["failures"],
            "converged": full[0]["per_p"][0]["converged_shots"]}}},
    }
    resumed = run_accuracy_confirmation(
        lambda point_, case_: ExplodingDecoder(), [(point, {})], cases, seeds, 1000, 0.99,
        log=lambda *_: None, progress=progress)
    assert resumed[0]["per_p"][0]["failures"] == full[0]["per_p"][0]["failures"]


def test_progress_with_other_case_seeds_is_ignored():
    cases = [_case()]
    seeds = {case["label"]: derive_seed(7, "confirm", case["seed_key"]) for case in cases}
    point = Point("bposd", 0.5, 1)
    fresh = run_accuracy_confirmation(DetectorDecoder_factory, [(point, {})], cases, seeds,
                                      500, 0.99, log=lambda *_: None)
    wrong = {
        "planned_shots": 500,
        "cases": [{"label": "case", "seed": 12345}],
        "candidates": {point.label(): {"case": {"shots": 500, "failures": 999,
                                                "converged": 0}}},
    }
    resumed = run_accuracy_confirmation(DetectorDecoder_factory, [(point, {})], cases, seeds,
                                        500, 0.99, log=lambda *_: None, progress=wrong)
    assert resumed[0]["per_p"][0]["failures"] == fresh[0]["per_p"][0]["failures"]


# -- the pruning rule ------------------------------------------------------
def test_pruning_rule_is_stricter_than_the_structural_one():
    from eldpctune.accuracy import dominator

    p3, p5 = Point("bposd", 0.5, 3), Point("bposd", 0.5, 5)
    shots = 2048

    def record(converged_shots):
        return {"status": "pass", "per_p": [{"shots": shots,
                                             "converged_shots": converged_shots}]}

    # p99 budget (q = 0.99): zero slow calls is below 1 % at this shot count, so it dominates
    assert dominator(p5, {p3: record(shots)}, min_converged=0.99) == p3
    # 20 slow calls in 2048 is 0.98 %: the point estimate is below 1 % but the 99 % upper bound
    # (about 1.2 %) is not, so the larger cap is evaluated
    assert stats.clopper_pearson_upper(20, shots, 0.99) > 0.01
    assert dominator(p5, {p3: record(shots - 20)}, min_converged=0.99) is None
    # p50-only budget (q = 0.5): both sides of the 50 % slow-share threshold
    assert stats.clopper_pearson_upper(900, shots, 0.99) < 0.5
    assert dominator(p5, {p3: record(shots - 900)}, min_converged=0.5) == p3
    assert stats.clopper_pearson_upper(1100, shots, 0.99) > 0.5
    assert dominator(p5, {p3: record(shots - 1100)}, min_converged=0.5) is None
    # the structural rule alone (no status data) keeps its old behaviour
    assert dominator(p5, {p3: {"status": "pass"}}) == p3
    assert dominator(p5, {p3: {"status": "pass"}}, min_converged=0.99) is None


def test_pruning_record_threshold_follows_the_budget_quantile():
    p99 = pruning_record(TuneConfig(out="/unused", p99_budget_us=100.0))
    assert p99["min_converged"] == 0.99 and p99["quantile"] == "p99" and p99["enabled"]
    p50 = pruning_record(TuneConfig(out="/unused", p50_budget_us=100.0))
    assert p50["min_converged"] == 0.5 and p50["quantile"] == "p50"
    none = pruning_record(TuneConfig(out="/unused"))
    assert none["min_converged"] == 0.5
    off = pruning_record(TuneConfig(out="/unused", p99_budget_us=100.0, no_prune=True))
    assert off["enabled"] is False and "--no-prune" in off["rule"]


# -- the slow-class budget rule -------------------------------------------
def test_a_p99_budget_uses_the_slow_median_when_the_p99_is_unreliable():
    from eldpctune.runner import _budget_status

    cfg = TuneConfig(out="/unused", p99_budget_us=100.0)
    unreliable = {"p50_us": 10.0, "p99_us": 300.0, "slow_fraction": 0.01,
                  "p99_unreliable": True, "slow_calls": 10, "slow_median_us": 60.0,
                  "mean_us": 20.0, "calls": 1000}
    assert _budget_status(unreliable, cfg)["within"] is True
    over = dict(unreliable, slow_median_us=150.0)
    assert _budget_status(over, cfg)["within"] is False
    # a stable tail uses its p99 directly, whatever the slow fraction
    stable = {"p50_us": 10.0, "p99_us": 150.0, "slow_fraction": 0.05,
              "p99_unreliable": False, "slow_calls": 50, "slow_median_us": 50.0,
              "mean_us": 20.0, "calls": 1000}
    assert _budget_status(stable, cfg)["within"] is False
    assert _budget_status(dict(stable, p99_us=90.0), cfg)["within"] is True
    assert _budget_status(unreliable, TuneConfig(out="/unused", p50_budget_us=20.0))["within"]


def test_slow_fraction_band_edges():
    """p99_unreliable is true exactly inside [0.005, 0.02]."""
    durations = list(range(1000))
    for slow, expected in ((4, False), (5, True), (20, True), (21, False), (0, False)):
        statuses = [0] * slow + [1] * (1000 - slow)
        out = stats.slow_class(durations, statuses)
        assert out["slow_fraction"] == pytest.approx(slow / 1000)
        assert out["p99_unreliable"] is expected
