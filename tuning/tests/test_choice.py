# SPDX-License-Identifier: Apache-2.0
"""The Pareto choice and the budget rule, on synthetic measurements."""
from eldpctune.grid import Point, coarse_points
from eldpctune.runner import TuneConfig, choose


def _record(status="pass", upper=1e-5, shots=1000, failures=0):
    return {"status": status, "cp99_upper_max": upper, "shots": shots, "failures": failures,
            "per_p": []}


def _stage(measurements, band_p50=None, band_p99=None):
    """measurements: list of (point, p50, p99); the bands are the A/A bands of the machine."""
    stage = {"_measured_points": {
        point: {"cand": {"p50_us": p50, "p99_us": p99, "mean_us": p50, "calls": 100},
                "verdict": "initial best"}
        for point, p50, p99 in measurements}}
    if band_p50 is not None or band_p99 is not None:
        stage["aa"] = {"band_p50": band_p50 or 0.0, "band_p99": band_p99 or 0.0}
    return stage


def test_fastest_point_within_both_budgets_wins():
    a = Point("bposd", 0.3125, 3)
    b = Point("bposd", 0.5, 5)
    c = Point("bposd", 0.5, 7)
    results = {a: _record(), b: _record(), c: _record()}
    # a is too slow at p50, b is too slow at p99, c meets both budgets
    cfg = TuneConfig(out="/unused", ler_target=1e-4, p50_budget_us=55.0, p99_budget_us=80.0)
    choice = choose(cfg, results, _stage([(a, 60.0, 90.0), (b, 40.0, 120.0), (c, 50.0, 70.0)]))
    assert choice["config"] == c.config()
    assert choice["p50_us"] == 50.0
    assert choice["budget"]["within"] is True


def test_a_point_over_budget_is_not_chosen():
    fast = Point("bposd", 0.3125, 3)
    slow = Point("bposd", 0.5, 5)
    results = {fast: _record(), slow: _record()}
    cfg = TuneConfig(out="/unused", ler_target=1e-4, p99_budget_us=100.0)
    choice = choose(cfg, results, _stage([(fast, 40.0, 150.0), (slow, 80.0, 90.0)]))
    assert choice["config"] == slow.config()


def test_no_qualifier_lists_nearest_points():
    a = Point("bposd", 0.3125, 3)
    b = Point("bposd", 0.5, 5)
    results = {a: _record(), b: _record()}
    cfg = TuneConfig(out="/unused", ler_target=1e-4, p99_budget_us=50.0)
    choice = choose(cfg, results, _stage([(a, 40.0, 90.0), (b, 60.0, 120.0)]))
    assert choice["config"] is None
    assert choice["reason"].startswith("no point")
    assert [n["label"] for n in choice["nearest"]] == [a.label(), b.label()]


def test_failed_points_are_not_candidates():
    good = Point("bposd", 0.5, 3)
    bad = Point("bposd", 0.3125, 1)
    results = {good: _record(), bad: _record(status="fail", upper=0.2)}
    cfg = TuneConfig(out="/unused", ler_target=1e-4)
    choice = choose(cfg, results, _stage([(good, 50.0, 70.0)]))
    assert choice["config"] == good.config()


def test_weight_ranks_by_the_documented_score():
    fast = Point("bposd", 0.5, 3)
    accurate = Point("bposd", 0.5, 15)
    results = {fast: _record(upper=8e-5), accurate: _record(upper=1e-6)}
    stage = _stage([(fast, 50.0, 70.0), (accurate, 150.0, 200.0)])
    without = choose(TuneConfig(out="/unused", ler_target=1e-4), results, stage)
    assert without["config"] == fast.config()
    with_weight = choose(TuneConfig(out="/unused", ler_target=1e-4, weight=3.0), results, stage)
    assert with_weight["config"] == accurate.config()
    assert with_weight["ranking"].startswith("score =")


def test_measured_mode_without_a_target_uses_the_fastest_point():
    a = Point("bposd", 0.5, 3)
    b = Point("bposd", 0.5, 5)
    results = {a: _record(status="measured"), b: _record(status="measured")}
    choice = choose(TuneConfig(out="/unused"), results, _stage([(a, 55.0, 80.0), (b, 45.0, 70.0)]))
    assert choice["config"] == b.config()
    assert choice["target_present"] is False


SHIPPED = [0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75]


def _tie_setup(rows):
    """rows: list of (point, ler_upper, p50, p99) -> (results, measurements)."""
    results = {point: _record(upper=upper) for point, upper, _p50, _p99 in rows}
    return results, [(point, p50, p99) for point, _upper, p50, p99 in rows]


def test_tie_set_takes_the_lowest_ler_bound_inside_the_band():
    # 0.011 us (0.03 %) apart, far inside a 0.2 % A/A band: the machine cannot tell them apart.
    fast = Point("bposd", 0.5, 3)
    close = Point("bposd", 0.5, 5)
    far = Point("bposd", 0.5, 7)
    results, measured = _tie_setup([(fast, 8e-5, 40.000, 60.0), (close, 3e-5, 40.011, 60.0),
                                    (far, 1e-9, 41.0, 60.0)])
    cfg = TuneConfig(out="/unused", ler_target=1e-4)
    choice = choose(cfg, results, _stage(measured, band_p50=0.002), SHIPPED)
    assert choice["config"] == close.config()          # lower LER bound beats 0.011 us
    assert [t["label"] for t in choice["tie_set"]] == [close.label(), fast.label()]
    assert choice["fastest"] == fast.label()
    assert [t["chosen"] for t in choice["tie_set"]] == [True, False]
    # the point outside the band never joins, whatever its LER bound
    assert far.label() not in [t["label"] for t in choice["tie_set"]]
    # without a measured band only exact latency ties remain: the fastest point wins
    exact = choose(cfg, results, _stage(measured), SHIPPED)
    assert exact["config"] == fast.config()
    assert [t["label"] for t in exact["tie_set"]] == [fast.label()]


def test_tie_order_is_ler_then_iterations_then_preset_alpha_then_name():
    cfg = TuneConfig(out="/unused", ler_target=1e-4)

    def pick(rows):
        results, measured = _tie_setup(rows)
        return choose(cfg, results, _stage(measured, band_p50=0.01), SHIPPED)["point"]["label"]

    # (2) equal LER bounds: fewer BP iterations
    assert pick([(Point("bposd", 0.5, 5), 2e-5, 40.0, 60.0),
                 (Point("bposd", 0.5, 3), 2e-5, 40.1, 60.0)]) == "bposd:a=0.5:c=3"
    # (3) equal LER bound and cap: the alpha closest to a shipped preset's alpha
    assert pick([(Point("bposd", 0.46875, 3), 2e-5, 40.0, 60.0),
                 (Point("bposd", 0.5, 3), 2e-5, 40.1, 60.0)]) == "bposd:a=0.5:c=3"
    # the preset distance decides before the name does: 0.625 is a preset, 0.40625 is not
    assert pick([(Point("bposd", 0.40625, 3), 2e-5, 40.0, 60.0),
                 (Point("bposd", 0.625, 3), 2e-5, 40.0, 60.0)]) == "bposd:a=0.625:c=3"
    # (4) everything else equal: name order (and not the input order)
    forward = [(Point("bposd", 0.5, 3), 2e-5, 40.0, 60.0), (Point("bposd", 0.375, 3), 2e-5, 40.0, 60.0)]
    assert pick(forward) == "bposd:a=0.375:c=3"
    assert pick(list(reversed(forward))) == "bposd:a=0.375:c=3"
    assert pick([(Point("bposd", 0.5, 3), 2e-5, 40.0, 60.0),
                 (Point("bp", 0.5, 3), 2e-5, 40.0, 60.0)]) == "bp:a=0.5:c=3"


def test_tie_set_applies_the_p99_band_when_the_budget_is_a_p99():
    low_tail = Point("bposd", 0.5, 3)
    long_tail = Point("bposd", 0.5, 5)
    results, measured = _tie_setup([(low_tail, 8e-5, 40.0, 60.0), (long_tail, 1e-5, 40.01, 90.0)])
    stage = _stage(measured, band_p50=0.01, band_p99=0.05)
    with_p99 = choose(TuneConfig(out="/unused", ler_target=1e-4, p99_budget_us=100.0),
                      results, stage, SHIPPED)
    assert with_p99["config"] == low_tail.config()      # p99 90 vs 60 is outside a 5 % band
    with_p50 = choose(TuneConfig(out="/unused", ler_target=1e-4, p50_budget_us=100.0),
                      results, stage, SHIPPED)
    assert with_p50["config"] == long_tail.config()     # only p50 matters: they tie, lower LER
    # a wide p99 band (sparse cadence) lets the longer tail join
    wide = choose(TuneConfig(out="/unused", ler_target=1e-4, p99_budget_us=100.0), results,
                  _stage(measured, band_p50=0.01, band_p99=0.6), SHIPPED)
    assert wide["config"] == long_tail.config()


def test_tie_set_with_a_weight_uses_the_score_slack():
    a = Point("bposd", 0.5, 3)
    b = Point("bposd", 0.5, 5)
    # b has the lower score (1 % faster for a 0.2 % higher LER bound), but the 1 % is inside the
    # 2 % A/A band, so the two tie and the lower LER bound (a) is chosen.
    results, measured = _tie_setup([(a, 4.0e-5, 40.4, 60.0), (b, 4.008e-5, 40.0, 60.0)])
    cfg = TuneConfig(out="/unused", ler_target=1e-4, weight=1.0)
    tied = choose(cfg, results, _stage(measured, band_p50=0.02), SHIPPED)
    assert tied["fastest"] == b.label()
    assert {t["label"] for t in tied["tie_set"]} == {a.label(), b.label()}
    assert tied["config"] == a.config()
    alone = choose(cfg, results, _stage(measured), SHIPPED)
    assert alone["config"] == b.config()
    assert len(alone["tie_set"]) == 1


def test_shipped_preset_alphas_come_from_the_package():
    import eldpc
    from eldpctune.runner import preset_alphas

    alphas = preset_alphas()
    assert alphas == sorted(set(alphas)) and len(alphas) >= 3
    assert {cfg["ms_scaling_factor"] for cfg in eldpc.presets().values()
            if "ms_scaling_factor" in cfg} == set(alphas)


def test_coarse_grid_points_are_hashable_and_ordered():
    points = coarse_points()
    assert len(set(points)) == len(points)
    assert sorted(points) == sorted(points)
