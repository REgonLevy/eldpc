# SPDX-License-Identifier: Apache-2.0
"""The fixed grid, the dominance rule and the deterministic refinement."""
import pytest

from eldpctune.accuracy import dominator
from eldpctune.grid import (BP_ALPHAS, BP_CAPS, COARSE_ALPHAS, COARSE_CAPS,
                            QUICK_ALPHAS, QUICK_CAPS, Point, coarse_points, refine_points)


def test_coarse_grid_is_fixed_and_unique():
    points = coarse_points()
    assert len(points) == len(COARSE_ALPHAS) * len(COARSE_CAPS) + len(BP_ALPHAS) * len(BP_CAPS)
    assert len(set(points)) == len(points)
    bposd = {(p.alpha, p.max_iter) for p in points if p.decoder == "bposd"}
    bp = {(p.alpha, p.max_iter) for p in points if p.decoder == "bp"}
    assert bposd == {(a, c) for a in COARSE_ALPHAS for c in COARSE_CAPS}
    assert bp == {(a, c) for a in BP_ALPHAS for c in BP_CAPS}
    for p in points:
        assert p.decoder in ("bposd", "bp")
        assert p.max_iter >= 1
        assert 0.125 <= p.alpha <= 0.875


def test_quick_grid_is_smaller():
    quick = coarse_points(quick=True)
    assert len(quick) == len(QUICK_ALPHAS) * len(QUICK_CAPS) + 2 * 2
    assert {p.max_iter for p in quick if p.decoder == "bposd"} == set(QUICK_CAPS)
    assert len(quick) < len(coarse_points())


def test_config_has_the_abi_keys():
    cfg = Point("bposd", 0.3125, 3).config()
    assert cfg["decoder"] == "bposd"
    assert cfg["bp_method"] == "ms"
    assert cfg["schedule"] == "parallel"
    assert cfg["osd_method"] == "osd0"
    assert cfg["osd_order"] == 0
    assert cfg["ms_scaling_factor"] == 0.3125
    assert cfg["max_iter"] == 3
    bp = Point("bp", 0.5, 7).config()
    assert bp["decoder"] == "bp"
    assert "osd_method" not in bp and "osd_order" not in bp


def test_point_rejects_bad_values():
    with pytest.raises(ValueError):
        Point("osd", 0.5, 3)
    with pytest.raises(ValueError):
        Point("bposd", 0.5, 0)


def test_dominance():
    p3 = Point("bposd", 0.3125, 3)
    p5 = Point("bposd", 0.3125, 5)
    p5_other_alpha = Point("bposd", 0.5, 5)
    p5_bp = Point("bp", 0.3125, 5)
    results = {p3: {"status": "pass"}}
    assert dominator(p5, results) == p3
    assert dominator(p5_other_alpha, results) is None
    assert dominator(p5_bp, results) is None
    assert dominator(Point("bposd", 0.3125, 1), results) is None
    results[p3]["status"] = "fail"
    assert dominator(p5, results) is None


def test_refinement_around_a_passing_frontier_is_deterministic():
    coarse = {p: {"status": "fail", "upper": 0.5} for p in coarse_points()}
    passing = Point("bposd", 0.3125, 3)
    coarse[passing] = {"status": "pass", "upper": 1e-5}
    already = set(coarse)
    first = refine_points(False, coarse, already)
    second = refine_points(False, coarse, coarse.keys())
    assert first == second
    assert first, "the refinement must add points around the passing frontier"
    assert all(p not in already for p in first)
    # neighbours of the passing alpha at the frontier cap and at the failing cap below it
    alphas = {p.alpha for p in first if p.decoder == "bposd" and p.max_iter == 3}
    assert any(abs(a - 0.28125) < 1e-9 for a in alphas)
    assert any(abs(a - 0.34375) < 1e-9 for a in alphas)
    caps = {p.max_iter for p in first if p.decoder == "bposd"}
    assert caps <= {2, 3}
    assert all(p.max_iter >= 1 for p in first)


def test_refinement_without_a_pass_refines_the_best_caps():
    coarse = {}
    for p in coarse_points():
        coarse[p] = {"status": "fail", "upper": 0.01 + 0.001 * p.max_iter}
    best = Point("bposd", 0.5, 1)
    coarse[best] = {"status": "fail", "upper": 0.0001}
    refined = refine_points(False, coarse, set(coarse))
    assert refined
    assert {p.max_iter for p in refined if p.decoder == "bposd"} <= {1, 2}


def test_quick_refinement_is_capped():
    coarse = {}
    for p in coarse_points(quick=True):
        coarse[p] = {"status": "fail", "upper": 0.5}
    coarse[Point("bposd", 0.5, 1)] = {"status": "pass", "upper": 1e-5}
    refined = refine_points(True, coarse, set(coarse))
    assert len([p for p in refined if p.decoder == "bposd"]) <= 4
    assert len([p for p in refined if p.decoder == "bp"]) <= 4
