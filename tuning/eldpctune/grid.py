# SPDX-License-Identifier: Apache-2.0
"""The fixed knob grid of the tuner and its deterministic refinement.

Knobs searched: ``ms_scaling_factor`` (alpha), ``max_iter``, and the decoder (BP+OSD-0, or BP alone).
The coarse ranges follow the presets the package ships:
alpha 0.25-0.75 covers the useful range (the stock setting is 0.625, the tuned presets use
0.3125-0.5), and the iteration caps 1-31 cover the cap curve, whose smallest useful caps are
typically 2-3 and whose reference point is 31.

The grid is fixed for a given mode; the refinement adds neighbours of the frontier.  Nothing here
depends on measured results except the refinement input, which is derived deterministically from the
accuracy stage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

#: Coarse alpha values (ms_scaling_factor).
COARSE_ALPHAS = (0.25, 0.3125, 0.375, 0.5, 0.625, 0.75)
#: Coarse iteration caps (max_iter).
COARSE_CAPS = (1, 2, 3, 5, 7, 11, 15, 31)
#: BP-only points are a subset: OSD-0 is what makes these codes decode, so BP alone is screened
#: where it has a chance (higher caps) and at the tuned alpha values.
BP_ALPHAS = (0.3125, 0.5, 0.625)
BP_CAPS = (3, 7, 15, 31)

#: Quick mode: a smoke test with small counts and a smaller grid.
QUICK_ALPHAS = (0.3125, 0.5, 0.625)
QUICK_CAPS = (1, 3, 7, 15)
QUICK_BP_ALPHAS = (0.3125, 0.5)
QUICK_BP_CAPS = (3, 15)
QUICK_REFINE_LIMIT = 4

#: Cap on the number of points a refinement stage may add per decoder kind.
REFINE_LIMIT = 16
#: Alpha offsets tried around the frontier (ms_scaling_factor is time-neutral, so a finer alpha
#: search costs accuracy shots only and can only find smaller qualifying caps).
REFINE_ALPHA_STEPS = (-0.0625, -0.03125, 0.03125, 0.0625)
ALPHA_MIN, ALPHA_MAX = 0.125, 0.875


@dataclass(frozen=True, order=True)
class Point:
    """One decoder configuration in the grid."""

    decoder: str   # "bposd" or "bp"
    alpha: float   # ms_scaling_factor
    max_iter: int

    def __post_init__(self):
        if self.decoder not in ("bposd", "bp"):
            raise ValueError(f"unknown decoder {self.decoder!r}")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")

    def config(self) -> dict:
        cfg = {
            "decoder": self.decoder,
            "bp_method": "ms",
            "schedule": "parallel",
            "ms_scaling_factor": float(self.alpha),
            "max_iter": int(self.max_iter),
        }
        if self.decoder == "bposd":
            cfg["osd_method"] = "osd0"
            cfg["osd_order"] = 0
        return cfg

    def label(self) -> str:
        return f"{self.decoder}:a={self.alpha:g}:c={self.max_iter}"

    def as_dict(self) -> dict:
        return {"decoder": self.decoder, "ms_scaling_factor": float(self.alpha),
                "max_iter": int(self.max_iter), "config": self.config(), "label": self.label()}


#: Stock: ldpc BP+OSD-0 at its common settings (min-sum alpha 0.625, 30 iterations), which is the
#: package preset ``ms30``.  Without --ler-target its screen LER is the default LER floor; it is
#: decoded on each screen stage's shots as a reference and is not a grid point.
STOCK_POINT = Point("bposd", 0.625, 30)
STOCK_LABEL = "ms30"


def coarse_points(quick: bool = False) -> list[Point]:
    """The fixed coarse grid, in evaluation order (decoder, alpha, cap ascending)."""
    alphas = QUICK_ALPHAS if quick else COARSE_ALPHAS
    caps = QUICK_CAPS if quick else COARSE_CAPS
    bp_alphas = QUICK_BP_ALPHAS if quick else BP_ALPHAS
    bp_caps = QUICK_BP_CAPS if quick else BP_CAPS
    points = [Point("bposd", a, c) for a in alphas for c in caps]
    points += [Point("bp", a, c) for a in bp_alphas for c in bp_caps]
    return sorted(points)


def refine_points(quick: bool, coarse_results: dict, already: set) -> list[Point]:
    """Deterministic refinement around the accuracy frontier.

    ``coarse_results`` maps Point -> {"status": "pass"|"fail"|"inconclusive", "upper": float},
    where ``upper`` is the worst (over physical error rates) upper confidence bound.  For each
    decoder kind:

    * if at least one point passes, the frontier cap is the smallest passing cap.  New points are
      the alpha neighbours (+-1/32, +-1/16, inside [0.125, 0.875]) of every alpha seen at that cap
      and at the cap just below it, evaluated at both caps (the cap below is the one that failed).
    * if nothing passes, the two caps with the smallest upper bound are refined the same way.

    Points already evaluated are never repeated.  The result is capped and sorted.
    """
    limit = QUICK_REFINE_LIMIT if quick else REFINE_LIMIT
    out: list[Point] = []
    for decoder in ("bposd", "bp"):
        kind = {p: r for p, r in coarse_results.items() if p.decoder == decoder}
        if not kind:
            continue
        passed = [(p.max_iter, p.alpha) for p, r in kind.items() if r["status"] == "pass"]
        if passed:
            cap_pass = min(c for c, _ in passed)
            alphas_pass = sorted({a for c, a in passed if c == cap_pass})
            cap_below = cap_pass - 1 if cap_pass > 1 else None
            alphas_below = sorted({p.alpha for p in kind if cap_below is not None
                                   and p.max_iter == cap_below})
            new_alphas = set()
            for a in alphas_pass + alphas_below:
                for step in REFINE_ALPHA_STEPS:
                    cand = round(a + step, 6)
                    if ALPHA_MIN <= cand <= ALPHA_MAX:
                        new_alphas.add(cand)
            caps = [cap_pass] + ([cap_below] if cap_below is not None else [])
        else:
            # Refine the two caps whose upper bounds are smallest at their own best alpha.
            best_by_cap: dict[int, tuple[float, float]] = {}
            for p, r in kind.items():
                cur = best_by_cap.get(p.max_iter)
                key = (r["upper"], p.alpha)
                if cur is None or key < (cur[0], cur[1]):
                    best_by_cap[p.max_iter] = (r["upper"], p.alpha)
            caps = sorted(best_by_cap, key=lambda c: (best_by_cap[c][0], c))[:2]
            new_alphas = set()
            for c in caps:
                a = best_by_cap[c][1]
                for step in REFINE_ALPHA_STEPS:
                    cand = round(a + step, 6)
                    if ALPHA_MIN <= cand <= ALPHA_MAX:
                        new_alphas.add(cand)
        kind_out = []
        for c in caps:
            for a in sorted(new_alphas):
                p = Point(decoder, a, c)
                if p not in already and p not in out + kind_out:
                    kind_out.append(p)
        out.extend(sorted(kind_out)[:limit])
    return sorted(out)


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def points_digest(points) -> str:
    """Identity of a list of grid points (order-insensitive): decoder, alpha and cap of each."""
    return _digest(sorted([p.decoder, float(p.alpha), int(p.max_iter)] for p in points))


def grid_digest(quick: bool) -> str:
    """Identity of the grid: the coarse point list and every constant of the refinement rule.

    A checkpoint made with a different grid has a different digest and is refused on resume.
    """
    return _digest({
        "coarse": sorted([p.decoder, float(p.alpha), int(p.max_iter)]
                         for p in coarse_points(quick)),
        "refine_limit": QUICK_REFINE_LIMIT if quick else REFINE_LIMIT,
        "refine_alpha_steps": [float(v) for v in REFINE_ALPHA_STEPS],
        "alpha_range": [float(ALPHA_MIN), float(ALPHA_MAX)],
    })
