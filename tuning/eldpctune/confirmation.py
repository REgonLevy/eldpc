# SPDX-License-Identifier: Apache-2.0
"""The fixed-size accuracy confirmation stage.

The screen stages stop a point at the first chunk whose one-sided Clopper-Pearson bound falls
below the target.  A passing point therefore stops with its bound just under the target, its
estimate is biased low, and choosing the fastest passing point favours the points whose early
chunks were luckiest.  The confirmation re-measures the leading candidates on a fixed number of
fresh shots, from a seed stream of its own, with no early stop, and the choice is made on those
numbers.

The shot count is declared before the first confirmation shot.  It comes from the target, the
confidence and a planned margin, by a fixed search (:func:`planned_shots`): N doubles from 64 until
a point whose true LER is ``target / 1.5`` passes (its upper bound is below the target) with
probability at least 0.9, then a bisection below that N returns an N at which the probability is
at least 0.9 and N - 1's is below it.  The pass probability is not monotone in N (it drops each
time the largest passing failure count steps up), so N is not necessarily the smallest shot count
that meets the margin; it is the deterministic result of that search.  ``--quick`` caps it at a
small value; ``--confirm-shots`` overrides it.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import binom

from . import accuracy, stats
from .problems import unpack_bits

#: The true LER, as a fraction of the target, that the shot-count rule plans for.
MARGIN = 1.5
#: The probability with which that point should pass at the planned shot count.
PASS_PROBABILITY = 0.9
#: ``--quick`` never plans more confirmation shots than this.
QUICK_MAX_SHOTS = 2048
#: Shot counts when there is no LER target (the confirmation only reports the measured LER).
NO_TARGET_SHOTS = {True: 512, False: 2048}
#: Largest confirmation batch decoded in one call (memory of the unpacked syndromes).
CHUNK_SHOTS = 4096


def pass_probability(shots: int, target: float, confidence: float,
                     true_rate: float) -> float:
    """P(a point with true LER ``true_rate`` passes) with ``shots`` fixed shots.

    A point passes when the one-sided Clopper-Pearson upper bound at ``confidence`` is below
    ``target``, which holds exactly when ``P(X <= failures | shots, target) < 1 - confidence``.
    """
    shots = int(shots)
    alpha = 1.0 - float(confidence)
    lo, hi = 0, shots
    if binom.cdf(0, shots, target) >= alpha:
        return 0.0
    while lo < hi:  # the largest failure count that still passes
        mid = (lo + hi + 1) // 2
        if binom.cdf(mid, shots, target) < alpha:
            lo = mid
        else:
            hi = mid - 1
    return float(binom.cdf(lo, shots, true_rate))


def planned_shots(target: float | None, confidence: float, quick: bool = False,
                  override: int | None = None) -> tuple[int, str]:
    """``(N, rule)`` for the confirmation: N shots per candidate and case and how N was chosen.

    With a target, N doubles from 64 until :func:`pass_probability` at ``target / MARGIN`` reaches
    ``PASS_PROBABILITY``, and a bisection over ``[1, that N]`` then returns an N whose probability
    is at least ``PASS_PROBABILITY`` while N - 1's is below it.  Because the probability is not
    monotone in N, a smaller N can also meet the margin (target 0.05 at confidence 0.99 gives
    N = 1944 although 1899 also does): N is the result of this search, not the smallest such N.
    """
    if override is not None:
        n = int(override)
        if n < 1:
            raise ValueError("--confirm-shots must be >= 1")
        return n, (f"--confirm-shots {n}: fixed by the command line, not derived from the target")
    if target is None:
        n = NO_TARGET_SHOTS[bool(quick)]
        return n, (f"no --ler-target: a fixed {n} shots per candidate and case")
    true_rate = float(target) / MARGIN
    n = 64
    while n < (1 << 31) and pass_probability(n, target, confidence, true_rate) < PASS_PROBABILITY:
        n *= 2
    lo, hi = 1, n
    # Bisection keeps pass_probability(hi) >= 0.9 and moves lo past every N tested below 0.9, so
    # it ends at an N with probability >= 0.9 whose predecessor is below 0.9.  The probability is
    # not monotone in N, so this is a crossing found by the search, not necessarily the first one.
    while lo < hi:
        mid = (lo + hi) // 2
        if pass_probability(mid, target, confidence, true_rate) >= PASS_PROBABILITY:
            hi = mid
        else:
            lo = mid + 1
    n = lo
    rule = (f"N doubles from 64 until a point whose true LER is target/{MARGIN:g} "
            f"({true_rate:.6g}) has its one-sided {confidence:.4g} upper bound below the target "
            f"{float(target):.6g} with probability >= {PASS_PROBABILITY:g}, then bisection returns "
            f"an N with probability >= {PASS_PROBABILITY:g} whose N - 1 is below it; the "
            f"probability is not monotone in N, so a smaller N can also meet the margin")
    if quick and n > QUICK_MAX_SHOTS:
        rule += f"; --quick caps N at {QUICK_MAX_SHOTS}"
        n = QUICK_MAX_SHOTS
    return int(n), rule


def run_accuracy_confirmation(make_decoder, candidates, cases, case_seeds: dict,
                              n_shots: int, confidence: float, log=print,
                              progress: dict | None = None,
                              save_progress=None) -> list[dict]:
    """Measure every candidate on the same fixed N fresh shots per case.

    ``candidates`` is the ordered list of points (the choice's first candidate first);
    ``case_seeds`` maps a case label to the confirmation seed of that case.  Every candidate
    decodes the same shot stream of a case (the sampler is re-created from the same seed), so the
    candidates are compared on the same shots.  No point stops early.

    ``progress`` is the saved state of an interrupted run -- per candidate and case the number of
    shots already decoded, their failures and their converged calls -- and ``save_progress(record)``
    persists the updated state after every decoded chunk.  A candidate/case whose saved count is
    the full N is not decoded again; a partial one re-creates the seeded source and samples away
    the decoded prefix, so the resumed counts are exactly those of an uninterrupted run.  The state
    is reused only when its planned shot count and case seeds match this call.
    """
    n_shots = int(n_shots)
    if n_shots < 1:
        raise ValueError("the confirmation shot count must be >= 1")
    chunk = max(1, min(int(CHUNK_SHOTS), n_shots))
    state = {
        "planned_shots": n_shots,
        "confidence": float(confidence),
        "cases": [{"label": case["label"], "seed": int(case_seeds[case["label"]])}
                  for case in cases],
        "candidates": {},
        "complete": False,
    }
    if progress is not None and int(progress.get("planned_shots", -1)) == n_shots:
        stored_cases = {entry.get("label"): int(entry.get("seed", -1))
                        for entry in progress.get("cases") or []}
        if stored_cases == {case["label"]: int(case_seeds[case["label"]]) for case in cases}:
            state["candidates"] = {
                label: {case_label: dict(values) for case_label, values in per_case.items()}
                for label, per_case in (progress.get("candidates") or {}).items()}
    records = []
    for rank, (point, screen) in enumerate(candidates, 1):
        label = point.label()
        per_case_state = state["candidates"].setdefault(label, {})
        per_p = []
        started = time.perf_counter()
        resumed_shots = sum(
            min(max(0, int((per_case_state.get(case["label"]) or {}).get("shots", 0))), n_shots)
            for case in cases)
        for case in cases:
            problem = case["problem"]
            case_label = case["label"]
            seed = int(case_seeds[case_label])
            saved = per_case_state.get(case_label) or {}
            done = min(max(0, int(saved.get("shots", 0))), n_shots)
            failures = int(saved.get("failures", 0)) if done else 0
            converged = int(saved.get("converged", 0)) if done else 0
            if done < n_shots:
                source = problem.shot_source(seed)
                remaining = done  # advance the seeded stream past the decoded prefix
                while remaining > 0:
                    take = min(chunk, remaining)
                    source.sample(take)
                    remaining -= take
                decoder = make_decoder(point, case)
                try:
                    while done < n_shots:
                        take = min(chunk, n_shots - done)
                        dets_packed, obs_packed = source.sample(take)
                        dets = unpack_bits(np.asarray(dets_packed, dtype=np.uint8), problem.m)
                        obs = (unpack_bits(np.asarray(obs_packed, dtype=np.uint8), problem.k)
                               if obs_packed is not None and problem.k > 0 else None)
                        _corr, predicted, statuses = decoder.decode_batch(dets)
                        words = np.asarray(statuses, dtype=np.uint64)
                        converged += int(np.count_nonzero(
                            words & np.uint64(accuracy.CONVERGED_FLAG)))
                        if decoder.num_observables:
                            if obs is None:
                                raise RuntimeError("the problem has observables but the shot "
                                                   "source returned none")
                            failures += int(np.count_nonzero(np.any(predicted != obs, axis=1)))
                        done += take
                        per_case_state[case_label] = {"shots": int(done),
                                                      "failures": int(failures),
                                                      "converged": int(converged)}
                        if save_progress is not None:
                            save_progress(state)
                finally:
                    decoder.close()
            per_case_state[case_label] = {"shots": int(done), "failures": int(failures),
                                          "converged": int(converged)}
            upper = stats.clopper_pearson_upper(failures, n_shots, confidence)
            lower = stats.clopper_pearson_lower(failures, n_shots, confidence)
            per_p.append({
                "label": case["label"], "p": case["p"], "shots": n_shots,
                "failures": int(failures), "ler": failures / n_shots,
                "lower": lower, "upper": upper,
                "converged_shots": int(converged),
                "converged_share": converged / n_shots,
            })
        worst_ler = max(entry["ler"] for entry in per_p)
        worst_upper = max(entry["upper"] for entry in per_p)
        records.append({
            "label": point.label(), "point": point.as_dict(), "rank": rank,
            "screen": screen, "per_p": per_p,
            "shots": sum(entry["shots"] for entry in per_p),
            "failures": sum(entry["failures"] for entry in per_p),
            "ler": worst_ler, "upper": worst_upper,
            "resumed_case_shots": int(resumed_shots),
            "wall_seconds": time.perf_counter() - started,
        })
        if resumed_shots:
            log(f"[confirm] {label}: resumed with {resumed_shots} of "
                f"{n_shots * len(cases)} case-shots already decoded")
        log(f"[confirm] {point.label()}: {records[-1]['failures']}/"
            f"{records[-1]['shots']} failures, worst upper {worst_upper:.3g}")
    state["complete"] = all(
        all(int((state["candidates"].get(point.label(), {}).get(case["label"]) or {})
                .get("shots", 0)) >= n_shots
            for case in cases)
        for point, _screen in candidates)
    if save_progress is not None:
        save_progress(state)
    return records
