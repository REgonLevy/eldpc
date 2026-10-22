# SPDX-License-Identifier: Apache-2.0
"""Accuracy stages: fresh shots, Clopper-Pearson bounds, sequential stopping.

Every stage samples its own shots from its own derived seed.  All points evaluated in one stage
decode exactly the same shot set (the store grows by fixed-size chunks, so the shot stream does not
depend on which point asks first).  A point stops as soon as a one-sided 99% Clopper-Pearson bound
decides it against the target: the upper bound below the target passes it, the lower bound above
the target rejects it.  Without a target, every point is measured to the stage's shot limit.

Each point also counts the calls whose status word has the converged flag, so its record carries
``converged_share``: the latency-stage dominance rule needs it (see :func:`dominator`).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import stats
from .problems import unpack_bits
from .grid import Point


#: The converged flag of the decoder status word (``eldpc.status.ST_BP_CONVERGED``, eldpc.h).
CONVERGED_FLAG = 0x0001


@dataclass
class StageSettings:
    max_shots: int
    chunk_shots: int
    target: float | None
    confidence: float = stats.CONFIDENCE

    def validated(self) -> "StageSettings":
        if self.max_shots < 1:
            raise ValueError("max_shots must be >= 1")
        if self.chunk_shots < 1:
            raise ValueError("chunk_shots must be >= 1")
        if self.target is not None and not (0.0 <= float(self.target) <= 1.0):
            raise ValueError("the LER target must be a probability in [0, 1]")
        return self


class ShotStore:
    """A growing, shared shot stream for one stage: fixed-size chunks, bit-packed rows."""

    def __init__(self, source, m: int, k: int, chunk_shots: int):
        self._source = source
        self._m = int(m)
        self._k = int(k)
        self._chunk = int(chunk_shots)
        self._parts: list[tuple[int, np.ndarray, np.ndarray | None]] = []
        self._shots = 0
        self.sample_seconds = 0.0

    @property
    def shots(self) -> int:
        return self._shots

    def ensure(self, shots: int) -> None:
        """Grow the stream in whole chunks until at least ``shots`` shots are available."""
        while self._shots < shots:
            take = min(self._chunk, shots - self._shots)
            t0 = time.perf_counter()
            dets, obs = self._source.sample(take)
            self.sample_seconds += time.perf_counter() - t0
            dets = np.asarray(dets, dtype=np.uint8)
            if dets.ndim != 2 or dets.shape[0] != take:
                raise RuntimeError(f"shot source returned detectors of shape {dets.shape} "
                                   f"for {take} shots")
            if obs is not None:
                obs = np.asarray(obs, dtype=np.uint8)
                if obs.ndim != 2 or obs.shape[0] != take:
                    raise RuntimeError(f"shot source returned observables of shape {obs.shape} "
                                       f"for {take} shots")
            self._parts.append((self._shots, dets, obs))
            self._shots += take

    def get(self, start: int, stop: int):
        """Unpacked (dets, obs) for shots [start, stop); obs is None when the problem has none."""
        if not (0 <= start <= stop <= self._shots):
            raise ValueError("shot range outside the store")
        det_parts = []
        obs_parts = []
        have_obs = True
        for part_start, dets, obs in self._parts:
            part_stop = part_start + dets.shape[0]
            if part_stop <= start or part_start >= stop:
                continue
            a = max(start, part_start) - part_start
            b = min(stop, part_stop) - part_start
            det_parts.append(dets[a:b])
            if obs is None:
                have_obs = False
            elif have_obs:
                obs_parts.append(obs[a:b])
        dets_out = np.concatenate(det_parts, axis=0) if det_parts else np.zeros((0, 0), np.uint8)
        dets_out = unpack_bits(dets_out, self._m)
        obs_out = None
        if have_obs and self._k > 0:
            obs_out = unpack_bits(np.concatenate(obs_parts, axis=0), self._k)
        return dets_out, obs_out


def decide(target: float | None, confidence: float, failures: int, shots: int,
           max_shots: int):
    """(status or None, upper bound, lower bound) for one look at the data."""
    upper = stats.clopper_pearson_upper(failures, shots, confidence)
    lower = stats.clopper_pearson_lower(failures, shots, confidence)
    status = None
    if target is not None:
        if upper < target:
            status = "pass"
        elif lower > target:
            status = "fail"
        elif shots >= max_shots:
            status = "inconclusive"
    elif shots >= max_shots:
        status = "measured"
    return status, upper, lower


class PointCaseRun:
    """Accuracy state of one point at one case, advanced one shot chunk at a time.

    The same class drives the sequential and the grouped concurrent evaluation, so both produce
    the same chunk records for the same shot stream.
    """

    def __init__(self, case: dict, settings: StageSettings):
        self.case = case
        self.settings = settings
        self.position = 0
        self.failures = 0
        self.converged = 0
        self.iteration_sum = 0.0
        self.decode_seconds = 0.0
        self.history: list[dict] = []
        self.status: str | None = None
        self.upper = 0.0
        self.lower = 0.0

    def apply(self, shots: int, failures: int, iterations: float, decode_seconds: float,
              converged: int = 0) -> None:
        self.position += int(shots)
        self.failures += int(failures)
        self.converged += int(converged)
        self.iteration_sum += float(iterations)
        self.decode_seconds += float(decode_seconds)
        status, self.upper, self.lower = decide(
            self.settings.target, self.settings.confidence, self.failures, self.position,
            self.settings.max_shots)
        self.history.append({
            "shots": self.position,
            "failures": self.failures,
            "converged": self.converged,
            "ler": self.failures / self.position,
            "cp99_lower": self.lower,
            "cp99_upper": self.upper,
            "decode_seconds": self.decode_seconds,
        })
        if status is not None:
            self.status = status

    def record(self, stage: str) -> dict:
        status = self.status
        if status is None:  # the shot limit was reached without a decision
            status = "inconclusive" if self.settings.target is not None else "measured"
            _s, self.upper, self.lower = decide(
                None, self.settings.confidence, self.failures, self.position,
                self.settings.max_shots)
        return {
            "stage": stage,
            "case": self.case["label"],
            "p": None if self.case["p"] is None else float(self.case["p"]),
            "status": status,
            "shots": self.position,
            "failures": self.failures,
            "converged_shots": self.converged,
            "converged_share": (self.converged / self.position if self.position else None),
            "ler": self.failures / self.position if self.position else None,
            "cp99_lower": self.lower,
            "cp99_upper": self.upper,
            "mean_iterations": self.iteration_sum / self.position if self.position else None,
            "decode_seconds": self.decode_seconds,
            "decode_us_per_shot": (1e6 * self.decode_seconds / self.position
                                   if self.position else None),
            "chunks": self.history,
        }


def evaluate_point(decoder, store: ShotStore, settings: StageSettings,
                   stage: str, case_label: str, p_value: float | None) -> dict:
    """Decode shared shots sequentially until a bound decides the point or the shot limit is hit."""
    run = PointCaseRun({"label": case_label, "p": p_value}, settings)
    while run.status is None and run.position < settings.max_shots:
        want = min(run.position + settings.chunk_shots, settings.max_shots)
        store.ensure(want)
        stop = min(want, store.shots)
        dets, obs = store.get(run.position, stop)
        if dets.shape[0] == 0:
            break
        t0 = time.perf_counter()
        _corrections, predicted, statuses = decoder.decode_batch(dets)
        seconds = time.perf_counter() - t0
        words = np.asarray(statuses, dtype=np.uint64)
        iterations = float(np.sum((words >> 16) & 0xFFFF))
        converged = int(np.count_nonzero(words & np.uint64(CONVERGED_FLAG)))
        failures = 0
        if decoder.num_observables:
            if obs is None:
                raise RuntimeError("the problem has observables but the shot source returned none")
            failures = int(np.count_nonzero(np.any(predicted != obs, axis=1)))
        run.apply(len(dets), failures, iterations, seconds, converged)
    return run.record(stage)


def combine_status(per_p: list[dict]) -> str:
    statuses = [r["status"] for r in per_p]
    if any(s == "fail" for s in statuses):
        return "fail"
    if statuses and all(s == "pass" for s in statuses):
        return "pass"
    if statuses and all(s == "measured" for s in statuses):
        return "measured"
    return "inconclusive"


def _assemble_record(point: Point, per_p: list[dict], create_seconds: float) -> dict:
    record = {
        "point": point.as_dict(),
        "label": point.label(),
        "status": combine_status(per_p),
        "create_seconds": create_seconds,
        "per_p": per_p,
    }
    record["shots"] = sum(r["shots"] for r in per_p)
    record["failures"] = sum(r["failures"] for r in per_p)
    record["decode_seconds"] = sum(r["decode_seconds"] for r in per_p)
    record["decode_us_per_shot"] = (1e6 * record["decode_seconds"] / record["shots"]
                                    if record["shots"] else None)
    record["mean_iterations"] = (sum(r["mean_iterations"] * r["shots"] for r in per_p) /
                                 record["shots"] if record["shots"] else None)
    record["cp99_upper_max"] = max(r["cp99_upper"] for r in per_p)
    record["cp99_lower_min"] = min(r["cp99_lower"] for r in per_p)
    shares = [r["converged_share"] for r in per_p if r.get("converged_share") is not None]
    record["converged_share"] = min(shares) if shares else None
    record["converged_shots"] = sum(int(r.get("converged_shots") or 0) for r in per_p)
    return record


def evaluate_point_all_p(make_decoder, point: Point, cases: list[dict], stores: dict,
                         settings: StageSettings, stage: str) -> dict:
    """Create one decoder handle per case (each physical error rate has its own matrices) and
    evaluate the point with the same knob settings on every case."""
    per_p = []
    create_seconds = 0.0
    for case in cases:
        t0 = time.perf_counter()
        decoder = make_decoder(point, case)
        create_seconds += time.perf_counter() - t0
        try:
            per_p.append(evaluate_point(decoder, stores[case["label"]], settings, stage,
                                        case["label"], case["p"]))
        finally:
            decoder.close()
    return _assemble_record(point, per_p, create_seconds)


def evaluate_group(make_decoder, points: list[Point], cases: list[dict], stores: dict,
                   settings: StageSettings, stage: str, workers: int) -> dict:
    """Evaluate a group of points concurrently, one shot chunk per round.

    Handles for the group are created in the same thread pool (handle creation is mostly
    host-side and overlaps well), then every still-undecided point decodes the same next shot
    chunk of every case.  The shot stores grow in exactly the same chunk sequence as in the
    sequential path and each point reads the same prefixes, so the chunk records and the accuracy
    decisions are identical to :func:`evaluate_point_all_p`; only the timing fields differ.
    """
    points = list(points)
    workers = max(1, int(workers))
    if not points:
        return {}
    keys = [(point, case) for point in points for case in cases]
    create_seconds = {point: 0.0 for point in points}
    handles: dict = {}
    states = {(point, case["label"]): PointCaseRun(case, settings)
              for point in points for case in cases}

    def create(point, case):
        t0 = time.perf_counter()
        decoder = make_decoder(point, case)
        return decoder, time.perf_counter() - t0

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [(point, case, pool.submit(create, point, case)) for point, case in keys]
        for point, case, future in futures:
            decoder, seconds = future.result()
            handles[(point, case["label"])] = decoder
            create_seconds[point] += seconds
        active = points
        while active:
            open_cases = [(point, case) for point in active for case in cases
                          if states[(point, case["label"])].status is None]
            if not open_cases:
                break
            position = states[(open_cases[0][0], open_cases[0][1]["label"])].position
            want = min(position + settings.chunk_shots, settings.max_shots)
            for case in cases:
                stores[case["label"]].ensure(want)

            def decode(point, case):
                run = states[(point, case["label"])]
                store = stores[case["label"]]
                stop = min(want, store.shots)
                dets, obs = store.get(run.position, stop)
                decoder = handles[(point, case["label"])]
                t0 = time.perf_counter()
                _corrections, predicted, statuses = decoder.decode_batch(dets)
                seconds = time.perf_counter() - t0
                words = np.asarray(statuses, dtype=np.uint64)
                iterations = float(np.sum((words >> 16) & 0xFFFF))
                converged = int(np.count_nonzero(words & np.uint64(CONVERGED_FLAG)))
                failures = 0
                if decoder.num_observables:
                    if obs is None:
                        raise RuntimeError("the problem has observables but the shot source "
                                           "returned none")
                    failures = int(np.count_nonzero(np.any(predicted != obs, axis=1)))
                return len(dets), failures, iterations, seconds, converged

            tasks = open_cases
            outcomes = list(pool.map(lambda key: decode(*key), tasks))
            for key, outcome in zip(tasks, outcomes):
                states[(key[0], key[1]["label"])].apply(*outcome)
            active = [point for point in active
                      if any(states[(point, case["label"])].status is None for case in cases)]
    finally:
        # Wait for every submitted chunk decode before closing the handles it uses.
        pool.shutdown(wait=True)
        for decoder in handles.values():
            decoder.close()
    records = {}
    for point in points:
        per_p = [states[(point, case["label"])].record(stage) for case in cases]
        records[point] = _assemble_record(point, per_p, create_seconds[point])
    return records


def slow_share_confident(record: dict, min_converged: float,
                         confidence: float = stats.CONFIDENCE) -> bool:
    """True when the point's non-converged share is below ``1 - min_converged`` with confidence.

    The one-sided Clopper-Pearson upper bound of the non-converged share must be below
    ``1 - min_converged`` at every case.  A record without per-case shot counts cannot prove it.
    """
    per_p = record.get("per_p") or []
    if not per_p:
        return False
    threshold = 1.0 - float(min_converged)
    for entry in per_p:
        shots = int(entry.get("shots") or 0)
        converged = entry.get("converged_shots")
        if shots < 1 or converged is None:
            return False
        slow = shots - int(converged)
        if not stats.clopper_pearson_upper(slow, shots, confidence) < threshold:
            return False
    return True


def dominator(point: Point, results: dict, min_converged: float | None = None,
              confidence: float = stats.CONFIDENCE):
    """An already-passing point with the same decoder and alpha and a smaller iteration cap.

    Under a latency budget at quantile q the larger caps of a row are skipped only when the passing
    point's non-converged share is below ``1 - q`` at the tuner's confidence (``min_converged = q``;
    p99 budget: 0.99, p50 only or no budget: 0.5).  The reason: while at least ``1 - q`` of the
    calls take the slow path, the q quantile of the pooled calls sits inside the slow class, and a
    larger cap that moves shots from OSD-0 into the converged class can be faster there.  With the
    rule satisfied the slow class is a small enough tail that the quantile is set by the converged
    calls, and raising the cap can only add BP work.

    ``min_converged=None`` keeps the structural rule alone (a lookup for callers that have no
    status words); the tuner always passes the quantile.
    """
    for other, record in results.items():
        if (other.decoder == point.decoder and other.alpha == point.alpha
                and other.max_iter < point.max_iter and record["status"] == "pass"):
            if min_converged is None or slow_share_confident(record, min_converged, confidence):
                return other
    return None
