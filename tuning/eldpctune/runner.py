# SPDX-License-Identifier: Apache-2.0
"""The fixed tuning procedure, end to end.

1. coarse grid over (alpha, max_iter) and the decoder, accuracy first, sequential Clopper-Pearson
   stopping on freshly sampled shots, with the converged-share dominance rule;
2. a refinement stage around the accuracy frontier with a fresh shot set;
3. latency of every surviving point, single-shot through the public API, paired against the
   current best in an ABBA schedule with an A/A band, at the requested call cadence, every
   measurement on its own freshly sampled syndromes and split into converged and slow calls;
4. budget membership: every survivor whose latency-stage measurement is within
   :data:`RETIME_MARGIN` of a budget is re-timed with the confirmation design before the in/out
   decision (both paths);
5. a fixed-size accuracy confirmation on fresh shots (no early stop) and a latency confirmation of
   the in-budget candidates, one at a time in the choice's order, until one is accepted or the
   in-budget list is exhausted (``--confirm-candidates`` caps the count);
6. the choice: the first accepted candidate, then the Pareto table;
7. the output files (written by :mod:`eldpctune.report`).

Without --ler-target (the weight and fastest modes) a point is also choosable only when its screen
LER is no worse than stock ``ms30`` decoded on the same screen shots (or at most --max-ler), a
candidate is accepted when its own latency confirmation is within the budget, and the reported LER
is the screen's fixed-size estimate.  With --ler-target a candidate is accepted when its accuracy
confirmation's upper bound is below the target at every case and its latency confirmation is
within the budget, and the reported LER is the confirmation's.

The accuracy path is deterministic: the same --seed gives the same shots, the same failures and the
same accuracy choices, and the confirmation's seeds derive from --seed with their own stage name.
Latency is a measurement of the machine and is not bit-reproducible.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import eldpc
import numpy as np

from . import __version__, accuracy, confirmation, env, grid, latency, problems, stats
from .report import write_json


@dataclass
class TuneConfig:
    problem: str | None = None
    noise_model: str | None = None
    dem: str | None = None
    circuit: str | None = None
    h: str | None = None
    o: str | None = None
    ps: list[float] = field(default_factory=list)
    rounds: int | None = None
    p50_budget_us: float | None = None
    p99_budget_us: float | None = None
    ler_target: float | None = None
    max_ler: float | None = None
    weight: float | None = None
    cadence_us: float = 0.0
    seed: int = 20261006
    out: str = ""
    auto: bool = False
    quick: bool = False
    max_shots: int | None = None
    workers: int = 1
    resume: bool = False
    confidence: float = stats.CONFIDENCE
    confirm_shots: int | None = None
    confirm_candidates: int | None = None
    no_prune: bool = False
    decompose: bool = False

    def validate(self) -> "TuneConfig":
        given = [name for name in ("problem", "dem", "circuit", "h")
                 if getattr(self, name)]
        if len(given) != 1:
            raise ValueError("give exactly one of --problem, --dem, --circuit or --h")
        if self.h and not self.ps:
            raise ValueError("--h needs --p (the code-capacity error rate)")
        if self.ps and not (self.h or self.problem):
            raise ValueError("--p applies to --problem (regenerated at that rate) and to --h")
        if self.rounds is not None and not self.problem:
            raise ValueError("--rounds applies to --problem (the built-in recipes)")
        if self.decompose and not (self.problem or self.circuit):
            raise ValueError("--decompose applies to --circuit and --problem, whose DEM this tool "
                             "derives; --dem FILE is used as it is and --h has no DEM")
        if self.noise_model is not None:
            if not self.problem:
                raise ValueError("--noise-model applies to --problem (the built-in recipes)")
            if self.noise_model not in ("uniform", "si1000"):
                raise ValueError("--noise-model must be 'uniform' or 'si1000'")
            spec = problems.PROBLEMS.get(self.problem)
            if spec is not None and spec["family"] != "surface" and self.noise_model != "uniform":
                raise ValueError(f"{self.problem} uses uniform depolarizing noise; its noise "
                                 f"model cannot be overridden")
        if self.o is not None and not self.h:
            raise ValueError("--o supplies the observable matrix of --h")
        if self.ps and any(not (0.0 < float(p) < 1.0) for p in self.ps):
            raise ValueError("every --p must be in (0, 1)")
        if len({float(p) for p in self.ps}) != len(self.ps):
            raise ValueError("--p values must be distinct")
        if self.ler_target is not None and not (0.0 < float(self.ler_target) < 1.0):
            raise ValueError("--ler-target must be in (0, 1)")
        if self.max_ler is not None:
            if self.ler_target is not None:
                raise ValueError("--max-ler replaces the default LER floor of a run without "
                                 "--ler-target; with --ler-target the target is the bar")
            if not (0.0 < float(self.max_ler) <= 1.0):
                raise ValueError("--max-ler must be in (0, 1]")
        if self.p50_budget_us is not None and self.p50_budget_us <= 0:
            raise ValueError("--p50-budget-us must be > 0")
        if self.p99_budget_us is not None and self.p99_budget_us <= 0:
            raise ValueError("--p99-budget-us must be > 0")
        if self.cadence_us < 0:
            raise ValueError("--cadence-us must be >= 0")
        if self.weight is not None and self.weight < 0:
            raise ValueError("--weight must be >= 0")
        if self.rounds is not None and self.rounds < 1:
            raise ValueError("--rounds must be >= 1")
        if self.max_shots is not None and self.max_shots < 1:
            raise ValueError("--max-shots must be >= 1")
        if self.workers < 1:
            raise ValueError("--workers must be >= 1")
        if self.confirm_shots is not None and self.confirm_shots < 1:
            raise ValueError("--confirm-shots must be >= 1")
        if self.confirm_candidates is not None and self.confirm_candidates < 1:
            raise ValueError("--confirm-candidates must be >= 1")
        if not (0.0 < float(self.confidence) < 1.0):
            raise ValueError("the confidence must be in (0, 1)")
        if not self.out:
            raise ValueError("--out DIR is required")
        return self


def pruning_quantile(cfg: TuneConfig) -> float:
    """The latency quantile the pruning rule protects (p99 budget: 0.99, else 0.5)."""
    return 0.99 if cfg.p99_budget_us is not None else 0.5


def pruning_record(cfg: TuneConfig) -> dict:
    """The results.json "pruning" block: the threshold, the quantile and the rule."""
    q = pruning_quantile(cfg)
    enabled = not cfg.no_prune
    rule = (
        "a point of a (decoder, alpha) row is skipped as dominated by a passing point of the same "
        "row with a smaller max_iter only when the one-sided Clopper-Pearson upper bound of the "
        "passing point's non-converged share, at the tuner's confidence, is below 1 - min_converged "
        "(it is then a tail small enough that the managed latency quantile is set by the converged "
        "calls, and raising the cap can only add BP work); while at least 1 - min_converged of the "
        "calls take the slow path the quantile sits in the slow class, and a larger cap that moves "
        "shots from OSD-0 into the converged class can be faster there")
    if not enabled:
        rule = "--no-prune: no point is skipped; every point of every row is evaluated.  " + rule
    return {"min_converged": float(q),
            "quantile": "p99" if cfg.p99_budget_us is not None else "p50",
            "confidence": float(cfg.confidence),
            "enabled": bool(enabled),
            "rule": rule}


def run_fingerprint(cfg: TuneConfig, cases: list[dict],
                    settings: accuracy.StageSettings) -> str:
    """Identity of everything that determines the accuracy results.

    The thread count is excluded: concurrency does not change a shot record.  The problem keys carry the
    content digest of a DEM/circuit/H input, so an edited input file changes the fingerprint, and the
    DEM digest covers the DEM options (decomposed or not).  The fingerprint also covers the tuner
    version, a digest of the grid itself (the coarse point list and the constants of the refinement
    rule) and the pruning rule, so a checkpoint made with another grid, another pruning rule or
    another tuner version is refused.  Each stage additionally records the digest of its own point
    list.
    """
    payload = {
        "tuner_version": __version__,
        "grid_digest": grid.grid_digest(bool(cfg.quick)),
        "seed": int(cfg.seed),
        "quick": bool(cfg.quick),
        "decompose": bool(cfg.decompose),
        "confirm_shots": None if cfg.confirm_shots is None else int(cfg.confirm_shots),
        "pruning": pruning_record(cfg),
        "cases": [{"label": case["label"], "p": case["p"], "key": case["seed_key"],
                   "m": case["problem"].m, "n": case["problem"].n, "k": case["problem"].k,
                   "dem_sha256": case["problem"].dem_sha256()}
                  for case in cases],
        "settings": {"max_shots": settings.max_shots, "chunk_shots": settings.chunk_shots,
                     "target": settings.target, "confidence": settings.confidence},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()


class Checkpoint:
    """Incremental progress of the accuracy stages, written after every group.

    A run always writes ``raw/checkpoint.json``; ``--resume`` makes the next run load it instead of
    starting from the beginning.  Only complete stages and points are reused; a partially evaluated
    group is simply evaluated again (its records are deterministic).  Latency is always measured
    again, because it describes the machine at the time of the run, not the computation.
    """

    SCHEMA = "eldpc.tune.checkpoint.v1"

    def __init__(self, path: str, fingerprint: str):
        self.path = path
        self.data = {"schema": self.SCHEMA, "fingerprint": fingerprint, "stages": {}}

    @classmethod
    def load(cls, path: str, fingerprint: str) -> "Checkpoint":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema") != cls.SCHEMA:
            raise ValueError(f"{path}: not an eldpc-tune checkpoint")
        if data.get("fingerprint") != fingerprint:
            raise ValueError(f"{path}: the checkpoint belongs to a different problem, seed, shot "
                             f"budget, grid or tuner version; rerun without --resume or use "
                             f"another --out")
        checkpoint = cls(path, fingerprint)
        checkpoint.data = data
        return checkpoint

    def saved_stage(self, name: str) -> dict | None:
        stage = self.data.get("stages", {}).get(name)
        if not stage or not stage.get("points"):
            return None
        return stage

    def save_stage(self, stage: dict, complete: bool) -> None:
        self.data.setdefault("stages", {})[stage["name"]] = {
            "points_digest": stage.get("points_digest"),
            "settings": stage["settings"],
            "cases": stage["cases"],
            "points": stage["points"],
            "references": stage.get("references") or [],
            "complete": bool(complete),
        }
        # Replace atomically, so an interruption never leaves a truncated checkpoint behind.
        temporary = self.path + ".tmp"
        write_json(temporary, self.data)
        os.replace(temporary, self.path)


class ConfirmationCheckpoint:
    """Incremental progress of the fixed-shot accuracy confirmation.

    A separate, small file: the confirmation saves after every decoded chunk, while the stage
    checkpoint holds every point's chunk history and is too large to rewrite that often.  The
    record is the one :func:`eldpctune.confirmation.run_accuracy_confirmation` keeps: the planned
    shot count, the case seeds and per candidate and case the shots, failures and converged calls
    already counted.  ``--resume`` continues from it up to the last complete chunk; a run with a
    different fingerprint or confirmation shot count is refused.
    """

    SCHEMA = "eldpc.tune.confirmation-checkpoint.v1"

    def __init__(self, path: str, fingerprint: str, planned_shots: int):
        self.path = path
        self.fingerprint = fingerprint
        self.planned_shots = int(planned_shots)
        self.data = {"schema": self.SCHEMA, "fingerprint": fingerprint,
                     "planned_shots": int(planned_shots), "cases": [],
                     "candidates": {}, "complete": False}

    @classmethod
    def load(cls, path: str, fingerprint: str, planned_shots: int) -> "ConfirmationCheckpoint":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema") != cls.SCHEMA:
            raise ValueError(f"{path}: not an eldpc-tune confirmation checkpoint")
        if data.get("fingerprint") != fingerprint:
            raise ValueError(f"{path}: the confirmation checkpoint belongs to a different "
                             f"problem, seed, shot budget, confirmation shot count, grid or tuner "
                             f"version; rerun without --resume or use another --out")
        if int(data.get("planned_shots", -1)) != int(planned_shots):
            raise ValueError(f"{path}: the confirmation checkpoint was made with "
                             f"{data.get('planned_shots')} shots per candidate and case, this run "
                             f"plans {int(planned_shots)}")
        checkpoint = cls(path, fingerprint, planned_shots)
        checkpoint.data = data
        return checkpoint

    def progress(self) -> dict | None:
        """The saved per-candidate state, or None when nothing has been decoded yet."""
        if not self.data.get("candidates"):
            return None
        return {"planned_shots": int(self.data.get("planned_shots", self.planned_shots)),
                "cases": list(self.data.get("cases") or []),
                "candidates": self.data["candidates"]}

    def save(self, record: dict) -> None:
        self.data = {"schema": self.SCHEMA, "fingerprint": self.fingerprint,
                     "planned_shots": int(self.planned_shots), **record}
        temporary = self.path + ".tmp"
        write_json(temporary, self.data)
        os.replace(temporary, self.path)


def _record_point(record: dict) -> grid.Point:
    point = record["point"]
    return grid.Point(point["decoder"], point["ms_scaling_factor"], point["max_iter"])


def build_cases(cfg: TuneConfig) -> list[dict]:
    """One case per physical error rate (or one case for a DEM/circuit file)."""
    if cfg.problem:
        ps = [float(p) for p in (cfg.ps or [1e-3])]
        cases = []
        for p in ps:
            prob = problems.build_problem(cfg.problem, p, cfg.rounds, cfg.noise_model,
                                          decompose=cfg.decompose)
            cases.append({"label": f"p={p:g}", "p": p, "problem": prob,
                          "seed_key": prob.key})
        return cases
    if cfg.dem:
        prob = problems.problem_from_dem_file(cfg.dem)
        return [{"label": "dem", "p": None, "problem": prob, "seed_key": prob.key}]
    if cfg.circuit:
        prob = problems.problem_from_circuit_file(cfg.circuit, decompose=cfg.decompose)
        return [{"label": "circuit", "p": None, "problem": prob, "seed_key": prob.key}]
    H = problems.read_matrix_file(cfg.h)
    if cfg.o is not None:
        O = problems.read_matrix_file(cfg.o)
    else:
        O = problems.logical_observables_from_h(H)
    cases = []
    for p in (float(x) for x in cfg.ps):
        prob = problems.problem_from_h_arrays(
            H, O, p, meta={"path": cfg.h, "observables_derived": cfg.o is None})
        cases.append({"label": f"p={p:g}", "p": p, "problem": prob, "seed_key": prob.key})
    return cases


def zero_failure_floor(shots: int, confidence: float = stats.CONFIDENCE) -> float:
    """The smallest upper bound provable with ``shots`` shots and zero failures."""
    return stats.clopper_pearson_upper(0, int(shots), confidence)


def auto_max_shots(cfg: TuneConfig) -> int | None:
    """A target-derived shot limit, or None when the default applies.

    The zero-failure upper bound of the default limit is about 3.5e-5; a target below that can
    never be proven, so for full runs the limit grows to about 4.7/target (a small margin above
    the exact 4.605/target).  --quick keeps its small smoke-test limit and --max-shots overrides.
    """
    if cfg.max_shots is not None or cfg.ler_target is None or cfg.quick:
        return None
    base = 131072
    needed = int(math.ceil(4.7 / float(cfg.ler_target)))
    return max(base, needed)


def stage_settings(cfg: TuneConfig) -> accuracy.StageSettings:
    automatic = auto_max_shots(cfg)
    if cfg.max_shots is not None:
        max_shots = int(cfg.max_shots)
    elif automatic is not None:
        max_shots = automatic
    else:
        max_shots = 2048 if cfg.quick else 131072
    chunk = 512 if cfg.quick else 2048
    chunk = min(chunk, max_shots)
    return accuracy.StageSettings(max_shots=max_shots, chunk_shots=chunk,
                                  target=None if cfg.ler_target is None else float(cfg.ler_target),
                                  confidence=cfg.confidence).validated()


def _make_decoder(point: grid.Point, case: dict):
    prob = case["problem"]
    return eldpc.Decoder(prob.H, prob.O, prob.priors, config=point.config())


def run_accuracy_stage(stage_name: str, points: list[grid.Point], cases: list[dict],
                       settings: accuracy.StageSettings, cfg: TuneConfig, log,
                       prior_results: dict | None = None, workers: int = 1,
                       checkpoint: "Checkpoint | None" = None,
                       pruning: dict | None = None,
                       reference_points: list[grid.Point] | None = None) -> dict:
    """One accuracy stage: a shared shot stream per case, a fresh stage seed per case.

    ``pruning`` is the results.json pruning block (or None for the structural rule alone).  The
    dominance lookup then also requires the passing point's non-converged share to be below
    ``1 - min_converged`` at the tuner's confidence.

    ``reference_points`` (the LER floor's stock point) are decoded on the same shots after the
    stage's points and recorded in ``stage["references"]``, not among the points: they are not
    candidates, take no part in the refinement and are not latency-measured.  A reference that is
    also a stage point reuses that point's record; a resumed stage reuses a saved reference.
    """
    stores = {}
    case_records = []
    for case in cases:
        seed = stats.derive_seed(cfg.seed, stage_name, case["seed_key"])
        problem = case["problem"]
        source = problem.shot_source(seed)
        stores[case["label"]] = accuracy.ShotStore(source, problem.m, problem.k,
                                                   settings.chunk_shots)
        case_records.append({"label": case["label"], "p": case["p"], "seed": seed,
                             "problem": problem.summary()})
    stage = {"name": stage_name,
             "settings": {"max_shots": settings.max_shots,
                          "chunk_shots": settings.chunk_shots,
                          "cp_confidence": settings.confidence,
                          "ler_target": settings.target},
             "cases": case_records, "points": []}
    results: dict[grid.Point, dict] = {}
    known = dict(prior_results or {})
    workers = max(1, int(workers))
    # Cap-major order: within a (decoder, alpha) row the caps still ascend, so every dominance
    # decision sees the row's smaller caps first, while the rows of one cap level are adjacent
    # and can fill a concurrent group with distinct rows.
    order = sorted(points, key=lambda p: (p.max_iter, p.decoder, p.alpha))
    stage["points_digest"] = grid.points_digest(order)
    completed: dict[grid.Point, dict] = {}
    prior_sample_seconds: dict[str, float] = {}
    saved = checkpoint.saved_stage(stage_name) if checkpoint is not None else None
    if saved is not None:
        if saved.get("points_digest") != stage["points_digest"]:
            raise ValueError(f"the checkpoint's {stage_name} stage was made with a different "
                             f"point list (another grid or tuner version); rerun without "
                             f"--resume or use another --out")
        prior_sample_seconds = {c["label"]: float(c.get("sample_seconds") or 0.0)
                                for c in saved.get("cases", [])}
        for record in saved["points"]:
            completed[_record_point(record)] = record
        resumed = 0
        for point in order:
            if point in completed:
                stage["points"].append(completed[point])
                results[point] = completed[point]
                resumed += 1
        log(f"[{stage_name}] resuming: {resumed} of {len(order)} points already recorded")
        stage["resumed_points"] = resumed
    started = time.perf_counter()

    def update_case_counters() -> None:
        # shots_sampled is the length of the shot stream the recorded decisions used: the larger of
        # what this run sampled and what the recorded (possibly resumed) points decoded.
        # sample_seconds accumulates over the interrupted runs; the *_this_run fields are this run.
        for record in case_records:
            store = stores[record["label"]]
            decoded = [entry["shots"] for point_record in stage["points"]
                       for entry in point_record.get("per_p", [])
                       if entry["case"] == record["label"]]
            record["shots_sampled"] = max([store.shots] + decoded)
            record["shots_sampled_this_run"] = store.shots
            record["sample_seconds"] = (prior_sample_seconds.get(record["label"], 0.0)
                                        + store.sample_seconds)
            record["sample_seconds_this_run"] = store.sample_seconds
    index = 0
    while index < len(order):
        group: list[grid.Point] = []
        alpha_rows: set = set()
        # A group never holds two points of the same (decoder, alpha) row, so every dominance
        # decision is made against points of earlier groups -- exactly as in a sequential run,
        # whose skipped set is therefore reproduced.
        while index < len(order) and len(group) < workers:
            point = order[index]
            alpha_row = (point.decoder, point.alpha)
            if alpha_row in alpha_rows:
                break
            index += 1
            if point in completed:
                continue
            if settings.target is not None and (pruning is None or pruning.get("enabled", True)):
                lookup = {}
                if pruning is None:
                    lookup = {}
                else:
                    lookup = {"min_converged": float(pruning["min_converged"]),
                              "confidence": float(pruning.get("confidence", cfg.confidence))}
                dominant = (accuracy.dominator(point, known, **lookup)
                            or accuracy.dominator(point, results, **lookup))
                if dominant is not None:
                    dominator_record = known.get(dominant) or results.get(dominant) or {}
                    record = {"point": point.as_dict(), "label": point.label(),
                              "status": "skipped", "dominated_by": dominant.label(),
                              "dominator_converged_share":
                                  dominator_record.get("converged_share"),
                              "reason": "a passing point with the same decoder and alpha has a "
                                        "smaller max_iter and its non-converged share is below "
                                        "1 - min_converged at the tuner's confidence, so the "
                                        "latency quantile is set by the converged calls and the "
                                        "larger cap can only add BP work (dominance pruning, see "
                                        "README); it is skipped"}
                    stage["points"].append(record)
                    results[point] = record
                    log(f"[{stage_name}] {point.label()}: skipped "
                        f"(dominated by {dominant.label()})")
                    continue
            alpha_rows.add(alpha_row)
            group.append(point)
        if not group:
            continue
        if len(group) == 1 and workers == 1:
            t0 = time.perf_counter()
            record = accuracy.evaluate_point_all_p(_make_decoder, group[0], cases, stores,
                                                   settings, stage_name)
            record["wall_seconds"] = time.perf_counter() - t0
            records = {group[0]: record}
        else:
            t0 = time.perf_counter()
            records = accuracy.evaluate_group(_make_decoder, group, cases, stores, settings,
                                              stage_name, workers)
            wall = time.perf_counter() - t0
            for point in group:
                records[point]["wall_seconds"] = wall
                records[point]["group_size"] = len(group)
        for point in group:
            record = records[point]
            stage["points"].append(record)
            results[point] = record
            log(f"[{stage_name}] {point.label()}: {record['status']} "
                f"({record['failures']}/{record['shots']} failures, "
                f"upper {record['cp99_upper_max']:.3g}, {record['wall_seconds']:.1f} s)")
        if checkpoint is not None:
            update_case_counters()
            checkpoint.save_stage(stage, complete=False)
    if reference_points:
        saved_references = {record.get("label"): record
                            for record in ((saved or {}).get("references") or [])}
        references = []
        for point in reference_points:
            if point in results:
                record = dict(results[point])
                record["same_as_stage_point"] = True
            elif point.label() in saved_references:
                record = saved_references[point.label()]
            else:
                t0 = time.perf_counter()
                record = accuracy.evaluate_point_all_p(_make_decoder, point, cases, stores,
                                                       settings, stage_name)
                record["wall_seconds"] = time.perf_counter() - t0
            record["role"] = "ler_floor_reference"
            references.append(record)
            log(f"[{stage_name}] reference {point.label()} ({grid.STOCK_LABEL}): "
                f"{record['failures']}/{record['shots']} failures on the stage's shots")
        stage["references"] = references
    stage["wall_seconds"] = time.perf_counter() - started
    update_case_counters()
    if checkpoint is not None:
        checkpoint.save_stage(stage, complete=True)
    return stage


def _survivors(results: dict, target_present: bool) -> list[grid.Point]:
    """The screen survivors: passing points (or, with no target, measured points), minus every
    point that a recorded LER floor verdict excludes (only runs without --ler-target have one)."""
    ok = ("pass",) if target_present else ("pass", "measured")
    return [p for p, r in results.items()
            if r["status"] in ok and (r.get("ler_floor") or {}).get("passed", True)]


def _aggregate_latency(per_case: list[dict], side: str) -> dict:
    """Worst case over physical error rates, for every reported statistic.

    The slow-class share is the largest over the cases (the worst case, like the quantiles), and
    ``p99_unreliable`` is computed from that share so the two fields always agree.
    """
    entries = [entry[side] for entry in per_case]
    slow_fractions = [e.get("slow_fraction") for e in entries]
    slow_fractions = [float(x) for x in slow_fractions if x is not None]
    fraction = max(slow_fractions) if slow_fractions else None
    slow_medians = [e.get("slow_median_us") for e in entries]
    slow_medians = [float(x) for x in slow_medians if x is not None]
    return {
        "p50_us": max(float(e["p50_us"]) for e in entries),
        "p99_us": max(float(e["p99_us"]) for e in entries),
        "mean_us": max(float(e["mean_us"]) for e in entries),
        "calls": min(int(e["calls"]) for e in entries),
        "slow_fraction": fraction,
        "p99_unreliable": bool(
            fraction is not None and stats.SLOW_FRACTION_LOW <= fraction <= stats.SLOW_FRACTION_HIGH),
        "slow_calls": max(int(e.get("slow_calls") or 0) for e in entries),
        "slow_median_us": max(slow_medians) if slow_medians else None,
        "aggregation": "worst case over physical error rates",
    }


def _budget_status(summary: dict, cfg: TuneConfig) -> dict:
    """The latency-budget verdict of one measurement.

    Under a p99 budget a measurement whose slow-class share is within 2x of 1 % has an unreliable
    p99 (it sits on the cliff between the converged calls and the slow class); it counts as within
    the budget only when the median of its slow calls is within the budget.  With at most 2 % slow
    calls, all slower than the converged ones, the p99 cannot exceed that median.
    """
    checks = {}
    ok = True
    if cfg.p50_budget_us is not None:
        passed = summary["p50_us"] <= cfg.p50_budget_us
        checks["p50"] = {"budget_us": cfg.p50_budget_us, "measured_us": summary["p50_us"],
                         "within": passed}
        ok = ok and passed
    if cfg.p99_budget_us is not None:
        unreliable = bool(summary.get("p99_unreliable"))
        slow_median = summary.get("slow_median_us")
        if unreliable:
            passed = slow_median is not None and float(slow_median) <= cfg.p99_budget_us
            basis = ("slow_fraction is within 2x of 1 %, so the p99 is unreliable; the verdict "
                     "uses the median of the slow (non-converged) calls")
        else:
            passed = summary["p99_us"] <= cfg.p99_budget_us
            basis = "p99"
        checks["p99"] = {"budget_us": cfg.p99_budget_us, "measured_us": summary["p99_us"],
                         "within": passed, "basis": basis,
                         "slow_fraction": summary.get("slow_fraction"),
                         "p99_unreliable": unreliable,
                         "slow_calls": summary.get("slow_calls"),
                         "slow_median_us": slow_median}
        ok = ok and passed
    return {"within": ok, "checks": checks}


#: Above this many handles the latency stage creates them lazily instead of all at once, to keep
#: the resident decoder memory bounded on the largest problems.
PRECREATE_HANDLE_LIMIT = 32


def _create_handles(points: list[grid.Point], cases: list[dict], workers: int) -> dict:
    """Create one decoder handle per (point, case); concurrently when workers > 1."""
    handles: dict = {}
    keys = [(point, case) for point in points for case in cases]
    if workers <= 1 or len(keys) <= 1:
        for point, case in keys:
            handles.setdefault(point, {})[case["label"]] = _make_decoder(point, case)
        return handles
    try:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(keys))) as pool:
            futures = [(point, case, pool.submit(_make_decoder, point, case))
                       for point, case in keys]
            for point, case, future in futures:
                handles.setdefault(point, {})[case["label"]] = future.result()
    except BaseException:
        for by_case in handles.values():
            for decoder in by_case.values():
                decoder.close()
        raise
    return handles


def run_latency_stage(cfg: TuneConfig, cases: list[dict], results: dict, log) -> dict:
    """Measure every surviving point, paired against the current best, with an A/A band.

    Every paired measurement (every A/A cycle and every candidate-versus-best experiment) times its
    own freshly sampled syndromes, with a seed derived from --seed and a deterministic experiment
    tag; both handles of a measurement decode the same set, so the pairing stays exact and no two
    measurements share a syndrome.  The status word of every timed call splits the samples into the
    converged calls and the slow (non-converged) class.
    """
    survivors = _survivors(results, cfg.ler_target is not None)
    out = {"cadence_us": cfg.cadence_us, "experiments": [], "aa": None, "survivors": []}
    if not survivors:
        out["note"] = ("no point met the accuracy target; latency was not measured"
                       if cfg.ler_target is not None else
                       "no point passed the LER floor; latency was not measured")
        return out
    ordered = sorted(survivors, key=lambda p: (
        results[p]["decode_us_per_shot"] if results[p]["decode_us_per_shot"] else float("inf"),
        p.max_iter, p.alpha, p.decoder))
    out["order"] = [p.label() for p in ordered]
    out["order_basis"] = "mean decode time per shot measured in the accuracy stages"

    calls_base = latency.scale_calls(cfg.cadence_us, 400 if cfg.quick else 2000)
    calls_large = min(calls_base * 4,
                      latency.scale_calls(cfg.cadence_us, 1600 if cfg.quick else 8000))
    out["cases"] = [{"label": case["label"], "p": case["p"]} for case in cases]
    out["calls_per_side"] = calls_base
    out["calls_top"] = calls_large

    experiment_counter = [0]

    def fresh_meters(calls: int):
        """One LatencyMeter per case over ``calls`` fresh, distinct syndromes; and their seeds."""
        tag = f"e{experiment_counter[0]}"
        experiment_counter[0] += 1
        meters, seeds = {}, {}
        for case in cases:
            seed = stats.derive_seed(cfg.seed, "latency-shots", case["seed_key"], tag)
            source = case["problem"].shot_source(seed)
            dets_packed, _obs = source.sample(int(calls))
            dets = problems.unpack_bits(np.asarray(dets_packed, dtype=np.uint8),
                                        case["problem"].m)
            meters[case["label"]] = latency.LatencyMeter(dets, cfg.cadence_us)
            seeds[case["label"]] = seed
        return meters, seeds

    def syndrome_fields(calls: int, seeds: dict) -> dict:
        return {"distinct_syndromes": int(calls),
                "syndrome_seeds": {label: int(seed) for label, seed in seeds.items()}}

    n_handles = len(survivors) * len(cases)
    precreate = cfg.workers > 1 and n_handles <= PRECREATE_HANDLE_LIMIT
    out["handle_creation"] = {"mode": "pre_created" if precreate else "lazy",
                              "handles": n_handles, "workers": int(cfg.workers)}
    pre_handles = (_create_handles(survivors, cases, cfg.workers) if precreate else {})
    handles: dict[grid.Point, dict] = {}

    def get_handles(point):
        if point in pre_handles:
            return pre_handles[point]
        if point not in handles:
            handles[point] = {case["label"]: _make_decoder(point, case) for case in cases}
        return handles[point]

    def drop(point):
        if point in pre_handles:
            return
        for decoder in handles.pop(point, {}).values():
            decoder.close()

    try:
        first = ordered[0]
        first_handles = get_handles(first)
        aa_cycles = 2 if cfg.cadence_us <= 1000 else 1
        out["aa_cycles"] = aa_cycles

        def measure_aa(calls_n, label):
            second = _create_handles([first], cases, cfg.workers)[first]
            per_case = []
            cycle_meters = [fresh_meters(calls_n) for _ in range(aa_cycles)]
            try:
                for case in cases:
                    band = latency.aa_band(
                        cycle_meters[0][0][case["label"]],
                        first_handles[case["label"]], second[case["label"]], calls_n,
                        cycles=aa_cycles,
                        meters=[cm[0][case["label"]] for cm in cycle_meters])
                    band["case"] = case["label"]
                    band["syndrome_seeds"] = [int(cm[1][case["label"]])
                                              for cm in cycle_meters]
                    per_case.append(band)
            finally:
                for decoder in second.values():
                    decoder.close()
            return {"cases": per_case,
                    "band_p50": max(b["band_p50"] for b in per_case),
                    "band_p99": max(b["band_p99"] for b in per_case),
                    "aggregation": "max over physical error rates",
                    "calls_per_side": calls_n, "set": label,
                    "aa_cycles": aa_cycles,
                    "distinct_syndromes": int(calls_n) * aa_cycles,
                    "syndrome_seeds": {case["label"]: [int(cm[1][case["label"]])
                                                       for cm in cycle_meters]
                                       for case in cases}}

        # The points most likely to win are measured with four times the calls: the A/A band of
        # the pooled p99 shrinks roughly with the square root of the call count, so the budget
        # decision on the fastest candidates rests on a tighter tail estimate.
        top_k = min(3, len(ordered)) if calls_large > calls_base else 0
        aa = measure_aa(calls_large if top_k else calls_base, "top" if top_k else "all")
        out["aa"] = aa
        out["top_candidates"] = top_k
        aa_base = aa
        if 0 < top_k < len(ordered):
            aa_base = measure_aa(calls_base, "rest")
        out["aa_base"] = aa_base
        log(f"[latency] A/A band: p50 +-{aa['band_p50'] * 100:.1f} %, "
            f"p99 +-{aa['band_p99'] * 100:.1f} % ({aa['calls_per_side']} calls per side, "
            f"top {top_k} candidates); base band p99 +-{aa_base['band_p99'] * 100:.1f} % "
            f"({aa_base['calls_per_side']} calls)")

        first_stats = {"point": first.as_dict(), "label": first.label(),
                       "per_case": [{"case": b["case"], "ref": b["ref"], "cand": b["cand"],
                                     "ratio_p50": b["ratio_p50"], "ratio_p99": b["ratio_p99"]}
                                    for b in aa["cases"]],
                       "distinct_syndromes": aa["distinct_syndromes"],
                       "syndrome_seeds": aa["syndrome_seeds"]}
        first_stats["cand"] = _aggregate_latency(first_stats["per_case"], "cand")
        first_stats["ref"] = _aggregate_latency(first_stats["per_case"], "ref")
        first_stats["role"] = "initial best (measured as the reference of the A/A experiment)"
        first_stats["calls_per_side"] = aa["calls_per_side"]
        measured = {first: first_stats}
        best = first
        for position, point in enumerate(ordered[1:], 1):
            top = position < top_k
            pair_calls = calls_large if top else calls_base
            band = aa if top else aa_base
            cand_handles = get_handles(point)
            ref_handles = get_handles(best)
            meters, seeds = fresh_meters(pair_calls)
            per_case = []
            for case in cases:
                pair = latency.measure_pair(meters[case["label"]],
                                            ref_handles[case["label"]],
                                            cand_handles[case["label"]], pair_calls)
                pair["case"] = case["label"]
                per_case.append(pair)
            cand = _aggregate_latency(per_case, "cand")
            ref = _aggregate_latency(per_case, "ref")
            ratio_p50 = stats.ratio(cand["p50_us"], ref["p50_us"])
            ratio_p99 = stats.ratio(cand["p99_us"], ref["p99_us"])
            verdict = "within_noise"
            if ratio_p50 < 1.0 - band["band_p50"] and ratio_p99 < 1.0 - band["band_p99"]:
                verdict = "faster"
            elif ratio_p50 > 1.0 + band["band_p50"] and ratio_p99 > 1.0 + band["band_p99"]:
                verdict = "slower"
            entry = {"point": point.as_dict(), "label": point.label(), "ref_label": best.label(),
                     "per_case": per_case, "cand": cand, "ref": ref,
                     "ratio_p50": ratio_p50, "ratio_p99": ratio_p99, "verdict": verdict,
                     "calls_per_side": pair_calls, "top_candidate": bool(top),
                     "band_p50": band["band_p50"], "band_p99": band["band_p99"],
                     **syndrome_fields(pair_calls, seeds)}
            out["experiments"].append(entry)
            measured[point] = entry
            if verdict == "faster":
                drop(best)
                best = point
            else:
                drop(point)
            log(f"[latency] {point.label()} vs {entry['ref_label']}: "
                f"p50 {cand['p50_us']:.1f}/{ref['p50_us']:.1f} us "
                f"(x{ratio_p50:.3f}), p99 {cand['p99_us']:.1f}/{ref['p99_us']:.1f} us "
                f"(x{ratio_p99:.3f}): {verdict}")
        out["measured"] = {p.label(): entry for p, entry in measured.items()}
        out["_measured_points"] = measured  # internal, stripped before writing
        out["_reference_point"] = first     # the A/A reference (internal)
    finally:
        for point in list(pre_handles):
            for decoder in pre_handles.pop(point).values():
                decoder.close()
        for point in list(handles):
            drop(point)
    return out


def preset_alphas() -> list[float]:
    """The ms_scaling_factor values of the presets the package ships, ascending."""
    return sorted({float(cfg["ms_scaling_factor"]) for cfg in eldpc.presets().values()
                   if "ms_scaling_factor" in cfg})


def _alpha_distance(alpha: float, shipped: list[float]) -> float:
    return round(min((abs(float(alpha) - a) for a in shipped), default=0.0), 9)


def _tie_key(row: dict, shipped: list[float]) -> tuple:
    """Order inside a tie set: lower LER upper bound, fewer BP iterations, alpha closest to a
    shipped preset, then the point's name."""
    point = row["point"]
    return (row["ler_upper"], point.max_iter, _alpha_distance(point.alpha, shipped),
            point.label())


def select_tie_set(rows: list[dict], cfg: TuneConfig, shipped: list[float]) -> dict:
    """Pick one row from ``rows`` (all within the budgets) with the deterministic tie rule.

    The fastest row is the one with the lowest p50 (then p99).  Every row whose p50 is within the
    measured A/A band of the fastest -- and, when the budget is a p99 budget, whose p99 is within
    the A/A band of the lowest p99 -- is indistinguishable from it on this machine and joins the
    tie set.  With ``--weight`` the same idea applies to the score: rows whose score is within
    ``weight * log10(1 + p50 band)`` of the best score tie.  Inside the tie set the choice is the
    lowest LER upper bound, then the fewest BP iterations, then the alpha closest to a shipped
    preset's alpha, then the name order.  A row without a recorded band has band 0, so only exact
    ties remain.
    """
    if cfg.weight is not None:
        for row in rows:
            row["score"] = (stats.log10_or_floor(row["ler_upper"])
                            + float(cfg.weight) * stats.log10_or_floor(row["p50"], 1e-9))
        fastest = min(rows, key=lambda r: (r["score"], r["p50"], r["p99"],
                                           r["point"].max_iter, r["point"].alpha,
                                           r["point"].label()))

        def slack(r):
            return float(cfg.weight) * math.log10(1.0 + max(r["band_p50"], fastest["band_p50"]))

        ties = [r for r in rows if r["score"] <= fastest["score"] + slack(r) + 1e-12]
        basis = "score"
    else:
        fastest = min(rows, key=lambda r: (r["p50"], r["p99"], r["point"].max_iter,
                                           r["point"].alpha, r["point"].label()))
        use_p99 = cfg.p99_budget_us is not None
        # Every row is compared with the fastest row itself, so the fastest always joins its own
        # tie set (comparing p99 with the lowest p99 of another row could leave the set empty).
        ties = []
        for r in rows:
            ok = r["p50"] <= fastest["p50"] * (1.0 + max(r["band_p50"], fastest["band_p50"]))
            if ok and use_p99:
                ok = r["p99"] <= fastest["p99"] * (1.0 + max(r["band_p99"], fastest["band_p99"]))
            if ok:
                ties.append(r)
        basis = "p50 and p99" if use_p99 else "p50"
    if not any(r is fastest for r in ties):
        ties.append(fastest)
    ties = sorted(ties, key=lambda r: _tie_key(r, shipped))
    chosen = ties[0]
    return {"chosen": chosen, "fastest": fastest, "ties": ties, "basis": basis,
            "shipped_alphas": shipped}


def decision_measurement(entry: dict, decided: dict | None) -> dict:
    """The measurement a row's in/out decision rests on, with its own p50 / p99.

    A re-timed point: its re-timing; any other point: its latency-stage measurement.  Returns
    ``{p50_us, p99_us, calls, measurement}``, ``measurement`` naming the source and its calls per
    side, so a nearest-list line prints the p50 / p99 of the same measurement as its budget ratio.
    """
    retime = (decided or {}).get("retime")
    if retime is not None:
        summary = retime["cand"]
        calls = int(retime.get("calls_per_side") or summary.get("calls") or 0)
        source = f"re-timed at {calls} calls per side"
    else:
        summary = entry["cand"]
        calls = int(entry.get("calls_per_side") or summary.get("calls") or 0)
        source = f"latency stage, {calls} calls per side"
    return {"p50_us": summary["p50_us"], "p99_us": summary["p99_us"], "calls": calls,
            "measurement": source}


def latency_rows(cfg: TuneConfig, results: dict, latency_stage: dict) -> list[dict]:
    """One row per latency-measured screen survivor: its record, latency summary and budget.

    The budget verdict is the latency-stage measurement's, unless the run decided budget
    membership separately (``latency_stage["_membership"]``, written by the re-timing stage of a
    run with a latency budget): then it is that decision, which for a point near the budget rests
    on its re-timing.  ``decision`` holds the p50 / p99 of the measurement the verdict rests on.
    """
    aa = latency_stage.get("aa") or {}
    survivors = _survivors(results, cfg.ler_target is not None)
    measured = latency_stage.get("_measured_points", {})
    membership = latency_stage.get("_membership") or {}
    rows = []
    for point in survivors:
        record = results[point]
        entry = measured.get(point)
        if entry is None:
            continue
        summary = entry["cand"]
        budget = (membership[point]["budget"] if point in membership
                  else _budget_status(summary, cfg))
        rows.append({"point": point, "record": record, "latency": entry, "summary": summary,
                     "budget": budget,
                     "decision": decision_measurement(entry, membership.get(point)),
                     "band_p50": float(entry.get("band_p50", aa.get("band_p50", 0.0)) or 0.0),
                     "band_p99": float(entry.get("band_p99", aa.get("band_p99", 0.0)) or 0.0),
                     "ler_upper": record.get("cp99_upper_max", float("inf")),
                     "p50": summary["p50_us"], "p99": summary["p99_us"]})
    return rows


def choose(cfg: TuneConfig, results: dict, latency_stage: dict,
           shipped_alphas: list[float] | None = None) -> dict:
    """The Pareto choice: fastest qualifying point, or the nearest points if none qualifies.

    Candidates whose latency is within the measured A/A band of the fastest form a tie set; the
    choice inside it is deterministic (see :func:`select_tie_set`).  This is the screen choice; the
    accuracy confirmation then keeps or rejects its leading candidates.
    """
    target_present = cfg.ler_target is not None
    shipped = preset_alphas() if shipped_alphas is None else sorted(shipped_alphas)
    aa = latency_stage.get("aa") or {}
    rows = latency_rows(cfg, results, latency_stage)
    eligible = [r for r in rows if r["budget"]["within"]]
    choice: dict = {"target_present": target_present, "eligible": len(eligible),
                    "measured": len(rows)}
    if eligible:
        selection = select_tie_set(eligible, cfg, shipped)
        chosen = selection["chosen"]
        if cfg.weight is not None:
            choice["ranking"] = ("score = log10(LER upper bound) + weight * log10(p50_us), "
                                 "smallest wins; scores within weight * log10(1 + A/A p50 band) "
                                 "of the best tie")
        else:
            choice["ranking"] = (f"lowest {selection['basis']}; candidates within the measured "
                                 f"A/A band of the fastest tie")
        choice["tie_rule"] = ("within a tie: lower LER upper bound, then fewer BP iterations, "
                              "then alpha closest to a shipped preset's alpha, then name order")
        choice["fastest"] = selection["fastest"]["point"].label()
        choice["shipped_preset_alphas"] = shipped
        choice["tie_set"] = [
            {"label": r["point"].label(), "ler_upper": r["ler_upper"], "p50_us": r["p50"],
             "p99_us": r["p99"], "max_iter": r["point"].max_iter,
             "ms_scaling_factor": r["point"].alpha,
             "alpha_distance_to_preset": _alpha_distance(r["point"].alpha, shipped),
             "band_p50": r["band_p50"], "band_p99": r["band_p99"],
             "chosen": r is chosen}
            for r in selection["ties"]]
        choice["config"] = chosen["point"].config()
        choice["point"] = chosen["point"].as_dict()
        choice["reason"] = ("the fastest measured point whose LER upper bound is below the "
                            "target, or a point tied with it within the A/A band (tie rule)"
                            if target_present else
                            "the fastest measured point, or a point tied with it within the "
                            "A/A band (tie rule)")
        choice["ler_upper"] = chosen["ler_upper"]
        choice["p50_us"] = chosen["p50"]
        choice["p99_us"] = chosen["p99"]
        choice["budget"] = chosen["budget"]
        if cfg.weight is not None:
            choice["score"] = chosen.get("score")
    else:
        choice["config"] = None
        if target_present:
            choice["reason"] = ("no measured point meets the accuracy target"
                                if not rows else
                                "no point that meets the accuracy target is within the latency "
                                "budget")
        else:
            choice["reason"] = ("no point passes the LER floor" if not rows else
                                "no point that passes the LER floor is within the latency budget")
        if not rows:
            failed = [(p, r) for p, r in results.items() if r["status"] != "skipped"]
            failed.sort(key=lambda pr: pr[1].get("cp99_upper_max", float("inf")))
            choice["nearest"] = [
                {"point": p.as_dict(), "label": p.label(), "status": r["status"],
                 "ler_upper": r.get("cp99_upper_max"), "latency": None,
                 "budget_distance": None}
                for p, r in failed[:5]]
        else:
            def distance(row):
                checks = (row["budget"] or {}).get("checks") or {}
                if not checks:
                    return 1.0
                return max(check["measured_us"] / check["budget_us"]
                           for check in checks.values())
            # p50 / p99 from the measurement the budget verdict (and so the ratio) rests on
            nearest = sorted(rows, key=lambda r: (distance(r), r["decision"]["p50_us"],
                                                  r["decision"]["p99_us"]))[:5]
            choice["nearest"] = [
                {"point": r["point"].as_dict(), "label": r["point"].label(),
                 "status": r["record"]["status"], "ler_upper": r["ler_upper"],
                 "p50_us": r["decision"]["p50_us"], "p99_us": r["decision"]["p99_us"],
                 "calls": r["decision"]["calls"], "measurement": r["decision"]["measurement"],
                 "budget": r["budget"], "budget_distance": distance(r)}
                for r in nearest]
            for entry, r in zip(choice["nearest"], nearest):
                if r["budget"].get("basis"):   # a run with a budget re-timing: the decision's basis
                    entry["basis"] = r["budget"]["basis"]
    return choice


#: Distinct syndromes a full-run latency confirmation of a candidate times per case.
CONFIRMATION_MIN_SYNDROMES = 8000


def run_latency_confirmation(cfg: TuneConfig, cases: list[dict], point: grid.Point,
                             log) -> dict:
    """An independent single-sided run of one configuration, per case, on fresh syndromes.

    The full run times at least ``CONFIRMATION_MIN_SYNDROMES`` distinct syndromes per case; a
    quick run uses the small call count.  The budget verdict is this run's, including the
    slow-class rule for a p99 budget.
    """
    out = {"point": point.as_dict(), "label": point.label(), "per_case": []}
    calls = latency.scale_calls(cfg.cadence_us, 400 if cfg.quick else CONFIRMATION_MIN_SYNDROMES)
    if not cfg.quick:
        calls = max(int(calls), CONFIRMATION_MIN_SYNDROMES)
    seeds = {}
    for case in cases:
        seed = stats.derive_seed(cfg.seed, "latency-confirm", case["seed_key"])
        seeds[case["label"]] = seed
        source = case["problem"].shot_source(seed)
        dets_packed, _obs = source.sample(int(calls))
        dets = problems.unpack_bits(np.asarray(dets_packed, dtype=np.uint8), case["problem"].m)
        meter = latency.LatencyMeter(dets, cfg.cadence_us)
        decoder = _make_decoder(point, case)
        try:
            run = meter.run(decoder, calls)
            variant = getattr(decoder, "core_variant", None)
        finally:
            decoder.close()
        run.pop("_samples", None)
        run.pop("_statuses", None)
        run["case"] = case["label"]
        run["seed"] = seed
        run["distinct_syndromes"] = int(calls)
        run["core_variant"] = variant
        out["per_case"].append(run)
    out["summary"] = _aggregate_latency([{"cand": r} for r in out["per_case"]], "cand")
    out["distinct_syndromes"] = min(int(r["distinct_syndromes"]) for r in out["per_case"])
    out["syndrome_seeds"] = {label: int(seed) for label, seed in seeds.items()}
    variants = sorted({str(r["core_variant"]) for r in out["per_case"]
                       if r["core_variant"] is not None})
    out["core_variant"] = (variants[0] if len(variants) == 1 else variants) if variants else None
    out["budget"] = _budget_status(out["summary"], cfg)
    log(f"[latency] confirmation {point.label()}: p50 {out['summary']['p50_us']:.1f} us, "
        f"p99 {out['summary']['p99_us']:.1f} us, slow fraction "
        f"{out['summary']['slow_fraction']}, within budget: {out['budget']['within']}")
    return out


def confirmation_latency_record(lat: dict) -> dict:
    """The latency record kept with every confirmed candidate (chosen or not).

    The pooled summary plus the measurement's syndrome fields, so every latency measurement of the
    run records how many distinct syndromes it timed and from which seeds.
    """
    return {**lat["summary"],
            "distinct_syndromes": int(lat["distinct_syndromes"]),
            "syndrome_seeds": dict(lat["syndrome_seeds"]),
            "core_variant": lat.get("core_variant")}


def p99_warnings(accuracy_confirmation: dict, pareto_rows: list[dict]) -> list[str]:
    """One warning per reported candidate whose p99 is unreliable.

    The reported candidates are every confirmed candidate (its latency confirmation) and every
    measured row of the Pareto table (its latency-stage measurement); the chosen point is one of
    the confirmed candidates.
    """
    chosen = accuracy_confirmation.get("chosen")
    out = []

    def text(label, where, summary):
        median = summary.get("slow_median_us")
        tail = f"{float(median):.1f} us" if median is not None else "not recorded"
        return (f"{label}{' (chosen)' if label == chosen else ''}: the p99 of its {where} is "
                f"unreliable: slow_fraction {float(summary.get('slow_fraction') or 0.0):.4g} is "
                f"within 2x of 1 %, so the p99 sits on the cliff between the converged calls and "
                f"the slow class; the median of its slow calls ({tail}) is the stable tail number")

    for cand in accuracy_confirmation.get("candidates") or []:
        summary = cand.get("latency") or {}
        if summary.get("p99_unreliable"):
            out.append(text(cand["label"], "latency confirmation", summary))
    for row in pareto_rows:
        if row.get("p99_unreliable"):
            out.append(text(row["label"], "latency-stage measurement", row))
    return out


#: The variable the front reads at every create to force a GPU core variant (docs/BUILD.md).
VARIANT_VARIABLE = "ELDPC_DEVICE_CLASS"
CORE_VARIANTS = ("default", "ada64")


def table_core_variant(device: dict | None) -> str:
    """The variant the device-class table gives the target device (device 0), ignoring overrides."""
    from eldpc._variant import core_variant_for_device

    try:
        major, minor = (int(x) for x in str((device or {}).get("compute_capability")).split("."))
        return core_variant_for_device(major, minor, int((device or {}).get("sm_count") or 0))
    except (TypeError, ValueError):
        return "default"


@contextlib.contextmanager
def forced_core_variant(variant: str):
    """Set ELDPC_DEVICE_CLASS for the creates inside the block and restore the previous value."""
    previous = os.environ.get(VARIANT_VARIABLE)
    os.environ[VARIANT_VARIABLE] = variant
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(VARIANT_VARIABLE, None)
        else:
            os.environ[VARIANT_VARIABLE] = previous


def run_core_variant_comparison(cfg: TuneConfig, cases: list[dict], point: grid.Point,
                                calls: int, aa_cycles: int, log, device: dict | None = None) -> dict:
    """Time the chosen point under both GPU core variants, paired on the same fresh syndromes.

    Only a device whose class has two variants (the table gives it ``ada64``) is measured; on any
    other device the stage is skipped and recorded.  One handle of each variant is created with
    ELDPC_DEVICE_CLASS set around its create (the front reads it at every create; the previous
    value is restored), the two are paired in the ABBA schedule of the latency stage, and an A/A of
    two handles of the table's variant gives the band.  The variants run the same decoder sources,
    so only latency is compared.  The faster variant is the one whose p50 and p99 are both beyond
    the band (worst case over the cases); otherwise there is "no difference".
    """
    if device is None:
        found = eldpc.devices()
        device = found[0] if found else None
    table = table_core_variant(device)
    out = {"evaluated": False, "label": point.label(), "variable": VARIANT_VARIABLE,
           "table_variant": table, "environment_value": os.environ.get(VARIANT_VARIABLE) or None,
           "device": ({k: device.get(k) for k in ("name", "compute_capability", "sm_count")}
                      if device else None),
           "config_json": ("not written into config.json: Decoder(config=...) has no variant key; "
                           "the front reads ELDPC_DEVICE_CLASS from the process environment at "
                           "every create")}
    if table == "default":
        out["reason"] = ("this device class has one core variant (default); only compute "
                         "capability 8.9 with at least 128 SMs has a second (ada64): skipped")
        log(f"[variants] {out['reason']}")
        return out
    out.update({"evaluated": True, "calls_per_side": int(calls), "aa_cycles": int(aa_cycles),
                "per_case": [], "aa": []})
    opened = []

    def create(variant):
        with forced_core_variant(variant):
            decoder = _make_decoder(point, case)
        opened.append(decoder)
        actual = getattr(decoder, "core_variant", variant)
        if actual != variant:
            raise RuntimeError(f"{VARIANT_VARIABLE}={variant} gave a handle of variant {actual!r}")
        return decoder

    def meter(tag):
        seed = stats.derive_seed(cfg.seed, "variant-shots", case["seed_key"], tag)
        dets_packed, _obs = case["problem"].shot_source(seed).sample(int(calls))
        dets = problems.unpack_bits(np.asarray(dets_packed, dtype=np.uint8), case["problem"].m)
        return latency.LatencyMeter(dets, cfg.cadence_us), seed

    try:
        for case in cases:
            ref, cand = create("default"), create("ada64")
            pair_meter, pair_seed = meter("pair")
            pair = latency.measure_pair(pair_meter, ref, cand, calls)
            pair.update(case=case["label"], syndrome_seed=pair_seed, distinct_syndromes=int(calls))
            out["per_case"].append(pair)
            first, second = (ref if table == "default" else cand), create(table)
            aa_meters = [meter(f"aa{i}") for i in range(int(aa_cycles))]
            band = latency.aa_band(None, first, second, calls, cycles=aa_cycles,
                                   meters=[m for m, _seed in aa_meters])
            out["aa"].append({"case": case["label"], "band_p50": band["band_p50"],
                              "band_p99": band["band_p99"], "ratios": band["aa_ratios"],
                              "syndrome_seeds": [int(seed) for _m, seed in aa_meters]})
    finally:
        for decoder in opened:
            decoder.close()
    out["default"] = _aggregate_latency(out["per_case"], "ref")
    out["ada64"] = _aggregate_latency(out["per_case"], "cand")
    out["ratio_p50"] = stats.ratio(out["ada64"]["p50_us"], out["default"]["p50_us"])
    out["ratio_p99"] = stats.ratio(out["ada64"]["p99_us"], out["default"]["p99_us"])
    out["band_p50"] = max(entry["band_p50"] for entry in out["aa"])
    out["band_p99"] = max(entry["band_p99"] for entry in out["aa"])
    out["distinct_syndromes"] = int(calls)
    faster = "no difference"
    if out["ratio_p50"] < 1.0 - out["band_p50"] and out["ratio_p99"] < 1.0 - out["band_p99"]:
        faster = "ada64"
    elif out["ratio_p50"] > 1.0 + out["band_p50"] and out["ratio_p99"] > 1.0 + out["band_p99"]:
        faster = "default"
    out["faster"] = faster
    out["rule"] = ("ratio = ada64 / default of the worst-case pooled p50 and p99; a variant is "
                   "faster when both ratios are beyond the A/A band, else no difference")
    if faster == "no difference" or faster == table:
        out["recommended_env"] = None
        out["recommendation"] = (f"keep the table's choice ({table}); no variable needed"
                                 + ("" if faster == table else
                                    ": the variants do not differ beyond the A/A band"))
    else:
        out["recommended_env"] = f"{VARIANT_VARIABLE}={faster}"
        out["recommendation"] = (f"set {VARIANT_VARIABLE}={faster} in the environment of the "
                                 f"process that creates the decoder (the table picks {table})")
    log(f"[variants] {point.label()}: ada64/default p50 x{out['ratio_p50']:.3f}, p99 "
        f"x{out['ratio_p99']:.3f} (A/A band p50 +-{out['band_p50'] * 100:.1f} %, p99 "
        f"+-{out['band_p99'] * 100:.1f} %): {faster}; {out['recommendation']}")
    return out


def run_keep_warm(cfg: TuneConfig, cases: list[dict], point: grid.Point, aa_band: dict,
                  log) -> dict:
    """Keep-warm on/off at the application's call cadence, for the chosen configuration."""
    out = {"evaluated": False, "recommended": False, "per_case": []}
    if cfg.cadence_us <= latency.KEEP_WARM_MIN_CADENCE_US:
        out["reason"] = (f"call cadence {cfg.cadence_us:g} us is at or below "
                         f"{latency.KEEP_WARM_MIN_CADENCE_US:g} us: the clock is held by the "
                         f"call stream itself and the keep-warm tick does not fit before a call")
        return out
    out["evaluated"] = True
    calls = latency.scale_calls(cfg.cadence_us, 400 if cfg.quick else 2000)
    out["calls_per_mode"] = calls
    for case in cases:
        seed = stats.derive_seed(cfg.seed, "keep-warm-shots", case["seed_key"])
        source = case["problem"].shot_source(seed)
        dets_packed, _obs = source.sample(int(calls))
        dets = problems.unpack_bits(np.asarray(dets_packed, dtype=np.uint8),
                                    case["problem"].m)
        meter = latency.LatencyMeter(dets, cfg.cadence_us)
        decoder = _make_decoder(point, case)
        try:
            modes = latency.measure_keep_warm(meter, decoder, calls)
        finally:
            decoder.close()
        modes["case"] = case["label"]
        modes["syndromes_seed"] = seed
        modes["distinct_syndromes"] = int(calls)
        out["per_case"].append(modes)
    off = {"p50_us": max(m["off"]["p50_us"] for m in out["per_case"]),
           "p99_us": max(m["off"]["p99_us"] for m in out["per_case"]),
           "calls": min(m["off"]["calls"] for m in out["per_case"])}
    on = {"p50_us": max(m["on"]["p50_us"] for m in out["per_case"]),
          "p99_us": max(m["on"]["p99_us"] for m in out["per_case"]),
          "calls": min(m["on"]["calls"] for m in out["per_case"])}
    out["off"] = off
    out["on"] = on
    out["ratio_p50"] = stats.ratio(on["p50_us"], off["p50_us"])
    out["ratio_p99"] = stats.ratio(on["p99_us"], off["p99_us"])
    band_p50 = float(aa_band.get("band_p50", 0.0)) if aa_band else 0.0
    band_p99 = float(aa_band.get("band_p99", 0.0)) if aa_band else 0.0
    out["aa_band_p50"] = band_p50
    out["aa_band_p99"] = band_p99
    out["aa_band_calls_per_side"] = (aa_band or {}).get("calls_per_side")
    # The clock effect of keep-warm shows up in the typical call, so the decision rests on p50
    # improving beyond its A/A band.  The p99 ratio is also required to stay within its band, but
    # at the call counts of sparse cadences the p99 is one of the few largest samples and its band
    # is wide, so that condition only catches a gross tail regression; it is not a tail guard.
    out["recommended"] = bool(out["ratio_p50"] < 1.0 - band_p50
                              and out["ratio_p99"] <= 1.0 + band_p99)
    out["rule"] = ("keep-warm is recommended when p50 improves beyond the A/A band; p99 must not "
                   "get worse beyond its own A/A band")
    out["decision_basis"] = "p50"
    out["p99_note"] = (f"the decision rests on p50: the p99 A/A band is +-{band_p99 * 100:.0f} % "
                       f"at the {calls} calls per mode of this cadence, so a p99 regression "
                       f"smaller than x{1.0 + band_p99:.2f} cannot be detected; this is not a "
                       f"p99 guard")
    log(f"[keep-warm] p50 off/on {off['p50_us']:.1f}/{on['p50_us']:.1f} us, "
        f"p99 {off['p99_us']:.1f}/{on['p99_us']:.1f} us -> "
        f"{'on' if out['recommended'] else 'off'}")
    return out


def _pareto_rows(results: dict, latency_stage: dict, cfg: TuneConfig) -> list[dict]:
    measured = latency_stage.get("_measured_points", {})
    membership = latency_stage.get("_membership") or {}
    rows = []
    for point, record in results.items():
        entry = measured.get(point)
        row = {
            "label": point.label(),
            "decoder": point.decoder,
            "alpha": point.alpha,
            "ms_scaling_factor": point.alpha,
            "max_iter": point.max_iter,
            "status": record["status"],
            "per_case": [{"case": r["case"], "p": r["p"], "shots": r["shots"],
                          "failures": r["failures"], "ler": r["ler"],
                          "cp99_lower": r["cp99_lower"], "cp99_upper": r["cp99_upper"]}
                         for r in record.get("per_p", [])],
            "ler_upper_max": record.get("cp99_upper_max"),
            "decode_us_per_shot": record.get("decode_us_per_shot"),
            "converged_share": record.get("converged_share"),
        }
        if record["status"] == "skipped":
            row["dominated_by"] = record.get("dominated_by")
            row["dominator_converged_share"] = record.get("dominator_converged_share")
        if record.get("ler_floor") is not None:
            row["ler_floor_passed"] = bool(record["ler_floor"]["passed"])
        if entry is not None:
            summary = entry["cand"]
            row["p50_us"] = summary["p50_us"]
            row["p99_us"] = summary["p99_us"]
            fraction = summary.get("slow_fraction")
            row["slow_fraction"] = fraction
            row["p99_unreliable"] = bool(
                fraction is not None
                and stats.SLOW_FRACTION_LOW <= float(fraction) <= stats.SLOW_FRACTION_HIGH)
            row["slow_calls"] = summary.get("slow_calls")
            row["slow_median_us"] = summary.get("slow_median_us")
            row["budget"] = _budget_status(summary, cfg)
            row["latency_verdict"] = entry.get("verdict", "initial best")
            decided = membership.get(point)
            if decided is not None:
                # the in/out decision; for a re-timed point it is the re-timing's verdict
                row["budget"] = decided["budget"]
                row["budget_basis"] = decided["basis"]
                if decided.get("retime") is not None:
                    row["retime_p50_us"] = decided["retime"]["cand"]["p50_us"]
                    row["retime_p99_us"] = decided["retime"]["cand"]["p99_us"]
        else:
            row["p50_us"] = None
            row["p99_us"] = None
            row["slow_fraction"] = None
            row["p99_unreliable"] = False
            row["budget"] = None
        rows.append(row)
    _mark_pareto(rows)
    return rows


def _mark_pareto(rows: list[dict]) -> None:
    """Mark non-dominated rows on (LER upper bound, p50) among measured rows."""
    measured = [r for r in rows if r["p50_us"] is not None and r["ler_upper_max"] is not None]
    for row in rows:
        row["pareto"] = False
    for a in measured:
        dominated = False
        for b in measured:
            if a is b:
                continue
            if (b["ler_upper_max"] <= a["ler_upper_max"] and b["p50_us"] <= a["p50_us"]
                    and (b["ler_upper_max"] < a["ler_upper_max"]
                         or b["p50_us"] < a["p50_us"])):
                dominated = True
                break
        a["pareto"] = not dominated


def screen_numbers(record: dict) -> dict:
    """The screen numbers of one point, next to the confirmation's in results.json."""
    shots = int(record.get("shots") or 0)
    failures = int(record.get("failures") or 0)
    return {"shots": shots, "failures": failures,
            "ler": failures / shots if shots else None,
            "upper": record.get("cp99_upper_max"), "source": "screen"}


# -- the budget re-timing (both paths); without --ler-target also the LER floor and the claim -----

#: A screen survivor (a pass with --ler-target; a point above the LER floor without) whose
#: latency-stage measurement is within this factor of a budget, on either side, is re-timed with
#: the confirmation design before the in/out decision.
#: The latency stage's base measurement is 2000 calls per side (two runs of 1000).  Where the p99
#: sits on a ~1 % per-call host tail or on a bimodal OSD tail (RTX 6000 Ada with a two-socket EPYC
#: 7542, development record; TUNER.md) that pooled p99 read down to 0.64x a point's 8000-call p99,
#: its 1000-call runs 0.53-1.9x the 4000-call runs, and two runs of one OSD-tail point differed 2x:
#: 0.5-2.0 covers that scatter, at a cost of a few seconds per re-timed point.
RETIME_MARGIN = (0.5, 2.0)
#: At most this many points are re-timed, the nearest to the budget first (by |log(ratio)|); the
#: rest of the margin keeps its latency-stage verdict and is listed as capped.
RETIME_MAX_POINTS = 32
#: Exit status of the command line when no configuration qualifies (with or without a target).
NO_QUALIFIER_EXIT = 3


def floor_reference_points(cfg: TuneConfig) -> list[grid.Point]:
    """The points the screen decodes as the LER floor's reference: stock ``ms30``, unless the run
    has --ler-target (the target is the bar) or --max-ler (an explicit floor)."""
    if cfg.ler_target is not None or cfg.max_ler is not None:
        return []
    return [grid.STOCK_POINT]


def _case_numbers(entry: dict) -> dict:
    shots = int(entry["shots"])
    failures = int(entry["failures"])
    return {"case": entry["case"], "p": entry.get("p"), "shots": shots, "failures": failures,
            "ler": failures / shots if shots else None,
            "cp99_lower": entry.get("cp99_lower"), "cp99_upper": entry.get("cp99_upper")}


def ler_floor_record(cfg: TuneConfig, stages: list[dict]) -> dict | None:
    """The results.json "ler_floor" block of a run without --ler-target (None with a target)."""
    if cfg.ler_target is not None:
        return None
    if cfg.max_ler is not None:
        return {"source": "--max-ler", "max_ler": float(cfg.max_ler),
                "rule": ("a point is choosable only when its screen LER (failures / shots) is at "
                         "most --max-ler at every case; this caps the screen's estimate and is "
                         "not a proven bound (--ler-target proves one)"),
                "references": []}
    references = []
    for stage in stages:
        for record in stage.get("references") or []:
            references.append({"stage": stage["name"], "label": grid.STOCK_LABEL,
                               "point": grid.STOCK_POINT.as_dict(),
                               "per_case": [_case_numbers(entry)
                                            for entry in record.get("per_p") or []]})
    return {"source": f"stock {grid.STOCK_LABEL} on the same screen shots",
            "reference": grid.STOCK_POINT.as_dict(),
            "rule": (f"a point is choosable only when, at every case, its screen LER is no worse "
                     f"than stock {grid.STOCK_LABEL} (ldpc's common settings: min-sum alpha "
                     f"0.625, 30 iterations, OSD-0) decoded on the same shots of the same screen "
                     f"stage (paired; failures/shots compared exactly); --max-ler X replaces "
                     f"the floor"),
            "references": references}


def floor_verdict(record: dict, floor: dict) -> dict:
    """Whether one screen record passes the LER floor: per case and overall (all cases)."""
    checks = []
    if floor["source"] == "--max-ler":
        limit = float(floor["max_ler"])
        for entry in record.get("per_p") or []:
            numbers = _case_numbers(entry)
            checks.append({"case": numbers["case"], "ler": numbers["ler"], "floor": limit,
                           "passed": bool(numbers["ler"] is not None
                                          and numbers["ler"] <= limit)})
    else:
        table = {(ref["stage"], case["case"]): case
                 for ref in floor.get("references") or [] for case in ref["per_case"]}
        for entry in record.get("per_p") or []:
            numbers = _case_numbers(entry)
            ref = table.get((entry.get("stage"), entry["case"]))
            if ref is None or not ref["shots"] or not numbers["shots"]:
                checks.append({"case": numbers["case"], "ler": numbers["ler"], "floor": None,
                               "passed": False,
                               "note": f"no {grid.STOCK_LABEL} reference on these shots"})
                continue
            # exact comparison of the two fractions on the same shot stream
            passed = numbers["failures"] * ref["shots"] <= ref["failures"] * numbers["shots"]
            checks.append({"case": numbers["case"], "stage": entry.get("stage"),
                           "ler": numbers["ler"], "floor": ref["ler"],
                           "failures": numbers["failures"], "floor_failures": ref["failures"],
                           "shots": numbers["shots"], "passed": bool(passed)})
    return {"passed": bool(checks) and all(check["passed"] for check in checks),
            "checks": checks}


def apply_ler_floor(results: dict, floor: dict, log) -> None:
    """Record the floor verdict on every screen record and summarize it in ``floor``."""
    choosable, excluded = [], []
    for point in sorted(results):
        record = results[point]
        if record.get("status") == "skipped":
            continue
        record["ler_floor"] = floor_verdict(record, floor)
        (choosable if record["ler_floor"]["passed"] else excluded).append(point.label())
    floor["choosable"] = len(choosable)
    floor["excluded"] = excluded
    log(f"[floor] {floor['source']}: {len(choosable)} choosable point(s), {len(excluded)} "
        f"below the floor")


def budget_ratio(budget: dict | None) -> float | None:
    """The largest measured / budget ratio of a budget verdict (None without a budget)."""
    checks = (budget or {}).get("checks") or {}
    if not checks:
        return None
    return max(float(check["measured_us"]) / float(check["budget_us"])
               for check in checks.values())


def plan_budget_retime(rows: list[dict], margin=RETIME_MARGIN,
                       max_points: int = RETIME_MAX_POINTS) -> dict:
    """Split latency rows by their latency-stage budget ratio: ``retime`` (inside the margin, the
    nearest to the budget first, at most ``max_points``), ``capped`` (inside the margin, beyond
    the cap) and ``decided`` (outside the margin, or no budget).  Each item is (row, ratio)."""
    low, high = float(margin[0]), float(margin[1])
    near, decided = [], []
    for row in rows:
        ratio = budget_ratio(row["budget"])
        if ratio is not None and low <= ratio <= high:
            near.append((row, ratio))
        else:
            decided.append((row, ratio))
    near.sort(key=lambda item: (abs(math.log(item[1])) if item[1] > 0 else float("inf"),
                                item[0]["point"].label()))
    return {"retime": near[:int(max_points)], "capped": near[int(max_points):],
            "decided": decided}


def run_budget_retime(cfg: TuneConfig, cases: list[dict], points: list[grid.Point],
                      reference: grid.Point, log) -> dict:
    """Re-time ``points`` with the confirmation design, each paired with the A/A reference.

    Every point gets at least ``CONFIRMATION_MIN_SYNDROMES`` calls per side and case in a full run
    (``--quick``: the small count) on its own freshly sampled syndromes (stage name
    ``budget-retime``), in the latency stage's ABBA schedule against a handle of the latency
    stage's A/A reference.  Returns {point: entry}; the entry's ``budget`` is this measurement's
    verdict (with the slow-class rule of a p99 budget).
    """
    calls = latency.scale_calls(cfg.cadence_us, 400 if cfg.quick else CONFIRMATION_MIN_SYNDROMES)
    if not cfg.quick:
        calls = max(int(calls), CONFIRMATION_MIN_SYNDROMES)
    out: dict = {}
    if not points:
        return out
    ref_handles = {case["label"]: _make_decoder(reference, case) for case in cases}
    try:
        for point in points:
            cand_handles = {case["label"]: _make_decoder(point, case) for case in cases}
            per_case, seeds = [], {}
            try:
                for case in cases:
                    seed = stats.derive_seed(cfg.seed, "budget-retime", case["seed_key"],
                                             point.label())
                    seeds[case["label"]] = seed
                    dets_packed, _obs = case["problem"].shot_source(seed).sample(int(calls))
                    dets = problems.unpack_bits(np.asarray(dets_packed, dtype=np.uint8),
                                                case["problem"].m)
                    meter = latency.LatencyMeter(dets, cfg.cadence_us)
                    pair = latency.measure_pair(meter, ref_handles[case["label"]],
                                                cand_handles[case["label"]], int(calls))
                    pair["case"] = case["label"]
                    pair["syndrome_seed"] = int(seed)
                    per_case.append(pair)
            finally:
                for decoder in cand_handles.values():
                    decoder.close()
            cand = _aggregate_latency(per_case, "cand")
            ref = _aggregate_latency(per_case, "ref")
            entry = {"label": point.label(), "ref_label": reference.label(),
                     "per_case": per_case, "cand": cand, "ref": ref,
                     "ratio_p50": stats.ratio(cand["p50_us"], ref["p50_us"]),
                     "ratio_p99": stats.ratio(cand["p99_us"], ref["p99_us"]),
                     "calls_per_side": int(calls), "distinct_syndromes": int(calls),
                     "syndrome_seeds": {label: int(seed) for label, seed in seeds.items()},
                     "budget": _budget_status(cand, cfg)}
            out[point] = entry
            log(f"[retime] {point.label()}: p50 {cand['p50_us']:.1f} us, p99 "
                f"{cand['p99_us']:.1f} us ({calls} calls per side; reference p99 "
                f"{ref['p99_us']:.1f} us): within budget: {entry['budget']['within']}")
    finally:
        for decoder in ref_handles.values():
            decoder.close()
    return out


def decide_budget_membership(cfg: TuneConfig, rows: list[dict], plan: dict,
                             retimed: dict) -> tuple[dict, dict]:
    """The in/out decision of every latency row: ({point: membership}, the results.json record).

    Outside the margin the latency-stage verdict stands; inside it the re-timing decides; a point
    inside the margin beyond the re-timing cap keeps its latency-stage verdict and is listed.
    """
    low, high = RETIME_MARGIN
    membership: dict = {}

    def entry(row, ratio, basis, budget, retime=None):
        decided = dict(budget)
        decided["basis"] = basis
        membership[row["point"]] = {"budget": decided, "basis": basis, "screen_ratio": ratio,
                                    "screen_budget": row["budget"], "retime": retime}

    for row, ratio in plan["decided"]:
        if ratio is None:
            basis = "latency stage (no latency budget)"
        elif ratio < low:
            basis = f"latency stage (below {low:g}x the budget)"
        else:
            basis = f"latency stage (above {high:g}x the budget)"
        entry(row, ratio, basis, row["budget"])
    for row, ratio in plan["retime"]:
        measured = retimed[row["point"]]
        entry(row, ratio, "re-timed", measured["budget"], measured)
    for row, ratio in plan["capped"]:
        entry(row, ratio, "latency stage (inside the margin, beyond the re-timing cap)",
              row["budget"])
    record = {
        "margin": [float(low), float(high)],
        "max_points": int(RETIME_MAX_POINTS),
        "rule": (f"every screen survivor (a screen pass with --ler-target, a point above the "
                 f"LER floor without) whose latency-stage measurement is within "
                 f"{low:g}x-{high:g}x of a budget (largest measured/budget ratio) is re-timed "
                 f"before the in/out decision: at least {CONFIRMATION_MIN_SYNDROMES} calls per "
                 f"side and case in a full run, on fresh syndromes, paired with the A/A "
                 f"reference in the ABBA schedule; the decision is that measurement's (with the "
                 f"slow-class rule of a p99 budget); at most {RETIME_MAX_POINTS} points, the "
                 f"nearest to the budget first; outside the margin the latency-stage verdict "
                 f"stands"),
        "retimed": [{"label": row["point"].label(), "screen_ratio": ratio,
                     "screen_p50_us": row["p50"], "screen_p99_us": row["p99"],
                     "screen_within": bool(row["budget"]["within"]),
                     "p50_us": retimed[row["point"]]["cand"]["p50_us"],
                     "p99_us": retimed[row["point"]]["cand"]["p99_us"],
                     "ref_p99_us": retimed[row["point"]]["ref"]["p99_us"],
                     "within": bool(retimed[row["point"]]["budget"]["within"]),
                     "measurement": retimed[row["point"]]}
                    for row, ratio in plan["retime"]],
        "capped": [{"label": row["point"].label(), "screen_ratio": ratio,
                    "within": bool(row["budget"]["within"])} for row, ratio in plan["capped"]],
        "decided_by_latency_stage": len(plan["decided"]),
    }
    return membership, record


def budget_retime_stage(cfg: TuneConfig, cases: list[dict], results: dict, latency_stage: dict,
                        log, retime=None) -> dict | None:
    """Plan, measure and decide budget membership; stores it in ``latency_stage["_membership"]``.

    ``retime(points, reference)`` measures the points (default: :func:`run_budget_retime`).
    """
    rows = latency_rows(cfg, results, latency_stage)
    if not rows:
        return None
    if cfg.p50_budget_us is None and cfg.p99_budget_us is None:
        return {"note": "no latency budget: nothing to decide", "retimed": [], "capped": []}
    plan = plan_budget_retime(rows)
    reference = latency_stage.get("_reference_point") or rows[0]["point"]
    points = [row["point"] for row, _ratio in plan["retime"]]
    if plan["retime"] or plan["capped"]:
        log(f"[retime] {len(points)} point(s) within {RETIME_MARGIN[0]:g}x-{RETIME_MARGIN[1]:g}x "
            f"of the budget are re-timed against {reference.label()}"
            + (f"; {len(plan['capped'])} beyond the cap of {RETIME_MAX_POINTS} keep their "
               f"latency-stage verdict" if plan["capped"] else ""))
    started = time.perf_counter()
    measure = retime if retime is not None else (
        lambda pts, ref: run_budget_retime(cfg, cases, pts, ref, log))
    retimed = measure(points, reference) if points else {}
    membership, record = decide_budget_membership(cfg, rows, plan, retimed)
    record["reference"] = reference.label()
    record["wall_seconds"] = time.perf_counter() - started
    latency_stage["_membership"] = membership
    log(f"[retime] {len(points)} re-timed in {record['wall_seconds']:.1f} s")
    return record


def screen_claim(record: dict) -> dict:
    """The LER claim of a run without --ler-target: the screen's fixed-size estimate.

    Worst case over the cases: the largest LER, with the largest of the lower and of the upper
    one-sided Clopper-Pearson bounds (each at the tuner's confidence) as its interval.
    """
    per_case = [_case_numbers(entry) for entry in record.get("per_p") or []]
    return {"ler": max(case["ler"] for case in per_case),
            "lower": max(float(case["cp99_lower"]) for case in per_case),
            "upper": max(float(case["cp99_upper"]) for case in per_case),
            "shots_per_case": min(case["shots"] for case in per_case),
            "per_case": per_case,
            "source": "screen",
            "interval": ("one-sided Clopper-Pearson bounds at the tuner's confidence, worst case "
                         "over the cases")}


def confirmation_consistency(record: dict, entry: dict) -> dict:
    """The small accuracy confirmation as a consistency check of the screen's estimate: the
    screen LER must lie inside the confirmation's one-sided bounds at every case."""
    screen = {e["case"]: _case_numbers(e) for e in record.get("per_p") or []}
    checks = []
    for case in entry.get("per_p") or []:
        numbers = screen.get(case["label"])
        ler = numbers["ler"] if numbers else None
        checks.append({"case": case["label"], "screen_ler": ler,
                       "confirmation_failures": case["failures"],
                       "confirmation_shots": case["shots"],
                       "confirmation_ler": case["ler"], "lower": case["lower"],
                       "upper": case["upper"],
                       "consistent": bool(ler is not None
                                          and case["lower"] <= ler <= case["upper"])})
    return {"consistent": bool(checks) and all(check["consistent"] for check in checks),
            "rule": ("consistent when the screen LER lies inside the confirmation's one-sided "
                     "Clopper-Pearson bounds (tuner's confidence) at every case"),
            "per_case": checks}


def _ranking_key(cfg: TuneConfig):
    """The order of candidates in a run without --ler-target: the weight score (or p50, p99)."""
    def key(row):
        point = row["point"]
        if cfg.weight is not None:
            score = (stats.log10_or_floor(row["ler_upper"])
                     + float(cfg.weight) * stats.log10_or_floor(row["p50"], 1e-9))
            return (score, row["ler_upper"], point.max_iter, point.alpha, point.label())
        return (row["p50"], row["p99"], point.max_iter, point.alpha, point.label())
    return key


def nearest_after_confirmation(cfg: TuneConfig, rows: list[dict], confirmed: list[dict],
                               limit: int = 5) -> list[dict]:
    """The nearest list when the confirmation accepted no candidate (both paths).

    Every measured screen survivor (above the LER floor without --ler-target) by its budget
    distance; a confirmed candidate by its latency confirmation, with ``accuracy_passed`` from its
    accuracy confirmation (always true without --ler-target).  A candidate that failed the
    accuracy confirmation sorts after every point that did not; ``config.nearest.json`` is the
    first entry.
    """
    by_label = {entry["label"]: entry for entry in confirmed}
    items = []
    for row in rows:
        label = row["point"].label()
        entry = by_label.get(label)
        if entry is not None:
            summary, budget, basis = entry["latency"], entry["latency_budget"], \
                "latency confirmation"
            calls = int(entry["latency"].get("distinct_syndromes")
                        or entry["latency"].get("calls") or 0)
            measurement = f"latency confirmation, {calls} calls"
            accuracy_passed = bool(entry["passed"])
        else:
            # p50 / p99 of the measurement the verdict rests on (the re-timing of a re-timed
            # point), never the latency-stage numbers next to a re-timed ratio
            summary, budget = row["decision"], row["budget"]
            calls, measurement = row["decision"]["calls"], row["decision"]["measurement"]
            basis = budget.get("basis") or "latency stage"
            accuracy_passed = None
        distance = budget_ratio(budget)
        item = {"point": row["point"].as_dict(), "label": label,
                "status": row["record"]["status"], "ler_upper": row["ler_upper"],
                "p50_us": summary["p50_us"], "p99_us": summary["p99_us"],
                "calls": calls, "measurement": measurement,
                "budget": budget, "budget_distance": distance, "basis": basis}
        if cfg.ler_target is not None and entry is not None:
            item["accuracy_passed"] = accuracy_passed
            item["confirmation_ler_upper"] = entry["upper"]
        items.append(item)
    items.sort(key=lambda item: (item.get("accuracy_passed") is False,
                                 item["budget_distance"] if item["budget_distance"] is not None
                                 else float("inf"), item["p50_us"], item["label"]))
    return items[:limit]


def candidate_order(cfg: TuneConfig, results: dict, latency_stage: dict,
                    screen_choice: dict) -> list[dict]:
    """Every candidate row the confirmation may measure, in the order the choice ranks them.

    The candidates are the screen survivors within the latency budget (after the re-timing): the
    screen choice's point first, then the other in-budget points, with --ler-target fastest p50
    first (then the tie order), without it in the choice's ranking (the weight score, or p50 then
    p99).  A point over the budget is never a candidate.  The confirmation measures them in this
    order until one is accepted or the list is exhausted (at most ``--confirm-candidates`` when
    that cap is given); the choice is the first accepted one.
    """
    rows = latency_rows(cfg, results, latency_stage)
    if not rows:
        return []
    by_point = {row["point"]: row for row in rows}

    def key(row):
        return (row["p50"], row["p99"], row["point"].max_iter, row["point"].alpha,
                row["point"].label())

    ordered, taken = [], set()
    chosen = screen_choice.get("point")
    if chosen is not None:
        point = grid.Point(chosen["decoder"], chosen["ms_scaling_factor"],
                           chosen["max_iter"])
        if point in by_point and by_point[point]["budget"]["within"]:
            ordered.append(by_point[point])
            taken.add(point)
    rank = _ranking_key(cfg) if cfg.ler_target is None else key
    for row in sorted((r for r in rows if r["budget"]["within"]), key=rank):
        if row["point"] not in taken:
            ordered.append(row)
            taken.add(row["point"])
    return ordered


def confirmation_failure_reasons(cfg: TuneConfig, entry: dict, lat: dict) -> list[str]:
    """Why one confirmed candidate was not accepted (empty when it was): its accuracy
    confirmation against the target (runs with --ler-target), its latency confirmation against
    each budget."""
    reasons = []
    if cfg.ler_target is not None and not entry["passed"]:
        worst = max(entry["per_p"], key=lambda e: e["upper"])
        reasons.append(f"accuracy: upper bound {worst['upper']:.3g} is not below the target "
                       f"{float(cfg.ler_target):g} ({worst['label']}: {worst['failures']}/"
                       f"{worst['shots']} failures)")
    calls = int(lat.get("distinct_syndromes") or 0)
    for name, check in sorted((lat["budget"].get("checks") or {}).items()):
        if check["within"]:
            continue
        if check.get("p99_unreliable"):
            median = check.get("slow_median_us")
            reasons.append(f"latency: p99 {check['measured_us']:.1f} us is unreliable and the "
                           f"slow-call median "
                           f"{'not recorded' if median is None else f'{float(median):.1f} us'} "
                           f"is over the {check['budget_us']:g} us budget (latency "
                           f"confirmation, {calls} calls)")
        else:
            reasons.append(f"latency: {name} {check['measured_us']:.1f} us is over the "
                           f"{check['budget_us']:g} us budget (latency confirmation, {calls} "
                           f"calls)")
    return reasons


def run(cfg: TuneConfig, log=print) -> dict:
    """Execute the procedure and return the results dictionary."""
    cfg.validate()
    settings = stage_settings(cfg)
    started = time.perf_counter()
    environment = env.capture_environment()
    cases = build_cases(cfg)
    for case in cases:
        if case["problem"].k < 1:
            raise ValueError(f"{case['label']}: the problem has no logical observables, so a "
                             f"logical error rate cannot be measured")
        if case["problem"].n < 1:
            raise ValueError(f"{case['label']}: the problem has no error mechanisms (a noiseless "
                             f"DEM or circuit)")
    if cfg.h and cfg.o is None:
        log("[warn] the observables were derived from H as a basis of ker(H)/row(H), which is a "
            "logical basis only when H is the full commuting stabilizer matrix of the code; for "
            "a CSS half such as Hx pass the observable matrix explicitly with --o")
    target_reachable = True
    if settings.target is not None:
        floor_upper = stats.clopper_pearson_upper(0, settings.max_shots, settings.confidence)
        target_reachable = bool(floor_upper < settings.target)
        if not target_reachable:
            log(f"[warn] with {settings.max_shots} shots the smallest provable upper bound is "
                f"{floor_upper:.3g}; no point can meet the target {settings.target:.3g}")

    pruning = pruning_record(cfg)
    coarse = grid.coarse_points(cfg.quick)
    log(f"[plan] problem cases: {', '.join(c['label'] for c in cases)}; "
        f"coarse grid: {len(coarse)} points; max {settings.max_shots} shots per point per case")
    if pruning["enabled"]:
        log(f"[plan] pruning: a larger cap is skipped only when the passing point's "
            f"non-converged share is below {1.0 - pruning['min_converged']:.3g} at confidence "
            f"{pruning['confidence']:.3g}")
    else:
        log("[plan] pruning disabled (--no-prune): every point is evaluated")

    checkpoint_path = os.path.join(cfg.out, "raw", "checkpoint.json")
    fingerprint = run_fingerprint(cfg, cases, settings)
    checkpoint = None
    if cfg.resume and os.path.isfile(checkpoint_path):
        checkpoint = Checkpoint.load(checkpoint_path, fingerprint)
        done = [name for name, stage in checkpoint.data.get("stages", {}).items()
                if stage.get("complete")]
        log(f"[resume] loading {checkpoint_path}"
            + (f"; completed stages: {', '.join(sorted(done))}" if done else ""))
    elif cfg.resume:
        log("[resume] no checkpoint found; starting from the beginning")
    if checkpoint is None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = Checkpoint(checkpoint_path, fingerprint)

    floor_references = floor_reference_points(cfg)
    coarse_stage = run_accuracy_stage("coarse", coarse, cases, settings, cfg, log,
                                      workers=cfg.workers, checkpoint=checkpoint,
                                      pruning=pruning, reference_points=floor_references)

    coarse_summary = {
        p: {"status": r["status"], "upper": r.get("cp99_upper_max", 1.0)}
        for p, r in _stage_points(coarse_stage).items() if r["status"] != "skipped"
    }
    already = {p for p, r in _stage_points(coarse_stage).items()}
    refined_points = grid.refine_points(cfg.quick, coarse_summary, already)
    refine_stage = None
    if refined_points:
        log(f"[plan] refinement grid: {len(refined_points)} points around the frontier")
        refine_stage = run_accuracy_stage("refine", refined_points, cases, settings, cfg, log,
                                          prior_results=_stage_points(coarse_stage),
                                          workers=cfg.workers, checkpoint=checkpoint,
                                          pruning=pruning, reference_points=floor_references)
    else:
        log("[plan] refinement adds no points")

    results = dict(_stage_points(coarse_stage))
    if refine_stage is not None:
        results.update(_stage_points(refine_stage))

    stages = [s for s in (coarse_stage, refine_stage) if s is not None]
    ler_floor = ler_floor_record(cfg, stages)
    if ler_floor is not None:
        apply_ler_floor(results, ler_floor, log)

    latency_stage = run_latency_stage(cfg, cases, results, log)
    # Both paths: every screen survivor near a budget is re-timed before the in/out decision.
    budget_retime = budget_retime_stage(cfg, cases, results, latency_stage, log)
    screen_choice = choose(cfg, results, latency_stage)
    screen_pick = (screen_choice.get("point") or {}).get("label")

    # -- accuracy confirmation: fixed N fresh shots per candidate and case, no early stop -------
    planned, shots_rule = confirmation.planned_shots(
        cfg.ler_target, cfg.confidence, cfg.quick, cfg.confirm_shots)
    ac_cases = []
    case_seeds = {}
    for case in cases:
        seed = stats.derive_seed(cfg.seed, "confirm", case["seed_key"])
        case_seeds[case["label"]] = seed
        ac_cases.append({"label": case["label"], "p": case["p"], "seed": seed})
    selected = candidate_order(cfg, results, latency_stage, screen_choice)
    cap = None if cfg.confirm_candidates is None else int(cfg.confirm_candidates)
    log(f"[confirm] plan: {planned} shots per candidate and case ({shots_rule}); "
        f"{len(selected)} in-budget candidate(s), confirmed in order until one is accepted"
        + (f" (at most {cap}: --confirm-candidates)" if cap is not None else "")
        + (f": {', '.join(row['point'].label() for row in selected)}" if selected else ""))
    confirmation_path = os.path.join(cfg.out, "raw", "confirmation-checkpoint.json")
    confirmation_checkpoint = None
    if cfg.resume and os.path.isfile(confirmation_path):
        confirmation_checkpoint = ConfirmationCheckpoint.load(
            confirmation_path, fingerprint, planned)
        saved = confirmation_checkpoint.progress() or {}
        saved_shots = sum(int(value.get("shots", 0))
                          for per_case in (saved.get("candidates") or {}).values()
                          for value in per_case.values())
        log(f"[resume] confirmation progress loaded from {confirmation_path}: "
            f"{saved_shots} case-shots already decoded")
    elif cfg.resume:
        log("[resume] no confirmation checkpoint found; the confirmation starts from the "
            "beginning")
    if confirmation_checkpoint is None:
        confirmation_checkpoint = ConfirmationCheckpoint(confirmation_path, fingerprint, planned)
    # Both paths: confirm the candidates one at a time in the choice's order (accuracy, then
    # latency) and stop at the first accepted one; the in-budget list (or the cap) ends it.
    confirm_entries: list[dict] = []
    latencies: dict[str, dict] = {}
    final_entry = None
    stopped = "exhausted" if selected else "no candidate"
    progress = confirmation_checkpoint.progress() if cfg.resume else None
    for rank, row in enumerate(selected, 1):
        if cap is not None and rank > cap:
            stopped = "cap"
            break
        entry = confirmation.run_accuracy_confirmation(
            _make_decoder, [(row["point"], screen_numbers(row["record"]))],
            cases, case_seeds, planned, cfg.confidence, log,
            progress=progress, save_progress=confirmation_checkpoint.save)[0]
        # the saved state now holds every candidate confirmed so far (a --resume reuses it)
        progress = confirmation_checkpoint.progress()
        entry["rank"] = rank
        lat = run_latency_confirmation(cfg, cases, row["point"], log)
        latencies[entry["label"]] = lat
        entry["latency"] = confirmation_latency_record(lat)
        entry["latency_within"] = bool(lat["budget"]["within"])
        entry["latency_budget"] = lat["budget"]
        if cfg.ler_target is None:
            entry["passed"] = True   # no accuracy bar; the LER floor was applied on the screen
            entry["consistency"] = confirmation_consistency(row["record"], entry)
        else:
            entry["passed"] = bool(all(e["upper"] < float(cfg.ler_target)
                                       for e in entry["per_p"]))
        entry["accepted"] = bool(entry["passed"] and entry["latency_within"])
        entry["failure_reasons"] = confirmation_failure_reasons(cfg, entry, lat)
        confirm_entries.append(entry)
        log(f"[confirm] candidate {rank} of {len(selected)} {entry['label']}: "
            + ("accepted" if entry["accepted"] else
               "not accepted: " + "; ".join(entry["failure_reasons"])))
        if entry["accepted"]:
            final_entry = entry
            stopped = "accepted"
            break
    unconfirmed = ([row["point"].label() for row in selected[len(confirm_entries):]]
                   if stopped == "cap" else [])
    final_latency = latencies.get(final_entry["label"]) if final_entry is not None else None

    accuracy_confirmation = {
        "planned_shots": int(planned),
        "shots_rule": shots_rule,
        "confidence": float(cfg.confidence),
        "target": None if cfg.ler_target is None else float(cfg.ler_target),
        "max_candidates": cap,
        "in_budget_candidates": len(selected),
        "in_budget_order": [row["point"].label() for row in selected],
        "confirmed": len(confirm_entries),
        "stopped": stopped,
        "stop_rule": ("the candidates are confirmed one at a time in the choice's order and the "
                      "first accepted one ends the confirmation (stopped: accepted); otherwise "
                      "every in-budget candidate is confirmed (exhausted), unless "
                      "--confirm-candidates caps the count (cap: the rest are listed in "
                      "unconfirmed); no candidate: nothing was within the budget"),
        "unconfirmed": unconfirmed,
        "cases": ac_cases,
        "candidates": confirm_entries,
        "chosen": final_entry["label"] if final_entry is not None else None,
        "no_early_stop": True,
        "resumed_case_shots": sum(int(entry.get("resumed_case_shots") or 0)
                                  for entry in confirm_entries),
        "seed_stage": "confirm",
        "candidate_order": [entry["label"] for entry in confirm_entries],
        "candidate_rule": ("the screen passes within the latency budget (after the re-timing): "
                           "the screen choice's point first, then the other in-budget points by "
                           "p50 (then the tie order); a point over the budget is never "
                           "confirmed"),
        "verdict_rule": ("passed: the one-sided Clopper-Pearson upper bound at the tuner's "
                         "confidence is below the target at every case; accepted: passed and the "
                         "latency confirmation is within the budget"),
    }
    if cfg.ler_target is None:
        accuracy_confirmation["candidate_rule"] = (
            "the screen choice's point first, then the other points above the LER floor and "
            "within the latency budget (after the re-timing) in the choice's ranking; a point "
            "over the budget is never confirmed")
        accuracy_confirmation["verdict_rule"] = (
            "no accuracy bar: passed is always true (the LER floor was applied on the screen); "
            "accepted: the latency confirmation is within the budget; the confirmation's "
            "failures are a consistency check of the screen's estimate (consistency), not the "
            "reported LER")

    # -- the choice: the first accepted confirmation candidate ---------------------------------
    no_target_warnings: list[str] = []
    if cfg.ler_target is None and final_entry is None and confirm_entries:
        # Weight / fastest mode: no confirmed candidate is within the budget, so nothing is
        # chosen; the nearest list is every choosable point by its budget distance.
        choice = {
            "target_present": False,
            "eligible": screen_choice.get("eligible"),
            "measured": screen_choice.get("measured"),
            "config": None,
            "point": None,
            "reason": ("no confirmed candidate is within the latency budget: every candidate's "
                       "latency confirmation is over it"
                       + (f" (--confirm-candidates {cap} left {len(unconfirmed)} in-budget "
                          f"candidate(s) unconfirmed)" if unconfirmed else "")),
            "ler_source": "screen",
            "screen_choice_label": screen_pick,
            "confirmation_candidates": [
                {"label": e["label"], "latency_within": e["latency_within"],
                 "accepted": e["accepted"], "p50_us": e["latency"]["p50_us"],
                 "p99_us": e["latency"]["p99_us"], "screen": e["screen"],
                 "failure_reasons": e["failure_reasons"]}
                for e in confirm_entries],
            "nearest": nearest_after_confirmation(
                cfg, latency_rows(cfg, results, latency_stage), confirm_entries),
        }
    elif cfg.ler_target is not None and final_entry is None:
        choice = {
            "target_present": True,
            "eligible": screen_choice.get("eligible"),
            "measured": screen_choice.get("measured"),
            "config": None,
            "point": None,
            "reason": (("no candidate passed the fixed-shot accuracy confirmation within the "
                        "latency budget"
                        + (f" (--confirm-candidates {cap} left {len(unconfirmed)} in-budget "
                           f"candidate(s) unconfirmed)" if unconfirmed else ""))
                       if confirm_entries else
                       screen_choice.get("reason") or "no screen survivor was available to "
                                                      "confirm"),
            "ler_source": "accuracy_confirmation",
            "screen_choice_label": screen_pick,
            "confirmation_candidates": [
                {"label": e["label"], "passed": e["passed"],
                 "latency_within": e["latency_within"], "accepted": e["accepted"],
                 "ler": e["ler"], "ler_upper": e["upper"],
                 "screen": e["screen"], "failure_reasons": e["failure_reasons"]}
                for e in confirm_entries],
        }
        if screen_choice.get("nearest"):
            choice["nearest"] = screen_choice["nearest"]
        elif confirm_entries:
            # the screen had in-budget points, but the confirmation accepted none: the same
            # nearest list (and so config.nearest.json) as a run without --ler-target
            choice["nearest"] = nearest_after_confirmation(
                cfg, latency_rows(cfg, results, latency_stage), confirm_entries)
    else:
        choice = dict(screen_choice)
        if final_entry is not None:
            choice["point"] = final_entry["point"]
            choice["config"] = dict(final_entry["point"]["config"])
            choice["ler"] = final_entry["ler"]
            choice["ler_upper"] = final_entry["upper"]
            choice["ler_source"] = "accuracy_confirmation"
            choice["screen"] = final_entry["screen"]
            choice["confirmation_label"] = final_entry["label"]
            choice["p50_us"] = final_latency["summary"]["p50_us"]
            choice["p99_us"] = final_latency["summary"]["p99_us"]
            choice["budget"] = final_latency["budget"]
            choice["slow_fraction"] = final_latency["summary"]["slow_fraction"]
            choice["p99_unreliable"] = final_latency["summary"]["p99_unreliable"]
            choice["reason"] = (
                "the first candidate in the choice's order that passed the fixed-shot accuracy "
                "confirmation at every case and whose latency confirmation is within the budget"
                if cfg.ler_target is not None else
                "the screen choice; with no accuracy bar the confirmation reports the point and "
                "does not change the choice")
            # The marks follow the accepted choice; the screen's own pick keeps its mark
            # ("screen pick") when the confirmation passed it over
            choice["screen_choice_label"] = screen_pick
            choice["tie_set"] = [dict(tie, chosen=tie["label"] == final_entry["label"],
                                      screen_pick=tie["label"] == screen_pick)
                                 for tie in screen_choice.get("tie_set") or []]
            if cfg.ler_target is None:
                final_point = _record_point(final_entry)
                claim = screen_claim(results[final_point])
                choice["ler"] = claim["ler"]
                choice["ler_lower"] = claim["lower"]
                choice["ler_upper"] = claim["upper"]
                choice["ler_interval"] = [claim["lower"], claim["upper"]]
                choice["ler_shots_per_case"] = claim["shots_per_case"]
                choice["ler_per_case"] = claim["per_case"]
                choice["ler_source"] = "screen"
                choice["ler_claim_rule"] = (
                    "without --ler-target the reported LER is the screen's fixed-size estimate "
                    "(every point decodes the full --max-shots, no early stop, so it is unbiased) "
                    "with its one-sided Clopper-Pearson bounds, worst case over the cases; the "
                    "accuracy confirmation's small sample is a consistency check")
                choice["confirmation_check"] = {
                    "shots_per_case": int(planned), "failures": final_entry["failures"],
                    "ler": final_entry["ler"], "upper": final_entry["upper"],
                    **final_entry["consistency"]}
                choice["ler_floor"] = (results[final_point].get("ler_floor") or {})
                choice["reason"] = (
                    "the first candidate in the choice's ranking (above the LER floor and within "
                    "the budget after the re-timing) whose latency confirmation is within the "
                    "budget")
                if cfg.weight is not None:
                    ranking_p50 = latency_stage["_measured_points"][final_point]["cand"]["p50_us"]
                    choice["score"] = (stats.log10_or_floor(claim["upper"]) + float(cfg.weight)
                                       * stats.log10_or_floor(ranking_p50, 1e-9))
                passed_over = [e for e in confirm_entries if e is not final_entry
                               and e["rank"] < final_entry["rank"]]
                if passed_over:
                    choice["passed_over"] = [
                        {"label": e["label"], "p50_us": e["latency"]["p50_us"],
                         "p99_us": e["latency"]["p99_us"], "latency_within": False}
                        for e in passed_over]
                    no_target_warnings.append(
                        f"{', '.join(e['label'] for e in passed_over)}: ranked above the choice "
                        f"but over the budget by its latency confirmation; the choice is the next "
                        f"confirmed candidate, {final_entry['label']}")
                if not final_entry["consistency"]["consistent"]:
                    no_target_warnings.append(
                        f"{final_entry['label']} (chosen): the accuracy confirmation "
                        f"({final_entry['failures']}/{final_entry['shots']} failures) is not "
                        f"consistent with the screen's estimate at the tuner's confidence; rerun "
                        f"with another --seed or more --confirm-shots before relying on the LER")

    confirm_latency = final_latency if final_entry is not None else None
    keep_warm = None
    core_variants = None
    if choice.get("config") is not None:
        chosen_point = grid.Point(choice["point"]["decoder"],
                                  choice["point"]["ms_scaling_factor"],
                                  choice["point"]["max_iter"])
        keep_warm = run_keep_warm(cfg, cases, chosen_point,
                                  latency_stage.get("aa") or {}, log)
        core_variants = run_core_variant_comparison(
            cfg, cases, chosen_point, int(latency_stage.get("calls_top") or 400),
            int(latency_stage.get("aa_cycles") or 2), log)

    pareto_rows = _pareto_rows(results, latency_stage, cfg)
    warnings = p99_warnings(accuracy_confirmation, pareto_rows)
    if budget_retime and budget_retime.get("capped"):
        capped = budget_retime["capped"]
        warnings.append(
            f"{len(capped)} point(s) inside the re-timing margin were not re-timed (cap "
            f"{RETIME_MAX_POINTS}, nearest to the budget first) and keep their latency-stage "
            f"verdict: {', '.join(item['label'] for item in capped)}")
    if unconfirmed:
        warnings.append(
            f"--confirm-candidates {cap} stopped the confirmation after {cap} candidate(s) with "
            f"{len(unconfirmed)} in-budget candidate(s) never confirmed "
            f"({', '.join(unconfirmed)}): no configuration was accepted, but one of them may "
            f"qualify; rerun without --confirm-candidates to confirm every in-budget candidate")
    warnings += no_target_warnings
    for warning in warnings:
        log(f"[warn] {warning}")
    results_out = {
        "schema": "eldpc.tune.v2",
        "tool": {"name": "eldpc-tune", "version": __version__, "seed": int(cfg.seed),
                 "quick": bool(cfg.quick), "auto": bool(cfg.auto),
                 "command": list(getattr(cfg, "command", [])),
                 "started_utc": environment["timestamp_utc"],
                 "duration_seconds": time.perf_counter() - started},
        "environment": environment,
        "inputs": {
            "problem": cfg.problem, "dem": cfg.dem, "circuit": cfg.circuit, "h": cfg.h,
            "o": cfg.o, "p": [float(p) for p in cfg.ps] if cfg.ps else [],
            "rounds": cfg.rounds, "noise_model": cfg.noise_model,
            "p50_budget_us": cfg.p50_budget_us,
            "p99_budget_us": cfg.p99_budget_us, "ler_target": cfg.ler_target,
            "max_ler": cfg.max_ler,
            "weight": cfg.weight, "cadence_us": cfg.cadence_us,
            "confidence": cfg.confidence, "max_shots": settings.max_shots,
            "max_shots_auto": auto_max_shots(cfg) is not None,
            "chunk_shots": settings.chunk_shots,
            "coarse_grid_points": len(coarse),
            "workers": int(cfg.workers),
            "resume": bool(cfg.resume),
            "refinement_points": len(refined_points),
            "target_reachable_with_max_shots": target_reachable,
            "confirm_shots": None if cfg.confirm_shots is None else int(cfg.confirm_shots),
            "confirm_candidates": cap,
            "no_prune": bool(cfg.no_prune),
            "decompose": bool(cfg.decompose),
        },
        "problem_cases": [c["problem"].summary() for c in cases],
        "pruning": pruning,
        "stages": stages,
        "latency": {k: v for k, v in latency_stage.items() if not k.startswith("_")},
        "ler_floor": ler_floor,
        "budget_retime": budget_retime,
        "accuracy_confirmation": accuracy_confirmation,
        "confirmation": confirm_latency,
        "keep_warm": keep_warm,
        "pareto": pareto_rows,
        "choice": choice,
        "core_variants": core_variants,
        "warnings": warnings,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return results_out


def _stage_points(stage: dict) -> dict[grid.Point, dict]:
    out = {}
    for record in stage["points"]:
        point = grid.Point(record["point"]["decoder"],
                           record["point"]["ms_scaling_factor"],
                           record["point"]["max_iter"])
        out[point] = record
    return out
