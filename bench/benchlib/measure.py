# SPDX-License-Identifier: Apache-2.0
"""The four measurements of the eldpc benchmark and the paired A/A design.

Three latency numbers are kept apart:
  (a) c_latency   -- eldpc_decode end to end: one C call per shot, transfers and synchronisation included
  (b) py_latency  -- Decoder.decode per shot from Python
  (c) throughput  -- decode_batch, shots per second
and (d) ler -- the logical error rate over sampled shots, with a 95% Wilson interval.

Every comparison (two configs, two builds, keep-warm on/off) runs both sides in the same sessions,
alternating the call order shot by shot and alternating which side is created first between sessions.
When a side runs on the CPU core, a session holds one live handle at a time instead (its worker
threads busy-wait, so two live CPU handles would compete for the pinned CPUs): each handle is timed
in its own block, and the session order alternates which side goes first (HANDLE_SCHEDULES).
The same machinery run as A against A gives the noise band: the spread of the difference statistic that
two identical measurements produce on this machine.  A reported difference is only meaningful if it is
larger than that band.

All timing statistics are pooled over the sessions of a pair (per-shot samples concatenated), reported as
nearest-rank p50/p99/p99.9 and the mean, in microseconds.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import cdriver, formats, problems, stats


class MeasurementError(RuntimeError):
    pass


# (quick, full) settings of the shot counts and session counts.  --quick is meant to finish in a minute or
# two on a laptop-class GPU; the default spends more shots for steadier p99.9 and LER numbers.
SETTINGS = {
    # An even session count keeps the two session orders equal in number, so the first-created handle's
    # order effect cancels from the pooled difference instead of biasing it (an odd count weighs one
    # order 2:1 or more).
    "quick": {
        "latency_shots": 2048, "c_shots": 300, "processes": 4, "warmup": 500, "warmup_ms": 800,
        "ler_shots": 4000, "throughput_batch": 1024, "throughput_reps": 16,
    },
    "full": {
        "latency_shots": 8192, "c_shots": 1500, "processes": 6, "warmup": 2000, "warmup_ms": 1500,
        "ler_shots": 40000, "throughput_batch": 4096, "throughput_reps": 24,
    },
}
GAP_US = 1000


def settings(quick: bool) -> dict:
    return dict(SETTINGS["quick" if quick else "full"])


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def library_ref(library: dict) -> dict:
    """A machine-independent identity of a measured library: its name and content hash.

    Absolute paths differ between machines and installations; the content hash is what identifies
    the binary in a results file, and compare.py already checks the problem, config and batch size
    rather than the path.
    """
    path = library.get("path")
    ref = {"source": library.get("source")}
    if path and os.path.isfile(path):
        ref["name"] = os.path.basename(path)
        try:
            ref["sha256"] = sha256_file(path)
        except OSError:
            pass
    elif path:
        ref["name"] = os.path.basename(str(path))
    return ref


def session_purpose(mode: str, pair: str, session: int) -> str:
    """The sampling purpose of one latency session; it makes the session's seed unique."""
    return f"latency:{mode}:{pair}:{int(session)}"


#: Distinct syndromes per handle per pair that a p99 budget should rest on.
FULL_DISTINCT_SYNDROMES = 8000


def control_sessions(processes: int, c_shots: int, quick: bool,
                     default_counts: bool) -> int:
    """Session count of the keep-warm pair's two A/A controls.

    The controls need only enough sessions for their noise band, so they run half as many as
    the treatment comparison.  In a full run with the default counts they are raised to the
    number of sessions that puts at least 8000 distinct syndromes behind each handle of a pair;
    an explicit --shots or --processes keeps the cost the user asked for.
    """
    sessions = max(2, int(processes) // 2)
    if not quick and default_counts:
        needed = (FULL_DISTINCT_SYNDROMES + int(c_shots) - 1) // max(1, int(c_shots))
        sessions = max(sessions, int(needed))
    return sessions


def syndrome_record(records: list[dict]) -> dict:
    """The ``syndromes`` block of a latency measurement.

    ``records`` are one dict per timed session with mode, pair, session, shots, seed and sha256.
    ``distinct`` is the number of distinct sampled shots timed per handle over the sessions of a
    pair -- every session draws its own seeded shots, and both handles of a session decode the
    same set -- taken as the minimum over pairs.
    """
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        groups.setdefault((record.get("mode"), record.get("pair")), []).append(record)
    per_pair = {key: sum(int(r["shots"]) for r in value) for key, value in groups.items()}
    distinct = min(per_pair.values()) if per_pair else 0
    return {
        "distinct": int(distinct),
        "distinct_definition": ("distinct sampled shots timed per handle over the sessions of a "
                                "pair, minimum over pairs; each session draws its own seeded "
                                "shots and both handles of a session decode the same set"),
        "per_session": [{"mode": r.get("mode"), "pair": r.get("pair"),
                         "session": int(r["session"]), "shots": int(r["shots"]),
                         "seed": int(r["seed"]), "sha256": r["sha256"]} for r in records],
    }


def config_string_from_dict(cfg: dict) -> str:
    """The ABI config string of a preset dict: keys sorted, bools true/false, floats repr()."""
    parts = []
    for k in sorted(cfg):
        v = cfg[k]
        if isinstance(v, bool):
            s = "true" if v else "false"
        elif isinstance(v, float):
            s = repr(v)
        else:
            s = str(v)
        parts.append(f"{k}={s}")
    return ";".join(parts)


def _presets(eldpc_module) -> dict:
    try:
        return eldpc_module.presets()
    except Exception as e:
        raise MeasurementError(f"cannot read the package's presets: {e}") from e


def resolve_config(name: str, eldpc_module) -> tuple[str, str]:
    """(label, ABI config string) for a preset name, an ABI string or a JSON config file.

    A preset name may be followed by extra ``key=value`` segments ("a03125c3;threads=8"): they add to
    the preset's keys or override them, and the result is the full ABI string in key order.
    """
    segments = [seg.strip() for seg in name.split(";") if seg.strip()]
    if ";" in name and segments and "=" not in segments[0]:
        presets = _presets(eldpc_module)
        if segments[0] not in presets:
            raise MeasurementError(f"unknown preset {segments[0]!r} in {name!r}; preset names are: "
                                   f"{', '.join(sorted(presets))}")
        cfg = dict(presets[segments[0]])
        seen = set()
        for seg in segments[1:]:
            key, eq, value = seg.partition("=")
            key = key.strip()
            if not eq or not key:
                raise MeasurementError(f"malformed config segment {seg!r} in {name!r}")
            if key in seen:
                raise MeasurementError(f"duplicate config key {key!r} in {name!r}")
            seen.add(key)
            cfg[key] = value.strip()
        return name, config_string_from_dict(cfg)
    if "=" in name:
        # An ABI config string, with its segments in key order as the package writes them.
        segments = sorted((seg.strip() for seg in name.split(";") if seg.strip()),
                          key=lambda seg: seg.split("=", 1)[0].strip())
        return name, ";".join(segments)
    if name.endswith(".json") and os.path.isfile(name):
        with open(name, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return name, config_string_from_dict(cfg)
    presets = _presets(eldpc_module)
    if name in presets:
        return name, config_string_from_dict(presets[name])
    raise MeasurementError(f"unknown config {name!r}; preset names are: {', '.join(sorted(presets))}")


def with_device(config_string: str, device: str) -> str:
    """The config string with the front's ``device=`` key for ``--device`` (none for "gpu", the
    front's default, so a GPU run's strings are unchanged). Both the C driver and the Python
    measurements take the same string. A config that names another device is refused."""
    segments = [seg for seg in config_string.split(";") if seg]
    named = [seg.split("=", 1)[1].strip() for seg in segments if seg.split("=", 1)[0].strip() == "device"]
    if named:
        if device != "gpu" and named[0] != device:
            raise MeasurementError(f"--device {device} conflicts with device={named[0]} in the config "
                                   f"{config_string!r}")
        return config_string
    if device == "gpu":
        return config_string
    return ";".join(sorted(segments + [f"device={device}"], key=lambda seg: seg.split("=", 1)[0]))


def config_keys(config_string: str) -> dict[str, str]:
    """``key -> value`` of an ABI config string (the last value of a repeated key)."""
    out = {}
    for seg in config_string.split(";"):
        if "=" in seg:
            key, value = seg.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def cpu_list_size(text: str) -> int | None:
    """The number of distinct CPUs of a ``cpus=`` list ("0-3,8"), None if it does not parse."""
    cpus = set()
    try:
        for item in text.split(","):
            a, _, b = item.strip().partition("-")
            lo, hi = int(a), int(b) if b else int(a)
            if hi < lo:
                return None
            cpus.update(range(lo, hi + 1))
    except ValueError:
        return None
    return len(cpus) or None


def resolve_device(config_string: str, eldpc_module) -> str:
    """The engine a handle of this config string runs on: "gpu" or "cpu".

    ``device=auto`` is resolved the way the package resolves it (``eldpc._auto``: the GPU when device 0
    is usable, else the CPU core); if that cannot be determined here it counts as "gpu", and the core
    variants the timed handles report are checked afterwards (``Bench.check_handle_schedule``).
    """
    device = config_keys(config_string).get("device", "gpu")
    if device != "auto":
        return device
    try:
        from eldpc import _auto

        chosen, _reason = _auto.choose(_auto.probe(eldpc_module.load().build_info()))
        return "cpu" if chosen == "cpu" else "gpu"
    except Exception:
        return "gpu"


# The two handle schedules of a session (results.json: <measurement>.handle_schedule.design).
HANDLE_SCHEDULES = {
    "interleaved": "both handles live in the session; calls alternate A, B, A, B, ... (the GPU core)",
    "sequential": "one live handle at a time: each handle is created, warmed up, timed for all its calls "
                  "and destroyed before the other is created, in the session's call order (the CPU core: "
                  "its worker threads busy-wait for the life of their handle, so two live CPU handles "
                  "would share the pinned CPUs); the session order still alternates which side goes first",
}


class Variant:
    def __init__(self, side: str, label: str, config_string: str, library: dict):
        self.side = side
        self.label = label
        self.config_string = config_string
        self.library = library

    def public(self) -> dict:
        return {"side": self.side, "label": self.label, "config_string": self.config_string,
                "library": self.library}


class Bench:
    def __init__(self, args, eldpc_module, bench_dir):
        self.args = args
        self.eldpc = eldpc_module
        self.bench_dir = Path(bench_dir).resolve()
        self.out = Path(args.out).resolve()
        self.data_dir = self.out / "data"
        self.cache_dir = self.out / "cache"
        self.S = settings(args.quick)
        self.quick = bool(args.quick)
        self.warnings: list[str] = []
        self.problem = None
        self.problem_info = {}
        self.variants: list[Variant] = []
        self.lib_a = None
        self.lib_b = None
        self.cpu = None
        self.env = None
        self.syndrome_seed_base = None
        # measurement -> side -> the core variants its timed handles reported (Decoder.core_variant)
        self.core_variants: dict[str, dict[str, set]] = {}
        # True when a handle runs on the CPU core: sessions then time one handle at a time
        # (HANDLE_SCHEDULES["sequential"]); the GPU core keeps the interleaved design.
        self.sequential = False
        self.cpu_core_sides: list[str] = []

    # -- shared ------------------------------------------------------------
    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def note_core_variants(self, measurement: str, sides, variants) -> None:
        """Record the core variant of each timed handle; ``sides[i]`` is the side of handle i."""
        per = self.core_variants.setdefault(measurement, {})
        variants = list(variants or [])
        for index, side in enumerate(sides):
            name = variants[index] if index < len(variants) and variants[index] else "unknown"
            per.setdefault(side, set()).add(str(name))

    def core_variant_record(self) -> dict:
        """The environment's ``core_variants``: {measurement: {side: variant}} for every timed handle.

        A side whose handles reported different variants gets the sorted list and a warning.
        """
        out = {}
        for measurement, per in self.core_variants.items():
            out[measurement] = {}
            for side in sorted(per):
                names = sorted(per[side])
                out[measurement][side] = names[0] if len(names) == 1 else names
                if len(names) > 1:
                    self.warn(f"the {measurement} handles of side {side} ran different core "
                              f"variants ({', '.join(names)})")
        return out

    def prepare_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def include_dirs_for_build(self) -> list[str]:
        """-I directories for the installed public header eldpc.h (raises DriverError if not found)."""
        return cdriver.find_include_dirs(self.eldpc)

    def prepare_problem(self) -> None:
        t0 = time.monotonic()
        self.prepare_dirs()
        if self.args.problem == problems.DEFAULT_PROBLEM and not os.path.isfile(self.args.problem):
            note = problems.resolve_default()[1]
            if note:
                self.warn(note)
        prob = problems.open_problem(self.args.problem)
        self.problem = prob
        S = self.S
        t_open = time.monotonic() - t0
        dets_lat, obs_lat, seed_lat = prob.sample(S["latency_shots"], "latency")
        t_sample = time.monotonic() - t0
        dets_ler, obs_ler, seed_ler = prob.sample(S["ler_shots"], "ler")
        t_ler = time.monotonic() - t0
        problem_file = self.data_dir / "problem.bin"
        shots_latency = self.data_dir / "shots_latency.bin"
        shots_ler = self.data_dir / "shots_ler.bin"
        formats.write_problem(problem_file, prob.H, prob.O, prob.priors, prob.dem_text)
        formats.write_shots(shots_latency, dets_lat, obs_lat)
        formats.write_shots(shots_ler, dets_ler, obs_ler)
        if prob.dem_text:
            (self.data_dir / "model.dem").write_text(prob.dem_text, encoding="utf-8")
        self.problem_file = problem_file
        self.shots_latency = shots_latency
        self.shots_ler = shots_ler
        # The base of the per-session syndrome seeds (each session derives its own from it).
        self.syndrome_seed_base = problems.session_seed(prob.name, "latency-syndromes")
        # Paths are recorded relative to the results directory, so a results file carries no
        # machine-dependent absolute path.
        self.problem_info = {
            **prob.info(),
            "shots": {
                "latency": {"shots": int(dets_lat.shape[0]), "seed": int(seed_lat),
                            "file": os.path.relpath(shots_latency, self.out),
                            "sha256": sha256_file(shots_latency)},
                "ler": {"shots": int(dets_ler.shape[0]), "seed": int(seed_ler),
                        "file": os.path.relpath(shots_ler, self.out),
                        "sha256": sha256_file(shots_ler)},
            },
            "problem_file": {"file": os.path.relpath(problem_file, self.out),
                             "sha256": sha256_file(problem_file)},
            "dem_sha256": hashlib.sha256(prob.dem_text.encode()).hexdigest() if prob.dem_text else None,
            "build_seconds": round(time.monotonic() - t0, 3),
            "timings_seconds": {"open": round(t_open, 3), "latency_shots": round(t_sample - t_open, 3),
                                "ler_shots": round(t_ler - t_sample, 3)},
        }

    def _session_shots(self, mode: str, pair: str, session: int, shots: int):
        """(dets, obs, seed, sha256) of one latency session: its own freshly sampled syndromes.

        The purpose includes the mode, the pair and the session index, so every session of every
        pair draws a different seeded shot set; the file written from it is decoded by both
        handles of the session.
        """
        dets, obs, seed = self.problem.sample(int(shots), session_purpose(mode, pair, session))
        digest = hashlib.sha256(np.ascontiguousarray(dets, dtype=np.uint8).tobytes()).hexdigest()
        return dets, obs, int(seed), digest

    def prepare_variants(self) -> None:
        names = self.args.config
        device = getattr(self.args, "device", "gpu")
        lib_a = cdriver.find_library(self.eldpc, explicit=self.args.lib)
        lib_b = cdriver.find_library(self.eldpc, explicit=self.args.lib_b) if self.args.lib_b else lib_a
        label_a, cfg_a = resolve_config(names[0], self.eldpc)
        cfg_a = with_device(cfg_a, device)
        self.lib_a, self.lib_b = lib_a, lib_b
        va = Variant("A", label_a, cfg_a, lib_a)
        self.variants = [va]
        if device == "cpu":
            self.warn("--device cpu: every handle runs on the CPU core, several times the GPU core's "
                      "time per shot (README, \"CPU core\"); a --quick run of a large problem such as bb144 "
                      "xyz takes far longer than on a GPU")
        if len(names) > 1:
            label_b, cfg_b = resolve_config(names[1], self.eldpc)
            cfg_b = with_device(cfg_b, device)
        else:  # a second library alone compares two builds at the same config
            label_b, cfg_b = label_a, cfg_a
        if cfg_b != cfg_a or lib_b["path"] != lib_a["path"]:
            self.variants.append(Variant("B", label_b, cfg_b, lib_b))
        elif len(names) > 1 or self.args.lib_b:
            self.warn("A and B are identical (same config and same library); only the A/A control is "
                      "measured")
        self.cpu_core_sides = [v.side for v in self.variants
                               if resolve_device(v.config_string, self.eldpc) == "cpu"]
        self.sequential = bool(self.cpu_core_sides)
        if self.sequential:
            print("bench.py: a handle runs on the CPU core: every session times one handle at a time "
                  "(handle_schedule 'sequential', docs/BENCH.md)", file=sys.stderr)
        for side in self.cpu_core_sides:
            v = next(x for x in self.variants if x.side == side)
            self._check_cpu_threads(v)

    def _check_cpu_threads(self, v: Variant) -> None:
        """Warn when a CPU-core handle has more threads than CPUs to run them on.

        Its threads - 1 workers busy-wait between decodes, each pinned to one CPU of ``cpus=`` or, without
        that key, of the creating thread's affinity (repeated when it has fewer CPUs than threads).  More
        threads than distinct CPUs puts spinning threads on one core and inflates every latency.
        """
        keys = config_keys(v.config_string)
        try:
            threads = int(keys.get("threads", "1"))
        except ValueError:
            return
        if "cpus" in keys:
            ncpu, where = cpu_list_size(keys["cpus"]), "its cpus= list"
        else:
            try:
                ncpu = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                ncpu = None
            where = "the CPUs this run may use (--cpu pins it to one; taskset sets a list)"
        if ncpu is not None and threads > ncpu:
            self.warn(f"side {v.side} runs the CPU core with threads={threads} on {ncpu} distinct CPU(s) of "
                      f"{where}: its busy-waiting workers share cores and every latency is inflated; give "
                      f"each thread its own CPU (e.g. cpus=0-{threads - 1} in the config and --cpu "
                      f"{threads - 1}, the list's last entry, for the calling thread; docs/BENCH.md "
                      f"\"CPU-core runs\") or lower threads=")

    def handle_schedule(self) -> dict:
        """The results record of the session design: interleaved (GPU) or sequential (CPU core)."""
        design = "sequential" if self.sequential else "interleaved"
        out = {"design": design, "what": HANDLE_SCHEDULES[design]}
        if self.sequential:
            out["cpu_core_sides"] = list(self.cpu_core_sides)
        return out

    def check_handle_schedule(self, measurement: str, variants) -> None:
        """After an interleaved session: warn loudly if both live handles ran on the CPU core anyway
        (a device=auto that resolved differently in the timed process)."""
        if not self.sequential and variants and all(str(x) == "cpu" for x in variants):
            self.warn(f"{measurement}: two CPU-core handles were live at the same time (interleaved "
                      f"sessions): their busy-waiting workers competed for the same CPUs, so these "
                      f"latencies are inflated; rerun with --device cpu")

    def python_variants(self) -> list[Variant]:
        """Variants measurable through the Python package (which loads one C library)."""
        if len(self.variants) == 1:
            return self.variants
        a, b = self.variants
        if a.library["path"] != b.library["path"]:
            self.warn("the Python-side measurements used one library only: the package loads one C library, "
                      "so build B is measured by c-latency alone")
            return [a]
        return [a, b]

    def _pairs(self, variants: list[Variant]) -> list[tuple[str, Variant, Variant]]:
        pairs = [("a_a", variants[0], variants[0])]
        if len(variants) > 1:
            pairs.append(("a_b", variants[0], variants[1]))
        return pairs

    # -- C driver ----------------------------------------------------------
    def _latency_modes(self) -> list[dict]:
        gap = int(self.args.gap_us)
        modes = [
            {"name": "back_to_back", "driver_mode": "b2b", "gap_us": 0, "kw": (0, 0)},
            {"name": "cadence", "driver_mode": "cadence", "gap_us": gap, "kw": (0, 0)},
            {"name": "cadence_keep_warm", "driver_mode": "cadence", "gap_us": gap, "kw": (1, 1)},
            # The keep-warm on/off comparison at a fixed cadence: the same decoder, warm against idle, in
            # the same sessions like any other pair.
            {"name": "cadence_keep_warm_pair", "driver_mode": "cadence", "gap_us": gap, "kw": (1, 0)},
        ]
        enabled = getattr(self, "enabled_modes", None)
        return [m for m in modes if enabled is None or m["name"] in enabled]

    def _session_order(self, session: int) -> tuple[str, str]:
        """(create order, call order) of one session under the configured order design.

        "alternating" (the original design) makes the create order and the call order the same and swaps
        both between sessions.  "all4" runs every combination of create order and call order once per
        four sessions, so both orders are balanced independently and the two effects can be separated.
        """
        design = getattr(self, "order_design", "alternating")
        if design == "all4":
            combos = (("ab", "ab"), ("ba", "ab"), ("ab", "ba"), ("ba", "ba"))
            return combos[session % 4]
        order = "ab" if session % 2 == 0 else "ba"
        return order, order

    def measure_c_latency(self) -> dict:
        if self.problem is None:
            self.prepare_problem()
        if not self.variants:
            self.prepare_variants()
        S = self.S
        include = self.include_dirs_for_build()
        driver = cdriver.build_driver(self.bench_dir, self.cache_dir, include)
        loaded = {v.side: v.library["path"] for v in self.variants}
        pairs = self._pairs(self.variants)
        modes_out = {}
        syndrome_sessions: list[dict] = []

        def sessions_for(mode, a, b, kw_a, kw_b, tag, pair, n_sessions=None):
            sessions = []
            for i in range(n_sessions or S["processes"]):
                create_order, order = self._session_order(i)
                if self.sequential:  # each handle is created right before its block of calls
                    create_order = order
                dets, obs, seed, digest = self._session_shots(
                    mode["name"], pair, i, S["c_shots"])
                shots_file = self.data_dir / f"c_{tag}_{i}.shots.bin"
                formats.write_shots(shots_file, dets, obs)
                out_file = self.data_dir / f"c_{tag}_{i}.bin"
                cmd = cdriver.driver_command(
                    driver, self.problem_file, shots_file, out_file,
                    loaded[a.side], loaded[b.side],
                    a.config_string, b.config_string, order, mode["driver_mode"], mode["gap_us"], kw_a,
                    kw_b, S["c_shots"], S["warmup"], self.cpu, S["warmup_ms"],
                    create_order=create_order, sequential=self.sequential)
                res = cdriver.run_driver(cmd)
                if res["exit"] == 3:
                    self.warn(f"the C driver reported a decode error in {tag}/session {i}")
                self.note_core_variants("c_latency", (a.side, b.side), res.get("core_variants"))
                self.check_handle_schedule("c_latency", res.get("core_variants"))
                session = _split_driver_session(res)
                session["create_order"] = create_order
                session["call_order"] = order
                session["syndrome"] = {"mode": mode["name"], "pair": pair, "session": i,
                                       "shots": int(dets.shape[0]), "seed": seed,
                                       "sha256": digest}
                syndrome_sessions.append(session["syndrome"])
                sessions.append(session)
            return sessions

        for mode in self._latency_modes():
            print(f"bench.py: c-latency mode {mode['name']} "
                  f"({S['processes']} sessions x {S['c_shots']} calls per variant)", file=sys.stderr)
            if mode["name"] == "cadence_keep_warm_pair":
                va = self.variants[0]
                # The treatment comparison gets the full session count; its two A/A controls only need
                # enough sessions to show the band, and each control is still order-balanced (an even
                # count of at least two).
                n_control = control_sessions(
                    S["processes"], S["c_shots"], self.quick,
                    default_counts=(getattr(self.args, "shots", None) is None
                                    and getattr(self.args, "processes", None) is None))
                per_pair = {
                    "keep_warm_on": sessions_for(mode, va, va, 1, 1, "kw_on", "keep_warm_on",
                                                 n_control),
                    "keep_warm_off": sessions_for(mode, va, va, 0, 0, "kw_off", "keep_warm_off",
                                                  n_control),
                    "keep_warm_ab": sessions_for(mode, va, va, 1, 0, "kw_ab", "keep_warm_ab"),
                }
                modes_out[mode["name"]] = _assemble_kw_pair_mode(
                    mode, per_pair, va, S, getattr(self, "order_design", "alternating"))
            else:
                per_pair = {}
                for kind, a, b in pairs:
                    per_pair[kind] = sessions_for(mode, a, b, mode["kw"][0], mode["kw"][1],
                                                  f"{mode['name']}_{kind}", kind)
                modes_out[mode["name"]] = _assemble_latency_mode(
                    mode, per_pair, self.variants, S, getattr(self, "order_design", "alternating"))
            for w in modes_out[mode["name"]]["warnings"]:
                self.warn(w)
        result = {
            "kind": "c_latency",
            "what_is_timed": "one eldpc_decode call per shot, CLOCK_MONOTONIC around the call: transfers "
                             "and synchronisation included",
            "problem": self.problem_info,
            "library": {v.side: library_ref(v.library) for v in self.variants},
            "configs": {v.side: {"label": v.label, "config_string": v.config_string} for v in self.variants},
            "cpu": self.cpu,
            "calls_per_variant_per_session": S["c_shots"],
            "sessions_per_pair": S["processes"],
            "warmup_calls_per_variant": S["warmup"],
            "order_design": getattr(self, "order_design", "alternating"),
            "handle_schedule": self.handle_schedule(),
            "syndromes": {**syndrome_record(syndrome_sessions),
                          "seed": self.syndrome_seed_base},
            "modes": modes_out,
        }
        return result

    # -- Python worker -----------------------------------------------------
    def _run_worker(self, job: dict, tag: str) -> dict:
        job = dict(job)
        job["out"] = str(self.data_dir / f"w_{tag}.json")
        job["cpu"] = self.cpu
        job_file = self.data_dir / f"w_{tag}.job.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")
        worker = str(self.bench_dir / "worker.py")
        try:
            p = subprocess.run([sys.executable, worker, str(job_file)], capture_output=True, text=True,
                               timeout=1800)
        except subprocess.TimeoutExpired as e:
            raise MeasurementError(f"the Python measurement worker ({tag}) timed out") from e
        if not os.path.exists(job["out"]):
            raise MeasurementError(f"the Python measurement worker ({tag}) wrote no result "
                                   f"(exit {p.returncode}): {p.stderr.strip()}")
        with open(job["out"], "r", encoding="utf-8") as f:
            res = json.load(f)
        res["stderr"] = p.stderr.strip()
        res["exit"] = p.returncode
        if "error" in res:
            raise MeasurementError(f"the Python measurement worker ({tag}) failed: {res['error']}")
        return res

    def measure_py_latency(self) -> dict:
        if self.problem is None:
            self.prepare_problem()
        if not self.variants:
            self.prepare_variants()
        S = self.S
        variants = self.python_variants()
        pairs = self._pairs(variants)
        per_pair = {}
        syndrome_sessions: list[dict] = []
        for kind, a, b in pairs:
            sessions = []
            for i in range(S["processes"]):
                print(f"bench.py: py-latency session {i + 1}/{S['processes']}", file=sys.stderr)
                create_order, call_order = self._session_order(i)
                if self.sequential:
                    create_order = call_order
                order = [0, 1] if call_order == "ab" else [1, 0]
                create = [0, 1] if create_order == "ab" else [1, 0]
                dets, obs, seed, digest = self._session_shots(
                    "back_to_back", kind, i, S["c_shots"])
                shots_file = self.data_dir / f"w_py_{kind}_{i}.shots.bin"
                formats.write_shots(shots_file, dets, obs)
                job = {
                    "kind": "py_latency",
                    "problem_file": str(self.problem_file),
                    "shots_file": str(shots_file),
                    "variants": [{"config_string": a.config_string}, {"config_string": b.config_string}],
                    "order": order,
                    "create_order": create,
                    "shots_per_variant": S["c_shots"],
                    "warmup": S["warmup"],
                    "gap_us": 0,
                    "keep_warm": False,
                    "sequential": self.sequential,
                }
                res = self._run_worker(job, f"py_{kind}_{i}")
                self.note_core_variants("py_latency", (a.side, b.side), res.get("core_variants"))
                self.check_handle_schedule("py_latency", res.get("core_variants"))
                session = _split_worker_latency(res)
                session["create_order"] = create_order
                session["call_order"] = call_order
                session["syndrome"] = {"mode": "back_to_back", "pair": kind, "session": i,
                                       "shots": int(dets.shape[0]), "seed": seed,
                                       "sha256": digest}
                syndrome_sessions.append(session["syndrome"])
                sessions.append(session)
            per_pair[kind] = sessions
        mode = {"name": "back_to_back", "gap_us": 0, "kw": (0, 0)}
        modes_out = {"back_to_back": _assemble_latency_mode(
            mode, per_pair, variants, S, getattr(self, "order_design", "alternating"))}
        for w in modes_out["back_to_back"]["warnings"]:
            self.warn(w)
        return {
            "kind": "py_latency",
            "what_is_timed": "Decoder.decode(syndrome) from Python, the syndrome already a uint8 numpy "
                             "array; Decoder.last_status is read after the call",
            "what_is_not_timed": "creating the decoder, warm-up, and converting the syndrome from another "
                                 "dtype",
            "problem": self.problem_info,
            "configs": {v.side: {"label": v.label, "config_string": v.config_string} for v in variants},
            "cpu": self.cpu,
            "calls_per_variant_per_session": S["c_shots"],
            "sessions_per_pair": S["processes"],
            "warmup_calls_per_variant": S["warmup"],
            "order_design": getattr(self, "order_design", "alternating"),
            "handle_schedule": self.handle_schedule(),
            "syndromes": {**syndrome_record(syndrome_sessions),
                          "seed": self.syndrome_seed_base},
            "modes": modes_out,
        }

    def measure_throughput(self) -> dict:
        if self.problem is None:
            self.prepare_problem()
        if not self.variants:
            self.prepare_variants()
        S = self.S
        variants = self.python_variants()
        pairs = self._pairs(variants)
        per_pair = {}
        for kind, a, b in pairs:
            sessions = []
            for i in range(S["processes"]):
                print(f"bench.py: throughput session {i + 1}/{S['processes']}", file=sys.stderr)
                create_order, call_order = self._session_order(i)
                if self.sequential:
                    create_order = call_order
                order = [0, 1] if call_order == "ab" else [1, 0]
                create = [0, 1] if create_order == "ab" else [1, 0]
                job = {
                    "kind": "throughput",
                    "problem_file": str(self.problem_file),
                    "shots_file": str(self.shots_latency),
                    "variants": [{"config_string": a.config_string}, {"config_string": b.config_string}],
                    "order": order,
                    "create_order": create,
                    "batch_shots": S["throughput_batch"],
                    "repetitions": S["throughput_reps"],
                    "warmup": S["warmup"],
                    "sequential": self.sequential,
                }
                res = self._run_worker(job, f"tp_{kind}_{i}")
                self.note_core_variants("throughput", (a.side, b.side), res.get("core_variants"))
                self.check_handle_schedule("throughput", res.get("core_variants"))
                sessions.append({"batches": res["batches"], "checksums": res["checksum"]})
            per_pair[kind] = sessions
        result = _assemble_throughput(per_pair, variants, S)
        result["handle_schedule"] = self.handle_schedule()
        if not getattr(self.args, "no_c_batch", False):
            try:
                result["c_driver"] = self._measure_c_batch(variants)
            except cdriver.DriverError as e:
                result["c_driver"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
                self.warn(f"the C-driver half of the throughput measurement could not run: {e}")
        return result

    def _measure_c_batch(self, variants: list) -> dict:
        """The same batch workload through the C driver, so the decoder's own batch throughput can be
        read next to the Python number (which also pays Python and output-allocation costs)."""
        S = self.S
        include = self.include_dirs_for_build()
        driver = cdriver.build_driver(self.bench_dir, self.cache_dir, include)
        loaded = {v.side: v.library["path"] for v in variants}
        pairs = self._pairs(variants)
        n_sessions = max(2, S["processes"] // 2)  # a diagnostic; the Python block keeps the full count
        per_pair = {}
        for kind, a, b in pairs:
            sessions = []
            for i in range(n_sessions):
                create_order, call_order = self._session_order(i)
                if self.sequential:
                    create_order = call_order
                print(f"bench.py: throughput C driver session {i + 1}/{n_sessions}", file=sys.stderr)
                out_file = self.data_dir / f"cb_{kind}_{i}.bin"
                cmd = cdriver.driver_command(
                    driver, self.problem_file, self.shots_latency, out_file,
                    loaded[a.side], loaded[b.side],
                    a.config_string, b.config_string, call_order, "batch", 0, False, False,
                    S["c_shots"], S["warmup"], self.cpu, S["warmup_ms"], create_order=create_order,
                    batch_shots=S["throughput_batch"], batch_reps=S["throughput_reps"],
                    sequential=self.sequential)
                res = cdriver.run_driver(cmd)
                if res["exit"] == 3:
                    self.warn(f"the C driver reported a decode error in the batch throughput pair {kind}")
                self.note_core_variants("throughput_c_driver", (a.side, b.side),
                                        res.get("core_variants"))
                self.check_handle_schedule("throughput_c_driver", res.get("core_variants"))
                shots = res["shots_per_record"]
                batches = [{"variant": int(res["variant"][j]), "shots": int(shots[j]),
                            "dur_ns": int(res["dur_ns"][j])} for j in range(len(shots))]
                sessions.append({"batches": batches,
                                 "checksums": [f"{res['checksums']['variant0']:016x}",
                                               f"{res['checksums']['variant1']:016x}"]})
            per_pair[kind] = sessions
        out = _assemble_throughput(per_pair, variants, S)
        out["kind"] = "throughput_c_driver"
        out["what_is_measured"] = ("eldpc_decode_batch called directly from the C driver with reused output "
                                   "buffers: the decoder's batch throughput without the Python call and "
                                   "its output allocation")
        out["sessions_per_pair"] = n_sessions
        out["handle_schedule"] = self.handle_schedule()
        return out

    def measure_ler(self) -> dict:
        if self.problem is None:
            self.prepare_problem()
        if not self.variants:
            self.prepare_variants()
        S = self.S
        if self.problem.k == 0:
            raise MeasurementError("the problem has no observables (k = 0): the LER check needs at least "
                                   "one observable to compare the predictions against")
        variants = self.python_variants()
        out = {"kind": "ler", "what_is_measured": "logical error rate over shots sampled from the "
                                                 "problem's model: a shot is wrong when any predicted "
                                                 "observable flip differs from the true one",
               "problem": self.problem_info, "shots": S["ler_shots"],
               "batch_shots": min(2048, max(1, S["ler_shots"])), "variants": {}}
        for v in variants:
            job = {"kind": "ler", "problem_file": str(self.problem_file), "shots_file": str(self.shots_ler),
                   "variants": [{"config_string": v.config_string}],
                   "batch_shots": out["batch_shots"]}
            res = self._run_worker(job, f"ler_{v.side}")
            self.note_core_variants("ler", (v.side,), res.get("core_variants"))
            if res["shots"] != S["ler_shots"]:
                raise MeasurementError(f"LER worker decoded {res['shots']} shots, expected {S['ler_shots']}")
            lo, hi = stats.wilson_interval(res["errors"], res["shots"])
            out["variants"][v.side] = {
                "label": v.label, "config_string": v.config_string,
                "shots": int(res["shots"]), "errors": int(res["errors"]),
                "ler": res["errors"] / res["shots"] if res["shots"] else None,
                "ci95": [round(lo, 8), round(hi, 8)],
                "checksum": res["checksum"],
                "decode_wall_s": res.get("decode_wall_s"),
                "shots_per_s_including_io": (round(res["shots"] / res["decode_wall_s"], 1)
                                             if res.get("decode_wall_s") else None),
            }
        if len(variants) > 1:
            a, b = out["variants"]["A"], out["variants"]["B"]
            out["comparison"] = {
                "a": "A", "b": "B",
                "delta_ler": round(a["ler"] - b["ler"], 8),
                "ci95_disjoint": bool(a["ci95"][1] < b["ci95"][0] or b["ci95"][1] < a["ci95"][0]),
            }
        return out


def _split_driver_session(res: dict) -> dict:
    v = res["variant"]
    out = {"meta": res["meta"], "keep": res["keep"], "stderr": res["stderr"], "exit": res["exit"]}
    for idx in (0, 1):
        mask = v == idx
        out[f"v{idx}"] = {"dur_ns": res["dur_ns"][mask], "status": res["status"][mask],
                          "checksum": f"{res['checksums']['variant%d' % idx]:016x}"}
    return out


def _split_worker_latency(res: dict) -> dict:
    out = {"stderr": res.get("stderr", ""), "exit": res.get("exit", 0)}
    for idx in (0, 1):
        w = res["variants"][idx]
        out[f"v{idx}"] = {"dur_ns": w["dur_ns"], "status": w["status"], "checksum": w["checksum"],
                          "keep_warm": w.get("keep_warm")}
    return out


def _status_summary(status_words) -> dict:
    st = np.asarray(status_words, dtype=np.uint64)
    flags = (st & np.uint64(0xFFFF)).astype(np.int64)
    iters = (st >> np.uint64(16)).astype(np.int64)
    hist = {}
    if iters.size:
        vals, counts = np.unique(iters, return_counts=True)
        hist = {str(int(k)): int(c) for k, c in zip(vals, counts)}
    return {
        "shots": int(st.size),
        "converged": int(np.count_nonzero(flags & 0x0001)),
        "postprocessed": int(np.count_nonzero(flags & 0x0002)),
        "higher_order_win": int(np.count_nonzero(flags & 0x0004)),
        "abstain": int(np.count_nonzero(flags & 0x0008)),
        "syndrome_unmatched": int(np.count_nonzero(flags & 0x0010)),
        "iterations_mean": round(float(iters.mean()), 4) if iters.size else None,
        "iterations_max": int(iters.max()) if iters.size else None,
        "iterations_histogram": hist,
    }


def _keep_summary(sessions: list[dict], shots_per_session: int) -> dict | None:
    if not sessions or "keep" not in sessions[0]:
        return None
    n = sum(s["keep"]["n"] for s in sessions)
    err = sum(s["keep"]["errors"] for s in sessions)
    psum = sum(s["keep"]["periodic_sum_ns"] for s in sessions)
    pmax = max((s["keep"]["periodic_max_ns"] for s in sessions), default=0)
    fsum = sum(s["keep"]["final_sum_ns"] for s in sessions)
    fmax = max((s["keep"]["final_max_ns"] for s in sessions), default=0)
    final_calls = 2 * shots_per_session * len(sessions)
    periodic_calls = max(0, n - final_calls)
    return {
        "calls": int(n), "errors": int(err),
        "periodic_calls": int(periodic_calls),
        "periodic_mean_us": round(psum / periodic_calls / 1000.0, 4) if periodic_calls else None,
        "periodic_max_us": round(pmax / 1000.0, 4),
        "final_calls": int(final_calls),
        "final_mean_us": round(fsum / final_calls / 1000.0, 4) if final_calls else None,
        "final_max_us": round(fmax / 1000.0, 4),
    }


LATENCY_KEYS = ("p50_us", "p99_us", "p99_9_us", "mean_us")


def _blocked_delta(rows: list[dict], sessions: list[dict]) -> tuple[dict | None, list[dict]]:
    """Order-balanced difference: average the per-session differences within each (create order, call
    order) block, then average the block means with equal weight.

    The first handle created in a session can be measurably faster or slower than the second, so a
    session's difference carries an order-dependent offset.  When the blocks have equal session counts,
    the plain mean of the sessions is already balanced; when they do not (for example three sessions
    with the alternating design: two 'ab' and one 'ba'), the plain mean is biased by the offset, while
    this estimate weights the two orders equally.
    """
    blocks: dict[tuple, list[dict]] = {}
    for row, s in zip(rows, sessions):
        key = (s.get("create_order"), s.get("call_order"))
        blocks.setdefault(key, []).append(row)
    if len(blocks) < 2 or any(k[0] is None for k in blocks):
        return None, []
    out = {}
    for key in LATENCY_KEYS:
        means = []
        for rs in blocks.values():
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                means.append(sum(vals) / len(vals))
        out[key] = round(sum(means) / len(means), 4) if means else None
    summary = [{"create_order": k[0], "call_order": k[1], "sessions": len(v),
                "mean_delta": {key: (round(sum(r[key] for r in v if r.get(key) is not None)
                                       / max(1, len([r for r in v if r.get(key) is not None])), 4))
                               for key in LATENCY_KEYS}} for k, v in blocks.items()]
    return out, summary


def _pair_stats(sessions: list[dict], label_a: str = "A", label_b: str = "A") -> dict:
    """Pooled statistics of variant sides 0 and 1 over a pair's sessions, plus the per-session deltas.

    The pooled statistics include the slow-class split of exactly the calls they summarize: the
    status words are concatenated in the same order as the durations.
    """
    if sessions:
        a_stats = stats.summarize_with_slow(
            np.concatenate([s["v0"]["dur_ns"] for s in sessions]),
            np.concatenate([s["v0"]["status"] for s in sessions]))
        b_stats = stats.summarize_with_slow(
            np.concatenate([s["v1"]["dur_ns"] for s in sessions]),
            np.concatenate([s["v1"]["status"] for s in sessions]))
    else:
        a_stats = stats.summarize_with_slow([], [])
        b_stats = stats.summarize_with_slow([], [])
    rows = []
    for s in sessions:
        sa = stats.summarize_ns(s["v0"]["dur_ns"])
        sb = stats.summarize_ns(s["v1"]["dur_ns"])
        rows.append(stats.delta(sa, sb))
    blocked, blocks = _blocked_delta(rows, sessions)
    a_status = _status_summary(np.concatenate([s["v0"]["status"] for s in sessions]))
    b_status = _status_summary(np.concatenate([s["v1"]["status"] for s in sessions]))
    checks_a = sorted({s["v0"]["checksum"] for s in sessions})
    checks_b = sorted({s["v1"]["checksum"] for s in sessions})
    # Every session decodes its own syndromes, so a handle's checksum differs between sessions by
    # design.  Within a session both handles decode exactly the same shots; that is the comparison
    # that shows the decoder returns identical outputs for identical inputs.
    session_checks_equal = [s["v0"]["checksum"] == s["v1"]["checksum"] for s in sessions]
    # Standard error of the mean per-session difference: the uncertainty of the pooled difference that
    # the order effects contribute to; unlike the range it shrinks as sessions are added.
    sem = {}
    for key in ("p50_us", "p99_us", "p99_9_us", "mean_us"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            sem[key] = round((var / len(vals)) ** 0.5, 4)
        else:
            sem[key] = None
    return {
        "a": label_a, "b": label_b,
        "a_stats": a_stats, "b_stats": b_stats,
        "delta": stats.delta(a_stats, b_stats),
        "delta_blocked": blocked,
        "delta_blocked_note": ("per-session differences averaged within each (create, call) order block "
                               "and then over blocks with equal weight; equal to delta when the order "
                               "blocks have equal session counts"),
        "order_blocks": blocks,
        "per_session_delta": rows,
        "band": stats.band(rows),
        "sem": sem,
        "orders": [{"session": i, "create_order": s.get("create_order"),
                    "call_order": s.get("call_order")} for i, s in enumerate(sessions)],
        "a_status": a_status, "b_status": b_status,
        "checksums": {"v0": checks_a, "v1": checks_b},
        "session_checksums_equal": session_checks_equal,
        # Sides with the same config and library must produce identical outputs for the same shots;
        # with per-session syndromes the comparison is made within each session.
        "sides_equal": bool(session_checks_equal) and all(session_checks_equal),
    }


def _outlier_warnings(out: dict, mode_name: str, stats_by_side: dict) -> None:
    """Flag a side whose worst shot is far above its p50: a host or GPU hiccup the clocks/busy checks may
    not have seen.  The 1 ms floor keeps the warning meaningful for very fast decodes, where the first
    launches' ~20 us latency is a normal part of the distribution rather than interference."""
    for side, st in stats_by_side.items():
        p50, mx = st.get("p50_us"), st.get("max_us")
        if p50 and mx and p50 > 0 and mx > 50.0 * p50 and mx > 1000.0:
            out["warnings"].append(
                f"{mode_name} {side}: one shot took {mx:.1f} us, {mx / p50:.0f} times the p50 "
                f"({p50:.3f} us); host or GPU interference during the run")


def _slow_class_warnings(out: dict, mode_name: str, variants: dict) -> None:
    """Warn when a side's p99 sits on the cliff between the converged calls and the slow class."""
    for side, variant in variants.items():
        pooled = variant.get("pooled") or {}
        if pooled.get("p99_unreliable"):
            fraction = pooled.get("slow_fraction")
            out["warnings"].append(
                f"{mode_name} {side}: slow_fraction "
                f"{fraction:.4g} is within 2x of 1 %: the p99 sits on the cliff between the "
                f"converged calls and the slow class and is not a stable number; the median of "
                f"the slow calls is {pooled.get('slow_p50_us')} us")


def _assemble_latency_mode(mode: dict, per_pair: dict, variants: list[Variant], S: dict,
                           order_design: str = "alternating") -> dict:
    out = {"mode": mode["name"], "gap_us": mode["gap_us"],
           "keep_warm_a": int(mode["kw"][0]), "keep_warm_b": int(mode["kw"][1]),
           "order_design": order_design,
           "shots_per_variant_per_session": S["c_shots"],
           "sessions_per_pair": S["processes"], "a_a_control": None, "comparison": None,
           "variants": {}, "warnings": []}
    control = _pair_stats(per_pair["a_a"], "A", "A")
    out["checksum_scope"] = ("every session decodes its own syndromes; the two handles of a "
                             "session are compared on the same shots, not across sessions")
    if not control["sides_equal"]:
        out["warnings"].append("the two handles of the A/A control produced different outputs for the "
                               "same shots: the decoder is not stateless across handles")
    out["a_a_control"] = control
    if "a_b" in per_pair:
        comp = _pair_stats(per_pair["a_b"], "A", "B")
        if variants[0].config_string == variants[1].config_string \
                and variants[0].library["path"] == variants[1].library["path"] and not comp["sides_equal"]:
            out["warnings"].append("two handles with the same config and library produced different "
                                   "outputs for the same shots")
        comp["aa_band"] = control["band"]
        comp["aa_band_sessions"] = len(control["per_session_delta"])
        comp["aa_band_note"] = ("range of the same per-session difference statistic over the A/A "
                                "control sessions (the noise band of this machine)")
        comp["within_band"] = {k: (comp["delta"].get(k) is not None
                                   and comp["aa_band"].get(k) is not None
                                   and comp["aa_band"][k][0] <= comp["delta"][k] <= comp["aa_band"][k][1])
                               for k in ("p50_us", "p99_us", "p99_9_us", "mean_us")}
        comp["within_band_blocked"] = {
            k: (comp["delta_blocked"] is not None and comp["delta_blocked"].get(k) is not None
                and comp["aa_band"].get(k) is not None
                and comp["aa_band"][k][0] <= comp["delta_blocked"][k] <= comp["aa_band"][k][1])
            for k in ("p50_us", "p99_us", "p99_9_us", "mean_us")}
        out["comparison"] = comp
        out["variants"] = {
            "A": {"label": variants[0].label, "config_string": variants[0].config_string,
                  "library": library_ref(variants[0].library), "pooled": comp["a_stats"], "status": comp["a_status"],
                  "checksum": comp["checksums"]["v0"][0] if len(comp["checksums"]["v0"]) == 1 else None},
            "B": {"label": variants[1].label, "config_string": variants[1].config_string,
                  "library": library_ref(variants[1].library), "pooled": comp["b_stats"], "status": comp["b_status"],
                  "checksum": comp["checksums"]["v1"][0] if len(comp["checksums"]["v1"]) == 1 else None},
        }
    else:
        both = np.concatenate([np.concatenate([s["v0"]["dur_ns"] for s in per_pair["a_a"]]),
                               np.concatenate([s["v1"]["dur_ns"] for s in per_pair["a_a"]])])
        status = np.concatenate([s["v0"]["status"] for s in per_pair["a_a"]] +
                                [s["v1"]["status"] for s in per_pair["a_a"]])
        out["variants"] = {"A": {"label": variants[0].label, "config_string": variants[0].config_string,
                                 "library": library_ref(variants[0].library),
                                 "pooled": stats.summarize_with_slow(both, status),
                                 "status": _status_summary(status),
                                 "checksum": control["checksums"]["v0"][0]
                                 if len(control["checksums"]["v0"]) == 1 else None}}
    if mode["kw"][0] and mode["kw"][1]:
        out["keep_warm_stats"] = _keep_summary(per_pair["a_a"], S["c_shots"])
        if "a_b" in per_pair:
            out["keep_warm_stats_b"] = _keep_summary(per_pair["a_b"], S["c_shots"])
    _outlier_warnings(out, mode["name"], {k: v.get("pooled", {}) for k, v in out["variants"].items()})
    _slow_class_warnings(out, mode["name"], out["variants"])
    return out


def _assemble_kw_pair_mode(mode: dict, per_pair: dict, variant: Variant, S: dict,
                           order_design: str = "alternating") -> dict:
    """The keep-warm on/off pair: warm against idle in the same sessions, with warm/warm and idle/idle
    controls giving the noise band of this cadence."""
    on = _pair_stats(per_pair["keep_warm_on"], "warm", "warm")
    off = _pair_stats(per_pair["keep_warm_off"], "idle", "idle")
    comp = _pair_stats(per_pair["keep_warm_ab"], "warm", "idle")
    band_rows = (on["per_session_delta"] + off["per_session_delta"])
    comp["aa_band"] = stats.band(band_rows)
    comp["aa_band_sessions"] = len(band_rows)
    comp["aa_band_note"] = ("range of the per-session A/A deltas of the keep-warm-on/on and keep-warm-off/off "
                            "controls at the same cadence (the noise band of this machine)")
    comp["within_band"] = {k: (comp["delta"].get(k) is not None and comp["aa_band"].get(k) is not None
                               and comp["aa_band"][k][0] <= comp["delta"][k] <= comp["aa_band"][k][1])
                           for k in ("p50_us", "p99_us", "p99_9_us", "mean_us")}
    comp["within_band_blocked"] = {
        k: (comp["delta_blocked"] is not None and comp["delta_blocked"].get(k) is not None
            and comp["aa_band"].get(k) is not None
            and comp["aa_band"][k][0] <= comp["delta_blocked"][k] <= comp["aa_band"][k][1])
        for k in ("p50_us", "p99_us", "p99_9_us", "mean_us")}
    warnings = []
    for ctl_name, ctl in (("keep-warm-on", on), ("keep-warm-off", off)):
        if not ctl["sides_equal"]:
            warnings.append(f"the two handles of the {ctl_name} control produced different outputs for "
                            "the same shots")
    if not comp["sides_equal"]:
        warnings.append("keep-warm on and keep-warm off produced different outputs for the same shots: "
                        "keep-warm must never change a decoder result")
    out = {
        "mode": mode["name"], "gap_us": mode["gap_us"], "keep_warm_a": 1, "keep_warm_b": 0,
        "order_design": order_design,
        "description": "the same decoder with keep-warm during the gap (side A) against no keep-warm "
                       "(side B), in the same sessions",
        "shots_per_variant_per_session": S["c_shots"], "sessions_per_pair": S["processes"],
        "control_sessions": len(per_pair["keep_warm_on"]),
        "a_a_controls": {"keep_warm_on": on, "keep_warm_off": off},
        "comparison": comp,
        "variants": {
            "A": {"label": variant.label, "config_string": variant.config_string,
                  "library": library_ref(variant.library), "treatment": "keep_warm", "pooled": comp["a_stats"],
                  "status": comp["a_status"],
                  "checksum": comp["checksums"]["v0"][0] if len(comp["checksums"]["v0"]) == 1 else None},
            "B": {"label": variant.label, "config_string": variant.config_string,
                  "library": library_ref(variant.library), "treatment": "idle", "pooled": comp["b_stats"],
                  "status": comp["b_status"],
                  "checksum": comp["checksums"]["v1"][0] if len(comp["checksums"]["v1"]) == 1 else None},
        },
        "keep_warm_stats": _keep_summary(per_pair["keep_warm_on"], S["c_shots"]),
        "warnings": warnings,
    }
    _outlier_warnings(out, mode["name"], {k: v.get("pooled", {}) for k, v in out["variants"].items()})
    _slow_class_warnings(out, mode["name"], out["variants"])
    return out


def _assemble_throughput(per_pair: dict, variants: list[Variant], S: dict) -> dict:
    def var_rates(sessions, side):
        shots = 0
        ns = 0
        for s in sessions:
            for b in s["batches"]:
                if b["variant"] == (0 if side == "A" else 1):
                    shots += b["shots"]
                    ns += b["dur_ns"]
        return {"shots": int(shots), "seconds": round(ns / 1e9, 6),
                "shots_per_s": round(shots / (ns / 1e9), 3) if ns else None}

    def session_rates(sessions, side):
        rates = []
        for s in sessions:
            idx = 0 if side == "A" else 1
            shots = sum(b["shots"] for b in s["batches"] if b["variant"] == idx)
            ns = sum(b["dur_ns"] for b in s["batches"] if b["variant"] == idx)
            rates.append(shots / (ns / 1e9) if ns else None)
        return rates

    control_sessions = per_pair["a_a"]
    control_ratios = _session_ratios(session_rates(control_sessions, "A"),
                                     session_rates(control_sessions, "B"))
    out = {
        "kind": "throughput",
        "what_is_measured": "Decoder.decode_batch over whole batches: shots per second, batch latency "
                            "included",
        "batch_shots": S["throughput_batch"],
        "repetitions_per_variant_per_session": S["throughput_reps"],
        "sessions_per_pair": S["processes"],
        "a_a_control": {
            "sessions": len(control_sessions),
            "A": var_rates(control_sessions, "A"),
            "B": var_rates(control_sessions, "B"),
            "per_session_shots_per_s": {"A": session_rates(control_sessions, "A"),
                                         "B": session_rates(control_sessions, "B")},
            "per_session_ratio_b_over_a": [round(r, 6) for r in control_ratios],
            "ratio_b_over_a": _ratio(var_rates(control_sessions, "B")["shots_per_s"],
                                      var_rates(control_sessions, "A")["shots_per_s"]),
            "band_ratio": _range(control_ratios),
        },
        "variants": {},
    }
    if "a_b" in per_pair:
        s = per_pair["a_b"]
        a, b = var_rates(s, "A"), var_rates(s, "B")
        ratios = _session_ratios(session_rates(s, "A"), session_rates(s, "B"))
        ratio = _ratio(b["shots_per_s"], a["shots_per_s"])
        aa_band = _range(control_ratios)
        out["comparison"] = {
            "rates_a": a, "rates_b": b,
            "ratio_b_over_a": ratio,
            # The noise band: the range of the same B/A ratio over the A/A control sessions, where both
            # sides are the same decoder.  It is not derived from the A/B sessions.
            "aa_band_ratio": aa_band,
            "aa_band_sessions": len(control_sessions),
            "aa_band_note": ("range of the per-session B/A shots-per-second ratio over the A/A control "
                             "sessions (both sides the same decoder): the noise band of this machine"),
            "within_aa_band": bool(ratio is not None and aa_band is not None
                                   and aa_band[0] <= ratio <= aa_band[1]),
            "ab_session_ratio_range": _range(ratios),
            "ab_sessions": len(s),
            "per_session_shots_per_s": {"A": session_rates(s, "A"), "B": session_rates(s, "B")},
            "checksums_equal": _checks_equal(s),
        }
        out["variants"] = {"A": {"label": variants[0].label, "config_string": variants[0].config_string,
                                 "library": library_ref(variants[0].library), **a},
                           "B": {"label": variants[1].label, "config_string": variants[1].config_string,
                                 "library": library_ref(variants[1].library), **b}}
    else:
        out["variants"] = {"A": {"label": variants[0].label, "config_string": variants[0].config_string,
                                 "library": library_ref(variants[0].library),
                                 **var_rates(control_sessions, "A")}}
    return out


def _ratio(num, den):
    return round(num / den, 6) if num and den else None


def _session_ratios(rates_a: list, rates_b: list) -> list[float]:
    """Per-session B/A ratios (sessions where either rate is missing are skipped)."""
    return [rb / ra for ra, rb in zip(rates_a, rates_b) if ra and rb]


def _range(values: list[float]) -> list[float] | None:
    return [round(min(values), 4), round(max(values), 4)] if values else None


def _checks_equal(sessions: list[dict]) -> bool:
    for idx in (0, 1):
        vals = {s["checksums"][idx] for s in sessions}
        if len(vals) != 1:
            return False
    return True
