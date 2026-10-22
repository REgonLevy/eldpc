#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect the outputs of the package's own verify/bench/tune commands into docs/results/.

``verify/run_verify.py`` and ``bench/bench.py`` write complete records of one run, including the
machine they ran on.  Those records name files by absolute path and the GPU by its UUID, which do
not belong in a published package, and they carry fields RESULTS.md does not use.  This tool keeps
the fields RESULTS.md is generated from, normalises the recorded command lines, adds the commit of
the package tree the measurement ran on, and writes one small JSON file per run under
docs/results/.

Usage (from the package root, with the package installed; the raw outputs may live anywhere,
for example in a directory next to the checkout):

    python docs/tools/collect_results.py \\
        --verify ../eldpc-results/verify/summary.json \\
        --bench  ../eldpc-results/bench/results.json \\
        --ler    ../eldpc-results/ler/results.json \\
        --tune   docs/results/tuning-bb144/results.json \\
        --out docs/results

Each of ``--verify``, ``--bench`` and ``--ler`` is optional (at least one is needed): the tool writes
the records of the runs it is given and leaves the others in ``--out`` as they are, so one record
can be replaced on its own, for example a machine's verification run:

    python docs/tools/collect_results.py \\
        --verify ../eldpc-results/target-verify/summary.json --out docs/results/target

``make_results.py`` reads the files this tool writes; it never runs a measurement itself, so
``make_results.py --check`` is a pure function of the committed results files.

With ``--tune`` the tool also writes ``provenance.json`` next to the tuner's own outputs and
cleans that directory in place: GPU UUIDs are removed from the tuner's JSON files and its resume
checkpoint (``raw/checkpoint.json``) is deleted.

The tool refuses to write a file that contains an absolute path of the machine it ran on, and a
recorded command that names any absolute path: run the commands with paths relative to the
package root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

RESULT_SCHEMA = "eldpc.results.v2"
# One generic check for an absolute path of the machine that produced a record: a POSIX path of at
# least two components (a "/" at the start of the string or after whitespace, a quote, "=", "(",
# ",", ":" or similar, not part of "://", then a name and one more "/") or a Windows drive path.
# Relative paths, URLs and ratios such as "us/shot" or "p50/p99" do not match.
_ABS_PATH = re.compile(
    r"""(?:^|[\s"'`=(\[<,:;])/(?!/)[\w.+@~-]+(?:/[\w.+@~-]*)+"""  # POSIX, >= 2 components
    r"""|\b[A-Za-z]:\\\S"""                                         # Windows drive path
)


def looks_like_abs_path(s: str) -> bool:
    return _ABS_PATH.search(s) is not None
# Keys that identify one physical device rather than a hardware class.
IDENTIFYING_KEYS = ("uuid",)
# The files of the package tree whose state decides whether a recorded run matches its commit.
WATCHED = ("src", "python", "include", "CMakeLists.txt", "pyproject.toml", "verify", "bench",
           "tuning")


# ---------------------------------------------------------------------------
# helpers


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git(*args: str, cwd: Path) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    return p.stdout.strip() if p.returncode == 0 else ""


def commit_of(package_dir: Path) -> str:
    """HEAD of the git repository that contains the package (the package may be its root)."""
    return git("rev-parse", "HEAD", cwd=package_dir)


def package_tree_clean(package_dir: Path) -> bool:
    """True when the package's sources and measurement code match the recorded commit.

    The pathspecs are relative to the package directory, so this works whether the package is the
    repository root or a subdirectory of it.
    """
    if not git("rev-parse", "--is-inside-work-tree", cwd=package_dir):
        return False
    out = git("status", "--porcelain", "--", *WATCHED, cwd=package_dir)
    return out == ""


def smi_gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    fields = ("index,name,driver_version,compute_cap,memory.total,"
              "clocks.current.graphics,clocks.current.memory")
    try:
        p = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if p.returncode != 0:
        return []
    rows = []
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue

        def num(s):
            try:
                return float(s)
            except ValueError:
                return None

        rows.append({
            "name": parts[1],
            "driver_version": parts[2],
            "compute_capability": parts[3],
            "memory_mib": num(parts[4]),
            "clocks": {"graphics_mhz": num(parts[5]), "memory_mhz": num(parts[6])},
        })
    return rows


def clock_snapshot() -> dict:
    gpus = smi_gpus()
    if not gpus:
        return {"graphics_mhz": None, "memory_mhz": None, "source": "nvidia-smi unavailable"}
    g = gpus[0]
    return {**g["clocks"], "source": "nvidia-smi snapshot taken by the collection step after the run"}


def toolkit_string(toolkit) -> str | None:
    """A single readable string from either tool's toolkit record."""
    if toolkit is None:
        return None
    if isinstance(toolkit, str):
        # verify records the last line of `nvcc --version`, e.g.
        # "Build cuda_13.3.r13.3/compiler.38244171_0" or
        # "Cuda compilation tools, release 13.3, V13.3.73".
        m = re.search(r"release\s+(\d+\.\d+)", toolkit)
        if m:
            return f"CUDA {m.group(1)} ({toolkit})"
        m = re.search(r"cuda_(\d+\.\d+)", toolkit)
        if m:
            return f"CUDA {m.group(1)} ({toolkit})"
        return toolkit
    if isinstance(toolkit, dict):
        version = toolkit.get("version")
        nvcc = toolkit.get("nvcc")
        if version and nvcc:
            return f"CUDA {version} ({nvcc})"
        return version or toolkit.get("first_line")
    return str(toolkit)


def static_runtime(build_info) -> str | None:
    """The CUDA runtime linked into the decoder cores, from the ``cuda=`` field of build_info().

    The cores link the runtime statically, so the runtime is the build toolkit's; a shared
    libcudart elsewhere on the machine is not used by the package and is not recorded here.
    """
    if not isinstance(build_info, str):
        return None
    m = re.search(r"cuda=([^;]+)", build_info)
    if not m:
        return None
    return f"{m.group(1).strip()}, linked statically (eldpc.build_info())"


def device_record(dev) -> dict:
    """The class-level fields of one ``eldpc.devices()`` entry (no UUID)."""
    if not isinstance(dev, dict):
        return {}
    keep = ("name", "compute_capability", "sm_count", "l2_bytes", "cuda_driver_version",
            "nvidia_driver_version", "nvidia_driver_version_reason")
    return {k: dev.get(k) for k in keep if k in dev}


def first_device(devices) -> dict:
    if isinstance(devices, list) and devices and isinstance(devices[0], dict):
        return devices[0]
    return {}


def clean_command(raw: str, script: str, display: str) -> str:
    """Drop the interpreter path from a recorded command line.

    ``display`` is the script's path from the package root (the benchmark records only its file
    name); the arguments are kept as they were recorded, so the command can be re-run from the
    package root.
    """
    tokens = raw.split()
    for i, tok in enumerate(tokens):
        if Path(tok).name == script:
            return " ".join(["python", display, *tokens[i + 1:]])
    return raw


def bench_record_name(raw: dict) -> str:
    """The file name of a benchmark record: ``bench-quick.json`` for a ``--quick`` run,
    ``bench.json`` for a full-length one."""
    return "bench-quick.json" if raw.get("quick") else "bench.json"


def reference_lib_display(path) -> str:
    """The separately built library's path as a recorded command shows it: relative to the
    package root (the working directory of collect_results) when it is inside it, else a
    placeholder that names docs/VERIFY.md's recipe."""
    if path:
        try:
            return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
        except (ValueError, OSError):
            pass
    return "<separately-built-library>"


def assert_relative_command(command: str, where: str) -> None:
    """Refuse a recorded command that names an absolute path (the machine's file layout)."""
    for tok in command.split():
        value = tok.split("=", 1)[1] if tok.startswith("--") and "=" in tok else tok
        if value.startswith("/"):
            raise SystemExit(f"collect_results: the command of {where} names the absolute path "
                             f"{value!r}; re-run it with paths relative to the package root")


def drop_identifying_keys(obj):
    """A copy of ``obj`` without the keys that identify one physical device."""
    if isinstance(obj, dict):
        return {k: drop_identifying_keys(v) for k, v in obj.items() if k not in IDENTIFYING_KEYS}
    if isinstance(obj, list):
        return [drop_identifying_keys(v) for v in obj]
    return obj


def assert_no_machine_paths(obj, where: str) -> None:
    """Refuse to write a committed file that names this machine by absolute path."""

    def walk(value, path):
        if isinstance(value, str):
            if looks_like_abs_path(value):
                raise SystemExit(f"collect_results: {where}{path} contains an absolute path: {value!r}")
        elif isinstance(value, dict):
            for k, v in value.items():
                if k in IDENTIFYING_KEYS:
                    raise SystemExit(f"collect_results: {where}{path}.{k} identifies a device")
                walk(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")

    walk(obj, "")


def common_environment(env: dict, clocks: dict | None, *, verify: bool) -> dict:
    """The environment fields RESULTS.md prints, from either tool's record."""
    gpu = env.get("gpu") or {}
    cpu = env.get("cpu") or {}
    freq = cpu.get("frequency") or {}
    if verify:
        eldpc = env.get("eldpc") or {}
        devices = eldpc.get("devices")
        dev = first_device(devices)
        cpu_out = {"model": cpu.get("model"), "logical_cores": cpu.get("logical_cpus")}
        gpu_out = {"name": gpu.get("name"), "compute_capability": gpu.get("compute_capability")}
        # The suite's eldpc probe reports the devices eldpc.devices() sees; use its SM count and
        # L2 size, which nvidia-smi does not report.
        for key in ("sm_count", "l2_bytes"):
            if dev.get(key) is not None:
                gpu_out[key] = dev[key]
        build_info = eldpc.get("build_info")
        return {
            "gpu": gpu_out,
            "cpu": cpu_out,
            "driver": gpu.get("driver_version") or dev.get("nvidia_driver_version"),
            "cuda_driver_version": dev.get("cuda_driver_version"),
            "cuda_runtime": static_runtime(build_info),
            "toolkit": toolkit_string(env.get("nvcc")),
            "clocks": clocks,
            "python": env.get("python"),
            "platform": env.get("platform"),
            "eldpc": {"version": eldpc.get("version"), "isa": eldpc.get("isa"),
                      "build_info": build_info,
                      "devices": [device_record(d) for d in devices]
                      if isinstance(devices, list) else devices},
        }
    dev = first_device(env.get("devices_api"))
    return {
        "gpu": {k: gpu.get(k) for k in ("name", "compute_capability", "sm_count", "l2_bytes")},
        "cpu": {"model": cpu.get("model"), "physical_cores": cpu.get("physical_cores"),
                "logical_cores": cpu.get("logical_cores"), "pinned_cpu": cpu.get("pinned_cpu")},
        "driver": env.get("driver") or dev.get("nvidia_driver_version"),
        "cuda_driver_version": dev.get("cuda_driver_version"),
        "cuda_runtime": static_runtime(env.get("build_info")),
        "toolkit": toolkit_string(env.get("toolkit")),
        "clocks": clocks or {
            "before": (env.get("clocks") or {}).get("before"),
            "after": (env.get("clocks") or {}).get("after"),
            "source": "recorded by the benchmark before and after the measurements",
        },
        "cpu_frequency": {k: freq.get(k) for k in ("governor", "cur_freq_khz", "boost")},
        "python_packages": env.get("python_packages"),
        "eldpc": {"version": env.get("version"), "isa": env.get("isa"),
                  "build_info": env.get("build_info")},
    }


# ---------------------------------------------------------------------------
# the collectors


VERIFY_ROW_KEYS = (
    "problem", "config", "shots", "num_detectors", "num_mechanisms", "num_observables",
    "mismatches", "correction_mismatches", "observables_mismatches",
    "batch_correction_mismatches", "batch_observables_mismatches", "single_batch_mismatches",
    "status_mismatches_vs_reference", "status_mismatches_vs_reference_lib",
    "reference_lib_correction_mismatches", "reference_lib_observables_mismatches",
)


def collect_verify(raw: dict, *, command: str, commit: str, tree_clean: bool,
                   package_version: str) -> dict:
    env = raw.get("environment") or {}
    rows = []
    for row in raw.get("rows") or []:
        rows.append({k: row.get(k) for k in VERIFY_ROW_KEYS if k in row})
    sel = raw.get("selection") or {}
    out = {
        "schema": RESULT_SCHEMA,
        "kind": "verify",
        "package_version": package_version,
        "commit": commit,
        "package_tree_clean": tree_clean,
        "command": command,
        "mode": raw.get("mode"),
        "started_utc": raw.get("started_utc"),
        "environment": common_environment(env, clock_snapshot(), verify=True),
        "status": raw.get("status"),
        "totals": raw.get("totals"),
        "selection": {"base_seed": sel.get("base_seed"), "problems": sel.get("problems"),
                      "configs": sel.get("configs"), "shots": sel.get("shots")},
        "rows": rows,
        "timings": raw.get("timings"),
    }
    reference_lib = env.get("reference_lib")
    if isinstance(reference_lib, dict):
        # The suite's separately built library of the same decoder sources (--reference-lib or
        # ELDPC_REFERENCE_LIB); its path is a property of the machine and is not kept, except as a
        # path relative to the package root when it lies inside it (docs/VERIFY.md builds it in
        # build-ref/), so that the recorded command names the variable the run was made with.
        out["reference_lib"] = {"name": reference_lib.get("name"),
                                "build_info": reference_lib.get("build_info")}
        out["command"] = f"ELDPC_REFERENCE_LIB={reference_lib_display(reference_lib.get('path'))} {command}"
    out["reference_build_info"] = env.get("reference_library_build_info")
    return out


def collect_bench(raw: dict, *, command: str, commit: str, tree_clean: bool,
                  package_version: str) -> dict:
    env = raw.get("environment") or {}
    problem = raw.get("problem") or {}
    shots = problem.get("shots") or {}

    def shot_record(side):
        rec = shots.get(side) or {}
        return {"shots": rec.get("shots"), "seed": rec.get("seed"), "sha256": rec.get("sha256")}

    out = {
        "schema": RESULT_SCHEMA,
        "kind": "bench",
        "package_version": package_version,
        "commit": commit,
        "package_tree_clean": tree_clean,
        "command": command,
        "quick": raw.get("quick"),
        "started_utc": raw.get("generated_utc"),
        "elapsed_s": raw.get("elapsed_s"),
        "settings": raw.get("settings"),
        "environment": common_environment(env, None, verify=False),
        "problem": {
            "name": problem.get("name"),
            "source": problem.get("source"),
            "m": problem.get("m"), "n": problem.get("n"), "k": problem.get("k"),
            "nnz_h": problem.get("nnz_h"), "nnz_o": problem.get("nnz_o"),
            "recipe": problem.get("recipe"),
            "dem_sha256": problem.get("dem_sha256"),
            "shots": {"latency": shot_record("latency"), "ler": shot_record("ler")},
        },
        "warnings": raw.get("warnings") or [],
        "failures": raw.get("failures") or [],
    }
    for key in ("c_latency", "py_latency", "throughput", "ler"):
        out[key] = _strip_variant_libraries(raw.get(key))
    return out


def _strip_variant_libraries(section):
    """Drop every machine path the benchmark records inside a measurement section.

    The sections name the measured library (by absolute path) and, on some records, the problem and
    shot files.  RESULTS.md needs none of them: the library is identified by its build info in the
    environment block, and the problem by its dimensions and DEM hash.
    """
    if not isinstance(section, dict):
        return section
    out = json.loads(json.dumps(section))

    def scrub(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("library"), dict) and "path" in obj["library"]:
                obj["library"] = {"source": obj["library"].get("source")}
            for key in list(obj):
                value = obj[key]
                if key in ("path", "file", "boost_source") and isinstance(value, str):
                    del obj[key]
                    continue
                scrub(value)
        elif isinstance(obj, list):
            for value in obj:
                scrub(value)

    scrub(out)
    return out


def _config_string(config: dict) -> str:
    """The canonical ABI config string of a tuning choice (keys sorted)."""
    return ";".join(f"{k}={v}" for k, v in sorted(config.items()))


def collect_tune(raw: dict, *, command: str, commit: str, tree_clean: bool,
                 package_version: str, dem_file_sha256: str | None = None) -> dict:
    """The provenance of one eldpc-tune run: commit, command, choice and environment."""
    case = (raw.get("problem_cases") or [{}])[0]
    choice = raw.get("choice") or {}
    environment = raw.get("environment") or {}
    machine = environment.get("machine") or {}
    eldpc_env = environment.get("eldpc") or {}
    gpu = first_device((machine.get("gpu") or {}).get("devices"))
    # The toolkit version from the recorded `$CUDACXX --version` / `nvcc --version` output.
    toolkit = None
    for rec in environment.get("commands") or []:
        if "$CUDACXX" in str(rec.get("command")) or "nvcc" in str(rec.get("command")):
            m = re.search(r"release\s+(\d+\.\d+),\s*(V[\d.]+)", str(rec.get("stdout", "")))
            if m:
                toolkit = f"CUDA {m.group(1)} ({m.group(2)})"
    if toolkit is None:
        # Tuner 2.2 reads the toolkit from the loaded library's build_info and records it as
        # environment.toolkit ({"version": "13.3", "nvcc": "V13.3.73", "source": "build_info"})
        # instead of running nvcc.
        toolkit = toolkit_string(environment.get("toolkit"))
    tie = []
    for item in choice.get("tie_set") or []:
        tie.append({k: item.get(k) for k in
                    ("label", "ler_upper", "p50_us", "p99_us", "max_iter",
                     "ms_scaling_factor", "chosen")})
    record = {
        "schema": RESULT_SCHEMA,
        "kind": "tuning",
        "package_version": package_version,
        "commit": commit,
        "package_tree_clean": tree_clean,
        "command": command,
        "tool": {k: (raw.get("tool") or {}).get(k) for k in
                 ("name", "version", "seed", "quick", "auto", "started_utc", "duration_seconds")},
        "elapsed_seconds": raw.get("elapsed_seconds"),
        "problem": {
            "label": case.get("label"), "source": case.get("source"),
            "num_detectors": case.get("num_detectors"),
            "num_mechanisms": case.get("num_mechanisms"),
            "num_observables": case.get("num_observables"),
            "dem_sha256": case.get("dem_sha256"),
            "dem_file_sha256": dem_file_sha256,
        },
        "choice": {
            "config": choice.get("config"),
            "config_string": _config_string(choice.get("config") or {}),
            "ler_upper": choice.get("ler_upper"),
            "p50_us": choice.get("p50_us"),
            "p99_us": choice.get("p99_us"),
            "eligible": choice.get("eligible"),
            "measured": choice.get("measured"),
            "fastest": choice.get("fastest"),
            "tie_set": tie,
        },
        "environment": {
            "gpu": {"name": gpu.get("name"), "compute_capability": gpu.get("compute_capability"),
                    "sm_count": gpu.get("sm_count"), "l2_bytes": gpu.get("l2_bytes")},
            "cpu": machine.get("cpu_model"),
            "logical_cores": machine.get("cpu_count"),
            "driver": gpu.get("nvidia_driver_version"),
            "cuda_driver_version": gpu.get("cuda_driver_version") or gpu.get("driver_version"),
            "cuda_runtime": static_runtime(eldpc_env.get("build_info")),
            "toolkit": toolkit,
            "eldpc": {"version": eldpc_env.get("version"), "isa": eldpc_env.get("isa"),
                      "build_info": eldpc_env.get("build_info")},
        },
    }
    if "accuracy_confirmation" in raw:  # the eldpc.tune.v2 records
        record["tune_schema"] = raw.get("schema")
        record["choice"].update({k: choice.get(k) for k in (
            "ler", "ler_source", "screen", "slow_fraction", "p99_unreliable", "reason")})
        ac = raw.get("accuracy_confirmation") or {}
        record["accuracy_confirmation"] = {
            **{k: ac.get(k) for k in ("planned_shots", "shots_rule", "confidence", "target",
                                      "chosen")},
            "candidates": [{k: c.get(k) for k in ("label", "shots", "failures", "ler", "upper",
                                                  "passed", "latency_within", "accepted")}
                           for c in ac.get("candidates") or []]}
        latency = raw.get("confirmation") or {}
        record["latency_confirmation"] = {
            "distinct_syndromes": latency.get("distinct_syndromes"),
            "core_variant": latency.get("core_variant")} if latency else None
        variants = raw.get("core_variants") or {}
        record["core_variants"] = {k: variants.get(k) for k in (
            "evaluated", "table_variant", "faster", "ratio_p50", "ratio_p99", "band_p50",
            "band_p99", "recommended_env", "reason")} if variants else None
        record["warnings"] = list(raw.get("warnings") or [])
    return record


def collect_ler(raw: dict, *, command: str, commit: str, tree_clean: bool,
                package_version: str) -> dict:
    bench = collect_bench(raw, command=command, commit=commit, tree_clean=tree_clean,
                          package_version=package_version)
    return {
        "schema": RESULT_SCHEMA,
        "kind": "ler",
        "package_version": package_version,
        "commit": commit,
        "package_tree_clean": tree_clean,
        "command": command,
        "quick": bench["quick"],
        "started_utc": bench["started_utc"],
        "elapsed_s": bench["elapsed_s"],
        "environment": bench["environment"],
        "problem": bench["problem"],
        "warnings": bench["warnings"],
        "failures": bench["failures"],
        "ler": bench["ler"],
    }


def clean_tuning_dir(tune_dir: Path) -> list[str]:
    """Remove GPU UUIDs from the tuner's JSON files in place and drop its resume checkpoints.

    The tuner's own records (results.json, raw/*.json) keep everything else as written.  Returns
    the files that were changed.
    """
    changed = []
    for name in ("checkpoint.json", "confirmation-checkpoint.json"):
        checkpoint = tune_dir / "raw" / name
        if checkpoint.is_file():
            checkpoint.unlink()
            changed.append(str(checkpoint))
    for path in sorted(tune_dir.rglob("*.json")):
        if path.name == "provenance.json":
            continue
        data = load_json(path)
        cleaned = drop_identifying_keys(data)
        assert_no_machine_paths(cleaned, str(path))
        if json.dumps(cleaned) != json.dumps(data):
            # The tuner's own format (report.write_json): indent 1, keys in insertion order.
            path.write_text(json.dumps(cleaned, indent=1) + "\n", encoding="utf-8")
            changed.append(str(path))
    for path in sorted(tune_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if looks_like_abs_path(text):
            raise SystemExit(f"collect_results: {path} contains an absolute path")
    return changed


# ---------------------------------------------------------------------------


def write_record(data: dict, path: Path) -> None:
    data = drop_identifying_keys(data)
    assert_no_machine_paths(data, str(path.name))
    assert_relative_command(str(data.get("command") or ""), str(path.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    print(f"collect_results: wrote {path}")


def main(argv=None) -> int:
    here = Path(__file__).resolve()
    package_dir = here.parent.parent.parent   # <package>/docs/tools/this -> <package>

    p = argparse.ArgumentParser(description="collect verify/bench/tune outputs into docs/results/")
    p.add_argument("--verify", default=None, help="verify/run_verify.py's summary.json")
    p.add_argument("--bench", default=None, help="bench/bench.py all's results.json")
    p.add_argument("--ler", default=None, help="bench/bench.py ler's results.json")
    p.add_argument("--tune", default=None,
                   help="optional eldpc-tune results.json, inside the directory to publish "
                        "(for example docs/results/tuning-bb144/results.json)")
    p.add_argument("--out", default="docs/results", help="output directory (default: %(default)s)")
    p.add_argument("--commit", default=None,
                   help="commit to record (default: HEAD of the repository containing this package)")
    p.add_argument("--version", default=None, help="package version to record (default: pyproject.toml)")
    args = p.parse_args(argv)
    if not (args.verify or args.bench or args.ler):
        p.error("give at least one of --verify, --bench and --ler")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    commit = args.commit or commit_of(package_dir)
    if not commit:
        raise SystemExit("collect_results: cannot determine the git commit (pass --commit)")
    tree_clean = package_tree_clean(package_dir)
    version = args.version
    if version is None:
        text = (package_dir / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'version\s*=\s*"([^"]+)"', text)
        version = m.group(1) if m else None

    files = {}
    if args.verify:
        verify_raw = load_json(Path(args.verify))
        files["verify-quick.json"] = collect_verify(
            verify_raw,
            command=clean_command(verify_raw.get("environment", {}).get("command", ""),
                                  "run_verify.py", "verify/run_verify.py"),
            commit=commit, tree_clean=tree_clean, package_version=version)
    if args.bench:
        bench_raw = load_json(Path(args.bench))
        # bench-quick.json for a --quick run, bench.json for a full-length one.
        files[bench_record_name(bench_raw)] = collect_bench(
            bench_raw, command=clean_command(bench_raw.get("command", ""), "bench.py",
                                             "bench/bench.py"),
            commit=commit, tree_clean=tree_clean, package_version=version)
    if args.ler:
        ler_raw = load_json(Path(args.ler))
        files["ler-check.json"] = collect_ler(
            ler_raw, command=clean_command(ler_raw.get("command", ""), "bench.py",
                                           "bench/bench.py"),
            commit=commit, tree_clean=tree_clean, package_version=version)
    for name, data in files.items():
        write_record(data, out_dir / name)
    if args.tune:
        tune_path = Path(args.tune)
        tune_raw = load_json(tune_path)
        tool_cmd = list((tune_raw.get("tool") or {}).get("command") or [])
        tune_command = "python tuning/eldpc_tune.py " + " ".join(tool_cmd[1:])
        case = (tune_raw.get("problem_cases") or [{}])[0]
        dem_arg = case.get("path")
        dem_file_sha256 = None
        if dem_arg:
            candidate = Path(dem_arg)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.is_file():
                dem_file_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
        data = collect_tune(tune_raw, command=tune_command, commit=commit,
                            tree_clean=tree_clean, package_version=version,
                            dem_file_sha256=dem_file_sha256)
        tune_dir = tune_path.parent
        for changed in clean_tuning_dir(tune_dir):
            print(f"collect_results: cleaned {changed}")
        write_record(data, tune_dir / "provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
