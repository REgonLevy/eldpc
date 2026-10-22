# SPDX-License-Identifier: Apache-2.0
"""Machine environment of a benchmark run, and the conditions that invalidate timings.

Everything is collected through the operating system and the driver (nvidia-smi, /proc, libcuda), never
assumed: the results file must name the hardware it was measured on.  The probe takes a snapshot before
the measurements and completes it afterwards, so GPU clocks can be compared and a warning issued when
they moved.  Conditions that make a timing unrepresentative -- an unpinned run, another process on the
GPU, a busy host, a frequency governor that idles clocks, a virtual machine -- become entries of the
results' `warnings` list; they are recorded, not silently fixed.
"""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess

# CUDA driver API attributes (cuda.h).
_CU_ATTR_SM_COUNT = 16
_CU_ATTR_L2_CACHE_SIZE = 38
_CU_ATTR_CC_MAJOR = 75
_CU_ATTR_CC_MINOR = 76


def _run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.SubprocessError) as e:
        return 127, "", str(e)


def _read(path, default=None):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return default


def _cuda_devices() -> list[dict]:
    """Devices through the CUDA driver API (cuInit/cuDeviceGet*), independent of the Python package."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return []
    try:
        if cu.cuInit(0) != 0:
            return []
        count = ctypes.c_int()
        if cu.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return []
        out = []
        for i in range(count.value):
            dev = ctypes.c_int()
            if cu.cuDeviceGet(ctypes.byref(dev), i) != 0:
                continue
            name = ctypes.create_string_buffer(256)
            cu.cuDeviceGetName(name, 256, dev)
            attrs = {}
            for key, attr in (("sm_count", _CU_ATTR_SM_COUNT), ("l2_bytes", _CU_ATTR_L2_CACHE_SIZE),
                              ("cc_major", _CU_ATTR_CC_MAJOR), ("cc_minor", _CU_ATTR_CC_MINOR)):
                v = ctypes.c_int()
                attrs[key] = int(v.value) if cu.cuDeviceGetAttribute(ctypes.byref(v), attr, dev) == 0 else None
            total = ctypes.c_size_t()
            mem = int(total.value) if cu.cuDeviceTotalMem_v2(ctypes.byref(total), dev) == 0 else None
            out.append({
                "index": i,
                "name": name.value.decode("utf-8", "replace"),
                "sm_count": attrs["sm_count"],
                "l2_bytes": attrs["l2_bytes"],
                "compute_capability": (f"{attrs['cc_major']}.{attrs['cc_minor']}"
                                       if attrs["cc_major"] is not None else None),
                "memory_bytes": mem,
            })
        return out
    except (AttributeError, OSError):
        return []


def _smi_gpus() -> list[dict]:
    """Per-GPU rows from nvidia-smi: name, driver, compute capability, memory, current clocks.

    When CUDA_VISIBLE_DEVICES selects GPUs, the rows are filtered and ordered to match it, so the first
    row is the GPU the benchmark's CUDA calls see as device 0.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    fields = ("index,uuid,name,driver_version,compute_cap,memory.total,clocks.current.graphics,"
              "clocks.current.memory")
    rc, out, _ = _run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if rc != 0:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue

        def num(s):
            try:
                return float(s)
            except ValueError:
                return None

        rows.append({
            "index": int(parts[0]) if parts[0].isdigit() else None,
            "uuid": parts[1],
            "name": parts[2],
            "driver_version": parts[3],
            "compute_capability": parts[4],
            "memory_mib": num(parts[5]),
            "clocks": {"graphics_mhz": num(parts[6]), "memory_mhz": num(parts[7])},
        })
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [v.strip() for v in visible.split(",") if v.strip()]
        if ids and all(i.isdigit() for i in ids):
            by_index = {r["index"]: r for r in rows}
            rows = [by_index[int(i)] for i in ids if int(i) in by_index]
        elif ids and all(i.startswith("GPU-") or i.startswith("MIG-") for i in ids):
            by_uuid = {r["uuid"]: r for r in rows}
            rows = [by_uuid[i] for i in ids if i in by_uuid]
    return rows


def _smi_compute_apps() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    rc, out, _ = _run([exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if rc != 0:
        return []
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        apps.append({"pid": pid, "process_name": parts[1], "used_memory_mib": parts[2]})
    return apps


def _toolkit(build_info: str | None = None) -> dict | None:
    """The CUDA toolkit of the measured library: ``{version, nvcc, first_line, source}``.

    The loaded library's ``build_info()`` names the toolkit that compiled its GPU cores (and whose
    runtime is linked into them) as ``cuda=<compiler id> <version>``; that is the toolkit of the
    measurement whether or not its ``nvcc`` is on ``PATH`` now, so it is recorded with
    ``source: "build_info"``. Only when ``build_info`` has no ``cuda=`` field (a CPU-only build, or
    no library loaded) is ``nvcc --version`` asked (``CUDACXX``, else the ``nvcc`` on ``PATH``;
    ``source: "nvcc"``); None when neither names a toolkit.
    """
    m = re.search(r"(?:^|[ ;])cuda=\S+ (\d+)\.(\d+)((?:\.\d+)*)", build_info or "")
    if m:
        full = f"{m.group(1)}.{m.group(2)}{m.group(3)}"
        return {"version": f"{m.group(1)}.{m.group(2)}", "nvcc": f"V{full}", "first_line": None,
                "source": "build_info"}
    nvcc = os.environ.get("CUDACXX") or shutil.which("nvcc")
    if not nvcc or not os.path.exists(nvcc):
        return None
    rc, out, err = _run([nvcc, "--version"])
    text = (out or err).strip()
    if rc != 0 or not text:
        return None
    m = re.search(r"release\s+([\d.]+),\s*(V[\d.]+)", text)
    return {"version": m.group(1) if m else None, "nvcc": m.group(2) if m else None,
            "first_line": text.splitlines()[0] if text else None, "source": "nvcc"}


def _cpu_info() -> dict:
    text = _read("/proc/cpuinfo", "") or ""
    model = None
    cpu_cores_field = None
    hypervisor = "hypervisor" in text
    hv_vendor = None
    physical_ids, core_pairs = set(), set()
    for block in text.split("\n\n"):
        phys = core = None
        for line in block.splitlines():
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k == "model name" and model is None:
                model = v
            elif k == "cpu cores" and cpu_cores_field is None and v.isdigit():
                cpu_cores_field = int(v)
            elif k == "physical id" and v.isdigit():
                phys = v
            elif k == "core id" and v.isdigit():
                core = v
            elif k == "hypervisor vendor" and hv_vendor is None:
                hv_vendor = v
        if phys is not None:
            physical_ids.add(phys)
        if phys is not None and core is not None:
            core_pairs.add((phys, core))
    # Count distinct (socket, core) pairs over the processor blocks; fall back to the per-socket "cpu cores".
    physical_cores = len(core_pairs) if core_pairs else cpu_cores_field
    return {"model": model, "physical_cores": physical_cores, "sockets": len(physical_ids) or None,
            "logical_cores": os.cpu_count(), "hypervisor": hypervisor,
            "hypervisor_vendor": hv_vendor}


def _cpu_freq(cpu: int | None) -> dict:
    out = {}
    root = f"/sys/devices/system/cpu/cpu{cpu if cpu is not None else 0}"
    base = os.path.join(root, "cpufreq")
    gov = _read(os.path.join(base, "scaling_governor"))
    if gov:
        out["governor"] = gov
    for key, path in (("cur_freq_khz", "scaling_cur_freq"), ("max_freq_khz", "cpuinfo_max_freq"),
                      ("min_freq_khz", "cpuinfo_min_freq")):
        v = _read(os.path.join(base, path))
        if v and v.isdigit():
            out[key] = int(v)
    # Power/boost state and the SMT sibling of the pinned CPU: both explain residual tail variance.
    for key, path in (("energy_performance_preference", "cpufreq/energy_performance_preference"),
                      ("scaling_driver", "cpufreq/scaling_driver"),
                      ("core_id", "topology/core_id"),
                      ("thread_siblings_list", "topology/thread_siblings_list")):
        v = _read(os.path.join(root, path))
        if v is not None:
            out[key] = v
    out["boost"], out["boost_source"] = _cpu_boost(root)
    return out


def _cpu_boost(cpu_root: str) -> tuple[str, str | None]:
    """(boost state, the file it was read from).

    The state is "1" (boost / turbo allowed) or "0" (disabled), read from the first of these that exists:
    the pinned CPU's cpufreq/boost, the global cpufreq/boost (acpi-cpufreq, amd-pstate), or intel_pstate's
    no_turbo (inverted: no_turbo=1 means boost is off).  When none of these files exists, e.g. in a
    virtual machine or on a driver without a switch, the state is "unknown" -- never absent.
    """
    for path in (os.path.join(cpu_root, "cpufreq", "boost"), "/sys/devices/system/cpu/cpufreq/boost"):
        v = _read(path)
        if v in ("0", "1"):
            return v, path
    path = "/sys/devices/system/cpu/intel_pstate/no_turbo"
    v = _read(path)
    if v in ("0", "1"):
        return ("0" if v == "1" else "1"), path
    return "unknown", None


def _affinity_size() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def parse_cpu_list(text: str | None) -> set[int] | None:
    """The CPUs of a kernel CPU list ("0-31,64-95"), or None when it is missing or malformed."""
    if not text or not text.strip():
        return None
    cpus: set[int] = set()
    try:
        for item in text.strip().split(","):
            a, sep, b = item.strip().partition("-")
            lo = int(a)
            hi = int(b) if sep else lo
            if hi < lo:
                return None
            cpus.update(range(lo, hi + 1))
    except ValueError:
        return None
    return cpus


def cpu_quota(cgroup_root: str = "/sys/fs/cgroup") -> float | None:
    """This process's cgroup CPU quota in CPUs (quota / period), or None when there is none.

    cgroup v2 writes it to ``cpu.max`` ("max 100000" when unlimited, "5375999 100000" for about
    54 CPUs); cgroup v1 to ``cpu/cpu.cfs_quota_us`` (-1 when unlimited) and ``cpu.cfs_period_us``.
    """
    text = _read(os.path.join(cgroup_root, "cpu.max"))
    try:
        if text:
            quota, _, period = text.strip().partition(" ")
            if quota == "max":
                return None
            return int(quota) / int(period or "100000")
        for sub in ("cpu", "cpu,cpuacct"):
            q = _read(os.path.join(cgroup_root, sub, "cpu.cfs_quota_us"))
            per = _read(os.path.join(cgroup_root, sub, "cpu.cfs_period_us"))
            if q and per:
                return None if int(q) < 0 else int(q) / int(per)
    except (ValueError, ZeroDivisionError):
        return None
    return None


#: The host-load warning fires when the 1-min load average, sampled before and after the run (not
#: during it, so a load peak inside the run is not seen), is above
#: LOAD_BASE + LOAD_FRACTION x the host's logical CPUs.  The load average counts the runnable
#: threads of the whole machine, and they disturb a timing through the machine's CPUs, so neither
#: the pinned affinity mask of a timing run nor a cgroup CPU quota (recorded as cpu_quota, for
#: information) is the right denominator: a container whose quota is a quarter of a quiet host
#: is not loaded, and one on a half-busy host is, whatever its quota.
LOAD_BASE = 1.0
LOAD_FRACTION = 0.05


def load_threshold(ncpu: int) -> float:
    return LOAD_BASE + LOAD_FRACTION * max(1, ncpu)


def load_warnings(la0: list[float] | None, la1: list[float] | None, ncpu: int) -> list[str]:
    """The host-load warnings for the 1-min load averages before (la0) and after (la1) a run."""
    limit = load_threshold(ncpu)
    high = [(when, la[0]) for when, la in (("before", la0), ("after", la1)) if la is not None and la[0] > limit]
    if high:
        when, value = max(high, key=lambda x: x[1])
        return [f"host load average (sampled before and after the run) is high: {value:.2f} {when} the "
                f"run on {ncpu} logical CPUs (the warning threshold is {limit:.2f}, 1 + 5% of them); other "
                "work on the host adds noise to the timings"]
    if la0 is not None and la1 is not None and la1[0] - la0[0] > 1.0:
        return [f"host load average (sampled before and after the run) rose during the run: "
                f"{la0[0]:.2f} -> {la1[0]:.2f}"]
    return []


def gpu_local_cpus() -> set[int] | None:
    """The CPUs of the NUMA node of the GPU this process sees as device 0 (its PCI device's
    ``local_cpulist`` in sysfs), or None when nvidia-smi or sysfs cannot say."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    rc, out, _ = _run([exe, "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"])
    if rc != 0:
        return None
    rows = [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.count(",") >= 2]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        first = visible.split(",")[0].strip()
        if not first:
            return None
        rows = [r for r in rows if r[0] == first or r[1] == first or r[1].startswith(first)]
    if not rows:
        return None
    domain, _, rest = rows[0][2].partition(":")
    bdf = f"{domain[-4:]}:{rest}".lower()
    return parse_cpu_list(_read(f"/sys/bus/pci/devices/{bdf}/local_cpulist"))


def smt_siblings(cpu: int) -> set[int]:
    """The SMT siblings of ``cpu`` (itself included), from sysfs; {cpu} when unknown."""
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    return parse_cpu_list(_read(path)) or {cpu}


def auto_cpu(allowed: list[int], gpu_local: set[int] | None, cpu0_siblings: set[int]) -> tuple[int, str]:
    """The CPU ``--cpu auto`` pins to, and why: the first allowed CPU of the GPU's NUMA node that is
    neither CPU 0 nor an SMT sibling of it (CPU 0 takes most interrupts and housekeeping work); else
    the first allowed CPU that is neither; else the first allowed CPU."""
    avoid = {0} | set(cpu0_siblings)
    near = [c for c in allowed if gpu_local and c in gpu_local and c not in avoid]
    if near:
        return near[0], "the first CPU of the GPU's NUMA node other than CPU 0 and its SMT sibling"
    rest = [c for c in allowed if c not in avoid]
    if rest:
        why = ("the first allowed CPU other than CPU 0 and its SMT sibling"
               + ("" if gpu_local is None else "; none of the GPU's NUMA node is allowed"))
        return rest[0], why
    return allowed[0], "the only CPUs allowed are CPU 0 and its SMT sibling"


def snapshot(pinned_cpu: int | None, eldpc_module) -> dict:
    """Everything environmental that is known before the first measurement."""
    gpus = _smi_gpus()
    cuda = _cuda_devices()
    for g, c in zip(gpus, cuda):
        if not g.get("sm_count"):
            g["sm_count"] = c.get("sm_count")
        if not g.get("l2_bytes"):
            g["l2_bytes"] = c.get("l2_bytes")
        if not g.get("memory_bytes"):
            g["memory_bytes"] = c.get("memory_bytes")
        if g.get("compute_capability") in (None, "N/A"):
            g["compute_capability"] = c.get("compute_capability")
    for c in cuda:
        if not any(g.get("name") == c.get("name") for g in gpus):
            gpus.append(c)
    gpu0 = gpus[0] if gpus else {}
    env = {
        "gpu": {
            "name": gpu0.get("name"),
            "sm_count": gpu0.get("sm_count"),
            "l2_bytes": gpu0.get("l2_bytes"),
            "memory_bytes": gpu0.get("memory_bytes"),
            "memory_mib": gpu0.get("memory_mib"),
            "compute_capability": gpu0.get("compute_capability"),
        },
        "gpus": gpus,
        "cpu": {
            **_cpu_info(),
            "pinned_cpu": pinned_cpu,
            "affinity_cpus": _affinity_size(),
            "cpu_quota": cpu_quota(),
            "frequency": _cpu_freq(pinned_cpu),
        },
        "driver": gpu0.get("driver_version"),
        "cuda_runtime": None,
        "cuda_runtime_library": None,
        "toolkit": None,
        "python_packages": _python_packages(),
        "clocks": {"before": gpu0.get("clocks"), "after": None},
        "isa": None,
        "version": None,
        "build_info": None,
        "loadavg": _loadavg(),
        "gpu_processes": _smi_compute_apps(),
        "pid": os.getpid(),
    }
    try:
        env["isa"] = eldpc_module.isa()
        env["version"] = eldpc_module.__version__
        env["build_info"] = eldpc_module.build_info()
        env["devices_api"] = eldpc_module.devices()
    except Exception as e:  # the package must stay usable even if a probe is not
        env["probe_error"] = f"{type(e).__name__}: {e}"
    # The CUDA runtime is linked statically into the eldpc cores: its version is the build
    # toolkit's, which build_info() records as "cuda=<compiler id> <version>".
    m = re.search(r"cuda=\S+ (\d+\.\d+(?:\.\d+)?)", env.get("build_info") or "")
    if m:
        env["cuda_runtime"] = m.group(1)
        env["cuda_runtime_library"] = "static (linked into the eldpc cores)"
    # The build toolkit, from the same build_info (nvcc --version only when it names none).
    env["toolkit"] = _toolkit(env.get("build_info"))
    return env


def _python_packages() -> dict:
    """Versions of the packages that build the problems and sample shots (when installed)."""
    out = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for name in ("numpy", "scipy", "stim", "qldpc"):
            try:
                out[name] = version(name)
            except PackageNotFoundError:
                out[name] = None
    except Exception:
        pass
    return out


def _loadavg() -> list[float] | None:
    text = _read("/proc/loadavg")
    if not text:
        return None
    try:
        return [float(x) for x in text.split()[:3]]
    except ValueError:
        return None


def note_gpu_processes(env: dict, after: str) -> None:
    """Record the other processes using the GPU right after a measurement finished (`after` names it).

    nvidia-smi is sampled between the measurements, not inside them, so a process that starts and ends
    within one measurement is still missed; the list is a lower bound.
    """
    others = [a for a in _smi_compute_apps() if a["pid"] != env.get("pid")]
    env.setdefault("gpu_processes_between", []).append({"after": after, "processes": others})


def finish(env: dict) -> list[str]:
    """Take the after-snapshot and return the warnings that apply to this run."""
    warnings = []
    gpus = _smi_gpus()
    env["clocks"]["after"] = gpus[0]["clocks"] if gpus else None
    env["loadavg_after"] = _loadavg()
    env["gpu_processes_after"] = _smi_compute_apps()

    if not env["gpu"].get("name"):
        warnings.append("no GPU was visible (nvidia-smi or the CUDA driver did not report one); "
                        "the GPU numbers were not measured on a GPU")
    if env["cpu"].get("hypervisor"):
        warnings.append("this is a virtual machine (hypervisor CPU flag): timings include virtualization "
                        "overhead and the GPU may be passed through")
    freq = env["cpu"].get("frequency", {})
    gov = freq.get("governor")
    if gov is not None and gov not in ("performance", "userspace"):
        warnings.append(f"CPU governor is '{gov}' on the pinned CPU: its frequency may change during the "
                        "run; use the 'performance' governor for publishable numbers")
    epp = freq.get("energy_performance_preference")
    if epp is not None and epp not in ("performance", "0"):
        warnings.append(f"CPU energy-performance preference is '{epp}' on the pinned CPU: its frequency "
                        "may be slower to rise; set it to 'performance' for stable latency")
    if env["cpu"].get("pinned_cpu") is None:
        warnings.append("the run was not pinned to a CPU (--allow-unpinned): process migration and "
                        "frequency changes add noise")

    apps_before = [a for a in env.get("gpu_processes", []) if a["pid"] != env["pid"]]
    apps_after = [a for a in env.get("gpu_processes_after", []) if a["pid"] != env["pid"]]
    apps_between = sum(len(x["processes"]) for x in env.get("gpu_processes_between", []))
    if apps_before or apps_after or apps_between:
        warnings.append("another process was using the GPU during the run (seen before: "
                        f"{len(apps_before)}, between measurements: {apps_between}, after: "
                        f"{len(apps_after)}); its occupancy adds gaps of milliseconds to the latency tail")

    cpu = env["cpu"]
    ncpu = max(1, int(cpu.get("logical_cores") or 1))
    warnings.extend(load_warnings(env.get("loadavg"), env.get("loadavg_after"), ncpu))

    c0, c1 = env["clocks"].get("before"), env["clocks"].get("after")
    if c0 and c1:
        for key in ("graphics_mhz", "memory_mhz"):
            a, b = c0.get(key), c1.get(key)
            if a and b and abs(b - a) > 0.1 * max(a, b):
                warnings.append(f"GPU clock changed during the run: {key} {a:g} -> {b:g} MHz "
                                "(the GPU was not at a steady clock)")
    rt, tk = env.get("cuda_runtime"), env.get("toolkit")
    if rt and tk and tk.get("version"):
        try:
            if [int(x) for x in rt.split(".")[:2]] != [int(x) for x in tk["version"].split(".")[:2]]:
                warnings.append(f"CUDA runtime {rt} (linked into eldpc) and the toolkit on PATH "
                                f"{tk['version']} differ in major.minor")
        except ValueError:
            pass
    return warnings
