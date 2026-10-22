# SPDX-License-Identifier: Apache-2.0
"""The CPU core's ``cpus`` list and the warning when it leaves the creating thread no CPU of its own.

The started threads take the first ``threads - 1`` entries of ``cpus``; the last entry is the
calling thread's CPU, which the library does not pin. A list that puts a busy-waiting worker on
every CPU the creating thread may run on makes decodes many times slower, so the Python layer warns
(eldpc._placement). These tests need no CUDA device.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from eldpc import _placement
from eldpc_testlib import cpu_core

PACKAGE = Path(__file__).resolve().parent.parent


@cpu_core
def test_cpu_list_parser_agrees_with_the_core_syntax():
    assert _placement.parse_cpu_list("1-3,0") == [1, 2, 3, 0]
    assert _placement.parse_cpu_list("5") == [5]
    assert _placement.parse_cpu_list("2,2") == [2, 2]
    for bad in ("", "1,", "a", "3-1", "1--2", " 1"):
        assert _placement.parse_cpu_list(bad) is None, bad


@cpu_core
def test_worker_cpus_are_the_first_threads_minus_one_entries():
    assert _placement.worker_cpus("1,2,3,0", 4, {0}) == [1, 2, 3]
    assert _placement.worker_cpus("0-3", 4, {0}) == [0, 1, 2]
    assert _placement.worker_cpus(None, 4, {6, 2}) == [2, 6, 2]   # the affinity mask, repeated
    assert _placement.worker_cpus(None, 1, {0}) == []
    assert _placement.worker_cpus("1,2", 3, {0}) is None          # the core rejects it at create


@cpu_core
def test_placement_warning_rule():
    # The caller pinned to CPU 0 and a worker on CPU 0: warn, and say what the last entry is for.
    text = _placement.placement_warning("0-7", 8, {0})
    assert text is not None and "last cpus entry is the calling thread's own CPU" in text
    # The documented form: workers elsewhere, the caller's CPU last.
    assert _placement.placement_warning("1,2,3,4,5,6,7,0", 8, {0}) is None
    # Without cpus, a pinned caller gets every worker on its own CPU.
    assert _placement.placement_warning(None, 4, {3}) is not None
    # An unpinned caller can still run on a CPU without a worker.
    assert _placement.placement_warning("0-7", 8, set(range(16))) is None
    assert _placement.placement_warning(None, 8, set(range(16))) is None
    # threads=1 starts no thread.
    assert _placement.placement_warning("0", 1, {0}) is None
    assert _placement.placement_warning(None, None, {0}) is None


SCRIPT = textwrap.dedent("""
    import os, warnings
    import numpy as np
    import eldpc
    a, b = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, {a})
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    O = np.array([[1, 0, 1]], dtype=np.uint8)
    def warned(**kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with eldpc.Decoder(H, O, [0.01] * 3, device="cpu", threads=2, **kw) as dec:
                out = dec.decode_full([1, 0])
        assert out[0].tolist() == [1, 0, 0]
        return sum("busy-wait" in str(w.message) for w in caught if w.category is RuntimeWarning)
    print("worker-on-caller", warned(config=dict(eldpc.presets()["a03125c3"], cpus=f"{a},{a}")))
    print("caller-last", warned(config=dict(eldpc.presets()["a03125c3"], cpus=f"{b},{a}")))
    print("string", warned(config=f"a03125c3;cpus={a},{a}"))
    print("default", warned(config="a03125c3"))
""")


@cpu_core
def test_a_cpu_decoder_whose_worker_takes_the_callers_cpu_warns():
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("the test needs two CPUs of the process's affinity mask")
    import eldpc

    env = os.environ.copy()
    env.pop("ELDPC_ISA", None)
    package_parent = str(Path(eldpc.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env,
                          timeout=300, cwd=str(PACKAGE))
    assert proc.returncode == 0, proc.stderr
    counts = dict(line.split() for line in proc.stdout.splitlines())
    assert counts == {"worker-on-caller": "1", "caller-last": "0", "string": "1", "default": "1"}


PINNING_SCRIPT = textwrap.dedent("""
    import os, warnings
    import numpy as np
    import eldpc
    cpus = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, {cpus[-1]})          # the caller on the last entry, as bench.py --cpu N does

    def allowed(tid):
        with open(f"/proc/self/task/{tid}/status") as f:
            return next(line.split()[1] for line in f if line.startswith("Cpus_allowed_list:"))

    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    O = np.array([[1, 0, 1]], dtype=np.uint8)
    before = set(os.listdir("/proc/self/task"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with eldpc.Decoder(H, O, [0.01] * 3, device="cpu", threads=4,
                           config="a03125c3;cpus=" + ",".join(map(str, cpus))) as dec:
            started = sorted(set(os.listdir("/proc/self/task")) - before, key=int)
            print("workers", ",".join(sorted((allowed(t) for t in started), key=int)))
            print("caller", allowed(os.getpid()))
            assert dec.decode_full([1, 0])[0].tolist() == [1, 0, 0]
    print("expected", ",".join(map(str, cpus[:3])), cpus[-1])
""")


@cpu_core
def test_workers_are_pinned_to_the_cpus_list_when_the_caller_is_pinned_to_its_last_entry():
    """With the creating thread pinned to one CPU (the last cpus entry; bench.py --cpu N, or
    os.sched_setaffinity), the started threads still run on the list's first threads - 1 CPUs, one
    each, and no placement warning is raised."""
    if len(os.sched_getaffinity(0)) < 4:
        pytest.skip("the test needs four CPUs of the process's affinity mask")
    import eldpc

    env = os.environ.copy()
    env.pop("ELDPC_ISA", None)
    package_parent = str(Path(eldpc.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run([sys.executable, "-c", PINNING_SCRIPT], capture_output=True, text=True,
                          env=env, timeout=300, cwd=str(PACKAGE))
    assert proc.returncode == 0, proc.stderr
    out = dict(line.split(" ", 1) for line in proc.stdout.splitlines())
    workers, caller = out["expected"].split()
    assert out["workers"] == workers
    assert out["caller"] == caller
