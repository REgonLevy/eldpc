# SPDX-License-Identifier: Apache-2.0
"""The CPU core's thread placement as the ``cpus`` and ``threads`` keys set it, and the warning when
it puts a busy-waiting worker thread on every CPU the creating thread may run on.

A CPU decoder with ``threads=T`` starts ``T - 1`` worker threads at construction and pins them to
the first ``T - 1`` entries of its CPU list; they busy-wait between shots until the decoder is
closed. The last entry is the calling thread's CPU: the library does not pin the calling thread, so
that entry is a statement of where the caller runs, and the caller must be pinned there by its
owner (``os.sched_setaffinity``). Without ``cpus`` the list is the creating thread's affinity mask
in ascending order, repeated to ``T`` entries (the CPU core's rule, src/cpu/fastbpcpu6g.cpp).

When every CPU the creating thread may run on also carries a worker, the caller shares a CPU with
a thread that never yields and each decode is many times slower. The core does not check that;
:func:`placement_warning` does, after a CPU decoder was created.
"""

from __future__ import annotations

import os


def parse_cpu_list(text: str) -> list[int] | None:
    """The CPU ids of a ``cpus`` value ("1-7,0"; items ``a`` or ``a-b``), or None when malformed.

    The core is the authority on the syntax (it rejects a malformed list at create); this parser
    only needs to agree with it on the lists the core accepts.
    """
    out: list[int] = []
    for item in str(text).split(","):
        a, sep, b = item.partition("-")
        if not a.isdigit() or (sep and not b.isdigit()):
            return None
        lo = int(a)
        hi = int(b) if sep else lo
        if hi < lo:
            return None
        out.extend(range(lo, hi + 1))
    return out


def worker_cpus(cpus, threads: int, affinity: set[int]) -> list[int] | None:
    """The CPUs the ``threads - 1`` started threads are pinned to, or None when unknown."""
    if threads <= 1:
        return []
    if cpus is None:
        mask = sorted(affinity)
        if not mask:
            return None
        listed = [mask[i % len(mask)] for i in range(threads)]
    else:
        listed = parse_cpu_list(cpus)
        if listed is None or len(listed) != threads:
            return None
    return listed[: threads - 1]


def placement_warning(cpus, threads, affinity: set[int] | None = None) -> str | None:
    """The warning text when every CPU the creating thread may run on carries a worker, else None.

    ``cpus`` is the config's ``cpus`` value (None when absent), ``threads`` the threads per shot
    (None means the core's default, 1), ``affinity`` the creating thread's affinity mask (default:
    ``os.sched_getaffinity(0)``, the calling thread's on Linux).
    """
    if threads is None or int(threads) <= 1:
        return None
    if affinity is None:
        try:
            affinity = set(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            return None
    workers = worker_cpus(cpus, int(threads), affinity)
    if not workers or not affinity or not set(affinity) <= set(workers):
        return None
    where = "the cpus list" if cpus is not None else "the default CPU list (this thread's affinity mask, repeated)"
    mine = ",".join(str(c) for c in sorted(affinity)[:8]) + ("..." if len(affinity) > 8 else "")
    return (f"eldpc: {where} pins the CPU decoder's worker threads (its first threads-1 = "
            f"{int(threads) - 1} entries) to every CPU this thread may run on ({mine}); the workers "
            f"busy-wait, so this thread shares a CPU with one of them and each decode is many times "
            f"slower. The last cpus entry is the calling thread's own CPU: pin this thread to it "
            f"(os.sched_setaffinity) and leave it out of the first threads-1 entries, for example "
            f"cpus=1,2,3,0 with threads=4 and this thread on CPU 0.")
