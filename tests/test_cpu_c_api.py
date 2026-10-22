# SPDX-License-Identifier: Apache-2.0
"""A C program built against the installed header creates a CPU handle through the C API.

This covers the C-level half of the device routing end to end: the config string carries
``device=cpu;threads=2;cpus=...``, ``eldpc_core_variant()`` is ``"cpu"``, ``eldpc_keep_warm()`` is
``ELDPC_OK``, and the correction, observables and status word equal the Python API's on the same
problem. It needs a C compiler and the installed header/library; without either it skips (it is not
one of the ``cpu_core`` marker's tests, which must never skip).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import eldpc

PROGRAM = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "eldpc.h"

/* H = [[1,1,0,0,1],[0,1,1,0,0],[0,0,1,1,1]] and O = [[0,1,0,1,1]] in CSC. */
static const uint32_t h_colptr[6] = {0, 1, 3, 5, 6, 8};
static const uint32_t h_rowind[8] = {0, 0, 1, 1, 2, 2, 0, 2};
static const uint32_t o_colptr[6] = {0, 0, 1, 1, 2, 3};
static const uint32_t o_rowind[3] = {0, 0, 0};
static const double priors[5] = {0.05, 0.10, 0.02, 0.20, 0.01};

int main(int argc, char **argv) {
    if (argc != 2) return 20;
    eldpc_problem p;
    memset(&p, 0, sizeof p);
    p.abi_version = ELDPC_API_VERSION;
    p.num_detectors = 3;
    p.num_mechanisms = 5;
    p.num_observables = 1;
    p.h_colptr = h_colptr;
    p.h_rowind = h_rowind;
    p.o_colptr = o_colptr;
    p.o_rowind = o_rowind;
    p.priors = priors;
    p.dem_text = NULL;

    char config[512];
    snprintf(config, sizeof config,
             "decoder=bposd;bp_method=ms;schedule=parallel;osd_method=osd0;osd_order=0;"
             "ms_scaling_factor=0.3125;max_iter=3;device=cpu;threads=2;cpus=%s", argv[1]);
    char err[512];
    err[0] = '\0';
    eldpc_decoder *dec = eldpc_create(config, &p, err, sizeof err);
    if (dec == NULL) {
        fprintf(stderr, "create: %s\n", err);
        return 4;
    }
    const char *variant = eldpc_core_variant(dec);
    if (variant == NULL || strcmp(variant, "cpu") != 0) {
        fprintf(stderr, "variant: %s\n", variant == NULL ? "(null)" : variant);
        return 12;
    }
    const uint8_t syndrome[3] = {1, 0, 0};
    uint8_t corr[5], obs[1];
    uint32_t status = 0;
    if (eldpc_decode(dec, syndrome, corr, obs, &status) != ELDPC_OK) return 5;
    if (eldpc_keep_warm(dec, ELDPC_KEEP_WARM_GPU) != ELDPC_OK) return 9;
    printf("%s\n", variant);
    printf("%s\n", eldpc_isa());
    for (int i = 0; i < 5; ++i) printf("%u", (unsigned)corr[i]);
    printf(" %u %u\n", (unsigned)obs[0], (unsigned)status);
    eldpc_destroy(dec);
    printf("ok\n");
    return 0;
}
"""


def test_c_program_creates_a_cpu_handle(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    package = Path(eldpc.__file__).resolve().parent
    header = package / "include" / "eldpc.h"
    libs = package / "_libs"
    if not header.is_file() or not (libs / "libeldpc.so").is_file():
        pytest.skip("the installed package does not ship the C header and library")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("the test needs two CPUs of the process's affinity mask")
    cpus = f"{allowed[0]},{allowed[1]}"

    src = tmp_path / "probe_cpu.c"
    src.write_text(PROGRAM)
    exe = tmp_path / "probe_cpu"
    build = subprocess.run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", f"-I{package / 'include'}",
                            str(src), f"-L{libs}", "-leldpc", f"-Wl,-rpath,{libs}", "-o", str(exe)],
                           capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stderr

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # the CPU handle needs no device
    run = subprocess.run([str(exe), cpus], capture_output=True, text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert lines[-1] == "ok"
    assert lines[0] == "cpu"
    assert lines[1] in ("avx2", "avx512")
    bits, obs, status = lines[2].split()

    H = np.array([[1, 1, 0, 0, 1], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=np.uint8)
    O = np.array([[0, 1, 0, 1, 1]], dtype=np.uint8)
    priors = np.array([0.05, 0.10, 0.02, 0.20, 0.01])
    config = dict(eldpc.presets()["a03125c3"], cpus=cpus)
    dec = eldpc.Decoder(H, O, priors, config=config, device="cpu", threads=2)
    try:
        corr, obs_py, st = dec.decode_full(np.array([1, 0, 0], dtype=np.uint8))
    finally:
        dec.close()
    assert bits == "".join(str(int(b)) for b in corr)
    assert int(obs) == int(obs_py[0]) and int(status) == st
