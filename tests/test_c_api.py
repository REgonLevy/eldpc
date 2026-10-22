# SPDX-License-Identifier: Apache-2.0
"""A C program built against the installed include/eldpc.h and libeldpc.so."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import eldpc
from eldpc_testlib import requires_gpu, tiny_problem

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

int main(void) {
    if (eldpc_api_version() != ELDPC_API_VERSION) return 2;
    if (strlen(eldpc_version()) == 0) return 3;
    const char *info = eldpc_build_info();
    if (info == NULL || strlen(info) == 0) return 10;

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

    char err[512];
    err[0] = '\0';
    eldpc_decoder *dec = eldpc_create(
        "decoder=bposd;bp_method=ms;schedule=parallel;osd_method=osd0;osd_order=0;"
        "ms_scaling_factor=0.3125;max_iter=3", &p, err, sizeof err);
    if (dec == NULL) {
        fprintf(stderr, "create: %s\n", err);
        return 4;
    }

    const char *variant = eldpc_core_variant(dec);
    if (variant == NULL) return 11;
    if (strcmp(variant, "default") != 0 && strcmp(variant, "ada64") != 0) return 12;
    if (strcmp(eldpc_core_variant(NULL), "") != 0) return 13;
    if (strcmp(eldpc_core_variant_for_device(8, 9, 142), "ada64") != 0) return 14;
    if (strcmp(eldpc_core_variant_for_device(8, 9, 76), "default") != 0) return 15;
    if (strcmp(eldpc_core_variant_for_device(12, 0, 84), "default") != 0) return 16;

    const uint8_t syndrome[3] = {1, 0, 0};
    uint8_t corr[5], obs[1];
    uint32_t status = 0;
    if (eldpc_decode(dec, syndrome, corr, obs, &status) != ELDPC_OK) return 5;
    printf("%s\n", eldpc_isa());
    printf("%s\n", variant);
    for (int i = 0; i < 5; ++i) printf("%u", (unsigned)corr[i]);
    printf(" %u %u\n", (unsigned)obs[0], (unsigned)status);

    double llr[5];
    if (eldpc_get_soft_output(dec, llr, 5) != ELDPC_OK) return 6;
    if (eldpc_set_priors(dec, llr, 5) != ELDPC_OK) return 7;
    if (eldpc_decode(dec, syndrome, corr, obs, &status) != ELDPC_OK) return 8;
    if (eldpc_keep_warm(dec, ELDPC_KEEP_WARM_GPU) != ELDPC_OK) return 9;
    eldpc_destroy(dec);
    eldpc_destroy(NULL);
    printf("ok\n");
    return 0;
}
"""


@requires_gpu
def test_c_program_links_and_decodes(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    package = Path(eldpc.__file__).resolve().parent
    header = package / "include" / "eldpc.h"
    libs = package / "_libs"
    if not header.is_file() or not (libs / "libeldpc.so").is_file():
        pytest.skip("the installed package does not ship the C header and library")
    src = tmp_path / "probe.c"
    src.write_text(PROGRAM)
    exe = tmp_path / "probe"
    build = subprocess.run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", f"-I{package / 'include'}",
                            str(src), f"-L{libs}", "-leldpc", f"-Wl,-rpath,{libs}", "-o", str(exe)],
                           capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert lines[-1] == "ok"
    assert lines[0] in ("avx2", "avx512")
    assert lines[1] in ("default", "ada64")
    bits, obs, status = lines[2].split()
    assert len(bits) == 5 and obs in ("0", "1")

    # The C program's result equals the Python API's on the same problem.
    H, O, priors = tiny_problem()
    dec = eldpc.Decoder(H, O, priors, config="a03125c3")
    try:
        corr, obs_py, st = dec.decode_full(np.array([1, 0, 0], dtype=np.uint8))
    finally:
        dec.close()
    assert bits == "".join(str(int(b)) for b in corr)
    assert int(obs) == int(obs_py[0]) and int(status) == st


@requires_gpu
def test_c_program_with_forced_avx2(tmp_path):
    import os

    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    package = Path(eldpc.__file__).resolve().parent
    if not (package / "include" / "eldpc.h").is_file():
        pytest.skip("the installed package does not ship the C header and library")
    src = tmp_path / "probe.c"
    src.write_text(PROGRAM)
    exe = tmp_path / "probe"
    build = subprocess.run([cc, "-std=c11", "-O2", f"-I{package / 'include'}", str(src),
                            f"-L{package / '_libs'}", "-leldpc", f"-Wl,-rpath,{package / '_libs'}",
                            "-o", str(exe)], capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stderr
    env = os.environ.copy()
    env["ELDPC_ISA"] = "avx2"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines()[0] == "avx2"


@requires_gpu
def test_c_program_device_class_override(tmp_path):
    """ELDPC_DEVICE_CLASS reaches a C consumer: a valid value selects, an invalid one fails create."""
    import os

    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    package = Path(eldpc.__file__).resolve().parent
    if not (package / "include" / "eldpc.h").is_file():
        pytest.skip("the installed package does not ship the C header and library")
    src = tmp_path / "probe.c"
    src.write_text(PROGRAM)
    exe = tmp_path / "probe"
    build = subprocess.run([cc, "-std=c11", "-O2", f"-I{package / 'include'}", str(src),
                            f"-L{package / '_libs'}", "-leldpc", f"-Wl,-rpath,{package / '_libs'}",
                            "-o", str(exe)], capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stderr
    env = os.environ.copy()
    env["ELDPC_DEVICE_CLASS"] = "ada64"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines()[1] == "ada64"
    env["ELDPC_DEVICE_CLASS"] = "bogus"
    bad = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300, env=env)
    assert bad.returncode != 0
    assert "ELDPC_DEVICE_CLASS" in bad.stderr and "bogus" in bad.stderr
