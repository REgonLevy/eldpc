# eldpc

Low-latency, exact BP+OSD decoding for quantum LDPC codes on NVIDIA GPUs, with a Python and a C
API.

`eldpc` wraps the `fastbpgpu6g` decoder: minimum-sum belief propagation (parallel schedule) on the
GPU and an exact, bit-packed OSD-0 post-processing on the host, decoding one syndrome per call.
The correction and the observable prediction of every shot are bit-identical to the reference
BP+OSD-0 implementation at the same settings, and the status word is bit-identical to a separately
built library of the same decoder sources.

## What eldpc is and is not

**It is**

- a decoder for binary qLDPC problems, given as a parity-check matrix `H` with an observable matrix
  `O` and one prior probability per column, or as a stim detector error model;
- an implementation of min-sum BP with the parallel schedule, followed by OSD-0 when BP does not
  converge, with the reference's arithmetic order and rounding;
- a `pip`-installable package: a `ctypes` Python package plus `libeldpc.so` with a C API
  (`include/eldpc.h`), pkg-config and CMake package files;
- an optional CPU core (`device="cpu"`) for machines without a GPU: the same decoder on the host,
  several times the GPU core's latency (about 6x on the AVX-512 hosts measured, about 10x on a
  Zen 2 host with AVX2 only), with every output bit-identical to the GPU core's;
- a qLDPC decoder specification (`eldpc.qldpc`) that plugs into qLDPC's estimators, Sinter
  decoders and sliding-window decoders.

**It is not**

- a maximum-likelihood decoder. "Exact" means bit-identical to the reference BP+OSD-0
  implementation, not optimal decoding: BP+OSD-0 is a heuristic, and its logical error rate
  depends on the problem, the presets and the noise;
- a batched GPU decoder: `decode_batch` decodes the shots one after another (see
  [docs/ARCH.md](docs/ARCH.md));
- a substitute for the full `ldpc` package: only minimum-sum BP, the parallel schedule and OSD
  are supported (OSD-0 natively, the exhaustive and combination-sweep methods through the
  vendored `ldpc` code); product-sum BP, serial schedules, `bplsd`, and erasure signalling are
  not;
- a loss-aware decoder: it decodes static priors. It can be given new priors with `set_priors`,
  but no workflow that decodes a loss model detector by detector is included;
- a multi-GPU decoder: one handle uses one CUDA device (device 0 of the visible devices);
- built or tested for Windows or macOS (Linux x86-64 only); see [docs/SUPPORT.md](docs/SUPPORT.md).

### What "exact" means, and how to check it

For every problem, every configuration and every shot, `eldpc` returns

- the same correction and observable prediction as the reference BP+OSD-0 implementation of
  `ldpc` 2.4.1 with the same deterministic OSD column order (ascending soft value, ties by
  ascending column index), and
- the same status word (BP converged / post-processing invoked / syndrome unmatched, and the BP
  iteration count) as a separately built library of the same decoder sources.

Both are checked shot by shot, single-shot and in batch, by the package's own verification suite;
see [docs/VERIFY.md](docs/VERIFY.md). To run it on your machine, from this directory with the
package and its `qldpc` extra installed (`pip install ".[test,qldpc]"`, see "Install" below):

```bash
python verify/run_verify.py --quick
```

The suite builds its own reference decoder from the sources in `verify/reference/`, generates its
corpora from public seeds, and exits 0 if and only if every comparison it made matches. As run
above it compares the corrections and observable predictions with the reference. The status-word
comparison needs a separately built library of the decoder sources as well, which a plain CMake
build of this repository provides; [docs/VERIFY.md](docs/VERIFY.md), "Status words", gives the
commands. The final `PASS` or `FAIL` line names the comparisons that were made. The package tests
(`python -m pytest tests`) cover the API and the C library.

## Install

This release is source only: build and install the package from this source tree with `pip`,
as below. No prebuilt wheels are published for now.

Requirements: Linux x86-64 with an AVX2-capable CPU (the AVX-512 core is used when the CPU and the
OS support it), an NVIDIA GPU of compute capability 8.9 (Ada) or 12.0 (Blackwell) with a driver
that supports CUDA 13.x for `device="gpu"` ([Supported GPUs](#supported-gpus) lists them by name),
a CUDA 13 toolkit to build with, and a C++17 host
compiler. The optional CPU core (`device="cpu"`, [below](#cpu-core)) needs none of the GPU parts at
run time: a machine with no NVIDIA driver can run the decoder on its CPU and the package's CPU-core
tests. The CUDA runtime is linked statically into the GPU decoder libraries, so no CUDA runtime
library or CUDA toolkit is needed at run time: only the NVIDIA driver. A machine without the CUDA
toolkit (and without a GPU) can still build and install the package for the CPU core alone with
`ELDPC_WITHOUT_CUDA=1 pip install .`; that build has no GPU cores (create
decoders with `device="cpu"`, or with `device="auto"`, which falls back to the CPU core), and the
package's CPU-core tests run on it ([docs/BUILD.md](docs/BUILD.md), "CPU-only build"). The
tested toolchains are listed in [docs/BUILD.md](docs/BUILD.md), the machines the package was tested
on in [docs/SUPPORT.md](docs/SUPPORT.md). The steps, in order: [Prerequisites](#prerequisites),
[CUDA 13 toolkit](#cuda-13-toolkit), [Build and install](#build-and-install) and [Check the
installation](#check-the-installation).

### Supported GPUs

The GPU cores carry SASS (compiled GPU code) for compute capabilities 8.9 and 12.0 and PTX for
`compute_120`. By CUDA's compatibility rules, SASS built for compute capability X.Y runs on devices
of the same major version X with a minor version of Y or higher, and PTX is compiled by the driver
for its own compute capability or a higher one, never for a lower one. So:

| GPUs | compute capability | GPU core (`device="gpu"`) |
|---|---|---|
| Ada: GeForce RTX 40-series, RTX 6000 Ada and the other RTX Ada workstation cards, L4, L40, L40S | 8.9 | supported (SASS `sm_89`); tested on an RTX 4090, an RTX 6000 Ada and L40S cards |
| Blackwell for desktops and workstations: GeForce RTX 50-series, the RTX PRO Blackwell cards | 12.0 | supported (SASS `sm_120`); tested on an RTX 5080, an RTX 5090 and an RTX PRO 6000 Blackwell Server Edition |
| Turing and Ampere: GeForce RTX 20- and 30-series, T4, A100, A30, A10, A40, RTX A6000 | 7.5, 8.0, 8.6 | not supported: no image for them (SASS needs the same major version and `compute_120` PTX cannot be compiled for a lower compute capability); the CPU core runs instead |
| Hopper: H100, H200 | 9.0 | not supported, as above |
| Blackwell for data centers: B200, B300 | 10.0, 10.3 | not supported, as above |

A device of another 12.x compute capability would run the `sm_120` SASS, and one of a higher major
version the `compute_120` PTX; no such device is in the test records. Check a GPU before installing
with `nvidia-smi --query-gpu=name,compute_cap --format=csv`. On a GPU that is not supported (or with
no GPU), the install still succeeds and the package runs on the CPU core: `device="cpu"` selects it,
`device="auto"` falls back to it with a one-time warning, `eldpc.devices()` reports
`supported: False` with the reason, and `eldpc.check()` ends with `CPU only: ...`. The tests then
skip the GPU tests, and `verify/run_verify.py` and `bench/bench.py` take `--device cpu`; the tuner
needs a supported GPU for now. Adding an architecture is a build change that must pass the
verification suite again ([docs/SUPPORT.md](docs/SUPPORT.md), "GPU architectures").

### Prerequisites

On Ubuntu or Debian, install the system packages first (of these, the fresh Ubuntu 24.04 machine
of the fresh-clone install test had only `git`):

```bash
sudo apt-get install -y python3-venv build-essential pkg-config cmake git
```

`python3-venv` provides `python3 -m venv`, `build-essential` the C and C++ compilers of the build,
and `git` the `qldpc` extra (it installs qLDPC from git). `pkg-config` is needed only by the C
quick start, and `cmake` only by `verify/run_verify.py` and the tests that build the reference
decoder (without it they fail with an error that names CMake): `pip install .` gets CMake and
ninja from PyPI for its own build, and `pip install cmake` inside the virtual environment works
instead of the system package. Other distributions need the same tools under their own package
names.

### CUDA 13 toolkit

The GPU cores are compiled with the `nvcc` of a CUDA 13 toolkit; the toolkit is needed only to
build them (at run time only the NVIDIA driver is used). Install the toolkit alone, without a
driver. On Ubuntu 24.04 x86-64, from NVIDIA's network repository:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3     # the toolkit only: no driver
```

- `cuda-toolkit-13-3` is CUDA 13.3, the tested toolkit ([docs/BUILD.md](docs/BUILD.md),
  "Requirements"). NVIDIA's download page, https://developer.nvidia.com/cuda-downloads, offers the
  newest release (13.4 at the time of writing) and has the commands for other distributions:
  `cuda-toolkit-13-4` also built the package and passed `verify --quick` in a fresh-clone install test on an RTX
  5090 ([docs/SUPPORT.md](docs/SUPPORT.md), "Tested configurations"); 13.3 has the full set of
  test records, 13.2 and 13.4 partial ones.
- Install a `cuda-toolkit-13-x` package, not the `cuda` or `cuda-13-x` metapackages: those also
  install an NVIDIA driver, which can replace the one the machine has.
- The GPU needs an NVIDIA driver that supports CUDA 13.x, release 580 or later (`nvidia-smi`
  prints the release; [docs/SUPPORT.md](docs/SUPPORT.md), "CUDA toolkit, runtime and driver").
- NVIDIA's packages install the toolkit under `/usr/local/cuda-13.3` and do not put `nvcc` on
  `PATH`. Before `pip install`, name its `nvcc` in `CUDACXX` (or put its `bin` directory on
  `PATH`) and check that it runs:

```bash
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc    # or: export PATH=/usr/local/cuda-13.3/bin:$PATH
$CUDACXX --version                               # "Cuda compilation tools, release 13.3, V13.3.73"
```

Installing a second toolkit repoints `/usr/local/cuda` to it; naming the toolkit in `CUDACXX`
keeps the build on the one you chose.

### Build and install

Clone the repository (the package is not on PyPI; it builds from source), then, in a virtual
environment and with `CUDACXX` set as above:

```bash
git clone https://github.com/REgonLevy/eldpc.git && cd eldpc
python3 -m venv .venv && . .venv/bin/activate
pip install ".[test,qldpc]"                # the package plus what the tests, the tools and eldpc.qldpc need
python -c "import eldpc; eldpc.check()"    # what was built; the last line says "GPU ready" or "CPU only: <why>"
```

or `pip install .` for the package alone (with numpy, scipy and stim); "Python packages for the
tests, the tools and qLDPC" below says what each part needs. `pip install .` builds the seven
native libraries with CMake -- the C API front `libeldpc.so`, four GPU decoder cores (AVX2 and
AVX-512 host code, each in a default and an `ada64` build) and the two CPU cores -- and
installs them inside the Python package; [docs/BUILD.md](docs/BUILD.md) covers the build options, a
plain CMake build, and how the C header, pkg-config file and CMake package config are installed.
It builds with the `nvcc` that `CUDACXX` names, or the one on `PATH`; check the `cuda=` field of
`eldpc.build_info()` (or the "GPU cores" line of `eldpc.check()`) for the toolkit it used.

If `pip install .` finds no CUDA compiler (no `nvcc` on `PATH` and `CUDACXX` unset), it builds
the CPU-only package instead of failing, and pip's default output does not show that:
`pip install -v .` shows CMake's configuration messages, among them `eldpc: GPU and CPU cores
(auto choice; nvcc ...)` or `eldpc: CPU-only build ...` with a warning that says what is missing.
On such a build `import eldpc` warns once when the CUDA driver reports a GPU (`eldpc: this build
has no GPU cores because nvcc was not found at build time, but the CUDA driver reports a GPU ...`),
`eldpc.check()` ends with `CPU only: ...`, and `device="gpu"` fails with a message that says the
build has no GPU cores; set `CUDACXX` and run `pip install` again. `ELDPC_WITHOUT_CUDA=0 pip
install .` makes a missing CUDA compiler a build error; `ELDPC_WITHOUT_CUDA=1` builds CPU-only on
purpose, with no warning at import.

On an offline machine, or in a prepared environment where the build dependencies (CMake, ninja,
scikit-build-core, a C++17 compiler, `nvcc`) and the runtime packages (numpy, scipy, stim) are
already installed, install from the source tree without an index and without build isolation:

```bash
python -m pip install --no-build-isolation --no-deps .
```

The install builds the native libraries with CMake and `nvcc`. How long it takes depends on the
machine, its load and whether pip's cache already holds the dependencies (with an empty cache pip
first downloads numpy, scipy, stim, qLDPC and their dependencies, which adds the download time):
45 s with the `qldpc` extra into a fresh virtual environment (its packages
from pip's cache) on the RTX 5090 and Ryzen 7 9700X desktop of one fresh-clone install test, 3 min 26 s with
the extra on the shared 48-core EPYC 7K62 host of another (RTX 3090, CUDA 13.0), and about 2
minutes (1 minute 10 seconds without the extra) on the shared L40S and dual Xeon Silver machine of
the fresh-clone test; `tools/results/matrix/tested-configurations.json` records the wall-clock
times of all three ([docs/SUPPORT.md](docs/SUPPORT.md), "Tested configurations"). The other times
in this README come from the same runs and are approximate.

### Check the installation

```bash
python -c "import eldpc; eldpc.check()"
```

prints what was built and what the process sees, runs a test decode on each usable core, and ends
with one line: `GPU ready: ...`, or `CPU only: <why>`. When no core can decode at all -- the library
or its decoder core cannot be loaded (for example under an `ELDPC_ISA` value this CPU cannot use,
or a missing core library), or the CPU core fails its test decode -- the last line is `NOT READY:
<why>` instead, and `check()` then raises `eldpc.EldpcError` with that line, so the command exits
non-zero. In the one-line command above Python then prints the exception's traceback after the
report, ending with the same `NOT READY` reason; that is the raise, not a second failure (in a
script, catch `eldpc.EldpcError` to keep the report without the traceback). On an RTX 5080 development machine with a
CUDA 13.3 build it printed (paths shortened):

```
eldpc 0.1.0 (.../site-packages/eldpc)
library:     .../site-packages/eldpc/_libs/libeldpc.so
host ISA:    avx512 (the avx512 build of each core is used; ELDPC_ISA=avx2 or avx512 forces one)
GPU cores:   yes, built with the CUDA 13.3.73 toolkit (build_info: cuda=NVIDIA 13.3.73)
             SASS for sm_89, sm_120 and PTX for compute_120 (compute capability 8.9; 12.0 or newer)
             libeldpc_core_avx2.so, libeldpc_core_avx2_ada64.so, libeldpc_core_avx512.so, libeldpc_core_avx512_ada64.so
CPU cores:   libeldpc_cpu_avx2.so, libeldpc_cpu_avx512.so
devices:     1 CUDA device (eldpc uses device 0)
  0: NVIDIA GeForce RTX 5080, compute capability 12.0, 84 SMs, NVIDIA driver 595.91.07 (CUDA 13.2), core variant default; supported by this build: yes
test decode: GPU core ok; CPU core ok
GPU ready: device="gpu" (the default) decodes on device 0 (NVIDIA GeForce RTX 5080); device="cpu" runs the CPU core.
```

On a build without GPU cores the "GPU cores" line says why and the last line says what to do; the
same machine with a build that found no `nvcc` printed (the first three lines as above):

```
GPU cores:   no: this build has no GPU cores because nvcc was not found at build time
CPU cores:   libeldpc_cpu_avx2.so, libeldpc_cpu_avx512.so
devices:     1 CUDA device (eldpc uses device 0)
  0: NVIDIA GeForce RTX 5080, compute capability 12.0, 84 SMs, NVIDIA driver 595.91.07 (CUDA 13.2), core variant default; supported by this build: NO
test decode: CPU core ok
CPU only: this build has no GPU cores because nvcc was not found at build time. device="gpu" fails; device="cpu" and device="auto" run on the CPU core. To build the GPU cores, see README.md#cuda-13-toolkit: install the CUDA 13 toolkit, put its nvcc on PATH or set CUDACXX, then pip install . again.
```

The CPU-only package still decodes: `device="cpu"` runs the CPU core, and `device="auto"` (opt-in;
the default stays `"gpu"`) runs on the GPU when it is usable and on the CPU core otherwise, with a
one-time warning that says why ([CPU core](#cpu-core)). With no device visible
(`CUDA_VISIBLE_DEVICES` empty, or no NVIDIA driver), a GPU build's last line says so (`CPU only:
no usable CUDA device: ...`) and a CPU-only build's adds it to the build's reason; neither warns at
import. On a GPU the build does not support, the device's line ends with `supported by this build:
NO` and the last line names its compute capability and the build's images (`CPU only: device 0
(NVIDIA GeForce RTX 3090) has compute capability 8.6, which the GPU cores of this build do not
support: ...`); that does not warn at import either.

The same information is available from Python:

```python
import eldpc
print(eldpc.__version__, eldpc.isa())
print(eldpc.build_info())
print(eldpc.devices())
```

`eldpc.isa()` reports the core that was loaded (`avx512` or `avx2`); `eldpc.build_info()` the
compiler, the CUDA toolkit the libraries were built (and their runtime linked) with, the GPU
architectures and the host ISA flags: a GPU build's contains `lib=fastbpgpu6g; cuda=NVIDIA 13.3.73;
cuda_arch=sm_89,sm_120,compute_120`, a CPU-only build's `lib=fastbpcpu6g` and no `cuda=` field.
When no core could be loaded, both raise `eldpc.EldpcError` with the library's reason (an
`ELDPC_ISA` value other than `avx2` and `avx512`, or one this CPU and OS do not support: the
message names the variable and what the CPU supports). `build_info()` describes the process's
default core, the default build of the chosen ISA, so it reports `gpu_sm_budget=0` also on a
142-SM Ada part, whose handles run the `ada64` build (`gpu_sm_budget=64`): `Decoder.core_variant`
reports the build a handle runs.
`eldpc.devices()` enumerates the CUDA devices through the CUDA driver API and returns one dict per
device with these keys:

| key | value |
|---|---|
| `name` | the device name |
| `uuid` | the device UUID (the `GPU-<uuid>` value of `nvidia-smi`, without the prefix), or `None` |
| `compute_capability` | `"major.minor"`, for example `"8.9"` |
| `sm_count` | the number of streaming multiprocessors |
| `l2_bytes` | the size of the L2 cache in bytes |
| `cuda_driver_version` | the newest CUDA version the installed driver supports (for example `"13.2"`), from the CUDA driver API |
| `driver_version` | the same value as `cuda_driver_version`, kept for compatibility; it is not the NVIDIA driver release |
| `nvidia_driver_version` | the NVIDIA driver release, read through NVML (for example `"595.84"`), or `None` |
| `nvidia_driver_version_reason` | `None` when the release was read, otherwise why `nvidia_driver_version` is `None` (NVML not installed or a failed NVML call) |
| `core_variant` | the core a handle on that GPU gets (`"default"` or `"ada64"`) |
| `supported` | whether this build's GPU cores can run on the device (`True`, `False`, or `None` when the build is unknown): the rule of `device="auto"` ([Supported GPUs](#supported-gpus)), read from the build record next to the libraries without loading them |
| `supported_reason` | why `supported` is not `True` (for example `device 0 (NVIDIA GeForce RTX 3090) has compute capability 8.6, which the GPU cores of this build do not support: ...`), else `None` |

An empty list means that no CUDA driver or no device is available
([docs/BUILD.md](docs/BUILD.md), "Device enumeration and reported versions", has the details). Each host
ISA ships two builds of the same decoder sources: the default core, and the `ada64` core for
142-SM Ada parts (compute capability 8.9 with at least 128 SMs), where a 64-SM launch plan is
faster at the default preset and bit-identical; `ELDPC_DEVICE_CLASS=default|ada64` forces the
choice and `Decoder.core_variant` reports a handle's (see [docs/BUILD.md](docs/BUILD.md)).
**Long iteration caps on Ada:** the 64-SM plan saves launch and observation time but halves the
blocks that run each iteration, so with many iterations it can lose. At `a03125c3` the `ada64`
core is faster on every measured problem whose launch it changes (bb144 xyz p50 x0.74 on both an
RTX 6000 Ada and an RTX 4090; also bb144 z and bb288 z on both, hgp98 xyz and sc9 z on the RTX 6000
Ada) and no different on bb72 z and hgp98 z, whose launch is the same in both builds. On bb144 xyz
it is slower at `a05c11` (x1.09 on the RTX 6000 Ada, x1.05 on the RTX 4090)
and at `ms30` (x1.37, x1.28), while bb144 z, bb288 z and hgp98 xyz stay level or faster
([tools/results/ada-sm-budget/](tools/results/ada-sm-budget/)). For long-cap presets such as
`ms30` on large joint problems like bb144 xyz, and for any comparison against a stock-equivalent
setting, run with `ELDPC_DEVICE_CLASS=default`, or time both variants (one environment variable;
on such a GPU the tuner times its chosen configuration under both and recommends the faster,
[tuning/TUNER.md](tuning/TUNER.md) section 5). The support matrix is in [docs/SUPPORT.md](docs/SUPPORT.md).

### Python packages for the tests, the tools and qLDPC

`pip install .` is all that the decoder, its Python and C APIs and stim detector error models need.
The tests, the tools in this source tree and the qLDPC integration need more; two optional extras
install it, `test` (pytest) and `qldpc` (qLDPC and `ldpc` 2.4.1):

| to run | install |
|---|---|
| `import eldpc`, the Python and C quick starts below | `pip install .` |
| `eldpc.qldpc`, the qLDPC quick starts, Sinter ([docs/QLDPC.md](docs/QLDPC.md)) | `pip install ".[qldpc]"` |
| `python -m pytest tests` | `pip install ".[test,qldpc]"`; with the `test` extra alone the qLDPC and corpus tests skip and say why |
| `python verify/run_verify.py` | `pip install ".[qldpc]"`, plus CMake >= 3.16 and a C++17 compiler |
| `bench/bench.py` and `tuning/eldpc_tune.py` on the built-in BB and HGP problems | `pip install ".[qldpc]"` (a DEM file needs nothing more) |

The `qldpc` extra installs qLDPC from git main at commit `4c4f801` (version 0.4.2.dev20261004), the
revision `eldpc.qldpc` and the verification corpora are tested with, so it needs `git`. The
released qLDPC 0.4.0 and 0.4.1 on PyPI lack the decoder-specification API that `eldpc.qldpc` builds
on (checked on their PyPI wheels; [docs/QLDPC.md](docs/QLDPC.md) names the missing parts); with
them, or with no qLDPC, `import eldpc.qldpc` fails with an error that names this
requirement. Every `pip install` of the source tree rebuilds the native libraries; to add the same
packages to an existing installation without a rebuild:

```bash
pip install "pytest>=8" "ldpc==2.4.1" \
    "qldpc @ git+https://github.com/qLDPCOrg/qLDPC@4c4f80109215fa4a28b4d94e2c285e4683487bc1"
```

## Quick start: Python from a stim detector error model

```python
import eldpc
import stim

dem = stim.DetectorErrorModel("""
    error(0.01) D0
    error(0.01) D1
    error(0.01) D0 D1 L0
""")

with eldpc.Decoder.from_dem(dem, config="a03125c3") as dec:
    correction, observables, status = dec.decode_full([1, 0])
    print(correction)                                  # uint8, one bit per error mechanism
    print(observables)                                 # uint8, one bit per observable
    print(eldpc.status.converged(status), eldpc.status.iterations(status))

    dec.set_priors([0.02, 0.02, 0.01])                 # optional: replace the priors
    more = dec.decode_batch([[1, 0], [0, 1]])          # (corrections, observables, statuses)
```

The three `print` lines show `[1 0 0]` (the correction), `[0]` (the observable prediction) and
`False 3` (whether the decode converged, and the BP iteration count).

`Decoder.from_dem` converts the flattened detector error model with the documented rule: one
column per `error` instruction, in order, never merged; a column's `H` and `O` entries are the
parity of the detectors and observables it flips; the prior is the instruction's probability.
`eldpc.dem_to_arrays(dem)` returns the same `(H, O, priors)` if you want to inspect or reuse them.

## Quick start: Python from H, O and priors

```python
import numpy as np
import eldpc

H = np.array([[1, 1, 0],
              [0, 1, 1]], dtype=np.uint8)      # m x n over GF(2)
O = np.array([[1, 0, 1]], dtype=np.uint8)      # k x n over GF(2)
priors = [0.01, 0.01, 0.01]                    # one probability per column, in (0, 1)

with eldpc.Decoder(H, O, priors, config="ms30") as dec:
    correction, observables, status = dec.decode_full([1, 0])
    print(correction, observables, status)     # [1 0 0] [1] 131073
```

The status word packs flags in its low 16 bits and the BP iteration count above them
(`eldpc.status.iterations(status)`). `O` is optional (`k = 0` when omitted); `error_rate=p` sets the same prior on every column.

## Quick start: C

The C API is declared in [`include/eldpc.h`](include/eldpc.h) and follows decoder ABI v1: `H` and
`O` are GF(2) matrices in compressed sparse column form, syndromes and outputs are one `uint8_t`
per bit, and the status word packs flags in the low 16 bits and the BP iteration count above
them.

```c
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

    char err[512];
    eldpc_decoder *dec = eldpc_create(
        "bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;"
        "osd_method=osd0;osd_order=0;schedule=parallel", &p, err, sizeof err);
    if (dec == NULL) {
        fprintf(stderr, "create: %s\n", err);
        return 1;
    }

    const uint8_t syndrome[3] = {1, 0, 0};
    uint8_t correction[5], observables[1];
    uint32_t status = 0;
    if (eldpc_decode(dec, syndrome, correction, observables, &status) != ELDPC_OK) return 1;

    printf("correction %u%u%u%u%u, observable %u, %u iterations\n",
           correction[0], correction[1], correction[2], correction[3], correction[4],
           observables[0], (unsigned)(status >> ELDPC_ST_ITER_SHIFT));
    eldpc_destroy(dec);
    return 0;
}
```

The same program is in [docs/tools/example.c](docs/tools/example.c). Build it against an
installed package with pkg-config, from the repository root (`libeldpc.so` looks for its cores in
its own directory, so no extra `rpath` entry is needed for them):

```bash
export PKG_CONFIG_PATH=$(python -c "import eldpc, os; print(os.path.join(os.path.dirname(eldpc.__file__), 'lib', 'pkgconfig'))")
cc docs/tools/example.c $(pkg-config --cflags --libs eldpc) -Wl,-rpath,$(pkg-config --variable=libdir eldpc) -o example
./example      # prints: correction 10000, observable 0, 3 iterations
```

A CMake build can use `find_package(eldpc REQUIRED)` with `CMAKE_PREFIX_PATH` set to the
`eldpc` package directory and link `eldpc::eldpc`; both files are installed with the package.
`eldpc_create` copies the problem; the caller may free its arrays afterwards. Distinct handles may
be used from different threads; one handle must not be used by two threads at once.

## Quick start: from a code to a circuit, a DEM and a decode

For a circuit-level memory experiment, build the circuit (with qLDPC for a qLDPC code, or with
stim), derive its detector error model with stim, and decode the sampled shots with
`Decoder.from_dem`. With the `qldpc` extra installed:

```python
import numpy as np
from qldpc import circuits, codes
from qldpc.objects import Pauli

import eldpc

code = codes.GrossCode()                         # the [[144, 12, 12]] bivariate bicycle code
circuit = circuits.get_memory_experiment(        # Z-basis memory experiment, 6 rounds, p = 3e-3
    code, basis=Pauli.Z, num_rounds=6, noise_model=circuits.DepolarizingNoiseModel(3e-3))
dem = circuit.detector_error_model(approximate_disjoint_errors=True)
with open("model.dem", "w") as f:                # the DEM file bench.py and the tuner take
    f.write(str(dem))

dets, obs = circuit.compile_detector_sampler(seed=1).sample(10000, separate_observables=True)
with eldpc.Decoder.from_dem(dem, config="a03125c3") as dec:
    print(dec)                                   # m detectors, n mechanisms, k observables
    _, predicted, _ = dec.decode_batch(dets)
print("logical error rate", np.any(predicted != obs, axis=1).mean())
```

With qLDPC git main commit `4c4f801` and stim 1.16.0 this prints a decoder with `m=504 n=5184
k=12` and a logical error rate of 0.0043 (43 of the 10,000 seeded shots); other qLDPC or stim
versions can build a different circuit or sample different shots. It runs in a few seconds.

`approximate_disjoint_errors=True` is the option the verification suite and the benchmark derive
their bivariate-bicycle and hypergraph-product DEMs with; the benchmark's `surface` built-in
additionally decomposes the errors (`decompose_errors=True`), so its DEM is a different model from
the undecomposed one.  The tuner's built-in recipes and `--circuit` pass `decompose_errors=False,
approximate_disjoint_errors=True` by default -- the undecomposed model BP+OSD decodes directly --
and `--decompose` switches them to `decompose_errors=True` with
`ignore_decomposition_failures=True` (tuning/TUNER.md, section 1). Write the DEM once and give the
same file to every step (`bench/bench.py --problem model.dem`, `tuning/eldpc_tune.py --dem
model.dem`, the deployment), so that they all decode the same problem. Without qLDPC, a stim
circuit gives a DEM file the same way, for example a distance-5 surface code (undecomposed, the
decoder's default), written to `surface.dem` so that it does not replace the `model.dem` above:

```bash
python -c "import stim; c = stim.Circuit.generated('surface_code:rotated_memory_z', distance=5, rounds=5, after_clifford_depolarization=1e-3); open('surface.dem', 'w').write(str(c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)))"
```

## Quick start: qLDPC

`eldpc.qldpc` exposes the decoder as a qLDPC decoder specification, so it can be passed wherever
qLDPC accepts `decoder=` (code-capacity estimators, `decoders.SinterDecoder`, the sliding-window
decoders). It needs the `qldpc` extra (qLDPC git main commit `4c4f801` or newer; see "Install").

```python
from eldpc.qldpc import eldpc
from qldpc import codes, decoders

spec = eldpc(config="a03125c3")
code = codes.GrossCode()                       # the [[144, 12, 12]] bivariate bicycle code
estimate = code.get_logical_error_rate_func(2000, decoder=spec)
logical_error_rate, stderr = estimate(0.05)    # code capacity: each qubit has a Pauli error w.p. 0.05
print(logical_error_rate, stderr)              # about 0.023 (typically within +-0.008) and 0.004; changes from run to run
print(*estimate(1e-3))                         # below about 1e-10, and about 3.5e-4

sinter_decoder = decoders.SinterDecoder(decoder=spec)
```

The estimator is qLDPC's code-capacity estimate: it samples 2,000 errors of fixed weights when it
is built (unseeded, so the numbers change from run to run) and reweights them for each physical
rate. At 0.05 it returns a logical error rate of about 0.023 with a standard error of about 0.004,
and successive runs spread by about as much, so expect a value typically within +-0.008 of 0.023:
29 runs of this block on three machines (12 on an RTX 6000 Ada host, 9 on an L40S host, 8 on an
RTX 5080 host) gave 0.018 to 0.031, with standard errors 0.0031 to 0.0044. These are samples of a
random estimate, not bounds; a run outside that range is not a fault.
At a rate where 2,000 samples see no failure the value means "too small to resolve": `estimate(1e-3)`
returns a value below about 1e-10 (from below 1e-17 to 9e-11 in the same 29 runs) with a standard error of
about 3.5e-4, the Jeffreys-posterior uncertainty of the sampled weights, so read it as an upper
bound of a few times 1e-4 (see `help(qldpc.codes.ErrorRateFunc)`).

The specification also builds decoders directly from a parity-check matrix
(`eldpc(error_channel=0.01).build(H)`) or a detector error model (`eldpc().build(dem)`), and
predicts observable flips natively for detector error models. The full option list, the accepted
inputs and the relation to qLDPC's `ldpc` integration are in [docs/QLDPC.md](docs/QLDPC.md).

## CPU core

eldpc also ships an optional CPU core: the same exact min-sum BP+OSD-0 decoder, run on the host,
for machines without a usable GPU (CI runners, laptops, a workstation whose GPU is busy or absent).
It is a fallback, not a replacement: in a same-session screen on a shared development machine
with an AVX-512 host (below) it ran at 223.5 us p50 per shot against 36.1 us for the GPU core,
6.19x the GPU core's latency at the p50 (6.05x at the p99). The ratio depends on the host: on a
dual AMD EPYC 7542 (Zen 2, AVX2 only) with an RTX 6000 Ada, a screen of the C API on bb144 xyz at
`a03125c3`, with the calling thread pinned to the last `cpus` entry, measured 608.6 / 796.9 us
p50 / p99 at `threads=4` on the four cores of one L3 against 63.9 / 84.8 us for the GPU core
(9.5x / 9.4x), and 1165.2 / 1595.6 us at `threads=8` across two L3s (18.2x / 18.8x;
`tools/results/cpu/zen2-avx2-screen/`). Every output is bit-identical to the
GPU core's and to the reference at the same settings.

```python
import numpy as np
import stim

import eldpc

circuit = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5,
                                 after_clifford_depolarization=0.01)
dets, obs = circuit.compile_detector_sampler(seed=7).sample(1000, separate_observables=True)
with eldpc.Decoder.from_dem(circuit.detector_error_model(), config="a03125c3",
                            device="cpu", threads=4) as dec:
    print(dec.device, dec.core_variant)          # cpu cpu
    _, predicted, _ = dec.decode_batch(dets)
print("logical error rate", np.any(predicted != obs, axis=1).mean())
```

With stim 1.16.0 it prints `cpu cpu` and `logical error rate 0.0` (no logical error in the 1,000
seeded shots of this small repetition code; `tests/test_readme_quick_starts.py` checks both lines).

`eldpc.Decoder(H, O, priors, device="cpu", threads=4)` does the same for H, O and priors, and
`eldpc.qldpc`'s specification takes `device=` and `threads=` too. `threads` (1..64, default 1) is
the number of threads that compute one shot; the calling thread is one of them. A CPU decoder
starts `threads - 1` threads at construction and they busy-wait (no sleep, no yield) for the next
shot until `close()`, so give each CPU decoder its own cores: two CPU decoders with `threads=8`
need 16 cores between them while both are open. The library does not pin the calling thread; pin
it yourself if it matters.

Placement matters: keep the threads and the calling thread on one CCD (one L3 cache) or one
socket. Across the two CCDs of an AMD Ryzen 9 7950X3D, `threads=16` was 2.41x the p50 latency of
`threads=8` on one CCD (2.20x at `threads=10` and 2.11x at `threads=12`, which also span both
CCDs). The installed wheel measured the same shape again (`tools/results/cpu/placement.md`,
written by `tools/cpu_placement.py`; 1000 shots per placement, 4 rounds, A/A control 1.005 at the
p50): `threads=16` across both CCDs 2.334x the p50 of `threads=8` on one CCD (2.008x at the p99);
`threads=16` on the one CCD, using its SMT second contexts, 1.017x (no gain); `threads=8` on the
SMT second contexts 1.006x; `threads=8` on the other CCD 0.929x in that run. The advice is about
not spanning CCDs or sockets, not about SMT: use the physical cores of one CCD. Pin through the
config key `cpus`: exactly `threads` CPU ids or ranges. The started threads are pinned to its
first `threads - 1` entries, and the last entry is the calling thread's CPU: the library does not
pin the calling thread, so pin it to that CPU yourself (`os.sched_setaffinity`, `taskset`), for
example `cpus=1,2,3,4,5,6,7,0` with `threads=8` and the caller on CPU 0. The default list is the
creating thread's affinity mask in ascending order, repeated to `threads` entries. A list whose
first `threads - 1` entries cover every CPU the creating thread may run on (the caller pinned to
CPU 0 with `cpus=0,1,...,7`, or a caller pinned to one CPU without `cpus`) puts a busy-waiting
worker on the caller's CPU and makes every decode many times slower; the Python package warns
about that (`RuntimeWarning`) when it creates such a decoder.

In Python, give `cpus` with the preset in the config string, or in a dict that starts from the
preset's keys:

```python
import os

import stim

import eldpc

dem = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5,
                             after_clifford_depolarization=0.01).detector_error_model()
os.sched_setaffinity(0, {4})        # the calling thread on CPU 4, the last cpus entry
with eldpc.Decoder.from_dem(dem, device="cpu", threads=4, config="a03125c3;cpus=1,2,3,4") as dec:
    print(dec.device)               # cpu: the started threads run on CPUs 1, 2 and 3
config = {**eldpc.presets()["a03125c3"], "cpus": "1,2,3,4"}   # the same configuration as a dict
with eldpc.Decoder.from_dem(dem, device="cpu", threads=4, config=config) as dec:
    print(dec.device)
```

A preset name with `;key=value` segments is the preset with those keys added (or replaced). A dict
is a whole configuration, not additions to a preset: `config={"cpus": "1,2,3,4"}` alone has no
decoder keys and fails with `missing required config key 'decoder'`, so start the dict from
`eldpc.presets()[name]` as above.

In C, the config string of `eldpc_create` takes the same keys:
`device=cpu;threads=8;cpus=1,2,3,4,5,6,7,0` alongside the decoder keys (the front removes
`device` before the core parses the string). `eldpc_core_variant()` returns `"cpu"` for such a
handle, `eldpc_keep_warm()` returns `ELDPC_OK` and does nothing, and the create makes no CUDA call
at all, so it works with `CUDA_VISIBLE_DEVICES` set empty. A `device=gpu` create that finds no
usable CUDA device fails with a message that says why and names `device="cpu"`.

`device="auto"` (Python; `device=auto` in a C config string or a config file) runs on the GPU
when device 0 is usable -- a driver and a visible device, a driver new enough for the GPU cores'
CUDA version, a compute capability the cores support, and a build with GPU cores -- and on the
CPU core otherwise. In Python the fallback raises a `RuntimeWarning` once per process that names
the reason, `threads` applies only to the CPU core, and `Decoder.device` / `Decoder.core_variant`
report the choice; in C, `eldpc_core_variant()` reports it ([docs/BUILD.md](docs/BUILD.md)).

The numbers above are a *screen* on a shared development machine (AMD Ryzen 9 7950X3D, GeForce
RTX 5080, driver 595.91.07, other processes running), not a benchmark of a quiet machine. The
records are `tools/results/cpu/gpu_same_machine.json` (the same-session CPU / GPU comparison of
the CPU core's sources as first built) and `tools/results/cpu/final_ab.json` (the placement A/B);
`tools/cpu_summary.py` prints the values quoted from them. The installed wheel's own screen,
`tools/results/cpu/screen.md` (written by `tools/cpu_screen.py`), gave 230.4 us at `threads=8`
against 42.0 us for the GPU core in its session (5.48x at the p50), and the packaged CPU core
within 1% of the same sources compiled again with the same flags. The CPU core's exactness record
(identical corrections, observables, status words and soft output against the reference on every
measured cell) is `tools/results/cpu/exactness*.json`, and the package's own tests check the same
identity without a GPU (`CUDA_VISIBLE_DEVICES= python -m pytest -m cpu_core tests`).

## Presets and choosing one

A preset is a named `(min-sum scaling factor, BP iteration cap)` setting of BP+OSD-0 with the
parallel schedule; `osd_method=osd0` and `osd_order=0` in every one.

| preset | `ms_scaling_factor` | `max_iter` | when to start with it |
|---|---:|---:|---|
| `a03125c3` | 0.3125 | 3 | the default: the fastest acceptable preset on the circuit-level bivariate-bicycle problem it was tuned on, at noise up to 2e-3 |
| `a04375c3` | 0.4375 | 3 | the tuned setting for a Z-basis bivariate-bicycle problem at 1e-3 |
| `a0375c7` | 0.375 | 7 | more accuracy margin at higher noise (2e-3 to 3e-3) |
| `a05c11` | 0.5 | 11 | within noise of the 31-iteration reference at 1e-3 and 2e-3 |
| `a05c15` | 0.5 | 15 | higher-noise Z-basis problems |
| `a05ms31` | 0.5 | 31 | the high-accuracy reference point of the tuning records |
| `ms30` | 0.625 | 30 | `ldpc`'s common setting; bit-identical to it, so its LER is stock's |
| `ms30a075` | 0.75 | 30 | the stock family with a larger scaling factor |
| `ms100` | 0.625 | 100 | the stock family with a large iteration cap |

The table is guidance, not a guarantee: the right point depends on the code, the noise level and
the latency budget. The tuning records behind it, with the machines they were measured on, are
in [tuning/KNOWLEDGE.md](tuning/KNOWLEDGE.md) (sections 2.2 and 5; its sources are in `tuning/data/` and
`tools/results/`). `a03125c3` is the default of
`Decoder(...)` and `Decoder.from_dem(...)`. The `.bp` suffix selects the same BP without OSD
(`a03125c3.bp`): it skips OSD-0 on the shots BP does not solve, so it is faster on those, and it
returns BP's hard decision for them (with `ST_SYNDROME_UNMATCHED` set). It is a different decoder.

```python
eldpc.presets()                                # name -> config dict
eldpc.Decoder(H, O, priors, config="ms30")     # a preset
eldpc.Decoder(H, O, priors, config={"decoder": "bposd", "max_iter": 7, "ms_scaling_factor": 0.375,
                                    "bp_method": "ms", "schedule": "parallel",
                                    "osd_method": "osd0", "osd_order": 0})
eldpc.Decoder(H, O, priors, config="bp_method=ms;decoder=bposd;max_iter=7;ms_scaling_factor=0.375;osd_method=osd0;osd_order=0;schedule=parallel")
eldpc.Decoder(H, O, priors, config="config.json")
eldpc.Decoder(H, O, priors, config="a03125c3;threads=8", device="cpu")   # a preset plus extra keys
```

A config string may start with a preset name and add `key=value` segments after it:
`"a03125c3;threads=8"` is the preset with `threads=8` added, and an extra key the preset already has
replaces its value (`"ms30;max_iter=50"`); a key given twice among the extra segments is an error,
as in any config string. The Python package and the C API expand the name the same way, so a C
caller can pass `"a03125c3"` or `"a03125c3;device=cpu;threads=8"` to `eldpc_create` too, and
`bench/bench.py --config` takes the same strings (docs/BENCH.md).

A C caller passes the same ABI string. The strings of the presets are
(`key=value` segments in sorted order):

```
a03125c3  bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;osd_method=osd0;osd_order=0;schedule=parallel
a04375c3  bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.4375;osd_method=osd0;osd_order=0;schedule=parallel
a0375c7   bp_method=ms;decoder=bposd;max_iter=7;ms_scaling_factor=0.375;osd_method=osd0;osd_order=0;schedule=parallel
a05c11    bp_method=ms;decoder=bposd;max_iter=11;ms_scaling_factor=0.5;osd_method=osd0;osd_order=0;schedule=parallel
a05c15    bp_method=ms;decoder=bposd;max_iter=15;ms_scaling_factor=0.5;osd_method=osd0;osd_order=0;schedule=parallel
a05ms31   bp_method=ms;decoder=bposd;max_iter=31;ms_scaling_factor=0.5;osd_method=osd0;osd_order=0;schedule=parallel
ms30      bp_method=ms;decoder=bposd;max_iter=30;ms_scaling_factor=0.625;osd_method=osd0;osd_order=0;schedule=parallel
ms30a075  bp_method=ms;decoder=bposd;max_iter=30;ms_scaling_factor=0.75;osd_method=osd0;osd_order=0;schedule=parallel
ms100     bp_method=ms;decoder=bposd;max_iter=100;ms_scaling_factor=0.625;osd_method=osd0;osd_order=0;schedule=parallel
```

A `.bp` variant is the same string with `decoder=bp` and without `osd_method` and `osd_order`.

### How to choose one for your problem

1. Start with `a03125c3` and measure the logical error rate on your code at your noise level with
   `bench/bench.py ler` (below). If it is acceptable, measure the latency and stop.
2. If it is not, try `a0375c7` and `a05ms31`; higher caps and larger scaling factors buy accuracy
   at a latency cost, and the frontier depends on the problem.
3. If the budget is tight, try smaller caps and the `.bp` variant; the OSD-0 post-processing is
   the expensive part of a shot that BP does not solve.
4. For a real deployment, run the tuner on a detector error model of the problem; it searches the
   scaling factor and the iteration cap directly against an LER target and a latency budget, and
   writes a config file the package loads. The procedure, the outputs and the decision rules are
   in [tuning/README.md](tuning/README.md) and [tuning/TUNER.md](tuning/TUNER.md); the measured
   effects behind the table above are in [tuning/KNOWLEDGE.md](tuning/KNOWLEDGE.md).

```bash
python tuning/eldpc_tune.py --dem model.dem --ler-target 1e-3 --p99-budget-us 250 --auto --out tune
python -c "import eldpc; print(eldpc.presets())"
```

`model.dem` is your problem's detector error model: "From a code to a circuit, a DEM and a
decode" above writes one, and every `bench/bench.py` run writes its problem's DEM to
`DIR/data/model.dem`. A full tuner run measures enough shots to certify the target: about 3 to 4
minutes for the first command above on the model that the code-to-DEM quick start writes (3 min
31 s on the RTX 6000 Ada host below; with a 200 us budget 3 min 2 s there, 2 min 40 s on the L40S
of the fresh-clone test and 3 min 9 s on the RTX 5090 of the fresh-clone install test;
`tuning/results/readme-example-rtx6000ada/`, `tools/results/matrix/tested-configurations.json`).

The point the tuner chooses depends on the machine (its latencies decide which points fit the
budget), on the tuner version and on the problem, so read it from your run rather than from this
README. `config.json` holds the choice, and the `config:` line of
section "Choice" in `pareto.md` names it (the console prints the same); section "Accuracy
confirmation" lists every confirmed candidate with `passed` (its LER bound met the target on the
fresh shots), `accepted` (and its latency confirmation stayed within the budget) and, when it was
not accepted, why. The candidates are the points within the budget (the points near it re-timed
on 8000 calls first), confirmed one at a time until one is accepted, so the screen's leader can
fail that confirmation and the next accepted candidate is then chosen; the tie set listed under
"Choice" is the screen's ranking, before the confirmation, with the accepted choice marked
`<- chosen` and a passed-over screen pick `(screen pick)`. The accuracy confirmation's shots are
seeded, so for the same model and tuner its failure counts are the same on every machine. With the
command above, tuner 2.2 (the one in this tree) confirms (0.5, 11) and (0.5, 15) with 82 and 80
failures in 99,992 shots (upper bounds 1.06e-3 and 1.03e-3, above the 1e-3 target), so neither is
accepted on any machine, and (0.5, 31) passes with 76 failures (upper bound 0.000989); whether
(0.5, 31) is chosen then depends on its latency on the machine. On an RTX 6000 Ada with a dual AMD
EPYC 7542 host (a shared host, governor `schedutil`, so the latencies are screens) the command
above chose (0.5, 31), with a latency-confirmation p50 / p99 of 183.6 / 217.7 us against the
250 us budget, and exited 0 (`tuning/results/readme-example-rtx6000ada/budget-250us/`).
When no measured point meets both the accuracy target and the p99 latency budget, the tuner
reports "No configuration qualifies", names the nearest point with how far it is from the budget
(its p50, p99 and budget ratio from one named measurement), writes `config.nearest.json` instead
of `config.json` and exits with status 3 (0 when it chose a point). Relax the latency budget or
the LER target, or use a faster GPU, and run it again. With a budget of 200 us the command above
ends that way on the same RTX 6000 Ada host: the nearest point, (0.5, 31), re-timed at a p99 of
212.1 us, 1.06x the budget (`tuning/results/readme-example-rtx6000ada/budget-200us/`); an RTX 5090
desktop, with an earlier tuner, chose (0.5, 31) within 200 us (p99 199.3 us;
`tools/results/matrix/tested-configurations.json`). Its `--quick`
mode decodes only 2,048 shots per point and so cannot certify a target below about 2.2e-3: it then
reports that no configuration qualifies and writes `config.nearest.json`. As a smoke test of the
tool (seconds; needs the `qldpc` extra), use a loose target:

```bash
python tuning/eldpc_tune.py --problem bb72 --p 0.001 --rounds 6 \
    --p99-budget-us 1000000 --ler-target 0.05 --auto --quick --out tune-quick
```

## Verify, benchmark and tune on your own machine

The verification suite, the benchmark and the tuner live in the source tree (they are not part of
the installed wheel). Run them from the package root with the package and its extras installed
(`pip install ".[test,qldpc]"`, see "Install").

```bash
python -m pytest tests                   # 20 s to 1 minute
python verify/run_verify.py --quick      # bit identity against the packaged reference; prints PASS/FAIL; 1.5 to 6 minutes
python bench/bench.py all --quick --out bench-results     # 2 to 5 minutes on a quiet host (a loaded host takes longer); writes bench-results/results.md and results.json
python tuning/eldpc_tune.py --dem model.dem --ler-target 1e-3 --p99-budget-us 250 --auto --out tune
```

Without a supported GPU ([Supported GPUs](#supported-gpus)), run the CPU core instead:

```bash
python -m pytest tests                   # the GPU tests skip and say why; the rest runs on the CPU core
CUDA_VISIBLE_DEVICES= python -m pytest -m cpu_core tests   # the CPU-core tests alone
python verify/run_verify.py --quick --device cpu           # the CPU core's outputs against the reference
python bench/bench.py all --quick --device cpu --problem surface --out bench-cpu   # CPU-core numbers
```

The tuner needs a supported GPU for now (it has no CPU-core option). The CPU core takes several
times the GPU core's time per shot ([CPU core](#cpu-core)), so pick a small problem for
`bench.py --device cpu`: on the default
bb144 problem one `--quick` latency session took about 2 minutes on the CPU core of the RTX 3090
fresh-clone install test's EPYC host. There, `CUDA_VISIBLE_DEVICES= python -m pytest -m cpu_core tests` passed 185
tests and skipped 4 (optional comparisons) in 15.5 s, at a revision before `--device` existed.

- The times vary with the machine and its load. On the RTX 5090 and Ryzen 7 9700X desktop of the
  fresh-clone install test (16 threads) the first three took 17 s, 1 min 20 s and 2 min 4 s; on the shared L40S
  with a dual Xeon Silver host of the fresh-clone test 52 s, 3 min 53 s and 4 min 32 s; the tuner
  command took about 3 minutes on both, with a 200 us budget
  (`tools/results/matrix/tested-configurations.json`). On the RTX 6000 Ada with a dual EPYC 7542
  host of `docs/results/` the verification took about 5 minutes with the status words compared
  (`timings.total_seconds` of `docs/results/verify-quick.json`); a first run on a machine also builds the reference and
  generates the corpora, and a loaded host adds to every step.
- Expect skipped tests: `pytest` skips, and says why, the tests that need an optional comparison
  input (`ELDPC_REFERENCE_LIB`, `ELDPC_CORE_TARBALL` and the other variables of
  [docs/BUILD.md](docs/BUILD.md), "Tests": a separately built reference library, the decoder's
  original sources, the reference construction's packages), the qLDPC tests without the `qldpc`
  extra, the GPU tests when no CUDA device is visible, the build has no GPU cores or the GPU is not
  supported, the forced-ISA tests that cannot fail on this CPU, and the tests whose tool is missing
  (a C compiler, `readelf`, `nvidia-smi`). `python -m pytest -rs tests` lists the reasons; the RTX
  5090 fresh-clone install test, with everything of "Prerequisites" installed, had 309 passed and 25 skipped:
  optional comparisons, and one forced-ISA test on its AVX-512 CPU. Without `cmake` on `PATH` the
  tests that build the reference decoder fail instead, with an error that names it.
- Verification needs a supported GPU (or `--device cpu`), qLDPC, CMake and a C++17 compiler (it
  builds the reference). It prints a line for each step from the start (the reference build, each
  corpus it generates or reuses, each row it verifies); the largest rows take a minute or more
  each with no new line, while the reference decoder runs on them; that is not a hang.
  Options, the corpora, the status-word comparison and the meaning of every column are in
  [docs/VERIFY.md](docs/VERIFY.md).
- `bench.py all --quick` measures its default problem (bb144, X and Z decoded jointly) and prints a
  progress line per session. It writes the report `results.md` and every number in `results.json`
  into the `--out` directory (`bench-results/` above; `eldpc-bench-results/` without `--out`);
  `python bench/bench.py summary --out bench-results` prints the headline numbers again. On a
  machine not prepared for timing it ends with a non-empty `warnings` list; that is expected, and
  its numbers are then screens. Without `--cpu N` the run is pinned automatically (`--cpu auto`)
  to the first CPU of the GPU's NUMA node other than CPU 0 and its SMT sibling (CPU 0 takes most
  interrupts and housekeeping work), and a warning names the choice; pass `--cpu N` to choose,
  for example a CPU whose SMT sibling is idle.
- The benchmark measures C single-shot latency, Python single-shot latency, batch throughput and
  the logical error rate, and records the machine, the clocks and the conditions it could check.
  Run it on an idle machine with the process pinned to a CPU. See [docs/BENCH.md](docs/BENCH.md).
- The tuner needs the same package plus the problem (a DEM, a circuit, a matrix or a built-in
  recipe) and writes `config.json`, `pareto.md` and `results.json`. See
  [tuning/README.md](tuning/README.md); the workflow, from collecting the inputs to reporting, is
  [tuning/TUNER.md](tuning/TUNER.md).

## Check the decoder sources

`src/core/SOURCES.sha256` pins the raw bytes of the six decoder sources. They are identical to the
benched sources except that development-history comments were neutralised: code tokens and line
structure are unchanged (`src/core/TOKENS.sha256` and `src/cpu/TOKENS.sha256` pin the text without
comments, checked by `tools/check_comment_only.py`), and the libraries compiled from them are
identical (`tools/results/comment-neutralisation/record.txt`). From the repository root, these
checks need nothing but the source tree:

```bash
(cd src/core && sha256sum -c SOURCES.sha256)
python3 tools/check_comment_only.py --check     # the text without comments, src/core and src/cpu
```

`tools/check_core_tarball.py` makes the comparison of the text without comments against the original
decoder source tarball (`fastbpgpu6g`), which it reads without extracting it. It is optional and only useful if
you have that tarball; the check above does not need it:

```bash
python tools/check_core_tarball.py /path/to/fastbpgpu6g-src.tar.gz    # from the repository root
```

## Results

The verification, benchmark, LER and tuning results recorded for this package, with the machine,
command and commit of every table, are in [docs/RESULTS.md](docs/RESULTS.md). That file is
generated from the records in `docs/results/` by `docs/tools/make_results.py`. Its timings were
taken on a shared development machine and are screens, not benchmark results; measure latency on
your own machine with `bench/bench.py`.

The timing-grade numbers come from the final bench of the shipped core's sources, built for the
measurement harness (not timed through the package's `libeldpc.so` front), on machine A of
[tuning/KNOWLEDGE.md](tuning/KNOWLEDGE.md) (an RTX 5080 with a Ryzen 7 9850X3D host in bench mode):
bb144 xyz at `a03125c3` took 32.12 / 42.37 µs p50 / p99 with the validated overclock (its profile,
validation and caveat: [docs/SUPPORT.md](docs/SUPPORT.md#the-validated-gpu-overclock)) and 34.02 /
44.15 µs at stock clocks, single handle, back to back
(`tools/results/final-bench/final-bench.json`), against 72.8 / 143.0 ms per shot for stock ldpc
2.4.1 at its common settings (`ms30`) through its Python `decode()` on one core of the same host
(`tools/results/stock-baseline/stock-baseline.json`). That comparison changes the settings as well
as the implementation: at the same settings, `a03125c3`, stock took 22.8 ms p50 in the same
record, about 670x the default preset's 34.02 µs. That machine is a development machine, not a
deployment class. The other measurement records behind the tuning knowledge (the SM-budget
sweeps on the Ada boards, the Xeon host study, joint against separate decoding, heralded loss) are
summarised under `tools/results/`, each directory with a README.

## Claims

Every user-facing claim this README makes, with its evidence, its status and what would refute
it, is listed in [docs/CLAIMS.md](docs/CLAIMS.md).

## Support

The built-for and tested support matrix (GPU architectures, host ISAs, CUDA toolkit and driver,
OS, glibc and Python versions), known limitations and how to check them are in
[docs/SUPPORT.md](docs/SUPPORT.md). The architecture of the decoder and of the package is in
[docs/ARCH.md](docs/ARCH.md).

## Documentation

- [docs/BUILD.md](docs/BUILD.md): building, installing, build options, the static CUDA runtime,
  device enumeration, tests.
- [docs/ARCH.md](docs/ARCH.md): the call path, the GPU kernel, the host OSD-0, the ISA choice,
  the CUDA runtime and driver, keep-warm, batch behaviour, memory, and the exactness rules of the
  build.
- [docs/SUPPORT.md](docs/SUPPORT.md): support matrix, limitations, checks.
- [docs/VERIFY.md](docs/VERIFY.md): the public verification suite.
- [docs/BENCH.md](docs/BENCH.md): the benchmark and how to read its numbers.
- [docs/QLDPC.md](docs/QLDPC.md): the qLDPC integration.
- [docs/RESULTS.md](docs/RESULTS.md): recorded verification, benchmark, LER and tuning results.
- [docs/CLAIMS.md](docs/CLAIMS.md): the claims of this README and their evidence.
- [tuning/README.md](tuning/README.md), [tuning/TUNER.md](tuning/TUNER.md),
  [tuning/KNOWLEDGE.md](tuning/KNOWLEDGE.md): the tuner and the measured tuning knowledge.

## Licence

`eldpc` is licensed under the Apache License, Version 2.0 (SPDX `Apache-2.0`); see
[LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

## Credits and third-party software

- [`ldpc`](https://github.com/quantumgizmos/ldpc) (Joschka Roffe): `eldpc` reproduces its BP+OSD-0
  decoding bit-exactly, and its C++ headers (2.4.1, with the tie-order patch in
  `verify/reference/`) are the verification reference. The headers, with `tsl::robin_map`, are
  vendored in `src/third_party/ldpc/` (which the decoder cores build on) and
  `verify/reference/ldpc/`, under the MIT licence; [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)
  holds their licence texts, the upstream commit and the changes.
- [qLDPC](https://github.com/qLDPCOrg/qLDPC): the integration layer `eldpc.qldpc` targets its
  decoder interface ([docs/QLDPC.md](docs/QLDPC.md)).
- [stim](https://github.com/quantumlib/Stim): circuits, detector error models and sampling, in the
  package's DEM input and in the tests, the verification suite, the benchmark and the tuner;
  C. Gidney, Quantum 5, 497 (2021), arXiv:2103.02202.
- The CUDA runtime is linked statically into the GPU decoder cores built from this source; it
  is not covered by the Apache licence ([THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)).

References:

- BP+OSD: P. Panteleev and G. Kalachev, "Degenerate Quantum LDPC Codes With Good Finite Length
  Performance", Quantum 5, 585 (2021), arXiv:1904.02703; J. Roffe, D. R. White, S. Burton,
  E. T. Campbell, "Decoding across the quantum low-density parity-check code landscape",
  Phys. Rev. Research 2, 043423 (2020), arXiv:2005.07016.
- Bivariate-bicycle codes: S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall,
  T. J. Yoder, "High-threshold and low-overhead fault-tolerant quantum memory", Nature 627,
  778-782 (2024), arXiv:2308.07915.
