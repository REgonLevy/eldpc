# Building eldpc

`eldpc` builds seven shared libraries with CMake, driven by scikit-build-core for `pip`:

| library | contents |
|---|---|
| `libeldpc_core_avx2.so` | the fastbpgpu6g decoder core, host code compiled with `-mavx2` |
| `libeldpc_core_avx512.so` | the same sources, host code compiled with `-mavx2 -mavx512f -mavx512vl -mavx512dq -mavx512bw` |
| `libeldpc_core_avx2_ada64.so` | the AVX2 core with `DBENCH_GPU_SM_BUDGET=64` (the Ada 142-SM class) |
| `libeldpc_core_avx512_ada64.so` | the AVX-512 core with `DBENCH_GPU_SM_BUDGET=64` |
| `libeldpc_cpu_avx2.so` | the optional CPU core (fastbpcpu6g), `-mavx2`, no CUDA code and no CUDA library |
| `libeldpc_cpu_avx512.so` | the same CPU sources, with the AVX-512 flag set of the GPU core |
| `libeldpc.so` | the C API front (`include/eldpc.h`): chooses the host ISA at load time, the GPU core variant per handle, and whether a handle runs on the GPU or the CPU |

A CPU-only build (`ELDPC_WITHOUT_CUDA`, or a build that finds no CUDA compiler; below) builds only
`libeldpc.so` and the two CPU cores.

All four cores contain CUDA SASS for `sm_89` (Ada) and `sm_120` (Blackwell) and PTX for
`compute_120`; the GPU driver picks the image that fits the device. See "Core variants" below for
the two builds of each ISA.

## Requirements

Tested with:

- Linux x86-64, glibc 2.39.
- Python 3.12; the build backend needs `scikit-build-core >= 0.11` (the first release that reads
  the PEP 639 licence fields of `pyproject.toml`), `cmake >= 3.21` and `ninja`.
- A C++17 host compiler: GCC 13.3 (tested).
- CUDA Toolkit 13.3 (tested; `nvcc` V13.3.73) to build. The build needs `nvcc`; the runtime
  library comes from the toolkit as a static archive and is linked into the cores, so no CUDA
  runtime library is needed at run time. Put the toolkit's `bin` directory on `PATH` or point
  `CUDACXX` at its `nvcc`. The build uses whichever `nvcc` that finds: on a machine with several
  toolkits, where `/usr/local/cuda` may point to another one, name it explicitly
  (`CUDACXX=/usr/local/cuda-13.3/bin/nvcc pip install .`) and check the `cuda=` field of
  `eldpc.build_info()`. CUDA 13.2 (`nvcc` V13.2.86) also built the package and passed the tests,
  `verify --quick` and `bench.py all --quick` in a fresh-clone install test on an L40S, and CUDA
  13.4 (`nvcc` V13.4.92) built it and passed `verify --quick` in a fresh-clone install test on an RTX 5090
  (docs/SUPPORT.md, "Tested configurations"; `tools/results/matrix/tested-configurations.json`);
  other 13.x toolkits are not tested. On Ubuntu, install the toolkit-only package
  (`cuda-toolkit-13-3` from NVIDIA's apt repository; the `cuda` and `cuda-13-x` metapackages also
  install a driver); NVIDIA's packages do not put `nvcc` on `PATH`. The README, "Prerequisites"
  and "CUDA 13 toolkit", has the commands, and `python -c "import eldpc; eldpc.check()"` says
  after the install whether the GPU cores were built.
- An NVIDIA driver that supports CUDA 13.x, and a GPU of compute capability 8.9 or 12.0.
  CUDA 13.3's own minimum driver is 610.43.02 (CUDA Toolkit 13.3 release notes, "CUDA Toolkit
  and Corresponding Driver Versions"; the toolkit's `version.json` names the same 610.43.02).
  Under CUDA minor-version compatibility a CUDA 13.x application also runs on the drivers of
  earlier 13.x releases, i.e. 580 and later; the package was tested with drivers 580.126.09,
  580.159.03, 595.71.05, 595.84, 595.91.07, 610.43.02 and 615.71.09 (docs/SUPPORT.md, "Tested
  configurations"; `tools/results/matrix/tested-configurations.json`).
- Python packages: numpy, scipy, stim (installed as dependencies). The tests and the tools of the
  source tree need the optional extras `test` (pytest) and `qldpc` (qLDPC at git main commit
  `4c4f801` and `ldpc` 2.4.1); see "Install from a fresh clone".

At run time the installed package needs only the NVIDIA driver: `libcuda.so.1` (and, for the
`nvidia_driver_version` key of `eldpc.devices()`, `libnvidia-ml.so.1` when NVML is installed).
No CUDA toolkit, no `libcudart`, no `LD_LIBRARY_PATH` entry for a CUDA `lib64` directory.

The build does not use `-march=native`, fast math or FMA contraction: the decoder's outputs must
not depend on the host it was compiled on. The host flags are the decoder's reference flags,
`-O2 -DNDEBUG -ffp-contract=off -fno-fast-math`, then `-O3`, then the ISA flags; the CUDA sources
get `-fmad=false -ftz=false -prec-div=true -prec-sqrt=true` last on every `nvcc` command line.

## Install from a fresh clone

```bash
python -m venv .venv && . .venv/bin/activate
pip install .                      # the package, with numpy, scipy and stim; or, instead:
pip install ".[test,qldpc]"        # the package plus pytest, qLDPC and ldpc 2.4.1 (needs git)
```

The extras are for the source-tree tools and the qLDPC integration: `test` installs pytest;
`qldpc` installs qLDPC from git main at commit `4c4f801` (0.4.2.dev20261004), the revision
`eldpc.qldpc` and the verification corpora are tested with, and `ldpc` 2.4.1, the version the
reference and the tests compare with. The released qLDPC 0.4.0 and 0.4.1 lack the
decoder-specification API `eldpc.qldpc` needs (their PyPI wheels have no
`qldpc.decoders.protocols.ErrorDecodeResult`, and 0.4.0 also no `decoder_spec`; check with
`pip download qldpc==0.4.1 --no-deps` and a search of the wheel). Each `pip install` of the source
tree rebuilds the libraries; to add the extras' packages to an existing installation without a
rebuild, install them directly:

```bash
pip install "pytest>=8" "ldpc==2.4.1" \
    "qldpc @ git+https://github.com/qLDPCOrg/qLDPC@4c4f80109215fa4a28b4d94e2c285e4683487bc1"
```

`pip install .` builds the seven libraries and installs them inside the Python package:

```
<site-packages>/eldpc/_libs/libeldpc.so
<site-packages>/eldpc/_libs/libeldpc_core_avx2.so
<site-packages>/eldpc/_libs/libeldpc_core_avx512.so
<site-packages>/eldpc/_libs/libeldpc_core_avx2_ada64.so
<site-packages>/eldpc/_libs/libeldpc_core_avx512_ada64.so
<site-packages>/eldpc/_libs/libeldpc_cpu_avx2.so
<site-packages>/eldpc/_libs/libeldpc_cpu_avx512.so
<site-packages>/eldpc/include/eldpc.h
<site-packages>/eldpc/lib/pkgconfig/eldpc.pc
<site-packages>/eldpc/lib/cmake/eldpc/eldpcConfig.cmake
```

A built wheel's listing and the `cuobjdump --list-elf` / `--list-ptx` output of its libraries are
recorded in `tools/results/build-listing/wheel-and-images.json`. `libeldpc.so` looks for its cores in its own
directory, so the seven library files must stay together. The CUDA runtime is not a shared dependency of the installed libraries: it is linked
statically into each GPU core, so the package uses the runtime of the toolkit it was built with
and cannot pick up a `libcudart.so` from the system, whatever the loader search path contains.
The only run-time dependency beyond the C++ runtime is the NVIDIA driver's `libcuda.so.1`, which
the static runtime loads itself, and only when a GPU handle is created. The CPU cores link no CUDA
library at all. Check with `readelf -d <prefix>/eldpc/_libs/libeldpc_core_avx2.so`: no
`libcudart` line, and `ldd` lists no CUDA library; the same for `libeldpc_cpu_avx2.so`, whose
`readelf -d` lists only `libstdc++`, `libm`, `libgcc_s`, `libc` and the dynamic loader.

## Static CUDA runtime

`CMakeLists.txt` sets `CUDA_RUNTIME_LIBRARY` to `Static` and links `CUDA::cudart_static` into
every core. The reason is deployability: a shared `libcudart.so.13` would be a run-time
dependency that a machine with only the driver does not have, and when several CUDA runtimes are
installed the loader would choose one by search path, so two processes could decode with
different runtimes from the same installed package. With the static runtime each core carries the
runtime bytes of the toolkit it was built from; the installed package needs the driver only, and
its behaviour does not depend on which CUDA libraries happen to be installed. The static archive
links into the shared cores without extra flags and leaves no text relocations
(`readelf -d` shows no `TEXTREL`). None of the seven libraries has a `RUNPATH` or `RPATH`: the
front finds its cores in its own directory (`dladdr`), and the cores need nothing but the C and
C++ runtime libraries. The tests check this on the installed libraries:
`tests/test_library_deps.py` (no `RUNPATH`/`RPATH`; no `libcudart` in any core's `NEEDED` and the
runtime's own strings inside every GPU core; no `GLIBC_` symbol version newer than the one
docs/SUPPORT.md states) and `tests/test_devices.py` (no `libcudart` in the dependency closure, and
none mapped by a process that enumerates devices or decodes). The statically linked runtime is governed by the NVIDIA CUDA
Toolkit End User License Agreement, not by the Apache licence of `eldpc`: anyone who redistributes
built cores (for example a wheel) must comply with it (`THIRD_PARTY_NOTICES`).

The wheel is tagged `py3-none-<platform>`: it contains no CPython extension module (the libraries
are loaded with ctypes), so it installs on any Python >= 3.9 on the same platform (tested with
3.12). Its metadata carries the licence expression `Apache-2.0` and the files `LICENSE`, `NOTICE`,
`THIRD_PARTY_NOTICES` and `AUTHORS` (in `*.dist-info/licenses/`). No prebuilt wheels are published
for now; the release is source only.

To build a wheel without installing it:

```bash
pip wheel --no-deps -w dist .
```

(with `--no-build-isolation` only in an environment that already has the build dependencies, as
for the offline install of the README, "Build and install").

## Build with CMake directly

For development on the native libraries:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The libraries land in `build/lib/`. The build compiles with `-Wall` and prints 88 warning lines
with GCC 13.3: 84 from the vendored `ldpc` headers under `src/third_party/ldpc/` (60
`-Wsign-compare` from 8 places, 18 `-Wparentheses`, 6 `-Wunused-variable`) and 4
`-Wunused-variable` from the frozen CPU core (`src/cpu/bp_engine.hpp`); they are expected, and
those sources are kept unmodified. The build options are:

- `-DCMAKE_BUILD_TYPE=Release` (the default when unset): the flags above.
- `CUDACXX` (environment): the `nvcc` to use.
- `-DCMAKE_CUDA_COMPILER=<path>`: the same choice as a CMake variable.
- `-DSKBUILD_PROJECT_VERSION`: not needed for a plain CMake build; the package version is read
  from `pyproject.toml`.
- `-DELDPC_WITHOUT_CUDA=ON|OFF` (or `ELDPC_WITHOUT_CUDA=1|0` in the environment): `ON` builds the
  CPU-only package, `OFF` requires the GPU cores; unset, the GPU cores are built when CMake finds a
  CUDA compiler and the CPU-only package otherwise (see "CPU-only build").

The GPU architectures are fixed in `CMakeLists.txt` (`89-real;120-real;120-virtual`): SASS for
`sm_89` and `sm_120` plus PTX for `compute_120`, in all four cores. The only other build variable
is the core variant (see "Core variants" below); there are no other knobs.

To check the result:

```bash
cuobjdump --list-elf build/lib/libeldpc_core_avx512.so   # sm_89 and sm_120 cubins
cuobjdump --list-ptx build/lib/libeldpc_core_avx512.so   # the compute_120 PTX image
```

The CUDA images carry the fatbin identifier `compute_120` (a naming-only `nvcc` option,
`-Xfatbin=--ident=compute_120`), so the listings name them `compute_120.sm_89.cubin`,
`compute_120.sm_120.cubin` and `compute_120.sm_120.ptx`. The PTX is the image nvcc generates for
the virtual architecture `compute_120`; its PTX target name is `sm_120`, CUDA's name for compute
capability 12.0.

## How the ISA choice works

`libeldpc.so` reads `ELDPC_ISA` once, when it is loaded, then checks the CPU at the same point:

1. `ELDPC_ISA=avx2` or `ELDPC_ISA=avx512` forces that core. If the CPU or the OS does not support
   every instruction set of that core, loading fails with a clear error and every
   `eldpc_create()` reports it.
2. Otherwise the AVX-512 core is used when the CPU reports AVX2 and AVX-512 F/VL/DQ/BW and the OS
   has enabled the opmask and ZMM register state (XGETBV); if not, the AVX2 core is used when AVX2
   and the OS's YMM state are available.
3. On a CPU with neither, loading fails with an error saying so.

The AVX-512 core is compiled with exactly the flag set above, and the check tests exactly that
set. AVX2 is always required as well, because the AVX-512 core is compiled with `-mavx2` too.

`eldpc.isa()` (Python) and `eldpc_isa()` (C) report the core that was loaded; `eldpc_build_info()`
reports the compiler, CUDA toolkit, GPU architectures, host ISA flags and the core identity. When
the front could load no core (step 1's error, step 3's, or a missing core library), the C API
returns `""` from `eldpc_isa()` and the error text from `eldpc_build_info()`; in Python
`eldpc.isa()` and `eldpc.build_info()` raise `eldpc.EldpcError` with that text, which names
`ELDPC_ISA` and what this CPU and OS support, and `eldpc.check()` ends with `NOT READY: ...` and
raises it (tests/test_isa.py).

In Python, `eldpc.load(isa="avx2")` sets `ELDPC_ISA` before the library is loaded. Because the
front reads the variable once, it must be the first load in the process.

## Core variants: the Ada 142-SM class

Each host ISA has two builds of the decoder core. They are compiled from the same six sources with
the same flags; the one difference is a compile definition: `DBENCH_GPU_SM_BUDGET=64` in the ada64
build instead of `DBENCH_GPU_SM_BUDGET=0` (the shared definition list is
`ELDPC_CORE_DEFINITIONS` in `CMakeLists.txt`, with that single value replaced for the ada64
builds). The definition caps the SM count the create path plans its grids, capacities and
formulation for (`src/core/bp_gpu.cu`, `g->sms`): 64 instead of the device's own SM count. On a
142-SM Ada part (RTX 6000 Ada, L40S) that plan is faster and bit-identical to the default build;
it is slower on smaller GPUs, which is why the default build stays the default. The build-info
strings of the two builds of one ISA differ only in `gpu_sm_budget`. `eldpc.build_info()` and
`eldpc_build_info()` report the process's default core, the default build (`gpu_sm_budget=0`), on
every GPU; a handle on a 142-SM Ada part runs the ada64 build (`gpu_sm_budget=64`), which
`Decoder.core_variant` and `eldpc_core_variant()` report per handle.

The two builds of one ISA ship byte-identical CUDA images: `cuobjdump -xelf all` and
`cuobjdump -xptx all` extract the same `compute_120.sm_89.cubin`, `compute_120.sm_120.cubin` and
`compute_120.sm_120.ptx` from the default and the ada64 core, because `DBENCH_GPU_SM_BUDGET` is
used only in host code. The variant changes the host-side launch plan and nothing that runs on
the device.

The front chooses the variant for each handle when the handle is created, from the target device:
device 0 of the process's visible devices, the device the decoder runs on. The choice is a small
table in `src/port/core_variant.h`, keyed by compute capability and an SM-count range:

| compute capability | SM count | variant |
|---|---|---|
| 8.9 (Ada) | at least 128 | `ada64` |
| anything else | any | `default` |

The `ada64` row was chosen at the default preset `a03125c3`, where the 64-SM plan is faster on every
measured problem (RTX 6000 Ada, 142 SMs: bb144 xyz p50 x0.739, bb144 z x0.962, bb288 z x0.647;
RTX 4090, 128 SMs: x0.740, x0.943, x0.661; the 142-SM L40S agrees). The table does not look at the
preset: with long iteration caps the halved per-iteration block count can outweigh the saving. On
bb144 xyz the `ada64` core is slower at `a05c11` (p50 x1.090 on the RTX 6000 Ada, x1.051 on the
RTX 4090) and at `ms30` (x1.373, x1.285). At `ms30` on the RTX 6000 Ada, bb144 z is level
(x0.994) and bb288 z (x0.861) and hgp98 xyz (x0.590) are faster
(`tools/results/ada-sm-budget/rtx6000ada-142sm.json`, `rtx4090-128sm.json`). Which variant is
faster at a long cap depends on the problem, so no rule keyed on the preset is right for every
measured problem. Set `ELDPC_DEVICE_CLASS=default` for long-cap presets on large joint problems
such as bb144 xyz and for comparisons at stock-equivalent settings, or time both variants with the
benchmark.

The first matching row wins; a row can also carry an upper SM bound (`0` means none). To change a
choice after measuring a device class, edit or add a row there and edit the mirror in
`python/eldpc/_variant.py` that `eldpc.devices()` reports from; `tests/test_core_variant.py`
compares the two over a grid of simulated device properties, so they cannot drift apart unnoticed.

`ELDPC_DEVICE_CLASS=default` or `ELDPC_DEVICE_CLASS=ada64` in the environment of the calling
process overrides the table; an empty value counts as unset (the table decides), and any other
value makes `eldpc_create()` fail with an error naming the variable and the value. The variable is read at every create, so one process can hold handles of
both variants; a handle keeps the variant it was created with. `eldpc_core_variant(handle)` (C)
and `Decoder.core_variant` (Python) report a handle's variant, and `eldpc.devices()` reports under
`core_variant` the variant a handle on each GPU would get. This composes with the host ISA choice
above: the variant is selected within the ISA the process loaded, and both builds of each ISA
exist.

The default core of the loaded ISA is loaded with the front, because it is what `eldpc_isa()` and
`eldpc_build_info()` report; the ada64 core of that ISA is loaded with `dlopen(RTLD_LOCAL)` on the
first handle that needs it. Handles of both variants can be alive in one process at the same time;
they return bit-identical corrections, observables and status words.

The two extra GPU cores grow the package: the wheel with all four cores is 8.85 MB, against
4.439 MB for the two-core package before them -- 4.411 MB more, most of it the two extra cores
(2,200,240 and 2,202,493 bytes compressed in the wheel). `tools/package_sizes.py` measures the
wheels and writes the result to `tools/results/wheel-sizes.json`; the byte counts there are one
exact build. A rebuild in another directory differs slightly, because the compiler's
anonymous-namespace symbol hashes depend on the source path; a two-core and a four-core wheel
built the same way differ by the same two cores either way.

The two CPU cores add much less: the seven-library wheel (four GPU cores, two CPU cores, the front)
is 9.082 MB, against 8.856 MB for the four-GPU-core package before them -- 0.227 MB more, of
which 106,350 bytes compressed for `libeldpc_cpu_avx2.so` and 112,816 bytes for
`libeldpc_cpu_avx512.so` (240,608 and 252,896 bytes uncompressed; the four GPU cores are
unchanged). `tools/package_sizes.py` measured the two wheels and wrote
`tools/results/cpu/wheel-sizes.json`; the byte counts there are that exact pair of builds.

The per-handle choice costs essentially no create time. On an RTX 5080 with driver 580.159.03,
the first create of a process takes about 75 ms -- the decoder core's CUDA context -- and later
creates about 0.9 ms for a small problem. The one-time device query is about 5.9 ms, almost all
of it CUDA driver initialisation, which the core pays anyway: the query runs before the core's
CUDA startup, so the first create is unchanged within its run-to-run spread. Once the device
properties are cached, a create with the table differs from one with `ELDPC_DEVICE_CLASS` by a
few tens of microseconds at most. `tools/measure_create_cost.py` measures this and writes
`tools/results/create-cost.json` (which records the hardware, the command and the raw samples).

The second core costs memory when it is loaded. In a fresh process the front maps the default
core (about 8.3 MB of address space, 0.6 MB resident on an RTX 5080); a first ada64 handle maps
the ada64 core on top, about 10 MB more address space and about 6 MB more resident memory than a
second handle of the default variant, which does the same allocation work without loading a core.
A process on an Ada part therefore maps both cores at its first create.
`tools/measure_variant_memory.py` measures this and writes `tools/results/variant-memory.json`.

These four records (`tools/results/wheel-sizes.json`, `tools/results/cpu/wheel-sizes.json`,
`tools/results/create-cost.json`, `tools/results/variant-memory.json`) are development-tree
records: they were measured in the development tree, whose history is not part of this
repository, so the `commit` each names is a revision that does not exist here (each record's
`commit_note` says so). The tools that wrote them are in `tools/` and measure the same quantities on a build of
this repository, but not at that revision.

## The CPU core

`src/cpu/` holds the CPU core's four sources, identical to the fastbpcpu6g sources they came
from except that development-history comments were neutralised (code tokens and line structure:
`src/cpu/TOKENS.sha256`; raw bytes: `src/cpu/SOURCES.sha256`; never edit them):

| file | what it is |
|---|---|
| `fastbpcpu6g.cpp` | the shim behind the decoder ABI: config parsing, the thread team, the per-shot status word |
| `bp_engine.hpp` | the multi-threaded exact min-sum BP engine |
| `osd0_engine.hpp`, `osd_caps.hpp` | the exact bit-packed OSD-0 engine (copies of `src/core/`'s files, which the CPU core needs because `osd0_engine.hpp` includes `bp_engine.hpp` from its own directory) |

`libeldpc_cpu_avx2.so` and `libeldpc_cpu_avx512.so` are built from those four files with exactly
the host flags of the GPU core of the same ISA: the release flags `-O2 -DNDEBUG`, then
`-ffp-contract=off -fno-fast-math`, then `-O3`, then the ISA list (`-mavx2`, or
`-mavx2 -mavx512f -mavx512vl -mavx512dq -mavx512bw`). There is no CUDA source, no `nvcc` step and
no CUDA library in either build; they link `Threads::Threads` for the thread team. Each library
carries a build-info string of the GPU cores' form:

```
compiler=...; flags=...; build_type=Release; isa=avx2; lib=fastbpcpu6g; threads=1..64
```

The `flags=` field is byte-identical to the GPU core's of the same ISA, so a CPU handle and a GPU
handle of one ISA are the same arithmetic on different engines; `lib=` names `fastbpcpu6g`. The
libraries are installed next to the GPU cores (the wheel layout above).

The installed wheel's CPU core was measured against its own GPU core in one session on the
recorded screen machine: 230.4 us p50 per shot at `threads=8` against 42.0 us for the GPU core
(5.48x at the p50, 3.15x at the p99); `threads=1` was 1180.3 us (28.1x). The A/A control (two
handles with the same settings) was 0.997 at the p50, and the packaged core against the same
sources compiled again with the same flags was 1.005 at the p50 and 1.008 at the p99: packaging
changes nothing measurable. Record: `tools/results/cpu/screen.json` and the table it generated,
`tools/results/cpu/screen.md` (AMD Ryzen 9 7950X3D, GeForce RTX 5080, driver 595.91.07, CUDA 13.3;
a screen on a shared development machine, not a benchmark).

The front (`src/port/eldpc_front.cpp`) reads the config key `device=gpu|cpu` and removes it before
the core parses the (unchanged) rest of the string: `device=gpu` is the default and the key never
reaches a core. A `device=cpu` create loads the CPU core of the host ISA the front chose
(`ELDPC_ISA`, CPUID / XGETBV — the same choice as for the GPU cores) with `dlopen(RTLD_LOCAL)` on
first use, and makes no CUDA call: no device query, no driver or runtime call. `threads=` and
`cpus=` pass through to the CPU core unchanged; the GPU core rejects them.
`eldpc_core_variant()` is `"cpu"` for a CPU handle and `eldpc_keep_warm()` returns `ELDPC_OK`
there and does nothing. `ELDPC_DEVICE_CLASS` applies to GPU handles only. GPU and CPU handles can
be alive in one process and return bit-identical results.

`device="auto"` (Python) and `device=auto` (C config string) are opt-in; the default stays
`"gpu"`. Auto runs on the GPU when device 0 is usable by the build's GPU cores (an NVIDIA driver
and a visible device, a driver that supports the cores' CUDA major version, and a compute
capability the cores carry SASS or PTX for, read from the `cuda_arch=` field of their build info)
and on the CPU core otherwise, including in a CPU-only build. In Python the choice is made before
the native create (`python/eldpc/_auto.py`): a fallback to the CPU core raises a `RuntimeWarning`
once per process that names the reason (no driver, no device, a driver too old, an unsupported
compute capability, or a build without GPU cores), `Decoder.device` and `Decoder.core_variant`
report the choice, `threads` applies only when the CPU core is chosen, and a config dict or JSON
file may say `"device": "auto"`. The C front makes the same choice for `device=auto` and drops
`threads=`/`cpus=` for a GPU handle; it has no warning channel, so `eldpc_core_variant()` (`"cpu"`)
reports a fallback and a `device=gpu` create on the same machine reports the reason.
`tests/test_device_auto.py` tests the rules with simulated probes and checks that Python and the
C front choose the same device on the test machine.

A `device="gpu"` create checks the same conditions before the GPU core is called and fails with a
message that says which one is missing and names `device="cpu"` and `device="auto"`; on a device of
a compute capability the cores carry no image for (for example 8.6) the message names that
compute capability, the cores' images (`SASS for sm_89, sm_120 and PTX for compute_120`) and the
compute capabilities they cover (8.9; 12.0 or newer). The message is built by
`src/port/gpu_arch.h`, which `tests/test_device_auto.py` compiles and checks; no device of an
unsupported compute capability is in the records.

### CPU-only build (no CUDA toolkit)

A machine without the CUDA toolkit can build and install the package for the CPU core alone:

```bash
ELDPC_WITHOUT_CUDA=1 pip install .   # or -DELDPC_WITHOUT_CUDA=ON for CMake
```

`pip install .` builds in an isolated environment and installs the build dependencies
(scikit-build-core, CMake, ninja) into it from the index. Add `--no-build-isolation` only in an
environment that already has them (the offline install of the README, "Build and install");
in a fresh virtual environment it fails with `ModuleNotFoundError: No module named
'scikit_build_core'`.

With `ELDPC_WITHOUT_CUDA` unset, a build that finds no CUDA compiler (CMake's
`check_language(CUDA)`: no `nvcc` on `PATH`, `CUDACXX` unset) builds the same CPU-only package
instead of failing, and CMake prints a warning that says what is missing, how to get the GPU cores
(install the CUDA 13 toolkit from https://developer.nvidia.com/cuda-downloads, put its `nvcc` on
`PATH` or set `CUDACXX`, rebuild) and that the result has the CPU cores only; `pip` shows it with
`pip install -v .`. CMake looks for `nvcc` in `CUDACXX`, on `PATH` and in `$CUDA_PATH/bin` only,
so a toolkit under `/usr/local/cuda-13.3` that none of them names is not found. To require
the GPU cores, so that a missing CUDA compiler stays a configure error, set
`ELDPC_WITHOUT_CUDA=0` (or `-DELDPC_WITHOUT_CUDA=OFF`, or
`--config-settings=cmake.define.ELDPC_WITHOUT_CUDA=OFF` for `pip`); `ELDPC_WITHOUT_CUDA=1`
builds the CPU-only package on purpose, without the warning.

The CPU-only build enables no CUDA language, does not look for the CUDA toolkit and never runs
`nvcc`; it builds `libeldpc.so` and the two CPU cores only (three libraries) and installs them with
the header, pkg-config and CMake files. The front then reports the CPU core as the process's core
(`eldpc.isa()` and `eldpc.build_info()` work; `build_info()`'s `core=` names `lib=fastbpcpu6g`);
`device="cpu"` and `device="auto"` run on the CPU core, and a `device="gpu"` create fails with
"eldpc: device=gpu is not available: this build has no GPU cores (built without a CUDA toolkit); to
run on the CPU core, create the decoder with device=cpu, or with device=auto to use the GPU only
when it is usable (Python: device=\"cpu\" or device=\"auto\"; C: device=cpu or device=auto in the
config string); for the GPU cores, install the CUDA 13 toolkit
(https://developer.nvidia.com/cuda-downloads), put its nvcc on PATH or set CUDACXX, and rebuild
eldpc (pip install .)". Every GPU create error ends with the same API-neutral advice (Python, C,
the benchmark and the tuner all show it). The Python API, `eldpc.qldpc`, the presets and
every CPU-core output are the same as in a full build.

Every build installs a build record next to the libraries, `eldpc/_libs/build.json`:
`{"schema": "eldpc.build.v1", "gpu_cores": true|false, "cuda_choice": "auto"|"explicit", "cuda":
"NVIDIA 13.3.73", "cuda_arch": "sm_89,sm_120,compute_120"}` (`"auto"`: CMake looked for a CUDA
compiler; `"explicit"`: `ELDPC_WITHOUT_CUDA` decided; `cuda` and `cuda_arch` are the GPU cores'
build-info fields, empty in a CPU-only build). `eldpc.devices()` reads `cuda` and `cuda_arch` from
it, while no library is loaded, to report whether the build supports each device (`supported`).
`import eldpc` reads it, without loading a library, and for a CPU-only build that CMake chose
because it found no CUDA compiler asks the CUDA driver for device 0 the way `device="auto"` does
(`cuInit`, `cuDeviceGetCount` and attribute queries; no CUDA context). When the driver reports a
device it warns once per process (a `RuntimeWarning`: "eldpc: this build has no GPU cores because
nvcc was not found at build time, but the CUDA driver reports a GPU (...)", with the README section
to follow). A build with GPU cores, a CPU-only build chosen with `ELDPC_WITHOUT_CUDA=1` and a
source-tree run without the record make no CUDA call at import, and the check never makes the
import fail. `python -c "import eldpc; eldpc.check()"` prints the build (GPU cores and their CUDA
images and toolkit, CPU cores, host ISA), the devices and a test decode per usable core, and ends
with `GPU ready: ...` or `CPU only: <why>` (README, "Check the installation");
`tests/test_install_check.py` tests both.

`tools/cpu_only_build.py` measures this end to end with a `PATH` whose `nvcc` is `/bin/false` and
`CUDACXX` unset: the CMake build and the wheel build succeed, the wheel is 277,192 bytes with three
libraries (`libeldpc.so`, `libeldpc_cpu_avx2.so`, `libeldpc_cpu_avx512.so`; the two CPU cores are
byte-identical to the full build's), and the installed wheel runs 162 `cpu_core` tests with 0
failures, 0 errors and 0 skips with `CUDA_VISIBLE_DEVICES=` empty. Record:
`tools/results/cpu/no-cuda-build.json` (revision `f40d6f9`, one build; zip timestamps move a rebuild's
byte count by a few hundred bytes). At revision `f44723c`, on the RTX 6000 Ada host with an AVX2-only
EPYC 7542, the same script built a 317,446-byte wheel (the front has grown since; the two CPU cores
are byte-identical to the earlier record's) and its `cpu_core` tests passed
(`tools/results/cpu/release/no-cuda-build.json`).

An existing build directory keeps an explicit choice (and the result of its CUDA compiler search);
pass `-DELDPC_WITHOUT_CUDA=OFF` (or `-DCMAKE_CUDA_COMPILER=<path to nvcc>`) to a build directory
that was configured CPU-only to build the GPU cores again. `pip install .` configures a fresh
build directory each time.

## Tests without a GPU

The package can be built and tested on a machine without a usable GPU. The test suite decides with
the rule of `device="auto"` (a build with GPU cores, a driver for their CUDA version and a compute
capability they carry an image for; `tests/eldpc_testlib.py`): with no CUDA device visible, with a
CPU-only build, or with a GPU the build does not support (an RTX 3090, compute capability 8.6, for
example), the API and qLDPC tests build their decoders on the CPU core (on a supported GPU they run
on the GPU core), the CPU-core tests run as always, and the tests that need a GPU skip with a
reason that says which of these it is instead of failing. The tests that need no GPU carry the
registered pytest marker `cpu_core`; run them alone with

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -m cpu_core tests
```

Everything they need (the packaged reference decoder of the verification suite included) is built
from the package tree, so this command needs no GPU and no other setup (the qLDPC tests among them
need the `qldpc` extra); only the optional comparisons whose variable is not set skip (for example
the `ELDPC_REFERENCE_PYTHON` ones: 185 passed and 4 skipped on the RTX 3090 fresh-clone install test of
`tools/results/matrix/tested-configurations.json`; the count grows with the suite). The other
tests, `-m "not cpu_core"`, either pass without a GPU (for example the verification-suite tests) or
skip: the GPU tests, and the comparisons with separately built libraries when `ELDPC_REFERENCE_LIB`
or `ELDPC_REFERENCE_LDPC_LIB` is not set. `ELDPC_REQUIRE_GPU=1` keeps its meaning: on a machine
that should have a working GPU installation, a missing or unsupported device fails the run instead
of skipping the GPU tests.

A recorded run of both ISA builds with `CUDA_VISIBLE_DEVICES=` empty
(`tools/results/cpu/no-gpu-tests.json`, written by `tools/cpu_no_gpu_tests.py`, with the
reference libraries given) gives, for each ISA, 162 `cpu_core` tests with 0 failures, 0
errors and 0 skips, and 170 other tests with 0 failures and 49 skips (the tests that
need a GPU), at revision `f40d6f9`. At revision `f44723c` on the AVX2-only RTX 6000 Ada host
(`tools/results/cpu/release/no-gpu-tests.json`, AVX2 only, `ELDPC_REFERENCE_LIB` given) the counts are
210 `cpu_core` tests (5 skipped, the optional comparisons) and 160 other tests (63 skipped), 0
failures; the run without any NVIDIA driver is `tools/results/cpu/release/no-driver-run.json`
(`python tools/cpu_no_gpu_tests.py --no-driver`).

## Device enumeration and reported versions

`eldpc.devices()` enumerates the CUDA devices through the CUDA driver API in `libcuda.so.1`
(`cuInit`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceGetUuid`,
`cuDriverGetVersion`), not through the CUDA runtime: no CUDA runtime is loaded anywhere in the
package, and enumeration needs no decoder core to be loaded (it also works on a host whose CPU
cannot run either core). Each entry is a dict with:

| key | meaning |
|---|---|
| `name` | the device name, e.g. `"NVIDIA GeForce RTX 5080"` |
| `uuid` | the device UUID as 8-4-4-4-12 lowercase hex (the `GPU-<uuid>` value nvidia-smi prints, without the `GPU-` prefix); `None` if the driver has no UUID entry point |
| `compute_capability` | `"major.minor"`, e.g. `"12.0"` |
| `sm_count` | streaming multiprocessors |
| `l2_bytes` | size of the device's L2 cache |
| `cuda_driver_version` | the CUDA version the installed driver supports, `"major.minor"`, from `cuDriverGetVersion` — for example a 580 driver reports `"13.0"` |
| `driver_version` | an alias of `cuda_driver_version`, kept for compatibility |
| `nvidia_driver_version` | the installed NVIDIA display driver's version from NVML (`libnvidia-ml.so.1`, `nvmlSystemGetDriverVersion`), e.g. `"580.159.03"`, or `None` |
| `nvidia_driver_version_reason` | `None` when the version was read; otherwise a sentence saying why (NVML not installed, `nvmlInit` failed, or the query failed) |
| `core_variant` | the decoder core variant a handle on this device would get (`"default"` or `"ada64"`): the table's choice for its compute capability and SM count, or `ELDPC_DEVICE_CLASS` when that variable holds a valid variant |

The two version keys answer different questions: the CUDA driver API version is the newest CUDA
release the installed driver supports, while the NVIDIA driver version identifies the display
driver package itself. An empty list means that no CUDA driver or no device is visible to the
process (for example `CUDA_VISIBLE_DEVICES=""`).

The NVIDIA driver version is read once per process and then cached: it cannot change while the
process runs, and an NVML init/query/shutdown cycle costs milliseconds (without the cache every
`devices()` call would pay it). A failed read is not cached, so a host where NVML becomes
available later reports the version at the next call.

## Check the decoder sources

The six decoder sources in `src/core/` are identical to the benched sources of the decoder except
that development-history comments were neutralised; code tokens and line structure are unchanged,
and the libraries compiled from them are identical
(`tools/results/comment-neutralisation/record.txt`). `src/core/SOURCES.sha256` lists the SHA-256
values of their raw bytes and `src/core/TOKENS.sha256` those of their text without comments; from
the package root:

```bash
(cd src/core && sha256sum -c SOURCES.sha256)
python3 tools/check_comment_only.py --check
```

To check them against the source tarball they came from (the tarball is not part of the package; the
check above does not need it), run this from the package root as well:

```bash
python tools/check_core_tarball.py /path/to/fastbpgpu6g-src.tar.gz
```

The tool reads the tarball in memory, finds each file under `candidates/fastbpgpu6g/` and
compares its SHA-256 with the manifest; it exits non-zero on any mismatch.

## Tests

```bash
pip install ".[test,qldpc]"        # once: pytest, qLDPC and ldpc 2.4.1 (see above)
python -m pytest tests
```

GPU tests skip with a clear reason when no CUDA device is visible; set `ELDPC_REQUIRE_GPU=1` on a
machine that has a GPU, so that a missing or broken driver fails the run instead. Without qLDPC,
or with a qLDPC older than `eldpc.qldpc` needs, `tests/qldpc` is skipped with a reason that names
the install command, and so are the `tests/verify` tests that build corpora; the rest of the
suite runs. Some tests compare against references outside the installed package and skip unless
their variable is set:

| variable | what it points to | tests |
|---|---|---|
| `ELDPC_REFERENCE_LIB` | a separately built library of the same decoder sources (decoder ABI v1, `dbench_*` symbols); a separate CMake build of this repository's `eldpc_core_avx2` target is one (docs/VERIFY.md, "Status words") | `test_bit_identity.py` |
| `ELDPC_REFERENCE_PYTHON` | the reference construction's `python/` directory (the reference DEM-conversion package) | `test_dem.py` |
| `ELDPC_CORE_TARBALL` | the decoder source tarball | `test_core_sources.py` |
| `ELDPC_CORE_SNAPSHOT` | a directory with the six decoder sources | `test_core_sources.py` |

The public bit-identity check against the reference BP+OSD-0 implementation needs none of these:
see `verify/`. `tests/test_core_variant.py` covers the device-class table (with simulated device
properties), `ELDPC_DEVICE_CLASS`, and handles of both variants in one process.

## C API

`include/eldpc.h` declares the API; link against `libeldpc.so` directly:

```bash
cc app.c -I<prefix>/eldpc/include -L<prefix>/eldpc/_libs -leldpc \
   -Wl,-rpath,<prefix>/eldpc/_libs -o app
```

where `<prefix>` is the site-packages directory of the installation. The installation also ships
a pkg-config file and a CMake package config:

```bash
export PKG_CONFIG_PATH=<prefix>/eldpc/lib/pkgconfig
cc app.c $(pkg-config --cflags --libs eldpc) \
   -Wl,-rpath,$(pkg-config --variable=libdir eldpc) -o app
```

```cmake
find_package(eldpc REQUIRED)          # configure with -DCMAKE_PREFIX_PATH=<prefix>/eldpc
target_link_libraries(app PRIVATE eldpc::eldpc)
```

The problem struct, return codes and status word are those of decoder ABI v1, renamed with
the `eldpc_` prefix; the `eldpc_problem` field order matches `dbench_problem`, so a C program
written against either works with both. `eldpc_core_variant(handle)` returns the core variant of
a handle (`"default"` or `"ada64"`), and `eldpc_core_variant_for_device(major, minor, sm_count)`
returns the device-class table's choice for a device without the `ELDPC_DEVICE_CLASS` override.
