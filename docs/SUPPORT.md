# eldpc support

What `eldpc` runs on, what was tested and what was only built for, and how to check each item on
your own machine. "Tested" means a command of this package was run on that configuration and the
document named in the row, or the table "Tested configurations" below, records it; "built for"
means the package contains code for it but no record of a run on it is included. The machine of
every recorded package run is named in [RESULTS.md](RESULTS.md) and, for the test runs of "Tested
configurations", in `tools/results/matrix/tested-configurations.json`; the machines behind the
tuning records are listed in [tuning/KNOWLEDGE.md](../tuning/KNOWLEDGE.md) section 0.2.

The deployment classes this package is built for are an RTX 6000 Ada class GPU (compute
capability 8.9) with a Sapphire Rapids Xeon host, and Blackwell RTX 5080 class GPUs (compute
capability 12.0) with Zen 4, Zen 5 or Intel hosts.

## Tested configurations

On each machine of this table the package was built from source with `pip install .` and every
check below passed; `tools/results/matrix/tested-configurations.json` records, per machine, the
GPU, CPU, driver and toolkit, the package revision and each check's result line (the CUDA 13.2
row and the RTX 5090 and RTX 3090 rows ran the checks their notes name instead):

- the full test suite, `ELDPC_REQUIRE_GPU=1 python -m pytest tests`, with the qLDPC tests;
- bit identity of the corrections, the observables and the status words, single shot and in
  batch, against a separately built library of the decoder (the decoder's own reference build):
  with the automatically chosen host core on the bb72 z, bb144 xyz and hgp98 z problems at
  `a03125c3` and `ms30` (2,000 shots per row), and with the AVX2 core forced (`ELDPC_ISA=avx2`)
  on bb144 xyz at both configs (500 shots per row);
- `python verify/run_verify.py --quick`: 0 mismatches.

| CPU | GPU | NVIDIA driver | CUDA toolkit | notes |
|---|---|---|---|---|
| Ryzen 9 9950X (Zen 5) | RTX 5080 (compute capability 12.0) | 580.159.03 | 13.3 | |
| Ryzen 9 7950X3D (Zen 4) | RTX 5080 (compute capability 12.0) | 595.91 | 13.3 | |
| Ryzen 9 7900 (Zen 4) | RTX 4090 (compute capability 8.9, 128 SMs) | 610.43 | 13.3 | virtual-machine guest; the `ada64` core variant was selected and is bit-identical |
| Xeon Platinum 8559C (Emerald Rapids) | RTX PRO 6000 Blackwell Server Edition (compute capability 12.0) | 615.71 | 13.3 | virtual-machine guest |
| 2x Xeon Silver 4514Y (Emerald Rapids) | L40S (compute capability 8.9, 142 SMs) | 595.84 | 13.3 | the `ada64` core variant was selected and is bit-identical |
| 2x Xeon Silver 4514Y (Emerald Rapids) | L40S (compute capability 8.9, 142 SMs) | 595.84 | 13.2 (the machine's default `nvcc`) | a fresh-clone install test that followed the README in a new account: `pip install ".[test,qldpc]"`, the test suite (without `ELDPC_REQUIRE_GPU`), `verify --quick` (0 mismatches; repeated with the status words compared against a CMake build of the same tree, 0 mismatches) and `bench/bench.py all --quick` passed; the corpus bit-identity check above was not run with this toolkit |
| 2x EPYC 7542 (Zen 2, AVX2 only) | RTX 6000 Ada Generation (compute capability 8.9, 142 SMs) | 580.126.09 | 13.3 | the `ada64` core variant was selected and is bit-identical; an AVX2-only host, so the front chose the AVX2 core automatically and both bit-identity runs used it; the screens in [RESULTS.md](RESULTS.md) were taken here; every check passed again at revision `894ebb3` (the release's code apart from the benchmark's host-load threshold, changed in `f44723c`) |
| Xeon w9-3495X (Sapphire Rapids-W, one socket) | L40S (compute capability 8.9, 142 SMs) | 595.91.07 | 13.3 | bare metal (no hypervisor flag); the `ada64` core variant is the one the device-class table gives this GPU, and the benchmark records of `tools/results/host-sprw/` show its handles used it; the records of [RESULTS.md](RESULTS.md) section 6 and `tools/results/host-sprw/` were taken here; the correctness checks (test suite, both bit-identity runs, `verify --quick`) passed again at revision `894ebb3`, on a loaded host |
| Ryzen 7 9700X (Zen 5) | RTX 5090 (compute capability 12.0, 170 SMs) | 595.71.05 | 13.3 and 13.4 | a fresh-clone install test on a desktop machine that followed the README alone on a fresh Ubuntu 24.04 desktop (revision `46cd59e`): the package built with each toolkit (`cuda-toolkit-13-3` and `cuda-toolkit-13-4` from NVIDIA's apt repository) and `verify --quick` passed with both (0 mismatches); with 13.3 the README quick starts reproduced, the test suite passed (without `ELDPC_REQUIRE_GPU` or the reference library: 309 passed, 25 skipped), and `bench/bench.py all --quick` and the tuner ran; the corpus bit-identity check above was not run. The benchmark's `--quick` **screen** (powersave governor, GPU clocks rising from idle; not a timing-grade run): bb144 xyz at `a03125c3`, C single shot back to back, p50 33.55 us, p99 43.14 us |
| EPYC 7K62 (Zen 2, AVX2 only; shared host) | RTX 3090 (compute capability 8.6: **not supported** on the GPU) | 580.159.03 | 13.0 (the machine's `nvcc` V13.0.88) | a fresh-clone install test on an unsupported GPU (revision `f9fb27c`): the package built with CUDA 13.0; every GPU create failed with the unsupported-device message (compute capability 8.6, the images and what they cover) and the README quick starts reproduced on the CPU core (`device="cpu"` or `"auto"`); `CUDA_VISIBLE_DEVICES= pytest -m cpu_core tests`: 185 passed, 4 skipped; the GPU checks could not run. At that revision `pytest tests` failed the GPU tests on the visible but unsupported GPU (they now skip, see "GPU architectures") |

On every machine except the EPYC 7542 the front chose the AVX-512 core automatically (the
`host_isa` of each run in the record), so the AVX2 core was tested there by forcing it (the EPYC
7K62 of the RTX 3090 row is AVX2-only too). Apart from the RTX 5090 row's `--quick` benchmark screen
and the RTX 3090 row's CPU-core screen, no timing results are part of these records; latency must
be measured on the deployment machine with `bench/bench.py`.

### What to expect from a first benchmark run

The latency records of this package for the two 142-SM Ada machines it has (bb144 xyz, the
benchmark's default problem, at `a03125c3`; C API single shot, back to back). A first run of the
same command on the same class that lands near these numbers shows the package performing as
recorded; neither is a timing-grade result, and the host and its load move them:

| machine | command (from the repository root) | C p50 / p99 | conditions | record |
|---|---|---|---|---|
| RTX 6000 Ada, 2x EPYC 7542 (Zen 2, AVX2 core) | `python bench/bench.py all --quick --problem bb144` (the README's command, with its default problem named; pinned automatically to CPU 1) | 64.59 / 86.55 us; A/A band of the p50 difference [-1.36, 1.69] us | a `--quick` screen on a shared host (1-min load average about 3.5 to 4.8 of 128 logical CPUs), governor `schedutil`, GPU clocks not locked; revision `f44723c` | `docs/results/bench-quick.json` ([RESULTS.md](RESULTS.md) section 2) |
| L40S (the RTX 6000 Ada's chip configuration on another board), Xeon w9-3495X (Sapphire Rapids-W, the deployment CPU family; AVX-512 core) | `python bench/bench.py all --problem docs/results/tuning-bb144/model.dem --cpu 8` (full length) | 59.03 / 68.57 us | a full-length run, pinned to CPU 8, governor `powersave` with EPP `balance_performance`, GPU clocks not locked; revision `21b2820` | `docs/results/target/bench.json` ([RESULTS.md](RESULTS.md) section 6) |

There is no `--quick` row for the L40S + Sapphire Rapids-W class: no `--quick` record of that class
was taken on a quiet host, so its full-length row is the reference for a first run there. A
`--quick` run times fewer shots than the full-length one, so its p99 moves more from run to run, and
a loaded host raises it further.

The two differ in host (Zen 2 against Sapphire Rapids-W, AVX2 against AVX-512 core), run length and
board, so they bound what to expect rather than predict a deployment machine's latency; on the
deployment pair (an RTX 6000 Ada with a Sapphire Rapids host) measure it there, pinned and on a quiet
host (docs/BENCH.md).

The CPU core (`device="cpu"`) was tested on the machine below: the package built from source (a
full build, and the CPU-only build of `ELDPC_WITHOUT_CUDA=1`), installed from its wheel, and every
check below passed for both CPU-core builds (`ELDPC_ISA=avx512` and `ELDPC_ISA=avx2`):

- with no GPU visible (`CUDA_VISIBLE_DEVICES=` empty): the 162 `cpu_core` tests (none skipped) and
  the rest of the suite (170 tests, 49 of them skipped because they need a GPU), recorded in
  `tools/results/cpu/no-gpu-tests.json`; the CPU-only build ran the same 162 `cpu_core` tests
  (`tools/results/cpu/no-cuda-build.json`);
- bit identity of the corrections, the observables, the status words and the soft output against
  the reference BP+OSD-0 on corpora of the bivariate-bicycle and hypergraph-product problems, at
  `threads=1` and `threads=4`, single shot and in batches, and of the corrections, observables and
  status words against a separately built library of the GPU decoder; 0 mismatches;
- with the GPU visible: the full test suite (`ELDPC_REQUIRE_GPU=1`), which includes GPU and CPU
  handles in one process returning bit-identical results;
- with every `dlopen` of the NVIDIA driver libraries (`libcuda.so.1`, `libnvidia-ml.so.1`) made to
  fail by an `LD_PRELOAD` shim, so that the process sees no driver: the 162 `cpu_core` tests, a CPU decode, an empty
  `eldpc.devices()` and a `device="gpu"` create failing with a message that names `device="cpu"`.

The same three checks were run again at revision `f44723c` on the RTX 6000 Ada host with the AVX2-only
EPYC 7542 (AVX2 core only), as recorded runs of the package's tools, in `tools/results/cpu/release/`:
`CUDA_VISIBLE_DEVICES= python tools/cpu_no_gpu_tests.py --isa avx2` (210 `cpu_core` tests, 5
skipped; 160 other tests, 63 skipped; `no-gpu-tests.json`), `python tools/cpu_no_gpu_tests.py
--no-driver --isa avx2` (the shim above, which the tool builds: 205 `cpu_core` tests passed and 5
skipped, `devices()` empty, the CPU decode, the GPU create's message, `check()`'s `CPU only:` line and no
driver library mapped; `no-driver-run.json`) and `python tools/cpu_only_build.py` (a three-library
wheel of 317,446 bytes whose two CPU cores have the SHA-256 of the full build's; `no-cuda-build.json`).

| CPU | GPU | NVIDIA driver | CUDA toolkit | notes |
|---|---|---|---|---|
| Ryzen 9 7950X3D (Zen 4) | RTX 5080 (compute capability 12.0), hidden or visible as above | 595.91 | 13.3 | the timing screens of the CPU core were taken on this machine too ([README.md](../README.md), "CPU core") |

## GPU architectures

The native libraries contain CUDA SASS for `sm_89` and `sm_120` and PTX for `compute_120`, in all
four decoder cores. The driver picks the image that fits the device.

| image | architecture | class | status |
|---|---|---|---|
| SASS `compute_120.sm_89.cubin` | compute capability 8.9 | Ada (RTX 6000 Ada class, L40S, RTX 4090) | tested: L40S (142 SMs) and RTX 4090 (128 SMs), both with the `ada64` core variant, and the RTX 6000 Ada itself (142 SMs, `ada64`) ("Tested configurations"; `tools/results/matrix/tested-configurations.json`); the RTX 6000 Ada's and RTX 4090's device-side timing ratios were also measured in the measurement harness (`tools/results/ada-sm-budget/`; tuning/KNOWLEDGE.md section 4.6) |
| SASS `compute_120.sm_120.cubin` | compute capability 12.0 | Blackwell consumer/workstation/server (RTX 5080 class, RTX PRO 6000) | tested: RTX 5080, RTX 5090 (a fresh-clone install test with `verify --quick`) and RTX PRO 6000 Blackwell Server Edition ("Tested configurations"; `tools/results/matrix/tested-configurations.json`); the decoder's tuning records also name RTX 5090, RTX 5070 Ti and RTX PRO 4000 cards (tuning/KNOWLEDGE.md section 0.2; development records and knob facts, not package tests) |
| PTX `compute_120.sm_120.ptx` | virtual architecture `compute_120` | JIT for devices of compute capability 12.0 or newer that the driver supports | built for; used only when no SASS image matches; no run on such a device is recorded |

A device with another compute capability (for example 8.0, 8.6, 8.7, 9.0 or 10.x) has no image in
the libraries. The front checks device 0's compute capability against the cores' own image list
(the `cuda_arch=` field of their build info) before the core is called: a `device="gpu"` create
fails with a message that names the device's compute capability, the images (SASS for sm_89,
sm_120 and PTX for compute_120) and the compute capabilities they cover (8.9; 12.0 or newer), and
says how to run on the CPU core (`device=cpu` or `device=auto`, in the words of Python and of the C
config string); `device="auto"` runs on the CPU core with a warning that says the same. Before any
create, `eldpc.devices()` reports such a device with `supported: False` and the same reason, and
`eldpc.check()` ends with `CPU only: ...` (the README, "Supported GPUs", lists the GPU families by
name). The test suite applies the same rule: on such a device the GPU tests skip and say why, and
the device-independent tests run on the CPU core. The message and the rule are checked by
`tests/test_device_auto.py` with simulated device properties (compute capabilities 7.5 to 12.1),
and the message was observed on an RTX 3090 (compute capability 8.6, "Tested configurations").
Adding an
architecture is a build change to the `CUDA_ARCHITECTURES` list in `CMakeLists.txt`; the new build
must pass the verification suite again before it is used.

Check on your machine:

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
python -c "import eldpc; print(eldpc.devices())"
cuobjdump --list-elf  <site-packages>/eldpc/_libs/libeldpc_core_avx512.so   # sm_89 and sm_120 SASS
cuobjdump --list-ptx  <site-packages>/eldpc/_libs/libeldpc_core_avx512.so   # compute_120 PTX
```

`eldpc.devices()` enumerates the devices through the CUDA driver API and reports, for each,
`name`, `uuid`, `compute_capability`, `sm_count`, `l2_bytes`, `cuda_driver_version` (the newest
CUDA version the installed driver supports, from `cuDriverGetVersion`; `driver_version` is an
alias of it) and `nvidia_driver_version` (the NVIDIA driver version that `nvidia-smi` also
prints, read through NVML; `None` with a reason in `nvidia_driver_version_reason` when NVML is not
installed). See [BUILD.md](BUILD.md), "Device enumeration and reported versions". The decoder
uses device 0 of the visible devices; select a GPU with `CUDA_VISIBLE_DEVICES`.

## Host ISAs

`libeldpc.so` is built for baseline x86-64 and chooses between two builds of the decoder core at
load time (see [ARCH.md](ARCH.md), section 4):

| core | required CPU features | status |
|---|---|---|
| `libeldpc_core_avx512.so` | AVX2 and AVX-512 F, VL, DQ, BW, and OS support (XGETBV: opmask/ZMM and XMM/YMM state) | tested on Zen 4 (Ryzen 9 7950X3D, Ryzen 9 7900), Zen 5 (Ryzen 9 9950X), Emerald Rapids Xeon (Platinum 8559C, Silver 4514Y) and Sapphire Rapids-W Xeon (w9-3495X) hosts ("Tested configurations"; `tools/results/matrix/tested-configurations.json`); the decoder's tuning records also used AVX-512 hosts of the first three classes (tuning/KNOWLEDGE.md sections 0.2 and 4.9; `tools/results/host-xeon/host-xeon.json`); on the Sapphire Rapids-W host it was also timed against the AVX2 core ("Host policy"; `tools/results/host-sprw/summary.json`) |
| `libeldpc_core_avx2.so` | AVX2 and OS support for the YMM state | tested by forcing it (`ELDPC_ISA=avx2`) on every AVX-512 machine of "Tested configurations" except two fresh-clone runs that did not force it (the CUDA 13.2 L40S run and the RTX 5090 / Ryzen 7 9700X test), and chosen automatically on the AVX2-only EPYC 7542 (Zen 2), where the [RESULTS.md](RESULTS.md) records were taken (`tools/results/matrix/tested-configurations.json`); the tuning records also include AVX2-only hosts (Zen 2, and an Intel Core i9-14900K; tuning/KNOWLEDGE.md section 0.2); `tests/test_isa.py` forces this core on any AVX2 machine and compares its results with the automatic choice |
| none | no AVX2 | loading fails with an error that names the missing state (the front's load check, `src/port/eldpc_front.cpp`; no machine without AVX2 is in the records) |

The AVX-512 core is compiled with exactly `-mavx2 -mavx512f -mavx512vl -mavx512dq -mavx512bw`;
the AVX2 core with `-mavx2`. Both are compiled from the same sources and produce identical
results (checked by `tests/test_isa.py`). Each also has an `ada64` build, which the front chooses
per handle on 8.9 parts with at least 128 SMs (see [BUILD.md](BUILD.md), "Core variants"). `aarch64` hosts are not supported: the
front contains x86 CPUID/XGETBV code and the cores are built for x86-64.

Check on your machine:

```bash
python -c "import eldpc; print(eldpc.isa()); print(eldpc.build_info())"
ELDPC_ISA=avx2    python -c "import eldpc; print(eldpc.isa())"   # forces the AVX2 core
ELDPC_ISA=avx512  python -c "import eldpc; print(eldpc.isa())"   # if unsupported: EldpcError naming ELDPC_ISA and what the CPU supports, exit 1
grep -m1 '^flags' /proc/cpuinfo                                   # avx2, avx512f, avx512vl, ...
```

`ELDPC_ISA` is read once, when `libeldpc.so` is loaded; forcing a core the CPU or the OS does not
support is a load error, not a fallback. In Python, `eldpc.load(isa="avx2")` must be the first
call that loads the library in the process.

## Host policy

What the measurements recommend for the host side of a GPU decode. The host rows come from a
two-socket Emerald Rapids Xeon Silver 4514Y with an L40S, shared with other tenants, so the
directions carry over and the sizes are that machine's (`tools/results/host-xeon/host-xeon.json`;
tuning/KNOWLEDGE.md sections 2.3, 4.4, 4.6, 4.9 and 4.10). The host ISA and core variant rows were also measured on a one-socket
Sapphire Rapids-W Xeon w9-3495X (the CPU family of the deployment class) with an L40S (the
RTX 6000 Ada's chip configuration on a board with other clocks and power limit), as alternating
full-length benchmark runs, and keep-warm at 1 ms and 10 ms gaps
(`tools/results/host-sprw/summary.json`); pinning, the SMT sibling and NUMA placement are not
measured on that host.

| item | policy | measured basis (the pinning, SMT, NUMA, host-ISA and keep-warm sizes: machine M's host study, `tools/results/host-xeon/host-xeon.json`; the core-variant row: `tools/results/ada-sm-budget/`; the Sapphire Rapids-W sizes: `tools/results/host-sprw/summary.json`) |
|---|---|---|
| pinning | pin the thread that calls the decoder to one core; `taskset -c <cpu>` is enough. The library does not pin the calling thread and starts no thread for a GPU handle | unpinned: +0.7 to +11.1 % at the p50, +2.2 to +15.1 % at the p99 |
| SMT sibling | keep that core's SMT sibling idle; either hardware thread of the core is equally good | a busy sibling: +3.3 to +6.4 % p50, +7.4 to +9.1 % p99, +14 to +17 % p99.9 |
| NUMA node | on a host with more than one NUMA node, use a core on the GPU's node | the other socket: +0.5 to +2.3 % p50; its p99 was not resolved |
| host ISA build | the shipped AVX-512 core, which the front chooses automatically when the CPU and OS support it; no `-march` flag and no other compiler | the AVX2 core was about 4-13 % slower on two benchmark problems at `a03125c3`; `-march=sapphirerapids` and `-march=x86-64-v4` builds measured no difference. On the Sapphire Rapids-W host with an L40S the AVX2 core was slower on bb288 z by 2.07 µs (C) and 1.97 µs (Python) at the p50 (+4.8 % and +3.8 %, outside the spread of the runs and the A/A band); on bb144 xyz its C p50 difference (+0.95 µs) was not resolved |
| keep-warm | call `keep_warm()` while the handle idles between decodes at a call cadence, at least every 0.5 ms (250 and 500 µs measured the same); not in a back-to-back loop | at a 10 ms gap: about -2.5 % at the p50 on one problem, not resolved on another; back to back: no p50 gain, and +13 % at the p99 on one problem. On the Sapphire Rapids-W host with an L40S, at a 10 ms gap: p50 -0.365 µs on bb144 xyz (inside the A/A band) and -0.341 µs on bb288 z (just outside its band, -0.331 to +0.447 µs); without keep-warm the 10 ms gap cost +0.77 and +1.13 µs at the p50 against back to back in the same runs (computed) |
| core variant on Ada | for long iteration caps (`a05c11`, `ms30` and similar) on large joint problems such as bb144 xyz, and for comparisons at stock-equivalent settings, set `ELDPC_DEVICE_CLASS=default` in the creating process, or time both variants | on the RTX 6000 Ada the `ada64` core was faster at `a03125c3` (bb144 xyz p50 x0.74) and slower on bb144 xyz at `a05c11` (x1.09) and `ms30` (x1.37); on the RTX 4090 x1.05 and x1.28 ([BUILD.md](BUILD.md), "Core variants"); on an L40S with a Sapphire Rapids-W host x0.544 (bb144 xyz) and x0.426 (bb288 z) at `a03125c3` and x1.244 on bb144 xyz at `ms30` |

Check on your machine:

```bash
nvidia-smi topo -m                                  # the GPU's CPU affinity and NUMA node
lscpu -e=CPU,NODE,SOCKET,CORE                       # CPUs with the same CORE are SMT siblings
taskset -c <cpu> python your_decoder_program.py     # pin the calling thread's process
python -c "import eldpc; print(eldpc.isa(), [d['core_variant'] for d in eldpc.devices()])"
```

These rules are for GPU handles. A CPU handle (`device="cpu"`) pins its own threads through
`cpus` and has its own placement rules (next section); its `keep_warm()` does nothing.

## The CPU core

`device="cpu"` runs the optional CPU core instead of a GPU; it needs no CUDA toolkit, no
`libcuda.so.1` and no NVIDIA driver at run time (a CPU handle makes no CUDA call). Its libraries
are `libeldpc_cpu_avx2.so` and `libeldpc_cpu_avx512.so`, built for the same two host ISA sets as
the GPU cores and chosen by the same `ELDPC_ISA` / CPUID / XGETBV decision.

| core | required CPU features | status |
|---|---|---|
| `libeldpc_cpu_avx512.so` | AVX2 and AVX-512 F, VL, DQ, BW, and OS support (XGETBV: opmask/ZMM and XMM/YMM state) | tested on a Ryzen 9 7950X3D (Zen 4): the package's CPU-core tests and the bit-identity check against the reference with the GPU hidden ("Tested configurations") |
| `libeldpc_cpu_avx2.so` | AVX2 and OS support for the YMM state | tested on the same host with `ELDPC_ISA=avx2` (tests and bit-identity check) |
| none | no AVX2 | a CPU create fails with the load error that names the missing state |

Build without the CUDA toolkit: `ELDPC_WITHOUT_CUDA=1` (or `-DELDPC_WITHOUT_CUDA=ON`) builds only
the front and the two CPU cores, with no CUDA language, no CUDA toolkit lookup and no `nvcc`
invocation. This was tested with a `PATH` whose `nvcc` is `/bin/false` and `CUDACXX` unset: the
CMake build and the wheel build succeeded, the three-library wheel (277,192 bytes; its two CPU
cores are byte-identical to the full build's) installed, and its `cpu_core` tests ran 162 with 0
failures, 0 errors and 0 skips with `CUDA_VISIBLE_DEVICES=` empty
(`tools/results/cpu/no-cuda-build.json`, written by `tools/cpu_only_build.py`; one build, so the
byte count moves a few hundred bytes between rebuilds; the SHA-256 of its CPU cores equal those in
`tools/results/cpu/no-gpu-tests.json`). A `device="gpu"` create on that build fails with a
message naming `device="cpu"`. With `ELDPC_WITHOUT_CUDA` unset, a build that finds no CUDA
compiler makes the same CPU-only package with a CMake warning instead of failing
([BUILD.md](BUILD.md), "CPU-only build").

The CPU core's latency is about 6x the GPU core's on the AVX-512 hosts measured (a screen on a
shared development machine; [README.md](../README.md), "CPU core"; the packaged wheel's own screen:
230.4 us against 42.0 us p50 at `threads=8`, `tools/results/cpu/screen.json`), and about 10x on a
Zen 2 host with AVX2 only (dual EPYC 7542 with an RTX 6000 Ada: 608.6 against 63.9 us p50 at
`threads=4` on the four cores of one L3, the calling thread pinned to the last `cpus` entry, 18.2x
at `threads=8` across two L3s; a screen, `tools/results/cpu/zen2-avx2-screen/`). Placement: keep the
threads and the calling thread on one CCD or one socket; across the two CCDs of the measured Ryzen
9 7950X3D, `threads=16` was 2.41x the p50 latency of `threads=8` on one CCD, and the packaged
wheel's placement screen measured 2.334x again (`tools/results/cpu/placement.json`). That screen
also found no gain from SMT: `threads=16` on one CCD using its SMT second contexts was 1.017x the
p50 of `threads=8` on the same CCD's physical cores, and `threads=8` on the second contexts was
1.006x. The advice to keep a two-socket Xeon's threads on one socket is **not measured** with this
package: there is no two-socket host in the records, so it is an expectation from the one-CCD
result, not a tested number.

A Zen 5 data point, also a **screen**: on machine A of [tuning/KNOWLEDGE.md](../tuning/KNOWLEDGE.md)
section 0.2 (a Ryzen 7 9850X3D, 8 cores, with an RTX 5080), outside its bench mode (a normal boot
with no isolated CPUs, performance governor, the GPU idle), the AVX-512 CPU core at revision
`f556bee` decoded bb144 xyz at `a03125c3` with one handle, back to back (25,000 distinct syndromes
each decoded twice, 50,000 calls pooled per run), at p50 / p99 148.6 / 185.0 us with `threads=8` on
CPUs 0-7, one hardware thread per core (two runs: 148.56 / 184.96 and 148.54 / 185.92 us), and
223.0 / 252.9 and 227.5 / 256.6 us with `threads=4` (two runs, on CPUs 0-3 and 0-4; a third run
with a transient load spike was discarded). The runs used a single-handle variant of
`bench/c/bench_driver.c` (each call its own shot) because the driver of that revision holds two
CPU handles at once, whose busy-waiting workers then share the pinned CPUs
(`tools/results/cpu/cpu-core-latency-screen.json`, which also keeps the discarded run).

## CUDA toolkit, runtime and driver

| item | value | status |
|---|---|---|
| Build toolkit | CUDA 13.3, `nvcc` V13.3.73 | tested on every machine of "Tested configurations" ([BUILD.md](BUILD.md), "Requirements"; `tools/results/matrix/tested-configurations.json`) |
| CUDA 13.2 | `nvcc` V13.2.86 | built the package and passed the tests, `verify --quick` (also with the status words against a CMake build of the same tree) and `bench.py all --quick` in the fresh-clone install test on the L40S machine ("Tested configurations"; `tools/results/matrix/tested-configurations.json`); the corpus bit-identity check was not run with this toolkit |
| CUDA 13.4 | `nvcc` V13.4.92 | built the package and passed `verify --quick` (0 mismatches) and the README's first quick start in the fresh-clone install test on the RTX 5090 machine ("Tested configurations"; `tools/results/matrix/tested-configurations.json`); the test suite, the benchmark and the corpus bit-identity check were not run with this toolkit |
| CUDA 13.0 | `nvcc` V13.0.88 | built the package in the fresh-clone install test on the RTX 3090 machine, whose GPU the build does not support; the CPU-core tests passed there; no GPU check ran with this toolkit ("Tested configurations"; `tools/results/matrix/tested-configurations.json`) |
| Other 13.x toolkits | the build needs `nvcc` and CMake's `find_package(CUDAToolkit)` | not tested with this package; the decoder's tuning records compare 13.0, 13.2 and 13.3 builds and show that the toolkit changes the kernels' timing ([tuning/KNOWLEDGE.md](../tuning/KNOWLEDGE.md) section 4.8), so re-run the verification and the benchmark after a toolkit change. With several toolkits installed, the build uses the `nvcc` on `PATH` or in `CUDACXX`; the `cuda=` field of `eldpc.build_info()` names the one used |
| CUDA runtime | linked statically into every core (`CUDA::cudart_static`); it is the runtime of the build toolkit (the `cuda=` field of `eldpc.build_info()`) | no CUDA runtime library is needed or loaded at run time ([BUILD.md](BUILD.md), "Static CUDA runtime") |
| Driver | an NVIDIA driver that supports CUDA 13.x. CUDA 13.3's own minimum is 610.43.02; under CUDA minor-version compatibility a CUDA 13.x application also runs on the drivers of earlier 13.x releases, i.e. 580 and later | tested with drivers 580.126.09, 580.159.03, 595.71.05, 595.84, 595.91.07, 610.43.02 and 615.71.09 ("Tested configurations"; `tools/results/matrix/tested-configurations.json`); the 580 and 595 drivers run CUDA 13.3 builds (and 595.71.05 a CUDA 13.4 build) under minor-version compatibility |

Where to get them: the CUDA 13 toolkit from https://developer.nvidia.com/cuda-downloads (only to
build the GPU cores; install the toolkit-only package `cuda-toolkit-13-3`, not the `cuda`
metapackage, which also installs a driver; the README, "CUDA 13 toolkit", has the Ubuntu commands
and how to point the build at its `nvcc`), and an NVIDIA driver that supports CUDA 13.x, release
580 or later (the "Driver" row above; `nvidia-smi` prints the installed release).

At run time the installed package needs only the NVIDIA driver: `libcuda.so.1`, which the static
runtime loads itself, and optionally `libnvidia-ml.so.1` (NVML) for the `nvidia_driver_version`
key of `eldpc.devices()`. No CUDA toolkit, no `libcudart` and no `LD_LIBRARY_PATH` entry for a
CUDA `lib64` directory are needed, and a `libcudart.so` installed elsewhere on the system is never
used. The installed libraries (from the wheel or from `cmake --install`) carry no `RUNPATH` that
names a CUDA directory (CMake's build-tree `RUNPATH`, which a library used straight from a CMake
build directory may carry, is removed at install).

Check on your machine:

```bash
python -c "import eldpc; eldpc.check()"              # the build, the devices, "GPU ready" or "CPU only: <why>"
nvcc --version                                       # only needed to build (often not on PATH: /usr/local/cuda-13.3/bin/nvcc)
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
python -c "import eldpc; print([(d['cuda_driver_version'], d['nvidia_driver_version']) for d in eldpc.devices()])"
readelf -d <site-packages>/eldpc/_libs/libeldpc_core_avx2.so | grep -E 'NEEDED|RUNPATH|RPATH'   # no libcudart, no CUDA RUNPATH
ldd <site-packages>/eldpc/_libs/libeldpc_core_avx2.so                                         # no CUDA library listed
```

## Operating system, C library and Python

| item | tested | minimum of the built package | how to check |
|---|---|---|---|
| OS | Linux x86-64 | Linux x86-64 | `uname -a` |
| glibc | 2.39 ([BUILD.md](BUILD.md)) | `GLIBC_2.34` for all seven libraries, as built on Ubuntu 24.04 (glibc 2.39, GCC 13.3); a build on another distribution needs that distribution's versions | `objdump -T <lib> \| grep -o 'GLIBC_[0-9.]*' \| sort -Vu \| tail -1` |
| libstdc++ | GCC 13.3 ([BUILD.md](BUILD.md)) | as built there: `GLIBCXX_3.4.32` (GCC 13) for the GPU and CPU cores, `GLIBCXX_3.4.21` for `libeldpc.so`; `CXXABI_1.3.9` (GPU cores and front), `CXXABI_1.3.11` (CPU cores) | `objdump -T <lib> \| grep -o 'GLIBCXX_[0-9.]*' \| sort -Vu \| tail -1` |
| Python | 3.12 ([BUILD.md](BUILD.md)) | 3.9 (`requires-python` in `pyproject.toml`; the wheel is `py3-none-<platform>`, no CPython extension) | `python -V` |
| Windows, macOS | not supported: never built or tested (the build and the front are Linux x86-64 only) | -- | -- |

The binaries need a C library and a `libstdc++` at least as new as the symbol versions they were
linked against, which are those of the build machine's toolchain. A distribution older than that
cannot load libraries built on a newer one; building the package on that distribution with its
own toolchain removes the constraint. Run the checks on all seven libraries
(`libeldpc.so`, the four GPU cores `libeldpc_core_{avx2,avx512}{,_ada64}.so` and the two CPU cores
`libeldpc_cpu_{avx2,avx512}.so`):

```bash
python -V
ldd --version | head -1
for f in <site-packages>/eldpc/_libs/*.so; do
  echo "$f"
  objdump -T "$f" | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1
  objdump -T "$f" | grep -o 'GLIBCXX_[0-9.]*' | sort -Vu | tail -1
done
```

## The validated GPU overclock

The overclocked rows of the package's timing-grade results are machine A's (an RTX 5080 with a
Ryzen 7 9850X3D host, [tuning/KNOWLEDGE.md](../tuning/KNOWLEDGE.md) section 0.2), given next
to the same rows at stock clocks. This section is the note that
[README.md](../README.md) ("Results"), [RESULTS.md](RESULTS.md) and [CLAIMS.md](CLAIMS.md) (H8,
H12) link to.

- **Gain**: in the final bench of the shipped core, bb144 xyz at `a03125c3`, single handle, back
  to back, p50 34.02 -> 32.12 us and p99 44.15 -> 42.37 us, i.e. -5.6 % / -4.0 % (80,000 shots per
  profile; `tools/results/final-bench/final-bench.json`, profiles `stock` and `oc`).
- **Profile** (`tuning/data/knob_facts.json`, the knob "GPU overclock profile";
  `final-bench.json`, `machine.profiles`):
  - power limit 110 %: 396 W on this RTX 5080, against 360 W at stock;
  - graphics clock offset +400 MHz with a 3,200 MHz cap: about 3,180 MHz under load (3,172-3,187
    MHz while decoding) against about 2,880 MHz at stock;
  - memory clock offset +3,000 in NVML units, which count the data rate (+1,500 MHz in MSI
    Afterburner terms): a memory clock of 16,301 MHz (16.3 GHz) as the driver reports it while
    decoding;
  - offsets, the cap and the power limit only: the graphics clock is not locked to a fixed value
    and still moves with load.
- **Validation**: outputs bit-identical. 12M shots (3,000 chunks) compared across machines against
  a stock-clocked reference machine, 0 mismatches (a silent error rate below 2.5e-7 per shot at
  95 % confidence); 1.6M shots on the same machine, 0 mismatches; 0 Xid errors over 13.3M
  overclocked shots (`tuning/data/knob_facts.json`, the same entry). The final bench added 2.6M
  paired shots per profile with 0 output mismatches and 0 Xid lines (`final-bench.json`).
- **Gotcha**: change the profile at most once per boot; a second switch in the same boot gave an
  Xid 62 error (`tuning/data/knob_facts.json`, the same entry).

> **This profile was validated as safe on our silicon. Do not apply it to your GPU without proper
> stability testing first.**
>
> Every chip differs, and an overclock that is stable in games can still corrupt compute results:
> check for bit-identical outputs (`python verify/run_verify.py --quick` must print PASS;
> [VERIFY.md](VERIFY.md)) as well as for crashes and Xid errors.

## Known limitations and open items

Each item is a property of the package as built; the last column says how to see it.

| limitation | detail | how to check |
|---|---|---|
| One syndrome per call | every call decodes one shot; `decode_batch` loops the shots on one handle, it is not a batched GPU kernel | [ARCH.md](ARCH.md) section 7; compare `bench.py c-latency` with `bench.py throughput` |
| Exactness is BP+OSD-0 exactness | "exact" means bit-identical to the reference BP+OSD-0, not maximum-likelihood decoding; the LER depends on the code, the noise and the preset | `verify/run_verify.py --quick` checks the identity; `bench.py ler` measures the LER |
| OSD methods beyond OSD-0 | `osd_cs` and `osd_e` are passed through to the vendored `ldpc` code; the bit-identity suite exercises OSD-0 presets, and the qLDPC tests compare one setting of each method | `python -m pytest tests/qldpc`; [QLDPC.md](QLDPC.md) |
| No loss-aware workflow | the decoder decodes static priors; new priors can be set with `set_priors`, but no workflow that decodes a loss model detector by detector is shipped | [VERIFY.md](VERIFY.md), "What the loss problem verifies" |
| One GPU per handle | the core uses device 0 of the visible devices; there is no multi-GPU split | `CUDA_VISIBLE_DEVICES=N`; `eldpc.devices()` |
| One thread per handle | distinct handles may be used concurrently; one handle must not be used by two threads at once | `include/eldpc.h`; `tests/test_api.py::test_concurrent_decoders` |
| The ISA is chosen once | `ELDPC_ISA` is read at load; the core cannot change in a process after the library is loaded | `eldpc.isa()`; `tests/test_isa.py` |
| Binary problems only | matrices over fields other than GF(2), and non-binary detector error models, are rejected | `python -m pytest tests/qldpc` |
| GPU architectures | only sm_89 and sm_120 SASS plus compute_120 PTX are in the fatbin, so a device of compute capability below 12.0 other than 8.9 has no image to run: a `device="gpu"` create fails with a message that names its compute capability, and `device="auto"` runs on the CPU core (no such device is in the records) | `cuobjdump --list-elf`; `eldpc.devices()` |
| Python batch memory | `decode_batch` allocates `shots x n` correction bytes; a million shots of the 75,600-column benchmark problem would need 76 GB | decode in chunks, or use the C API with `out_corrections = NULL` |
| Soft output scope | `soft_output()` is defined for the most recent single-shot decode, not after a batch | `include/eldpc.h`; `tests/test_api.py::test_soft_output` |
| Timing needs a quiet machine | latency and throughput depend on GPU clocks, other GPU users, CPU pinning and host load; the benchmark records warnings but cannot fix them | `bench.py`'s `warnings` list; [BENCH.md](BENCH.md) |
| The RTX 6000 Ada / Sapphire Rapids class is only partly measured | measured on an RTX 6000 Ada: device-side timing ratios of the decoder core in the measurement harness (the SM-budget sweep behind the `ada64` core and its long-cap caveat; `tools/results/ada-sm-budget/full-bench.json`; tuning/KNOWLEDGE.md section 4.6), taken on a Zen 2 host, so they are ratios, not deployment latencies; the "Host policy" above comes from an Emerald Rapids Xeon with an L40S (`tools/results/host-xeon/host-xeon.json`). Measured on a proxy: a Sapphire Rapids-W Xeon w9-3495X (the deployment CPU family) with an L40S (the same AD102 chip configuration, 142 SMs, on a board whose clocks and power limit differ) holds the full-length benchmark, verification and LER records of the RESULTS.md section "The RTX 6000 Ada class with a Sapphire Rapids Xeon" and the host ISA, core variant and cadence runs of `tools/results/host-sprw/`. Not measured: absolute latency on the RTX 6000 Ada board itself with a Sapphire Rapids host. The package itself is tested on the RTX 6000 Ada ("Tested configurations"), and RESULTS.md holds screens from that machine (its Zen 2 host makes them indicative, not deployment latencies); other Ada parts (L40S, RTX 4090) are tested for exactness | `tools/results/ada-sm-budget/rtx6000ada-142sm.json`; `tools/results/host-sprw/summary.json`; [RESULTS.md](RESULTS.md) |
| Source-only release | no prebuilt wheels are published for now: `pip install .` builds the native libraries from this source tree with CMake and `nvcc` (or, with `ELDPC_WITHOUT_CUDA=1`, the CPU-only build); the package is licensed under Apache-2.0 and the vendored MIT-licensed `ldpc` 2.4.1 and `tsl::robin_map` headers and the statically linked CUDA runtime are listed in `THIRD_PARTY_NOTICES` | [README.md](../README.md), "Install" and "Licence"; `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES` |
