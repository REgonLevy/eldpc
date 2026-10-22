# eldpc architecture

This document describes how `eldpc` is built from its code: the layers a decode call passes
through, what runs on the GPU and what runs on the host, how the two host builds of the decoder
are chosen, what the libraries need from the CUDA driver at run time, and which build rules exist
only to keep the outputs bit-identical.

The decoder sources in `src/core/` are the `fastbpgpu6g` sources, identical to the benched sources
except that development-history comments were neutralised (code tokens and line structure unchanged:
`src/core/TOKENS.sha256`, checked by `tools/check_comment_only.py`; raw bytes: `src/core/SOURCES.sha256`;
the libraries compiled from them are identical: `tools/results/comment-neutralisation/record.txt`). The package adds the Python layer (`python/eldpc/`), the C
API front (`src/port/eldpc_front.cpp`) and the build (`CMakeLists.txt`, `pyproject.toml`), plus
the verification suite, the benchmark and the tuner in the source tree; the core files are not
edited.

## 1. The layers

```
Python application
  eldpc.Decoder              python/eldpc/__init__.py
    Decoder.decode_full()    validates the syndrome, allocates the outputs
    _abi.Handle.decode()     ctypes call of eldpc_decode
        |
        v
  libeldpc.so                src/port/eldpc_front.cpp
    eldpc_decode()           forwards through a function pointer resolved at load time
        |
        v
  libeldpc_core_avx2.so      src/core/   (or libeldpc_core_avx512.so)
  or libeldpc_cpu_<isa>.so   src/cpu/    (device="cpu")
    dbench_decode()          src/core/fastbpgpu6g.cpp or src/cpu/fastbpcpu6g.cpp: decode_one()
      all-zero syndrome?     zeros + BP_CONVERGED, no device work
      otherwise              fastbp::Engine::run()     src/core/bp_gpu.cu (CPU: src/cpu/bp_engine.hpp)
        one cooperative CUDA kernel per shot, all BP iterations (CPU: T threads per shot)
        host: OSD-0 on non-converged shots            src/core/osd0_engine.hpp (src/cpu/ copy on the CPU)
        observables O*x                               compute_obs / compute_obs_support
```

A C application calls the same `eldpc_*` functions directly through `include/eldpc.h`; the Python
layer only adds validation, the ctypes crossing and allocation.

## 2. The call path of one decode

1. **Python: `Decoder.decode_full(syndrome)`** (`python/eldpc/__init__.py`). The syndrome is
   converted to a C-contiguous `uint8` array of `m` bytes and every entry is checked to be 0 or 1.
   `_abi.Handle.decode` (`python/eldpc/_abi.py`) allocates three fresh output arrays (correction
   `n`, observables `k`, status 1) and calls `eldpc_decode` through ctypes.
2. **Front: `eldpc_decode`** (`src/port/eldpc_front.cpp`). The front is a thin forwarder: at load
   time it resolved the core's `dbench_*` symbols with `dlsym` and stored them in function
   pointers; every `eldpc_*` call casts the handle and calls the matching pointer. The front does
   no decoding work of its own.
3. **Core: `decode_one`** (`src/core/fastbpgpu6g.cpp`). It checks the output pointers, scans the
   syndrome (which is read in place; no copy is made on the OSD-0 path), and short-circuits the
   all-zero syndrome: zero correction, zero observables, `ST_BP_CONVERGED`, 0 iterations, no GPU
   work at all.
4. **GPU BP: `fastbp::Engine::run`** (`src/core/bp_gpu.cu`). Everything shot-independent was done
   at `eldpc_create`; the decode path only packs the syndrome, launches, and hands back results.
   - The syndrome is bit-packed into the kernel's launch parameters when `m <= 2048` (32 64-bit
     words, `kSynWords`); a wider problem gets one host-to-device copy on the handle's stream
     instead.
   - One `cudaLaunchCooperativeKernel` launch runs the whole shot: all iterations of the
     min-sum BP, with a grid-wide barrier between the two half-iterations. The host is involved
     once per shot, not once per iteration.
   - The kernel ends by publishing the shot's status word (BP iterations run, converged flag) and
     a generation counter into a **host-mapped pinned block**; the host polls that block with a
     bounded spin and, once the generation matches, knows which device-to-host copies this shot
     needs before it issues them. A shot that converged needs only its decision; a non-converged
     OSD-0 shot needs its decision, the posterior and the front of the OSD-0 column order; a
     plain BP handle needs the base hand-off. A missed poll falls back to an unconditional copy
     and a stream synchronisation, so a slow kernel cannot hang the host.
5. **Host BP decision.** `Engine::run` copies back and exposes the hard decision
   (`Engine::decoding`), the posterior (`Engine::posterior`, lazily fetched by
   `Engine::fetch_posterior`) and `last_converged`.
6. **Host OSD-0** (`src/core/osd0_engine.hpp`, class `Osd0Engine`), only for `decoder=bposd`
   handles whose shot did not converge:
   - The device already collected a *prefix* of the OSD-0 column order while the BP kernel was
     running (rung one: up to `kOsdPrefixKA` = 384 candidates, cap `kOsdPrefixCapA` = 640), and
     the host sorts the collected records and walks them (`Osd0Engine::run_candidates`).
   - If the pivot walk runs past that prefix, `Engine::osd_repick` re-picks a larger prefix on
     the device (rung two: `kOsdPrefixK` = 1024, cap `kOsdPrefixCap` = 2048) from the same sample
     and histogram, and `Osd0Engine::resume_candidates` continues the walk from its state. A
     walk that runs past rung two fetches the full posterior and runs `Osd0Engine::run` over the
     complete order.
   - Every rung is a prefix of the same reference order (ascending posterior LLR, ties by
     ascending column index, NaN last), so all three paths walk exactly the columns the full
     order would, and produce the same correction bit for bit.
   - OSD-0 is a GF(2) elimination in the reference order: columns are visited in that order,
     a column that is independent of the current basis becomes a pivot, the syndrome is reduced
     as it goes, and the unique solution supported on the pivots is recovered by
     back-substitution. The engine keeps the columns as bit-packed bitsets and visits only the
     64-bit words a column actually spans, so the visited set is the reference's set with its
     zero words skipped.
   - The vendored `ldpc::osd::OsdDecoder` in `src/third_party/ldpc/` remains for the
     `osd_cs` and `osd_e` methods, and for an OSD-0 shot whose syndrome is outside the image of
     `H` (the reference's own answer there depends on its internal pivot choice).
7. **Observables.** `O * correction` over GF(2). For `k <= 64` the core keeps one 64-bit mask per
   column of `O` (`dbench_decoder::o_colmask`), so the observables are a masked XOR scan of the
   correction (`compute_obs`), or one XOR per set column of the OSD-0 support
   (`compute_obs_support`). For `k > 64` it walks `O`'s CSC lists.
8. **Status word.** `(flags & 0xFFFF) | (iterations << 16)`, with the flags of
   `include/eldpc.h`: `ST_BP_CONVERGED`, `ST_POSTPROC_INVOKED`, `ST_HIGHER_ORDER_WIN`,
   `ST_ABSTAIN`, `ST_SYNDROME_UNMATCHED`. The OSD-0 path cannot return an unmatched syndrome by
   construction (it solves `H x = s`), so the re-check `H x = s` runs only on the vendored path.
   A plain BP handle whose BP did not converge returns its hard decision with
   `ST_SYNDROME_UNMATCHED`.

## 3. The GPU BP kernel

The kernel (`bp_kernel`, `src/core/bp_gpu.cu`) implements min-sum BP with the parallel schedule:

- **Check pass.** For each row, reduce the incoming messages to the smallest magnitude, the
  second smallest and the sign parity. All three are order-free, so they may be reduced by
  parallel threads; every outgoing message is `(m2 if |v| == m1 else m1) * (+-alpha)`, with the
  final sign as the operand of the single multiply, exactly as the reference.
- **Variable pass.** One thread per column walks the column in increasing row order and performs
  the reference's exact sequence of additions and the `+0.0` restore.
- **Rounding.** The GPU has no floating-point rounding-mode state. The host reads
  `fegetround()` once at `eldpc_create` and launches the matching instantiation of the kernel
  (`__dadd_rn`/`rd`/`ru`/`rz`, `__dmul_rn`/...), so the device arithmetic rounds the way the
  caller's process does.
- **Two formulations, chosen at create.** The in-place formulation writes check-to-bit messages
  over the bit-to-check array (8 bytes per edge, the register path, up to 8 warps per long row).
  The row-summary formulation stores `(m1, m2, sign)` per row and recomputes each message in the
  column pass; it is chosen when the in-place grid does not fit the device's occupancy or the
  message array does not fit the L2 budget (`Engine::setup`). Both produce the same values.
- **Layout.** The interleaved 32-column layout groups columns into aligned blocks so a warp reads
  contiguous slots; it is used only when its padding waste is at most 12% (`kIlvWasteLimitPct`),
  otherwise the CSC layout is kept. A degree-aware permutation decides which warp group walks
  which columns. Both are create-time permutations: they move work, not values.
- **Sizing.** The grid, the number of warps per row, the formulation and the shared-memory window
  are derived at create from the device's SM count and L2 size and from `H`'s row degrees
  (`Engine::setup`); nothing is hard-coded to one GPU model.
- **OSD-0 prefix.** On a non-converged OSD-0 shot the kernel also computes, in a tail pass, the
  keys and histogram from which the host-side order prefix is collected. A converged shot, or a
  `decoder=bp` handle, pays nothing for it.

## 4. The two cores and the CPUID / XGETBV choice

`CMakeLists.txt` compiles the same core sources twice, with different host ISA flags:

| library | host flags |
|---|---|
| `libeldpc_core_avx2.so` | `-mavx2` |
| `libeldpc_core_avx512.so` | `-mavx2 -mavx512f -mavx512vl -mavx512dq -mavx512bw` |

Both contain the same CUDA images (SASS for `sm_89` and `sm_120`, PTX for `compute_120`), and both
export the same `dbench_*` symbols. `libeldpc.so` (`src/port/eldpc_front.cpp`) picks one at load
time, in its ELF constructor:

1. `ELDPC_ISA=avx2` or `ELDPC_ISA=avx512` forces that core. It is read once, in the constructor,
   and never again; the core sources never read the environment.
2. Otherwise `cpu_has_avx512_os()` is tried first: CPUID for AVX2 and AVX-512 F/VL/DQ/BW, plus
   `XGETBV` checking that the OS has enabled the opmask and ZMM register state (XCR0 bits 5-7)
   and the XMM/YMM state (bits 1-2).
3. If that fails, `cpu_has_avx2_os()` is tried: CPUID for AVX2 plus OSXSAVE, AVX and the XCR0
   XMM/YMM bits.
4. If neither holds, loading fails with a message, and every `eldpc_create` reports it.

The front then `dlopen`s `libeldpc_core_<isa>.so` from its own directory with
`RTLD_NOW | RTLD_LOCAL`, resolves the ten required symbols (`dbench_create`, `dbench_decode`,
`dbench_decode_batch`, `dbench_set_priors`, `dbench_get_soft_output`, `dbench_destroy`,
`dbench_abi_version`, `dbench_decoder_name`, `dbench_capabilities`, `fastbp_keep_warm`) plus the
optional `dbench_decoder_build_info`, and checks the ABI version. `eldpc_isa()` reports the
selected core; `eldpc_build_info()` reports the front compiler, the core's compiler, flags, CUDA
toolkit, GPU architectures and ISA. `eldpc.load(isa=...)` (Python) sets `ELDPC_ISA` before the
first load, because the choice is made once per process.

`tests/test_isa.py` checks that both cores give identical results, that an unknown value and a
forced AVX-512 on a CPU without it fail with a clear error, and that the choice cannot change
after the first load.

### 4.1 The device-class variant: default and ada64

Each ISA core is in fact built twice: the default build, and an `ada64` build that differs
only by the compile-time macro `DBENCH_GPU_SM_BUDGET=64` (the core sources are the same files,
byte for byte). On a 142-SM Ada part the 64-SM launch plan is faster and gives bit-identical
outputs. The front picks the variant per decoder handle at create time from the device-class
table in `src/port/core_variant.h` (compute capability and an SM-count range; today one row:
cc 8.9 with at least 128 SMs -> `ada64`, anything else -> default), within the ISA the process
loaded. `ELDPC_DEVICE_CLASS=default|ada64` overrides the table (an empty value counts as unset; any
other value makes the create fail). `eldpc_core_variant()` / `Decoder.core_variant` report a
handle's variant and `eldpc.devices()` the variant each GPU would get; docs/BUILD.md has the
details and how to change the table after measuring a new device class.

### 4.2 The CPU core and the device routing

`src/cpu/` holds a second decoder behind the same decoder ABI (the `dbench_*` entry points), the
CPU core `fastbpcpu6g`: the four sources are identical to the sources they came from except for the same
comment-only difference (code tokens and line structure: `src/cpu/TOKENS.sha256`; raw bytes:
`src/cpu/SOURCES.sha256`). It is the multi-threaded exact min-sum BP engine with the
same bit-packed OSD-0 engine the GPU core uses on the host, compiled twice with the GPU cores'
host flags (`libeldpc_cpu_avx2.so`, `libeldpc_cpu_avx512.so`), with no CUDA source and no CUDA
library.

The front routes each create by the config key `device=gpu|cpu|auto`, which it reads and removes
before the core parses the rest of the string (so no core knows the key):

- `device=gpu` (the default): the front queries device 0's properties (as section 4.1
  describes), chooses the variant, and calls the GPU core's `dbench_create`. A create that finds
  no usable CUDA device -- no driver or device, a driver older than the cores' CUDA major
  version, or a compute capability the cores carry no image for (`src/port/gpu_arch.h`) -- fails
  before the core is called, with a message that says which and names `device="cpu"`.
- `device=auto`: `device=gpu` when those checks pass, else `device=cpu` (always `cpu` in a
  CPU-only build); `threads=` and `cpus=` are dropped for a GPU handle. The Python
  `device="auto"` makes the same choice in `python/eldpc/_auto.py` and warns once on a fallback.
- `device=cpu`: the front loads `libeldpc_cpu_<isa>.so` for the host ISA it chose at load time
  (the same `ELDPC_ISA` / CPUID / XGETBV choice as for the GPU cores; `eldpc_isa()` is unchanged)
  with `dlopen(RTLD_LOCAL)` on first use, and calls its `dbench_create` with the unchanged rest of
  the config string (`threads=`, `cpus=` and the decoder keys). It makes no CUDA call at all — no
  device query, no driver or runtime call — so it works with no driver and with
  `CUDA_VISIBLE_DEVICES` set empty. `ELDPC_DEVICE_CLASS` is not read on this path.

A CPU-only build (`ELDPC_WITHOUT_CUDA`) compiles the front with `ELDPC_CPU_ONLY`: the CPU core is
then the process's core loaded by the front's constructor (`eldpc_isa()` and `eldpc_build_info()`
report it), and a `device=gpu` create fails with a message that names `device="cpu"` instead of
looking for a GPU core.

The handle records the core it came from and whether it is a CPU handle; every later `eldpc_*`
call forwards to that core. `eldpc_core_variant()` is `"cpu"` for a CPU handle, and
`eldpc_keep_warm()` returns `ELDPC_OK` and does nothing there (the CPU core exports no
`fastbp_keep_warm`; the GPU cores do). GPU and CPU handles can be alive in one process; all of
them use the one host ISA the front chose for the process.

The call path of one CPU decode is:

```
eldpc_decode()                  src/port/eldpc_front.cpp: forwards to the CPU core's dbench_decode
  dbench_decode()               src/cpu/fastbpcpu6g.cpp: decode_one()
    all-zero syndrome?          zeros + BP_CONVERGED, 0 iterations
    otherwise                   fastbp::Engine (src/cpu/bp_engine.hpp), threads T
      the calling thread computes as thread 0; threads 1..T-1 were started at create and
      pinned to the first T-1 entries of cpus, and busy-wait for the next shot between decodes
    host OSD-0                   src/cpu/osd0_engine.hpp (the same engine as src/core/'s)
    observables O*x              same code as src/core/fastbpgpu6g.cpp
```

The outputs are bit-identical to the GPU core's and to the reference at the same settings; the
package's tests check the identity against the packaged reference and, with a GPU, against the
GPU core (`tests/test_cpu_core.py`, `tests/test_cpu_gpu_agreement.py`). Placement, threads and
the busy-wait are covered in the README's CPU core section.

## 5. The CUDA runtime, the driver and device enumeration

All cores link the CUDA runtime statically (`CMAKE_CUDA_RUNTIME_LIBRARY Static` and
`CUDA::cudart_static` in `CMakeLists.txt`). Each core therefore carries the runtime of the
toolkit it was built with (the `cuda=` field of `eldpc.build_info()` names that toolkit), and the
installed libraries have no `libcudart` dependency: no CUDA runtime library and no CUDA toolkit
is needed on the machine that runs the package, and a `libcudart.so` elsewhere on the system can
never be picked up, whatever the loader search path contains. The only CUDA library loaded at run
time is the NVIDIA driver's `libcuda.so.1`, which the static runtime opens itself. The installed
libraries carry no `RUNPATH` that names a CUDA directory. `docs/BUILD.md` ("Static CUDA runtime")
gives the reasons and the `readelf` check.

`eldpc.devices()` (`python/eldpc/_devices.py`) does not go through the decoder at all: it opens
`libcuda.so.1` with ctypes and uses the CUDA driver API (`cuInit`, `cuDeviceGetCount`,
`cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceGetUuid`, `cuDriverGetVersion`), so it works
before any core is loaded and on a host whose CPU can run neither core. Besides the name, UUID,
compute capability, SM count and L2 size of each device it reports two different versions:

- `cuda_driver_version` (and its alias `driver_version`): the newest CUDA version the installed
  driver supports, from `cuDriverGetVersion`;
- `nvidia_driver_version`: the version of the NVIDIA driver package, from NVML
  (`libnvidia-ml.so.1`, `nvmlSystemGetDriverVersion`), or `None` with the reason in
  `nvidia_driver_version_reason` when NVML is not installed or the query fails. A successful read
  is cached for the life of the process.

The decoder itself uses device 0 of the devices the process sees (`Engine::setup` calls
`cudaSetDevice(0)`); `CUDA_VISIBLE_DEVICES` selects which physical GPU that is.
`tests/test_devices.py` checks both version keys, a simulated missing NVML and enumeration
without a loaded core, and also that the installed libraries have no `libcudart` dependency
(`readelf -d`, `ldd`), that neither enumeration nor a decode maps a `libcudart`, and that a broken
`libcudart.so.13` on `LD_LIBRARY_PATH` does not affect a decode.

## 6. Keep-warm

A GPU that has been idle lowers its clocks, so a decoder called every few milliseconds can be
slower than one called back to back. `eldpc_keep_warm(h, ELDPC_KEEP_WARM_GPU)` (Python
`Decoder.keep_warm()`) calls the core's `fastbp_keep_warm`, which launches one empty 1-block kernel
on the handle's stream (`Engine::keep_warm_gpu`, `keep_warm_kernel` in `src/core/bp_gpu.cu`) and
returns. It does not synchronise, allocate, start a thread or touch any state a decode reads, so
it cannot change a result (`tests/test_api.py::test_keep_warm_does_not_change_results` decodes
with and without it and compares). The tuning records call it every 0.5 ms while the handle is
idle (`tuning/KNOWLEDGE.md` section 4.4, which also records that every 2 ms was not enough on the
machine measured there); a call still in flight when a syndrome arrives delays the decode by its
launch time.

## 7. The batch path

`eldpc_decode_batch` (`src/core/fastbpgpu6g.cpp`) is a loop over `decode_one`: the shots are
decoded one after another on the same handle, each with its own cooperative kernel launch. It is
not a batched GPU kernel, and its results are exactly the single-shot results in order (the ABI
requires it, `tests/test_api.py::test_shapes_and_dtypes` and the verification suite check it).
The Python `Decoder.decode_batch` allocates the `(shots, n)` and `(shots, k)` output arrays and
makes one ctypes call, so it amortises the Python and ctypes overhead, not the GPU work.
Throughput is therefore a property of the batch size and of the per-shot cost, and is not the
reciprocal of the single-shot latency.

## 8. Memory per decoder

All allocation happens at `eldpc_create` (and in `set_priors` for the four n-sized host arrays);
the decode path allocates nothing besides the caller's output buffers. The dominant arrays for a
problem with `m` detectors, `n` mechanisms, `nnz` entries in `H` and `k` observables, with the
device tables in the interleaved layout (`nslots` = the number of 32-column groups times 32 times
9 slots per column, `kRegDeg`):

| where | array | bytes |
|---|---|---:|
| device | `H` column pointers, row pointers | `4(n+1) + 4(m+1)` |
| device | edge lists `redge`, `rcol` | `8 nnz` |
| device | interleaved slot rows, `b2c`, `c2b` | `4 nslots + 16 nslots` |
| device | priors, prior positions, per-row accumulators | `16 n + O(m)` |
| device | decisions, status, posterior (two parity buffers) | `~18 n` |
| device | OSD-0 only: order keys, histogram, candidate and hand-over blocks | `8 n + ~72 KiB` |
| host | `H` row-major copy, `O` CSC | `4 nnz + 4(m+1) + 4 n + 4 nnz(O)` |
| host | `channel_probs`, `prior_llr`, `posterior`, two scratch vectors | `40 n` |
| host | pinned output staging (`decisions + status + posterior`) | `9 n + 8` |
| host | OSD-0 only: dense column table, basis, parents, per-shot buffers | `64 n + O(m²/4)` |

`Engine::setup` chooses `nslots` and the formulation; `dbench_create` builds the host copies.
The fixed cost of a CUDA context is per process, not per handle; a second handle adds only the
tables above. For scale, on the benchmark problem (`m` = 1872, `n` = 75600, `nnz(H)` = 437184,
`nnz(O)` = 288776, `k` = 24) these terms add up to about 20 MiB of device tables and about
12 MiB of host tables per handle, before the CUDA context (computed from the table, not
measured). `docs/tools/measure_memory.py` measures what a handle adds to the process -- RSS and
the driver's per-process GPU accounting, including the create-time temporaries and the
allocator's retained space that the table does not count -- and writes
[results/memory/decoder-memory.md](results/memory/decoder-memory.md) with its command, commit and
hardware.

The Python `decode_batch` allocates `shots × n` bytes for corrections and `shots × k` for
observables: on the same problem a million shots would need 76 GB of correction data. Decode
large shot sets in chunks, or use the C API with `out_corrections = NULL` when the correction is
not needed.

## 9. The exactness rules of the build

The decoder's outputs must not depend on the machine that compiled it. The build rules that
enforce this are in `CMakeLists.txt`:

- **Host flags**: `-O2 -DNDEBUG -ffp-contract=off -fno-fast-math` (the decoder's reference
  flags), then `-O3`, then the ISA flags of the core. No `-march=native`, no FMA contraction, no
  fast math: `-ffp-contract=off` keeps a multiply-add written as two operations from being fused
  into one instruction, and `-fno-fast-math` keeps the compiler from reassociating or replacing
  floating-point operations. The OSD-0 walk and the observable masks depend on these values.
- **CUDA flags**: `-fmad=false -ftz=false -prec-div=true -prec-sqrt=true`, last on every `nvcc`
  command line. `-fmad=false` keeps a multiply and an add from becoming an FMA on the device
  (the reference computes them separately); `-ftz=false` preserves subnormals; `-prec-div` and
  `-prec-sqrt` keep division and square root IEEE-correct. The kernel additionally uses explicit
  `__dadd_*` / `__dmul_*` intrinsics for the messages, so the arithmetic does not depend on
  `nvcc`'s optimisation of an expression.
- **No `-march=native`**: the AVX2 and AVX-512 cores use exactly the flag sets in section 4, so
  a build on a wider CPU does not emit instructions the target CPU may lack, and the two cores
  differ only in the code paths they enable.
- **Pinned sources**: `src/core/SOURCES.sha256` lists the raw bytes of the six decoder sources and
  `src/core/TOKENS.sha256` the text without comments (`tools/check_comment_only.py`);
  `tests/test_core_sources.py` and `tools/check_core_tarball.py` compare them with the snapshot
  (the text without comments; the comments were neutralised). The core sources are never edited.
- **No I/O or environment in the core**: per the decoder ABI, the decode path performs no file,
  environment or process call. The only environment variable the package's own native code reads
  is `ELDPC_ISA`, once, in the front's load-time constructor; the CUDA driver and the statically
  linked CUDA runtime read their own variables, such as `CUDA_VISIBLE_DEVICES`.

The exactness itself (same correction, observables and status word as the reference) is checked
by `verify/run_verify.py`, which compares against a reference built from the pinned sources with
the same determinism flags; see [VERIFY.md](VERIFY.md).

## 10. Source map

| file | contents |
|---|---|
| `include/eldpc.h` | the public C API: problem struct, status word, entry points |
| `src/port/eldpc_front.cpp` | CPUID/XGETBV choice, core loading, the `device` routing, forwarding |
| `src/port/decoder_abi.h` | decoder ABI v1 (frozen), the core's internal contract |
| `src/core/fastbpgpu6g.cpp` | the ABI over the engine: config parsing, problem validation, `decode_one`, observables, status |
| `src/core/bp_engine.hpp` | the engine interface and the exactness argument of the message schedule |
| `src/core/bp_gpu.cu` | the CUDA kernel, device memory, the mapped status hand-off, OSD-0 prefix, keep-warm |
| `src/core/osd0_engine.hpp` | the exact bit-packed OSD-0 with the reference column order |
| `src/core/osd_caps.hpp` | the device prefix rung sizes |
| `src/cpu/fastbpcpu6g.cpp` | the CPU core's ABI: config parsing, the thread team, `decode_one`, observables, status |
| `src/cpu/bp_engine.hpp` | the multi-threaded exact min-sum BP engine of the CPU core |
| `src/cpu/osd0_engine.hpp`, `src/cpu/osd_caps.hpp` | the CPU core's copies of the two OSD-0 files (byte-identical to `src/core/`'s) |
| `python/eldpc/__init__.py` | `Decoder`, `load`, `build_info`, `isa`, `devices` |
| `python/eldpc/_abi.py` | ctypes binding of `eldpc_*`, handle lifetime, buffers |
| `python/eldpc/_config.py` | the presets and the canonical ABI config string |
| `python/eldpc/_dem.py` | the DEM-to-(H, O, priors) rule |
| `python/eldpc/_devices.py` | device enumeration through the CUDA driver API, the NVIDIA driver version through NVML |
| `python/eldpc/_status.py` | the status-word constants and the `status` helpers |
| `python/eldpc/qldpc.py` | the qLDPC decoder specification |
| `CMakeLists.txt`, `pyproject.toml` | the seven libraries, their flags, the static CUDA runtime, the wheel layout |
